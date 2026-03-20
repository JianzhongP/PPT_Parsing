"""
Phase 1: 版面检测与类型清洗 (Layout Detection & Type Refinement)

核心目的：在低算力（CPU）条件下快速获取所有元素的坐标，并修正 MinerU 将表格/图表误判为图片的问题。

包含组件：
1. MinerUClient: MinerU版面分析客户端（支持标准版和VLM版）
2. ROICropper: ROI区域裁剪器
3. TypeRefinementEngine: VLM类型清洗引擎
4. Phase1_LayoutDetector: Phase 1 总控制器
"""

import os
import sys
import json
import time
import base64
import asyncio
import subprocess
import threading
import shutil
import re
import numpy as np
from PIL import Image
from pathlib import Path
from contextlib import contextmanager, nullcontext
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .pipeline_state import (
    ElementBBox, DetectedElement, ROIImage, CleanedLayoutJSON
)


def _fallback_vlm_model_from_env() -> str:
    provider = (os.getenv("MULTIMODAL_PROVIDER", "gpt4o") or "").strip().lower()
    if provider in {"qwen", "qwen3-vl-plus", "dashscope"}:
        return (os.getenv("VLM_MODEL_NAME", "qwen3-vl-plus") or "qwen3-vl-plus").strip()
    return (os.getenv("G4O_MODEL_NAME", "gpt-4o") or "gpt-4o").strip()


def _read_gpu_mem_mb(gpu_index: int = 0) -> Tuple[Optional[int], Optional[int]]:
    """读取指定 GPU 的已用/总显存 (MB)，失败时返回 (None, None)。"""
    try:
        if not shutil.which("nvidia-smi"):
            return None, None

        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu_index),
        ]
        out = subprocess.check_output(cmd, text=True, timeout=2).strip()
        used_str, total_str = [x.strip() for x in out.split(",", 1)]
        return int(used_str), int(total_str)
    except Exception:
        return None, None


def _read_gpu_process_mem_map(gpu_index: int = 0) -> Dict[int, int]:
    """读取指定 GPU 上各计算进程显存占用（MB）。"""
    try:
        if not shutil.which("nvidia-smi"):
            return {}

        cmd = [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu_index),
        ]
        out = subprocess.check_output(cmd, text=True, timeout=2).strip()
        if not out:
            return {}

        mem_map: Dict[int, int] = {}
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
                used = int(parts[1])
                mem_map[pid] = used
            except Exception:
                continue
        return mem_map
    except Exception:
        return {}


def _collect_descendant_pids(root_pid: int) -> set[int]:
    """收集 root_pid 及其后代进程 PID（Linux/macOS best effort）。"""
    if root_pid <= 0:
        return set()

    if sys.platform.startswith("win"):
        return {root_pid}

    try:
        out = subprocess.check_output(["ps", "-e", "-o", "pid=,ppid="], text=True, timeout=2)
        children_map: Dict[int, List[int]] = {}
        for line in out.splitlines():
            cols = line.split()
            if len(cols) < 2:
                continue
            try:
                pid = int(cols[0])
                ppid = int(cols[1])
            except Exception:
                continue
            children_map.setdefault(ppid, []).append(pid)

        seen = set()
        stack = [root_pid]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(children_map.get(current, []))
        return seen
    except Exception:
        return {root_pid}


class _GpuProcessMemSampler:
    """按 MinerU 主进程及其子进程采样显存，同时保留整卡对照值。"""

    def __init__(self, root_pid: int, gpu_index: int = 0, interval_sec: float = 0.2):
        self.root_pid = int(root_pid)
        self.gpu_index = gpu_index
        self.interval_sec = max(0.05, float(interval_sec))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.baseline_proc_mb: Optional[int] = None
        self.peak_proc_mb: Optional[int] = None
        self.baseline_total_mb: Optional[int] = None
        self.peak_total_mb: Optional[int] = None
        self.total_mb: Optional[int] = None
        self.samples: int = 0

    def start(self):
        self.baseline_total_mb, self.total_mb = _read_gpu_mem_mb(self.gpu_index)
        self.peak_total_mb = self.baseline_total_mb

        pid_set = _collect_descendant_pids(self.root_pid)
        mem_map = _read_gpu_process_mem_map(self.gpu_index)
        proc_used = sum(mem_map.get(pid, 0) for pid in pid_set)
        self.baseline_proc_mb = proc_used
        self.peak_proc_mb = proc_used

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            used_mb, total_mb = _read_gpu_mem_mb(self.gpu_index)
            if used_mb is not None:
                self.total_mb = total_mb
                if self.peak_total_mb is None:
                    self.peak_total_mb = used_mb
                else:
                    self.peak_total_mb = max(self.peak_total_mb, used_mb)

            pid_set = _collect_descendant_pids(self.root_pid)
            mem_map = _read_gpu_process_mem_map(self.gpu_index)
            proc_used = sum(mem_map.get(pid, 0) for pid in pid_set)
            if self.peak_proc_mb is None:
                self.peak_proc_mb = proc_used
            else:
                self.peak_proc_mb = max(self.peak_proc_mb, proc_used)

                self.samples += 1
            time.sleep(self.interval_sec)

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def summary(self) -> Optional[Dict[str, int]]:
        if self.baseline_proc_mb is None or self.peak_proc_mb is None:
            return None

        proc_delta_mb = max(0, self.peak_proc_mb - self.baseline_proc_mb)
        total_delta_mb = max(0, (self.peak_total_mb or 0) - (self.baseline_total_mb or 0))
        return {
            "pid": int(self.root_pid),
            "baseline_proc_mb": self.baseline_proc_mb,
            "peak_proc_mb": self.peak_proc_mb,
            "delta_proc_mb": proc_delta_mb,
            "baseline_total_mb": int(self.baseline_total_mb) if self.baseline_total_mb is not None else 0,
            "peak_total_mb": int(self.peak_total_mb) if self.peak_total_mb is not None else 0,
            "delta_total_mb": int(total_delta_mb),
            "total_mb": int(self.total_mb) if self.total_mb is not None else 0,
            "samples": int(self.samples),
        }


def _write_gpu_mem_debug(debug_dir: Optional[str], record: Dict[str, Any]) -> None:
    """将 GPU 显存采样结果追加写入 debug 文件（JSONL + 易读文本）。"""
    if not debug_dir:
        return
    try:
        os.makedirs(debug_dir, exist_ok=True)

        jsonl_path = os.path.join(debug_dir, "mineru_gpu_mem.jsonl")
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        txt_path = os.path.join(debug_dir, "mineru_gpu_mem.log")
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(
                f"[{record.get('timestamp','')}] page={record.get('page','')} gpu={record.get('gpu_index','')} "
                f"baseline={record.get('baseline_mb','-')}MB "
                f"peak={record.get('peak_mb','-')}MB "
                f"delta={record.get('delta_mb','-')}MB "
                f"total={record.get('total_mb','-')}MB "
                f"samples={record.get('samples','-')} "
                f"note={record.get('note','')}\n"
            )
    except Exception as e:
        print(f"[MinerU][GPU] 写入 debug 显存日志失败: {e}")


# ============================================================================
# MinerU 客户端
# ============================================================================

class MinerUClient:
    """
    MinerU 版面分析客户端
    
    修复：使用 python -m 模块方式调用，避免 Windows exe 包装器问题
    """

    _vlm_semaphore_lock = threading.Lock()
    _vlm_semaphore: Optional[threading.Semaphore] = None
    _vlm_semaphore_limit: int = 1
    
    def __init__(self, 
                 mode: str = "standard",
                 mineru_path: Optional[str] = None,
                 docker_image: str = "mineru-cpu:latest",
                 use_docker: bool = False,
                 vlm_client=None):
        self.mode = mode
        self.mineru_path = mineru_path
        self.docker_image = docker_image
        self.use_docker = use_docker
        self.vlm_client = vlm_client  # VLM 客户端，用于回退版面检测

        # 运行状态记录
        self.last_run_debug_dir: Optional[str] = None
        self.last_run_output_dir: Optional[str] = None
        self.last_run_cmd: Optional[List[str]] = None
        self.last_run_returncode: Optional[int] = None
        self.last_run_pid: Optional[int] = None
        self.last_run_stdout_path: Optional[str] = None
        self.last_run_stderr_path: Optional[str] = None
        self.last_run_used_fallback: bool = False
        
        self._check_availability()
    
    def _check_availability(self):
        """检查 MinerU (Magic-PDF) 是否可用"""
        self.is_available = False
        self.launch_module = None
        
        if self.use_docker:
            # Docker 检查保持不变
            try:
                result = subprocess.run(["docker", "images", "-q", self.docker_image], capture_output=True, text=True)
                if result.stdout.strip():
                    self.is_available = True
            except:
                pass
        else:
            # 优先检查 python 模块是否存在，这比检查 exe 更可靠
            try:
                # 尝试导入 magic_pdf 模块以确认安装
                import importlib.util
                # 注意：有些环境里可能存在 magic_pdf 包名但不包含 cli 子模块，
                # 因此这里要检查到“可执行子模块”级别。
                magic_cli_spec = None
                mineru_cli_spec = None
                try:
                    magic_cli_spec = importlib.util.find_spec("magic_pdf.cli.magicpdf")
                except ModuleNotFoundError:
                    magic_cli_spec = None

                try:
                    mineru_cli_spec = importlib.util.find_spec("mineru.cli.client")
                except ModuleNotFoundError:
                    mineru_cli_spec = None

                if magic_cli_spec:
                    self.is_available = True
                    self.launch_module = "magic_pdf.cli.magicpdf"
                    print("[MinerU] 检测到 magic_pdf.cli.magicpdf，将使用 python -m magic_pdf.cli.magicpdf 调用")
                elif mineru_cli_spec:
                    self.is_available = True
                    self.launch_module = "mineru.cli.client"
                    print("[MinerU] 检测到 mineru.cli.client，将使用 python -m mineru.cli.client 调用")
                else:
                    # 回退到 CLI 检查
                    if shutil.which("magic-pdf") or shutil.which("mineru"):
                        self.is_available = True
                        print(f"[MinerU] 未检测到 Python 模块，但发现本地 CLI")
                    else:
                        print("[MinerU] 未检测到本地 MinerU/Magic-PDF，将使用模拟模式")
            except Exception as e:
                print(f"[MinerU] 环境检测异常: {e}")

    def analyze(self, image_path: str, debug_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        # 重置状态
        self.last_run_debug_dir = os.path.abspath(debug_dir) if debug_dir else None
        self.last_run_output_dir = None
        self.last_run_used_fallback = False

        if self.is_available:
            if self.use_docker:
                return self._analyze_docker(image_path, debug_dir=debug_dir)
            else:
                return self._analyze_local(image_path, debug_dir=debug_dir)
        else:
            return self._analyze_vlm_fallback(image_path, vlm_client=self.vlm_client)

    def _gpu_is_available(self) -> bool:
        """Best-effort check whether CUDA GPU is usable.

        Notes:
        - If CUDA_VISIBLE_DEVICES is explicitly set to -1, treat as unavailable.
        - Prefer nvidia-smi when present; fall back to torch.cuda.is_available().
        """
        if os.getenv("CUDA_VISIBLE_DEVICES", "").strip() == "-1":
            return False

        # Fast path: nvidia-smi
        try:
            if shutil.which("nvidia-smi"):
                r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=2)
                if r.returncode == 0 and (r.stdout or "").strip():
                    return True
        except Exception:
            pass

        # Fallback: torch (may not be installed)
        try:
            import torch  # type: ignore
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @classmethod
    def _get_vlm_semaphore(cls) -> threading.Semaphore:
        """全局 VLM 并发闸门，默认 1，可用 MINERU_VLM_MAX_CONCURRENCY 覆盖。"""
        with cls._vlm_semaphore_lock:
            if cls._vlm_semaphore is None:
                raw = os.getenv("MINERU_VLM_MAX_CONCURRENCY", "1").strip()
                try:
                    limit = int(raw)
                except Exception:
                    limit = 1
                limit = max(1, limit)
                cls._vlm_semaphore_limit = limit
                cls._vlm_semaphore = threading.Semaphore(limit)
                print(f"[MinerU][VLM Gate] 初始化并发上限: {limit}")
            return cls._vlm_semaphore

    @contextmanager
    def _vlm_gate(self, page_tag: str):
        semaphore = self._get_vlm_semaphore()
        wait_start = time.time()
        print(f"[MinerU][VLM Gate] {page_tag}: 等待 VLM 槽位...")
        semaphore.acquire()
        waited_ms = int((time.time() - wait_start) * 1000)
        print(f"[MinerU][VLM Gate] {page_tag}: 已获取槽位 (wait={waited_ms}ms, limit={self._vlm_semaphore_limit})")
        try:
            yield
        finally:
            semaphore.release()
            print(f"[MinerU][VLM Gate] {page_tag}: 已释放槽位")

    def _analyze_local(self, image_path: str, debug_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """通过本地 Python 模块调用 MinerU"""
        try:
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)

            page_tag = Path(image_path).stem

            # 1. 检查并初始化 magic-pdf.json (关键步骤)
            self._ensure_config_file()

            # 2. 转换图片为 PDF
            input_file_path = self._image_to_pdf(image_path)
            abs_input_path = os.path.abspath(input_file_path)
            
            # 3. 准备输出目录
            base_dir = os.path.dirname(abs_input_path)
            output_dir = os.path.join(base_dir, "mineru_output")
            # 清理旧数据
            if os.path.exists(output_dir):
                try:
                    shutil.rmtree(output_dir)
                except Exception as e:
                    print(f"[MinerU] 清理旧目录警告: {e}")
            os.makedirs(output_dir, exist_ok=True)
            
            self.last_run_output_dir = output_dir

            # 4. 构造命令 (使用 python -m 方式)
            if self.launch_module:
                cmd = [sys.executable, "-m", self.launch_module]
            else:
                # 回退到 exe
                cmd = ["magic-pdf"] if shutil.which("magic-pdf") else ["mineru"]
            
            # 4.1 根据模式选择后端/设备
            # - standard: pipeline + cpu（尽量省资源）
            # - vlm: 默认走 MinerU 的 VLM 本地引擎（需要 GPU），也允许用户通过环境变量覆盖
            if self.mode == "vlm":
                backend = os.getenv("MINERU_VLM_BACKEND", "vlm-auto-engine")
                device = os.getenv("MINERU_VLM_DEVICE", "cuda:0")
            else:
                backend = os.getenv("MINERU_STD_BACKEND", "pipeline")
                # 默认 auto：有 GPU 就用 cuda，否则 cpu
                device_env = (os.getenv("MINERU_STD_DEVICE", "auto") or "auto").strip()
                if device_env.lower() in {"auto", ""}:
                    device = "cuda" if self._gpu_is_available() else "cpu"
                else:
                    device = device_env

            # 公式/表格开关：默认关闭公式以降低显存/内存占用
            enable_formula = os.getenv("MINERU_ENABLE_FORMULA", "0").strip().lower() in {"1", "true", "yes"}
            enable_table = os.getenv("MINERU_ENABLE_TABLE", "1").strip().lower() in {"1", "true", "yes"}

            # 模型来源：CLI 参数优先于 env
            model_source = os.getenv("MINERU_MODEL_SOURCE", "modelscope")
            vram_limit = os.getenv("MINERU_VRAM", "").strip()

            cmd.extend([
                "-p", abs_input_path,
                "-o", output_dir,
                "-b", backend,
                "-f", str(enable_formula),
                "-t", str(enable_table),
                "--source", model_source,
            ])

            # http-client 后端需要 server url
            if backend in {"vlm-http-client", "hybrid-http-client"}:
                server_url = os.getenv("MINERU_SERVER_URL") or os.getenv("MINERU_VLM_URL") or ""
                server_url = server_url.strip()
                if server_url:
                    cmd.extend(["-u", server_url])
                else:
                    raise RuntimeError(
                        f"[MinerU] backend={backend} 需要设置 MINERU_SERVER_URL (或 MINERU_VLM_URL)"
                    )

            # device/vram 仅对 backend=pipeline 生效（MinerU 帮助里有说明）
            if backend == "pipeline":
                cmd.extend(["-d", device])
                if vram_limit.isdigit():
                    cmd.extend(["--vram", vram_limit])

            # 5. 设置环境变量
            env = os.environ.copy()

            # 兼容旧版 MinerU：仍保留 env 方式，但不再强制覆盖 GPU 可见性
            env.setdefault("MINERU_MODEL_SOURCE", model_source)

            # device=cpu 时才显式禁用 GPU；否则尊重用户 CUDA_VISIBLE_DEVICES
            if backend == "pipeline" and str(device).lower().startswith("cpu"):
                env["CUDA_VISIBLE_DEVICES"] = "-1"
                env["MINERU_DEVICE_MODE"] = "cpu"
            else:
                env.pop("MINERU_DEVICE_MODE", None)

            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            # 允许 OpenMP 库重复加载，解决 DLL 冲突
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            # 强制单线程运行
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"

            # 记录命令
            self.last_run_cmd = cmd
            print(f"[MinerU] 执行命令: {' '.join(cmd)}")
            print(f"[MinerU] 输出目录: {output_dir}")
            if self.mode != "vlm":
                if backend == "pipeline":
                    print(f"[MinerU] Mode=standard: backend=pipeline, device={device}")
                else:
                    print(f"[MinerU] Mode=standard: backend={backend} (device 参数仅 pipeline 生效)")
            else:
                print(f"[MinerU] Mode=vlm: backend={backend}, device={device} (pipeline 时生效)")

            # 6. 执行命令 (使用文件重定向代替 Pipe，防止缓冲区溢出)
            stdout_path = os.path.join(debug_dir, "stdout.txt") if debug_dir else os.devnull
            stderr_path = os.path.join(debug_dir, "stderr.txt") if debug_dir else os.devnull
            
            self.last_run_stdout_path = stdout_path
            self.last_run_stderr_path = stderr_path

            needs_vlm_gate = bool(
                self.mode == "vlm" or backend in {"vlm-auto-engine", "vlm-http-client", "hybrid-http-client"}
            )

            gpu_index = 0
            if "cuda:" in str(device):
                try:
                    gpu_index = int(str(device).split("cuda:")[-1])
                except Exception:
                    gpu_index = 0

            with (self._vlm_gate(page_tag) if needs_vlm_gate else nullcontext()):
                gpu_sampler: Optional[_GpuProcessMemSampler] = None
                process: Optional[subprocess.Popen] = None

                try:
                    with open(stdout_path, "w", encoding="utf-8") as f_out, \
                         open(stderr_path, "w", encoding="utf-8") as f_err:
                        start_time = time.time()
                        process = subprocess.Popen(
                            cmd,
                            stdout=f_out,
                            stderr=f_err,
                            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
                            encoding="utf-8",
                            errors="replace",
                            env=env
                        )
                        self.last_run_pid = int(process.pid)

                        if "cuda" in str(device).lower() and shutil.which("nvidia-smi"):
                            gpu_sampler = _GpuProcessMemSampler(
                                root_pid=process.pid,
                                gpu_index=gpu_index,
                                interval_sec=0.2,
                            )
                            gpu_sampler.start()
                            print(f"[MinerU][GPU] {page_tag}: start pid monitor on gpu={gpu_index}, pid={process.pid}")

                        try:
                            return_code = process.wait(timeout=600)
                        except subprocess.TimeoutExpired:
                            try:
                                process.kill()
                            except Exception:
                                pass
                            raise

                        result = subprocess.CompletedProcess(args=cmd, returncode=return_code)
                        self.last_run_returncode = result.returncode

                finally:
                    if gpu_sampler is not None:
                        gpu_sampler.stop()
                        mem_stats = gpu_sampler.summary()
                        if mem_stats:
                            record = {
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "page": page_tag,
                                "gpu_index": gpu_index,
                                "pid": mem_stats["pid"],
                                "baseline_proc_mb": mem_stats["baseline_proc_mb"],
                                "peak_proc_mb": mem_stats["peak_proc_mb"],
                                "delta_proc_mb": mem_stats["delta_proc_mb"],
                                "baseline_total_mb": mem_stats["baseline_total_mb"],
                                "peak_total_mb": mem_stats["peak_total_mb"],
                                "delta_total_mb": mem_stats["delta_total_mb"],
                                "total_mb": mem_stats["total_mb"],
                                "samples": mem_stats["samples"],
                                "note": "pid-scope snapshot (root+children), with whole-gpu reference",
                            }
                            _write_gpu_mem_debug(debug_dir, record)
                            print(
                                f"[MinerU][GPU] {page_tag}: "
                                f"pid={mem_stats['pid']}, "
                                f"proc_base={mem_stats['baseline_proc_mb']}MB, "
                                f"proc_peak={mem_stats['peak_proc_mb']}MB, "
                                f"proc_delta≈{mem_stats['delta_proc_mb']}MB, "
                                f"gpu_peak={mem_stats['peak_total_mb']}MB, "
                                f"samples={mem_stats['samples']}"
                            )
                        else:
                            _write_gpu_mem_debug(debug_dir, {
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "page": page_tag,
                                "gpu_index": gpu_index,
                                "pid": int(process.pid) if process is not None else None,
                                "note": "pid monitor unavailable or nvidia-smi read failed",
                            })
                            print(f"[MinerU][GPU] {page_tag}: 无法读取 PID 显存数据")

            # 7. with 块已关闭，文件句柄释放，现在可以安全读取 stderr
            stderr_content = ""
            if stderr_path and os.path.exists(stderr_path) and stderr_path != os.devnull:
                try:
                    with open(stderr_path, "r", encoding="utf-8", errors="replace") as f:
                        stderr_content = f.read()
                except Exception as e:
                    print(f"[MinerU] 读取 stderr 文件失败: {e}")

            # 检查 stderr 中是否有严重错误（MinerU 可能返回码 0 但实际失败）
            stderr_has_error = False
            if stderr_content:
                error_keywords = ["OSError", "MemoryError", "RuntimeError", "CUDA error",
                                  "页面文件太小", "os error 1455", "Cannot allocate memory",
                                  "OutOfMemoryError", "Traceback (most recent call last)"]
                for kw in error_keywords:
                    if kw in stderr_content:
                        stderr_has_error = True
                        print(f"[MinerU] ⚠️ 在 stderr 中检测到严重错误关键字: {kw}")
                        break

            # 8. 查找并解析结果
            # 注意：即使 exit code 为 0，也可能因为内存不足等原因没生成文件
            # 所以这里必须检查 json 是否存在；同时 MinerU 不同 backend 可能产出多个 json，
            # 需要做“多候选重试解析”，避免某个 v2 结构变化导致误判失败。
            json_candidates = self._find_result_json_candidates(output_dir)

            if json_candidates:
                print(f"[MinerU] 找到结果文件: {json_candidates[0]}")
                last_parse_error: Optional[Exception] = None
                for json_path in json_candidates:
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        parsed = self._parse_mineru_json_output(data)

                        # 若能解析出 elements（即使为空也表示结构可读），就认为 MinerU 成功
                        if isinstance(parsed, dict) and "elements" in parsed:
                            return parsed
                        return parsed
                    except Exception as e:
                        last_parse_error = e
                        continue

                print(f"[MinerU] JSON 解析异常: {last_parse_error}")

            # 如果走到这里，说明失败了（没有产物或所有产物都解析失败）
            if json_candidates:
                print(f"[MinerU] ❌ 失败: 命令返回码 {result.returncode}，但所有 JSON 产物解析失败")
            else:
                print(f"[MinerU] ❌ 失败: 命令返回码 {result.returncode}，且未找到 JSON 产物")
            
            # 打印 STDERR 帮助调试 (从文件读取，非常重要)
            if stderr_content:
                print(f"[MinerU] STDERR (Last 1500 chars):")
                print(stderr_content[-1500:])
            else:
                print("[MinerU] STDERR 文件为空")
            
            if stderr_has_error:
                print(f"[MinerU] 💡 提示: MinerU 可能因内存不足而失败。")
                print(f"    建议: 1) 增大系统虚拟内存/页面文件  2) 关闭其他占用内存的程序  3) 使用 -d cpu 参数")
                
            self._debug_list_files(output_dir)
            self.last_run_used_fallback = True
            return self._analyze_vlm_fallback(image_path, vlm_client=self.vlm_client)

        except Exception as e:
            print(f"[MinerU] 执行异常: {e}")
            import traceback
            traceback.print_exc()
            self.last_run_used_fallback = True
            return self._analyze_vlm_fallback(image_path, vlm_client=self.vlm_client)

    def _ensure_config_file(self):
        """检查 magic-pdf.json 是否存在，如果不存在则提示或创建临时配置"""
        home_dir = os.path.expanduser("~")
        config_path = os.path.join(home_dir, "magic-pdf.json")
        
        if not os.path.exists(config_path):
            print(f"[MinerU] ⚠️ 未检测到配置文件 {config_path}")
            print(f"[MinerU] 尝试创建默认配置 (CPU/Modelscope)...")
            try:
                default_config = {
                    "bucket_info": {
                        "bucket-name": "magic-pdf",
                        "access-key": "",
                        "secret-key": "",
                        "endpoint": ""
                    },
                    "models-dir": os.path.join(home_dir, "magic-pdf-models"),
                    "device-mode": "cpu",
                    "table-config": {
                        "model": "TableMaster",
                        "is_table_recognize": True,
                        "max_time": 400
                    },
                    "layout-config": {
                        "model": "layoutlmv3"
                    },
                    # "formula-config": {
                    #     "mfd_model": "yolo_v8_n",
                    #     "mfr_model": "unimernet_small"
                    # }
                }
                # 暂时不自动创建，以免覆盖用户意图，只打印警告
                # with open(config_path, "w") as f:
                #     json.dump(default_config, f, indent=4)
                print(f"[MinerU] 请确保已运行 'magic-pdf --init' 或手动配置了 magic-pdf.json")
            except Exception as e:
                print(f"[MinerU] 配置检查失败: {e}")
        else:
            print(f"[MinerU] 配置文件已存在: {config_path}")

    def _find_result_json_candidates(self, output_root: str) -> List[str]:
        """递归查找 MinerU 生成的 JSON 产物（按优先级排序，供逐个尝试解析）。"""
        candidates: List[Tuple[int, str]] = []

        for root, _dirs, files in os.walk(output_root):
            for file in files:
                if not file.endswith(".json"):
                    continue
                if file in {"images.json", "mineru_output_index.json"}:
                    continue

                full_path = os.path.join(root, file)
                score = 0

                # 优先 content_list，其次 layout/model/middle
                if "content_list" in file:
                    score += 100
                if file.endswith("_content_list_v2.json"):
                    score += 10
                if "layout" in file:
                    score += 50
                if "model" in file:
                    score += 5
                if "middle" in file:
                    score += 1

                # 常见目录优先级
                if os.sep + "auto" + os.sep in (root + os.sep):
                    score += 20
                if os.sep + "vlm" + os.sep in (root + os.sep):
                    score += 15

                candidates.append((score, full_path))

        candidates.sort(key=lambda x: (x[0], -len(x[1])), reverse=True)
        return [p for _s, p in candidates]

    def _find_result_json(self, output_root: str) -> Optional[str]:
        """递归查找 MinerU 生成的最优 JSON 文件（兼容旧接口）。"""
        paths = self._find_result_json_candidates(output_root)
        return paths[0] if paths else None

    def _debug_list_files(self, root_dir):
        """调试辅助：打印目录结构"""
        print(f"[Debug] 目录结构 {root_dir}:")
        file_count = 0
        for root, dirs, files in os.walk(root_dir):
            level = root.replace(root_dir, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print(f"{indent}{os.path.basename(root)}/")
            for f in files:
                print(f"{indent}    {f}")
                file_count += 1
        if file_count == 0:
            print(f"[Debug] 目录为空或仅包含空文件夹")

    def _image_to_pdf(self, image_path: str) -> str:
        """将图片转换为PDF"""
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            pdf_path = os.path.splitext(image_path)[0] + ".pdf"
            image.save(pdf_path, "PDF", resolution=72.0)
            return pdf_path
        except Exception as e:
            print(f"[MinerU] 图片转PDF失败: {e}")
            return image_path

    def _analyze_docker(self, image_path: str, debug_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._analyze_mock(image_path)

    def _analyze_mock(self, image_path: str) -> List[Dict[str, Any]]:
        """静态模拟数据（仅在 VLM 回退也失败时使用）"""
        print(f"[MinerU] 使用静态模拟数据 (Mock)")
        img_width, img_height = 1920, 1080
        if PIL_AVAILABLE:
            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except: pass
            
        return [
            {"type": "Title", "bbox": [50, 50, 1800, 150], "text": "Mock Title", "confidence": 0.9},
            {"type": "Image", "bbox": [100, 200, 900, 800], "text": "", "confidence": 0.85},
            {"type": "Text", "bbox": [950, 200, 1800, 800], "text": "Mock Content", "confidence": 0.88}
        ]

    def _analyze_vlm_fallback(self, image_path: str, vlm_client=None) -> List[Dict[str, Any]]:
        """
        VLM 回退方案: 当 MinerU 因内存不足等原因失败时，
        使用 VLM 模型直接检测页面元素的位置和类型。
        
        返回格式与 MinerU 一致的元素列表。
        """
        if vlm_client is None:
            print("[VLM Fallback] 未提供 VLM 客户端，回退到静态 Mock")
            return self._analyze_mock(image_path)
        
        print("[VLM Fallback] MinerU 不可用，使用 VLM 进行版面检测...")
        
        try:
            # 读取图片
            with open(image_path, "rb") as f:
                base64_img = base64.b64encode(f.read()).decode('utf-8')
            
            img_width, img_height = 1920, 1080
            if PIL_AVAILABLE:
                try:
                    with Image.open(image_path) as img:
                        img_width, img_height = img.size
                except:
                    pass
            
            prompt = """你是一个专业的文档版面分析专家。请分析这张PPT幻灯片图片，检测其中所有视觉元素的位置和类型。

对于每个检测到的元素，请返回：
- type: 元素类型，必须是以下之一: Title, Text, Image, Table
- bbox: 边界框坐标 [x1, y1, x2, y2]，使用像素坐标（图片尺寸为 """ + f"{img_width}x{img_height}" + """）
- text: 如果是文本元素，提供文本内容；如果是图片/表格，留空字符串
- confidence: 置信度 0.0-1.0

请严格按照以下JSON格式返回，不要包含任何其他内容：
```json
[
  {"type": "Title", "bbox": [x1, y1, x2, y2], "text": "标题文本", "confidence": 0.95},
  {"type": "Image", "bbox": [x1, y1, x2, y2], "text": "", "confidence": 0.9}
]
```

注意：
1. bbox 坐标是 [左上角x, 左上角y, 右下角x, 右下角y]，单位为像素
2. 请检测所有可见元素，包括标题、正文、图表、表格、图片等
3. 不要遗漏任何重要元素"""
            
            # 获取模型名称（按 MULTIMODAL_PROVIDER 统一路由）
            model_name = None
            try:
                from ...config import get_config
                model_name = get_config().vlm_runtime_model_name
            except Exception:
                pass
            if not model_name:
                try:
                    # 兼容直接运行
                    import sys as _sys
                    _parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    if _parent not in _sys.path:
                        _sys.path.insert(0, _parent)
                    from config import get_config
                    model_name = get_config().vlm_runtime_model_name
                except Exception:
                    model_name = _fallback_vlm_model_from_env()
            
            response = vlm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                    ]}
                ],
                max_tokens=4096,
                temperature=0.0
            )
            
            raw_response = response.choices[0].message.content.strip()
            print(f"[VLM Fallback] VLM 原始返回长度: {len(raw_response)} 字符")
            
            # 解析 JSON（处理可能的 markdown 包装）
            json_str = raw_response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            elements = json.loads(json_str)
            
            if not isinstance(elements, list):
                elements = [elements]
            
            # 验证和清洗
            valid_elements = []
            valid_types = {"title", "text", "image", "table", "chart", "formula", "flowchart"}
            for elem in elements:
                e_type = elem.get("type", "Text")
                bbox = elem.get("bbox", [0, 0, 0, 0])
                
                # 类型标准化
                if e_type.lower() not in valid_types:
                    e_type = "Text"
                
                # bbox 验证
                if len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
                    x1, y1, x2, y2 = bbox
                    # 确保坐标在合理范围内
                    x1 = max(0, min(img_width, x1))
                    y1 = max(0, min(img_height, y1))
                    x2 = max(0, min(img_width, x2))
                    y2 = max(0, min(img_height, y2))
                    if x2 > x1 and y2 > y1:
                        valid_elements.append({
                            "type": e_type,
                            "bbox": [x1, y1, x2, y2],
                            "text": elem.get("text", ""),
                            "confidence": float(elem.get("confidence", 0.8))
                        })
            
            if valid_elements:
                print(f"[VLM Fallback] ✅ 检测到 {len(valid_elements)} 个元素")
                for e in valid_elements:
                    print(f"  - {e['type']}: bbox={e['bbox']}, text={e['text'][:30] if e['text'] else '(无)'}")
                return valid_elements
            else:
                print("[VLM Fallback] ⚠️ VLM 未返回有效元素，回退到静态 Mock")
                return self._analyze_mock(image_path)
                
        except Exception as e:
            print(f"[VLM Fallback] ❌ VLM 版面检测失败: {e}")
            import traceback
            traceback.print_exc()
            return self._analyze_mock(image_path)

    def _parse_mineru_json_output(self, raw_output: Any) -> Dict[str, Any]:
        """解析 JSON"""
        elements = []
        content_list = []
        page_info = None
        
        # 1. 尝试提取 page_info (坐标系的真理)
        if isinstance(raw_output, dict):
            # Magic-PDF 常见结构
            if "page_info" in raw_output:
                page_info = raw_output["page_info"]
            elif "image_info" in raw_output:
                page_info = raw_output["image_info"]
                
            # 提取内容列表
            if "content_list" in raw_output:
                content_list = raw_output["content_list"]
            elif "para_blocks" in raw_output:
                content_list = raw_output["para_blocks"]
            elif "layout_dets" in raw_output: # 某些版本字段
                content_list = raw_output["layout_dets"]
            else:
                # 兜底：遍历 value 找 list
                for k, v in raw_output.items():
                    if isinstance(v, list):
                        content_list = v
                        break
        elif isinstance(raw_output, list):
            content_list = raw_output

        # 1.1 兼容：某些版本 page_info/image_info 可能是 list（取第一个 dict）
        if isinstance(page_info, list):
            if page_info and isinstance(page_info[0], dict):
                page_info = page_info[0]
            else:
                page_info = None

        # 2. 标准化元素
        for item in content_list:
            elements.append(self._normalize_element(item))
            
        print(f"[MinerU] 解析得到 {len(elements)} 个元素")
        if isinstance(page_info, dict):
            print(f"[MinerU] 捕捉到模型坐标系: w={page_info.get('width')}, h={page_info.get('height')}")
        
        return {
            "elements": elements,
            "model_width": page_info.get("width") if isinstance(page_info, dict) else None,
            "model_height": page_info.get("height") if isinstance(page_info, dict) else None
        }

    def _normalize_element(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """标准化元素"""
        elem_type = item.get("type") or item.get("category") or "text"
        bbox = item.get("bbox") or [0,0,0,0]
        text = item.get("text") or item.get("content") or ""
        return {
            "type": elem_type,
            "bbox": bbox,
            "text": text,
            "confidence": 0.9
        }


# ============================================================================
# ROI 裁剪器
# ============================================================================

class ROICropper:
    """
    ROI (Region of Interest) 裁剪器
    
    从原图中裁剪出各元素的图片碎片，供后续专家节点使用
    """
    
    def __init__(self, output_dir: str = "roi_crops"):
        """
        初始化裁剪器
        
        Args:
            output_dir: ROI图片保存目录
        """
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    def crop_elements(self, 
                     image_path: str, 
                     elements: List[DetectedElement],
                     target_types: List[str] = None) -> List[ROIImage]:
        """
        裁剪指定类型的元素
        
        Args:
            image_path: 原图路径
            elements: 检测到的元素列表
            target_types: 需要裁剪的类型列表，None表示裁剪Image和Table
            
        Returns:
            ROI图片信息列表
        """
        if target_types is None:
            target_types = ["Image", "Table", "image", "table", "chart", "Chart"]
        
        roi_images = []
        
        abs_image_path = os.path.abspath(image_path)

        # 读取原图
        if CV2_AVAILABLE:
            # 处理中文路径问题：先读字节流再解码
            img = cv2.imdecode(np.fromfile(abs_image_path, dtype=np.uint8), -1)
            if img is None:
                print(f"[ROICropper] ❌ 无法读取图片: {abs_image_path}")
                return roi_images
            img_height, img_width = img.shape[:2]
        elif PIL_AVAILABLE:
            img = Image.open(abs_image_path)
            img_width, img_height = img.size
        else:
            return roi_images
        
        print(f"[ROICropper] 图片尺寸: {img_width}x{img_height}")

        for elem in elements:
            # 检查是否需要裁剪
            e_type = str(elem.original_type or "").lower()
            r_type = str(elem.refined_type or "").lower()
            if not any(t.lower() in e_type for t in target_types) and not any(t.lower() in r_type for t in target_types):
                continue
            
            # 获取像素坐标（从归一化坐标转换）
            bbox = elem.bbox
            ymin = int(bbox.ymin * img_height / 1000)
            xmin = int(bbox.xmin * img_width / 1000)
            ymax = int(bbox.ymax * img_height / 1000)
            xmax = int(bbox.xmax * img_width / 1000)
            
            # 边界保护
            ymin = max(0, ymin)
            xmin = max(0, xmin)
            ymax = min(img_height, ymax)
            xmax = min(img_width, xmax)
            
            # 无效区域检查
            if (xmax - xmin) < 10 or (ymax - ymin) < 10:
                print(f"[ROICropper] ⚠️ 忽略过小区域 {elem.element_id}: {xmax-xmin}x{ymax-ymin}")
                continue
            
            # 裁剪
            roi_path = os.path.join(self.output_dir, f"{elem.element_id}_roi.png")
            
            try:
                if CV2_AVAILABLE:
                    roi = img[ymin:ymax, xmin:xmax]
                    # cv2.imwrite 不支持中文路径，使用 imencode
                    is_success, buffer = cv2.imencode(".png", roi)
                    if is_success:
                        with open(roi_path, "wb") as f:
                            f.write(buffer)
                else:
                    roi = img.crop((xmin, ymin, xmax, ymax))
                    roi.save(roi_path)
                
                # 验证文件是否生成
                if os.path.exists(roi_path) and os.path.getsize(roi_path) > 0:
                    roi_images.append(ROIImage(
                        element_id=elem.element_id,
                        roi_path=roi_path,
                        original_type=elem.original_type,
                        bbox=elem.bbox
                    ))
                else:
                    print(f"[ROICropper] ❌ 文件生成失败: {roi_path}")

            except Exception as e:
                print(f"[ROICropper] 裁剪异常 {elem.element_id}: {e}")
        
        return roi_images


# ============================================================================
# VLM 类型清洗引擎
# ============================================================================

class TypeRefinementEngine:
    """
    VLM 类型清洗引擎
    
    使用 GPT-4o 等 VLM 模型判断元素的真实类型，
    修正 MinerU 将表格/图表误判为图片的问题
    """
    
    # 类型映射: VLM返回的中文 -> 标准类型
    TYPE_MAPPING = {
        "数据图表": "chart",
        "图表": "chart",
        "表格": "table",
        "纯图片": "image",
        "图片": "image",
        "公式": "formula",
        "流程图": "flowchart",
        "diagram": "diagram"
    }
    
    def __init__(self, vlm_client, use_gpu_branch: bool = False):
        """
        初始化类型清洗引擎
        
        Args:
            vlm_client: VLM客户端（GPT-4o 或 Qwen-VL）
            use_gpu_branch: 是否使用GPU分支（MinerU VLM版本）
        """
        self.vlm_client = vlm_client
        self.use_gpu_branch = use_gpu_branch
        
        # 获取模型名称
        try:
            from config import get_config
            self.model = get_config().vlm_runtime_model_name
        except Exception:
            self.model = _fallback_vlm_model_from_env()
    
    def refine_types(self, 
                    roi_images: List[ROIImage],
                    max_workers: int = 4) -> Dict[str, str]:
        """
        并行调用VLM清洗类型
        
        Args:
            roi_images: ROI图片列表
            max_workers: 最大并行数
            
        Returns:
            element_id -> refined_type 的映射
        """
        if not roi_images:
            return {}
        
        print(f"[TypeRefinement] 开始类型清洗，共 {len(roi_images)} 个元素")
        
        refined_types = {}
        
        # 并行处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._refine_single, roi): roi.element_id
                for roi in roi_images
            }
            
            for future in as_completed(futures):
                element_id = futures[future]
                try:
                    refined_type, raw_response = future.result()
                    refined_types[element_id] = {
                        "refined_type": refined_type,
                        "raw_response": raw_response
                    }
                except Exception as e:
                    print(f"[TypeRefinement] {element_id} 清洗失败: {e}")
                    refined_types[element_id] = {
                        "refined_type": None,
                        "raw_response": str(e)
                    }
        
        return refined_types
    
    def _refine_single(self, roi: ROIImage) -> Tuple[str, str]:
        """清洗单个元素的类型"""
        try:
            # 读取ROI图片并编码
            with open(roi.roi_path, "rb") as f:
                base64_img = base64.b64encode(f.read()).decode('utf-8')
            
            # 构造Prompt
            prompt = """这是一张 PPT 局部截图，请结合视觉特征判断其真实类型。

选项：[数据图表, 表格, 纯图片, 公式, 流程图]

判断依据：
- 数据图表：包含坐标轴、柱状/折线/饼图等数据可视化
- 表格：包含行列结构、单元格边框
- 纯图片：照片、插图、示意图（无数据结构）
- 公式：数学公式、化学式
- 流程图：包含箭头连接的流程/步骤图

请只返回选项中的一个词，不要其他内容。"""
            
            # 调用VLM
            response = self.vlm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                    ]}
                ],
                max_tokens=50,
                temperature=0.0
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            # 映射到标准类型
            refined_type = self.TYPE_MAPPING.get(raw_response, raw_response.lower())
            
            print(f"[TypeRefinement] {roi.element_id}: {roi.original_type} -> {refined_type}")
            
            return refined_type, raw_response
            
        except Exception as e:
            print(f"[TypeRefinement] VLM调用失败: {e}")
            return roi.original_type.lower(), str(e)


# ============================================================================
# Phase 1 总控制器
# ============================================================================

class Phase1_LayoutDetector:
    """
    Phase 1 版面检测与类型清洗 - 总控制器
    
    整合 MinerU 检测、ROI 裁剪、VLM 类型清洗
    """
    
    def __init__(self, 
                 vlm_client,
                 output_dir: str = "processing_artifacts",
                 use_mineru_vlm: bool = False):
        """
        初始化 Phase 1 控制器
        
        Args:
            vlm_client: VLM客户端
            output_dir: 输出目录
            use_mineru_vlm: 是否使用MinerU VLM版本（需要GPU）
        """
        self.vlm_client = vlm_client
        self.output_dir = output_dir
        self.use_mineru_vlm = use_mineru_vlm
        
        # 初始化组件
        self.mineru_client = MinerUClient(
            mode="vlm" if use_mineru_vlm else "standard",
            vlm_client=vlm_client
        )
        
        roi_output_dir = os.path.join(output_dir, "roi_crops")
        self.roi_cropper = ROICropper(output_dir=roi_output_dir)
        
        self.type_refiner = TypeRefinementEngine(vlm_client)
    
    def run(self, 
            image_path: str, 
            page_id: int,
            global_analysis: Optional[Dict[str, Any]] = None,
            precomputed_layout: Optional[Dict[str, Any]] = None) -> CleanedLayoutJSON:
        """
        执行 Phase 1 完整流程
        
        Args:
            image_path: PPT页面图片路径
            page_id: 页面ID
            global_analysis: Step1传入的全局分析结果（辅助参考）
            
        Returns:
            CleanedLayoutJSON: 包含所有元素精准BBox和修正后Type的JSON
        """
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"[Phase 1] 开始版面检测与类型清洗 (Page {page_id})")
        print(f"{'='*60}")
        
        # 确保图片路径是绝对路径
        image_path = os.path.abspath(image_path)

        # 为 MinerU 运行准备工作目录：把输入图片复制到本页 processing_artifacts 下
        # 这样 mineru_output / stdout / stderr 都会落在同一页目录里，便于对比官网结果
        mineru_work_dir = os.path.join(self.output_dir, "mineru_work")
        os.makedirs(mineru_work_dir, exist_ok=True)
        mineru_image_path = os.path.join(mineru_work_dir, os.path.basename(image_path))
        try:
            if (not os.path.exists(mineru_image_path)) or (os.path.getmtime(mineru_image_path) < os.path.getmtime(image_path)):
                shutil.copy2(image_path, mineru_image_path)
        except Exception as e:
            print(f"[Phase 1] 复制图片到 MinerU 工作目录失败，将直接使用原图: {e}")
            mineru_image_path = image_path
        
        # Step 1: MinerU 基础检测（支持全局预计算结果复用）
        print("\n[Phase 1.1] MinerU 版面分析...")
        mineru_debug_dir = os.path.join(self.output_dir, "mineru_debug")

        if isinstance(precomputed_layout, dict) and "elements" in precomputed_layout:
            print(f"[Phase 1.1] 命中全局预计算布局，跳过本页 MinerU 调用 (Page {page_id})")
            mineru_result = precomputed_layout
        else:
            # [Fix]: 获取包含元数据的完整结果
            mineru_result = self.mineru_client.analyze(mineru_image_path, debug_dir=mineru_debug_dir)
        
        # 兼容旧代码：如果 analyze 返回的是 list，说明是 VLM fallback 或 Mock
        if isinstance(mineru_result, list):
            raw_elements = mineru_result
            # VLM fallback 返回像素坐标，使用图片尺寸进行归一化
            model_w, model_h = self._get_image_size(image_path)
            is_vlm_fallback = True
        else:
            raw_elements = mineru_result.get("elements", [])
            model_w = mineru_result.get("model_width")
            model_h = mineru_result.get("model_height")
            is_vlm_fallback = False

        # 获取图片尺寸用于坐标归一化 (作为兜底)
        img_width, img_height = self._get_image_size(image_path)
        
        # 获取 PDF 的真实逻辑尺寸 (作为第二兜底)
        pdf_filename = os.path.splitext(os.path.basename(image_path))[0] + ".pdf"
        mineru_pdf_path = os.path.join(mineru_work_dir, pdf_filename)
        if os.path.exists(mineru_pdf_path):
            pdf_w, pdf_h = self._get_pdf_mediabox(mineru_pdf_path)
        else:
            pdf_w, pdf_h = 1382.4, 777.6

        # Step 2: 转换为标准格式并归一化坐标
        print("\n[Phase 1.2] 坐标归一化...")
        # [Fix]: 传入所有可能的尺寸参考，由转换函数决定信谁
        detected_elements = self._convert_to_detected_elements(
            raw_elements, 
            pdf_w=pdf_w, pdf_h=pdf_h,
            model_w=model_w, model_h=model_h
        )
        print(f"  检测到 {len(detected_elements)} 个元素")
        
        # Step 3: ROI 裁剪（针对 Image 和 Table）
        print("\n[Phase 1.3] ROI 裁剪...")
        roi_images = self.roi_cropper.crop_elements(
            image_path, 
            detected_elements,
            target_types=["Image", "Table", "image", "table"]
        )
        print(f"  裁剪了 {len(roi_images)} 个 ROI")
        
        # Step 4: VLM 类型清洗
        print("\n[Phase 1.4] VLM 类型清洗...")
        if roi_images:
            refined_types = self.type_refiner.refine_types(roi_images)
            
            # 更新元素类型
            for elem in detected_elements:
                if elem.element_id in refined_types:
                    result = refined_types[elem.element_id]
                    elem.refined_type = result.get("refined_type")
                    elem.vlm_refinement_response = result.get("raw_response")
            
            # 更新ROI的类型
            for roi in roi_images:
                if roi.element_id in refined_types:
                    roi.refined_type = refined_types[roi.element_id].get("refined_type")
        
        # Step 4.5: 全局 Type Correction 纠偏 (Strategy B) + 大框包小框清洗
        print("\n[Phase 1.4.5] 全局 Type Correction 纠偏...")
        try:
            from config import get_config
            cfg = get_config()
            if getattr(cfg, "enable_type_correction", True):
                from nodes.step2_locator import _correct_types_with_page_vlm, _canonical_dispatch_type
                from state import PageElement, BBox
                
                # Mock state for Type Correction
                class DummyGlobal: pass
                g_analysis = DummyGlobal()
                g_analysis.core_summary = global_analysis.get("core_summary", "") if global_analysis else ""
                g_analysis.section_title = global_analysis.get("section_title", "") if global_analysis else ""
                g_analysis.elements = []
                g_analysis.is_pure_text = False # conservative default
                
                class DummyState: pass
                d_state = DummyState()
                d_state.image_path = image_path
                d_state.global_analysis = g_analysis
                
                # Setup map to revert changes if needed
                mock_map = {}
                mock_elements = []
                for e in detected_elements:
                    pel = PageElement(
                        element_id=e.element_id,
                        type=e.refined_type or e.original_type,
                        description=e.ocr_text or "",
                        bbox=BBox(box_2d=e.bbox.box_2d)
                    )
                    mock_elements.append(pel)
                    mock_map[e.element_id] = e
                
                corrected_mock, corrected_count = _correct_types_with_page_vlm(d_state, mock_elements, self.vlm_client, cfg)
                
                print(f"  全局纠偏修改/确定了 {corrected_count} 个元素的类型")
                
                # Apply changes and drop missing elements
                valid_ids = {m.element_id: _canonical_dispatch_type(m.type) for m in corrected_mock}
                filtered_elements = []
                new_visual_elements = []
                
                for e in detected_elements:
                    if e.element_id in valid_ids:
                        new_type = valid_ids[e.element_id]
                        old_type = _canonical_dispatch_type(e.refined_type or e.original_type)
                        e.refined_type = new_type
                        filtered_elements.append(e)
                        
                        # Check if a text element was upgraded to visual, needing ROI crop
                        if old_type == "text" and new_type in {"chart", "image", "table"}:
                            new_visual_elements.append(e)
                
                detected_elements = filtered_elements
                
                # IF new visual elements were found, we MUST crop them so Phase 3 ChartExpert has ROIs
                if new_visual_elements:
                    print(f"  发现 {len(new_visual_elements)} 个由 Text 修正为视觉容器的元素，补充裁剪 ROI...")
                    new_rois = self.roi_cropper.crop_elements(
                        image_path,
                        new_visual_elements,
                        target_types=["chart", "image", "table", "Chart", "Image", "Table"]
                    )
                    if new_rois:
                        roi_images.extend(new_rois)
                        print(f"  成功补充裁剪 {len(new_rois)} 个 ROI")

        except Exception as e:
            print(f"[Phase 1] 全局 Type Correction 失败，跳过: {e}")
            import traceback
            traceback.print_exc()

        # Step 5: 辅助参考全局分析（如果提供）
        if global_analysis:
            self._merge_with_global_analysis(detected_elements, global_analysis)
        
        # 构建输出
        processing_time = int((time.time() - start_time) * 1000)
        
        result = CleanedLayoutJSON(
            page_id=page_id,
            image_path=image_path,
            elements=detected_elements,
            roi_images=roi_images,
            detection_source=(
                "fallback" if getattr(self.mineru_client, "last_run_used_fallback", False)
                else ("mineru_vlm" if self.use_mineru_vlm else "mineru_std")
            ),
            processing_time_ms=processing_time,
            mineru_debug_dir=getattr(self.mineru_client, "last_run_debug_dir", None),
            mineru_output_dir=getattr(self.mineru_client, "last_run_output_dir", None),
            mineru_cmd=(" ".join(getattr(self.mineru_client, "last_run_cmd", []) or [])) or None,
            mineru_returncode=getattr(self.mineru_client, "last_run_returncode", None),
            mineru_stdout_path=getattr(self.mineru_client, "last_run_stdout_path", None),
            mineru_stderr_path=getattr(self.mineru_client, "last_run_stderr_path", None),
            mineru_used_fallback=getattr(self.mineru_client, "last_run_used_fallback", False)
        )
        
        print(f"\n[Phase 1] 完成！耗时 {processing_time}ms")
        print(f"  - 检测元素: {len(detected_elements)}")
        print(f"  - ROI图片: {len(roi_images)}")
        
        # 输出元素摘要
        type_counts = {}
        for elem in detected_elements:
            t = elem.refined_type or elem.original_type
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"  - 类型分布: {type_counts}")
        
        return result
    
    def _get_image_size(self, image_path: str) -> Tuple[int, int]:
        """获取图片尺寸"""
        if PIL_AVAILABLE:
            with Image.open(image_path) as img:
                return img.size
        elif CV2_AVAILABLE:
            img = cv2.imread(image_path)
            if img is not None:
                return img.shape[1], img.shape[0]
        return 1920, 1080  # 默认
    
    def _get_pdf_mediabox(self, pdf_path: str) -> Tuple[float, float]:
        """
        [核心黑科技] 不依赖重型库，直接正则读取 PDF 的 MediaBox 尺寸
        这比猜测 DPI 准确 10000 倍，且耗时几乎为 0
        """
        try:
            with open(pdf_path, 'rb') as f:
                # 读取前 2KB 足够找到 MediaBox
                header = f.read(2048).decode('latin1', errors='ignore')
                
            # 匹配 /MediaBox [0 0 595.28 841.89] 格式
            match = re.search(r'/MediaBox\s*\[\s*([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s*\]', header)
            
            if match:
                x1, y1, x2, y2 = map(float, match.groups())
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                print(f"[PDF Debug] 读取到真实 PDF 尺寸: {width} x {height}")
                return width, height
            
            # 如果没找到，尝试在 MinerU 的输出中找线索，或者回退到默认
            # 这里的回退值应该和 image_to_pdf 的逻辑一致 (1920/100*72 = 1382.4)
            print("[PDF Debug] 未找到 MediaBox，使用默认推断尺寸")
            return 1382.4, 777.6 
            
        except Exception as e:
            print(f"[PDF Debug] 读取 PDF 尺寸失败: {e}")
            return 1382.4, 777.6
    
    def _convert_to_detected_elements(self, 
                                      raw_elements: List[Dict], 
                                      pdf_w: float, pdf_h: float,
                                      model_w: Optional[float] = None,
                                      model_h: Optional[float] = None) -> List[DetectedElement]:
        """转换MinerU输出为标准格式"""
        detected = []
        
        # 1. 确定归一化的分母 (Base Dimension)
        base_w, base_h = pdf_w, pdf_h
        
        # 策略 A: 优先使用模型 JSON 自带的尺寸 (最准确)
        if model_w and model_h:
            print(f"  [Coords] 使用模型元数据尺寸: {model_w}x{model_h}")
            base_w, base_h = model_w, model_h
        else:
            # 策略 B: 启发式推断
            # 扫描所有元素的坐标，看最大值分布
            max_x = 0
            max_y = 0
            for raw in raw_elements:
                bbox = raw.get("bbox", [0,0,0,0])
                if len(bbox) == 4:
                    max_x = max(max_x, bbox[2])
                    max_y = max(max_y, bbox[3])
            
            # 如果最大坐标在 900-1000 之间，且 PDF 宽度远大于 1000，说明是归一化坐标
            if 800 < max_x <= 1000 and pdf_w > 1200:
                print(f"  [Coords] 检测到 0-1000 归一化坐标 (MaxX={max_x}, PDFW={pdf_w})")
                base_w, base_h = 1000.0, 1000.0 # 假设高度也是归一化，或者按比例
                # 注意：MinerU 有时高度是真实的，只有宽度归一化，这里简化处理，通常 width 对了 height 也会对
            else:
                print(f"  [Coords] 使用 PDF MediaBox 尺寸: {pdf_w}x{pdf_h}")
        
        print(f"  [Debug] 归一化计算: Coord / ({base_w}x{base_h}) * 1000")
        
        for i, raw in enumerate(raw_elements):
            bbox_px = raw.get("bbox", [0, 0, 0, 0])
            if len(bbox_px) == 4:
                x1, y1, x2, y2 = bbox_px
            else:
                continue
            
            # 执行归一化
            ymin = int(y1 * 1000 / base_h)
            xmin = int(x1 * 1000 / base_w)
            ymax = int(y2 * 1000 / base_h)
            xmax = int(x2 * 1000 / base_w)
            
            # 边界保护
            ymin = max(0, min(1000, ymin))
            xmin = max(0, min(1000, xmin))
            ymax = max(0, min(1000, ymax))
            xmax = max(0, min(1000, xmax))
            
            element = DetectedElement(
                element_id=f"elem_{i:03d}",
                original_type=raw.get("type", "Unknown"),
                bbox=ElementBBox(box_2d=[ymin, xmin, ymax, xmax]),
                ocr_text=raw.get("text", ""),
                confidence=raw.get("confidence", 0.5)
            )
            
            detected.append(element)
        
        return detected
    
    def _merge_with_global_analysis(self, 
                                    elements: List[DetectedElement], 
                                    global_analysis: Dict[str, Any]):
        """合并Step1的全局分析结果（辅助参考）"""
        # 可以根据全局分析中识别的元素类型辅助校正
        # 例如：如果全局分析说有"生存曲线"，可以帮助确认某个Image是chart
        ga_elements = global_analysis.get("elements", [])
        
        for ga_elem in ga_elements:
            ga_type = ga_elem.get("type", "").lower()
            ga_desc = ga_elem.get("description", "")
            
            # 简单的匹配策略：根据描述中的关键词
            keywords_to_type = {
                "曲线": "chart",
                "柱状": "chart",
                "折线": "chart",
                "饼图": "chart",
                "表格": "table",
                "公式": "formula",
                "流程": "flowchart"
            }
            
            for kw, target_type in keywords_to_type.items():
                if kw in ga_desc:
                    # 找到对应的元素并辅助判断
                    for elem in elements:
                        if elem.refined_type is None and elem.original_type.lower() == "image":
                            # 可以考虑更精细的匹配逻辑
                            pass
