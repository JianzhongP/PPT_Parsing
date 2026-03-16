"""PPT解析工作流的入口文件和示例使用"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

from config import get_config, init_api_clients
from graph import build_ppt_workflow
from state import PPTWorkflowState
from debug_logger import get_debug_logger, reset_debug_logger


def _rewrite_markdown_image_paths(markdown: str, *, artifacts_dir: str | None, output_md_path: Path, project_root: Path) -> str:
    """Rewrite local image paths in markdown to be relative to the merged output file.

    - For complex pipeline markdown, images are usually like: roi_crops/xxx.png
      and should be resolved relative to processing_artifacts/page_XXX.
    - For other markdown, images might be relative to repo root.
    """
    if not markdown:
        return ""

    img_pat = re.compile(r"(!\[[^\]]*\]\()\s*([^\)\s]+)([^\)]*)\)")
    out_dir = output_md_path.parent
    artifacts_path = (project_root / artifacts_dir) if artifacts_dir else None

    def _is_url(p: str) -> bool:
        return p.startswith("http://") or p.startswith("https://") or p.startswith("data:")

    def repl(m: re.Match) -> str:
        prefix, raw_path, suffix = m.group(1), m.group(2), m.group(3)
        path = raw_path.strip().strip('"').strip("'")
        if not path or _is_url(path):
            return m.group(0)

        # If absolute, just relativize to output.md
        candidate_paths: list[Path] = []
        p = Path(path)
        if p.is_absolute():
            candidate_paths.append(p)
        else:
            # Try artifacts_dir-relative first
            if artifacts_path:
                candidate_paths.append(artifacts_path / path)
            # Then try project-root-relative
            candidate_paths.append(project_root / path)

        target = None
        for c in candidate_paths:
            if c.exists():
                target = c
                break
        if target is None:
            return m.group(0)

        rel = os.path.relpath(str(target), str(out_dir))
        rel = rel.replace(os.sep, "/")
        return f"{prefix}{rel}{suffix})"

    return img_pat.sub(repl, markdown)


def _selected_markdown_from_output(output: Any) -> str:
    """优先返回人工修订后的 markdown，否则返回自动生成 markdown。"""
    reviewed = getattr(output, "reviewed_markdown_content", None)
    if reviewed and str(reviewed).strip():
        return str(reviewed)
    return getattr(output, "markdown_content", "") or ""


def _selected_summary_from_output(output: Any) -> str:
    """优先返回人工修订后的 summary，否则返回自动 summary。"""
    reviewed = getattr(output, "reviewed_summary", None)
    if reviewed and str(reviewed).strip():
        return str(reviewed)
    return getattr(output, "summary", "") or ""


def _selected_elements_from_output(output: Any) -> Dict[str, Any]:
    """优先返回人工修订后的 elements，否则返回自动 elements。"""
    reviewed = getattr(output, "reviewed_elements", None)
    if isinstance(reviewed, dict) and reviewed:
        return reviewed
    return getattr(output, "elements", {}) or {}


def _detect_review_signals(output: Any, low_conf_threshold: float = 0.7) -> Tuple[bool, List[str], float]:
    """根据规则判定是否需要人工审核。"""
    reasons: List[str] = []
    score = 0.0

    elements = getattr(output, "elements", {}) or {}
    markdown = (getattr(output, "markdown_content", "") or "").strip()
    summary = (getattr(output, "summary", "") or "").strip()

    visual_count = 0
    low_conf_count = 0
    for _, value in elements.items():
        if not isinstance(value, dict):
            continue
        et = str(value.get("type", "")).lower()
        if et in {"chart", "table", "diagram", "flowchart", "image"}:
            visual_count += 1
        conf = value.get("confidence")
        try:
            conf_val = float(conf)
            if conf_val < low_conf_threshold:
                low_conf_count += 1
        except Exception:
            continue

    if low_conf_count > 0:
        reasons.append(f"存在 {low_conf_count} 个低置信度元素(conf<{low_conf_threshold:.2f})")
        score += min(0.5, 0.2 * low_conf_count)

    uncertain_patterns = [
        r"可能", r"疑似", r"无法(?:完全)?判断", r"不确定", r"unknown", r"未识别", r"待确认"
    ]
    uncertain_hits = sum(1 for p in uncertain_patterns if re.search(p, markdown, flags=re.IGNORECASE))
    if uncertain_hits > 0:
        reasons.append("输出包含不确定性措辞")
        score += 0.2

    if visual_count >= 4:
        reasons.append(f"可视元素较多({visual_count})，页面可能较复杂")
        score += 0.15

    if visual_count > 0 and len(summary) < 24:
        reasons.append("页面有可视元素但摘要过短")
        score += 0.2

    if visual_count > 0 and len(markdown) < 120:
        reasons.append("页面有可视元素但详细描述偏少")
        score += 0.15

    needs_review = score >= 0.25 or len(reasons) > 0
    return needs_review, reasons, round(min(score, 1.0), 3)

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

class PPTParsingPipeline:
    """PPT解析管道
    
    使用示例：
    ```python
    pipeline = PPTParsingPipeline("test.pptx", enable_debug=True)
    results = pipeline.run()
    pipeline.save_results("output.json")
    ```
    """
    
    def __init__(self, ppt_path: str, max_concurrent_pages: int = 3, enable_debug: bool = True):
        """
        初始化PPT解析管道
        
        参数：
            ppt_path: PPT文件路径
            max_concurrent_pages: 最大并发处理页面数
            enable_debug: 是否启用详细的调试日志
        """
        self.ppt_path = ppt_path
        self.max_concurrent_pages = max_concurrent_pages
        self.enable_debug = enable_debug
        
        # 初始化调试logger
        if enable_debug:
            reset_debug_logger()
            self.debug_logger = get_debug_logger(output_dir="debug_logs")
        else:
            self.debug_logger = None
        
        # 初始化配置和客户端
        self.config = get_config()
        # self.llm_client = init_api_clients()

        self.llm_client, self.vlm_client = init_api_clients()
        
        # 构建工作流
        self.workflow = build_ppt_workflow(self.llm_client, self.vlm_client, debug_logger=self.debug_logger)
        
        # 结果存储
        self.results: Dict[int, Any] = {}
    
    def run(self) -> Dict[int, Any]:
        """
        运行PPT解析工作流
        
        返回：
            {page_index: final_output} 的字典
        """
        print("\n" + "="*70)
        print(f"开始PPT解析工作流: {self.ppt_path}")
        print("="*70)
        
        if self.debug_logger:
            self.debug_logger.logger.info(f"启动工作流: {self.ppt_path}")
        
        # 初始化状态
        initial_state = PPTWorkflowState(
            ppt_path=self.ppt_path,
            max_concurrent_pages=self.max_concurrent_pages,
            page_queue=[],  # 将在 ingestion 节点填充
            total_pages=0,
            chapter_info={}
        )
        
        try:
            import uuid
            # 运行工作流
            final_state = self.workflow.invoke(
                initial_state,
                config={"recursion_limit": 100,
                        "configurable": {
                        "thread_id": str(uuid.uuid4())
                        }}
            )
            
            # 收集结果
            self.results = final_state.get("completed_outputs", {})
            
            print("\n" + "="*70)
            print(f"[OK] 工作流完成！处理 {final_state.get('completed_pages', 0)} 页")
            print("="*70)
            
            # 记录工作流总结
            if self.debug_logger:
                summary = {
                    "total_pages": final_state.get('completed_pages', 0),
                    "ppt_file": self.ppt_path,
                    "status": "completed",
                    "timestamp": str(__import__('datetime').datetime.now())
                }
                self.debug_logger.log_workflow_summary(summary)
                self.debug_logger.save()
            
            return self.results
        
        except Exception as e:
            print(f"\n[ERROR] 工作流执行失败: {e}")
            if self.debug_logger:
                self.debug_logger.log_error(0, "workflow", e, {"ppt_path": self.ppt_path})
                self.debug_logger.save()
            raise
    
    def save_results(self, output_path: str = None, output_dir: str = None):
        """
        保存解析结果
        
        参数：
            output_path: 单个JSON文件路径（不推荐）
            output_dir: 输出目录（每页一个JSON文件，推荐）
        """
        if not output_dir:
            output_dir = self.config.output_dir
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # 保存每一页的结果
        for page_idx, output in self.results.items():
            output_file = Path(output_dir) / f"page_{page_idx:03d}.json"
            
            # 将Pydantic模型转换为dict
            if hasattr(output, 'model_dump'):
                # Pydantic V2
                output_dict = output.model_dump()
            elif hasattr(output, 'dict'):
                output_dict = output.dict()
            else:
                output_dict = output
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_dict, f, ensure_ascii=False, indent=2)
            
            print(f"[OK] 保存: {output_file}")
        
        # 生成总结报告
        self._generate_summary_report(output_dir)
    
    def _generate_summary_report(self, output_dir: str):
        """生成总结报告"""
        summary = {
            "ppt_file": self.ppt_path,
            "total_pages": len(self.results),
            "pages": {}
        }
        
        for page_idx, output in self.results.items():
            summary["pages"][page_idx] = {
                "section_title": output.section_title if hasattr(output, 'section_title') else "Unknown",
                "status": "completed"
            }
        
        summary_file = Path(output_dir) / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"[OK] 总结报告: {summary_file}")
    
    def get_page_result(self, page_index: int) -> Any:
        """获取单页的解析结果"""
        return self.results.get(page_index)

    def prepare_human_review(
        self,
        review_output_dir: str | None = None,
        low_conf_threshold: float = 0.7,
        force: bool = False,
    ) -> Dict[str, Any]:
        """生成人工审核任务（JSON + Markdown）。

        返回：
            {
              "review_required_pages": int,
              "review_tasks_json": str,
              "review_queue_md": str,
              "total_pages": int
            }
        """
        if not self.results:
            raise ValueError("当前无解析结果，请先运行 run()")

        if not review_output_dir:
            review_output_dir = self.config.output_dir

        out_dir = Path(review_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        review_tasks: List[Dict[str, Any]] = []
        for page_idx in sorted(self.results.keys()):
            output = self.results[page_idx]

            existing_status = getattr(output, "review_status", "not_required")
            if existing_status == "reviewed" and not force:
                continue

            needs_review, reasons, review_score = _detect_review_signals(
                output,
                low_conf_threshold=low_conf_threshold,
            )

            output.needs_review = bool(needs_review)
            output.review_reasons = reasons
            if needs_review:
                output.review_status = "pending"
            elif output.review_status != "reviewed":
                output.review_status = "not_required"

            if not needs_review:
                continue

            review_tasks.append({
                "page_index": page_idx,
                "page_number": page_idx + 1,
                "section_title": getattr(output, "section_title", ""),
                "review_score": review_score,
                "reasons": reasons,
                "auto_summary": getattr(output, "summary", "") or "",
                "auto_markdown": getattr(output, "markdown_content", "") or "",
                "auto_elements": getattr(output, "elements", {}) or {},
                "review_status": "pending",
                "review_notes": "",
                "reviewed_summary": "",
                "reviewed_markdown_content": "",
                "reviewed_elements": {},
            })

        review_tasks_json = out_dir / "review_tasks.json"
        with open(review_tasks_json, "w", encoding="utf-8") as f:
            json.dump(review_tasks, f, ensure_ascii=False, indent=2)

        review_queue_md = out_dir / "review_queue.md"
        lines: List[str] = []
        lines.append("# 人工审核队列")
        lines.append("")
        lines.append("说明：请根据每个任务补充修订内容，并将结果保存到 review_results.json。")
        lines.append("")
        lines.append(f"待审核页面数：{len(review_tasks)} / 总页数：{len(self.results)}")
        lines.append("")

        for item in review_tasks:
            lines.append(f"## Page {item['page_number']} (page_index={item['page_index']}) - {item['section_title']}")
            lines.append("")
            lines.append(f"- 审核评分：{item['review_score']}")
            lines.append(f"- 触发原因：{'；'.join(item['reasons'])}")
            lines.append("")
            lines.append("### 自动摘要")
            lines.append(item["auto_summary"])
            lines.append("")
            lines.append("### 自动输出(节选)")
            snippet = (item["auto_markdown"] or "")[:1200]
            lines.append(snippet if snippet else "(空)")
            lines.append("")
            lines.append("### 人工修订建议")
            lines.append("- 在 review_results.json 中填写 reviewed_summary / reviewed_markdown_content / reviewed_elements")
            lines.append("- review_status 设置为 reviewed")
            lines.append("")
            lines.append("---")
            lines.append("")

        with open(review_queue_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # 提供空模板，方便人工直接填写
        review_results_json = out_dir / "review_results.json"
        if not review_results_json.exists() or force:
            result_template = [
                {
                    "page_index": t["page_index"],
                    "review_status": "pending",
                    "review_notes": "",
                    "reviewed_summary": "",
                    "reviewed_markdown_content": "",
                    "reviewed_elements": {}
                }
                for t in review_tasks
            ]
            with open(review_results_json, "w", encoding="utf-8") as f:
                json.dump(result_template, f, ensure_ascii=False, indent=2)

        print(f"[HITL] 已生成审核任务: {review_tasks_json}")
        print(f"[HITL] 已生成审核说明: {review_queue_md}")
        print(f"[HITL] 待填写审核结果: {review_results_json}")

        return {
            "review_required_pages": len(review_tasks),
            "review_tasks_json": str(review_tasks_json),
            "review_queue_md": str(review_queue_md),
            "review_results_json": str(review_results_json),
            "total_pages": len(self.results),
        }

    def apply_human_review(self, review_results_path: str) -> Dict[str, Any]:
        """合并人工审核结果到内存结果集中。"""
        if not self.results:
            raise ValueError("当前无解析结果，请先运行 run()")

        review_file = Path(review_results_path)
        if not review_file.exists():
            raise FileNotFoundError(f"审核结果文件不存在: {review_results_path}")

        with open(review_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, list):
            raise ValueError("review_results.json 格式错误：应为 list")

        updated_pages = 0
        skipped_pages = 0
        for item in payload:
            if not isinstance(item, dict):
                skipped_pages += 1
                continue

            page_idx = item.get("page_index")
            if not isinstance(page_idx, int) or page_idx not in self.results:
                skipped_pages += 1
                continue

            output = self.results[page_idx]
            status = str(item.get("review_status", "pending")).strip().lower()
            if status not in {"pending", "reviewed", "not_required"}:
                status = "pending"

            output.review_status = status
            output.review_notes = str(item.get("review_notes", "") or "")

            reviewed_summary = item.get("reviewed_summary")
            if isinstance(reviewed_summary, str) and reviewed_summary.strip():
                output.reviewed_summary = reviewed_summary.strip()

            reviewed_md = item.get("reviewed_markdown_content")
            if isinstance(reviewed_md, str) and reviewed_md.strip():
                output.reviewed_markdown_content = reviewed_md.strip()

            reviewed_elements = item.get("reviewed_elements")
            if isinstance(reviewed_elements, dict) and reviewed_elements:
                output.reviewed_elements = reviewed_elements

            if output.review_status == "reviewed":
                output.needs_review = False

            updated_pages += 1

        return {
            "updated_pages": updated_pages,
            "skipped_records": skipped_pages,
            "total_pages": len(self.results),
        }

    def apply_human_review_if_available(self, review_output_dir: str | None = None) -> Dict[str, Any]:
        """如果 review_results.json 中有 reviewed 记录，则自动合并。"""
        if not review_output_dir:
            review_output_dir = self.config.output_dir

        review_results_json = Path(review_output_dir) / "review_results.json"
        if not review_results_json.exists():
            return {"applied": False, "reason": "review_results.json 不存在"}

        try:
            with open(review_results_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            return {"applied": False, "reason": f"读取 review_results.json 失败: {exc}"}

        if not isinstance(payload, list):
            return {"applied": False, "reason": "review_results.json 格式非法"}

        has_reviewed = False
        for item in payload:
            if not isinstance(item, dict):
                continue
            status = str(item.get("review_status", "")).strip().lower()
            if status == "reviewed":
                has_reviewed = True
                break

        if not has_reviewed:
            return {"applied": False, "reason": "暂无 reviewed 记录"}

        merged = self.apply_human_review(str(review_results_json))
        merged["applied"] = True
        merged["review_results_json"] = str(review_results_json)
        return merged
    
    def export_markdown(self, output_path: str = "output.md"):
        """
        导出所有结果为单个Markdown文件
        
        按照PPT章节、页号、标题进行组织
        """
        lines = []
        lines.append("# PPT解析结果")
        lines.append("")

        output_md_path = Path(output_path)
        project_root = Path(__file__).parent.resolve()
        
        for page_idx in sorted(self.results.keys()):
            output = self.results[page_idx]

            # Always add a page header so merged markdown is easy to navigate.
            # Avoid duplicating if the page content already starts with a page header.
            page_no = page_idx + 1
            page_title = None
            if hasattr(output, "section_title") and getattr(output, "section_title", None):
                page_title = str(output.section_title).strip()
            header = f"## 第 {page_no} 页" + (f"：{page_title}" if page_title else "")
            
            if hasattr(output, 'markdown_content'):
                artifacts_dir = None
                try:
                    artifacts_dir = (output.slice_reference or {}).get("artifacts_dir")
                except Exception:
                    artifacts_dir = None

                md = _selected_markdown_from_output(output)
                md_stripped = (md or "").lstrip()
                has_page_header = bool(re.match(r"^##\s*(第\s*\d+\s*页|页面\s*\d+)\b", md_stripped))
                if not has_page_header:
                    lines.append(header)
                    lines.append("")
                md = _rewrite_markdown_image_paths(
                    md,
                    artifacts_dir=artifacts_dir,
                    output_md_path=output_md_path,
                    project_root=project_root,
                )
                lines.append(md)
            else:
                # 降级处理
                lines.append(header)
            
            lines.append("")
            lines.append("---")
            lines.append("")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        print(f"[OK] Markdown导出: {output_path}")


def main():
    """主函数示例"""
    # 配置PPT文件路径
    ppt_path = "FH1701表格.pptx"  # 修改为实际的PPT文件路径
    
    if not Path(ppt_path).exists():
        print(f"错误: PPT文件不存在 - {ppt_path}")
        return
    
    # 创建解析管道（启用调试日志）
    pipeline = PPTParsingPipeline(
        ppt_path=ppt_path,
        max_concurrent_pages=3,
        enable_debug=True  # 启用详细调试日志
    )
    
    # 运行解析
    results = pipeline.run()
    
    # 保存结果
    pipeline.save_results()

    # 生成人工审核任务（复杂图表/低置信度页面）
    hitl_meta = pipeline.prepare_human_review(review_output_dir="ppt_parsing_output")
    print(f"[HITL] 待人工审核页面: {hitl_meta['review_required_pages']}")

    auto_merge = pipeline.apply_human_review_if_available(review_output_dir="ppt_parsing_output")
    if auto_merge.get("applied"):
        print(f"[HITL] 已自动合并人工审核结果: {auto_merge.get('review_results_json')}")
        print(f"[HITL] 已更新页面数: {auto_merge.get('updated_pages', 0)}")
    else:
        print(f"[HITL] 尚未合并人工审核结果: {auto_merge.get('reason')}")
    
    # 导出为Markdown
    pipeline.export_markdown("ppt_parsing_output/output.md")
    
    # 打印样本结果
    if results:
        first_page_idx = list(results.keys())[0]
        first_result = results[first_page_idx]
        
        print("\n" + "="*70)
        print("[Sample Results] Page " + str(first_page_idx) + ":")
        print("="*70)
        # 简化输出以避免编码问题
        print("[Note] Results saved to: ppt_parsing_output/")
        print("[Note] Debug logs saved to: debug_logs/")


if __name__ == "__main__":
    main()
