"""PPT解析工作流的入口文件和示例使用"""

import json
import html
import os
import re
import sys
import xml.etree.ElementTree as ET
import difflib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from config import get_config, init_api_clients
from graph import build_ppt_workflow
from state import PPTWorkflowState
from debug_logger import get_debug_logger, reset_debug_logger


def _sanitize_name_for_path(name: str) -> str:
    """将任意文件名清洗为可用于目录名的安全片段。"""
    s = str(name or "").strip()
    if not s:
        return "default"
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\-\u4e00-\u9fff]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80] or "default"


def _build_run_directories(ppt_path: str) -> Tuple[str, str, str]:
    """根据 ppt_path 生成独立目录名（processing_artifacts/output/layout_cache）。"""
    stem = _sanitize_name_for_path(Path(ppt_path).stem)
    processing_dir = f"processing_artifacts_{stem}"
    output_dir = f"ppt_parsing_output_{stem}"
    layout_cache_dir = f"{processing_dir}/layout_cache"
    return processing_dir, output_dir, layout_cache_dir


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


def _split_merged_slides_sections(markdown: str) -> List[Dict[str, str]]:
    """将 merged_slides.md 按一级标题切分为页面段落。"""
    text = str(markdown or "")
    if not text.strip():
        return []

    sections: List[Dict[str, str]] = []
    heading_pattern = re.compile(r"^#\s+(.+?)\s*$", flags=re.MULTILINE)
    matches = list(heading_pattern.finditer(text))
    if not matches:
        return [{"title": "", "content": text.strip()}]

    for idx, match in enumerate(matches):
        title = (match.group(1) or "").strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        sections.append({"title": title, "content": content})

    return sections


def _html_table_to_markdown(table_html: str) -> str:
    """将 HTML table 转换为 markdown table（简化版，足够对齐用途）。"""
    raw = str(table_html or "").strip()
    if not raw:
        return ""

    cleaned = html.unescape(raw)
    cleaned = re.sub(r"<br\s*/?>", " ", cleaned, flags=re.IGNORECASE)

    table = None
    try:
        root = ET.fromstring(f"<root>{cleaned}</root>")
        table = root.find(".//table")
    except Exception:
        table = None

    if table is None:
        return ""

    rows: List[List[str]] = []
    for tr in table.findall(".//tr"):
        cells = tr.findall("./td") + tr.findall("./th")
        row = ["".join((c.itertext() or [])).strip() for c in cells]
        if row:
            rows.append(row)

    if not rows:
        return ""

    width = max(len(r) for r in rows)
    norm_rows = [r + [""] * (width - len(r)) for r in rows]
    head = norm_rows[0]
    body = norm_rows[1:]

    out = [
        "| " + " | ".join(head) + " |",
        "|" + "|".join([" --- " for _ in range(width)]) + "|",
    ]
    for r in body:
        out.append("| " + " | ".join(r) + " |")

    return "\n".join(out).strip()


def _build_mineru_pages_from_content_list(content_list: List[Dict[str, Any]]) -> Dict[int, Dict[str, str]]:
    """基于 merged_slides_content_list.json 的 page_idx 重建逐页内容。"""
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for item in content_list:
        if not isinstance(item, dict):
            continue
        page_idx = item.get("page_idx")
        if not isinstance(page_idx, int):
            continue
        grouped.setdefault(page_idx, []).append(item)

    pages: Dict[int, Dict[str, str]] = {}
    for page_idx, items in grouped.items():
        title = ""
        lines: List[str] = []

        for item in items:
            typ = str(item.get("type", "") or "").strip().lower()
            if typ == "text":
                text = str(item.get("text", "") or "").strip()
                if not text:
                    continue
                if not title and int(item.get("text_level", 0) or 0) == 1:
                    title = text.replace("\n", " ").strip()
                lines.append(text)
            elif typ == "table":
                table_html = str(item.get("table_body", "") or "").strip()
                md_table = _html_table_to_markdown(table_html)
                if md_table:
                    lines.append(md_table)

        page_content = "\n\n".join([ln for ln in lines if str(ln).strip()]).strip()
        pages[page_idx] = {
            "title": title,
            "content": page_content,
        }

    return pages


def _render_labeled_mineru_markdown(pages: Dict[int, Dict[str, str]]) -> str:
    """输出显式页标记的 merged markdown，便于人工核对。"""
    out: List[str] = []
    for page_idx in sorted(pages.keys()):
        sec = pages[page_idx]
        title = str(sec.get("title", "") or "").strip()
        header = f"## 第 {page_idx + 1} 页" + (f"：{title}" if title else "")
        out.append(header)
        out.append("")
        out.append(str(sec.get("content", "") or "").strip())
        out.append("")
        out.append("---")
        out.append("")
    return "\n".join(out).strip()


def _extract_markdown_tables(markdown: str) -> List[str]:
    """提取 markdown 中的表格块（至少包含表头+分隔行）。"""
    lines = str(markdown or "").splitlines()
    tables: List[str] = []
    cur: List[str] = []

    def _flush() -> None:
        nonlocal cur
        if len(cur) >= 2:
            sep_like = any(re.match(r"^\s*\|?\s*[:\-\s\|]+\|?\s*$", ln) for ln in cur)
            if sep_like:
                table = "\n".join([ln.rstrip() for ln in cur]).strip()
                if table:
                    tables.append(table)
        cur = []

    for line in lines:
        if "|" in line:
            cur.append(line)
        else:
            _flush()
    _flush()
    return tables


def _normalize_table_for_compare(table_md: str) -> str:
    s = str(table_md or "").strip().lower()
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"\s+", "", s)
    s = s.replace("，", ",").replace("：", ":")
    return s


def _replace_first_table(markdown: str, replacement_table: str) -> str:
    """用 replacement_table 替换 markdown 第一个表格；若不存在则追加到末尾。"""
    text = str(markdown or "")
    new_table = str(replacement_table or "").strip()
    if not new_table:
        return text

    lines = text.splitlines()
    blocks: List[Tuple[int, int]] = []
    start = -1

    def _is_sep_line(line: str) -> bool:
        return bool(re.match(r"^\s*\|?\s*[:\-\s\|]+\|?\s*$", line))

    for i, line in enumerate(lines):
        if "|" in line:
            if start < 0:
                start = i
        else:
            if start >= 0:
                chunk = lines[start:i]
                if len(chunk) >= 2 and any(_is_sep_line(ln) for ln in chunk):
                    blocks.append((start, i))
                start = -1
    if start >= 0:
        chunk = lines[start:len(lines)]
        if len(chunk) >= 2 and any(_is_sep_line(ln) for ln in chunk):
            blocks.append((start, len(lines)))

    if blocks:
        s, e = blocks[0]
        merged_lines = lines[:s] + [new_table] + lines[e:]
        return "\n".join(merged_lines).strip()

    out = text.rstrip()
    if out:
        out += "\n\n"
    out += new_table
    return out.strip()


def _find_first_table_range(lines: List[str]) -> Optional[Tuple[int, int]]:
    start = -1

    def _is_sep_line(line: str) -> bool:
        return bool(re.match(r"^\s*\|?\s*[:\-\s\|]+\|?\s*$", line))

    for i, line in enumerate(lines):
        if "|" in line:
            if start < 0:
                start = i
        else:
            if start >= 0:
                chunk = lines[start:i]
                if len(chunk) >= 2 and any(_is_sep_line(ln) for ln in chunk):
                    return (start, i)
                start = -1

    if start >= 0:
        chunk = lines[start:len(lines)]
        if len(chunk) >= 2 and any(_is_sep_line(ln) for ln in chunk):
            return (start, len(lines))
    return None


def _table_analysis_presence(markdown: str) -> Dict[str, bool]:
    """检查首个表格附近是否已有洞察/关键数据块。"""
    lines = str(markdown or "").splitlines()
    table_range = _find_first_table_range(lines)
    if not table_range:
        return {"has_table": False, "has_insight": False, "has_key_data": False}

    s, _ = table_range
    window = "\n".join(lines[max(0, s - 28):s])
    has_insight = bool(re.search(r">\s*\*\*洞察\*\*\s*[:：]", window))
    has_key_data = bool(re.search(r"\*\*关键数据\s*[:：]\*\*", window))
    return {"has_table": True, "has_insight": has_insight, "has_key_data": has_key_data}


def _inject_table_analysis_block(markdown: str, insight: str, key_values: Dict[str, str]) -> str:
    """在首个表格前注入洞察和关键数据（仅补缺失项）。"""
    text = str(markdown or "")
    lines = text.splitlines()
    table_range = _find_first_table_range(lines)
    if not table_range:
        return text

    s, _ = table_range
    presence = _table_analysis_presence(text)
    blocks: List[str] = []

    if (not presence.get("has_insight")) and str(insight or "").strip():
        blocks.append(f"> **洞察**: {str(insight).strip()}")
        blocks.append("")

    if (not presence.get("has_key_data")) and isinstance(key_values, dict) and key_values:
        blocks.append("**关键数据:**")
        for k, v in list(key_values.items())[:8]:
            ks = str(k or "").strip()
            vs = str(v or "").strip()
            if ks and vs:
                blocks.append(f"- {ks}: {vs}")
        blocks.append("")

    if not blocks:
        return text

    merged_lines = lines[:s] + blocks + lines[s:]
    return "\n".join(merged_lines).strip()


def _append_missing_texts(markdown: str, spans: List[str], mode: str = "after_last_table") -> str:
    """将缺失文本追加到页面 markdown，默认插入到最后一个表格之后。"""
    base = str(markdown or "")
    add_lines = [str(x).strip() for x in spans if str(x or "").strip()]
    if not add_lines:
        return base

    existing_norm = re.sub(r"\s+", "", base)
    deduped: List[str] = []
    for line in add_lines:
        norm = re.sub(r"\s+", "", line)
        if norm and norm not in existing_norm:
            deduped.append(line)
    if not deduped:
        return base

    bullet_block = "\n".join([f"- {x}" for x in deduped])
    insert_block = f"\n\n**补充信息:**\n{bullet_block}\n"

    if mode == "end":
        return (base.rstrip() + insert_block).strip()

    lines = base.splitlines()
    table_ranges: List[Tuple[int, int]] = []
    start = -1

    def _is_sep_line(line: str) -> bool:
        return bool(re.match(r"^\s*\|?\s*[:\-\s\|]+\|?\s*$", line))

    for i, line in enumerate(lines):
        if "|" in line:
            if start < 0:
                start = i
        else:
            if start >= 0:
                chunk = lines[start:i]
                if len(chunk) >= 2 and any(_is_sep_line(ln) for ln in chunk):
                    table_ranges.append((start, i))
                start = -1
    if start >= 0:
        chunk = lines[start:len(lines)]
        if len(chunk) >= 2 and any(_is_sep_line(ln) for ln in chunk):
            table_ranges.append((start, len(lines)))

    if not table_ranges:
        return (base.rstrip() + insert_block).strip()

    _, end_idx = table_ranges[-1]
    merged_lines = lines[:end_idx] + ["", "**补充信息:**"] + [f"- {x}" for x in deduped] + lines[end_idx:]
    return "\n".join(merged_lines).strip()


def _normalize_for_fuzzy_compare(text: str) -> str:
    s = str(text or "").strip().lower()
    if not s:
        return ""
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("：", ":").replace("，", ",").replace("；", ";")
    s = s.replace("•", " ").replace("√", " ")
    s = re.sub(r"^[-*+\s]+", "", s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^\w\u4e00-\u9fff%.,:;()\-+/]", "", s)
    return s


def _extract_non_table_text_lines(markdown: str) -> List[str]:
    """提取页面中可用于去重的文本行（过滤表格行）。"""
    lines = str(markdown or "").splitlines()
    out: List[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if "|" in s:
            continue
        if s.startswith("#"):
            continue
        if s.startswith("---"):
            continue
        out.append(s)
    return out


def _is_semantically_present(candidate: str, existing_lines: List[str], threshold: float = 0.86) -> bool:
    c_norm = _normalize_for_fuzzy_compare(candidate)
    if not c_norm:
        return True

    for line in existing_lines:
        e_norm = _normalize_for_fuzzy_compare(line)
        if not e_norm:
            continue
        if c_norm == e_norm:
            return True
        if c_norm in e_norm or e_norm in c_norm:
            if min(len(c_norm), len(e_norm)) >= 8:
                return True
        ratio = difflib.SequenceMatcher(None, c_norm, e_norm).ratio()
        if ratio >= threshold:
            return True

    return False


def _filter_missing_text_spans(spans: List[str], markdown: str) -> List[str]:
    """过滤误报缺失文本：若与页面已有文本语义近似则不补充。"""
    existing_lines = _extract_non_table_text_lines(markdown)
    kept: List[str] = []

    for raw in spans or []:
        s = str(raw or "").strip()
        if not s:
            continue
        # 过短文本几乎都属于噪声或标题碎片
        if len(_normalize_for_fuzzy_compare(s)) < 10:
            continue
        # 和原页面已存在文本近似 -> 丢弃
        if _is_semantically_present(s, existing_lines):
            continue
        # 和已经保留的补充文本近似 -> 丢弃
        if _is_semantically_present(s, kept):
            continue
        kept.append(s)

    return kept

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

        processing_dir, output_dir, layout_cache_dir = _build_run_directories(ppt_path)
        os.environ["PPT_PROCESSING_ARTIFACTS_DIR"] = processing_dir
        os.environ["PPT_OUTPUT_DIR"] = output_dir
        os.environ["PPT_LAYOUT_CACHE_DIR"] = layout_cache_dir
        
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

    def _default_merged_slides_path(self) -> Path:
        ppt_dir = Path(self.ppt_path).resolve().parent
        processing_dir = str(getattr(self.config, "processing_artifacts_dir", "processing_artifacts") or "processing_artifacts")
        return ppt_dir / processing_dir / "global_mineru_precompute" / "mineru_output" / "merged_slides" / "auto" / "merged_slides.md"

    def _default_merged_content_list_path(self) -> Path:
        ppt_dir = Path(self.ppt_path).resolve().parent
        processing_dir = str(getattr(self.config, "processing_artifacts_dir", "processing_artifacts") or "processing_artifacts")
        return ppt_dir / processing_dir / "global_mineru_precompute" / "mineru_output" / "merged_slides" / "auto" / "merged_slides_content_list.json"

    def _load_mineru_page_map(self, merged_slides_path: Optional[str] = None) -> Tuple[Dict[int, Dict[str, str]], Optional[Path], Optional[Path], str]:
        """优先从 content_list(page_idx) 还原逐页内容；失败时回退到 merged_slides 标题切分。"""
        content_list_path = self._default_merged_content_list_path()
        if content_list_path.exists():
            try:
                payload = json.loads(content_list_path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    pages = _build_mineru_pages_from_content_list(payload)
                    if pages:
                        labeled_path = content_list_path.parent / "merged_slides_labeled.md"
                        labeled_path.write_text(_render_labeled_mineru_markdown(pages), encoding="utf-8")
                        return pages, content_list_path, labeled_path, "content_list_page_idx"
            except Exception:
                pass

        merged_path = Path(merged_slides_path).resolve() if merged_slides_path else self._default_merged_slides_path()
        if merged_path.exists():
            text = merged_path.read_text(encoding="utf-8")
            sections = _split_merged_slides_sections(text)
            aligned_sections = self._align_mineru_sections(sections)
            if aligned_sections:
                return aligned_sections, None, None, "merged_slides_heading_fallback"

        return {}, None, None, "none"

    def _align_mineru_sections(self, sections: List[Dict[str, str]]) -> Dict[int, Dict[str, str]]:
        """将 merged_slides 段落对齐到当前结果页。优先按顺序，数量不匹配时按标题相似度兜底。"""
        page_indices = sorted(self.results.keys())
        aligned: Dict[int, Dict[str, str]] = {}
        if not page_indices or not sections:
            return aligned

        if len(sections) >= len(page_indices):
            for i, page_idx in enumerate(page_indices):
                aligned[page_idx] = sections[i]
            return aligned

        # 兜底：段落少于页面数时，按标题相似度匹配可用段
        from difflib import SequenceMatcher

        used = set()
        for page_idx in page_indices:
            output = self.results[page_idx]
            title = str(getattr(output, "section_title", "") or "").strip().lower()
            best_i = -1
            best_score = -1.0
            for idx, sec in enumerate(sections):
                if idx in used:
                    continue
                sec_title = str(sec.get("title", "") or "").strip().lower()
                score = SequenceMatcher(None, title, sec_title).ratio() if (title or sec_title) else 0.0
                if score > best_score:
                    best_score = score
                    best_i = idx
            if best_i >= 0:
                used.add(best_i)
                aligned[page_idx] = sections[best_i]

        return aligned

    def _fallback_reconcile_decision(
        self,
        output_md: str,
        mineru_md: str,
        output_table: str,
        mineru_table: str,
    ) -> Dict[str, Any]:
        """模型不可用时的保守规则兜底。"""
        out_no_ws = re.sub(r"\s+", "", output_md or "")
        missing: List[str] = []
        for line in str(mineru_md or "").splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            if "|" in s:
                continue
            if len(s) < 8:
                continue
            if re.sub(r"\s+", "", s) in out_no_ws:
                continue
            missing.append(s)
            if len(missing) >= 5:
                break

        table_diff = False
        if mineru_table and output_table:
            table_diff = _normalize_table_for_compare(mineru_table) != _normalize_table_for_compare(output_table)
        elif mineru_table and not output_table:
            table_diff = True

        return {
            "missing_text_spans": missing,
            "missing_insert_mode": "after_last_table",
            "table_decision": "different" if table_diff else "same",
            "confidence": 0.5,
            "notes": "fallback_rule_based",
        }

    def _judge_reconcile_with_llm(
        self,
        page_no: int,
        output_md: str,
        mineru_md: str,
        output_table: str,
        mineru_table: str,
    ) -> Dict[str, Any]:
        """调用 LLM 对页内“缺失文本+表格冲突”做结构化判定。"""
        fallback = self._fallback_reconcile_decision(output_md, mineru_md, output_table, mineru_table)

        try:
            prompt = f"""你是文档对齐审校器。请对比同一页的 two markdown sources：
1) output.md页面内容
2) MinerU页面内容

目标（仅限这两项）：
- 找出 MinerU 中存在、output 缺失的文本片段（允许没有“注/脚注/注意”等关键词）。
- 判断两者表格是否不同（结构或内容差异均算不同）。

严格要求：
1. 只能输出 JSON，不要输出解释。
2. missing_text_spans 必须是 MinerU 原文中的连续片段（原样摘录），不要改写。
3. 不要处理图片、链接、样式差异。
4. 若不确定，宁可少报，不要臆造。
5. 如果某条信息在 output 中已出现（即使表达略有差异、标点不同、含项目符号），不要再放入 missing_text_spans。
6. 优先返回“真正新增信息”，不要返回与“关键数据/图中说明文字/表格说明”重复的内容。
7. 若 output_primary_table 为空且 mineru_primary_table 非空，table_decision 必须返回 "different"。

JSON schema:
{{
  "missing_text_spans": ["..."],
  "missing_insert_mode": "after_last_table" | "end",
  "table_decision": "same" | "different",
  "confidence": 0.0,
  "notes": "optional"
}}

page_no: {page_no}

[output_page_markdown]
{output_md}

[mineru_page_markdown]
{mineru_md}

[output_primary_table]
{output_table}

[mineru_primary_table]
{mineru_table}
"""

            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model_name,
                messages=[
                    {"role": "system", "content": "你是严格的JSON输出助手。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            raw = (response.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            if not isinstance(data, dict):
                return fallback

            spans = data.get("missing_text_spans", [])
            if not isinstance(spans, list):
                spans = []
            spans = [str(x).strip() for x in spans if str(x or "").strip()]

            mode = str(data.get("missing_insert_mode", "after_last_table") or "after_last_table").strip().lower()
            if mode not in {"after_last_table", "end"}:
                mode = "after_last_table"

            table_decision = str(data.get("table_decision", "same") or "same").strip().lower()
            if table_decision not in {"same", "different"}:
                table_decision = "same"

            try:
                confidence = float(data.get("confidence", 0.0))
            except Exception:
                confidence = 0.0

            notes = str(data.get("notes", "") or "").strip()

            return {
                "missing_text_spans": spans,
                "missing_insert_mode": mode,
                "table_decision": table_decision,
                "confidence": confidence,
                "notes": notes,
            }
        except Exception:
            return fallback

    def _fallback_table_semantics(self, table_md: str) -> Dict[str, Any]:
        """表格语义兜底：在模型不可用时提供保守洞察与关键数据。"""
        rows = [ln.strip() for ln in str(table_md or "").splitlines() if ln.strip() and "|" in ln]
        headers: List[str] = []
        key_values: Dict[str, str] = {}

        if rows:
            first = [c.strip() for c in rows[0].strip("|").split("|")]
            headers = [h for h in first if h and not re.match(r"^[-: ]+$", h)]

        data_rows = []
        for ln in rows[2:]:
            cols = [c.strip() for c in ln.strip("|").split("|")]
            if any(cols):
                data_rows.append(cols)

        if headers and data_rows:
            first_data = data_rows[0]
            usable = min(len(headers), len(first_data), 6)
            for i in range(usable):
                k = str(headers[i]).strip()
                v = str(first_data[i]).strip()
                if k and v:
                    key_values[k] = v

        if not key_values and headers:
            for h in headers[:6]:
                key_values[h] = "见表格对应列"

        insight = "该表展示了本页关键决策/预算信息，建议重点关注首列主题项与右侧对应结论或数值，以支持页面核心判断。"
        if headers:
            insight = f"该表围绕“{'、'.join(headers[:3])}”等字段组织信息，核心结论与关键数值已在表中给出，可用于支撑本页决策结论。"

        return {
            "insight": insight,
            "key_values": key_values,
            "notes": "fallback_table_semantics",
        }

    def _generate_table_semantics_with_llm(self, page_no: int, table_md: str, mineru_md: str) -> Dict[str, Any]:
        """基于表格 markdown 生成“洞察 + 关键数据”用于输出补全。"""
        fallback = self._fallback_table_semantics(table_md)
        if not str(table_md or "").strip():
            return fallback

        try:
            prompt = f"""你是医疗商业汇报审校助手。请仅基于给定表格内容生成用于 markdown 展示的结构化结果。

要求：
1. 只输出 JSON，不要解释。
2. insight 用 1-2 句中文，先写核心观点，再写结构/趋势描述。
3. key_values 返回 3-8 项，键名应简洁且来自表头/行名语义，不要臆造不存在的数据。
4. 若某项数值不明确，不要编造；可省略该项。

JSON schema:
{{
  "insight": "...",
  "key_values": {{"字段": "值"}},
  "notes": "optional"
}}

page_no: {page_no}

[table_markdown]
{table_md}

[page_context_markdown]
{mineru_md}
"""

            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model_name,
                messages=[
                    {"role": "system", "content": "你是严格JSON输出助手。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            raw = (response.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            if not isinstance(data, dict):
                return fallback

            insight = str(data.get("insight", "") or "").strip()
            key_values = data.get("key_values", {})
            if not isinstance(key_values, dict):
                key_values = {}

            norm_kv: Dict[str, str] = {}
            for k, v in key_values.items():
                ks = str(k or "").strip()
                vs = str(v or "").strip()
                if ks and vs:
                    norm_kv[ks] = vs

            if not insight and not norm_kv:
                return fallback

            return {
                "insight": insight or fallback.get("insight", ""),
                "key_values": norm_kv or fallback.get("key_values", {}),
                "notes": str(data.get("notes", "") or "").strip(),
            }
        except Exception:
            return fallback

    def reconcile_with_mineru(
        self,
        merged_slides_path: str | None = None,
        use_model_judge: bool = True,
    ) -> Dict[str, Any]:
        """按页对齐 output 与 merged_slides：补缺失文本；表格冲突默认采用 MinerU。"""
        if not self.results:
            return {"applied": False, "reason": "当前无解析结果"}

        aligned, content_list_path, labeled_path, source_mode = self._load_mineru_page_map(merged_slides_path)
        if not aligned:
            default_merged = self._default_merged_slides_path()
            return {"applied": False, "reason": f"无法加载 MinerU 页映射（checked: {default_merged}）"}

        updated_pages = 0
        conflict_pages = 0

        for page_idx in sorted(self.results.keys()):
            output = self.results[page_idx]
            sec = aligned.get(page_idx)
            if not sec:
                continue

            output_md = str(getattr(output, "markdown_content", "") or "")
            mineru_md = str(sec.get("content", "") or "").strip()
            if not mineru_md:
                continue

            output_tables = _extract_markdown_tables(output_md)
            mineru_tables = _extract_markdown_tables(mineru_md)
            output_table = output_tables[0] if output_tables else ""
            mineru_table = mineru_tables[0] if mineru_tables else ""

            if use_model_judge:
                decision = self._judge_reconcile_with_llm(
                    page_no=page_idx + 1,
                    output_md=output_md,
                    mineru_md=mineru_md,
                    output_table=output_table,
                    mineru_table=mineru_table,
                )
            else:
                decision = self._fallback_reconcile_decision(output_md, mineru_md, output_table, mineru_table)

            missing_texts_raw = decision.get("missing_text_spans", []) if isinstance(decision.get("missing_text_spans", []), list) else []
            missing_texts = _filter_missing_text_spans(missing_texts_raw, output_md)
            insert_mode = str(decision.get("missing_insert_mode", "after_last_table") or "after_last_table")
            table_decision = str(decision.get("table_decision", "same") or "same")

            # 硬约束：若 MinerU 有表格而 output 无 markdown 表格，则必须按差异处理。
            # 避免模型误判为 same 导致“只有图片、无表格正文”的页面漏补。
            forced_table_replace = False
            if mineru_table and not output_table:
                table_decision = "different"
                forced_table_replace = True

            merged_md = output_md
            table_conflict = False
            if table_decision == "different" and mineru_table:
                merged_md = _replace_first_table(merged_md, mineru_table)
                table_conflict = True

            presence = _table_analysis_presence(merged_md)
            table_semantic_filled = False
            table_semantic_notes = ""
            if mineru_table and presence.get("has_table") and (not presence.get("has_insight") or not presence.get("has_key_data")):
                table_semantics = self._generate_table_semantics_with_llm(
                    page_no=page_idx + 1,
                    table_md=mineru_table,
                    mineru_md=mineru_md,
                )
                merged_with_semantic = _inject_table_analysis_block(
                    merged_md,
                    insight=str(table_semantics.get("insight", "") or ""),
                    key_values=table_semantics.get("key_values", {}) if isinstance(table_semantics.get("key_values", {}), dict) else {},
                )
                if merged_with_semantic.strip() != merged_md.strip():
                    merged_md = merged_with_semantic
                    table_semantic_filled = True
                    table_semantic_notes = str(table_semantics.get("notes", "") or "")

            if missing_texts:
                merged_md = _append_missing_texts(merged_md, missing_texts, mode=insert_mode)

            # 更新页面自动输出（人工审核仍可覆盖）
            if merged_md.strip() != output_md.strip():
                output.markdown_content = merged_md
                updated_pages += 1

            output.reconcile_applied = True
            output.reconcile_mineru_page_markdown = mineru_md
            output.reconcile_missing_texts = [str(x).strip() for x in missing_texts if str(x or "").strip()]
            output.reconcile_table_conflict = bool(table_conflict)
            output.reconcile_table_output = output_table or None
            output.reconcile_table_mineru = mineru_table or None
            output.reconcile_table_merged = _extract_markdown_tables(merged_md)[0] if _extract_markdown_tables(merged_md) else None
            output.reconcile_notes = [
                f"table_decision={table_decision}",
                f"missing_count={len(output.reconcile_missing_texts)}",
                f"insert_mode={insert_mode}",
                f"forced_table_replace={forced_table_replace}",
                f"table_semantic_filled={table_semantic_filled}",
                f"table_semantic_notes={table_semantic_notes}",
            ]

            if table_conflict:
                conflict_pages += 1

        return {
            "applied": True,
            "merged_slides_path": str(self._default_merged_slides_path()),
            "merged_content_list_path": str(content_list_path) if content_list_path else None,
            "merged_labeled_path": str(labeled_path) if labeled_path else None,
            "source_mode": source_mode,
            "updated_pages": updated_pages,
            "conflict_pages": conflict_pages,
            "aligned_pages": len(aligned),
        }

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

            if bool(getattr(output, "reconcile_table_conflict", False)):
                needs_review = True
                reasons.append("与MinerU表格存在差异，已默认采用MinerU表格，需人工确认")
                review_score = max(review_score, 0.7)

            if getattr(output, "reconcile_missing_texts", None):
                reasons.append(f"已补充 {len(getattr(output, 'reconcile_missing_texts', []) or [])} 条MinerU缺失文本，建议人工复核")

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
                "reconcile_applied": bool(getattr(output, "reconcile_applied", False)),
                "reconcile_notes": getattr(output, "reconcile_notes", []) or [],
                "table_conflict": bool(getattr(output, "reconcile_table_conflict", False)),
                "table_from_output": getattr(output, "reconcile_table_output", None),
                "table_from_mineru": getattr(output, "reconcile_table_mineru", None),
                "table_merged_result": getattr(output, "reconcile_table_merged", None),
                "mineru_missing_texts": getattr(output, "reconcile_missing_texts", []) or [],
                "review_status": "pending",
                "review_notes": "",
                "reviewed_summary": "",
                "reviewed_markdown_content": "",
                "reviewed_elements": {},
                "selected_table_source": "merged",
            })

        review_tasks_json = out_dir / "review_tasks.json"
        with open(review_tasks_json, "w", encoding="utf-8") as f:
            json.dump(review_tasks, f, ensure_ascii=False, indent=2)

        review_queue_md = out_dir / "review_queue.md"
        project_root = Path(__file__).parent.resolve()
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
            artifacts_dir = None
            try:
                output_obj = self.results.get(item["page_index"])
                artifacts_dir = (getattr(output_obj, "slice_reference", {}) or {}).get("artifacts_dir")
            except Exception:
                artifacts_dir = None

            rewritten_auto_md = _rewrite_markdown_image_paths(
                item["auto_markdown"] or "",
                artifacts_dir=artifacts_dir,
                output_md_path=review_queue_md,
                project_root=project_root,
            )
            snippet = rewritten_auto_md[:1200]
            lines.append(snippet if snippet else "(空)")
            lines.append("")
            lines.append("### 人工修订建议")
            if item.get("table_conflict"):
                lines.append("- 检测到 output 与 MinerU 表格冲突：请在 review_results.json 中确认最终表格来源，并可微调")
                lines.append("")
                lines.append("### 表格对比（Output）")
                lines.append(item.get("table_from_output") or "(无)")
                lines.append("")
                lines.append("### 表格对比（MinerU）")
                lines.append(item.get("table_from_mineru") or "(无)")
                lines.append("")
                lines.append("### 当前合并结果（默认）")
                lines.append(item.get("table_merged_result") or "(无)")
                lines.append("")
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
                    "reviewed_elements": {},
                    "selected_table_source": "merged"
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
            else:
                selected_source = str(item.get("selected_table_source", "") or "").strip().lower()
                if selected_source in {"output", "mineru", "merged"} and bool(getattr(output, "reconcile_table_conflict", False)):
                    base_md = str(getattr(output, "markdown_content", "") or "")
                    source_table = ""
                    if selected_source == "output":
                        source_table = str(getattr(output, "reconcile_table_output", "") or "")
                    elif selected_source == "mineru":
                        source_table = str(getattr(output, "reconcile_table_mineru", "") or "")
                    else:
                        source_table = str(getattr(output, "reconcile_table_merged", "") or "")

                    if source_table.strip():
                        output.reviewed_markdown_content = _replace_first_table(base_md, source_table)

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
    ppt_path = "FH1701 KRAS突变NSCLC开发_GPT上升会问题跟进_20260312_vPre-read.pptx"  # 修改为实际的PPT文件路径
    
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

    # 与全局 MinerU 输出对齐：补缺失文本 + 表格冲突默认采用 MinerU
    reconcile_meta = pipeline.reconcile_with_mineru(use_model_judge=True)
    if reconcile_meta.get("applied"):
        print(
            f"[Reconcile] 对齐完成: source={reconcile_meta.get('source_mode')}, "
            f"updated={reconcile_meta.get('updated_pages', 0)}, conflicts={reconcile_meta.get('conflict_pages', 0)}"
        )
        if reconcile_meta.get("merged_labeled_path"):
            print(f"[Reconcile] 带页标记MinerU文本: {reconcile_meta.get('merged_labeled_path')}")
    else:
        print(f"[Reconcile] 未执行: {reconcile_meta.get('reason')}")
    
    # 保存结果
    pipeline.save_results()

    # 生成人工审核任务（复杂图表/低置信度页面）
    hitl_meta = pipeline.prepare_human_review(review_output_dir=pipeline.config.output_dir)
    print(f"[HITL] 待人工审核页面: {hitl_meta['review_required_pages']}")

    auto_merge = pipeline.apply_human_review_if_available(review_output_dir=pipeline.config.output_dir)
    if auto_merge.get("applied"):
        print(f"[HITL] 已自动合并人工审核结果: {auto_merge.get('review_results_json')}")
        print(f"[HITL] 已更新页面数: {auto_merge.get('updated_pages', 0)}")
    else:
        print(f"[HITL] 尚未合并人工审核结果: {auto_merge.get('reason')}")
    
    # 导出为Markdown
    pipeline.export_markdown(str(Path(pipeline.config.output_dir) / "output.md"))
    
    # 打印样本结果
    if results:
        first_page_idx = list(results.keys())[0]
        first_result = results[first_page_idx]
        
        print("\n" + "="*70)
        print("[Sample Results] Page " + str(first_page_idx) + ":")
        print("="*70)
        # 简化输出以避免编码问题
        print(f"[Note] Results saved to: {pipeline.config.output_dir}/")
        print("[Note] Debug logs saved to: debug_logs/")


if __name__ == "__main__":
    main()
