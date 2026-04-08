"""Step 4: 最终输出生成节点，生成结构化markdown报告"""

from state import (
    PPTPageState, PageAnalysisResult, FinalPageOutput
)
from config import get_config
from typing import Dict, Any
import re


import os


_VISUAL_ELEMENT_TYPES = {
    "chart",
    "table",
    "diagram",
    "flowchart",
    "image",
}


_DECORATIVE_PAGE_PATTERNS = [
    r"(^|\n)\s*(目录|目\s*录|contents?)\s*$",
    r"(title\s*or\s*decorative\s*page|decorative\s*page)",
    r"本页为(目录|封面|章节|过渡)页",
    r"(目录页|封面页|章节页|过渡页)",
]


def _strip_hypothesis_sections(text: str) -> str:
    """移除 key_insight 中的“假设验证/验证假设”等调试段落。

    说明：该信息仍保留在 ElementInsight.hypothesis_verification / debug logs 中，
    但不应出现在最终用户可见输出。
    """
    if not text:
        return ""

    # 常见标题/加粗写法
    patterns = [
        r"\n?#{1,6}\s*(假设验证|验证假设)\s*[:：]?\s*\n[\s\S]*$",
        r"\n?\*\*(假设验证|验证假设)\*\*\s*[:：]?\s*\n[\s\S]*$",
        r"\n?(假设验证|验证假设)\s*[:：]\s*\n[\s\S]*$",
    ]
    cleaned = text
    for p in patterns:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)

    return cleaned.strip()


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _extract_raw_text_content(global_analysis: Any, element_insights: list) -> tuple[str, str]:
    """提取页面文本内容。

    返回 (text, source):
    - source: "xml_full" | "text_aggregate" | "vlm_extracted" | ""
    """
    # 1) Prefer XML-parsed markdown/text from TextWorker
    for ins in element_insights or []:
        if getattr(ins, "element_id", "") == "full_text_page" and (getattr(ins, "element_type", "") or "") == "text_page":
            xml_md = (getattr(ins, "data_evidence", "") or "").strip()
            if xml_md:
                return xml_md, "xml_full"

    # 2) Aggregated text blocks from locator (non-pure text pages)
    for ins in element_insights or []:
        if (getattr(ins, "element_type", "") or "") == "text_aggregate":
            merged = (getattr(ins, "key_insight", "") or "").strip()
            if merged:
                return merged, "text_aggregate"

    # 3) Fallback to Step1 extracted_text
    extracted_text = (getattr(global_analysis, "extracted_text", "") or "").strip()
    if extracted_text:
        return extracted_text, "vlm_extracted"

    return "", ""

class Step4_OutputGenerator:
    """最终输出生成器
    
    职责：
    1. 接收Supervisor的校验结果
    2. 按照PPT章节、页号、标题进行组织
    3. 生成结构化的markdown输出
    4. 包含元素的引用、分析内容引用、第一步总结引用等
    """
    
    @staticmethod
    def generate_output(page_state: PPTPageState, 
                       analysis_result: PageAnalysisResult,
                       chapter_title: str = None) -> FinalPageOutput:
        """生成最终输出"""
        print(f"\n[Step4] 生成页面 {page_state.page_index} 的最终输出")
        
        # 提取信息
        page_index = analysis_result.page_index
        section_title = analysis_result.section_title
        global_analysis = analysis_result.global_analysis
        element_insights = analysis_result.element_insights

        # 证据门禁：封面/章节/装饰页或证据不足时，降级输出，避免生成推断性结论
        if Step4_OutputGenerator._should_degrade_output(global_analysis, element_insights):
            inferred_title, inferred_source = Step4_OutputGenerator._infer_title_from_text_worker(element_insights)
            section_title = (global_analysis.section_title or "").strip() or (analysis_result.section_title or "").strip() or inferred_title or "封面/章节页"

            summary = (getattr(global_analysis, "core_summary", "") or "").strip()
            # 将占位符替换为更准确的表述（优先使用XML标题）
            if "Title or Decorative Page" in summary or summary == "一句话总结" or summary == "一句话总结 Title or Decorative Page":
                if inferred_title:
                    summary = f"本页为封面/章节页：{inferred_title}" if section_title == inferred_title else f"本页为封面/章节页：{section_title}"
                else:
                    summary = "本页为封面/章节页（未识别到明确标题）。"

            markdown = Step4_OutputGenerator._generate_degraded_markdown(
                analysis_result.page_index,
                chapter_title,
                section_title,
                summary,
                inferred_source
            )
            return FinalPageOutput(
                page_index=analysis_result.page_index,
                chapter_title=chapter_title,
                section_title=section_title,
                summary=summary,
                elements={},
                markdown_content=markdown,
                slice_reference={
                    "summary": {
                        "source": "全局分析（Step 1）",
                        "note": "降级输出：证据不足/装饰页",
                        "title_source": inferred_source
                    }
                }
            )
        
        # 如果该页由复杂 Pipeline 产出 markdown，优先复用它（更贴近 MinerU 的图表/图片展示）
        cfg = get_config()
        processing_root = str(getattr(page_state, "processing_artifacts_dir", "") or "").strip()
        if not processing_root:
            processing_root = str(getattr(cfg, "processing_artifacts_dir", "processing_artifacts") or "processing_artifacts")
        artifacts_dir = os.path.join(processing_root, f"page_{page_index:03d}")
        pipeline_md_path_abs = os.path.join(artifacts_dir, f"page_{page_index:03d}_output.md")

        use_pipeline_md = os.path.exists(pipeline_md_path_abs)

        # 构建最终元素字典：仅保留可视元素，且剥离假设验证文本
        elements_dict: Dict[str, Any] = {}
        for insight in element_insights or []:
            et = (getattr(insight, "element_type", "") or "").lower()
            if et not in _VISUAL_ELEMENT_TYPES:
                continue

            elements_dict[insight.element_id] = {
                "type": insight.element_type,
                "key_insight": _strip_hypothesis_sections(getattr(insight, "key_insight", "")),
                "data_evidence": getattr(insight, "data_evidence", ""),
                "confidence": getattr(insight, "confidence", 0.0),
            }
        
        # 构建 slice reference（用于调试/路径重写）
        slice_reference: Dict[str, Any] = {
            "summary": {
                "source": "全局分析（Step 1）",
                "line_range": [1, 3],
            },
            "artifacts_dir": artifacts_dir,
            "render_mode": "complex_pipeline_md" if use_pipeline_md else "visual_only",
        }

        for idx, insight in enumerate(element_insights or [], 1):
            slice_reference[insight.element_id] = {
                "type": getattr(insight, "element_type", ""),
                "section": f"元素分析 ({idx})",
                "confidence": getattr(insight, "confidence", 0.0),
            }
        
        # 生成 markdown：复杂页直接复用 pipeline 输出；否则生成“仅可视元素”的精简版
        if use_pipeline_md:
            markdown = _read_text_file(pipeline_md_path_abs)
        else:
            markdown = Step4_OutputGenerator._generate_markdown(
                page_index,
                chapter_title,
                section_title,
                global_analysis,
                element_insights,
            )
        
        # 构建最终输出
        final_output = FinalPageOutput(
            page_index=page_index,
            chapter_title=chapter_title,
            section_title=section_title,
            summary=global_analysis.core_summary,
            elements=elements_dict,
            markdown_content=markdown,
            slice_reference=slice_reference
        )
        
        return final_output

    @staticmethod
    def _should_degrade_output(global_analysis: Any, element_insights: list) -> bool:
        """判断是否需要降级：避免对章节/封面/证据不足页面过度分析。"""
        if global_analysis is None:
            return True

        extracted_text = (getattr(global_analysis, "extracted_text", "") or "").strip()
        elements = getattr(global_analysis, "elements", []) or []
        insights = element_insights or []

        # 只要已经有可视证据（Step1元素或Step2/3洞察），就不应降级为“封面/目录页”。
        has_visual_evidence = any(
            str(getattr(e, "type", "") or "").lower() in _VISUAL_ELEMENT_TYPES
            for e in elements
        ) or any(
            str(getattr(ins, "element_type", "") or "").lower() in _VISUAL_ELEMENT_TYPES
            for ins in insights
        )
        if has_visual_evidence:
            return False

        core = (getattr(global_analysis, "core_summary", "") or "").lower()
        title = (getattr(global_analysis, "section_title", "") or "").lower()
        decorative_hint = any(
            re.search(p, core, flags=re.IGNORECASE) or re.search(p, title, flags=re.IGNORECASE)
            for p in _DECORATIVE_PAGE_PATTERNS
        )

        # 纯文本页且没有提取文本，且无元素/洞察 -> 基本属于章节/装饰/空页
        is_pure_text = bool(getattr(global_analysis, "is_pure_text", False))
        no_evidence = (not extracted_text) and (len(elements) == 0) and (len(insights) == 0)

        return decorative_hint or (is_pure_text and no_evidence)

    @staticmethod
    def _infer_title_from_text_worker(element_insights: list) -> tuple[str, str]:
        """从 TextWorker 的 XML Markdown 中推断标题。

        返回: (title, source)
        - title: 推断的标题（可能为空）
        - source: 'xml' | ''
        """
        if not element_insights:
            return "", ""

        xml_md = ""
        for ins in element_insights:
            if getattr(ins, "element_id", "") == "full_text_page":
                xml_md = (getattr(ins, "data_evidence", "") or "").strip()
                break
        if not xml_md:
            return "", ""

        # 优先匹配 Markdown H1
        for line in xml_md.splitlines():
            s = line.strip()
            if s.startswith("# ") and len(s) > 2:
                title = s[2:].strip()
                if title:
                    return title, "xml"

        # 次优：取第一条短文本行作为标题（避免把大段正文当标题）
        for line in xml_md.splitlines():
            s = line.strip().lstrip("-*")
            if not s:
                continue
            if len(s) <= 40:
                return s, "xml"

        return "", "xml"

    @staticmethod
    def _generate_degraded_markdown(page_index: int,
                                   chapter_title: str,
                                   section_title: str,
                                   summary: str,
                                   title_source: str) -> str:
        display_page_num = page_index + 1
        lines = []
        if chapter_title:
            lines.append(f"# {chapter_title}")
            lines.append("")

        lines.append(f"## 第 {display_page_num} 页：{section_title}")
        lines.append("")
        lines.append("### 1. 核心观点")
        lines.append(summary if summary else "本页未检测到可用于结构化解析的有效内容（可能为封面/章节分隔/装饰页）。")
        return "\n".join(lines)
    
    @staticmethod
    def _generate_markdown(page_index: int,
                          chapter_title: str,
                          section_title: str,
                          global_analysis: Any,
                          element_insights: list) -> str:
        """生成结构化markdown内容。

        约定：
        - 默认输出：核心观点 + 关键可视元素。
        - 对纯文本页（或存在 TextWorker 的 full_text_page）额外输出页面原始内容。
        """
        display_page_num = page_index + 1
        lines = []
        
        # 文档头
        if chapter_title:
            lines.append(f"# {chapter_title}")
            lines.append("")
        
        lines.append(f"## 第 {display_page_num} 页：{section_title}")
        lines.append("")
        # 核心观点
        lines.append("### 1. 核心观点")
        lines.append((getattr(global_analysis, "core_summary", "") or "").strip())
        lines.append("")

        # 页面文本内容（纯文本页输出原始内容；非纯文本页输出 text 元素聚合）
        raw_text, raw_source = _extract_raw_text_content(global_analysis, element_insights)
        if raw_text:
            text_section_title = "页面原始内容" if raw_source == "xml_full" else "页面文本"
            lines.append(f"### 2. {text_section_title}")
            lines.append("```text")
            lines.append(raw_text)
            lines.append("```")
            lines.append("")

        # 关键可视元素
        visual_insights = []
        for ins in element_insights or []:
            et = (getattr(ins, "element_type", "") or "").lower()
            if et in _VISUAL_ELEMENT_TYPES:
                visual_insights.append(ins)

        visual_section_no = 3 if raw_text else 2
        if visual_insights:
            lines.append(f"### {visual_section_no}. 关键图表/图片")
            for ins in visual_insights:
                lines.append("")
                lines.append(f"- **[{(ins.element_type or '').upper()}] {ins.element_id}**")

                # 仅对可视元素展示图片（且不强制存在 crop；复杂页一般会走 pipeline_md）
                if getattr(ins, "crop_path", None) and os.path.exists(ins.crop_path):
                    try:
                        rel_path = os.path.relpath(ins.crop_path, os.getcwd())
                        lines.append(f"\n  ![{ins.element_id}]({rel_path})")
                    except ValueError:
                        lines.append(f"\n  ![{ins.element_id}]({ins.crop_path})")

                key_insight = _strip_hypothesis_sections(getattr(ins, "key_insight", ""))
                if key_insight:
                    lines.append(f"\n  - 核心洞察：{key_insight}")

                data_evidence = (getattr(ins, "data_evidence", "") or "").strip()
                if data_evidence:
                    lines.append(f"  - 数据支撑：{data_evidence}")
        else:
            lines.append(f"### {visual_section_no}. 关键图表/图片")
            lines.append("本页未检测到需要展示的图表/流程图/图片元素。")
        
        return "\n".join(lines)


def node_step4_output_generation(state: PPTPageState, debug_logger=None) -> dict:
    """节点：Step 4 最终输出生成"""
    page_index = state.page_index
    
    print(f"\n[Step4] 页面 {page_index} 最终输出生成")
    
    # 记录步骤开始
    if debug_logger:
        debug_logger.log_step_start(
            page_index,
            "Step 4: Output Generation",
            input_data={
                "page_index": page_index,
                "section_title": state.analysis_result.section_title if state.analysis_result else None,
                "element_insights_count": len(state.element_insights),
                "analysis_result_status": state.analysis_result.status if state.analysis_result else None
            },
            previous_step="Step 3: Supervisor"
        )
    
    analysis_result = state.analysis_result
    
    # 生成最终输出
    final_output = Step4_OutputGenerator.generate_output(
        state,
        analysis_result,
        state.chapter_title
    )
    
    # 记录步骤结束
    if debug_logger:
        debug_logger.log_step_end(
            page_index,
            "Step 4: Output Generation",
            output_data={
                "page_index": final_output.page_index,
                "section_title": final_output.section_title,
                "summary_preview": final_output.summary[:200],
                "elements_count": len(final_output.elements) if isinstance(final_output.elements, dict) else 0,
                "markdown_length": len(final_output.markdown_content)
            },
            status="success"
        )
    
    print(f"[Step4] [OK] 页面 {page_index} 完成")
    
    return {
        "final_output": final_output
    }
