"""
Phase 4: 统一组装与输出 (Normalization & Assembly)

核心目的：生成对 RAG 友好且对人类阅读友好的最终格式

包含组件：
1. SchemaValidator: Schema 校验器
2. Phase4_Assembler: Phase 4 组装器
"""

import json
import os
import time
import re
import difflib
from typing import List, Dict, Any, Optional

from .pipeline_state import (
    ValidatedSemanticJSON, ExtractedDataMap, SemanticGroup,
    ExtractedContent, DetectedElement,
    SemanticBlock, FinalStructuredJSON, FinalMarkdownOutput
)


# ============================================================================
# Schema 校验器
# ============================================================================

class SchemaValidator:
    """
    输出 Schema 校验器
    
    确保输出符合预定义的 JSON 结构
    """
    
    REQUIRED_BLOCK_FIELDS = ["block_id", "block_type", "reading_order", "elements"]
    REQUIRED_JSON_FIELDS = ["page_id", "layout_confidence", "semantic_blocks"]
    
    @staticmethod
    def validate_structured_json(data: FinalStructuredJSON) -> tuple[bool, List[str]]:
        """
        验证结构化 JSON 输出
        
        Args:
            data: 待验证的 FinalStructuredJSON
            
        Returns:
            (是否有效, 错误列表)
        """
        errors = []
        
        # 检查必填字段
        if data.page_id is None:
            errors.append("Missing required field: page_id")
        
        if data.layout_confidence not in ["high", "medium", "low"]:
            errors.append(f"Invalid layout_confidence: {data.layout_confidence}")
        
        # 检查语义块
        for i, block in enumerate(data.semantic_blocks):
            if not block.block_id:
                errors.append(f"Block {i}: missing block_id")
            if not block.block_type:
                errors.append(f"Block {i}: missing block_type")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @staticmethod
    def validate_markdown(content: str) -> tuple[bool, List[str]]:
        """
        验证 Markdown 输出
        
        Args:
            content: Markdown 内容
            
        Returns:
            (是否有效, 警告列表)
        """
        warnings = []
        
        if not content.strip():
            warnings.append("Markdown content is empty")
        
        # 检查是否有标题
        if not content.startswith("#"):
            warnings.append("Markdown should start with a heading")
        
        # 检查图片引用格式
        import re
        img_pattern = r'!\[.*?\]\(.*?\)'
        if "![" in content and not re.search(img_pattern, content):
            warnings.append("Invalid image reference format detected")
        
        is_valid = "Markdown content is empty" not in warnings
        return is_valid, warnings


# ============================================================================
# Phase 4 组装器
# ============================================================================

class Phase4_Assembler:
    """
    Phase 4 统一组装与输出 - 总控制器
    
    整合所有数据，生成最终的 Markdown 和 JSON 输出
    """
    
    def __init__(self, image_base_url: str = "", llm_client=None):
        """
        初始化组装器
        
        Args:
            image_base_url: 图片基础URL（用于生成图片链接）
        """
        self.image_base_url = image_base_url
        self.llm_client = llm_client
        self._translation_cache: Dict[str, str] = {}
        try:
            from config import get_config
            self.translation_model = get_config().llm_model_name
        except Exception:
            self.translation_model = None
        self.schema_validator = SchemaValidator()

    def _contains_cjk(self, text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text or ""))

    @staticmethod
    def _contains_failure_marker(text: str) -> bool:
        s = str(text or "").strip().lower()
        if not s:
            return False
        markers = (
            "分析失败",
            "提取失败",
            "mixed提取降级",
            "混合提取降级",
            "unterminated string",
            "jsondecodeerror",
            "expecting value",
            "line 1 column",
            "char ",
            "traceback",
            "error:",
        )
        return any(m in s for m in markers)

    @staticmethod
    def _is_footer_noise_text(text: str) -> bool:
        s = str(text or "").strip()
        if not s:
            return False
        if re.fullmatch(r"\d+", s):
            return True
        if re.fullmatch(r"\d+\s+[A-Z][A-Z0-9\-]{2,}", s):
            return True
        if re.fullmatch(r"[A-Z][A-Z0-9\-]{4,}", s):
            return True
        if s.upper() in {"DIVAMICS", "ADIS"}:
            return True
        return False

    def _is_chart_payload_valid(self, chart_data: Dict[str, Any]) -> bool:
        if not isinstance(chart_data, dict):
            return False

        insight = str(chart_data.get("insight", "") or "").strip()
        chart_type = str(chart_data.get("type", "") or "").strip().lower()

        if self._contains_failure_marker(insight):
            return False

        if chart_type in {"table", "grid", "matrix"}:
            return False

        key_values = chart_data.get("key_values", {})
        panel_titles = chart_data.get("panel_titles", []) if isinstance(chart_data.get("panel_titles", []), list) else []
        axis_labels = chart_data.get("axis_labels", []) if isinstance(chart_data.get("axis_labels", []), list) else []
        legend_items = chart_data.get("legend_items", []) if isinstance(chart_data.get("legend_items", []), list) else []
        annotation_texts = chart_data.get("annotation_texts", []) if isinstance(chart_data.get("annotation_texts", []), list) else []
        x_axis = str(chart_data.get("x_axis", "") or "").strip()
        y_axis = str(chart_data.get("y_axis", "") or "").strip()

        has_signal = bool(
            (insight and len(insight) >= 4)
            or (isinstance(key_values, dict) and len(key_values) > 0)
            or panel_titles
            or axis_labels
            or legend_items
            or annotation_texts
            or x_axis
            or y_axis
        )

        if chart_type in {"", "unknown", "other", "未知"} and not has_signal:
            return False

        return has_signal or chart_type not in {"", "unknown", "other", "未知"}

    def _is_table_payload_valid(self, table_item: Any) -> bool:
        markdown = str(getattr(table_item, "markdown_table", "") or "").strip()
        return bool(markdown)

    def _has_zh_paren_translation(self, text: str) -> bool:
        if not text:
            return False
        # Matches (...) or （...） that contains at least one CJK char.
        return bool(re.search(r"[\(（][^\)）]*[\u4e00-\u9fff][^\)）]*[\)）]", text))

    def _translate_title(self, text: str) -> str:
        if not text:
            return ""
        if text in self._translation_cache:
            return self._translation_cache[text]
        if not self.llm_client or not self.translation_model:
            return ""

        prompt = (
            "将以下标题翻译为中文，仅输出中文译文，不要加引号或解释。"
            "保留原有缩写和专业名词的缩写形式。\n标题："
            f"{text}"
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.translation_model,
                messages=[
                    {"role": "system", "content": "你是专业的中英标题翻译助手。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            translation = response.choices[0].message.content.strip()
        except Exception:
            translation = ""

        self._translation_cache[text] = translation
        return translation

    def _with_zh_translation(self, text: str) -> str:
        if not text:
            return text
        if self._contains_cjk(text) or self._has_zh_paren_translation(text):
            return text
        translation = self._translate_title(text)
        if not translation:
            return text
        return f"{text}（{translation}）"

    def _translate_markdown_headings(self, text: str) -> str:
        if not text:
            return text
        lines = []
        for line in text.splitlines():
            m = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
            if m:
                level = m.group(1)
                title = m.group(2).strip()
                if title:
                    line = f"{level} {self._with_zh_translation(title)}"
            lines.append(line)
        return "\n".join(lines)
    
    def run(self,
           validated_semantic: ValidatedSemanticJSON,
           extracted_data: ExtractedDataMap,
           page_title: str = "") -> tuple[FinalStructuredJSON, FinalMarkdownOutput]:
        """
        执行 Phase 4 完整流程
        
        Args:
            validated_semantic: Phase 2 的输出
            extracted_data: Phase 3 的输出
            page_title: 页面标题（可选）
            
        Returns:
            (FinalStructuredJSON, FinalMarkdownOutput)
        """
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"[Phase 4] 开始统一组装与输出 (Page {validated_semantic.page_id})")
        print(f"{'='*60}")
        
        page_id = validated_semantic.page_id
        
        # Step 1: 构建语义块
        print("\n[Phase 4.1] 构建语义块...")
        semantic_blocks = self._build_semantic_blocks(
            validated_semantic,
            extracted_data
        )
        
        # Step 2: 生成结构化 JSON
        print("\n[Phase 4.2] 生成结构化 JSON...")
        final_json = self._build_structured_json(
            page_id,
            validated_semantic.layout_confidence,
            semantic_blocks,
            page_title
        )
        
        # Step 3: 生成 Markdown
        print("\n[Phase 4.3] 生成 Markdown 输出...")
        final_markdown = self._build_markdown(
            page_id,
            page_title,
            semantic_blocks,
            extracted_data
        )
        
        # Step 4: Schema 校验
        print("\n[Phase 4.4] Schema 校验...")
        json_valid, json_errors = self.schema_validator.validate_structured_json(final_json)
        md_valid, md_warnings = self.schema_validator.validate_markdown(final_markdown.markdown_content)
        
        if json_errors:
            print(f"  ⚠️ JSON 校验错误: {json_errors}")
        if md_warnings:
            print(f"  ⚠️ Markdown 警告: {md_warnings}")
        
        processing_time = int((time.time() - start_time) * 1000)
        
        print(f"\n[Phase 4] 完成！耗时 {processing_time}ms")
        print(f"  - 语义块数: {len(semantic_blocks)}")
        print(f"  - Markdown长度: {len(final_markdown.markdown_content)} 字符")
        print(f"  - 表格数: {final_markdown.table_count}")
        print(f"  - 图表数: {final_markdown.chart_count}")
        
        return final_json, final_markdown
    
    def _build_semantic_blocks(self,
                              validated_semantic: ValidatedSemanticJSON,
                              extracted_data: ExtractedDataMap) -> List[SemanticBlock]:
        """构建语义块列表"""
        blocks = []
        elem_map = {e.element_id: e for e in validated_semantic.elements}
        
        # 按阅读顺序处理分组
        sorted_groups = sorted(validated_semantic.groups, key=lambda g: g.reading_order)
        
        for group in sorted_groups:
            # 获取提取的内容
            primary_content = extracted_data.extractions.get(group.primary_element_id)
            
            # 构建元素内容
            elements_content = self._build_block_elements(
                group, extracted_data, validated_semantic.id_to_text_map, elem_map
            )
            
            # 计算联合边界框
            member_bboxes = [
                elem_map[mid].bbox 
                for mid in group.member_ids 
                if mid in elem_map
            ]
            bbox_union = self._compute_union_bbox(member_bboxes) if member_bboxes else None
            
            # 确定块类型和角色
            block_type = self._determine_block_type(group, primary_content)
            role = self._determine_role(group, primary_content)
            
            block = SemanticBlock(
                block_id=group.group_id,
                block_type=block_type,
                reading_order=group.reading_order,
                role=role,
                elements=elements_content,
                bbox_union=bbox_union
            )
            
            blocks.append(block)
        
        return blocks

    @staticmethod
    def _axis_to_text(axis_value: Any) -> str:
        """将多态轴字段（str/dict/list）转换为可读文本。"""
        if axis_value is None:
            return ""
        if isinstance(axis_value, str):
            return axis_value.strip()
        if isinstance(axis_value, list):
            return "；".join(str(v).strip() for v in axis_value if str(v).strip())
        if isinstance(axis_value, dict):
            pairs = []
            for k, v in axis_value.items():
                k_s = str(k).strip()
                v_s = str(v).strip()
                if k_s and v_s:
                    pairs.append(f"{k_s}: {v_s}")
            return "；".join(pairs)
        return str(axis_value).strip()

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
        normalized = normalized.replace("（", "(").replace("）", ")")
        return normalized

    @staticmethod
    def _canonical_ocr_text(text: str) -> str:
        s = str(text or "").strip().lower()
        s = s.replace("（", "(").replace("）", ")")
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[\-–—_]+", "", s)
        s = re.sub(r"([a-z])0([a-z])", r"\1o\2", s)
        s = re.sub(r"([a-z])[1l|]([a-z])", r"\1i\2", s)
        s = re.sub(r"[^\w\u4e00-\u9fff\s\(\)%/\.:,+]", "", s)
        return s.strip()

    @staticmethod
    def _looks_joined_english(text: str) -> bool:
        s = str(text or "").strip()
        if len(s) < 35:
            return False
        letters = re.findall(r"[A-Za-z]", s)
        if not letters:
            return False
        letter_ratio = len(letters) / max(len(s), 1)
        space_ratio = s.count(" ") / max(len(s), 1)
        long_word = max((len(w) for w in re.findall(r"[A-Za-z]+", s)), default=0)
        return letter_ratio > 0.75 and (space_ratio < 0.06 or long_word >= 20)

    @staticmethod
    def _repair_joined_english(text: str) -> str:
        s = str(text or "").strip()
        if not s:
            return ""
        if not Phase4_Assembler._looks_joined_english(s):
            return s

        repaired = s
        repaired = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", repaired)
        repaired = re.sub(r",(?=[A-Za-z])", ", ", repaired)
        repaired = re.sub(r"\)(?=[A-Za-z])", ") ", repaired)
        repaired = re.sub(r"(?<=[A-Za-z])\((?=[A-Za-z])", " (", repaired)
        repaired = re.sub(r"\s+", " ", repaired).strip()
        return repaired

    @staticmethod
    def _is_noise_symbol_line(text: str) -> bool:
        s = str(text or "").strip()
        if not s:
            return True
        return bool(re.fullmatch(r"[\*\s\-_=+]{1,8}", s))

    def _axis_labels_to_lines(self, axis_labels: List[str], x_axis_text: str, y_axis_text: str, chart_type: str) -> List[str]:
        labels = [str(x).strip() for x in (axis_labels or []) if str(x).strip()]
        labels = [x for x in labels if not self._is_noise_symbol_line(x)]
        labels = self._dedup_texts(labels)
        if not labels:
            return []

        if chart_type in {"flowchart", "timeline", "process_diagram", "gantt_like"}:
            semantic = [x for x in labels if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", x)]
            return semantic[:8]

        numeric = []
        non_numeric = []
        for value in labels:
            if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value):
                try:
                    numeric.append(float(value))
                except Exception:
                    non_numeric.append(value)
            else:
                non_numeric.append(value)

        compressed: List[str] = []
        if numeric and len(numeric) >= 4 and len(non_numeric) <= 2:
            small = [v for v in numeric if abs(v) <= 20]
            large = [v for v in numeric if abs(v) > 20]
            if len(small) >= 3 and len(large) >= 3:
                compressed.append(f"X轴刻度范围：{int(min(small))} - {int(max(small))}")
                compressed.append(f"Y轴刻度范围：{int(min(large))} - {int(max(large))}")
                return compressed
            compressed.append(f"刻度范围：{min(numeric):g} - {max(numeric):g}")
            return compressed + non_numeric[:4]

        return labels[:10]

    def _dedup_texts(self, items: List[str]) -> List[str]:
        deduped: List[str] = []
        seen = set()
        canon_seen: List[str] = []
        for item in items:
            text = str(item or "").strip()
            if not text:
                continue
            if self._is_noise_symbol_line(text):
                continue

            text = self._repair_joined_english(text)
            key = self._normalize_text(text)
            if not key or key in seen:
                continue

            ckey = self._canonical_ocr_text(text)
            is_near_dup = False
            for existing in canon_seen:
                if ckey == existing:
                    is_near_dup = True
                    break
                if len(ckey) >= 12 and len(existing) >= 12:
                    sim = difflib.SequenceMatcher(None, ckey, existing).ratio()
                    if sim >= 0.95:
                        is_near_dup = True
                        break
            if is_near_dup:
                continue

            seen.add(key)
            canon_seen.append(ckey)
            deduped.append(text)
        return deduped

    def _filter_texts_not_in_table(self, items: List[str], markdown_table: str, known_texts: List[str]) -> List[str]:
        normalized_table = self._normalize_text(markdown_table)
        known = {self._normalize_text(text) for text in known_texts if str(text or "").strip()}
        kept: List[str] = []
        for item in self._dedup_texts(items):
            norm_item = self._normalize_text(item)
            if not norm_item:
                continue
            if self._contains_failure_marker(item):
                continue
            if self._is_footer_noise_text(item):
                continue
            if norm_item in known:
                continue
            if len(norm_item) <= 2:
                continue
            if norm_item in normalized_table:
                continue
            kept.append(item)
        return kept

    @staticmethod
    def _split_markdown_row(line: str) -> List[str]:
        s = str(line or "").strip()
        if not s or "|" not in s:
            return []
        if s.startswith("|"):
            s = s[1:]
        if s.endswith("|"):
            s = s[:-1]
        return [c.strip() for c in s.split("|")]

    @staticmethod
    def _is_markdown_separator_row(cells: List[str]) -> bool:
        if not cells:
            return False
        for cell in cells:
            c = str(cell or "").strip()
            if not c:
                return False
            if re.sub(r"[-:\s]", "", c) != "":
                return False
        return True

    @staticmethod
    def _looks_like_header_row(cells: List[str]) -> bool:
        if not cells:
            return False
        non_empty = [c for c in cells if str(c).strip()]
        if not non_empty:
            return False
        numeric_like = 0
        for c in non_empty:
            t = str(c).strip().replace(",", "")
            if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:%|[a-zA-Z]+)?", t):
                numeric_like += 1
        return (numeric_like / max(len(non_empty), 1)) <= 0.35

    @staticmethod
    def _expand_group_header(top_cells: List[str], sub_cells: List[str], col_count: int) -> List[str]:
        labels = [str(c or "").strip() for c in top_cells if str(c or "").strip()]
        if not labels:
            return [""] * col_count
        if len(labels) >= col_count:
            return labels[:col_count]

        expanded = [""] * col_count

        has_row_header_hint = bool(sub_cells and not str(sub_cells[0] or "").strip() and len(labels) >= 2)
        start_col = 0
        if has_row_header_hint:
            expanded[0] = labels[0]
            labels = labels[1:]
            start_col = 1

        remaining_cols = max(col_count - start_col, 0)
        if remaining_cols == 0:
            return expanded

        if len(labels) == 1:
            for i in range(start_col, col_count):
                expanded[i] = labels[0]
            return expanded

        base = remaining_cols // len(labels)
        rem = remaining_cols % len(labels)

        cursor = start_col
        for i, label in enumerate(labels):
            span = base + (1 if i < rem else 0)
            span = max(span, 1)
            for _ in range(span):
                if cursor >= col_count:
                    break
                expanded[cursor] = label
                cursor += 1

        # 若仍有空位，用最后一个标签补齐
        if labels:
            last = labels[-1]
            for i in range(col_count):
                if not expanded[i]:
                    expanded[i] = last

        return expanded

    def _normalize_markdown_table(self, markdown_table: str) -> str:
        text = str(markdown_table or "").strip()
        if not text:
            return ""

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        table_lines = [ln for ln in lines if "|" in ln]
        if len(table_lines) < 2:
            return text

        raw_rows = [self._split_markdown_row(ln) for ln in table_lines]
        raw_rows = [r for r in raw_rows if r]
        if len(raw_rows) < 2:
            return text

        sep_idx = -1
        for i, row in enumerate(raw_rows):
            if self._is_markdown_separator_row(row):
                sep_idx = i
                break

        if sep_idx <= 0:
            header = raw_rows[0]
            data_rows = raw_rows[1:]
        else:
            header = raw_rows[0]
            data_rows = raw_rows[sep_idx + 1:]

        all_rows = [header] + data_rows
        col_count = max(len(r) for r in all_rows) if all_rows else len(header)
        if col_count <= 0:
            return text

        def pad(row: List[str]) -> List[str]:
            return row + [""] * (col_count - len(row))

        header = pad(header)
        data_rows = [pad(r) for r in data_rows if any(str(c).strip() for c in r)]

        # 多级表头展平：首个数据行疑似“子表头”且主表头列数明显不足
        if data_rows:
            first_data = data_rows[0]
            top_non_empty = sum(1 for c in header if str(c).strip())
            second_non_empty = sum(1 for c in first_data if str(c).strip())
            if top_non_empty < col_count and second_non_empty >= max(2, col_count - 1) and self._looks_like_header_row(first_data):
                group = self._expand_group_header(header, first_data, col_count)
                merged: List[str] = []
                for i in range(col_count):
                    g = str(group[i] or "").strip()
                    s = str(first_data[i] or "").strip()
                    if s and g and self._normalize_text(s) != self._normalize_text(g):
                        merged.append(f"{s} ({g})")
                    elif s:
                        merged.append(s)
                    elif g:
                        merged.append(g)
                    else:
                        merged.append(f"Col{i+1}")
                header = merged
                data_rows = data_rows[1:]

        if not any(str(c).strip() for c in header):
            header = [f"Col{i+1}" for i in range(col_count)]

        out = []
        out.append("| " + " | ".join((str(c).replace("|", "\\|").strip()) for c in header) + " |")
        out.append("|" + "|".join([" --- " for _ in range(col_count)]) + "|")
        for row in data_rows:
            out.append("| " + " | ".join((str(c).replace("|", "\\|").strip()) for c in row) + " |")

        return "\n".join(out)

    def _is_redundant_table_title(self, table_title: str, markdown_table: str) -> bool:
        title = str(table_title or "").strip()
        if not title:
            return False
        norm_title = self._normalize_text(title)
        if not norm_title:
            return False

        lines = [ln.strip() for ln in str(markdown_table or "").splitlines() if ln.strip() and "|" in ln]
        row_cells = [self._split_markdown_row(ln) for ln in lines]
        row_cells = [r for r in row_cells if r]
        candidates: List[str] = []
        for row in row_cells[:2]:
            for cell in row:
                c = str(cell or "").strip()
                if c:
                    candidates.append(c)

        for c in candidates:
            n = self._normalize_text(c)
            if not n:
                continue
            if norm_title == n or norm_title in n or n in norm_title:
                return True
        return False
    
    def _build_block_elements(self,
                             group: SemanticGroup,
                             extracted_data: ExtractedDataMap,
                             id_to_text_map: Dict[str, str],
                             elem_map: Dict[str, DetectedElement]) -> Dict[str, Any]:
        """构建块内元素内容"""
        elements = {}
        
        # 1. 基础 OCR 文本收集 (仅作备用，不再盲目全拼接到 text_content)
        raw_ocr_texts = []
        for mid in group.member_ids:
            txt = id_to_text_map.get(mid, "").strip()
            if txt:
                raw_ocr_texts.append(txt)
        fallback_text = "\n".join(raw_ocr_texts)
        
        # 2. 收集所有提取内容（支持多图表/多表格）
        collected_charts = []
        collected_tables = []
        primary_content = extracted_data.extractions.get(group.primary_element_id)
        
        extraction_success = False
        
        # 遍历组内所有成员，收集提取结果
        for mid in group.member_ids:
            content = extracted_data.extractions.get(mid)
            if not content or content.processing_status != "success":
                continue
                
            extraction_success = True

            # 混合内容（一个ROI内包含多图/图+表）优先处理
            if content.mixed_data:
                if content.mixed_data.summary and "mixed_summary" not in elements and not self._contains_failure_marker(content.mixed_data.summary):
                    elements["mixed_summary"] = content.mixed_data.summary
                if content.mixed_data.validation_notes:
                    elements.setdefault("mixed_notes", content.mixed_data.validation_notes)
                if content.mixed_data.text_items:
                    elements.setdefault("mixed_external_texts", [])
                    elements["mixed_external_texts"].extend(content.mixed_data.text_items)

                for idx, chart_item in enumerate(content.mixed_data.chart_items):
                    chart_info = {
                        "element_id": f"{mid}#chart_{idx+1}",
                        "data": {
                            "panel_name": chart_item.panel_name,
                            "type": chart_item.chart_type,
                            "x_axis": chart_item.x_axis,
                            "y_axis": chart_item.y_axis,
                            "insight": chart_item.insight,
                            "key_values": chart_item.key_values,
                            "chart_title": chart_item.chart_title,
                            "panel_titles": chart_item.panel_titles,
                            "legend_items": chart_item.legend_items,
                            "axis_labels": chart_item.axis_labels,
                            "annotation_texts": chart_item.annotation_texts,
                            "footnotes": chart_item.footnotes,
                        },
                    }
                    if not self._is_chart_payload_valid(chart_info["data"]):
                        continue
                    if mid in elem_map:
                        chart_info["image_path"] = f"roi_crops/{mid}_roi.png"
                    collected_charts.append(chart_info)

                for idx, table_item in enumerate(content.mixed_data.table_items):
                    if not self._is_table_payload_valid(table_item):
                        continue
                    table_info = {
                        "element_id": f"{mid}#table_{idx+1}",
                        "markdown": table_item.markdown_table,
                        "insight": getattr(table_item, 'insight', ''),
                        "key_values": getattr(table_item, 'key_values', {}),
                        "table_title": table_item.table_title,
                        "external_texts": table_item.external_texts,
                        "footnotes": table_item.footnotes,
                        "notes": table_item.notes,
                        "internal_headers": table_item.internal_headers,
                    }
                    if mid in elem_map:
                        table_info["image_path"] = f"roi_crops/{mid}_roi.png"
                    collected_tables.append(table_info)
                continue
            
            # 收集图表数据
            if content.chart_data:
                chart_info = {
                    "element_id": mid,
                    "data": {
                        "panel_name": content.chart_data.panel_name,
                        "type": content.chart_data.chart_type,
                        "x_axis": content.chart_data.x_axis,
                        "y_axis": content.chart_data.y_axis,
                        "insight": content.chart_data.insight,
                        "key_values": content.chart_data.key_values,
                        "chart_title": content.chart_data.chart_title,
                        "panel_titles": content.chart_data.panel_titles,
                        "legend_items": content.chart_data.legend_items,
                        "axis_labels": content.chart_data.axis_labels,
                        "annotation_texts": content.chart_data.annotation_texts,
                        "footnotes": content.chart_data.footnotes,
                    }
                }
                if not self._is_chart_payload_valid(chart_info["data"]):
                    chart_info = None
                
                # 检查图片路径
                if chart_info and mid in elem_map:
                    # 使用相对路径
                    chart_info["image_path"] = f"roi_crops/{mid}_roi.png"
                
                if chart_info:
                    collected_charts.append(chart_info)
            
            # 收集表格数据
            if content.table_data:
                table_info = {
                    "element_id": mid,
                    "markdown": content.table_data.markdown_table,
                    "insight": getattr(content.table_data, 'insight', ''),
                    "key_values": getattr(content.table_data, 'key_values', {}),
                    "table_title": content.table_data.table_title,
                    "external_texts": content.table_data.external_texts,
                    "footnotes": content.table_data.footnotes,
                    "notes": content.table_data.notes,
                    "internal_headers": content.table_data.internal_headers,
                }
                if mid in elem_map:
                    table_info["image_path"] = f"roi_crops/{mid}_roi.png"
                collected_tables.append(table_info)

            # 收集文本数据 (主要是TextProcessor的结果)
            if content.text_data:
                # 文本通常归并，这里只记录主要的
                if mid == group.primary_element_id:
                    elements["text_content"] = content.text_data.merged_text
        
        # 填充集合数据
        if collected_charts:
            elements["charts"] = collected_charts
            primary_chart = next((c for c in collected_charts if c["element_id"] == group.primary_element_id), collected_charts[0])
            elements["chart_data"] = primary_chart["data"]
            if "image_path" in primary_chart:
                elements["image_path"] = primary_chart["image_path"]
        
        if collected_tables:
            elements["tables"] = collected_tables
            primary_table = next((t for t in collected_tables if t["element_id"] == group.primary_element_id), collected_tables[0])
            elements["table_data"] = {
                "markdown": primary_table["markdown"],
                "insight": primary_table.get("insight", ""),
                "key_values": primary_table.get("key_values", {}),
                "table_title": primary_table.get("table_title", ""),
                "external_texts": primary_table.get("external_texts", []),
                "footnotes": primary_table.get("footnotes", []),
                "notes": primary_table.get("notes", []),
                "internal_headers": primary_table.get("internal_headers", []),
            }

        # 3. 兜底文本逻辑与标题智能推断
        if not extraction_success:
            # 这是一个纯文本组（或者提取失败的视觉组）
            if "text_content" not in elements:
                # 启发式寻找真正的“小标题”：通常是组内最短的那句话
                texts_with_ids = [(mid, id_to_text_map.get(mid, "").strip()) for mid in group.member_ids if id_to_text_map.get(mid, "").strip()]
                
                if not texts_with_ids:
                    pass
                elif len(texts_with_ids) == 1:
                    # 只有一个文本块，直接作为正文
                    elements["text_content"] = texts_with_ids[0][1]
                else:
                    # 有多个文本块，尝试分离标题和正文
                    # 按照文本长度排序，找到最短的
                    shortest_text_tuple = min(texts_with_ids, key=lambda x: len(x[1]))
                    
                    # 如果最短的文本不超过 30 个字，我们认为它是标题
                    if len(shortest_text_tuple[1]) < 30:
                        elements["title"] = shortest_text_tuple[1]
                        # 剩下的全部拼接成正文，不会再包含刚刚提取的标题
                        body_texts = [txt for mid, txt in texts_with_ids if mid != shortest_text_tuple[0]]
                        elements["text_content"] = "\n".join(body_texts)
                    else:
                        # 全都很长，没有明显的标题特征，直接全部拼接作为正文
                        elements["text_content"] = "\n".join([txt for mid, txt in texts_with_ids])
            
            elements["is_fallback"] = True
        else:
            # 如果是视觉元素组（图/表）且高级提取成功，尝试获取辅助标题
            if "title" not in elements:
                for sec_id in group.secondary_element_ids:
                    text = id_to_text_map.get(sec_id, "").strip()
                    if text and len(text) < 50: # 避免把长正文当做标题
                        elements["title"] = text
                        break
        
        # 4. 添加图片引用 (兜底：如果还未设置image_path且它是视觉元素)
        if "image_path" not in elements:
            primary_id = group.primary_element_id
            if primary_id and primary_id in elem_map:
                elem = elem_map[primary_id]
                e_type = elem.original_type.lower()
                crop_targets = ["image", "table", "chart", "figure", "graph"]
                should_have_image = any(t in e_type for t in crop_targets)
                
                if should_have_image:
                    elements["image_path"] = f"roi_crops/{primary_id}_roi.png"

        elements["semantic_desc"] = group.semantic_desc

        if "mixed_external_texts" in elements:
            filtered_mixed_texts = []
            for txt in self._dedup_texts(elements.get("mixed_external_texts", [])):
                if self._contains_failure_marker(txt):
                    continue
                if self._is_footer_noise_text(txt):
                    continue
                filtered_mixed_texts.append(txt)
            elements["mixed_external_texts"] = filtered_mixed_texts
        
        return elements
    
    def _build_structured_json(self,
                              page_id: int,
                              layout_confidence: str,
                              semantic_blocks: List[SemanticBlock],
                              page_title: str) -> FinalStructuredJSON:
        """构建最终结构化JSON"""
        metadata = {
            "page_title": page_title,
            "block_count": len(semantic_blocks),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return FinalStructuredJSON(
            page_id=page_id,
            layout_confidence=layout_confidence,
            semantic_blocks=semantic_blocks,
            metadata=metadata
        )
    
    def _build_markdown(self,
                       page_id: int,
                       page_title: str,
                       semantic_blocks: List[SemanticBlock],
                       extracted_data: ExtractedDataMap) -> FinalMarkdownOutput:
        """构建最终Markdown输出"""
        md_lines = []
        image_refs = []
        table_count = 0
        chart_count = 0
        
        # 页面标题
        if page_title:
            md_lines.append(f"# {page_title}")
        else:
            md_lines.append(f"# 页面 {page_id}")
        md_lines.append("")

        seen_signatures = set()
        
        # 按阅读顺序输出各语义块
        for block in sorted(semantic_blocks, key=lambda b: b.reading_order):
            signature = self._block_render_signature(block)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            block_md, block_images, block_tables, block_charts = self._block_to_markdown(block)
            md_lines.extend(block_md)
            image_refs.extend(block_images)
            table_count += block_tables
            chart_count += block_charts
        
        markdown_content = "\n".join(md_lines)
        
        return FinalMarkdownOutput(
            page_id=page_id,
            title=page_title,
            markdown_content=markdown_content,
            image_references=image_refs,
            table_count=table_count,
            chart_count=chart_count
        )
    
    def _block_to_markdown(self, block: SemanticBlock) -> tuple[List[str], List[str], int, int]:
        """将语义块转换为Markdown"""
        md_lines = []
        image_refs = []
        table_count = 0
        chart_count = 0
        rendered_images = set()
        
        elements = block.elements

        has_tables = bool(elements.get("tables"))

        def add_image_once(image_path: str) -> None:
            if not image_path or image_path in rendered_images:
                return
            img_filename = os.path.basename(image_path)
            md_lines.append(f"![{block.block_type}](roi_crops/{img_filename})")
            md_lines.append("")
            image_refs.append(image_path)
            rendered_images.add(image_path)
        
        # 标题
        if elements.get("title"):
            md_lines.append(f"## {elements['title']}")
            md_lines.append("")

        # mixed 组整体洞察（优先于子图明细展示）
        if elements.get("mixed_summary"):
            md_lines.append(f"> **整体洞察**: {elements['mixed_summary']}")
            md_lines.append("")

        if elements.get("mixed_external_texts"):
            mixed_external = self._dedup_texts(elements.get("mixed_external_texts", []))
            if mixed_external:
                md_lines.append("**版面说明文字:**")
                for text in mixed_external:
                    md_lines.append(f"- {text}")
                md_lines.append("")
            
        # ====================================================================
        # 新逻辑：优先处理列表形式的内容 (charts, tables)
        # ====================================================================
        
        # 1. 处理图表列表
        if "charts" in elements:
            seen_chart_items = set()
            for chart_info in elements["charts"]:
                # 数据分析
                chart_data = chart_info.get("data", {})
                if not self._is_chart_payload_valid(chart_data):
                    continue
                chart_type = str(chart_data.get("type", "")).strip().lower()
                panel_name = str(chart_data.get("panel_name", "")).strip()
                # 如果同块存在表格，过滤掉 table-like chart，避免重复解释
                if has_tables and ("table" in chart_type or chart_type in {"grid", "matrix"}):
                    continue

                chart_key = (
                    panel_name.lower(),
                    chart_type,
                    self._axis_to_text(chart_data.get("x_axis")),
                    self._axis_to_text(chart_data.get("y_axis")),
                    str(chart_data.get("insight", "")).strip(),
                    json.dumps(chart_data.get("key_values", {}), ensure_ascii=False, sort_keys=True),
                )
                if chart_key in seen_chart_items:
                    continue
                seen_chart_items.add(chart_key)

                chart_count += 1

                # 图片
                if "image_path" in chart_info:
                    add_image_once(chart_info["image_path"])

                if panel_name:
                    md_lines.append(f"**{panel_name}**")
                    md_lines.append("")
                
                insight = chart_data.get("insight", "")
                if insight:
                    md_lines.append(f"> **洞察**: {insight}")
                    md_lines.append("")

                chart_title = str(chart_data.get("chart_title", "") or "").strip()
                panel_titles = self._dedup_texts(chart_data.get("panel_titles", []) if isinstance(chart_data.get("panel_titles", []), list) else [])
                legend_items = self._dedup_texts(chart_data.get("legend_items", []) if isinstance(chart_data.get("legend_items", []), list) else [])
                axis_labels = self._dedup_texts(chart_data.get("axis_labels", []) if isinstance(chart_data.get("axis_labels", []), list) else [])
                annotation_texts = self._dedup_texts(chart_data.get("annotation_texts", []) if isinstance(chart_data.get("annotation_texts", []), list) else [])
                footnotes = self._dedup_texts(chart_data.get("footnotes", []) if isinstance(chart_data.get("footnotes", []), list) else [])
                x_axis_text = self._axis_to_text(chart_data.get("x_axis"))
                y_axis_text = self._axis_to_text(chart_data.get("y_axis"))
                axis_lines = self._axis_labels_to_lines(axis_labels, x_axis_text, y_axis_text, chart_type)

                if chart_title or panel_titles:
                    md_lines.append("**图标题/面板标题:**")
                    if chart_title:
                        md_lines.append(f"- {chart_title}")
                    for line in panel_titles:
                        if line != chart_title:
                            md_lines.append(f"- {line}")
                    md_lines.append("")

                if legend_items:
                    md_lines.append("**图例:**")
                    for line in legend_items:
                        md_lines.append(f"- {line}")
                    md_lines.append("")

                if axis_lines:
                    md_lines.append("**坐标与标签:**")
                    for line in axis_lines:
                        md_lines.append(f"- {line}")
                    md_lines.append("")

                if annotation_texts:
                    md_lines.append("**图中说明文字:**")
                    for line in annotation_texts:
                        md_lines.append(f"- {line}")
                    md_lines.append("")

                if footnotes:
                    md_lines.append("**图下注释/脚注:**")
                    for line in footnotes:
                        md_lines.append(f"- {line}")
                    md_lines.append("")
                
                # 元数据
                md_lines.append(f"*图表类型: {chart_data.get('type', '未知')}*")
                if x_axis_text:
                    md_lines.append(f"*X轴: {x_axis_text}*")
                if y_axis_text:
                    md_lines.append(f"*Y轴: {y_axis_text}*")
                md_lines.append("")
                
                # 关键数据
                key_values = chart_data.get("key_values", {})
                if key_values:
                    md_lines.append("**关键数据:**")
                    for k, v in key_values.items():
                        md_lines.append(f"- {k}: {v}")
                    md_lines.append("")
        
        # 2. 处理表格列表
        if "tables" in elements:
            seen_tables = set()
            for table_info in elements["tables"]:
                table_key = (
                    str(table_info.get("markdown", "")).strip(),
                    str(table_info.get("insight", "")).strip(),
                )
                if table_key in seen_tables:
                    continue
                seen_tables.add(table_key)

                table_count += 1

                # 图片（表格裁剪图）
                if "image_path" in table_info:
                    add_image_once(table_info["image_path"])
                
                if table_info.get("insight"):
                    md_lines.append(f"> **洞察**: {table_info['insight']}")
                    md_lines.append("")
                
                table_key_values = table_info.get("key_values", {})
                if table_key_values:
                    md_lines.append("**关键数据:**")
                    for k, v in table_key_values.items():
                        md_lines.append(f"- {k}: {v}")
                    md_lines.append("")

                md_table = self._normalize_markdown_table(table_info.get("markdown", ""))
                table_title = str(table_info.get("table_title", "") or "").strip()
                if self._is_redundant_table_title(table_title, md_table):
                    table_title = ""
                external_texts = table_info.get("external_texts", []) if isinstance(table_info.get("external_texts", []), list) else []
                footnotes = table_info.get("footnotes", []) if isinstance(table_info.get("footnotes", []), list) else []
                notes = table_info.get("notes", []) if isinstance(table_info.get("notes", []), list) else []

                external_texts = self._filter_texts_not_in_table(
                    external_texts,
                    markdown_table=md_table,
                    known_texts=[table_info.get("insight", ""), table_title],
                )
                footnotes = self._filter_texts_not_in_table(
                    footnotes,
                    markdown_table=md_table,
                    known_texts=[table_info.get("insight", ""), table_title],
                )
                notes = self._filter_texts_not_in_table(
                    notes,
                    markdown_table=md_table,
                    known_texts=[table_info.get("insight", ""), table_title],
                )

                if table_title:
                    md_lines.append(f"**表题/表头说明:** {table_title}")
                if external_texts:
                    md_lines.append("**表格外部说明文字:**")
                    for line in external_texts:
                        md_lines.append(f"- {line}")
                    md_lines.append("")
                if footnotes:
                    md_lines.append("**表注/脚注:**")
                    for line in footnotes:
                        md_lines.append(f"- {line}")
                    md_lines.append("")
                if notes:
                    md_lines.append("**表格补充说明:**")
                    for line in notes:
                        md_lines.append(f"- {line}")
                    md_lines.append("")

                if md_table:
                    md_lines.append(md_table)
                    md_lines.append("")

        # ====================================================================
        # 旧逻辑兼兜底：如果没有检测到列表形式，回退到处理单元素
        # (通常发生在非聚合提取或旧版本数据中)
        # ====================================================================
        
        handled_via_list = ("charts" in elements) or ("tables" in elements)
        
        if not handled_via_list:
            # 优先展示图表/图片
            if "image_path" in elements:
                add_image_once(elements["image_path"])
            
            # 图表内容
            if "chart_data" in elements:
                chart_count += 1
                chart = elements["chart_data"]
                
                # 洞察引用块
                insight = chart.get("insight", "")
                if insight:
                    md_lines.append(f"> **洞察**: {insight}")
                    md_lines.append("")
                
                # 图表元数据
                md_lines.append(f"*图表类型: {chart.get('type', '未知')}*")
                x_axis_text = self._axis_to_text(chart.get("x_axis"))
                y_axis_text = self._axis_to_text(chart.get("y_axis"))
                if x_axis_text:
                    md_lines.append(f"*X轴: {x_axis_text}*")
                if y_axis_text:
                    md_lines.append(f"*Y轴: {y_axis_text}*")
                md_lines.append("")
                
                # 关键数值
                key_values = chart.get("key_values", {})
                if key_values:
                    md_lines.append("**关键数据:**")
                    for k, v in key_values.items():
                        md_lines.append(f"- {k}: {v}")
                    md_lines.append("")
            
            # 表格内容
            if "table_data" in elements:
                table_count += 1
                table = elements["table_data"]
                
                # 表格分析
                if table.get("insight"):
                    md_lines.append(f"> **洞察**: {table['insight']}")
                    md_lines.append("")
                
                table_key_values = table.get("key_values", {})
                if table_key_values:
                    md_lines.append("**关键数据:**")
                    for k, v in table_key_values.items():
                        md_lines.append(f"- {k}: {v}")
                    md_lines.append("")
                
                # Markdown表格
                md_table = self._normalize_markdown_table(table.get("markdown", ""))
                if md_table:
                    md_lines.append(md_table)
                    md_lines.append("")
        
        # --------------------------------------------------------------------
        # 通用内容 (无论是否列表处理，都可能存在的其余部分)
        # --------------------------------------------------------------------

        # 公式内容
        if "formula_data" in elements:
            formula = elements["formula_data"]
            latex = formula.get("latex", "")
            if latex:
                md_lines.append(f"$$")
                md_lines.append(latex)
                md_lines.append(f"$$")
                md_lines.append("")
        
        # 文本内容
        if "text_content" in elements:
            text = elements["text_content"]
            if text:
                md_lines.append(self._repair_joined_english(str(text)))
                md_lines.append("")

        # 说明文字
        caption = elements.get("caption", "")
        if caption:
            md_lines.append(f"*{caption}*")
            md_lines.append("")
        
        # 分隔线
        md_lines.append("---")
        md_lines.append("")
        
        return md_lines, image_refs, table_count, chart_count

    def _block_render_signature(self, block: SemanticBlock) -> str:
        """生成块级渲染签名，用于跨块去重。"""
        elements = block.elements or {}
        image_paths: List[str] = []

        for chart_info in elements.get("charts", []) or []:
            path = chart_info.get("image_path")
            if path:
                image_paths.append(path)

        for table_info in elements.get("tables", []) or []:
            path = table_info.get("image_path")
            if path:
                image_paths.append(path)

        if elements.get("image_path"):
            image_paths.append(elements.get("image_path"))

        image_paths = sorted(set(image_paths))
        if image_paths:
            return "img:" + "|".join(image_paths)

        fallback = {
            "block_type": block.block_type,
            "title": elements.get("title", ""),
            "semantic_desc": elements.get("semantic_desc", ""),
            "reading_order": block.reading_order,
        }
        return "fallback:" + json.dumps(fallback, ensure_ascii=False, sort_keys=True)
    
    def _compute_union_bbox(self, bboxes) -> Optional[List[int]]:
        """计算联合边界框"""
        if not bboxes:
            return None
        
        ymin = min(b.ymin for b in bboxes)
        xmin = min(b.xmin for b in bboxes)
        ymax = max(b.ymax for b in bboxes)
        xmax = max(b.xmax for b in bboxes)
        
        return [ymin, xmin, ymax, xmax]
    
    def _determine_block_type(self, 
                             group: SemanticGroup, 
                             content: Optional[ExtractedContent]) -> str:
        """确定块类型"""
        if content:
            if content.mixed_data:
                has_charts = any(
                    self._is_chart_payload_valid({
                        "type": getattr(item, "chart_type", ""),
                        "x_axis": getattr(item, "x_axis", ""),
                        "y_axis": getattr(item, "y_axis", ""),
                        "key_values": getattr(item, "key_values", {}),
                        "insight": getattr(item, "insight", ""),
                        "chart_title": getattr(item, "chart_title", ""),
                        "panel_titles": getattr(item, "panel_titles", []),
                        "legend_items": getattr(item, "legend_items", []),
                        "axis_labels": getattr(item, "axis_labels", []),
                        "annotation_texts": getattr(item, "annotation_texts", []),
                        "footnotes": getattr(item, "footnotes", []),
                    })
                    for item in (content.mixed_data.chart_items or [])
                )
                has_tables = any(self._is_table_payload_valid(item) for item in (content.mixed_data.table_items or []))
                if has_charts and has_tables:
                    return "mixed_group"
                if has_tables:
                    return "table_group"
                if has_charts:
                    return "chart_group"
                return "mixed_group"
            if content.table_data:
                return "table_group"
            elif content.chart_data:
                return "chart_group"
            elif content.formula_data:
                return "formula_group"
            elif content.text_data:
                return "text_block"
        
        return group.group_type
    
    def _determine_role(self,
                       group: SemanticGroup,
                       content: Optional[ExtractedContent]) -> str:
        """确定块角色"""
        if content:
            if content.mixed_data or content.chart_data or content.table_data:
                return "primary_content"
            elif content.text_data:
                return "supporting"
        
        return "content"
