"""
nodes/step2_locator.py
Step 2.5: 负责为元素获取精确坐标 (BBox) 的组件
"""

import json
import os
import base64
from pathlib import Path
from openai import AzureOpenAI
from state import PPTPageState, PageElement, BBox, LogicalGroup, ElementInsight
from vlm_client import VLMClient
from typing import List, Dict, Any, Tuple
from tools.visual_marker import mark_image_with_bboxes


_DECORATIVE_KWS = (
    "背景", "底纹", "装饰", "边框", "分割线", "页眉", "页脚", "页码", "水印",
    "logo", "LOGO", "Icon", "icon", "ICON",
)


def _is_decorative_candidate(el: PageElement) -> bool:
    desc = (getattr(el, "description", "") or "").strip()
    if desc and any(k in desc for k in _DECORATIVE_KWS):
        return True
    return False


def _bbox_is_unreasonable(box_2d: list[int], cfg) -> bool:
    """基于 0-1000 归一化坐标的 bbox 规则过滤。

    统一使用 [ymin, xmin, ymax, xmax]。
    """
    if not isinstance(box_2d, list) or len(box_2d) != 4:
        return False
    ymin, xmin, ymax, xmax = [int(x) for x in box_2d]
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    if w <= 0 or h <= 0:
        return True

    area_ratio = (w * h) / float(1000 * 1000)
    if area_ratio < float(getattr(cfg, "bbox_min_area_ratio", 0.005)):
        return True
    if area_ratio > float(getattr(cfg, "bbox_max_area_ratio", 0.60)):
        return True

    aspect = (w / h) if h > 0 else 9999.0
    extreme = float(getattr(cfg, "bbox_extreme_aspect_ratio", 10.0))
    if aspect > extreme or aspect < (1.0 / extreme):
        return True

    # 贴边且很薄：页眉/页脚/边框/分割线
    edge_margin = float(getattr(cfg, "bbox_edge_margin_ratio", 0.02))
    thin_ratio = float(getattr(cfg, "bbox_edge_thin_ratio", 0.05))
    near_edge = (
        xmin < 1000 * edge_margin or ymin < 1000 * edge_margin or
        xmax > 1000 * (1 - edge_margin) or ymax > 1000 * (1 - edge_margin)
    )
    thin = (w < 1000 * thin_ratio) or (h < 1000 * thin_ratio)
    if near_edge and thin:
        return True

    # 贴边大条幅：常见的顶部/底部背景横条或左右侧边栏（即便不“很薄”也应剔除）
    # 仅基于几何启发式：接近全宽(或全高) + 靠边 + 相对较短
    if w >= int(1000 * 0.92) and h <= int(1000 * 0.25) and (ymin <= int(1000 * 0.05) or ymax >= int(1000 * 0.95)):
        return True
    if h >= int(1000 * 0.92) and w <= int(1000 * 0.25) and (xmin <= int(1000 * 0.05) or xmax >= int(1000 * 0.95)):
        return True

    return False


def _bbox_iou(a: list[int], b: list[int]) -> float:
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _safe_text(s: str) -> str:
    return (s or "").strip()


def _text_similarity(a: str, b: str) -> float:
    a = _safe_text(a)
    b = _safe_text(b)
    if not a or not b:
        return 0.0
    try:
        from difflib import SequenceMatcher
        return float(SequenceMatcher(None, a, b).ratio())
    except Exception:
        return 0.0


def _load_xml_text_safe(state: PPTPageState) -> str:
    try:
        from nodes.step2_workers import convert_slide_to_markdown
        return _safe_text(convert_slide_to_markdown(state.ppt_path, state.page_index))
    except Exception:
        return ""


def _normalize_elements(elements: List[PageElement], cfg) -> tuple[List[PageElement], List[PageElement]]:
    """统一执行去噪+去重；返回 (保留元素, 被聚合的文本元素)。"""
    cleaned: List[PageElement] = []
    text_candidates: List[PageElement] = []

    for el in elements or []:
        el_type = (getattr(el, "type", "") or "").strip().lower()
        desc = _safe_text(getattr(el, "description", "") or "")
        box = getattr(getattr(el, "bbox", None), "box_2d", None)

        # 丢弃显式无效类型
        if el_type in {"discarded", "noise", "background"}:
            continue

        # 文本元素先收集，后续统一聚合
        is_text_like = any(k in el_type for k in ["text", "title", "caption"])

        # 装饰关键词过滤
        if _is_decorative_candidate(el):
            continue

        if box and _bbox_is_unreasonable(box, cfg):
            continue

        # 基于位置的页脚/角标规则（需要 bbox）
        if box and len(box) == 4:
            ymin, xmin, ymax, xmax = [int(v) for v in box]
            w = max(0, xmax - xmin)
            h = max(0, ymax - ymin)
            area_ratio = (w * h) / float(1000 * 1000)
            footer_ratio = float(getattr(cfg, "element_footer_zone_ratio", 0.10))
            corner_ratio = float(getattr(cfg, "element_corner_zone_ratio", 0.12))
            corner_max_area = float(getattr(cfg, "element_corner_max_area_ratio", 0.02))
            in_footer = ymin >= int(1000 * (1 - footer_ratio))
            in_left = xmax <= int(1000 * corner_ratio)
            in_right = xmin >= int(1000 * (1 - corner_ratio))
            in_top = ymax <= int(1000 * corner_ratio)
            in_bottom = ymin >= int(1000 * (1 - corner_ratio))
            in_corner = (in_left or in_right) and (in_top or in_bottom)

            if in_footer and h <= int(1000 * 0.12):
                continue
            if in_corner and area_ratio <= corner_max_area:
                continue

        # 短文本噪声
        min_len = int(getattr(cfg, "element_text_min_len", 3))
        if is_text_like and len(desc) < min_len:
            continue

        if is_text_like:
            text_candidates.append(el)
        else:
            cleaned.append(el)

    # 去重（同类元素 + IoU 高）
    deduped: List[PageElement] = []
    iou_th = float(getattr(cfg, "element_dedup_iou_threshold", 0.85))
    for cur in cleaned:
        cur_box = getattr(getattr(cur, "bbox", None), "box_2d", None)
        cur_type = (getattr(cur, "type", "") or "").lower()
        duplicate = False
        if cur_box and len(cur_box) == 4:
            for prev in deduped:
                prev_box = getattr(getattr(prev, "bbox", None), "box_2d", None)
                prev_type = (getattr(prev, "type", "") or "").lower()
                if prev_box and len(prev_box) == 4 and prev_type == cur_type:
                    if _bbox_iou(prev_box, cur_box) >= iou_th:
                        duplicate = True
                        break
        if not duplicate:
            deduped.append(cur)

    return deduped, text_candidates


def _canonical_dispatch_type(raw_type: str) -> str:
    t = (raw_type or "").strip().lower()
    if any(k in t for k in ["table", "grid", "matrix"]):
        return "table"
    if any(k in t for k in ["chart", "diagram", "flow", "plot", "graph"]):
        return "chart"
    if any(k in t for k in ["text", "title", "caption", "header", "footer", "paragraph", "bullet"]):
        return "text"
    if any(k in t for k in ["image", "picture", "photo", "figure", "icon", "logo"]):
        return "image"
    return "image"


def _clean_elements_no_text_split(elements: List[PageElement], cfg) -> List[PageElement]:
    """执行基础去噪/去重，但不提前剔除 text。"""
    cleaned: List[PageElement] = []

    for el in elements or []:
        el_type = (getattr(el, "type", "") or "").strip().lower()
        desc = _safe_text(getattr(el, "description", "") or "")
        box = getattr(getattr(el, "bbox", None), "box_2d", None)

        if el_type in {"discarded", "noise", "background"}:
            continue

        if _is_decorative_candidate(el):
            continue

        if box and _bbox_is_unreasonable(box, cfg):
            continue

        if box and len(box) == 4:
            ymin, xmin, ymax, xmax = [int(v) for v in box]
            w = max(0, xmax - xmin)
            h = max(0, ymax - ymin)
            area_ratio = (w * h) / float(1000 * 1000)
            footer_ratio = float(getattr(cfg, "element_footer_zone_ratio", 0.10))
            corner_ratio = float(getattr(cfg, "element_corner_zone_ratio", 0.12))
            corner_max_area = float(getattr(cfg, "element_corner_max_area_ratio", 0.02))
            in_footer = ymin >= int(1000 * (1 - footer_ratio))
            in_left = xmax <= int(1000 * corner_ratio)
            in_right = xmin >= int(1000 * (1 - corner_ratio))
            in_top = ymax <= int(1000 * corner_ratio)
            in_bottom = ymin >= int(1000 * (1 - corner_ratio))
            in_corner = (in_left or in_right) and (in_top or in_bottom)

            if in_footer and h <= int(1000 * 0.12):
                continue
            if in_corner and area_ratio <= corner_max_area:
                continue

        min_len = int(getattr(cfg, "element_text_min_len", 3))
        if _canonical_dispatch_type(el_type) == "text" and len(desc) < min_len:
            continue

        cleaned.append(el)

    deduped: List[PageElement] = []
    iou_th = float(getattr(cfg, "element_dedup_iou_threshold", 0.85))
    for cur in cleaned:
        cur_box = getattr(getattr(cur, "bbox", None), "box_2d", None)
        cur_type = _canonical_dispatch_type(getattr(cur, "type", ""))
        duplicate = False
        if cur_box and len(cur_box) == 4:
            for prev in deduped:
                prev_box = getattr(getattr(prev, "bbox", None), "box_2d", None)
                prev_type = _canonical_dispatch_type(getattr(prev, "type", ""))
                if prev_box and len(prev_box) == 4 and prev_type == cur_type:
                    if _bbox_iou(prev_box, cur_box) >= iou_th:
                        duplicate = True
                        break
        if not duplicate:
            deduped.append(cur)

    return deduped


def _bbox_area(box: List[int]) -> int:
    y1, x1, y2, x2 = [int(v) for v in box]
    return max(0, x2 - x1) * max(0, y2 - y1)


def _bbox_contains(outer: List[int], inner: List[int], margin: int = 6) -> bool:
    oy1, ox1, oy2, ox2 = [int(v) for v in outer]
    iy1, ix1, iy2, ix2 = [int(v) for v in inner]
    return (
        iy1 >= oy1 - margin
        and ix1 >= ox1 - margin
        and iy2 <= oy2 + margin
        and ix2 <= ox2 + margin
    )


def _encode_image_base64(image_path: str) -> str:
    try:
        from nodes.step1_global_analysis import ImageUtils
        return ImageUtils.encode_image(image_path)
    except Exception:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


def _correct_types_with_page_vlm(
    state: PPTPageState,
    elements: List[PageElement],
    vlm_client: VLMClient,
    cfg,
) -> Tuple[List[PageElement], int]:
    """策略B：整页标框后，让 VLM 统一校验元素类型。"""
    if not elements:
        return [], 0

    for el in elements:
        el.type = _canonical_dispatch_type(el.type)

    if not bool(getattr(cfg, "enable_type_correction", True)):
        return elements, 0

    max_elements = int(getattr(cfg, "type_correction_max_elements", 24))
    target_elements = elements[:max_elements]

    if not target_elements:
        return elements, 0

    corrected_count = 0
    before_types = {el.element_id: _canonical_dispatch_type(el.type) for el in target_elements}

    try:
        marked_image_path = mark_image_with_bboxes(state.image_path, target_elements)
        base64_img = _encode_image_base64(marked_image_path)

        catalog_lines = []
        for idx, el in enumerate(target_elements, start=1):
            desc = _safe_text(getattr(el, "description", ""))
            if len(desc) > 120:
                desc = desc[:120] + "..."
            catalog_lines.append(
                f"#{idx} | id={el.element_id} | current={_canonical_dispatch_type(el.type)} | desc={desc}"
            )

        summary = _safe_text(state.global_analysis.core_summary if state.global_analysis else "")
        section_title = _safe_text(state.global_analysis.section_title if state.global_analysis else "")

        system_prompt = """你是PPT版面元素类型校验器。
任务：根据整页标框图，对每个框做类型纠偏。

输出要求（严格JSON）：
{
  "labels": [
    {"idx": 1, "element_id": "...", "type": "text|chart|table|image", "confidence": 0.0-1.0}
  ]
}

规则：
1) 只允许 type 为 text/chart/table/image。
2) 含流程、箭头、结构关系的内容，即使文字很多，也标为 chart。
3) 若无法判断，保守返回当前类型。
4) 不要输出 markdown，不要输出解释性文本。"""

        user_prompt = f"""页面标题：{section_title}
页面全局摘要：{summary}
以下是候选框列表：
{os.linesep.join(catalog_lines)}

请对这些框逐一给出最终类型。"""

        from config import get_config
        model = get_config().vlm_runtime_model_name
        response = vlm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}},
                    ],
                },
            ],
            max_tokens=1200,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        labels = parsed.get("labels", []) if isinstance(parsed, dict) else []

        by_idx: Dict[int, str] = {}
        by_id: Dict[str, str] = {}
        if isinstance(labels, list):
            for item in labels:
                if not isinstance(item, dict):
                    continue
                t = _canonical_dispatch_type(str(item.get("type", "")))
                idx = item.get("idx")
                eid = str(item.get("element_id", "")).strip()
                if isinstance(idx, int):
                    by_idx[idx] = t
                if eid:
                    by_id[eid] = t

        for idx, el in enumerate(target_elements, start=1):
            new_t = by_id.get(el.element_id) or by_idx.get(idx)
            if new_t:
                el.type = new_t

    except Exception as e:
        print(f"[TypeNormalization] VLM类型纠偏失败，回退原类型: {e}")

    # Step1语义兜底：非纯文本页面且全是 text 时，提升最大 text 框为 chart
    try:
        is_pure_text = bool(state.global_analysis.is_pure_text) if state.global_analysis else False
        all_text = all(_canonical_dispatch_type(el.type) == "text" for el in target_elements) if target_elements else False
        if (not is_pure_text) and all_text and target_elements:
            has_visual_hint = False
            if state.global_analysis:
                for ge in state.global_analysis.elements or []:
                    gt = _canonical_dispatch_type(getattr(ge, "type", ""))
                    if gt in {"chart", "table", "image"}:
                        has_visual_hint = True
                        break
            if has_visual_hint:
                largest = None
                largest_area = -1
                for el in target_elements:
                    box = getattr(getattr(el, "bbox", None), "box_2d", None)
                    if box and len(box) == 4:
                        area = _bbox_area(box)
                        if area > largest_area:
                            largest_area = area
                            largest = el
                if largest is not None:
                    largest.type = "chart"
    except Exception:
        pass

    # 大框包小框：非text容器内的小 text 去重
    try:
        container_area_ratio = float(getattr(cfg, "type_correction_container_area_ratio", 0.10))
        to_drop_ids = set()
        non_text = [e for e in elements if _canonical_dispatch_type(e.type) in {"chart", "table", "image"}]
        text_like = [e for e in elements if _canonical_dispatch_type(e.type) == "text"]

        for parent in non_text:
            pbox = getattr(getattr(parent, "bbox", None), "box_2d", None)
            if not pbox or len(pbox) != 4:
                continue
            p_area = _bbox_area(pbox)
            if p_area < int(1000 * 1000 * container_area_ratio):
                continue
            inside_count = 0
            for child in text_like:
                cbox = getattr(getattr(child, "bbox", None), "box_2d", None)
                if not cbox or len(cbox) != 4:
                    continue
                c_area = _bbox_area(cbox)
                if c_area <= 0:
                    continue
                if _bbox_contains(pbox, cbox) and c_area <= int(p_area * 0.45):
                    inside_count += 1
                    to_drop_ids.add(child.element_id)
            if inside_count < 2:
                for child in text_like:
                    to_drop_ids.discard(child.element_id)

        if to_drop_ids:
            elements = [e for e in elements if e.element_id not in to_drop_ids]
    except Exception:
        pass

    for el in elements:
        el.type = _canonical_dispatch_type(el.type)

    for el in target_elements:
        after_t = _canonical_dispatch_type(el.type)
        if before_types.get(el.element_id) and before_types[el.element_id] != after_t:
            corrected_count += 1

    return elements, corrected_count


def _build_mineru_text_bypass_insight(text_elements: List[PageElement], state: PPTPageState) -> ElementInsight | None:
    if not text_elements:
        return None

    sortable = []
    for el in text_elements:
        box = getattr(getattr(el, "bbox", None), "box_2d", None)
        y = int(box[0]) if box and len(box) == 4 else 999
        x = int(box[1]) if box and len(box) == 4 else 999
        sortable.append((y, x, el))
    sortable.sort(key=lambda t: (t[0], t[1]))

    chunks: List[str] = []
    seen = set()
    for _, _, el in sortable:
        text = _safe_text(getattr(el, "description", "") or getattr(el, "legend_text", ""))
        if text and text not in seen:
            chunks.append(text)
            seen.add(text)

    if not chunks:
        return None

    merged = "\n".join(chunks)
    preview = merged[:300] + ("..." if len(merged) > 300 else "")

    return ElementInsight(
        element_id=f"mineru_text_page_{state.page_index}",
        element_type="text_page",
        key_insight=preview,
        data_evidence=merged,
        confidence=0.9,
        status="pass",
        hypothesis_verification=f"MinerU OCR直通文本，共{text_elements.__len__()}块",
    )


def _build_text_aggregate_insight(text_elements: List[PageElement], state: PPTPageState, cfg) -> ElementInsight | None:
    if not text_elements or not getattr(cfg, "aggregate_text_elements", True):
        return None

    chunks = []
    for el in text_elements:
        t = _safe_text(getattr(el, "description", ""))
        if t:
            chunks.append(t)
    if not chunks:
        return None

    # 去重保序
    seen = set()
    dedup_chunks = []
    for c in chunks:
        if c not in seen:
            dedup_chunks.append(c)
            seen.add(c)

    merged = "\n".join(dedup_chunks)
    step1_text = _safe_text(state.global_analysis.extracted_text if state.global_analysis else "")
    xml_text = _load_xml_text_safe(state)
    sim_step1 = _text_similarity(merged, step1_text)
    sim_xml = _text_similarity(merged, xml_text)

    threshold = float(getattr(cfg, "text_compare_similarity_threshold", 0.60))
    status = "pass" if max(sim_step1, sim_xml) >= threshold else "retry"

    compare_note = (
        f"text_compare(sim_step1={sim_step1:.3f}, sim_xml={sim_xml:.3f}, threshold={threshold:.3f})"
    )

    return ElementInsight(
        element_id=f"aggregated_text_page_{state.page_index}",
        element_type="text_aggregate",
        key_insight=merged,
        data_evidence=compare_note,
        confidence=0.8 if status == "pass" else 0.6,
        status=status,
        hypothesis_verification=compare_note,
    )


def _cache_file_path(page_index: int, mineru_mode: str, cfg) -> str:
    cache_dir = Path(getattr(cfg, "layout_cache_dir", "processing_artifacts/layout_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / f"page_{page_index:03d}_{mineru_mode}.json")


def _contains_failure_marker(text: str) -> bool:
    s = str(text or "").strip().lower()
    if not s:
        return False
    markers = (
        "分析失败",
        "提取失败",
        "unterminated string",
        "jsondecodeerror",
        "expecting value",
        "traceback",
        "error:",
        "line 1 column",
        "char ",
    )
    return any(m in s for m in markers)


def _is_visual_element_type(element_type: str) -> bool:
    et = (element_type or "").strip().lower()
    return et in {"chart", "image", "mixed", "diagram", "flowchart", "figure", "graph"}


def _is_cache_payload_healthy(result: dict) -> bool:
    insights = result.get("element_insights", []) or []
    for ins in insights:
        element_type = getattr(ins, "element_type", "")
        if not _is_visual_element_type(element_type):
            continue

        status = (getattr(ins, "status", "") or "").strip().lower()
        key_insight = getattr(ins, "key_insight", "")
        data_evidence = getattr(ins, "data_evidence", "")

        if status in {"fail", "retry"}:
            return False
        if _contains_failure_marker(key_insight) or _contains_failure_marker(data_evidence):
            return False

    return True


def _serialize_result(result: dict) -> dict:
    out = {
        "pending_elements": [],
        "element_insights": [],
    }
    for el in result.get("pending_elements", []) or []:
        if hasattr(el, "model_dump"):
            out["pending_elements"].append(el.model_dump())
    for insight in result.get("element_insights", []) or []:
        if hasattr(insight, "model_dump"):
            out["element_insights"].append(insight.model_dump())
    return out


def _deserialize_result(payload: dict) -> dict:
    pending = [PageElement(**x) for x in payload.get("pending_elements", [])]
    insights = [ElementInsight(**x) for x in payload.get("element_insights", [])]
    return {
        "pending_elements": pending,
        "element_insights": insights,
    }

class SimpleLocator:
    def __init__(self, vlm_client: VLMClient):
        self.vlm_client = vlm_client
        from config import get_config
        self.model = get_config().vlm_runtime_model_name

    def locate_elements(self, image_path: str, elements: list[PageElement]) -> list[PageElement]:
        """
        专门的任务：只负责找坐标。
        输入：Step 1 识别出的元素描述列表。
        输出：带 BBox 的元素列表。
        """
        if not elements:
            return []
            
        print(f"[Locator] 回调 VLM 定位 {len(elements)} 个元素")
        
        # 1. 准备图片
        # 注意：这里需要引用 nodes.step1_global_analysis 中的 ImageUtils
        # 或者为了解耦，简单的 base64 编码逻辑可以重复写在这里
        try:
            from nodes.step1_global_analysis import ImageUtils
            base64_img = ImageUtils.encode_image(image_path)
        except ImportError:
            # 简单的 fallback
            import base64
            with open(image_path, "rb") as image_file:
                base64_img = base64.b64encode(image_file.read()).decode('utf-8')

        # 2. 构造 Prompt
        elements_desc = "\n".join([f"- ID: {e.element_id} | 类型: {e.type} | 描述: {e.description}" for e in elements])
        
        system_prompt = f"""你是一个精确的坐标定位助手。
        用户会提供一张 PPT 图片和一个元素列表（包含 ID 和描述）。
        【重要任务】
        1. 请在 1000x1000 的归一化坐标系中找到这些元素。
        2. **必须使用列表提供的 ID 作为 JSON 的 Key**，严禁修改 ID (不要把 element_1 改成 chart_1)。
        3. 坐标格式：请返回 [ymin, xmin, ymax, xmax]，取值范围 0-1000。例如：整个页面的左上角为 [0, 0, 200, 200]。
        4. 如果找不到某个元素，不要返回该 ID。

        待定位元素列表：
        {elements_desc}

        请返回严格的 JSON 格式，不要包含 Markdown 格式符（如 ```json），格式如下：
        {{
            "element_id_1": [ymin, xmin, ymax, xmax],
            "element_id_2": [ymin, xmin, ymax, xmax]
        }}"""

        try:
            # 3. 调用 API
            response = self.vlm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": "请定位列表中的元素。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]}
                ],
                max_tokens=1000,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            coord_map = json.loads(content)
            
            # 4. 回填坐标
            normalized_map = {}
            for k, v in coord_map.items():
                # 去除空格，转小写
                clean_key = k.lower().replace(" ", "").replace("_", "")
                normalized_map[clean_key] = v
            
            # 4. 回填坐标
            success_count = 0
            located_elements = []
            for e in elements:
                # 尝试匹配
                target_key = e.element_id.lower().replace(" ", "").replace("_", "")
                
                bbox_data = None
                # 1. 精确匹配
                if e.element_id in coord_map:
                    bbox_data = coord_map[e.element_id]
                # 2. 模糊匹配
                elif target_key in normalized_map:
                    bbox_data = normalized_map[target_key]
                
                if bbox_data:
                    try:
                        if isinstance(bbox_data, list) and len(bbox_data) == 4:
                            # 强制限制在 0-1000 范围内
                            ymin, xmin, ymax, xmax = [max(0, min(1000, int(float(x)))) for x in bbox_data]
                            # 存储时建议明确字段名，避免后续解包出错
                            e.bbox = BBox(box_2d=[ymin, xmin, ymax, xmax])
                            success_count += 1
                            # safe_bbox = [
                            #     max(0, min(1000, int(float(x)))) 
                            #     for x in bbox_data
                            # ]
                            # # 坐标矫正
                            # if safe_bbox[0] > safe_bbox[2]: safe_bbox[0], safe_bbox[2] = safe_bbox[2], safe_bbox[0]
                            # if safe_bbox[1] > safe_bbox[3]: safe_bbox[1], safe_bbox[3] = safe_bbox[3], safe_bbox[1]
                            
                            # e.bbox = BBox(box_2d=safe_bbox)
                            # success_count += 1
                        else:
                            print(f"⚠️ [Locator] 坐标格式错误 {e.element_id}: {bbox_data}")
                    except Exception as cast_err:
                        print(f"⚠️ [Locator] 坐标转换异常 {e.element_id}: {cast_err}")
                else:
                    print(f"⚠️ [Locator] VLM 未返回 ID: {e.element_id} (Available keys: {list(coord_map.keys())})")
                
                located_elements.append(e)
            
            print(f"[Locator] 成功定位: {success_count}/{len(elements)}")
            return located_elements

        except Exception as e:
            print(f"❌ [Locator] 定位失败: {e}")
            return elements
        
class LayoutAnalysisEngine:
    """布局分析引擎：硬检测 + 软聚合"""

    def __init__(self, vlm_client: VLMClient):
        self.vlm_client = vlm_client
        from config import get_config
        self.model = get_config().vlm_runtime_model_name

    def run_pipeline(self, state: PPTPageState) -> dict:
        print(f"[LayoutEngine] 启动布局分析 Pipeline (Page {state.page_index})")
        
        # --- Phase 1: 硬检测 (Mock YOLO) ---
        # 目前还没有 YOLO，所以我们“假装”检测到了元素。
        # 策略：直接使用 Step 1 全局分析中 VLM 识别出的元素。
        # 注意：Step 1 的元素可能没有 BBox，如果 Step 2 Router 之前跑过 vlm_localization，
        # 这里的 state.pending_elements 可能已经有粗略坐标了。
        
        # 我们优先使用已经带坐标的元素，如果没有，就回退到 Step 1 的元素
        candidates = state.pending_elements if state.pending_elements else state.global_analysis.elements
        
        # 必须确保元素有 BBox 才能进行 SoM 标记
        # 如果当前没有任何坐标（比如是第一次进这个节点），我们需要先快速搞个坐标
        # 这里为了跑通，我们调用一个简化的 Locator (复用 SimpleLocator)
        if not all(e.bbox for e in candidates):
            print("[LayoutEngine] 元素缺少坐标，调用 VLM 快速定位...")
            locator = SimpleLocator(self.vlm_client)
            raw_elements = locator.locate_elements(state.image_path, candidates)
        else:
            raw_elements = candidates

        print(f"[LayoutEngine] Mock YOLO 检测到 {len(raw_elements)} 个元素")

        # --- Phase 2: Set-of-Marks 视觉标记 (Real) ---
        # 这步是真实的，会在图片上画框和ID
        try:
            marked_image_path = mark_image_with_bboxes(state.image_path, raw_elements)
            print(f"[LayoutEngine] SoM 标记图已生成: {marked_image_path}")
        except Exception as e:
            print(f"[LayoutEngine] ⚠️ SoM 标记失败 (可能是cv2问题): {e}")
            marked_image_path = state.image_path # 降级：使用原图

        # --- Phase 3: 语义聚合 (VLM Grouping) (Real) ---
        # 让 VLM 看着画了框的图，决定哪些元素是一组
        logical_groups = self._semantic_grouping(
            marked_image_path, 
            raw_elements, 
            state.global_analysis.core_summary
        )
        
        return {
            "raw_detected_elements": raw_elements,
            "logical_groups": logical_groups,
            # 将 raw_elements 传递下去，或者如果 grouping 成功，后续 worker 应该基于 groups 工作
            # 为了兼容现有 worker，我们暂时还是传 elements
            "pending_elements": raw_elements 
        }

    def _semantic_grouping(self, marked_image_path: str, elements: list, summary: str) -> List[LogicalGroup]:
        """让 VLM 看着画了框的图，分析元素间的逻辑关系"""
        if not elements:
            return []
            
        print("[LayoutEngine] 正在进行语义分组 (Semantic Grouping)...")
        
        # 构造 ID 列表供 VLM 参考
        element_list_str = "\n".join([f"#{i+1}: {e.type} (初步识别)" for i, e in enumerate(elements)])
        
        system_prompt = f"""你是一个文档布局逻辑分析师。
        我提供了一张标注了 ID (如 #1, #2) 的 PPT 页面。
        请分析这些元素之间的**逻辑关系**，并将它们归类为有意义的“逻辑组”。

        全局主题：{summary}

        检测到的元素：
        {element_list_str}

        你的任务：
        1. **识别组合**：哪些元素应该放在一起看？（例如：#1是图表，#2是该图表的图例，#3是该图表的结论文字 -> 它们是一组）。
        2. **识别对比**：#4和#5是否是左右并列的对比关系？
        3. **识别独立**：哪些元素是独立的？

        请返回 JSON 格式的分组列表：
        [
            {{
                "group_id": "group_1",
                "element_ids": ["#1", "#2"],
                "group_type": "chart_with_explanation",
                "semantic_desc": "左侧的生存曲线及其下方的P值标注"
            }},
            {{
                "group_id": "group_2",
                "element_ids": ["#3", "#4"],
                "group_type": "comparison",
                "semantic_desc": "右侧的两张柱状图，展示了不同年份的对比"
            }}
        ]"""
        # ... 调用 VLM (传入 marked_image_path) ...
        # ... 解析 JSON 并返回 LogicalGroup 对象 ...
        return [] # Mock return

def node_vlm_locator(state: PPTPageState, vlm_client: VLMClient, debug_logger=None) -> dict:
    """节点：调用 VLM 获取坐标"""
    page_index = state.page_index
    
    # 记录步骤开始
    if debug_logger:
        debug_logger.log_step_start(
            page_index,
            "Step 2.5: VLM Locator",
            input_data={
                "page_index": page_index,
                "pending_elements_count": len(state.pending_elements),
                "image_path": state.image_path
            },
            previous_step="Step 2: Router"
        )
    
    locator = SimpleLocator(vlm_client)
    # 定位 pending_elements
    elements_with_bbox = locator.locate_elements(state.image_path, state.pending_elements)

    # Locator 后统一基础清洗：仅做去噪/去重，不在此阶段聚合文本
    try:
        from config import get_config
        cfg = get_config()
        if getattr(cfg, "filter_decorative_elements", True):
            elements_with_bbox = _clean_elements_no_text_split(elements_with_bbox, cfg)
    except Exception:
        # 过滤失败不影响主流程
        pass
    
    # 记录步骤结束
    if debug_logger:
        debug_logger.log_step_end(
            page_index,
            "Step 2.5: VLM Locator",
            output_data={
                "elements_with_bbox_count": len(elements_with_bbox),
                "elements_sample": [
                    {
                        "element_id": e.element_id,
                        "type": e.type,
                        "bbox": e.bbox.box_2d if e.bbox else None
                    } for e in elements_with_bbox[:3]
                ]
            },
            status="success"
        )
    
    # 更新 state 中的 pending_elements，现在它们有坐标了！
    return {"pending_elements": elements_with_bbox}

def node_layout_pipeline(state: PPTPageState, vlm_client: VLMClient, debug_logger=None) -> dict:
    """
    节点：调用复杂页面处理 Pipeline
    
    该节点现在集成了完整的四阶段处理流程：
    - Phase 1: 版面检测与类型清洗 (MinerU + VLM)
    - Phase 2: 语义挂载与逻辑校验 (SoM + VLM分组 + Supervisor)
    - Phase 3: 并行特征提取 (表格/图表/公式/文本专家)
    - Phase 4: 统一组装与输出 (Markdown + JSON)
    """
    page_index = state.page_index
    
    plan = getattr(state, "step2_plan", {}) or {}
    mineru_mode = plan.get("mineru_mode")
    # 兜底：若 Router 未设置，则根据复杂度推断
    if not mineru_mode:
        try:
            if state.global_analysis and state.global_analysis.complexity_score == "high":
                mineru_mode = "vlm"
            elif state.global_analysis and state.global_analysis.complexity_score == "medium":
                mineru_mode = "pipeline"
        except Exception:
            mineru_mode = None
    if mineru_mode not in {"pipeline", "vlm"}:
        mineru_mode = "pipeline"

    use_mineru_vlm = (mineru_mode == "vlm")
    from config import get_config
    cfg = get_config()
    cache_enabled = bool(getattr(cfg, "enable_layout_cache", True))
    cache_file = _cache_file_path(page_index, mineru_mode, cfg)
    bypass_cache_read = bool(getattr(state, "is_retry_mode", False) or getattr(state, "retry_count", 0) > 0)

    if cache_enabled and not bypass_cache_read and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            loaded = _deserialize_result(cached)
            if not _is_cache_payload_healthy(loaded):
                print(f"[LayoutPipeline] 缓存命中但质量不达标，忽略并重跑: {cache_file}")
                loaded = None
            if loaded is None:
                raise ValueError("cached payload unhealthy")
            print(f"[LayoutPipeline] 命中缓存: {cache_file}")
            if debug_logger:
                try:
                    debug_logger.log_step_end(
                        page_index,
                        "Step 2.5: Complex Layout Pipeline",
                        output_data={
                            "pending_elements_count": len(loaded.get("pending_elements", [])),
                            "element_insights_count": len(loaded.get("element_insights", [])),
                            "pipeline_mode": "complex_cache",
                            "mineru_mode": mineru_mode,
                        },
                        status="success"
                    )
                except Exception:
                    pass
            return loaded
        except Exception as e:
            print(f"[LayoutPipeline] 缓存读取失败，继续实时执行: {e}")

    # 记录步骤开始
    if debug_logger:
        debug_logger.log_step_start(
            page_index,
            "Step 2.5: Complex Layout Pipeline",
            input_data={
                "page_index": page_index,
                "pending_elements_count": len(state.pending_elements),
                "image_path": state.image_path,
                "has_global_analysis": state.global_analysis is not None,
                "mineru_mode": mineru_mode,
                "use_mineru_vlm": use_mineru_vlm,
            },
            previous_step="Step 2: Router"
        )
    
    print(f"\n[LayoutPipeline] 启动复杂页面处理 Pipeline (Page {page_index}) | MinerU={mineru_mode}")
    
    try:
        # 尝试导入复杂Pipeline模块
        from .complex_pipeline import node_complex_pipeline
        
        # 调用复杂Pipeline
        result = node_complex_pipeline(
            state=state,
            vlm_client=vlm_client,
            llm_client=None,  # 如果需要可以传入LLM客户端
            debug_logger=debug_logger,
            use_mineru_vlm=use_mineru_vlm,
        )
        
        print(f"[LayoutPipeline] 复杂Pipeline处理完成")

        # 统一基础清洗（适配 MinerU 结果）：仅去噪/去重，不提前剔除 text
        result["pending_elements"] = _clean_elements_no_text_split(result.get("pending_elements", []), cfg)

        # 写缓存（用于重试和重复运行复用）
        if cache_enabled:
            try:
                if _is_cache_payload_healthy(result):
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(_serialize_result(result), f, ensure_ascii=False, indent=2)
                else:
                    print(f"[LayoutPipeline] 跳过缓存写入（检测到失败视觉提取）: {cache_file}")
            except Exception as e:
                print(f"[LayoutPipeline] 缓存写入失败: {e}")
        
        # 记录步骤结束
        if debug_logger:
            pending_elements = result.get("pending_elements", [])
            element_insights = result.get("element_insights", [])
            debug_logger.log_step_end(
                page_index,
                "Step 2.5: Complex Layout Pipeline",
                output_data={
                    "pending_elements_count": len(pending_elements),
                    "element_insights_count": len(element_insights),
                    "pipeline_mode": "complex",
                    "mineru_mode": mineru_mode,
                },
                status="success"
            )
        
        return result
        
    except ImportError as e:
        print(f"[LayoutPipeline] ⚠️ 复杂Pipeline模块导入失败: {e}")
        print(f"[LayoutPipeline] 回退到简化模式...")
        
        # 回退到原有的简化模式
        engine = LayoutAnalysisEngine(vlm_client)
        result = engine.run_pipeline(state)

        result["pending_elements"] = _clean_elements_no_text_split(result.get("pending_elements", []), cfg)

        if cache_enabled:
            try:
                if _is_cache_payload_healthy(result):
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(_serialize_result(result), f, ensure_ascii=False, indent=2)
                else:
                    print(f"[LayoutPipeline] 跳过缓存写入（检测到失败视觉提取）: {cache_file}")
            except Exception as e:
                print(f"[LayoutPipeline] 缓存写入失败: {e}")
        
        # 记录步骤结束
        if debug_logger:
            pending_elements = result.get("pending_elements", [])
            debug_logger.log_step_end(
                page_index,
                "Step 2.5: Complex Layout Pipeline",
                output_data={
                    "pending_elements_count": len(pending_elements),
                    "pipeline_mode": "fallback_simple",
                    "fallback_reason": str(e)
                },
                status="success"
            )
        
        return result
    
    except Exception as e:
        print(f"[LayoutPipeline] ❌ Pipeline执行失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 记录错误
        if debug_logger:
            debug_logger.log_step_end(
                page_index,
                "Step 2.5: Complex Layout Pipeline",
                output_data={
                    "error": str(e),
                    "pipeline_mode": "error"
                },
                status="failed"
            )
        
        # 返回空结果，让后续流程能继续
        return {
            "pending_elements": state.pending_elements,
            "element_insights": []
        }


def node_type_normalization(state: PPTPageState, vlm_client: VLMClient, debug_logger=None) -> dict:
    """节点：Locator 后执行类型纠偏 + MinerU Text 直通。"""
    page_index = state.page_index
    elements = list(state.pending_elements or [])

    if not elements:
        return {"pending_elements": []}

    if debug_logger:
        debug_logger.log_step_start(
            page_index,
            "Step 2.6: Type Normalization",
            input_data={
                "page_index": page_index,
                "pending_elements_count": len(elements),
            },
            previous_step="Step 2.5: Locator",
        )

    try:
        from config import get_config
        cfg = get_config()
    except Exception:
        cfg = None

    cleaned = elements
    if cfg and getattr(cfg, "filter_decorative_elements", True):
        try:
            cleaned = _clean_elements_no_text_split(cleaned, cfg)
        except Exception:
            pass

    corrected = cleaned
    corrected_count = 0
    if cfg is not None:
        corrected, corrected_count = _correct_types_with_page_vlm(state, cleaned, vlm_client, cfg)
    else:
        for el in corrected:
            el.type = _canonical_dispatch_type(el.type)

    text_elements = [el for el in corrected if _canonical_dispatch_type(el.type) == "text"]
    non_text_elements = [el for el in corrected if _canonical_dispatch_type(el.type) in {"chart", "table", "image"}]

    text_insight = _build_mineru_text_bypass_insight(text_elements, state)

    if debug_logger:
        debug_logger.log_step_end(
            page_index,
            "Step 2.6: Type Normalization",
            output_data={
                "corrected_count": corrected_count,
                "total_after_clean": len(corrected),
                "text_bypass_count": len(text_elements),
                "worker_dispatch_count": len(non_text_elements),
            },
            status="success",
        )

    out: Dict[str, Any] = {"pending_elements": non_text_elements}
    if text_insight is not None:
        out["element_insights"] = [text_insight]
    return out