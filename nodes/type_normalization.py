"""统一元素类型规范化（Single Source of Truth）。

将上游识别到的细分类型映射到 6 大类：
- text / table / chart / formula / image / mixed
"""

from typing import Optional

TEXT = "text"
TABLE = "table"
CHART = "chart"
FORMULA = "formula"
IMAGE = "image"
MIXED = "mixed"

CANONICAL_TYPES = {TEXT, TABLE, CHART, FORMULA, IMAGE, MIXED}

_TYPE_MAP = {
    # Text
    "text": TEXT,
    "title": TEXT,
    "text_block": TEXT,
    "header": TEXT,
    "footer": TEXT,
    "page_number": TEXT,

    # Table
    "table": TABLE,

    # Chart / Diagram
    "chart": CHART,
    "data_chart": CHART,
    "km_curve": CHART,
    "forest_plot": CHART,
    "bar_chart": CHART,
    "line_chart": CHART,
    "pie_chart": CHART,
    "diagram": CHART,
    "flowchart": CHART,

    # Formula
    "formula": FORMULA,
    "equation": FORMULA,
    "math": FORMULA,

    # Image
    "image": IMAGE,
    "picture": IMAGE,
    "photo": IMAGE,
    "figure": IMAGE,
    "molecular_structure": IMAGE,

    # Mixed (one ROI contains multiple element kinds)
    "mixed": MIXED,
    "mixed_group": MIXED,
    "multi_panel": MIXED,
    "composite": MIXED,
    "chart_and_table": MIXED,

    # Legacy
    "unknown": IMAGE,
}


def normalize_element_type(raw_type: Optional[str]) -> str:
    """将原始类型归一为 6 大类。"""
    type_key = (raw_type or "").strip().lower()
    if not type_key:
        return IMAGE
    return _TYPE_MAP.get(type_key, IMAGE)


def is_text_type(raw_type: Optional[str]) -> bool:
    """是否属于文本类（应走文本聚合/跳过视觉抽取）。"""
    return normalize_element_type(raw_type) == TEXT
