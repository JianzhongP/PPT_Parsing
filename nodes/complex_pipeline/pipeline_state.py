"""
复杂页面处理 Pipeline 状态定义

定义Pipeline各阶段的输入/输出数据结构
"""

from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field
from dataclasses import dataclass, field


# ============================================================================
# Phase 1 数据结构
# ============================================================================

class ElementBBox(BaseModel):
    """元素边界框 (归一化坐标 0-1000)"""
    box_2d: List[int] = Field(
        ..., 
        description="[ymin, xmin, ymax, xmax], 0-1000 scale"
    )
    
    @property
    def ymin(self) -> int:
        return self.box_2d[0]
    
    @property
    def xmin(self) -> int:
        return self.box_2d[1]
    
    @property
    def ymax(self) -> int:
        return self.box_2d[2]
    
    @property
    def xmax(self) -> int:
        return self.box_2d[3]
    
    @property
    def width(self) -> int:
        return self.xmax - self.xmin
    
    @property
    def height(self) -> int:
        return self.ymax - self.ymin


class DetectedElement(BaseModel):
    """MinerU检测到的元素"""
    element_id: str = Field(..., description="唯一标识符，如 elem_001")
    original_type: str = Field(..., description="MinerU原始类型: Text, Image, Table 等")
    refined_type: Optional[str] = Field(None, description="VLM清洗后类型: 数据图表, 表格, 纯图片, 公式, 流程图")
    bbox: ElementBBox = Field(..., description="边界框")
    ocr_text: Optional[str] = Field(None, description="OCR提取的文本")
    confidence: float = Field(default=1.0, description="检测置信度")
    
    # 用于调试
    vlm_refinement_response: Optional[str] = Field(None, description="VLM类型清洗的原始响应")


class ROIImage(BaseModel):
    """裁剪的ROI图片信息"""
    element_id: str
    roi_path: str = Field(..., description="ROI图片保存路径")
    original_type: str
    refined_type: Optional[str] = None
    bbox: ElementBBox


class CleanedLayoutJSON(BaseModel):
    """Phase 1 输出: 清洗后的版面布局JSON"""
    page_id: int
    image_path: str
    elements: List[DetectedElement] = Field(default_factory=list)
    roi_images: List[ROIImage] = Field(default_factory=list)
    detection_source: Literal['mineru_std', 'mineru_vlm', 'fallback', 'vlm_fallback'] = "mineru_std"
    processing_time_ms: Optional[int] = None

    # MinerU 调试信息（可选）
    mineru_debug_dir: Optional[str] = Field(None, description="MinerU 调试目录，包含 stdout/stderr 与输出文件索引")
    mineru_output_dir: Optional[str] = Field(None, description="MinerU 输出目录（mineru_output），通常包含 layout/json 等中间结果")
    mineru_cmd: Optional[str] = Field(None, description="实际执行的 docker 命令（字符串化）")
    mineru_returncode: Optional[int] = Field(None, description="docker 命令返回码")
    mineru_stdout_path: Optional[str] = Field(None, description="MinerU stdout 落盘路径")
    mineru_stderr_path: Optional[str] = Field(None, description="MinerU stderr 落盘路径")
    mineru_used_fallback: Optional[bool] = Field(None, description="本次是否回退到 mock/fallback")


# ============================================================================
# Phase 2 数据结构
# ============================================================================

class SemanticGroup(BaseModel):
    """语义分组"""
    group_id: str = Field(..., description="分组ID，如 group_1")
    member_ids: List[str] = Field(..., description="组内元素ID列表")
    group_type: str = Field(..., description="分组类型: chart_and_title, table_group, comparison 等")
    semantic_desc: str = Field(default="", description="分组的语义描述")
    reading_order: int = Field(default=0, description="阅读顺序索引")
    
    # 组内角色定义
    primary_element_id: Optional[str] = Field(None, description="主元素ID")
    secondary_element_ids: List[str] = Field(default_factory=list, description="从元素ID列表")


class DraftSemanticJSON(BaseModel):
    """Phase 2-1 输出: 初步语义分组结果"""
    page_id: int
    groups: List[SemanticGroup] = Field(default_factory=list)
    ungrouped_element_ids: List[str] = Field(default_factory=list, description="未分组的孤立元素")
    id_to_text_map: Dict[str, str] = Field(default_factory=dict, description="ID->OCR文本映射")
    som_image_path: Optional[str] = Field(None, description="带SoM标记的图片路径")


class ValidationIssue(BaseModel):
    """验证问题"""
    issue_type: str = Field(..., description="问题类型: orphan_chart, over_spanning, missing_title, wrong_order")
    element_ids: List[str] = Field(default_factory=list, description="涉及的元素ID")
    description: str = Field(..., description="问题描述")
    suggestion: str = Field(default="", description="修正建议")


class GroupingValidationResult(BaseModel):
    """Phase 2-2 分组验证结果"""
    status: Literal["pass", "fail", "meltdown"] = "pass"
    issues: List[ValidationIssue] = Field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 2
    fallback_used: bool = False
    validation_notes: str = ""


class ValidatedSemanticJSON(BaseModel):
    """Phase 2-2 输出: 验证后的语义分组"""
    page_id: int
    layout_confidence: Literal["high", "medium", "low"] = "high"
    groups: List[SemanticGroup] = Field(default_factory=list)
    validation_result: GroupingValidationResult
    
    # 来自Phase 1的数据引用
    elements: List[DetectedElement] = Field(default_factory=list)
    id_to_text_map: Dict[str, str] = Field(default_factory=dict)


# ============================================================================
# Phase 3 数据结构
# ============================================================================

class TableExtraction(BaseModel):
    """表格提取结果"""
    markdown_table: str = Field(default="", description="Markdown格式表格")
    json_structure: Optional[Dict[str, Any]] = Field(None, description="JSON结构化数据")
    insight: str = Field(default="", description="表格洞察信息")
    key_values: Dict[str, str] = Field(default_factory=dict, description="关键数据")
    table_title: str = Field(default="", description="表题/表头说明原文")
    external_texts: List[str] = Field(default_factory=list, description="表格外部说明文字原文（不包含表内单元格）")
    footnotes: List[str] = Field(default_factory=list, description="表格脚注/注释原文")
    notes: List[str] = Field(default_factory=list, description="补充说明原文（单位、统计说明等）")
    internal_headers: List[str] = Field(default_factory=list, description="表内表头文本（仅用于内部去重，不默认展示）")


class ChartExtraction(BaseModel):
    """图表提取结果"""
    chart_type: str = Field(default="", description="图表类型: bar_chart, line_chart, pie_chart 等")
    panel_name: str = Field(default="", description="子图名称/方位（如左图、右图、上图）")
    x_axis: Union[str, Dict[str, str], List[str]] = Field(default="", description="X轴含义（支持多子图）")
    y_axis: Union[str, Dict[str, str], List[str]] = Field(default="", description="Y轴含义（支持多子图）")
    key_values: Dict[str, Any] = Field(default_factory=dict, description="关键数值")
    insight: str = Field(default="", description="图表洞察分析")
    context_title: Optional[str] = Field(None, description="上下文标题")
    chart_title: str = Field(default="", description="图标题原文")
    panel_titles: List[str] = Field(default_factory=list, description="子图标题原文")
    legend_items: List[str] = Field(default_factory=list, description="图例原文")
    axis_labels: List[str] = Field(default_factory=list, description="坐标轴标签/刻度相关原文")
    annotation_texts: List[str] = Field(default_factory=list, description="图中解释性文字原文")
    footnotes: List[str] = Field(default_factory=list, description="图下注释/脚注原文")


class MixedExtraction(BaseModel):
    """混合元素提取结果（单个ROI内包含图表/表格等多种元素）"""
    summary: str = Field(default="", description="混合内容整体洞察")
    chart_items: List[ChartExtraction] = Field(default_factory=list, description="图表子元素列表")
    table_items: List[TableExtraction] = Field(default_factory=list, description="表格子元素列表")
    text_items: List[str] = Field(default_factory=list, description="混合区域中的原文描述文字，不翻译")
    contains_multiple_elements: bool = Field(default=True, description="是否确认存在多种元素")
    validation_notes: str = Field(default="", description="混合抽取自检说明")


class FormulaExtraction(BaseModel):
    """公式提取结果"""
    latex: str = Field(default="", description="LaTeX格式公式")
    plain_text: Optional[str] = Field(None, description="纯文本近似")


class TextExtraction(BaseModel):
    """文本提取结果"""
    merged_text: str = Field(default="", description="合并后的文本")
    paragraphs: List[str] = Field(default_factory=list, description="分段落文本")


class ExtractedContent(BaseModel):
    """单个元素的提取内容"""
    element_id: str
    element_type: str
    
    # 根据类型填充不同字段
    table_data: Optional[TableExtraction] = None
    chart_data: Optional[ChartExtraction] = None
    mixed_data: Optional[MixedExtraction] = None
    formula_data: Optional[FormulaExtraction] = None
    text_data: Optional[TextExtraction] = None
    
    # 通用字段
    raw_response: Optional[str] = Field(None, description="API原始响应")
    processing_status: Literal["success", "failed", "skipped"] = "success"
    error_message: Optional[str] = None


class ExtractedDataMap(BaseModel):
    """Phase 3 输出: 提取数据映射"""
    page_id: int
    extractions: Dict[str, ExtractedContent] = Field(
        default_factory=dict, 
        description="Key为element_id，Value为提取内容"
    )
    group_context_map: Dict[str, str] = Field(
        default_factory=dict,
        description="group_id -> 组内上下文文本（用于图表分析）"
    )


# ============================================================================
# Phase 4 数据结构
# ============================================================================

class SemanticBlock(BaseModel):
    """语义块 - 最终输出单元"""
    block_id: str
    block_type: str = Field(..., description="chart_group, table_group, text_block 等")
    reading_order: int
    role: str = Field(default="content", description="primary_content, supporting, decoration")
    
    # 元素内容
    elements: Dict[str, Any] = Field(default_factory=dict)
    
    # 联合边界框
    bbox_union: Optional[List[int]] = Field(None, description="组内所有元素的联合bbox")


class FinalStructuredJSON(BaseModel):
    """最终结构化JSON输出"""
    page_id: int
    layout_confidence: Literal["high", "medium", "low"] = "high"
    semantic_blocks: List[SemanticBlock] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FinalMarkdownOutput(BaseModel):
    """最终Markdown输出"""
    page_id: int
    title: Optional[str] = None
    markdown_content: str = ""
    
    # 引用信息
    image_references: List[str] = Field(default_factory=list)
    table_count: int = 0
    chart_count: int = 0


# ============================================================================
# Pipeline 总状态
# ============================================================================

class ComplexPipelineState(BaseModel):
    """复杂页面Pipeline完整状态"""
    
    # 基础信息
    page_id: int
    origin_image_path: str
    global_analysis: Optional[Dict[str, Any]] = Field(None, description="Step1传入的全局分析结果")
    
    # Phase 1 输出
    phase1_output: Optional[CleanedLayoutJSON] = None
    
    # Phase 2 输出
    phase2_draft: Optional[DraftSemanticJSON] = None
    phase2_validated: Optional[ValidatedSemanticJSON] = None
    
    # Phase 3 输出
    phase3_output: Optional[ExtractedDataMap] = None
    
    # Phase 4 输出
    final_json: Optional[FinalStructuredJSON] = None
    final_markdown: Optional[FinalMarkdownOutput] = None
    
    # 流程控制
    current_phase: Literal["phase1", "phase2_grouping", "phase2_validation", "phase3", "phase4", "completed", "failed"] = "phase1"
    phase2_retry_count: int = 0
    error_log: List[str] = Field(default_factory=list)
    
    # 性能监控
    phase_timings: Dict[str, int] = Field(default_factory=dict, description="各阶段耗时(ms)")
