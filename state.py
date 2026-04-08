"""
state.py
定义 PPT 解析工作流中的所有数据结构 (State)。
"""

import operator
from typing import List, Dict, Optional, Any, Annotated, Literal, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# 1. 基础实体对象 (Entities)
# ============================================================================

class BBox(BaseModel):
    """边界框定义"""
    box_2d: Optional[List[int]] = Field(
        default=None, 
        description="[ymin, xmin, ymax, xmax], 0-1000 scale"
    )

class PageElement(BaseModel):
    """页面元素定义"""
    element_id: str = Field(..., description="唯一ID，如 chart_01, table_02")
    type: str = Field(..., description="类型：chart, table, diagram, image, text_block")
    description: str = Field(..., description="元素的视觉描述")
    legend_text: Optional[str] = Field(None, description="该元素对应的图例或说明文字")
    bbox: Optional[BBox] = None
    medical_context: Optional[str] = ""

class ElementInsight(BaseModel):
    """Worker 分析后的单个元素洞察"""
    element_id: str
    element_type: str
    key_insight: str = Field(..., description="提取的核心汇报信息")
    data_evidence: str = Field(..., description="支持该结论的数据或文本证据")
    confidence: float = Field(default=1.0, description="置信度 0.0-1.0")
    status: Literal["pass", "fail", "retry"] = "pass"
    hypothesis_verification: Optional[str] = ""
    crop_path: Optional[str] = Field(default=None, description="元素的裁剪图片路径")

class GlobalPageAnalysis(BaseModel):
    """Step 1 全局语义分析结果"""
    page_index: int
    section_title: str
    is_pure_text: bool = Field(..., description="是否为纯文本页面")
    complexity_score: Literal["low", "medium", "high"] = "low"
    has_unclear_boundaries: bool = Field(False, description="元素边界是否模糊/重叠")
    core_summary: str = Field(..., description="页面核心观点 (Key Takeaway)")
    extracted_text: str = Field(default="", description="页面上的所有可见文本")
    elements: List[PageElement] = Field(default_factory=list)
    research_hypothesis: Optional[str] = Field(default="", description="基于页面标题推测的科研假设，例如：'验证实验组生存获益'")

class SupervisorOutput(BaseModel):
    status: str = Field(default="fail", description="'pass' 或 'fail'")
    # LLM 输出经常不稳定：issues 可能是 str / list[str] / list[dict] / 混合 list。
    # 这里统一归一化为 list[str]，避免 Step3 因 Pydantic 校验失败而降级通过。
    issues: List[str] = Field(default_factory=list, description="发现的问题列表")
    feedback: Union[str, List[Any], Dict[str, Any]] = Field(default="", description="详细反馈")
    consistency_score: int = Field(default=0, description="0-100, 全局一致性打分")
    warning: Union[str, List[Any], Dict[str, Any], None] = Field(default="", description="交叉验证警告（如果有）")

    @field_validator("warning", mode="before")
    @classmethod
    def _normalize_warning(cls, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            parts = [str(x).strip() for x in v if str(x).strip()]
            return " | ".join(parts)
        if isinstance(v, dict):
            try:
                import json as _json
                return _json.dumps(v, ensure_ascii=False)
            except Exception:
                return str(v)
        return str(v)

    @field_validator("issues", mode="before")
    @classmethod
    def _normalize_issues(cls, v: Any) -> List[str]:
        def _stringify_issue(item: Any) -> Optional[str]:
            if item is None:
                return None
            if isinstance(item, str):
                s = item.strip()
                if not s or s in {":", "-", "--"}:
                    return None
                return s
            if isinstance(item, (int, float, bool)):
                # 过滤掉明显的噪声数值（例如 -45）
                return None
            if isinstance(item, dict):
                element = item.get("element") or item.get("element_id") or item.get("id")
                msg = (
                    item.get("issue")
                    or item.get("message")
                    or item.get("problem")
                    or item.get("desc")
                    or item.get("detail")
                    or item.get("feedback")
                )
                if isinstance(msg, str) and msg.strip():
                    if element:
                        return f"[{element}] {msg.strip()}"
                    return msg.strip()
                # 尽量不要把整坨 dict 原样塞回去；但至少保证可读
                try:
                    import json as _json
                    return _json.dumps(item, ensure_ascii=False)
                except Exception:
                    return str(item)
            if isinstance(item, list):
                # 不期望嵌套 list，摊平
                parts: List[str] = []
                for sub in item:
                    s = _stringify_issue(sub)
                    if s:
                        parts.append(s)
                return " | ".join(parts) if parts else None
            # 兜底：转字符串
            s = str(item).strip()
            return s if s else None

        if v is None:
            return []
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            # 按行/分号粗分，尽量让 issues 是多条
            seps = ["\n", "；", ";"]
            for sep in seps:
                if sep in s:
                    items = [x.strip(" -\t") for x in s.split(sep)]
                    return [x for x in items if x]
            return [s]
        if isinstance(v, dict):
            s = _stringify_issue(v)
            return [s] if s else []
        if isinstance(v, list):
            out: List[str] = []
            for item in v:
                s = _stringify_issue(item)
                if s:
                    out.append(s)
            return out
        # 其它类型：尽量转为单条字符串
        s = _stringify_issue(v)
        return [s] if s else []

    @field_validator("consistency_score", mode="before")
    @classmethod
    def _normalize_score(cls, v: Any) -> int:
        try:
            return int(v)
        except Exception:
            return 0

class LogicalGroup(BaseModel):
    group_id: str
    element_ids: List[str] 
    group_type: str        
    semantic_desc: str   
# ============================================================================
# 2. 最终输出结构 (Output)
# ============================================================================

class FinalPageOutput(BaseModel):
    """最终输出给用户的页面结构"""
    page_index: int
    chapter_title: Optional[str]
    section_title: str
    summary: str
    elements: Dict[str, Any]
    markdown_content: str
    slice_reference: Dict[str, Any]
    needs_review: bool = False
    review_status: Literal["not_required", "pending", "reviewed"] = "not_required"
    review_reasons: List[str] = Field(default_factory=list)
    review_notes: str = ""
    reviewed_markdown_content: Optional[str] = None
    reviewed_summary: Optional[str] = None
    reviewed_elements: Optional[Dict[str, Any]] = None
    reconcile_applied: bool = False
    reconcile_notes: List[str] = Field(default_factory=list)
    reconcile_missing_texts: List[str] = Field(default_factory=list)
    reconcile_table_conflict: bool = False
    reconcile_mineru_page_markdown: Optional[str] = None
    reconcile_table_output: Optional[str] = None
    reconcile_table_mineru: Optional[str] = None
    reconcile_table_merged: Optional[str] = None


class PageAnalysisResult(BaseModel):
    """Step 3 Supervisor 的中间校验结果"""
    page_index: int
    section_title: str
    status: Literal["pass", "failed", "retry"]
    global_analysis: GlobalPageAnalysis
    element_insights: List[ElementInsight]
    validation_passed: bool
    validation_notes: str = ""
    consistency_score: int = 0
    fused_markdown: str = ""


# ============================================================================
# 3. Graph 状态定义 (States)
# ============================================================================

class PPTPageState(BaseModel):
    """
    【页面级状态】
    在 create_page_workflow 子图中流转的状态。
    """
    # --- 基础信息 ---
    page_index: int
    ppt_path: str  # 保留此字段用于文本解析，使用 Annotated 处理合并
    image_path: str
    run_id: Optional[str] = None
    output_dir: Optional[str] = None
    processing_artifacts_dir: Optional[str] = None
    layout_cache_dir: Optional[str] = None
    max_retries: int = 2
    retry_count: int = 0
    chapter_title: Optional[str] = None
    
    # --- 流程控制 ---
    is_retry_mode: bool = False

    # --- 上下文信息 ---
    previous_context: str = ""
    
    # --- Step 1 数据 ---
    global_analysis: Optional[GlobalPageAnalysis] = None
    
    # --- Step 2 数据 ---
    step2_plan: Dict[str, Any] = Field(default_factory=dict)
    pending_elements: List[PageElement] = Field(default_factory=list)
    native_page_evidence: Dict[str, Any] = Field(default_factory=dict)
    precomputed_mineru_layout: Optional[Dict[str, Any]] = None
    precomputed_mineru_source: Optional[str] = None

    # 从 YOLO 得到的原始元素（硬切割）
    raw_detected_elements: List[PageElement] = field(default_factory=list)
    
    # 经过 VLM 聚合后的逻辑组（软语义）
    logical_groups: List[LogicalGroup] = field(default_factory=list)
    
    # --- Worker 结果 ---
    element_insights: Annotated[List[ElementInsight], operator.add] = []
    
    # --- Step 3 数据 ---
    supervisor_feedback: Dict[str, str] = Field(default_factory=dict)
    analysis_result: Optional[PageAnalysisResult] = None
    
    # --- Step 4 数据 ---
    final_output: Optional[FinalPageOutput] = None


class PPTWorkflowState(BaseModel):
    """
    【全局工作流状态】
    在主 Graph 中流转的状态。
    """
    ppt_path: str
    max_concurrent_pages: int = 3
    run_id: Optional[str] = None
    output_dir: Optional[str] = None
    processing_artifacts_dir: Optional[str] = None
    layout_cache_dir: Optional[str] = None
    
    # 页面队列
    page_queue: List[int] = Field(default_factory=list)
    total_pages: int = 0
    page_image_map: Dict[int, str] = Field(default_factory=dict)
    
    # 当前批次
    current_batch: List[int] = Field(default_factory=list)

    # 全局 MinerU 预计算结果（统一预测，拆分使用）
    global_mineru_pdf_path: Optional[str] = None
    global_mineru_output_dir: Optional[str] = None
    precomputed_mineru_layouts: Dict[int, Dict[str, Any]] = Field(default_factory=dict)
    
    # 章节信息
    chapter_info: Dict[str, Any] = Field(default_factory=dict)

    # 全局知识库
    global_knowledge_base: str = ""
    
    # 统计信息
    # completed_pages 使用 operator.add 来累加（多个页面返回时相加）
    completed_pages: Annotated[int, operator.add] = 0
    failed_pages: int = 0
    
    # 结果存储 {page_index: FinalPageOutput}
    # 使用 Annotated 和 operator.or_ 来合并多个页面的结果
    completed_outputs: Annotated[dict, lambda a, b: {**a, **b}] = field(default_factory=dict)
    global_knowledge_base: Annotated[str, lambda a, b: a + "\n" + b] = ""
