"""Step 2: 规划/路由节点，根据页面特性决定处理策略"""

from state import GlobalPageAnalysis, PageElement, PPTPageState
from langgraph.constants import Send
from typing import List
from config import get_config


class Step2_RouterEngine:
    """第二步路由引擎
    
    根据第一步的分析结果，决定：
    1. 纯文本页面 -> 调用 MarkItDown 解析
    2. 复杂布局 -> 调用布局分析pipeline（未完成）或标记为需要人工审核
    3. 正常布局 -> 分发各元素到对应的Agent（表格、图表、流程图等）
    """
    @staticmethod
    def _extract_medical_terms(text: str) -> str:
        """简单的医学术语/关键词提取，用于辅助定位"""
        # 这里可以使用更轻量级的 NLP 库，或者简单的规则
        import re
        candidates = set(re.findall(r'\b[A-Z][a-zA-Z0-9-]{2,}\b', text))
        # 过滤常见非实义词
        stop_words = {"THE", "AND", "FOR", "WITH", "FIG", "TABLE"}
        terms = [w for w in candidates if w.upper() not in stop_words]
        return ", ".join(terms[:5]) # 取前5个
    
    @staticmethod
    def plan_processing(analysis: GlobalPageAnalysis) -> dict:
        """规划页面处理方案，根据 Step 1 的定性分析，决定定位策略"""
        cfg = get_config()
        plan = {
            "route": None,
            "elements_to_process": [],
            "strategy_reason": "",
            # 对复杂页面 Pipeline 的运行模式进行细分：
            # - "pipeline": MinerU 标准 pipeline（更快，适用于 medium）
            # - "vlm": MinerU VLM/GPU 分支（更强，适用于 high）
            "mineru_mode": None,
        }
        
        # 注入医学上下文
        medical_context = Step2_RouterEngine._extract_medical_terms(analysis.extracted_text)
        for el in analysis.elements:
            el.medical_context = medical_context

        # 1. 纯文本 -> MarkItDown
        if analysis.is_pure_text:
            plan["route"] = "text_only"
            # plan["strategy_reason"] = "纯文本页面，无需视觉定位"
            return plan

        # 1.5 临时策略：非纯文本页面统一走 MinerU pipeline（禁用 VLM 路由）
        if getattr(cfg, "force_all_non_text_to_pipeline", False):
            plan["route"] = "layout_pipeline"
            plan["mineru_mode"] = "pipeline"
            plan["strategy_reason"] = "临时策略：所有非纯文本页面统一走 MinerU-pipeline"
            plan["elements_to_process"] = analysis.elements
            return plan

        # 2. 非纯文本页面的策略：
        # - high 或边界不清：优先 MinerU-VLM
        # - low/medium：优先 MinerU 标准 pipeline
        # - 若 enable_layout_pipeline 关闭，统一退回 VLM locator
        if not getattr(cfg, "enable_layout_pipeline", True):
            plan["route"] = "vlm_localization"
            plan["strategy_reason"] = "配置关闭布局 pipeline，回退 VLM 定位"
            plan["elements_to_process"] = analysis.elements
            return plan

        if analysis.complexity_score == "high" or analysis.has_unclear_boundaries:
            if getattr(cfg, "route_high_to_vlm", True):
                plan["route"] = "layout_pipeline"
                plan["mineru_mode"] = "vlm"
                plan["strategy_reason"] = "布局复杂或边界不清，使用 MinerU-VLM 模式精细切分"
            else:
                plan["route"] = "layout_pipeline"
                plan["mineru_mode"] = "pipeline"
                plan["strategy_reason"] = "高复杂页面但已禁用 VLM 模式，降级到 MinerU-pipeline"
            plan["elements_to_process"] = analysis.elements
            return plan

        if analysis.complexity_score == "medium":
            if getattr(cfg, "route_medium_to_pipeline", True):
                plan["route"] = "layout_pipeline"
                plan["mineru_mode"] = "pipeline"
                plan["strategy_reason"] = "中等复杂度页面，使用 MinerU-pipeline 模式"
            else:
                plan["route"] = "vlm_localization"
                plan["strategy_reason"] = "中等复杂度页面，按配置回退到 VLM 定位"
            plan["elements_to_process"] = analysis.elements
            return plan

        # low
        if getattr(cfg, "route_low_to_pipeline", True):
            plan["route"] = "layout_pipeline"
            plan["mineru_mode"] = "pipeline"
            plan["strategy_reason"] = "低复杂度页面，使用 MinerU-pipeline 保证切框稳定性"
            plan["elements_to_process"] = analysis.elements
            return plan

        # 3. 配置要求 low 不走 pipeline 时，回调 VLM 获取坐标
        plan["route"] = "vlm_localization"
        plan["strategy_reason"] = "低复杂度页面，按配置回调 VLM 定位"

        # 医药 PPT 核心关注的视觉元素类型
        VISUAL_TYPES = [
            "chart", "table", "diagram", "image", "picture", 
            "figure", "graph", "plot", 
            "molecular", "structure", "formula" # 针对医药场景新增
        ]
        filtered_elements = []
        for el in analysis.elements:
            el_type_lower = el.type.lower()
            
            # 逻辑：如果类型包含白名单中的任意关键词，则保留, 例如 "data_chart" 包含 "chart"，保留
            is_visual = any(vt in el_type_lower for vt in VISUAL_TYPES)
            
            if is_visual:
                filtered_elements.append(el)
            else:
                # 过滤掉 text, text_block, title, bullet_list 等
                print(f"[Router] 🚫 跳过纯文本元素: {el.element_id} ({el.type})")
        
        plan["elements_to_process"] = filtered_elements
        
        return plan


def node_step2_router(state: PPTPageState, debug_logger=None) -> dict:
    """节点：Step 2 路由和规划"""
    print(f"\n[Step2-Router] 页面 {state.page_index} 规划处理策略")
    
    page_index = state.page_index
    
    # 记录步骤开始
    if debug_logger:
        debug_logger.log_step_start(
            page_index,
            "Step 2: Router",
            input_data={
                "page_index": page_index,
                "section_title": state.global_analysis.section_title if state.global_analysis else None,
                "is_pure_text": state.global_analysis.is_pure_text if state.global_analysis else None,
                "complexity_score": state.global_analysis.complexity_score if state.global_analysis else None,
                "elements_count": len(state.global_analysis.elements) if state.global_analysis else 0
            },
            previous_step="Step 1: Global Analysis"
        )
    
    analysis = state.global_analysis
    plan = Step2_RouterEngine.plan_processing(analysis)
    
    print(f"[Step2-Router] 决策: {plan['route']} ({plan['strategy_reason']})")
    
    # 记录步骤结束
    if debug_logger:
        debug_logger.log_step_end(
            page_index,
            "Step 2: Router",
            output_data={
                "route": plan.get("route"),
                "strategy_reason": plan.get("strategy_reason"),
                "elements_to_process_count": len(plan.get("elements_to_process", []))
            },
            status="success"
        )
    
    return {
        "step2_plan": plan,
        # 注意：此时 pending_elements 里的元素还没有 BBox
        # 它们需要经过后续的 "Locator" 节点处理后，才能分发给 Workers
        "pending_elements": plan.get("elements_to_process", [])
    }
