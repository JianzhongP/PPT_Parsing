"""PPT解析工作流的主图定义

架构：
START -> Step1(全局语义) -> Step2(Router决策) 
       -> [分支A: 纯文本] -> MarkItDown Worker
       -> [分支B: 简单] -> VLM Locator (找坐标) -> [Map分发] -> Element Worker
       -> [分支C: 复杂] -> Layout Pipeline (找坐标) -> [Map分发] -> Element Worker
       -> [汇聚] -> Step3(Supervisor校验) 
       -> [循环] -> (如果校验失败) -> 回到 Step2 Router (仅重试失败元素)
       -> Step4(输出) -> END

逐页处理流程（每页独立）：
  Step1(全局分析) → Step2-Router(规划) → [Step2-Workers(并行)] → Step3(Supervisor) → Step4(输出生成)
  
控制逻辑：
- 最大并发页面数: max_concurrent_pages
- 优先级: 页面索引越小越高（前面的页面优先处理）
- Supervisor失败时: 重试最多max_retries次
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
from typing import List, Dict

from state import PPTWorkflowState, PPTPageState
from nodes.step1_global_analysis import (
    node_ingestion, 
    node_step1_global_analysis,
    PPTIngestionEngine,
    Step1_GlobalAnalysisEngine
)
from nodes.step2_router import node_step2_router
from nodes.step2_workers import (
    node_text_worker,      # 处理纯文本
    node_element_worker,   # 处理带坐标的元素 (Worker)
)
from nodes.step2_locator import node_vlm_locator, node_layout_pipeline, node_type_normalization
from nodes.step3_supervisor import node_step3_supervisor
from nodes.step4_output import node_step4_output_generation
from nodes.step5_editor import node_step5_editor


# ============================================================================
# 1. 页面级子图 (Page Workflow)
# ============================================================================

def create_page_workflow(llm_client, vlm_client, debug_logger=None):
    """创建处理单个页面的子工作流"""
    
    workflow = StateGraph(PPTPageState)
    
    # --- 添加节点 ---
    
    # Step 1: 全局语义分析 (无坐标)
    workflow.add_node("step1_global", lambda state: node_step1_global_analysis(state, vlm_client, debug_logger))
    
    # Step 2: 路由决策
    workflow.add_node("step2_router", lambda state: node_step2_router(state, debug_logger))
    
    # Step 2.5: 定位分支 (Locators)
    workflow.add_node("locator_vlm", lambda state: node_vlm_locator(state, vlm_client, debug_logger))
    # workflow.add_node("locator_pipeline", node_layout_pipeline)
    workflow.add_node("locator_pipeline", lambda state: node_layout_pipeline(state, vlm_client, debug_logger))
    
    # Step 2: 执行 Workers
    workflow.add_node("worker_markitdown", lambda state: node_text_worker(state, llm_client, debug_logger))
    workflow.add_node("worker_element", lambda state: node_element_worker(state, vlm_client, debug_logger))
    
    # Step 3: 监督与校验
    workflow.add_node("step3_supervisor", lambda state: node_step3_supervisor(state, llm_client, debug_logger))
    
    # Step 4: 输出生成
    workflow.add_node("step4_output", lambda state: node_step4_output_generation(state, debug_logger))

    # Step 5: 结果汇总
    workflow.add_node("step5_editor", lambda state: node_step5_editor(state, llm_client, debug_logger))
    
    # --- 定义边 (逻辑流转) ---
    
    # 启动 -> Step 1
    workflow.add_edge(START, "step1_global")
    workflow.add_edge("step1_global", "step2_router")
    
    # Router 后的条件分支
    def route_after_router(state: PPTPageState):
        plan = state.step2_plan
        route = plan.get("route")
        
        if route == "text_only":
            return "worker_markitdown"
        elif route == "layout_pipeline":
            return "locator_pipeline"
        else:
            # 默认兜底走 pipeline，避免误入 VLM 分支
            return "locator_pipeline"

    workflow.add_conditional_edges(
        "step2_router",
        route_after_router,
        {
            "worker_markitdown": "worker_markitdown",
            "locator_pipeline": "locator_pipeline",
            "locator_vlm": "locator_vlm"
        }
    )
    
    # 定位完成后的 Map (分发) 逻辑
    def map_elements_to_workers(state: PPTPageState) -> List[Send]:
        """将定位好的 pending_elements 分发给 workers"""
        latest_status: Dict[str, str] = {}
        for insight in state.element_insights or []:
            latest_status[insight.element_id] = insight.status

        sends = []
        for element in state.pending_elements:
            # 已成功处理的元素不再重复分析（包括重试场景）
            if latest_status.get(element.element_id) == "pass":
                continue

            # 检查是否有 supervisor 的反馈意见
            feedback = state.supervisor_feedback.get(element.element_id)
            
            # 构造 Payload
            payload = {
                "element": element,
                "global_summary": state.global_analysis.core_summary,
                "hypothesis": state.global_analysis.research_hypothesis,
                "image_path": state.image_path,
                "feedback": feedback
            }
            sends.append(Send("worker_element", payload))

        # 没有可分发元素时，发送一个 no-op worker 触发后续汇聚
        if not sends:
            sends.append(Send("worker_element", {"skip_worker": True, "page_index": state.page_index}))
        
        return sends

    # Locator 后进行 Map 分发
    workflow.add_conditional_edges("locator_vlm", map_elements_to_workers, ["worker_element"])
    workflow.add_conditional_edges("locator_pipeline", map_elements_to_workers, ["worker_element"])
    # worker 分发是在 map_elements_to_workers 时动态决定

    
    # Workers 汇聚到 Supervisor
    workflow.add_edge("worker_markitdown", "step3_supervisor")
    workflow.add_edge("worker_element", "step3_supervisor")
    
    # Supervisor 后的条件分支 (重试逻辑)
    def route_after_supervisor(state: PPTPageState):
        # 检查校验结果，如果失败且还有重试次数，进入重试模式
        if state.is_retry_mode and state.retry_count < state.max_retries:
            print(f"[Router] 页面 {state.page_index} 校验失败，准备重试 (第 {state.retry_count}/{state.max_retries} 次)")
            return "step2_router"
        
        # 否则直接输出（已达最大重试次数或校验通过）
        return "step4_output"

    workflow.add_conditional_edges(
        "step3_supervisor",
        route_after_supervisor,
        {
            "step2_router": "step2_router",
            "step4_output": "step4_output"
        }
    )
    
    workflow.add_edge("step4_output", "step5_editor")
    workflow.add_edge("step5_editor", END)
    
    return workflow.compile()


# ============================================================================
# 2. 主工作流 (Main Workflow)
# ============================================================================
# 这部分逻辑主要负责读取文件和批处理，保持原有的简洁性即可

def node_ingestion(state: PPTWorkflowState) -> dict:
    """读取PPT节点"""
    from nodes.step1_global_analysis import PPTIngestionEngine
    image_paths = PPTIngestionEngine.extract_all_slides(state.ppt_path)
    total_pages = len(image_paths)
    return {
        "page_queue": list(range(total_pages)),
        "total_pages": total_pages,
        # 假设这里有一个 mapping 存储页面图片路径
        # 实际项目中建议在 state 中维护一个 page_idx -> image_path 的字典
    }

def node_prepare_batch(state: PPTWorkflowState) -> dict:
    """准备批次"""
    batch = state.page_queue[:state.max_concurrent_pages]
    remaining = state.page_queue[state.max_concurrent_pages:]
    return {"current_batch": batch, "page_queue": remaining}

def node_dispatch_pages(state: PPTWorkflowState) -> List[Send]:
    """分发页面任务"""
    batch = state.current_batch
    sends = []
    for page_idx in batch:
        # 构造页面初始状态
        image_path = f"ppt_slides_temp/slide_{page_idx:03d}.png" # 示例路径
        
        page_state = PPTPageState(
            page_index=page_idx,
            ppt_path=state.ppt_path,
            image_path=image_path,
            max_retries=getattr(state, "max_retries_per_page", 2)
        )
        sends.append(Send("page_workflow", page_state))
    return sends

# 将 Node 变为纯占位符/透传节点
def node_dispatch_placeholder(state: PPTWorkflowState) -> dict:
    """
    分发任务的占位节点。
    实际的 Send 逻辑将在离开此节点的 Conditional Edge 中执行。
    """
    print(f"\n[Dispatch] 准备分发批次: {state.current_batch}")
    return {} # 返回空字典，仅作为流程锚点

#  将生成 Send 的逻辑移动到独立的映射函数中
def map_pages_to_workflow(state: PPTWorkflowState) -> List[Send]:
    """生成并行任务的映射函数
    
    为每个页面创建独立的 PPTPageState，发送到 page_workflow 节点进行处理
    """
    from pathlib import Path
    
    batch = state.current_batch
    sends = []
    
    # 根据 PPT 路径计算 ppt_slides_temp 目录
    ppt_dir = Path(state.ppt_path).parent.resolve()
    slides_dir = ppt_dir / "ppt_slides_temp"
    
    for page_idx in batch:
        # 使用绝对路径或相对于 PPT 目录的路径
        image_path = str(slides_dir / f"slide_{page_idx:03d}.png")
        
        # 创建页面状态
        page_state = PPTPageState(
            page_index=page_idx,
            ppt_path=state.ppt_path,  # 传递 PPT 路径供文本解析使用
            image_path=image_path,
            max_retries=2,
            previous_context=state.global_knowledge_base
        )

        sends.append(Send("page_workflow", page_state))
    
    return sends


def node_collect_results(state: PPTWorkflowState) -> dict:
    """汇总结果
    
    这个节点在所有批次处理完毕时被调用，主要用于统计和收尾。
    实际的结果收集是通过 completed_outputs reducer 在 LangGraph 中自动完成的。
    """
    print(f"\n[CollectResults] 已完成 {state.completed_pages} 页")
    print(f"[CollectResults] 已收集 {len(state.completed_outputs)} 页的输出")
    return {}


def node_merge_page_result(state: PPTWorkflowState) -> dict:
    """
    从页面工作流返回后的结果合并节点
    
    LangGraph 在处理 Send 返回的 PPTPageState 时，会自动尝试合并到主状态。
    但由于字段不兼容，我们需要手动处理。
    这个节点接收来自子图的输出并进行转换。
    """
    # 注意：这个函数在 page_workflow 返回后被调用
    # 此时 state 可能包含了来自子图的部分数据
    # 我们的目标是确保 completed_outputs 被正确填充
    
    # 实际上，这个函数不会被正常调用，因为 Send 不是标准的节点返回
    # 真正的解决方案是让子图不返回 ppt_path 等冲突字段
    # 这已经在 PPTPageState 定义中通过移除 ppt_path 来解决了
    pass


def build_ppt_workflow(llm_client, vlm_client, debug_logger=None):
    """构建完整工作流"""
    
    # 1. 预编译子图
    page_graph = create_page_workflow(llm_client, vlm_client, debug_logger)

    # 2. 创建一个包装函数，处理子图返回值并转换为主工作流兼容的格式
    def wrapped_page_workflow(page_state: PPTPageState) -> dict:
        """
        包装页面工作流的调用，处理状态转换
        
        PPTPageState -> 执行页面工作流 -> 提取 final_output -> 返回主工作流兼容格式
        """
        page_idx = page_state.page_index
        
        # 记录页面开始
        if debug_logger:
            debug_logger.logger.info(f"[Page {page_idx}] 开始处理页面")
        
        # 调用子图
        result_state = page_graph.invoke(page_state)
        
        # 提取最终输出
        final_output = result_state.get("final_output")
        page_idx = result_state.get("page_index")

        # 提取本页的核心结论，供更新全局知识库
        page_summary = final_output.summary if final_output else ""
        
        # 记录页面完成
        if debug_logger:
            debug_logger.logger.info(f"[Page {page_idx}] 页面处理完成")
        
        # 返回只包含主工作流兼容字段的字典
        # 这样 LangGraph 就只会尝试合并这些字段，不会尝试合并 image_path 等
        return {
            "completed_outputs": {page_idx: final_output},
            "completed_pages": 1,  # 这个页面已完成
            "global_knowledge_base": f"\n[Page {page_idx}] {page_summary}"
        }

    # 3. 创建主工作流
    workflow = StateGraph(PPTWorkflowState)
    # 初始化 Checkpointer
    checkpointer = MemorySaver()
    
    # 注册节点
    workflow.add_node("ingestion", node_ingestion)
    workflow.add_node("prepare_batch", node_prepare_batch)
    workflow.add_node("dispatch_pages", node_dispatch_placeholder)
    workflow.add_node("collect_results", node_collect_results)
    
    # 注册包装后的子图节点，而不是直接注册子图
    # 这样可以控制返回什么字段
    workflow.add_node("page_workflow", wrapped_page_workflow)
    
    # 定义边
    workflow.set_entry_point("ingestion")
    workflow.add_edge("ingestion", "prepare_batch")
    
    def check_has_pages(state):
        """检查是否还有页面待处理"""
        if not state.current_batch:
            return "collect_results"
        return "dispatch_pages"
    
    workflow.add_conditional_edges(
        "prepare_batch",
        check_has_pages,
        {
            "dispatch_pages": "dispatch_pages",
            "collect_results": "collect_results"
        }
    )
    
    # 从 dispatch_pages 分发任务给 page_workflow
    workflow.add_conditional_edges(
        "dispatch_pages", 
        map_pages_to_workflow, 
        ["page_workflow"]
    )
    
    # page_workflow 完成后，回到 prepare_batch 获取下一批
    workflow.add_edge("page_workflow", "prepare_batch")
    workflow.add_edge("collect_results", END)
    
    return workflow.compile(checkpointer=checkpointer)