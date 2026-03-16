"""
node_wrappers.py
为所有节点函数添加自动日志记录的包装器
"""

from functools import wraps
from typing import Callable, Any, Dict
from debug_logger import get_debug_logger


def log_node_execution(node_name: str):
    """
    装饰器：自动为节点函数添加日志记录
    
    用法:
    @log_node_execution("step1_global")
    def node_step1_global_analysis(state, vlm_client):
        ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state: Any, *args, **kwargs) -> Dict[str, Any]:
            # 获取debug logger
            debug_logger = None
            
            # 尝试从kwargs中获取debug_logger
            if 'debug_logger' in kwargs:
                debug_logger = kwargs.pop('debug_logger')
            
            # 获取page_index
            page_index = getattr(state, 'page_index', 0)
            
            # 记录开始
            if debug_logger:
                input_summary = {
                    "page_index": page_index,
                    "state_fields": list(state.__dict__.keys()) if hasattr(state, '__dict__') else []
                }
                debug_logger.log_step_start(page_index, node_name, input_summary)
            
            try:
                # 执行原函数
                result = func(state, *args, **kwargs)
                
                # 记录结束
                if debug_logger:
                    output_summary = {
                        "result_fields": list(result.keys()) if isinstance(result, dict) else []
                    }
                    debug_logger.log_step_end(page_index, node_name, output_summary, status="success")
                
                return result
                
            except Exception as e:
                # 记录错误
                if debug_logger:
                    debug_logger.log_error(page_index, node_name, e)
                raise
        
        return wrapper
    return decorator


# 包装所有节点函数
def wrap_all_nodes(llm_client, vlm_client, debug_logger=None):
    """
    导入所有节点函数并返回带有日志功能的版本
    """
    from nodes.step1_global_analysis import (
        node_step1_global_analysis as original_step1,
        node_ingestion as original_ingestion
    )
    from nodes.step2_router import node_step2_router as original_router
    from nodes.step2_locator import (
        node_vlm_locator as original_vlm_locator,
        node_layout_pipeline as original_layout_pipeline
    )
    from nodes.step2_workers import (
        node_text_worker as original_text_worker,
        node_element_worker as original_element_worker
    )
    from nodes.step3_supervisor import node_step3_supervisor as original_supervisor
    from nodes.step4_output import node_step4_output_generation as original_output
    from nodes.step5_editor import node_step5_editor as original_editor
    
    # 创建包装版本
    def node_step1_global_analysis_wrapped(state, vlm_client_arg=None, debug_logger_arg=None):
        debug_logger_arg = debug_logger_arg or debug_logger
        if debug_logger_arg:
            page_index = getattr(state, 'page_index', 0)
            debug_logger_arg.log_step_start(page_index, "Step 1: Global Analysis")
        
        result = original_step1(state, vlm_client_arg or vlm_client)
        
        if debug_logger_arg:
            page_index = getattr(state, 'page_index', 0)
            debug_logger_arg.log_step_end(page_index, "Step 1: Global Analysis", status="success")
        
        return result
    
    return {
        'node_step1_global_analysis': node_step1_global_analysis_wrapped,
        'node_ingestion': original_ingestion,
        'node_step2_router': original_router,
        'node_vlm_locator': original_vlm_locator,
        'node_layout_pipeline': original_layout_pipeline,
        'node_text_worker': original_text_worker,
        'node_element_worker': original_element_worker,
        'node_step3_supervisor': original_supervisor,
        'node_step4_output_generation': original_output,
        'node_step5_editor': original_editor,
    }
