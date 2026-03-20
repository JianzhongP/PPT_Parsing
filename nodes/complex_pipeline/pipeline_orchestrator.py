"""
复杂页面处理 Pipeline 编排器

整合 Phase 1-4，提供统一的入口和节点函数
"""

import os
import time
from typing import Dict, Any, Optional

from config import get_config

from .pipeline_state import (
    ComplexPipelineState,
    CleanedLayoutJSON, DraftSemanticJSON, ValidatedSemanticJSON,
    ExtractedDataMap, FinalStructuredJSON, FinalMarkdownOutput
)
from .phase1_layout_detection import Phase1_LayoutDetector
from .phase2_semantic_grouping import Phase2_SemanticGrouper
from .phase2_supervisor import Phase2_SupervisorValidator
from .phase3_expert_extraction import Phase3_ParallelExtractor
from .phase4_assembly import Phase4_Assembler
from .pipeline_debug_logger import PipelineDebugLogger


class ComplexPipelineOrchestrator:
    """
    复杂页面处理 Pipeline 编排器
    
    整合四个阶段的处理流程：
    - Phase 1: 版面检测与类型清洗
    - Phase 2-1: 多模态语义分组
    - Phase 2-2: 监督节点验证
    - Phase 3: 并行特征提取
    - Phase 4: 统一组装与输出
    """
    
    def __init__(self,
                 vlm_client,
                 llm_client=None,
                 output_dir: str = "processing_artifacts",
                 textin_api_key: Optional[str] = None,
                 use_mineru_vlm: bool = False,
                 max_workers: int = 4,
                 debug_mode: bool = False,
                 debug_verbose: bool = True):
        """
        初始化编排器
        
        Args:
            vlm_client: VLM客户端（GPT-4o）
            llm_client: LLM客户端（可选）
            output_dir: 输出目录
            textin_api_key: TextIn API密钥
            use_mineru_vlm: 是否使用MinerU VLM版本（GPU分支）
            max_workers: 并行工作线程数
            debug_mode: 是否启用调试模式（记录每步中间输出）
            debug_verbose: 调试模式下是否输出详细日志到控制台
        """
        self.vlm_client = vlm_client
        self.llm_client = llm_client or vlm_client
        self.output_dir = output_dir
        self.debug_mode = debug_mode
        self.debug_verbose = debug_verbose
        self.debug_logger = None  # 运行时初始化
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化各阶段控制器
        self.phase1_detector = Phase1_LayoutDetector(
            vlm_client=vlm_client,
            output_dir=output_dir,
            use_mineru_vlm=use_mineru_vlm
        )
        
        self.phase2_grouper = Phase2_SemanticGrouper(
            vlm_client=vlm_client,
            output_dir=output_dir
        )
        
        self.phase2_supervisor = Phase2_SupervisorValidator(
            vlm_client=vlm_client,
            llm_client=llm_client
        )
        
        self.phase3_extractor = Phase3_ParallelExtractor(
            vlm_client=vlm_client,
            textin_api_key=textin_api_key,
            max_workers=max_workers
        )
        
        self.phase4_assembler = Phase4_Assembler(llm_client=self.llm_client)
    
    def run(self,
           image_path: str,
           page_id: int,
           global_analysis: Optional[Dict[str, Any]] = None,
            page_title: str = "",
            precomputed_mineru_layout: Optional[Dict[str, Any]] = None) -> ComplexPipelineState:
        """
        执行完整的复杂页面处理流程
        
        Args:
            image_path: PPT页面图片路径
            page_id: 页面ID
            global_analysis: Step1传入的全局分析结果
            page_title: 页面标题
            
        Returns:
            ComplexPipelineState: 完整的处理状态和结果
        """
        total_start = time.time()
        
        print("\n" + "="*70)
        print(f"  复杂页面处理 Pipeline 启动")
        print(f"  页面ID: {page_id}")
        print(f"  图片路径: {image_path}")
        print(f"  调试模式: {'开启' if self.debug_mode else '关闭'}")
        print("="*70)
        
        # 初始化调试日志器
        if self.debug_mode:
            self.debug_logger = PipelineDebugLogger(
                output_dir=self.output_dir,
                page_id=page_id,
                verbose=self.debug_verbose
            )
        
        # 初始化状态
        state = ComplexPipelineState(
            page_id=page_id,
            origin_image_path=image_path,
            global_analysis=global_analysis,
            current_phase="phase1"
        )
        
        try:
            # ============================================================
            # Phase 1: 版面检测与类型清洗
            # ============================================================
            phase1_start = time.time()
            state.current_phase = "phase1"
            
            # 调试日志: Phase 1 开始
            if self.debug_logger:
                self.debug_logger.start_phase("phase1", {
                    "image_path": image_path,
                    "page_id": page_id,
                    "global_analysis": global_analysis
                })
            
            state.phase1_output = self.phase1_detector.run(
                image_path=image_path,
                page_id=page_id,
                global_analysis=global_analysis,
                precomputed_layout=precomputed_mineru_layout,
            )
            
            state.phase_timings["phase1"] = int((time.time() - phase1_start) * 1000)
            
            # 调试日志: Phase 1 结束
            if self.debug_logger:
                # 记录 MinerU 原始结果
                mineru_client = getattr(self.phase1_detector, 'mineru_client', None)
                if mineru_client:
                    self.debug_logger.log_phase1_mineru_result(
                        raw_elements=[elem.model_dump() for elem in state.phase1_output.elements],
                        mineru_output_dir=getattr(mineru_client, 'last_run_output_dir', None),
                        mineru_cmd=str(getattr(mineru_client, 'last_run_cmd', None)),
                        returncode=getattr(mineru_client, 'last_run_returncode', None)
                    )
                
                # 记录 ROI 裁剪
                self.debug_logger.log_phase1_roi_crops(state.phase1_output.roi_images)
                
                # 记录每个元素的类型清洗结果
                for elem in state.phase1_output.elements:
                    if elem.vlm_refinement_response:
                        self.debug_logger.log_phase1_vlm_refinement(
                            element_id=elem.element_id,
                            original_type=elem.original_type,
                            refined_type=elem.refined_type or "未清洗",
                            prompt="(见ROI图片)",
                            response=elem.vlm_refinement_response
                        )
                
                self.debug_logger.end_phase("phase1", state.phase1_output)
            
            # ============================================================
            # Phase 2-1: 多模态语义分组
            # ============================================================
            phase2_1_start = time.time()
            state.current_phase = "phase2_grouping"
            
            # 调试日志: Phase 2-1 开始
            if self.debug_logger:
                self.debug_logger.start_phase("phase2_grouping", {
                    "elements_count": len(state.phase1_output.elements),
                    "roi_images_count": len(state.phase1_output.roi_images)
                })
            
            global_summary = ""
            if global_analysis:
                global_summary = global_analysis.get("core_summary", "")
            
            state.phase2_draft = self.phase2_grouper.run(
                cleaned_layout=state.phase1_output,
                global_summary=global_summary
            )
            
            state.phase_timings["phase2_grouping"] = int((time.time() - phase2_1_start) * 1000)
            
            # 调试日志: Phase 2-1 结束
            if self.debug_logger:
                if state.phase2_draft.som_image_path:
                    self.debug_logger.log_phase2_som_image(
                        state.phase2_draft.som_image_path,
                        len(state.phase1_output.elements)
                    )
                self.debug_logger.log_phase2_grouping_result(
                    state.phase2_draft.groups,
                    state.phase2_draft.ungrouped_element_ids
                )
                self.debug_logger.end_phase("phase2_grouping", state.phase2_draft)
            
            # ============================================================
            # Phase 2-2: 监督节点验证（带重试循环）
            # ============================================================
            phase2_2_start = time.time()
            state.current_phase = "phase2_validation"
            
            # 调试日志: Phase 2-2 开始
            if self.debug_logger:
                self.debug_logger.start_phase("phase2_validation", {
                    "groups_count": len(state.phase2_draft.groups),
                    "ungrouped_count": len(state.phase2_draft.ungrouped_element_ids)
                })
            
            need_retry = True
            while need_retry:
                validated_result, need_retry = self.phase2_supervisor.validate(
                    draft_semantic=state.phase2_draft,
                    elements=state.phase1_output.elements,
                    som_image_path=state.phase2_draft.som_image_path,
                    retry_count=state.phase2_retry_count,
                    page_title=page_title,
                )
                
                state.phase2_validated = validated_result
                
                # 调试日志: 记录验证结果
                if self.debug_logger:
                    self.debug_logger.log_phase2_validation(
                        validated_result.validation_result,
                        validated_result.validation_result.issues
                    )
                
                if need_retry:
                    state.phase2_retry_count += 1
                    print(f"\n[Pipeline] Phase 2 重试 (第 {state.phase2_retry_count} 次)...")
                    
                    if self.debug_logger:
                        self.debug_logger.log_warning("phase2_validation", 
                            f"触发重试 (第 {state.phase2_retry_count} 次)")
                    
                    # 应用修正后重新分组
                    corrected_draft = self.phase2_supervisor.apply_corrections(
                        state.phase2_draft,
                        state.phase1_output.elements,
                        validated_result.validation_result.issues
                    )
                    state.phase2_draft = corrected_draft
            
            state.phase_timings["phase2_validation"] = int((time.time() - phase2_2_start) * 1000)
            
            # 调试日志: Phase 2-2 结束
            if self.debug_logger:
                self.debug_logger.end_phase("phase2_validation", state.phase2_validated)
            
            # ============================================================
            # Phase 3: 并行特征提取
            # ============================================================
            phase3_start = time.time()
            state.current_phase = "phase3"
            
            # 调试日志: Phase 3 开始
            if self.debug_logger:
                self.debug_logger.start_phase("phase3", {
                    "groups_to_extract": len(state.phase2_validated.groups),
                    "elements_to_extract": [e.element_id for e in state.phase1_output.elements]
                })
            
            state.phase3_output = self.phase3_extractor.run(
                validated_semantic=state.phase2_validated,
                roi_images=state.phase1_output.roi_images,
                all_elements=state.phase1_output.elements,
                page_title=page_title,
            )
            
            state.phase_timings["phase3"] = int((time.time() - phase3_start) * 1000)
            
            # 调试日志: Phase 3 结束
            if self.debug_logger:
                for elem_id, content in state.phase3_output.extractions.items():
                    self.debug_logger.log_phase3_extraction(
                        elem_id,
                        content.element_type,
                        content,
                        content.raw_response
                    )
                self.debug_logger.end_phase("phase3", state.phase3_output)
            
            # ============================================================
            # Phase 4: 统一组装与输出
            # ============================================================
            phase4_start = time.time()
            state.current_phase = "phase4"
            
            # 调试日志: Phase 4 开始
            if self.debug_logger:
                self.debug_logger.start_phase("phase4", {
                    "extractions_count": len(state.phase3_output.extractions)
                })
            
            final_json, final_markdown = self.phase4_assembler.run(
                validated_semantic=state.phase2_validated,
                extracted_data=state.phase3_output,
                page_title=page_title
            )
            
            state.final_json = final_json
            state.final_markdown = final_markdown
            
            state.phase_timings["phase4"] = int((time.time() - phase4_start) * 1000)
            
            # 调试日志: Phase 4 结束
            if self.debug_logger:
                md_preview = final_markdown.markdown_content[:500] if final_markdown else ""
                self.debug_logger.log_phase4_assembly(
                    final_json.semantic_blocks if final_json else [],
                    md_preview
                )
                self.debug_logger.end_phase("phase4", {
                    "final_json": final_json.model_dump() if final_json else None,
                    "final_markdown_length": len(final_markdown.markdown_content) if final_markdown else 0
                })
            
            # 完成
            state.current_phase = "completed"
            
        except Exception as e:
            state.current_phase = "failed"
            state.error_log.append(f"Pipeline执行失败: {str(e)}")
            print(f"\n❌ [Pipeline] 执行失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 调试日志: 记录错误
            if self.debug_logger:
                self.debug_logger.log_error(state.current_phase, str(e))
        
        # 总结
        total_time = int((time.time() - total_start) * 1000)
        state.phase_timings["total"] = total_time
        
        print("\n" + "="*70)
        print(f"  复杂页面处理 Pipeline 完成")
        print(f"  状态: {state.current_phase}")
        print(f"  总耗时: {total_time}ms")
        print(f"  各阶段耗时: {state.phase_timings}")
        print("="*70)
        
        # 生成调试汇总报告
        if self.debug_logger:
            self.debug_logger.generate_summary_report()
            # 将调试目录路径存储到 state 中，便于后续访问
            state.error_log.append(f"[DEBUG] 调试目录: {self.debug_logger.get_debug_dir()}")
        
        return state
    
    def save_results(self, state: ComplexPipelineState, output_dir: Optional[str] = None):
        """
        保存处理结果到文件
        
        Args:
            state: 完整的处理状态
            output_dir: 输出目录（默认使用初始化时的目录）
        """
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        page_id = state.page_id
        
        # 保存结构化JSON
        if state.final_json:
            json_path = os.path.join(output_dir, f"page_{page_id:03d}_structured.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                import json
                f.write(state.final_json.model_dump_json(indent=2))
            print(f"[Pipeline] 已保存结构化JSON: {json_path}")
        
        # 保存Markdown
        if state.final_markdown:
            md_path = os.path.join(output_dir, f"page_{page_id:03d}_output.md")
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(state.final_markdown.markdown_content)
            print(f"[Pipeline] 已保存Markdown: {md_path}")
        
        # 保存完整状态（用于调试）
        state_path = os.path.join(output_dir, f"page_{page_id:03d}_state.json")
        with open(state_path, 'w', encoding='utf-8') as f:
            import json
            f.write(state.model_dump_json(indent=2))
        print(f"[Pipeline] 已保存完整状态: {state_path}")


# ============================================================================
# 节点函数（供 LangGraph 调用）
# ============================================================================

def node_complex_pipeline(state, vlm_client, llm_client=None, debug_logger=None, use_mineru_vlm: bool = False) -> dict:
    """
    复杂页面处理节点函数
    
    用于集成到现有的 LangGraph 工作流中
    
    Args:
        state: PPTPageState
        vlm_client: VLM客户端
        llm_client: LLM客户端
        debug_logger: 调试日志记录器
        
    Returns:
        更新后的状态字典
    """
    from state import PPTPageState, PageElement, ElementInsight, BBox
    
    page_index = state.page_index
    image_path = state.image_path
    
    print(f"\n[ComplexPipeline] 处理页面 {page_index}")
    
    # 记录步骤开始
    if debug_logger:
        debug_logger.log_step_start(
            page_index,
            "Complex Pipeline",
            input_data={
                "page_index": page_index,
                "image_path": image_path,
                "has_global_analysis": state.global_analysis is not None
            },
            previous_step="Step 2: Router"
        )
    
    # 准备全局分析数据
    global_analysis_dict = None
    if state.global_analysis:
        global_analysis_dict = {
            "page_index": state.global_analysis.page_index,
            "section_title": state.global_analysis.section_title,
            "core_summary": state.global_analysis.core_summary,
            "elements": [
                {
                    "element_id": e.element_id,
                    "type": e.type,
                    "description": e.description
                }
                for e in state.global_analysis.elements
            ]
        }
    
    # 初始化并运行Pipeline
    cfg = get_config()
    processing_root = str(getattr(cfg, "processing_artifacts_dir", "processing_artifacts") or "processing_artifacts")

    orchestrator = ComplexPipelineOrchestrator(
        vlm_client=vlm_client,
        llm_client=llm_client,
        output_dir=f"{processing_root}/page_{page_index:03d}",
        use_mineru_vlm=bool(use_mineru_vlm),
    )
    
    pipeline_state = orchestrator.run(
        image_path=image_path,
        page_id=page_index,
        global_analysis=global_analysis_dict,
        page_title=state.global_analysis.section_title if state.global_analysis else "",
        precomputed_mineru_layout=getattr(state, "precomputed_mineru_layout", None),
    )
    
    # 保存结果
    orchestrator.save_results(pipeline_state)
    
    # 转换为 ElementInsight 格式（兼容现有流程）
    element_insights = []
    
    if pipeline_state.phase3_output:
        for elem_id, content in pipeline_state.phase3_output.extractions.items():
            # 构建 key_insight
            if content.table_data:
                table_insight = (content.table_data.insight or "").strip()
                if not table_insight and content.table_data.key_values:
                    kv_preview = "，".join(
                        f"{k}: {v}" for k, v in list(content.table_data.key_values.items())[:3]
                    )
                    table_insight = kv_preview
                if not table_insight:
                    table_insight = "已提取表格结构"
                key_insight = f"表格分析: {table_insight}"
                data_evidence = content.table_data.markdown_table
            elif content.chart_data:
                key_insight = f"图表洞察: {content.chart_data.insight}"
                data_evidence = f"类型: {content.chart_data.chart_type}, X轴: {content.chart_data.x_axis}, Y轴: {content.chart_data.y_axis}"
            elif content.formula_data:
                key_insight = f"公式: {content.formula_data.latex}"
                data_evidence = content.formula_data.plain_text or ""
            elif content.text_data:
                key_insight = content.text_data.merged_text[:200]
                data_evidence = ""
            else:
                key_insight = "已处理"
                data_evidence = ""
            
            insight = ElementInsight(
                element_id=elem_id,
                element_type=content.element_type,
                key_insight=key_insight,
                data_evidence=data_evidence,
                confidence=1.0 if content.processing_status == "success" else 0.5,
                status="pass" if content.processing_status == "success" else "fail"
            )
            element_insights.append(insight)
    
    # 转换 pending_elements 格式（如果需要）
    pending_elements = []
    if pipeline_state.phase1_output:
        for elem in pipeline_state.phase1_output.elements:
            page_elem = PageElement(
                element_id=elem.element_id,
                type=elem.refined_type or elem.original_type,
                description=elem.ocr_text or "",
                bbox=BBox(box_2d=elem.bbox.box_2d)
            )
            pending_elements.append(page_elem)
    
    # 记录步骤结束
    if debug_logger:
        debug_logger.log_step_end(
            page_index,
            "Complex Pipeline",
            output_data={
                "status": pipeline_state.current_phase,
                "element_insights_count": len(element_insights),
                "phase_timings": pipeline_state.phase_timings,
                "errors": pipeline_state.error_log
            },
            status="success" if pipeline_state.current_phase == "completed" else "failed"
        )
    
    return {
        "pending_elements": pending_elements,
        "element_insights": element_insights
    }
