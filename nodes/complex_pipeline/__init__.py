"""
复杂页面处理 Pipeline 模块

该模块实现了针对复杂PPT页面的四阶段处理流程：
- Phase 1: 版面检测与类型清洗 (Layout Detection & Type Refinement)
- Phase 2: 语义挂载与逻辑校验 (Semantic Mounting & Validation)
- Phase 3: 并行特征提取 (Parallel Expert Extraction)
- Phase 4: 统一组装与输出 (Normalization & Assembly)
"""

from .phase1_layout_detection import (
    MinerUClient,
    ROICropper,
    TypeRefinementEngine,
    Phase1_LayoutDetector
)

from .phase2_semantic_grouping import (
    CoordinateNormalizer,
    SoMGenerator,
    SemanticGroupingEngine,
    Phase2_SemanticGrouper
)

from .phase2_supervisor import (
    Phase2_SupervisorValidator,
    GroupingValidationResult
)

from .phase3_expert_extraction import (
    TextInAPIClient,
    TableExpertAgent,
    ChartExpertAgent,
    FormulaExpertAgent,
    TextProcessorAgent,
    Phase3_ParallelExtractor
)

from .phase4_assembly import (
    Phase4_Assembler,
    SchemaValidator
)

from .pipeline_state import (
    CleanedLayoutJSON,
    ROIImage,
    DraftSemanticJSON,
    ValidatedSemanticJSON,
    ExtractedDataMap,
    ComplexPipelineState
)

from .pipeline_orchestrator import (
    ComplexPipelineOrchestrator,
    node_complex_pipeline
)

from .pipeline_debug_logger import (
    PipelineDebugLogger,
    PhaseLogEntry
)

__all__ = [
    # Phase 1
    "MinerUClient",
    "ROICropper", 
    "TypeRefinementEngine",
    "Phase1_LayoutDetector",
    
    # Phase 2-1
    "CoordinateNormalizer",
    "SoMGenerator",
    "SemanticGroupingEngine",
    "Phase2_SemanticGrouper",
    
    # Phase 2-2
    "Phase2_SupervisorValidator",
    "GroupingValidationResult",
    
    # Phase 3
    "TextInAPIClient",
    "TableExpertAgent",
    "ChartExpertAgent",
    "FormulaExpertAgent",
    "TextProcessorAgent",
    "Phase3_ParallelExtractor",
    
    # Phase 4
    "Phase4_Assembler",
    "SchemaValidator",
    
    # State
    "CleanedLayoutJSON",
    "ROIImage",
    "DraftSemanticJSON",
    "ValidatedSemanticJSON",
    "ExtractedDataMap",
    "ComplexPipelineState",
    
    # Orchestrator
    "ComplexPipelineOrchestrator",
    "node_complex_pipeline",
    
    # Debug
    "PipelineDebugLogger",
    "PhaseLogEntry"
]
