"""节点模块初始化"""

try:
    from .step1_global_analysis import (
        PPTIngestionEngine,
        ImageUtils,
        Step1_GlobalAnalysisEngine,
        node_ingestion,
        node_step1_global_analysis,
    )
    STEP1_AVAILABLE = True
except ImportError:
    STEP1_AVAILABLE = False

try:
    from .step2_router import (
        Step2_RouterEngine,
        node_step2_router,
    )
    STEP2_ROUTER_AVAILABLE = True
except ImportError:
    STEP2_ROUTER_AVAILABLE = False

try:
    from .step2_locator import (
        SimpleLocator,
        LayoutAnalysisEngine,
        node_vlm_locator,
        node_layout_pipeline,
    )
    STEP2_LOCATOR_AVAILABLE = True
except ImportError:
    STEP2_LOCATOR_AVAILABLE = False

try:
    from .step2_workers import (
        ElementAnalysisAgent,
        ChartAnalysisAgent,
        TableAnalysisAgent,
        DiagramAnalysisAgent,
        ImageAnalysisAgent,
        get_agent,
        node_element_worker,
    )
    STEP2_WORKERS_AVAILABLE = True
except ImportError:
    STEP2_WORKERS_AVAILABLE = False

try:
    from .step3_supervisor import (
        Step3_SupervisorEngine,
        node_step3_supervisor,
    )
    STEP3_AVAILABLE = True
except ImportError:
    STEP3_AVAILABLE = False

try:
    from .step4_output import (
        Step4_OutputGenerator,
        node_step4_output_generation,
    )
    STEP4_AVAILABLE = True
except ImportError:
    STEP4_AVAILABLE = False

# 复杂页面处理Pipeline
try:
    from .complex_pipeline import (
        # Phase 1
        MinerUClient,
        ROICropper,
        TypeRefinementEngine,
        Phase1_LayoutDetector,
        
        # Phase 2
        CoordinateNormalizer,
        SoMGenerator,
        SemanticGroupingEngine,
        Phase2_SemanticGrouper,
        Phase2_SupervisorValidator,
        
        # Phase 3
        TextInAPIClient,
        TableExpertAgent,
        ChartExpertAgent,
        FormulaExpertAgent,
        TextProcessorAgent,
        Phase3_ParallelExtractor,
        
        # Phase 4
        Phase4_Assembler,
        SchemaValidator,
        
        # Orchestrator
        ComplexPipelineOrchestrator,
        node_complex_pipeline
    )
    COMPLEX_PIPELINE_AVAILABLE = True
except ImportError:
    COMPLEX_PIPELINE_AVAILABLE = False

__all__ = [
    "STEP1_AVAILABLE",
    "STEP2_ROUTER_AVAILABLE",
    "STEP2_LOCATOR_AVAILABLE",
    "STEP2_WORKERS_AVAILABLE",
    "STEP3_AVAILABLE",
    "STEP4_AVAILABLE",
    "COMPLEX_PIPELINE_AVAILABLE",
]

if STEP1_AVAILABLE:
    __all__.extend([
        "PPTIngestionEngine",
        "ImageUtils",
        "Step1_GlobalAnalysisEngine",
        "node_ingestion",
        "node_step1_global_analysis",
    ])

if STEP2_ROUTER_AVAILABLE:
    __all__.extend([
        "Step2_RouterEngine",
        "node_step2_router",
    ])

if STEP2_LOCATOR_AVAILABLE:
    __all__.extend([
        "SimpleLocator",
        "LayoutAnalysisEngine",
        "node_vlm_locator",
        "node_layout_pipeline",
    ])

if STEP2_WORKERS_AVAILABLE:
    __all__.extend([
        "ElementAnalysisAgent",
        "ChartAnalysisAgent",
        "TableAnalysisAgent",
        "DiagramAnalysisAgent",
        "ImageAnalysisAgent",
        "get_agent",
        "node_element_worker",
    ])

if STEP3_AVAILABLE:
    __all__.extend([
        "Step3_SupervisorEngine",
        "node_step3_supervisor",
    ])

if STEP4_AVAILABLE:
    __all__.extend([
        "Step4_OutputGenerator",
        "node_step4_output_generation",
    ])

# 如果复杂Pipeline可用，添加到导出列表
if COMPLEX_PIPELINE_AVAILABLE:
    __all__.extend([
        "MinerUClient",
        "ROICropper",
        "TypeRefinementEngine",
        "Phase1_LayoutDetector",
        "CoordinateNormalizer",
        "SoMGenerator",
        "SemanticGroupingEngine",
        "Phase2_SemanticGrouper",
        "Phase2_SupervisorValidator",
        "TextInAPIClient",
        "TableExpertAgent",
        "ChartExpertAgent",
        "FormulaExpertAgent",
        "TextProcessorAgent",
        "Phase3_ParallelExtractor",
        "Phase4_Assembler",
        "SchemaValidator",
        "ComplexPipelineOrchestrator",
        "node_complex_pipeline"
    ])
