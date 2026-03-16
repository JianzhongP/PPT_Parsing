"""PPT解析工作流配置文件"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal
from openai import AzureOpenAI, OpenAI

# 加载 .env 文件
from dotenv import load_dotenv
dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path, override=True)
print(f"[Config] 从 {dotenv_path} 加载环境变量")

# 导入 VLM 客户端
from vlm_client import VLMClient


@dataclass
class PPTWorkflowConfig:
    """PPT工作流配置"""
    
    # ============ LLM API 配置 (Azure OpenAI) ============
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_azure_endpoint: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", ""))
    llm_model_name: str = field(default_factory=lambda: os.getenv("LLM_MODEL_NAME", "gpt-4-turbo"))
    api_version: str = "2024-08-01-preview"
    
    # ============ VLM API 配置 (Dashscope Qwen) ============
    vlm_api_key: str = field(default_factory=lambda: os.getenv("VLM_API_KEY", ""))
    vlm_base_url: str = field(default_factory=lambda: os.getenv("VLM_BASE_URL", ""))
    vlm_model_name: str = field(default_factory=lambda: os.getenv("VLM_MODEL_NAME", "qwen3-vl-plus"))
    
    g4o_api_key: str = field(default_factory=lambda: os.getenv("G4O_API_KEY", ""))
    g4o_azure_endpoint: str = field(default_factory=lambda: os.getenv("G4O_AZURE_ENDPOINT", ""))
    g4o_api_version: str = field(default_factory=lambda: os.getenv("G4O_API_VERSION", "2024-02-15-preview"))
    g4o_model_name: str = field(default_factory=lambda: os.getenv("G4O_MODEL_NAME", "gpt-4o"))

    # ============ 多模态模型路由配置 ============
    multimodal_provider: str = field(default_factory=lambda: os.getenv("MULTIMODAL_PROVIDER", "gpt4o"))
    """多模态模型提供方：gpt4o 或 qwen3-vl-plus（支持别名 qwen）"""
    # ============ 第三方工具 API ============
    ppt_converter_api_url: str = field(default_factory=lambda: os.getenv("PPT_CONVERTER_API_URL", ""))
    ppt_converter_api_key: str = field(default_factory=lambda: os.getenv("PPT_CONVERTER_API_KEY", ""))

    # ============ 工作流参数 ============
    max_concurrent_pages: int = 3
    """最大并发处理的页面数，越靠前的页面优先级越高"""
    
    max_retries_per_page: int = 2
    """每个页面最多重试次数（Supervisor校验失败时）"""
    
    output_dir: str = "ppt_parsing_output"
    """输出目录"""
    
    save_intermediate: bool = True
    """是否保存中间过程结果"""
    
    # ============ 解析策略参数 ============
    enable_layout_pipeline: bool = True
    """是否启用布局分析pipeline（Router 硬开关）"""

    route_low_to_pipeline: bool = True
    """low 复杂度页面是否走 MinerU pipeline（默认开启）"""

    route_medium_to_pipeline: bool = True
    """medium 复杂度页面是否走 MinerU pipeline（默认开启）"""

    route_high_to_vlm: bool = True
    """high/边界不清页面是否走 MinerU VLM 模式（默认开启）"""

    force_all_non_text_to_pipeline: bool = True
    """临时开关：非纯文本页面统一走 MinerU pipeline，禁用 VLM 路由"""
    
    use_markitdown_for_text: bool = True
    """纯文本页面是否使用MarkItDown解析"""

    # ============ 元素过滤/裁剪防噪参数 ============
    filter_decorative_elements: bool = True
    """是否过滤背景/装饰/边框/页眉页脚等非语义元素，避免进入裁剪与分析"""

    # Locator 返回 bbox 后的通用规则过滤（基于 0-1000 归一化坐标）
    bbox_min_area_ratio: float = 0.025
    """过小 bbox 过滤阈值（面积占比），用于剔除 icon/噪声点"""
    bbox_max_area_ratio: float = 0.60
    """过大 bbox 过滤阈值（面积占比），用于剔除背景块/整页底纹"""
    bbox_extreme_aspect_ratio: float = 10.0
    """极端长宽比过滤阈值，用于剔除分割线/边框长条"""
    bbox_edge_margin_ratio: float = 0.05
    """贴边判定阈值（边距占比），用于识别页眉/页脚/边框"""
    bbox_edge_thin_ratio: float = 0.12
    """贴边且很薄的判定阈值（宽/高占比），用于剔除页眉页脚/边框"""

    # ============ 元素去噪/去重与文本聚合参数 ============
    element_text_min_len: int = 3
    """短文本最小长度，低于该值默认按噪声处理"""

    element_footer_zone_ratio: float = 0.10
    """页面底部区域比例，位于该区域且满足细条条件时视为 footer 噪声"""

    element_corner_zone_ratio: float = 0.12
    """页面四角区域比例，位于四角且面积很小时优先视为角标噪声"""

    element_corner_max_area_ratio: float = 0.02
    """角标噪声最大面积占比"""

    element_dedup_iou_threshold: float = 0.85
    """元素去重 IoU 阈值"""

    aggregate_text_elements: bool = True
    """是否将 text 元素聚合为单条洞察，避免逐块进入 LLM"""

    aggregate_text_to_llm: bool = False
    """聚合文本是否再走一次 LLM（默认关闭，先做轻量聚合）"""

    text_compare_similarity_threshold: float = 0.60
    """聚合文本与 Step1/XML 的最小相似度阈值"""

    # ============ Step2.6 类型纠偏参数 ============
    enable_type_correction: bool = True
    """是否启用 Locator 后的整页类型纠偏（策略B：标框图 + VLM）"""

    type_correction_max_elements: int = 24
    """单页参与类型纠偏的最大元素数，避免超长 prompt"""

    type_correction_container_area_ratio: float = 0.10
    """大框包小框判定阈值（父框最小面积占比）"""

    # ============ Step2.5 缓存参数 ============
    enable_layout_cache: bool = True
    """是否启用 Step2.5 布局结果缓存（重试时复用）"""

    layout_cache_dir: str = "processing_artifacts/layout_cache"
    """Step2.5 布局缓存目录"""
    
    # ============ 日志参数 ============
    verbose: bool = True
    """是否输出详细日志"""
    
    log_file: str = "ppt_workflow.log"
    """日志文件路径"""

    @property
    def normalized_multimodal_provider(self) -> Literal["gpt4o", "qwen3-vl-plus"]:
        provider = (self.multimodal_provider or "").strip().lower()
        if provider in {"qwen", "qwen3-vl-plus", "dashscope"}:
            return "qwen3-vl-plus"
        return "gpt4o"

    @property
    def vlm_runtime_model_name(self) -> str:
        """返回当前多模态路由下的实际模型名。"""
        if self.normalized_multimodal_provider == "qwen3-vl-plus":
            return self.vlm_model_name
        return self.g4o_model_name


def get_config() -> PPTWorkflowConfig:
    """获取配置实例"""
    return PPTWorkflowConfig()


def init_api_clients():
    """初始化API客户端
    
    LLM: Azure OpenAI GPT-4
    VLM: Qwen 视觉模型 (优先使用原生 Dashscope SDK，降级到 OpenAI 兼容模式)
    """
    config = get_config()
    
    # 诊断配置
    print("\n" + "="*70)
    print("[Config] API 配置诊断")
    print("="*70)
    
    # 1. LLM Client (GPT-4 via Azure OpenAI)
    print(f"\n[LLM] LLM 配置:")
    print(f"  - API Key: {'[OK]' if config.llm_api_key else '[MISSING]'}")
    print(f"  - 端点: {config.llm_azure_endpoint if config.llm_azure_endpoint else '[MISSING]'}")
    print(f"  - 模型: {config.llm_model_name}")
    print(f"  - API版本: {config.api_version}")
    
    if not config.llm_api_key:
        raise ValueError("[ERROR] LLM_API_KEY 未设置，请检查 .env 文件")
    if not config.llm_azure_endpoint:
        raise ValueError("[ERROR] LLM_BASE_URL 未设置，请检查 .env 文件")
    
    try:
        llm_client = AzureOpenAI(
            api_key=config.llm_api_key,
            azure_endpoint=config.llm_azure_endpoint,
            api_version=config.api_version
        )
        print(f"  [OK] Azure OpenAI 客户端初始化成功")
    except Exception as e:
        print(f"  [ERROR] Azure OpenAI 客户端初始化失败: {e}")
        raise

    # 2. VLM Client (可配置：GPT-4o 或 Qwen3-VL-Plus)
    selected_provider = config.normalized_multimodal_provider
    print(f"\n[VLM] VLM 配置:")
    print(f"  - 提供方: {selected_provider}")
    print(f"  - 运行模型: {config.vlm_runtime_model_name}")

    try:
        if selected_provider == "qwen3-vl-plus":
            print(f"  - Qwen API Key: {'[OK]' if config.vlm_api_key else '[MISSING]'}")
            print(f"  - Qwen 端点: {config.vlm_base_url if config.vlm_base_url else '[MISSING]'}")
            if not config.vlm_api_key:
                raise ValueError("[ERROR] VLM_API_KEY 未设置，请检查 .env 文件")
            if not config.vlm_base_url:
                raise ValueError("[ERROR] VLM_BASE_URL 未设置，请检查 .env 文件")

            vlm_client = VLMClient(
                api_key=config.vlm_api_key,
                base_url=config.vlm_base_url,
                model_name=config.vlm_runtime_model_name,
            )
        else:
            print(f"  - GPT-4o API Key: {'[OK]' if config.g4o_api_key else '[MISSING]'}")
            print(f"  - GPT-4o 端点: {config.g4o_azure_endpoint if config.g4o_azure_endpoint else '[MISSING]'}")
            print(f"  - API版本: {config.g4o_api_version}")
            if not config.g4o_api_key:
                raise ValueError("[ERROR] G4O_API_KEY 未设置，请检查 .env 文件")
            if not config.g4o_azure_endpoint:
                raise ValueError("[ERROR] G4O_AZURE_ENDPOINT 未设置，请检查 .env 文件")

            vlm_client = AzureOpenAI(
                api_key=config.g4o_api_key,
                azure_endpoint=config.g4o_azure_endpoint,
                api_version=config.g4o_api_version,
            )

        print(f"  [OK] VLM 客户端初始化成功")
    except Exception as e:
        print(f"  [ERROR] VLM 客户端初始化失败: {e}")
        raise
    
    print("\n" + "="*70 + "\n")
    return llm_client, vlm_client
