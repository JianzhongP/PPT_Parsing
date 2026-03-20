"""
nodes/step1_global_analysis.py
Step 1: PPT读取、图片转换和全局分析
"""

import os
import cv2
import json
import base64
import sys
import subprocess
from pathlib import Path
from typing import List
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from openai import AzureOpenAI

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

# 1. 补充引入 PPTWorkflowState (修复 ImportError 关键)
from state import GlobalPageAnalysis, PageElement, BBox, PPTWorkflowState, PPTPageState

# 2. 导入 VLM 客户端
from vlm_client import VLMClient
from tools.ppt_converter import PPTConverter
from config import get_config


# 装饰/噪声元素关键词（用于后处理过滤，双保险）
_DECORATIVE_KWS = (
    "背景", "底纹", "装饰", "边框", "分割线", "页眉", "页脚", "页码", "水印",
    "logo", "LOGO", "Icon", "icon", "ICON", "纹理", "渐变", "花纹", "点阵",
)


def _filter_decorative_elements(elements: List[PageElement]) -> List[PageElement]:
    """过滤明显的装饰性元素，避免进入定位/裁剪/分析链路。"""
    if not elements:
        return []
    filtered: List[PageElement] = []
    for el in elements:
        desc = (el.description or "").strip()
        if desc and any(k in desc for k in _DECORATIVE_KWS):
            continue
        # 额外：一些模型会把页码/页脚当成 text_block/image
        if el.type and ("footer" in el.type.lower() or "header" in el.type.lower()):
            continue
        filtered.append(el)
    return filtered

class PPTIngestionEngine:
    """PPT文件读取和页面转图片引擎"""
    
    @staticmethod
    def extract_all_slides(ppt_path: str) -> List[str]:
        """
        将PPT的所有页面转换为图片
        策略：优先使用工具集中的 PPTConverter (支持 API + 本地作为 fallback)
        """
        print(f"[PPTIngestion] 开始转换PPT: {ppt_path}")
        
        ppt_abs_path = os.path.abspath(ppt_path)
        output_dir = Path(ppt_path).parent / "ppt_slides_temp"
        output_dir.mkdir(exist_ok=True)

        suffix = Path(ppt_abs_path).suffix.lower()
        if suffix == ".pdf":
            print("[PPTIngestion] 检测到PDF，直接进行页面转图片")
            return PPTIngestionEngine._convert_pdf_to_images(ppt_abs_path, output_dir)
        if suffix not in (".ppt", ".pptx"):
            raise ValueError(f"不支持的文件类型: {suffix} (仅支持 .ppt/.pptx/.pdf)")
        
        # 加载配置
        config = get_config()
        
        # 初始化转换工具
        converter = PPTConverter(
            api_url=config.ppt_converter_api_url,
            api_key=config.ppt_converter_api_key
        )
        
        # 执行转换
        image_paths = converter.convert_ppt_to_images(ppt_abs_path, str(output_dir))
        
        return image_paths

    @staticmethod
    def _convert_pdf_to_images(pdf_path: str, output_dir: Path) -> List[str]:
        """将 PDF 转换为图片序列（跳过 PPT 转换流程）"""
        if convert_from_path is None:
            raise ImportError("未安装 pdf2image，无法处理 PDF 输入")

        image_paths: List[str] = []
        pages = convert_from_path(pdf_path, dpi=200)
        for i, img in enumerate(pages):
            image_filename = f"slide_{i:03d}.png"
            image_path = output_dir / image_filename
            img.save(image_path, "PNG")
            image_paths.append(str(image_path))

        print(f"[PPTIngestion] PDF 转换完成，获得 {len(image_paths)} 张图片")
        return image_paths



class ImageUtils:
    """图片工具函数"""

    @staticmethod
    def encode_image(image_path: str) -> str:
        """将图片编码为base64"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
            
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    @staticmethod
    def crop_image(image_path: str, bbox: List[int], padding_ratio: float = 0.02) -> str:
        """
        根据边界框裁剪图片
        改进：
        1. 增加 padding 防止切掉边缘
        2. 将裁剪图片存放到独立的结构化目录
        """
        try:
            if not bbox or sum(bbox) == 0:
                return image_path

            path_obj = Path(image_path)
            # 定义清晰的输出结构: ../processing_artifacts/crops/<filename>/
            # 假设 image_path 是 .../ppt_slides_temp/slide_001.png
            # 输出将会是 .../processing_artifacts/crops/slide_001/element_x_y.png
            
            cfg = get_config()
            processing_root = str(getattr(cfg, "processing_artifacts_dir", "processing_artifacts") or "processing_artifacts")
            output_dir = path_obj.parent.parent / processing_root / "crops" / path_obj.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with Image.open(image_path) as img:
                w, h = img.size
                
                # Step 2 Locator 返回的是 [ymin, xmin, ymax, xmax] (0-1000 归一化坐标)
                ymin_n, xmin_n, ymax_n, xmax_n = bbox
                
                # --- [改进] 添加 Padding ---
                pad_w = int(w * padding_ratio)
                pad_h = int(h * padding_ratio)
                
                # 转换为像素坐标
                left = int((xmin_n / 1000) * w)
                top = int((ymin_n / 1000) * h)
                right = int((xmax_n / 1000) * w)
                bottom = int((ymax_n / 1000) * h)

                # 应用 Padding (建议在像素层面加，而不是在归一化层面)
                pad_w = int(w * padding_ratio)
                pad_h = int(h * padding_ratio)
                left, top = max(0, left - pad_w), max(0, top - pad_h)
                right, bottom = min(w, right + pad_w), min(h, bottom + pad_h)
                
                # 防止坐标反转或无效
                if right <= left or bottom <= top:
                    print(f"[ImageUtils] ⚠️ 无效的裁剪区域 (after padding): {bbox}, 使用原图")
                    return image_path
                
                crop_img = img.crop((left, top, right, bottom))
                
                # 生成有意义的文件名
                crop_filename = f"crop_{xmin_n}_{ymin_n}_{xmax_n}_{ymax_n}.png"
                target_path = output_dir / crop_filename
                
                crop_img.save(target_path)
                
                return str(target_path)
                
        except Exception as e:
            print(f"[ImageUtils] ⚠️ 裁剪异常: {e}, 使用原始图片")
            return image_path


class Step1_GlobalAnalysisEngine:
    """第一步：全局页面分析引擎"""
    
    def __init__(self, vlm_client):
        self.vlm_client = vlm_client
        from config import get_config
        self.model = get_config().vlm_runtime_model_name
    
    def analyze_page(self, image_path: str, page_index: int) -> GlobalPageAnalysis:
        """分析单个页面"""
        print(f"[Step1] 全局语义分析页面 {page_index}")
        
        try:
            base64_img = ImageUtils.encode_image(image_path)
        except FileNotFoundError:
            print(f"[ERROR] [Step1] 图片丢失: {image_path}, 跳过视觉分析")
            return self._create_empty_analysis(page_index, "Image Missing")
        
        system_prompt = """你是一个专业的 PPT 内容分析师。请对当前页面进行全局语义理解。
        语言要求（必须严格遵守）：
        - 所有输出字段必须使用中文表述，不得夹杂英文句子。
        - 若标题/术语为英文或主要是英文，请保留英文原文，并在其后追加中文翻译（括号内），例如："Overall Goal（总体目标）"。
        核心任务：
        基于页面标题和整体视觉，提出一个**核心科研假设 (Research Hypothesis)**。
        例如：如果标题是“OS Analysis”，假设可能是“旨在证明实验组 OS 显著优于对照组”。
        这也将指导后续的数据提取。

        任务清单：
        首先区分分析的当前页面是标题页、目录页、装饰页还是内容页，然后根据页面内容执行以下操作：
        
        1. **标题页或目录页或纯装饰页处理**：
                - 如果页面为标题(章节)页/封面/目录页或仅包含装饰性元素：
                    - 仍需尽力提取页面上可见的主标题到 section_title（不要留空；若确实没有标题，填 "封面/章节页"）。若标题为英文，请按上述规则追加中文翻译。
                    - 设定 is_pure_text=True，complexity_score=low，elements=[]。
                    - core_summary 用中文简洁说明这是封面/章节页，并包含标题（例如："本页为封面/章节页：2023年业绩概览"），不要输出占位符英文句子。
                    - 若为目录页，不仅需要列出所有章节标题，还需提取加粗或者高亮的章节标题并解释其是后续内容页的标题。
        - 该步执行完后，无需执行后续内容页的分析任务。

        2. **内容页处理**：
        - 1. **章节标题识别**：提取页面的章节标题 (section_title)，确保准确反映页面主题；若标题为英文，请保留原文并追加中文翻译。
        - 2. **内容概览**：用 1-2 句话总结页面核心观点 (Key Takeaway)。
          - **注意**：若页面中单独出现并排的几个小标题，且其中一个小标题被高亮或加粗，请优先考虑该小标题作为改页面的中心标题。
        - 3. **文本提取**：
          - 若页面中存在纯文本块，提取出其完整内容，不要有其他改动。
          - **注意**：页面中存在的表格、数据图、流程图等内部包含的文字及其图例文字不要提取；纯文本块的提取需要考虑块间的关系，避免含义割裂。
        - 4. **结构识别**：
          - 判断页面是否为纯文本 (is_pure_text)。
        - 5. **专业元素识别**：
          请使用以下**医药垂直领域分类**来标记元素类型 (type)，不要使用通用的 image/chart：
            - `km_curve`: Kaplan-Meier 生存曲线 (用于展示 OS, PFS 等)。
            - `forest_plot`: 森林图 (用于展示亚组分析、OR/HR 值)。
            - `molecular_structure`: 化学分子式、药物结构图。
            - `flowchart`: 临床试验设计图、研究流程图、机制通路图。
            - `data_chart`: 常规数据图表 (柱状图、折线图、饼图，用于展示销量、患者比例等)。
            - `table`: 各种数据表格 (基线表、AE表、PK参数表)。
            - `image`: 其他的根据页面信息判断与页面核心内容紧密相关的图片元素。**注意**：页面中包含其他的图片元素，例如logo标志、装饰性图片等，**不需要纳入元素列表,也无需裁剪（重要）**。
            **注意**：
              - 为每个元素生成一个 ID 和描述（例如：“左侧的销售趋势图”），并提取其**图例 (Legend)**。
              - **不需要**提供元素的具体坐标 (BBox)
              - 如果元素很小，模糊不清，大概率为装饰元素，请忽略它，不要强行识别成某个类型。
        - 6. **复杂度评估**：
          - 评估页面布局复杂度 (low/medium/high)。
          - 如果元素非常密集、重叠或非结构化，标记 has_unclear_boundaries=True。
        3. **注意**：
        - 请忽略页面中的背景装饰图、公司 LOGO、页码、纯装饰性的色块。只识别包含实际业务信息、数据或关键结论的元素（如：数据图表、流程图、关键架构图、核心实物图片）。
        4. **输出格式**：
        - 请以严格的 JSON 格式输出: {
            "page_index": int,
            "section_title": "string",
            "is_pure_text": bool,
            "complexity_score": "low" | "medium" | "high",
            "has_unclear_boundaries": bool,
            "core_summary": "string",
            "research_hypothesis": "string",
            "extracted_text": "string",
            "elements": [
                {
                    "element_id": "string",
                    "type": "km_curve" | "forest_plot" | "molecular_structure" | "flowchart" | "data_chart" | "table" | "image",
                    "description": "string",
                    "legend_text": "string (optional)"
                }
            ]
        }"""
                    
        try:
            print(f"[Step1] 页面 {page_index} 准备调用VLM API...")
            print(f"[Step1] 图片大小: {len(base64_img)} bytes")
            
            response = self.vlm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": "请分析这张PPT页面的结构和内容。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            
            # 手动解析 JSON
            if not response or not response.choices or not response.choices[0].message:
                print(f"[Step1] [ERROR] 页面 {page_index} VLM返回格式错误: response={response}")
                return self._create_empty_analysis(page_index, "Invalid response format")
            
            content = response.choices[0].message.content
            if not content:
                print(f"[Step1] [WARNING] 页面 {page_index} VLM返回内容为空")
                return self._create_empty_analysis(page_index, "VLM returned empty content")
            
            print(f"[Step1] 页面 {page_index} 获得VLM响应，长度: {len(content)}")
            
            # 清理可能的 markdown 标记
            content = content.replace("```json", "").replace("```", "")
            data = json.loads(content)
            
            # 转换为 GlobalPageAnalysis 对象
            # 补全 page_index，防止模型漏掉
            data["page_index"] = page_index
            result = GlobalPageAnalysis(**data)

            # 双保险：过滤装饰性元素，避免进入后续裁剪/分析
            try:
                from config import get_config
                if get_config().filter_decorative_elements:
                    result.elements = _filter_decorative_elements(result.elements)
            except Exception:
                # 过滤失败不应影响主流程
                pass

            try:
                print(f"[Step1] [OK] 页面 {page_index} 分析完成: {result.section_title} ({len(result.elements)} 元素)")
            except UnicodeEncodeError:
                # 如果打印标题失败，回退到安全打印，不要中断流程
                safe_title = result.section_title.encode('utf-8', 'ignore').decode('utf-8')
                print(f"[Step1] [OK] 页面 {page_index} 分析完成 (Title Encoding Issue)")
            
            print(f"[Step1] [OK] 页面 {page_index} 分析完成: {result.section_title} ({len(result.elements)} 元素)")
            return result
            
        except json.JSONDecodeError as e:
            print(f"[Step1] [ERROR] 页面 {page_index} JSON解析失败: {e}")
            print(f"[Step1] 原始响应内容前500字: {content[:500] if content else 'None'}")
            return self._create_empty_analysis(page_index, f"JSON Parse Error: {str(e)}")
        except Exception as e:
            print(f"[Step1] [ERROR] 页面 {page_index} 分析失败: {e}")
            import traceback
            traceback.print_exc()
            return self._create_empty_analysis(page_index, f"Analysis Failed: {str(e)}")

    def _create_empty_analysis(self, page_index: int, error_msg: str) -> GlobalPageAnalysis:
        """生成默认的空分析结果"""
        return GlobalPageAnalysis(
            page_index=page_index,
            section_title="Error Page",
            is_pure_text=True, # 失败默认当纯文本处理
            complexity_score="low",
            core_summary=f"Error: {error_msg}",
            extracted_text="",
            elements=[]
        )

# ============================================================================
# 2. 节点函数 (修复：补全缺失的 node_ingestion)
# ============================================================================

def node_step1_global_analysis(state: PPTPageState, vlm_client, debug_logger=None) -> dict:
    """节点：Step 1 全局分析"""
    print(f"\n[Step1] 开始分析页面 {state.page_index}")
    
    page_index = state.page_index
    
    # 记录步骤开始
    if debug_logger:
        debug_logger.log_step_start(
            page_index,
            "Step 1: Global Analysis",
            input_data={
                "page_index": page_index,
                "image_path": state.image_path,
                "previous_context": state.previous_context[:100] if state.previous_context else None
            },
            previous_step="Ingestion"
        )
    
    engine = Step1_GlobalAnalysisEngine(vlm_client)
    analysis = engine.analyze_page(state.image_path, page_index)
    
    # 记录步骤结束
    if debug_logger:
        debug_logger.log_step_end(
            page_index,
            "Step 1: Global Analysis",
            output_data={
                "page_index": analysis.page_index,
                "section_title": analysis.section_title,
                "is_pure_text": analysis.is_pure_text,
                "complexity_score": analysis.complexity_score,
                "elements_count": len(analysis.elements),
                "core_summary": analysis.core_summary[:200],
                "research_hypothesis": analysis.research_hypothesis,
                "has_unclear_boundaries": analysis.has_unclear_boundaries
            },
            status="success"
        )
    
    return {
        "global_analysis": analysis,
        # 初始时，待处理元素即为所有识别出的元素（无坐标）
        "pending_elements": analysis.elements
    }

def node_ingestion(state: PPTWorkflowState) -> dict:
    """节点：PPT读取和页面转换"""
    print("\n" + "="*60)
    print("[节点] 读取PPT文件")
    print("="*60)
    
    # 读取PPT并转换为图片
    image_paths = PPTIngestionEngine.extract_all_slides(state.ppt_path)
    
    total_pages = len(image_paths)
    print(f"[节点] 共读取 {total_pages} 页")
    
    return {
        "page_queue": list(range(total_pages)),
        "total_pages": total_pages,
    }