"""
Phase 2-1: 多模态语义分组 (Multimodal Semantic Grouping)

核心目的：解决"零散元素"到"逻辑组合"的问题

包含组件：
1. CoordinateNormalizer: 坐标归一化工具
2. SoMGenerator: Set-of-Mark 图片生成器  
3. SemanticGroupingEngine: VLM 语义聚类引擎
4. Phase2_SemanticGrouper: Phase 2-1 总控制器
"""

import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .pipeline_state import (
    CleanedLayoutJSON, DetectedElement, ElementBBox,
    SemanticGroup, DraftSemanticJSON
)


# ============================================================================
# 坐标归一化工具
# ============================================================================

class CoordinateNormalizer:
    """
    坐标归一化工具
    
    将所有 BBox 统一转为 [0, 1000] 相对坐标，方便后续绘图和 LLM 理解
    """
    
    SCALE = 1000  # 归一化范围
    
    @staticmethod
    def normalize_bbox(bbox: List[int], 
                      img_width: int, 
                      img_height: int) -> List[int]:
        """
        将像素坐标归一化到 0-1000
        
        Args:
            bbox: [x1, y1, x2, y2] 像素坐标
            img_width: 图片宽度
            img_height: 图片高度
            
        Returns:
            [ymin, xmin, ymax, xmax] 归一化坐标
        """
        x1, y1, x2, y2 = bbox
        
        xmin = int(x1 * CoordinateNormalizer.SCALE / img_width)
        ymin = int(y1 * CoordinateNormalizer.SCALE / img_height)
        xmax = int(x2 * CoordinateNormalizer.SCALE / img_width)
        ymax = int(y2 * CoordinateNormalizer.SCALE / img_height)
        
        # 边界保护
        xmin = max(0, min(CoordinateNormalizer.SCALE, xmin))
        ymin = max(0, min(CoordinateNormalizer.SCALE, ymin))
        xmax = max(0, min(CoordinateNormalizer.SCALE, xmax))
        ymax = max(0, min(CoordinateNormalizer.SCALE, ymax))
        
        return [ymin, xmin, ymax, xmax]
    
    @staticmethod
    def denormalize_bbox(bbox: List[int],
                        img_width: int,
                        img_height: int) -> List[int]:
        """
        将归一化坐标转换回像素坐标
        
        Args:
            bbox: [ymin, xmin, ymax, xmax] 归一化坐标
            img_width: 图片宽度
            img_height: 图片高度
            
        Returns:
            [x1, y1, x2, y2] 像素坐标
        """
        ymin, xmin, ymax, xmax = bbox
        
        x1 = int(xmin * img_width / CoordinateNormalizer.SCALE)
        y1 = int(ymin * img_height / CoordinateNormalizer.SCALE)
        x2 = int(xmax * img_width / CoordinateNormalizer.SCALE)
        y2 = int(ymax * img_height / CoordinateNormalizer.SCALE)
        
        return [x1, y1, x2, y2]
    
    @staticmethod
    def get_bbox_center(bbox: ElementBBox) -> Tuple[int, int]:
        """获取bbox中心点坐标（归一化）"""
        cx = (bbox.xmin + bbox.xmax) // 2
        cy = (bbox.ymin + bbox.ymax) // 2
        return cx, cy
    
    @staticmethod
    def compute_union_bbox(bboxes: List[ElementBBox]) -> List[int]:
        """计算多个bbox的联合边界框"""
        if not bboxes:
            return [0, 0, 1000, 1000]
        
        ymin = min(b.ymin for b in bboxes)
        xmin = min(b.xmin for b in bboxes)
        ymax = max(b.ymax for b in bboxes)
        xmax = max(b.xmax for b in bboxes)
        
        return [ymin, xmin, ymax, xmax]


# ============================================================================
# Set-of-Mark 图片生成器
# ============================================================================

class SoMGenerator:
    """
    Set-of-Mark (SoM) 增强图片生成器
    
    在原图上绘制：
    1. 半透明红框标记每个元素
    2. 在框的左上角标注鲜艳的数字 ID
    """
    
    # 颜色配置 (BGR格式 for cv2, RGB for PIL)
    COLORS = {
        "text": (0, 255, 0),      # 绿色
        "chart": (0, 165, 255),   # 橙色
        "table": (0, 0, 255),     # 红色
        "image": (255, 0, 0),     # 蓝色
        "formula": (255, 0, 255), # 紫色
        "flowchart": (255, 255, 0),# 青色
        "default": (128, 128, 128) # 灰色
    }
    
    def __init__(self, output_dir: str = "processing_artifacts/som_images"):
        """
        初始化SoM生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate(self, 
                image_path: str,
                elements: List[DetectedElement],
                page_id: int) -> str:
        """
        生成带SoM标记的图片
        
        Args:
            image_path: 原图路径
            elements: 元素列表
            page_id: 页面ID
            
        Returns:
            生成的SoM图片路径
        """
        if CV2_AVAILABLE:
            return self._generate_cv2(image_path, elements, page_id)
        elif PIL_AVAILABLE:
            return self._generate_pil(image_path, elements, page_id)
        else:
            print("[SoMGenerator] 缺少图像处理库")
            return image_path
    
    def _generate_cv2(self, 
                     image_path: str,
                     elements: List[DetectedElement],
                     page_id: int) -> str:
        """使用OpenCV生成SoM图片"""
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"[SoMGenerator] 无法读取图片: {image_path}")
            return image_path
        
        img_height, img_width = img.shape[:2]
        overlay = img.copy()
        
        for i, elem in enumerate(elements):
            # 获取像素坐标
            bbox = elem.bbox
            x1 = int(bbox.xmin * img_width / 1000)
            y1 = int(bbox.ymin * img_height / 1000)
            x2 = int(bbox.xmax * img_width / 1000)
            y2 = int(bbox.ymax * img_height / 1000)
            
            # 边界保护
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width - 1, x2), min(img_height - 1, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # 获取颜色
            elem_type = (elem.refined_type or elem.original_type).lower()
            color = self.COLORS.get(elem_type, self.COLORS["default"])
            
            # 1. 绘制半透明填充
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            
            # 2. 绘制粗边框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            # 3. 绘制ID标签
            label = f"#{i + 1}"
            font_scale = 1.2
            thickness = 3
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # 标签背景
            text_y = max(y1, text_h + 10)
            cv2.rectangle(
                img, 
                (x1, text_y - text_h - 10), 
                (x1 + text_w + 10, text_y + 5),
                color, -1
            )
            
            # 标签文字（白色）
            cv2.putText(
                img, label, (x1 + 5, text_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
            )
        
        # 混合原图和半透明层
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # 保存
        output_path = os.path.join(self.output_dir, f"page_{page_id:03d}_som.png")
        cv2.imwrite(output_path, img)
        
        print(f"[SoMGenerator] 已生成SoM图片: {output_path}")
        return output_path
    
    def _generate_pil(self,
                     image_path: str,
                     elements: List[DetectedElement],
                     page_id: int) -> str:
        """使用PIL生成SoM图片"""
        # 读取图片
        img = Image.open(image_path).convert("RGBA")
        img_width, img_height = img.size
        
        # 创建绘图层
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except:
            font = ImageFont.load_default()
        
        for i, elem in enumerate(elements):
            # 获取像素坐标
            bbox = elem.bbox
            x1 = int(bbox.xmin * img_width / 1000)
            y1 = int(bbox.ymin * img_height / 1000)
            x2 = int(bbox.xmax * img_width / 1000)
            y2 = int(bbox.ymax * img_height / 1000)
            
            # 边界保护
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width - 1, x2), min(img_height - 1, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # 获取颜色 (RGB for PIL)
            elem_type = (elem.refined_type or elem.original_type).lower()
            bgr_color = self.COLORS.get(elem_type, self.COLORS["default"])
            rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
            
            # 1. 绘制半透明填充
            draw.rectangle([x1, y1, x2, y2], fill=(*rgb_color, 80))
            
            # 2. 绘制边框
            draw.rectangle([x1, y1, x2, y2], outline=rgb_color, width=3)
            
            # 3. 绘制ID标签
            label = f"#{i + 1}"
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            
            # 标签背景和文字
            text_y = max(y1, text_h + 10)
            draw.rectangle(
                [x1, text_y - text_h - 10, x1 + text_w + 10, text_y + 5],
                fill=(*rgb_color, 255)
            )
            draw.text((x1 + 5, text_y - text_h - 5), label, fill=(255, 255, 255), font=font)
        
        # 合并图层
        img = Image.alpha_composite(img, overlay)
        img = img.convert("RGB")
        
        # 保存
        output_path = os.path.join(self.output_dir, f"page_{page_id:03d}_som.png")
        img.save(output_path)
        
        print(f"[SoMGenerator] 已生成SoM图片: {output_path}")
        return output_path


# ============================================================================
# VLM 语义聚类引擎
# ============================================================================

class SemanticGroupingEngine:
    """
    VLM 语义聚类推理引擎
    
    使用 GPT-4o 分析 SoM 图片和文本内容，完成：
    1. 分组 (Grouping)：将语义强相关的元素分为一组
    2. 层级定义 (Hierarchy)：在组内区分主元素和从元素
    3. 阅读顺序 (Reading Order)：根据PPT布局给出阅读顺序
    """
    
    def __init__(self, vlm_client):
        """
        初始化语义聚类引擎
        
        Args:
            vlm_client: VLM客户端（GPT-4o）
        """
        self.vlm_client = vlm_client
        
        # 获取模型名称
        try:
            from config import get_config
            self.model = get_config().vlm_runtime_model_name
        except:
            self.model = "gpt-4o"
    
    def group_elements(self,
                      som_image_path: str,
                      elements: List[DetectedElement],
                      id_to_text_map: Dict[str, str],
                      global_summary: str = "") -> List[SemanticGroup]:
        """
        执行语义分组
        
        Args:
            som_image_path: SoM标记图片路径
            elements: 元素列表
            id_to_text_map: ID到OCR文本的映射
            global_summary: 页面全局摘要（来自Step1）
            
        Returns:
            语义分组列表
        """
        print("[SemanticGrouping] 开始VLM语义聚类推理...")
        
        # 读取SoM图片
        try:
            with open(som_image_path, "rb") as f:
                base64_img = base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"[SemanticGrouping] 读取SoM图片失败: {e}")
            return self._fallback_grouping(elements)
        
        # 构建元素列表描述
        element_list = []
        for i, elem in enumerate(elements):
            elem_type = elem.refined_type or elem.original_type
            ocr_text = id_to_text_map.get(elem.element_id, elem.ocr_text or "")
            text_preview = ocr_text[:50] + "..." if len(ocr_text) > 50 else ocr_text
            element_list.append(f"#{i+1} ({elem.element_id}): 类型={elem_type}, 文本=\"{text_preview}\"")
        
        elements_desc = "\n".join(element_list)
        
        # 构造Prompt
        prompt = f"""你是一个专业的PPT版面语义分析师。

我提供了一张带有数字ID标记的PPT页面截图（#1, #2, ...）以及每个元素的信息。

【页面全局主题】
{global_summary if global_summary else "未提供"}

【检测到的元素】
{elements_desc}

【你的任务】
1. **分组 (Grouping)**：将语义强相关的元素分为一组。例如：
   - 图表 + 图表的标题/图例 = 一组
   - 表格 + 表格的说明文字 = 一组
   - 左右并列对比的两个图 = 一组
    - 如果一个ROI本身同时包含多个元素（如"两张图+一张表"），请标记为 mixed_group

2. **层级定义 (Hierarchy)**：在组内区分：
   - primary_element: 主元素（如图表、表格本身）
   - secondary_elements: 从元素（如标题、图例、说明）

3. **阅读顺序 (Reading Order)**：根据PPT布局（通常是Z字形或分栏），给出各组的阅读顺序索引（从0开始）。

【输出格式】
请返回严格的JSON，不要包含markdown标记：
{{
    "groups": [
        {{
            "group_id": "group_1",
            "member_ids": ["#1", "#2"],
            "group_type": "chart_and_title",
            "semantic_desc": "左上角的销售趋势图及其标题",
            "reading_order": 0,
            "primary_element_id": "#1",
            "secondary_element_ids": ["#2"]
        }},
        {{
            "group_id": "group_2",
            "member_ids": ["#3"],
            "group_type": "standalone_text",
            "semantic_desc": "右侧的结论文字",
            "reading_order": 1,
            "primary_element_id": "#3",
            "secondary_element_ids": []
        }}
    ],
    "ungrouped_element_ids": ["#5"]
}}

注意：
- member_ids 使用 "#N" 格式
- 如果元素是独立的（无关联），可以单独成组
- group_type 建议使用：chart_group / table_group / text_group / mixed_group / standalone_xxx
- 如果实在无法分组的元素，放入 ungrouped_element_ids"""
        
        try:
            # 调用VLM
            response = self.vlm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是专业的PPT版面分析师，擅长理解元素间的语义关系。"},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                    ]}
                ],
                max_tokens=2000,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # 解析分组结果
            groups = []
            for g in result.get("groups", []):
                # 转换 #N 格式为 element_id
                member_ids = self._convert_ids(g.get("member_ids", []), elements)
                primary_id = self._convert_single_id(g.get("primary_element_id"), elements)
                secondary_ids = self._convert_ids(g.get("secondary_element_ids", []), elements)
                
                group = SemanticGroup(
                    group_id=g.get("group_id", f"group_{len(groups)+1}"),
                    member_ids=member_ids,
                    group_type=g.get("group_type", "unknown"),
                    semantic_desc=g.get("semantic_desc", ""),
                    reading_order=g.get("reading_order", len(groups)),
                    primary_element_id=primary_id,
                    secondary_element_ids=secondary_ids
                )
                groups.append(group)
            
            print(f"[SemanticGrouping] VLM返回 {len(groups)} 个分组")
            return groups
            
        except Exception as e:
            print(f"[SemanticGrouping] VLM调用失败: {e}")
            return self._fallback_grouping(elements)
    
    def _convert_ids(self, hash_ids: List[str], elements: List[DetectedElement]) -> List[str]:
        """将 #N 格式转换为 element_id"""
        result = []
        for hid in hash_ids:
            elem_id = self._convert_single_id(hid, elements)
            if elem_id:
                result.append(elem_id)
        return result
    
    def _convert_single_id(self, hash_id: Optional[str], elements: List[DetectedElement]) -> Optional[str]:
        """将单个 #N 转换为 element_id"""
        if not hash_id:
            return None
        
        try:
            # 提取数字
            num_str = hash_id.replace("#", "").strip()
            idx = int(num_str) - 1  # #1 -> index 0
            
            if 0 <= idx < len(elements):
                return elements[idx].element_id
        except:
            pass
        
        return None
    
    def _fallback_grouping(self, elements: List[DetectedElement]) -> List[SemanticGroup]:
        """
        降级分组策略：按几何坐标从上到下、从左到右生成线性分组
        每个元素单独成组
        """
        print("[SemanticGrouping] 使用降级分组策略（几何排序）")
        
        # 按坐标排序（先上后下，先左后右）
        sorted_elements = sorted(
            elements,
            key=lambda e: (e.bbox.ymin, e.bbox.xmin)
        )
        
        groups = []
        for i, elem in enumerate(sorted_elements):
            group = SemanticGroup(
                group_id=f"group_{i+1}",
                member_ids=[elem.element_id],
                group_type=f"standalone_{elem.refined_type or elem.original_type}".lower(),
                semantic_desc=f"独立元素（降级分组）",
                reading_order=i,
                primary_element_id=elem.element_id,
                secondary_element_ids=[]
            )
            groups.append(group)
        
        return groups


# ============================================================================
# Phase 2-1 总控制器
# ============================================================================

class Phase2_SemanticGrouper:
    """
    Phase 2-1 多模态语义分组 - 总控制器
    
    整合坐标归一化、SoM生成、VLM语义聚类
    """
    
    def __init__(self, vlm_client, output_dir: str = "processing_artifacts"):
        """
        初始化 Phase 2-1 控制器
        
        Args:
            vlm_client: VLM客户端
            output_dir: 输出目录
        """
        self.vlm_client = vlm_client
        self.output_dir = output_dir
        
        # 初始化组件
        som_output_dir = os.path.join(output_dir, "som_images")
        self.som_generator = SoMGenerator(output_dir=som_output_dir)
        self.grouping_engine = SemanticGroupingEngine(vlm_client)
    
    def run(self,
           cleaned_layout: CleanedLayoutJSON,
           global_summary: str = "") -> DraftSemanticJSON:
        """
        执行 Phase 2-1 完整流程
        
        Args:
            cleaned_layout: Phase 1 的输出
            global_summary: 页面全局摘要
            
        Returns:
            DraftSemanticJSON: 初步语义分组结果
        """
        print(f"\n{'='*60}")
        print(f"[Phase 2-1] 开始多模态语义分组 (Page {cleaned_layout.page_id})")
        print(f"{'='*60}")
        
        elements = cleaned_layout.elements
        image_path = cleaned_layout.image_path
        page_id = cleaned_layout.page_id
        
        # Step 1: 构建 ID -> OCR Text 映射表
        print("\n[Phase 2-1.1] 构建上下文映射...")
        id_to_text_map = {}
        for elem in elements:
            id_to_text_map[elem.element_id] = elem.ocr_text or ""
        print(f"  映射了 {len(id_to_text_map)} 个元素的文本")
        
        # Step 2: 生成 SoM 增强图片
        print("\n[Phase 2-1.2] 生成 Set-of-Mark 图片...")
        som_image_path = self.som_generator.generate(
            image_path, elements, page_id
        )
        
        # Step 3: VLM 聚类推理
        print("\n[Phase 2-1.3] VLM 语义聚类推理...")
        groups = self.grouping_engine.group_elements(
            som_image_path,
            elements,
            id_to_text_map,
            global_summary
        )
        
        # 识别未分组的元素
        grouped_ids = set()
        for g in groups:
            grouped_ids.update(g.member_ids)
        
        ungrouped_ids = [
            e.element_id for e in elements 
            if e.element_id not in grouped_ids
        ]
        
        # 构建输出
        result = DraftSemanticJSON(
            page_id=page_id,
            groups=groups,
            ungrouped_element_ids=ungrouped_ids,
            id_to_text_map=id_to_text_map,
            som_image_path=som_image_path
        )
        
        print(f"\n[Phase 2-1] 完成！")
        print(f"  - 分组数: {len(groups)}")
        print(f"  - 未分组元素: {len(ungrouped_ids)}")
        
        # 输出分组摘要
        for g in groups:
            print(f"    [{g.group_id}] {g.group_type}: {g.member_ids} (order={g.reading_order})")
        
        return result
