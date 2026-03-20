# tools/visual_marker.py
import cv2
import os
from pathlib import Path
from typing import List
from state import PageElement

def mark_image_with_bboxes(image_path: str, elements: List[PageElement]) -> str:
    """
    SoM 核心技术：在图片上绘制半透明的 BBox 和 ID 标签。
    """
    if not os.path.exists(image_path):
        print(f"[VisualMarker] 图片不存在: {image_path}")
        return image_path

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        return image_path
        
    h, w = img.shape[:2]
    overlay = img.copy()
    
    # 准备输出目录
    original_path = Path(image_path)
    # output_dir = original_path.parent / "debug_layouts" # 如果不想太深，改浅一点
    output_dir = original_path.parent.parent / "processing_artifacts" / "debug_layouts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    drawn_count = 0
    for i, el in enumerate(elements):
        if not el.bbox or not hasattr(el.bbox, "box_2d"): continue
        
        # 严格按照 [ymin, xmin, ymax, xmax] 解包
        ymin_n, xmin_n, ymax_n, xmax_n = el.bbox.box_2d

        # 核心修复：归一化坐标 -> 像素坐标
        x1 = int(xmin_n * w / 1000)
        y1 = int(ymin_n * h / 1000)
        x2 = int(xmax_n * w / 1000)
        y2 = int(ymax_n * h / 1000)
        
        # 防御：避免反向框
        if x2 <= x1 or y2 <= y1:
            continue
        
        # 强制边界保护
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # 颜色映射 (BGR)
        color_map = {
            "table": (0, 0, 255),      # 红色
            "chart": (0, 165, 255),    # 橙色
            "image": (255, 0, 0),      # 蓝色
            "default": (0, 255, 0)     # 绿色
        }
        color = color_map.get("default")
        for k, v in color_map.items():
            if k in el.type.lower():
                color = v
                break

        # 1. 绘制半透明填充
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 2. 绘制粗边框 (Thickness = 3)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # 3. 绘制文字标签 ID
        label = f"#{i+1} {el.element_id}" # 显示 ID 方便调试
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # 保证文字背景不跑出图片上方
        text_y = max(y1, text_h + 5)
        
        cv2.rectangle(img, (x1, text_y - text_h - 5), (x1 + text_w, text_y + 5), color, -1)
        cv2.putText(img, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        drawn_count += 1

    # 混合原图和半透明层 (加重 alpha 使得颜色更明显)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    filename = original_path.stem + "_layout.png"
    marked_path = output_dir / filename
    
    cv2.imwrite(str(marked_path), img)
    print(f"[VisualMarker] 已绘制 {drawn_count} 个框 -> {marked_path}")
    return str(marked_path)