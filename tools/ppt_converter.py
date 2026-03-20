"""
PPT 文件转换工具 - 支持本地与远程 API 转换
"""
import os
import requests
import zipfile
import io
import time
from pathlib import Path
from typing import List, Optional

class PPTConverter:
    """PPT 转换器工具类"""
    
    def __init__(self, api_url: str = None, api_key: str = None):
        self.api_url = api_url
        self.api_key = api_key

    def convert_ppt_to_images(self, ppt_path: str, output_dir: str) -> List[str]:
        """
        将 PPT 转换为图片序列
        
        Args:
            ppt_path: PPT 文件路径
            output_dir: 输出目录
            
        Returns:
            List[str]: 生成的图片文件路径列表
        """
        # 1. 如果配置了 API URL，优先尝试远程转换
        if self.api_url:
            try:
                print(f"[PPTConverter] 尝试使用远程 API 转换: {self.api_url}")
                return self._convert_via_api(ppt_path, output_dir)
            except Exception as e:
                print(f"[PPTConverter] ⚠️ 远程转换失败: {e}，尝试本地降级方案")
        
        # 2. 降级到本地 Aspose (有水印但可用)
        try:
            print(f"[PPTConverter] 使用本地 Aspose.Slides 转换 (注意：可能包含水印)")
            return self._convert_via_aspose(ppt_path, output_dir)
        except ImportError:
             print("[PPTConverter] ⚠️ 未安装 aspose.slides")
        except Exception as e:
             print(f"[PPTConverter] ⚠️ Aspose 转换失败: {e}")

        # 3. 最后的兜底：生成占位图
        print(f"[PPTConverter] ⚠️ 所有转换方法均失败，生成占位图")
        return self._generate_placeholders(output_dir, ppt_path)

    def _convert_via_api(self, ppt_path: str, output_dir: str) -> List[str]:
        """
        通过通用文件转换 API 转换
        
        协议假设：
        - POST multiform-data: file=@test.pptx
        - Header: X-API-Key: <key>
        - Response: application/zip (包含 slide_000.png, slide_001.png...)
        """
        if not os.path.exists(ppt_path):
            raise FileNotFoundError(f"File not found: {ppt_path}")

        ppt_path_obj = Path(ppt_path)
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-API-Key"] = self.api_key

        print(f"  - 上传文件: {ppt_path} ({os.path.getsize(ppt_path)/1024/1024:.2f} MB)")
        
        with open(ppt_path, 'rb') as f:
            files = {'file': (ppt_path_obj.name, f, 'application/vnd.openxmlformats-officedocument.presentationml.presentation')}
            
            # 设置较长的超时时间 (PPT转换可能耗时)
            response = requests.post(self.api_url, files=files, headers=headers, timeout=120)
            
        if response.status_code != 200:
            raise ValueError(f"API Error {response.status_code}: {response.text[:200]}")
            
        print(f"  - 接收响应: {len(response.content)} bytes, 解压中...")
        
        # 解压 ZIP
        image_paths = []
        try:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # 过滤出图片文件
                image_files = [n for n in z.namelist() if n.lower().endswith(('.png', '.jpg', '.jpeg'))]
                # 排序 (slide_0.png, slide_1.png...)
                # 尝试智能排序
                try:
                    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
                except:
                    image_files.sort()
                
                for i, filename in enumerate(image_files):
                    # 重命名标准化: slide_000.png
                    std_filename = f"slide_{i:03d}{Path(filename).suffix}"
                    target_path = Path(output_dir) / std_filename
                    
                    with z.open(filename) as source, open(target_path, 'wb') as target:
                        target.write(source.read())
                    
                    image_paths.append(str(target_path))
        except zipfile.BadZipFile:
            raise ValueError("API 返回的不是有效的 ZIP 文件")

        print(f"  - [OK] API 转换成功，获得 {len(image_paths)} 张图片")
        return image_paths

    def _convert_via_aspose(self, ppt_path: str, output_dir: str) -> List[str]:
        """本地 Aspose.Slides 转换"""
        import aspose.slides as slides
        import aspose.pydrawing as drawing
        
        image_paths = []
        
        # 简单获取页面数量预估 (可选)
        
        with slides.Presentation(ppt_path) as presentation:
            total = len(presentation.slides)
            print(f"  - PPT共 {total} 页")
            
            for i, slide in enumerate(presentation.slides):
                image_filename = f"slide_{i:03d}.png"
                image_path = Path(output_dir) / image_filename
                
                # 导出图片 (2.0x 缩放确保清晰度)
                bmp = slide.get_thumbnail(2.0, 2.0)
                bmp.save(str(image_path), drawing.imaging.ImageFormat.png)
                
                image_paths.append(str(image_path))
                if i % 5 == 0:
                    print(f"    - 处理进度: {i+1}/{total}")
                    
        return image_paths

    def _generate_placeholders(self, output_dir: str, ppt_path: str) -> List[str]:
        """生成占位图 (在所有手段失败时)"""
        import aspose.slides as slides
        try:
            # 尝试至少获取页数
             with slides.Presentation(ppt_path) as prs:
                 count = len(prs.slides)
        except:
            count = 1 # 无法读取则默认1页
            
        from PIL import Image, ImageDraw
        paths = []
        for i in range(count):
            p = Path(output_dir) / f"slide_{i:03d}.png"
            img = Image.new('RGB', (1024, 768), color='white')
            d = ImageDraw.Draw(img)
            d.text((50, 50), f"Slide {i}\nConversion Failed", fill=(0,0,0))
            img.save(p)
            paths.append(str(p))
            
        return paths
