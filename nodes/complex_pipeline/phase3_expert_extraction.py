"""
Phase 3: 并行特征提取 (Parallel Expert Extraction)

核心目的：注入 Phase 2 识别出的上下文（如标题），让专家模型"带着答案找线索"

包含组件：
1. TextInAPIClient: 合合TextIn API客户端（表格/公式识别）
2. TableExpertAgent: 表格专家
3. ChartExpertAgent: 图表专家
4. FormulaExpertAgent: 公式专家
5. TextProcessorAgent: 文本处理器
6. Phase3_ParallelExtractor: Phase 3 总控制器
"""

import os
import json
import base64
import time
import requests
import re
import html
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tempfile import NamedTemporaryFile
from PIL import Image
from nodes.type_normalization import normalize_element_type, is_text_type, TABLE, CHART, FORMULA, IMAGE, MIXED

from .pipeline_state import (
    ValidatedSemanticJSON, SemanticGroup, DetectedElement, ROIImage,
    ExtractedDataMap, ExtractedContent,
    TableExtraction, ChartExtraction, FormulaExtraction, TextExtraction, MixedExtraction
)


# ============================================================================
# 合合 TextIn API 客户端
# ============================================================================

class TextInAPIClient:
    """
    合合 TextIn API 客户端
    
    支持功能：
    - 表格识别（结构化JSON/Markdown）
    - 公式识别（LaTeX）
    - 通用OCR
    
    API文档参考：https://www.textin.com/document/table_recognize
    """
    
    # 合合信息 TextIn API 配置
    # 不要在代码里硬编码真实凭证：请通过环境变量提供（见 config.py 里 .env 加载）。
    # 支持：TEXTIN_APP_ID / TEXTIN_SECRET_CODE 或 TEXTIN_API_KEY="app_id:secret_code"
    DEFAULT_APP_ID = ""
    DEFAULT_SECRET_CODE = ""
    BASE_URL = "https://api.textin.com"
    
    def __init__(self, api_key: Optional[str] = None, app_id: Optional[str] = None, secret_code: Optional[str] = None):
        """
        初始化TextIn客户端
        
        Args:
            api_key: API密钥（格式: app_id:secret_code）
            app_id: 应用ID（优先使用）
            secret_code: 密钥（优先使用）
        """
        # 优先级：显式传参 > api_key(app_id:secret) > 环境变量 > 默认值(空)
        if app_id and secret_code:
            self.app_id = app_id
            self.secret_code = secret_code
        elif api_key and ":" in api_key:
            self.app_id = api_key.split(":", 1)[0]
            self.secret_code = api_key.split(":", 1)[1]
        else:
            env_app_id = os.getenv("TEXTIN_APP_ID", "").strip()
            env_secret_code = os.getenv("TEXTIN_SECRET_CODE", "").strip()
            env_api_key = os.getenv("TEXTIN_API_KEY", "").strip()

            if env_app_id and env_secret_code:
                self.app_id = env_app_id
                self.secret_code = env_secret_code
            elif env_api_key and ":" in env_api_key:
                self.app_id = env_api_key.split(":", 1)[0]
                self.secret_code = env_api_key.split(":", 1)[1]
            else:
                self.app_id = self.DEFAULT_APP_ID
                self.secret_code = self.DEFAULT_SECRET_CODE
        
        self.headers = {
            "x-ti-app-id": self.app_id,
            "x-ti-secret-code": self.secret_code,
            "Content-Type": "application/json",
        }

        if self.app_id and self.secret_code:
            masked = (self.app_id[:4] + "..." + self.app_id[-2:]) if len(self.app_id) >= 6 else "***"
            print(f"[TextIn] 客户端初始化 (已配置), app_id: {masked}")
        else:
            print("[TextIn] 客户端初始化 (未配置，将使用 mock/降级)")
    
    def recognize_table(self, image_path: str) -> Dict[str, Any]:
        """
        识别表格
        
        Args:
            image_path: 图片路径
            
        Returns:
            包含 markdown_table 和 json_structure 的字典
        """
        # 如果未配置凭证，直接 mock（避免对外请求失败浪费时间）
        if not (self.app_id and self.secret_code):
            mock = self._mock_table_response()
            mock["textin_error"] = {"reason": "missing_credentials"}
            return mock

        last_error: Optional[Dict[str, Any]] = None

        # 优先使用新版 v2 二进制接口（文档：/ai/service/v2/recognize/table/multipage），
        # 该接口返回结构化单元格信息，我们在本地重建 Markdown 表格。
        try:
            v2_url = f"{self.BASE_URL}/ai/service/v2/recognize/table/multipage?output_order=table_only"
            with open(image_path, "rb") as f:
                data = f.read()
            v2_headers = dict(self.headers)
            v2_headers["Content-Type"] = "application/octet-stream"

            response = requests.post(v2_url, headers=v2_headers, data=data, timeout=60)
            if response.status_code == 200:
                result = response.json()

                # TextIn 有时会在 HTTP 200 下返回业务错误码
                if isinstance(result, dict) and "code" in result and str(result.get("code")) not in {"0", "200"}:
                    last_error = {
                        "endpoint": "v2",
                        "code": result.get("code"),
                        "message": result.get("msg") or result.get("message"),
                        "x_request_id": result.get("x_request_id") or result.get("x_request_id") or result.get("x_request_id"),
                    }
                    print(f"[TextIn] v2 业务错误: {last_error.get('code')} {last_error.get('message')}")
                else:
                    last_error = None

                md, js = self._parse_v2_table_result(result)
                # v2 可能返回结构化信息但字段形态变化导致 md 为空；此时继续尝试 v1 拿 markdown。
                if md and md.strip():
                    return {
                        "markdown_table": md,
                        "json_structure": js,
                        "raw_response": result,
                    }
                extracted_md = self._extract_markdown_text(result)
                if extracted_md:
                    return {
                        "markdown_table": extracted_md,
                        "json_structure": js,
                        "raw_response": result,
                    }
                # 如果 v2 明确返回“非表格区域”(plain/无 cells)，直接返回空结果，避免误走 v1
                if self._looks_like_non_table_v2(result):
                    print("[TextIn] v2 返回非表格区域，跳过 v1 兜底")
                    return {
                        "markdown_table": "",
                        "json_structure": {"raw": result, "non_table": True},
                        "raw_response": result,
                        "textin_error": {"reason": "non_table"},
                    }

                print("[TextIn] v2 表格识别未生成可用Markdown，尝试旧版接口兜底")
            else:
                print(f"[TextIn] v2 表格识别失败: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[TextIn] v2 API请求失败: {e}")
        except Exception as e:
            print(f"[TextIn] v2 表格识别异常: {e}")

        # 旧版接口兜底（部分环境仍可用）。如果解析结果为空，则视为失败并走 mock。
        try:
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")

            url = f"{self.BASE_URL}/ai/service/v1/table/recognize"
            payload = {
                "image": image_base64,
                "output_format": "json_and_markdown",
            }

            response = requests.post(url, headers=self.headers, json=payload, timeout=60)

            if response.status_code != 200:
                print(f"[TextIn] 表格识别失败: {response.status_code}")
                return self._mock_table_response()

            result = response.json()

            if isinstance(result, dict) and "code" in result and str(result.get("code")) not in {"0", "200"}:
                last_error = {
                    "endpoint": "v1",
                    "code": result.get("code"),
                    "message": result.get("msg") or result.get("message"),
                }
                print(f"[TextIn] v1 业务错误: {last_error.get('code')} {last_error.get('message')}")

                # v1 返回“文件类型不支持”等业务错误时，不再降级 mock，避免假表格污染结果
                if str(result.get("code")) in {"40301", "40302", "40303", "40304", "40305"}:
                    return {
                        "markdown_table": "",
                        "json_structure": {},
                        "raw_response": result,
                        "textin_error": last_error,
                    }

            data_obj = result.get("data") or result.get("result") or {}
            markdown = (
                data_obj.get("markdown")
                or data_obj.get("md")
                or data_obj.get("table_markdown")
                or data_obj.get("tableMarkdown")
                or ""
            )
            json_struct = (
                data_obj.get("json")
                or data_obj.get("table_json")
                or data_obj.get("tableJson")
                or data_obj.get("tables")
                or {}
            )

            if (not markdown.strip()) and json_struct:
                rebuilt = self._rebuild_markdown_from_json(json_struct)
                if rebuilt:
                    markdown = rebuilt

            if not (markdown.strip() or json_struct):
                # 200 但解析为空：不再降级 mock，直接返回空结果给上层兜底
                print("[TextIn] v1 表格识别响应解析为空，返回空结果")
                return {
                    "markdown_table": "",
                    "json_structure": {},
                    "raw_response": result,
                    "textin_error": last_error or {"reason": "empty_parse"},
                }

            return {
                "markdown_table": markdown,
                "json_structure": json_struct,
                "raw_response": result,
            }

        except requests.exceptions.RequestException as e:
            print(f"[TextIn] API请求失败: {e}")
            mock = self._mock_table_response()
            mock["textin_error"] = {"reason": "request_exception", "message": str(e)}
            return mock
        except Exception as e:
            print(f"[TextIn] 表格识别异常: {e}")
            mock = self._mock_table_response()
            mock["textin_error"] = {"reason": "exception", "message": str(e)}
            return mock

    def _parse_v2_table_result(self, payload: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """解析 v2 表格识别结果，并重建 Markdown。

        v2 接口以 pages/tables/table_cells 形式返回结构化信息。
        """
        # v2 返回结构在不同版本/账号下可能略有差异：
        # - {result: {pages: [...]}}
        # - {data: {result: {pages: [...]}}}
        # - {data: {pages: [...]}}
        # - {pages: [...]}
        root: Any = payload
        if isinstance(payload.get("data"), dict):
            root = payload.get("data")

        result: Any = root.get("result") if isinstance(root, dict) and isinstance(root.get("result"), dict) else root

        def _find_first_list(obj: Any, key: str) -> Optional[list]:
            if isinstance(obj, dict):
                v = obj.get(key)
                if isinstance(v, list):
                    return v
                for vv in obj.values():
                    found = _find_first_list(vv, key)
                    if found is not None:
                        return found
            elif isinstance(obj, list):
                for vv in obj:
                    found = _find_first_list(vv, key)
                    if found is not None:
                        return found
            return None

        def _find_first_dict(obj: Any, key: str) -> Optional[dict]:
            if isinstance(obj, dict):
                v = obj.get(key)
                if isinstance(v, dict):
                    return v
                for vv in obj.values():
                    found = _find_first_dict(vv, key)
                    if found is not None:
                        return found
            elif isinstance(obj, list):
                for vv in obj:
                    found = _find_first_dict(vv, key)
                    if found is not None:
                        return found
            return None

        pages = _find_first_list(result, "pages") or _find_first_list(result, "page_list") or []

        # 有些账号/版本可能不会返回 pages，而是直接返回 tables 列表
        if pages and isinstance(pages, list):
            page0 = pages[0] if pages else {}
            tables = page0.get("tables") or page0.get("table_list") or []
        else:
            tables = _find_first_list(result, "tables") or _find_first_list(result, "table_list") or []

        if not isinstance(tables, list) or not tables:
            return "", {"raw": payload}

        # 选择第一个包含 cells 的区域（跳过非表格区域）
        def _has_cells(t: dict) -> bool:
            return bool(
                t.get("table_cells")
                or t.get("cells")
                or t.get("tableCells")
                or t.get("table_cells_list")
                or t.get("cell_list")
            )

        table_regions = [t for t in tables if isinstance(t, dict) and _has_cells(t)]
        if not table_regions:
            return "", {"raw": payload, "tables": tables}

        table0 = table_regions[0]
        cells = (
            table0.get("table_cells")
            or table0.get("cells")
            or table0.get("tableCells")
            or table0.get("table_cells_list")
            or table0.get("cell_list")
            or []
        )
        if not isinstance(cells, list) or not cells:
            return "", {"raw": payload, "table": table0}

        def _cell_int(c: dict, keys: list[str], default: int = 0) -> int:
            try:
                for k in keys:
                    if k in c and c.get(k) is not None:
                        return int(c.get(k))
                return int(default)
            except Exception:
                return default

        max_row = 0
        max_col = 0
        for c in cells:
            if not isinstance(c, dict):
                continue
            start_row = _cell_int(c, ["start_row", "startRow", "row", "row_index"], 0)
            start_col = _cell_int(c, ["start_col", "startCol", "col", "col_index"], 0)
            end_row = _cell_int(c, ["end_row", "endRow"], start_row)
            end_col = _cell_int(c, ["end_col", "endCol"], start_col)
            max_row = max(max_row, end_row)
            max_col = max(max_col, end_col)

        rows = max_row + 1
        cols = max_col + 1
        if rows <= 0 or cols <= 0:
            return "", {"raw": payload, "table": table0}

        grid = [["" for _ in range(cols)] for _ in range(rows)]
        header_span_end = 0
        for c in cells:
            if not isinstance(c, dict):
                continue
            r0 = _cell_int(c, ["start_row", "startRow", "row", "row_index"], 0)
            c0 = _cell_int(c, ["start_col", "startCol", "col", "col_index"], 0)
            r1 = _cell_int(c, ["end_row", "endRow"], r0)
            c1 = _cell_int(c, ["end_col", "endCol"], c0)
            text = (c.get("text") or c.get("cell_text") or c.get("cellText") or "").strip()
            if r0 == 0 and r1 > header_span_end:
                header_span_end = r1

            if not text:
                continue

            for rr in range(r0, r1 + 1):
                if rr < 0 or rr >= rows:
                    continue
                for cc in range(c0, c1 + 1):
                    if cc < 0 or cc >= cols:
                        continue
                    if not grid[rr][cc]:
                        grid[rr][cc] = text

        def esc(s: str) -> str:
            return (s or "").replace("\n", " ").replace("|", "\\|").strip()

        header_rows = max(0, header_span_end) + 1
        header_rows = min(header_rows, rows)

        header = []
        for col in range(cols):
            parts = []
            seen = set()
            for r in range(header_rows):
                val = esc(grid[r][col])
                if not val or val in seen:
                    continue
                seen.add(val)
                parts.append(val)
            header.append(" / ".join(parts).strip())

        if all(not h for h in header):
            header = [f"Col{i+1}" for i in range(cols)]

        md_lines = []
        md_lines.append("| " + " | ".join(header) + " |")
        md_lines.append("|" + "|".join([" --- " for _ in range(cols)]) + "|")
        for r in range(header_rows, rows):
            row = [esc(x) for x in grid[r]]
            md_lines.append("| " + " | ".join(row) + " |")

        markdown = "\n".join(md_lines)
        return markdown, {"table": table0, "tables": tables}

    def _looks_like_non_table_v2(self, payload: Any) -> bool:
        """判断 v2 返回是否明显为非表格区域（plain/无 cells 且有文本行）。"""
        root: Any = payload
        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            root = payload.get("data")

        result: Any = root.get("result") if isinstance(root, dict) and isinstance(root.get("result"), dict) else root
        pages = None
        if isinstance(result, dict):
            pages = result.get("pages") or result.get("page_list")
        if not isinstance(pages, list) or not pages:
            return False

        page0 = pages[0] if isinstance(pages[0], dict) else {}
        tables = page0.get("tables") or page0.get("table_list") or []
        if not isinstance(tables, list) or not tables:
            return False

        table0 = tables[0] if isinstance(tables[0], dict) else {}
        table_type = (table0.get("type") or "").lower()
        cells = (
            table0.get("table_cells")
            or table0.get("cells")
            or table0.get("tableCells")
            or table0.get("table_cells_list")
            or table0.get("cell_list")
            or []
        )
        lines = table0.get("lines") or []
        return table_type == "plain" and (not cells) and bool(lines)

    def _extract_markdown_text(self, payload: Any) -> str:
        """递归地从返回 payload 中提取 markdown 字段（兼容不同接口/版本的字段命名）。"""
        keys = {"markdown", "md", "table_markdown", "tableMarkdown"}
        if isinstance(payload, dict):
            for k, v in payload.items():
                if k in keys and isinstance(v, str) and v.strip():
                    return v.strip()
            for v in payload.values():
                found = self._extract_markdown_text(v)
                if found:
                    return found
        elif isinstance(payload, list):
            for v in payload:
                found = self._extract_markdown_text(v)
                if found:
                    return found
        return ""

    def _rebuild_markdown_from_json(self, json_struct: Any) -> str:
        """从 JSON 结构尝试重建 Markdown 表格（当 API 未直接返回 markdown 时）。"""
        # 常见：{"rows": [[...], [...]]}
        if isinstance(json_struct, dict):
            rows = json_struct.get("rows")
            if isinstance(rows, list) and rows:
                return self._markdown_from_rows(rows)

            # 有些返回会把表格包装成 tables/list
            tables = json_struct.get("tables") or json_struct.get("table_list")
            if isinstance(tables, list) and tables:
                return self._rebuild_markdown_from_json(tables[0])

            # 尝试从 cells 结构重建
            cells = json_struct.get("cells") or json_struct.get("table_cells") or json_struct.get("tableCells")
            if isinstance(cells, list) and cells:
                md, _ = self._parse_v2_table_result({"tables": [{"cells": cells}]})
                return md

        # 另一种：list[dict] 行记录
        if isinstance(json_struct, list) and json_struct and all(isinstance(x, dict) for x in json_struct):
            keys = []
            seen = set()
            for row in json_struct:
                for k in row.keys():
                    if k not in seen:
                        seen.add(k)
                        keys.append(k)
            if not keys:
                return ""
            rows = [keys]
            for row in json_struct[:50]:
                rows.append([str(row.get(k, "")) for k in keys])
            return self._markdown_from_rows(rows)

        return ""

    def _markdown_from_rows(self, rows: list) -> str:
        """rows 支持 list[list[Any]]。第一行作为表头。"""
        if not rows or not isinstance(rows, list):
            return ""
        # 过滤掉非 list 行
        norm_rows: list[list[str]] = []
        for r in rows:
            if isinstance(r, list):
                norm_rows.append([str(x) if x is not None else "" for x in r])
        if not norm_rows:
            return ""
        cols = max(len(r) for r in norm_rows)
        if cols <= 0:
            return ""
        def esc(s: str) -> str:
            return (s or "").replace("\n", " ").replace("|", "\\|").strip()
        header = [esc(x) for x in (norm_rows[0] + [""] * cols)[:cols]]
        if all(not h for h in header):
            header = [f"Col{i+1}" for i in range(cols)]
        md_lines = []
        md_lines.append("| " + " | ".join(header) + " |")
        md_lines.append("|" + "|".join([" --- " for _ in range(cols)]) + "|")
        for r in norm_rows[1:]:
            row = [esc(x) for x in (r + [""] * cols)[:cols]]
            md_lines.append("| " + " | ".join(row) + " |")
        return "\n".join(md_lines)
    
    def recognize_formula(self, image_path: str) -> Dict[str, Any]:
        """
        识别公式
        
        Args:
            image_path: 图片路径
            
        Returns:
            包含 latex 的字典
        """
        if not (self.app_id and self.secret_code):
            return self._mock_formula_response()
        try:
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            url = f"{self.BASE_URL}/ai/service/v1/formula/recognize"
            payload = {"image": image_base64}
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "latex": result.get("data", {}).get("latex", ""),
                    "plain_text": result.get("data", {}).get("text", ""),
                    "raw_response": result
                }
            else:
                return self._mock_formula_response()
                
        except Exception as e:
            print(f"[TextIn] 公式识别异常: {e}")
            return self._mock_formula_response()
    
    def _mock_table_response(self) -> Dict[str, Any]:
        """模拟表格识别响应"""
        return {
            "markdown_table": "| Column1 | Column2 |\n|---------|----------|\n| Data1 | Data2 |",
            "json_structure": {"rows": [["Column1", "Column2"], ["Data1", "Data2"]]},
            "raw_response": "mock_response",
            "source": "mock",
        }
    
    def _mock_formula_response(self) -> Dict[str, Any]:
        """模拟公式识别响应"""
        return {
            "latex": "E = mc^2",
            "plain_text": "E equals m c squared",
            "raw_response": "mock_response"
        }


# ============================================================================
# 表格专家
# ============================================================================

class TableExpertAgent:
    """
    表格专家 Agent
    
    使用合合TextIn进行表格结构化提取，并描述表格内容
    """
    
    def __init__(self, textin_client: TextInAPIClient, vlm_client=None):
        """
        初始化表格专家
        
        Args:
            textin_client: TextIn API客户端
            vlm_client: VLM客户端（用于表格语义分析）
        """
        self.textin_client = textin_client
        self.vlm_client = vlm_client
        
        if vlm_client:
            try:
                from config import get_config
                self.model = get_config().vlm_runtime_model_name
            except:
                self.model = "gpt-4o"
    
    def extract(self, 
               roi_path: str,
               context_text: str = "") -> TableExtraction:
        """
        提取表格内容
        
        Args:
            roi_path: 表格ROI图片路径
            context_text: 上下文文本（标题/说明）
            
        Returns:
            TableExtraction 对象
        """
        print(f"[TableExpert] 处理表格: {roi_path}")
        
        # 1. 调用TextIn识别表格结构
        table_result = self.textin_client.recognize_table(roi_path)

        markdown_table = (table_result.get("markdown_table") or "").strip()
        json_structure = table_result.get("json_structure", {})
        mineru_fallback_markdown = ""
        fallback_reason = ""

        # 兜底：当合合调用失败/解析为空时，尝试使用 MinerU 解析结果（slide_XXX.md）
        # 典型失败表现：返回 mock_response 或固定占位表格。
        if self._looks_like_mock_table(markdown_table, table_result):
            mineru_fallback_markdown = self._fallback_markdown_from_mineru(roi_path)
            if mineru_fallback_markdown:
                markdown_table = mineru_fallback_markdown
                fallback_reason = "mock_or_invalid_textin"
                # 记录来源，便于调试
                if isinstance(json_structure, dict):
                    json_structure = dict(json_structure)
                    json_structure.setdefault("_fallback", "mineru")
                    json_structure.setdefault("_fallback_reason", fallback_reason)
            else:
                # 保持原结果（可能是 mock），避免影响后续流程
                pass
        else:
            # 方案A：交叉对比 TextIn 与 MinerU 的结构规模，识别“看似合法但明显截断”的表格。
            mineru_fallback_markdown = self._fallback_markdown_from_mineru(roi_path)
            should_use_mineru, compare_metrics = self._should_prefer_mineru_markdown(
                textin_markdown=markdown_table,
                mineru_markdown=mineru_fallback_markdown,
            )
            if should_use_mineru:
                markdown_table = mineru_fallback_markdown
                fallback_reason = "quality_guard_textin_truncated"
                if isinstance(json_structure, dict):
                    json_structure = dict(json_structure)
                    json_structure.setdefault("_fallback", "mineru")
                    json_structure.setdefault("_fallback_reason", fallback_reason)
                    json_structure.setdefault("_quality_compare", compare_metrics)
                print(
                    "[TableExpert] TextIn表格疑似截断，切换 MinerU 兜底 | "
                    f"textin_rows={compare_metrics.get('textin', {}).get('data_rows', 0)}, "
                    f"mineru_rows={compare_metrics.get('mineru', {}).get('data_rows', 0)}, "
                    f"textin_chars={compare_metrics.get('textin', {}).get('non_space_chars', 0)}, "
                    f"mineru_chars={compare_metrics.get('mineru', {}).get('non_space_chars', 0)}"
                )
        
        # 2. 如果有VLM，进行语义分析
        insight = ""
        key_values = {}
        table_title = ""
        external_texts: List[str] = []
        footnotes: List[str] = []
        notes: List[str] = []
        internal_headers: List[str] = self._extract_table_headers(markdown_table)
        
        if self.vlm_client and markdown_table:
            analysis = self._analyze_table_semantics(roi_path, markdown_table, context_text)
            insight = analysis.get("insight", "")
            key_values = analysis.get("key_values", {})
            table_title = str(analysis.get("table_title", "") or "").strip()

            external_raw = analysis.get("external_texts", [])
            footnotes_raw = analysis.get("footnotes", [])
            notes_raw = analysis.get("notes", [])
            headers_raw = analysis.get("internal_headers", [])

            if isinstance(external_raw, list):
                external_texts = [str(item).strip() for item in external_raw if str(item).strip()]
            if isinstance(footnotes_raw, list):
                footnotes = [str(item).strip() for item in footnotes_raw if str(item).strip()]
            if isinstance(notes_raw, list):
                notes = [str(item).strip() for item in notes_raw if str(item).strip()]
            if isinstance(headers_raw, list):
                internal_headers.extend(str(item).strip() for item in headers_raw if str(item).strip())

        external_texts = self._filter_table_external_texts(
            external_texts,
            markdown_table=markdown_table,
            known_texts=[context_text, insight, table_title],
        )
        footnotes = self._filter_table_external_texts(
            footnotes,
            markdown_table=markdown_table,
            known_texts=[context_text, insight, table_title],
        )
        notes = self._filter_table_external_texts(
            notes,
            markdown_table=markdown_table,
            known_texts=[context_text, insight, table_title],
        )
        internal_headers = MixedExpertAgent._dedup_text_items(internal_headers)
        
        return TableExtraction(
            markdown_table=markdown_table,
            json_structure=json_structure,
            insight=insight,
            key_values=key_values,
            table_title=table_title,
            external_texts=external_texts,
            footnotes=footnotes,
            notes=notes,
            internal_headers=internal_headers,
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
        normalized = normalized.replace("（", "(").replace("）", ")")
        return normalized

    def _extract_table_headers(self, markdown_table: str) -> List[str]:
        if not markdown_table:
            return []

        lines = [ln.strip() for ln in markdown_table.splitlines() if ln.strip()]
        if not lines:
            return []

        header_lines = [lines[0]]
        if len(lines) > 2:
            second_data_line = lines[2]
            if "|" in second_data_line:
                header_lines.append(second_data_line)

        headers: List[str] = []
        for line in header_lines:
            if "|" not in line:
                continue
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            headers.extend(cells)

        return MixedExpertAgent._dedup_text_items(headers)

    def _filter_table_external_texts(self, items: List[str], markdown_table: str, known_texts: List[str]) -> List[str]:
        if not items:
            return []

        normalized_markdown = self._normalize_text(markdown_table)
        normalized_known = {self._normalize_text(text) for text in known_texts if str(text or "").strip()}

        kept: List[str] = []
        for item in MixedExpertAgent._dedup_text_items(items):
            norm_item = self._normalize_text(item)
            if not norm_item:
                continue

            if norm_item in normalized_known:
                continue

            if len(norm_item) <= 2:
                continue

            if norm_item in normalized_markdown:
                continue

            kept.append(item)

        return kept

    def _looks_like_mock_table(self, markdown_table: str, table_result: Dict[str, Any]) -> bool:
        """判断当前表格是否为 mock/无效结果。"""
        if not markdown_table:
            return True
        raw = table_result.get("raw_response")
        if raw == "mock_response" or table_result.get("source") == "mock":
            return True

        # 固定占位表格（兼容历史版本 mock 输出）
        normalized = "\n".join([ln.strip() for ln in markdown_table.strip().splitlines() if ln.strip()])
        if "| Column1 | Column2 |" in normalized and "| Data1 | Data2 |" in normalized:
            # mock 表格通常只有 3 行
            if len(normalized.splitlines()) <= 4:
                return True
        return False

    def _markdown_quality_metrics(self, markdown_table: str) -> Dict[str, int]:
        """统计 Markdown 表格规模，用于 TextIn/MinerU 质量对比。"""
        if not markdown_table:
            return {
                "line_count": 0,
                "table_line_count": 0,
                "data_rows": 0,
                "pipe_count": 0,
                "non_space_chars": 0,
            }

        lines = [ln.strip() for ln in str(markdown_table).splitlines() if ln.strip()]
        table_lines = [ln for ln in lines if "|" in ln]

        def _is_separator(line: str) -> bool:
            normalized = line.replace(" ", "")
            return bool(re.fullmatch(r"\|?[:\-\|]+\|?", normalized))

        data_rows = [ln for ln in table_lines if not _is_separator(ln)]
        non_space_chars = len(re.sub(r"\s+", "", markdown_table))

        return {
            "line_count": len(lines),
            "table_line_count": len(table_lines),
            "data_rows": len(data_rows),
            "pipe_count": markdown_table.count("|"),
            "non_space_chars": non_space_chars,
        }

    def _should_prefer_mineru_markdown(self, textin_markdown: str, mineru_markdown: str) -> tuple[bool, Dict[str, Any]]:
        """对比 TextIn 与 MinerU 表格规模，判断是否应切换至 MinerU。"""
        if not mineru_markdown:
            return False, {}

        textin_metrics = self._markdown_quality_metrics(textin_markdown)
        mineru_metrics = self._markdown_quality_metrics(mineru_markdown)

        # TextIn 空结果但没被 mock 命中时，仍允许切换
        if textin_metrics["data_rows"] == 0 and mineru_metrics["data_rows"] >= 2:
            return True, {"textin": textin_metrics, "mineru": mineru_metrics, "signals": ["textin_empty"]}

        signals: List[str] = []

        # 行数显著差异：MinerU 至少多 2 行，且至少是 TextIn 的 1.5 倍
        if mineru_metrics["data_rows"] >= max(textin_metrics["data_rows"] + 2, int(textin_metrics["data_rows"] * 1.5)):
            signals.append("rows_gap")

        # 内容长度显著差异：去空白字符数明显更多
        min_char_target = max(textin_metrics["non_space_chars"] + 40, int(textin_metrics["non_space_chars"] * 1.8))
        if mineru_metrics["non_space_chars"] >= min_char_target:
            signals.append("chars_gap")

        # 单元格规模差异
        if mineru_metrics["pipe_count"] >= textin_metrics["pipe_count"] + 8:
            signals.append("pipe_gap")

        # 快速规则：TextIn 极短而 MinerU 明显完整时直接切换
        quick_switch = textin_metrics["data_rows"] <= 2 and mineru_metrics["data_rows"] >= 4
        should_switch = quick_switch or (
            len(signals) >= 2 and mineru_metrics["data_rows"] >= 3
        )

        compare_payload = {
            "textin": textin_metrics,
            "mineru": mineru_metrics,
            "signals": signals,
            "quick_switch": quick_switch,
        }
        return should_switch, compare_payload

    def _fallback_markdown_from_mineru(self, roi_path: str) -> str:
        """从 MinerU 产物中抽取 HTML table 并转换为 Markdown。"""
        base_dir, page_id = self._infer_page_base_and_id(roi_path)
        if not base_dir or page_id is None:
            return ""

        candidates = [
            os.path.join(base_dir, "mineru_work", "mineru_output", f"slide_{page_id:03d}", "auto", f"slide_{page_id:03d}.md"),
            os.path.join(base_dir, "mineru_work", "mineru_output", f"slide_{page_id:03d}", "vlm", f"slide_{page_id:03d}.md"),
        ]
        md_path = ""
        for p in candidates:
            if os.path.exists(p):
                md_path = p
                break

        if not md_path:
            slide_dir = os.path.join(base_dir, "mineru_work", "mineru_output", f"slide_{page_id:03d}")
            if os.path.isdir(slide_dir):
                for root, _dirs, files in os.walk(slide_dir):
                    for fn in files:
                        if fn.lower().endswith(".md"):
                            md_path = os.path.join(root, fn)
                            break
                    if md_path:
                        break

        if not md_path or not os.path.exists(md_path):
            return ""

        try:
            with open(md_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            return ""

        tables = self._extract_html_tables(text)
        if not tables:
            return ""

        # 选择“最像表格”的那个：按行数/单元格数排序
        best_md = ""
        best_score = -1
        for t in tables:
            md = self._html_table_to_markdown(t)
            if not md:
                continue
            score = md.count("\n") + md.count("|")
            if score > best_score:
                best_score = score
                best_md = md
        return best_md

    def _infer_page_base_and_id(self, roi_path: str) -> tuple[str, Optional[int]]:
        """从 ROI 路径推断 processing_artifacts/page_XXX 目录与 page_id。"""
        # 常见形态：.../processing_artifacts/page_000/roi_crops/elem_000_roi.png
        m = re.search(r"(?P<base>.*/page_(?P<pid>\d{3}))(?:/|$)", roi_path.replace("\\", "/"))
        if m:
            base_dir = m.group("base")
            try:
                pid = int(m.group("pid"))
            except Exception:
                pid = None
            return base_dir, pid

        # 兜底：用上两级目录推断
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(roi_path)))
        # 尝试从 base_dir 名称解析 page_id
        m2 = re.search(r"page_(\d{3})", base_dir)
        pid2 = int(m2.group(1)) if m2 else None
        return base_dir, pid2

    def _extract_html_tables(self, text: str) -> List[str]:
        if not text:
            return []
        # MinerU 输出通常直接嵌入 <table>...</table>
        matches = re.findall(r"<table[\s\S]*?</table>", text, flags=re.IGNORECASE)
        return [m.strip() for m in matches if m and m.strip()]


    def _html_table_to_markdown(self, table_html: str) -> str:
        if not table_html or "<table" not in table_html.lower():
            return ""

        cleaned = table_html.strip()
        cleaned = html.unescape(cleaned)
        cleaned = re.sub(r"&(?![a-zA-Z]+;|#\d+;)", "&amp;", cleaned)
        
        # Optional: handle <br> tags, converting them to spaces or similar
        cleaned = re.sub(r"<br\s*/?>", " ", cleaned, flags=re.IGNORECASE)

        root = None
        try:
            root = ET.fromstring(f"<root>{cleaned}</root>")
        except Exception:
            root = None

        table = root.find(".//table") if root is not None else None
        if table is None:
            try:
                table_root = ET.fromstring(f"<table>{cleaned}</table>")
                table = table_root
            except Exception:
                return ""

        # Parse trs
        trs = table.findall(".//tr")
        if not trs:
            return ""

        grid_data = {}  # (row_idx, col_idx) -> text
        max_col = 0
        max_row = len(trs)

        for row_idx, tr in enumerate(trs):
            col_idx = 0
            for cell in list(tr):
                tag = (getattr(cell, "tag", "") or "").lower()
                if tag not in {"td", "th"}:
                    continue
                
                txt = "".join(cell.itertext())
                txt = (txt or "").replace("\n", " ").strip()
                
                colspan = int(cell.get("colspan", "1"))
                rowspan = int(cell.get("rowspan", "1"))
                
                # Find the next available col_idx in this row_idx
                while (row_idx, col_idx) in grid_data:
                    col_idx += 1
                
                # Fill the grid
                for r in range(row_idx, row_idx + rowspan):
                    for c in range(col_idx, col_idx + colspan):
                        if r == row_idx and c == col_idx:
                            grid_data[(r, c)] = txt
                        else:
                            grid_data[(r, c)] = txt # Duplicate the text or keep empty. MinerU output is better with duplicated text or empty strings. Usually better to keep empty strings for spanned cells except the top-left one, or duplicate it. Let's keep empty string to not mess up Markdown formatting of text, but let's duplicate for now so data isn't lost if users read row by row. Wait, standard markdown doesn't support colspans. Best to duplicate the content.
                            pass
                
                col_idx += colspan
            
            # update max_col based on how far col_idx went
            # wait, we need to find the max column index across all rows
            
        for r, c in grid_data.keys():
            if c >= max_col:
                 max_col = c + 1

        if max_row == 0 or max_col == 0:
            return ""

        rows = []
        for r in range(max_row):
            row = []
            for c in range(max_col):
                row.append(grid_data.get((r, c), ""))
            rows.append(row)

        if not rows:
            return ""

        def esc(s: str) -> str:
            return (s or "").replace("|", "\\|").strip()

        header = [esc(x) for x in rows[0]]
        if all(not h for h in header):
            header = [f"Col{i+1}" for i in range(max_col)]
            
        md_lines = []
        md_lines.append("| " + " | ".join(header) + " |")
        md_lines.append("|" + "|".join([" --- " for _ in range(max_col)]) + "|")
        for r in rows[1:]:
            md_lines.append("| " + " | ".join(esc(x) for x in r) + " |")
        return "\n".join(md_lines)

    @staticmethod
    def _extract_markdown_table_stats(markdown_table: str) -> Dict[str, int]:
        lines = [ln.strip() for ln in str(markdown_table or "").splitlines() if ln.strip()]
        table_lines = [ln for ln in lines if "|" in ln]
        if not table_lines:
            return {"rows": 0, "cols": 0}

        def _cells(line: str) -> List[str]:
            return [c.strip() for c in line.split("|")[1:-1]] if line.count("|") >= 2 else []

        header_cells = _cells(table_lines[0])
        cols = max(0, len(header_cells))

        data_lines = table_lines[1:]
        if data_lines and re.fullmatch(r"\|?\s*:?[-]+:?\s*(\|\s*:?[-]+:?\s*)+\|?", data_lines[0]):
            data_lines = data_lines[1:]
        rows = max(0, len(data_lines))
        return {"rows": rows, "cols": cols}

    @staticmethod
    def _compact_markdown_for_semantics(markdown_table: str, max_rows: int = 10, max_cols: int = 10) -> str:
        lines = [ln.strip() for ln in str(markdown_table or "").splitlines() if ln.strip()]
        table_lines = [ln for ln in lines if "|" in ln]
        if not table_lines:
            return str(markdown_table or "")[:1500]

        def _split_cells(line: str) -> List[str]:
            if line.count("|") < 2:
                return []
            return [c.strip() for c in line.split("|")[1:-1]]

        rows = [_split_cells(ln) for ln in table_lines]
        rows = [r for r in rows if r]
        if not rows:
            return str(markdown_table or "")[:1500]

        def _is_sep_row(row: List[str]) -> bool:
            return bool(row) and all(re.fullmatch(r":?-+:?", c.replace(" ", "")) for c in row if c is not None)

        if len(rows) >= 2 and _is_sep_row(rows[1]):
            rows.pop(1)

        max_len = max(len(r) for r in rows)
        norm_rows = [r + [""] * (max_len - len(r)) for r in rows]

        # 限制列：优先保留前后列，避免超宽表导致提示过长
        keep_cols: List[int]
        if max_len <= max_cols:
            keep_cols = list(range(max_len))
        else:
            head = max_cols // 2
            tail = max_cols - head
            keep_cols = list(range(head)) + list(range(max_len - tail, max_len))

        trimmed_rows = [[row[i] for i in keep_cols] for row in norm_rows]
        trimmed_rows = trimmed_rows[: max_rows + 2]  # header + separator + data rows

        if not trimmed_rows:
            return str(markdown_table or "")[:1500]

        md_lines: List[str] = []
        header = trimmed_rows[0]
        md_lines.append("| " + " | ".join(header) + " |")
        md_lines.append("|" + "|".join([" --- " for _ in header]) + "|")
        for r in trimmed_rows[1:]:
            md_lines.append("| " + " | ".join(r) + " |")

        return "\n".join(md_lines)

    @staticmethod
    def _safe_parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
        text = str(raw or "").strip()
        if not text:
            return None

        # 去掉 markdown code fence
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

        # 尝试提取首个 JSON 对象
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        candidate = m.group(0)
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    @staticmethod
    def _normalize_semantic_analysis(result: Dict[str, Any]) -> Dict[str, Any]:
        insight = str(result.get("insight", "") or "").strip()
        key_values = result.get("key_values", {})
        if not isinstance(key_values, dict):
            key_values = {}

        table_title = str(result.get("table_title", "") or "").strip()
        external_texts = result.get("external_texts", [])
        footnotes = result.get("footnotes", [])
        notes = result.get("notes", [])
        internal_headers = result.get("internal_headers", [])

        if not isinstance(external_texts, list):
            external_texts = []
        if not isinstance(footnotes, list):
            footnotes = []
        if not isinstance(notes, list):
            notes = []
        if not isinstance(internal_headers, list):
            internal_headers = []

        return {
            "insight": insight,
            "key_values": key_values,
            "table_title": table_title,
            "external_texts": [str(x).strip() for x in external_texts if str(x).strip()],
            "footnotes": [str(x).strip() for x in footnotes if str(x).strip()],
            "notes": [str(x).strip() for x in notes if str(x).strip()],
            "internal_headers": [str(x).strip() for x in internal_headers if str(x).strip()],
        }

    def _build_table_semantic_fallback(self, markdown_table: str, context_text: str) -> Dict[str, Any]:
        stats = self._extract_markdown_table_stats(markdown_table)

        # 取表头前几个字段作为结构摘要
        header_cells: List[str] = []
        for ln in str(markdown_table or "").splitlines():
            if "|" in ln and ln.count("|") >= 2:
                header_cells = [c.strip() for c in ln.split("|")[1:-1] if c.strip()]
                break
        header_preview = "、".join(header_cells[:4]) if header_cells else "多指标字段"

        context_hint = str(context_text or "").strip()
        if context_hint:
            insight = (
                f"结合页面上下文，本表围绕“{context_hint}”展开，"
                f"采用按时间/指标展开的结构（约{stats['rows']}行、{stats['cols']}列），"
                f"核心字段包括{header_preview}。"
            )
        else:
            insight = (
                f"该表为多指标对照表，结构约为{stats['rows']}行、{stats['cols']}列，"
                f"核心字段包括{header_preview}。"
            )

        kv: Dict[str, str] = {
            "表格结构": f"约{stats['rows']}行 × {stats['cols']}列",
            "核心字段": header_preview or "未识别到明确表头",
        }

        m_irr = re.search(r"IRR\s*[=:：]\s*([\d\.]+%)", context_hint, flags=re.IGNORECASE)
        m_npv = re.search(r"NPV\s*[=:：]\s*([\d,\.]+\s*万元?)", context_hint, flags=re.IGNORECASE)
        if m_irr:
            kv["上下文IRR"] = m_irr.group(1)
        if m_npv:
            kv["上下文NPV"] = m_npv.group(1)

        return {
            "insight": insight,
            "key_values": kv,
            "table_title": "",
            "external_texts": [],
            "footnotes": [],
            "notes": [],
            "internal_headers": [],
        }



    def _analyze_table_semantics(self, 
                                roi_path: str,
                                markdown_table: str,
                                context_text: str) -> Dict[str, Any]:
        """使用VLM分析表格语义"""
        default_empty = {
            "insight": "",
            "key_values": {},
            "table_title": "",
            "external_texts": [],
            "footnotes": [],
            "notes": [],
            "internal_headers": [],
        }

        if not self.vlm_client:
            return self._build_table_semantic_fallback(markdown_table, context_text)

        try:
            with open(roi_path, "rb") as f:
                base64_img = base64.b64encode(f.read()).decode('utf-8')
        except Exception:
            return self._build_table_semantic_fallback(markdown_table, context_text)

        stats = self._extract_markdown_table_stats(markdown_table)
        compact_md = self._compact_markdown_for_semantics(markdown_table, max_rows=10, max_cols=10)

        prompt_variants = [
            {
                "table_md": str(markdown_table or "")[:1800],
                "extra": "请优先保证 insight 与 key_values 非空；若图片不清晰，请结合上下文与表格结构给出保守结论。",
            },
            {
                "table_md": compact_md[:1800],
                "extra": (
                    "这是压缩后的宽表摘要（原表约"
                    f"{stats.get('rows', 0)}行×{stats.get('cols', 0)}列）。"
                    "请基于摘要字段与上下文提炼洞察，并输出3-6个关键键值对。"
                ),
            },
        ]

        last_error: Optional[str] = None
        for attempt_idx, variant in enumerate(prompt_variants, start=1):
            try:
                prompt = f"""你是一名数据分析师。请分析这个表格。
语言要求（必须严格遵守）：
- 所有输出必须为中文，不得夹杂英文句子。
- 若引用的标题/术语为英文或主要是英文，请保留英文原文，并在其后追加中文翻译（括号内）。
- 原文摘录字段必须保留图中原文，不可翻译。

【上下文】
{context_text if context_text else "无"}

【表格内容（Markdown）】
{variant['table_md']}

【补充要求】
{variant['extra']}

请提供：
1. insight: 两段式洞察（先核心观点，再整体结构/趋势描述）
2. key_values: 与核心观点强相关的关键数据键值对（3-6项）
3. table_title: 表题/表头说明原文（若有）
4. external_texts: 仅包含“表格外部”的说明原文（不包含表格单元格与列名）
5. footnotes: 表格脚注/注释原文
6. notes: 其他补充说明原文（如单位说明、统计说明）
7. internal_headers: 表内表头原文（仅用于内部，不用于最终展示）

注意：
- external_texts 不能重复 markdown 表格中已经出现的列名/单元格文本。
- 所有原文字段保留原始语言，不翻译，不改写。
- 仅返回 JSON 对象，不要包含任何解释文字。

返回JSON格式：
{{
  "insight": "...",
  "key_values": {{"关键项": "值"}},
  "table_title": "...",
  "external_texts": ["..."],
  "footnotes": ["..."],
  "notes": ["..."],
  "internal_headers": ["..."]
}}"""

                response = self.vlm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是严谨的数据分析助手，请只返回合法JSON对象。"},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                        ]}
                    ],
                    max_tokens=900,
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )

                raw = response.choices[0].message.content if response and response.choices else ""
                parsed = self._safe_parse_json_object(raw)
                if not parsed:
                    raise ValueError("模型返回非JSON对象或JSON解析失败")

                normalized = self._normalize_semantic_analysis(parsed)
                if normalized.get("insight") or normalized.get("key_values"):
                    return normalized

                # 返回空结构时允许重试下一轮
                last_error = "模型返回空 insight/key_values"
                print(f"[TableExpert] 语义分析第{attempt_idx}轮为空，准备重试")
            except Exception as e:
                last_error = str(e)
                print(f"[TableExpert] 语义分析第{attempt_idx}轮失败: {e}")

        # 最终兜底：保证下游至少可展示洞察+关键数据
        fallback = self._build_table_semantic_fallback(markdown_table, context_text)
        # 避免返回完全空结构
        if not fallback.get("insight") and not fallback.get("key_values"):
            return default_empty
        return fallback


# ============================================================================
# 图表专家
# ============================================================================

class ChartExpertAgent:
    """
    图表专家 Agent
    
    使用 GPT-4o 分析数据图表，提取结构化数据和洞察
    """
    
    def __init__(self, vlm_client):
        """
        初始化图表专家
        
        Args:
            vlm_client: VLM客户端（GPT-4o）
        """
        self.vlm_client = vlm_client
        
        try:
            from config import get_config
            self.model = get_config().vlm_runtime_model_name
        except:
            self.model = "gpt-4o"

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
        normalized = normalized.replace("（", "(").replace("）", ")")
        return normalized

    @staticmethod
    def _normalize_chart_type(
        chart_type: str,
        context_text: str,
        chart_title: str,
        panel_titles: List[str],
        axis_labels: List[str],
        annotation_texts: List[str],
    ) -> str:
        ct = str(chart_type or "").strip().lower()
        if ct in {"flowchart", "timeline", "process_diagram", "gantt_like", "line_chart", "bar_chart", "pie_chart", "km_curve", "forest_plot", "scatter", "heatmap"}:
            return ct

        clues = " ".join([
            str(context_text or ""),
            str(chart_title or ""),
            " ".join(str(x) for x in (panel_titles or [])),
            " ".join(str(x) for x in (axis_labels or [])),
            " ".join(str(x) for x in (annotation_texts or [])),
        ]).lower()

        has_timeline = any(k in clues for k in ["timeline", "时间线", "milestone", "阶段", "month", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "202", "2025", "2026"])
        has_process = any(k in clues for k in ["流程", "workflow", "process", "step", "阶段", "synthesis", "validation"]) 

        if has_timeline:
            return "timeline"
        if has_process:
            return "flowchart"

        if ct in {"", "unknown", "other", "未知", "其他"}:
            return "other"
        return ct

    @staticmethod
    def _classify_chart_texts(raw_texts: List[str]) -> Dict[str, List[str]]:
        panel_titles: List[str] = []
        legends: List[str] = []
        axis_labels: List[str] = []
        annotations: List[str] = []
        footnotes: List[str] = []

        for raw in raw_texts:
            text = str(raw or "").strip()
            if not text:
                continue
            lower = text.lower()

            if re.search(r"(^note[:\s])|(^备注)|(^注[:：])|\bp\s*[<=>]\s*0?\.\d+|\bn\s*=", lower):
                footnotes.append(text)
                continue

            if any(k in lower for k in ["legend", "图例"]) or re.match(r"^[A-Za-z0-9\-_/]{2,20}$", text):
                legends.append(text)
                continue

            if any(k in lower for k in ["axis", "conc", "inhibition", "%", "(nm", "(ng", "(h", "x轴", "y轴", "单位"]):
                axis_labels.append(text)
                continue

            if len(text) >= 20 or any(k in lower for k in ["trend", "increase", "decrease", "显示", "说明", "表明"]):
                annotations.append(text)
                continue

            panel_titles.append(text)

        return {
            "panel_titles": MixedExpertAgent._dedup_text_items(panel_titles),
            "legend_items": MixedExpertAgent._dedup_text_items(legends),
            "axis_labels": MixedExpertAgent._dedup_text_items(axis_labels),
            "annotation_texts": MixedExpertAgent._dedup_text_items(annotations),
            "footnotes": MixedExpertAgent._dedup_text_items(footnotes),
        }

    @staticmethod
    def _filter_chart_texts(items: List[str], known_texts: List[str]) -> List[str]:
        normalized_known = {ChartExpertAgent._normalize_text(k) for k in known_texts if str(k or "").strip()}
        kept: List[str] = []
        for item in MixedExpertAgent._dedup_text_items(items):
            norm_item = ChartExpertAgent._normalize_text(item)
            if not norm_item:
                continue
            if norm_item in normalized_known:
                continue
            kept.append(item)
        return kept
    
    def extract(self,
               roi_path: str,
               context_text: str = "") -> ChartExtraction:
        """
        提取图表内容
        
        Args:
            roi_path: 图表ROI图片路径
            context_text: 上下文文本（标题/描述）- 非常重要！
            
        Returns:
            ChartExtraction 对象
        """
        print(f"[ChartExpert] 处理图表: {roi_path}")
        
        try:
            with open(roi_path, "rb") as f:
                base64_img = base64.b64encode(f.read()).decode('utf-8')
            
            prompt = f"""你是一名数据分析师。
            语言要求（必须严格遵守）：
            - 所有输出必须为中文，不得夹杂英文句子。
            - 若引用的标题/术语为英文或主要是英文，请保留英文原文，并在其后追加中文翻译（括号内）。
            - 原文摘录字段必须保留图中原文，不可翻译。

【背景上下文】
此图表的标题/描述是：'{context_text}'（非常重要，请基于此确定图表主题）。

【任务】
1. 结构化提取：识别图表类型、坐标轴含义、关键数值
2. 洞察分析：结合背景上下文，用一句话总结图表展示的核心趋势或异常点

【输出JSON格式】
{{
    "chart_type": "bar_chart/line_chart/pie_chart/km_curve/forest_plot/flowchart/timeline/process_diagram/gantt_like/other",
    "x_axis": "X轴含义（中文，必要时附英文原文翻译）",
    "y_axis": "Y轴含义（中文，必要时附英文原文翻译）",
    "key_values": {{"系列名": "关键数值", ...}},
    "insight": "一句话洞察分析（中文）",
    "chart_title": "图标题原文",
    "panel_titles": ["子图标题原文"],
    "legend_items": ["图例项原文"],
    "axis_labels": ["坐标轴标签/刻度原文"],
    "annotation_texts": ["图中解释性文字原文"],
    "footnotes": ["图下注释/脚注原文"]
}}"""
            
            response = self.vlm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是专业的数据分析师，擅长从图表中提取结构化信息。"},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                    ]}
                ],
                max_tokens=1000,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            panel_titles = result.get("panel_titles", [])
            legend_items = result.get("legend_items", [])
            axis_labels = result.get("axis_labels", [])
            annotation_texts = result.get("annotation_texts", [])
            footnotes = result.get("footnotes", [])

            if not isinstance(panel_titles, list):
                panel_titles = []
            if not isinstance(legend_items, list):
                legend_items = []
            if not isinstance(axis_labels, list):
                axis_labels = []
            if not isinstance(annotation_texts, list):
                annotation_texts = []
            if not isinstance(footnotes, list):
                footnotes = []

            # 兼容旧版输出：仅返回 verbatim_texts 时做启发式分类
            if not (panel_titles or legend_items or axis_labels or annotation_texts or footnotes):
                legacy_verbatim = result.get("verbatim_texts", [])
                if isinstance(legacy_verbatim, list):
                    classified = self._classify_chart_texts([str(t).strip() for t in legacy_verbatim if str(t).strip()])
                    panel_titles = classified["panel_titles"]
                    legend_items = classified["legend_items"]
                    axis_labels = classified["axis_labels"]
                    annotation_texts = classified["annotation_texts"]
                    footnotes = classified["footnotes"]

            chart_title = str(result.get("chart_title", "") or "").strip()
            if context_text.strip() and not chart_title:
                chart_title = context_text.strip()

            known_texts = [chart_title, context_text]
            panel_titles = self._filter_chart_texts(panel_titles, known_texts)
            legend_items = self._filter_chart_texts(legend_items, known_texts)
            axis_labels = self._filter_chart_texts(axis_labels, known_texts)
            annotation_texts = self._filter_chart_texts(annotation_texts, known_texts)
            footnotes = self._filter_chart_texts(footnotes, known_texts)
            chart_type = self._normalize_chart_type(
                chart_type=result.get("chart_type", "unknown"),
                context_text=context_text,
                chart_title=chart_title,
                panel_titles=panel_titles,
                axis_labels=axis_labels,
                annotation_texts=annotation_texts,
            )
            
            return ChartExtraction(
                chart_type=chart_type,
                panel_name="",
                x_axis=result.get("x_axis", ""),
                y_axis=result.get("y_axis", ""),
                key_values=result.get("key_values", {}),
                insight=result.get("insight", ""),
                context_title=context_text,
                chart_title=chart_title,
                panel_titles=panel_titles,
                legend_items=legend_items,
                axis_labels=axis_labels,
                annotation_texts=annotation_texts,
                footnotes=footnotes,
            )
            
        except Exception as e:
            print(f"[ChartExpert] 图表分析失败: {e}")
            return ChartExtraction(
                chart_type="unknown",
                insight=f"分析失败: {str(e)}",
                chart_title=context_text.strip() if context_text else "",
            )


class MixedExpertAgent:
    """混合元素专家 Agent（强校验/强覆盖）：一个ROI内同时提取图表与表格。"""

    def __init__(self, vlm_client, table_expert: TableExpertAgent, chart_expert: ChartExpertAgent):
        self.vlm_client = vlm_client
        self.table_expert = table_expert
        self.chart_expert = chart_expert
        try:
            from config import get_config
            self.model = get_config().vlm_runtime_model_name
        except Exception:
            self.model = "gpt-4o"

    @staticmethod
    def _dedup_text_items(items: List[str]) -> List[str]:
        deduped: List[str] = []
        seen = set()
        for item in items:
            text = str(item or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(text)
        return deduped

    @staticmethod
    def _coerce_chart_item(item: Dict[str, Any], context_text: str) -> ChartExtraction:
        panel_name = (
            item.get("panel_name")
            or item.get("panel_title")
            or item.get("subplot_name")
            or ""
        )
        panel_titles = item.get("panel_titles", [])
        legend_items = item.get("legend_items", [])
        axis_labels = item.get("axis_labels", [])
        annotation_texts = item.get("annotation_texts", [])
        footnotes = item.get("footnotes", [])

        if not isinstance(panel_titles, list):
            panel_titles = []
        if not isinstance(legend_items, list):
            legend_items = []
        if not isinstance(axis_labels, list):
            axis_labels = []
        if not isinstance(annotation_texts, list):
            annotation_texts = []
        if not isinstance(footnotes, list):
            footnotes = []

        if not (panel_titles or legend_items or axis_labels or annotation_texts or footnotes):
            legacy_texts = item.get("verbatim_texts")
            if not isinstance(legacy_texts, list):
                legacy_texts = item.get("raw_texts")
            if isinstance(legacy_texts, list):
                classified = ChartExpertAgent._classify_chart_texts([str(t).strip() for t in legacy_texts if str(t).strip()])
                panel_titles = classified["panel_titles"]
                legend_items = classified["legend_items"]
                axis_labels = classified["axis_labels"]
                annotation_texts = classified["annotation_texts"]
                footnotes = classified["footnotes"]

        chart_title = str(item.get("chart_title", "") or "").strip()
        if not chart_title and context_text.strip():
            chart_title = context_text.strip()
        chart_type = ChartExpertAgent._normalize_chart_type(
            chart_type=item.get("chart_type", "unknown"),
            context_text=context_text,
            chart_title=chart_title,
            panel_titles=panel_titles,
            axis_labels=axis_labels,
            annotation_texts=annotation_texts,
        )

        return ChartExtraction(
            chart_type=chart_type,
            panel_name=str(panel_name).strip(),
            x_axis=item.get("x_axis", ""),
            y_axis=item.get("y_axis", ""),
            key_values=item.get("key_values", {}) if isinstance(item.get("key_values"), dict) else {},
            insight=item.get("insight", ""),
            context_title=context_text,
            chart_title=chart_title,
            panel_titles=MixedExpertAgent._dedup_text_items(panel_titles),
            legend_items=MixedExpertAgent._dedup_text_items(legend_items),
            axis_labels=MixedExpertAgent._dedup_text_items(axis_labels),
            annotation_texts=MixedExpertAgent._dedup_text_items(annotation_texts),
            footnotes=MixedExpertAgent._dedup_text_items(footnotes),
        )

    @staticmethod
    def _coerce_table_item(item: Dict[str, Any]) -> TableExtraction:
        external_texts = item.get("external_texts", [])
        footnotes = item.get("footnotes", [])
        notes = item.get("notes", [])
        internal_headers = item.get("internal_headers", [])

        if not isinstance(external_texts, list):
            external_texts = []
        if not isinstance(footnotes, list):
            footnotes = []
        if not isinstance(notes, list):
            notes = []
        if not isinstance(internal_headers, list):
            internal_headers = []

        if not (external_texts or footnotes or notes or internal_headers):
            legacy_texts = item.get("verbatim_texts")
            if not isinstance(legacy_texts, list):
                legacy_texts = item.get("raw_texts")
            if isinstance(legacy_texts, list):
                notes = [str(t).strip() for t in legacy_texts if str(t).strip()]

        return TableExtraction(
            markdown_table=item.get("markdown_table", ""),
            json_structure=item.get("json_structure") if isinstance(item.get("json_structure"), dict) else None,
            insight=item.get("insight", ""),
            key_values=item.get("key_values", {}),
            table_title=str(item.get("table_title", "") or "").strip(),
            external_texts=MixedExpertAgent._dedup_text_items(external_texts),
            footnotes=MixedExpertAgent._dedup_text_items(footnotes),
            notes=MixedExpertAgent._dedup_text_items(notes),
            internal_headers=MixedExpertAgent._dedup_text_items(internal_headers),
        )

    @staticmethod
    def _is_table_like_chart(chart: ChartExtraction) -> bool:
        ct = (chart.chart_type or "").strip().lower()
        return "table" in ct or ct in {"grid", "matrix"}

    @staticmethod
    def _dedup_chart_items(items: List[ChartExtraction]) -> List[ChartExtraction]:
        deduped: List[ChartExtraction] = []
        seen = set()
        for item in items:
            key = (
                (item.panel_name or "").strip().lower(),
                (item.chart_type or "").strip().lower(),
                str(item.x_axis).strip(),
                str(item.y_axis).strip(),
                (item.insight or "").strip(),
                json.dumps(item.key_values or {}, ensure_ascii=False, sort_keys=True),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @staticmethod
    def _table_markdown_score(markdown_table: str) -> int:
        md = (markdown_table or "").strip()
        if not md:
            return 0
        lines = [ln for ln in md.splitlines() if ln.strip()]
        if len(lines) < 2:
            return 1
        pipe_lines = [ln for ln in lines if "|" in ln]
        row_score = min(len(pipe_lines), 12)
        text_score = min(len(md) // 40, 8)
        return row_score + text_score

    @staticmethod
    def _contains_numeric_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, dict):
            return any(MixedExpertAgent._contains_numeric_value(v) for v in value.values())
        if isinstance(value, list):
            return any(MixedExpertAgent._contains_numeric_value(v) for v in value)
        text = str(value)
        return bool(re.search(r"\d", text))

    @staticmethod
    def _is_weak_chart_item(chart: ChartExtraction) -> bool:
        chart_type = (chart.chart_type or "").strip().lower()
        insight = (chart.insight or "").strip()
        x_axis = str(chart.x_axis or "").strip()
        y_axis = str(chart.y_axis or "").strip()

        weak_phrases = [
            "未提供", "未明确", "更多信息", "无法判断", "不清晰", "看不清", "无具体数值",
            "需要补充", "图表展示了", "分组情况", "可能为", "未显示"
        ]
        has_weak_phrase = any(p in insight for p in weak_phrases)
        has_numeric = MixedExpertAgent._contains_numeric_value(chart.key_values)
        weak_axis = (not x_axis or "未明确" in x_axis) and (not y_axis or "未明确" in y_axis)

        if chart_type in {"", "unknown"}:
            return True

        if chart_type == "other" and (has_weak_phrase or (not has_numeric and weak_axis)):
            return True

        if has_weak_phrase and not has_numeric:
            return True

        return False

    def _suppress_weak_charts_for_table_dominant(
        self,
        chart_items: List[ChartExtraction],
        table_items: List[TableExtraction]
    ) -> List[ChartExtraction]:
        if not table_items or not chart_items:
            return chart_items

        best_table_score = max(self._table_markdown_score(t.markdown_table) for t in table_items)
        if best_table_score < 6:
            return chart_items

        weak_count = sum(1 for c in chart_items if self._is_weak_chart_item(c))
        if weak_count == len(chart_items):
            return []

        return [c for c in chart_items if not self._is_weak_chart_item(c)]

    def _extract_table_from_bottom_regions(self, roi_path: str, context_text: str) -> Optional[TableExtraction]:
        """对 mixed ROI 的底部区域做表格兜底抽取，解决“图+表”整图识别漏表。"""
        try:
            with Image.open(roi_path) as img:
                width, height = img.size
                if width <= 0 or height <= 0:
                    return None

                ratio_candidates = [
                    (0.50, 1.00),
                    (0.55, 1.00),
                    (0.60, 1.00),
                    (0.45, 0.90),
                ]

                best_table: Optional[TableExtraction] = None
                best_score = 0

                for y_start_ratio, y_end_ratio in ratio_candidates:
                    top = int(height * y_start_ratio)
                    bottom = int(height * y_end_ratio)
                    if bottom - top < max(80, int(height * 0.12)):
                        continue

                    crop = img.crop((0, top, width, bottom))
                    tmp_path = ""
                    try:
                        with NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                            tmp_path = tmp.name
                        crop.save(tmp_path)
                        table_data = self.table_expert.extract(tmp_path, context_text)
                        score = self._table_markdown_score(table_data.markdown_table)
                        if score > best_score:
                            best_score = score
                            best_table = table_data
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass

                return best_table if best_score > 0 else None
        except Exception as e:
            print(f"[MixedExpert] 底部区域表格兜底失败: {e}")
            return None

    def _extract_dual_chart_fallback(self, roi_path: str, context_text: str) -> List[ChartExtraction]:
        """对宽幅 mixed ROI 顶部区域做左右子图兜底，提升双图拆解稳定性。"""
        try:
            with Image.open(roi_path) as img:
                width, height = img.size
                if width <= 0 or height <= 0:
                    return []

                if width < int(height * 1.25):
                    return []

                top_end = int(height * 0.62)
                if top_end < max(120, int(height * 0.25)):
                    return []

                panel_boxes = [
                    ("左图", (0, 0, width // 2, top_end)),
                    ("右图", (width // 2, 0, width, top_end)),
                ]

                fallback_items: List[ChartExtraction] = []
                for panel_name, box in panel_boxes:
                    crop = img.crop(box)
                    tmp_path = ""
                    try:
                        with NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                            tmp_path = tmp.name
                        crop.save(tmp_path)
                        panel_context = f"{context_text} | {panel_name}"
                        chart_data = self.chart_expert.extract(tmp_path, panel_context)
                        chart_data.panel_name = panel_name
                        if (chart_data.chart_type or "").strip() or (chart_data.insight or "").strip():
                            fallback_items.append(chart_data)
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass

                return fallback_items
        except Exception as e:
            print(f"[MixedExpert] 双子图兜底失败: {e}")
            return []

    @staticmethod
    def _dedup_table_items(items: List[TableExtraction]) -> List[TableExtraction]:
        deduped: List[TableExtraction] = []
        seen = set()
        for item in items:
            key = (
                (item.markdown_table or "").strip(),
                (item.insight or "").strip(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def extract(self, roi_path: str, context_text: str = "") -> MixedExtraction:
        print(f"[MixedExpert] 处理混合元素: {roi_path}")

        try:
            with open(roi_path, "rb") as f:
                base64_img = base64.b64encode(f.read()).decode("utf-8")

            prompt = f"""你是“复杂科研图像拆解专家”，请对单张ROI做多元素拆解与结构化提取。
语言要求（必须严格遵守）：
- 所有输出必须为中文（术语可保留英文并附中文解释）。
- 若识别到图片中的原文文字，请在指定字段中按原样保留，不要翻译或改写。

【背景上下文】
{context_text if context_text else "无"}

【任务】
1) 判断该ROI是否为混合内容（例如一张图里含多个子图、图+表、双图对比）。
2) 若包含图表：按“每个子图”分别提取 chart_type/x_axis/y_axis/key_values/insight。洞察须包含试验目的核心观点与详细图表描述，关键数据以键值对形式返回。
3) 若包含表格：输出 markdown_table（必须是标准Markdown表格），并提取 insight和key_values。洞察须包含试验目的核心观点与详细表格描述，关键数据以键值对展示。
4) 输出覆盖性自检：是否遗漏任何主要元素。

【返回JSON】
{{
  "is_mixed": true,
  "summary": "整体一句话洞察",
    "texts": ["ROI中可见的原文描述/图注/脚注，按原文输出"],
  "charts": [
    {{
            "panel_name": "子图标识（如左图/右图/上图）",
      "chart_type": "line_chart/bar_chart/pie_chart/km_curve/forest_plot/other",
      "x_axis": "X轴含义，若多子图可用{{\"左图\":\"...\",\"右图\":\"...\"}}",
      "y_axis": "Y轴含义，若多子图可用{{\"左图\":\"...\",\"右图\":\"...\"}}",
      "key_values": {{"关键项": "值"}},
                        "insight": "该子图一句话洞察",
                        "chart_title": "图标题原文",
                        "panel_titles": ["子图标题原文"],
                        "legend_items": ["图例项原文"],
                        "axis_labels": ["坐标轴标签原文"],
                        "annotation_texts": ["图中解释性文字原文"],
                        "footnotes": ["图下注释/脚注原文"]
    }}
  ],
  "tables": [
    {{
      "markdown_table": "|A|B|\\n|---|---|\\n|...|...|",
      "insight": "包含核心观点和整体描述的两部分洞察",
                        "key_values": {"围绕核心观点的关键项": "值"},
                        "table_title": "表题/表头说明原文",
                        "external_texts": ["表格外部说明文字原文"],
                        "footnotes": ["表格脚注原文"],
                        "notes": ["单位或统计说明原文"],
                        "internal_headers": ["表内表头原文"]
    }}
  ],
  "validation_notes": "覆盖检查结果"
}}"""

            response = self.vlm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你擅长复杂科研图像中多元素拆解，请只返回JSON对象。"},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                    ]}
                ],
                max_tokens=2000,
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            chart_items_raw = result.get("charts", [])
            table_items_raw = result.get("tables", [])
            text_items_raw = result.get("texts", [])

            chart_items: List[ChartExtraction] = []
            if isinstance(chart_items_raw, list):
                for item in chart_items_raw:
                    if isinstance(item, dict):
                        chart_items.append(self._coerce_chart_item(item, context_text))

            table_items: List[TableExtraction] = []
            if isinstance(table_items_raw, list):
                for item in table_items_raw:
                    if isinstance(item, dict):
                        table_items.append(self._coerce_table_item(item))

            text_items: List[str] = []
            if isinstance(text_items_raw, list):
                text_items.extend(str(item).strip() for item in text_items_raw if str(item).strip())
            if context_text.strip():
                text_items.append(context_text.strip())

            # 兜底：如果疑似有表格但未产出可用 markdown，优先对底部区域尝试补提取
            markdown_exists = any((t.markdown_table or "").strip() for t in table_items)
            if (result.get("is_mixed") or table_items_raw) and not markdown_exists:
                fallback_candidates: List[TableExtraction] = []
                full_roi_table = self.table_expert.extract(roi_path, context_text)
                if (full_roi_table.markdown_table or "").strip():
                    fallback_candidates.append(full_roi_table)

                bottom_region_table = self._extract_table_from_bottom_regions(roi_path, context_text)
                if bottom_region_table and (bottom_region_table.markdown_table or "").strip():
                    fallback_candidates.append(bottom_region_table)

                if fallback_candidates:
                    best_table = max(
                        fallback_candidates,
                        key=lambda t: self._table_markdown_score(t.markdown_table)
                    )
                    table_items.append(best_table)

            # 当模型未能稳定拆出多子图且当前非表格主导时，对宽幅 ROI 执行左右子图兜底
            has_table_already = any((t.markdown_table or "").strip() for t in table_items)
            if len(chart_items) < 2 and not has_table_already:
                dual_items = self._extract_dual_chart_fallback(roi_path, context_text)
                chart_items.extend(dual_items)

            # 去重 + 语义规范：当已识别出表格时，过滤掉“chart_type=table”的伪图表，避免双重解释
            table_items = self._dedup_table_items(table_items)
            if table_items:
                chart_items = [c for c in chart_items if not self._is_table_like_chart(c)]
            chart_items = self._dedup_chart_items(chart_items)
            chart_items = self._suppress_weak_charts_for_table_dominant(chart_items, table_items)

            # 兜底：仅在没有表格时补单图，避免“纯表格ROI被强行图解”
            if not chart_items and not table_items:
                fallback_chart = self.chart_expert.extract(roi_path, context_text)
                if fallback_chart.chart_type or fallback_chart.insight:
                    chart_items.append(fallback_chart)

            for chart in chart_items:
                text_items.extend(chart.annotation_texts)
                text_items.extend(chart.footnotes)
            for table in table_items:
                text_items.extend(table.external_texts)
                text_items.extend(table.footnotes)
                text_items.extend(table.notes)

            text_items = self._dedup_text_items(text_items)

            contains_multiple = bool(result.get("is_mixed")) or (
                (len(chart_items) + len(table_items)) >= 2
            ) or (bool(chart_items) and bool(table_items))

            return MixedExtraction(
                summary=result.get("summary", ""),
                chart_items=chart_items,
                table_items=table_items,
                text_items=text_items,
                contains_multiple_elements=contains_multiple,
                validation_notes=result.get("validation_notes", ""),
            )

        except Exception as e:
            print(f"[MixedExpert] 混合提取失败: {e}")
            # 失败兜底：降级为普通图表+尝试表格
            chart_fallback = self.chart_expert.extract(roi_path, context_text)
            table_fallback = self.table_expert.extract(roi_path, context_text)
            table_items = [table_fallback] if (table_fallback.markdown_table or "").strip() else []
            return MixedExtraction(
                summary=f"混合提取降级: {str(e)}",
                chart_items=[chart_fallback],
                table_items=table_items,
                text_items=self._dedup_text_items(
                    ([context_text] if context_text else [])
                    + chart_fallback.annotation_texts
                    + chart_fallback.footnotes
                    + [t for table in table_items for t in (table.external_texts + table.footnotes + table.notes)]
                ),
                contains_multiple_elements=bool(table_items),
                validation_notes="fallback",
            )


# ============================================================================
# 公式专家
# ============================================================================

class FormulaExpertAgent:
    """
    公式专家 Agent
    
    使用 TextIn/Mathpix 提取公式的 LaTeX 格式
    """
    
    def __init__(self, textin_client: TextInAPIClient):
        """
        初始化公式专家
        
        Args:
            textin_client: TextIn API客户端
        """
        self.textin_client = textin_client
    
    def extract(self, roi_path: str) -> FormulaExtraction:
        """
        提取公式
        
        Args:
            roi_path: 公式ROI图片路径
            
        Returns:
            FormulaExtraction 对象
        """
        print(f"[FormulaExpert] 处理公式: {roi_path}")
        
        result = self.textin_client.recognize_formula(roi_path)
        
        return FormulaExtraction(
            latex=result.get("latex", ""),
            plain_text=result.get("plain_text")
        )


# ============================================================================
# 文本处理器
# ============================================================================

class TextProcessorAgent:
    """
    文本处理器 Agent
    
    处理文本元素：
    1. 直接使用 MinerU 的 OCR 结果
    2. 结合 Phase 2 的分组信息，将散落的文本块按语义拼接
    """
    
    def __init__(self):
        """初始化文本处理器"""
        pass
    
    def process(self,
               elements: List[DetectedElement],
               group: SemanticGroup,
               id_to_text_map: Dict[str, str]) -> TextExtraction:
        """
        处理文本元素
        
        Args:
            elements: 组内的元素列表
            group: 分组信息
            id_to_text_map: ID到文本的映射
            
        Returns:
            TextExtraction 对象
        """
        print(f"[TextProcessor] 处理文本组: {group.group_id}")
        
        # 获取组内文本元素
        text_pieces = []
        elem_map = {e.element_id: e for e in elements}
        
        # 按阅读顺序排序组内元素
        sorted_member_ids = sorted(
            group.member_ids,
            key=lambda mid: (
                elem_map[mid].bbox.ymin if mid in elem_map else 0,
                elem_map[mid].bbox.xmin if mid in elem_map else 0
            )
        )
        
        for member_id in sorted_member_ids:
            text = id_to_text_map.get(member_id, "")
            if text.strip():
                text_pieces.append(text.strip())
        
        # 智能合并文本
        merged_text = self._smart_merge(text_pieces)
        
        return TextExtraction(
            merged_text=merged_text,
            paragraphs=text_pieces
        )
    
    def _smart_merge(self, pieces: List[str]) -> str:
        """
        智能合并文本片段
        
        规则：
        - 如果片段以句号/问号/感叹号结尾，加换行
        - 否则用空格连接
        """
        if not pieces:
            return ""
        
        result = []
        for i, piece in enumerate(pieces):
            result.append(piece)
            
            if i < len(pieces) - 1:
                # 判断是否应该换行
                if piece.endswith(("。", "？", "！", ".", "?", "!")):
                    result.append("\n")
                else:
                    result.append(" ")
        
        return "".join(result)


# ============================================================================
# Phase 3 总控制器
# ============================================================================

class Phase3_ParallelExtractor:
    """
    Phase 3 并行特征提取 - 总控制器
    
    根据元素类型分发任务到不同的专家，并行处理
    """
    
    def __init__(self, 
                 vlm_client,
                 textin_api_key: Optional[str] = None,
                 max_workers: int = 4):
        """
        初始化 Phase 3 控制器
        
        Args:
            vlm_client: VLM客户端
            textin_api_key: TextIn API密钥
            max_workers: 最大并行工作线程数
        """
        self.vlm_client = vlm_client
        self.max_workers = max_workers
        
        # 初始化各专家
        self.textin_client = TextInAPIClient(api_key=textin_api_key)
        self.table_expert = TableExpertAgent(self.textin_client, vlm_client)
        self.chart_expert = ChartExpertAgent(vlm_client)
        self.mixed_expert = MixedExpertAgent(vlm_client, self.table_expert, self.chart_expert)
        self.formula_expert = FormulaExpertAgent(self.textin_client)
        self.text_processor = TextProcessorAgent()

    @staticmethod
    def _is_mixed_candidate(elem: DetectedElement, group: SemanticGroup, context_text: str) -> bool:
        """启发式判断是否应走 mixed worker。"""
        raw_type = (elem.refined_type or elem.original_type or "").lower()
        group_type = (group.group_type or "").lower()
        semantic_desc = (group.semantic_desc or "").lower()
        context_l = (context_text or "").lower()

        if "mixed" in raw_type or "mixed" in group_type:
            return True

        multi_keywords = [
            "左图", "右图", "上图", "下图", "并列", "对比", "comparison",
            "table", "表格", "双图", "多图", "multi-panel", "composite"
        ]
        if any(k in semantic_desc for k in multi_keywords) or any(k in context_l for k in multi_keywords):
            return True

        return False
    
    def run(self,
           validated_semantic: ValidatedSemanticJSON,
           roi_images: List[ROIImage],
            all_elements: List[DetectedElement],
            page_title: str = "") -> ExtractedDataMap:
        """
        执行 Phase 3 完整流程
        
        Args:
            validated_semantic: Phase 2 的输出
            roi_images: Phase 1 裁剪的ROI图片
            all_elements: 所有检测到的元素
            
        Returns:
            ExtractedDataMap: 提取数据映射
        """
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"[Phase 3] 开始并行特征提取 (Page {validated_semantic.page_id})")
        print(f"{'='*60}")
        
        # 创建ROI路径映射
        roi_map = {r.element_id: r.roi_path for r in roi_images}
        elem_map = {e.element_id: e for e in all_elements}
        
        # 构建组上下文映射（用于图表分析）
        group_context_map = self._build_group_context(
            validated_semantic.groups,
            validated_semantic.id_to_text_map,
            elem_map,
            page_title=page_title,
        )
        
        # 准备任务
        tasks = self._prepare_tasks(
            validated_semantic,
            roi_map,
            elem_map,
            group_context_map
        )
        
        print(f"[Phase 3] 准备了 {len(tasks)} 个提取任务")
        
        # 并行执行
        extractions = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(task["func"], **task["kwargs"]): task["element_id"]
                for task in tasks
            }
            
            for future in as_completed(futures):
                element_id = futures[future]
                try:
                    result = future.result()
                    extractions[element_id] = result
                    print(f"  ✓ {element_id}: 提取完成")
                except Exception as e:
                    print(f"  ✗ {element_id}: 提取失败 - {e}")
                    extractions[element_id] = ExtractedContent(
                        element_id=element_id,
                        element_type="unknown",
                        processing_status="failed",
                        error_message=str(e)
                    )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        result = ExtractedDataMap(
            page_id=validated_semantic.page_id,
            extractions=extractions,
            group_context_map=group_context_map
        )
        
        print(f"\n[Phase 3] 完成！耗时 {processing_time}ms")
        print(f"  - 成功: {sum(1 for e in extractions.values() if e.processing_status == 'success')}")
        print(f"  - 失败: {sum(1 for e in extractions.values() if e.processing_status == 'failed')}")
        
        return result
    
    def _build_group_context(self,
                            groups: List[SemanticGroup],
                            id_to_text_map: Dict[str, str],
                            elem_map: Dict[str, DetectedElement],
                            page_title: str = "") -> Dict[str, str]:
        """
        构建组上下文映射
        
        对于每个组，提取其中的文本元素作为上下文
        """
        group_context = {}
        
        for group in groups:
            # 获取组内的文本内容
            text_parts = []
            
            # 优先使用从元素（通常是标题）
            for sec_id in group.secondary_element_ids:
                text = id_to_text_map.get(sec_id, "")
                if text.strip():
                    text_parts.append(text.strip())
            
            # 如果没有从元素文本，尝试主元素的OCR
            if not text_parts and group.primary_element_id:
                text = id_to_text_map.get(group.primary_element_id, "")
                if text.strip():
                    text_parts.append(text.strip())
            
            context = " | ".join(text_parts) if text_parts else ""
            if not context and page_title:
                # 单图/单表页面通常没有局部标题，使用页面标题作为上下文兜底
                context = page_title.strip()
            group_context[group.group_id] = context
        
        return group_context
    
    def _prepare_tasks(self,
                      validated_semantic: ValidatedSemanticJSON,
                      roi_map: Dict[str, str],
                      elem_map: Dict[str, DetectedElement],
                      group_context_map: Dict[str, str]) -> List[Dict]:
        """
        准备提取任务
        
        根据元素类型分配到不同的专家
        """
        tasks_by_element: Dict[str, Dict[str, Any]] = {}
        task_priority: Dict[str, int] = {}

        def add_task(task: Dict[str, Any], priority: int) -> None:
            element_id = task.get("element_id")
            if not element_id:
                return
            prev = task_priority.get(element_id, -1)
            if priority >= prev:
                tasks_by_element[element_id] = task
                task_priority[element_id] = priority
        
        for group in validated_semantic.groups:
            # 策略调整：遍历组内所有成员，寻找可提取的元素（图表、表格、图片、公式）
            # 而不仅仅是处理 primary_element_id
            # 这解决了单个分组包含多个图表时，只有主图表被提取的问题
            
            # 1. 特殊处理：纯文本组作为整体处理
            # 如果主元素是文本类型，且组被认为是文本组，则聚合处理
            primary_id = group.primary_element_id
            is_text_group = False
            
            if primary_id and primary_id in elem_map:
                primary_elem = elem_map[primary_id]
                primary_type = normalize_element_type(primary_elem.refined_type or primary_elem.original_type)
                
                if primary_type == "text":
                    # 获取组内所有元素
                    group_elements = [elem_map[mid] for mid in group.member_ids if mid in elem_map]
                    add_task({
                        "element_id": primary_id,
                        "func": self._process_text,
                        "kwargs": {
                            "elements": group_elements,
                            "group": group,
                            "id_to_text_map": validated_semantic.id_to_text_map,
                            "element_id": primary_id
                        }
                    }, priority=10)
                    is_text_group = True
            
            if is_text_group:
                continue

            # 2. 对于混合组/图表组，遍历成员逐个提取视觉/结构化内容
            context_text = group_context_map.get(group.group_id, "")
            group_type_l = (group.group_type or "").lower()

            group_mixed_hint = ("mixed" in group_type_l)
            
            for member_id in group.member_ids:
                if member_id not in elem_map:
                    continue
                    
                elem = elem_map[member_id]
                raw_elem_type = elem.refined_type or elem.original_type
                elem_type = normalize_element_type(raw_elem_type)
                
                # 跳过纯文本（文本通常作为上下文或在Phase4中作为OCR文本补充）
                if is_text_type(raw_elem_type):
                    continue

                use_mixed_worker = False
                if member_id in roi_map:
                    # [策略优化] 只对真正有"混合"嫌疑的元素使用 MixedExpert。
                    # 如果 MinerU 已经明确判定它是一个独立的 TABLE 或 CHART（通过其 bbox 推断），
                    # 我们直接将其发往更专业的单体专家（表格自带兜底回退算法，图表自带精确提取），
                    # 避免使用 VLM 视觉大模型进行无约束提取导致严重幻觉。
                    if elem_type == MIXED:
                        use_mixed_worker = True
                    elif elem_type == IMAGE and (group_mixed_hint or self._is_mixed_candidate(elem, group, context_text)):
                        use_mixed_worker = True
                    elif "mixed" in (raw_elem_type or "").lower():
                        use_mixed_worker = True

                if use_mixed_worker:
                    add_task({
                        "element_id": member_id,
                        "func": self._extract_mixed,
                        "kwargs": {
                            "roi_path": roi_map[member_id],
                            "context_text": context_text,
                            "element_id": member_id,
                        }
                    }, priority=60)
                    continue
                
                # 根据类型分配任务
                if elem_type == TABLE:
                    if member_id in roi_map:
                        add_task({
                            "element_id": member_id,
                            "func": self._extract_table,
                            "kwargs": {
                                "roi_path": roi_map[member_id],
                                "context_text": context_text,
                                "element_id": member_id
                            }
                        }, priority=50)
                
                elif elem_type == CHART:
                    if member_id in roi_map:
                        # 流程图/机制图等被归一为 chart，注入少量上下文增强提示
                        chart_context = context_text
                        raw_type_norm = (raw_elem_type or "").lower()
                        if raw_type_norm in ["flowchart", "diagram"] and context_text:
                            chart_context = f"流程图: {context_text}"

                        add_task({
                            "element_id": member_id,
                            "func": self._extract_chart,
                            "kwargs": {
                                "roi_path": roi_map[member_id],
                                "context_text": chart_context,
                                "element_id": member_id
                            }
                        }, priority=45)
                
                elif elem_type == FORMULA:
                    if member_id in roi_map:
                        add_task({
                            "element_id": member_id,
                            "func": self._extract_formula,
                            "kwargs": {
                                "roi_path": roi_map[member_id],
                                "element_id": member_id
                            }
                        }, priority=40)
                
                else:
                    # 其他类型：尝试作为图片处理（如图表专家通用处理）
                    # 只有当在ROI map中时才处理
                    if member_id in roi_map:
                        add_task({
                            "element_id": member_id,
                            "func": self._extract_image,
                            "kwargs": {
                                "roi_path": roi_map[member_id],
                                "context_text": context_text,
                                "element_id": member_id
                            }
                        }, priority=30)
        
        return list(tasks_by_element.values())
    
    def _extract_table(self, roi_path: str, context_text: str, element_id: str) -> ExtractedContent:
        if not os.path.exists(roi_path):
            return ExtractedContent(element_id=element_id, element_type="table", processing_status="failed", error_message="ROI file not found")
        table_data = self.table_expert.extract(roi_path, context_text)
        return ExtractedContent(element_id=element_id, element_type="table", table_data=table_data, processing_status="success")
    
    def _extract_chart(self, roi_path: str, context_text: str, element_id: str) -> ExtractedContent:
        """提取图表 (带文件检查)"""
        # 检查文件是否存在
        if not os.path.exists(roi_path):
            print(f"[Phase 3] ⚠️ 文件缺失，跳过图表提取: {roi_path}")
            return ExtractedContent(
                element_id=element_id,
                element_type="chart",
                processing_status="failed",
                error_message="ROI file not found"
            )

        chart_data = self.chart_expert.extract(roi_path, context_text)
        return ExtractedContent(
            element_id=element_id,
            element_type="chart",
            chart_data=chart_data,
            processing_status="success"
        )

    def _extract_mixed(self, roi_path: str, context_text: str, element_id: str) -> ExtractedContent:
        """提取混合元素（图+表/多子图）"""
        if not os.path.exists(roi_path):
            return ExtractedContent(
                element_id=element_id,
                element_type="mixed",
                processing_status="failed",
                error_message="ROI file not found"
            )

        mixed_data = self.mixed_expert.extract(roi_path, context_text)
        return ExtractedContent(
            element_id=element_id,
            element_type="mixed",
            mixed_data=mixed_data,
            processing_status="success"
        )
    
    def _extract_formula(self, roi_path: str, element_id: str) -> ExtractedContent:
        """提取公式"""
        formula_data = self.formula_expert.extract(roi_path)
        return ExtractedContent(
            element_id=element_id,
            element_type="formula",
            formula_data=formula_data,
            processing_status="success"
        )
    
    def _process_text(self, elements: List[DetectedElement], group: SemanticGroup, 
                     id_to_text_map: Dict[str, str], element_id: str) -> ExtractedContent:
        """处理文本"""
        text_data = self.text_processor.process(elements, group, id_to_text_map)
        return ExtractedContent(
            element_id=element_id,
            element_type="text",
            text_data=text_data,
            processing_status="success"
        )
    
    def _extract_image(self, roi_path: str, context_text: str, element_id: str) -> ExtractedContent:
        """提取图片（作为图表处理）"""
        chart_data = self.chart_expert.extract(roi_path, context_text)
        return ExtractedContent(
            element_id=element_id,
            element_type="image",
            chart_data=chart_data,
            processing_status="success"
        )
