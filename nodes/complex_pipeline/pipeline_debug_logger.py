"""
Pipeline 调试日志器

专门用于记录复杂页面处理 Pipeline 每一步的中间输出，便于调试

使用方式:
    logger = PipelineDebugLogger(output_dir="debug_output", page_id=3)
    logger.log_phase1(phase1_output, mineru_raw_result, vlm_refinement_details)
    logger.log_phase2_grouping(phase2_draft, som_image_path, vlm_prompt, vlm_response)
    logger.log_phase2_validation(validated_result, issues_found)
    logger.log_phase3(extractions, api_responses)
    logger.log_phase4(final_json, final_markdown)
    logger.generate_summary_report()
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class PhaseLogEntry:
    """单个阶段的日志条目"""
    phase_name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[int] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    intermediate_data: Dict[str, Any] = field(default_factory=dict)
    prompts: List[Dict[str, str]] = field(default_factory=list)  # VLM/LLM prompts
    responses: List[Dict[str, str]] = field(default_factory=list)  # VLM/LLM responses
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, success, failed


class PipelineDebugLogger:
    """
    Pipeline 调试日志器
    
    为每个 Phase 记录详细的输入、输出、中间结果、VLM/LLM prompts 和 responses
    """
    
    def __init__(self, 
                 output_dir: str = "debug_output",
                 page_id: int = 0,
                 verbose: bool = True,
                 save_images: bool = True):
        """
        初始化调试日志器
        
        Args:
            output_dir: 调试输出目录
            page_id: 页面ID
            verbose: 是否输出详细日志到控制台
            save_images: 是否保存中间图片
        """
        self.output_dir = output_dir
        self.page_id = page_id
        self.verbose = verbose
        self.save_images = save_images
        
        # 创建带时间戳的调试目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_dir = os.path.join(output_dir, f"debug_page_{page_id:03d}_{timestamp}")
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # 创建各阶段子目录
        self.phase_dirs = {}
        for phase in ["phase1", "phase2_grouping", "phase2_validation", "phase3", "phase4"]:
            phase_dir = os.path.join(self.debug_dir, phase)
            os.makedirs(phase_dir, exist_ok=True)
            self.phase_dirs[phase] = phase_dir
        
        # 日志条目
        self.phase_logs: Dict[str, PhaseLogEntry] = {}
        self.pipeline_start_time = time.time()
        
        self._print(f"\n{'='*70}")
        self._print(f"  Pipeline Debug Logger 初始化")
        self._print(f"  调试目录: {self.debug_dir}")
        self._print(f"  页面ID: {page_id}")
        self._print(f"{'='*70}")
    
    def _print(self, msg: str):
        """条件性打印"""
        if self.verbose:
            print(msg)
    
    def _save_json(self, data: Any, filename: str, phase: str):
        """保存JSON数据到文件"""
        filepath = os.path.join(self.phase_dirs.get(phase, self.debug_dir), filename)
        try:
            if hasattr(data, 'model_dump_json'):
                content = data.model_dump_json(indent=2)
            elif hasattr(data, 'model_dump'):
                content = json.dumps(data.model_dump(), indent=2, ensure_ascii=False)
            else:
                content = json.dumps(data, indent=2, ensure_ascii=False, default=str)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return filepath
        except Exception as e:
            self._print(f"  ⚠️ 保存JSON失败: {e}")
            return None
    
    def _save_text(self, text: str, filename: str, phase: str):
        """保存文本到文件"""
        filepath = os.path.join(self.phase_dirs.get(phase, self.debug_dir), filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            return filepath
        except Exception as e:
            self._print(f"  ⚠️ 保存文本失败: {e}")
            return None
    
    def start_phase(self, phase_name: str, input_data: Dict[str, Any] = None):
        """开始记录一个阶段"""
        self.phase_logs[phase_name] = PhaseLogEntry(
            phase_name=phase_name,
            input_data=input_data or {},
            status="running"
        )
        self._print(f"\n{'─'*60}")
        self._print(f"  📍 开始 {phase_name}")
        self._print(f"{'─'*60}")
        
        # 保存输入数据
        if input_data:
            self._save_json(input_data, "input.json", phase_name)
    
    def end_phase(self, phase_name: str, output_data: Any = None, status: str = "success"):
        """结束一个阶段的记录"""
        if phase_name not in self.phase_logs:
            return
        
        entry = self.phase_logs[phase_name]
        entry.end_time = time.time()
        entry.duration_ms = int((entry.end_time - entry.start_time) * 1000)
        entry.status = status
        
        if output_data:
            if hasattr(output_data, 'model_dump'):
                entry.output_data = output_data.model_dump()
            else:
                entry.output_data = output_data
            self._save_json(output_data, "output.json", phase_name)
        
        status_icon = "✅" if status == "success" else "❌"
        self._print(f"\n  {status_icon} {phase_name} 完成 (耗时: {entry.duration_ms}ms)")
    
    def log_prompt_response(self, phase_name: str, prompt: str, response: str, 
                           label: str = "VLM", save_to_file: bool = True):
        """记录 VLM/LLM 的 prompt 和 response"""
        if phase_name not in self.phase_logs:
            self.start_phase(phase_name)
        
        entry = self.phase_logs[phase_name]
        entry.prompts.append({"label": label, "content": prompt})
        entry.responses.append({"label": label, "content": response})
        
        if save_to_file:
            idx = len(entry.prompts)
            self._save_text(prompt, f"{label.lower()}_prompt_{idx:02d}.txt", phase_name)
            self._save_text(response, f"{label.lower()}_response_{idx:02d}.txt", phase_name)
        
        # 在控制台显示摘要
        prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
        response_preview = response[:300] + "..." if len(response) > 300 else response
        
        self._print(f"\n  📝 {label} Prompt (前200字符):")
        self._print(f"     {prompt_preview}")
        self._print(f"\n  📨 {label} Response (前300字符):")
        self._print(f"     {response_preview}")
    
    def log_intermediate(self, phase_name: str, key: str, data: Any, description: str = ""):
        """记录中间数据"""
        if phase_name not in self.phase_logs:
            self.start_phase(phase_name)
        
        entry = self.phase_logs[phase_name]
        
        if hasattr(data, 'model_dump'):
            entry.intermediate_data[key] = data.model_dump()
        else:
            entry.intermediate_data[key] = data
        
        # 保存到文件
        self._save_json(data, f"intermediate_{key}.json", phase_name)
        
        if description:
            self._print(f"  📦 {key}: {description}")
    
    def log_error(self, phase_name: str, error: str):
        """记录错误"""
        if phase_name not in self.phase_logs:
            self.start_phase(phase_name)
        
        self.phase_logs[phase_name].errors.append(error)
        self._print(f"  ❌ 错误: {error}")
    
    def log_warning(self, phase_name: str, warning: str):
        """记录警告"""
        if phase_name not in self.phase_logs:
            self.start_phase(phase_name)
        
        self.phase_logs[phase_name].warnings.append(warning)
        self._print(f"  ⚠️ 警告: {warning}")
    
    # ========================================================================
    # Phase 1 专用日志方法
    # ========================================================================
    
    def log_phase1_mineru_result(self, raw_elements: List[Dict], 
                                  mineru_output_dir: str = None,
                                  mineru_cmd: str = None,
                                  returncode: int = None):
        """记录 Phase 1 MinerU 原始结果"""
        phase = "phase1"
        
        self._print(f"\n  🔍 MinerU 检测结果:")
        self._print(f"     检测到 {len(raw_elements)} 个元素")
        
        if mineru_cmd:
            self._print(f"     命令: {mineru_cmd}")
        if returncode is not None:
            self._print(f"     返回码: {returncode}")
        if mineru_output_dir:
            self._print(f"     输出目录: {mineru_output_dir}")
        
        # 列出每个元素
        for i, elem in enumerate(raw_elements):
            elem_type = elem.get('type', 'unknown')
            bbox = elem.get('bbox', [])
            text_preview = elem.get('text', '')[:50] if elem.get('text') else ''
            self._print(f"     [{i}] {elem_type}: bbox={bbox}, text=\"{text_preview}...\"")
        
        self.log_intermediate(phase, "mineru_raw_elements", raw_elements, 
                             f"MinerU检测到 {len(raw_elements)} 个元素")
    
    def log_phase1_vlm_refinement(self, element_id: str, original_type: str,
                                   refined_type: str, prompt: str, response: str):
        """记录 Phase 1 VLM 类型清洗"""
        phase = "phase1"
        
        self._print(f"\n  🎯 VLM 类型清洗: {element_id}")
        self._print(f"     原始类型: {original_type}")
        self._print(f"     清洗后类型: {refined_type}")
        
        # 保存 prompt 和 response
        self._save_text(prompt, f"vlm_refine_{element_id}_prompt.txt", phase)
        self._save_text(response, f"vlm_refine_{element_id}_response.txt", phase)
    
    def log_phase1_roi_crops(self, roi_images: List[Any]):
        """记录 Phase 1 ROI 裁剪结果"""
        phase = "phase1"
        
        self._print(f"\n  🖼️ ROI 裁剪结果: {len(roi_images)} 个")
        for roi in roi_images:
            if hasattr(roi, 'element_id'):
                self._print(f"     - {roi.element_id}: {roi.roi_path}")
            elif isinstance(roi, dict):
                self._print(f"     - {roi.get('element_id')}: {roi.get('roi_path')}")
    
    # ========================================================================
    # Phase 2 专用日志方法
    # ========================================================================
    
    def log_phase2_som_image(self, som_image_path: str, elements_marked: int):
        """记录 Phase 2 SoM 图片生成"""
        phase = "phase2_grouping"
        
        self._print(f"\n  🏷️ SoM 图片生成:")
        self._print(f"     路径: {som_image_path}")
        self._print(f"     标记元素数: {elements_marked}")
        
        self.log_intermediate(phase, "som_image_path", som_image_path)
    
    def log_phase2_grouping_result(self, groups: List[Any], ungrouped: List[str]):
        """记录 Phase 2 分组结果"""
        phase = "phase2_grouping"
        
        self._print(f"\n  📦 语义分组结果:")
        self._print(f"     分组数: {len(groups)}")
        self._print(f"     未分组元素: {ungrouped}")
        
        for i, group in enumerate(groups):
            if hasattr(group, 'group_id'):
                self._print(f"     [{i}] {group.group_id}: "
                          f"类型={group.group_type}, "
                          f"成员={group.member_ids}")
            elif isinstance(group, dict):
                self._print(f"     [{i}] {group.get('group_id')}: "
                          f"类型={group.get('group_type')}, "
                          f"成员={group.get('member_ids')}")
    
    def log_phase2_validation(self, validation_result: Any, issues: List[Any]):
        """记录 Phase 2 验证结果"""
        phase = "phase2_validation"
        
        status = "pass"
        if hasattr(validation_result, 'status'):
            status = validation_result.status
        elif isinstance(validation_result, dict):
            status = validation_result.get('status', 'pass')
        
        self._print(f"\n  ✓ 验证结果: {status}")
        self._print(f"     发现问题数: {len(issues)}")
        
        for issue in issues:
            if hasattr(issue, 'issue_type'):
                self._print(f"     - {issue.issue_type}: {issue.description}")
            elif isinstance(issue, dict):
                self._print(f"     - {issue.get('issue_type')}: {issue.get('description')}")
    
    # ========================================================================
    # Phase 3 专用日志方法
    # ========================================================================
    
    def log_phase3_extraction(self, element_id: str, element_type: str,
                              extraction_result: Any, api_response: str = None):
        """记录 Phase 3 单个元素提取"""
        phase = "phase3"
        
        status = "success"
        if hasattr(extraction_result, 'processing_status'):
            status = extraction_result.processing_status
        
        status_icon = "✅" if status == "success" else "❌"
        self._print(f"\n  {status_icon} 提取 {element_id} ({element_type}):")
        
        # 根据类型显示不同信息
        if hasattr(extraction_result, 'table_data') and extraction_result.table_data:
            table = extraction_result.table_data
            preview = (getattr(table, 'insight', '') or '').strip()
            if not preview and getattr(table, 'key_values', None):
                kv = list((table.key_values or {}).items())[:2]
                preview = "，".join(f"{k}: {v}" for k, v in kv)
            if not preview:
                preview = "已提取表格结构"
            self._print(f"     表格: {preview[:50]}...")
            if table.markdown_table:
                self._print(f"     Markdown表格长度: {len(table.markdown_table)} 字符")
        elif hasattr(extraction_result, 'chart_data') and extraction_result.chart_data:
            chart = extraction_result.chart_data
            self._print(f"     图表类型: {chart.chart_type}")
            self._print(f"     洞察: {chart.insight[:100]}...")
        elif hasattr(extraction_result, 'formula_data') and extraction_result.formula_data:
            formula = extraction_result.formula_data
            self._print(f"     公式: {formula.latex[:80]}...")
        elif hasattr(extraction_result, 'text_data') and extraction_result.text_data:
            text = extraction_result.text_data
            self._print(f"     文本: {text.merged_text[:100]}...")
        
        # 保存 API response
        if api_response:
            self._save_text(api_response, f"api_response_{element_id}.txt", phase)
    
    # ========================================================================
    # Phase 4 专用日志方法
    # ========================================================================
    
    def log_phase4_assembly(self, semantic_blocks: List[Any], 
                            markdown_preview: str = None):
        """记录 Phase 4 组装结果"""
        phase = "phase4"
        
        self._print(f"\n  🧩 组装结果:")
        self._print(f"     语义块数: {len(semantic_blocks)}")
        
        for block in semantic_blocks:
            if hasattr(block, 'block_id'):
                self._print(f"     - {block.block_id}: {block.block_type} (order={block.reading_order})")
            elif isinstance(block, dict):
                self._print(f"     - {block.get('block_id')}: {block.get('block_type')}")
        
        if markdown_preview:
            self._print(f"\n  📄 Markdown 预览 (前500字符):")
            self._print(f"     {markdown_preview[:500]}...")
    
    # ========================================================================
    # 汇总报告
    # ========================================================================
    
    def generate_summary_report(self) -> str:
        """生成调试汇总报告"""
        total_time = int((time.time() - self.pipeline_start_time) * 1000)
        
        report_lines = [
            "=" * 70,
            f"  Pipeline 调试汇总报告",
            f"  页面ID: {self.page_id}",
            f"  调试目录: {self.debug_dir}",
            f"  总耗时: {total_time}ms",
            "=" * 70,
            ""
        ]
        
        # 各阶段摘要
        report_lines.append("📊 各阶段摘要:")
        report_lines.append("-" * 50)
        
        for phase_name, entry in self.phase_logs.items():
            status_icon = {"success": "✅", "failed": "❌", "running": "🔄", "pending": "⏳"}.get(entry.status, "❓")
            duration = f"{entry.duration_ms}ms" if entry.duration_ms else "N/A"
            
            report_lines.append(f"\n{status_icon} {phase_name}:")
            report_lines.append(f"   状态: {entry.status}")
            report_lines.append(f"   耗时: {duration}")
            
            if entry.prompts:
                report_lines.append(f"   VLM/LLM调用次数: {len(entry.prompts)}")
            
            if entry.errors:
                report_lines.append(f"   ❌ 错误: {len(entry.errors)} 个")
                for err in entry.errors:
                    report_lines.append(f"      - {err}")
            
            if entry.warnings:
                report_lines.append(f"   ⚠️ 警告: {len(entry.warnings)} 个")
                for warn in entry.warnings:
                    report_lines.append(f"      - {warn}")
        
        # 保存的文件列表
        report_lines.append("\n" + "-" * 50)
        report_lines.append("📁 生成的调试文件:")
        
        for phase_name, phase_dir in self.phase_dirs.items():
            files = os.listdir(phase_dir) if os.path.exists(phase_dir) else []
            if files:
                report_lines.append(f"\n  {phase_name}/")
                for f in sorted(files):
                    fpath = os.path.join(phase_dir, f)
                    fsize = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
                    report_lines.append(f"    - {f} ({fsize} bytes)")
        
        report_lines.append("\n" + "=" * 70)
        
        report = "\n".join(report_lines)
        
        # 保存报告
        report_path = os.path.join(self.debug_dir, "debug_summary.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 同时保存完整的日志条目为JSON
        full_log = {
            "page_id": self.page_id,
            "total_time_ms": total_time,
            "debug_dir": self.debug_dir,
            "phases": {}
        }
        for phase_name, entry in self.phase_logs.items():
            full_log["phases"][phase_name] = {
                "status": entry.status,
                "duration_ms": entry.duration_ms,
                "input_data": entry.input_data,
                "output_data": entry.output_data,
                "intermediate_data": entry.intermediate_data,
                "prompt_count": len(entry.prompts),
                "errors": entry.errors,
                "warnings": entry.warnings
            }
        
        full_log_path = os.path.join(self.debug_dir, "debug_full_log.json")
        with open(full_log_path, 'w', encoding='utf-8') as f:
            json.dump(full_log, f, indent=2, ensure_ascii=False, default=str)
        
        self._print(report)
        self._print(f"\n📝 调试报告已保存到: {report_path}")
        self._print(f"📝 完整日志已保存到: {full_log_path}")
        
        return report
    
    def get_debug_dir(self) -> str:
        """获取调试目录路径"""
        return self.debug_dir
