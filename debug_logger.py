"""
debug_logger.py
调试日志记录模块 - 记录每一步的详细信息供检查
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List
import logging


class DebugLogger:
    """调试日志记录器 - 记录工作流每一步的详细信息"""
    
    def __init__(self, output_dir: str = "debug_logs", run_id: str = "default"):
        """
        初始化调试日志记录器
        
        参数:
            output_dir: 日志输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id or "default")
        
        # 创建主日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.log_file = self.output_dir / f"workflow_debug_{timestamp}.md"
        self.json_log_file = self.output_dir / f"workflow_debug_{timestamp}.json"
        
        # 日志数据结构
        self.log_data = {
            "timestamp": timestamp,
            "run_id": self.run_id,
            "pages": {},  # page_index -> page_logs
            "summary": {}
        }
        
        # 初始化markdown文档
        self._init_markdown()
        
        # Python logging配置
        self.logger = self._setup_python_logger(timestamp)
        
    def _init_markdown(self):
        """初始化markdown文档"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("# PPT解析工作流 - 调试日志\n\n")
            f.write(f"**Run ID**: {self.run_id}\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
    
    def _setup_python_logger(self, timestamp: str):
        """设置Python标准logging"""
        log_file_path = self.output_dir / f"workflow_{timestamp}.log"
        
        logger = logging.getLogger(f"ppt_workflow.{self.run_id}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # 避免重复挂载 handler 导致重复日志
        if logger.handlers:
            for h in list(logger.handlers):
                logger.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass
        
        # 文件处理器
        fh = logging.FileHandler(str(log_file_path), encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        # 格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - [run_id=%(run_id)s] - %(message)s'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logging.LoggerAdapter(logger, {"run_id": self.run_id})
    
    def log_step_start(self, page_index: int, step_name: str, input_data: Dict[str, Any] = None, previous_step: str = None):
        """记录步骤开始
        
        参数:
            page_index: 页面索引
            step_name: 当前步骤名称
            input_data: 输入数据
            previous_step: 前驱步骤名称（这个步骤的输入来自哪个节点）
        """
        if page_index not in self.log_data["pages"]:
            self.log_data["pages"][page_index] = {
                "page_index": page_index,
                "steps": {}
            }
        
        step_key = f"{step_name}"
        self.log_data["pages"][page_index]["steps"][step_key] = {
            "name": step_name,
            "start_time": datetime.now().isoformat(),
            "input": self._serialize_data(input_data),
            "output": None,
            "duration": None,
            "status": "running",
            "previous_step": previous_step,
            "input_summary": self._get_data_summary(input_data)
        }
        
        # 追加到markdown
        self._append_markdown(f"## 📄 页面 {page_index}\n\n")
        self._append_markdown(f"### 节点: {step_name}\n\n")
        self._append_markdown(f"**开始时间**: {datetime.now().strftime('%H:%M:%S')}\n\n")
        
        # 记录前驱节点
        if previous_step:
            self._append_markdown(f"**前驱节点**: {previous_step}\n\n")
        
        if input_data:
            self._append_markdown(f"**输入数据**:\n```json\n{json.dumps(self._serialize_data(input_data), ensure_ascii=False, indent=2)}\n```\n\n")
        
        self.logger.info(f"[Page {page_index}] {step_name} - START (previous: {previous_step})")
    
    def log_step_end(self, page_index: int, step_name: str, output_data: Dict[str, Any] = None, status: str = "success"):
        """记录步骤结束"""
        if page_index not in self.log_data["pages"]:
            return
        
        step_key = f"{step_name}"
        if step_key in self.log_data["pages"][page_index]["steps"]:
            step_data = self.log_data["pages"][page_index]["steps"][step_key]
            step_data["output"] = self._serialize_data(output_data)
            step_data["status"] = status
            step_data["output_summary"] = self._get_data_summary(output_data)
            step_data["end_time"] = datetime.now().isoformat()
            
            # 计算耗时
            start_time = datetime.fromisoformat(step_data["start_time"])
            end_time = datetime.fromisoformat(step_data["end_time"])
            duration = (end_time - start_time).total_seconds()
            step_data["duration"] = f"{duration:.2f}s"
            
            # 追加到markdown - 更详细的格式
            self._append_markdown(f"**状态**: {status}\n")
            self._append_markdown(f"**耗时**: {duration:.2f}秒\n\n")
            
            if output_data:
                self._append_markdown(f"**输出数据**:\n```json\n{json.dumps(self._serialize_data(output_data), ensure_ascii=False, indent=2)}\n```\n\n")
            else:
                self._append_markdown(f"**输出数据**: (empty)\n\n")
            
            self._append_markdown("---\n\n")
        
        self.logger.info(f"[Page {page_index}] {step_name} - END ({status}) - {duration:.2f}s")
    
    def log_element_processing(self, page_index: int, element_id: str, step_name: str, details: Dict[str, Any]):
        """记录单个元素的处理"""
        if page_index not in self.log_data["pages"]:
            self.log_data["pages"][page_index] = {"page_index": page_index, "steps": {}}
        
        self._append_markdown(f"#### 元素处理: {element_id}\n\n")
        self._append_markdown(f"**步骤**: {step_name}\n\n")
        self._append_markdown(f"**详情**:\n```json\n{json.dumps(self._serialize_data(details), ensure_ascii=False, indent=2)}\n```\n\n")
        
        self.logger.debug(f"[Page {page_index}] Element {element_id} - {step_name}")
    
    def log_decision(self, page_index: int, decision_point: str, decision: str, reasoning: str = ""):
        """记录决策点"""
        self._append_markdown(f"#### 决策: {decision_point}\n\n")
        self._append_markdown(f"**决策**: {decision}\n\n")
        if reasoning:
            self._append_markdown(f"**理由**: {reasoning}\n\n")
        
        self.logger.info(f"[Page {page_index}] Decision - {decision_point}: {decision}")
    
    def log_error(self, page_index: int, step_name: str, error: Exception, context: Dict[str, Any] = None):
        """记录错误"""
        self._append_markdown(f"### ❌ 错误: {step_name}\n\n")
        self._append_markdown(f"**错误信息**: {str(error)}\n\n")
        if context:
            self._append_markdown(f"**上下文**:\n```json\n{json.dumps(self._serialize_data(context), ensure_ascii=False, indent=2)}\n```\n\n")
        
        self.logger.error(f"[Page {page_index}] {step_name} - ERROR: {str(error)}", exc_info=True)
    
    def log_workflow_summary(self, summary: Dict[str, Any]):
        """记录工作流总结"""
        self._append_markdown("# 🎯 工作流总结\n\n")
        self._append_markdown(f"**完成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        self._append_markdown(f"**总结信息**:\n```json\n{json.dumps(self._serialize_data(summary), ensure_ascii=False, indent=2)}\n```\n\n")
        
        self.log_data["summary"] = summary
        
        self.logger.info(f"Workflow Summary: {summary}")
    
    def _append_markdown(self, content: str):
        """追加内容到markdown文件"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(content)
    
    def _serialize_data(self, data: Any) -> Any:
        """序列化数据为可JSON化的格式"""
        if data is None:
            return None
        
        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                result[k] = self._serialize_data(v)
            return result
        
        if isinstance(data, (list, tuple)):
            return [self._serialize_data(item) for item in data]
        
        # 处理Pydantic模型
        if hasattr(data, 'dict'):
            return self._serialize_data(data.dict())
        
        if hasattr(data, '__dict__'):
            try:
                return self._serialize_data(data.__dict__)
            except:
                return str(data)
        
        # 基础类型
        if isinstance(data, (str, int, float, bool)):
            return data
        
        return str(data)
    
    def _get_data_summary(self, data: Any) -> str:
        """获取数据的简洁摘要"""
        if data is None:
            return "None"
        
        if isinstance(data, dict):
            keys = list(data.keys())
            if len(keys) <= 3:
                return f"dict with keys: {', '.join(keys)}"
            else:
                return f"dict with {len(keys)} keys: {', '.join(keys[:3])}..."
        
        if isinstance(data, (list, tuple)):
            return f"{type(data).__name__} with {len(data)} items"
        
        if hasattr(data, '__class__'):
            class_name = data.__class__.__name__
            if hasattr(data, '__dict__'):
                fields = list(data.__dict__.keys())
                if len(fields) <= 3:
                    return f"{class_name}({', '.join(fields)})"
                else:
                    return f"{class_name} with {len(fields)} fields"
            return class_name
        
        return str(data)[:50]
    
    def save(self):
        """保存JSON格式的日志"""
        with open(self.json_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.log_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"日志已保存到: {self.log_file} 和 {self.json_log_file}")
        print(f"\n✅ [{self.run_id}] 调试日志已保存:")
        print(f"   Markdown: {self.log_file}")
        print(f"   JSON: {self.json_log_file}")
        
        return str(self.log_file), str(self.json_log_file)


def get_debug_logger(output_dir: str = "debug_logs", run_id: str = "default") -> DebugLogger:
    """兼容入口：返回新的 run 级 logger 实例（不复用全局单例）。"""
    return DebugLogger(output_dir=output_dir, run_id=run_id)


def reset_debug_logger():
    """兼容占位：已无全局 logger 可重置。"""
    return None
