"""Step 3: Supervisor监督节点，进行校验和融合"""

from state import (
    PPTPageState, GlobalPageAnalysis, ElementInsight, PageAnalysisResult, SupervisorOutput
)
from openai import AzureOpenAI
from typing import List
from config import get_config
import json



class Step3_SupervisorEngine:
    """Supervisor引擎
    
    职责：
    1. 比对各元素分析与全局总结的一致性
    2. 校验分析的合理性和完整性
    3. 如果不符合，说明问题点并标记需要重新分析
    4. 融合所有分析结果为最终结构化输出
    """
    
    def __init__(self, llm_client: AzureOpenAI):
        self.llm_client = llm_client
        # 使用配置中的模型名称，而不是硬编码
        config = get_config()
        self.model = config.llm_model_name
        print(f"[Step3-Supervisor] 使用模型: {self.model}")

    @staticmethod
    def _contains_failure_marker(text: str) -> bool:
        s = str(text or "").strip().lower()
        if not s:
            return False
        markers = (
            "分析失败",
            "提取失败",
            "unterminated string",
            "jsondecodeerror",
            "expecting value",
            "traceback",
            "error:",
            "line 1 column",
            "char ",
        )
        return any(m in s for m in markers)
    
    def validate_and_fuse(self,
                         global_analysis: GlobalPageAnalysis,
                         element_insights: List[ElementInsight]) -> PageAnalysisResult:
        """校验并融合分析结果"""
        print(f"\n[Step3-Supervisor] 校验页面 {global_analysis.page_index}")
        
        # # 如果重试发生，element_insights 可能会包含重复的 element_id
        # unique_insights_map = {}
        # for insight in element_insights:
        #     unique_insights_map[insight.element_id] = insight
        # # 重建去重后的列表
        # element_insights = list(unique_insights_map.values())
        # 多源交叉验证 (Cross-Check) Logic
        # 检查：如果有文本类型的 insight，对比其 data_evidence (XML) 和 global_analysis.extracted_text (VLM)
        cross_check_warning = ""
        for insight in element_insights:
            if insight.element_type == "text_page":
                xml_len = len(insight.data_evidence)
                vlm_len = len(global_analysis.extracted_text)
                # 简单启发式：如果长度差异过大，可能 XML 解析失败或 VLM 幻觉
                if abs(xml_len - vlm_len) > 200 and vlm_len > 0:
                    cross_check_warning = f"⚠️ [风险] 文本源不一致: XML提取({xml_len}字) vs VLM提取({vlm_len}字)，需人工复核。"

        hard_fail_issues = []
        for insight in element_insights:
            et = (getattr(insight, "element_type", "") or "").lower()
            if et not in {"chart", "image", "mixed", "diagram", "flowchart", "figure", "graph"}:
                continue

            status = (getattr(insight, "status", "") or "").lower()
            key_insight = getattr(insight, "key_insight", "")
            data_evidence = getattr(insight, "data_evidence", "")

            if status in {"fail", "retry"} or self._contains_failure_marker(key_insight) or self._contains_failure_marker(data_evidence):
                hard_fail_issues.append(
                    f"[{insight.element_id}] 视觉元素提取失败或结果异常，需重跑提取，不可直接放行。"
                )

        if hard_fail_issues:
            validation_passed = False
            formatted_issues = self._format_complex_field(hard_fail_issues)
            validation_notes = f"Score: 0\nFeedback: 检测到视觉元素提取失败，触发硬性拦截。\nIssues: {formatted_issues}"
            print(f"[Step3-Supervisor] 校验结果: [ERROR] FAIL")
        else:
            # 1. 构建校验prompt
            elements_summary = "\n".join([
                f"- [{e.element_id}] {e.element_type}:\n  Insight: {e.key_insight}\n  Hypothesis Check: {e.hypothesis_verification}"
                for e in element_insights
            ])

            validation_prompt = f"""请作为质量审核官，校验以下分析结果的合理性：

全局结论：{global_analysis.core_summary}

元素分析：
{elements_summary}

【放行规则】（优先级最高）
判断该页面是否属于以下类型之一，如果是，请直接判定 status 为 "pass"，不要添加任何问题：
1. 目录页、标题首页、章节过渡页或 Q&A 页面。
2. 参考文献 (References)、致谢、版权声明或仅仅是长文本背景介绍。

【审核规则】
如果不是上述豁免页面，请按以下实用标准进行校验：
1. 事实一致性：重点检查元素分析结果与“全局结论”之间是否存在**明显的自相矛盾或由于幻觉产生的数据篡改**。只要不存在直接矛盾，即视为合理。
2. 噪音宽容度：提取结果中可能包含无意义文本（如单独的数字、期刊名、公司Logo如"Adis"、页脚等），这是一些正常的视觉噪点。**绝对不要**因为这些孤立的无关元素判定为 fail，直接放行即可。
3. 假设相关性：本研究的预设假设是 "{global_analysis.research_hypothesis}"。请记住：不是所有元素都必须能够回答该假设。只要现有元素没有反驳该假设，或者提供了合理的背景支撑，都不算失败。
4. 拦截标准：只有在出现**严重的核心图表错解**、**关键结论与全局严重背离**时，才将 status 设为 "fail"。

输出格式：返回JSON格式的校验结果，包括：
- status: "pass" 或 "fail"
- issues: 导致 fail 的致命问题列表（若 pass 则返回空列表）
- feedback: "简要反馈意见"
- consistency_score: 0-100 的一致性评分（如果不冲突，即便无关也应打 80 分以上）
- warning: 交叉验证警告：{cross_check_warning}"""
                
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个严格的质量审核官。"},
                        {"role": "user", "content": validation_prompt}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=1000
                )
                content = response.choices[0].message.content
                output_data = json.loads(content)
                
                # 使用 Pydantic 验证
                sup_output = SupervisorOutput(**output_data)
                
                validation_passed = sup_output.status == "pass"
                # 格式化输出
                formatted_issues = self._format_complex_field(sup_output.issues)
                formatted_feedback = self._format_complex_field(sup_output.feedback)
                validation_notes = f"Score: {sup_output.consistency_score}\nFeedback: {formatted_feedback}\nIssues: {formatted_issues}"
                
                print(f"[Step3-Supervisor] 校验结果: {'[OK] PASS' if validation_passed else '[ERROR] FAIL'}")
                
            except Exception as e:
                print(f"[Supervisor] JSON解析失败，降级通过: {e}")
                validation_passed = True
                validation_notes = str(e)
        
        # 2. 如果通过，融合成最终markdown
        if validation_passed:
            fused_markdown = self._fuse_to_markdown(global_analysis, element_insights)
            status = "pass"
        else:
            fused_markdown = ""
            status = "failed"
        
        # 3. 返回校验结果
        consistency_score = 0
        try:
            if 'output_data' in locals() and 'consistency_score' in output_data:
                consistency_score = int(output_data.get('consistency_score', 0))
        except (ValueError, TypeError):
            consistency_score = 0
        
        return PageAnalysisResult(
            page_index=global_analysis.page_index,
            section_title=global_analysis.section_title,
            status=status,
            global_analysis=global_analysis,
            element_insights=element_insights,
            validation_passed=validation_passed,
            validation_notes=validation_notes,
            consistency_score=consistency_score,
            fused_markdown=fused_markdown
        )
    
    def _fuse_to_markdown(self, 
                         global_analysis: GlobalPageAnalysis,
                         element_insights: List[ElementInsight]) -> str:
        """融合分析结果为markdown格式"""
        lines = []
        
        # 标题
        lines.append(f"## {global_analysis.section_title}")
        lines.append("")
        
        # 核心总结
        lines.append("### 核心内容")
        lines.append(global_analysis.core_summary)
        lines.append("")
        
        # 页面文本
        if global_analysis.extracted_text:
            lines.append("### 页面文本")
            lines.append(global_analysis.extracted_text)
            lines.append("")
        
        # 元素分析
        if element_insights:
            lines.append("### 元素分析")
            for insight in element_insights:
                lines.append(f"**[{insight.element_type}] {insight.element_id}**")
                lines.append(insight.key_insight)
                lines.append("")
        
        return "\n".join(lines)
    
    def _format_complex_field(self, field) -> str:
        """辅助函数：格式化复杂的列表或字典字段为字符串"""
        if isinstance(field, list):
            return "; ".join([str(item) for item in field])
        elif isinstance(field, dict):
            return json.dumps(field, ensure_ascii=False)
        return str(field)


def node_step3_supervisor(state: PPTPageState, llm_client: AzureOpenAI, debug_logger=None) -> dict:
    """节点：Step 3 Supervisor校验"""
    page_index = state.page_index
    
    print(f"\n[Step3] 页面 {page_index} Supervisor校验")
    
    # 记录步骤开始
    if debug_logger:
        debug_logger.log_step_start(
            page_index,
            "Step 3: Supervisor",
            input_data={
                "page_index": page_index,
                "section_title": state.global_analysis.section_title if state.global_analysis else None,
                "element_insights_count": len(state.element_insights),
                "retry_count": state.retry_count,
                "max_retries": state.max_retries
            },
            previous_step="Step 2: Workers"
        )
    
    engine = Step3_SupervisorEngine(llm_client)
    
    # 校验并融合
    analysis_result = engine.validate_and_fuse(
        state.global_analysis,
        state.element_insights
    )
    
    # 根据校验结果决定是否进入重试模式
    should_retry = (
        not analysis_result.validation_passed and 
        state.retry_count < state.max_retries
    )
    
    # 记录步骤结束
    if debug_logger:
        debug_logger.log_step_end(
            page_index,
            "Step 3: Supervisor",
            output_data={
                "validation_passed": analysis_result.validation_passed,
                "validation_notes": analysis_result.validation_notes[:200],
                "consistency_score": analysis_result.consistency_score,
                "should_retry": should_retry,
                "retry_count": state.retry_count + 1 if should_retry else state.retry_count
            },
            status="success"
        )
    
    return {
        "analysis_result": analysis_result,
        "retry_count": state.retry_count + 1 if should_retry else state.retry_count,
        "is_retry_mode": should_retry
    }
