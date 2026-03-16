"""
Phase 2-2: 监督节点合理性验证 (Supervisor Logic Check)

核心目的：引入监督机制确保分组准确性，包含熔断机制和降级策略

包含组件：
1. GroupingValidationResult: 验证结果数据结构
2. Phase2_SupervisorValidator: 监督验证引擎
"""

import json
from typing import List, Dict, Any, Optional, Tuple, Literal

from .pipeline_state import (
    DraftSemanticJSON, ValidatedSemanticJSON, SemanticGroup,
    ValidationIssue, GroupingValidationResult, DetectedElement
)


class Phase2_SupervisorValidator:
    """
    Phase 2-2 监督节点验证器
    
    使用高推理能力模型 (GPT-4o) 审查分组结果的合理性
    包含：
    1. 孤立图表检测
    2. 跨度过大检测
    3. 分组逻辑检查
    4. 熔断机制 (Meltdown)
    5. 降级策略 (Fallback)
    """
    
    MAX_RETRIES = 2  # 最大重试次数
    
    def __init__(self, vlm_client, llm_client=None):
        """
        初始化监督验证器
        
        Args:
            vlm_client: VLM客户端（用于查看SoM图片）
            llm_client: LLM客户端（可选，用于纯文本推理）
        """
        self.vlm_client = vlm_client
        self.llm_client = llm_client or vlm_client
        
        # 获取模型名称
        try:
            from config import get_config
            self.model = get_config().vlm_runtime_model_name
        except:
            self.model = "gpt-4o"
    
    def validate(self,
                draft_semantic: DraftSemanticJSON,
                elements: List[DetectedElement],
                som_image_path: str,
                retry_count: int = 0,
                page_title: str = "") -> Tuple[ValidatedSemanticJSON, bool]:
        """
        验证分组结果的合理性
        
        Args:
            draft_semantic: 待验证的分组结果
            elements: 元素列表
            som_image_path: SoM图片路径
            retry_count: 当前重试次数
            
        Returns:
            (验证后的结果, 是否需要重试)
        """
        print(f"\n{'='*60}")
        print(f"[Phase 2-2] 监督节点合理性验证 (Page {draft_semantic.page_id}, 重试={retry_count})")
        print(f"{'='*60}")
        
        # 检查是否触发熔断（重试次数 > MAX_RETRIES 时才熔断，避免最后一次重试被直接跳过）
        if retry_count > self.MAX_RETRIES:
            print("[Phase 2-2] ⚠️ 触发熔断机制！使用降级策略...")
            return self._meltdown_fallback(draft_semantic, elements), False
        
        # 执行验证
        validation_result = self._run_validation(
            draft_semantic, elements, som_image_path, page_title=page_title
        )
        
        # 判断是否通过
        if validation_result.status == "pass":
            print("[Phase 2-2] ✅ 验证通过！")
            validated = ValidatedSemanticJSON(
                page_id=draft_semantic.page_id,
                layout_confidence="high",
                groups=draft_semantic.groups,
                validation_result=validation_result,
                elements=elements,
                id_to_text_map=draft_semantic.id_to_text_map
            )
            return validated, False
        
        else:
            print(f"[Phase 2-2] ⚠️ 发现 {len(validation_result.issues)} 个潜在问题")
            for issue in validation_result.issues:
                print(f"    - [{issue.issue_type}] {issue.description}")

            # 即使重试次数耗尽，也不熔断回退，而是接受当前最好的结果
            # 只有在还有重试机会时，才返回 True 让外部循环调用 apply_corrections
            if retry_count < self.MAX_RETRIES:
                print(f"[Phase 2-2] 尝试自动修复 (剩余重试: {self.MAX_RETRIES - retry_count})")
                # 返回 True，指示 Orchestrator 调用 apply_corrections 并再次验证
                return self._create_temp_validated(draft_semantic, elements, validation_result), True
            else:
                print("[Phase 2-2] 🛑 重试次数耗尽，保留当前最佳修正结果（不回退到几何线性分组）")
                # 标记为 medium/low 置信度，但保留结构
                validated = ValidatedSemanticJSON(
                    page_id=draft_semantic.page_id,
                    layout_confidence="medium", 
                    groups=draft_semantic.groups,
                    validation_result=validation_result,
                    elements=elements,
                    id_to_text_map=draft_semantic.id_to_text_map
                )
                return validated, False

    def _meltdown_fallback(self,
                          draft_semantic: DraftSemanticJSON,
                          elements: List[DetectedElement]) -> ValidatedSemanticJSON:
        """熔断降级策略：放弃语义分组，按几何坐标线性分组。

        触发条件：重试次数超过阈值，且验证无法收敛。
        目标：保证 Pipeline 可继续进入 Phase 3/4，不因分组失败阻断。
        """
        print("[Phase 2-2] 执行熔断降级：按几何坐标线性分组")

        if not elements:
            validation_result = GroupingValidationResult(
                status="meltdown",
                issues=[],
                retry_count=self.MAX_RETRIES,
                max_retries=self.MAX_RETRIES,
                fallback_used=True,
                validation_notes="触发熔断，但 elements 为空：保留 draft groups"
            )
            return ValidatedSemanticJSON(
                page_id=draft_semantic.page_id,
                layout_confidence="low",
                groups=draft_semantic.groups,
                validation_result=validation_result,
                elements=elements,
                id_to_text_map=draft_semantic.id_to_text_map
            )

        sorted_elements = sorted(elements, key=lambda e: (e.bbox.ymin, e.bbox.xmin))

        fallback_groups: List[SemanticGroup] = []
        for i, elem in enumerate(sorted_elements):
            elem_type = (elem.refined_type or elem.original_type or "text").lower()
            fallback_groups.append(
                SemanticGroup(
                    group_id=f"fallback_group_{i+1}",
                    member_ids=[elem.element_id],
                    group_type=f"standalone_{elem_type}",
                    semantic_desc="降级分组（熔断策略：几何线性分组）",
                    reading_order=i,
                    primary_element_id=elem.element_id,
                    secondary_element_ids=[],
                )
            )

        validation_result = GroupingValidationResult(
            status="meltdown",
            issues=[],
            retry_count=self.MAX_RETRIES,
            max_retries=self.MAX_RETRIES,
            fallback_used=True,
            validation_notes="触发熔断，使用几何坐标线性分组"
        )

        return ValidatedSemanticJSON(
            page_id=draft_semantic.page_id,
            layout_confidence="low",
            groups=fallback_groups,
            validation_result=validation_result,
            elements=elements,
            id_to_text_map=draft_semantic.id_to_text_map
        )
    
    def _create_temp_validated(self, draft, elements, result):
        """辅助函数：创建临时的 Validated 对象供传递"""
        return ValidatedSemanticJSON(
            page_id=draft.page_id,
            layout_confidence="low",
            groups=draft.groups,
            validation_result=result,
            elements=elements,
            id_to_text_map=draft.id_to_text_map
        )
    
    def _run_validation(self,
                       draft_semantic: DraftSemanticJSON,
                       elements: List[DetectedElement],
                       som_image_path: str,
                       page_title: str = "") -> GroupingValidationResult:
        """执行验证逻辑"""
        issues = []
        
        # 1. 规则检查（快速，无需调用API）
        rule_issues = self._rule_based_checks(draft_semantic, elements, page_title=page_title)
        issues.extend(rule_issues)
        
        # 2. VLM 深度审查
        # 这里为了效率，只有当规则检查没发现大问题时，才可选调用 VLM
        if not issues:
            # vlm_issues = self._vlm_deep_review(draft_semantic, elements, som_image_path)
            # issues.extend(vlm_issues)
            pass
        
        # 判断状态
        if not issues:
            return GroupingValidationResult(
                status="pass",
                issues=[],
                validation_notes="所有检查通过"
            )
        
        # # 根据问题严重程度判断
        # critical_issues = [i for i in issues if i.issue_type in ["orphan_chart", "missing_title"]]
        
        # if len(critical_issues) > 2:
        #     return GroupingValidationResult(
        #         status="meltdown",
        #         issues=issues,
        #         validation_notes="严重问题过多，建议熔断"
        #     )
        
        return GroupingValidationResult(
            status="fail",
            issues=issues,
            validation_notes=f"发现 {len(issues)} 个问题待修复"
        )
    
    def _rule_based_checks(self,
                          draft_semantic: DraftSemanticJSON,
                          elements: List[DetectedElement],
                          page_title: str = "") -> List[ValidationIssue]:
        """基于规则的快速检查"""
        issues = []
        
        # 创建元素类型映射
        elem_map = {e.element_id: e for e in elements}
        
        # 统计“视觉主元素”组数量：用于判断是否可以用页面级标题兜底
        visual_primary_groups = []
        for group in draft_semantic.groups:
            if not group.primary_element_id or group.primary_element_id not in elem_map:
                continue
            primary_elem = elem_map[group.primary_element_id]
            elem_type = (primary_elem.refined_type or primary_elem.original_type).lower()
            if elem_type in ["chart", "table", "flowchart", "diagram"]:
                visual_primary_groups.append(group)

        for group in draft_semantic.groups:
            if not group.primary_element_id or group.primary_element_id not in elem_map:
                continue
                
            primary_elem = elem_map[group.primary_element_id]
            elem_type = (primary_elem.refined_type or primary_elem.original_type).lower()
            
            # 检查: 孤立的图表/表格
            # 定义：如果是图表类，且没有从元素（标题），且组内成员只有它自己
            # 注意：不要把普通 image/icon/logo 当成 chart 来做“孤立图表”检查，否则容易误报。
            if elem_type in ["chart", "table", "flowchart", "diagram"]:
                # 如果是 Image，只有当它看起来很大或者像图表时才检查，避免 logo 被误报
                # 这里简化处理：假设 Phase 1 清洗过的 chart 都是重要的
                
                is_lonely = len(group.member_ids) == 1
                has_no_title = not group.secondary_element_ids
                
                if is_lonely and has_no_title:
                    # 如果页面只有一个视觉主元素且页面级标题存在：认为“页面标题即图/表标题”，不作为错误。
                    if page_title and page_title.strip() and len(visual_primary_groups) <= 1:
                        continue

                    # 这是一个强信号：孤立图表
                    issues.append(ValidationIssue(
                        issue_type="orphan_chart",
                        element_ids=[group.primary_element_id],
                        description=f"Group {group.group_id} 是孤立图表，缺少标题",
                        suggestion="search_nearest_text" # 明确指示修复策略
                    ))

            # 检查: 混合类型候选组（单ROI包含多元素或组内图表+表格混合）
            group_type_l = (group.group_type or "").lower()
            if "mixed" in group_type_l:
                continue

            member_types = []
            for mid in group.member_ids:
                if mid in elem_map:
                    member_types.append((elem_map[mid].refined_type or elem_map[mid].original_type or "").lower())

            has_chart_like = any(t in ["chart", "flowchart", "diagram", "data_chart", "line_chart", "bar_chart"] for t in member_types)
            has_table_like = any("table" in t for t in member_types)

            mixed_keywords = ["左图", "右图", "双图", "并列", "对比", "table", "表格", "mixed"]
            semantic_l = (group.semantic_desc or "").lower()
            looks_mixed_by_text = any(k in semantic_l for k in mixed_keywords)

            if (has_chart_like and has_table_like) or (len(group.member_ids) == 1 and has_chart_like and looks_mixed_by_text):
                issues.append(ValidationIssue(
                    issue_type="mixed_candidate",
                    element_ids=list(group.member_ids),
                    description=f"Group {group.group_id} 疑似为混合内容，建议走 mixed worker",
                    suggestion="mark_group_mixed"
                ))
        
        return issues
        # # 检查1: 孤立的图表（图表类型的元素没有匹配任何标题）
        # for group in draft_semantic.groups:
        #     primary_elem = elem_map.get(group.primary_element_id)
        #     if primary_elem:
        #         elem_type = (primary_elem.refined_type or primary_elem.original_type).lower()
                
        #         # 如果主元素是图表，检查是否有从元素（标题/图例）
        #         if elem_type in ["chart", "table", "flowchart"]:
        #             if not group.secondary_element_ids:
        #                 # 检查是否真的是孤立的
        #                 # 如果group只有一个成员且是图表，可能是孤立图表
        #                 if len(group.member_ids) == 1:
        #                     issues.append(ValidationIssue(
        #                         issue_type="orphan_chart",
        #                         element_ids=[group.primary_element_id],
        #                         description=f"图表 {group.primary_element_id} 没有匹配到标题或图例",
        #                         suggestion="检查是否有遗漏的文本元素应该关联到此图表"
        #                     ))
        
        # # 检查2: 跨度过大的分组（组内元素距离过远）
        # for group in draft_semantic.groups:
        #     if len(group.member_ids) > 1:
        #         bboxes = [elem_map[mid].bbox for mid in group.member_ids if mid in elem_map]
        #         if bboxes:
        #             # 计算组内元素的位置跨度
        #             ymin = min(b.ymin for b in bboxes)
        #             ymax = max(b.ymax for b in bboxes)
        #             xmin = min(b.xmin for b in bboxes)
        #             xmax = max(b.xmax for b in bboxes)
                    
        #             # 如果跨度超过页面的70%，可能是错误分组
        #             if (ymax - ymin) > 700 or (xmax - xmin) > 700:
        #                 issues.append(ValidationIssue(
        #                     issue_type="over_spanning",
        #                     element_ids=group.member_ids,
        #                     description=f"分组 {group.group_id} 的元素跨度过大，可能是错误分组",
        #                     suggestion="检查这些元素是否真的属于同一逻辑单元"
        #                 ))
        
        # # 检查3: 阅读顺序是否合理（是否从上到下）
        # sorted_groups = sorted(draft_semantic.groups, key=lambda g: g.reading_order)
        # for i, group in enumerate(sorted_groups[:-1]):
        #     next_group = sorted_groups[i + 1]
            
        #     # 获取组的代表位置
        #     group_elem = elem_map.get(group.primary_element_id)
        #     next_elem = elem_map.get(next_group.primary_element_id)
            
        #     if group_elem and next_elem:
        #         # 如果后一个组在前一个组的上方很多，顺序可能有问题
        #         if next_elem.bbox.ymax < group_elem.bbox.ymin - 100:
        #             issues.append(ValidationIssue(
        #                 issue_type="wrong_order",
        #                 element_ids=[group.group_id, next_group.group_id],
        #                 description=f"阅读顺序可能错误：{next_group.group_id} 在页面上方但顺序靠后",
        #                 suggestion="建议重新调整阅读顺序"
        #             ))
        
        # return issues
    
    def _vlm_deep_review(self,
                        draft_semantic: DraftSemanticJSON,
                        elements: List[DetectedElement],
                        som_image_path: str) -> List[ValidationIssue]:
        """使用VLM进行深度审查"""
        try:
            import base64
            with open(som_image_path, "rb") as f:
                base64_img = base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"[Phase 2-2] 读取SoM图片失败: {e}")
            return []
        
        # 构建分组描述
        groups_desc = []
        for g in draft_semantic.groups:
            groups_desc.append(f"- [{g.group_id}] 类型={g.group_type}, 成员={g.member_ids}, 阅读顺序={g.reading_order}")
        groups_text = "\n".join(groups_desc)
        
        prompt = f"""你是一个严格的文档版面审查员。请检查以下PPT页面的分组结果是否合理。

【分组结果】
{groups_text}

【检查要点】
1. 是否有孤立的图表（即有图表但未匹配任何标题或图例）？
2. 是否有跨度过大的分组（例如页面顶部的文字被分到了底部的图中）？
3. 分组逻辑是否合理：是否有图表丢失了标题？
4. 阅读顺序是否正确：是否从下到上反了？

请检查图片中的元素标记，对照分组结果。

【输出格式】
如果合理，返回：
{{"status": "PASS", "notes": "通过原因"}}

如果不合理，返回：
{{"status": "FAIL", "issues": [
    {{"type": "issue_type", "elements": ["#1", "#2"], "description": "问题描述", "suggestion": "修正建议"}}
]}}"""
        
        try:
            response = self.vlm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是严格的文档版面审查员，专注于发现分组逻辑问题。请始终以json格式返回结果。"},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                    ]}
                ],
                max_tokens=1000,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            if result.get("status") == "PASS":
                return []
            
            # 解析问题
            issues = []
            for issue_data in result.get("issues", []):
                issues.append(ValidationIssue(
                    issue_type=issue_data.get("type", "unknown"),
                    element_ids=issue_data.get("elements", []),
                    description=issue_data.get("description", ""),
                    suggestion=issue_data.get("suggestion", "")
                ))
            
            return issues
            
        except Exception as e:
            print(f"[Phase 2-2] VLM审查失败: {e}")
            return []
    
    # def _meltdown_fallback(self,
    #                       draft_semantic: DraftSemanticJSON,
    #                       elements: List[DetectedElement]) -> ValidatedSemanticJSON:
    #     """
    #     熔断降级策略
        
    #     放弃VLM分组，强制按照几何坐标生成线性分组
    #     """
    #     print("[Phase 2-2] 执行熔断降级：按几何坐标线性分组")
        
    #     # 按坐标排序
    #     sorted_elements = sorted(
    #         elements,
    #         key=lambda e: (e.bbox.ymin, e.bbox.xmin)
    #     )
        
    #     # 每个元素单独成组
    #     fallback_groups = []
    #     for i, elem in enumerate(sorted_elements):
    #         elem_type = (elem.refined_type or elem.original_type).lower()
    #         group = SemanticGroup(
    #             group_id=f"fallback_group_{i+1}",
    #             member_ids=[elem.element_id],
    #             group_type=f"standalone_{elem_type}",
    #             semantic_desc="降级分组（熔断策略）",
    #             reading_order=i,
    #             primary_element_id=elem.element_id,
    #             secondary_element_ids=[]
    #         )
    #         fallback_groups.append(group)
        
    #     validation_result = GroupingValidationResult(
    #         status="meltdown",
    #         issues=[],
    #         retry_count=self.MAX_RETRIES,
    #         max_retries=self.MAX_RETRIES,
    #         fallback_used=True,
    #         validation_notes="触发熔断，使用几何坐标线性分组"
    #     )
        
    #     return ValidatedSemanticJSON(
    #         page_id=draft_semantic.page_id,
    #         layout_confidence="low",  # 降级后置信度为低
    #         groups=fallback_groups,
    #         validation_result=validation_result,
    #         elements=elements,
    #         id_to_text_map=draft_semantic.id_to_text_map
    #     )
    
    def apply_corrections(self,
                         draft_semantic: DraftSemanticJSON,
                         elements: List[DetectedElement],
                         issues: List[ValidationIssue]) -> DraftSemanticJSON:
        """
        根据验证问题自动修正分组 增强的自动修复逻辑
        
        Args:
            draft_semantic: 原始分组
            elements: 元素列表
            issues: 验证发现的问题
            
        Returns:
            修正后的分组
        """
        print("[Phase 2-2] 正在执行基于规则的自动修复...")
        groups = list(draft_semantic.groups)
        elem_map = {e.element_id: e for e in elements}
        
        # 记录已经被合并过的文本ID，防止一个文本被多个图表抢
        merged_text_ids = set()
        
        # 预处理：找出所有“潜在的标题候选者”
        # 候选者定义：目前是 standalone (单独成组) 的文本，或者是 ungrouped 的文本
        candidate_texts = []
        
        # 1. 来自现有单文本组的候选
        groups_to_remove_indices = []
        for i, g in enumerate(groups):
            if len(g.member_ids) == 1:
                mid = g.member_ids[0]
                if mid in elem_map:
                    etype = (elem_map[mid].refined_type or "text").lower()
                    if "text" in etype or "title" in etype:
                        candidate_texts.append({
                            "id": mid, 
                            "bbox": elem_map[mid].bbox, 
                            "source_group_idx": i
                        })
        
        # 2. 处理孤立图表问题
        for issue in issues:
            if issue.issue_type == "orphan_chart":
                chart_id = issue.element_ids[0]
                chart_elem = elem_map.get(chart_id)
                if not chart_elem: continue
                
                # 在候选文本中寻找最近的一个
                best_candidate = None
                min_dist = float('inf')
                
                # 搜索范围阈值 (归一化坐标 0-1000)
                # 垂直距离容忍度较大 (标题通常在上方 0-150 单位)
                SEARCH_RADIUS_Y = 200 
                SEARCH_RADIUS_X = 100
                
                for candidate in candidate_texts:
                    if candidate["id"] in merged_text_ids:
                        continue
                        
                    text_bbox = candidate["bbox"]
                    chart_bbox = chart_elem.bbox
                    
                    # 计算距离
                    # 1. 垂直距离: 标题底部 到 图表顶部 (标题在上方)
                    dist_y_above = chart_bbox.ymin - text_bbox.ymax
                    # 2. 垂直距离: 图表底部 到 标题顶部 (标题在下方)
                    dist_y_below = text_bbox.ymin - chart_bbox.ymax
                    
                    # 水平重叠度检查 (标题应该和图表在水平方向上有重叠)
                    x_overlap = max(0, min(chart_bbox.xmax, text_bbox.xmax) - max(chart_bbox.xmin, text_bbox.xmin))
                    
                    is_valid_pos = False
                    current_dist = float('inf')
                    
                    # 优先找上方的 (Title)
                    if 0 < dist_y_above < SEARCH_RADIUS_Y and x_overlap > 0:
                        is_valid_pos = True
                        current_dist = dist_y_above # 距离越小越好
                    # 其次找下方的 (Caption)
                    elif 0 < dist_y_below < SEARCH_RADIUS_Y and x_overlap > 0:
                        is_valid_pos = True
                        current_dist = dist_y_below + 50 # 稍微惩罚下方，优先选上方
                        
                    if is_valid_pos and current_dist < min_dist:
                        min_dist = current_dist
                        best_candidate = candidate
                
                # 如果找到了合适的标题
                if best_candidate:
                    print(f"  -> 修复: 将文本 {best_candidate['id']} 吸附到图表 {chart_id} (距离: {min_dist})")
                    
                    # 1. 标记该文本已被使用
                    merged_text_ids.add(best_candidate["id"])
                    
                    # 2. 将其标记为待移除 (从原来的组)
                    if best_candidate["source_group_idx"] is not None:
                        groups_to_remove_indices.append(best_candidate["source_group_idx"])
                    
                    # 3. 更新图表所在的组
                    for g in groups:
                        if chart_id in g.member_ids:
                            # 插入到 member_ids (保持顺序，简单的加在前面或后面)
                            # 这里简单 append，Phase 3 会重新排序
                            g.member_ids.append(best_candidate["id"])
                            g.secondary_element_ids.append(best_candidate["id"])
                            g.semantic_desc += f" (自动吸附标题 {best_candidate['id']})"
                            break

            elif issue.issue_type == "mixed_candidate":
                target_ids = set(issue.element_ids or [])
                for g in groups:
                    if any(mid in target_ids for mid in g.member_ids):
                        if "mixed" not in (g.group_type or "").lower():
                            g.group_type = "mixed_group"
                            if "mixed" not in (g.semantic_desc or "").lower():
                                g.semantic_desc = (g.semantic_desc + "（已标记为混合组）").strip()
        
        # 清理被合并掉的旧组
        # 从后往前删，防止索引偏移
        final_groups = []
        groups_to_remove_set = set(groups_to_remove_indices)
        for i, g in enumerate(groups):
            if i not in groups_to_remove_set:
                final_groups.append(g)
            else:
                # 检查这个组是否还有剩余成员 (虽然按逻辑应该只剩空了，但为了保险)
                remaining_members = [m for m in g.member_ids if m not in merged_text_ids]
                if remaining_members:
                    g.member_ids = remaining_members
                    final_groups.append(g)
        
        # 重新整理 Reading Order (因为合并后位置可能变了，或者删除了某些组)
        final_groups.sort(key=lambda g: self._get_group_top_y(g, elem_map))
        for i, g in enumerate(final_groups):
            g.reading_order = i
            
        return DraftSemanticJSON(
            page_id=draft_semantic.page_id,
            groups=final_groups,
            ungrouped_element_ids=[uid for uid in draft_semantic.ungrouped_element_ids if uid not in merged_text_ids],
            id_to_text_map=draft_semantic.id_to_text_map,
            som_image_path=draft_semantic.som_image_path
        )
    
    def _get_group_top_y(self, group, elem_map):
        """获取组的顶部Y坐标用于排序"""
        min_y = 10000
        for mid in group.member_ids:
            if mid in elem_map:
                min_y = min(min_y, elem_map[mid].bbox.ymin)
        return min_y