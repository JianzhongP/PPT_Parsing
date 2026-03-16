import re
import difflib

from state import PPTPageState, FinalPageOutput
from openai import AzureOpenAI


_NUM_PATTERN = re.compile(r"(?<![\w\u4e00-\u9fff])(\d+(?:\.\d+)?%?)(?![\w\u4e00-\u9fff])")

_FAILURE_MARKERS = (
    "分析失败",
    "提取失败",
    "混合提取降级",
    "mixed提取降级",
    "unterminated string",
    "jsondecodeerror",
    "expecting value",
    "traceback",
    "error:",
)


def _numbers_set(s: str) -> set[str]:
    return {m.group(1) for m in _NUM_PATTERN.finditer(s or "")}


def _contains_failure_marker(text: str) -> bool:
    s = str(text or "").strip().lower()
    if not s:
        return False
    return any(m in s for m in _FAILURE_MARKERS)


def _canonical_line(text: str) -> str:
    s = str(text or "").strip().lower()
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\-–—_]+", "", s)
    s = re.sub(r"([a-z])[1l|]([a-z])", r"\1i\2", s)
    s = re.sub(r"([a-z])0([a-z])", r"\1o\2", s)
    s = re.sub(r"[^\w\u4e00-\u9fff\s\(\)%/\.:,+#]", "", s)
    return s.strip()


def _looks_joined_english(text: str) -> bool:
    s = str(text or "").strip()
    if len(s) < 35:
        return False
    letters = re.findall(r"[A-Za-z]", s)
    if not letters:
        return False
    letter_ratio = len(letters) / max(len(s), 1)
    space_ratio = s.count(" ") / max(len(s), 1)
    long_word = max((len(w) for w in re.findall(r"[A-Za-z]+", s)), default=0)
    return letter_ratio > 0.75 and (space_ratio < 0.06 or long_word >= 20)


def _repair_joined_english(text: str) -> str:
    s = str(text or "").strip()
    if not s or not _looks_joined_english(s):
        return s
    repaired = s
    repaired = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", repaired)
    repaired = re.sub(r",(?=[A-Za-z])", ", ", repaired)
    repaired = re.sub(r"\)(?=[A-Za-z])", ") ", repaired)
    repaired = re.sub(r"(?<=[A-Za-z])\((?=[A-Za-z])", " (", repaired)
    repaired = re.sub(r"\s+", " ", repaired).strip()
    return repaired


def _is_footer_noise_line(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    if re.fullmatch(r"#{1,6}\s*\d+", s):
        return True
    if re.fullmatch(r"\d+", s):
        return True
    if re.fullmatch(r"\d+\s+[A-Z][A-Z0-9\-]{2,}", s):
        return True
    if s.upper() in {"DIVAMICS", "ADIS"}:
        return True
    return False


def _sanitize_markdown_output(markdown: str) -> str:
    if not markdown:
        return ""

    cleaned_lines = []
    seen_recent: list[str] = []
    for raw in (markdown or "").splitlines():
        line = _repair_joined_english(raw.rstrip())
        compact = line.strip()
        compact_lower = compact.lower()

        if _contains_failure_marker(compact):
            continue
        if _is_footer_noise_line(compact):
            continue
        if "图表类型" in compact and compact_lower.endswith("unknown*"):
            continue
        if "图表类型" in compact and compact_lower.endswith("未知*"):
            continue

        if _looks_joined_english(compact) and max((len(w) for w in re.findall(r"[A-Za-z]+", compact)), default=0) >= 24:
            continue

        if re.fullmatch(r"[\*\s\-_=+]{1,8}", compact):
            continue

        key = _canonical_line(compact)
        if key:
            is_dup = False
            for prev in seen_recent[-8:]:
                if key == prev:
                    is_dup = True
                    break
                if len(key) >= 12 and len(prev) >= 12:
                    if difflib.SequenceMatcher(None, key, prev).ratio() >= 0.96:
                        is_dup = True
                        break
            if is_dup:
                continue
            seen_recent.append(key)

        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _is_decorative_or_evidence_poor(state: PPTPageState) -> bool:
    ga = state.global_analysis
    if not ga:
        return True
    extracted_text = (ga.extracted_text or "").strip()
    elements = ga.elements or []
    insights = state.element_insights or []

    core = (ga.core_summary or "").lower()
    title = (ga.section_title or "").lower()
    decorative_hint = (
        "decorative" in core or "title" in core or "目录" in core or "封面" in core or "章节" in core or
        "decorative" in title or "title" in title
    )

    no_evidence = (not extracted_text) and (len(elements) == 0) and (len(insights) == 0)
    return decorative_hint or (ga.is_pure_text and no_evidence)

def node_step5_editor(state: PPTPageState, llm_client: AzureOpenAI, debug_logger=None) -> dict:
    """Step 5: 最终润色节点"""
    page_index = state.page_index
    page_new = page_index + 1
    
    print(f"\n[Step5] 正在润色页面 {page_new} ...")
    
    # 记录步骤开始
    if debug_logger:
        debug_logger.log_step_start(
            page_index,
            "Step 5: Editor",
            input_data={
                "page_index": page_index,
                "section_title": state.final_output.section_title if state.final_output else None,
                "markdown_length": len(state.final_output.markdown_content) if state.final_output else 0
            },
            previous_step="Step 4: Output Generation"
        )
    
    # 以 Step4 的最终 markdown 作为润色输入，避免对 fused_markdown 进行“再创作”
    raw_output = ""
    if state.final_output and state.final_output.markdown_content:
        raw_output = state.final_output.markdown_content
    elif state.analysis_result and state.analysis_result.fused_markdown:
        raw_output = state.analysis_result.fused_markdown

    # 如果 Step4 已选择复用复杂 Pipeline 的 markdown，则跳过润色：
    # 1) 避免 LLM 改写图片引用/破坏相对路径
    # 2) 避免把调试信息再“扩写”回用户输出
    render_mode = ""
    try:
        render_mode = (state.final_output.slice_reference or {}).get("render_mode", "") if state.final_output else ""
    except Exception:
        render_mode = ""
    if render_mode in {"complex_pipeline_md", "visual_only"}:
        final_output = state.final_output
        if final_output:
            final_output.markdown_content = _sanitize_markdown_output(raw_output)
        return {"final_output": final_output}

    # 封面/章节/证据不足页：跳过润色，避免无证据扩写
    if _is_decorative_or_evidence_poor(state):
        final_output = state.final_output
        if final_output:
            final_output.markdown_content = _sanitize_markdown_output(raw_output)
        return {"final_output": final_output}
    
    system_prompt = f"""你是一位严谨的医药研发文档编辑。
你的任务是清洗并润色 PPT 解析的原始数据，输出一份**结构清晰、层级分明**的 Markdown 报告。

### 核心原则
1. **严禁融合**：必须严格保留“核心摘要”与“元素分析”的独立板块，**不要**将它们写成一段通文。
2. **结构规范**：严格遵守下方的[输出模板]。
3. **去除噪点**：删除“Step 1分析显示”、“根据 XML 解析”等过程性废话，直接展示结论。
4. **严禁新增事实**：不得新增原文中不存在的结论、数字、统计学指标（HR/CI/P 值/百分比等）。只能改写表达与重排结构。
5. **页码修正**：当前处理的是第 {page_new} 页。
6. **注意**：尽量不要更改原始输出的内容，只做格式和结构上的润色；如遇信息缺失，请保持空缺，不要补写。

### 输出模板 (请严格按此格式输出)

## 第 {page_new} 页：<这里填标题>

### 1. 核心观点
<这里用精炼的语言总结本页旨在传达的核心商业或科研结论，2-3句话>

### 2. 页面布局与逻辑
<简要描述页面结构，例如：左侧为文字介绍，右侧为两张生存曲线图。说明各部分之间的逻辑关系>

### 3. 关键元素详情
<遍历每一个有价值的元素，按顺序排列>

- **[<元素类型中文名>] <元素ID或简述>**
  - **核心洞察**：<直接写出该图表/文本块得出的具体结论>
  - **关键数据**：<列出支持结论的数据，如 'HR=0.65, P<0.001' 或 '同比增长 58%'>
  - *(如果有)* **图例/备注**：<补充信息>

*(如果有下一个元素，继续列出)*
...

---
"""

    from config import get_config
    response = llm_client.chat.completions.create(
        model=get_config().llm_model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_output}
        ]
    )
    
    polished_content = response.choices[0].message.content

    # 防事实漂移：若润色后新增任何数字 token，回滚到原始输出
    before_nums = _numbers_set(raw_output)
    after_nums = _numbers_set(polished_content)
    if len(after_nums - before_nums) > 0:
        polished_content = raw_output

    polished_content = _sanitize_markdown_output(polished_content)
    
    # 更新 FinalOutput
    final_output = state.final_output
    final_output.markdown_content = polished_content
    
    # 记录步骤结束
    if debug_logger:
        debug_logger.log_step_end(
            page_index,
            "Step 5: Editor",
            output_data={
                "page_index": page_index,
                "section_title": final_output.section_title,
                "polished_markdown_length": len(polished_content),
                "status": "completed"
            },
            status="success"
        )
    
    return {"final_output": final_output}