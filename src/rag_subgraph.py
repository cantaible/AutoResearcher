"""RAG 子图：Plan → 并行 Execute-with-Retry → Compress。"""

import asyncio
import sys
from pathlib import Path

from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

from configuration import Configuration
from state import (
    RAGExecuteState, RAGQueryPlan, RAGResearcherState,
    ResearcherOutputState, SearchEvaluation,
)

def _get_model():
    """延迟导入 configurable_model 以避免循环依赖。"""
    from graph import configurable_model
    return configurable_model
from utils import get_api_key_for_model

# ── Plan 阶段的系统提示 ──
RAG_PLAN_PROMPT = """你是一个查询规划助手。将用户的研究主题拆分为多个子查询，
以提高在本地新闻数据库中的召回率。今天是 {today}。

## 搜索系统能力说明
你的每个子查询会被发送到一个混合检索系统，该系统会：
1. 用 query 文本做**向量语义匹配**（适合自然语言描述性短句）
2. 同时做**关键词全文检索**（适合精确术语和名称）
3. 用 start_date/end_date 参数在搜索层做**时间过滤**——因此 query 文本中不需要包含日期信息
4. 用 category 参数按分类过滤

## 拆分原则
1. **按角度/维度拆分**：每个子查询聚焦一个具体的子主题、厂商群或技术方向，而非重复同一个泛化 query
2. **不要按时间窗口拆分**：时间过滤已交给搜索引擎参数处理，所有子查询共享用户指定的完整时间范围即可
3. **query 应为描述性自然语言短句**：向量检索对完整语义的句子效果远好于松散的关键词堆叠
4. **适度补充英文名/别名**：对于知名厂商和模型，在 query 中混入英文名有助于提升召回率
6. **子查询之间应尽量正交**：不同子查询的预期返回结果重叠度应尽可能低

## 字段说明
- search_intent: 一句话说明这个子查询想找什么信息
- query: 描述性自然语言搜索短句（中文为主，可混合英文术语）
- start_date: 时间范围起始日期，格式 YYYY-MM-DD
- end_date: 时间范围结束日期，格式 YYYY-MM-DD
- category: 新闻分类，"AI" / "GAMES" / ""（不限）"""


async def plan(state: RAGResearcherState, config) -> dict:
    """Plan 节点：LLM 将研究主题拆分为子查询列表。"""
    configurable = Configuration.from_runnable_config(config)
    # Plan 是 RAG 流程中最关键的环节，使用 hard_model 确保查询质量
    model = _get_model().with_config({
        "model": configurable.hard_model,
        "max_tokens": configurable.hard_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.hard_model, config),
    })
    structured_model = model.with_structured_output(RAGQueryPlan)

    prompt = RAG_PLAN_PROMPT.format(today=date.today().isoformat())
    result = await structured_model.ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"请将以下研究主题拆分为子查询：\n\n{state['research_topic']}"),
    ])

    sub_queries = [q.model_dump() for q in result.sub_queries]
    print(f"  📝 RAG Plan: 拆分为 {len(sub_queries)} 个子查询")
    return {"sub_queries": sub_queries}


def route_plan(state: RAGResearcherState) -> list[Send]:
    """路由函数：将 plan 产出的子查询通过 Send 并行分发到 execute。"""
    sub_queries = state.get("sub_queries", [])
    if not sub_queries:
        print("  ⚠️ RAG Plan: 未生成子查询，直接进入 compress")
        return [Send("compress", {"research_topic": state["research_topic"]})]
    return [
        Send("execute", {
            "sub_query": sq,
            "research_topic": state["research_topic"],
        })
        for sq in sub_queries
    ]


async def _run_single_rag_query(sub_query: dict, top_k: int = 20) -> str | dict:
    """在线程池中执行单个 RAG 查询。

    返回 str（默认）或 dict（当需要检索详情时）。
    """
    from rag_search import rag_search
    result = await asyncio.to_thread(
        rag_search.invoke,
        {
            "query": sub_query["query"],
            "start_date": sub_query.get("start_date", ""),
            "end_date": sub_query.get("end_date", ""),
            "category": sub_query.get("category", ""),
            "top_k": top_k,
            "return_details": True,  # 启用检索详情记录
        },
    )
    return result


# ── Execute 阶段的评估提示 ──
RAG_EVALUATE_PROMPT = """你是一个搜索结果质量评估助手。评估以下搜索结果是否充分回答了子查询。

评估标准：
1. **good**：结果直接相关且信息充足，无需补搜
2. **insufficient**：结果部分相关但信息不足，需要补充搜索
3. **off_topic**：结果偏离主题，需要改写查询

如果判断为 insufficient 或 off_topic，请提供修正后的查询（refined_query）：
- 保留原查询的时间范围和主题边界
- 可调整关键词、补充中文/英文别名、缩窄或扩展范围
- 修正后的查询应具体、可搜索"""


async def execute(state: RAGExecuteState, config) -> dict:
    """执行单个子查询，带结构化重试。

    由 Send 分发，每个实例独立处理一个子查询。
    内部用 Python 循环实现重试，不需要图的边来编排。
    """
    sub_query = state["sub_query"]
    research_topic = state["research_topic"]
    configurable = Configuration.from_runnable_config(config)
    max_retries = configurable.max_rag_retries

    model = _get_model().with_config({
        "model": configurable.simple_model,
        "max_tokens": configurable.simple_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.simple_model, config),
    })
    evaluator = model.with_structured_output(SearchEvaluation)

    all_results = []
    all_retrieval_details = []  # 收集所有轮次的检索详情
    all_evidence_items = []  # 收集所有轮次的证据条目
    current_query = sub_query["query"]

    for attempt in range(max_retries + 1):
        # 1. 执行搜索
        result = await _run_single_rag_query(
            {**sub_query, "query": current_query},
            top_k=configurable.rag_top_k,
        )

        # 处理返回结果（可能是 str 或 dict）
        if isinstance(result, dict):
            formatted_output = result["formatted_output"]
            retrieval_details = result.get("retrieval_details")
            if retrieval_details:
                all_retrieval_details.append(retrieval_details)

            # 提取原始文章列表，构建 evidence_pool
            articles = result.get("articles", [])
            if articles:
                from datetime import datetime
                from utils import generate_evidence_id

                for article in articles:
                    evidence = {
                        "source": "rag",
                        "article_id": article.get("id"),
                        "url": article.get("url"),
                        "title": article.get("title", ""),
                        "content": article.get("content", ""),
                        "published_date": article.get("published_date"),
                        "used_by_node": "rag_researcher",
                        "query": current_query,
                        "timestamp": datetime.now().isoformat(),
                    }
                    evidence["id"] = generate_evidence_id(evidence)
                    all_evidence_items.append(evidence)
        else:
            formatted_output = result

        all_results.append(f"[第{attempt+1}轮] 查询: {current_query}\n{formatted_output}")
        print(f"  🔍 RAG execute [{attempt+1}/{max_retries+1}]: {current_query}")

        # 最后一轮不再评估
        if attempt == max_retries:
            break

        # 2. 结构化评估
        evaluation = await evaluator.ainvoke([
            SystemMessage(content=RAG_EVALUATE_PROMPT),
            HumanMessage(content=(
                f"研究主题：{research_topic}\n"
                f"子查询：{current_query}\n"
                f"搜索结果：\n{formatted_output}"
            )),
        ])

        if evaluation.quality == "good":
            print(f"  ✅ RAG evaluate: {evaluation.reason}")
            break

        # 3. 不满意 → 用修正后的 query 重试
        current_query = evaluation.refined_query or current_query
        print(f"  🔄 RAG retry: {evaluation.reason} → {current_query}")

    combined = "\n\n".join(all_results)

    # 去重 evidence_pool（同一篇文章可能在多轮中重复出现）
    from utils import deduplicate_evidence
    all_evidence_items = deduplicate_evidence(all_evidence_items)

    return {
        "raw_results": [f"--- 查询: {sub_query['query']} ---\n{combined}"],
        "raw_notes": [f"[RAG] {sub_query['query']}"],
        "retrieval_details": all_retrieval_details,  # 返回所有轮次的检索详情
        "evidence_pool": all_evidence_items,  # 返回结构化证据
    }


# ── Compress 阶段的系统提示 ──
RAG_COMPRESS_PROMPT = """你是一个 RAG 证据保全与事件整理助手。你的目标不是写最终报告，而是把搜索结果中所有可能相关的模型发布事件完整保留下来，供后续 Writer 使用。

核心原则：
1. 宁可多保留，不要漏掉。只要搜索结果中出现“发布、推出、上线、开源、预览版、Beta、模型家族、released、launched、introduced、open-weight”等表述，并且涉及模型名称或模型能力，就必须列入候选事件。
2. 不要因为厂商不够头部、证据较弱、事件小众、属于图像/视频/语音/代码/embedding/agent 模型就删除。低置信事件放入“待核验/低置信候选”，也不能直接丢弃。
3. 每个模型或模型家族必须单独成条，不要合并成“多家公司发布了若干模型”这种概括句。
4. 每条事实必须保留文章来源标记，格式为 [article:ID]（ID 为搜索结果中的 ArticleID 数字）。没有 article 引用的事实不要写。
5. 保留关键细节：厂商、模型名称、发布日期/新闻日期、模型类型、发布状态、参数/能力/benchmark/应用场景。
6. 如果多篇文章支持同一事件，保留多个 [article:ID]；如果不同文章互相矛盾，明确标为“待核验”。

请严格按以下结构输出：

## 高置信模型发布事件
逐条列出证据充分、明确属于 2026 年 3 月模型发布/推出/开源/上线的事件。每条格式：
- 事件：厂商 - 模型名
  - 日期：
  - 类型：
  - 发布状态：
  - 关键事实：
  - 证据：[article:ID] [article:ID]

## 待核验/低置信候选
逐条列出可能是模型发布，但证据不完整、时间不明确、或可能只是版本更新/生态接入/评测解读的事件。不要省略小众模型。

## 明确非发布或排除项
列出和主题相关但不应计入“新模型发布”的内容，例如行业新闻、融资、评测、旧模型解读、应用产品发布等。

## 覆盖检查
- 写出你最终保留了多少个候选事件。
- 写出仍可能遗漏的方向。
- 写出本摘要中实际使用过的 ArticleID 列表。

再次强调：这是给后续报告生成器看的证据账本，不是面向用户的精简摘要。不要为了简洁而删除候选事件。"""


async def compress(state: RAGResearcherState, config) -> dict:
    """Compress 节点：合并去重所有子查询结果，压缩为摘要。"""
    configurable = Configuration.from_runnable_config(config)
    model = _get_model().with_config({
        "model": configurable.simple_model,
        "max_tokens": configurable.simple_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.simple_model, config),
    })

    all_results = "\n\n".join(state.get("raw_results", []))

    response = await model.ainvoke([
        SystemMessage(content=RAG_COMPRESS_PROMPT),
        HumanMessage(
            content=f"研究主题：{state['research_topic']}\n\n搜索结果：\n\n{all_results}"
        ),
    ])

    result_text = response.content
    print(f"  📦 RAG Compress: {len(result_text)} 字符")
    return {"compressed_research": result_text}


# ── 图组装：plan → [Send ×N] execute → compress ──
from langgraph.graph import END, START, StateGraph  # noqa: E402

rag_researcher_builder = StateGraph(
    RAGResearcherState, output=ResearcherOutputState
)
rag_researcher_builder.add_node("plan", plan)
rag_researcher_builder.add_node("execute", execute)
rag_researcher_builder.add_node("compress", compress)

rag_researcher_builder.add_edge(START, "plan")
rag_researcher_builder.add_conditional_edges("plan", route_plan, ["execute", "compress"])
rag_researcher_builder.add_edge("execute", "compress")
rag_researcher_builder.add_edge("compress", END)
