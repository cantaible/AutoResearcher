"""LATS-R (Language Agent Tree Search for Retrieval) 子图。

树状发散搜索 → 评估剪枝 → 聚合结果。
适合需要多维度深入探索的复杂调研场景。
"""

import asyncio
import math
import uuid
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from state import (
    TreeNode, NodeEvaluation, LATSExpandResult, LATSResearcherState,
)
from configuration import Configuration

def _get_model():
    """延迟导入 configurable_model 以避免循环依赖。"""
    from graph import configurable_model
    return configurable_model

# ── 参数 ──
MAX_ITERATIONS = 12
MAX_DEPTH = 3
MAX_CHILDREN = 5
PRUNE_THRESHOLD = 0.3     # relevance < 此值则剪枝
LEAF_THRESHOLD = 0.3       # completeness < 此值则标记为 leaf
UCB_C = 1.414              # sqrt(2)


def _make_node(query: str, depth: int = 0, parent_id: str | None = None,
               dimension: str = "") -> TreeNode:
    """创建一个新的树节点。"""
    return TreeNode(
        id=str(uuid.uuid4())[:8],
        query=query, depth=depth, parent_id=parent_id, dimension=dimension,
    )


def _ucb1(node: TreeNode, parent_visits: int) -> float:
    """UCB1 选择公式。未访问节点返回无穷大优先探索。"""
    if node.visits == 0:
        return float("inf")
    exploit = node.value / node.visits
    explore = UCB_C * math.sqrt(math.log(parent_visits) / node.visits)
    return exploit + explore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 节点 1: INITIALIZE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def initialize(state: LATSResearcherState):
    """创建根节点，初始化搜索树。"""
    root = _make_node(query=state["research_topic"], depth=0)
    root.status = "pending"
    return {
        "tree": {root.id: root.model_dump()},
        "root_id": root.id,
        "current_node_id": root.id,
        "iteration": 0,
        "collected_findings": [],
        "compressed_research": "",
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 节点 2: SELECT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def select(state: LATSResearcherState):
    """用 UCB1 选择最有探索价值的 pending 节点。"""
    tree = state["tree"]
    total_visits = sum(n.get("visits", 0) for n in tree.values()) + 1

    best_id, best_score = None, -1.0
    for nid, ndata in tree.items():
        if ndata["status"] != "pending":
            continue
        node = TreeNode(**ndata)
        score = _ucb1(node, total_visits)
        if score > best_score:
            best_score = score
            best_id = nid

    # 如果没有 pending 节点，current_node_id 设为空，触发终止
    return {"current_node_id": best_id or ""}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 节点 3: EVALUATE (搜索 + 评估)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def evaluate(state: LATSResearcherState, config):
    """对当前节点进行 Tavily 网络搜索，并用 LLM 评估价值。"""
    from utils import tavily_search_async, get_api_key_for_model
    configurable = Configuration.from_runnable_config(config)
    
    node_id = state["current_node_id"]
    tree = state["tree"]
    node = tree[node_id]

    # 1. 将中文查询翻译为英文以提高 Tavily 召回率（翻译用 Worker 层级即可）
    model = _get_model().with_config({
        "model": configurable.simple_model,
        "api_key": get_api_key_for_model(configurable.simple_model, config),
    })
    en_query = await model.ainvoke([HumanMessage(
        f"将以下研究查询翻译为精准的英文搜索查询（只返回英文查询本身，不要解释）：\n{node['query']}"
    )])
    search_q = getattr(en_query, 'content', node['query'])

    # 2. Tavily 网络搜索（英文查询 + 原始中文查询，双路召回）
    raw_results = await tavily_search_async(
        search_queries=[search_q, node["query"]], max_results=5, topic="news"
    )
    snippets = []
    seen_urls = set()
    for resp in raw_results:
        for r in resp.get("results", []):
            if r.get('url') not in seen_urls:
                seen_urls.add(r.get('url'))
                snippets.append(f"[{r.get('title','')}]({r.get('url','')})\n{r.get('content','')}")
    results_str = "\n\n".join(snippets) if snippets else "无搜索结果"
    node["search_results"] = results_str
    node["result_count"] = len(snippets)

    # 2. LLM 结构化打分
    eval_prompt = f"""
    研究主题：{state['research_topic']}
    当前子查询：{node['query']} (维度: {node['dimension']})
    搜索结果：\n{results_str[:5500]}...
    """
    model = _get_model().with_config({
        "model": configurable.hard_model,
        "api_key": get_api_key_for_model(configurable.hard_model, config),
    })
    evaluator = model.with_structured_output(NodeEvaluation)
    evaluation = await evaluator.ainvoke([
        SystemMessage("评估上述搜索结果。相关性为内容与研究主题匹配度，完整性为是否仍需向下细分搜索。"),
        HumanMessage(eval_prompt)
    ])

    node["relevance_score"] = evaluation.relevance_score
    node["completeness_score"] = evaluation.completeness_score
    node["visits"] = 1

    # 3. 决定节点状态
    if evaluation.relevance_score < PRUNE_THRESHOLD:
        node["status"] = "pruned"
    elif node["depth"] == 0:
        # 根节点强制展开，保证至少发散一层
        node["status"] = "expanded"
    elif evaluation.completeness_score < LEAF_THRESHOLD or node["depth"] >= MAX_DEPTH:
        node["status"] = "leaf"
    else:
        node["status"] = "expanded"

    tree[node_id] = node
    return {"tree": tree}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 节点 4: EXPAND (维度展开)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def expand(state: LATSResearcherState, config):
    """如果值得深入，生成子查询。"""
    from utils import get_api_key_for_model
    configurable = Configuration.from_runnable_config(config)
    node_id = state["current_node_id"]
    tree = state["tree"]
    node = tree[node_id]

    # 只针对 status 变成 expanded 的节点生成子节点
    if node["status"] != "expanded":
        return {}

    prompt = f"""研究主题：{state['research_topic']}
当前查询：{node['query']}（深度 {node['depth']}）
已有搜索结果摘要：{node['search_results'][:800]}
"""
    expand_sys = f"""你是一个研究规划助手。你的任务是为搜索树的下一层生成子查询。

**层级策略（由宽到窄）**：
- 深度1：按大维度拆分（如：按地区=美国/中国/欧洲、按类型=开源/闭源、按时间段=上旬/中旬/下旬）
- 深度2：按具体实体拆分（如：具体公司/厂商/机构名称）
- 深度3：按具体产品/事件拆分

**要求**：
- 生成 3~{MAX_CHILDREN} 个子查询，覆盖面尽量广
- 不要只关注头部厂商，也要覆盖中小厂商和新兴玩家
- 每个子查询必须具体、可搜索，包含实体名或明确范围
- 子查询之间不要重叠"""
    # 子查询生成用 Worker 层级模型（结构化拆分任务，不需要复杂推理）
    model = _get_model().with_config({
        "model": configurable.simple_model,
        "api_key": get_api_key_for_model(configurable.simple_model, config),
    })
    expander = model.with_structured_output(LATSExpandResult)
    result = await expander.ainvoke([
        SystemMessage(expand_sys),
        HumanMessage(prompt)
    ])

    new_nodes = {}
    for sq, dim in zip(result.sub_queries[:MAX_CHILDREN], result.dimensions[:MAX_CHILDREN]):
        child = _make_node(query=sq, depth=node["depth"]+1, parent_id=node_id, dimension=dim)
        child.status = "pending"
        new_nodes[child.id] = child.model_dump()
        node["children_ids"].append(child.id)

    # 存回状态
    tree[node_id] = node
    tree.update(new_nodes)
    return {"tree": tree}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 节点 5: BACKPROPAGATE (回传)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def backpropagate(state: LATSResearcherState):
    """将新评估的节点价值沿父指针链向上回溯，用于后续 UCB1 选择。"""
    node_id = state["current_node_id"]
    tree = state["tree"]

    if not node_id:
        return {"iteration": state["iteration"] + 1}

    node = tree[node_id]
    # 基础奖赏：相关性越高，且有命中结果，价值才大
    reward = node["relevance_score"] * (1.0 if node["result_count"] > 0 else 0.1)

    # 沿由子至父链往上累加
    curr_id = node_id
    while curr_id:
        curr = tree[curr_id]
        # 当前节点在 evaluate 时 visit 已设为 1，只需更新 value
        # 其余祖先节点 visit + 1，value + reward
        if curr_id != node_id:
            curr["visits"] += 1
        curr["value"] += reward
        curr_id = curr["parent_id"]

    return {"tree": tree, "iteration": state["iteration"] + 1}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 判断路由: SHOULD_CONTINUE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def should_continue(state: LATSResearcherState) -> str:
    """终止条件：预算耗尽或无 pending 节点可展开。"""
    if not state.get("current_node_id"):
        return "aggregate"
    
    return "select"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 节点 6: AGGREGATE (大聚合)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def aggregate(state: LATSResearcherState, config):
    """收集所有 leaf (及有价值但不完整) 节点的搜索结果，一并压缩。"""
    from utils import get_api_key_for_model
    configurable = Configuration.from_runnable_config(config)
    tree = state["tree"]
    findings = []
    
    for ndata in tree.values():
        # 这里提取叶子节点或已展开有结果但不算剪枝的节点
        if ndata["status"] in ("leaf", "expanded") and ndata.get("search_results"):
            findings.append(f"【子查询: {ndata['query']}】\n{ndata['search_results']}")
            
    # 利用 LLM 做摘要压缩，合并重复信息，留存核心结论
    compress_prompt = f"研究主题：{state['research_topic']}\n\n以下是收集到的多维度发散结果：\n" + "\n\n".join(findings)
    # 摘要压缩用 Worker 层级模型（纯文本合并任务）
    model = _get_model().with_config({
        "model": configurable.simple_model,
        "api_key": get_api_key_for_model(configurable.simple_model, config),
    })
    compress_result = await model.ainvoke([
        SystemMessage("你是一个研究助手。请将发散收集回来的零散原始结果，梳理成逻辑连贯、要点清晰的详细情报摘要，务必保留所有的事实(如厂商、发布时间、产品特点)。提取结构清晰，便于主代理阅读。"),
        HumanMessage(compress_prompt[:80000])  # 防截断
    ])
    
    return {
        "collected_findings": findings,
        "compressed_research": getattr(compress_result, "content", ""),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图组装
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
lats_builder = StateGraph(LATSResearcherState)

lats_builder.add_node("initialize", initialize)
lats_builder.add_node("select", select)
lats_builder.add_node("evaluate", evaluate)
lats_builder.add_node("expand", expand)
lats_builder.add_node("backpropagate", backpropagate)
lats_builder.add_node("aggregate", aggregate)

lats_builder.add_edge(START, "initialize")
lats_builder.add_edge("initialize", "select")

# 选择后，判断路由
lats_builder.add_conditional_edges(
    "select",
    should_continue,
    {"select": "evaluate", "aggregate": "aggregate"}
)

lats_builder.add_edge("evaluate", "expand")
lats_builder.add_edge("expand", "backpropagate")
lats_builder.add_edge("backpropagate", "select")
lats_builder.add_edge("aggregate", END)

# 暴露给主图
lats_researcher_builder = lats_builder
