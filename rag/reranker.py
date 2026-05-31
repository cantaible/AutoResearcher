"""本地 reranker：对候选文档做统一重排。"""
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 在导入 transformers/sentence_transformers 前设置离线模式，避免库初始化阶段探测远端。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from sentence_transformers import CrossEncoder

from config import RERANKER_MAX_LENGTH, RERANKER_MODEL
from clusterer import cluster_candidates
from config import CLUSTER_PER_GROUP_TOP_N


MODEL_KEY_PATTERNS = [
    (r"\bGPT[-\s]?\d+(?:\.\d+)?(?:\s*Instant)?\b", "GPT"),
    (r"\bGemini(?:\s+\d+(?:\.\d+)?)?(?:\s+Flash[-\s]?Lite|\s+Embedding\s+\d+)?\b", "Gemini"),
    (r"\bClaude(?:\s+\d+(?:\.\d+)?)?\b", "Claude"),
    (r"\bDeepSeek(?:[-\s]?[A-Za-z0-9.]+)?\b", "DeepSeek"),
    (r"\bQwen(?:[-\s]?\d+(?:\.\d+)?)?\b", "Qwen"),
    (r"\bGLM[-\s]?\d+(?:\.\d+)?(?:[-\s]?Turbo)?\b", "GLM"),
    (r"\bGrok(?:\s+\d+(?:\.\d+)?)?\b", "Grok"),
    (r"\bMAI[-\s]?Image[-\s]?\d+\b", "MAI-Image"),
    (r"\bSeedance(?:\s+\d+(?:\.\d+)?)?\b", "Seedance"),
    (r"\bSkyReels(?:\s+[A-Za-z0-9.]+)?\b", "SkyReels"),
    (r"\bVidu(?:\s+[A-Za-z0-9.]+)?\b", "Vidu"),
    (r"\bMistral(?:\s+Small)?(?:\s+\d+(?:\.\d+)?)?\b", "Mistral"),
    (r"\bNemotron(?:\s+\d+(?:\.\d+)?)?\b", "Nemotron"),
    (r"\bLongCat(?:[-\s]?[A-Za-z0-9.]+)?\b", "LongCat"),
    (r"\bMiMo(?:[-\s]?[A-Za-z0-9.]+)?\b", "MiMo"),
    (r"\bMidjourney(?:\s+[A-Za-z0-9.]+)?\b", "Midjourney"),
    (r"\bMiniMax(?:\s+[A-Za-z0-9.]+)?\b", "MiniMax"),
    (r"\bComposer\b", "Composer"),
    (r"\bASMR\b|\bSupermemory\b", "ASMR"),
]

CHINESE_MODEL_KEYS = {
    "通义": "Qwen",
    "千问": "Qwen",
    "阿里": "Qwen",
    "智谱": "GLM",
    "深度求索": "DeepSeek",
    "字节": "Seedance",
    "豆包": "Seedance",
    "微软": "MAI-Image",
    "美团": "LongCat",
    "小米": "MiMo",
    "昆仑": "SkyReels",
    "生数科技": "Vidu",
    "英伟达": "Nemotron",
}

RELEASE_TERMS = [
    "发布", "推出", "上线", "开源", "开放权重", "升级", "新模型", "大模型",
    "release", "released", "launch", "launched", "introduced", "open-source",
]

ACTION_RELEASE_TERMS = [
    "发布", "推出", "上线", "开源", "开放权重",
    "release", "released", "launch", "launched", "introduced", "open-source",
]

NOISE_TERMS = [
    "融资", "股价", "招聘", "传闻", "预测", "评论", "教程", "榜单", "营销",
    "白皮书", "指南", "陪聊", "情感", "感情", "恋爱", "请愿", "下载榜", "封面",
]

HARD_NOISE_TERMS = ["白皮书", "指南", "陪聊", "情感", "感情", "恋爱", "请愿", "下载榜", "封面"]

EVIDENCE_TERMS = [
    "参数", "能力", "性能", "基准", "发布", "推出", "上线", "开源", "开放权重",
    "model", "release", "launch", "open source", "benchmark",
]

CONTROVERSY_TERMS = [
    "争议", "质疑", "对比", "超过", "领先", "挑战", "监管", "版权", "安全",
    "controversy", "benchmark", "lawsuit", "safety", "copyright",
]

TIER1_VENDOR_TERMS = [
    "OpenAI", "Anthropic", "Google", "Gemini", "Meta", "Llama", "xAI", "Grok",
    "Microsoft", "微软", "NVIDIA", "英伟达", "Alibaba", "阿里", "通义", "Qwen",
    "Tencent", "腾讯", "百度", "ByteDance", "字节", "豆包", "DeepSeek",
]

TIER2_VENDOR_TERMS = [
    "Mistral", "Midjourney", "MiniMax", "智谱", "GLM", "小米", "MiMo",
    "美团", "LongCat", "昆仑", "SkyReels", "Vidu", "生数科技",
]

AI_RERANK_FORMULA = {
    "impact": 0.65,
    "prominence": 0.50,
    "heat": 0.20,
    "controversy": 0.10,
    "penalty": 0.75,
}

QUERY_STOP_TERMS = {
    "2026", "3月", "发布", "推出", "上线", "新闻", "消息", "资讯", "模型", "大模型",
    "new", "model", "models", "llm", "foundation", "released", "release", "launch",
}


def get_reranker_device() -> str:
    # 强制 CPU：MPS 在 asyncio 并发 rerank 时会 segfault
    return "cpu"


_reranker = None
_reranker_lock = __import__("threading").Lock()


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                _reranker = CrossEncoder(
                    RERANKER_MODEL,
                    max_length=RERANKER_MAX_LENGTH,
                    device=get_reranker_device(),
                )
    return _reranker


def rerank_candidates(query: str, candidates: list[dict]) -> list[dict]:
    """对候选文档统一打分并按分数降序返回。"""
    if not candidates:
        return []

    model = get_reranker()
    pairs = [(query, candidate.get("doc", "")) for candidate in candidates]
    scores = model.predict(pairs)

    reranked = []
    for candidate, score in zip(candidates, scores):
        updated = dict(candidate)
        updated["rerank_score"] = float(score)
        updated["_event_key"] = _event_key(updated)
        reranked.append(updated)

    _apply_business_rubric_scores(query, reranked)
    reranked.sort(key=lambda item: item.get("hybrid_score", item["rerank_score"]), reverse=True)
    for item in reranked:
        item.pop("_event_key", None)
    return reranked


def _candidate_text(candidate: dict) -> str:
    metadata = candidate.get("metadata") or {}
    return " ".join(
        str(part)
        for part in (
            candidate.get("doc", ""),
            metadata.get("title", ""),
            metadata.get("preview", ""),
            metadata.get("source_name", ""),
        )
        if part
    )


def _has_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _event_key(candidate: dict) -> str:
    text = _candidate_text(candidate)
    for pattern, key in MODEL_KEY_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return key
    for term, key in CHINESE_MODEL_KEYS.items():
        if term in text:
            return key
    return ""


def _source_name(candidate: dict) -> str:
    metadata = candidate.get("metadata") or {}
    return str(metadata.get("source_name") or "").strip()


def _normalize_semantic_score(score: float) -> float:
    """把 CrossEncoder 分数压到 0-5，兼容 0-1 与 logit 两类输出。"""
    if 0.0 <= score <= 1.0:
        return score * 5.0
    return 5.0 / (1.0 + pow(2.718281828, -score))


def _impact_score(query: str, candidate: dict, semantic_score: float) -> float:
    """impact：语义相关性 + 是否真是模型发布/能力升级，而不是泛泛命中关键词。"""
    text = _candidate_text(candidate)
    score = _normalize_semantic_score(semantic_score)

    if _has_any(text, ACTION_RELEASE_TERMS):
        score += 0.35
    if _has_any(text, EVIDENCE_TERMS):
        score += 0.20
    if _event_key(candidate):
        score += 0.20

    term_hits = sum(1 for term in _query_terms(query) if term.lower() in text.lower())
    score += min(0.45, term_hits * 0.09)

    if _has_any(text, NOISE_TERMS) and not _has_any(text, ACTION_RELEASE_TERMS):
        score -= 0.70
    if _has_any(text, HARD_NOISE_TERMS):
        score -= 0.55

    return max(0.0, min(5.0, score))


def _prominence_score(candidate: dict, impact: float) -> float:
    """prominence：头部主体规则打分，并按 impact + 1 封顶。"""
    text = _candidate_text(candidate)
    if _has_any(text, TIER1_VENDOR_TERMS):
        score = 5.0
    elif _has_any(text, TIER2_VENDOR_TERMS):
        score = 3.5
    else:
        score = 0.0
    return min(score, impact + 1.0)


def _heat_score(candidate: dict, event_counts: Counter[str]) -> float:
    """heat：同一事件/模型在候选池中被重复报道的强度。"""
    cluster_size = int(candidate.get("_cluster_size") or 0)
    key = candidate.get("_event_key") or _event_key(candidate)
    count = cluster_size or (event_counts.get(key, 1) if key else 1)
    if count <= 1:
        return 0.0
    # 2 篇约 2.2 分，4 篇约 3.4 分，8 篇约 4.6 分，避免热度过度支配。
    return min(5.0, 1.0 + 1.2 * (count.bit_length() - 1))


def _controversy_score(candidate: dict) -> float:
    """controversy：无 LLM 时用争议/对比/监管/安全等显式信号近似。"""
    text = _candidate_text(candidate)
    hits = sum(1 for term in CONTROVERSY_TERMS if term.lower() in text.lower())
    return min(5.0, hits * 1.25)


def _source_penalty_scores(candidates: list[dict]) -> dict[str, float]:
    """来源灌水惩罚：同一媒体在候选池占比过高时扣分。"""
    sources = [_source_name(candidate) for candidate in candidates if _source_name(candidate)]
    counts = Counter(sources)
    total = len(candidates) or 1
    penalties: dict[str, float] = {}
    for source, count in counts.items():
        share = count / total
        if count <= 3 and share <= 0.25:
            penalties[source] = 0.0
            continue
        penalties[source] = min(2.0, max(0.0, (share - 0.25) * 5.0 + max(0, count - 8) * 0.08))
    return penalties


def _apply_business_rubric_scores(query: str, candidates: list[dict]) -> None:
    """按原版业务 rubric 计算 hybrid_score，排序用 hybrid_score，保留 rerank_score 原始语义分。"""
    event_counts = Counter(_event_key(candidate) for candidate in candidates)
    event_counts.pop("", None)
    source_penalties = _source_penalty_scores(candidates)

    for candidate in candidates:
        semantic_score = float(candidate.get("rerank_score", 0.0))
        impact = _impact_score(query, candidate, semantic_score)
        prominence = _prominence_score(candidate, impact)
        heat = _heat_score(candidate, event_counts)
        controversy = _controversy_score(candidate)
        penalty = source_penalties.get(_source_name(candidate), 0.0)

        common_score = (
            AI_RERANK_FORMULA["impact"] * impact
            + AI_RERANK_FORMULA["prominence"] * prominence
            + AI_RERANK_FORMULA["heat"] * heat
            + AI_RERANK_FORMULA["controversy"] * controversy
        )
        final_score = common_score - AI_RERANK_FORMULA["penalty"] * penalty
        candidate["hybrid_score"] = final_score
        candidate["rubric_scores"] = {
            "impact": impact,
            "prominence": prominence,
            "heat": heat,
            "controversy": controversy,
            "penalty_score": penalty,
            "common_score": common_score,
            "final_score": final_score,
        }


def _is_excluded_term(query: str, term: str) -> bool:
    return bool(re.search(rf"[-－]\s*{re.escape(term)}", query, flags=re.IGNORECASE))


def _query_terms(query: str) -> list[str]:
    terms = re.findall(r"[A-Za-z][A-Za-z0-9.\-]+|[\u4e00-\u9fff]{2,}", query)
    result = []
    for term in terms:
        normalized = term.lower()
        if normalized in QUERY_STOP_TERMS or term in QUERY_STOP_TERMS:
            continue
        if _is_excluded_term(query, term):
            continue
        result.append(term)
    return result


def _business_boost(query: str, candidate: dict) -> float:
    text = _candidate_text(candidate).lower()
    sources = set(candidate.get("sources", []))
    boost = 0.0

    if len(sources) >= 2:
        boost += 0.03
    if any(term.lower() in text for term in ACTION_RELEASE_TERMS):
        boost += 0.04
    if any(term in text for term in ("模型", "大模型", "多模态", "推理", "视频", "图像", "agent")):
        boost += 0.02

    term_hits = 0
    for term in _query_terms(query):
        if term.lower() in text:
            term_hits += 1
    boost += min(0.08, term_hits * 0.015)

    if any(term in text for term in NOISE_TERMS) and not any(term.lower() in text for term in ACTION_RELEASE_TERMS):
        boost -= 0.08
    if any(term in text for term in HARD_NOISE_TERMS):
        boost -= 0.05

    return boost


def diversify_reranked_candidates(query: str, reranked: list[dict], top_k: int) -> list[dict]:
    """在 CrossEncoder TopK 内做轻量事件多样性重排。

    不从 TopK 外替换候选，避免牺牲 reference recall；只调整 TopK 内部顺序，
    让不同模型/厂商的证据更靠前，便于后续压缩阶段保留。
    """
    if top_k <= 0 or len(reranked) <= top_k:
        return reranked

    head = list(reranked[:top_k])
    tail = list(reranked[top_k:])

    for item in head:
        item["_event_key"] = _event_key(item)
        item["_selection_boost"] = _business_boost(query, item)

    selected: list[dict] = []
    selected_keys: dict[str, int] = {}
    selected_sources: dict[str, int] = {}
    remaining = head

    while remaining:
        best_index = 0
        best_score = float("-inf")

        for index, item in enumerate(remaining):
            base_score = float(item.get("hybrid_score", item.get("rerank_score", 0.0)))
            key = item.get("_event_key", "")
            key_count = selected_keys.get(key, 0) if key else 0
            diversity_adjustment = 0.03 if key and key_count == 0 else -0.02 * key_count
            source = _source_name(item)
            source_count = selected_sources.get(source, 0) if source else 0
            source_adjustment = -0.18 * source_count

            selection_score = (
                base_score
                + float(item.get("_selection_boost", 0.0))
                + diversity_adjustment
                + source_adjustment
            )
            if selection_score > best_score:
                best_score = selection_score
                best_index = index

        chosen = remaining.pop(best_index)
        selected.append(chosen)
        key = chosen.get("_event_key", "")
        if key:
            selected_keys[key] = selected_keys.get(key, 0) + 1
        source = _source_name(chosen)
        if source:
            selected_sources[source] = selected_sources.get(source, 0) + 1

    for item in selected:
        item.pop("_event_key", None)
        item.pop("_selection_boost", None)
    return selected + tail


def rerank_and_select_candidates(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """先统一 rerank，再选择事件覆盖更稳的 TopK。"""
    reranked = rerank_candidates(query, candidates)
    return diversify_reranked_candidates(query, reranked, top_k)[:top_k]


def _cluster_rep_score(candidate: dict, query: str) -> float:
    """组内代表评分：业务信号分，不依赖 CrossEncoder。"""
    boost = _business_boost(query, candidate)
    # 加分来源数（已在 business_boost 里考虑）作为基础
    sources = len(candidate.get("sources", []))
    return boost + 0.01 * min(sources, 5)


def rerank_with_clusters(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """聚类→组内择优→组间Rerank→展开的 TopK 选择。

    1. 对 Merge 候选做语义聚类，每簇 = 同一事件的不同报道
    2. 每簇选 1 个业务分最高的代表
    3. CrossEncoder 只对代表做 Rerank
    4. 按事件排名依次从各簇取 N 篇，凑够 top_k
    """
    if not candidates:
        return []

    clusters, tail = cluster_candidates(candidates)

    if not clusters:
        return rerank_and_select_candidates(query, candidates, top_k)

    per_group = CLUSTER_PER_GROUP_TOP_N
    max_covered_clusters = top_k // per_group
    # 按簇大小排序，看前 max_covered_clusters 个最大簇覆盖了多少候选
    cluster_sizes = sorted([len(c) for c in clusters], reverse=True)
    covered = sum(cluster_sizes[:max_covered_clusters])
    total = sum(cluster_sizes)
    # 前 N 大簇覆盖不到 40%，说明聚类质量差（大多孤点），回退贪心多样性
    if covered < total * 0.4:
        return rerank_and_select_candidates(query, candidates, top_k)

    # 组内先按业务信号排序，再选代表。后续展开时沿用这个簇内顺序。
    clusters = [
        sorted(cluster, key=lambda c: _cluster_rep_score(c, query), reverse=True)
        for cluster in clusters
    ]

    # 组内择优：每簇选代表，并把簇大小交给 heat 规则项。
    reps: list[dict] = []
    for cluster in clusters:
        best = dict(cluster[0])
        best["_cluster_size"] = len(cluster)
        reps.append(best)

    # 组间 Rerank（只跑 ~N 篇代表）
    ranked_reps = rerank_candidates(query, reps)

    # 构建 rep → cluster 映射（用 article_id 做 key，因为 rerank_candidates 会复制 dict）
    def _rep_key(rep: dict) -> str:
        meta = rep.get("metadata") or {}
        return str(meta.get("article_id", rep.get("id", id(rep))))

    rep_to_cluster: dict[str, int] = {}
    for i, rep in enumerate(reps):
        rep_to_cluster[_rep_key(rep)] = i

    cluster_score_context: dict[int, dict] = {}
    for rep in ranked_reps:
        cluster_idx = rep_to_cluster.get(_rep_key(rep))
        if cluster_idx is not None:
            cluster_score_context[cluster_idx] = rep

    def _copy_with_rep_scores(item: dict, rep: dict, decay: float) -> dict:
        copied = dict(item)
        copied.setdefault("rerank_score", float(rep.get("rerank_score", 0.0)))
        copied.setdefault("hybrid_score", float(rep.get("hybrid_score", 0.0)) - decay)
        if "rubric_scores" not in copied and "rubric_scores" in rep:
            copied["rubric_scores"] = dict(rep["rubric_scores"])
        return copied

    # 展开：按事件排名依次从各簇取 N 篇
    selected: list[dict] = []
    selected_ids: set[str] = set()
    round_idx = 0

    while len(selected) < top_k:
        added_this_round = False
        for rep in ranked_reps:
            if len(selected) >= top_k:
                break
            cluster_idx = rep_to_cluster.get(_rep_key(rep))
            if cluster_idx is None:
                continue
            cluster = clusters[cluster_idx]
            rep_context = cluster_score_context.get(cluster_idx, rep)
            # 跳过在当前轮次索引之前没有元素的簇
            start = round_idx * per_group
            end = start + per_group
            group_slice = cluster[start:end]
            if not group_slice:
                continue
            for item in group_slice:
                if len(selected) >= top_k:
                    break
                item_id = str(item.get("metadata", {}).get("article_id", item.get("id", "")))
                if item_id in selected_ids:
                    continue
                selected.append(_copy_with_rep_scores(item, rep_context, decay=0.03 * round_idx))
                selected_ids.add(item_id)
                added_this_round = True
        if not added_this_round:
            break
        round_idx += 1

    # tail（无语义文本的候选）排在末尾
    for item in tail:
        if len(selected) >= top_k:
            break
        item_id = str(item.get("metadata", {}).get("article_id", item.get("id", "")))
        if item_id in selected_ids:
            continue
        copied = dict(item)
        copied.setdefault("rerank_score", 0.0)
        copied.setdefault("hybrid_score", 0.0)
        selected.append(copied)
        selected_ids.add(item_id)

    return selected[:top_k]
