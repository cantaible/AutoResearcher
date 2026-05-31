"""RAG 四节点评测：Query Rewrite / Dense / BM25(OpenSearch) / Rerank。

评测目标：
- Query Rewrite：无硬真值时用 rubric + rerank 弱反馈打分
- Dense/Chroma：用标注事件计算 recall / precision / F1
- BM25/OpenSearch：用标注事件计算 recall / precision / F1
- Rerank：无硬真值时用 rubric 打分，保留 NDCG/retention 等参考指标

用法：
    python eval/eval_four_nodes.py --run-dir logs/<run_dir>
    python eval/eval_four_nodes.py --run-dir logs/<run_dir> --synthesize-rubrics --business-context-path docs/interview.md
    python eval/eval_four_nodes.py --run-dir logs/<run_dir> --judge-with-llm
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

load_dotenv(PROJECT_ROOT / ".env", override=False)


DEFAULT_TOPIC = (
    "搜索本地新闻数据库，查找2026年3月1日至3月31日期间发布的大模型相关新闻，"
    "看看有哪些新的大模型发布了，尤其要关注头部厂商。"
)

QUERY_REWRITE_WEIGHTS = {
    "coverage": 0.35,
    "constraint_accuracy": 0.20,
    "diversity": 0.20,
    "rerank_feedback": 0.25,
}

RERANK_WEIGHTS = {
    "impact": 0.65,
    "prominence": 0.50,
    "heat": 0.20,
    "controversy": 0.10,
    "penalty_score": 0.75,
}
RERANK_MAX_COMMON_SCORE = (
    RERANK_WEIGHTS["impact"]
    + RERANK_WEIGHTS["prominence"]
    + RERANK_WEIGHTS["heat"]
    + RERANK_WEIGHTS["controversy"]
) * 5

DEFAULT_RUBRICS = {
    "query_rewrite": {
        "coverage_dimensions": [
            {
                "name": "海外头部厂商",
                "description": "覆盖 OpenAI、Anthropic、Google、Meta、xAI、Microsoft 等头部厂商。",
                "keywords": ["OpenAI", "Anthropic", "Google", "Gemini", "Meta", "Llama", "xAI", "Grok", "Microsoft"],
            },
            {
                "name": "国内厂商",
                "description": "覆盖阿里、百度、腾讯、字节、DeepSeek、智谱等国内模型发布。",
                "keywords": ["阿里", "通义", "Qwen", "百度", "文心", "腾讯", "混元", "字节", "豆包", "DeepSeek", "智谱", "GLM"],
            },
            {
                "name": "开源/开放权重",
                "description": "覆盖开源或开放权重模型。",
                "keywords": ["开源", "开放权重", "open source", "Llama", "Mistral", "Qwen", "DeepSeek"],
            },
            {
                "name": "多模态/图像",
                "description": "覆盖图像理解、图像生成、视觉多模态模型。",
                "keywords": ["多模态", "图像", "图片", "image", "vision", "视觉"],
            },
            {
                "name": "视频/语音",
                "description": "覆盖视频生成、语音、音频相关模型。",
                "keywords": ["视频", "video", "语音", "audio", "Sora", "Vidu", "Seedance", "SkyReels"],
            },
            {
                "name": "代码/Agent/推理",
                "description": "覆盖代码模型、Agent、推理模型或能力升级。",
                "keywords": ["代码", "coding", "Agent", "智能体", "推理", "reasoning"],
            },
        ],
        "weights": QUERY_REWRITE_WEIGHTS,
    },
    "rerank": {
        "dimensions": [
            {
                "key": "impact",
                "name": "实际影响力",
                "weight": 0.65,
                "description": "事件是否对月报主题有实际价值，语义相关且是模型发布、能力升级、开源或重要产品进展。",
            },
            {
                "key": "prominence",
                "name": "主体头部性",
                "weight": 0.50,
                "description": "事件主体是否为 OpenAI、Google、Anthropic、Meta、xAI、Microsoft、阿里、腾讯、百度、DeepSeek 等头部主体；该项按 impact + 1 封顶。",
            },
            {
                "key": "heat",
                "name": "报道热度",
                "weight": 0.20,
                "description": "同一事件是否被多篇报道重复覆盖，用于衡量事件热度，而不是让 LLM 主观猜测。",
            },
            {
                "key": "controversy",
                "name": "争议性",
                "weight": 0.10,
                "description": "是否包含争议、对比、监管、安全、版权、基准挑战等需要在月报中提示的语义信号。",
            },
            {
                "key": "penalty_score",
                "name": "来源灌水惩罚",
                "weight": -0.75,
                "description": "同一来源占比过高时扣分，防止高产媒体刷榜。",
            },
        ],
        "weights": RERANK_WEIGHTS,
        "formula": "CommonScore = 0.65*impact + 0.50*prominence + 0.20*heat + 0.10*controversy; FinalScore = CommonScore - 0.75*penalty_score",
    },
}

RELEASE_TERMS = [
    "发布", "推出", "上线", "开源", "开放权重", "升级", "更新", "模型", "大模型",
    "release", "launch", "model", "open source", "reasoning", "multimodal",
]

ACTION_RELEASE_TERMS = [
    "发布", "推出", "上线", "开源", "开放权重",
    "release", "launch", "open source",
]

NOISE_TERMS = [
    "融资", "股价", "招聘", "传闻", "预测", "评论", "观点", "教程", "榜单", "营销",
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

EVENT_HINTS = {
    "ASMR": ["ASMR", "Supermemory", "永久记忆", "memory"],
    "Composer": ["Cursor", "Composer", "coding", "代码", "编程"],
    "DeepSeek V4": ["DeepSeek", "DeepSeek V4", "深度求索", "多模态", "国产", "国内"],
    "GLM-5-Turbo": ["智谱", "ChatGLM", "GLM", "GLM-5-Turbo", "智能体", "Agent", "国产", "国内"],
    "GPT-5.3 Instant": ["OpenAI", "ChatGPT", "GPT", "GPT-5.3", "GPT-5.3 Instant"],
    "GPT-5.4": ["OpenAI", "ChatGPT", "Codex", "GPT", "GPT-5.4", "reasoning", "推理"],
    "Gemini 3.1 Flash-Lite": ["Google", "谷歌", "Gemini", "Gemini 3.1", "Flash-Lite"],
    "Gemini Embedding 2": ["Google", "谷歌", "Gemini", "Embedding", "嵌入", "多模态"],
    "Grok 4.20": ["xAI", "Grok", "Grok 4.20", "推理", "reasoning"],
    "LongCat-Flash-Prover": ["美团", "LongCat", "LongCat-Flash-Prover", "开源", "数学", "证明", "推理"],
    "MAI-Image-2": ["Microsoft", "微软", "MAI", "MAI-Image", "图像", "生图", "image"],
    "MiMo": ["小米", "MiMo", "MiMo-V2", "国产", "国内", "多模态"],
    "Midjourney V8": ["Midjourney", "Midjourney V8", "图像", "生图", "image"],
    "MiniMax M2.7": ["MiniMax", "M2.7", "国产", "国内", "Agent", "智能体"],
    "Mistral Small 4": ["Mistral", "Mistral Small", "开源", "open source", "Apache"],
    "Nemotron 3": ["NVIDIA", "英伟达", "Nemotron", "开源", "推理", "reasoning"],
    "Qwen3.5": ["阿里", "Alibaba", "通义", "千问", "Qwen", "Qwen3.5", "开源", "国产", "国内", "多模态", "Agent"],
    "Seedance 2.0": ["字节", "ByteDance", "CapCut", "Dreamina", "Seedance", "视频", "video"],
    "SkyReels V4": ["昆仑", "SkyReels", "SkyReels V4", "视频", "video", "国产", "国内"],
    "Vidu Q3": ["Vidu", "Vidu Q3", "生数科技", "视频", "video", "国产", "国内"],
}

FOCUS_GROUPS = {
    "open_source": {
        "query": ["开源", "开放权重", "open source", "open-weight", "open weight", "apache"],
        "event": ["开源", "开放权重", "open source", "apache", "hugging face"],
    },
    "video": {
        "query": ["视频", "video"],
        "event": ["视频", "video", "seedance", "skyreels", "vidu", "sora"],
    },
    "image": {
        "query": ["图像", "图片", "生图", "image", "vision", "视觉"],
        "event": ["图像", "图片", "生图", "image", "vision", "midjourney", "mai-image"],
    },
    "coding": {
        "query": ["代码", "coding", "编程"],
        "event": ["代码", "coding", "编程", "cursor", "composer", "codex"],
    },
    "reasoning": {
        "query": ["推理", "reasoning", "证明", "prover"],
        "event": ["推理", "reasoning", "证明", "prover", "longcat", "nemotron", "grok"],
    },
    "domestic": {
        "query": ["国内", "国产", "阿里", "百度", "腾讯", "字节", "智谱", "小米", "美团"],
        "event": ["国产", "国内", "阿里", "通义", "qwen", "deepseek", "智谱", "glm", "minimax", "小米", "mimo", "美团", "longcat", "字节", "seedance", "skyreels", "vidu"],
    },
    "head_vendor": {
        "query": ["头部厂商", "openai", "google", "anthropic", "meta", "xai", "microsoft", "amazon", "阿里", "腾讯", "百度"],
        "event": ["openai", "google", "gemini", "meta", "llama", "xai", "grok", "microsoft", "微软", "qwen", "阿里", "通义"],
    },
}

QUERY_COMMON_TERMS = [
    "2026", "3月", "3", "发布", "推出", "上线", "新", "新品", "消息", "新闻", "资讯",
    "本地", "数据库", "site", "or", "and", "模型", "大模型", "llm", "foundation model",
    "正式", "版本", "更新", "能力", "升级", "参数", "规模", "月报",
]

BROAD_HINT_TERMS = {
    "开源", "开放权重", "open source", "open-weight", "apache", "国产", "国内",
    "多模态", "视频", "video", "图像", "图片", "生图", "image", "vision", "视觉",
    "推理", "reasoning", "代码", "coding", "编程", "智能体", "agent", "数学", "证明",
}


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def harmonic(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def format_pct(value: float) -> str:
    return f"{value:.1%}"


def ordered_unique(items: list[int]) -> list[int]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def normalize_query(text: str) -> str:
    text = text.lower()
    return re.sub(r"[\s\-_，。,.；;:：/\\]+", "", text)


def char_ngrams(text: str, n: int = 3) -> set[str]:
    text = normalize_query(text)
    if len(text) <= n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def load_ground_truth(eval_dir: Path) -> tuple[dict[int, dict], dict[str, str], dict[str, set[int]]]:
    labels_path = eval_dir / "article_labels_v2.json"
    event_path = eval_dir / "event_to_articles.json"
    labels_raw = load_json(labels_path, [])
    event_index = load_json(event_path, {})
    labels = {int(item["article_id"]): item for item in labels_raw}
    event_articles: dict[str, set[int]] = {}
    for aid_str, event_name in event_index.items():
        event_articles.setdefault(event_name, set()).add(int(aid_str))
    return labels, event_index, event_articles


def ids_to_events(article_ids: list[int] | set[int], event_index: dict[str, str]) -> set[str]:
    events = set()
    for aid in article_ids:
        event_name = event_index.get(str(aid))
        if event_name:
            events.add(event_name)
    return events


def stage_article_ids(retrieval_details: list[dict], stage: str) -> list[int]:
    ids: list[int] = []
    for detail in retrieval_details:
        ids.extend(int(aid) for aid in detail.get(stage, {}).get("article_ids", []) if aid)
    return ordered_unique(ids)


def build_event_catalog(
    labels: dict[int, dict],
    event_articles: dict[str, set[int]],
) -> dict[str, dict]:
    catalog = {}
    for event_name, article_ids in event_articles.items():
        parts = [event_name]
        aliases = set(EVENT_HINTS.get(event_name, []))
        aliases.add(event_name)
        for aid in article_ids:
            item = labels.get(aid, {})
            parts.extend([
                str(item.get("title", "")),
                str(item.get("gold_evidence", "")),
                str(item.get("canonical_name", "") or ""),
            ])
            for value in item.get("entities", []) or []:
                parts.append(str(value))
            for value in item.get("aliases", []) or []:
                aliases.add(str(value))
                parts.append(str(value))
        for hint in EVENT_HINTS.get(event_name, []):
            parts.append(hint)
        catalog[event_name] = {
            "article_ids": set(article_ids),
            "aliases": sorted(alias for alias in aliases if alias),
            "hint_aliases": sorted(
                hint for hint in EVENT_HINTS.get(event_name, [])
                if hint and hint.lower() not in BROAD_HINT_TERMS
            ),
            "search_text": " ".join(part for part in parts if part).lower(),
            "search_text_norm": normalize_query(" ".join(part for part in parts if part)),
        }
    return catalog


def is_excluded_term(query: str, term: str) -> bool:
    term = term.strip()
    if not term:
        return False
    escaped = re.escape(term)
    return bool(re.search(rf"[-－]\s*{escaped}", query, flags=re.IGNORECASE))


def is_broad_alias(term: str) -> bool:
    term_lower = term.lower().strip()
    term_norm = normalize_query(term)
    return (
        term_lower in BROAD_HINT_TERMS
        or term_norm in {normalize_query(item) for item in BROAD_HINT_TERMS}
    )


def is_valid_alias(term: str) -> bool:
    term_norm = normalize_query(term)
    if re.search(r"[\u4e00-\u9fff]", term):
        return len(term_norm) >= 2
    return len(term_norm) >= 3


def event_alias_score(query: str, event_data: dict) -> float:
    q = query.lower()
    q_norm = normalize_query(query)
    score = 0.0
    for alias in sorted(set([*event_data.get("aliases", []), *event_data.get("hint_aliases", [])])):
        if is_broad_alias(alias):
            continue
        alias_lower = alias.lower()
        alias_norm = normalize_query(alias)
        if not is_valid_alias(alias):
            continue
        if is_excluded_term(query, alias):
            continue
        if alias_lower in q or alias_norm in q_norm:
            score += 3.0 if len(alias_norm) >= 5 else 1.5
    return score


def query_has_focus(query: str) -> bool:
    q = query.lower()
    for group in FOCUS_GROUPS.values():
        if any(term.lower() in q for term in group["query"]):
            return True
    return False


def query_has_event_alias(query: str, catalog: dict[str, dict]) -> bool:
    for item in catalog.values():
        if event_alias_score(query, item) > 0:
            return True
    return False


def should_use_all_events(query: str, catalog: dict[str, dict]) -> bool:
    q = query.lower()
    broad_markers = ["新大模型", "大模型相关新闻", "有哪些新的大模型", "旗舰基础模型"]
    if not any(marker in q for marker in broad_markers):
        return False
    return not query_has_focus(query) and not query_has_event_alias(query, catalog)


def score_event_for_query(query: str, event_data: dict) -> float:
    q = query.lower()
    q_norm = normalize_query(query)
    event_text = event_data["search_text"]
    event_text_norm = event_data["search_text_norm"]
    score = 0.0

    score += event_alias_score(query, event_data)

    for group in FOCUS_GROUPS.values():
        query_match = any(term.lower() in q for term in group["query"])
        if not query_match:
            continue
        event_match = any(
            term.lower() in event_text or normalize_query(term) in event_text_norm
            for term in group["event"]
        )
        if event_match:
            score += 1.25

    meaningful_terms = [
        term for term in re.findall(r"[A-Za-z][A-Za-z0-9.\-]+|[\u4e00-\u9fff]{2,}", query)
        if not any(common.lower() == term.lower() for common in QUERY_COMMON_TERMS)
    ]
    for term in meaningful_terms:
        term_lower = term.lower()
        term_norm = normalize_query(term)
        if len(term_norm) < 3:
            continue
        if term_lower in event_text or term_norm in event_text_norm:
            score += 0.5

    return score


def query_ground_truth(
    query: str,
    catalog: dict[str, dict],
    event_articles: dict[str, set[int]],
) -> dict:
    if should_use_all_events(query, catalog):
        selected = sorted(event_articles)
        return {
            "strategy": "broad_query_all_events",
            "expected_events": selected,
            "expected_articles": sorted(
                aid for event_name in selected for aid in event_articles[event_name]
            ),
            "event_scores": {event_name: 1.0 for event_name in selected},
        }

    alias_scores = {
        event_name: event_alias_score(query, event_data)
        for event_name, event_data in catalog.items()
    }
    alias_selected = sorted(event_name for event_name, score in alias_scores.items() if score > 0)
    if alias_selected:
        return {
            "strategy": "query_alias_matched_events",
            "expected_events": alias_selected,
            "expected_articles": sorted(
                aid for event_name in alias_selected for aid in event_articles[event_name]
            ),
            "event_scores": {
                event_name: score
                for event_name, score in sorted(alias_scores.items())
                if score > 0
            },
        }

    event_scores = {
        event_name: score_event_for_query(query, event_data)
        for event_name, event_data in catalog.items()
    }
    selected = sorted(event_name for event_name, score in event_scores.items() if score >= 1.0)

    if not selected:
        # 没能识别出 query 的目标事件时退回全量 GT，避免虚高。
        selected = sorted(event_articles)
        strategy = "fallback_all_events"
    else:
        strategy = "query_matched_events"

    return {
        "strategy": strategy,
        "expected_events": selected,
        "expected_articles": sorted(
            aid for event_name in selected for aid in event_articles[event_name]
        ),
        "event_scores": {
            event_name: score
            for event_name, score in sorted(event_scores.items())
            if score > 0
        },
    }


def build_query_ground_truths(
    retrieval_details: list[dict],
    labels: dict[int, dict],
    event_articles: dict[str, set[int]],
) -> list[dict]:
    catalog = build_event_catalog(labels, event_articles)
    return [
        query_ground_truth(detail.get("query", ""), catalog, event_articles)
        for detail in retrieval_details
    ]


def compute_ndcg(article_ids: list[int], relevant_articles: set[int], k: int) -> float:
    if not article_ids:
        return 0.0
    ranked = article_ids[:k]
    dcg = 0.0
    for idx, aid in enumerate(ranked):
        rel = 1 if aid in relevant_articles else 0
        dcg += rel / math.log2(idx + 2)

    ideal_hits = min(k, len(relevant_articles))
    idcg = sum(1 / math.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def load_sub_queries(run_dir: Path) -> list[dict]:
    for name in ("sub_queries.json", "rag_sub_queries.json"):
        sub_queries = load_json(run_dir / name, [])
        if sub_queries:
            return sub_queries

    rag_outputs = load_json(run_dir / "rag_outputs.json", [])
    merged = []
    for output in rag_outputs:
        merged.extend(output.get("sub_queries", []))
    return merged


def collect_raw_result_texts(run_dir: Path) -> list[str]:
    raw_results = load_json(run_dir / "raw_results.json", [])
    if isinstance(raw_results, list) and raw_results:
        return [str(item) for item in raw_results]

    texts: list[str] = []
    rag_outputs = load_json(run_dir / "rag_outputs.json", [])
    for output in rag_outputs:
        texts.extend(str(item) for item in output.get("raw_results", []))
    return texts


def parse_result_text(text: str) -> list[dict]:
    pattern = re.compile(
        r"--- 结果\s+(\d+)\s+\[(.*?)\]\s+---\s*"
        r"ArticleID:\s*(\d+)\s*"
        r"标题:\s*(.*?)\s*"
        r"元数据:\s*(.*?)\s*"
        r"Rerank分数:\s*([-\d.]+)\s*"
        r"预览:\s*(.*?)(?=\n--- 结果\s+\d+\s+\[|\Z)",
        re.DOTALL,
    )
    items = []
    for match in pattern.finditer(text):
        items.append({
            "rank": int(match.group(1)),
            "source": match.group(2).strip(),
            "article_id": int(match.group(3)),
            "title": match.group(4).strip(),
            "metadata": match.group(5).strip(),
            "rerank_score": float(match.group(6)),
            "preview": re.sub(r"\s+", " ", match.group(7)).strip(),
        })
    return items


def build_result_items_by_query(run_dir: Path, retrieval_details: list[dict]) -> list[list[dict]]:
    raw_texts = collect_raw_result_texts(run_dir)
    parsed = [parse_result_text(text) for text in raw_texts]
    rows: list[list[dict]] = []
    for idx, detail in enumerate(retrieval_details):
        if idx < len(parsed) and parsed[idx]:
            rows.append(parsed[idx])
            continue

        ids = detail.get("reranked", {}).get("article_ids", [])
        scores = detail.get("reranked", {}).get("scores", [])
        rows.append([
            {
                "rank": rank,
                "article_id": int(aid),
                "title": "",
                "metadata": "",
                "preview": "",
                "rerank_score": float(scores[rank - 1]) if rank - 1 < len(scores) else 0.0,
            }
            for rank, aid in enumerate(ids, start=1)
        ])
    return rows


def extract_json_object(text: str) -> dict:
    text = str(text).strip()
    text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def call_llm_json(messages: list[Any]) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 未配置")

    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    model_name = os.getenv("SYNTHETIC_EVAL_MODEL", "gpt-5.4-mini-ca")
    kwargs = {
        "model": model_name,
        "temperature": 0,
        "api_key": api_key,
    }
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    model = ChatOpenAI(**kwargs)

    converted = []
    for role, content in messages:
        if role == "system":
            converted.append(SystemMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    response = model.invoke(converted)
    return extract_json_object(str(response.content))


def synthesize_rubrics(topic: str, business_context: str, output_path: Path) -> dict:
    if not business_context.strip():
        rubric = dict(DEFAULT_RUBRICS)
        rubric["synthesis_note"] = "未提供业务访谈记录，使用默认 rubric。"
        write_json(output_path, rubric)
        return rubric

    prompt = f"""
你是 RAG 评测负责人。请根据业务访谈记录，为四节点评测生成 rubric。

任务主题：
{topic}

业务访谈/聊天记录：
{business_context}

只输出 JSON，不要 Markdown。JSON 格式必须是：
{{
  "query_rewrite": {{
    "coverage_dimensions": [
      {{"name":"维度名","description":"为什么重要","keywords":["可匹配 query 的关键词"]}}
    ],
    "weights": {{"coverage":0.35,"constraint_accuracy":0.20,"diversity":0.20,"rerank_feedback":0.25}}
  }},
  "rerank": {{
    "dimensions": [
      {{"key":"impact","name":"实际影响力","weight":0.65,"description":"评分标准"}},
      {{"key":"prominence","name":"主体头部性","weight":0.50,"description":"评分标准"}},
      {{"key":"heat","name":"报道热度","weight":0.20,"description":"评分标准"}},
      {{"key":"controversy","name":"争议性","weight":0.10,"description":"评分标准"}},
      {{"key":"penalty_score","name":"来源灌水惩罚","weight":-0.75,"description":"评分标准"}}
    ],
    "weights": {{"impact":0.65,"prominence":0.50,"heat":0.20,"controversy":0.10,"penalty_score":0.75}},
    "formula": "CommonScore = 0.65*impact + 0.50*prominence + 0.20*heat + 0.10*controversy; FinalScore = CommonScore - 0.75*penalty_score"
  }}
}}

要求：
1. coverage_dimensions 必须来自业务关注点，不要只写抽象词。
2. keywords 要能用于自动匹配 query。
3. rerank 必须沿用 impact/prominence/heat/controversy/source penalty 公式，只允许改写维度说明和关键词，不要改键名。
"""
    try:
        rubric = call_llm_json([
            ("system", "只输出可解析 JSON。"),
            ("human", prompt),
        ])
        if "query_rewrite" not in rubric or "rerank" not in rubric:
            raise ValueError("rubric 缺少 query_rewrite 或 rerank")
        write_json(output_path, rubric)
        return rubric
    except Exception as exc:
        rubric = dict(DEFAULT_RUBRICS)
        rubric["synthesis_error"] = str(exc)
        write_json(output_path, rubric)
        return rubric


def load_rubrics(args: argparse.Namespace, run_dir: Path) -> dict:
    def normalize_rubric(rubric: dict) -> dict:
        if "query_rewrite" not in rubric:
            rubric["query_rewrite"] = DEFAULT_RUBRICS["query_rewrite"]
        rerank_weights = rubric.get("rerank", {}).get("weights", {})
        if "impact" not in rerank_weights:
            rubric["rerank"] = DEFAULT_RUBRICS["rerank"]
        return rubric

    if args.rubric_path:
        return normalize_rubric(load_json(Path(args.rubric_path), DEFAULT_RUBRICS))

    output_path = run_dir / "four_node_rubrics.json"
    if args.synthesize_rubrics:
        business_context = ""
        if args.business_context_path:
            business_context = Path(args.business_context_path).read_text(encoding="utf-8")
        return normalize_rubric(synthesize_rubrics(args.topic, business_context, output_path))

    if output_path.exists():
        return normalize_rubric(load_json(output_path, DEFAULT_RUBRICS))
    return normalize_rubric(dict(DEFAULT_RUBRICS))


def score_query_rewrite_with_llm(
    topic: str,
    sub_queries: list[dict],
    retrieval_feedback: dict,
    rubric: dict,
) -> dict:
    prompt = {
        "topic": topic,
        "rubric": rubric.get("query_rewrite", {}),
        "sub_queries": sub_queries,
        "retrieval_feedback": retrieval_feedback,
        "instruction": (
            "请按 coverage、constraint_accuracy、diversity、rerank_feedback 四项各打 0-5 分，"
            "并给出 overall_score_100。只输出 JSON。"
        ),
    }
    return call_llm_json([
        ("system", "你是严谨的 RAG Query Rewrite 评测员。只输出可解析 JSON。"),
        ("human", json.dumps(prompt, ensure_ascii=False)),
    ])


def evaluate_query_rewrite(
    topic: str,
    sub_queries: list[dict],
    retrieval_details: list[dict],
    rubric: dict,
    expected_start_date: str,
    expected_end_date: str,
    expected_category: str,
    high_score_threshold: float,
    judge_with_llm: bool,
) -> dict:
    if not sub_queries:
        sub_queries = [{"query": detail.get("query", "")} for detail in retrieval_details]

    query_texts = [
        f"{q.get('search_intent', '')} {q.get('query', '')}".strip()
        for q in sub_queries
    ]
    all_query_text = "\n".join(query_texts).lower()

    coverage_dimensions = rubric.get("query_rewrite", {}).get(
        "coverage_dimensions",
        DEFAULT_RUBRICS["query_rewrite"]["coverage_dimensions"],
    )
    covered_dimensions = []
    missed_dimensions = []
    for dim in coverage_dimensions:
        keywords = [str(k).lower() for k in dim.get("keywords", [])]
        covered = any(keyword and keyword in all_query_text for keyword in keywords)
        if covered:
            covered_dimensions.append(dim)
        else:
            missed_dimensions.append(dim)
    coverage_rate = safe_div(len(covered_dimensions), len(coverage_dimensions))

    date_checks = [
        q.get("start_date") == expected_start_date and q.get("end_date") == expected_end_date
        for q in sub_queries
        if "start_date" in q or "end_date" in q
    ]
    category_checks = [
        q.get("category", "") == expected_category
        for q in sub_queries
        if "category" in q
    ]
    release_focus = [
        any(term.lower() in text.lower() for term in RELEASE_TERMS)
        for text in query_texts
    ]
    constraint_parts = [
        safe_div(sum(date_checks), len(date_checks)) if date_checks else None,
        safe_div(sum(category_checks), len(category_checks)) if category_checks else None,
        safe_div(sum(release_focus), len(release_focus)) if release_focus else None,
    ]
    constraint_values = [part for part in constraint_parts if part is not None]
    constraint_accuracy = sum(constraint_values) / len(constraint_values) if constraint_values else 0.0

    duplicate_pairs = []
    grams = [char_ngrams(text) for text in query_texts]
    for i in range(len(grams)):
        for j in range(i + 1, len(grams)):
            sim = jaccard(grams[i], grams[j])
            if sim >= 0.75:
                duplicate_pairs.append({
                    "left": i,
                    "right": j,
                    "similarity": sim,
                    "left_query": query_texts[i],
                    "right_query": query_texts[j],
                })
    duplicate_rate = safe_div(len(duplicate_pairs), max(1, len(query_texts) * (len(query_texts) - 1) / 2))
    diversity_score = 1.0 - duplicate_rate

    seen_high_docs: set[int] = set()
    query_feedback_rows = []
    all_scores: list[float] = []
    high_score_hits = 0
    reranked_count = 0
    marginal_high_docs = 0
    for detail in retrieval_details:
        ids = [int(aid) for aid in detail.get("reranked", {}).get("article_ids", []) if aid]
        scores = [float(score) for score in detail.get("reranked", {}).get("scores", [])]
        pairs = list(zip(ids, scores))
        high_pairs = [(aid, score) for aid, score in pairs if score >= high_score_threshold]
        new_high = [aid for aid, _ in high_pairs if aid not in seen_high_docs]
        seen_high_docs.update(aid for aid, _ in high_pairs)

        all_scores.extend(scores)
        high_score_hits += len(high_pairs)
        reranked_count += len(scores)
        marginal_high_docs += len(new_high)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        query_feedback_rows.append({
            "query": detail.get("query", ""),
            "reranked_count": len(scores),
            "avg_rerank_score": avg_score,
            "max_rerank_score": max(scores) if scores else 0.0,
            "high_score_count": len(high_pairs),
            "high_score_rate": safe_div(len(high_pairs), len(scores)),
            "new_high_score_doc_count": len(new_high),
        })

    avg_rerank_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    high_score_rate = safe_div(high_score_hits, reranked_count)
    marginal_high_doc_rate = safe_div(marginal_high_docs, max(1, high_score_hits))
    rerank_feedback = clamp(0.55 * avg_rerank_score + 0.30 * high_score_rate + 0.15 * marginal_high_doc_rate)

    weights = rubric.get("query_rewrite", {}).get("weights", QUERY_REWRITE_WEIGHTS)
    dimension_scores = {
        "coverage": coverage_rate * 5,
        "constraint_accuracy": constraint_accuracy * 5,
        "diversity": diversity_score * 5,
        "rerank_feedback": rerank_feedback * 5,
    }
    overall_score = sum(
        safe_div(dimension_scores[key], 5.0) * float(weights.get(key, 0.0))
        for key in dimension_scores
    ) * 100
    precision = constraint_accuracy * diversity_score
    recall = coverage_rate
    f1 = harmonic(precision, recall)

    result = {
        "mode": "heuristic_rubric",
        "sub_query_count": len(sub_queries),
        "rubric_dimension_count": len(coverage_dimensions),
        "covered_dimension_count": len(covered_dimensions),
        "missed_dimension_count": len(missed_dimensions),
        "covered_dimensions": covered_dimensions,
        "missed_dimensions": missed_dimensions,
        "recall_proxy": recall,
        "precision_proxy": precision,
        "f1_proxy": f1,
        "coverage_rate": coverage_rate,
        "constraint_accuracy": constraint_accuracy,
        "date_accuracy": safe_div(sum(date_checks), len(date_checks)) if date_checks else None,
        "category_accuracy": safe_div(sum(category_checks), len(category_checks)) if category_checks else None,
        "release_focus_rate": safe_div(sum(release_focus), len(release_focus)) if release_focus else 0.0,
        "duplicate_pair_count": len(duplicate_pairs),
        "duplicate_rate": duplicate_rate,
        "duplicate_pairs": duplicate_pairs[:20],
        "avg_rerank_score": avg_rerank_score,
        "high_score_threshold": high_score_threshold,
        "high_score_rate": high_score_rate,
        "marginal_high_doc_rate": marginal_high_doc_rate,
        "dimension_scores_0_5": dimension_scores,
        "overall_score_100": overall_score,
        "query_feedback": query_feedback_rows,
    }

    if judge_with_llm:
        try:
            llm_result = score_query_rewrite_with_llm(
                topic,
                sub_queries,
                {
                    "avg_rerank_score": avg_rerank_score,
                    "high_score_rate": high_score_rate,
                    "marginal_high_doc_rate": marginal_high_doc_rate,
                    "query_feedback": query_feedback_rows[:30],
                },
                rubric,
            )
            result["mode"] = "llm_rubric"
            result["llm_judge"] = llm_result
            if isinstance(llm_result.get("overall_score_100"), (int, float)):
                result["overall_score_100"] = float(llm_result["overall_score_100"])
        except Exception as exc:
            result["llm_judge_error"] = str(exc)

    return result


def evaluate_retrieval_node(
    retrieval_details: list[dict],
    stage: str,
    event_index: dict[str, str],
    event_articles: dict[str, set[int]],
    query_gts: list[dict],
    ks: list[int],
) -> dict:
    global_relevant_articles = {int(aid) for aid in event_index}

    def score_ids(
        article_ids: list[int],
        expected_events: set[str],
        expected_articles: set[int],
    ) -> dict:
        retrieved_set = set(article_ids)
        relevant_hits = retrieved_set & expected_articles
        hit_events = ids_to_events(article_ids, event_index)
        expected_hit_events = hit_events & expected_events
        article_precision = safe_div(len(relevant_hits), len(retrieved_set))
        article_recall = safe_div(len(relevant_hits), len(expected_articles))
        metrics = {
            "count": len(article_ids),
            "event_recall": safe_div(len(expected_hit_events), len(expected_events)),
            "article_precision": article_precision,
            "article_recall": article_recall,
            "article_f1": harmonic(article_precision, article_recall),
            "hit_event_count": len(expected_hit_events),
            "total_event_count": len(expected_events),
            "hit_events": sorted(expected_hit_events),
            "missed_events": sorted(expected_events - expected_hit_events),
            "at_k": {},
        }
        for k in ks:
            top_ids = article_ids[:k]
            top_set = set(top_ids)
            top_relevant = top_set & expected_articles
            top_events = ids_to_events(top_ids, event_index)
            top_expected_events = top_events & expected_events
            top_precision = safe_div(len(top_relevant), len(top_set))
            top_recall = safe_div(len(top_relevant), len(expected_articles))
            metrics["at_k"][str(k)] = {
                "count": len(top_ids),
                "event_recall": safe_div(len(top_expected_events), len(expected_events)),
                "article_precision": top_precision,
                "article_recall": top_recall,
                "article_f1": harmonic(top_precision, top_recall),
                "ndcg": compute_ndcg(article_ids, expected_articles, k),
                "hit_events": sorted(top_expected_events),
            }
        return metrics

    per_execute = []
    for index, detail in enumerate(retrieval_details, start=1):
        article_ids = [
            int(aid)
            for aid in detail.get(stage, {}).get("article_ids", [])
            if aid
        ]
        query_gt = query_gts[index - 1] if index - 1 < len(query_gts) else {}
        expected_events = set(query_gt.get("expected_events", event_articles.keys()))
        expected_articles = set(query_gt.get("expected_articles", global_relevant_articles))
        row = score_ids(article_ids, expected_events, expected_articles)
        row["execute_index"] = index
        row["query"] = detail.get("query", "")
        row["gt_strategy"] = query_gt.get("strategy", "")
        row["expected_events"] = sorted(expected_events)
        row["expected_event_count"] = len(expected_events)
        row["expected_article_count"] = len(expected_articles)
        per_execute.append(row)

    union_ids = stage_article_ids(retrieval_details, stage)
    union_metrics = score_ids(union_ids, set(event_articles), global_relevant_articles)

    macro = {
        "execute_count": len(per_execute),
        "avg_count": sum(row["count"] for row in per_execute) / len(per_execute)
        if per_execute else 0.0,
        "avg_expected_event_count": (
            sum(row["expected_event_count"] for row in per_execute) / len(per_execute)
            if per_execute else 0.0
        ),
        "avg_expected_article_count": (
            sum(row["expected_article_count"] for row in per_execute) / len(per_execute)
            if per_execute else 0.0
        ),
        "event_recall": sum(row["event_recall"] for row in per_execute) / len(per_execute)
        if per_execute else 0.0,
        "article_precision": sum(row["article_precision"] for row in per_execute) / len(per_execute)
        if per_execute else 0.0,
        "article_recall": sum(row["article_recall"] for row in per_execute) / len(per_execute)
        if per_execute else 0.0,
        "article_f1": sum(row["article_f1"] for row in per_execute) / len(per_execute)
        if per_execute else 0.0,
        "hit_event_count": sum(row["hit_event_count"] for row in per_execute) / len(per_execute)
        if per_execute else 0.0,
        "at_k": {},
    }
    for k in ks:
        key = str(k)
        macro["at_k"][key] = {
            "event_recall": sum(row["at_k"][key]["event_recall"] for row in per_execute) / len(per_execute)
            if per_execute else 0.0,
            "article_precision": sum(row["at_k"][key]["article_precision"] for row in per_execute) / len(per_execute)
            if per_execute else 0.0,
            "article_recall": sum(row["at_k"][key]["article_recall"] for row in per_execute) / len(per_execute)
            if per_execute else 0.0,
            "article_f1": sum(row["at_k"][key]["article_f1"] for row in per_execute) / len(per_execute)
            if per_execute else 0.0,
            "ndcg": sum(row["at_k"][key]["ndcg"] for row in per_execute) / len(per_execute)
            if per_execute else 0.0,
        }

    return {
        "stage": stage,
        "aggregation": "macro_average_by_execute",
        **macro,
        "per_execute": per_execute,
        "union": union_metrics,
    }


def item_text(item: dict) -> str:
    return f"{item.get('title', '')} {item.get('preview', '')} {item.get('metadata', '')}".lower()


def keyword_rate(items: list[dict], keywords: list[str]) -> float:
    if not items:
        return 0.0
    hits = 0
    for item in items:
        text = item_text(item)
        if any(keyword.lower() in text for keyword in keywords):
            hits += 1
    return hits / len(items)


def item_source(item: dict) -> str:
    metadata = str(item.get("metadata", ""))
    match = re.search(r"\|\s*\[(.*?)\]", metadata)
    if match:
        return match.group(1).strip()
    return str(item.get("source", "")).strip()


def normalized_rerank_score(score: float) -> float:
    if 0.0 <= score <= 1.0:
        return score * 5.0
    return 5.0 / (1.0 + math.exp(-score))


def score_item_impact(item: dict) -> float:
    text = item_text(item)
    score = normalized_rerank_score(float(item.get("rerank_score", 0.0)))
    if any(term.lower() in text for term in ACTION_RELEASE_TERMS):
        score += 0.35
    if any(term.lower() in text for term in EVIDENCE_TERMS):
        score += 0.20
    if any(any(hint.lower() in text for hint in hints) for hints in EVENT_HINTS.values()):
        score += 0.20
    if any(term.lower() in text for term in NOISE_TERMS) and not any(
        term.lower() in text for term in ACTION_RELEASE_TERMS
    ):
        score -= 0.70
    if any(term.lower() in text for term in HARD_NOISE_TERMS):
        score -= 0.55
    return clamp(score, 0.0, 5.0)


def score_item_prominence(item: dict, impact: float) -> float:
    text = item_text(item)
    if any(term.lower() in text for term in TIER1_VENDOR_TERMS):
        score = 5.0
    elif any(term.lower() in text for term in TIER2_VENDOR_TERMS):
        score = 3.5
    else:
        score = 0.0
    return min(score, impact + 1.0)


def item_event_key(item: dict) -> str:
    text = item_text(item)
    for event_name, hints in EVENT_HINTS.items():
        if any(hint.lower() in text for hint in hints):
            return event_name
    return ""


def score_item_controversy(item: dict) -> float:
    text = item_text(item)
    hits = sum(1 for term in CONTROVERSY_TERMS if term.lower() in text)
    return min(5.0, hits * 1.25)


def source_penalty(items: list[dict]) -> dict[str, float]:
    sources = [item_source(item) for item in items if item_source(item)]
    counts = Counter(sources)
    total = len(items) or 1
    penalties = {}
    for source, count in counts.items():
        share = count / total
        if count <= 3 and share <= 0.25:
            penalties[source] = 0.0
            continue
        penalties[source] = min(2.0, max(0.0, (share - 0.25) * 5.0 + max(0, count - 8) * 0.08))
    return penalties


def score_heat(items: list[dict]) -> dict[int, float]:
    event_keys = [item_event_key(item) for item in items]
    counts = Counter(key for key in event_keys if key)
    heat_by_index = {}
    for index, key in enumerate(event_keys):
        count = counts.get(key, 1) if key else 1
        if count <= 1:
            heat_by_index[index] = 0.0
        else:
            heat_by_index[index] = min(5.0, 1.0 + 1.2 * (count.bit_length() - 1))
    return heat_by_index


def score_rerank_heuristic(items_by_query: list[list[dict]], high_score_threshold: float) -> dict:
    query_scores = []
    for items in items_by_query:
        if not items:
            continue
        top_items = items[: min(10, len(items))]
        raw_scores = [float(item.get("rerank_score", 0.0)) for item in top_items]
        avg_score = sum(raw_scores) / len(raw_scores) if raw_scores else 0.0
        high_score_rate = safe_div(sum(1 for score in raw_scores if score >= high_score_threshold), len(raw_scores))

        impacts = [score_item_impact(item) for item in top_items]
        prominences = [
            score_item_prominence(item, impact)
            for item, impact in zip(top_items, impacts)
        ]
        heat_by_index = score_heat(top_items)
        heats = [heat_by_index[index] for index in range(len(top_items))]
        controversies = [score_item_controversy(item) for item in top_items]
        penalties_by_source = source_penalty(top_items)
        penalties = [penalties_by_source.get(item_source(item), 0.0) for item in top_items]

        dimension_scores = {
            "impact": sum(impacts) / len(impacts) if impacts else 0.0,
            "prominence": sum(prominences) / len(prominences) if prominences else 0.0,
            "heat": sum(heats) / len(heats) if heats else 0.0,
            "controversy": sum(controversies) / len(controversies) if controversies else 0.0,
            "penalty_score": sum(penalties) / len(penalties) if penalties else 0.0,
        }
        common_score = (
            RERANK_WEIGHTS["impact"] * dimension_scores["impact"]
            + RERANK_WEIGHTS["prominence"] * dimension_scores["prominence"]
            + RERANK_WEIGHTS["heat"] * dimension_scores["heat"]
            + RERANK_WEIGHTS["controversy"] * dimension_scores["controversy"]
        )
        final_score = common_score - RERANK_WEIGHTS["penalty_score"] * dimension_scores["penalty_score"]
        overall = clamp(final_score / RERANK_MAX_COMMON_SCORE) * 100
        query_scores.append({
            "article_count": len(top_items),
            "avg_rerank_score": avg_score,
            "high_score_rate": high_score_rate,
            "dimension_scores_0_5": dimension_scores,
            "common_score": common_score,
            "final_score": final_score,
            "overall_score_100": overall,
        })

    if not query_scores:
        return {
            "mode": "heuristic_rubric",
            "overall_score_100": 0.0,
            "dimension_scores_0_5": {key: 0.0 for key in RERANK_WEIGHTS},
            "formula": DEFAULT_RUBRICS["rerank"]["formula"],
            "query_scores": [],
        }

    avg_dimensions = {
        key: sum(row["dimension_scores_0_5"][key] for row in query_scores) / len(query_scores)
        for key in RERANK_WEIGHTS
    }
    common_score = (
        RERANK_WEIGHTS["impact"] * avg_dimensions["impact"]
        + RERANK_WEIGHTS["prominence"] * avg_dimensions["prominence"]
        + RERANK_WEIGHTS["heat"] * avg_dimensions["heat"]
        + RERANK_WEIGHTS["controversy"] * avg_dimensions["controversy"]
    )
    final_score = common_score - RERANK_WEIGHTS["penalty_score"] * avg_dimensions["penalty_score"]
    overall = clamp(final_score / RERANK_MAX_COMMON_SCORE) * 100
    return {
        "mode": "heuristic_rubric",
        "overall_score_100": overall,
        "dimension_scores_0_5": avg_dimensions,
        "formula": DEFAULT_RUBRICS["rerank"]["formula"],
        "common_score": common_score,
        "final_score": final_score,
        "query_scores": query_scores,
    }


def score_rerank_with_llm(
    topic: str,
    retrieval_details: list[dict],
    items_by_query: list[list[dict]],
    rubric: dict,
    judge_limit: int,
) -> dict:
    cases = []
    for detail, items in zip(retrieval_details[:judge_limit], items_by_query[:judge_limit]):
        cases.append({
            "query": detail.get("query", ""),
            "top_results": [
                {
                    "rank": item.get("rank"),
                    "article_id": item.get("article_id"),
                    "title": item.get("title", ""),
                    "preview": item.get("preview", ""),
                    "rerank_score": item.get("rerank_score", 0.0),
                }
                for item in items[:10]
            ],
        })

    prompt = {
        "topic": topic,
        "rubric": rubric.get("rerank", DEFAULT_RUBRICS["rerank"]),
        "cases": cases,
        "instruction": (
            "请对每个 query 的 rerank TopK 按 rubric 评分。每个维度 0-5 分，"
            "按公式计算 overall_score_100，并给出简短问题诊断。只输出 JSON。"
        ),
        "output_schema": {
            "overall_score_100": 0,
            "dimension_scores_0_5": {
                "impact": 0,
                "prominence": 0,
                "heat": 0,
                "controversy": 0,
                "penalty_score": 0,
            },
            "query_scores": [],
            "diagnosis": "",
        },
    }
    return call_llm_json([
        ("system", "你是严谨的 RAG Rerank 评测员。只输出可解析 JSON。"),
        ("human", json.dumps(prompt, ensure_ascii=False)),
    ])


def evaluate_rerank_node(
    topic: str,
    run_dir: Path,
    retrieval_details: list[dict],
    event_index: dict[str, str],
    event_articles: dict[str, set[int]],
    query_gts: list[dict],
    rubric: dict,
    ks: list[int],
    high_score_threshold: float,
    judge_with_llm: bool,
    judge_limit: int,
) -> dict:
    items_by_query = build_result_items_by_query(run_dir, retrieval_details)
    heuristic = score_rerank_heuristic(items_by_query, high_score_threshold)

    relevant_articles = {int(aid) for aid in event_index}
    per_execute_reference = []
    for index, detail in enumerate(retrieval_details, start=1):
        query_gt = query_gts[index - 1] if index - 1 < len(query_gts) else {}
        expected_events = set(query_gt.get("expected_events", event_articles.keys()))
        expected_articles = set(query_gt.get("expected_articles", relevant_articles))
        merged_ids = [
            int(aid)
            for aid in detail.get("merged", {}).get("article_ids", [])
            if aid
        ]
        reranked_ids = [
            int(aid)
            for aid in detail.get("reranked", {}).get("article_ids", [])
            if aid
        ]
        merged_events = ids_to_events(merged_ids, event_index)
        reranked_events = ids_to_events(reranked_ids, event_index)
        expected_merged_events = merged_events & expected_events
        expected_reranked_events = reranked_events & expected_events
        reranked_set = set(reranked_ids)
        precision = safe_div(len(reranked_set & expected_articles), len(reranked_set))
        per_execute_reference.append({
            "execute_index": index,
            "query": detail.get("query", ""),
            "gt_strategy": query_gt.get("strategy", ""),
            "expected_events": sorted(expected_events),
            "expected_event_count": len(expected_events),
            "expected_article_count": len(expected_articles),
            "merged_count": len(merged_ids),
            "reranked_count": len(reranked_ids),
            "merged_event_count": len(expected_merged_events),
            "reranked_event_count": len(expected_reranked_events),
            "event_retention_from_merged": safe_div(
                len(expected_reranked_events & expected_merged_events),
                len(expected_merged_events),
            ),
            "lost_events_from_merged": sorted(expected_merged_events - expected_reranked_events),
            "reference_event_recall": safe_div(len(expected_reranked_events), len(expected_events)),
            "reference_article_precision": precision,
            "reference_article_recall": safe_div(len(reranked_set & expected_articles), len(expected_articles)),
            "ndcg_at_k": {
                str(k): compute_ndcg(reranked_ids, expected_articles, k)
                for k in ks
            },
        })

    union_merged_events = ids_to_events(stage_article_ids(retrieval_details, "merged"), event_index)
    union_reranked_ids = stage_article_ids(retrieval_details, "reranked")
    union_reranked_events = ids_to_events(union_reranked_ids, event_index)

    reference_metrics = {
        "aggregation": "macro_average_by_execute",
        "execute_count": len(per_execute_reference),
        "avg_expected_event_count": (
            sum(row["expected_event_count"] for row in per_execute_reference) / len(per_execute_reference)
            if per_execute_reference else 0.0
        ),
        "event_retention_from_merged": (
            sum(row["event_retention_from_merged"] for row in per_execute_reference) / len(per_execute_reference)
            if per_execute_reference else 0.0
        ),
        "reference_event_recall": (
            sum(row["reference_event_recall"] for row in per_execute_reference) / len(per_execute_reference)
            if per_execute_reference else 0.0
        ),
        "reference_article_precision": (
            sum(row["reference_article_precision"] for row in per_execute_reference) / len(per_execute_reference)
            if per_execute_reference else 0.0
        ),
        "reference_article_recall": (
            sum(row["reference_article_recall"] for row in per_execute_reference) / len(per_execute_reference)
            if per_execute_reference else 0.0
        ),
        "per_execute": per_execute_reference,
        "union": {
            "event_retention_from_merged": safe_div(
                len(union_reranked_events & union_merged_events),
                len(union_merged_events),
            ),
            "lost_events_from_merged": sorted(union_merged_events - union_reranked_events),
            "reference_event_recall": safe_div(len(union_reranked_events), len(event_articles)),
            "reference_article_precision": safe_div(
                len(set(union_reranked_ids) & relevant_articles),
                len(set(union_reranked_ids)),
            ),
        },
        "at_k": {
            str(k): {
                "ndcg": (
                    sum(row["ndcg_at_k"][str(k)] for row in per_execute_reference) / len(per_execute_reference)
                    if per_execute_reference else 0.0
                ),
            }
            for k in ks
        },
    }
    result = {
        **heuristic,
        "reference_metrics": reference_metrics,
        "note": "Rerank 主指标是 rubric score；reference_metrics 只在存在标注时用于诊断。",
    }

    if judge_with_llm:
        try:
            llm_result = score_rerank_with_llm(
                topic,
                retrieval_details,
                items_by_query,
                rubric,
                judge_limit,
            )
            result["mode"] = "llm_rubric"
            result["llm_judge"] = llm_result
            if isinstance(llm_result.get("overall_score_100"), (int, float)):
                result["overall_score_100"] = float(llm_result["overall_score_100"])
            if isinstance(llm_result.get("dimension_scores_0_5"), dict):
                result["dimension_scores_0_5"] = llm_result["dimension_scores_0_5"]
        except Exception as exc:
            result["llm_judge_error"] = str(exc)

    return result


def sparse_backend(retrieval_details: list[dict]) -> str:
    for detail in retrieval_details:
        backend = detail.get("sparse", {}).get("backend")
        if backend:
            return str(backend)
    return "unknown"


def generate_report(metrics: dict, run_dir: Path) -> str:
    query = metrics["query_rewrite"]
    dense = metrics["dense"]
    sparse = metrics["bm25"]
    rerank = metrics["rerank"]

    lines = [
        "# RAG 四节点评估报告",
        "",
        "## 总览",
        "",
        "| 节点 | 主指标 | Precision | F1 | 说明 |",
        "|---|---:|---:|---:|---|",
        (
            f"| Plan/Query Rewrite | {query.get('overall_score_100', 0):.1f}/100 | "
            f"{format_pct(query.get('precision_proxy', 0.0))} | "
            f"{format_pct(query.get('f1_proxy', 0.0))} | rubric 自动评分，无硬真值 |"
        ),
        (
            f"| Dense/Chroma | {format_pct(dense.get('event_recall', 0.0))} | "
            f"{format_pct(dense.get('article_precision', 0.0))} | "
            f"{format_pct(dense.get('article_f1', 0.0))} | 每个 execute 单独评估后取平均 |"
        ),
        (
            f"| BM25/OpenSearch | {format_pct(sparse.get('event_recall', 0.0))} | "
            f"{format_pct(sparse.get('article_precision', 0.0))} | "
            f"{format_pct(sparse.get('article_f1', 0.0))} | 每个 execute 单独评估后取平均，backend={metrics.get('sparse_backend', 'unknown')} |"
        ),
        (
            f"| Rerank | {rerank.get('overall_score_100', 0):.1f}/100 | "
            f"{format_pct(rerank.get('reference_metrics', {}).get('reference_article_precision', 0.0))} | "
            f"N/A | 每个 execute 都会 rerank，rubric score 为 per-execute 平均 |"
        ),
        "",
        "## Plan/Query Rewrite",
        "",
        f"- 评分模式: {query.get('mode')}",
        f"- 子查询数: {query.get('sub_query_count', 0)}",
        f"- Rubric Recall Proxy: {format_pct(query.get('recall_proxy', 0.0))}",
        f"- Precision Proxy: {format_pct(query.get('precision_proxy', 0.0))}",
        f"- F1 Proxy: {format_pct(query.get('f1_proxy', 0.0))}",
        f"- 约束准确率: {format_pct(query.get('constraint_accuracy', 0.0))}",
        f"- 重复率: {format_pct(query.get('duplicate_rate', 0.0))}",
        f"- Rerank 弱反馈: 平均分 {query.get('avg_rerank_score', 0.0):.3f}, 高分率 {format_pct(query.get('high_score_rate', 0.0))}",
        "",
        "未覆盖业务维度:",
    ]
    missed = query.get("missed_dimensions", [])
    if missed:
        lines.extend(f"- {item.get('name', '')}: {item.get('description', '')}" for item in missed)
    else:
        lines.append("- 无")

    lines.extend([
        "",
        "## Dense/Chroma",
        "",
        f"- 聚合口径: {dense.get('aggregation')}",
        f"- Execute 数: {dense.get('execute_count', 0)}",
        f"- 平均候选文章数/execute: {dense.get('avg_count', 0.0):.1f}",
        f"- 平均 GT 事件数/execute: {dense.get('avg_expected_event_count', 0.0):.1f}",
        f"- Macro Event Recall: {format_pct(dense.get('event_recall', 0.0))}",
        f"- Article Precision / Recall / F1: "
        f"{format_pct(dense.get('article_precision', 0.0))} / "
        f"{format_pct(dense.get('article_recall', 0.0))} / "
        f"{format_pct(dense.get('article_f1', 0.0))}",
        f"- Union Event Recall（诊断）: {format_pct(dense.get('union', {}).get('event_recall', 0.0))}",
        f"- Union 命中事件: {dense.get('union', {}).get('hit_event_count', 0)}/{dense.get('union', {}).get('total_event_count', 0)}",
        "",
        "## BM25/OpenSearch",
        "",
        f"- 聚合口径: {sparse.get('aggregation')}",
        f"- Execute 数: {sparse.get('execute_count', 0)}",
        f"- 平均候选文章数/execute: {sparse.get('avg_count', 0.0):.1f}",
        f"- 平均 GT 事件数/execute: {sparse.get('avg_expected_event_count', 0.0):.1f}",
        f"- Macro Event Recall: {format_pct(sparse.get('event_recall', 0.0))}",
        f"- Article Precision / Recall / F1: "
        f"{format_pct(sparse.get('article_precision', 0.0))} / "
        f"{format_pct(sparse.get('article_recall', 0.0))} / "
        f"{format_pct(sparse.get('article_f1', 0.0))}",
        f"- Union Event Recall（诊断）: {format_pct(sparse.get('union', {}).get('event_recall', 0.0))}",
        f"- Union 命中事件: {sparse.get('union', {}).get('hit_event_count', 0)}/{sparse.get('union', {}).get('total_event_count', 0)}",
        "",
        "## Rerank",
        "",
        f"- 评分模式: {rerank.get('mode')}",
        f"- 评分公式: {rerank.get('formula', DEFAULT_RUBRICS['rerank']['formula'])}",
        f"- Rubric Score: {rerank.get('overall_score_100', 0):.1f}/100",
        f"- CommonScore / FinalScore: {rerank.get('common_score', 0.0):.3f} / {rerank.get('final_score', 0.0):.3f}",
        f"- 聚合口径: {rerank.get('reference_metrics', {}).get('aggregation')}",
        f"- 平均 GT 事件数/execute: {rerank.get('reference_metrics', {}).get('avg_expected_event_count', 0.0):.1f}",
        f"- Macro 从 Merge 保留事件率: {format_pct(rerank.get('reference_metrics', {}).get('event_retention_from_merged', 0.0))}",
        f"- Macro 标注参考 Event Recall: {format_pct(rerank.get('reference_metrics', {}).get('reference_event_recall', 0.0))}",
        f"- Union 标注参考 Event Recall（诊断）: {format_pct(rerank.get('reference_metrics', {}).get('union', {}).get('reference_event_recall', 0.0))}",
        "",
        "| 维度 | 分数(0-5) |",
        "|---|---:|",
    ])
    for key, value in rerank.get("dimension_scores_0_5", {}).items():
        lines.append(f"| {key} | {float(value):.2f} |")

    lost = rerank.get("reference_metrics", {}).get("union", {}).get("lost_events_from_merged", [])
    lines.extend(["", "Rerank 从 Merge 丢失事件（union 诊断）:"])
    lines.extend(f"- {event}" for event in lost) if lost else lines.append("- 无")

    lines.extend([
        "",
        "## 产物",
        "",
        f"- 指标 JSON: `{run_dir / 'four_node_metrics.json'}`",
        f"- 本报告: `{run_dir / 'FOUR_NODE_EVAL_REPORT.md'}`",
        f"- Rubric: `{run_dir / 'four_node_rubrics.json'}`",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 四节点评测")
    parser.add_argument("--run-dir", required=True, help="运行日志目录")
    parser.add_argument("--eval-dir", default=str(Path(__file__).parent), help="评测数据目录")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="研究主题")
    parser.add_argument("--start-date", default="2026-03-01", help="期望开始日期")
    parser.add_argument("--end-date", default="2026-03-31", help="期望结束日期")
    parser.add_argument("--category", default="AI", help="期望分类")
    parser.add_argument("--ks", default="5,10,20,50", help="逗号分隔的 K 值")
    parser.add_argument("--high-score-threshold", type=float, default=0.5, help="rerank 高分阈值")
    parser.add_argument("--rubric-path", default="", help="已有 rubric JSON")
    parser.add_argument("--business-context-path", default="", help="业务访谈/聊天记录文件，用于合成 rubric")
    parser.add_argument("--synthesize-rubrics", action="store_true", help="用大模型从业务记录合成 rubric")
    parser.add_argument("--judge-with-llm", action="store_true", help="用大模型按 rubric 评估 Query Rewrite 与 Rerank")
    parser.add_argument("--judge-limit", type=int, default=12, help="LLM 评估 Rerank 时最多评估多少个 query")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"找不到运行目录: {run_dir}")

    retrieval_details = load_json(run_dir / "retrieval_details.json", [])
    if not retrieval_details:
        raise FileNotFoundError(f"找不到或为空: {run_dir / 'retrieval_details.json'}")

    labels, event_index, event_articles = load_ground_truth(Path(args.eval_dir))
    if not event_index:
        raise FileNotFoundError("找不到 ground truth，请先生成 eval/event_to_articles.json")

    sub_queries = load_sub_queries(run_dir)
    ks = [int(item.strip()) for item in args.ks.split(",") if item.strip()]
    rubric = load_rubrics(args, run_dir)
    write_json(run_dir / "four_node_rubrics.json", rubric)
    query_gts = build_query_ground_truths(retrieval_details, labels, event_articles)

    metrics = {
        "run_dir": str(run_dir),
        "topic": args.topic,
        "ground_truth": {
            "event_count": len(event_articles),
            "article_count": len(event_index),
        },
        "sparse_backend": sparse_backend(retrieval_details),
        "query_ground_truths": query_gts,
        "query_rewrite": evaluate_query_rewrite(
            args.topic,
            sub_queries,
            retrieval_details,
            rubric,
            args.start_date,
            args.end_date,
            args.category,
            args.high_score_threshold,
            args.judge_with_llm,
        ),
        "dense": evaluate_retrieval_node(retrieval_details, "dense", event_index, event_articles, query_gts, ks),
        "bm25": evaluate_retrieval_node(retrieval_details, "sparse", event_index, event_articles, query_gts, ks),
        "rerank": evaluate_rerank_node(
            args.topic,
            run_dir,
            retrieval_details,
            event_index,
            event_articles,
            query_gts,
            rubric,
            ks,
            args.high_score_threshold,
            args.judge_with_llm,
            args.judge_limit,
        ),
    }

    write_json(run_dir / "four_node_metrics.json", metrics)
    report = generate_report(metrics, run_dir)
    (run_dir / "FOUR_NODE_EVAL_REPORT.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"\n已保存: {run_dir / 'FOUR_NODE_EVAL_REPORT.md'}")
    print(f"已保存: {run_dir / 'four_node_metrics.json'}")


if __name__ == "__main__":
    main()
