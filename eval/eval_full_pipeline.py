"""RAG 全链路分阶段评测。

评测对象：
- Plan 子查询
- Dense/Chroma 与 Sparse/OpenSearch
- Merge 候选池
- Rerank Top K
- RAG Compress
- Supervisor 交接

用法：
    python eval/eval_full_pipeline.py --run-dir logs/<run_dir>
    python eval/eval_full_pipeline.py --run-dir logs/<run_dir> --synthesize-plan-rubric
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
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

DEFAULT_PLAN_RUBRIC = {
    "dimensions": [
        {
            "name": "头部闭源厂商",
            "keywords": ["OpenAI", "Anthropic", "Google", "Meta", "xAI"],
            "description": "覆盖主流闭源或头部厂商模型发布。",
        },
        {
            "name": "国内厂商",
            "keywords": ["阿里", "通义", "Qwen", "百度", "文心", "DeepSeek", "字节", "豆包", "腾讯"],
            "description": "覆盖国内大模型厂商与模型发布。",
        },
        {
            "name": "开源模型",
            "keywords": ["开源", "open source", "Llama", "Mistral", "Qwen", "DeepSeek"],
            "description": "覆盖开源或开放权重模型发布。",
        },
        {
            "name": "多模态/图像",
            "keywords": ["多模态", "图像", "image", "vision", "视觉"],
            "description": "覆盖图像理解、图像生成、多模态模型。",
        },
        {
            "name": "视频/语音",
            "keywords": ["视频", "video", "语音", "audio", "Sora", "Vidu", "Seedance"],
            "description": "覆盖视频、语音、音频相关模型。",
        },
        {
            "name": "代码/Agent/推理",
            "keywords": ["代码", "coding", "Agent", "智能体", "推理", "reasoning"],
            "description": "覆盖代码模型、Agent、推理模型或能力升级。",
        },
    ]
}


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def ordered_unique(items: list[int]) -> list[int]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def ids_to_events(article_ids: list[int] | set[int], event_index: dict[str, str]) -> set[str]:
    events = set()
    for aid in article_ids:
        event_name = event_index.get(str(aid))
        if event_name:
            events.add(event_name)
    return events


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


def stage_article_ids(retrieval_details: list[dict], stage: str) -> list[int]:
    ids: list[int] = []
    for detail in retrieval_details:
        ids.extend(int(aid) for aid in detail.get(stage, {}).get("article_ids", []) if aid)
    return ordered_unique(ids)


def stage_metrics(
    article_ids: list[int],
    event_index: dict[str, str],
    event_articles: dict[str, set[int]],
    relevant_articles: set[int],
    ks: list[int],
) -> dict:
    hit_events_all = ids_to_events(article_ids, event_index)
    metrics = {
        "count": len(article_ids),
        "event_recall": len(hit_events_all) / len(event_articles) if event_articles else 0.0,
        "hit_events": sorted(hit_events_all),
        "missed_events": sorted(set(event_articles) - hit_events_all),
        "article_recall": len(set(article_ids) & relevant_articles) / len(relevant_articles)
        if relevant_articles else 0.0,
        "article_precision": len(set(article_ids) & relevant_articles) / len(article_ids)
        if article_ids else 0.0,
        "at_k": {},
    }
    for k in ks:
        top_ids = article_ids[:k]
        hit_events = ids_to_events(top_ids, event_index)
        metrics["at_k"][str(k)] = {
            "count": len(top_ids),
            "event_recall": len(hit_events) / len(event_articles) if event_articles else 0.0,
            "article_precision": len(set(top_ids) & relevant_articles) / len(top_ids)
            if top_ids else 0.0,
            "ndcg": compute_ndcg(article_ids, relevant_articles, k),
            "hit_events": sorted(hit_events),
        }
    return metrics


def evaluate_retrieval(
    retrieval_details: list[dict],
    event_index: dict[str, str],
    event_articles: dict[str, set[int]],
    ks: list[int],
) -> dict:
    relevant_articles = {int(aid) for aid in event_index}
    stages = {}
    for stage in ("dense", "sparse", "merged", "reranked"):
        ids = stage_article_ids(retrieval_details, stage)
        stages[stage] = stage_metrics(ids, event_index, event_articles, relevant_articles, ks)

    dense_events = set(stages["dense"]["hit_events"])
    sparse_events = set(stages["sparse"]["hit_events"])
    merged_events = set(stages["merged"]["hit_events"])
    reranked_events = set(stages["reranked"]["hit_events"])

    return {
        "stages": stages,
        "merge": {
            "event_recall_gain_vs_dense": stages["merged"]["event_recall"] - stages["dense"]["event_recall"],
            "event_recall_gain_vs_sparse": stages["merged"]["event_recall"] - stages["sparse"]["event_recall"],
            "dense_only_events": sorted(dense_events - sparse_events),
            "sparse_only_events": sorted(sparse_events - dense_events),
            "both_events": sorted(dense_events & sparse_events),
        },
        "rerank": {
            "event_retention_from_merged": len(reranked_events & merged_events) / len(merged_events)
            if merged_events else 0.0,
            "lost_events_from_merged": sorted(merged_events - reranked_events),
        },
    }


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


def synthesize_plan_rubric(topic: str, output_path: Path) -> dict:
    """用大模型合成 Plan 评测维度；失败时回退内置 rubric。"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return DEFAULT_PLAN_RUBRIC

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

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
        prompt = (
            "你是 RAG 查询规划评测专家。请为下面的月报研究任务合成 Plan 子查询评测 rubric。\n"
            "只输出 JSON，不要 Markdown。JSON 格式：\n"
            "{\"dimensions\":[{\"name\":\"维度名\",\"description\":\"为什么需要覆盖\","
            "\"keywords\":[\"用于匹配 query 的关键词或英文别名\"]}]}\n"
            "要求：6-10 个维度，覆盖厂商、模型类型、模态、开源/闭源、地区等关键方向。"
            "每个维度的 keywords 必须同时包含抽象词和可直接匹配 query 的具体例子，"
            "例如头部厂商维度必须包含 OpenAI、Google、Anthropic、Meta、xAI、Microsoft、阿里、腾讯、百度等具体名称。\n\n"
            f"研究任务：{topic}"
        )
        response = model.invoke([
            SystemMessage(content="只输出可解析 JSON。"),
            HumanMessage(content=prompt),
        ])
        text = str(response.content).strip()
        text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.DOTALL)
        rubric = json.loads(text)
        if not isinstance(rubric.get("dimensions"), list):
            raise ValueError("rubric 缺少 dimensions")
        write_json(output_path, rubric)
        return rubric
    except Exception as exc:
        fallback = dict(DEFAULT_PLAN_RUBRIC)
        fallback["synthesis_error"] = str(exc)
        write_json(output_path, fallback)
        return fallback


def load_plan_rubric(args, run_dir: Path) -> dict:
    if args.rubric_path:
        return load_json(Path(args.rubric_path), DEFAULT_PLAN_RUBRIC)
    output_path = run_dir / "synthetic_plan_rubric.json"
    if args.synthesize_plan_rubric:
        return synthesize_plan_rubric(args.topic, output_path)
    if output_path.exists():
        return load_json(output_path, DEFAULT_PLAN_RUBRIC)
    return DEFAULT_PLAN_RUBRIC


def evaluate_plan(
    sub_queries: list[dict],
    rubric: dict,
    expected_start_date: str,
    expected_end_date: str,
    expected_category: str,
    retrieval_details: list[dict],
    event_index: dict[str, str],
) -> dict:
    query_texts = [
        f"{q.get('search_intent', '')} {q.get('query', '')}"
        for q in sub_queries
    ]
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
                    "left_query": sub_queries[i].get("query", ""),
                    "right_query": sub_queries[j].get("query", ""),
                })

    date_ok = [
        q.get("start_date") == expected_start_date and q.get("end_date") == expected_end_date
        for q in sub_queries
    ]
    category_ok = [q.get("category", "") == expected_category for q in sub_queries]

    covered_dimensions = []
    missed_dimensions = []
    all_query_text = "\n".join(query_texts).lower()
    for dim in rubric.get("dimensions", []):
        keywords = [str(k).lower() for k in dim.get("keywords", [])]
        covered = any(keyword and keyword in all_query_text for keyword in keywords)
        item = {
            "name": dim.get("name", ""),
            "description": dim.get("description", ""),
            "keywords": dim.get("keywords", []),
        }
        if covered:
            covered_dimensions.append(item)
        else:
            missed_dimensions.append(item)

    query_event_hits = []
    for detail in retrieval_details:
        query = detail.get("query", "")
        reranked_ids = detail.get("reranked", {}).get("article_ids", [])
        events = ids_to_events([int(aid) for aid in reranked_ids if aid], event_index)
        query_event_hits.append({
            "query": query,
            "hit_event_count": len(events),
            "hit_events": sorted(events),
        })

    return {
        "sub_query_count": len(sub_queries),
        "date_accuracy": sum(date_ok) / len(date_ok) if date_ok else 0.0,
        "category_accuracy": sum(category_ok) / len(category_ok) if category_ok else 0.0,
        "duplicate_pair_count": len(duplicate_pairs),
        "duplicate_rate": len(duplicate_pairs) / max(1, len(sub_queries) * (len(sub_queries) - 1) / 2),
        "duplicate_pairs": duplicate_pairs[:20],
        "rubric_dimension_count": len(rubric.get("dimensions", [])),
        "coverage_rate": len(covered_dimensions) / len(rubric.get("dimensions", []))
        if rubric.get("dimensions") else 0.0,
        "covered_dimensions": covered_dimensions,
        "missed_dimensions": missed_dimensions,
        "query_event_hits": query_event_hits,
    }


def extract_article_ids_from_text(text: str) -> list[int]:
    ids = []
    for pattern in (r"ArticleID:\s*(\d+)", r"\[article:(\d+)\]"):
        ids.extend(int(match) for match in re.findall(pattern, text))
    return ordered_unique(ids)


def read_report_text(run_dir: Path, filename: str) -> str:
    path = run_dir / filename
    return path.read_text(encoding="utf-8") if path.exists() else ""


def collect_raw_results_text(run_dir: Path) -> str:
    texts = []
    raw_results = load_json(run_dir / "raw_results.json", [])
    if isinstance(raw_results, list):
        texts.extend(str(item) for item in raw_results)
    rag_outputs = load_json(run_dir / "rag_outputs.json", [])
    for output in rag_outputs:
        texts.extend(str(item) for item in output.get("raw_results", []))
    return "\n\n".join(texts)


def collect_compressed_text(run_dir: Path) -> str:
    compressed = read_report_text(run_dir, "compressed.md")
    if compressed:
        return compressed
    rag_outputs = load_json(run_dir / "rag_outputs.json", [])
    return "\n\n".join(str(item.get("compressed_research", "")) for item in rag_outputs)


def evaluate_compression(
    run_dir: Path,
    retrieval_details: list[dict],
    event_index: dict[str, str],
    event_articles: dict[str, set[int]],
) -> dict:
    raw_text = collect_raw_results_text(run_dir)
    compressed_text = collect_compressed_text(run_dir)

    raw_ids = extract_article_ids_from_text(raw_text)
    if not raw_ids and retrieval_details:
        raw_ids = stage_article_ids(retrieval_details, "reranked")
    compressed_ids = extract_article_ids_from_text(compressed_text)

    raw_events = ids_to_events(raw_ids, event_index)
    compressed_events = ids_to_events(compressed_ids, event_index)
    retained_events = raw_events & compressed_events

    return {
        "raw_article_count": len(raw_ids),
        "compressed_citation_count": len(compressed_ids),
        "raw_event_count": len(raw_events),
        "compressed_event_count": len(compressed_events),
        "raw_event_recall": len(raw_events) / len(event_articles) if event_articles else 0.0,
        "compressed_event_recall": len(compressed_events) / len(event_articles) if event_articles else 0.0,
        "event_retention_rate": len(retained_events) / len(raw_events) if raw_events else 0.0,
        "citation_retention_rate": len(set(compressed_ids) & set(raw_ids)) / len(set(raw_ids))
        if raw_ids else 0.0,
        "retained_events": sorted(retained_events),
        "dropped_events": sorted(raw_events - compressed_events),
        "new_events_from_compression": sorted(compressed_events - raw_events),
        "raw_chars": len(raw_text),
        "compressed_chars": len(compressed_text),
        "compression_ratio": len(compressed_text) / len(raw_text) if raw_text else 0.0,
    }


def evaluate_supervisor(
    run_dir: Path,
    event_index: dict[str, str],
    event_articles: dict[str, set[int]],
) -> dict:
    tool_calls = load_json(run_dir / "supervisor_tool_calls.json", [])
    evidence_pool = load_json(run_dir / "evidence_pool.json", [])
    final_notes = "\n\n".join(load_json(run_dir / "final_report_notes.json", []))
    final_report = read_report_text(run_dir, "report.md")
    compressed_text = collect_compressed_text(run_dir)

    available = bool(tool_calls or evidence_pool or final_notes or final_report)

    evidence_ids = {
        int(item["article_id"])
        for item in evidence_pool
        if item.get("article_id")
    }
    report_ids = set(extract_article_ids_from_text(final_report))
    note_ids = set(extract_article_ids_from_text(final_notes))
    compressed_ids = set(extract_article_ids_from_text(compressed_text))

    compressed_events = ids_to_events(compressed_ids, event_index)
    note_events = ids_to_events(note_ids, event_index)
    report_events = ids_to_events(report_ids, event_index)

    rag_calls = [call for call in tool_calls if call.get("name") == "ConductRAGResearch"]
    web_calls = [call for call in tool_calls if call.get("name") == "ConductResearch"]

    return {
        "available": available,
        "tool_call_count": len(tool_calls),
        "conduct_rag_count": len(rag_calls),
        "conduct_web_count": len(web_calls),
        "rag_success_count": sum(1 for call in rag_calls if call.get("status") == "ok"),
        "evidence_pool_count": len(evidence_pool),
        "final_report_citation_count": len(report_ids),
        "citation_whitelist_rate": len(report_ids & evidence_ids) / len(report_ids)
        if report_ids else 0.0,
        "notes_event_count": len(note_events),
        "report_event_count": len(report_events),
        "supervisor_handoff_retention": len(note_events & compressed_events) / len(compressed_events)
        if compressed_events else 0.0,
        "final_report_retention_from_notes": len(report_events & note_events) / len(note_events)
        if note_events else 0.0,
        "final_report_event_recall": len(report_events) / len(event_articles) if event_articles else 0.0,
        "events_in_compressed": sorted(compressed_events),
        "events_in_notes": sorted(note_events),
        "events_in_report": sorted(report_events),
        "events_lost_before_notes": sorted(compressed_events - note_events),
        "events_lost_before_report": sorted(note_events - report_events),
        "citations_outside_evidence_pool": sorted(report_ids - evidence_ids),
    }


def load_sub_queries(run_dir: Path) -> list[dict]:
    sub_queries = load_json(run_dir / "sub_queries.json", [])
    if sub_queries:
        return sub_queries
    sub_queries = load_json(run_dir / "rag_sub_queries.json", [])
    if sub_queries:
        return sub_queries
    rag_outputs = load_json(run_dir / "rag_outputs.json", [])
    merged = []
    for output in rag_outputs:
        merged.extend(output.get("sub_queries", []))
    return merged


def format_pct(value: float) -> str:
    return f"{value:.1%}"


def generate_markdown_report(metrics: dict, run_dir: Path) -> str:
    retrieval = metrics.get("retrieval", {})
    stages = retrieval.get("stages", {})
    compression = metrics.get("compression", {})
    supervisor = metrics.get("supervisor", {})
    plan = metrics.get("plan", {})

    lines = [
        "# RAG 全链路分阶段评测报告",
        "",
        "## 总览",
        "",
        "| 环节 | 核心指标 | 结果 |",
        "|---|---:|---:|",
        f"| Plan | 覆盖率 / 重复率 | {format_pct(plan.get('coverage_rate', 0.0))} / {format_pct(plan.get('duplicate_rate', 0.0))} |",
        f"| Dense/Chroma | Event Recall | {format_pct(stages.get('dense', {}).get('event_recall', 0.0))} |",
        f"| Sparse/OpenSearch | Event Recall | {format_pct(stages.get('sparse', {}).get('event_recall', 0.0))} |",
        f"| Merge | Event Recall | {format_pct(stages.get('merged', {}).get('event_recall', 0.0))} |",
        f"| Rerank | Event Recall | {format_pct(stages.get('reranked', {}).get('event_recall', 0.0))} |",
        f"| Compress | 事件保留率 | {format_pct(compression.get('event_retention_rate', 0.0))} |",
        f"| Supervisor | RAG 到 notes 保留率 | "
        f"{format_pct(supervisor.get('supervisor_handoff_retention', 0.0)) if supervisor.get('available') else 'N/A'} |",
        f"| Final Report | 引用白名单率 | "
        f"{format_pct(supervisor.get('citation_whitelist_rate', 0.0)) if supervisor.get('available') else 'N/A'} |",
        "",
        "## Plan 子查询",
        "",
        f"- 子查询数: {plan.get('sub_query_count', 0)}",
        f"- 日期正确率: {format_pct(plan.get('date_accuracy', 0.0))}",
        f"- 分类正确率: {format_pct(plan.get('category_accuracy', 0.0))}",
        f"- Rubric 覆盖率: {format_pct(plan.get('coverage_rate', 0.0))}",
        f"- 重复 query 对数: {plan.get('duplicate_pair_count', 0)}",
        "",
        "未覆盖维度:",
    ]
    missed = plan.get("missed_dimensions", [])
    if missed:
        lines.extend(f"- {item.get('name', '')}: {item.get('description', '')}" for item in missed)
    else:
        lines.append("- 无")

    lines.extend([
        "",
        "## 检索与重排",
        "",
        "| 阶段 | 候选数 | Event Recall | Article Precision | NDCG@20 |",
        "|---|---:|---:|---:|---:|",
    ])
    for key, label in (
        ("dense", "Dense/Chroma"),
        ("sparse", "Sparse/OpenSearch"),
        ("merged", "Merge"),
        ("reranked", "Rerank"),
    ):
        stage = stages.get(key, {})
        at20 = stage.get("at_k", {}).get("20", {})
        lines.append(
            f"| {label} | {stage.get('count', 0)} | "
            f"{format_pct(stage.get('event_recall', 0.0))} | "
            f"{format_pct(stage.get('article_precision', 0.0))} | "
            f"{at20.get('ndcg', 0.0):.3f} |"
        )

    merge = retrieval.get("merge", {})
    rerank = retrieval.get("rerank", {})
    lines.extend([
        "",
        f"- Merge 相对 Dense 召回增益: {format_pct(merge.get('event_recall_gain_vs_dense', 0.0))}",
        f"- Merge 相对 Sparse 召回增益: {format_pct(merge.get('event_recall_gain_vs_sparse', 0.0))}",
        f"- Rerank 从 Merged 的事件保留率: {format_pct(rerank.get('event_retention_from_merged', 0.0))}",
        "",
        "Rerank 丢失事件:",
    ])
    lost = rerank.get("lost_events_from_merged", [])
    lines.extend(f"- {event}" for event in lost) if lost else lines.append("- 无")

    lines.extend([
        "",
        "## RAG Compress",
        "",
        f"- Raw 命中事件数: {compression.get('raw_event_count', 0)}",
        f"- Compressed 引用命中事件数: {compression.get('compressed_event_count', 0)}",
        f"- 事件保留率: {format_pct(compression.get('event_retention_rate', 0.0))}",
        f"- 引用保留率: {format_pct(compression.get('citation_retention_rate', 0.0))}",
        f"- 压缩比: {compression.get('compression_ratio', 0.0):.3f}",
        "",
        "Compress 丢失事件:",
    ])
    dropped = compression.get("dropped_events", [])
    lines.extend(f"- {event}" for event in dropped) if dropped else lines.append("- 无")

    lines.extend([
        "",
        "## Supervisor",
        "",
    ])
    if not supervisor.get("available"):
        lines.extend([
            "当前 run 未包含完整主图的 Supervisor 产物；该部分只适用于通过 `src/runner.py` 或 TUI 跑出的完整端到端日志。",
            "",
            "需要的文件包括：`supervisor_tool_calls.json`、`rag_outputs.json`、`final_report_notes.json`、`report.md`。",
        ])
        lines.extend([
            "",
            "## 产物",
            "",
            f"- 指标 JSON: `{run_dir / 'stage_eval_metrics.json'}`",
            f"- 本报告: `{run_dir / 'STAGE_EVAL_REPORT.md'}`",
        ])
        return "\n".join(lines)

    lines.extend([
        f"- Supervisor 工具调用数: {supervisor.get('tool_call_count', 0)}",
        f"- ConductRAGResearch 调用数: {supervisor.get('conduct_rag_count', 0)}",
        f"- ConductResearch 调用数: {supervisor.get('conduct_web_count', 0)}",
        f"- Evidence Pool 条数: {supervisor.get('evidence_pool_count', 0)}",
        f"- RAG compressed 到 final notes 的事件保留率: {format_pct(supervisor.get('supervisor_handoff_retention', 0.0))}",
        f"- final notes 到 report 的事件保留率: {format_pct(supervisor.get('final_report_retention_from_notes', 0.0))}",
        f"- 报告引用白名单率: {format_pct(supervisor.get('citation_whitelist_rate', 0.0))}",
        "",
        "报告引用不在 evidence_pool 的 ArticleID:",
    ])
    outside = supervisor.get("citations_outside_evidence_pool", [])
    lines.extend(f"- {aid}" for aid in outside[:50]) if outside else lines.append("- 无")

    lines.extend([
        "",
        "## 产物",
        "",
        f"- 指标 JSON: `{run_dir / 'stage_eval_metrics.json'}`",
        f"- 本报告: `{run_dir / 'STAGE_EVAL_REPORT.md'}`",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 全链路分阶段评测")
    parser.add_argument("--run-dir", required=True, help="运行日志目录")
    parser.add_argument("--eval-dir", default=str(Path(__file__).parent), help="评测数据目录")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="研究主题，用于 Plan rubric 合成")
    parser.add_argument("--start-date", default="2026-03-01", help="期望开始日期")
    parser.add_argument("--end-date", default="2026-03-31", help="期望结束日期")
    parser.add_argument("--category", default="AI", help="期望分类")
    parser.add_argument("--ks", default="5,10,20,50", help="逗号分隔的 K 值")
    parser.add_argument("--rubric-path", default="", help="人工或合成 plan rubric JSON")
    parser.add_argument("--synthesize-plan-rubric", action="store_true", help="用大模型合成 Plan rubric")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    eval_dir = Path(args.eval_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"找不到运行目录: {run_dir}")

    _, event_index, event_articles = load_ground_truth(eval_dir)
    retrieval_details = load_json(run_dir / "retrieval_details.json", [])
    sub_queries = load_sub_queries(run_dir)
    ks = [int(item.strip()) for item in args.ks.split(",") if item.strip()]
    rubric = load_plan_rubric(args, run_dir)

    metrics = {
        "run_dir": str(run_dir),
        "topic": args.topic,
        "ground_truth": {
            "event_count": len(event_articles),
            "article_count": len(event_index),
        },
        "plan": evaluate_plan(
            sub_queries,
            rubric,
            args.start_date,
            args.end_date,
            args.category,
            retrieval_details,
            event_index,
        ),
        "retrieval": evaluate_retrieval(retrieval_details, event_index, event_articles, ks)
        if retrieval_details else {"stages": {}, "merge": {}, "rerank": {}, "missing": "retrieval_details.json not found"},
        "compression": evaluate_compression(run_dir, retrieval_details, event_index, event_articles),
        "supervisor": evaluate_supervisor(run_dir, event_index, event_articles),
    }

    write_json(run_dir / "stage_eval_metrics.json", metrics)
    report = generate_markdown_report(metrics, run_dir)
    (run_dir / "STAGE_EVAL_REPORT.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"\n已保存: {run_dir / 'STAGE_EVAL_REPORT.md'}")
    print(f"已保存: {run_dir / 'stage_eval_metrics.json'}")


if __name__ == "__main__":
    main()
