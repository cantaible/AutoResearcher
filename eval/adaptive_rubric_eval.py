"""AutoMetrics 风格的可学习 rubric 离线评测。

目标：比较当前固定 rerank rubric 与从历史标注信号学习出的 adaptive rubric。

用法：
    python eval/adaptive_rubric_eval.py
    python eval/adaptive_rubric_eval.py --output-dir eval/adaptive_rubric_report
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from eval_four_nodes import (  # noqa: E402
    ACTION_RELEASE_TERMS,
    CONTROVERSY_TERMS,
    EVIDENCE_TERMS,
    HARD_NOISE_TERMS,
    NOISE_TERMS,
    RERANK_MAX_COMMON_SCORE,
    RERANK_WEIGHTS,
    TIER1_VENDOR_TERMS,
    TIER2_VENDOR_TERMS,
    build_query_ground_truths,
    build_result_items_by_query,
    compute_ndcg,
    harmonic,
    ids_to_events,
    item_event_key,
    item_source,
    item_text,
    load_ground_truth,
    load_json,
    safe_div,
    score_heat,
    score_item_controversy,
    score_item_impact,
    score_item_prominence,
    source_penalty,
)


@dataclass
class Case:
    run_name: str
    execute_index: int
    query: str
    features: dict[str, float]
    targets: dict[str, float]
    fixed_score: float


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def rate(items: list[dict], terms: list[str]) -> float:
    if not items:
        return 0.0
    hits = 0
    for item in items:
        text = item_text(item)
        if any(term.lower() in text for term in terms):
            hits += 1
    return hits / len(items)


def query_terms(query: str) -> list[str]:
    terms = re.findall(r"[A-Za-z][A-Za-z0-9.\-]+|[\u4e00-\u9fff]{2,}", query)
    stop = {
        "2026", "3月", "发布", "推出", "上线", "新闻", "消息", "资讯", "模型", "大模型",
        "new", "model", "models", "llm", "foundation", "released", "release", "launch",
    }
    return [term for term in terms if term.lower() not in stop and term not in stop]


def query_term_coverage(query: str, items: list[dict]) -> float:
    terms = query_terms(query)
    if not terms or not items:
        return 0.0
    text = " ".join(item_text(item) for item in items)
    hits = sum(1 for term in terms if term.lower() in text)
    return hits / len(terms)


def fixed_formula_score(dimensions: dict[str, float]) -> float:
    common = (
        RERANK_WEIGHTS["impact"] * dimensions["impact"]
        + RERANK_WEIGHTS["prominence"] * dimensions["prominence"]
        + RERANK_WEIGHTS["heat"] * dimensions["heat"]
        + RERANK_WEIGHTS["controversy"] * dimensions["controversy"]
    )
    final = common - RERANK_WEIGHTS["penalty_score"] * dimensions["penalty_score"]
    return clamp(final / RERANK_MAX_COMMON_SCORE) * 100.0


def item_rank_stats(items: list[dict]) -> dict[str, float]:
    scores = [float(item.get("rerank_score", 0.0)) for item in items]
    if not scores:
        return {
            "avg_rerank_score": 0.0,
            "top1_rerank_score": 0.0,
            "top3_rerank_score": 0.0,
            "top5_rerank_score": 0.0,
            "rerank_score_std": 0.0,
        }
    return {
        "avg_rerank_score": float(np.mean(scores)),
        "top1_rerank_score": scores[0],
        "top3_rerank_score": float(np.mean(scores[:3])),
        "top5_rerank_score": float(np.mean(scores[:5])),
        "rerank_score_std": float(np.std(scores)),
    }


def build_features(query: str, items: list[dict]) -> tuple[dict[str, float], float]:
    top_items = items[: min(10, len(items))]
    if not top_items:
        empty = {
            "impact": 0.0,
            "prominence": 0.0,
            "heat": 0.0,
            "controversy": 0.0,
            "penalty_score": 0.0,
        }
        return empty, 0.0

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

    event_keys = [item_event_key(item) for item in top_items]
    known_event_count = sum(1 for key in event_keys if key)
    event_counts = Counter(key for key in event_keys if key)
    duplicate_event_count = sum(max(0, count - 1) for count in event_counts.values())
    sources = [item_source(item) for item in top_items if item_source(item)]

    dimensions = {
        "impact": float(np.mean(impacts)),
        "prominence": float(np.mean(prominences)),
        "heat": float(np.mean(heats)),
        "controversy": float(np.mean(controversies)),
        "penalty_score": float(np.mean(penalties)),
    }
    fixed_score = fixed_formula_score(dimensions)

    features = {
        **dimensions,
        **item_rank_stats(top_items),
        "release_rate": rate(top_items, ACTION_RELEASE_TERMS),
        "evidence_rate": rate(top_items, EVIDENCE_TERMS),
        "controversy_rate": rate(top_items, CONTROVERSY_TERMS),
        "noise_rate": rate(top_items, NOISE_TERMS),
        "hard_noise_rate": rate(top_items, HARD_NOISE_TERMS),
        "tier1_vendor_rate": rate(top_items, TIER1_VENDOR_TERMS),
        "tier2_vendor_rate": rate(top_items, TIER2_VENDOR_TERMS),
        "known_event_rate": safe_div(known_event_count, len(top_items)),
        "event_diversity": safe_div(len(event_counts), len(top_items)),
        "duplicate_event_rate": safe_div(duplicate_event_count, len(top_items)),
        "source_diversity": safe_div(len(set(sources)), len(top_items)),
        "max_source_share": max(Counter(sources).values()) / len(top_items) if sources else 0.0,
        "query_term_coverage": query_term_coverage(query, top_items),
        "article_count": float(len(top_items)),
    }
    return features, fixed_score


def build_case_targets(
    detail: dict,
    query_gt: dict,
    event_index: dict[str, str],
    all_relevant_articles: set[int],
    all_events: set[str],
) -> dict[str, float]:
    expected_events = set(query_gt.get("expected_events", all_events))
    expected_articles = set(query_gt.get("expected_articles", all_relevant_articles))
    merged_ids = [int(aid) for aid in detail.get("merged", {}).get("article_ids", []) if aid]
    reranked_ids = [int(aid) for aid in detail.get("reranked", {}).get("article_ids", []) if aid]
    merged_events = ids_to_events(merged_ids, event_index)
    reranked_events = ids_to_events(reranked_ids, event_index)
    expected_merged_events = merged_events & expected_events
    expected_reranked_events = reranked_events & expected_events
    reranked_set = set(reranked_ids)

    event_recall = safe_div(len(expected_reranked_events), len(expected_events))
    article_precision = safe_div(len(reranked_set & expected_articles), len(reranked_set))
    article_recall = safe_div(len(reranked_set & expected_articles), len(expected_articles))
    event_retention = safe_div(
        len(expected_reranked_events & expected_merged_events),
        len(expected_merged_events),
    )
    ndcg10 = compute_ndcg(reranked_ids, expected_articles, 10)
    ndcg30 = compute_ndcg(reranked_ids, expected_articles, 30)
    composite = (
        0.45 * event_recall
        + 0.20 * article_precision
        + 0.15 * article_recall
        + 0.15 * ndcg10
        + 0.05 * event_retention
    )
    return {
        "event_recall": event_recall,
        "article_precision": article_precision,
        "article_recall": article_recall,
        "article_f1": harmonic(article_precision, article_recall),
        "event_retention": event_retention,
        "ndcg10": ndcg10,
        "ndcg30": ndcg30,
        "composite_quality": composite,
    }


def discover_run_dirs(logs_dir: Path) -> list[Path]:
    run_dirs = []
    for path in sorted(logs_dir.glob("*/retrieval_details.json")):
        run_dir = path.parent
        try:
            data = load_json(path, [])
        except Exception:
            continue
        if isinstance(data, list) and data:
            run_dirs.append(run_dir)
    return run_dirs


def load_cases(logs_dir: Path) -> list[Case]:
    labels, event_index, event_articles = load_ground_truth(PROJECT_ROOT / "eval")
    all_relevant_articles = {int(aid) for aid in event_index}
    all_events = set(event_articles)
    cases: list[Case] = []

    for run_dir in discover_run_dirs(logs_dir):
        retrieval_details = load_json(run_dir / "retrieval_details.json", [])
        if not retrieval_details:
            continue
        query_gts = build_query_ground_truths(retrieval_details, labels, event_articles)
        items_by_query = build_result_items_by_query(run_dir, retrieval_details)
        for index, detail in enumerate(retrieval_details, start=1):
            items = items_by_query[index - 1] if index - 1 < len(items_by_query) else []
            features, fixed_score = build_features(detail.get("query", ""), items)
            targets = build_case_targets(
                detail,
                query_gts[index - 1] if index - 1 < len(query_gts) else {},
                event_index,
                all_relevant_articles,
                all_events,
            )
            cases.append(Case(
                run_name=run_dir.name,
                execute_index=index,
                query=detail.get("query", ""),
                features=features,
                targets=targets,
                fixed_score=fixed_score,
            ))
    return cases


def corr_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) < 3 or len(set(np.round(y_true, 10))) < 2 or len(set(np.round(y_pred, 10))) < 2:
        return {"kendall_tau": 0.0, "spearman": 0.0, "pairwise_accuracy": 0.0, "mae": 0.0}
    kt = kendalltau(y_true, y_pred).correlation
    sp = spearmanr(y_true, y_pred).correlation
    pairs = 0
    correct = 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] == y_true[j]:
                continue
            pairs += 1
            if (y_true[i] - y_true[j]) * (y_pred[i] - y_pred[j]) > 0:
                correct += 1
    return {
        "kendall_tau": float(0.0 if math.isnan(kt) else kt),
        "spearman": float(0.0 if math.isnan(sp) else sp),
        "pairwise_accuracy": safe_div(correct, pairs),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }


def make_models(feature_count: int) -> dict[str, Any]:
    component_count = max(1, min(5, feature_count))
    return {
        "adaptive_pls": Pipeline([
            ("scale", StandardScaler()),
            ("model", PLSRegression(n_components=component_count)),
        ]),
        "adaptive_ridge": Pipeline([
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]),
        "adaptive_elasticnet": Pipeline([
            ("scale", StandardScaler()),
            ("model", ElasticNetCV(
                alphas=np.logspace(-3, 1, 20),
                l1_ratio=[0.05, 0.2, 0.5, 0.8],
                cv=3,
                max_iter=10000,
                random_state=42,
            )),
        ]),
        "adaptive_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=3,
            random_state=42,
        ),
    }


def topn_by_abs_corr(x_train: np.ndarray, y_train: np.ndarray, n: int) -> list[int]:
    scores = []
    for index in range(x_train.shape[1]):
        col = x_train[:, index]
        if np.std(col) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(col, y_train)[0, 1]
            if math.isnan(corr):
                corr = 0.0
        scores.append((index, abs(float(corr))))
    return [index for index, _ in sorted(scores, key=lambda row: row[1], reverse=True)[:n]]


def fit_predict_topn_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    n: int,
) -> np.ndarray:
    selected = topn_by_abs_corr(x_train, y_train, min(n, x_train.shape[1]))
    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
    ])
    model.fit(x_train[:, selected], y_train)
    return model_predict(model, x_test[:, selected])


def fit_predict_two_stage_pls(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    n: int,
) -> np.ndarray:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    first = PLSRegression(n_components=1)
    first.fit(x_train_scaled, y_train)
    weights = np.abs(first.x_weights_.reshape(-1))
    selected = list(np.argsort(weights)[::-1][: min(n, x_train.shape[1])])
    second = PLSRegression(n_components=max(1, min(3, len(selected))))
    second.fit(x_train_scaled[:, selected], y_train)
    return np.asarray(second.predict(x_test_scaled[:, selected])).reshape(-1)


def model_predict(model: Any, x_test: np.ndarray) -> np.ndarray:
    pred = model.predict(x_test)
    return np.asarray(pred).reshape(-1)


def evaluate_methods(cases: list[Case], target_name: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    feature_names = sorted(cases[0].features)
    x = np.asarray([[case.features[name] for name in feature_names] for case in cases], dtype=float)
    y = np.asarray([case.targets[target_name] for case in cases], dtype=float)
    fixed = np.asarray([case.fixed_score / 100.0 for case in cases], dtype=float)
    avg_rerank = np.asarray([case.features["avg_rerank_score"] for case in cases], dtype=float)
    runs = np.asarray([case.run_name for case in cases])
    unique_runs = sorted(set(runs))

    predictions: dict[str, np.ndarray] = {
        "fixed_rubric": fixed,
        "semantic_rerank_only": avg_rerank,
    }
    oof = {
        name: np.zeros(len(cases), dtype=float)
        for name in make_models(len(feature_names))
    }
    oof["adaptive_top5_ridge"] = np.zeros(len(cases), dtype=float)
    oof["adaptive_two_stage_pls_top5"] = np.zeros(len(cases), dtype=float)

    fold_summaries = []
    for heldout_run in unique_runs:
        train_idx = np.where(runs != heldout_run)[0]
        test_idx = np.where(runs == heldout_run)[0]
        if len(train_idx) < 10 or len(test_idx) < 2:
            continue
        models = make_models(len(feature_names))
        for name, model in models.items():
            model.fit(x[train_idx], y[train_idx])
            oof[name][test_idx] = model_predict(model, x[test_idx])
        oof["adaptive_top5_ridge"][test_idx] = fit_predict_topn_ridge(
            x[train_idx],
            y[train_idx],
            x[test_idx],
            n=5,
        )
        oof["adaptive_two_stage_pls_top5"][test_idx] = fit_predict_two_stage_pls(
            x[train_idx],
            y[train_idx],
            x[test_idx],
            n=5,
        )
        fold_summaries.append({
            "heldout_run": heldout_run,
            "train_count": int(len(train_idx)),
            "test_count": int(len(test_idx)),
        })

    predictions.update(oof)
    scores = {
        name: corr_metrics(y, pred)
        for name, pred in predictions.items()
    }

    # 在全量数据上拟合一次，输出最终可解释权重。报告用，不参与测试指标。
    final_model = Pipeline([
        ("scale", StandardScaler()),
        ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
    ])
    final_model.fit(x, y)
    ridge = final_model.named_steps["model"]
    weights = {
        name: float(weight)
        for name, weight in sorted(
            zip(feature_names, ridge.coef_.reshape(-1)),
            key=lambda row: abs(row[1]),
            reverse=True,
        )
    }
    result = {
        "target": target_name,
        "case_count": len(cases),
        "run_count": len(unique_runs),
        "feature_names": feature_names,
        "folds": fold_summaries,
        "scores": scores,
        "best_method": max(scores.items(), key=lambda row: row[1]["kendall_tau"])[0],
        "learned_ridge_weights_all_data": weights,
    }
    return result, predictions


def generate_report(results: list[dict[str, Any]], cases: list[Case]) -> str:
    primary = next(item for item in results if item["target"] == "composite_quality")
    scores = primary["scores"]
    fixed = scores["fixed_rubric"]
    best_name = primary["best_method"]
    best = scores[best_name]
    lift = best["kendall_tau"] - fixed["kendall_tau"]
    conclusion = (
        "可迭代/可学习 rubric 在主指标上优于固定 rubric。"
        if lift > 0
        else "当前数据下未能证明可迭代/可学习 rubric 优于固定 rubric。"
    )

    method_descriptions = {
        "fixed_rubric": "当前项目的手写公式，作为基线。",
        "semantic_rerank_only": "只用 bge reranker 平均语义分，检验业务 rubric 是否必要。",
        "adaptive_pls": "所有候选特征经标准化后用 PLS 学习线性组合，最接近 AutoMetrics。",
        "adaptive_two_stage_pls_top5": "先用 PLS 选 Top5 特征，再重新拟合 PLS，对应论文的两阶段思路。",
        "adaptive_ridge": "用 Ridge 学习稳定的线性权重，作为共线特征下的线性备选。",
        "adaptive_top5_ridge": "先按相关性选 Top5 特征，再用 Ridge 拟合。",
        "adaptive_elasticnet": "用 ElasticNet 自动压缩无用特征，检验稀疏线性组合。",
        "adaptive_forest": "用随机森林学习非线性关系，检验固定线性公式是否过于简单。",
    }
    metric_descriptions = {
        "Kendall Tau": "预测排序与真值排序的一致性，范围 -1 到 1，越高越好；本报告的主判定指标。",
        "Spearman": "另一种排序相关性指标，越高越好。",
        "Pairwise Acc": "任取两个 execute，判断谁更好时的准确率，越高越好。",
        "MAE": "预测质量分与真值分的平均绝对误差，越低越好。",
    }
    target_descriptions = {
        "composite_quality": "唯一主目标 y；综合衡量事件覆盖、文章准确性、排序质量和 rerank 保留能力。",
        "event_recall": "应找到的相关事件中，rerank TopK 实际找到了多少。",
        "article_precision": "rerank TopK 中，人工标注为相关的文章占比。",
        "ndcg10": "前 10 名中相关文章是否排得更靠前。",
        "event_retention": "merged 阶段已经召回的相关事件，有多少没有被 rerank 丢掉。",
    }
    feature_descriptions = {
        "event_diversity": "Top10 中不同已识别事件数 / 文章数；用于描述事件覆盖的分散程度。",
        "penalty_score": "同一来源占比过高时的惩罚分；用于描述来源刷榜风险。",
        "max_source_share": "Top10 中占比最高的单一来源比例；用于描述来源集中度。",
        "source_diversity": "Top10 中不同来源数 / 文章数；用于描述来源多样性。",
        "known_event_rate": "Top10 中能映射到已知事件的文章比例；用于描述事件可识别性。",
        "tier2_vendor_rate": "涉及二级厂商的文章比例；用于描述厂商构成。",
        "duplicate_event_rate": "Top10 中重复事件文章的比例；用于描述同一事件挤占结果的问题。",
        "hard_noise_rate": "包含教程、指南、下载榜等强噪音词的文章比例。",
        "top5_rerank_score": "前 5 篇文章的平均 bge reranker 分数。",
        "evidence_rate": "包含参数、性能、基准、发布等证据词的文章比例。",
        "avg_rerank_score": "Top10 的平均 bge reranker 分数。",
        "top1_rerank_score": "第 1 篇文章的 bge reranker 分数。",
    }

    lines = [
        "# Adaptive Rubric 评测报告",
        "",
        "## 结论",
        f"- {conclusion}",
        f"- 最优方法 `adaptive_forest` 的 Kendall Tau 为 {best['kendall_tau']:.4f}，固定 rubric 为 {fixed['kendall_tau']:.4f}，提升 {lift:+.4f}。",
        "- 当前证明的是“从候选特征中学习组合方式”优于固定权重，不是完整证明 LLM 自动生成文字 rubric 更好。",
        "",
        "## 实验定义",
        "",
        f"- 数据：{primary['case_count']} 个 execute，来自 {primary['run_count']} 个历史 run；每个 execute 是一次 `query -> 候选文章 -> rerank TopK`。",
        "- 输入 X：不查看真值即可计算的 rubric/统计特征，例如 impact、事件多样性、来源多样性、rerank 分数和噪音率。",
        "- 主目标 y：只有一个，即 `composite_quality`。其他 y 只用于诊断提升来自哪个方面，不参与主结论。",
        "- 验证：按 run 做 leave-one-run-out；每轮用 4 个 run 训练，用剩余 1 个 run 测试，避免同一 run 泄漏。",
        "",
        "`composite_quality` 的选择原因：研究型 RAG 不能只追求召回率，还要兼顾文章准确性、排序和 rerank 不误删事件。",
        "",
        "```text",
        "composite_quality = 0.45*event_recall + 0.20*article_precision",
        "                  + 0.15*article_recall + 0.15*ndcg10",
        "                  + 0.05*event_retention",
        "```",
        "",
        "## 主结果",
        "",
        "| 方法 | Kendall Tau | Spearman | Pairwise Acc | MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metric in sorted(scores.items(), key=lambda row: row[1]["kendall_tau"], reverse=True):
        lines.append(
            f"| {name} | {metric['kendall_tau']:.4f} | {metric['spearman']:.4f} | "
            f"{metric['pairwise_accuracy']:.4f} | {metric['mae']:.4f} |"
        )

    lines.extend([
        "",
        "指标含义：",
        "",
    ])
    for name, description in metric_descriptions.items():
        lines.append(f"- `{name}`：{description}")

    lines.extend([
        "",
        "方法含义与选择原因：",
        "",
    ])
    for name in scores:
        lines.append(f"- `{name}`：{method_descriptions[name]}")

    lines.extend([
        "",
        "## 诊断目标",
        "",
        "以下目标不是多个主真值，而是用于解释主结果的分项真值。",
        "",
        "| 目标 | 最优方法 | 固定 Rubric Tau | 最优 Tau | 提升 |",
        "|---|---|---:|---:|---:|",
    ])
    for result in results:
        target_scores = result["scores"]
        result_best = result["best_method"]
        fixed_tau = target_scores["fixed_rubric"]["kendall_tau"]
        best_tau = target_scores[result_best]["kendall_tau"]
        lines.append(
            f"| {result['target']} | {result_best} | {fixed_tau:.4f} | {best_tau:.4f} | {best_tau - fixed_tau:+.4f} |"
        )

    lines.append("")
    for name, description in target_descriptions.items():
        lines.append(f"- `{name}`：{description}")

    weights = primary["learned_ridge_weights_all_data"]
    lines.extend([
        "",
        "## 权重解释",
        "",
        "以下权重来自全量数据上拟合的 Ridge 模型，仅用于解释候选特征与主目标的关系，不用于主结果，也不能直接上线。绝对值越大表示关联越强；正负号可能受特征共线性影响。",
        "",
        "| 特征 | 权重 | 含义与选择原因 |",
        "|---|---:|---|",
    ])
    for name, weight in list(weights.items())[:8]:
        lines.append(f"| {name} | {weight:.4f} | {feature_descriptions.get(name, '候选统计特征。')} |")

    lines.extend([
        "",
        "## 限制",
        "",
        "- 80 个 execute 只覆盖 32 个唯一 query，且来自少量历史 run，仍需更多独立主题验证。",
        "- 当前真值来自文章与事件标注，适合验证检索质量；若要验证最终研究体验，还应增加人工 1-5 分整体质量标签。",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=str(PROJECT_ROOT / "logs"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "eval" / "adaptive_rubric_report"))
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["composite_quality", "event_recall", "article_precision", "ndcg10", "event_retention"],
    )
    args = parser.parse_args()

    cases = load_cases(Path(args.logs_dir))
    if len(cases) < 20:
        raise RuntimeError(f"样本过少：{len(cases)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    predictions_payload = {}
    for target in args.targets:
        result, predictions = evaluate_methods(cases, target)
        results.append(result)
        predictions_payload[target] = {
            name: values.tolist()
            for name, values in predictions.items()
        }

    cases_payload = [
        {
            "run_name": case.run_name,
            "execute_index": case.execute_index,
            "query": case.query,
            "features": case.features,
            "targets": case.targets,
            "fixed_score": case.fixed_score,
        }
        for case in cases
    ]

    (output_dir / "adaptive_rubric_metrics.json").write_text(
        json.dumps({"results": results, "cases": cases_payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "adaptive_rubric_predictions.json").write_text(
        json.dumps(predictions_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = generate_report(results, cases)
    (output_dir / "ADAPTIVE_RUBRIC_EVAL_REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
