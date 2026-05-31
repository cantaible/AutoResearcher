"""语义聚类模块：对 Merge 候选做事件级聚类，将同事件的不同报道归组。

优先用关键词规则（_event_key）快速归组，无法识别的事件再用语义向量聚类。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sentence_transformers import SentenceTransformer

from config import CLUSTER_SIMILARITY_THRESHOLD, EMBEDDING_MODEL

_embed_model = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _embed_model


def _semantic_text(candidate: dict) -> str | None:
    """构造聚类用的语义文本：title + preview，都空则返回 None。"""
    meta = candidate.get("metadata") or {}
    title = (meta.get("title") or "").strip()
    preview = (meta.get("preview") or "").strip()
    # preview 空则用 doc 前 200 字兜底
    if not preview:
        doc = (candidate.get("doc") or "").strip()
        preview = doc[:200]
    text = f"{title}\n{preview}".strip()
    return text if text else None


def _get_embeddings(texts: list[str]) -> list[list[float]]:
    model = _get_embed_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def _pairwise_cosine_similarity(embeddings: list[list[float]]) -> np.ndarray:
    """L2 归一化后逐对点积 = cosine 相似度矩阵。"""
    vecs = np.asarray(embeddings, dtype=np.float32)
    return vecs @ vecs.T


def _hierarchical_clusters(
    sim_matrix: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    """用 scipy 的 complete-linkage 层次聚类（C 实现，O(n²)）。

    距离 = 1 - 余弦相似度，complete linkage，按 threshold 切分。
    """
    n = sim_matrix.shape[0]
    if n <= 1:
        return [[0]] if n == 1 else []

    dist = 1.0 - sim_matrix
    # scipy 需要 condensed 形式（上三角不含对角线）
    condensed = dist[np.triu_indices(n, k=1)]
    Z = linkage(condensed, method="complete")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    # 按 label 聚合成簇列表
    clusters: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(i)
    return sorted(clusters.values(), key=lambda c: c[0])


def cluster_candidates(
    candidates: list[dict],
    threshold: float | None = None,
) -> tuple[list[list[dict]], list[dict]]:
    """对候选文档做事件级聚类：先关键词归组，再语义向量兜底。

    返回 (簇列表, 未归类候选)。
    """
    if threshold is None:
        threshold = CLUSTER_SIMILARITY_THRESHOLD

    if not candidates:
        return [], []

    # 第一层：关键词事件归组（导入在函数内避免循环依赖）
    from reranker import _event_key as _get_event_key
    key_groups: dict[str, list[int]] = {}
    no_key_indices: list[int] = []
    for idx, candidate in enumerate(candidates):
        key = _get_event_key(candidate)
        if key:
            key_groups.setdefault(key, []).append(idx)
        else:
            no_key_indices.append(idx)

    key_clusters: list[list[dict]] = [
        [candidates[i] for i in indices]
        for indices in key_groups.values()
    ]

    # 第二层：无语义标签的用语义向量聚类
    if len(no_key_indices) <= 2:
        semantic_clusters = [[candidates[i]] for i in no_key_indices] if no_key_indices else []
        return key_clusters + semantic_clusters, []

    no_key_candidates = [candidates[i] for i in no_key_indices]
    # 构造语义文本
    valid_texts: list[str] = []
    valid_indices: list[int] = []  # 在 no_key_candidates 中的位置
    tail: list[dict] = []
    for j, candidate in enumerate(no_key_candidates):
        text = _semantic_text(candidate)
        if text:
            valid_texts.append(text)
            valid_indices.append(j)
        else:
            tail.append(candidate)

    if len(valid_texts) <= 1:
        semantic_clusters = [[no_key_candidates[j]] for j in valid_indices] if valid_indices else []
        return key_clusters + semantic_clusters, tail

    embeddings = _get_embeddings(valid_texts)
    sim_matrix = _pairwise_cosine_similarity(embeddings)
    index_clusters = _hierarchical_clusters(sim_matrix, threshold)

    for idx_cluster in index_clusters:
        cluster = [no_key_candidates[valid_indices[i]] for i in idx_cluster]
        key_clusters.append(cluster)

    return key_clusters, tail
