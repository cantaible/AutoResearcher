"""单独测试 reranker 重排效果的脚本。"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bm25_search import bm25_search
from config import RETRIEVAL_CANDIDATE_MULTIPLIER
from rag_search import _collect_candidates, embed_query, get_collection
from reranker import rerank_candidates


def main():
    parser = argparse.ArgumentParser(description="单独测试 reranker 对候选集的重排效果。")
    parser.add_argument(
        "query",
        nargs="?",
        default="2026年3月发布了哪些模型？",
        help="要测试的查询语句。",
    )
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--category", default="AI")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=0,
        help="候选集大小。0 表示自动按 top_k * multiplier 计算。",
    )
    args = parser.parse_args()

    threshold = None
    if args.days > 0:
        threshold = int(time.time() - args.days * 24 * 3600)

    where_clauses = []
    if threshold is not None:
        where_clauses.append({"published_ts": {"$gte": threshold}})
    if args.category:
        where_clauses.append({"category": {"$eq": args.category}})

    where = None
    if len(where_clauses) == 1:
        where = where_clauses[0]
    elif len(where_clauses) > 1:
        where = {"$and": where_clauses}

    candidate_k = args.candidate_k or max(args.top_k * RETRIEVAL_CANDIDATE_MULTIPLIER, args.top_k)

    vec = get_collection().query(
        query_embeddings=[embed_query(args.query)],
        where=where,
        n_results=candidate_k,
    )
    bm25 = bm25_search(
        query=args.query,
        top_k=candidate_k,
        category=args.category,
        published_ts_gte=threshold,
    )
    candidates = _collect_candidates(vec, bm25)
    reranked = rerank_candidates(args.query, candidates)[: args.top_k]

    print(f"\n{'=' * 60}")
    print("Reranker 单独测试")
    print(f"{'=' * 60}")
    print(f"查询: {args.query}")
    print(f"候选集大小: {len(candidates)}")

    for i, item in enumerate(reranked, start=1):
        meta = item["metadata"]
        title = (item.get("doc") or "").split("\n")[0]
        print(f"\n--- 重排结果 {i} ---")
        print(f"来源: {'+'.join(item.get('sources', []))}")
        print(f"Rerank分数: {item.get('rerank_score', 0.0):.4f}")
        print(f"标题: {title}")
        print(f"元数据: [{meta.get('category')}] | [{meta.get('source_name')}]")
        print(f"预览: {(meta.get('preview') or '')[:300]}")


if __name__ == "__main__":
    main()
