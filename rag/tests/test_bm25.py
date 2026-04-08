"""单独测试 BM25 分支的脚本。"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bm25_search import bm25_search
from text_analyzer import analyze_text


def _format_timestamp(ts: int | None) -> str:
    if not ts:
        return "unknown"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def main():
    parser = argparse.ArgumentParser(description="单独测试 BM25 检索分支。")
    parser.add_argument(
        "query",
        nargs="?",
        default="2026年3月发布了哪些模型？",
        help="要测试的查询语句。",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="时间范围。0 表示不过滤时间。",
    )
    parser.add_argument(
        "--category",
        default="AI",
        help="分类过滤。空字符串表示不过滤分类。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="返回结果数量。",
    )
    parser.add_argument(
        "--preview-length",
        type=int,
        default=300,
        help="预览文本最大字符数。",
    )
    args = parser.parse_args()

    threshold = None
    if args.days > 0:
        threshold = int(time.time() - args.days * 24 * 3600)

    print(f"\n{'=' * 60}")
    print("BM25 单独测试")
    print(f"{'=' * 60}")
    print(f"查询: {args.query}")
    print(f"Analyzer Tokens: {analyze_text(args.query)}")
    print(f"分类过滤: {args.category or '不限'}")
    print(f"时间过滤: {args.days} 天" if args.days > 0 else "时间过滤: 不限")
    if threshold is not None:
        print(f"时间阈值: {_format_timestamp(threshold)}")

    hits = bm25_search(
        query=args.query,
        top_k=args.top_k,
        category=args.category,
        published_ts_gte=threshold,
    )

    if not hits:
        print("\n没有命中结果。")
        return

    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        doc = hit.get("doc", "")
        title = doc.split("\n")[0] if doc else hit["id"]
        preview = (meta.get("preview") or "")[: args.preview_length]

        print(f"\n--- BM25 结果 {i} ---")
        print(f"id: {hit['id']}")
        print(f"score: {hit['score']:.4f}")
        print(f"标题: {title}")
        print(f"分类: {meta.get('category')}")
        print(f"来源: {meta.get('source_name')}")
        print(f"发布时间: {_format_timestamp(meta.get('published_ts'))}")
        print(f"预览: {preview}")


if __name__ == "__main__":
    main()
