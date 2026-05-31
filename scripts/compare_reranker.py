"""对比新旧 Reranker 的快速脚本。"""
import importlib.util
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "rag"))

os.environ["LEXICAL_BACKEND"] = "bm25"

# eval 没有 __init__.py，用 importlib 加载
def _load_eval_module():
    spec = importlib.util.spec_from_file_location(
        "eval_four_nodes",
        PROJECT_ROOT / "eval" / "eval_four_nodes.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_em = _load_eval_module()
load_ground_truth = _em.load_ground_truth
ids_to_events = _em.ids_to_events
safe_div = _em.safe_div
build_event_catalog = _em.build_event_catalog
query_ground_truth = _em.query_ground_truth
load_json = _em.load_json

from rag.reranker import rerank_and_select_candidates, rerank_with_clusters
from rag.rag_search import _collect_candidates, _lexical_search, embed_query, get_collection

def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "logs/eval搜索本地新闻数据库查找2026年3月1-20260531-110457"
    )
    retrieval_details = load_json(run_dir / "retrieval_details.json", [])
    if not retrieval_details:
        print("找不到 retrieval_details.json")
        return

    eval_dir = Path("eval")
    labels, event_index, event_articles = load_ground_truth(eval_dir)
    catalog = build_event_catalog(labels, event_articles)
    all_gt_events = set(event_articles.keys())
    k = 30

    old_union = set()
    new_union = set()
    old_macros = []
    new_macros = []

    for idx, detail in enumerate(retrieval_details):
        query = detail.get("query", "")
        try:
            vec = get_collection().query(
                query_embeddings=[embed_query(query)],
                n_results=80,
            )
            lexical_hits = _lexical_search(
                query, top_k=80, category="AI",
                published_ts_gte=None, published_ts_lte=None,
            )
            candidates = _collect_candidates(vec, lexical_hits)

            gt = query_ground_truth(query, catalog, event_articles)
            gt_events = set(gt["expected_events"])

            # 旧方法
            old = rerank_and_select_candidates(query, candidates, k)
            old_ids = [int(item["metadata"].get("article_id", 0)) for item in old if item["metadata"].get("article_id")]
            old_events = ids_to_events(old_ids, event_index) & all_gt_events
            old_recall = safe_div(len(old_events & gt_events), len(gt_events)) if gt_events else 0

            # 新方法
            new = rerank_with_clusters(query, candidates, k)
            new_ids = [int(item["metadata"].get("article_id", 0)) for item in new if item["metadata"].get("article_id")]
            new_events = ids_to_events(new_ids, event_index) & all_gt_events
            new_recall = safe_div(len(new_events & gt_events), len(gt_events)) if gt_events else 0

            old_macros.append(old_recall)
            new_macros.append(new_recall)
            old_union |= (old_events & gt_events)
            new_union |= (new_events & gt_events)

            print(f"[{idx+1}/{len(retrieval_details)}] {query[:60]}... old={old_recall:.2f} new={new_recall:.2f}")
        except Exception as e:
            print(f"[{idx+1}] ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("对比结果 (TopK=30)")
    print("=" * 60)
    old_m = sum(old_macros) / len(old_macros) if old_macros else 0
    new_m = sum(new_macros) / len(new_macros) if new_macros else 0
    old_u = len(old_union) / len(all_gt_events)
    new_u = len(new_union) / len(all_gt_events)

    print(f"旧方法 (贪心多样性): Macro={old_m:.1%}  Union={old_u:.1%} ({len(old_union)}/{len(all_gt_events)})")
    print(f"新方法 (语义聚类):   Macro={new_m:.1%}  Union={new_u:.1%} ({len(new_union)}/{len(all_gt_events)})")
    print(f"Macro 变化: {new_m - old_m:+.1%}")
    print(f"Union 变化: {new_u - old_u:+.1%}")

    print("\n逐 query (new-old):")
    for i, (old_r, new_r) in enumerate(zip(old_macros, new_macros)):
        diff = new_r - old_r
        sig = "+" if diff > 0.01 else ("-" if diff < -0.01 else "=")
        print(f"  [{i+1}] {sig} {diff:+.2%}")

if __name__ == "__main__":
    main()
