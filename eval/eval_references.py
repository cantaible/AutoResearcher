"""结构化引用评测：从报告中提取 [article:ID]，与 ground truth 匹配计算 Recall/Precision。

用法：
    python eval/eval_references.py --run-dir logs/<运行目录>
"""

import argparse
import json
import re
from pathlib import Path


def extract_article_ids(report_text: str) -> set[int]:
    """从报告文本中提取所有 [article:ID] 引用。"""
    matches = re.findall(r"\[article:(\d+)\]", report_text)
    return set(int(m) for m in matches)


def load_ground_truth(eval_dir: Path) -> dict[str, set[int]]:
    """加载 ground truth：event_name -> set of article_ids。"""
    event_to_articles_path = eval_dir / "event_to_articles.json"
    with open(event_to_articles_path) as f:
        raw = json.load(f)
    # raw: {article_id_str: canonical_name}
    event_articles: dict[str, set[int]] = {}
    for aid_str, event_name in raw.items():
        event_articles.setdefault(event_name, set()).add(int(aid_str))
    return event_articles


def evaluate(
    cited_ids: set[int],
    event_articles: dict[str, set[int]],
) -> dict:
    """计算 Reference Recall 和 Precision。

    - Event Recall: 命中了多少个事件（至少引用了该事件的一篇文章）
    - Article Precision: 引用的文章中，有多少属于某个 GT 事件
    """
    all_gt_article_ids = set()
    for aids in event_articles.values():
        all_gt_article_ids.update(aids)

    total_events = len(event_articles)
    hit_events = []
    for event_name, aids in event_articles.items():
        if cited_ids & aids:
            hit_events.append(event_name)

    # Precision: 引用的文章中属于 GT 的比例
    cited_in_gt = cited_ids & all_gt_article_ids
    precision = len(cited_in_gt) / len(cited_ids) if cited_ids else 0.0
    recall = len(hit_events) / total_events if total_events else 0.0

    return {
        "total_cited": len(cited_ids),
        "cited_in_gt": len(cited_in_gt),
        "precision": precision,
        "total_events": total_events,
        "hit_events": len(hit_events),
        "recall": recall,
        "hit_event_names": hit_events,
        "missed_event_names": [
            e for e in event_articles if e not in hit_events
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="结构化引用评测")
    parser.add_argument("--run-dir", required=True, help="运行日志目录")
    parser.add_argument(
        "--eval-dir",
        default=str(Path(__file__).parent),
        help="评测数据目录（默认 eval/）",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    eval_dir = Path(args.eval_dir)

    # 加载报告
    report_path = run_dir / "report.md"
    if not report_path.exists():
        print(f"错误：找不到报告文件 {report_path}")
        return

    report_text = report_path.read_text(encoding="utf-8")
    cited_ids = extract_article_ids(report_text)
    print(f"从报告中提取到 {len(cited_ids)} 个唯一 article 引用")

    if not cited_ids:
        print("报告中没有 [article:ID] 格式的引用，无法评测。")
        print("请确保报告使用了新的引用格式。")
        return

    # 加载 ground truth
    event_articles = load_ground_truth(eval_dir)
    print(f"Ground Truth: {len(event_articles)} 个事件")

    # 评测
    results = evaluate(cited_ids, event_articles)

    # 输出结果
    print("\n" + "=" * 50)
    print("Reference Evaluation Results")
    print("=" * 50)
    print(f"Event Recall:      {results['hit_events']}/{results['total_events']}"
          f" = {results['recall']:.1%}")
    print(f"Article Precision: {results['cited_in_gt']}/{results['total_cited']}"
          f" = {results['precision']:.1%}")
    print(f"\n命中事件 ({results['hit_events']}):")
    for name in results["hit_event_names"]:
        print(f"  + {name}")
    print(f"\n未命中事件 ({len(results['missed_event_names'])}):")
    for name in results["missed_event_names"]:
        print(f"  - {name}")

    # 保存结果到运行目录
    output_path = run_dir / "REFERENCE_EVAL.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {output_path}")


if __name__ == "__main__":
    main()
