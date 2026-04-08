"""LATS 子图测试：实际运行树状发散搜索 + 导出搜索树和状态历史。

需要环境变量：OPENAI_API_KEY
需要本地 RAG 数据库已构建（rag/vectordb/）
运行后在 logs/ 目录生成独立 run 目录。
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))


async def test_lats_subgraph():
    """实际调用 LATS 子图，测试完整 树搜索-评估-展开-剪枝-聚合 流程。"""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from lats_subgraph import lats_researcher_builder

    topic = "2026年3月1日到31日之间有哪些基础大语言模型发布了"

    # 创建 run 目录
    from runner import make_run_dir
    run_dir = make_run_dir(f"lats-test-{topic[:20]}")
    thread_id = run_dir.name
    db_path = str(run_dir / "checkpoints.db")

    print(f"📋 研究主题: {topic}")
    print(f"📁 Run 目录: {run_dir}")
    print(f"🔑 Thread ID: {thread_id}")
    print("⏳ 启动 LATS 子图...\n")

    start_time = datetime.now()

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        subgraph = lats_researcher_builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        result = await subgraph.ainvoke(
            {"research_topic": topic},
            config=config,
        )

        # 打印结果
        compressed = result.get("compressed_research", "")
        findings = result.get("collected_findings", [])
        tree = result.get("tree", {})
        iteration = result.get("iteration", 0)

        print("=" * 60)
        print("📝 压缩研究摘要（前 800 字）:")
        print("-" * 60)
        print(compressed[:800] if compressed else "❌ 无摘要输出")
        print(f"\n📏 摘要总长度: {len(compressed)} 字符")
        print(f"📌 收集到的 findings 条数: {len(findings)}")
        print(f"🌳 搜索树节点总数: {len(tree)}")
        print(f"🔄 迭代轮次: {iteration}")

        # 搜索树统计
        status_counts = {}
        for ndata in tree.values():
            s = ndata.get("status", "unknown")
            status_counts[s] = status_counts.get(s, 0) + 1
        print(f"📊 节点状态分布: {status_counts}")

        # 打印搜索树结构
        print("\n🌲 搜索树结构:")
        print("-" * 60)
        root_id = result.get("root_id", "")
        _print_tree(tree, root_id, indent=0)

        # 导出状态历史
        history = []
        async for snapshot in subgraph.aget_state_history(config):
            step = {
                "step": snapshot.metadata.get("step", -1),
                "node": snapshot.metadata.get("source", "unknown"),
                "writes_keys": list((snapshot.metadata.get("writes") or {}).keys()),
                "tree_size": len(snapshot.values.get("tree", {})),
                "iteration": snapshot.values.get("iteration", 0),
                "current_node_id": snapshot.values.get("current_node_id", ""),
            }
            history.append(step)

    elapsed = (datetime.now() - start_time).total_seconds()

    # 保存运行元数据
    meta = {
        "test": "lats_subgraph",
        "topic": topic,
        "thread_id": thread_id,
        "elapsed_seconds": round(elapsed, 1),
        "iterations": iteration,
        "tree_node_count": len(tree),
        "status_distribution": status_counts,
        "compressed_length": len(compressed),
        "findings_count": len(findings),
        "steps": history,
    }
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 保存压缩摘要
    if compressed:
        with open(run_dir / "compressed.md", "w", encoding="utf-8") as f:
            f.write(compressed)

    # 保存收集到的原始 findings
    with open(run_dir / "findings.json", "w", encoding="utf-8") as f:
        json.dump(findings, f, ensure_ascii=False, indent=2)

    # 保存完整搜索树（核心调试产物）
    with open(run_dir / "search_tree.json", "w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)

    # 生成搜索树可视化报告
    _generate_tree_report(run_dir, topic, tree, root_id, findings, compressed,
                          elapsed, iteration, status_counts)

    print(f"\n⏱️  耗时: {elapsed:.1f}s")
    print(f"📁 结果已保存: {run_dir}")
    print(f"   ├── run_meta.json      (运行元数据)")
    print(f"   ├── compressed.md      (压缩摘要)")
    print(f"   ├── findings.json      (原始 findings)")
    print(f"   ├── search_tree.json   (完整搜索树)")
    print(f"   ├── report.md          (搜索树可视化报告)")
    print(f"   └── checkpoints.db     (完整 checkpoint)")

    assert compressed, "❌ compressed_research 不应为空"
    assert len(findings) > 0, "❌ collected_findings 不应为空"
    assert len(tree) >= 1, "❌ 搜索树不应为空"
    print("\n🎉 LATS 子图测试通过！")


def _print_tree(tree: dict, node_id: str, indent: int):
    """递归打印搜索树结构。"""
    if not node_id or node_id not in tree:
        return
    node = tree[node_id]
    status_icon = {
        "leaf": "🟢", "expanded": "🔵", "pruned": "❌", "pending": "⏳"
    }.get(node["status"], "❓")
    rel = f"rel={node.get('relevance_score', 0):.2f}"
    comp = f"comp={node.get('completeness_score', 0):.2f}"
    prefix = "  " * indent + ("├── " if indent > 0 else "")
    print(f"{prefix}{status_icon} [{node['status']}] {node['query'][:50]}  ({rel}, {comp})")
    for child_id in node.get("children_ids", []):
        _print_tree(tree, child_id, indent + 1)


def _generate_tree_report(run_dir, topic, tree, root_id, findings,
                          compressed, elapsed, iteration, status_counts):
    """生成 Markdown 格式的搜索树可视化报告。"""
    lines = [
        f"# LATS 搜索树报告",
        f"",
        f"**研究主题**: {topic}",
        f"**耗时**: {elapsed:.1f}s | **迭代**: {iteration} 轮 | **节点数**: {len(tree)}",
        f"**节点状态**: {' / '.join(f'{s}: {c}' for s, c in status_counts.items())}",
        f"",
        f"---",
        f"",
        f"## 搜索树",
        f"",
    ]

    # 递归生成 Markdown 树
    def _md_tree(nid, depth):
        if not nid or nid not in tree:
            return
        n = tree[nid]
        icon = {"leaf": "🟢", "expanded": "🔵", "pruned": "❌", "pending": "⏳"}.get(n["status"], "❓")
        indent = "  " * depth
        lines.append(f"{indent}- {icon} **{n['query'][:60]}**  "
                      f"`rel={n.get('relevance_score',0):.2f} comp={n.get('completeness_score',0):.2f} "
                      f"d={n['depth']} hits={n.get('result_count',0)}`")
        for cid in n.get("children_ids", []):
            _md_tree(cid, depth + 1)

    _md_tree(root_id, 0)

    lines += [
        f"",
        f"---",
        f"",
        f"## 各节点搜索结果摘要",
        f"",
    ]
    for nid, n in tree.items():
        if n.get("search_results") and n["status"] != "pruned":
            lines.append(f"### {n['query'][:60]}")
            lines.append(f"- 状态: {n['status']} | 相关性: {n.get('relevance_score',0):.2f} | 完整性: {n.get('completeness_score',0):.2f}")
            lines.append(f"")
            lines.append(f"```text")
            lines.append(n["search_results"][:1000])
            lines.append(f"```")
            lines.append(f"")

    lines += [
        f"---",
        f"",
        f"## 最终压缩摘要",
        f"",
        compressed[:3000] if compressed else "（无摘要）",
    ]

    with open(run_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(test_lats_subgraph())
