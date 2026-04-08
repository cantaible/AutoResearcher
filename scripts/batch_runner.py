"""批量实验运行器：自动跑多组配置，无人值守。

用法：
    python scripts/batch_runner.py experiments/experiment_config.example.yaml
    python scripts/batch_runner.py experiments/my_config.yaml --dry-run   # 预览不执行
    python scripts/batch_runner.py experiments/my_config.yaml --only baseline_gpt41  # 只跑指定实验
    nohup python scripts/batch_runner.py experiments/my_config.yaml > batch.log 2>&1 &  # 后台运行

特性：
  - 按顺序执行每组 (experiment × topic) 组合
  - 单组失败不影响后续执行
  - 实时写入进度到 progress.json（可随时查看跑到哪了）
  - 结束后自动生成 summary.csv 对比表
"""

import argparse
import asyncio
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# 确保 src 在 import 路径中
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_config(config_path: str) -> dict:
    """加载 YAML 实验配置。"""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_experiment_dir(output_base: Path, experiment_name: str, topic: str) -> Path:
    """为单次实验创建目录：output_base/experiment_name/topic_slug-timestamp/"""
    import re
    slug = re.sub(r'[^\w\s]', '', topic).strip()
    slug = re.sub(r'\s+', '-', slug)[:30]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_base / experiment_name / f"{slug}-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


async def run_single_experiment(
    topic: str,
    experiment_config: dict,
    global_config: dict,
    run_dir: Path,
) -> dict:
    """执行单次研究实验。

    Returns:
        结果字典，包含耗时、报告长度、是否成功等。
    """
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from graph import deep_researcher_builder     # noqa: 需在 sys.path 设置后导入

    # 合并配置：global < experiment-level config
    merged_config = {**global_config}
    merged_config.update(experiment_config.get("config", {}))
    # 批量运行必须禁用澄清
    merged_config["allow_clarification"] = False

    thread_id = run_dir.name
    config = {"configurable": {"thread_id": thread_id, **merged_config}}
    db_path = str(run_dir / "checkpoints.db")
    events_path = run_dir / "events.jsonl"

    start_time = time.time()
    final_report = ""
    error_msg = ""
    token_usage = {"total_input": 0, "total_output": 0}

    try:
        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
            graph = deep_researcher_builder.compile(checkpointer=checkpointer)
            messages = [HumanMessage(content=topic)]

            event_stream = graph.astream_events(
                {"messages": messages}, config, version="v2"
            )

            async for raw in event_stream:
                # 标准化事件并持久化
                from runner import normalize_event, append_event
                evt = normalize_event(raw)
                if evt:
                    append_event(events_path, evt)
                    # 累计 token 用量
                    if evt["type"] == "llm_end":
                        usage = evt.get("usage", {})
                        token_usage["total_input"] += usage.get("input_tokens", 0)
                        token_usage["total_output"] += usage.get("output_tokens", 0)

            # 获取最终状态
            state = await graph.aget_state(config)
            result = state.values
            final_report = result.get("final_report", "")

            if final_report:
                with open(run_dir / "report.md", "w", encoding="utf-8") as f:
                    f.write(final_report)

    except Exception as e:
        error_msg = str(e)
        # 记录错误详情
        with open(run_dir / "error.log", "w", encoding="utf-8") as f:
            import traceback
            f.write(traceback.format_exc())

    elapsed = time.time() - start_time

    # 保存运行元数据
    meta = {
        "topic": topic,
        "experiment": experiment_config.get("name", "unknown"),
        "config": merged_config,
        "elapsed_seconds": round(elapsed, 1),
        "completed": bool(final_report),
        "report_length": len(final_report),
        "token_usage": token_usage,
        "error": error_msg,
        "timestamp": datetime.now().isoformat(),
    }
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def generate_summary(output_base: Path):
    """扫描所有实验结果，生成 summary.csv 对比表。"""
    rows = []
    for meta_path in sorted(output_base.rglob("run_meta.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        config = meta.get("config", {})
        token = meta.get("token_usage", {})
        rows.append({
            "experiment": meta.get("experiment", ""),
            "topic": meta.get("topic", "")[:40],
            "completed": meta.get("completed", False),
            "elapsed_min": round(meta.get("elapsed_seconds", 0) / 60, 1),
            "report_chars": meta.get("report_length", 0),
            "input_tokens": token.get("total_input", 0),
            "output_tokens": token.get("total_output", 0),
            "research_model": config.get("research_model", ""),
            "final_report_model": config.get("final_report_model", ""),
            "max_iterations": config.get("max_researcher_iterations", ""),
            "error": meta.get("error", "")[:60],
        })

    if not rows:
        print("⚠️  未找到任何实验结果")
        return

    csv_path = output_base / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # 同时打印到终端
    print(f"\n{'='*80}")
    print(f"📊 实验结果汇总 ({len(rows)} 组)")
    print(f"{'='*80}")
    header = f"{'实验名':<20} {'主题':<25} {'完成':^6} {'耗时':>6} {'报告长度':>8} {'总token':>8}"
    print(header)
    print("-" * 80)
    for r in rows:
        total_tok = r['input_tokens'] + r['output_tokens']
        status = "✅" if r['completed'] else "❌"
        print(f"{r['experiment']:<20} {r['topic']:<25} {status:^6} "
              f"{r['elapsed_min']:>5.1f}m {r['report_chars']:>8} {total_tok:>8}")
    print(f"\n📁 详细结果: {csv_path}")


def update_progress(progress_path: Path, current: int, total: int, info: dict):
    """更新进度文件，方便随时 cat 查看。"""
    progress = {
        "current": current,
        "total": total,
        "percent": round(current / total * 100, 1) if total else 0,
        "current_experiment": info.get("experiment", ""),
        "current_topic": info.get("topic", ""),
        "last_update": datetime.now().isoformat(),
        "completed_results": info.get("completed_results", []),
    }
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


async def main():
    parser = argparse.ArgumentParser(description="Deep Research 批量实验运行器")
    parser.add_argument("config", help="YAML 实验配置文件路径")
    parser.add_argument("--dry-run", action="store_true", help="预览实验列表，不实际执行")
    parser.add_argument("--only", type=str, help="只运行指定名称的实验")
    parser.add_argument("--summary-only", action="store_true", help="只生成汇总报告，不运行实验")
    args = parser.parse_args()

    # 加载配置
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)

    exp_config = load_config(str(config_path))
    global_config = exp_config.get("global", {})
    topics = exp_config.get("topics", [])
    experiments = exp_config.get("experiments", [])

    # 过滤指定实验
    if args.only:
        experiments = [e for e in experiments if e["name"] == args.only]
        if not experiments:
            print(f"❌ 未找到实验: {args.only}")
            sys.exit(1)

    # 计算总任务数
    total_runs = len(experiments) * len(topics)
    output_base = PROJECT_ROOT / global_config.get("output_dir", "experiments/results")
    output_base.mkdir(parents=True, exist_ok=True)

    # 汇总模式
    if args.summary_only:
        generate_summary(output_base)
        return

    # 预览模式
    print(f"\n🔬 Deep Research 批量实验")
    print(f"   实验组数: {len(experiments)}")
    print(f"   研究主题: {len(topics)}")
    print(f"   总运行数: {total_runs}")
    print(f"   输出目录: {output_base}\n")

    for i, exp in enumerate(experiments, 1):
        print(f"   [{i}] {exp['name']}: {exp.get('description', '')}")
        for k, v in exp.get("config", {}).items():
            print(f"       {k}: {v}")
    print()

    for topic in topics:
        print(f"   📝 {topic}")

    if args.dry_run:
        print("\n🏃 --dry-run 模式，不实际执行")
        return

    # 开始执行
    print(f"\n{'='*60}")
    print(f"🚀 开始批量实验 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'='*60}\n")

    progress_path = output_base / "progress.json"
    completed_results = []
    current = 0

    for exp in experiments:
        for topic in topics:
            current += 1
            exp_name = exp["name"]

            print(f"\n[{current}/{total_runs}] 🧪 {exp_name} | {topic[:40]}...")
            print(f"   ⏱️  开始: {datetime.now().strftime('%H:%M:%S')}")

            # 更新进度
            update_progress(progress_path, current, total_runs, {
                "experiment": exp_name,
                "topic": topic,
                "completed_results": completed_results,
            })

            # 创建实验目录并运行
            run_dir = make_experiment_dir(output_base, exp_name, topic)
            result = await run_single_experiment(
                topic=topic,
                experiment_config=exp,
                global_config=global_config,
                run_dir=run_dir,
            )

            # 记录结果
            status = "✅ 完成" if result["completed"] else "❌ 失败"
            elapsed_min = round(result["elapsed_seconds"] / 60, 1)
            print(f"   {status} | 耗时 {elapsed_min}分钟 | 报告 {result['report_length']} 字符")
            if result["error"]:
                print(f"   ⚠️  错误: {result['error'][:100]}")

            completed_results.append({
                "experiment": exp_name,
                "topic": topic[:30],
                "completed": result["completed"],
                "elapsed_min": elapsed_min,
            })

    # 生成汇总
    print(f"\n{'='*60}")
    print(f"✨ 所有实验完成 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'='*60}")
    generate_summary(output_base)

    # 最终进度标记
    update_progress(progress_path, total_runs, total_runs, {
        "experiment": "ALL_DONE",
        "topic": "",
        "completed_results": completed_results,
    })


if __name__ == "__main__":
    asyncio.run(main())
