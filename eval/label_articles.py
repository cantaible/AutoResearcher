"""为数据库中的新闻文章打标签，用于 RAG 检索效果评估。

完全独立脚本，不依赖项目中其他文件。
从 MariaDB 逐批读取文章，调用 LLM 打标签，结果保存为 JSON。

用法：
    python eval/label_articles.py                    # 打标全部 2026年3月 AI 文章
    python eval/label_articles.py --limit 50         # 只打前 50 篇（测试用）
    python eval/label_articles.py --resume           # 从上次中断处继续
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import pymysql
import requests

# ── 配置 ──

OPENROUTER_API_KEY = "DE3FDDAF-472D-473A-80BB-7A4BB5599DB8"
OPENROUTER_BASE_URL = "https://capi.quan2go.com/v1/chat/completions"
MODEL = "gpt-5.4"

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3307,
    "user": "root",
    "password": "rootpass",
    "database": "news_reader",
    "charset": "utf8mb4",
}

# 每次 LLM 调用处理的文章数（批量打标省钱）
BATCH_SIZE = 20

# 输出路径
OUTPUT_DIR = Path(__file__).parent
OUTPUT_FILE = OUTPUT_DIR / "article_labels.json"

# ── 标签体系 ──

EVENT_TYPES = [
    "model_release",     # 模型发布、开源、版本升级
    "product_launch",    # 产品/服务/平台上线、功能更新（非模型本身）
    "funding_business",  # 融资、营收、估值、商业合作
    "industry_news",     # 事实性行业新闻（人事、政策、事件、标准）
    "other",             # 观点分析、评论、教程、榜单汇总等
]

SYSTEM_PROMPT = """你是一个新闻分类标注助手。对每篇 AI 新闻文章进行两个维度的标注。

## 维度 1：event_type（单选，必须为以下 5 个值之一）
- model_release：一个新的 AI 模型被发布、开源或版本升级（无论语言/图像/机器人模型）
- product_launch：一款 AI 产品、服务或平台上线，或已有产品发布了重要功能更新（不是模型本身）
- funding_business：融资、营收披露、估值变化、商业合作等商业事件
- industry_news：事实性行业新闻，如人事变动、政策发布、标准制定、行业事件报道
- other：观点文章、趋势分析、评论、教程、排行榜汇总、与 AI 无直接关联的内容

## 维度 2：entities（多选，开放集合）
提取文章涉及的最核心 1-3 个实体（公司/模型/产品名称）。
用最常见的简称即可。如果是泛行业分析无特定实体，填 ["general"]。

## 输出格式（严格遵守）
返回一个 JSON 数组，每个元素格式如下（注意 id 为整数）：
{"id": 4299, "event_type": "industry_news", "entities": ["Anthropic"]}

不要输出任何解释文字，只输出 JSON 数组。"""


def load_existing_labels(path: Path) -> dict[int, dict]:
    """加载已有标签，支持断点续打。"""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["article_id"]: item for item in data}


def save_labels(labels: dict[int, dict], path: Path):
    """保存标签到 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_items = sorted(labels.values(), key=lambda x: x["article_id"])
    path.write_text(
        json.dumps(sorted_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def fetch_articles(limit: int = 0) -> list[dict]:
    """从数据库拉取 2026年3月 AI 类全部文章。"""
    conn = pymysql.connect(**DB_CONFIG)
    cur = conn.cursor(pymysql.cursors.DictCursor)
    sql = """
        SELECT id, title, summary, source_name, published_at
        FROM news_article
        WHERE category = 'AI'
          AND published_at >= '2026-03-01'
          AND published_at < '2026-04-01'
        ORDER BY id
    """
    if limit > 0:
        sql += f" LIMIT {limit}"
    cur.execute(sql)
    articles = cur.fetchall()
    conn.close()
    return articles


def build_batch_prompt(articles: list[dict]) -> str:
    """构建批量打标的用户消息。"""
    lines = []
    for art in articles:
        title = (art["title"] or "").strip()[:100]
        summary = (art["summary"] or "").strip()[:150]
        line = f'[ID={art["id"]}] {title}'
        if summary:
            line += f"\n  摘要: {summary}"
        lines.append(line)
    return "请对以下文章逐一打标：\n\n" + "\n\n".join(lines)


def call_llm(user_message: str, retry: int = 6) -> list[dict]:
    """调用 LLM，支持 SSE 流式响应，返回解析后的标签列表。

    重试策略：指数退避，最多 6 次（5s, 10s, 20s, 40s, 80s, 160s）。
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
        "stream": False,
    }

    for attempt in range(retry):
        try:
            resp = requests.post(
                OPENROUTER_BASE_URL,
                headers=headers,
                json=payload,
                timeout=180,  # 加长超时到 3 分钟
            )
            resp.raise_for_status()

            # 尝试标准 JSON 解析（非流式）
            try:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
            except (json.JSONDecodeError, KeyError):
                # 回退：解析 SSE 流式格式（data: {...}\n）
                content_parts = []
                for line in resp.text.strip().split("\n"):
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    chunk_str = line[len("data: "):]
                    if chunk_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(chunk_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            content_parts.append(delta["content"])
                    except json.JSONDecodeError:
                        continue
                content = "".join(content_parts)

            if not content.strip():
                raise ValueError("LLM 返回空内容")

            # 提取 JSON 数组（可能被 markdown 包裹）
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(
                    line for line in lines
                    if not line.strip().startswith("```")
                )
            return json.loads(content)

        except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError) as e:
            wait = 5 * (2 ** attempt)  # 5s, 10s, 20s, 40s, 80s, 160s
            print(f"    ⚠️ 调用失败 (attempt {attempt + 1}/{retry}): {e}")
            if attempt < retry - 1:
                print(f"    ⏳ 等待 {wait}s 后重试...")
                time.sleep(wait)
    return []


def label_batch(articles: list[dict]) -> list[dict]:
    """对一批文章调用 LLM 打标。"""
    prompt = build_batch_prompt(articles)
    results = call_llm(prompt)

    # 构建 id→article 映射，用于补充元数据
    art_map = {art["id"]: art for art in articles}

    labeled = []
    for item in results:
        # 兼容 "id"/"ID" 和 字符串/整数
        raw_id = item.get("id") or item.get("ID")
        try:
            art_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if art_id not in art_map:
            continue
        art = art_map[art_id]

        # entities 可能是字符串列表或对象列表
        entities = item.get("entities", ["general"])
        if entities and isinstance(entities[0], dict):
            entities = [e.get("entity", e.get("name", str(e))) for e in entities]

        # event_type 校验
        event_type = item.get("event_type", "other")
        if event_type not in EVENT_TYPES:
            event_type = "other"

        labeled.append({
            "article_id": art_id,
            "title": (art["title"] or "").strip()[:120],
            "published_at": str(art["published_at"])[:10],
            "source_name": art["source_name"],
            "event_type": event_type,
            "entities": entities,
        })
    return labeled


def main():
    parser = argparse.ArgumentParser(description="AI 新闻文章打标工具")
    parser.add_argument("--limit", type=int, default=0, help="限制处理文章数（0=全部）")
    parser.add_argument("--resume", action="store_true", help="从上次中断处继续")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="每批文章数")
    parser.add_argument("--workers", type=int, default=10, help="并发数")
    args = parser.parse_args()

    # 默认总是加载已有标签（断点续打）
    existing = load_existing_labels(OUTPUT_FILE)
    if args.resume is False and existing:
        print(f"⚠️ 发现已有 {len(existing)} 条标签，自动启用断点续打")
    print(f"📂 已有标签: {len(existing)} 条")

    # 拉取文章
    articles = fetch_articles(limit=args.limit)
    print(f"📰 数据库文章: {len(articles)} 条")

    # 过滤已打标的
    articles = [a for a in articles if a["id"] not in existing]
    print(f"🔄 待打标: {len(articles)} 条")

    if not articles:
        print("✅ 全部已打标完成")
        return

    # 分批处理
    labels = dict(existing)
    batches = [articles[i : i + args.batch_size] for i in range(0, len(articles), args.batch_size)]
    total_batches = len(batches)
    failed_articles = []  # 收集失败的文章

    def process_batch(batch, batch_num):
        id_range = f"{batch[0]['id']}~{batch[-1]['id']}"
        time.sleep(0.2 * (batch_num % args.workers))  # 稍微错开并发请求
        results = label_batch(batch)
        return batch_num, batch, results, id_range

    print(f"🚀 以 {args.workers} 个并发线程开始处理...")
    completed_batches = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_batch = {
            executor.submit(process_batch, batch, idx + 1): batch
            for idx, batch in enumerate(batches)
        }

        for future in concurrent.futures.as_completed(future_to_batch):
            batch = future_to_batch[future]
            completed_batches += 1
            try:
                batch_num, _, results, id_range = future.result()
            except Exception as exc:
                print(f"⚠️ 批次处理遇到异常: {exc}")
                results = []
                batch_num = "?"
                id_range = f"{batch[-1]['id']}"

            for item in results:
                labels[item["article_id"]] = item

            # 记录本批失败的文章
            labeled_ids = {item["article_id"] for item in results}
            batch_failed = [a for a in batch if a["id"] not in labeled_ids]
            if batch_failed:
                failed_articles.extend(batch_failed)

            # 每批保存一次（断点续打）
            save_labels(labels, OUTPUT_FILE)
            print(f"  [{completed_batches}/{total_batches}] 批次 {batch_num} (ID: {id_range}) ✅ {len(results)}/{len(batch)} 标签" + (f" ⚠️ {len(batch_failed)}篇失败" if batch_failed else ""))


    # ── 补打失败的文章（缩小 batch 到 5 篇，增加成功率）──
    if failed_articles:
        print(f"\n{'─' * 50}")
        print(f"🔄 补打 {len(failed_articles)} 篇失败文章（batch=5）...")
        retry_batch_size = 5
        still_failed = []
        retry_batches = [failed_articles[i : i + retry_batch_size] for i in range(0, len(failed_articles), retry_batch_size)]
        
        completed_retries = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_rbatch = {
                executor.submit(process_batch, batch, idx + 1): batch
                for idx, batch in enumerate(retry_batches)
            }
            for future in concurrent.futures.as_completed(future_to_rbatch):
                batch = future_to_rbatch[future]
                completed_retries += 1
                try:
                    batch_num, _, results, id_range = future.result()
                except Exception as exc:
                    results = []
                    batch_num = "?"
                    id_range = f"{batch[-1]['id']}"

                for item in results:
                    labels[item["article_id"]] = item

                labeled_ids = {item["article_id"] for item in results}
                batch_still_failed = [a for a in batch if a["id"] not in labeled_ids]
                still_failed.extend(batch_still_failed)

                save_labels(labels, OUTPUT_FILE)
                print(f"  [补打 {completed_retries}/{len(retry_batches)}] 批次 {batch_num} (ID: {id_range}) ✅ {len(results)}/{len(batch)}")

        if still_failed:
            failed_ids = [a["id"] for a in still_failed]
            print(f"\n⚠️ 仍有 {len(still_failed)} 篇无法打标: {failed_ids}")
            print(f"   请稍后用 --resume 重新运行来补打这些文章")

    # 统计
    all_labels = list(labels.values())
    type_counts = {}
    for item in all_labels:
        t = item["event_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n{'=' * 50}")
    print(f"✅ 打标完成: {len(all_labels)} / {len(fetch_articles())} 篇")
    print(f"📁 保存至: {OUTPUT_FILE}")
    print(f"\n事件类型分布:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:20s} {c:5d} ({c / len(all_labels) * 100:.1f}%)")


if __name__ == "__main__":
    main()
