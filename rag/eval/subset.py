"""从 MariaDB 抽取 100 篇 AI 类最新文章作为实验子集。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymysql
from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
from scripts.clean_html import clean_html

def fetch_subset(n=100):
    conn = pymysql.connect(host=DB_HOST, port=DB_PORT, user=DB_USER,
        password=DB_PASSWORD, database=DB_NAME, charset="utf8mb4")
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("""
                SELECT id, title, summary, raw_content, category, source_name, published_at
                FROM news_article
                WHERE category = 'AI' AND title IS NOT NULL AND title != ''
                    AND raw_content IS NOT NULL AND LENGTH(raw_content) > 500
                ORDER BY published_at DESC
                LIMIT %s
            """, (n,))
            return cur.fetchall()
    finally:
        conn.close()

if __name__ == "__main__":
    articles = fetch_subset(100)
    # 清洗并保存为 JSON，后续实验直接读这个文件
    out = []
    for a in articles:
        out.append({
            "id": a["id"],
            "title": a["title"],
            "summary": clean_html(a["summary"] or ""),
            "raw_content": clean_html(a["raw_content"] or ""),
            "category": a["category"],
            "source_name": a["source_name"],
            "published_at": str(a["published_at"]) if a["published_at"] else "",
        })
    path = Path(__file__).parent / "results" / "subset_100.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # 统计
    avg_len = sum(len(a["raw_content"]) for a in out) // len(out)
    print(f"✅ 已保存 {len(out)} 篇文章到 {path}")
    print(f"   平均正文长度: {avg_len} 字符")
    print(f"   时间范围: {out[-1]['published_at'][:10]} ~ {out[0]['published_at'][:10]}")
