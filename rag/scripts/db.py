"""数据库访问模块：封装 MariaDB 查询。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymysql
from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME


def get_connection():
    """获取 MariaDB 连接。"""
    return pymysql.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER,
        password=DB_PASSWORD, database=DB_NAME, charset="utf8mb4"
    )


def fetch_all_articles():
    """读取所有文章，返回 dict 列表。"""
    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("""
                SELECT id, title, summary, category, source_name,
                       published_at, raw_content
                FROM news_article
                WHERE title IS NOT NULL AND title != ''
                ORDER BY id
            """)
            return cur.fetchall()
    finally:
        conn.close()


if __name__ == "__main__":
    rows = fetch_all_articles()
    print(f"共 {len(rows)} 篇文章")
    print(f"示例: {rows[0]['title'][:50]}...")
