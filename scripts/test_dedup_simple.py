"""简单测试 evidence_pool 去重逻辑（不依赖其他模块）"""

import hashlib
from urllib.parse import urlparse


def generate_evidence_id(evidence: dict) -> str:
    """生成唯一 ID（复制自 utils.py）"""
    source = evidence.get("source")

    if source == "rag":
        article_id = evidence.get("article_id")
        return f"rag_{article_id}"

    elif source == "web_search":
        url = evidence.get("url", "")
        parsed = urlparse(url)
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        url_hash = hashlib.md5(clean_url.encode()).hexdigest()[:8]
        return f"web_{url_hash}"

    else:
        content = evidence.get("content", "")
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"unknown_{content_hash}"


def deduplicate_evidence(evidence_list: list[dict]) -> list[dict]:
    """基于唯一 ID 去重（复制自 utils.py）"""
    seen = {}
    for item in evidence_list:
        if "id" not in item:
            item["id"] = generate_evidence_id(item)

        item_id = item["id"]
        if item_id not in seen:
            seen[item_id] = item

    return list(seen.values())


# 测试
print("测试 1: RAG 来源去重")
evidence_list = [
    {"source": "rag", "article_id": 123, "title": "文章1"},
    {"source": "rag", "article_id": 456, "title": "文章2"},
    {"source": "rag", "article_id": 123, "title": "文章1（重复）"},
]
result = deduplicate_evidence(evidence_list)
print(f"  去重前: {len(evidence_list)} 条")
print(f"  去重后: {len(result)} 条")
assert len(result) == 2, "应该去重为 2 条"
print("  ✅ 通过\n")

print("测试 2: Web Search 来源去重")
evidence_list = [
    {"source": "web_search", "url": "https://example.com/news/1", "title": "新闻1"},
    {"source": "web_search", "url": "https://example.com/news/2", "title": "新闻2"},
    {"source": "web_search", "url": "https://example.com/news/1?utm=123", "title": "新闻1（带参数）"},
]
result = deduplicate_evidence(evidence_list)
print(f"  去重前: {len(evidence_list)} 条")
print(f"  去重后: {len(result)} 条")
assert len(result) == 2, "应该去重为 2 条（URL 参数被忽略）"
print("  ✅ 通过\n")

print("测试 3: 混合来源")
evidence_list = [
    {"source": "rag", "article_id": 123, "title": "RAG文章"},
    {"source": "web_search", "url": "https://example.com/news/1", "title": "Web文章"},
    {"source": "rag", "article_id": 123, "title": "RAG文章（重复）"},
]
result = deduplicate_evidence(evidence_list)
print(f"  去重前: {len(evidence_list)} 条")
print(f"  去重后: {len(result)} 条")
assert len(result) == 2, "应该去重为 2 条"
print("  ✅ 通过\n")

print("=" * 60)
print("✅ 所有测试通过！去重逻辑正确。")
print("=" * 60)
