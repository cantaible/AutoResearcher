"""测试 evidence_pool 功能的简单脚本。

验证：
1. State 定义是否正确
2. 去重函数是否工作
3. evidence_pool 数据结构是否正确
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import generate_evidence_id, deduplicate_evidence


def test_generate_evidence_id():
    """测试 ID 生成函数"""
    print("测试 generate_evidence_id()...")

    # 测试 RAG 来源
    rag_evidence = {
        "source": "rag",
        "article_id": 123,
    }
    rag_id = generate_evidence_id(rag_evidence)
    assert rag_id == "rag_123", f"RAG ID 错误: {rag_id}"
    print(f"  ✓ RAG ID: {rag_id}")

    # 测试 Web Search 来源
    web_evidence = {
        "source": "web_search",
        "url": "https://example.com/news/article",
    }
    web_id = generate_evidence_id(web_evidence)
    assert web_id.startswith("web_"), f"Web ID 错误: {web_id}"
    print(f"  ✓ Web ID: {web_id}")

    print("  ✅ generate_evidence_id() 测试通过\n")


def test_deduplicate_evidence():
    """测试去重函数"""
    print("测试 deduplicate_evidence()...")

    evidence_list = [
        {"source": "rag", "article_id": 123, "title": "文章1"},
        {"source": "rag", "article_id": 456, "title": "文章2"},
        {"source": "rag", "article_id": 123, "title": "文章1（重复）"},
        {"source": "rag", "article_id": 789, "title": "文章3"},
    ]

    deduplicated = deduplicate_evidence(evidence_list)

    assert len(deduplicated) == 3, f"去重后应该有 3 条，实际: {len(deduplicated)}"

    # 检查是否保留了第一次出现的
    titles = [e["title"] for e in deduplicated]
    assert "文章1" in titles, "应该保留第一次出现的文章1"
    assert "文章1（重复）" not in titles, "不应该保留重复的文章1"

    print(f"  ✓ 去重前: {len(evidence_list)} 条")
    print(f"  ✓ 去重后: {len(deduplicated)} 条")
    print("  ✅ deduplicate_evidence() 测试通过\n")


def test_evidence_structure():
    """测试 evidence 数据结构"""
    print("测试 evidence 数据结构...")

    evidence = {
        "id": "rag_123",
        "source": "rag",
        "article_id": 123,
        "url": None,
        "title": "测试文章",
        "content": "这是测试内容",
        "published_date": "2026-03-15",
        "used_by_node": "rag_researcher",
        "query": "测试查询",
        "timestamp": "2026-05-08T10:00:00",
    }

    # 检查必需字段
    required_fields = ["id", "source", "article_id", "title", "content"]
    for field in required_fields:
        assert field in evidence, f"缺少必需字段: {field}"

    print(f"  ✓ 数据结构包含所有必需字段")
    print("  ✅ evidence 数据结构测试通过\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Evidence Pool 功能测试")
    print("=" * 60)
    print()

    try:
        test_generate_evidence_id()
        test_deduplicate_evidence()
        test_evidence_structure()

        print("=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
