"""验证 RAG 检索功能的测试脚本。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_search import rag_search

def test():
    queries = [
        ("2026年3月发布了哪些模型？", 30, "AI"),
    ]

    for q, days, cat in queries:
        print(f"\n{'='*50}")
        print(f"查询: '{q}' (范围: {days}天前, 分类: {cat or '不限'})")
        print(f"{'='*50}")
        
        # 因为 rag_search 是通过 @tool 包装的，我们需要调用它的 .invoke() 方法
        # 传入的参数必需是一个 dict
        result = rag_search.invoke({
            "query": q,
            "days": days,
            "category": cat,
            "top_k": 6
        })
        print(result)

if __name__ == "__main__":
    test()
