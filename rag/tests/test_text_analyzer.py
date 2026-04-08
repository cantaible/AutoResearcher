"""BM25 analyzer 最小回归测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from text_analyzer import analyze_text, normalize_text


def test_normalize_text_compacts_dates():
    normalized = normalize_text("OpenAI GPT-5.4 在 2026 年 3 月发布了吗？")
    assert "2026年3月" in normalized


def test_analyze_text_extracts_dates_and_versions():
    tokens = analyze_text("OpenAI GPT-5.4 在 2026 年 3 月发布了吗？")
    assert "openai" in tokens
    assert "gpt-5.4" in tokens
    assert "gpt" in tokens
    assert "5.4" in tokens
    assert "2026年3月" in tokens
    assert "3月" in tokens
    assert "发布" in tokens


def test_analyze_text_keeps_month_only_queries():
    tokens = analyze_text("4月将发布哪些模型？")
    assert "4月" in tokens
    assert "发布" in tokens
    assert "模型" in tokens
