"""BM25 分析器：负责标准化、结构化 token 抽取和中英文分词。"""
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pkuseg

RAG_DIR = Path(__file__).parent
STOPWORDS_PATH = RAG_DIR / "resources" / "bm25_stopwords_zh.txt"

ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
FULL_DATE_RE = re.compile(r"(?<!\d)(\d{4})\s*年\s*(\d{1,2})\s*月(?:\s*(\d{1,2})\s*日)?")
MONTH_DAY_RE = re.compile(r"(?<!\d)(\d{1,2})\s*月(?:\s*(\d{1,2})\s*日)?")
COMPOUND_TOKEN_RE = re.compile(
    r"\b[a-z0-9]+(?:[-_.][a-z0-9]+)+\b"
    r"|\b[a-z]+\d+[a-z0-9]*\b"
    r"|\bv\d+(?:\.\d+)*\b"
    r"|\b\d+(?:\.\d+)+(?:[a-z]+)?\b"
    r"|\b\d+[a-z]+\b",
    re.IGNORECASE,
)
ASCII_WORD_RE = re.compile(r"\b[a-z]{2,}\b", re.IGNORECASE)
NUMBER_TOKEN_RE = re.compile(r"\d+(?:\.\d+)*")

DASH_TRANSLATION = str.maketrans({
    "—": "-",
    "–": "-",
    "－": "-",
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "﹣": "-",
})


def _replace_full_date(match: re.Match[str]) -> str:
    year = int(match.group(1))
    month = int(match.group(2))
    day = match.group(3)
    if day is None:
        return f"{year}年{month}月"
    return f"{year}年{month}月{int(day)}日"


def _replace_month_day(match: re.Match[str]) -> str:
    month = int(match.group(1))
    day = match.group(2)
    if day is None:
        return f"{month}月"
    return f"{month}月{int(day)}日"


@lru_cache(maxsize=1)
def load_stopwords() -> set[str]:
    stopwords = set()
    if not STOPWORDS_PATH.exists():
        return stopwords

    for raw_line in STOPWORDS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            stopwords.add(line)
    return stopwords


@lru_cache(maxsize=1)
def load_segmenter():
    """延迟加载 pkuseg，避免模块导入时就初始化模型。"""
    return pkuseg.pkuseg()


def normalize_text(text: str) -> str:
    """统一大小写、空白、全半角和日期表达，减少 analyzer 分支复杂度。"""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = text.translate(DASH_TRANSLATION)
    text = FULL_DATE_RE.sub(_replace_full_date, text)
    text = MONTH_DAY_RE.sub(_replace_month_day, text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _append_once(tokens: list[str], seen: set[str], token: str) -> None:
    if token and token not in seen:
        tokens.append(token)
        seen.add(token)


def _extract_date_tokens(text: str) -> list[str]:
    tokens: list[str] = []

    for match in FULL_DATE_RE.finditer(text):
        local_seen: set[str] = set()
        year = int(match.group(1))
        month = int(match.group(2))
        day = match.group(3)
        month_token = f"{year}年{month}月"
        _append_once(tokens, local_seen, month_token)
        _append_once(tokens, local_seen, f"{year}年")
        _append_once(tokens, local_seen, f"{month}月")
        if day is not None:
            day_int = int(day)
            _append_once(tokens, local_seen, f"{month_token}{day_int}日")
            _append_once(tokens, local_seen, f"{month}月{day_int}日")

    masked_text = FULL_DATE_RE.sub(" ", text)
    for match in MONTH_DAY_RE.finditer(masked_text):
        local_seen: set[str] = set()
        month = int(match.group(1))
        day = match.group(2)
        _append_once(tokens, local_seen, f"{month}月")
        if day is not None:
            _append_once(tokens, local_seen, f"{month}月{int(day)}日")

    return tokens


def _expand_compound_token(token: str) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    _append_once(expanded, seen, token)

    for part in re.split(r"[-_]", token):
        part = part.strip()
        if not part:
            continue
        _append_once(expanded, seen, part)

        if part.startswith("v") and NUMBER_TOKEN_RE.fullmatch(part[1:]):
            _append_once(expanded, seen, part[1:])
            continue

        prefix_match = re.fullmatch(r"([a-z]+)(\d+(?:\.\d+)*)", part)
        if prefix_match:
            prefix, version = prefix_match.groups()
            _append_once(expanded, seen, prefix)
            if "." in version or len(version) > 1:
                _append_once(expanded, seen, version)
            continue

        suffix_match = re.fullmatch(r"(\d+(?:\.\d+)*)([a-z]+)", part)
        if suffix_match:
            version, suffix = suffix_match.groups()
            _append_once(expanded, seen, suffix)
            if "." in version or len(version) > 1:
                _append_once(expanded, seen, version)
            continue

        if NUMBER_TOKEN_RE.fullmatch(part) and ("." in part or len(part) > 1):
            _append_once(expanded, seen, part)

    return expanded


def _extract_compound_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for match in COMPOUND_TOKEN_RE.finditer(text):
        tokens.extend(_expand_compound_token(match.group(0)))
    return tokens


def _mask_patterns(text: str, patterns: list[re.Pattern[str]]) -> str:
    masked = text
    for pattern in patterns:
        masked = pattern.sub(" ", masked)
    return masked


def _is_noise_token(token: str, stopwords: set[str]) -> bool:
    if not token:
        return True
    if token in stopwords:
        return True

    if all(
        unicodedata.category(char).startswith(("P", "S")) or char.isspace()
        for char in token
    ):
        return True

    if len(token) == 1:
        if token in {"年", "月", "日"}:
            return True
        return False

    return False


def analyze_text(text: str) -> list[str]:
    """产出适合 BM25 的 token 序列。"""
    normalized = normalize_text(text)
    if not normalized:
        return []

    stopwords = load_stopwords()
    tokens: list[str] = []
    tokens.extend(_extract_date_tokens(normalized))
    tokens.extend(_extract_compound_tokens(normalized))

    masked_for_ascii = _mask_patterns(
        normalized,
        [FULL_DATE_RE, MONTH_DAY_RE, COMPOUND_TOKEN_RE],
    )
    tokens.extend(ASCII_WORD_RE.findall(masked_for_ascii))

    masked_for_cjk = _mask_patterns(masked_for_ascii, [ASCII_WORD_RE, NUMBER_TOKEN_RE])
    tokens.extend(load_segmenter().cut(masked_for_cjk))

    return [token for token in tokens if not _is_noise_token(token.strip(), stopwords)]
