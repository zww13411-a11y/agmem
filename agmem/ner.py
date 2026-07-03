"""木行 — hybrid entity extraction for AgMem memory graph.

**触 → 冒 → 蠢 → 动 → 震**

Three-tier approach (no LLM calls):

1. **Precoded patterns** — regex rules for common expressions in both EN and ZH
2. **Known-entity patterns** — structural patterns (acronyms, CamelCase, project names)
3. **Noun-chunk fallback** — spaCy for EN, jieba for ZH

Plus:
- **Synonym map** — normalize aliases (DB → 数据库 → PostgreSQL → postgres)
- **Implicit reference resolver** — find mentions from recent context
"""

from __future__ import annotations

import re
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Tier 0 — Synonym map (木行: 蠢 — 群起响应, resonance across aliases)
# ---------------------------------------------------------------------------

SYNONYM_MAP: Dict[str, str] = {
    # Database / storage
    "sqlite": "SQLite",
    "sqlite3": "SQLite",
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "pg": "PostgreSQL",
    "pg database": "PostgreSQL",
    "db": "DB",
    "database": "DB",
    "数据库": "DB",
    "redis": "Redis",
    "缓存": "cache",
    "cache": "cache",
    "caching": "cache",
    "redis cache": "Redis",
    "记忆": "memory",
    "记忆系统": "memory system",
    "memory system": "memory system",
    "mem": "memory",
    # Python ecosystem
    "python": "Python",
    "python3": "Python",
    "fastapi": "FastAPI",
    "pydantic": "Pydantic",
    "sqlalchemy": "SQLAlchemy",
    "pandas": "Pandas",
    "numpy": "NumPy",
    # ML/AI
    "llm": "LLM",
    "lm": "LM",
    "ai": "AI",
    "人工智能": "AI",
    "大模型": "LLM",
    "机器学习": "ML",
    "ml": "ML",
    "transformer": "Transformer",
    "neural network": "NN",
    "神经网络": "NN",
    "梯度": "gradient",
    "gradient": "gradient",
    "loss": "loss",
    "损失": "loss",
    "embedding": "embedding",
    "embed": "embedding",
    "向量": "vector",
    "vector": "vector",
    # Programming
    "async": "async",
    "await": "async",
    "异步": "async",
    "docker": "Docker",
    "容器": "container",
    "container": "container",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "api": "API",
    "restful": "REST API",
    "rest": "REST API",
    "rest api": "REST API",
    # Human / user
    "zhouwang": "周旺",
    "旺旺": "周旺",
    "user": "user",
    "用户": "user",
    "developer": "developer",
    "工程师": "engineer",
    "engineer": "engineer",
    # Concepts
    "open source": "open source",
    "开源": "open source",
    "privacy": "privacy",
    "隐私": "privacy",
    "security": "security",
    "安全": "security",
    "performance": "performance",
    "性能": "performance",
    "memory": "memory",
    "记忆": "memory",
    "五行": "五行",
    "wuxing": "五行",
    "five elements": "五行",
}

# Reverse map for checking known synonyms
_NORM_LOOKUP: Dict[str, str] = {}
for k, v in SYNONYM_MAP.items():
    _NORM_LOOKUP[k.lower()] = v
    _NORM_LOOKUP[v.lower()] = v


def normalize_entity(entity: str) -> str:
    """Normalize an entity through the synonym map.

    'postgresql' → 'PostgreSQL', '数据库' → 'DB', etc.
    """
    key = entity.lower().strip()
    return _NORM_LOOKUP.get(key, entity.strip())


# ---------------------------------------------------------------------------
# Tier 1 — Precoded sentence patterns (EN + ZH)
# ---------------------------------------------------------------------------

_STOP_WORDS: FrozenSet[str] = frozenset({
    "and", "or", "but", "for", "with", "without", "than", "over",
    "the", "a", "an", "in", "on", "at", "to", "of", "by", "is", "was",
    "的", "了", "是", "在", "有", "和", "与", "就", "也", "还",
    "都", "要", "会", "能", "可以", "这个", "那个", "一个",
})


def _split_at_stop(phrase: str) -> List[str]:
    """Split a captured phrase on stop words, returning non-empty fragments."""
    parts = re.split(r"\s+", phrase)
    fragments: List[str] = []
    current: List[str] = []
    for part in parts:
        if part.lower() in _STOP_WORDS:
            if current:
                fragments.append(" ".join(current))
                current = []
        else:
            current.append(part)
    if current:
        fragments.append(" ".join(current))
    return [f for f in fragments if not _is_low_value(f)]


def _split_zh(phrase: str) -> List[str]:
    """Split a Chinese phrase into segments around stop words."""
    for stop in _STOP_WORDS:
        if stop in phrase:
            parts = phrase.split(stop)
            return [p.strip() for p in parts if p.strip() and not _is_low_value(p.strip())]
    return [phrase.strip()] if phrase.strip() and not _is_low_value(phrase.strip()) else []


_SENTENCE_PATTERNS: List[re.Pattern] = [
    # English patterns
    re.compile(r"\bI\s+(?:like|love|enjoy|prefer|favor)\s+(.+)", re.I),
    re.compile(r"\bI\s+(?:use|use\s+to|work\s+with)\s+(.+)", re.I),
    re.compile(r"\b(?:prefer|preferred)\s+(.+?)\s+(?:over|to|than)", re.I),
    re.compile(r"\bI\s+don'?t\s+(?:like|prefer|use)\s+(.+)", re.I),
    re.compile(r"\bMy\s+favorite\s+\w+\s+is\s+(.+)", re.I),
    re.compile(r"\b\w+\s+is\s+(?:good|great|excellent)\s+(?:for|at)\s+(.+)", re.I),
    re.compile(r"\b(?:using|working\s+with|dealing\s+with)\s+(.+)", re.I),
    re.compile(r"\b(?:known\s+for|famous\s+for)\s+(.+)", re.I),
    # Chinese patterns
    re.compile(r"(?:我|我们|本人)\s*(?:喜欢|偏好|偏爱|推荐|习惯使用|常用)\s*(.+)"),
    re.compile(r"(?:不喜欢|不太用|不用|少用)\s*(.+)"),
    re.compile(r"(?:觉得|认为|感觉)\s*(.+?)\s*(?:很|非常|比较|挺)\s*(?:好|棒|方便|实用|厉害)"),
]

# ---------------------------------------------------------------------------
# Tier 2 — Known-entity / structural patterns
# ---------------------------------------------------------------------------

_ENTITY_PATTERNS: List[re.Pattern] = [
    # "the [Something] project / framework / library / tool"
    re.compile(r"the\s+([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\s+(?:project|framework|library|tool|system|platform|protocol|language|database|server)", re.I),
    # Quoted multi-word terms (EN)
    re.compile(r"[\"']([A-Za-z][A-Za-z0-9\s.-]{2,})[\"']"),
    # Quoted terms (ZH)
    re.compile(r"「([^」]{2,})」"),
    re.compile(r"『([^』]{2,})』"),
    # Acronyms (2-6 uppercase letters, possibly with dots)
    re.compile(r"\b([A-Z][A-Z0-9.]{1,5})\b"),
    # CamelCase / PascalCase identifiers
    re.compile(r"\b([A-Z][a-z]+[A-Z][a-zA-Z0-9]+)\b"),
    # Versioned references: "Python 3.12", "PostgreSQL 16"
    re.compile(r"\b([A-Za-z]\w{1,})\s+(\d+[.\d]*)\b"),
    # Chinese-specific: proper nouns with 的
    re.compile(r"\b([A-Z]\w{1,})(?:的|之)(?:问题|方案|框架|方法|设计|模式|架构)"),
    # Short lowercase compounds where both words are in the synonym map
    re.compile(r"\b([a-z]{2,10})\s+([a-z]{2,12})\b"),
]

# Words that are known synoym keys — helps the short-compound pattern
_KNOWN_SYNONYM_WORDS: FrozenSet[str] = frozenset(
    w.lower() for w in SYNONYM_MAP.keys()
    if " " not in w and len(w) >= 2
)

_LOW_VALUE_WORDS: FrozenSet[str] = frozenset({
    "this", "that", "these", "those", "it", "they", "them",
    "we", "you", "he", "she", "i", "me", "my", "mine",
    "your", "our", "their", "its", "the", "a", "an",
    "and", "or", "but", "if", "then", "else", "when",
    "where", "what", "why", "how", "who", "whom",
    "thing", "things", "stuff", "something", "anything",
    "way", "ways", "time", "times", "kind", "kinds",
    "type", "types", "part", "parts", "example", "examples",
    "really", "very", "quite", "much", "also", "just",
    "here", "there", "then", "now", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "above",
    "below", "more", "less", "most", "least", "few", "many",
    "one", "two", "three", "first", "second", "third",
    "do", "does", "did", "done", "doing", "make", "makes",
    "made", "get", "gets", "got", "go", "goes", "went", "gone",
    "好", "很", "太", "非常", "比较", "也", "还", "就",
    "这个", "那个", "什么", "怎么", "为什么", "如何",
})


def _is_low_value(word: str) -> bool:
    return word.lower() in _LOW_VALUE_WORDS

# ---------------------------------------------------------------------------
# Tier 3 — spaCy EN + jieba ZH fallback
# ---------------------------------------------------------------------------

_NLP = None
_JIEBA = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm", disable=["lemmatizer", "textcat"])
        except ImportError:
            pass  # graceful fallback
        except OSError:
            pass
    return _NLP


def _get_jieba():
    global _JIEBA
    if _JIEBA is None:
        try:
            import jieba
            _JIEBA = jieba
        except ImportError:
            logger.warning("jieba not installed — Chinese entity extraction disabled")
    return _JIEBA


# ---------------------------------------------------------------------------
# Chinese-text detection
# ---------------------------------------------------------------------------

_ZH_RE = re.compile(r"[\u4e00-\u9fff]")


def _has_chinese(text: str) -> bool:
    return bool(_ZH_RE.search(text))


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def _normalise(e: str) -> str:
    return e.lower().strip()


def _deduplicate(entities: List[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for e in entities:
        key = _normalise(e)
        if key not in seen and e.strip():
            seen.add(key)
            result.append(e.strip())
    return result


def _split_chinese_compounds(entities: List[str]) -> List[str]:
    """Break Chinese multi-character strings into known synonym pieces.

    '数据库设计' → ['数据库', '设计'] → '数据库' is in synonym map → keep
    """
    result: List[str] = []
    for e in entities:
        if not _has_chinese(e):
            result.append(e)
            continue
        # Check if this exact entity is already a known synonym target
        norm = normalize_entity(e)
        if norm != e or e.lower() in _NORM_LOOKUP:
            result.append(norm)
            continue
        # Try to split at 2-char boundaries for known synonym keywords
        if len(e) >= 4 and len(e) <= 8:
            found_parts = False
            for i in range(2, len(e) - 1, 2):
                prefix = e[:i]
                suffix = e[i:]
                p_norm = normalize_entity(prefix)
                s_norm = normalize_entity(suffix)
                if p_norm != prefix:
                    result.append(p_norm)
                    if s_norm != suffix and not _is_low_value(s_norm):
                        result.append(s_norm)
                    found_parts = True
                    break
            if not found_parts:
                result.append(e)
        else:
            result.append(e)
    return result


def _apply_synonyms(entities: List[str]) -> List[str]:
    """Normalize all entities through the synonym map.

    After normalization, dedup again (multiple aliases → one canonical form).
    Also expands known compound terms.
    """
    entities = _split_chinese_compounds(entities)
    return _deduplicate([normalize_entity(e) for e in entities])


# ---------------------------------------------------------------------------
# Implicit reference resolver (木行: 触 — 从语境中触碰)
# ---------------------------------------------------------------------------

_IMPLICIT_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # "那个[thing]" — anaphora: 那个缩放的问题 → 缩放
    (re.compile(r"那个\s*[的]?\s*(.{2,8}?)(?:的\s*(?:问题|方案|方法|设计|模式|框架|架构|想法))?"), "anaphora_ref"),
    # "刚才说的[thing]"
    (re.compile(r"刚才\s*(?:说|提|讲)\s*[的]?\s*(.{2,12})"), "contextual_ref"),
    # "上次的[thing]"
    (re.compile(r"上次\s*[的]?\s*(.{2,12})"), "prior_ref"),
    # "这个[thing]" — also anaphora
    (re.compile(r"这个\s*(.{2,8}?)(?:问题|方案|方法|设计|模式|想法)"), "deictic_ref"),
]


def extract_implicit_references(text: str) -> List[str]:
    """Extract implicit references that point to prior entities.

    '那个缩放的问题' → ['缩放']
    '刚才说的五行映射' → ['五行映射']
    """
    references: List[str] = []
    for pat, _kind in _IMPLICIT_PATTERNS:
        for m in pat.finditer(text):
            ref = m.group(1).strip()
            if ref and not _is_low_value(ref) and len(ref) >= 2:
                references.append(ref)
    return _deduplicate(references)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_entities(text: str) -> List[str]:
    """Run all extraction tiers and return a deduplicated, synonym-normalized entity list.

    Supports mixed EN/ZH input.  Tiers are applied in order;
    noun-chunk fallback only fires when regex found nothing.
    """
    if not text or not text.strip():
        return []

    entities: List[str] = []

    # Tier 1 — sentence patterns
    for pat in _SENTENCE_PATTERNS:
        for m in pat.finditer(text):
            raw = m.group(1).strip()
            if raw:
                if _has_chinese(raw):
                    entities.extend(_split_zh(raw))
                else:
                    entities.extend(_split_at_stop(raw))
    # Tier 2 — known-entity patterns
    for pat in _ENTITY_PATTERNS:
        for m in pat.finditer(text):
            # 'versioned reference' groups: "Python 3.12"
            candidate = m.group(1) + " " + m.group(2) if len(m.groups()) >= 2 else m.group(1)

            # Short compound filter: only keep if at least one word is a known synonym
            # AND the first word isn't a stop word
            if len(m.groups()) >= 2 and re.match(r"^[a-z\s]+$", candidate.strip()):
                words = candidate.strip().split()
                first_lower = words[0].lower()
                if first_lower in {"for", "to", "in", "on", "at", "by", "with", "from", "of", "and", "or", "the", "a", "an", "use", "using"}:
                    continue
                if not any(w.lower() in _KNOWN_SYNONYM_WORDS for w in words):
                    continue

            candidate = candidate.strip()
            if candidate and not _is_low_value(candidate) and len(candidate) >= 2:
                entities.append(candidate)

    entities = _deduplicate(entities)

    # Implicit references — always run, always add
    entities.extend(extract_implicit_references(text))

    # Tier 3 — noun-chunk fallback (only if regex found nothing substantive)
    if not [e for e in entities if len(e) >= 3]:
        found_zh = False
        if _has_chinese(text):
            jieba = _get_jieba()
            if jieba:
                for word in jieba.cut(text, cut_all=False):
                    w = word.strip()
                    if len(w) >= 2 and not _is_low_value(w):
                        entities.append(w)
                        found_zh = True

        if not found_zh:
            nlp = _get_nlp()
            if nlp:
                doc = nlp(text)
                for chunk in doc.noun_chunks:
                    candidate = chunk.text.strip()
                    if len(candidate) >= 2 and not _is_low_value(candidate):
                        if candidate.lower() not in _LOW_VALUE_WORDS:
                            entities.append(candidate)

    entities = _deduplicate(entities)

    # Apply synonym normalization — 木行: 蠢 (群起响应)
    entities = _apply_synonyms(entities)

    return entities
