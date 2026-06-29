# AgMem — Agent Memory Graph

> 五行驱动的 LLM Agent 长期记忆系统。  
> Five Elements (Wu Xing) inspired long-term memory for LLM agents.

SQLite-backed, FTS5 full-text search, graph-walk retrieval, Hebbian weight dynamics, inspection-based pruning, and consolidation — all in pure Python with zero external dependencies (optional spaCy/jieba for NER).

## Architecture (五行动力链)

```
木 — NER + entity resolution     (触→冒→蠢→动→震)
火 — FTS5 + graph-walk retrieval  (温→烟→燃→炎)
土 — SQLite WAL + consolidation   (凝→载→化→育→返)
金 — inspection + pruning         (涩→剥→断→割→决)
水 — user profile injection       (凝→润→陷→流→渊)
```

### Quick Start

```python
from agmem import AgMemDB, RecallTier, write_memory, retrieve, run_inspection

db = AgMemDB("memories.db")

# Write a memory
write_memory(db, "User prefers Python for backend work", "preference")

# Retrieve (4 tiers)
recall = retrieve(db, "python backend", tier=RecallTier.BURN)

# Inspection pass (dry-run first, then prune)
report = run_inspection(db, dry_run=True)
```

### Tiered Recall (火行)

| Tier | Hops | Use Case |
|------|:----:|----------|
| 温 (warm) | 0 | Quick FTS5 check |
| 烟 (smoke) | 1 | Light context |
| 燃 (burn) | 3 | Standard retrieval |
| 炎 (inferno) | 5 | Deep dive |

### Key Features

- **FTS5 full-text search** — instant keyword matching
- **Synonym normalization** — 60+ aliases (postgresql↔PostgreSQL, 数据库↔DB)
- **Chinese + English** — mixed-language entity extraction
- **Implicit references** — "那个缩放的问题" → "缩放"
- **Graph walk retrieval** — multi-hop memory association
- **Hebbian dynamics** — weights strengthen on successful recall
- **Inspection pass** — 4 verdicts: keep / archive / prune / review
- **Consolidation** — dedup + merge + orphan link cleanup
- **User profile** — generates system-prompt-ready summary

## Dependencies

- **Required:** Python 3.11+, SQLite3 (stdlib)
- **Optional:** `spacy` + `en_core_web_sm` (English noun-chunk NER)
- **Optional:** `jieba` (Chinese word segmentation)

## Tests

```bash
uv run python test.py
```

12 tests covering all modules.

## LICENSE

MIT
