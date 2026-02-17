# AI Tracker — Knowledge Graph Pipeline

End-to-end pipeline that turns an Excel spreadsheet of AI research papers into a queryable **Neo4j knowledge graph**, with a natural-language chat interface.

```
Excel → JSON → LLM Extraction → Neo4j Ingestion → NL Query Chat
```

## Architecture

```
┌──────────────┐     ┌─────────────────────┐     ┌──────────────┐     ┌──────────┐
│ AI_Tracker   │     │ relation_extraction  │     │  ingestion   │     │  nlq.py  │
│   .xlsx      │────▶│      .py             │────▶│     .py      │────▶│  (chat)  │
│              │     │  LLM entity/relation │     │  Neo4j load  │     │  NL→Cypher│
└──────────────┘     │  extraction          │     │              │     └──────────┘
                     └─────────────────────┘     └──────────────┘
                              │                         │
                     ┌────────▼────────┐       ┌────────▼────────┐
                     │ llm_providers.py│       │     Neo4j       │
                     │ Groq → Gemini   │       │  :Paper :Entity │
                     │ auto-fallback   │       │  :MENTIONS      │
                     └─────────────────┘       │  :RELATION      │
                                               └─────────────────┘
```

## Graph Schema

| Node | Properties | Description |
|------|-----------|-------------|
| `:Paper` | title, date, category, impact, enhancement, link | Research paper / article |
| `:Entity` | name, type, last_source | AI_Model, Technology, Concept, Industry, Tool, Organization, etc. |

| Edge | Direction | Description |
|------|-----------|-------------|
| `:MENTIONS` | Paper → Entity | Paper discusses this entity |
| `:RELATION` | Entity → Entity | Typed relationship (UTILIZES, ENHANCES, APPLICABLE_IN, etc.) |

## Setup

### Prerequisites
- Python 3.11+
- Neo4j (local or Aura)
- At least one LLM API key (Groq or Gemini)

### Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install pandas openpyxl python-dotenv neo4j openai
```

### Configure

```bash
cp .env.example .env
# Edit .env with your Neo4j credentials and API keys
```

## Usage

### 1. Extract entities & relations from Excel

```bash
python relation_extraction.py
```

- Reads `data/AI_Tracker.xlsx` → extracts entities & relations via LLM
- **Incremental**: only processes new rows (compares against existing output)
- Output: `data/AI_Tracker_relations.json`

| Flag | Description |
|------|-------------|
| `--full` | Re-extract all rows (ignore existing) |
| `--limit N` | Process only first N new rows |
| `--input PATH` | Custom Excel path |

### 2. Ingest into Neo4j

```bash
python ingestion.py
```

- Loads `data/AI_Tracker_relations.json` → creates Paper, Entity, MENTIONS, RELATION in Neo4j
- **Incremental**: uses MERGE (safe to re-run)

| Flag | Description |
|------|-------------|
| `--clear` | Wipe entire graph before ingesting |
| `--apoc` | Use APOC for dynamic entity labels |

### 3. Query with natural language

```bash
python nlq.py
```

Interactive chat that converts questions to Cypher, runs against Neo4j, and summarizes results.

**Example questions:**
```
❓ latest news
❓ what's new in reinforcement learning
❓ papers about GPT
❓ how are knowledge graphs and RAG related
❓ which AI models apply to cybersecurity
❓ what do WarpGrep and COSMO have in common
```

## LLM Providers

Configured in `llm_providers.py` with automatic fallback:

| Priority | Provider | Model | Env Var |
|----------|----------|-------|---------|
| 1 | Groq | llama-3.3-70b-versatile | `GROQ_API_KEY` |
| 2 | Gemini | gemini-2.0-flash | `GEMINI_API_KEY` |

On 429 rate-limit → retries with backoff → falls through to next provider.

Add more providers by extending the `PROVIDERS` list in `llm_providers.py`.

## Project Structure

```
├── .env.example              # Template for API keys & Neo4j config
├── .gitignore                # Ignores .env, .xlsx, .venv, __pycache__
├── llm_providers.py          # LLM abstraction with auto-fallback
├── relation_extraction.py    # Excel → JSON → LLM extraction
├── ingestion.py              # Relations JSON → Neo4j graph
├── nlq.py                    # Natural language query chat
└── data/
    ├── AI_Tracker.xlsx       # Source spreadsheet (gitignored)
    ├── AI_Tracker.json       # Raw JSON from Excel
    └── AI_Tracker_relations.json  # Extracted entities & relations
```
