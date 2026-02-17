"""xlsx → json → Gemini entity/relation extraction."""

from __future__ import annotations

import asyncio
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# ── Groq config ────────────────────────────────────────────────────────
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """\
You are an expert knowledge-graph builder.
Given a JSON record about an AI research paper / product, extract:

1. **entities** – named things (models, orgs, techniques, industries, tools, concepts, metrics).
2. **relations** – directional relationships between two *distinct* entities you extracted.

Return ONLY valid JSON (no markdown fences):
{
  "entities": [
    {"name": "...", "type": "AI_Model|Technology|Organization|Industry|Concept|Metric|Tool"}
  ],
  "relations": [
    {"source": "...", "target": "...", "relation": "SCREAMING_SNAKE_CASE", "fact": "one-line description"}
  ]
}

Rules:
- Names must be proper nouns or short technical terms, NOT long phrases.
- source / target must exactly match an entity name you listed.
- Do NOT extract dates as entities.
- Be thorough: cover ALL fields in the record."""


# ── Excel helpers ─────────────────────────────────────────────────────────

def load_excel(path: Path, sheet: str | int | None = 0) -> list[dict[str, Any]]:
    df = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    if isinstance(df, dict):
        df = next(iter(df.values()))
    return [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _ref_time(rec: Mapping[str, Any]) -> str:
    for key in ("Paper Date", "Log Date"):
        raw = rec.get(key)
        if raw is None:
            continue
        if isinstance(raw, datetime):
            return raw.isoformat()
        try:
            return datetime.fromisoformat(str(raw)).isoformat()
        except (ValueError, TypeError):
            pass
    return datetime.now().isoformat()


# ── Gemini caller with retry ─────────────────────────────────────────────

async def _call_llm(record_json: str) -> dict[str, Any]:
    text = await achat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record_json},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"entities": [], "relations": [], "_raw": text}


# ── Main extraction loop ─────────────────────────────────────────────────

async def extract(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for i, rec in enumerate(records):
        title = rec.get("Title", f"row-{i}")
        print(f"  [{i+1}/{len(records)}] {title}")

        try:
            extracted = await _call_llm(json.dumps(rec, default=str))
        except Exception as e:
            print(f"    ⚠ failed: {e}")
            extracted = {"entities": [], "relations": []}

        results.append({
            "title": title,
            "reference_time": _ref_time(rec),
            "source": rec,
            "entities": extracted.get("entities", []),
            "relations": extracted.get("relations", []),
        })

        # stay under free-tier 15 RPM
        if i < len(records) - 1:
            await asyncio.sleep(5)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="xlsx → json → relation extraction")
    p.add_argument("--input",  type=Path, default=Path("data/AI_Tracker.xlsx"))
    p.add_argument("--output", type=Path, default=Path("data/AI_Tracker.json"))
    p.add_argument("--relations-output", type=Path, default=Path("data/AI_Tracker_relations.json"))
    p.add_argument("--limit",  type=int,  default=None, help="only process first N rows")
    p.add_argument("--full",   action="store_true", help="re-extract ALL rows (ignore existing)")
    args = p.parse_args()

    print(f"LLM providers (fallback order):\n{list_providers()}")

    # Step 1: xlsx → json
    records = load_excel(args.input)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    print(f"✔ {len(records)} rows → {args.output}")

    # Step 2: load existing relations (incremental mode)
    existing: list[dict[str, Any]] = []
    existing_titles: set[str] = set()

    if not args.full and args.relations_output.exists():
        existing = json.loads(args.relations_output.read_text(encoding="utf-8"))
        existing_titles = {r.get("title", "") for r in existing}
        print(f"  Found {len(existing)} already-extracted records")

    # Step 3: filter to new rows only
    new_records = [
        r for r in records
        if r.get("Title", f"row-?") not in existing_titles
    ]

    if args.limit:
        new_records = new_records[:args.limit]

    if not new_records:
        print("✔ Nothing new to extract — all rows already processed.")
        return

    print(f"Extracting relations from {len(new_records)} NEW records (skipping {len(existing_titles)} existing) …")
    new_relations = asyncio.run(extract(new_records))

    # Step 4: merge existing + new and save
    all_relations = existing + new_relations
    args.relations_output.write_text(json.dumps(all_relations, indent=2, default=str), encoding="utf-8")
    ok = sum(1 for r in new_relations if r["entities"])
    print(f"✔ {ok}/{len(new_relations)} new records extracted")
    print(f"✔ Total: {len(all_relations)} records → {args.relations_output}")


if __name__ == "__main__":
    main()
