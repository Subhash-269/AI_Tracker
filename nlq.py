"""Natural-language query chat over the AI Tracker knowledge graph."""

from __future__ import annotations

import json
import os
import textwrap

from dotenv import load_dotenv
from neo4j import GraphDatabase

from llm_providers import chat, list_providers

load_dotenv()

SYSTEM_PROMPT = """\
You are a Cypher query generator for a Neo4j knowledge graph about AI research.
Today's date is 2026-02-16. Output ONLY a valid Cypher query. No markdown, no explanation, no backticks.

SCHEMA:
  (:Paper {title, date, category, impact, enhancement, link})  — 72 nodes, dates 2025-08-07 to 2026-02-16
  (:Entity {name, type, last_source})  — types: AI_Model, Technology, Concept, Industry, Technique, Tool, Organization, Metric
  (:Paper)-[:MENTIONS]->(:Entity)  — 701 edges
  (:Entity)-[:RELATION {type, fact}]->(:Entity)  — 640 edges, types: APPLICABLE_IN, UTILIZES, ENHANCES, ENABLES, etc.

CRITICAL RULES:
1. Questions about "latest", "recent", "news", "new", "updates", "advancements"
   MUST use this exact pattern:
   MATCH (p:Paper) RETURN p.title, p.date, p.category, p.impact ORDER BY p.date DESC LIMIT 10
2. NEVER answer "latest news" by querying Entity or Organization nodes.
3. Always include p.date when returning Paper results.
4. Use toLower() + CONTAINS for fuzzy matching.
5. Limit to 25 rows unless user asks for more.
6. Read-only: no CREATE/DELETE/SET/MERGE/REMOVE.
7. If unanswerable: // CANNOT_ANSWER

EXAMPLES:
User: latest news
MATCH (p:Paper) RETURN p.title, p.date, p.category, p.impact ORDER BY p.date DESC LIMIT 10

User: latest news with dates
MATCH (p:Paper) RETURN p.title, p.date, p.category, p.impact ORDER BY p.date DESC LIMIT 10

User: what's new in reinforcement learning
MATCH (p:Paper) WHERE toLower(p.category) CONTAINS 'reinforcement' RETURN p.title, p.date, p.category, p.impact ORDER BY p.date DESC LIMIT 10

User: papers about GPT
MATCH (p:Paper)-[:MENTIONS]->(e:Entity) WHERE toLower(e.name) CONTAINS 'gpt' RETURN DISTINCT p.title, p.date, p.category, p.impact ORDER BY p.date DESC

User: what entities does the latest paper mention
MATCH (p:Paper) WITH p ORDER BY p.date DESC LIMIT 1 MATCH (p)-[:MENTIONS]->(e:Entity) RETURN p.title, p.date, e.name, e.type

User: list all AI models
MATCH (e:Entity) WHERE e.type = 'AI_Model' RETURN e.name ORDER BY e.name

User: how are knowledge graphs and RAG related
MATCH (a:Entity)-[r:RELATION]->(b:Entity) WHERE toLower(a.name) CONTAINS 'knowledge graph' AND toLower(b.name) CONTAINS 'rag' RETURN a.name, r.type, r.fact, b.name UNION MATCH (a:Entity)-[r:RELATION]->(b:Entity) WHERE toLower(a.name) CONTAINS 'rag' AND toLower(b.name) CONTAINS 'knowledge graph' RETURN a.name, r.type, r.fact, b.name"""

ANSWER_PROMPT = """\
You are a helpful AI research assistant. Given the user's question and knowledge-graph query results,
provide a clear, well-structured answer. Use bullet points, group by category or date where helpful.
Include paper titles, dates, and impact summaries when available.
If the results are empty, say so and suggest rephrasing."""


def build_driver():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pw = os.getenv("NEO4J_PASSWORD", "")
    driver = GraphDatabase.driver(uri, auth=(user, pw))
    driver.verify_connectivity()
    return driver


def nl_to_cypher(question: str, history: list[dict]) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-4:])  # last 2 turns for context
    messages.append({"role": "user", "content": question})

    return chat(messages, temperature=0)


def run_cypher(driver, query: str) -> list[dict]:
    with driver.session() as session:
        result = session.run(query)
        return [r.data() for r in result]


def answer_question(question: str, rows: list[dict]) -> str:
    content = f"Question: {question}\n\nQuery results ({len(rows)} rows):\n{json.dumps(rows[:50], indent=2, default=str)}"
    return chat(
        [
            {"role": "system", "content": ANSWER_PROMPT},
            {"role": "user", "content": content},
        ],
        temperature=0.3,
    )


def main():
    driver = build_driver()

    print("=" * 60)
    print("  AI Tracker — Natural Language Query")
    print("  Type your question or 'quit' to exit.")
    print("  Generated Cypher is shown for each query.")
    print("  LLM providers (fallback order):")
    print(list_providers())
    print("=" * 60)

    history: list[dict] = []

    while True:
        try:
            q = input("\n❓ ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q or q.lower() in ("quit", "exit", "q"):
            break

        # Step 1: NL → Cypher
        cypher = nl_to_cypher(q, history)

        if "CANNOT_ANSWER" in cypher:
            print("  ⚠ This question can't be answered from the graph.")
            continue

        print(f"  📝 Cypher: {cypher}")

        # Step 2: Execute
        try:
            rows = run_cypher(driver, cypher)
        except Exception as e:
            print(f"  ⚠ Query error: {e}")
            print(f"  Query was: {cypher}")
            continue

        # Step 3: Summarize
        answer = answer_question(q, rows)
        print(f"\n{answer}")

        # Track history
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": cypher})

    driver.close()
    print("\nBye! 👋")


if __name__ == "__main__":
    main()
