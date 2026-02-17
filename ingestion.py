"""Ingest extracted entities & relations JSON into Neo4j."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


# ── Neo4j helpers ─────────────────────────────────────────────────────────

def _drop_conflicting_index(tx):
    tx.run("DROP INDEX name_entity_index IF EXISTS")


def _ensure_constraints(tx):
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")


def _merge_entity(tx, name: str, entity_type: str, source_title: str):
    tx.run(
        """
        MERGE (e:Entity {name: $name})
        SET e.type = $type, e.last_source = $source_title
        WITH e
        CALL apoc.create.addLabels(e, [$type]) YIELD node
        RETURN node
        """,
        name=name, type=entity_type, source_title=source_title,
    )


def _merge_entity_simple(tx, name: str, entity_type: str, source_title: str):
    """Fallback if APOC is not installed — uses a single Entity label."""
    tx.run(
        """
        MERGE (e:Entity {name: $name})
        SET e.type = $type, e.last_source = $source_title
        """,
        name=name, type=entity_type, source_title=source_title,
    )


def _merge_relation(tx, source: str, target: str, relation: str, fact: str):
    tx.run(
        """
        MATCH (a:Entity {name: $source})
        MATCH (b:Entity {name: $target})
        MERGE (a)-[r:RELATION {type: $relation}]->(b)
        SET r.fact = $fact
        """,
        source=source, target=target, relation=relation, fact=fact,
    )


def _merge_paper(tx, title: str, date: str, category: str, impact: str, enhancement: str, link: str):
    tx.run(
        """
        MERGE (p:Paper {title: $title})
        SET p.date = date($date),
            p.category = $category,
            p.impact = $impact,
            p.enhancement = $enhancement,
            p.link = $link
        """,
        title=title, date=date[:10], category=category,
        impact=impact, enhancement=enhancement, link=link,
    )


def _link_entity_to_paper(tx, entity_name: str, paper_title: str):
    tx.run(
        """
        MATCH (e:Entity {name: $entity_name})
        MATCH (p:Paper {title: $paper_title})
        MERGE (p)-[:MENTIONS]->(e)
        """,
        entity_name=entity_name, paper_title=paper_title,
    )


# ── Main ingestion ───────────────────────────────────────────────────────

def ingest(records: list[dict], driver, use_apoc: bool = False):
    merge_fn = _merge_entity if use_apoc else _merge_entity_simple
    total_e, total_r = 0, 0

    total_p = 0

    with driver.session() as session:
        session.execute_write(_drop_conflicting_index)
        session.execute_write(_ensure_constraints)
        # Paper constraint
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.title IS UNIQUE")

        for i, rec in enumerate(records):
            title = rec.get("title", f"row-{i}")
            entities = rec.get("entities", [])
            relations = rec.get("relations", [])
            src_rec = rec.get("source", {})

            if not entities and not relations:
                continue

            # Paper node
            ref_time = rec.get("reference_time", "2026-01-01T00:00:00")
            category = src_rec.get("Category", "") or ""
            impact = src_rec.get("Impact", "") or ""
            enhancement = src_rec.get("Enhancement", "") or ""
            link = src_rec.get("Link", "") or ""
            session.execute_write(_merge_paper, title, ref_time, category, impact, enhancement, link)
            total_p += 1

            # Entities
            for ent in entities:
                name = ent.get("name", "").strip()
                etype = ent.get("type", "Concept")
                if name:
                    session.execute_write(merge_fn, name, etype, title)
                    session.execute_write(_link_entity_to_paper, name, title)
                    total_e += 1

            # Relations
            for rel in relations:
                src = rel.get("source", "").strip()
                tgt = rel.get("target", "").strip()
                rtype = rel.get("relation", "RELATED_TO")
                fact = rel.get("fact", "")
                if src and tgt:
                    session.execute_write(_merge_relation, src, tgt, rtype, fact)
                    total_r += 1

            print(f"  [{i+1}] {title}  ({len(entities)} entities, {len(relations)} relations)")

    return total_p, total_e, total_r


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="relations JSON → Neo4j")
    p.add_argument("--input", type=Path, default=Path("data/AI_Tracker_relations.json"))
    p.add_argument("--neo4j-uri",  default=None)
    p.add_argument("--neo4j-user", default=None)
    p.add_argument("--neo4j-pass", default=None)
    p.add_argument("--apoc", action="store_true", help="use APOC to add dynamic labels")
    p.add_argument("--clear", action="store_true", help="wipe all nodes/edges before ingesting")
    args = p.parse_args()

    uri  = args.neo4j_uri  or os.getenv("NEO4J_URI",  "bolt://localhost:7687")
    user = args.neo4j_user or os.getenv("NEO4J_USER", "neo4j")
    pw   = args.neo4j_pass or os.getenv("NEO4J_PASSWORD", "")
    if not pw:
        raise SystemExit("Set NEO4J_PASSWORD in .env or pass --neo4j-pass")

    data = json.loads(args.input.read_text(encoding="utf-8"))
    print(f"Loaded {len(data)} records from {args.input}")

    driver = GraphDatabase.driver(uri, auth=(user, pw))
    driver.verify_connectivity()
    print(f"✔ Connected to {uri}")

    if args.clear:
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
            # Drop constraints first (they own backing indexes)
            for rec in s.run("SHOW CONSTRAINTS YIELD name RETURN name"):
                s.run(f"DROP CONSTRAINT {rec['name']} IF EXISTS")
            # Then drop remaining non-system indexes
            for rec in s.run("SHOW INDEXES YIELD name RETURN name"):
                idx = rec["name"]
                if not idx.startswith("index_"):
                    s.run(f"DROP INDEX {idx} IF EXISTS")
        print("  (cleared existing graph + indexes)")

    total_p, total_e, total_r = ingest(data, driver, use_apoc=args.apoc)
    driver.close()

    print(f"✔ Ingested {total_p} papers, {total_e} entities, {total_r} relations into Neo4j")


if __name__ == "__main__":
    main()
