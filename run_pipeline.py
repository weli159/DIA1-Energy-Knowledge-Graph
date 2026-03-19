"""
run_pipeline.py — Full project pipeline runner
Usage:  python run_pipeline.py [--step STEP]

Steps:
  crawl   → src/crawl/crawler.py           → data/crawler_output.jsonl
  ie      → src/ie/ner_extractor.py        → data/extracted_knowledge.csv
  kg      → src/kg/build_kg.py             → kg_artifacts/alignment.ttl
  expand  → src/kg/expand_kb.py            → kg_artifacts/expanded.nt
  reason  → src/reason/swrl_reasoning.py   → kg_artifacts/ontology.ttl
  kge     → src/kge/prepare_kge.py +       → data/train.txt, valid.txt, test.txt
             src/kge/train_kge.py           → data/kge_results.csv
  rag     → src/rag/lab_rag_sparql_gen.py  → interactive CLI demo
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def step_crawl():
    from src.crawl.crawler import crawl
    crawl()


def step_ie():
    from src.ie.ner_extractor import extract, load_nlp
    DATA = ROOT / "data"
    IN   = DATA / "crawler_output.jsonl"
    if not IN.exists():
        print(f"ERROR: {IN} not found. Run --step crawl first.")
        return
    pages = [json.loads(l) for l in IN.read_text(encoding="utf-8").splitlines() if l.strip()]
    extract(pages, load_nlp())


def step_kg():
    from src.kg.build_kg import main as build_main
    build_main()


def step_expand():
    from src.kg.expand_kb import main as expand_main
    expand_main()


def step_reason():
    from src.reason.swrl_reasoning import run_family_swrl, run_energy_swrl
    run_family_swrl()
    run_energy_swrl()


def step_kge():
    from src.kge.prepare_kge import prepare
    from src.kge.train_kge import main as train_main
    prepare()
    train_main()


def step_rag():
    # Import and run the CLI demo from lab_rag_sparql_gen.py
    from src.rag.lab_rag_sparql_gen import (
        load_graph, build_schema_summary, answer_no_rag,
        answer_with_sparql_generation, pretty_print_result, TTL_FILE
    )
    g      = load_graph(TTL_FILE)
    schema = build_schema_summary(g)
    print(f"Loaded {len(g):,} triples | Model: gemma:2b")
    print("Type 'quit' to exit.\n")
    while True:
        try:
            q = input("Question (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        print("\n--- Baseline (No RAG) ---")
        print(answer_no_rag(q))
        print("\n--- SPARQL-generation RAG (Gemma 2B + rdflib) ---")
        pretty_print_result(answer_with_sparql_generation(g, schema, q, try_repair=True))
        print()


STEPS = {
    "crawl":  step_crawl,
    "ie":     step_ie,
    "kg":     step_kg,
    "expand": step_expand,
    "reason": step_reason,
    "kge":    step_kge,
    "rag":    step_rag,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Energy KG Project Pipeline")
    parser.add_argument(
        "--step", default="all",
        choices=list(STEPS.keys()) + ["all"],
        help="Which step to run  (default: all, in order)"
    )
    args = parser.parse_args()

    if args.step == "all":
        for name, fn in STEPS.items():
            print(f"\n{'#'*60}\n# STEP: {name.upper()}\n{'#'*60}")
            fn()
    else:
        STEPS[args.step]()

    print("\n✅ Done.")
