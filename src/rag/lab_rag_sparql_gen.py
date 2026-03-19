"""
src/rag/lab_rag_sparql_gen.py
─────────────────────────────────────────────────────────────
TD6 — RAG with RDF/SPARQL and a Local Small LLM
─────────────────────────────────────────────────────────────
INPUT  : kg_artifacts/alignment.ttl  (or expanded.nt from TD4)
OUTPUT : CLI demo + evaluation table

Implements exactly the structure from the professor's lab PDF:
  ask_local_llm, load_graph, build_schema_summary,
  generate_sparql, run_sparql, repair_sparql,
  answer_with_sparql_generation, answer_no_rag,
  pretty_print_result, CLI main loop.

Usage:
  python -m src.rag.lab_rag_sparql_gen
"""

import re
from pathlib import Path
from typing import List, Tuple

import requests
from rdflib import Graph

# ----------------------------
# Configuration
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]

# Load from best available KB artifact
def _find_ttl() -> str:
    for candidate in [
        ROOT / "kg_artifacts" / "expanded.nt",
        ROOT / "kg_artifacts" / "ontology.ttl",
        ROOT / "kg_artifacts" / "alignment.ttl",
    ]:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("No KG file found. Run TD4 pipeline first.")

TTL_FILE       = _find_ttl()
OLLAMA_URL     = "http://localhost:11434/api/generate"
GEMMA_MODEL    = "gemma:2b"   # If 'model not found', try "gemma2:2b"
MAX_PREDICATES = 15
MAX_CLASSES    = 10
SAMPLE_TRIPLES = 0


# ----------------------------
# 0) Utility: call local LLM (Ollama)
# ----------------------------
def ask_local_llm(prompt: str, model: str = GEMMA_MODEL) -> str:
    """
    Send a prompt to a local Ollama model using the REST API.
    Returns the full text response as a single string.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,   # important: disable streaming for simpler integration
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error {response.status_code}: {response.text}")
        data = response.json()
        return data.get("response", "")
    except requests.exceptions.ConnectionError:
        return "[ERROR] Cannot connect to Ollama. Run: ollama serve"
    except Exception as e:
        return f"[ERROR] {e}"


# ----------------------------
# 1) Load RDF graph
# ----------------------------
def load_graph(ttl_path: str) -> Graph:
    g = Graph()
    fmt = "nt" if ttl_path.endswith(".nt") else "turtle"
    g.parse(ttl_path, format=fmt)
    print(f"Loaded {len(g)} triples from {ttl_path}")
    return g


# ----------------------------
# 2) Build a small schema summary
# ----------------------------
def get_prefix_block(g: Graph) -> str:
    """Collect prefixes registered in the graph's namespace manager.
    Ensure common prefixes exist to help the LLM."""
    defaults = {
        "rdf":  "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "xsd":  "http://www.w3.org/2001/XMLSchema#",
        "owl":  "http://www.w3.org/2002/07/owl#",
        "wd":   "http://www.wikidata.org/entity/",
        "wdt":  "http://www.wikidata.org/prop/direct/",
    }
    ns_map = {p: str(ns) for p, ns in g.namespace_manager.namespaces()}
    for k, v in defaults.items():
        ns_map.setdefault(k, v)
    lines = [f"PREFIX {p}: <{ns}>" for p, ns in ns_map.items()]
    return "\n".join(sorted(lines))


def list_distinct_predicates(g: Graph, limit=MAX_PREDICATES) -> List[str]:
    q = f"""
    SELECT DISTINCT ?p WHERE {{
        ?s ?p ?o .
    }} LIMIT {limit}
    """
    return [str(row.p) for row in g.query(q)]


def list_distinct_classes(g: Graph, limit=MAX_CLASSES) -> List[str]:
    q = f"""
    SELECT DISTINCT ?cls WHERE {{
        ?s a ?cls .
    }} LIMIT {limit}
    """
    return [str(row.cls) for row in g.query(q)]


def sample_triples(g: Graph, limit=SAMPLE_TRIPLES) -> List[Tuple[str, str, str]]:
    q = f"""
    SELECT ?s ?p ?o WHERE {{
        ?s ?p ?o .
    }} LIMIT {limit}
    """
    return [(str(r.s), str(r.p), str(r.o)) for r in g.query(q)]


def build_schema_summary(g: Graph) -> str:
    prefixes = get_prefix_block(g)
    preds    = list_distinct_predicates(g)
    clss     = list_distinct_classes(g)
    samples  = sample_triples(g)

    pred_lines   = "\n".join(f"- {p}" for p in preds)
    cls_lines    = "\n".join(f"- {c}" for c in clss)
    sample_lines = "\n".join(f"- {s} {p} {o}" for s, p, o in samples)

    summary = f"""
{prefixes}

# Predicates (sampled, unique up to {MAX_PREDICATES})
{pred_lines}

# Classes / rdf:type (sampled, unique up to {MAX_CLASSES})
{cls_lines}

# Sample triples (up to {SAMPLE_TRIPLES})
{sample_lines}
"""
    return summary.strip()


# ----------------------------
# 3) Prompting Gemma: NL → SPARQL
# ----------------------------
SPARQL_INSTRUCTIONS = """
Task: Convert the user's QUESTION into a SPARQL 1.1 SELECT query.
Schema: Entities use <http://private.org/energy/entity/> and <http://private.org/energy/type/> namespaces.

STRICT SCHEMA RULES:
1. For organizations, use: <http://private.org/energy/type/ORG>
2. For persons, use: <http://private.org/energy/type/PERSON>
3. Use rdfs:label for all text matching.
4. DO NOT use SERVICE wikibase:label (local graph only).
5. DO NOT use wdt:P21 or wd: namespaces unless specifically seen in the schema.

Rules:
1. Use ONLY the prefixes provided.
2. Return ONLY the SELECT query in a ```sparql code block.
3. If you do not know the answer, return an empty code block.

Example 1:
Question: "List solar energy entities."
Answer:
```sparql
SELECT ?entity WHERE {
  ?entity rdfs:label ?label .
  FILTER(CONTAINS(LCASE(?label), "solar"))
}
```

Example 2:
Question: "What is the location of EDF?"
Answer:
```sparql
SELECT ?loc WHERE {
  ?edf rdfs:label "EDF" .
  ?edf schema:location ?loc .
}
```

Example 3:
Question: "How many organizations are there?"
Answer:
```sparql
SELECT (COUNT(?s) AS ?count) WHERE {
  ?s a <http://private.org/energy/type/ORG> .
}
```

IMPORTANT RULE: For any "search" or "related to" questions, ALWAYS use FILTER(CONTAINS(LCASE(?label), "keyword")) on rdfs:label.
"""


def make_sparql_prompt(schema_summary: str, question: str) -> str:
    return f"""{SPARQL_INSTRUCTIONS}

SCHEMA SUMMARY:
{schema_summary}

QUESTION: "{question}"
Final Answer (SPARQL code block only):
"""


CODE_BLOCK_RE = re.compile(r"```(?:sparql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sparql_from_text(text: str) -> str:
    m = CODE_BLOCK_RE.search(text)
    if m: return m.group(1).strip()
    m_fallback = re.search(r"(SELECT\s.*?\})", text, re.IGNORECASE | re.DOTALL)
    if m_fallback: return m_fallback.group(1).strip()
    return ""

def generate_sparql(question: str, schema_summary: str) -> str:
    raw = ask_local_llm(make_sparql_prompt(schema_summary, question))
    return extract_sparql_from_text(raw)

def run_sparql(g: Graph, query: str) -> Tuple[List[str], List[Tuple]]:
    res   = g.query(query)
    vars_ = [str(v) for v in res.vars]
    rows  = [tuple(str(cell) for cell in r) for r in res]
    return vars_, rows

REPAIR_INSTRUCTIONS = "The previous SPARQL failed to execute. Using the SCHEMA SUMMARY and the ERROR MESSAGE, return a corrected SPARQL 1.1 SELECT query. Follow strictly: Use only known prefixes/IRIs. Keep it simple. Return only a single code block with the corrected SPARQL."

def repair_sparql(schema_summary: str, question: str, bad_query: str, error_msg: str) -> str:
    prompt = f"{REPAIR_INSTRUCTIONS}\nSCHEMA SUMMARY:\n{schema_summary}\nQUESTION:\n{question}\nBAD SPARQL:\n{bad_query}\nERROR MESSAGE:\n{error_msg}\nReturn only the corrected SPARQL in a code block."
    raw = ask_local_llm(prompt)
    return extract_sparql_from_text(raw)

def synthesize_answer(question: str, vars_: list, rows: list) -> str:
    if not rows: return "I found no direct matches for that in the Knowledge Graph."
    data_str = ""
    for r in rows[:5]: data_str += ", ".join(str(x) for x in r) + "\n"
    prompt = f"Task: You are a helpful assistant. Synthesize a natural language answer based on the given Data.\nUser Question: \"{question}\"\nSPARQL Results Data:\n{data_str}\nInstruction: Answer the question naturally using the data provided. Be concise."
    return ask_local_llm(prompt)

def answer_with_sparql_generation(g: Graph, schema_summary: str, question: str, try_repair: bool = True) -> dict:
    prefixes = get_prefix_block(g)
    raw_sparql = generate_sparql(question, schema_summary)
    if not raw_sparql:
        return {"query": "", "vars": [], "rows": [], "answer": "I couldn't generate a query.", "repaired": False, "error": "LLM failed to generate a SPARQL query."}
    sparql = raw_sparql
    if "PREFIX" not in sparql.upper(): sparql = f"{prefixes}\n\n{raw_sparql}"
    try:
        vars_, rows = run_sparql(g, sparql)
        str_rows = [[str(cell) for cell in row] for row in rows]
        ans = synthesize_answer(question, vars_, str_rows)
        return {"query": sparql, "vars": vars_, "rows": str_rows, "answer": ans, "repaired": False, "error": None}
    except Exception as e:
        err = str(e)
        if try_repair:
            repaired = repair_sparql(schema_summary, question, raw_sparql, err)
            if "PREFIX" not in repaired.upper(): repaired = f"{prefixes}\n\n{repaired}"
            try:
                vars_, rows = run_sparql(g, repaired)
                str_rows = [[str(cell) for cell in row] for row in rows]
                ans = synthesize_answer(question, vars_, str_rows)
                return {"query": repaired, "vars": vars_, "rows": str_rows, "answer": ans, "repaired": True, "error": None}
            except Exception as e2:
                return {"query": repaired, "vars": [], "rows": [], "answer": "Error after repair.", "repaired": True, "error": str(e2)}
        else:
            return {"query": sparql, "vars": [], "rows": [], "answer": "Error executing query.", "repaired": False, "error": err}

def answer_no_rag(question: str) -> str:
    prompt = f"Answer the following question as best as you can:\n\n{question}"
    return ask_local_llm(prompt)

def pretty_print_result(result: dict):
    if result.get("error"):
        print("\n[Execution Error]", result["error"])
    query = result.get("query", "")
    clean_lines = [line for line in query.split("\n") if not line.strip().startswith("PREFIX")]
    clean_query = "\n".join(clean_lines).strip()

    print("\n[SPARQL Query (Logic Only)]")
    print(clean_query)
    if result.get("repaired"):
        print("[Repaired?]", result["repaired"])
    vars_ = result.get("vars", [])
    rows  = result.get("rows", [])
    
    if not rows:
        print("\n[No rows returned]")
        return
        
    print("\n[Results]")
    print(" | ".join(vars_))
    for r in rows[:20]:
        print(" | ".join(r))
    if len(rows) > 20:
        print(f"... (showing 20 of {len(rows)})")
        
    print("\n[Synthesized Answer]")
    print(result.get("answer", ""))

if __name__ == "__main__":
    g      = load_graph(TTL_FILE)
    schema = build_schema_summary(g)

    while True:
        q = input("\nQuestion (or 'quit'): ").strip()
        if q.lower() == "quit":
            break
        print("\n--- Baseline (No RAG) ---")
        print(answer_no_rag(q))
        print("\n--- SPARQL-generation RAG (Gemma 2B + rdflib) ---")
        result = answer_with_sparql_generation(g, schema, q, try_repair=True)
        pretty_print_result(result)