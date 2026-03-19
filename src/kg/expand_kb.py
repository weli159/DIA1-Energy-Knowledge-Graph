"""
src/kg/expand_kb.py
KB Expansion — BFS over Wikidata SPARQL to grow the private RDF graph.
Outputs: kg_artifacts/expanded.ttl, kg_artifacts/expanded.nt,
         data/kb_stats.csv
"""

import time
from collections import deque
from pathlib import Path

import pandas as pd
from rdflib import Graph, URIRef
from SPARQLWrapper import SPARQLWrapper, JSON

ROOT = Path(__file__).resolve().parents[2]
KGA  = ROOT / "kg_artifacts"
DATA = ROOT / "data"

ALIGNMENT_TTL  = KGA / "alignment.ttl"
MAPPING_CSV    = DATA / "entity_mapping.csv"
EXPANDED_TTL   = KGA / "expanded.ttl"
EXPANDED_NT    = KGA / "expanded.nt"
KB_STATS_CSV   = DATA / "kb_stats.csv"

MIN_CONF        = 0.75
TARGET_TRIPLES  = 55_000
PER_ENTITY_LIMIT= 50
MAX_REQUESTS    = 4_000


def fetch_one_hop(sparql_ep, wd_uri: str, limit: int = PER_ENTITY_LIMIT) -> list[tuple]:
    q = f"""
    SELECT ?p ?o WHERE {{
      <{wd_uri}> ?p ?o .
      FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
      FILTER(STRSTARTS(STR(?o), "http://www.wikidata.org/entity/Q"))
    }} LIMIT {limit}
    """
    sparql_ep.setQuery(q)
    res = sparql_ep.query().convert()
    return [(b["p"]["value"], b["o"]["value"])
            for b in res.get("results",{}).get("bindings",[])
            if "p" in b and "o" in b]


def expand(g: Graph, mapping_csv: Path = MAPPING_CSV) -> Graph:
    sparql_ep = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql_ep.setReturnFormat(JSON)
    sparql_ep.setTimeout(30)
    sparql_ep.addCustomHttpHeader("User-Agent", "EnergyKG-Expand/1.0")

    df_map = pd.read_csv(mapping_csv)
    eligible = df_map[
        df_map["external_uri"].astype(str).str.startswith("http") &
        (df_map["confidence_score"] >= MIN_CONF)
    ]
    seeds    = eligible["external_uri"].astype(str).dropna().tolist()
    frontier = deque(seeds)
    visited  = set()
    req_cnt  = 0

    print(f"Eligible seeds : {len(seeds)}")
    print(f"Starting from  : {len(g):,} triples")

    while len(g) < TARGET_TRIPLES and frontier and req_cnt < MAX_REQUESTS:
        uri = frontier.popleft()
        if uri in visited:
            continue
        visited.add(uri)

        ok = False
        for attempt in range(1, 4):
            try:
                neighbors = fetch_one_hop(sparql_ep, uri)
                ok = True
                break
            except Exception:
                time.sleep(2 * attempt)
        req_cnt += 1

        if ok:
            for p_str, o_str in neighbors:
                g.add((URIRef(uri), URIRef(p_str), URIRef(o_str)))
                if o_str not in visited:
                    frontier.append(o_str)

        if req_cnt % 20 == 0:
            print(f"  req={req_cnt} | triples={len(g):,} | frontier={len(frontier)}")
        time.sleep(1.0)

    print(f"\nExpansion done. Triples={len(g):,}, SPARQL requests={req_cnt}")
    return g


def main():
    print("Loading alignment graph…")
    g = Graph()
    g.parse(str(ALIGNMENT_TTL), format="turtle")

    g = expand(g)

    import warnings; warnings.filterwarnings("ignore")
    g.serialize(destination=str(EXPANDED_TTL), format="turtle")
    g.serialize(destination=str(EXPANDED_NT),  format="nt", encoding="utf-8")

    entities  = {str(s) for s, _, _ in g if hasattr(s, "toPython")} | {str(o) for _,_,o in g if str(o).startswith("http")}
    relations = {str(p) for _, p, _ in g}

    stats = pd.DataFrame([{"total_triplets": len(g), "total_entities": len(entities), "total_relations": len(relations)}])
    stats.to_csv(KB_STATS_CSV, index=False)

    print(f"Expanded KB  → {EXPANDED_TTL}")
    print(f"Expanded NT  → {EXPANDED_NT}")
    print(f"Stats        → {KB_STATS_CSV}")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
