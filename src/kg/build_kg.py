"""
src/kg/build_kg.py
─────────────────────────────────────────────────────────────
Pipeline position:  TD4  —  takes TD1 outputs, produces alignment.ttl
─────────────────────────────────────────────────────────────
INPUT  (from TD1 / src/ie/ner_extractor.py):
  data/extracted_knowledge.csv   — named entities (Entity_Name, Type, Source_URL)
  data/relation_candidates.csv   — S-P-O triples (subject, predicate, object, …)

OUTPUT (required kg_artifacts):
  kg_artifacts/alignment.ttl     — private RDF graph + owl:sameAs + owl:equivalentProperty
  data/entity_mapping.csv        — QID lookup table
  data/predicate_alignment.csv   — verb → Wikidata property

Next step: src/kg/expand_kb.py  → kg_artifacts/expanded.nt
"""

import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
from rdflib import Graph, Literal, Namespace, OWL, RDF, RDFS, URIRef

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
KGA  = ROOT / "kg_artifacts"
KGA.mkdir(parents=True, exist_ok=True)

# ── Input: from TD1 ──
ENTITIES_CSV  = DATA / "extracted_knowledge.csv"
RELATIONS_CSV = DATA / "relation_candidates.csv"

# ── Output: required kg_artifacts ──
ALIGNMENT_TTL      = KGA / "alignment.ttl"
MAPPING_CSV        = DATA / "entity_mapping.csv"
PRED_ALIGNMENT_CSV = DATA / "predicate_alignment.csv"

MYKB = Namespace("http://private.org/energy/")
WDT  = Namespace("http://www.wikidata.org/prop/direct/")
WD   = Namespace("http://www.wikidata.org/entity/")

PREDICATE_MAP = {
    "produce":"P1056","generate":"P1056","use":"P2283","locate":"P276",
    "include":"P527","develop":"P178","create":"P170","build":"P178",
    "join":"P463","publish":"P123","contribute":"P3712","convene":"P664",
    "seek":"P101","increase":"P3362","select":"P361","establish":"P571",
    "fund":"P8324","invest":"P8324","supply":"P1056","install":"P1056",
    "reduce":"P3362","comprise":"P527","consist":"P527","employ":"P108",
}


def slugify(t: str) -> str:
    t = re.sub(r"[^A-Za-z0-9]+", "_", str(t).strip())
    return re.sub(r"_+", "_", t).strip("_") or "entity"

def entity_uri(label: str) -> URIRef:
    return MYKB[f"entity/{slugify(label)}"]

def prop_uri(rel: str) -> URIRef:
    return MYKB[f"prop/{slugify(rel.lower())}"]


def build_graph(df_ent: pd.DataFrame, df_rel: pd.DataFrame) -> Graph:
    """Build the private RDF graph from TD1 entity/relation tables."""
    g = Graph()
    g.bind("mykb", MYKB); g.bind("owl", OWL); g.bind("rdfs", RDFS)
    g.bind("rdf", RDF); g.bind("wdt", WDT); g.bind("wd", WD)

    ecol = "Entity_Name" if "Entity_Name" in df_ent.columns else "Entity"
    tcol = "Type"        if "Type"        in df_ent.columns else "label"
    for _, row in df_ent.iterrows():
        uri = entity_uri(str(row[ecol]))
        g.add((uri, RDFS.label, Literal(str(row[ecol]))))
        g.add((uri, RDF.type,   MYKB[f"type/{str(row[tcol]).upper()}"]))

    col_s = "subject"   if "subject"   in df_rel.columns else "source"
    col_p = "predicate" if "predicate" in df_rel.columns else "relation"
    col_t = "object"    if "object"    in df_rel.columns else "target"
    for _, row in df_rel.iterrows():
        src = str(row.get(col_s, "")).strip()
        rel = str(row.get(col_p, "")).strip()
        tgt = str(row.get(col_t, "")).strip()
        if not (src and rel and tgt):
            continue
        g.add((entity_uri(src), prop_uri(rel),    entity_uri(tgt)))
        g.add((entity_uri(src), RDFS.label,       Literal(src)))
        g.add((entity_uri(tgt), RDFS.label,       Literal(tgt)))
        g.add((prop_uri(rel),   RDFS.label,       Literal(rel)))

    print(f"Private KG triples: {len(g):,}")
    return g


# ── Wikidata entity linking (owl:sameAs) ────────────────────────────────────
_session = requests.Session()
_session.headers.update({"User-Agent": "EnergyKG-Builder/1.0 (academic)"})

def _wikidata_search(label: str) -> list:
    r = _session.get("https://www.wikidata.org/w/api.php", params={
        "action":"wbsearchentities","format":"json","language":"en",
        "type":"item","search":label,"limit":"3",
    }, timeout=20)
    return r.json().get("search", [])

def _confidence(priv: str, wd: str) -> float:
    a, b = priv.lower().strip(), wd.lower().strip()
    if a == b:           return 0.97
    if a in b or b in a: return 0.90
    return 0.75

def _is_valid(label: str) -> bool:
    v = str(label).strip()
    return (2 <= len(v) <= 80 and bool(re.search(r"[A-Za-z]", v))
            and sum(1 for c in v if not c.isalnum() and not c.isspace()) / max(1, len(v)) <= 0.25)


def align_entities(g: Graph, df_ent: pd.DataFrame, df_rel: pd.DataFrame,
                   max_labels: int = 500) -> pd.DataFrame:
    """Link entity labels to Wikidata QIDs; add owl:sameAs triples to the graph."""
    ecol  = "Entity_Name" if "Entity_Name" in df_ent.columns else "Entity"
    col_s = "subject"     if "subject"     in df_rel.columns else "source"
    col_t = "object"      if "object"      in df_rel.columns else "target"

    all_labels = ([str(x).strip() for x in df_ent[ecol]]
                + [str(x).strip() for x in df_rel[col_s]]
                + [str(x).strip() for x in df_rel[col_t]])
    all_labels = [l for l in all_labels if _is_valid(l)]

    counts    = Counter(l.lower() for l in all_labels)
    canonical = {}
    for l in all_labels:
        canonical.setdefault(l.lower(), l)

    labels = [canonical[k] for k, _ in
              sorted(counts.items(), key=lambda x: x[1], reverse=True)[:max_labels]]
    print(f"Linking {len(labels)} entity labels to Wikidata (owl:sameAs)…")

    rows = []
    for idx, label in enumerate(labels, 1):
        priv_uri = entity_uri(label)
        ext_uri, conf = "", 0.0
        for attempt in range(1, 4):
            try:
                results = _wikidata_search(label)
                if results:
                    best = results[0]
                    qid  = str(best.get("id", "")).strip()
                    if qid:
                        ext_uri = f"http://www.wikidata.org/entity/{qid}"
                        conf    = _confidence(label, str(best.get("label", "")))
                        g.add((priv_uri, OWL.sameAs, URIRef(ext_uri)))
                break
            except Exception:
                time.sleep(1.5 * attempt)
        rows.append({"private_label": label, "external_uri": ext_uri, "confidence_score": conf})
        if idx % 50 == 0:
            print(f"  {idx}/{len(labels)} linked…")
        time.sleep(0.3)

    df_map = pd.DataFrame(rows)
    df_map.to_csv(MAPPING_CSV, index=False)
    print(f"Entity alignment: {int((df_map['external_uri'] != '').sum())}/{len(labels)} linked")
    return df_map


def align_predicates(g: Graph, df_rel: pd.DataFrame) -> pd.DataFrame:
    """Map relation verbs → Wikidata properties; add owl:equivalentProperty triples."""
    col_p = "predicate" if "predicate" in df_rel.columns else "relation"
    freq  = df_rel[col_p].astype(str).str.strip().str.lower().value_counts().head(30)

    rows = []
    for rel, cnt in freq.items():
        qprop = PREDICATE_MAP.get(rel)
        if not qprop:
            continue
        g.add((prop_uri(rel), OWL.equivalentProperty, WDT[qprop]))
        rows.append({"private_relation": rel, "wikidata_property": f"wdt:{qprop}",
                     "frequency": int(cnt)})

    df = pd.DataFrame(rows)
    df.to_csv(PRED_ALIGNMENT_CSV, index=False)
    print(f"Predicate alignment: {len(df)} verbs mapped to Wikidata properties")
    return df


def main():
    print("=" * 60)
    print("TD4 — KG Construction & Alignment")
    print(f"  ← {ENTITIES_CSV}")
    print(f"  ← {RELATIONS_CSV}")
    print("=" * 60)

    df_ent = pd.read_csv(ENTITIES_CSV)
    df_rel = pd.read_csv(RELATIONS_CSV)
    print(f"  {len(df_ent)} entities  |  {len(df_rel)} relation triples\n")

    g = build_graph(df_ent, df_rel)
    align_entities(g, df_ent, df_rel)
    align_predicates(g, df_rel)

    g.serialize(destination=str(ALIGNMENT_TTL), format="turtle")
    print(f"\n✅ → {ALIGNMENT_TTL}  ({len(g):,} triples)")
    print(f"✅ → {MAPPING_CSV}")
    print(f"✅ → {PRED_ALIGNMENT_CSV}")
    print(f"\nNext: python -m src.kg.expand_kb  →  kg_artifacts/expanded.nt")


if __name__ == "__main__":
    main()
