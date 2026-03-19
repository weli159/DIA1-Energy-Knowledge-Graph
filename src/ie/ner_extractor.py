"""
src/ie/ner_extractor.py
Information Extraction — NER (PERSON, ORG, GPE, DATE) + Relation Extraction
via dependency parsing (nsubj → ROOT-verb → dobj).
Outputs: data/extracted_knowledge.csv, data/relation_candidates.csv
"""

import json
import re
from pathlib import Path

import pandas as pd
import spacy

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

IN_JSONL      = DATA / "crawler_output.jsonl"
OUT_ENTITIES  = DATA / "extracted_knowledge.csv"
OUT_RELATIONS = DATA / "relation_candidates.csv"

VALID_LABELS = {"PERSON", "ORG", "GPE", "DATE"}

STOP_ENTITIES = {
    "twh","gwh","mwh","kwh","gw","mw","kw","pv","csp","ac","dc",
    "co2","co₂","ghg","un","eu","us","uk","usd","eur",
}


def load_nlp():
    for model in ["en_core_web_trf", "en_core_web_lg", "en_core_web_md", "en_core_web_sm"]:
        try:
            m = spacy.load(model)
            print(f"Loaded: {model}")
            return m
        except OSError:
            continue
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"])
    return spacy.load("en_core_web_sm")


def is_valid_entity(text: str, label: str) -> bool:
    t = text.strip()
    if len(t) < 2:
        return False
    if t.lower() in STOP_ENTITIES:
        return False
    if label in ("PERSON","ORG","GPE") and re.fullmatch(r"[A-Z]{1,4}", t):
        return False
    return bool(re.search(r"[A-Za-z]", t))


def extract(pages: list[dict], nlp) -> tuple[pd.DataFrame, pd.DataFrame]:
    entity_rows   = []
    relation_rows = []

    for page in pages:
        url  = page["url"]
        text = page.get("text","")[:100_000]
        if not text:
            continue
        doc = nlp(text)

        # ── NER ──
        seen = set()
        for ent in doc.ents:
            if ent.label_ not in VALID_LABELS:
                continue
            v = ent.text.strip()
            if not is_valid_entity(v, ent.label_):
                continue
            key = (v.lower(), ent.label_, url)
            if key in seen:
                continue
            seen.add(key)
            entity_rows.append({"Entity_Name": v, "Type": ent.label_, "Source_URL": url})

        # ── Relation extraction via dependency parsing ──
        tok_to_ent = {}
        for ent in doc.ents:
            if ent.label_ in VALID_LABELS and is_valid_entity(ent.text.strip(), ent.label_):
                for tok in ent:
                    tok_to_ent[tok.i] = ent

        for sent in doc.sents:
            root = next((t for t in sent if t.dep_ == "ROOT" and t.pos_ == "VERB"), None)
            if root is None:
                continue
            subjects = [t for t in root.children if t.dep_ in ("nsubj","nsubjpass")]
            objects  = [t for t in root.children if t.dep_ in ("dobj","attr","pobj","obj")]

            for subj_tok in subjects:
                subj_ent = tok_to_ent.get(subj_tok.i)
                if subj_ent is None:
                    for c in subj_tok.subtree:
                        if c.i in tok_to_ent:
                            subj_ent = tok_to_ent[c.i]; break
                if subj_ent is None:
                    continue
                for obj_tok in objects:
                    obj_ent = tok_to_ent.get(obj_tok.i)
                    if obj_ent is None:
                        for c in obj_tok.subtree:
                            if c.i in tok_to_ent:
                                obj_ent = tok_to_ent[c.i]; break
                    if obj_ent is None or obj_ent == subj_ent:
                        continue
                    src = subj_ent.text.strip()
                    tgt = obj_ent.text.strip()
                    rel = root.lemma_.lower().strip()
                    if src.lower() == tgt.lower() or not rel:
                        continue
                    relation_rows.append({
                        "subject": src, "predicate": rel, "object": tgt,
                        "subject_type": subj_ent.label_, "object_type": obj_ent.label_,
                        "sentence": sent.text.strip()[:300], "source_url": url,
                    })

    df_ent = pd.DataFrame(entity_rows).drop_duplicates()
    df_rel = pd.DataFrame(relation_rows).drop_duplicates(
        subset=["subject","predicate","object","source_url"])

    df_ent.to_csv(OUT_ENTITIES, index=False)
    df_rel.to_csv(OUT_RELATIONS, index=False)

    print(f"Entities  : {len(df_ent)} → {OUT_ENTITIES}")
    print(f"Relations : {len(df_rel)} → {OUT_RELATIONS}")
    return df_ent, df_rel


if __name__ == "__main__":
    pages = [json.loads(l) for l in IN_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    nlp   = load_nlp()
    extract(pages, nlp)
