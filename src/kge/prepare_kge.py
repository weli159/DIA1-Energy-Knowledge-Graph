"""
src/kge/prepare_kge.py
Knowledge Graph Embedding — Data Preparation
Converts the RDF graph triples into the train/valid/test split format
required by pykeen / PyKEEN.
Outputs: data/train.txt, data/valid.txt, data/test.txt
"""

import random
from pathlib import Path

from rdflib import Graph, URIRef

ROOT = Path(__file__).resolve().parents[2]
KGA  = ROOT / "kg_artifacts"
DATA = ROOT / "data"

TRAIN_TXT = DATA / "train.txt"
VALID_TXT = DATA / "valid.txt"
TEST_TXT  = DATA / "test.txt"

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
# TEST_RATIO  = 0.10


def load_triples(ttl_path: Path) -> list[tuple[str, str, str]]:
    """Load (subject, predicate, object) as short URI strings."""
    g = Graph()
    g.parse(str(ttl_path), format="turtle")

    triples = []
    for s, p, o in g:
        if not isinstance(s, URIRef) or not isinstance(o, URIRef):
            continue
        triples.append((str(s), str(p), str(o)))
    return triples


def save_triples(triples: list[tuple], path: Path):
    with path.open("w", encoding="utf-8") as f:
        for s, p, o in triples:
            f.write(f"{s}\t{p}\t{o}\n")
    print(f"  Saved {len(triples):,} triples → {path}")


def prepare(ttl_path: Path = None, seed: int = 42) -> dict[str, list]:
    if ttl_path is None:
        for candidate in [KGA / "expanded.ttl", KGA / "ontology.ttl",
                          KGA / "alignment.ttl", KGA / "initial_kg.ttl"]:
            if candidate.exists():
                ttl_path = candidate
                break

    if ttl_path is None or not ttl_path.exists():
        raise FileNotFoundError("No KG TTL file found. Run build_kg.py and expand_kb.py first.")

    print(f"Loading triples from: {ttl_path}")
    triples = load_triples(ttl_path)
    print(f"Total URI triples: {len(triples):,}")

    # Remove duplicate triples and sort for reproducibility
    triples = list(set(triples))
    random.seed(seed)
    random.shuffle(triples)

    n       = len(triples)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)

    train = triples[:n_train]
    valid = triples[n_train:n_train + n_valid]
    test  = triples[n_train + n_valid:]

    save_triples(train, TRAIN_TXT)
    save_triples(valid, VALID_TXT)
    save_triples(test,  TEST_TXT)

    print(f"\nSplit summary:")
    print(f"  Train : {len(train):,}  ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Valid : {len(valid):,}  ({VALID_RATIO*100:.0f}%)")
    print(f"  Test  : {len(test):,}   ({(1-TRAIN_RATIO-VALID_RATIO)*100:.0f}%)")
    return {"train": train, "valid": valid, "test": test}


if __name__ == "__main__":
    prepare()
