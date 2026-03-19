"""
src/kge/train_kge.py
─────────────────────────────────────────────────────────────
TD5 Part 2 — Knowledge Graph Embedding (KGE)
─────────────────────────────────────────────────────────────
INPUT:  data/train.txt, data/valid.txt, data/test.txt  (from prepare_kge.py)
OUTPUT: data/kge_results.csv, data/kge_tsne.png, data/kge_neighbors.txt

Implements all required experiments from the lab PDF:
  1. Two models: TransE + RotatE (via PyKEEN)
  2. Link prediction metrics: MRR, Hits@1/3/10 (filtered)
  3. KB size sensitivity: 20k / 50k / full dataset
  4. Nearest neighbor analysis
  5. t-SNE clustering by entity type
  6. Relation behavior discussion (symmetric, inverse, composition)
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

TRAIN_TXT    = DATA / "train.txt"
VALID_TXT    = DATA / "valid.txt"
TEST_TXT     = DATA / "test.txt"
RESULTS_CSV  = DATA / "kge_results.csv"
TSNE_PNG     = DATA / "kge_tsne.png"
NEIGHBORS_TXT= DATA / "kge_neighbors.txt"

EMBEDDING_DIM = 100          # use 100 or 200 as per lab spec
NUM_EPOCHS    = 100
BATCH_SIZE    = 256
LEARNING_RATE = 0.01


def ensure_pykeen():
    try:
        import pykeen
    except ImportError:
        print("Installing pykeen (this may take a few minutes)…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pykeen", "-q"])


def load_dataset():
    from pykeen.triples import TriplesFactory
    if not TRAIN_TXT.exists():
        raise FileNotFoundError(f"{TRAIN_TXT} not found. Run prepare_kge.py first.")
    tf_train = TriplesFactory.from_path(str(TRAIN_TXT))
    tf_valid = TriplesFactory.from_path(str(VALID_TXT),
                                        entity_to_id=tf_train.entity_to_id,
                                        relation_to_id=tf_train.relation_to_id)
    tf_test  = TriplesFactory.from_path(str(TEST_TXT),
                                        entity_to_id=tf_train.entity_to_id,
                                        relation_to_id=tf_train.relation_to_id)
    print(f"Entities  : {tf_train.num_entities:,}")
    print(f"Relations : {tf_train.num_relations:,}")
    print(f"Train     : {tf_train.num_triples:,}")
    print(f"Valid     : {tf_valid.num_triples:,}")
    print(f"Test      : {tf_test.num_triples:,}")
    return tf_train, tf_valid, tf_test


def train_model(model_name: str, tf_train, tf_valid, tf_test,
                epochs: int = NUM_EPOCHS, dim: int = EMBEDDING_DIM) -> dict:
    from pykeen.pipeline import pipeline

    print(f"\n{'='*60}")
    print(f"Training {model_name}  (dim={dim}, epochs={epochs})")
    print(f"{'='*60}")

    result = pipeline(
        training=tf_train,
        validation=tf_valid,
        testing=tf_test,
        model=model_name,
        model_kwargs={"embedding_dim": dim},
        training_kwargs={
            "num_epochs": epochs,
            "batch_size": BATCH_SIZE,
        },
        optimizer_kwargs={"lr": LEARNING_RATE},
        stopper="early",
        stopper_kwargs={"patience": 5, "frequency": 10},
        random_seed=42,
    )

    metrics = result.metric_results.to_flat_dict()
    mrr     = metrics.get("both.realistic.inverse_harmonic_mean_rank", 0)
    h1      = metrics.get("both.realistic.hits_at_1",  0)
    h3      = metrics.get("both.realistic.hits_at_3",  0)
    h10     = metrics.get("both.realistic.hits_at_10", 0)

    print(f"\n  MRR    : {mrr:.4f}")
    print(f"  Hits@1 : {h1:.4f}")
    print(f"  Hits@3 : {h3:.4f}")
    print(f"  Hits@10: {h10:.4f}")

    return {
        "model": model_name, "embedding_dim": dim, "epochs": epochs,
        "dataset": "full",
        "train_triples": tf_train.num_triples,
        "mrr": round(mrr, 4), "hits@1": round(h1, 4),
        "hits@3": round(h3, 4), "hits@10": round(h10, 4),
        "pipeline_result": result,
    }


# ── Size Sensitivity ────────────────────────────────────────────────────────
def size_sensitivity(tf_train, tf_valid, tf_test) -> list[dict]:
    """
    Train TransE on subsets: 20k / 50k / full dataset.
    Observe how performance scales with KB size.
    """
    import numpy as np
    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline

    rows = []
    total = tf_train.num_triples
    sizes = {"20k":  20_000, "50k": 50_000, "full": total}

    for label, size in sizes.items():
        size = min(size, total)
        idx  = np.random.default_rng(42).choice(total, size=size, replace=False)
        tf_sub = TriplesFactory.from_labeled_triples(
            tf_train.triples[idx],
            entity_to_id=tf_train.entity_to_id,
            relation_to_id=tf_train.relation_to_id,
        )
        print(f"\nSize sensitivity — {label} ({size:,} triples)")
        result = pipeline(
            training=tf_sub, validation=tf_valid, testing=tf_test,
            model="TransE",
            model_kwargs={"embedding_dim": EMBEDDING_DIM},
            training_kwargs={"num_epochs": 30, "batch_size": BATCH_SIZE},
            optimizer_kwargs={"lr": LEARNING_RATE},
            stopper="early",
            stopper_kwargs={"patience": 3, "frequency": 10},
            random_seed=42,
        )
        m = result.metric_results.to_flat_dict()
        mrr = m.get("both.realistic.inverse_harmonic_mean_rank", 0)
        h10 = m.get("both.realistic.hits_at_10", 0)
        print(f"  → MRR={mrr:.4f}, Hits@10={h10:.4f}")
        rows.append({
            "model": f"TransE-{label}", "embedding_dim": EMBEDDING_DIM,
            "epochs": 30, "dataset": label, "train_triples": size,
            "mrr": round(mrr, 4), "hits@1": None,
            "hits@3": None, "hits@10": round(h10, 4),
        })
    return rows


# ── Nearest Neighbors ────────────────────────────────────────────────────────
def nearest_neighbors(pipeline_result, tf_train, n_entities: int = 5,
                      k: int = 5) -> str:
    """
    For selected entities, retrieve top-k nearest neighbors in embedding space.
    Analyzes semantic coherence.
    """
    import numpy as np

    try:
        model = pipeline_result.model
        emb   = model.entity_representations[0](indices=None).detach().numpy()
    except Exception as e:
        return f"Could not extract embeddings: {e}"

    # Pick n_entities entities (first ones from vocab)
    ids   = list(tf_train.entity_to_id.items())[:n_entities]
    lines = []
    for name, idx in ids:
        vec      = emb[idx]
        # Cosine similarity
        norms    = np.linalg.norm(emb, axis=1, keepdims=True)
        norm_v   = np.linalg.norm(vec)
        if norm_v < 1e-9:
            continue
        cosine   = (emb @ vec) / (norms.squeeze() * norm_v + 1e-9)
        cosine[idx] = -1  # exclude self
        top_k    = np.argsort(cosine)[::-1][:k]
        id_to_name = {v: k for k, v in tf_train.entity_to_id.items()}
        neighbors = [id_to_name.get(i, "?") for i in top_k]
        lines.append(f"  {name[:60]}")
        for j, nb in enumerate(neighbors, 1):
            lines.append(f"    {j}. {nb[:60]}  (cos={cosine[top_k[j-1]]:.3f})")
    return "\n".join(lines)


# ── t-SNE Clustering ─────────────────────────────────────────────────────────
def plot_tsne(pipeline_result, tf_train):
    """
    Extract embeddings, apply t-SNE to 2D, color by entity URI prefix
    (proxy for ontology class since run without owlrl).
    """
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        model = pipeline_result.model
        emb   = model.entity_representations[0](indices=None).detach().numpy()

        n     = min(300, len(emb))
        idx   = np.random.default_rng(42).choice(len(emb), size=n, replace=False)
        emb_s = emb[idx]
        id_to = {v: k for k, v in tf_train.entity_to_id.items()}

        # Color by URI prefix (proxy for ontology type)
        colors = []
        for i in idx:
            uri = id_to.get(i, "")
            if "/entity/" in uri:   colors.append("steelblue")
            elif "/prop/"  in uri:  colors.append("tomato")
            elif "wikidata" in uri: colors.append("seagreen")
            else:                   colors.append("gray")

        tsne   = TSNE(n_components=2, random_state=42,
                      perplexity=min(30, n - 1))
        coords = tsne.fit_transform(emb_s)

        plt.figure(figsize=(10, 8))
        for color, label in [("steelblue", "Energy entities"),
                              ("tomato",    "Predicates"),
                              ("seagreen",  "Wikidata nodes"),
                              ("gray",      "Other")]:
            mask = [c == color for c in colors]
            if any(mask):
                plt.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=color, label=label, alpha=0.7, s=25,
                )
        plt.title("t-SNE of Entity Embeddings (colored by URI type)")
        plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
        plt.legend(); plt.tight_layout()
        plt.savefig(str(TSNE_PNG), dpi=150)
        plt.close()
        print(f"t-SNE plot → {TSNE_PNG}")
    except Exception as e:
        print(f"t-SNE skipped: {e}")


# ── Relation Behavior Discussion ─────────────────────────────────────────────
RELATION_BEHAVIOR = """
=== Relation Behavior Analysis ===

TransE assumptions: t ≈ h + r  (translational model)
RotatE assumptions: t = h ∘ r  (rotation in complex space)

Symmetric relations (e.g., mykb:prop/include):
  - TransE: h + r ≈ t AND t + r ≈ h → r ≈ 0 vector → can't distinguish direction
  - RotatE: represents symmetry as 180° rotation → handles correctly ✅

Inverse relations (e.g., produce ↔ supply):
  - TransE: r1 ≈ -r2  (approximation, not exact)
  - RotatE: r2 = conj(r1) in complex space → exact inverse ✅

Composition (e.g., locate ∘ produce → hasEnergyProducer):
  - TransE: r3 ≈ r1 + r2 → supports composition naturally ✅
  - RotatE: r3 = r1 ∘ r2 → also supports ✅

Conclusion: RotatE is more expressive than TransE for complex relation patterns.
TransE is faster to train and more interpretable for simple domains.
"""


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    import pandas as pd

    ensure_pykeen()

    if not TRAIN_TXT.exists():
        print("train.txt not found. Running prepare_kge.py first…")
        from src.kge.prepare_kge import prepare
        prepare()

    tf_train, tf_valid, tf_test = load_dataset()

    # ── 1. Train both required models ──────────────────────────────
    rows   = []
    models = ["TransE", "RotatE"]
    best   = None

    for model_name in models:
        r = train_model(model_name, tf_train, tf_valid, tf_test)
        rows.append({k: v for k, v in r.items() if k != "pipeline_result"})
        if best is None:
            best = r

    # ── 2. Size sensitivity ────────────────────────────────────────
    print("\n=== KB Size Sensitivity (TransE) ===")
    sensitivity_rows = size_sensitivity(tf_train, tf_valid, tf_test)
    rows.extend(sensitivity_rows)

    # ── 3. Nearest neighbors ───────────────────────────────────────
    print("\n=== Nearest Neighbor Analysis ===")
    nn_text = nearest_neighbors(best["pipeline_result"], tf_train)
    print(nn_text)
    NEIGHBORS_TXT.write_text(nn_text, encoding="utf-8")
    print(f"Neighbors saved → {NEIGHBORS_TXT}")

    # ── 4. t-SNE ──────────────────────────────────────────────────
    print("\n=== t-SNE Embedding Visualization ===")
    plot_tsne(best["pipeline_result"], tf_train)

    # ── 5. Relation behavior ───────────────────────────────────────
    print(RELATION_BEHAVIOR)

    # ── 6. Save results ───────────────────────────────────────────
    df = pd.DataFrame({k: v for k, v in r.items() if k != "pipeline_result"}
                      for r in rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n✅ KGE results → {RESULTS_CSV}")
    print(df[["model", "dataset", "train_triples", "mrr", "hits@1",
              "hits@3", "hits@10"]].to_string(index=False))


if __name__ == "__main__":
    main()
