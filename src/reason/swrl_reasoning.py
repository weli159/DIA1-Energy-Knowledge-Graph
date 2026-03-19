"""
src/reason/swrl_reasoning.py
─────────────────────────────────────────────────────────────
TD5 Part 1 — Knowledge Reasoning with SWRL
─────────────────────────────────────────────────────────────
Part A: family.owl  — SWRL rule: oldPerson (age > 60)
Part B: Energy KB  — Horn rule with 2 conditions on your KB
        + comparison with embedding vectors

Required by grading guide:
  - SWRL rule on family.owl + output
  - One SWRL rule on your KB
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KGA  = ROOT / "kg_artifacts"

# ─────────────────────────────────────────────────────────────
# PART A  —  family.owl  (required by grading guide Section 3)
# ─────────────────────────────────────────────────────────────
def run_family_swrl():
    """
    Load family.owl with OWLReady2 and apply a SWRL rule:
    'A person who is older than 60 years old is an oldPerson.'

    SWRL Rule:
        Person(?p) ∧ age(?p, ?a) ∧ swrlb:greaterThan(?a, 60) → oldPerson(?p)
    """
    try:
        from owlready2 import get_ontology, sync_reasoner_hermit, default_world
        import owlready2
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "owlready2", "-q"])
        from owlready2 import get_ontology, sync_reasoner_hermit, default_world
        import owlready2

    # Find family_lab.owl
    candidates = [
        KGA / "family_lab.owl",
        ROOT.parent / "td3" / "deliverables_Ouali Youssef Smash Yassin" / "family_lab.owl",
    ]
    owl_path = next((p for p in candidates if p.exists()), None)
    if owl_path is None:
        print("WARNING: family_lab.owl not found. Copy it to kg_artifacts/family_lab.owl")
        return

    print("=" * 60)
    print("PART A — family.owl SWRL Reasoning")
    print("=" * 60)
    print(f"Loading: {owl_path}\n")

    onto = get_ontology("http://test.org/family.owl").load(fileobj=open(owl_path, "rb"))

    # Define SWRL rule: Person(?p) ∧ age(?p,?a) ∧ swrlb:greaterThan(?a,60) → oldPerson(?p)
    with onto:
        # Check if oldPerson class exists, create if not
        if not onto.search_one(iri="*oldPerson"):
            from owlready2 import Thing
            class oldPerson(Thing):
                pass


        # Add the SWRL rule
        try:
            rule = owlready2.Imp()
            rule.set_as_rule(
                "Person(?p), age(?p, ?a), swrlb:greaterThan(?a, 60) -> oldPerson(?p)"
            )
            print("SWRL Rule added:")
            print("  Person(?p) ∧ age(?p, ?a) ∧ swrlb:greaterThan(?a, 60) → oldPerson(?p)\n")
        except Exception as e:
            print(f"Note: SWRL rule syntax error ({e}). Using HermiT for OWL reasoning only.\n")

    # Run HermiT reasoner
    try:
        with onto:
            sync_reasoner_hermit(infer_property_values=True)
        print("✅ HermiT reasoning complete.\n")
    except Exception as e:
        print(f"Note: HermiT error: {e}\n")

    # Print inferred individuals
    print("--- Inferred family tree links ---")
    for person in onto.individuals():
        parents = [p.name for p in getattr(person, "isChildOf", [])]
        if parents:
            print(f"  {person.name} isChildOf: {', '.join(parents)}")

    # Print oldPerson instances (if rule was applied)
    print("\n--- oldPerson instances (age > 60, inferred by SWRL rule) ---")
    try:
        old_cls = onto.search_one(iri="*oldPerson")
        if old_cls:
            instances = list(old_cls.instances())
            if instances:
                for inst in instances:
                    age_val = getattr(inst, "age", [None])
                    a = age_val[0] if isinstance(age_val, list) else age_val
                    print(f"  {inst.name}  (age={a})")
            else:
                # Fallback: manual check
                print("  (SWRL rule not fired by HermiT — checking manually)")
                for person in onto.individuals():
                    age_prop = getattr(person, "age", None)
                    if age_prop is not None:
                        a = age_prop[0] if isinstance(age_prop, list) else age_prop
                        try:
                            if float(a) > 60:
                                print(f"  {person.name}  (age={a}) → qualifies as oldPerson")
                        except (TypeError, ValueError):
                            pass
    except Exception as e:
        print(f"  Could not query oldPerson: {e}")

    print()


# ─────────────────────────────────────────────────────────────
# PART B  —  Energy KB SWRL rule (Section 8 of lab)
# ─────────────────────────────────────────────────────────────
def run_energy_swrl():
    """
    Apply one SWRL-style horn rule on the energy KB (2 conditions).

    Rule (simulated via SPARQL CONSTRUCT — OWL reasoning without a triplestore):
      produce(?X, ?Z) ∧ locatedIn(?X, ?Y) → hasEnergyProducer(?Y, ?X)

    Meaning: if an entity X produces something AND is located in country Y,
             then Y hasEnergyProducer X.

    This is the SPARQL-CONSTRUCT equivalent of the SWRL horn rule:
      mykb:produce(?X, ?Z) ∧ mykb:locate(?X, ?Y) → mykb:hasEnergyProducer(?Y, ?X)
    """
    try:
        from rdflib import Graph, Namespace, URIRef
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rdflib", "-q"])
        from rdflib import Graph, Namespace, URIRef

    # Load best available KB
    ttl_path = None
    for candidate in [KGA / "expanded.nt", KGA / "alignment.ttl"]:
        if candidate.exists():
            ttl_path = candidate
            break

    if ttl_path is None:
        print("WARNING: No energy KB file found. Run TD4 pipeline first.")
        return

    MYKB = Namespace("http://private.org/energy/")

    print("=" * 60)
    print("PART B — Energy KB SWRL Rule (Horn Rule)")
    print("=" * 60)
    fmt = "nt" if str(ttl_path).endswith(".nt") else "turtle"
    g = Graph()
    g.parse(str(ttl_path), format=fmt)
    print(f"Loaded {len(g):,} triples from {ttl_path.name}\n")

    print("SWRL Rule:")
    print("  produce(?X, ?Z) ∧ locate(?X, ?Y) → hasEnergyProducer(?Y, ?X)")
    print("  (If X produces Z and is located in Y → Y has energy producer X)\n")

    # SPARQL CONSTRUCT equivalent of the SWRL rule
    RULE = """
    PREFIX mykb: <http://private.org/energy/>
    PREFIX prop: <http://private.org/energy/prop/>
    CONSTRUCT {
        ?country prop:hasEnergyProducer ?producer .
    }
    WHERE {
        ?producer prop:produce ?product .
        ?producer prop:locate  ?country .
    }
    """
    new_facts = list(g.query(RULE))
    print(f"New triples inferred by rule: {len(new_facts)}")
    for s, p, o in new_facts[:10]:
        print(f"  <{str(s).split('/')[-1]}> hasEnergyProducer <{str(o).split('/')[-1]}>")
    if len(new_facts) > 10:
        print(f"  ... ({len(new_facts)} total)")

    for s, p, o in new_facts:
        g.add((s, p, o))

    # Save enriched ontology
    out = KGA / "ontology.ttl"
    g.serialize(destination=str(out), format="turtle")
    print(f"\n✅ Saved enriched ontology → {out}  ({len(g):,} triples)")

    print("\n--- Embedding comparison (Section 8) ---")
    print("Rule:  produce(?X,?Z) ∧ locate(?X,?Y) → hasEnergyProducer(?Y,?X)")
    print("Embedding check: vector(produce) + vector(locate) ≈ vector(hasEnergyProducer)?")
    print("(Run src/kge/train_kge.py and inspect relation vectors to verify.)")
    return g


if __name__ == "__main__":
    run_family_swrl()
    run_energy_swrl()
    print("\n✅ SWRL reasoning complete.")
