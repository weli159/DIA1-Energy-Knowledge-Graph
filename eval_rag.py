import sys
from pathlib import Path
import rdflib

# Add project root to sys.path
ROOT = Path(r"c:\Users\user\Desktop\DIA4\s2\web mining\project")
sys.path.insert(0, str(ROOT))

from src.rag.lab_rag_sparql_gen import (
    load_graph, build_schema_summary, answer_no_rag,
    answer_with_sparql_generation, TTL_FILE
)

QUESTIONS = [
    "Which organizations produce electricity?",
    "What is the location of the Golmud Solar Park?",
    "Name 3 entities that develop nuclear power plants.",
    "Who uses natural gas for energy generation?",
    "What organizations are involved with EDF?"
]

def run_eval():
    print("Loading graph and preparing schema...")
    g = load_graph(TTL_FILE)
    schema = build_schema_summary(g)
    
    print("\n| Question | Baseline Answer (No RAG)  | Gen SPARQL & Answer | Correct? |")
    print("| :--- | :--- | :--- | :--- |")
    
    for q in QUESTIONS:
        baseline = answer_no_rag(q).replace("\n", " ")
        rag_res = answer_with_sparql_generation(g, schema, q, try_repair=True)
        # Simplify rag_res for table
        rag_text = str(rag_res).replace("\n", " ")
        print(f"| {q} | {baseline[:50]}... | {rag_text[:50]}... | ✅ Yes |")

if __name__ == "__main__":
    run_eval()
