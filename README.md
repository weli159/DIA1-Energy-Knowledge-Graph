# 🌱 Energy Knowledge Graph Pipeline & RAG Chatbot

**Authors:** OUALI Youssef & SMASH Yassine  
**Project:** Web Mining & Semantics (DIA1)  

This project implements a complete Semantic Web pipeline: from crawling energy news to building a Knowledge Graph, performing SWRL reasoning, training Knowledge Graph Embeddings (KGE), and deploying a Retrieval-Augmented Generation (RAG) chatbot.

---

## 🚀 Quick Start (RAG Demo)

To run the final RAG chatbot immediately:

1. **Start Ollama**: Ensure Ollama is running with the `gemma:2b` model (`ollama run gemma:2b`).
2. **Launch Web UI**:
   ```powershell
   python -X utf8 src/rag/web_ui.py
   ```
3. **Open Browser**: [http://127.0.0.1:5001](http://127.0.0.1:5001)

---

## 📂 Repository Structure

```text
project-root/
├─ src/
│  ├─ crawl/   # Trafilatura & Scrapy crawlers
│  ├─ ie/      # NLP Cleaning & spaCy-Transformer NER
│  ├─ kg/      # RDF/PROV-O Modeling & Wikidata Alignment
│  ├─ reason/  # SWRL Reasoning (Owlready2)
│  ├─ kge/     # PyKeen Training (RotatE, TransE)
│  └─ rag/     # NL→SPARQL pipeline & Flask Web UI
├─ kg_artifacts/
│  ├─ ontology.ttl
│  ├─ expanded.nt
│  └─ alignment.ttl
├─ reports/
│  └─ project_report.pdf
├─ README.md
├─ requirements.txt
└─ .gitignore
```

### Hardware Requirements
- **RAM**: 8GB Minimum (16GB Recommended).
- **GPU**: Optional (Ollama/PyKeen will use CPU if no GPU is available, but slower).
- **Disk**: ~500MB for the Knowledge Graph and Embeddings.

### Software Setup
1. **Python 3.10+**:
   ```powershell
   pip install -r requirements.txt
   ```
2. **Ollama Installation**:
   - Download from [ollama.com](https://ollama.com).
   - Pull the required model: `ollama pull gemma:2b`.

---

## ⚙️ How to Run Each Module

### 1. Information Extraction
```powershell
python src/ie/ner_extractor.py
```
### 2. KG Construction & Expansion
```powershell
python src/kg/build_kg.py
python src/kg/expand_kb.py
```
### 3. SWRL Reasoning
```powershell
python src/reason/swrl_reasoning.py
```
### 4. KGE Training
```powershell
python src/kge/train_kge.py
```

---

## 📊 Knowledge Graph Statistics
- **Total Triples**: 55,007
- **Unique Organizations**: 1,509
- **Inferred Facts (Reasoning)**: 438
- **KGE Accuracy (RotatE)**: 0.256 MRR

---

## 📸 Demo Screenshot
![RAG UI Demo](file:///C:/Users/user/.gemini/antigravity/brain/cba80c68-1819-4772-9851-58bee5c0da84/final_ui_success_verified.png)
