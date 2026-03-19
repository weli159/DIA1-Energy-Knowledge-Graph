from flask import Flask, request, jsonify, render_template_string
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.rag.lab_rag_sparql_gen import (
    load_graph, build_schema_summary,
    answer_with_sparql_generation, answer_no_rag, TTL_FILE
)

app = Flask(__name__)

print(f"Loading KB for Web UI from {TTL_FILE} ...")
g = load_graph(TTL_FILE)
schema = build_schema_summary(g)
print(f"KB loaded: {len(g)} triples ready for RAG.")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Energy Knowledge Graph RAG</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f6f9; color: #333; margin: 0; padding: 2rem; }
        .container { max-width: 800px; margin: auto; background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; font-size: 1.8rem; text-align: center; margin-bottom: 1.5rem; }
        .chat-box { height: 400px; overflow-y: auto; background: #fcfcfc; border: 1px solid #ddd; padding: 1rem; border-radius: 6px; margin-bottom: 1.5rem; display: flex; flex-direction: column; gap: 1rem; }
        .message { max-width: 80%; padding: 0.8rem 1rem; border-radius: 6px; line-height: 1.4; }
        .user-msg { align-self: flex-end; background: #3498db; color: white; }
        .bot-msg { align-self: flex-start; background: #ecf0f1; border-left: 4px solid #e74c3c; width: 95%; max-width: 95%; }
        .baseline-answer { font-style: italic; color: #666; background: #fff; padding: 0.6rem; border-radius: 4px; border: 1px dashed #bbb; margin-bottom: 0.8rem; font-size: 0.9rem; }
        .rag-answer { background: #dff9fb; padding: 0.8rem; border-radius: 6px; border: 1px solid #c7ecee; font-weight: 500; margin-bottom: 0.5rem; }
        .sparql-details, .data-details { margin-top: 0.5rem; cursor: pointer; font-size: 0.85rem; color: #2980b9; }
        .sparql-code { background: #2c3e50; color: #ecf0f1; padding: 0.8rem; border-radius: 4px; font-family: 'Consolas', monospace; font-size: 0.8rem; white-space: pre-wrap; margin-top: 0.3rem; }
        .result-table { width: 100%; border-collapse: collapse; margin-top: 0.3rem; font-size: 0.8rem; background: white; }
        .result-table th, .result-table td { border: 1px solid #bdc3c7; padding: 0.4rem; text-align: left; }
        .result-table th { background: #e0e6ed; }
        .input-area { display: flex; gap: 0.5rem; }
        input[type="text"] { flex: 1; padding: 0.8rem; border: 1px solid #ccc; border-radius: 4px; outline: none; font-size: 1rem; }
        button { padding: 0.8rem 1.5rem; background: #27ae60; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 1rem; transition: background 0.2s; }
        button:hover { background: #2ecc71; }
        button:disabled { background: #95a5a6; cursor: not-allowed; }
        .loading { font-style: italic; color: #7f8c8d; font-size: 0.9rem; align-self: center; display: none; margin-top: -0.5rem; }
    </style>
</head>
<body>

<div class="container">
    <h1>🌱 Energy KG — RAG Chatbot</h1>
    <div class="chat-box" id="chatbox">
        <div class="message bot-msg">
            Hello! I am connected to the Energy Knowledge Graph ({{ triples }} triples). 
            Ask me a question, and I will generate a SPARQL query to answer it!
        </div>
    </div>
    <div class="loading" id="loading">Generating SPARQL and executing...</div>
    <div class="input-area">
        <input type="text" id="userInput" placeholder="E.g., Which organizations produce electricity?" onkeypress="if(event.key === 'Enter') sendMessage()">
        <button id="sendBtn" onclick="sendMessage()">Send</button>
    </div>
</div>

<script>
    async function sendMessage() {
        const input = document.getElementById("userInput");
        const btn = document.getElementById("sendBtn");
        const chatbox = document.getElementById("chatbox");
        const loading = document.getElementById("loading");

        const q = input.value.trim();
        if (!q) return;

        chatbox.innerHTML += `<div class="message user-msg">${q}</div>`;
        input.value = "";
        btn.disabled = true;
        loading.style.display = "block";
        chatbox.scrollTop = chatbox.scrollHeight;

        try {
            const response = await fetch("/api/rag", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({question: q})
            });
            
            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`Server Error (${response.status}): ${errText}`);
            }

            const data = await response.json();
            let botReply = ""; 

            if (data.baseline) {
                botReply += `<div class="baseline-answer"><b>Baseline Answer (General Knowledge):</b><br>${data.baseline}</div>`;
            }

            if (data.answer) {
                botReply += `<div class="rag-answer"><b>Final Answer (from Knowledge Graph):</b><br>${data.answer}</div>`;
            }

            if (data.query) {
                botReply += `
                <details class="sparql-details">
                    <summary>🔍 View SPARQL Logic</summary>
                    <div class="sparql-code">${data.query.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</div>
                </details>`;
            }

            if (data.rows && data.rows.length > 0) {
                let tableHtml = `<table class="result-table"><thead><tr>`;
                data.vars.forEach(v => tableHtml += `<th>${v}</th>`);
                tableHtml += `</tr></thead><tbody>`;
                data.rows.slice(0, 20).forEach(r => {
                    tableHtml += `<tr>`;
                    r.forEach(c => tableHtml += `<td>${c}</td>`);
                    tableHtml += `</tr>`;
                });
                tableHtml += `</tbody></table>`;
                if(data.rows.length > 20) tableHtml += `<div style="font-size:0.75rem; margin-top:5px; color:#7f8c8d;">... showing 20 of ${data.rows.length} results.</div>`;
                
                botReply += `
                <details class="data-details">
                    <summary>📊 View Raw Data Results</summary>
                    ${tableHtml}
                </details>`;
            } else if (!data.error) {
                botReply += `<br><b>No factual matches found in graph.</b>`;
            }

            chatbox.innerHTML += `<div class="message bot-msg">${botReply}</div>`;
        } catch (err) {
            console.error(err);
            // THIS IS THE NEW ERROR MESSAGE WE ARE LOOKING FOR:
            chatbox.innerHTML += `<div class="message bot-msg" style="color:red; font-size: 0.9rem;"><b>NEW ERROR:</b> ${err.message}</div>`;
        }

        btn.disabled = false;
        loading.style.display = "none";
        chatbox.scrollTop = chatbox.scrollHeight;
    }
</script>

</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, triples=len(g))

@app.route("/api/rag", methods=["POST"])
def rag_api():
    try:
        q = request.json.get("question", "")
        if not q:
            return jsonify({"error": "Empty question"}), 400
        
        baseline = answer_no_rag(q)
        result = answer_with_sparql_generation(g, schema, q, try_repair=True)
        result["baseline"] = baseline
        
        if "query" in result and result["query"]:
            clean_lines = [line for line in result["query"].split("\n") if not line.strip().startswith("PREFIX")]
            result["query"] = "\n".join(clean_lines).strip()
            
        return jsonify(result)
    
    except Exception as e:
        traceback.print_exc()
        return str(e), 500

def main():
    print("\n" + "="*50)
    print("Starting BRAND NEW Web UI on http://127.0.0.1:5001")
    print("="*50 + "\n")
    app.run(host="127.0.0.1", port=5001, debug=False)

if __name__ == "__main__":
    main()