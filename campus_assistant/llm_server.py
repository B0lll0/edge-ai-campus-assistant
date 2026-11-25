import time
from flask import Flask, request, Response, jsonify
import ollama
from rag_inference import stream_grounded_answer, seed_sample_data

app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ready", "model": "campus-assistant"})

@app.route("/ask_stream", methods=["POST"])
def ask_stream():
    """Streaming sentence endpoint for Student 4 (TTS) to achieve <2s latency."""
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    def generate():
        for sentence in stream_grounded_answer(query):
            yield sentence + "\n"

    return Response(generate(), mimetype="text/plain; charset=utf-8")

@app.route("/chat", methods=["POST"])
def chat():
    """Full JSON endpoint for Student 5 (Kiosk Text UI) & end-to-end testing."""
    t0 = time.perf_counter()
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    sentences = list(stream_grounded_answer(query))
    return jsonify({
        "query": query,
        "answer": " ".join(sentences),
        "sentences": sentences,
        "latency_sec": round(time.perf_counter() - t0, 2)
    })

if __name__ == "__main__":
    seed_sample_data()
    ollama.generate(model="campus-assistant", prompt="", keep_alive=-1)
    ollama.embeddings(model="nomic-embed-text", prompt="warmup", keep_alive=-1)
    app.run(host="0.0.0.0", port=5000, threaded=True)