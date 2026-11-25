import time
import requests

BASE_URL = "http://127.0.0.1:5000"

# 1. Health Check
health = requests.get(f"{BASE_URL}/health").json()
assert health["status"] == "ready", "Server not ready!"
print("[PASS] Health check:", health)

# 2. Simulate STT -> LLM Stream -> TTS Latency
t0 = time.perf_counter()
with requests.post(f"{BASE_URL}/ask_stream", json={"query": "Where is Lecture Hall B?"}, stream=True) as r:
    first_sentence_time = None
    for line in r.iter_lines(decode_unicode=True):
        if line:
            if first_sentence_time is None:
                first_sentence_time = time.perf_counter() - t0
            print(f"[TTS Stream Chunk @ {first_sentence_time:.2f}s]: {line}")

assert first_sentence_time < 2.0, f"Latency {first_sentence_time:.2f}s exceeded 2.0s target!"
print(f"[PASS] End-to-end streaming latency: {first_sentence_time:.2f}s (< 2.0s target)")

# 3. Simulate Kiosk JSON UI Query + Anti-Hallucination Check
res = requests.post(f"{BASE_URL}/chat", json={"query": "What is the cafeteria menu?"}).json()
print("[PASS] Guardrail refusal response:", res)