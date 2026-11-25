import time
import ollama

MODELS = ["llama3.2:1b-instruct-q4_K_M", "phi3:3.8b-mini-4k-instruct-q4_K_M"]
PROMPTS = [
    "Where is the admissions office and what documents do I need to register?",
    "ممكن تقولي مكان شؤون الطلاب إيه والأوراق المطلوبة للتقديم؟",
    "إيه الفرق بين قسم ميكاترونكس وقسم حاسبات؟"
]

for model in MODELS:
    print(f"\n=== Benchmarking: {model} ===")
    for prompt in PROMPTS:
        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0

        stream = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"num_ctx": 2048}
        )

        response_text = ""
        for chunk in stream:
            if first_token_time is None:
                first_token_time = time.perf_counter() - start_time
            token_count += 1
            response_text += chunk["message"]["content"]

        total_time = time.perf_counter() - start_time
        gen_time = total_time - first_token_time
        tps = token_count / gen_time if gen_time > 0 else 0
        print(f"Prompt: {prompt[:35]}... | TTFT: {first_token_time:.2f}s | Speed: {tps:.1f} tok/s")
        print(f"Sample Reply: {response_text.strip()[:120]}...\n")