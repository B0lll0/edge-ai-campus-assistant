import time
import chromadb
import ollama

# 1. Connect to local ChromaDB persistent folder
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="faculty_knowledge")

def seed_sample_data():
    """Seeds initial test records if the database is empty."""
    if collection.count() == 0:
        sample_docs = [
            "The Student Affairs and Admissions office is located on the Ground Floor of Building A, Room 104. Required documents for registration: original high school certificate, national ID copy, 6 personal photos, and birth certificate.",
            "Lecture Hall B (مدرج ب) is on the First Floor of the Main Engineering Building next to the elevators.",
            "مكتب شؤون الطلاب والتسجيل موجود في الدور الأرضي مبنى A غرفة 104، والأوراق المطلوبة: أصل شهادة الثانوية، صورة البطاقة، 6 صور شخصية، وشهادة الميلاد."
        ]
        for idx, doc in enumerate(sample_docs):
            emb = ollama.embeddings(model="nomic-embed-text", prompt=doc)["embedding"]
            collection.add(ids=[f"doc_{idx}"], embeddings=[emb], documents=[doc])

def retrieve_context(query: str, top_k: int = 2) -> str:
    """Embeds the student query and pulls top matching faculty chunks."""
    emb = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]
    results = collection.query(query_embeddings=[emb], n_results=top_k)
    docs = results.get("documents", [[]])[0]
    return "\n---\n".join(docs)

def stream_grounded_answer(user_query: str):
    """Streams complete sentences immediately to hit <2s first-sentence latency."""
    t0 = time.perf_counter()
    context = retrieve_context(user_query, top_k=2)
    prompt = f"[CONTEXT]\n{context}\n\n[STUDENT QUESTION]\n{user_query}"

    stream = ollama.chat(
        model="campus-assistant",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    sentence_buffer = ""
    first_token_logged = False

    for chunk in stream:
        token = chunk["message"]["content"]
        if not first_token_logged:
            print(f"\n[Latency to first token: {time.perf_counter() - t0:.2f}s]")
            first_token_logged = True

        sentence_buffer += token
        if any(punct in token for punct in [".", "!", "?", "؟", "\n"]):
            clean = sentence_buffer.strip()
            if clean:
                yield clean
            sentence_buffer = ""

    if sentence_buffer.strip():
        yield sentence_buffer.strip()

if __name__ == "__main__":
    seed_sample_data()
    # Pre-warm models in RAM so cold-start doesn't inflate latency
    ollama.generate(model="campus-assistant", prompt="", keep_alive=-1)
    ollama.embeddings(model="nomic-embed-text", prompt="warmup", keep_alive=-1)

    test_queries = [
        "Where is Lecture Hall B?",
        "مكتب شؤون الطلاب فين والأوراق المطلوبة إيه؟",
        "What is the cafeteria menu?"
    ]
    for q in test_queries:
        print(f"\nStudent: {q}")
        for sentence in stream_grounded_answer(q):
            print(f"Avatar TTS Chunk -> {sentence}")