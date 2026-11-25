import csv
import glob
import json
import os
import chromadb
import ollama
from pypdf import PdfReader

chroma_client = chromadb.PersistentClient(path="./chroma_db")

def chunk_text(text: str, chunk_size: int = 220, overlap: int = 35):
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size]).strip()
        # Cap at 1500 characters so dense assembly/code tables never overflow nomic-embed-text
        if chunk:
            chunks.append(chunk[:1500])
    return chunks

def ingest_all():
    os.makedirs("./dataset", exist_ok=True)
    all_paths = glob.glob("./dataset/**/*", recursive=True)
    files = [p for p in all_paths if os.path.isfile(p)]

    if not files:
        print("No files found in ./dataset/ or its subfolders.")
        return

    try:
        chroma_client.delete_collection("faculty_knowledge")
    except Exception:
        pass
    collection = chroma_client.create_collection(name="faculty_knowledge")

    documents = []
    metadatas = []

    for path in files:
        ext = os.path.splitext(path)[1].lower()
        source_name = os.path.basename(path)
        print(f"Reading: {path}")

        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    text = item.get("text") or f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}"
                    if text.strip():
                        documents.append(text.strip()[:1500])
                        metadatas.append({"source": source_name})

        elif ext == ".csv":
            with open(path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    text = row.get("text") or f"Q: {row.get('question', '')}\nA: {row.get('answer', '')}"
                    if text.strip():
                        documents.append(text.strip()[:1500])
                        metadatas.append({"source": source_name})

        elif ext == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                for chunk in chunk_text(f.read()):
                    documents.append(chunk)
                    metadatas.append({"source": source_name})

        elif ext == ".pdf":
            reader = PdfReader(path)
            full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
            for chunk in chunk_text(full_text):
                documents.append(chunk)
                metadatas.append({"source": source_name})

    print(f"Embedding {len(documents)} chunks with nomic-embed-text...")
    for idx, (doc, meta) in enumerate(zip(documents, metadatas)):
        emb = ollama.embeddings(model="nomic-embed-text", prompt=doc)["embedding"]
        collection.add(ids=[f"doc_{idx}"], embeddings=[emb], documents=[doc], metadatas=[meta])

    print(f"Done! Indexed {len(documents)} chunks from {len(files)} files into ./chroma_db.")

if __name__ == "__main__":
    ingest_all()