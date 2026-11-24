import json, pickle, re
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

chunks = json.load(open("processed/chunks.json", encoding="utf-8"))
texts = [c["text"] for c in chunks]

def tokenize(t):
    return re.findall(r"\w+(?:[.,]\w+)*", t.lower())  # keep on 215,938 and 4.93

# Dense
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
emb = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)

client = chromadb.PersistentClient(path="chroma_db")
try:
    client.delete_collection("tenk")
except Exception:
    pass
col = client.create_collection("tenk", metadata={"hnsw:space": "cosine"})
for i in range(0, len(chunks), 500):
    b = chunks[i:i + 500]
    col.add(
        ids=[c["id"] for c in b],
        documents=[c["text"] for c in b],
        embeddings=emb[i:i + 500].tolist(),
        metadatas=[{"company": c["company"], "page": c["page"]} for c in b],
    )

# Sparse
bm25 = BM25Okapi([tokenize(t) for t in texts])
pickle.dump({"bm25": bm25, "ids": [c["id"] for c in chunks]}, open("processed/bm25.pkl", "wb"))
print("INDEX V1 | chroma:", col.count(), "| bm25 docs:", len(chunks))

# Sanity test
q = "What was NVIDIA's revenue in fiscal 2026?"
qe = model.encode(["Represent this sentence for searching relevant passages: " + q], normalize_embeddings=True)
print("DENSE:", col.query(query_embeddings=qe.tolist(), n_results=3)["ids"][0])
scores = bm25.get_scores(tokenize(q))
top = sorted(range(len(scores)), key=lambda i: -scores[i])[:3]
print("BM25 :", [chunks[i]["id"] for i in top])