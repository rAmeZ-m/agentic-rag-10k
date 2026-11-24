import json, pickle, re
import chromadb
from sentence_transformers import SentenceTransformer

chunks = json.load(open("processed/chunks.json", encoding="utf-8"))
by_id = {c["id"]: c for c in chunks}
_bm = pickle.load(open("processed/bm25.pkl", "rb"))
bm25, bm_ids = _bm["bm25"], _bm["ids"]
col = chromadb.PersistentClient(path="chroma_db").get_collection("tenk")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def tokenize(t):
    return re.findall(r"\w+(?:[.,]\w+)*", t.lower())

def hybrid_search(query, k=5, pool=20, rrf_k=60, company=None):
    qe = model.encode(["Represent this sentence for searching relevant passages: " + query],
                      normalize_embeddings=True)
    where = {"company": company.upper()} if company else None
    dense = col.query(query_embeddings=qe.tolist(), n_results=pool, where=where)["ids"][0]

    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    sparse = [bm_ids[i] for i in order if not company or by_id[bm_ids[i]]["company"] == company.upper()][:pool]

    fused = {}
    for ranking in (dense, sparse):
        for r, cid in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0) + 1 / (rrf_k + r)
    top = sorted(fused, key=lambda c: -fused[c])[:k]
    return [{**by_id[c], "score": fused[c],
             "dense_rank": dense.index(c) + 1 if c in dense else None,
             "bm25_rank": sparse.index(c) + 1 if c in sparse else None} for c in top]

if __name__ == "__main__":
    print("RETRIEVER V1")
    for q in ["What was NVIDIA's revenue in fiscal 2026?",
              "Basic net income per share 4.93",
              "How does AMD describe growth in its data center business?"]:
        print("\n###", q)
        for r in hybrid_search(q, k=5):
            print(f"{r['id']:<16} dense={r['dense_rank']} bm25={r['bm25_rank']} | {r['text'].splitlines()[1][:70]}")