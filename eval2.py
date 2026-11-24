from retriever import *

PREFIX = "Represent this sentence for searching relevant passages: "
STOP = set("what was were is are the a an of in for at as to did do does how much many by and or its their which that this with from on".split())

def detect_company(q):
    found = [c for c in ("nvidia", "amd", "intel") if c in q.lower()]
    return found[0].upper() if len(found) == 1 else None

def dense_ids(q, n=20, company=None):
    qe = model.encode([PREFIX + q], normalize_embeddings=True)
    where = {"company": company} if company else None
    return col.query(query_embeddings=qe.tolist(), n_results=n, where=where)["ids"][0]

def bm25_ids(q, n=20, company=None):
    s = bm25.get_scores([w for w in tokenize(q) if w not in STOP])
    order = sorted(range(len(s)), key=lambda i: -s[i])
    return [bm_ids[i] for i in order if not company or by_id[bm_ids[i]]["company"] == company][:n]

def fuse(rankings, weights, k=5, rrf_k=60):
    sc = {}
    for r, w in zip(rankings, weights):
        for pos, cid in enumerate(r, start=1):
            sc[cid] = sc.get(cid, 0) + w / (rrf_k + pos)
    return sorted(sc, key=lambda c: -sc[c])[:k]

QA = [
    ("What was NVIDIA's revenue in fiscal 2026?", "215,938"),
    ("What was NVIDIA's gross profit for the year ended January 25, 2026?", "153,463"),
    ("What was NVIDIA's net income in fiscal 2026?", "120,067"),
    ("What was NVIDIA's diluted earnings per share in fiscal 2026?", "4.90"),
    ("What was AMD's total net revenue in 2025?", "34,639"),
    ("What was AMD's Data Center segment net revenue in 2025?", "16,635"),
    ("What were AMD's total assets as of December 27, 2025?", "76,926"),
    ("What were Intel's total assets as of December 27, 2025?", "211,429"),
    ("How much cash and cash equivalents did Intel have at the end of 2025?", "14,265"),
    ("What was Intel Products CCG revenue in 2025?", "32,228"),
    ("How much did NVIDIA spend on research and development in fiscal 2026?", "18,497"),
    ("What were NVIDIA's total operating expenses in fiscal 2026?", "23,076"),
    ("What was NVIDIA's operating income in fiscal 2026?", "130,387"),
    ("What were AMD's total current assets as of December 27, 2025?", "26,947"),
    ("How much goodwill did AMD report at the end of 2025?", "25,126"),
    ("What was AMD's total operating income in 2025?", "3,694"),
    ("How much goodwill did Intel report at the end of 2025?", "23,912"),
    ("What was Intel Products DCAI revenue in 2025?", "16,919"),
    ("What were Intel's total current assets as of December 27, 2025?", "63,688"),
    ("Basic net income per share 4.93", "4.93"),
]

def run(q, co):
    return {
        "dense":    dense_ids(q)[:5],
        "dense+co": dense_ids(q, company=co)[:5],
        "bm25+co":  bm25_ids(q, company=co)[:5],
        "rrf+co":   fuse([dense_ids(q, company=co), bm25_ids(q, company=co)], [1, 1]),
        "wrrf+co":  fuse([dense_ids(q, company=co), bm25_ids(q, company=co)], [2, 1]),
    }

print("EVAL V2 | hit@5")
names = ["dense", "dense+co", "bm25+co", "rrf+co", "wrrf+co"]
hits = {n: [0, 0] for n in names}
for idx, (q, gold) in enumerate(QA):
    res = run(q, detect_company(q))
    row = []
    for n in names:
        ok = any(gold in by_id[i]["text"] for i in res[n])
        hits[n][idx >= 10] += ok
        row.append(f"{n}={'Y' if ok else '-'}")
    print(f"{gold:<8} " + " ".join(row))
print("first 10 (seen):", {n: f"{h[0]}/10" for n, h in hits.items()})
print("last 10 (new)  :", {n: f"{h[1]}/10" for n, h in hits.items()})