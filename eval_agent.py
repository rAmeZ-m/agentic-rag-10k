import json, time, os
from agent import ask

SEEN = [
    ("What was NVIDIA's operating income in fiscal 2026?", ["130387"]),
    ("How much goodwill did Intel report at the end of 2025?", ["23912", "23.9"]),
    ("What was Intel Products CCG revenue in 2025?", ["32228"]),
    ("What was Intel Products DCAI revenue in 2025?", ["16919"]),
    ("What were AMD's total assets as of December 27, 2025?", ["76926"]),
    ("What was AMD's Data Center segment net revenue in 2025?", ["16635"]),
    ("What was NVIDIA's diluted earnings per share in fiscal 2026?", ["4.90"]),
    ("How much cash and cash equivalents did Intel have at the end of 2025?", ["14265"]),
    ("What was NVIDIA's gross margin percentage in fiscal 2026?", ["71.1", "71.07"]),
    ("What share of AMD's total net revenue in 2025 came from the Data Center segment?", ["48.0", "48%", "48.02"]),
]
NEW = [
    ("How much did NVIDIA spend on research and development in fiscal 2026?", ["18497"]),
    ("What was AMD's total operating income in 2025?", ["3694"]),
    ("What was AMD's Embedded segment net revenue in 2025?", ["3454"]),
    ("What were Intel's short-term investments at the end of 2025?", ["23151"]),
    ("What was NVIDIA's income tax expense in fiscal 2026?", ["21383"]),
]
QA = [(q, g, "seen") for q, g in SEEN] + [(q, g, "new") for q, g in NEW]

OUT = "processed/agent_eval2.json"
done = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}

print("AGENT EVAL 2")
for q, gold, tag in QA:
    if q in done:
        continue
    try:
        r = ask(q)
        ans = r["answer"].replace(",", "")
        ok = any(g in ans for g in gold)
        done[q] = {"ok": ok, "tag": tag, "seconds": r["seconds"], "calls": len(r["calls"]), "answer": r["answer"]}
        print(f"{'PASS' if ok else 'FAIL'} [{tag}] {r['seconds']:>5}s calls={len(r['calls'])} | {q[:55]}", flush=True)
        if not ok:
            print("   ANSWER:", r["answer"][:300].replace("\n", " "), flush=True)
    except Exception as e:
        print("ERROR |", q[:50], "|", str(e)[:200], flush=True)
    json.dump(done, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    time.sleep(20)

for tag in ("seen", "new"):
    rows = [v for v in done.values() if v["tag"] == tag]
    print(f"{tag}: {sum(v['ok'] for v in rows)}/{len(rows)}")