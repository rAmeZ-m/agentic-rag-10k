import json, statistics
import re

pages = json.load(open("processed/pages_clean.json", encoding="utf-8"))
MAX, OVERLAP = 1500, 2

chunks = []
DATE_LINE = re.compile(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? ?\d{1,2}, ?\d{4}")

for p in pages:
    parts, cur, size = [], [], 0
    for ln in p["text"].splitlines():
        if cur and size + len(ln) + 1 > MAX:
            parts.append(cur)
            cur = cur[-OVERLAP:]
            size = sum(len(l) + 1 for l in cur)
        cur.append(ln)
        size += len(ln) + 1
    if cur:
        parts.append(cur)
    last_dates = ""
    for j, part in enumerate(parts):
        body = "\n".join(part)
        if j > 0 and not DATE_LINE.search(body) and last_dates:
            body = f"(period header from earlier on page: {last_dates})\n" + body
        for ln in part:
            if len(DATE_LINE.findall(ln)) >= 2 or (DATE_LINE.search(ln) and len(ln) < 40):
                last_dates = ln
        chunks.append({
            "id": f"{p['company']}-p{p['page']}-c{j}",
            "company": p["company"],
            "page": p["page"],
            "text": f"[{p['company']} 10-K | PDF page {p['page']}]\n{body}",
        })

json.dump(chunks, open("processed/chunks.json", "w", encoding="utf-8"), ensure_ascii=False)
lens = [len(c["text"]) for c in chunks]
print("CHUNK V1 | chunks:", len(chunks), "| avg:", int(statistics.mean(lens)), "| max:", max(lens), "| under 300:", sum(l < 300 for l in lens))

for c in chunks:
    if c["company"] == "NVIDIA" and c["page"] == 53:
        print("=====", c["id"], len(c["text"]))
        print(c["text"])