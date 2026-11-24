import fitz, json, re
from pathlib import Path

DATE = re.compile(r"\d{4}/\u200f?\d{1,2}/\u200f?\d{1,2}")

def page_lines(page, tol=3):
    items = sorted(((w[1] + w[3]) / 2, w[0], w[4]) for w in page.get_text("words"))
    rows = []
    for y, x, t in items:
        if rows and abs(y - rows[-1][0]) <= tol:
            rows[-1][1].append((x, t))
        else:
            rows.append([y, [(x, t)]])
    return [" ".join(t for x, t in sorted(r[1])) for r in rows]

def keep(s):
    if DATE.search(s) or s.startswith("https://www.sec.gov/Archives"):
        return False
    if re.fullmatch(r"\d+/\d+", s) or s == "Table of Contents":
        return False
    return True

pages = []
for pdf in sorted(Path("data").glob("*_10k.pdf")):
    company = pdf.stem.replace("_10k", "").upper()
    for i, page in enumerate(fitz.open(pdf), start=1):
        lines = [s for s in (l.strip() for l in page_lines(page)) if s and keep(s)]
        text = "\n".join(lines)
        if len(text) >= 50:
            pages.append({"company": company, "page": i, "text": text})

Path("processed").mkdir(exist_ok=True)
json.dump(pages, open("processed/pages_clean.json", "w", encoding="utf-8"), ensure_ascii=False)
print("EXTRACT V2 | pages kept:", len(pages))

for p in pages:
    if p["company"] == "INTEL" and "Intel Products Financial Performance" in p["text"]:
        print(f"--- INTEL PDF page {p['page']} ---")
        print(p["text"][:1200])
        break