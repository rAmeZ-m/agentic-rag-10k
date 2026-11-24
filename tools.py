from dotenv import load_dotenv
load_dotenv()
import ast, operator
import subprocess, sys
from typing import Optional
from langchain.tools import tool
from retriever import model, col, bm25, bm_ids, by_id, tokenize

PREFIX = "Represent this sentence for searching relevant passages: "
STOP = set("what was were is are the a an of in for at as to did do does how much many by and or its their which that this with from on".split())
COMPANIES = ("NVIDIA", "AMD", "INTEL")

def _dense(q, n, company):
    qe = model.encode([PREFIX + q], normalize_embeddings=True)
    where = {"company": company} if company else None
    return col.query(query_embeddings=qe.tolist(), n_results=n, where=where)["ids"][0]

def _bm25(q, n, company):
    s = bm25.get_scores([w for w in tokenize(q) if w not in STOP])
    order = sorted(range(len(s)), key=lambda i: -s[i])
    return [bm_ids[i] for i in order if not company or by_id[bm_ids[i]]["company"] == company][:n]

@tool
def search_knowledge_base(query: str, company: str = "") -> str:
    """Search the 10-K annual reports of NVIDIA, AMD and Intel (fiscal 2025/2026).
    Use a specific query with the exact line-item or topic name (e.g. 'Consolidated Balance Sheets total assets').
    Set company to NVIDIA, AMD or INTEL to search only that company's report; leave empty to search all three.
    Returns text passages, each starting with [COMPANY 10-K | PDF page N]. Numbers are in millions of USD unless stated."""
    company = company.upper() if company else None
    if company and company not in COMPANIES:
        return f"Error: company must be one of {COMPANIES} or empty."
    sc = {}
    for ranking in (_dense(query, 20, company), _bm25(query, 20, company)):
        for pos, cid in enumerate(ranking, start=1):
            sc[cid] = sc.get(cid, 0) + 1 / (60 + pos)
    top = sorted(sc, key=lambda c: -sc[c])[:4]
    return "\n\n---\n\n".join(by_id[c]["text"] for c in top)

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}

def _eval(n):
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return n.value
    if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
        return _OPS[type(n.op)](_eval(n.left), _eval(n.right))
    if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
        return _OPS[type(n.op)](_eval(n.operand))
    raise ValueError("unsupported expression")

@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression with + - * / ** and parentheses, e.g. '(215938 - 130497) / 130497 * 100'.
    Use it for every calculation instead of computing in your head."""
    try:
        return str(round(_eval(ast.parse(expression.replace(",", ""), mode="eval").body), 6))
    except Exception as e:
        return f"Error: {e}"
    
import uuid

@tool
def analyze_data(code: str) -> str:
    """Run short Python code for tabular analysis: growth rates, margins and comparisons across companies or years.
    pandas (pd), numpy (np) and math are already imported. Build a small DataFrame from figures copied EXACTLY
    from the retrieved passages, compute what is needed, and print() the result. Only printed output is returned.
    For a single arithmetic expression use the calculator tool instead."""
    prelude = "import math\nimport numpy as np\nimport pandas as pd\npd.set_option('display.width', 200)\n"
    name = "rag-sbx-" + uuid.uuid4().hex[:8]
    cmd = ["docker", "run", "--rm", "-i", "--name", name, "--network", "none", "--read-only",
           "--memory", "256m", "--cpus", "0.5", "--pids-limit", "64", "--cap-drop", "ALL",
           "--security-opt", "no-new-privileges", "--tmpfs", "/tmp:size=16m",
           "rag-sandbox", "python", "-"]
    try:
        r = subprocess.run(cmd, input=prelude + code, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True)
        return "Error: timeout."
    except FileNotFoundError:
        return "Error: Docker is not installed or not on PATH, so analyze_data is disabled."
    if r.returncode == 137:
        return "Error: killed (out of memory)."
    out = (r.stdout + r.stderr).strip()
    return out[:3000] if out else "Error: no output. Use print()."

@tool
def summarize_topic(topic: str, company: str = "") -> str:
    """Summarize what a company's 10-K says about a topic (e.g. risk factors, AI strategy, competition, manufacturing).
    Use it for descriptive questions, not for exact figures. Set company to NVIDIA, AMD or INTEL; leave empty to cover all three.
    Returns a short bullet summary with (COMPANY, PDF page N) citations."""
    from langchain_groq import ChatGroq
    passages = search_knowledge_base.invoke({"query": topic, "company": company})
    llm = ChatGroq(model=os.getenv("SUMMARY_MODEL", "openai/gpt-oss-20b"), temperature=0, timeout=60, max_retries=6)
    prompt = ("Summarize in 4 to 6 bullet points what the passages below say about: " + topic +
              ". Copy every number together with its year or period exactly as written "
              "(for example '43% in 2025, 45% in 2024'); never merge numbers into ranges. "
              "Use only the passages. After each bullet cite (COMPANY, PDF page N) taken from the passage header. "
              "If the passages do not cover the topic, say so plainly. Do not add outside knowledge.\n\n" + passages)
    return llm.invoke(prompt).text