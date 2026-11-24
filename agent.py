import time
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import search_knowledge_base, calculator, analyze_data, summarize_topic

SYSTEM = """You answer questions about the 10-K reports of NVIDIA, AMD and Intel using the tools.

Process:
- Search before answering. If the passages lack the exact figure, refine the query and search again.
- For comparisons, search each company separately.
- Use the calculator only for the numbers the question needs. Do not compute extra metrics nobody asked for.
- Use the calculator for every ratio, percentage or difference, even simple ones. Never compute in your head.
- For figures from the main financial statements, put the statement name in the query (e.g. 'Consolidated Statements of Income operating income', 'Consolidated Balance Sheets total assets').
- For tables or comparisons across several companies or years (growth, margins), build a small DataFrame in analyze_data from the exact figures you retrieved and print the result. Use calculator for single expressions.
- Balance sheets and income statements list the MOST RECENT period in the FIRST numeric column. Before copying a figure, name the column it comes from (e.g. 'Dec 27, 2025') and check it matches the period asked. Never mix columns from different dates in one calculation.
- When you build a DataFrame in analyze_data, include a 'period' column with the date or year of every figure.
- For questions about strategy, risks, competition, products or business descriptions, you MUST call summarize_topic (with the company) instead of search_knowledge_base, and base your answer on its output. Use search_knowledge_base only for exact figures and for finding specific numbers.

Periods:
- NVIDIA's fiscal year ends in January (FY2026 ended Jan 25, 2026). AMD and Intel end in December (2025 ended Dec 27, 2025).
- When comparing companies, compare the periods that overlap most in time (NVIDIA FY2026 with AMD/Intel 2025), say so in one line, and do not add other period pairings unless asked.

Answer style:
- Short and direct: the answer first, then the key figures. Plain text only, no LaTeX, no headers.
- Cite as (COMPANY, PDF page N), using the page of the passage the figure came from.
- Figures are in millions of USD unless the passage says otherwise.
- If the knowledge base does not contain the answer, say so plainly. Never invent numbers and never answer from memory.
- When ranking or comparing, state the figure for every company you compared."""

def make_llm(name):
    if name.startswith("gemini"):
        return ChatGoogleGenerativeAI(model=name, timeout=60, max_retries=2)
    return ChatGroq(model=name, temperature=0, timeout=60, max_retries=6)

def ask(question, model="openai/gpt-oss-20b"):
    agent = create_agent(make_llm(model), tools=[search_knowledge_base, calculator, analyze_data, summarize_topic], system_prompt=SYSTEM)
    t0 = time.time()
    r = agent.invoke({"messages": [{"role": "user", "content": question}]})
    calls = [(c["name"], c["args"]) for m in r["messages"] for c in (getattr(m, "tool_calls", None) or [])]
    return {"model": model, "answer": r["messages"][-1].text, "calls": calls, "seconds": round(time.time() - t0, 1)}

if __name__ == "__main__":
    out = ask("Which of the three companies had the highest revenue in its latest fiscal year, and how many times larger is it than AMD's?")
    print("AGENT.PY |", out["model"], "|", out["seconds"], "s |", len(out["calls"]), "tool calls")
    for name, args in out["calls"]:
        print("  CALL", name, args)
    print("\n" + out["answer"])