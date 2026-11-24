import streamlit as st
from agent import ask

st.set_page_config(page_title="Agentic RAG - 10-K Assistant", page_icon="📊")
st.markdown("""
<style>
.stApp {
  background:
    radial-gradient(ellipse at 100% 0%, #1E4A5A 0%, rgba(30,74,90,0) 55%),
    radial-gradient(ellipse at 100% 100%, #6B7480 0%, rgba(107,116,128,0) 45%),
    linear-gradient(135deg, #171C26 0%, #1F2D45 55%, #26384F 100%);
  background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: rgba(23,28,38,0.85); border-right: 1px solid rgba(242,240,237,0.08); }
[data-testid="stChatMessage"] { background: rgba(242,240,237,0.05); border: 1px solid rgba(242,240,237,0.08); border-radius: 14px; }
h1 { letter-spacing: -0.5px; font-weight: 700; }
.stButton > button { border-radius: 10px; border: 1px solid rgba(242,240,237,0.15); background: rgba(242,240,237,0.04); color: #F2F0ED; text-align: left; }
.stButton > button:hover { border-color: #F26207; color: #F26207; }
[data-testid="stBottom"] > div { background: transparent; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(14,19,29,0.97) 0%, rgba(24,38,58,0.93) 100%); }
.side-title { font-size: 1.1rem; font-weight: 700; letter-spacing: 0.2px; padding-top: 6px; }
.side-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 1.4px; color: rgba(242,240,237,0.5); margin: 20px 0 6px; }
.side-note { font-size: 0.78rem; line-height: 1.55; color: rgba(242,240,237,0.65); background: rgba(242,240,237,0.04); border: 1px solid rgba(242,240,237,0.08); border-left: 3px solid #F26207; border-radius: 10px; padding: 10px 12px; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background: rgba(242,240,237,0.05); border: 1px solid rgba(242,240,237,0.12); border-radius: 10px; }
[data-testid="stSidebar"] .stButton > button { padding: 8px 12px; font-size: 0.86rem; border-left: 3px solid transparent; }
[data-testid="stSidebar"] .stButton > button:hover { border-left-color: #F26207; background: rgba(242,96,7,0.08); }
</style>
""", unsafe_allow_html=True)
st.title("Agentic RAG: 10-K Assistant")
st.caption("Answers come from the 10-K reports of NVIDIA (FY2026), AMD and Intel (2025). Figures in millions of USD.")

MODELS = {
    "Groq · gpt-oss-20b": "openai/gpt-oss-20b",
    "Gemini · 3.5-flash": "gemini-3.5-flash",
    "Gemini · 3.5-flash-lite": "gemini-3.5-flash-lite",
}
EXAMPLES = [
    ("Revenue growth: NVIDIA vs AMD", "Compare NVIDIA's and AMD's revenue growth in their latest fiscal year. Which grew faster?"),
    ("Total assets: Intel vs AMD", "Which has more total assets, Intel or AMD, and what is the difference?"),
    ("Intel's competitive risks", "Summarize the main competitive risks Intel describes in its 10-K."),
    ("Out-of-scope check: Tesla", "What was Tesla's revenue in 2025?"),
]

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown('<div class="side-title">10-K Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Model</div>', unsafe_allow_html=True)
    label = st.selectbox("Model", list(MODELS), label_visibility="collapsed")
    st.markdown('<div class="side-label">Try an example</div>', unsafe_allow_html=True)
    for title, ex in EXAMPLES:
        if st.button(title, use_container_width=True, key=title):
            st.session_state["pending"] = ex
    st.markdown('<div class="side-label">Notes</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-note">Groq is faster but often skips the summary tool. Gemini picks tools better, but its free daily quota is small. Each question is answered independently. The data-analysis tool needs Docker Desktop running.</div>', unsafe_allow_html=True)
    if st.button("Clear conversation", use_container_width=True, key="clear"):
        st.session_state.history = []
        st.rerun()
def show_calls(calls):
    for name, args in calls:
        if name == "analyze_data":
            st.markdown("**analyze_data**")
            st.code(args.get("code", ""), language="python")
        else:
            st.markdown(f"**{name}**")
            st.code(str(args), language=None)

def md(text):
    return text.replace("$", "\\$")

for m in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(md(m["q"]))
    with st.chat_message("assistant"):
        st.markdown(md(m["answer"]))
        with st.expander(f"Tools used: {len(m['calls'])} | {m['model']} | {m['seconds']}s"):
            show_calls(m["calls"])

q = st.chat_input("Ask about NVIDIA, AMD or Intel...") or st.session_state.pop("pending", None)
if q:
    with st.chat_message("user"):
        st.markdown(md(q))
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                out = ask(q, model=MODELS[label])
            except Exception as e:
                out = None
                st.error(f"{type(e).__name__}: {str(e)[:400]}")
    if out:
        st.session_state.history.append({"q": q, **out})
        st.rerun()