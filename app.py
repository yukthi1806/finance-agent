"""Statement-to-Insights Finance Agent - Streamlit GUI (Criterion 2)."""
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.llm import PROVIDERS, DEFAULT_MODELS, get_llm
from src.parser import parse_statement
from src.categorizer import categorize
from src.analysis import compute_summary
from src.vectorstore import get_client, index_guidance, index_transactions
from src.agents import run_agent, insights_report, PROMPTS

load_dotenv()
st.set_page_config(page_title="Statement-to-Insights Finance Agent", page_icon="💰", layout="wide")
st.title("💰 Statement-to-Insights Finance Agent")
st.caption("Upload a bank/UPI statement → agents parse, categorise, analyse and advise.")

ss = st.session_state
for k, v in {"df": None, "summary": None, "tx_col": None, "guide_col": None, "chat": [], "report": None}.items():
    ss.setdefault(k, v)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ LLM settings")
    provider = st.selectbox("Provider", PROVIDERS)
    model = st.text_input("Model", DEFAULT_MODELS[provider])
    if provider == "Anthropic (API)":
        key = st.text_input("ANTHROPIC_API_KEY", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))
        if key: os.environ["ANTHROPIC_API_KEY"] = key
    elif provider == "OpenAI (API)":
        key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        if key: os.environ["OPENAI_API_KEY"] = key
    else:
        st.info("Run `ollama pull llama3.2` once, keep Ollama running.")
    use_llm_cat = st.checkbox("Use LLM for unknown merchants", value=True)

    st.header("📄 Statement")
    up = st.file_uploader("CSV or PDF", type=["csv", "pdf"])
    use_sample = st.button("Use sample statement")
    process = st.button("Process statement", type="primary")

def get_model():
    return get_llm(provider, model)

# ------------------------------------------------------------------ pipeline
if process or use_sample:
    try:
        if use_sample:
            df = parse_statement("sample_data/sample_statement.csv", "sample.csv")
        elif up is not None:
            df = parse_statement(up, up.name)
        else:
            st.sidebar.error("Upload a file or click 'Use sample statement'."); st.stop()
        with st.spinner("Categorising transactions..."):
            df = categorize(df, get_model() if use_llm_cat else None)
        with st.spinner("Indexing into ChromaDB..."):
            client = get_client()
            ss.guide_col = index_guidance(client)
            ss.tx_col = index_transactions(client, df)
        ss.df, ss.summary, ss.chat, ss.report = df, compute_summary(df), [], None
        st.sidebar.success(f"Loaded {len(df)} transactions")
    except Exception as e:
        st.sidebar.error(f"Failed: {e}")

if ss.df is None:
    st.info("👈 Upload a statement (or use the sample) and click **Process statement**.")
    st.stop()

df, s = ss.df, ss.summary
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🧠 AI Insights", "💬 Chat Advisor", "🔎 Prompts & Architecture"])

# ------------------------------------------------------------------ dashboard
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Income", f"₹{s['income']:,.0f}")
    c2.metric("Spending", f"₹{s['total_spend']:,.0f}")
    c3.metric("Invested", f"₹{s['invested']:,.0f}")
    c4.metric("Savings rate", f"{s['savings_rate']:.1f}%" if s["savings_rate"] is not None else "n/a")
    if s["alerts"]:
        st.subheader("🚨 Overspending alerts")
        for a in s["alerts"]:
            st.warning(a)
    l, r = st.columns(2)
    l.subheader("Spending by category"); l.bar_chart(s["by_category"])
    r.subheader("Monthly spending"); r.line_chart(s["monthly"])
    st.subheader("Transactions")
    cats = st.multiselect("Filter category", sorted(df.category.unique()))
    st.dataframe(df[df.category.isin(cats)] if cats else df, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ insights
with tab2:
    st.write("Advisor agent drafts a report, then a **reflection** step verifies it against the pandas facts.")
    if st.button("Generate insights report"):
        with st.spinner("Advisor + reviewer agents working..."):
            try:
                ss.report = insights_report(get_model(), s, ss.guide_col)
            except Exception as e:
                st.error(f"LLM error: {e}")
    if ss.report:
        draft, final = ss.report
        st.markdown(final)
        with st.expander("Show first draft (before reflection)"):
            st.markdown(draft)

# ------------------------------------------------------------------ chat
with tab3:
    st.write("Try: *How much did I spend on Swiggy?* · *Which category grew most last month?* · *How can I save more?*")
    for m in ss.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("trace"):
                with st.expander("Agent trace"):
                    for t in m["trace"]: st.write("•", t)
    if q := st.chat_input("Ask about your money..."):
        with st.chat_message("user"): st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    res = run_agent(get_model(), q, df, s, ss.chat, ss.tx_col, ss.guide_col)
                    st.markdown(res["answer"])
                    with st.expander("Agent trace"):
                        for t in res["trace"]: st.write("•", t)
                    ss.chat += [{"role": "user", "content": q},
                                {"role": "assistant", "content": res["answer"], "trace": res["trace"]}]
                except Exception as e:
                    st.error(f"LLM error: {e}")

# ------------------------------------------------------------------ architecture
with tab4:
    st.markdown("""
**Agents:** Parser → Categorizer → Analysis (pandas + calculator) → Advisor (+ reviewer)  
**Flow for a chat question:** Intent router → retrieval planner (agent decides) → ChromaDB / pandas → advisor answer  
**Memory:** last 6 chat turns are passed through `MessagesPlaceholder("history")`.
""")
    for name, text in PROMPTS.items():
        with st.expander(f"System prompt — {name}"):
            st.code(text, language="text")
    st.caption("User prompts: the router/advisor user prompt is the chat message itself; "
               "the report/review user prompts are fixed strings in src/agents.py.")
