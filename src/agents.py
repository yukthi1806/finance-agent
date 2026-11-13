"""Agents, prompts, intent routing and Agentic RAG (Criteria 3, 4, 5, 6).

Every chain is LCEL:  ChatPromptTemplate | llm | StrOutputParser()
Prompt techniques used:
  * Few-shot prompting ........ ROUTER_SYSTEM, CAT_SYSTEM (categorizer.py)
  * Zero-shot + Chain of Thought  ADVISOR_SYSTEM (reason privately, answer concisely)
  * ReAct-style decision ...... RETRIEVAL_SYSTEM (agent decides WHEN to retrieve, then observes context)
  * Reflection ................ REVIEW_SYSTEM (checks the draft against FACTS and fixes it)
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from .analysis import summary_text, exact_lookup
from .vectorstore import query

# ------------------------------------------------------------------ prompts
ROUTER_SYSTEM = """You are the intent router of a personal-finance assistant.
Classify the user's message into EXACTLY one label:
transaction_lookup  - asks about specific transactions, merchants, or "how much did I spend on X"
spending_analysis   - asks for totals, trends, comparisons, category breakdowns, month-over-month
savings_guidance    - asks for advice, budgeting tips, how to save or cut spending

Examples (few-shot):
User: How much did I spend on Swiggy? -> transaction_lookup
User: Show my Amazon payments -> transaction_lookup
User: Which category grew the most last month? -> spending_analysis
User: Compare my spending in March and April -> spending_analysis
User: How can I save more money? -> savings_guidance
User: Am I overspending on food and what should I do? -> savings_guidance

Reply with ONLY the label."""
ROUTER_PROMPT = ChatPromptTemplate.from_messages([("system", ROUTER_SYSTEM), ("user", "{question}")])

RETRIEVAL_SYSTEM = """You are the planning step of an agent (ReAct style).
Thought: decide what extra information is needed to answer the user's question.
Action: reply with ONE word only:
NONE          - the FACTS summary already contains the answer
TRANSACTIONS  - need to look up specific transactions in the statement index
GUIDANCE      - need budgeting/saving guidance from the knowledge base
BOTH          - need both"""
RETRIEVAL_PROMPT = ChatPromptTemplate.from_messages(
    [("system", RETRIEVAL_SYSTEM), ("user", "Intent: {intent}\nQuestion: {question}")])

ADVISOR_SYSTEM = """You are a friendly personal-finance advisor for an Indian user. Amounts are in rupees (Rs).
Rules:
1. Think step by step privately (Chain of Thought) but output only the final answer.
2. Use ONLY the numbers in FACTS and RETRIEVED CONTEXT. Never invent or recompute figures.
3. Plain language, short paragraphs or bullets, specific and practical suggestions.
4. If the data cannot answer the question, say so.

FACTS (computed exactly with pandas):
{facts}

RETRIEVED CONTEXT:
{retrieved}"""
ADVISOR_PROMPT = ChatPromptTemplate.from_messages(
    [("system", ADVISOR_SYSTEM), MessagesPlaceholder("history"), ("user", "{question}")])

REPORT_SYSTEM = """You are a personal-finance advisor. Write a short insights report with these sections:
1. Where your money went (3 bullets)
2. Overspending alerts (use the alerts in FACTS; if none say 'No major alerts')
3. 4 practical savings actions, each with an estimated monthly saving in Rs based only on FACTS.
Use plain language. Use only numbers in FACTS.

FACTS:
{facts}

BUDGETING GUIDANCE:
{retrieved}"""
REPORT_PROMPT = ChatPromptTemplate.from_messages([("system", REPORT_SYSTEM), ("user", "Generate my insights report.")])

REVIEW_SYSTEM = """You are a strict reviewer (reflection step).
Compare the DRAFT with FACTS. Fix any number, claim or suggestion that is not supported by FACTS.
Keep the same structure and tone. Return ONLY the corrected final report.

FACTS:
{facts}

DRAFT:
{draft}"""
REVIEW_PROMPT = ChatPromptTemplate.from_messages([("system", REVIEW_SYSTEM), ("user", "Review and return the final report.")])

PROMPTS = {
    "Router (few-shot)": ROUTER_SYSTEM,
    "Retrieval planner (ReAct-style)": RETRIEVAL_SYSTEM,
    "Advisor (CoT, system prompt)": ADVISOR_SYSTEM,
    "Report (system prompt)": REPORT_SYSTEM,
    "Reviewer (reflection)": REVIEW_SYSTEM,
}

# ------------------------------------------------------------------ chains
def _chain(prompt, llm):
    return prompt | llm | StrOutputParser()   # LCEL: prompt -> model -> output parser


INTENTS = ["transaction_lookup", "spending_analysis", "savings_guidance"]
DEFAULT_PLAN = {"transaction_lookup": "TRANSACTIONS", "spending_analysis": "NONE", "savings_guidance": "GUIDANCE"}


def route_intent(llm, question: str) -> str:
    raw = _chain(ROUTER_PROMPT, llm).invoke({"question": question}).lower()
    for label in INTENTS:
        if label in raw:
            return label
    return "spending_analysis"


def decide_retrieval(llm, question: str, intent: str) -> str:
    raw = _chain(RETRIEVAL_PROMPT, llm).invoke({"question": question, "intent": intent}).upper()
    for opt in ("BOTH", "TRANSACTIONS", "GUIDANCE", "NONE"):
        if opt in raw:
            return opt
    return DEFAULT_PLAN[intent]


def to_messages(chat, limit=6):
    """Conversation memory: last N turns -> LangChain messages."""
    msgs = []
    for m in chat[-limit:]:
        msgs.append(HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]))
    return msgs


def run_agent(llm, question, df, summary, chat, tx_col, guide_col):
    trace = []
    intent = route_intent(llm, question)
    trace.append(f"Intent router -> {intent}")

    plan = decide_retrieval(llm, question, intent)
    trace.append(f"Agent decided retrieval -> {plan}")

    parts = []
    if plan in ("TRANSACTIONS", "BOTH") or intent == "transaction_lookup":
        exact = exact_lookup(df, question)
        if exact:
            parts.append(exact)
        hits = query(tx_col, question, 8)
        if hits:
            parts.append("Similar transactions (ChromaDB):\n" + "\n".join(hits))
        trace.append(f"Retrieved {len(hits)} transactions from ChromaDB" + (" + exact pandas match" if exact else ""))
    if plan in ("GUIDANCE", "BOTH"):
        docs = query(guide_col, question, 3)
        parts.append("Budgeting guidance (ChromaDB):\n" + "\n".join(docs))
        trace.append(f"Retrieved {len(docs)} guidance passages from ChromaDB")

    answer = _chain(ADVISOR_PROMPT, llm).invoke({
        "facts": summary_text(summary),
        "retrieved": "\n\n".join(parts) or "(none)",
        "history": to_messages(chat),
        "question": question,
    })
    return dict(intent=intent, plan=plan, answer=answer, trace=trace)


def insights_report(llm, summary, guide_col):
    docs = query(guide_col, "reduce overspending and increase savings", 4)
    facts = summary_text(summary)
    retrieved = "\n".join(docs) or "(none)"
    draft = _chain(REPORT_PROMPT, llm).invoke({"facts": facts, "retrieved": retrieved})
    final = _chain(REVIEW_PROMPT, llm).invoke({"facts": facts, "draft": draft})   # reflection
    return draft, final
