# 💰 Statement-to-Insights Finance Agent

Agentic AI assignment (Assignment 4). Upload a bank/UPI statement (CSV/PDF); a team of agents parses it,
categorises transactions, computes exact numbers with pandas + a calculator tool, flags overspending and
suggests savings. A chat advisor answers questions using intent routing and Agentic RAG over ChromaDB.

## Agents
| Agent | Job | Where |
|---|---|---|
| Parser | CSV/PDF → structured transactions | `src/parser.py` |
| Categorizer | rules + few-shot LLM for unknown merchants | `src/categorizer.py` |
| Analysis | totals, trends, alerts via pandas + safe calculator | `src/analysis.py` |
| Advisor (+Reviewer) | insights, savings tips, reflection check | `src/agents.py` |

## Mandatory criteria → where demonstrated
1. **LLM integration** – Ollama (local) / Anthropic / OpenAI: `src/llm.py`
2. **Interactive GUI** – Streamlit: `app.py`
3. **Prompt engineering** – few-shot (router, categorizer), CoT (advisor), ReAct-style retrieval planner, reflection (reviewer). All system prompts visible in the app's *Prompts & Architecture* tab and in `src/agents.py`
4. **LangChain core** – `ChatPromptTemplate` + LCEL `prompt | llm | StrOutputParser()` in every chain
5. **Agentic RAG** – `decide_retrieval()` lets the agent choose NONE / TRANSACTIONS / GUIDANCE / BOTH before retrieving
6. **Intent routing** – `route_intent()` → `transaction_lookup | spending_analysis | savings_guidance`
7. **Vector DB** – ChromaDB (`chroma_db/`), collections `transactions` and `budget_guidance`: `src/vectorstore.py`

## Setup
```bash
git clone <your-repo-url> && cd finance-agent
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```
Pick ONE LLM:
- **Local (free):** install Ollama, then `ollama pull llama3.2`
- **API:** copy `.env.example` to `.env` and add `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`

## Run
```bash
streamlit run app.py
```
Click **Use sample statement** → **Process statement**, then explore the tabs.

## CSV format
Columns: `Date, Description, Debit, Credit` (or `Date, Narration, Amount, Type`). See `sample_data/sample_statement.csv`.

## Project structure
```
app.py                 Streamlit UI
src/                   llm, parser, categorizer, analysis, vectorstore, agents
knowledge/             budgeting guidance indexed into ChromaDB
sample_data/           demo statement
```
