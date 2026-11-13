"""Categorizer Agent: rules first, LLM (few-shot, LCEL) for unknown merchants."""
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

CATEGORIES = ["Food & Dining", "Groceries", "Transport", "Shopping", "Bills & Utilities",
              "Entertainment", "Health", "Education", "Rent & Housing",
              "Investments & Savings", "Cash & Transfers", "Other"]

RULES = {
    "Food & Dining": ["swiggy", "zomato", "restaurant", "cafe", "dominos", "mcdonald", "kfc", "starbucks", "bakery", "biryani", "pizza"],
    "Groceries": ["bigbasket", "blinkit", "zepto", "dmart", "grocery", "instamart", "more retail", "reliance fresh"],
    "Transport": ["uber", "ola", "rapido", "metro", "irctc", "redbus", "petrol", "fuel", "indian oil", "hpcl", "bpcl", "fastag"],
    "Shopping": ["amazon", "flipkart", "myntra", "ajio", "nykaa", "meesho", "decathlon"],
    "Bills & Utilities": ["electricity", "bescom", "airtel", "jio", "vodafone", "broadband", "recharge", "fibernet", "water bill"],
    "Entertainment": ["netflix", "bookmyshow", "spotify", "hotstar", "prime video", "pvr", "inox", "youtube"],
    "Health": ["pharmacy", "apollo", "medplus", "hospital", "clinic", "practo", "1mg"],
    "Education": ["udemy", "coursera", "college", "tuition", "unacademy", "byju"],
    "Rent & Housing": ["rent", "maintenance", "society"],
    "Investments & Savings": ["sip", "mutual fund", "zerodha", "groww", "ppf", "lic "],
    "Cash & Transfers": ["atm", "cash withdrawal"],
}

CAT_SYSTEM = """You are a transaction categorizer for Indian bank/UPI statements.
Allowed categories: {cats}.
Examples (few-shot):
1|SWIGGY ORDER 8821 -> Food & Dining
2|UPI/RAMESH KIRANA STORE -> Groceries
3|BESCOM BILL PAY -> Bills & Utilities
For every numbered line the user sends, answer with exactly one line 'number|category'.
Use 'Other' if unsure. No explanations."""
CAT_PROMPT = ChatPromptTemplate.from_messages([("system", CAT_SYSTEM), ("user", "{lines}")])


def rule_category(desc: str, direction: str) -> str:
    if direction == "credit":
        return "Income"
    d = desc.lower()
    for cat, kws in RULES.items():
        if any(k in d for k in kws):
            return cat
    return "Other"


def categorize(df: pd.DataFrame, llm=None) -> pd.DataFrame:
    df = df.copy()
    df["category"] = [rule_category(d, r) for d, r in zip(df["description"], df["direction"])]
    if llm is not None:
        unknown = df.loc[(df["category"] == "Other"), "description"].unique()[:40]
        if len(unknown):
            chain = CAT_PROMPT | llm | StrOutputParser()           # LCEL
            lines = "\n".join(f"{i}|{d}" for i, d in enumerate(unknown))
            try:
                reply = chain.invoke({"cats": ", ".join(CATEGORIES), "lines": lines})
                mapping = {}
                for ln in reply.splitlines():
                    if "|" in ln:
                        i, c = ln.split("|", 1)
                        c = c.strip()
                        if i.strip().isdigit() and int(i) < len(unknown) and c in CATEGORIES:
                            mapping[unknown[int(i)]] = c
                m = df["description"].isin(mapping)
                df.loc[m, "category"] = df.loc[m, "description"].map(mapping)
            except Exception:
                pass  # keep rule-based result if LLM unavailable
    return df
