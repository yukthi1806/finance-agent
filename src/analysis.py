"""Analysis Agent tools: pandas + safe calculator. All numbers come from here, never from the LLM."""
import ast
import operator as op
import pandas as pd

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}


def safe_calc(expr: str) -> float:
    """Calculator tool: evaluates arithmetic safely (no eval)."""
    def ev(n):
        if isinstance(n, ast.Constant): return n.value
        if isinstance(n, ast.BinOp): return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp): return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("Unsupported expression")
    return ev(ast.parse(expr, mode="eval").body)


# Guideline caps as a share of monthly income (used for overspending alerts)
CAPS = {"Food & Dining": 0.12, "Shopping": 0.10, "Entertainment": 0.06, "Transport": 0.10, "Groceries": 0.15}
SAVING_CATS = {"Investments & Savings"}


def compute_summary(df: pd.DataFrame) -> dict:
    debit = df[df.direction == "debit"]
    spend = debit[~debit.category.isin(SAVING_CATS)].copy()
    invest = debit[debit.category.isin(SAVING_CATS)].amount.sum()
    income = df[df.direction == "credit"].amount.sum()
    total_spend = spend.amount.sum()
    months = max(df.date.dt.to_period("M").nunique(), 1)

    by_cat = spend.groupby("category").amount.sum().sort_values(ascending=False)
    spend["month"] = spend.date.dt.to_period("M").astype(str)
    monthly = spend.groupby("month").amount.sum()
    cat_month = spend.pivot_table(index="month", columns="category", values="amount", aggfunc="sum", fill_value=0)
    top_merchants = spend.groupby("description").amount.agg(["sum", "count"]).sort_values("sum", ascending=False).head(5)

    savings_rate = safe_calc(f"({income} - {total_spend}) / {income} * 100") if income > 0 else None

    alerts = []
    base = income / months if income > 0 else total_spend / months
    for cat, cap in CAPS.items():
        if cat in by_cat.index and base > 0:
            share = by_cat[cat] / months / base
            if share > cap:
                alerts.append(f"{cat}: {share*100:.0f}% of monthly {'income' if income>0 else 'spend'} (guideline {cap*100:.0f}%)")
    if len(cat_month) >= 2:
        last, prev = cat_month.iloc[-1], cat_month.iloc[-2]
        for cat in cat_month.columns:
            if prev[cat] > 0 and last[cat] > prev[cat] * 1.3 and last[cat] - prev[cat] > 1000:
                alerts.append(f"{cat} jumped {((last[cat]-prev[cat])/prev[cat])*100:.0f}% in {cat_month.index[-1]} vs {cat_month.index[-2]}")
    if savings_rate is not None and savings_rate < 20:
        alerts.append(f"Savings rate is only {savings_rate:.0f}% (target 20%+)")

    return dict(total_spend=total_spend, income=income, invested=invest, savings_rate=savings_rate,
                months=months, by_category=by_cat, monthly=monthly, top_merchants=top_merchants, alerts=alerts)


def summary_text(s: dict) -> str:
    lines = [f"Months covered: {s['months']}",
             f"Total income: Rs {s['income']:.0f}",
             f"Total spending (excl. investments): Rs {s['total_spend']:.0f}",
             f"Invested/saved: Rs {s['invested']:.0f}",
             f"Savings rate: {s['savings_rate']:.1f}%" if s["savings_rate"] is not None else "Savings rate: n/a",
             "Spending by category:"]
    lines += [f"  - {c}: Rs {v:.0f}" for c, v in s["by_category"].items()]
    lines.append("Monthly spending:")
    lines += [f"  - {m}: Rs {v:.0f}" for m, v in s["monthly"].items()]
    lines.append("Top merchants:")
    lines += [f"  - {d}: Rs {r['sum']:.0f} ({int(r['count'])} txns)" for d, r in s["top_merchants"].iterrows()]
    lines.append("Overspending alerts: " + ("; ".join(s["alerts"]) if s["alerts"] else "none"))
    return "\n".join(lines)


def exact_lookup(df: pd.DataFrame, question: str) -> str:
    """Pandas keyword filter for exact totals on lookup questions."""
    stop = {"how", "much", "what", "did", "spend", "spent", "show", "all", "the", "for", "on", "my", "in", "was", "were",
            "transactions", "payments", "paid", "total", "many", "times", "from", "with", "last", "month", "this"}
    words = [w.strip("?.,!").lower() for w in question.split()]
    words = [w for w in words if len(w) > 2 and w not in stop]
    if not words:
        return ""
    pattern = "|".join(words)
    m = df[df.description.str.lower().str.contains(pattern, regex=True) | df.category.str.lower().str.contains(pattern, regex=True)]
    if m.empty:
        return ""
    tot = m[m.direction == "debit"].amount.sum()
    rows = "\n".join(f"{r.date:%Y-%m-%d} | {r.description} | Rs {r.amount:.0f} | {r.direction}" for r in m.head(15).itertuples())
    return f"Exact pandas match for {words}: {len(m)} rows, total debits Rs {tot:.0f}\n{rows}"
