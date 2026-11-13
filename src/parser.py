"""Parser Agent (tool part): CSV / PDF statement -> clean transactions table.
Output columns: date, description, amount (always positive), direction ('debit'|'credit')."""
import pandas as pd


def _find(cols, keys):
    for c in cols:
        if any(k in c.lower() for k in keys):
            return c
    return None


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce").fillna(0.0)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    dcol = _find(df.columns, ["date"])
    ncol = _find(df.columns, ["desc", "narration", "particular", "remark", "detail"])
    if not dcol or not ncol:
        raise ValueError("Could not find 'date' and 'description/narration' columns in the statement.")
    debit = _find(df.columns, ["debit", "withdraw"])
    credit = _find(df.columns, ["credit", "deposit"])
    amt = _find(df.columns, ["amount"])
    typ = _find(df.columns, ["type", "dr/cr", "cr/dr"])

    out = pd.DataFrame({
        "date": pd.to_datetime(df[dcol], dayfirst=True, errors="coerce"),
        "description": df[ncol].astype(str).str.strip(),
    })
    if debit or credit:
        d = _num(df[debit]) if debit else pd.Series(0.0, index=df.index)
        c = _num(df[credit]) if credit else pd.Series(0.0, index=df.index)
        is_credit = (c > 0) & (d == 0)
        out["amount"] = d.where(~is_credit, c)
    elif amt:
        raw = _num(df[amt])
        if typ:
            is_credit = df[typ].astype(str).str.lower().str.contains("cr|credit")
        else:
            is_credit = raw > 0
        out["amount"] = raw.abs()
    else:
        raise ValueError("No amount / debit / credit column found.")
    out["direction"] = is_credit.map({True: "credit", False: "debit"})
    out = out.dropna(subset=["date"])
    out = out[out["amount"] > 0].sort_values("date").reset_index(drop=True)
    return out


def parse_statement(file, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".pdf"):
        import pdfplumber
        rows = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables():
                    rows.extend(table)
        if not rows:
            raise ValueError("No tables found in PDF. Try the CSV version of the statement.")
        header, body = rows[0], [r for r in rows[1:] if r != header]
        return _normalize(pd.DataFrame(body, columns=header))
    return _normalize(pd.read_csv(file))
