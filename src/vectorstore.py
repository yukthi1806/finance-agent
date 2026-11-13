"""Criterion 7 - Vector DB: ChromaDB with similarity search.
Two collections: budget_guidance (static knowledge) and transactions (the user's statement)."""
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

_EF = embedding_functions.DefaultEmbeddingFunction()  # all-MiniLM-L6-v2 (ONNX), downloads once


def get_client():
    return chromadb.PersistentClient(path="chroma_db")


def index_guidance(client):
    col = client.get_or_create_collection("budget_guidance", embedding_function=_EF)
    if col.count() == 0:
        ids, docs, metas = [], [], []
        for f in Path("knowledge").glob("*.md"):
            for i, para in enumerate(p.strip() for p in f.read_text(encoding="utf-8").split("\n\n")):
                if len(para) > 40:
                    ids.append(f"{f.stem}-{i}"); docs.append(para); metas.append({"source": f.name})
        if ids:
            col.add(ids=ids, documents=docs, metadatas=metas)
    return col


def index_transactions(client, df):
    try:
        client.delete_collection("transactions")
    except Exception:
        pass
    col = client.create_collection("transactions", embedding_function=_EF)
    docs = [f"{r.date:%Y-%m-%d} | {r.description} | {r.category} | Rs {r.amount:.0f} | {r.direction}" for r in df.itertuples()]
    for i in range(0, len(docs), 500):
        batch = docs[i:i + 500]
        col.add(ids=[f"t{j}" for j in range(i, i + len(batch))], documents=batch)
    return col


def query(col, text: str, k: int = 5):
    if col is None or col.count() == 0:
        return []
    return col.query(query_texts=[text], n_results=min(k, col.count()))["documents"][0]
