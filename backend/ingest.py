"""Rebuild the vector DB from scratch using the bundled research_docs/.

Docs added through the extension live in the same collection, so this wipes
everything first — run it only when you want a clean, known index.
Day-to-day, add and remove docs through the extension's Sources panel instead.
"""

import os
import shutil

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from doc_store import DocStore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "db")
SEED_DIR = os.path.join(BASE_DIR, "research_docs")


def build_vector_db():
    if os.path.exists(DB_DIR):
        print(f"Removing existing database at {DB_DIR}...")
        shutil.rmtree(DB_DIR)

    print("Reading research documents...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    store = DocStore(
        vectorstore,
        os.path.join(DB_DIR, "docs.json"),
        seed_dir=SEED_DIR,
    )

    docs = store.list()
    total_chunks = sum(d["chunks"] for d in docs)
    print(f"Created vector database with {total_chunks} chunks from {len(docs)} docs:")
    for d in docs:
        print(f"  - {d['name']} ({d['chunks']} chunks)")
    print(f"Database saved to {DB_DIR}!")


if __name__ == "__main__":
    build_vector_db()
