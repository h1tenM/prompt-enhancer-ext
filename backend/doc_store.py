"""Manages the RAG corpus: which research docs exist, which are switched on,
and how their chunks map into the Chroma collection.

The manifest (docs.json) is the source of truth for names/enabled flags and for
the chunk ids each doc owns. Keeping the ids here means we can delete a doc via
the public Chroma API instead of reaching into the collection's where-filters.
"""

import json
import os
import uuid
from datetime import datetime, timezone

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# L2 distance cutoff for "actually relevant" (lower = closer).
# Calibrated against the bundled corpus: on-topic queries scored 0.83-1.74,
# off-topic ones 1.37-1.65. The distributions overlap, because all-MiniLM-L6-v2
# discriminates poorly over docs this short, so 1.30 is set to favour rejecting
# irrelevant context over catching every relevant hit. Recalibrate if you swap
# the embedding model or add substantially longer docs.
MAX_DISTANCE = 1.30

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)


class DocStore:
    def __init__(self, vectorstore, manifest_path, seed_dir=None):
        self.vectorstore = vectorstore
        self.manifest_path = manifest_path
        self.seed_dir = seed_dir
        self.docs = self._load_manifest()

        if not self.docs and seed_dir and os.path.isdir(seed_dir):
            self._seed_from_dir(seed_dir)

    # --- persistence ---------------------------------------------------

    def _load_manifest(self):
        if not os.path.exists(self.manifest_path):
            return []
        with open(self.manifest_path) as f:
            return json.load(f).get("docs", [])

    def _save_manifest(self):
        with open(self.manifest_path, "w") as f:
            json.dump({"docs": self.docs}, f, indent=2)

    def _seed_from_dir(self, seed_dir):
        """First run: pull the bundled research_docs in as built-in sources."""
        for filename in sorted(os.listdir(seed_dir)):
            if not filename.endswith(".txt"):
                continue
            with open(os.path.join(seed_dir, filename)) as f:
                text = f.read()
            if text.strip():
                self.add(
                    name=filename[:-4].replace("_", " ").title(),
                    text=text,
                    builtin=True,
                )

    # --- queries -------------------------------------------------------

    def list(self):
        return [
            {
                "id": d["id"],
                "name": d["name"],
                "enabled": d["enabled"],
                "builtin": d.get("builtin", False),
                "chunks": len(d["chunk_ids"]),
                "chars": d["chars"],
                "added_at": d["added_at"],
            }
            for d in self.docs
        ]

    def enabled_ids(self):
        return [d["id"] for d in self.docs if d["enabled"]]

    def _find(self, doc_id):
        return next((d for d in self.docs if d["id"] == doc_id), None)

    def search(self, query, k=2, sources=None, max_distance=MAX_DISTANCE):
        """Retrieve chunks, restricted to the enabled docs (or an explicit subset).

        Returns [] when nothing is enabled — an unfiltered search would silently
        pull from docs the user just switched off.

        Results beyond max_distance are dropped. Chroma returns the k nearest
        chunks whether or not they are actually relevant, and injecting an
        unrelated doc measurably degrades the enhanced prompt: pre-threshold, a
        "write a professional email" request retrieved the academic-writing doc
        and produced an email with Abstract/Methodology/Results headings. No
        context beats wrong context.
        """
        allowed = self.enabled_ids()
        if sources is not None:
            allowed = [s for s in allowed if s in sources]
        if not allowed:
            return []

        # Chroma rejects $in with a single-element list on some versions.
        where = (
            {"source": allowed[0]}
            if len(allowed) == 1
            else {"source": {"$in": allowed}}
        )
        scored = self.vectorstore.similarity_search_with_score(query, k=k, filter=where)
        return [doc for doc, distance in scored if distance <= max_distance]

    # --- mutations -----------------------------------------------------

    def add(self, name, text, builtin=False):
        text = text.strip()
        if not text:
            raise ValueError("Document is empty.")

        doc_id = str(uuid.uuid4())
        chunks = _splitter.split_text(text)
        if not chunks:
            raise ValueError("Document produced no chunks.")

        chunk_ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
        self.vectorstore.add_texts(
            texts=chunks,
            metadatas=[{"source": doc_id, "name": name} for _ in chunks],
            ids=chunk_ids,
        )

        entry = {
            "id": doc_id,
            "name": name,
            "enabled": True,
            "builtin": builtin,
            "chunk_ids": chunk_ids,
            "chars": len(text),
            "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.docs.append(entry)
        self._save_manifest()
        return entry

    def set_enabled(self, doc_id, enabled):
        doc = self._find(doc_id)
        if not doc:
            return None
        doc["enabled"] = bool(enabled)
        self._save_manifest()
        return doc

    def delete(self, doc_id):
        doc = self._find(doc_id)
        if not doc:
            return False
        if doc["chunk_ids"]:
            self.vectorstore.delete(ids=doc["chunk_ids"])
        self.docs = [d for d in self.docs if d["id"] != doc_id]
        self._save_manifest()
        return True
