"""
Day 23 - Step 5.1: Dense Retrieval

Flow:

    User question
          ↓
    BGE embedding
          ↓
    FAISS similarity search
          ↓
    Top-K child chunks

Input:
    data/vector_databases/faiss/index.faiss
    data/vector_databases/faiss/metadata.json

Model:
    BAAI/bge-large-en-v1.5
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FAISS_DIR = (
    PROJECT_ROOT
    / "data"
    / "vector_databases"
    / "faiss"
)

INDEX_FILE = FAISS_DIR / "index.faiss"
METADATA_FILE = FAISS_DIR / "metadata.json"


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "BAAI/bge-large-en-v1.5"


# ============================================================
# DENSE RETRIEVER
# ============================================================

class DenseRetriever:

    def __init__(self, top_k=5):

        self.top_k = top_k

        print("[dense] Loading FAISS index...")

        self.index = faiss.read_index(
            str(INDEX_FILE)
        )

        print(
            f"[dense] FAISS vectors: {self.index.ntotal}"
        )

        print("[dense] Loading metadata...")

        data = json.loads(
            METADATA_FILE.read_text(
                encoding="utf-8"
            )
        )

        self.metadata = data["metadata"]

        print(
            f"[dense] Metadata records: {len(self.metadata)}"
        )

        print(
            f"[dense] Loading embedding model: {MODEL_NAME}"
        )

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        print("[dense] Ready.")

    # ========================================================
    # SEARCH
    # ========================================================

    def search(self, query: str, top_k=None):

        if not query.strip():
            return []

        if top_k is None:
            top_k = self.top_k

        # ----------------------------------------------------
        # Convert query to vector
        # ----------------------------------------------------

        query_vector = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        query_vector = np.asarray(
            query_vector,
            dtype="float32"
        )

        # ----------------------------------------------------
        # Search FAISS
        # ----------------------------------------------------

        scores, positions = self.index.search(
            query_vector,
            top_k
        )

        results = []

        # ----------------------------------------------------
        # Build retrieval results
        # ----------------------------------------------------

        for score, position in zip(
            scores[0],
            positions[0]
        ):

            if position < 0:
                continue

            metadata = self.metadata[position]

            results.append(
                {
                    "rank": len(results) + 1,
                    "score": float(score),
                    "faiss_position": int(position),
                    "child_id": metadata["child_id"],
                    "parent_id": metadata["parent_id"],
                    "element_type": metadata["element_type"],
                    "page": metadata["page"],
                    "section": metadata["section"],
                    "content": metadata["content"],
                    "embedding_text": metadata["embedding_text"],
                }
            )

        return results

    # ========================================================
    # ADD CHUNKS (live ingestion — used by /api/rag/ingest)
    # ========================================================

    def add_chunks(self, new_chunks: list[dict], persist: bool = True) -> int:
        """
        Embeds new child chunks with the SAME model used to build the
        original index, appends the vectors to the live FAISS index,
        and appends matching metadata rows so FAISS position N always
        equals self.metadata[N] -- exactly the same invariant the
        original offline build (04_vector_databases/01_faiss_vector_database.py)
        relies on.

        IndexFlatIP supports .add() at any time (no training/quantization
        step required), so this is safe to call on a running index.

        Returns the number of chunks added.
        """

        if not new_chunks:
            return 0

        print(f"[dense] Embedding {len(new_chunks)} new chunk(s)...")

        texts = [chunk["embedding_text"] for chunk in new_chunks]

        vectors = self.model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
        )

        vectors = np.asarray(vectors, dtype="float32")

        start_position = self.index.ntotal

        self.index.add(vectors)

        for offset, chunk in enumerate(new_chunks):
            self.metadata.append(
                {
                    "faiss_position": start_position + offset,
                    "child_id": chunk["child_id"],
                    "parent_id": chunk["parent_id"],
                    "element_type": chunk["element_type"],
                    "page": chunk["page"],
                    "section": chunk["section"],
                    "content": chunk["content"],
                    "embedding_text": chunk["embedding_text"],
                }
            )

        print(
            f"[dense] Added {len(new_chunks)} vectors. "
            f"Total now: {self.index.ntotal}"
        )

        if persist:
            self._persist()

        return len(new_chunks)

    def _persist(self):
        """
        Writes the live index + metadata back to disk, in the exact
        same shape 04_vector_databases/01_faiss_vector_database.py
        produces, so the offline scripts and a server restart both
        keep working against the enlarged index.
        """
        FAISS_DIR.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(INDEX_FILE))

        METADATA_FILE.write_text(
            json.dumps(
                {
                    "model": MODEL_NAME,
                    "embedding_dimension": self.index.d,
                    "total_vectors": self.index.ntotal,
                    "metadata": self.metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"[dense] Persisted index.faiss + metadata.json "
            f"({self.index.ntotal} vectors)"
        )


# ============================================================
# TEST
# ============================================================

def main():

    retriever = DenseRetriever(
        top_k=5
    )

    query = input(
        "\nEnter your question: "
    ).strip()

    results = retriever.search(
        query
    )

    print(
        f"\n[dense] Results for: {query}"
    )

    print("=" * 70)

    for result in results:

        print(
            f"\nRank       : {result['rank']}"
        )

        print(
            f"Score      : {result['score']:.4f}"
        )

        print(
            f"Child ID   : {result['child_id']}"
        )

        print(
            f"Parent ID  : {result['parent_id']}"
        )

        print(
            f"Type       : {result['element_type']}"
        )

        print(
            f"Page       : {result['page']}"
        )

        print(
            f"Section    : {result['section']}"
        )

        print(
            f"Content    : {result['content'][:500]}"
        )

        print("-" * 70)


if __name__ == "__main__":
    main()