"""
Day 23 - Step 4A: FAISS Vector Database

Input:
    data/embeddings/hierarchical_embeddings.json

Output:
    data/vector_databases/faiss/
        index.faiss
        metadata.json

Purpose:
    Store dense embedding vectors in FAISS so they can be
    searched efficiently during dense retrieval.
"""

import json
from pathlib import Path

import faiss
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "hierarchical_embeddings.json"
)

FAISS_DIR = (
    PROJECT_ROOT
    / "data"
    / "vector_databases"
    / "faiss"
)

INDEX_FILE = FAISS_DIR / "index.faiss"
METADATA_FILE = FAISS_DIR / "metadata.json"


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings():

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embeddings file not found:\n{EMBEDDINGS_FILE}"
        )

    print(f"[load] Reading: {EMBEDDINGS_FILE}")

    data = json.loads(
        EMBEDDINGS_FILE.read_text(
            encoding="utf-8"
        )
    )

    chunks = data["chunks"]

    print(f"[load] Embeddings: {len(chunks)}")

    return data, chunks


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_index(chunks):

    vectors = np.array(
        [
            chunk["embedding"]
            for chunk in chunks
        ],
        dtype="float32",
    )

    dimension = vectors.shape[1]

    print(
        f"[faiss] Vector count: {vectors.shape[0]}"
    )

    print(
        f"[faiss] Vector dimension: {dimension}"
    )

    # BGE embeddings were normalized during generation.
    # Inner Product therefore behaves like cosine similarity.
    index = faiss.IndexFlatIP(dimension)

    index.add(vectors)

    print(
        f"[faiss] Vectors stored: {index.ntotal}"
    )

    return index


# ============================================================
# SAVE METADATA
# ============================================================

def build_metadata(chunks):

    metadata = []

    for position, chunk in enumerate(chunks):

        metadata.append(
            {
                "faiss_position": position,
                "child_id": chunk["child_id"],
                "parent_id": chunk["parent_id"],
                "element_type": chunk["element_type"],
                "page": chunk["page"],
                "section": chunk["section"],
                "content": chunk["content"],
                "embedding_text": chunk["embedding_text"],
            }
        )

    return metadata


# ============================================================
# MAIN
# ============================================================

def main():

    data, chunks = load_embeddings()

    index = create_faiss_index(chunks)

    metadata = build_metadata(chunks)

    FAISS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    METADATA_FILE.write_text(
        json.dumps(
            {
                "model": data["model"],
                "embedding_dimension": data["embedding_dimension"],
                "total_vectors": len(chunks),
                "metadata": metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n[save] FAISS database created")

    print(
        f"  Index    : {INDEX_FILE}"
    )

    print(
        f"  Metadata : {METADATA_FILE}"
    )

    print(
        f"  Vectors  : {index.ntotal}"
    )

    print(
        f"  Dimension: {index.d}"
    )

    print("\n[done] FAISS vector database complete.")


if __name__ == "__main__":
    main()