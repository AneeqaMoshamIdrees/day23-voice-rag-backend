"""
Day 23 - Step 3: Embedding Generation

Reads hierarchical child chunks and converts their embedding_text
into dense vectors using BAAI/bge-large-en-v1.5.

Input:
    data/structured_documents/hierarchical_chunks.json

Output:
    data/embeddings/hierarchical_embeddings.json
"""

import json
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "structured_documents"
    / "hierarchical_chunks.json"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings"

OUTPUT_FILE = OUTPUT_DIR / "hierarchical_embeddings.json"


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "BAAI/bge-large-en-v1.5"

BATCH_SIZE = 16


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Hierarchical chunks not found:\n{INPUT_FILE}"
        )

    print(f"[load] Reading: {INPUT_FILE}")

    data = json.loads(
        INPUT_FILE.read_text(
            encoding="utf-8"
        )
    )

    children = data["searchable_children"]

    print(
        f"[load] Searchable child chunks: {len(children)}"
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print(
        f"\n[model] Loading: {MODEL_NAME}"
    )

    start = time.time()

    model = SentenceTransformer(MODEL_NAME)

    load_time = time.time() - start

    print(
        f"[model] Loaded in {load_time:.2f}s"
    )

    # --------------------------------------------------------
    # Prepare text
    # --------------------------------------------------------

    texts = [
        child["embedding_text"]
        for child in children
    ]

    print(
        f"\n[embed] Generating embeddings for {len(texts)} chunks..."
    )

    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embedding_time = time.time() - start

    print(
        f"\n[embed] Completed in {embedding_time:.2f}s"
    )

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    embedding_dimension = len(embeddings[0])

    print(
        f"[embed] Vector dimension: {embedding_dimension}"
    )

    # --------------------------------------------------------
    # Build output
    # --------------------------------------------------------

    embedded_chunks = []

    for child, vector in zip(children, embeddings):

        embedded_chunks.append(
            {
                "child_id": child["child_id"],
                "parent_id": child["parent_id"],
                "element_type": child["element_type"],
                "page": child["page"],
                "section": child["section"],
                "content": child["content"],
                "embedding_text": child["embedding_text"],
                "embedding": vector.tolist(),
            }
        )

    output = {
        "model": MODEL_NAME,
        "embedding_dimension": embedding_dimension,
        "chunk_count": len(embedded_chunks),
        "model_load_time_seconds": round(
            load_time,
            3
        ),
        "embedding_time_seconds": round(
            embedding_time,
            3
        ),
        "chunks": embedded_chunks,
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        f"\n[save] Embeddings written to:"
    )

    print(
        f"       {OUTPUT_FILE}"
    )

    print("\n[summary]")

    print(
        f"  Model       : {MODEL_NAME}"
    )

    print(
        f"  Chunks      : {len(embedded_chunks)}"
    )

    print(
        f"  Dimensions  : {embedding_dimension}"
    )

    print(
        f"  Load time   : {load_time:.2f}s"
    )

    print(
        f"  Embed time  : {embedding_time:.2f}s"
    )

    print("\n[done] Embedding stage complete.")


if __name__ == "__main__":
    main()