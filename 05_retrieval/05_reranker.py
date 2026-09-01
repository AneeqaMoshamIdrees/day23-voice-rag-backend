"""
Day 23 - Step 5.5: Cross-Encoder Reranker

Pipeline:

    Dense Retrieval
          +
    BM25 Retrieval
          ↓
    Hybrid Retrieval
          ↓
    RRF Fusion
          ↓
    Cross-Encoder Reranker
          ↓
    Best Relevant Chunks

Purpose:
    RRF produces a strong candidate list.

    The cross-encoder then reads:

        Question + Chunk

    together and gives each candidate a relevance score.

    This improves the final ranking.
"""

from pathlib import Path
import importlib.util


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RETRIEVAL_DIR = PROJECT_ROOT / "05_retrieval"


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RRF_TOP_K = 10
FINAL_TOP_K = 5

_model = None


# ============================================================
# LOAD RRF MODULE
# ============================================================

def load_rrf_module():
    """
    Load:

        04_rrf_fusion.py

    We use importlib because the filename starts with a number.
    """

    file_path = RETRIEVAL_DIR / "04_rrf_fusion.py"

    spec = importlib.util.spec_from_file_location(
        "rrf_fusion",
        file_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load RRF module: {file_path}"
        )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


# ============================================================
# LOAD CROSS ENCODER
# ============================================================

def load_reranker():
    """
    Load the cross-encoder only once.

    This prevents loading the model repeatedly.
    """

    global _model

    if _model is None:

        from sentence_transformers import CrossEncoder

        print()
        print("[reranker] Loading cross-encoder...")
        print(f"[reranker] Model: {MODEL_NAME}")

        _model = CrossEncoder(MODEL_NAME)

        print("[reranker] Model loaded.")

    return _model


# ============================================================
# RRF CANDIDATES
# ============================================================

def get_rrf_candidates(
    question: str,
    top_k: int = RRF_TOP_K
):
    """
    Run Dense + BM25 + RRF and return RRF candidates.
    """

    rrf_module = load_rrf_module()

    RRFFusion = rrf_module.RRFFusion

    rrf = RRFFusion(
        dense_k=10,
        bm25_k=10,
        rrf_k=60,
        top_k=top_k
    )

    # Run Dense and BM25
    dense_results = rrf.dense_retriever.search(
        question,
        top_k=10
    )

    bm25_results = rrf.bm25_retriever.search(
        question,
        top_k=10
    )

    # Apply RRF
    results = rrf.fuse(
        dense_results,
        bm25_results
    )

    return results


# ============================================================
# EXTRACT ORIGINAL RESULT
# ============================================================

def prepare_candidates(rrf_results):
    """
    Convert RRF results into candidates suitable
    for the cross-encoder.

    RRF stores the original retrieval result inside:

        result

    """

    candidates = []

    for item in rrf_results:

        original = item.get("result", {})

        content = original.get(
            "content",
            ""
        ).strip()

        if not content:
            continue

        candidate = dict(original)

        # Preserve RRF information
        candidate["rrf_score"] = item.get(
            "rrf_score"
        )

        candidate["rrf_rank"] = item.get(
            "rrf_rank"
        )

        candidate["dense_rank"] = item.get(
            "dense_rank"
        )

        candidate["bm25_rank"] = item.get(
            "bm25_rank"
        )

        candidates.append(candidate)

    return candidates


# ============================================================
# RERANK
# ============================================================

def rerank(
    question: str,
    candidates: list[dict],
    top_k: int = FINAL_TOP_K
):
    """
    Rerank candidates using:

        CrossEncoder(question, content)
    """

    if not candidates:
        return []

    model = load_reranker()

    pairs = []

    for candidate in candidates:

        content = candidate.get(
            "content",
            ""
        ).strip()

        pairs.append(
            [
                question,
                content
            ]
        )

    print()
    print(
        f"[reranker] Scoring {len(pairs)} candidates..."
    )

    scores = model.predict(pairs)

    results = []

    for candidate, score in zip(
        candidates,
        scores
    ):

        item = dict(candidate)

        item["reranker_score"] = float(score)

        results.append(item)

    # Highest cross-encoder score first
    results.sort(
        key=lambda item: item[
            "reranker_score"
        ],
        reverse=True
    )

    # Final rank
    results = results[:top_k]

    for rank, result in enumerate(
        results,
        start=1
    ):

        result["reranker_rank"] = rank

    return results


# ============================================================
# DISPLAY
# ============================================================

def display_results(
    question: str,
    results: list[dict]
):

    print()
    print("=" * 75)
    print("CROSS-ENCODER RERANKER RESULTS")
    print("=" * 75)

    print()
    print(f"Question: {question}")

    for result in results:

        print()
        print(
            f"Reranker Rank : "
            f"{result['reranker_rank']}"
        )

        print(
            f"Reranker Score: "
            f"{result['reranker_score']:.4f}"
        )

        print(
            f"RRF Rank      : "
            f"{result.get('rrf_rank')}"
        )

        print(
            f"RRF Score     : "
            f"{result.get('rrf_score')}"
        )

        print(
            f"Child ID      : "
            f"{result.get('child_id')}"
        )

        print(
            f"Parent ID     : "
            f"{result.get('parent_id')}"
        )

        print(
            f"Type          : "
            f"{result.get('element_type')}"
        )

        print(
            f"Page          : "
            f"{result.get('page')}"
        )

        print(
            f"Section       : "
            f"{result.get('section')}"
        )

        print(
            "Content       : "
            f"{result.get('content', '')[:800]}"
        )

        print("-" * 75)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print("DAY 23 - CROSS-ENCODER RERANKER TEST")
    print("=" * 75)

    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:

        print(
            "Question cannot be empty."
        )

        return

    # --------------------------------------------------------
    # 1. RRF
    # --------------------------------------------------------

    print()
    print(
        "[reranker] Getting RRF candidates..."
    )

    rrf_results = get_rrf_candidates(
        question,
        top_k=RRF_TOP_K
    )

    print(
        f"[reranker] RRF candidates: "
        f"{len(rrf_results)}"
    )

    # --------------------------------------------------------
    # 2. Prepare candidates
    # --------------------------------------------------------

    candidates = prepare_candidates(
        rrf_results
    )

    # --------------------------------------------------------
    # 3. Cross-Encoder
    # --------------------------------------------------------

    results = rerank(
        question,
        candidates,
        top_k=FINAL_TOP_K
    )

    # --------------------------------------------------------
    # 4. Display
    # --------------------------------------------------------

    display_results(
        question,
        results
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()