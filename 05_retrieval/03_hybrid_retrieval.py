"""
Day 23 - Step 5.3: Hybrid Retrieval

Combines:

    Dense Retrieval
          +
    BM25 Retrieval

Why?

Dense Retrieval:
    Finds semantically similar content.

BM25:
    Finds content containing important keywords.

Hybrid Retrieval:
    Uses both retrieval signals.

Important:
    Dense and BM25 scores have different scales.
    Therefore, we do NOT directly add their scores.

    The next step (RRF) will combine their rankings.

Flow:

    Question
       |
       +-------------------+
       |                   |
       v                   v
    Dense                BM25
       |                   |
       v                   v
   Results              Results
       |                   |
       +---------+---------+
                 |
                 v
              Hybrid
                 |
                 v
                RRF
"""


from pathlib import Path
import sys


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


# ============================================================
# IMPORT EXISTING RETRIEVERS
# ============================================================

# Allows Python to import the retrieval modules when this
# file is executed directly.

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from importlib.util import (
    module_from_spec,
    spec_from_file_location,
)


def load_retriever_module(
    filename: str,
    module_name: str
):
    """
    Load another retrieval file safely.

    This is useful because our files have numbered names
    such as:

        01_dense_retrieval.py
        02_bm25_retrieval.py

    Python cannot import those names using normal import syntax.
    """

    file_path = (
        PROJECT_ROOT
        / "05_retrieval"
        / filename
    )

    spec = spec_from_file_location(
        module_name,
        file_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load {file_path}"
        )

    module = module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# LOAD DENSE RETRIEVER
# ============================================================

dense_module = load_retriever_module(
    "01_dense_retrieval.py",
    "dense_retrieval"
)

DenseRetriever = (
    dense_module.DenseRetriever
)


# ============================================================
# LOAD BM25 RETRIEVER
# ============================================================

bm25_module = load_retriever_module(
    "02_bm25_retrieval.py",
    "bm25_retrieval"
)

BM25Retriever = (
    bm25_module.BM25Retriever
)


# ============================================================
# HYBRID RETRIEVER
# ============================================================

class HybridRetriever:

    def __init__(
        self,
        top_k: int = 5,
        dense_k: int = 10,
        bm25_k: int = 10
    ):

        self.top_k = top_k

        print()
        print(
            "[hybrid] Initializing Dense Retrieval..."
        )

        self.dense_retriever = DenseRetriever(
            top_k=dense_k
        )

        print()
        print(
            "[hybrid] Initializing BM25 Retrieval..."
        )

        self.bm25_retriever = BM25Retriever(
            top_k=bm25_k
        )

        print()
        print(
            "[hybrid] Hybrid retriever ready."
        )


    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int | None = None
    ) -> dict:

        if not query.strip():
            return {
                "dense_results": [],
                "bm25_results": [],
                "combined_results": [],
            }

        if top_k is None:
            top_k = self.top_k

        # ----------------------------------------------------
        # 1. Dense Retrieval
        # ----------------------------------------------------

        print(
            "\n[hybrid] Running Dense Retrieval..."
        )

        dense_results = (
            self.dense_retriever.search(
                query,
                top_k=self.dense_retriever.top_k
            )
        )

        # ----------------------------------------------------
        # 2. BM25 Retrieval
        # ----------------------------------------------------

        print(
            "[hybrid] Running BM25 Retrieval..."
        )

        bm25_results = (
            self.bm25_retriever.search(
                query,
                top_k=self.bm25_retriever.top_k
            )
        )

        # ----------------------------------------------------
        # 3. Combine candidates
        # ----------------------------------------------------

        combined = {}

        # Dense candidates
        for result in dense_results:

            child_id = result[
                "child_id"
            ]

            combined.setdefault(
                child_id,
                {
                    "child_id": child_id,
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "result": result,
                }
            )

            combined[
                child_id
            ][
                "dense_rank"
            ] = result["rank"]

            combined[
                child_id
            ][
                "dense_score"
            ] = result["score"]

        # BM25 candidates
        for result in bm25_results:

            child_id = result[
                "child_id"
            ]

            combined.setdefault(
                child_id,
                {
                    "child_id": child_id,
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "result": result,
                }
            )

            combined[
                child_id
            ][
                "bm25_rank"
            ] = result["rank"]

            combined[
                child_id
            ][
                "bm25_score"
            ] = result["score"]

        # ----------------------------------------------------
        # 4. Sort candidates
        #
        # For now we use:
        #
        #   Dense rank
        #   BM25 rank
        #
        # The next file will replace this simple ordering
        # with proper Reciprocal Rank Fusion.
        # ----------------------------------------------------

        def hybrid_sort_key(item):

            dense_rank = (
                item["dense_rank"]
                if item["dense_rank"] is not None
                else 999999
            )

            bm25_rank = (
                item["bm25_rank"]
                if item["bm25_rank"] is not None
                else 999999
            )

            return (
                min(
                    dense_rank,
                    bm25_rank
                ),
                dense_rank,
                bm25_rank,
            )

        combined_results = sorted(
            combined.values(),
            key=hybrid_sort_key
        )[:top_k]

        # ----------------------------------------------------
        # 5. Add final hybrid rank
        # ----------------------------------------------------

        for index, item in enumerate(
            combined_results
        ):

            item["hybrid_rank"] = (
                index + 1
            )

        return {
            "dense_results": dense_results,
            "bm25_results": bm25_results,
            "combined_results": combined_results,
        }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    query: str,
    results: dict
):

    print()
    print(
        "=" * 75
    )

    print(
        f"HYBRID RETRIEVAL RESULTS"
    )

    print(
        "=" * 75
    )

    print(
        f"\nQuestion: {query}"
    )

    # --------------------------------------------------------
    # Dense
    # --------------------------------------------------------

    print()
    print(
        "DENSE RESULTS"
    )

    print(
        "-" * 75
    )

    for result in results[
        "dense_results"
    ]:

        print(
            f"Rank {result['rank']} | "
            f"Score {result['score']:.4f} | "
            f"{result['child_id']} | "
            f"Page {result['page']}"
        )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    print()
    print(
        "BM25 RESULTS"
    )

    print(
        "-" * 75
    )

    for result in results[
        "bm25_results"
    ]:

        print(
            f"Rank {result['rank']} | "
            f"Score {result['score']:.4f} | "
            f"{result['child_id']} | "
            f"Page {result['page']}"
        )

    # --------------------------------------------------------
    # Combined
    # --------------------------------------------------------

    print()
    print(
        "COMBINED HYBRID CANDIDATES"
    )

    print(
        "-" * 75
    )

    for item in results[
        "combined_results"
    ]:

        print(
            f"\nHybrid Rank : "
            f"{item['hybrid_rank']}"
        )

        print(
            f"Child ID    : "
            f"{item['child_id']}"
        )

        print(
            f"Dense Rank  : "
            f"{item['dense_rank']}"
        )

        print(
            f"Dense Score : "
            f"{item['dense_score']}"
        )

        print(
            f"BM25 Rank   : "
            f"{item['bm25_rank']}"
        )

        print(
            f"BM25 Score  : "
            f"{item['bm25_score']}"
        )

        result = item[
            "result"
        ]

        print(
            f"Section     : "
            f"{result['section']}"
        )

        print(
            f"Content     : "
            f"{result['content'][:400]}"
        )

        print(
            "-" * 75
        )


# ============================================================
# TEST
# ============================================================

def main():

    print()
    print(
        "=" * 75
    )

    print(
        "DAY 23 - HYBRID RETRIEVAL TEST"
    )

    print(
        "=" * 75
    )

    retriever = HybridRetriever(
        top_k=5,
        dense_k=10,
        bm25_k=10
    )

    query = input(
        "\nEnter your question: "
    ).strip()

    results = retriever.search(
        query
    )

    print_results(
        query,
        results
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
