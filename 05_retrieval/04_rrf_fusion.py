"""
Day 23 - Step 5.4: Reciprocal Rank Fusion (RRF)

Purpose:
    Combine Dense Retrieval and BM25 Retrieval rankings.

Why RRF?

    Dense Retrieval and BM25 produce different score ranges.

    Example:

        Dense score = 0.7466
        BM25 score  = 17.2165

    These scores cannot safely be added directly.

    RRF solves this by using RANK instead of raw scores.

Formula:

    RRF(d) = Σ 1 / (k + rank)

Where:

    d    = document / child chunk
    rank = position returned by a retriever
    k    = constant, normally 60

Flow:

    User Question
          ↓
    ┌───────────────┐
    │               │
    ↓               ↓
 Dense             BM25
    ↓               ↓
 Dense ranks     BM25 ranks
    └───────┬───────┘
            ↓
           RRF
            ↓
     Unified ranking
"""


from pathlib import Path
import sys
from importlib.util import (
    module_from_spec,
    spec_from_file_location,
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


# ============================================================
# LOAD RETRIEVER MODULES
# ============================================================

def load_module(
    filename: str,
    module_name: str
):
    """
    Load one of our numbered retrieval files.

    Example:

        01_dense_retrieval.py
        02_bm25_retrieval.py
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
            f"Could not load module: "
            f"{file_path}"
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

dense_module = load_module(
    "01_dense_retrieval.py",
    "dense_retrieval"
)

DenseRetriever = (
    dense_module.DenseRetriever
)


# ============================================================
# LOAD BM25 RETRIEVER
# ============================================================

bm25_module = load_module(
    "02_bm25_retrieval.py",
    "bm25_retrieval"
)

BM25Retriever = (
    bm25_module.BM25Retriever
)


# ============================================================
# RRF FUSION
# ============================================================

class RRFFusion:

    def __init__(
        self,
        dense_k: int = 10,
        bm25_k: int = 10,
        rrf_k: int = 60,
        top_k: int = 5,
    ):

        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.rrf_k = rrf_k
        self.top_k = top_k

        print()
        print(
            "[rrf] Initializing Dense Retriever..."
        )

        self.dense_retriever = (
            DenseRetriever(
                top_k=dense_k
            )
        )

        print()
        print(
            "[rrf] Initializing BM25 Retriever..."
        )

        self.bm25_retriever = (
            BM25Retriever(
                top_k=bm25_k
            )
        )

        print()
        print(
            f"[rrf] RRF constant k = "
            f"{self.rrf_k}"
        )

        print(
            "[rrf] RRF fusion ready."
        )


    # ========================================================
    # RRF SCORE
    # ========================================================

    def calculate_rrf_score(
        self,
        rank: int
    ) -> float:
        """
        Calculate:

            1 / (k + rank)
        """

        return 1.0 / (
            self.rrf_k + rank
        )


    # ========================================================
    # FUSE RESULTS
    # ========================================================

    def fuse(
        self,
        dense_results: list[dict],
        bm25_results: list[dict],
    ) -> list[dict]:

        fused = {}

        # ----------------------------------------------------
        # Dense results
        # ----------------------------------------------------

        for result in dense_results:

            child_id = result[
                "child_id"
            ]

            if child_id not in fused:

                fused[child_id] = {
                    "child_id": child_id,

                    "rrf_score": 0.0,

                    "dense_rank": None,
                    "dense_score": None,

                    "bm25_rank": None,
                    "bm25_score": None,

                    "result": result,
                }

            rank = result[
                "rank"
            ]

            fused[
                child_id
            ][
                "dense_rank"
            ] = rank

            fused[
                child_id
            ][
                "dense_score"
            ] = result[
                "score"
            ]

            # RRF contribution
            fused[
                child_id
            ][
                "rrf_score"
            ] += self.calculate_rrf_score(
                rank
            )

        # ----------------------------------------------------
        # BM25 results
        # ----------------------------------------------------

        for result in bm25_results:

            child_id = result[
                "child_id"
            ]

            if child_id not in fused:

                fused[child_id] = {
                    "child_id": child_id,

                    "rrf_score": 0.0,

                    "dense_rank": None,
                    "dense_score": None,

                    "bm25_rank": None,
                    "bm25_score": None,

                    "result": result,
                }

            rank = result[
                "rank"
            ]

            fused[
                child_id
            ][
                "bm25_rank"
            ] = rank

            fused[
                child_id
            ][
                "bm25_score"
            ] = result[
                "score"
            ]

            # RRF contribution
            fused[
                child_id
            ][
                "rrf_score"
            ] += self.calculate_rrf_score(
                rank
            )

        # ----------------------------------------------------
        # Sort by RRF score
        # ----------------------------------------------------

        ranked_results = sorted(
            fused.values(),
            key=lambda item: item[
                "rrf_score"
            ],
            reverse=True,
        )

        # ----------------------------------------------------
        # Keep top-K
        # ----------------------------------------------------

        ranked_results = ranked_results[
            :self.top_k
        ]

        # ----------------------------------------------------
        # Add final RRF rank
        # ----------------------------------------------------

        for index, item in enumerate(
            ranked_results
        ):

            item[
                "rrf_rank"
            ] = index + 1

        return ranked_results


    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str
    ) -> list[dict]:

        print()
        print(
            "[rrf] Running Dense Retrieval..."
        )

        dense_results = (
            self.dense_retriever.search(
                query,
                top_k=self.dense_k
            )
        )

        print(
            "[rrf] Running BM25 Retrieval..."
        )

        bm25_results = (
            self.bm25_retriever.search(
                query,
                top_k=self.bm25_k
            )
        )

        print(
            "[rrf] Applying Reciprocal Rank Fusion..."
        )

        fused_results = self.fuse(
            dense_results,
            bm25_results,
        )

        return fused_results


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    query: str,
    results: list[dict]
):

    print()
    print(
        "=" * 75
    )

    print(
        "RRF FUSION RESULTS"
    )

    print(
        "=" * 75
    )

    print(
        f"\nQuestion: {query}"
    )

    print(
        f"\nRRF constant k = 60"
    )

    print(
        "-" * 75
    )

    for result in results:

        print()

        print(
            f"RRF Rank    : "
            f"{result['rrf_rank']}"
        )

        print(
            f"RRF Score   : "
            f"{result['rrf_score']:.6f}"
        )

        print(
            f"Child ID    : "
            f"{result['child_id']}"
        )

        print(
            f"Dense Rank  : "
            f"{result['dense_rank']}"
        )

        print(
            f"Dense Score : "
            f"{result['dense_score']}"
        )

        print(
            f"BM25 Rank   : "
            f"{result['bm25_rank']}"
        )

        print(
            f"BM25 Score  : "
            f"{result['bm25_score']}"
        )

        retrieved = result[
            "result"
        ]

        print(
            f"Parent ID   : "
            f"{retrieved['parent_id']}"
        )

        print(
            f"Type        : "
            f"{retrieved['element_type']}"
        )

        print(
            f"Page        : "
            f"{retrieved['page']}"
        )

        print(
            f"Section     : "
            f"{retrieved['section']}"
        )

        print(
            f"Content     : "
            f"{retrieved['content'][:500]}"
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
        "DAY 23 - RRF FUSION TEST"
    )

    print(
        "=" * 75
    )

    fusion = RRFFusion(
        dense_k=10,
        bm25_k=10,
        rrf_k=60,
        top_k=5,
    )

    query = input(
        "\nEnter your question: "
    ).strip()

    results = fusion.search(
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
