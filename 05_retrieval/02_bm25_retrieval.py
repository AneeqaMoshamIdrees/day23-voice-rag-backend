"""
Day 23 - Step 5.2: BM25 Retrieval

Purpose:
    Perform keyword-based retrieval over the hierarchical child chunks.

Flow:
    User Question
          ↓
    Tokenization
          ↓
    BM25 Search
          ↓
    Top-K Relevant Child Chunks

Input:
    data/structured_documents/hierarchical_chunks.json

Output:
    Ranked retrieval results

BM25 is different from Dense Retrieval:

    Dense Retrieval:
        understands semantic similarity

    BM25:
        focuses on matching important words
"""


import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HIERARCHICAL_CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "structured_documents"
    / "hierarchical_chunks.json"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

DEFAULT_TOP_K = 5


# ============================================================
# 3. TEXT TOKENIZATION
# ============================================================

def tokenize(text: str) -> list[str]:
    """
    Convert text into lowercase tokens.

    Example:

        "What is the LMS training rule?"

    becomes approximately:

        ["what", "is", "the", "lms", "training", "rule"]
    """

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


# ============================================================
# 4. BM25 RETRIEVER
# ============================================================

class BM25Retriever:

    def __init__(
        self,
        top_k: int = DEFAULT_TOP_K
    ):
        self.top_k = top_k

        # ----------------------------------------------------
        # Load hierarchical chunks
        # ----------------------------------------------------

        print(
            "[bm25] Loading hierarchical chunks..."
        )

        if not HIERARCHICAL_CHUNKS_FILE.exists():
            raise FileNotFoundError(
                f"Hierarchical chunks not found:\n"
                f"{HIERARCHICAL_CHUNKS_FILE}"
            )

        data = json.loads(
            HIERARCHICAL_CHUNKS_FILE.read_text(
                encoding="utf-8"
            )
        )

        # searchable_children contains:
        #
        # text
        # image
        # table
        # equation
        #
        # child chunks.

        self.children = data[
            "searchable_children"
        ]

        print(
            f"[bm25] Searchable children: "
            f"{len(self.children)}"
        )

        # ----------------------------------------------------
        # Prepare documents for BM25
        # ----------------------------------------------------

        print(
            "[bm25] Tokenizing child chunks..."
        )

        self.tokenized_documents = [
            tokenize(
                child["embedding_text"]
            )
            for child in self.children
        ]

        # ----------------------------------------------------
        # Build BM25 index
        # ----------------------------------------------------

        print(
            "[bm25] Building BM25 index..."
        )

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

        print(
            "[bm25] BM25 index ready."
        )


    # ========================================================
    # 5. SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int | None = None
    ) -> list[dict]:
        """
        Search the hierarchical children using BM25.
        """

        if not query.strip():
            return []

        if top_k is None:
            top_k = self.top_k

        # ----------------------------------------------------
        # Tokenize user query
        # ----------------------------------------------------

        query_tokens = tokenize(
            query
        )

        # ----------------------------------------------------
        # Calculate BM25 scores
        # ----------------------------------------------------

        scores = self.bm25.get_scores(
            query_tokens
        )

        # ----------------------------------------------------
        # Sort results by score
        # Highest score first
        # ----------------------------------------------------

        ranked_positions = sorted(
            range(len(scores)),
            key=lambda position: scores[position],
            reverse=True
        )

        ranked_positions = ranked_positions[
            :top_k
        ]

        # ----------------------------------------------------
        # Build result objects
        # ----------------------------------------------------

        results = []

        for position in ranked_positions:

            child = self.children[position]

            result = {
                "rank": len(results) + 1,

                "score": float(
                    scores[position]
                ),

                "bm25_position": int(
                    position
                ),

                "child_id": child[
                    "child_id"
                ],

                "parent_id": child[
                    "parent_id"
                ],

                "element_type": child[
                    "element_type"
                ],

                "page": child[
                    "page"
                ],

                "section": child[
                    "section"
                ],

                "content": child[
                    "content"
                ],

                "embedding_text": child[
                    "embedding_text"
                ],
            }

            results.append(
                result
            )

        return results

    # ========================================================
    # 6. ADD CHUNKS (live ingestion — used by /api/rag/ingest)
    # ========================================================

    def add_chunks(self, new_chunks: list[dict]) -> int:
        """
        BM25Okapi has no incremental "add one document" operation --
        its scoring depends on corpus-wide statistics (document
        frequency, average document length), so any new document
        genuinely changes those numbers for every existing document
        too. The only correct option is to extend the token corpus
        and rebuild.

        This is cheap: even a few thousand short chunks tokenize and
        rebuild in a fraction of a second, so doing this synchronously
        inside a single upload request is fine.

        hierarchical_chunks.json itself is persisted centrally by
        LangChainRetrievalPipeline.ingest_document() -- not here --
        since that file is shared with the parent expander too and
        should only be written once per ingest.
        """

        if not new_chunks:
            return 0

        self.children.extend(new_chunks)

        new_tokenized = [
            tokenize(chunk["embedding_text"])
            for chunk in new_chunks
        ]

        self.tokenized_documents.extend(new_tokenized)

        print(
            f"[bm25] Rebuilding BM25 index over "
            f"{len(self.tokenized_documents)} documents..."
        )

        self.bm25 = BM25Okapi(self.tokenized_documents)

        print(
            f"[bm25] Added {len(new_chunks)} new documents. "
            f"Total now: {len(self.children)}"
        )

        return len(new_chunks)


# ============================================================
# 7. PRINT RESULTS
# ============================================================

def print_results(
    query: str,
    results: list[dict]
):

    print()
    print(
        f"[bm25] Results for: {query}"
    )

    print(
        "=" * 70
    )

    if not results:
        print(
            "[bm25] No results found."
        )
        return

    for result in results:

        print()

        print(
            f"Rank       : {result['rank']}"
        )

        print(
            f"Score      : "
            f"{result['score']:.4f}"
        )

        print(
            f"Child ID   : "
            f"{result['child_id']}"
        )

        print(
            f"Parent ID  : "
            f"{result['parent_id']}"
        )

        print(
            f"Type       : "
            f"{result['element_type']}"
        )

        print(
            f"Page       : "
            f"{result['page']}"
        )

        print(
            f"Section    : "
            f"{result['section']}"
        )

        print(
            f"Content    : "
            f"{result['content'][:500]}"
        )

        print(
            "-" * 70
        )


# ============================================================
# 8. TEST BM25 RETRIEVAL
# ============================================================

def main():

    print()
    print(
        "=" * 70
    )

    print(
        "DAY 23 - BM25 RETRIEVAL TEST"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Create BM25 retriever
    # --------------------------------------------------------

    retriever = BM25Retriever(
        top_k=5
    )

    # --------------------------------------------------------
    # Ask question
    # --------------------------------------------------------

    query = input(
        "\nEnter your question: "
    ).strip()

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    results = retriever.search(
        query
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print_results(
        query,
        results
    )


# ============================================================
# 9. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()