"""
DAY 23 - STEP 5.6: PARENT EXPANDER

Pipeline:

    Dense Retrieval
          ↓
    BM25 Retrieval
          ↓
    Hybrid Retrieval
          ↓
    RRF Fusion
          ↓
    Cross-Encoder Reranker
          ↓
    Parent Expander
          ↓
    Expanded Context
          ↓
    LLM

Purpose:

    The reranker selects the most relevant CHILD chunks.

    Parent Expander then finds the PARENT section belonging
    to those selected children and returns the complete
    parent context.

Example:

    child_0054_0002
           ↓
    parent_0054
           ↓
    all children belonging to parent_0054
           ↓
    expanded context
"""


from pathlib import Path
import json
import sys


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

RETRIEVAL_DIR = (
    PROJECT_ROOT
    / "05_retrieval"
)


# ============================================================
# ADD PROJECT PATH
# ============================================================

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )

if str(RETRIEVAL_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(RETRIEVAL_DIR)
    )


# ============================================================
# FILE PATH
# ============================================================

HIERARCHICAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "structured_documents"
    / "hierarchical_chunks.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

RERANKER_TOP_K = 5


# ============================================================
# LOAD HIERARCHICAL DOCUMENT
# ============================================================

def load_hierarchical_data():

    print(
        "[parent] Loading hierarchical chunks..."
    )

    if not HIERARCHICAL_FILE.exists():

        raise FileNotFoundError(
            f"File not found:\n"
            f"{HIERARCHICAL_FILE}"
        )

    with open(
        HIERARCHICAL_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    print(
        "[parent] File loaded successfully."
    )

    return data


# ============================================================
# BUILD PARENT LOOKUP
# ============================================================

def build_parent_lookup(data):

    parents = data.get(
        "parents",
        []
    )

    lookup = {}

    for parent in parents:

        parent_id = parent.get(
            "parent_id"
        )

        if parent_id:

            lookup[parent_id] = parent

    print(
        f"[parent] Parent lookup created: "
        f"{len(lookup)} parents"
    )

    return lookup


# ============================================================
# GET RERANKED RESULTS
# ============================================================

def get_reranked_results(question):

    print()
    print("[parent] Running RRF → Reranker...")

    # --------------------------------------------------------
    # Load RRF module
    # --------------------------------------------------------

    from importlib.util import (
        spec_from_file_location,
        module_from_spec,
    )

    rrf_file = (
        RETRIEVAL_DIR
        / "04_rrf_fusion.py"
    )

    spec = spec_from_file_location(
        "rrf_module",
        rrf_file
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load: {rrf_file}"
        )

    rrf_module = module_from_spec(spec)

    spec.loader.exec_module(
        rrf_module
    )

    # --------------------------------------------------------
    # Create RRF retriever
    # --------------------------------------------------------

    rrf = rrf_module.RRFFusion(
        dense_k=10,
        bm25_k=10,
        rrf_k=60,
        top_k=10,
    )

    # --------------------------------------------------------
    # Run Dense + BM25 → RRF
    # --------------------------------------------------------

    print()
    print(
        "[parent] Running Dense + BM25 → RRF..."
    )

    rrf_results = rrf.search(
        question
    )

    print(
        f"[parent] RRF results: "
        f"{len(rrf_results)}"
    )

    # --------------------------------------------------------
    # Load Reranker
    # --------------------------------------------------------

    reranker_file = (
        RETRIEVAL_DIR
        / "05_reranker.py"
    )

    reranker_spec = spec_from_file_location(
        "reranker_module",
        reranker_file
    )

    if (
        reranker_spec is None
        or reranker_spec.loader is None
    ):
        raise ImportError(
            f"Could not load: {reranker_file}"
        )

    reranker_module = (
        module_from_spec(
            reranker_spec
        )
    )

    reranker_spec.loader.exec_module(
        reranker_module
    )

    # --------------------------------------------------------
    # RRF output → Cross Encoder
    # --------------------------------------------------------

    candidates = []

    for item in rrf_results:

        # 04_rrf_fusion stores the original
        # retrieval result inside "result".

        result = item.get(
            "result",
            {}
        )

        candidate = dict(result)

        # Preserve RRF information.

        candidate["rrf_rank"] = item.get(
            "rrf_rank"
        )

        candidate["rrf_score"] = item.get(
            "rrf_score"
        )

        candidate["dense_rank"] = item.get(
            "dense_rank"
        )

        candidate["bm25_rank"] = item.get(
            "bm25_rank"
        )

        candidate["dense_score"] = item.get(
            "dense_score"
        )

        candidate["bm25_score"] = item.get(
            "bm25_score"
        )

        candidates.append(
            candidate
        )

    print(
        f"[parent] Candidates sent to "
        f"reranker: {len(candidates)}"
    )

    # --------------------------------------------------------
    # Cross-encoder reranking
    # --------------------------------------------------------

    reranked = reranker_module.rerank(
        question,
        candidates,
        top_k=RERANKER_TOP_K
    )

    print(
        f"[parent] Reranked children: "
        f"{len(reranked)}"
    )

    return reranked


# ============================================================
# EXPAND CHILDREN TO PARENTS
# ============================================================

def expand_to_parents(
    reranked_children,
    parent_lookup
):

    expanded = []

    seen_parents = set()

    for child in reranked_children:

        child_id = child.get(
            "child_id"
        )

        parent_id = child.get(
            "parent_id"
        )

        if not parent_id:

            print(
                f"[parent] WARNING: "
                f"{child_id} has no parent_id"
            )

            continue

        # ----------------------------------------------------
        # Avoid duplicate parent sections
        # ----------------------------------------------------

        if parent_id in seen_parents:

            continue

        parent = parent_lookup.get(
            parent_id
        )

        if parent is None:

            print(
                f"[parent] WARNING: "
                f"{parent_id} not found"
            )

            continue

        seen_parents.add(
            parent_id
        )

        expanded_parent = {
            "parent_id": parent_id,

            "section": parent.get(
                "section"
            ),

            "page": parent.get(
                "page"
            ),

            "children": parent.get(
                "children",
                []
            ),

            "matched_child_id": child_id,

            "reranker_score": child.get(
                "reranker_score"
            ),

            "element_type": child.get(
                "element_type"
            ),

            # Which document this parent/section came from. Live-ingested
            # documents (via /api/rag/ingest) tag every parent with this
            # field; the original startup document predates that field,
            # so we fall back to "MACHINE LEARNING.pdf" for it.
            "source": parent.get(
                "source",
                "MACHINE LEARNING.pdf"
            ),
        }

        expanded.append(
            expanded_parent
        )

    return expanded


# ============================================================
# BUILD EXPANDED CONTEXT
# ============================================================

def build_expanded_context(
    expanded_parents
):

    context_parts = []

    for rank, parent in enumerate(
        expanded_parents,
        start=1
    ):

        parent_id = parent.get(
            "parent_id",
            ""
        )

        section = parent.get(
            "section",
            ""
        )

        page = parent.get(
            "page",
            ""
        )

        children = parent.get(
            "children",
            []
        )

        content_parts = []

        for child in children:

            content = child.get(
                "content",
                ""
            )

            if content:

                content_parts.append(
                    content.strip()
                )

        parent_content = (
            "\n\n".join(
                content_parts
            )
        )

        block = (
            f"--- Parent {rank} ---\n"
            f"Parent ID: {parent_id}\n"
            f"Section: {section}\n"
            f"Page: {page}\n\n"
            f"{parent_content}"
        )

        context_parts.append(
            block
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    question,
    reranked_children,
    expanded_parents,
    context
):

    print()
    print("=" * 75)
    print(
        "PARENT EXPANDER RESULTS"
    )
    print("=" * 75)

    print()
    print(
        f"Question: {question}"
    )

    print()
    print(
        f"Reranked children : "
        f"{len(reranked_children)}"
    )

    print(
        f"Expanded parents  : "
        f"{len(expanded_parents)}"
    )

    # --------------------------------------------------------
    # CHILD → PARENT
    # --------------------------------------------------------

    print()
    print(
        "CHILD → PARENT MAPPING"
    )

    print(
        "-" * 75
    )

    for rank, parent in enumerate(
        expanded_parents,
        start=1
    ):

        score = parent.get(
            "reranker_score"
        )

        print()

        print(
            f"Parent Rank       : {rank}"
        )

        print(
            f"Parent ID         : "
            f"{parent.get('parent_id')}"
        )

        print(
            f"Matched Child     : "
            f"{parent.get('matched_child_id')}"
        )

        if score is not None:

            print(
                f"Reranker Score    : "
                f"{score:.4f}"
            )

        else:

            print(
                "Reranker Score    : None"
            )

        print(
            f"Section           : "
            f"{parent.get('section')}"
        )

        print(
            f"Page              : "
            f"{parent.get('page')}"
        )

        print(
            f"Source            : "
            f"{parent.get('source')}"
        )

        print(
            f"Parent Children   : "
            f"{len(parent.get('children', []))}"
        )

        print(
            "-" * 75
        )

    # --------------------------------------------------------
    # EXPANDED CONTEXT
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print(
        "EXPANDED CONTEXT"
    )
    print("=" * 75)

    print()

    if not context:

        print(
            "[parent] No expanded context."
        )

        return

    # Don't flood the terminal.

    print(
        context[:5000]
    )

    if len(context) > 5000:

        print()
        print(
            "... [context display truncated]"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print(
        "DAY 23 - STEP 5.6: "
        "PARENT EXPANDER TEST"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Question
    # --------------------------------------------------------

    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:

        print(
            "Question cannot be empty."
        )

        return

    # --------------------------------------------------------
    # Load document
    # --------------------------------------------------------

    data = load_hierarchical_data()

    # --------------------------------------------------------
    # Parent lookup
    # --------------------------------------------------------

    parent_lookup = (
        build_parent_lookup(data)
    )

    # --------------------------------------------------------
    # Reranker
    # --------------------------------------------------------

    reranked_children = (
        get_reranked_results(
            question
        )
    )

    # --------------------------------------------------------
    # Parent expansion
    # --------------------------------------------------------

    print()
    print(
        "[parent] Expanding children → parents..."
    )

    expanded_parents = (
        expand_to_parents(
            reranked_children,
            parent_lookup
        )
    )

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    print()
    print(
        "[parent] Building expanded context..."
    )

    context = (
        build_expanded_context(
            expanded_parents
        )
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_results(
        question,
        reranked_children,
        expanded_parents,
        context
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()