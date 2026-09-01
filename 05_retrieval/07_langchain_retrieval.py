"""
DAY 23 - STEP 5.7: LANGCHAIN RETRIEVAL PIPELINE

This is the ONE main retrieval entry point.

Flow:

    User Question
          |
          +--------------------+
          |                    |
          v                    v
       Dense               BM25
    (01_dense)          (02_bm25)
          |                    |
          +---------+----------+
                    |
                    v
               RRF Fusion
              (04_rrf_fusion)
                    |
                    v
          Cross-Encoder Reranker
              (05_reranker)
                    |
                    v
             Parent Expansion
           (06_parent_expander)
                    |
                    v
          LangChain Documents
                    |
                    v
        (handed to 08_gemini_generation.py)

This file does NOT call Gemini.
Generation lives entirely in 08_gemini_generation.py.

ALSO in this file: LangChainRetrievalPipeline.ingest_document(), which
lets a new PDF be uploaded at runtime (via /api/rag/ingest in api.py)
and become searchable immediately, without restarting the server. It
reuses the exact same PDF-parsing / chunking / embedding building
blocks the offline Day 17-23 scripts use, so a live-uploaded document
is processed identically to the one loaded at startup.
"""

from pathlib import Path
import sys
import json
from importlib.util import (
    module_from_spec,
    spec_from_file_location,
)

from langchain_core.documents import Document


# ============================================================
# PATHS
# ============================================================

RETRIEVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RETRIEVAL_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HIERARCHICAL_CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "structured_documents"
    / "hierarchical_chunks.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

DENSE_TOP_K = 10
BM25_TOP_K = 10
RRF_TOP_K = 10
RERANK_TOP_K = 5
PARENT_TOP_K = 5


# ============================================================
# LOAD NUMBERED MODULES
# ============================================================

def load_module(filename, module_name):
    """
    Loads a module by filename, relative to THIS file's own
    directory (05_retrieval/). Used for the retrieval-stage
    modules that already live next to this file.
    """
    file_path = RETRIEVAL_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required module not found: {file_path}"
        )

    spec = spec_from_file_location(module_name, file_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {file_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def load_module_at(file_path: Path, module_name: str):
    """
    Same as load_module(), but takes a full path -- used for
    ingest_document() to reach into sibling folders like
    01_ingestion/ and 02_chunking/ that this file doesn't
    otherwise need to know about.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required module not found: {file_path}"
        )

    spec = spec_from_file_location(module_name, file_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {file_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


print()
print("[langchain_retrieval] Loading retrieval components...")

rrf_module = load_module("04_rrf_fusion.py", "day23_rrf")
reranker_module = load_module("05_reranker.py", "day23_reranker")
parent_module = load_module("06_parent_expander.py", "day23_parent")

RRFFusion = rrf_module.RRFFusion


# ============================================================
# THE MAIN RETRIEVAL PIPELINE CLASS
# ============================================================

class LangChainRetrievalPipeline:

    def __init__(self):

        print()
        print("=" * 75)
        print("INITIALIZING RETRIEVAL PIPELINE")
        print("=" * 75)

        print()
        print("[pipeline] Loading RRF (Dense + BM25 inside)...")

        self.rrf = RRFFusion(
            dense_k=DENSE_TOP_K,
            bm25_k=BM25_TOP_K,
            rrf_k=60,
            top_k=RRF_TOP_K,
        )

        print()
        print("[pipeline] Loading hierarchical document data...")

        self.hierarchical_data = parent_module.load_hierarchical_data()

        self.parent_lookup = parent_module.build_parent_lookup(
            self.hierarchical_data
        )

        self.child_lookup = {
            child["child_id"]: child
            for child in self.hierarchical_data.get(
                "searchable_children", []
            )
            if child.get("child_id")
        }

        print(
            f"[pipeline] Parent lookup ready: "
            f"{len(self.parent_lookup)} parents"
        )
        print(
            f"[pipeline] Child lookup ready: "
            f"{len(self.child_lookup)} children"
        )

        print()
        print("[pipeline] Retrieval pipeline ready.")


    def _restore_metadata(self, reranked_results):
        for result in reranked_results:

            child_id = result.get("child_id")
            original = self.child_lookup.get(child_id)

            if not original:
                continue

            if not result.get("parent_id"):
                result["parent_id"] = original.get("parent_id")

            if not result.get("content"):
                result["content"] = original.get("content", "")

            if not result.get("section"):
                result["section"] = original.get("section")

            if not result.get("page"):
                result["page"] = original.get("page")

            if not result.get("element_type"):
                result["element_type"] = original.get("element_type")

        return reranked_results


    def retrieve(self, question: str) -> dict:

        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        print()
        print("=" * 75)
        print("RETRIEVAL PIPELINE")
        print("=" * 75)
        print(f"\nQuestion: {question}")

        print()
        print("[1/4] Dense + BM25 retrieval")

        dense_results = self.rrf.dense_retriever.search(
            question, top_k=DENSE_TOP_K
        )
        bm25_results = self.rrf.bm25_retriever.search(
            question, top_k=BM25_TOP_K
        )

        print(f"[dense] {len(dense_results)} results")
        print(f"[bm25]  {len(bm25_results)} results")

        print()
        print("[2/4] RRF fusion")

        rrf_results = self.rrf.fuse(dense_results, bm25_results)

        print(f"[rrf] {len(rrf_results)} candidates")

        print()
        print("[3/4] Cross-encoder reranking")

        reranked_results = reranker_module.rerank(
            question, rrf_results, top_k=RERANK_TOP_K
        )

        print(f"[reranker] {len(reranked_results)} results")

        reranked_results = self._restore_metadata(reranked_results)

        print()
        print("[4/4] Parent expansion")

        expanded_parents = parent_module.expand_to_parents(
            reranked_results, self.parent_lookup
        )

        expanded_parents = expanded_parents[:PARENT_TOP_K]

        print(f"[parent] {len(expanded_parents)} expanded parents")

        documents = parents_to_documents(expanded_parents)

        return {
            "question": question,
            "dense_results": dense_results,
            "bm25_results": bm25_results,
            "rrf_results": rrf_results,
            "reranked_results": reranked_results,
            "expanded_parents": expanded_parents,
            "documents": documents,
        }


    def invoke(self, question: str):
        return self.retrieve(question)["documents"]


    # ========================================================
    # LIVE INGESTION — used by /api/rag/ingest
    # ========================================================

    def ingest_document(self, pdf_path: Path, document_name: str) -> dict:
        """
        Runs a newly uploaded PDF through the SAME stages the offline
        Day 17-23 scripts use:

            PDF -> structured markdown      (01_ingestion/01_pdf_ingestion.py)
                -> typed elements           (01_ingestion/02_document_elements.py)
                -> hierarchical chunks      (02_chunking/01_hierarchical_chunker.py)
                -> embeddings + FAISS add   (05_retrieval/01_dense_retrieval.py)
                -> BM25 rebuild             (05_retrieval/02_bm25_retrieval.py)

        ...then merges the result into this pipeline's live in-memory
        state (hierarchical_data, parent_lookup, child_lookup) and
        persists hierarchical_chunks.json so the new document survives
        a server restart.

        No uvicorn restart is required for the new document to become
        searchable — the very next /api/rag/chat or /api/rag/chat/voice
        call already sees it.
        """

        print()
        print("=" * 75)
        print(f"[pipeline] INGESTING NEW DOCUMENT: {document_name}")
        print("=" * 75)

        # --------------------------------------------------------
        # Stage 1: PDF -> structured markdown
        # --------------------------------------------------------

        pdf_module = load_module_at(
            PROJECT_ROOT / "01_ingestion" / "01_pdf_ingestion.py",
            "day23_pdf_ingestion_live",
        )

        print("[pipeline] Converting PDF to structured markdown...")
        markdown_text = pdf_module.ingest_pdf_to_markdown(pdf_path)

        # --------------------------------------------------------
        # Stage 2: markdown -> typed elements
        # --------------------------------------------------------

        elements_module = load_module_at(
            PROJECT_ROOT / "01_ingestion" / "02_document_elements.py",
            "day23_document_elements_live",
        )

        print("[pipeline] Parsing document elements...")
        elements = elements_module.parse_document_elements(markdown_text)
        print(f"[pipeline] Elements found: {len(elements)}")

        if not elements:
            raise ValueError(
                "No content could be extracted from this PDF."
            )

        # --------------------------------------------------------
        # Stage 3: elements -> hierarchical parent/child chunks
        # --------------------------------------------------------

        chunker_module = load_module_at(
            PROJECT_ROOT / "02_chunking" / "01_hierarchical_chunker.py",
            "day23_hierarchical_chunker_live",
        )

        print("[pipeline] Building hierarchical chunks...")
        new_hierarchy = chunker_module.create_hierarchy(elements)
        new_children = chunker_module.flatten_children(new_hierarchy)

        if not new_children:
            raise ValueError(
                "No searchable content was extracted from this PDF."
            )

        # --------------------------------------------------------
        # Remap parent_id / child_id so they never collide with IDs
        # already in the index (both start counting from 0000), and
        # tag every parent + child with which document it came from
        # (this is what fixes source attribution for citations).
        # --------------------------------------------------------

        parent_offset = len(self.hierarchical_data.get("parents", []))

        for i, parent in enumerate(new_hierarchy):
            new_parent_num = parent_offset + i
            new_parent_id = f"parent_{new_parent_num:04d}"

            parent["parent_id"] = new_parent_id
            parent["source"] = document_name

            for child in parent["children"]:
                # child_id format is "child_{parent_idx:04d}_{child_idx:04d}"
                # -- keep the child-index suffix, replace the parent-index part.
                child_suffix = child["child_id"].split("_")[-1]
                child["child_id"] = f"child_{new_parent_num:04d}_{child_suffix}"
                child["parent_id"] = new_parent_id
                child["source"] = document_name

        print(
            f"[pipeline] New parents: {len(new_hierarchy)} | "
            f"New children: {len(new_children)}"
        )

        # --------------------------------------------------------
        # Stage 4: embed + append to the LIVE FAISS index
        # --------------------------------------------------------

        print("[pipeline] Embedding and indexing new children (FAISS)...")
        added_count = self.rrf.dense_retriever.add_chunks(
            new_children, persist=True
        )

        # --------------------------------------------------------
        # Stage 5: rebuild the LIVE BM25 index
        # --------------------------------------------------------

        print("[pipeline] Rebuilding BM25 index...")
        self.rrf.bm25_retriever.add_chunks(new_children)

        # --------------------------------------------------------
        # Update in-memory hierarchical data + lookups
        # --------------------------------------------------------

        self.hierarchical_data.setdefault("parents", []).extend(new_hierarchy)
        self.hierarchical_data.setdefault(
            "searchable_children", []
        ).extend(new_children)

        documents_list = self.hierarchical_data.setdefault(
            "documents",
            [self.hierarchical_data.get("document", "MACHINE LEARNING.pdf")],
        )
        documents_list.append(document_name)

        self.hierarchical_data["parent_count"] = len(
            self.hierarchical_data["parents"]
        )
        self.hierarchical_data["child_count"] = len(
            self.hierarchical_data["searchable_children"]
        )

        for parent in new_hierarchy:
            self.parent_lookup[parent["parent_id"]] = parent

        for child in new_children:
            self.child_lookup[child["child_id"]] = child

        # --------------------------------------------------------
        # Persist hierarchical_chunks.json so a restart doesn't
        # lose the newly ingested document. This is the single
        # writer for this file during ingestion -- BM25Retriever
        # and the parent expander both just read it at startup.
        # --------------------------------------------------------

        HIERARCHICAL_CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        HIERARCHICAL_CHUNKS_PATH.write_text(
            json.dumps(self.hierarchical_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(
            f"[pipeline] Persisted hierarchical_chunks.json "
            f"({self.hierarchical_data['child_count']} total children, "
            f"{len(documents_list)} documents)"
        )

        print(f"[pipeline] Ingestion complete: {document_name}")

        return {
            "document_name": document_name,
            "chunks_added": added_count,
            "parents_added": len(new_hierarchy),
            "total_chunks": self.hierarchical_data["child_count"],
        }


def parents_to_documents(expanded_parents):

    documents = []

    for parent in expanded_parents:

        parent_id = parent.get("parent_id", "")
        section = parent.get("section", "")
        page = parent.get("page", "")
        matched_child_id = parent.get("matched_child_id", "")
        reranker_score = parent.get("reranker_score")
        children = parent.get("children", [])

        content_parts = [
            child.get("content", "").strip()
            for child in children
            if child.get("content", "").strip()
        ]

        if not content_parts:
            continue

        page_content = "\n\n".join(content_parts)

        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "parent_id": parent_id,
                    "section": section,
                    "page": page,
                    "matched_child_id": matched_child_id,
                    "reranker_score": reranker_score,
                    # Was hardcoded to "MACHINE LEARNING.pdf" before --
                    # now reflects whichever document this parent
                    # actually came from (set in ingest_document(),
                    # defaulting to the original startup document for
                    # data predating this field).
                    "source": parent.get("source", "MACHINE LEARNING.pdf"),
                },
            )
        )

    print(f"[langchain] Documents created: {len(documents)}")

    return documents


def documents_to_context(documents):

    parts = []

    for rank, doc in enumerate(documents, start=1):
        meta = doc.metadata
        parts.append(
            f"[Source {rank}]\n"
            f"Parent ID: {meta.get('parent_id')}\n"
            f"Section: {meta.get('section')}\n"
            f"Page: {meta.get('page')}\n\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(parts)


def get_sources(documents):

    seen = set()
    sources = []

    for doc in documents:
        meta = doc.metadata
        key = (meta.get("section"), meta.get("page"))

        if key in seen:
            continue

        seen.add(key)
        sources.append(
            {
                "section": meta.get("section"),
                "page": meta.get("page"),
                "parent_id": meta.get("parent_id"),
            }
        )

    return sources


def display_documents(documents):

    print()
    print("=" * 75)
    print("LANGCHAIN DOCUMENTS")
    print("=" * 75)

    if not documents:
        print("No documents retrieved.")
        return

    for rank, doc in enumerate(documents, start=1):
        print()
        print(f"Document {rank}")
        print("-" * 75)
        print("Parent ID:", doc.metadata.get("parent_id"))
        print("Section  :", doc.metadata.get("section"))
        print("Page     :", doc.metadata.get("page"))
        print("Source   :", doc.metadata.get("source"))
        print()
        print(doc.page_content[:800])
        if len(doc.page_content) > 800:
            print("...")


def main():

    print()
    print("=" * 75)
    print("DAY 23 - LANGCHAIN RETRIEVAL PIPELINE TEST")
    print("=" * 75)

    question = input("\nEnter your question: ").strip()

    if not question:
        print("Question cannot be empty.")
        return

    pipeline = LangChainRetrievalPipeline()

    result = pipeline.retrieve(question)

    display_documents(result["documents"])


if __name__ == "__main__":
    main()