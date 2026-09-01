"""
DAY 23 - STEP 5.9b: FULL PIPELINE STREAMING TEST

Same as the full pipeline test, but uses
generate_answer_stream() instead of generate_answer(),
so we can confirm real retrieval + real Gemini streaming
work together before touching api.py.

Question
   ↓
07_langchain_retrieval.py  (Dense + BM25 + RRF + Reranker + Parent Expansion)
   ↓
LangChain Documents
   ↓
08_gemini_generation.py  (generate_answer_stream)
   ↓
Chunk
Chunk
Chunk
   ↓
Final Answer (assembled)
"""

from pathlib import Path
import sys
from importlib.util import module_from_spec, spec_from_file_location

RETRIEVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RETRIEVAL_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_module(filename, module_name):
    file_path = RETRIEVAL_DIR / filename
    spec = spec_from_file_location(module_name, file_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


print("Loading retrieval pipeline...")
retrieval_module = load_module("07_langchain_retrieval.py", "day23_retrieval")

print("Loading Gemini generation...")
gemini_module = load_module("08_gemini_generation.py", "day23_gemini")


def main():

    print()
    print("=" * 75)
    print("DAY 23 - FULL RAG PIPELINE (RETRIEVAL + STREAMING GENERATION)")
    print("=" * 75)

    question = input("\nEnter your question: ").strip()

    if not question:
        print("Question cannot be empty.")
        return

    pipeline = retrieval_module.LangChainRetrievalPipeline()

    result = pipeline.retrieve(question)

    documents = result["documents"]

    print()
    print("=" * 75)
    print("GENERATING ANSWER WITH GEMINI (STREAMING)")
    print("=" * 75)
    print()

    chunk_count = 0
    full_answer_parts = []

    for chunk in gemini_module.generate_answer_stream(
        question=question,
        documents=documents,
        chat_history=[],
    ):
        chunk_count += 1
        full_answer_parts.append(chunk)
        print(f"[chunk {chunk_count}]: {chunk!r}")

    answer = "".join(full_answer_parts)

    print()
    print("=" * 75)
    print(f"TOTAL CHUNKS RECEIVED: {chunk_count}")
    print("=" * 75)

    print()
    print("=" * 75)
    print("FINAL ASSEMBLED ANSWER")
    print("=" * 75)
    print()
    print(answer)

    print()
    print("=" * 75)
    print("SOURCES")
    print("=" * 75)

    sources = retrieval_module.get_sources(documents)

    for source in sources:
        print(f"- Section: {source['section']} | Page: {source['page']}")


if __name__ == "__main__":
    main()