"""
DAY 23 - STEP 5.8
GEMINI GENERATION

Purpose:
    Send LangChain Documents/context to Gemini.

This file does NOT perform retrieval.

Retrieval is handled by:

    01_dense_retrieval.py
    02_bm25_retrieval.py
    04_rrf_fusion.py
    05_reranker.py
    06_parent_expander.py
    07_langchain_retrieval.py

This file handles only:

    LangChain Documents
            ↓
        Prompt
            ↓
      Gemini through
         LangChain
            ↓
          Answer
"""

from pathlib import Path
import os

from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI
)

from langchain_core.prompts import (
    ChatPromptTemplate
)


# ============================================================
# ENVIRONMENT
# ============================================================

BACKEND_ROOT = Path(
    __file__
).resolve().parent.parent

ENV_PATH = BACKEND_ROOT / ".env"

load_dotenv(
    dotenv_path=ENV_PATH
)


# ============================================================
# CHECK API KEY
# ============================================================

def check_api_key():
    """
    Check whether Gemini API credentials exist.
    """

    google_key = os.getenv(
        "GOOGLE_API_KEY"
    )

    gemini_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not google_key and not gemini_key:

        raise RuntimeError(
            "\n"
            "Gemini API key was not found.\n\n"
            "Create:\n"
            f"    {ENV_PATH}\n\n"
            "and add:\n"
            "    GOOGLE_API_KEY=your_key_here\n"
        )

    return True


# ============================================================
# LOAD GEMINI
# ============================================================

def create_gemini():

    check_api_key()

    print()
    print(
        "[gemini] Loading Gemini through LangChain..."
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        temperature=0.2,
    )

    print(
        "[gemini] Gemini loaded."
    )

    return llm


# ============================================================
# FORMAT DOCUMENTS
# ============================================================

def documents_to_context(documents):
    """
    Convert LangChain Documents into a context string.
    """

    if not documents:
        return ""

    context_parts = []

    for rank, document in enumerate(
        documents,
        start=1
    ):

        parent_id = document.metadata.get(
            "parent_id",
            "unknown"
        )

        section = document.metadata.get(
            "section",
            "unknown"
        )

        page = document.metadata.get(
            "page",
            "unknown"
        )

        content = document.page_content.strip()

        block = (
            f"[Source {rank}]\n"
            f"Parent ID: {parent_id}\n"
            f"Section: {section}\n"
            f"Page: {page}\n\n"
            f"{content}"
        )

        context_parts.append(
            block
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# BUILD PROMPT
# ============================================================

def _build_prompt():
    """
    Build the shared Gemini prompt.

    Both normal generation and streaming generation
    use the same prompt.
    """

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an enterprise Retrieval-Augmented
Generation assistant.

Answer the user's question using ONLY the
retrieved document context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not present in the
   retrieved context, say that you do not
   know based on the provided documents.
4. Be precise and direct. Answer in 2-4
   sentences for straightforward questions.
   Do not pad the answer with extra
   background, restatements of the question,
   or minor details not central to the answer.
5. Only use a bulleted or numbered list if the
   user explicitly asks for a list, a
   step-by-step explanation, or multiple
   distinct items.
6. Use the conversation history only to
   understand follow-up questions.
7. Do not mention internal retrieval
   implementation unless explicitly asked.
"""
            ),
            (
                "human",
                """
Retrieved document context:

{context}

Conversation history:

{history}

User question:

{question}

Answer:
"""
            ),
        ]
    )


# ============================================================
# GENERATE COMPLETE ANSWER
# ============================================================

def generate_answer(
    question,
    documents,
    chat_history=None
):
    """
    Generate the complete answer using Gemini.

    This is the normal non-streaming version.
    """

    if not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    context = documents_to_context(documents)

    if not context:
        return (
            "I could not find relevant information "
            "in the retrieved documents."
        )

    if chat_history is None:
        chat_history = []

    history_text = "\n".join(
        f"{role}: {text}"
        for role, text in chat_history
    )

    prompt = _build_prompt()

    llm = create_gemini()

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "history": history_text,
            "question": question,
        }
    )

    content = response.content

    if isinstance(content, list):

        parts = []

        for part in content:

            if isinstance(part, str):
                parts.append(part)

            elif isinstance(part, dict):
                parts.append(
                    part.get("text", "")
                )

        content = "".join(parts)

    return content.strip()


# ============================================================
# GENERATE STREAMING ANSWER
# ============================================================

def generate_answer_stream(
    question,
    documents,
    chat_history=None
):
    """
    Stream Gemini's answer chunk-by-chunk.

    Instead of waiting for the complete answer,
    this function yields text as Gemini generates it.

    Example:

        Gemini
           ↓
        "Machine learning"
           ↓
        " is a field"
           ↓
        " of artificial intelligence."
    """

    if not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    context = documents_to_context(documents)

    if not context:
        yield (
            "I could not find relevant information "
            "in the retrieved documents."
        )
        return

    if chat_history is None:
        chat_history = []

    history_text = "\n".join(
        f"{role}: {text}"
        for role, text in chat_history
    )

    prompt = _build_prompt()

    llm = create_gemini()

    chain = prompt | llm

    for chunk in chain.stream(
        {
            "context": context,
            "history": history_text,
            "question": question,
        }
    ):

        if not chunk.content:
            continue

        content = chunk.content

        if isinstance(content, list):

            parts = []

            for part in content:

                if isinstance(part, str):
                    parts.append(part)

                elif isinstance(part, dict):
                    parts.append(
                        part.get("text", "")
                    )

            content = "".join(parts)

        if content:
            yield content


# ============================================================
# QUESTION REWRITING (for follow-up questions)
# ============================================================

def rewrite_question(question: str, chat_history) -> str:
    """
    Turn a follow-up like "what about its disadvantages?" into
    a standalone question the retriever can actually search on,
    using the conversation so far.

    If there is no chat history yet (first question), the
    original question is returned unchanged.
    """

    if not chat_history:
        return question

    llm = create_gemini()

    history_text = "\n".join(
        f"{role}: {text}" for role, text in chat_history
    )

    prompt = (
        "Given this conversation so far:\n"
        f"{history_text}\n\n"
        "Rewrite this follow-up question so it can be understood "
        "on its own, without needing the conversation above. "
        "If the question is already standalone, return it "
        "unchanged. Only output the rewritten question, "
        "nothing else.\n\n"
        f"Follow-up question: {question}\n"
        f"Standalone question:"
    )

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        content = "".join(parts)

    rewritten = content.strip()

    return rewritten if rewritten else question

# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 75)
    print("DAY 23 - GEMINI GENERATION TEST")
    print("=" * 75)

    try:

        # ----------------------------------------------------
        # Test normal generation
        # ----------------------------------------------------

        print()
        print("[TEST 1] Normal Gemini generation")
        print("-" * 75)

        llm = create_gemini()

        response = llm.invoke(
            "What is machine learning? Answer in 2 sentences."
        )

        print()
        print("Gemini response:")
        print(response.content)

        # ----------------------------------------------------
        # Test streaming generation
        # ----------------------------------------------------

        print()
        print("=" * 75)
        print("[TEST 2] Gemini streaming generation")
        print("=" * 75)

        print()
        print("Streaming response:")
        print("-" * 75)

        # For this isolated test we don't need real RAG
        # documents. We will provide a small test document.
        from langchain_core.documents import Document

        test_documents = [
            Document(
                page_content=(
                    "Machine learning is a branch of artificial "
                    "intelligence that allows computers to learn "
                    "patterns from data."
                ),
                metadata={
                    "parent_id": "test",
                    "section": "Machine Learning",
                    "page": 1,
                },
            )
        ]

        chunk_count = 0
        for chunk in generate_answer_stream(
            question="What is machine learning?",
            documents=test_documents,
            chat_history=[],
        ):
            chunk_count += 1
            print(f"\n[chunk {chunk_count}]: {chunk!r}")

        print(f"\nTotal chunks received: {chunk_count}")

        print()
        print("-" * 75)
        print("[TEST] Streaming completed successfully.")

    except Exception as e:

        print()
        print("[gemini] ERROR:")
        print(e)
