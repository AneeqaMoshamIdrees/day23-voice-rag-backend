"""
DAY 23 - MAIN API

Matches the existing frontend contract exactly:

    GET  /api/rag/sources
    POST /api/rag/ingest        (stubbed for now)
    POST /api/rag/chat
    POST /api/rag/chat/voice    (combined streamed text + streamed audio)
    POST /api/transcribe
"""

import os
import re
import json
import base64
import tempfile
import time
from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import logging

logging.basicConfig(
    filename="rag_api.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ============================================================
# PATHS
# ============================================================

BACKEND_ROOT = Path(__file__).resolve().parent
RETRIEVAL_DIR = BACKEND_ROOT / "05_retrieval"
VOICE_DIR = BACKEND_ROOT / "06_voice"
IMAGES_DIR = BACKEND_ROOT / "data" / "images"

TTS_SERVICE_URL = "http://127.0.0.1:8001/api/tts/speak-one"


# ============================================================
# LOAD NUMBERED MODULES
# ============================================================

def load_module(file_path: Path, module_name: str):
    if not file_path.exists():
        raise FileNotFoundError(f"Module not found: {file_path}")
    spec = spec_from_file_location(module_name, file_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


print()
print("=" * 75)
print("STARTING DAY 23 API")
print("=" * 75)

print()
print("[api] Loading retrieval pipeline...")
retrieval_module = load_module(
    RETRIEVAL_DIR / "07_langchain_retrieval.py", "day23_retrieval"
)

print()
print("[api] Loading Gemini generation...")
gemini_module = load_module(
    RETRIEVAL_DIR / "08_gemini_generation.py", "day23_gemini"
)

print()
print("[api] Loading speech-to-text...")
stt_module = load_module(
    VOICE_DIR / "01_speech_to_text.py", "day23_stt"
)

print()
print("[api] Initializing RAG pipeline...")
rag_pipeline = retrieval_module.LangChainRetrievalPipeline()
print()
print("[api] RAG pipeline ready.")

transcription_service = stt_module.transcription_service

print()
print("[api] All components loaded. API is ready.")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="Day 23 Voice RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve figure images so the frontend can do:
#   API_BASE + "/" + "images/figure_017.png"
if IMAGES_DIR.exists():
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


# ============================================================
# SESSION MEMORY
# ============================================================

sessions: dict[str, list] = {}


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):
    session_id: str
    question: str
    voice: bool = True  # False = text-only, skip TTS generation entirely


# ============================================================
# HELPERS: match old frontend's expected shapes
# ============================================================

def format_sources_string(documents):
    """
    Frontend parses the FIRST line with regex:
        /-\s*(.+?),\s*page\s*(\S+)/i
    So we must return a "- file, page N" style string,
    one unique (file, page) per line, most relevant first.
    """
    seen = set()
    lines = []

    for doc in documents:
        source_file = doc.metadata.get("source", "MACHINE LEARNING.pdf")
        page = doc.metadata.get("page", "unknown")
        key = (source_file, page)

        if key not in seen:
            seen.add(key)
            lines.append(f"- {source_file}, page {page}")

    return "\n".join(lines)


def extract_image_paths(reranked_results, max_images=3):
    """
    Only pull images that were part of the SPECIFIC reranked
    child chunks (the ones that actually scored well for this
    question) -- not every image in the whole matched parent
    section, which was causing unrelated figures to appear.
    """
    image_paths = []
    seen = set()

    for result in reranked_results:
        element_type = result.get("element_type")
        content = result.get("content", "").strip()

        if element_type == "image" and content:
            if content not in seen:
                seen.add(content)
                image_paths.append(content)

    return image_paths[:max_images]


def extract_complete_sentences(buffer: str):
    """
    Given accumulated streamed text, returns (complete_sentences, remainder).
    Splits on sentence-ending punctuation followed by a space or end of string.
    """
    matches = list(re.finditer(r'[^.!?]*[.!?]+(?:\s+|$)', buffer))
    if not matches:
        return [], buffer
    last_end = matches[-1].end()
    sentences = [m.group().strip() for m in matches if m.group().strip()]
    remainder = buffer[last_end:]
    return sentences, remainder


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {"status": "Day 23 Voice RAG API is running"}


# ============================================================
# SOURCES LIST
# ============================================================

@app.get("/api/rag/sources")
def list_sources():
    total_children = len(
        rag_pipeline.hierarchical_data.get("searchable_children", [])
    )

    document_name = rag_pipeline.hierarchical_data.get(
        "document", "MACHINE LEARNING.pdf"
    )

    return {
        "sources": [document_name],
        "total_chunks": total_children,
    }


# ============================================================
# INGEST (stub)
# ============================================================

@app.post("/api/rag/ingest")
async def ingest_document(file: UploadFile = File(...)):
    raise HTTPException(
        status_code=501,
        detail=(
            "Dynamic document ingestion is not yet connected "
            "to the Day 23 hierarchical pipeline. The indexed "
            "document is currently fixed at startup."
        ),
    )


# ============================================================
# CHAT ENDPOINT (text only, non-streaming)
# ============================================================

@app.post("/api/rag/chat")
def rag_chat(request: ChatRequest):

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    history = sessions.get(request.session_id, [])

    print()
    print(f"[api] /api/rag/chat  question: {request.question}")

    standalone_question = gemini_module.rewrite_question(
        request.question, history
    )

    rewritten_flag = (
        standalone_question
        if standalone_question.strip() != request.question.strip()
        else None
    )

    if rewritten_flag:
        print(f"[api] rewritten question: {standalone_question}")

    result = rag_pipeline.retrieve(standalone_question)
    documents = result["documents"]
    reranked_results = result["reranked_results"]

    answer = gemini_module.generate_answer(
        question=standalone_question,
        documents=documents,
        chat_history=history,
    )

    sources_string = format_sources_string(documents)
    image_paths = extract_image_paths(reranked_results)

    history.append(("Human", request.question))
    history.append(("AI", answer))
    sessions[request.session_id] = history

    return {
        "answer": answer,
        "sources": sources_string,
        "rewritten_question": rewritten_flag,
        "images": image_paths,
    }


# ============================================================
# COMBINED CHAT + STREAMING VOICE ENDPOINT
#
# Day 23 requirement: the chat API returns both the streamed
# text answer AND streamed audio TOGETHER, in one response.
#
# Implementation: a single Server-Sent Events (SSE) stream
# with four event types, always in this order per request:
#
#   event: meta        -> sent once, immediately
#                          { sources, rewritten_question, images }
#   event: text_chunk  -> sent repeatedly as Gemini streams
#                          { text }
#   event: audio_chunk -> sent once per completed sentence,
#                          interleaved with text_chunk events
#                          { audio: <base64-encoded wav> }
#   event: done         -> sent once, at the very end
#                          { answer }
#
# The frontend reads this single stream and updates the text
# bubble and the audio queue from the SAME events as they
# arrive -- that is what makes text and audio "simultaneous":
# both are driven by one connection, not two separate requests.
# ============================================================

@app.post("/api/rag/chat/stream")
def rag_chat_stream(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    history = sessions.get(request.session_id, [])

    standalone_question = gemini_module.rewrite_question(request.question, history)
    rewritten_flag = (
        standalone_question
        if standalone_question.strip() != request.question.strip()
        else None
    )

    result = rag_pipeline.retrieve(standalone_question)
    documents = result["documents"]
    reranked_results = result["reranked_results"]

    sources_string = format_sources_string(documents)
    image_paths = extract_image_paths(reranked_results)

    def event_stream():
        meta = {
            "sources": sources_string,
            "rewritten_question": rewritten_flag,
            "images": image_paths,
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        full_answer = []
        for piece in gemini_module.generate_answer_stream(
            question=standalone_question,
            documents=documents,
            chat_history=history,
        ):
            full_answer.append(piece)
            yield f"event: text_chunk\ndata: {json.dumps({'text': piece})}\n\n"

        answer = "".join(full_answer)
        history.append(("Human", request.question))
        history.append(("AI", answer))
        sessions[request.session_id] = history

        yield f"event: done\ndata: {json.dumps({'answer': answer})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/rag/chat/voice")
def rag_chat_voice(request: ChatRequest):

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    history = sessions.get(request.session_id, [])

    print()
    print(f"[api] /api/rag/chat/voice  question: {request.question}")
    request_start = time.time()
    logging.info(f"Voice chat request received | question={request.question}")

    standalone_question = gemini_module.rewrite_question(request.question, history)
    rewritten_flag = (
        standalone_question
        if standalone_question.strip() != request.question.strip()
        else None
    )

    retrieval_start = time.time()
    result = rag_pipeline.retrieve(standalone_question)
    retrieval_time = time.time() - retrieval_start

    documents = result["documents"]
    reranked_results = result["reranked_results"]

    logging.info(
        f"Retrieval completed | time={retrieval_time:.2f}s | docs_found={len(documents)}"
    )

    sources_string = format_sources_string(documents)
    image_paths = extract_image_paths(reranked_results)

    def event_stream():
        # --- meta, sent immediately, before any generation ---
        meta = {
            "sources": sources_string,
            "rewritten_question": rewritten_flag,
            "images": image_paths,
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        full_answer = []
        sentence_buffer = ""
        generation_start = time.time()

        # --- text_chunk + audio_chunk, interleaved ---
        for piece in gemini_module.generate_answer_stream(
            question=standalone_question,
            documents=documents,
            chat_history=history,
        ):
            full_answer.append(piece)
            yield f"event: text_chunk\ndata: {json.dumps({'text': piece})}\n\n"

            sentence_buffer += piece
            sentences, sentence_buffer = extract_complete_sentences(sentence_buffer)

            if request.voice:
                for sentence in sentences:
                    try:
                        tts_resp = requests.post(
                            TTS_SERVICE_URL,
                            json={"text": sentence},
                            timeout=120,
                        )
                        if tts_resp.status_code == 200:
                            audio_b64 = base64.b64encode(tts_resp.content).decode("utf-8")
                            yield f"event: audio_chunk\ndata: {json.dumps({'audio': audio_b64})}\n\n"
                        else:
                            print(f"[api] TTS returned status {tts_resp.status_code} for sentence")
                    except requests.exceptions.RequestException as e:
                        print(f"[api] TTS request failed for sentence: {e}")

        # Flush whatever text didn't end in sentence punctuation.
        if request.voice and sentence_buffer.strip():
            try:
                tts_resp = requests.post(
                    TTS_SERVICE_URL,
                    json={"text": sentence_buffer.strip()},
                    timeout=120,
                )
                if tts_resp.status_code == 200:
                    audio_b64 = base64.b64encode(tts_resp.content).decode("utf-8")
                    yield f"event: audio_chunk\ndata: {json.dumps({'audio': audio_b64})}\n\n"
            except requests.exceptions.RequestException as e:
                print(f"[api] TTS request failed for final buffer: {e}")

        generation_time = time.time() - generation_start
        answer = "".join(full_answer)

        history.append(("Human", request.question))
        history.append(("AI", answer))
        sessions[request.session_id] = history

        total_time = time.time() - request_start
        logging.info(
            f"Voice chat request completed | total_time={total_time:.2f}s | "
            f"retrieval={retrieval_time:.2f}s | generation={generation_time:.2f}s"
        )

        # --- done, sent once, at the very end ---
        yield f"event: done\ndata: {json.dumps({'answer': answer})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering if any sits in front
        },
    )


# ============================================================
# TRANSCRIBE ENDPOINT
# ============================================================

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    allowed_extensions = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".mp4"}
    extension = os.path.splitext(file.filename)[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, detail=f"Unsupported audio format: {extension}"
        )

    temp_path = None

    try:
        audio_bytes = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as f:
            f.write(audio_bytes)
            temp_path = f.name

        result = transcription_service.transcribe(temp_path)

        return {
            "success": True,
            "text": result["text"],
            "language": result["language"],
            "language_probability": result["language_probability"],
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(exc)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)