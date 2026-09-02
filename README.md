# Day 23 — Voice RAG Backend

A Retrieval-Augmented Generation system with hierarchical document
retrieval, conversational memory, and streaming voice output using
a cloned reference voice.

## Architecture Overview

This backend consists of **two independent services** that must run
simultaneously in **two separate Python virtual environments**,
because the TTS library (`coqui-tts`) requires a different
`transformers` version than the RAG/embedding stack, and mixing
them in one environment breaks both.

┌─────────────────────────────────────────────────────────┐
│ Frontend (React) │
│ http://localhost:5173 │
└───────────────────────────┬───────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────┐
│ Main API — venv18 │
│ http://127.0.0.1:8000 │
│ │
│ - Document retrieval (Dense + BM25 + RRF + Reranker │
│ + Hierarchical Parent Expansion) │
│ - Gemini (via LangChain) — answer generation, streaming │
│ - Whisper (faster-whisper) — speech-to-text │
│ - Internally calls the TTS service below for audio │
└───────────────────────────┬───────────────────────────────┘
│ (internal HTTP call)
▼
┌─────────────────────────────────────────────────────────┐
│ TTS Service — venv_tts │
│ http://127.0.0.1:8001 │
│ │
│ - XTTS v2 (Coqui) — voice cloning + speech synthesis │
│ - Uses a recorded reference voice sample │
└─────────────────────────────────────────────────────────┘


The main API and the TTS service are genuinely separate processes,
each with their own virtual environment, running on different
ports. The main API calls the TTS service over local HTTP — they
never share a Python process or dependency set.

---

## 1. Prerequisites

- Python 3.12
- `ffmpeg` installed and on PATH (`sudo apt install ffmpeg` on Ubuntu/Debian)
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com))
- A short (6–15 second) clean audio recording of your own voice, for
  voice cloning
- ~7GB+ free RAM recommended (XTTS v2 is memory-heavy; the project
  has been tested and runs, with some slowness, on a machine with
  7.2GB total RAM)

---

## 2. Directory Structure

backend/
│
├── 01_ingestion/ # PDF -> structured Markdown -> document elements
├── 02_chunking/ # Hierarchical parent/child chunking
├── 03_embeddings/ # BAAI/bge-large-en-v1.5 embedding generation
├── 04_vector_databases/ # FAISS index construction
├── 05_retrieval/
│ ├── 01_dense_retrieval.py
│ ├── 02_bm25_retrieval.py
│ ├── 03_hybrid_retrieval.py
│ ├── 04_rrf_fusion.py
│ ├── 05_reranker.py
│ ├── 06_parent_expander.py
│ ├── 07_langchain_retrieval.py # main retrieval orchestrator
│ ├── 08_gemini_generation.py # Gemini generation + streaming
│ └── 09_full_pipeline_test.py # standalone end-to-end test script
├── 06_voice/
│ ├── 01_speech_to_text.py # faster-whisper wrapper
│ ├── 02_convert_to_wav.py # audio format converter utility
│ └── reference_voice.wav # YOUR recorded voice sample (required)
│
├── data/
│ ├── MACHINE LEARNING.pdf # source document
│ ├── structured.md # Docling output (Stage 1)
│ ├── images/ # extracted figures
│ ├── structured_documents/ # document_elements.json, hierarchical_chunks.json
│ ├── embeddings/ # hierarchical_embeddings.json
│ └── vector_databases/faiss/ # index.faiss, metadata.json
│
├── docs/
│ └── tts_latency_findings.md # streaming vs non-streaming latency results
│
├── api.py # Main API entry point (port 8000)
├── tts_api.py # TTS service entry point (port 8001)
├── requirements.txt # dependencies for venv18 (main API)
├── .env # GOOGLE_API_KEY (create this yourself, see below)
├── venv18/ (or wherever your main venv lives)
└── venv_tts/ # separate venv for TTS


---

## 3. One-Time Setup

### 3a. Main environment (RAG + Gemini + Whisper)

```bash
python3 -m venv venv18
source venv18/bin/activate
pip install -r requirements.txt
```

### 3b. TTS environment (separate, isolated)

```bash
python3 -m venv venv_tts
source venv_tts/bin/activate
pip install coqui-tts torch torchaudio "coqui-tts[codec]"
pip install "transformers==4.57.1"
pip install fastapi uvicorn python-multipart requests
```

> **Why two environments?** `coqui-tts` requires `transformers>=4.57`
> but breaks with `transformers>=5.0`. The main RAG stack (BGE
> embeddings, cross-encoder reranker) uses a newer `transformers`
> release. These two requirements are incompatible in one
> environment — hence the split.

### 3c. Environment variables

Create `backend/.env`:

GOOGLE_API_KEY=your_gemini_api_key_here


### 3d. Reference voice

Record or obtain a clean 6–15 second WAV/audio file of a voice you
want to clone. Convert it to the format XTTS expects:

```bash
source venv18/bin/activate   # ffmpeg conversion doesn't need venv_tts
python3 06_voice/02_convert_to_wav.py /path/to/your/recording.ext
```

This produces a WAV file (22050 Hz, mono). Rename/move it to:

06_voice/reference_voice.wav


This file is **required** — both `tts_api.py` and the main pipeline
expect it at this exact path.

### 3e. Build the document index (one-time, or when the source PDF changes)

Place your PDF at `data/MACHINE LEARNING.pdf` (or edit the path in
`01_ingestion/01_pdf_ingestion.py`), then run each stage in order,
using `venv18`:

```bash
source venv18/bin/activate

python3 01_ingestion/01_pdf_ingestion.py
python3 01_ingestion/02_document_elements.py
python3 02_chunking/01_hierarchical_chunker.py
python3 03_embeddings/01_embedding_generator.py
python3 04_vector_databases/01_faiss_vector_database.py
```

Each stage reads the previous stage's output from `data/` and
writes its own output there. You only need to re-run this sequence
if you change the source document.

---

## 4. Running the Backend (every time)

You need **two terminals**, each with its own environment active.

### Terminal 1 — Main API (port 8000)

```bash
cd backend
source venv18/bin/activate
uvicorn api:app --reload
```

Wait for:

[api] All components loaded. API is ready.
Uvicorn running on http://127.0.0.1:8000


### Terminal 2 — TTS Service (port 8001)

```bash
cd backend
source venv_tts/bin/activate
uvicorn tts_api:app --port 8001
```

(`--reload` is optional here — omit it once the file is stable, to
reduce overhead on memory-constrained machines.)

Wait for:

[tts] Model loaded. Service is ready.
Uvicorn running on http://127.0.0.1:8001


**Both must be running** for voice features to work. If the TTS
service is down, `/api/rag/chat/voice` still returns the text
answer — it just won't include audio.

---

## 5. API Endpoints

| Method | Path                    | Purpose                                              |
|--------|-------------------------|-------------------------------------------------------|
| GET    | `/`                      | Health check                                          |
| GET    | `/api/rag/sources`        | List indexed documents + total passage count           |
| POST   | `/api/rag/ingest`         | Stubbed (501) — live ingestion not enabled; see notes  |
| POST   | `/api/rag/chat`           | Legacy non-streaming text chat                          |
| POST   | `/api/rag/chat/voice`      | **Main endpoint.** SSE stream: retrieval + streamed text + streamed cloned-voice audio. Pass `"voice": false` to skip audio generation. |
| POST   | `/api/transcribe`          | Upload audio, get back a Whisper transcript             |
| POST   | `/api/tts/speak`            | (TTS service, port 8001) Non-streaming single-shot synthesis |
| POST   | `/api/tts/speak-one`         | (TTS service, port 8001) Synthesize one sentence, used internally by `/api/rag/chat/voice` |
| POST   | `/api/tts/speak-stream`       | (TTS service, port 8001) Standalone sentence-by-sentence streaming |

### `/api/rag/chat/voice` request body

```json
{
  "session_id": "any-string-you-choose",
  "question": "What is the LMS training rule?",
  "voice": true
}
```

### `/api/rag/chat/voice` response (Server-Sent Events)

event: meta
data: {"sources": "...", "rewritten_question": null, "images": []}

event: text_chunk
data: {"text": "Based on the "}

event: text_chunk
data: {"text": "provided documents..."}

event: audio_chunk
data: {"audio": "<base64 WAV bytes for one completed sentence>"}

event: done
data: {"answer": "<the complete final answer text>"}


---

## 6. Known Limitations (documented, not bugs)

- **CPU-only inference is slow.** XTTS v2 was tested with no GPU
  available. Per-sentence synthesis ranges roughly 15–50 seconds
  depending on sentence length and system load. See
  `docs/tts_latency_findings.md` for measured numbers.
- **Accent drift.** XTTS v2's zero-shot cloning preserves voice
  timbre well but tends to shift toward a generic English accent
  rather than fully preserving the reference speaker's accent —
  a known model limitation, not a bug in this implementation.
- **Live document upload is disabled** (`/api/rag/ingest` returns
  `501`). Per the project's Day 21 amendment, the RAG index is
  built once from a fixed source PDF rather than rebuilt on every
  upload.
- **7GB RAM is the practical minimum tested.** Running the TTS
  service alongside a code editor and browser can strain memory on
  smaller machines; consider closing other heavy applications while
  running both backend services together.

---

## 7. Troubleshooting

**"API key required for Gemini Developer API"**
`.env` is missing or `GOOGLE_API_KEY` isn't set. Confirm the file
exists at `backend/.env` and contains a valid key.

**TTS service won't import (`ModuleNotFoundError` / transformers
import errors)**
Confirm you're in `venv_tts`, not `venv18`, and that
`transformers==4.57.1` specifically is installed (not the latest
version — newer releases removed functions `coqui-tts` depends on).

**Voice endpoint returns text but no audio**
Check that `tts_api.py` is actually running on port 8001. The main
API logs `[api] TTS request failed for sentence: ...` if it can't
reach it — check Terminal 1's logs.

**System freezes / editor crashes while TTS is running**
Likely RAM exhaustion. Close other applications (especially a
second code editor, if running one), and consider running
`tts_api.py` without `--reload`.
