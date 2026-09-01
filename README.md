# Day 23 — Voice RAG Backend

Backend service for a Retrieval-Augmented Generation (RAG) system with
streaming text responses and voice (speech-to-text + text-to-speech)
support.

## Architecture

```
Question (text or voice)
        |
        v
FastAPI (api.py)
        |
        +-- Speech-to-text (Whisper, voice input only)
        |
        +-- Retrieval pipeline
        |     Dense + BM25 + RRF fusion + Reranker + Parent expansion
        |
        +-- Gemini (streaming generation)
        |     yields answer chunks as they're generated
        |
        +-- Text-only: chunks streamed directly to frontend (SSE)
        |
        +-- Voice: each completed sentence is also sent to TTS
              (XTTS) and the resulting audio is streamed back
              alongside the text (SSE)
```

## Key components

- `api.py` — FastAPI app; retrieval + generation orchestration;
  SSE streaming endpoints for both text and voice chat
- `tts_api.py` — Text-to-speech service (XTTS), synthesizes
  sentence-by-sentence audio
- `05_retrieval/07_langchain_retrieval.py` — Combined retrieval
  pipeline (dense, BM25, RRF, reranking, parent expansion)
- `05_retrieval/08_gemini_generation.py` — Gemini answer
  generation, both streaming and non-streaming
- `06_voice/01_speech_to_text.py` — Whisper-based speech-to-text

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/rag/ingest` | Upload and index a PDF |
| `POST /api/rag/sources` | List ingested documents |
| `POST /api/rag/chat` | Text chat (non-streaming) |
| `POST /api/rag/chat/stream` | Text chat (SSE streaming) |
| `POST /api/rag/chat/voice` | Voice chat (SSE streaming text + audio) |
| `POST /api/transcribe` | Speech-to-text (Whisper) |
| `POST /api/tts/speak` | Full-text TTS (non-streaming) |
| `POST /api/tts/speak-stream` | Sentence-by-sentence TTS |
| `POST /api/tts/speak-one` | Synthesize a single sentence |

## Setup

```bash
python -m venv venv18
source venv18/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the backend root with:

```
GOOGLE_API_KEY=your_key_here
```

## Running

Start both services (in separate terminals):

```bash
# Terminal 1: main API (retrieval + generation + orchestration)
uvicorn api:app --reload --port 8000

# Terminal 2: TTS service
python tts_api.py
```

## Notes

- TTS uses XTTS-v2 for voice cloning; a reference voice sample is
  required at `06_voice/reference_voice.wav`.
- Speech-to-text uses faster-whisper (`small.en` model) locked to
  English for accuracy.
- Logs are written to `tts_service.log` and `rag_api.log`,
  including per-chunk RTF (Real-Time Factor) for TTS performance
  monitoring.
```

