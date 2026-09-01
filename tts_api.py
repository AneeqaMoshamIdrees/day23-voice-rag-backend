"""
DAY 23 - TTS SERVICE (Step 4: real streaming endpoint)

Two endpoints now exist:

    POST /api/tts/speak
        Non-streaming. Waits for the ENTIRE text to be
        synthesized, then sends back one audio file.

    POST /api/tts/speak-stream
        Streaming. Splits text into sentences, synthesizes
        them one at a time, and sends each chunk's audio
        as soon as it's ready -- instead of waiting for
        everything to finish first.
"""

import re
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from TTS.api import TTS
import logging
import wave

logging.basicConfig(
    filename="tts_service.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def get_wav_duration(path):
    """Returns the duration of a WAV file in seconds."""
    with wave.open(path, "rb") as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate) if rate > 0 else 0
# ============================================================
# PATHS
# ============================================================

BACKEND_ROOT = Path(__file__).resolve().parent
REFERENCE_VOICE = BACKEND_ROOT / "06_voice" / "reference_voice.wav"

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"


# ============================================================
# LOAD MODEL ONCE, AT STARTUP
# ============================================================

print()
print("=" * 75)
print("STARTING TTS SERVICE")
print("=" * 75)

print()
print(f"[tts] Reference voice: {REFERENCE_VOICE}")
print(f"[tts] Loading XTTS model (this takes ~20 seconds)...")

tts_model = TTS(MODEL_NAME)

print("[tts] Model loaded. Service is ready.")


# ============================================================
# SENTENCE SPLITTER
# ============================================================

def clean_text_for_speech(text: str) -> str:
    """
    Strips markdown formatting that makes no sense spoken
    aloud, WITHOUT removing any actual content -- nothing is
    shortened or summarized here, only formatting symbols
    are removed so the words themselves come through clean.

    Examples:
        "* **Goal:** Find patterns"    -> "Goal: Find patterns"
        "1. First point"               -> "First point"
        "# Heading"                    -> "Heading"
        "`code`"                       -> "code"
    """

    cleaned = text

    # Remove markdown bold/italic markers (**text**, *text*, _text_)
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*(.+?)\*", r"\1", cleaned)
    cleaned = re.sub(r"_(.+?)_", r"\1", cleaned)

    # Remove markdown headings (#, ##, ###...)
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)

    # Remove bullet markers at the start of a line (*, -, +)
    cleaned = re.sub(r"^\s*[\*\-\+]\s+", "", cleaned, flags=re.MULTILINE)

    # Remove numbered list markers at the start of a line (1., 2), etc.)
    cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", cleaned, flags=re.MULTILINE)

    # Remove inline code backticks
    cleaned = cleaned.replace("`", "")

    # Remove leftover markdown table pipes
    cleaned = cleaned.replace("|", ", ")

    # Collapse multiple blank lines/spaces into single spaces,
    # since TTS reads based on sentence punctuation, not
    # visual line breaks.
    cleaned = re.sub(r"\n+", ". ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)

    return cleaned.strip()


def split_into_sentences(text: str) -> list[str]:
    """
    Splits CLEANED text into sentences on '.', '!', '?'.

    Also merges any fragment shorter than 15 characters into
    the next sentence, so short leftover pieces (like a bare
    "2" from an old numbered list) don't become their own
    tiny, wasteful synthesis request.

    Example:
        "Hello. How are you?" -> ["Hello.", "How are you?"]
    """

    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    raw_sentences = [s.strip() for s in raw_sentences if s.strip()]

    # Merge very short fragments into the following sentence.
    merged = []
    buffer = ""

    for sentence in raw_sentences:
        candidate = (buffer + " " + sentence).strip() if buffer else sentence

        if len(candidate) < 15:
            buffer = candidate
        else:
            merged.append(candidate)
            buffer = ""

    if buffer:
        if merged:
            merged[-1] = (merged[-1] + " " + buffer).strip()
        else:
            merged.append(buffer)

    return merged


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="Day 23 TTS Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class SpeakRequest(BaseModel):
    text: str


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {"status": "TTS service is running"}


# ============================================================
# NON-STREAMING ENDPOINT (unchanged from before)
# ============================================================

@app.post("/api/tts/speak")
def speak(request: SpeakRequest):
    """
    Synthesizes the ENTIRE text as one audio file.
    Caller must wait for everything before hearing anything.
    """

    text = request.text.strip()

    if not text:
        return {"error": "Text cannot be empty."}

    print()
    print(f"[tts] (non-streaming) Synthesizing: {text[:80]}...")

    start = time.time()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    output_path = temp_file.name
    temp_file.close()

    tts_model.tts_to_file(
        text=text,
        speaker_wav=str(REFERENCE_VOICE),
        language="en",
        file_path=output_path,
    )

    elapsed = time.time() - start
    print(f"[tts] (non-streaming) Done in {elapsed:.1f}s")

    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename="speech.wav",
    )

from fastapi import Response

@app.post("/api/tts/speak-one")
def speak_one(request: SpeakRequest):
    text = request.text.strip()
    if not text:
        return {"error": "Text cannot be empty."}

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    chunk_path = temp_file.name
    temp_file.close()

    start = time.time()
    tts_model.tts_to_file(
        text=text,
        speaker_wav=str(REFERENCE_VOICE),
        language="en",
        file_path=chunk_path,
    )
    elapsed = time.time() - start
    audio_duration = get_wav_duration(chunk_path)
    rtf = elapsed / audio_duration if audio_duration > 0 else 0

    logging.info(
        f"TTS single-sentence | text_len={len(text)} | "
        f"synth_time={elapsed:.2f}s | audio_duration={audio_duration:.2f}s | RTF={rtf:.2f}"
    )

    with open(chunk_path, "rb") as f:
        audio_bytes = f.read()

    return Response(content=audio_bytes, media_type="audio/wav")

# ============================================================
# STREAMING ENDPOINT (new)
# ============================================================

@app.post("/api/tts/speak-stream")
def speak_stream(request: SpeakRequest):
    """
    Splits text into sentences and synthesizes them ONE AT
    A TIME, sending each chunk's audio bytes as soon as it's
    ready -- instead of waiting for the whole answer.

    The response is a sequence of complete WAV files, one
    per sentence, sent back to back. The frontend reads them
    one at a time as they arrive.
    """

    text = request.text.strip()

    if not text:
        return {"error": "Text cannot be empty."}

    sentences = split_into_sentences(text)

    print()
    print(f"[tts] (streaming) {len(sentences)} sentence(s) to synthesize")
    logging.info(f"TTS stream request started | {len(sentences)} sentences | text_len={len(text)} chars")

    def audio_chunk_generator():
        """
        This function runs DURING the response -- it produces
        audio chunks one at a time, and each one is sent to
        the caller immediately, without waiting for the rest.
        """

        for i, sentence in enumerate(sentences, start=1):

            start = time.time()

            print(f"[tts] (streaming) Synthesizing chunk {i}: "
                  f"{sentence[:60]}...")

            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav"
            )
            chunk_path = temp_file.name
            temp_file.close()

            tts_model.tts_to_file(
                text=sentence,
                speaker_wav=str(REFERENCE_VOICE),
                language="en",
                file_path=chunk_path,
            )

            elapsed = time.time() - start
            audio_duration = get_wav_duration(chunk_path)
            rtf = elapsed / audio_duration if audio_duration > 0 else 0

            print(f"[tts] (streaming) Chunk {i} ready in {elapsed:.1f}s "
                  f"(audio: {audio_duration:.1f}s, RTF: {rtf:.2f})")

            logging.info(
                f"TTS chunk {i} | text_len={len(sentence)} chars | "
                f"synth_time={elapsed:.2f}s | audio_duration={audio_duration:.2f}s | "
                f"RTF={rtf:.2f}"
            )

            # Read the audio bytes and send them immediately.
            with open(chunk_path, "rb") as f:
                audio_bytes = f.read()

            # Send a small header so the frontend knows how
            # many bytes belong to this chunk, then the bytes
            # themselves. This lets the browser split the
            # continuous stream back into separate WAV files.
            size_header = len(audio_bytes).to_bytes(4, "big")

            yield size_header
            yield audio_bytes
        logging.info(f"TTS stream request completed | {len(sentences)} chunks sent")
        
    return StreamingResponse(
        audio_chunk_generator(),
        media_type="application/octet-stream",
    )
