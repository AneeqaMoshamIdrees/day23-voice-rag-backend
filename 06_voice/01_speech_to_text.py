"""
DAY 22/23 - STEP 6.1: SPEECH TO TEXT

Purpose:
    Convert recorded audio into text using faster-whisper.

This replaces the old speech/transcription.py with the
same working logic, moved into the numbered structure.
"""

from faster_whisper import WhisperModel


class TranscriptionService:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        print(f"[stt] Loading Whisper model: {model_size}")

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        print("[stt] Whisper model loaded successfully.")

    def transcribe(self, audio_path: str):

        print(f"[stt] Transcribing: {audio_path}")

        segments, info = self.model.transcribe(
         audio_path,
         beam_size=5,
         language="en",
         task="transcribe",) 

        text = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        print(f"[stt] Detected language: {info.language} "
              f"(confidence: {info.language_probability:.2f})")

        print(f"[stt] Transcript: {text}")

        return {
            "text": text,
            "language": info.language,
            "language_probability": info.language_probability,
        }

transcription_service = TranscriptionService(model_size="small.en")

def main():

    from pathlib import Path

    print()
    print("=" * 75)
    print("DAY 22/23 - SPEECH TO TEXT TEST")
    print("=" * 75)

    audio_path = input(
        "\nEnter path to an audio file to transcribe: "
    ).strip()

    if not audio_path or not Path(audio_path).exists():
        print("File not found.")
        return

    result = transcription_service.transcribe(audio_path)

    print()
    print("=" * 75)
    print("RESULT")
    print("=" * 75)
    print(f"Text: {result['text']}")
    print(f"Language: {result['language']}")
    print(f"Confidence: {result['language_probability']:.2f}")


if __name__ == "__main__":
    main()
