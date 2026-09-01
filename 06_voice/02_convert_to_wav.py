"""
DAY 23 - UTILITY: CONVERT ANY AUDIO FILE TO XTTS-READY WAV

Purpose:
    Convert any downloaded/recorded reference voice file
    (mp3, m4a, aac, ogg, etc.) into a WAV file matching
    XTTS v2's expected format:

        - WAV container
        - 22050 Hz sample rate
        - Mono (1 channel)
        - 16-bit PCM

    Output is saved in the SAME folder as the input file,
    with the same base filename but a .wav extension.

Requires:
    ffmpeg must be installed and available on PATH.

Usage:
    python3 03_convert_to_wav.py /path/to/audio_file.aac
    python3 03_convert_to_wav.py /path/to/audio_file.mp3

    Or run with no arguments to be prompted interactively.
"""

import subprocess
import sys
from pathlib import Path


TARGET_SAMPLE_RATE = 22050
TARGET_CHANNELS = 1


def check_ffmpeg_available() -> bool:
    """
    Confirm ffmpeg is installed and callable.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def convert_to_wav(input_path: Path) -> Path:
    """
    Convert the given audio file to a WAV file with the
    same base name, saved in the same directory.

    Returns the path to the resulting WAV file.
    """

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    output_path = input_path.with_suffix(".wav")

    if output_path == input_path:
        # Input is already .wav -- still re-encode to force
        # the correct sample rate / channel count, saving to
        # a distinctly named file to avoid overwriting the
        # original before we can compare.
        output_path = input_path.with_name(
            input_path.stem + "_converted.wav"
        )

    print()
    print(f"[convert] Input : {input_path}")
    print(f"[convert] Output: {output_path}")
    print(
        f"[convert] Target: {TARGET_SAMPLE_RATE} Hz, "
        f"{TARGET_CHANNELS} channel(s)"
    )

    command = [
        "ffmpeg",
        "-y",                          # overwrite output if it exists
        "-i", str(input_path),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", str(TARGET_CHANNELS),
        str(output_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if result.returncode != 0:
        print()
        print("[convert] ffmpeg output:")
        print(result.stdout)
        raise RuntimeError(
            f"ffmpeg conversion failed (exit code {result.returncode})"
        )

    print()
    print(f"[convert] Success -- saved to: {output_path}")

    size_kb = output_path.stat().st_size / 1024
    print(f"[convert] File size: {size_kb:.1f} KB")

    return output_path


def main():

    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1]).expanduser().resolve()
    else:
        raw = input(
            "\nEnter the path to the audio file to convert: "
        ).strip()
        input_path = Path(raw).expanduser().resolve()

    print()
    print("=" * 75)
    print("AUDIO TO WAV CONVERTER")
    print("=" * 75)

    if not check_ffmpeg_available():
        print()
        print(
            "[error] ffmpeg was not found on PATH. "
            "Install it first, e.g.:"
        )
        print("    sudo apt install ffmpeg")
        return

    try:
        output_path = convert_to_wav(input_path)
    except Exception as exc:
        print()
        print(f"[error] {exc}")
        return

    print()
    print("=" * 75)
    print("DONE")
    print("=" * 75)
    print(f"WAV file ready at: {output_path}")


if __name__ == "__main__":
    main()
