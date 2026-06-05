#!/usr/bin/env python3
"""Mic → turing transcript server (live transcription + website update).

Usage:
    python3 live_transcribe.py [--device N] [--silence 0.05]

POSTs audio chunks directly to turing's transcript server over LAN.
Captures screenshots and posts them when the slide changes significantly.
No SSH in the hot path.
"""
import argparse
import io
import sys
import urllib.request
import wave

import numpy as np
import sounddevice as sd
from PIL import Image

SERVER = "http://172.24.17.90:7103"
TOKEN = "sbi4galev"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 8
MIN_CHUNK_SECONDS = 2
BLOCK_SIZE = int(SAMPLE_RATE * 0.1)  # 100ms blocks
SLIDE_CHANGE_THRESHOLD = 8.0  # mean pixel diff (0-255) to count as a new slide
THUMB_SIZE = (64, 64)

_last_thumb = None


def _pcm_to_wav(pcm: np.ndarray) -> bytes:
    pcm16 = (pcm * 32768).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0


def capture_and_push_screenshot():
    global _last_thumb
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "x11grab", "-video_size", "1920x1080",
             "-i", ":0.0", "-frames:v", "1", "-f", "image2", "-update", "1",
             "/tmp/_sbi_screenshot.png", "-y"],
            capture_output=True, timeout=5
        )
        if result.returncode != 0:
            return
        img = Image.open("/tmp/_sbi_screenshot.png")
        thumb = np.array(img.resize(THUMB_SIZE, Image.LANCZOS).convert("L"), dtype=float)
        if _last_thumb is not None:
            diff = np.mean(np.abs(thumb - _last_thumb))
            if diff < SLIDE_CHANGE_THRESHOLD:
                return
        _last_thumb = thumb
        # encode as JPEG and POST
        buf = io.BytesIO()
        img.resize((1280, int(img.height * 1280 / img.width)), Image.LANCZOS).save(buf, format="JPEG", quality=75)
        jpeg = buf.getvalue()
        req = urllib.request.Request(
            f"{SERVER}/image",
            data=jpeg,
            headers={"Content-Type": "image/jpeg", "X-Token": TOKEN},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"\n[screenshot warn] {e}", file=sys.stderr)


def transcribe_and_push(pcm: np.ndarray):
    wav = _pcm_to_wav(pcm)
    print("  …transcribing…", end="\r", flush=True)
    try:
        req = urllib.request.Request(
            f"{SERVER}/transcribe",
            data=wav,
            headers={"Content-Type": "audio/wav", "X-Token": TOKEN},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            import json
            text = json.loads(resp.read()).get("text", "").strip()
        if text:
            print(f"  {text}          ")
    except Exception as e:
        print(f"\n[warn] {e}", file=sys.stderr)


def run(device: int, silence_threshold: float):
    buffer = []
    silence_frames = 0
    frames_per_chunk = int(SAMPLE_RATE * CHUNK_SECONDS)
    min_frames = int(SAMPLE_RATE * MIN_CHUNK_SECONDS)

    print(f"Recording on device {device} — speak now. Ctrl+C to stop.\n")

    try:
        with sd.InputStream(device=device, channels=1, samplerate=SAMPLE_RATE,
                            dtype="float32", blocksize=BLOCK_SIZE) as stream:
            while True:
                block, _ = stream.read(BLOCK_SIZE)
                block = block[:, 0]
                buffer.append(block)

                if _rms(block) < silence_threshold:
                    silence_frames += BLOCK_SIZE
                else:
                    silence_frames = 0

                total = sum(len(b) for b in buffer)
                if (silence_frames >= SAMPLE_RATE * 0.5 and total >= min_frames) \
                        or total >= frames_per_chunk:
                    capture_and_push_screenshot()
                    transcribe_and_push(np.concatenate(buffer))
                    buffer = []
                    silence_frames = 0

    except KeyboardInterrupt:
        if buffer and sum(len(b) for b in buffer) >= min_frames:
            transcribe_and_push(np.concatenate(buffer))
        print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=8)
    parser.add_argument("--silence", type=float, default=0.05)
    args = parser.parse_args()
    run(args.device, args.silence)
