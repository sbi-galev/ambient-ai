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
import json
import os
import re
import subprocess
import sys
import urllib.request
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "sbi4galev"))
from screen_capture import ScreenCapture

SERVER = "http://172.24.17.90:7103"
TOKEN = "sbi4galev"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 8
MIN_CHUNK_SECONDS = 2
BLOCK_SIZE = int(SAMPLE_RATE * 0.1)  # 100ms blocks
SAVE_DIR = Path.home() / "Pictures" / "SBI4GALEV"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
LAST_SENT_PATH = SAVE_DIR / "last_sent.jpg"
SLIDE_CHANGE_THRESHOLD = 8.0
THUMB_SIZE = (64, 64)

_last_thumb = None
_screen = None


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
    global _last_thumb, _screen
    try:
        if _screen is None:
            return
        img = _screen.grab()
        if img is None:
            return

        thumb = np.array(img.resize(THUMB_SIZE, Image.LANCZOS).convert("L"), dtype=float)
        if _last_thumb is not None:
            diff = np.mean(np.abs(thumb - _last_thumb))
            if diff < SLIDE_CHANGE_THRESHOLD:
                return

        _last_thumb = thumb

        buf = io.BytesIO()
        img.resize((1280, int(img.height * 1280 / img.width)), Image.LANCZOS).save(buf, format="JPEG", quality=75)
        jpeg = buf.getvalue()

        LAST_SENT_PATH.write_bytes(jpeg)

        req = urllib.request.Request(
            f"{SERVER}/image",
            data=jpeg,
            headers={"Content-Type": "image/jpeg", "X-Token": TOKEN},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        print("  [slide captured]")
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


def _find_default_sink() -> str:
    """Return the PipeWire node name of the default audio output sink."""
    try:
        out = subprocess.run(
            ["pw-metadata", "0", "default.audio.sink"],
            capture_output=True, text=True,
        ).stdout
        m = re.search(r'"name"\s*:\s*"([^"]+)"', out)
        if m:
            return m.group(1)
    except Exception:
        pass
    # Fallback: first alsa_output sink from pw-dump
    try:
        data = json.loads(subprocess.run(["pw-dump"], capture_output=True, text=True).stdout)
        for o in data:
            props = o.get("info", {}).get("props", {})
            if (o.get("type") == "PipeWire:Interface:Node"
                    and props.get("media.class") == "Audio/Sink"
                    and "alsa_output" in props.get("node.name", "")):
                return props["node.name"]
    except Exception:
        pass
    return None


def _zoom_blocks():
    """Generator: yield float32 mono numpy blocks from pw-record on the default sink monitor."""
    sink = _find_default_sink()
    if sink is None:
        print("[zoom] Could not find default audio sink", file=sys.stderr)
        return

    print(f"  [zoom] capturing monitor of: {sink}")
    cmd = ["pw-record", f"--target={sink}", "--format=s16",
           f"--rate={SAMPLE_RATE}", "--channels=2", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Skip WAV header: scan forward for the "data" chunk marker
    hdr = proc.stdout.read(64)
    data_offset = hdr.find(b"data")
    if data_offset != -1:
        proc.stdout.read(4)  # skip the 4-byte data-chunk size field
    # else: non-WAV output — treat everything as raw PCM (shouldn't happen)

    bytes_per_block = BLOCK_SIZE * 2 * 2  # frames × 2 ch × 2 bytes/sample
    try:
        while True:
            raw = proc.stdout.read(bytes_per_block)
            if not raw:
                break
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if arr.size >= 2:
                # Boost system audio (typically ~0.01–0.04 RMS) to mic-comparable levels
                yield np.clip(arr.reshape(-1, 2).mean(axis=1) * 8.0, -1.0, 1.0)
    finally:
        proc.terminate()
        proc.wait()


def run(device, silence_threshold: float, zoom: bool = False):
    global _screen
    buffer = []
    silence_frames = 0
    frames_per_chunk = int(SAMPLE_RATE * CHUNK_SECONDS)
    min_frames = int(SAMPLE_RATE * MIN_CHUNK_SECONDS)

    try:
        _screen = ScreenCapture()
    except Exception as e:
        print(f"[warn] Screen capture unavailable: {e}", file=sys.stderr)

    if zoom:
        print("Recording from system audio monitor via PipeWire — Ctrl+C to stop.\n")
    else:
        print(f"Recording on device {device} — speak now. Ctrl+C to stop.\n")

    def _process_blocks(blocks):
        for block in blocks:
            buffer.append(block)

            if _rms(block) < silence_threshold:
                silence_frames_ref[0] += len(block)
            else:
                silence_frames_ref[0] = 0

            total = sum(len(b) for b in buffer)
            if (silence_frames_ref[0] >= SAMPLE_RATE * 0.5 and total >= min_frames) \
                    or total >= frames_per_chunk:
                capture_and_push_screenshot()
                transcribe_and_push(np.concatenate(buffer))
                buffer.clear()
                silence_frames_ref[0] = 0

    silence_frames_ref = [0]

    try:
        if zoom:
            _process_blocks(_zoom_blocks())
        else:
            with sd.InputStream(device=device, channels=1, samplerate=SAMPLE_RATE,
                                dtype="float32", blocksize=BLOCK_SIZE) as stream:
                def _sd_blocks():
                    while True:
                        blk, _ = stream.read(BLOCK_SIZE)
                        yield blk[:, 0]
                _process_blocks(_sd_blocks())

    except KeyboardInterrupt:
        if buffer and sum(len(b) for b in buffer) >= min_frames:
            transcribe_and_push(np.concatenate(buffer))
        if _screen:
            _screen.stop()
        print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=8)
    parser.add_argument("--silence", type=float, default=None)
    parser.add_argument("--zoom", action="store_true",
                        help="Capture Zoom output via PipeWire (ignores --device)")
    args = parser.parse_args()
    silence = args.silence if args.silence is not None else (0.01 if args.zoom else 0.05)
    run(args.device, silence, zoom=args.zoom)
