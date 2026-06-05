#!/usr/bin/env python3
"""Watch ~/Pictures for new screenshots and POST them to the transcript server.

Run in a second terminal. Press Print Screen whenever you want to capture a slide.
"""
import io
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

SERVER = "http://172.24.17.90:7103"
TOKEN = "sbi4galev"
WATCH_DIR = Path.home() / "Pictures" / "Screenshots"
SLIDE_CHANGE_THRESHOLD = 8.0
THUMB_SIZE = (64, 64)

_seen = set()
_last_thumb = None


def send_image(path: Path):
    global _last_thumb
    try:
        img = Image.open(path).convert("RGB")
        thumb = np.array(img.resize(THUMB_SIZE, Image.LANCZOS).convert("L"), dtype=float)
        if _last_thumb is not None:
            diff = np.mean(np.abs(thumb - _last_thumb))
            if diff < SLIDE_CHANGE_THRESHOLD:
                print(f"  skipped (no significant change, diff={diff:.1f})")
                return
        _last_thumb = thumb

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
        print(f"  sent: {path.name} ({len(jpeg)//1024}KB)")
    except Exception as e:
        print(f"  error: {e}", file=sys.stderr)


def scan():
    for f in WATCH_DIR.glob("*.png"):
        if f not in _seen:
            _seen.add(f)
            return f
    return None


def main():
    print(f"Watching {WATCH_DIR} — press Print Screen to capture a slide.\n")
    # seed with existing files so we don't send old screenshots
    for f in WATCH_DIR.glob("*.png"):
        _seen.add(f)
    print(f"  ({len(_seen)} existing files ignored)")

    while True:
        new = scan()
        if new:
            print(f"  new screenshot: {new.name}")
            time.sleep(0.3)  # let GNOME finish writing
            send_image(new)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
