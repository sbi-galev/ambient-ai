#!/usr/bin/env python3
"""Generate summaries for saved talks that don't have one yet.

Useful for talks saved before summaries existed, or while SUMMARIES=0, or to
regenerate failed ones. Reuses transcript_server's summariser; temporarily
unlocks each read-only folder, writes summary.json / summary.md, then re-locks.

Usage:
    python3 backfill_summaries.py [--force] [folder-name ...]

With no folder names, scans every talk folder under transcripts/. --force
regenerates even folders that already have a good summary.
"""
import json
import os
import sys
import time
from pathlib import Path

import transcript_server as ts


def _unlock(d: Path):
    for root, dirs, files in os.walk(d):
        os.chmod(root, 0o755)
        for f in files:
            os.chmod(os.path.join(root, f), 0o644)


def _needs_summary(d: Path, force: bool) -> bool:
    if force:
        return True
    if not (d / "questions.json").exists():   # talks saved before questions existed
        return True
    sj = d / "summary.json"
    if not sj.exists():
        return True
    try:
        return json.loads(sj.read_text()).get("status") == "failed"
    except Exception:
        return True


def main():
    args = sys.argv[1:]
    force = "--force" in args
    names = [a for a in args if not a.startswith("--")]

    base = ts.SAVE_DIR
    folders = ([base / n for n in names] if names
               else sorted(d for d in base.iterdir() if d.is_dir()))
    todo = [d for d in folders
            if d.is_dir() and (d / "metadata.json").exists() and _needs_summary(d, force)]

    print(f"{len(todo)} folder(s) to summarise via {ts.LLM_MODEL} at {ts.LLM_URL}\n")
    for d in todo:
        label = ""
        try:
            label = json.loads((d / "metadata.json").read_text()).get("label", "")
        except Exception:
            pass
        n_slides = len(list((d / "slides").glob("slide_*.jpg"))) if (d / "slides").is_dir() else 0
        print(f"  → {d.name}  ({n_slides} slides) ...", flush=True)
        t0 = time.time()
        try:
            _unlock(d)
            data = ts._generate_summary(d, label)
            title = data["title"]
            ts._write_summary_outputs(d, data)
            # Question text changed, so any votes keyed to the old q1..q3 are stale.
            (ts.VOTES_DIR / f"{d.name}.json").unlink(missing_ok=True)
            print(f"    ✓ {time.time() - t0:5.1f}s  {title!r}", flush=True)
        except Exception as e:
            print(f"    ✗ {time.time() - t0:5.1f}s  {e}", flush=True)
        finally:
            ts._make_readonly(d)
    print("\ndone.")


if __name__ == "__main__":
    main()
