#!/usr/bin/env python3
"""Generate (or refresh) the per-day editorial overviews shown on /summaries.

Each conference day ("Day 1", "Day 2", …) gets a short overview synthesised from
that day's talk summaries, cached under day_summaries/<date>.json. The server
also builds these lazily in the background when the page is viewed; this script
just lets you warm or force them up front (e.g. after re-cutting talks).

Usage:
    python3 backfill_day_summaries.py [--force] [YYYY-MM-DD ...]

With no dates, processes every day that has talks. --force regenerates even days
whose cached overview already matches the current set of talks.
"""
import json
import sys
import time

import transcript_server as ts


def main():
    args = sys.argv[1:]
    force = "--force" in args
    want = {a for a in args if not a.startswith("--")}

    talks = ts._load_talks()
    grouped = ts._group_by_day(talks)
    if want:
        grouped = [g for g in grouped if g[0] in want]

    print(f"{len(grouped)} day(s) via {ts.LLM_MODEL} at {ts.LLM_URL}\n")
    for date, day_label, day_talks in grouped:
        sig = ts._day_signature(day_talks)
        cached = ts._load_day_summary(date)
        if not force and cached and cached.get("signature") == sig and cached.get("overview"):
            print(f"  · {date} ({day_label}) up to date — skipping", flush=True)
            continue
        print(f"  → {date} ({day_label}, {len(day_talks)} talks) ...", flush=True)
        t0 = time.time()
        try:
            data = ts._generate_day_summary(date, day_label, day_talks)
            data["signature"] = sig
            ts._atomic_write_bytes(ts.DAYS_DIR / f"{date}.json",
                                   json.dumps(data, indent=2).encode())
            print(f"    ✓ {time.time() - t0:5.1f}s  {len(data['themes'])} themes", flush=True)
        except Exception as e:
            print(f"    ✗ {time.time() - t0:5.1f}s  {e}", flush=True)
    print("\ndone.")


if __name__ == "__main__":
    main()
