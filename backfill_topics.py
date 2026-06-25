#!/usr/bin/env python3
"""Generate (or refresh) the conference-wide key-topics synthesis shown on /topics.

The server builds this lazily in the background when the page is viewed; this
script lets you warm or force it up front (e.g. after re-cutting or re-summarising
talks). Writes topic_summaries/topics.json.

Usage:
    python3 backfill_topics.py [--force] [--no-llm]

--force rebuilds even when the cache is already up to date. --no-llm skips the
model and clusters purely by keyword (also the automatic fallback if the model
is unreachable).
"""
import json
import sys
import time

import transcript_server as ts


def main():
    args = sys.argv[1:]
    force = "--force" in args
    use_llm = "--no-llm" not in args and ts.SUMMARIES_ENABLED

    talks = ts._load_talks()
    if not talks:
        print("No talks found.")
        return

    sig = ts._topics_signature(talks)
    cached = ts._load_topics_summary()
    if not force and cached and cached.get("signature") == sig and cached.get("topics"):
        print(f"Key topics up to date ({len(cached['topics'])} topics) — use --force to rebuild.")
        return

    how = f"LLM {ts.LLM_MODEL}" if use_llm else "keyword aggregation"
    print(f"Synthesising key topics from {len(talks)} talks via {how} ...", flush=True)
    t0 = time.time()
    data = ts._generate_topics_summary(talks, use_llm)
    data["signature"] = sig
    ts._atomic_write_bytes(ts.TOPICS_PATH, json.dumps(data, indent=2).encode())
    print(f"  ✓ {time.time() - t0:5.1f}s  {len(data['topics'])} topics, "
          f"{len(data['edges'])} links ({data['source']})")


if __name__ == "__main__":
    main()
