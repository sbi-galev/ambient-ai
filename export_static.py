#!/usr/bin/env python3
"""Export the SBI4GALEV archive as a static site for GitHub Pages.

Renders the same pages the live server serves at /summaries, /summaries/<date>
and /topics into a self-contained `site/` directory of plain HTML + the slide
thumbnails they reference, with all server-only behaviour (absolute links, the
live SSE viewer, audience voting, the topics generation timer) stripped out.

The live transcript viewer (/, /admin, /stream) and voting (/vote) are dynamic
and are intentionally NOT exported.

This reuses transcript_server's render functions and summary generators, so it
must run on the machine that has the saved talks (transcripts/) and, for fresh
day overviews / key topics, the local LLM. Nothing in transcript_server.py is
modified — the static-specific transforms are post-processing here.

Output layout (flat, relative links → works under any base path, incl.
https://<owner>.github.io/<repo>/):
    site/
      index.html            (= /summaries, the archive home)
      topics.html           (= /topics)
      day-YYYY-MM-DD.html   (= /summaries/<date>, one per day)
      talks/<folder>/slides/slide_001.jpg   (referenced thumbnails only)
      .nojekyll

Usage:
    python3 export_static.py [--out DIR] [--clean] [--no-llm]
                            [--all-slides] [--include-transcripts] [--strict] [-v]

Deploy: the built site/ is git-ignored and lives on its own `gh-pages` branch,
which GitHub Pages serves directly. The publish_site.sh helper builds and pushes
it in one step:
    ./publish_site.sh        # = export_static.py --clean, then push site/ -> gh-pages
One-time: repo Settings -> Pages -> Source: Deploy from a branch -> gh-pages /
(root). publish_site.sh force-pushes a single fresh commit, so the branch only
ever holds the latest site; GitHub rebuilds Pages on each push (no workflow).
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import unquote

import transcript_server as ts


def _warn(msg):
    print(f"  ! {msg}", file=sys.stderr, flush=True)


# ── Permissions allowlist (fail-closed) ──────────────────────────────────────

ALLOWLIST_FILE = Path(__file__).parent / "public_talks.txt"


def load_allowlist(path: Path = ALLOWLIST_FILE):
    """Read the approved-talk folder names from public_talks.txt.

    Returns a set of folder names, or None if the file is absent (caller decides
    what that means). Blank lines and # comments are ignored; an inline '#'
    trailing a folder name is stripped too."""
    if not path.exists():
        return None
    names = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(line)
    return names


def apply_allowlist(verbose):
    """Wrap ts._load_talks so EVERY consumer (render functions, cache
    generation) only ever sees talks approved for public export. Fail-closed:
    a missing allowlist exports nothing."""
    allowed = load_allowlist()
    if allowed is None:
        raise SystemExit(
            f"No allowlist at {ALLOWLIST_FILE.name} — refusing to export (fail-closed). "
            f"Create it with one approved folder name per line.")
    if not allowed:
        raise SystemExit(
            f"{ALLOWLIST_FILE.name} lists no approved talks — nothing to export (fail-closed).")

    _orig_load = ts._load_talks

    def _ok(folder):
        # Match an allowlist entry against the full folder name OR its readable
        # slug (the part after the YYYY-MM-DD_HH-MM-SS_ timestamp prefix).
        slug = re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_", "", folder)
        return folder in allowed or slug in allowed

    def _filtered():
        talks = _orig_load()
        kept = [t for t in talks if _ok(t["folder"])]
        if verbose:
            print(f"  allowlist: keeping {len(kept)}/{len(talks)} talks; "
                  f"excluding {len(talks) - len(kept)}", flush=True)
        matched = set()
        for t in talks:
            slug = re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_", "", t["folder"])
            matched |= ({t["folder"], slug} & allowed)
        missing = allowed - matched
        if missing:
            _warn(f"allowlist names {len(missing)} entr(y/ies) with no talk on disk: "
                  f"{', '.join(sorted(missing))}")
        return kept

    ts._load_talks = _filtered


# ── Cache freshness (run before rendering, so pages render fully not "pending") ─

def ensure_caches(talks, use_llm, strict, verbose):
    """Make sure the per-day overviews and key-topics synthesis are present and
    fresh for the current talks, generating any that are stale. Per-talk
    summaries are only checked (generate those with backfill_summaries.py)."""
    missing = [t["folder"] for t in talks if not t["summary"]]
    if missing:
        _warn(f"{len(missing)} talk(s) have no summary.json — run backfill_summaries.py "
              f"first (e.g. {missing[0]})")
        if strict:
            raise SystemExit("strict: refusing to export with unsummarised talks")

    # Per-day overviews (LLM-only: no deterministic fallback).
    for date, label, dtalks in ts._group_by_day(talks):
        sig = ts._day_signature(dtalks)
        cached = ts._load_day_summary(date)
        if cached and cached.get("signature") == sig and cached.get("overview"):
            if verbose:
                print(f"  · day {date} overview up to date", flush=True)
            continue
        if not use_llm:
            _warn(f"day {date} overview is stale/missing and --no-llm — page will show "
                  f"a placeholder")
            if strict:
                raise SystemExit(f"strict: day {date} overview unavailable")
            continue
        print(f"  → generating day {date} overview …", flush=True)
        try:
            data = ts._generate_day_summary(date, label, dtalks)
            data["signature"] = sig
            ts._atomic_write_bytes(ts.DAYS_DIR / f"{date}.json",
                                   json.dumps(data, indent=2).encode())
        except Exception as e:
            _warn(f"day {date} overview generation failed: {e}")
            if strict:
                raise

    # Key topics (LLM with a built-in keyword fallback).
    sig = ts._topics_signature(talks)
    cached = ts._load_topics_summary()
    if cached and cached.get("signature") == sig and "topics" in cached:
        if verbose:
            print("  · key topics up to date", flush=True)
    else:
        print("  → generating key topics …", flush=True)
        try:
            data = ts._generate_topics_summary(talks, use_llm)
            data["signature"] = sig
            ts._atomic_write_bytes(ts.TOPICS_PATH, json.dumps(data, indent=2).encode())
            print(f"    {len(data['topics'])} topics, {len(data['edges'])} links "
                  f"({data['source']})", flush=True)
        except Exception as e:
            _warn(f"key topics generation failed: {e}")
            if strict:
                raise


# ── HTML post-processing: server URLs → static, voting → read-only ─────────────

def staticize(s: str) -> str:
    """Rewrite the server-rendered HTML into a standalone static page."""
    # 1. Dated day links (with optional #anchor) — BEFORE the bare /summaries rule.
    s = re.sub(r'href="/summaries/(\d{4}-\d{2}-\d{2})(#[^"]*)?"',
               lambda m: f'href="day-{m.group(1)}.html{m.group(2) or ""}"', s)
    # 2. Summaries index (anchor fallback first, then bare).
    s = s.replace('href="/summaries#', 'href="index.html#')
    s = s.replace('href="/summaries"', 'href="index.html"')
    # 3. Topics page.
    s = s.replace('href="/topics"', 'href="topics.html"')
    # 4. Slide assets (both src= and href=) → relative.
    s = s.replace('="/talks/', '="talks/')
    # 5. Drop the live-viewer nav link (not part of the static archive).
    s = re.sub(r'<a href="/">[^<]*</a>', '', s)
    # 6. Voting → read-only: strip the vote buttons and remove the voting JS.
    s = re.sub(r'<span class="qvote">.*?</span>', '', s, flags=re.S)
    s = s.replace(ts.SUMMARIES_JS, "")
    return s


# ── Assets ─────────────────────────────────────────────────────────────────────

def copy_assets(refs, talks, out: Path, all_slides, include_transcripts, verbose):
    copied = 0
    if all_slides:
        for t in talks:
            sd = ts.SAVE_DIR / t["folder"] / "slides"
            for f in sorted(sd.glob("slide_*.jpg")) if sd.is_dir() else []:
                dst = out / "talks" / t["folder"] / "slides" / f.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, dst)
                copied += 1
    else:
        for rel in refs:                       # rel = "talks/<folder>/slides/<file>"
            relp = unquote(rel)
            src = ts.SAVE_DIR / relp[len("talks/"):]
            if not src.is_file():
                _warn(f"referenced asset missing on disk: {relp}")
                continue
            dst = out / relp
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            copied += 1

    if include_transcripts:
        _warn("--include-transcripts: PUBLISHING FULL VERBATIM TRANSCRIPTS with no "
              "access control")
        for t in talks:
            for name in ts.PRIVATE_FILES:
                src = ts.SAVE_DIR / t["folder"] / name
                if src.is_file():
                    dst = out / "talks" / t["folder"] / name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                    copied += 1
    if verbose:
        print(f"  copied {copied} asset file(s)", flush=True)
    return copied


# ── Verification ───────────────────────────────────────────────────────────────

_FORBIDDEN = ['="/summaries', '="/topics"', '="/talks/', 'href="/"',
              'function vote(', '/vote', '/topics.json', '/stream',
              'EventSource', 'class="qvote"', 'class="generating"', 'id="gen-timer"']


def verify(out: Path, pages: dict, refs, include_transcripts) -> list:
    problems = []
    for name, htmlstr in pages.items():
        for bad in _FORBIDDEN:
            if bad in htmlstr:
                problems.append(f"{name}: leftover {bad!r}")
    if 'class="tcard"' not in pages.get("topics.html", ""):
        problems.append("topics.html: no topic cards (generation may have failed)")
    for rel in refs:
        if not (out / unquote(rel)).is_file():
            problems.append(f"missing copied asset: {rel}")
    if not include_transcripts:
        for p in out.rglob("*"):
            if p.name in ts.PRIVATE_FILES:
                problems.append(f"private file leaked into export: {p}")
    return problems


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Export the archive as a static site.")
    ap.add_argument("--out", default="site", help="output directory (default: site)")
    ap.add_argument("--clean", action="store_true", help="remove the output dir first")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip the LLM (topics use the keyword fallback; stale day "
                         "overviews render as placeholders)")
    ap.add_argument("--all-slides", action="store_true",
                    help="copy every slide of every talk, not just referenced thumbnails")
    ap.add_argument("--include-transcripts", action="store_true",
                    help="also publish full transcripts (PUBLIC, no access control)")
    ap.add_argument("--strict", action="store_true",
                    help="treat missing summaries / unavailable overviews as errors")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    apply_allowlist(args.verbose)   # fail-closed: only public_talks.txt talks
    talks = ts._load_talks()
    if not talks:
        print("No approved talks found (check public_talks.txt) — nothing to export.")
        return

    use_llm = (not args.no_llm) and ts.SUMMARIES_ENABLED
    if not use_llm:
        # Stop the render functions from spawning background generation threads.
        ts.SUMMARIES_ENABLED = False

    print(f"Exporting {len(talks)} talks "
          f"({'LLM ' + ts.LLM_MODEL if use_llm else 'no LLM'}) → {out}/", flush=True)
    ensure_caches(talks, use_llm, args.strict, args.verbose)

    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # Render every page through the static post-processor.
    pages = {"index.html": staticize(ts.render_summaries_page()),
             "topics.html": staticize(ts.render_topics_page())}
    for date, _label, _dtalks in ts._group_by_day(talks):
        pages[f"day-{date}.html"] = staticize(ts.render_day_page(date))
    for name, body in pages.items():
        (out / name).write_text(body)
        if args.verbose:
            print(f"  wrote {name} ({len(body)} bytes)", flush=True)

    # Copy only the slide thumbnails the pages reference.
    combined = "\n".join(pages.values())
    refs = sorted(set(re.findall(r'(?:src|href)="(talks/[^"]+)"', combined)))
    copy_assets(refs, talks, out, args.all_slides, args.include_transcripts, args.verbose)

    (out / ".nojekyll").write_text("")

    problems = verify(out, pages, refs, args.include_transcripts)
    if problems:
        print("\nVERIFY FAILED:", flush=True)
        for p in problems:
            print(f"  ✗ {p}", flush=True)
        raise SystemExit(1)

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\n✓ {len(pages)} pages, {len(refs)} thumbnails, {total / 1e6:.1f} MB → {out}/")
    print("  verify OK · serve locally with:  python3 -m http.server -d "
          f"{out} 8000")


if __name__ == "__main__":
    main()
