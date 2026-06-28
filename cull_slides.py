#!/usr/bin/env python3
"""Cull non-content slides from saved talks using the local multimodal LLM.

Slide capture is noisy: it grabs the desktop while a speaker sets up screen
share, the presentation-software editor, app switchers, file pickers, video-call
windows and blank/duplicate transition frames. This tool asks the on-box vision
model to flag those, and moves them out of the way so the archive (and its
lightbox) shows only real presentation content — and so slide 1 is the title
slide, not a screen-share dialog.

Culling is NON-DESTRUCTIVE and reversible: flagged slides are moved into
slides/.culled/ inside the talk folder (a hidden dir, so transcript_server and
export_static never serve or copy them). `restore` moves them back.

Nothing here calls Claude — only the local LLM at config.LLM_URL, the same model
the summaries use. Run it on the GPU box, where the talks and the model live.

Workflow:
    python3 cull_slides.py analyze                 # slide counts per talk
    python3 cull_slides.py propose -o cull.json    # model flags slides -> plan
    python3 cull_slides.py apply cull.json --dry-run
    python3 cull_slides.py apply cull.json         # move flagged -> .culled/
    python3 cull_slides.py restore [--talk SLUG]   # undo

Scope a run with --talk <slug-or-folder> (repeatable) to a subset of talks.
"""
import argparse
import base64
import json
import os
import re
import shutil
import sys
from pathlib import Path

import transcript_server as ts  # single source of truth for paths + the local LLM

ROOT = ts.SAVE_DIR
CULLED_DIRNAME = ".culled"

SYSTEM = (
    "You audit screenshots captured automatically during a conference talk. Each "
    "image is one captured frame. Decide, per frame, whether it is REAL "
    "PRESENTATION CONTENT (a title slide, figure, plot, equation, bulleted slide, "
    "demo output the speaker is presenting) or NON-CONTENT noise that should be "
    "removed from the archive.\n\n"
    "Cull (NON-CONTENT) examples: the OS desktop or home screen; screen-share / "
    "'share your screen' setup dialogs; the presentation editor UI (PowerPoint, "
    "Keynote, Google Slides edit view with panels/thumbnails); video-call windows "
    "(Zoom, Teams, Meet); app switchers, file pickers, browser chrome, terminals "
    "used for setup; blank, black or white frames; near-duplicate transition "
    "frames mid-animation.\n\n"
    "Keep anything that a viewer would recognise as a slide being presented, even "
    "if simple. When genuinely unsure, KEEP it.\n\n"
    "Reply with ONLY a JSON object: {\"cull\": [\"<filename>\", ...]} listing the "
    "filenames (exactly as labelled) to remove. If every frame is content, reply "
    "{\"cull\": []}."
)


def _slug(name: str) -> str:
    return re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_", "", name)


def talk_dirs(only=None):
    """Saved-talk dirs (those with metadata.json), optionally filtered to `only`
    (a set matched against full folder name or readable slug)."""
    out = []
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir() or not (d / "metadata.json").exists():
            continue
        if only and not (d.name in only or _slug(d.name) in only):
            continue
        out.append(d)
    return out


def _slides(d: Path):
    sd = d / "slides"
    return sorted(sd.glob("slide_*.jpg")) if sd.is_dir() else []


def _parse_cull(raw: str, valid: set) -> list:
    """Pull a {"cull": [...]} list out of the model reply, keeping only names we
    actually sent (the model occasionally invents or reformats)."""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return []
    names = obj.get("cull", []) if isinstance(obj, dict) else []
    return [n for n in names if n in valid]


# ── analyze ──────────────────────────────────────────────────────────────────

def analyze(only=None):
    rows = []
    for d in talk_dirs(only):
        slides = _slides(d)
        culled = len(list((d / "slides" / CULLED_DIRNAME).glob("slide_*.jpg"))) \
            if (d / "slides" / CULLED_DIRNAME).is_dir() else 0
        rows.append((len(slides), culled, _slug(d.name)))
    if not rows:
        print("No talks found.")
        return
    print(f"{'live':>5} {'culled':>7}  talk")
    for n, c, slug in rows:
        print(f"{n:>5} {c:>7}  {slug}")
    print(f"\n{len(rows)} talks · {sum(r[0] for r in rows)} live slides · "
          f"{sum(r[1] for r in rows)} already culled")


# ── propose ──────────────────────────────────────────────────────────────────

def _flag_talk(d: Path, batch: int, timeout: int) -> list:
    slides = _slides(d)
    flagged = []
    for i in range(0, len(slides), batch):
        chunk = slides[i:i + batch]
        content = [{"type": "text",
                    "text": f"{len(chunk)} captured frames from this talk follow, "
                            f"each labelled with its filename."}]
        for f in chunk:
            b64 = base64.b64encode(f.read_bytes()).decode()
            content.append({"type": "text", "text": f"Filename: {f.name}"})
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + b64}})
        try:
            raw = ts._llm_chat(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": content}],
                max_tokens=400, temperature=0.0, timeout=timeout)
        except Exception as e:
            print(f"    ! batch {i // batch + 1} failed: {e}", file=sys.stderr)
            continue
        flagged += _parse_cull(raw, {f.name for f in chunk})
    return sorted(set(flagged))


def propose(only=None, batch=8, timeout=240, out=None):
    plan = {"model": ts.LLM_MODEL, "culled": {}}
    for d in talk_dirs(only):
        slides = _slides(d)
        if not slides:
            continue
        print(f"  → {_slug(d.name)} ({len(slides)} slides) …", flush=True)
        flagged = _flag_talk(d, batch, timeout)
        # Safety: never propose removing (almost) the whole talk — that's a model
        # misfire, not a setup-heavy talk. Skip and warn for a human to inspect.
        if flagged and len(flagged) > 0.6 * len(slides):
            print(f"    ! flagged {len(flagged)}/{len(slides)} (>60%) — skipping "
                  f"this talk, inspect manually", file=sys.stderr)
            continue
        if flagged:
            plan["culled"][d.name] = flagged
            print(f"    flagged {len(flagged)}: {', '.join(flagged)}")
    text = json.dumps(plan, indent=2)
    if out:
        Path(out).write_text(text)
        n = sum(len(v) for v in plan["culled"].values())
        print(f"\nplan → {out}  ({n} slides across {len(plan['culled'])} talks)")
        print("review it, then:  python3 cull_slides.py apply " + out)
    else:
        print(text)


# ── lock helpers (saved talks are chmod 0o555/0o444) ─────────────────────────

def _unlock(d: Path):
    os.chmod(d, 0o755)
    for r, ds, fs in os.walk(d):
        for x in ds + fs:
            p = os.path.join(r, x)
            os.chmod(p, 0o755 if os.path.isdir(p) else 0o644)


def _lock(d: Path):
    ts._make_readonly(d)


# ── apply / restore ──────────────────────────────────────────────────────────

def apply(plan_path, dry_run=False):
    plan = json.loads(Path(plan_path).read_text())
    culled = plan.get("culled", {})
    if not culled:
        print("plan lists nothing to cull.")
        return
    moved = 0
    for folder, names in culled.items():
        d = ROOT / folder
        sd = d / "slides"
        if not sd.is_dir():
            print(f"  ! {folder}: no slides/ — skipping", file=sys.stderr)
            continue
        present = [n for n in names if (sd / n).is_file()]
        if not present:
            print(f"  · {_slug(folder)}: already culled / not found")
            continue
        print(f"  {'[dry-run] ' if dry_run else ''}{_slug(folder)}: "
              f"{len(present)} → .culled/  ({', '.join(present)})")
        if dry_run:
            continue
        locked = not os.access(sd, os.W_OK)
        if locked:
            _unlock(d)
        try:
            dest = sd / CULLED_DIRNAME
            dest.mkdir(exist_ok=True)
            for n in present:
                shutil.move(str(sd / n), str(dest / n))
                moved += 1
        finally:
            if locked:
                _lock(d)
    if not dry_run:
        print(f"\nmoved {moved} slide(s) to .culled/. "
              f"Re-export to publish:  ./publish_site.sh")


def restore(only=None):
    restored = 0
    for d in talk_dirs(only):
        dest = d / "slides" / CULLED_DIRNAME
        if not dest.is_dir():
            continue
        files = list(dest.glob("slide_*.jpg"))
        if not files:
            continue
        locked = not os.access(d / "slides", os.W_OK)
        if locked:
            _unlock(d)
        try:
            for f in files:
                shutil.move(str(f), str(d / "slides" / f.name))
                restored += 1
            try:
                dest.rmdir()
            except OSError:
                pass
            print(f"  {_slug(d.name)}: restored {len(files)}")
        finally:
            if locked:
                _lock(d)
    print(f"\nrestored {restored} slide(s)." if restored else "nothing to restore.")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_talk(p):
        p.add_argument("--talk", action="append", metavar="SLUG",
                       help="limit to this talk (folder name or slug); repeatable")

    pa = sub.add_parser("analyze", help="list live/culled slide counts per talk")
    add_talk(pa)

    pp = sub.add_parser("propose", help="local LLM flags non-content slides → plan JSON")
    add_talk(pp)
    pp.add_argument("--batch", type=int, default=8, help="frames per model call (default 8)")
    pp.add_argument("--timeout", type=int, default=240, help="per-call timeout seconds")
    pp.add_argument("-o", "--out", help="write the plan here (else stdout)")

    px = sub.add_parser("apply", help="move flagged slides into slides/.culled/")
    px.add_argument("plan", help="plan JSON from `propose`")
    px.add_argument("--dry-run", action="store_true", help="show what would move")

    pr = sub.add_parser("restore", help="move .culled/ slides back")
    add_talk(pr)

    args = ap.parse_args()
    only = set(args.talk) if getattr(args, "talk", None) else None
    if args.cmd == "analyze":
        analyze(only)
    elif args.cmd == "propose":
        propose(only, batch=args.batch, timeout=args.timeout, out=args.out)
    elif args.cmd == "apply":
        apply(args.plan, dry_run=args.dry_run)
    elif args.cmd == "restore":
        restore(only)


if __name__ == "__main__":
    main()
