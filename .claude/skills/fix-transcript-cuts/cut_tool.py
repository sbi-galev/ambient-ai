#!/usr/bin/env python3
"""Mechanical primitives for inspecting and re-cutting saved talk folders.

Talk folders are written by transcript_server.py and finalised read-only
(files 0444, dirs 0555). Each holds:
    session.json    ordered list of {"type":"text","ts","text"}
                                  or {"type":"slide","ts","file":"slides/slide_NNN.jpg"}
    transcript.txt  " ".join(text chunks).strip() + "\n"
    transcript.html rendered archive page
    metadata.json   {label, saved_at, n_slides, n_text_chunks, first_ts, last_ts}
    slides/slide_NNN.jpg

Two subcommands:

  analyze   Print a chronological timeline, the tail/head of every consecutive
            pair of talks, and transition markers (applause / chair handoffs /
            speaker self-intros). Use this to locate boundary leakage and
            mega-folders before deciding on cuts.

  apply     Execute a declarative plan (JSON). Every folder that changes is
            rebuilt from scratch as an ordered concatenation of segments sliced
            out of (backed-up copies of) source folders. Slides are copied and
            renumbered into each destination; transcript.txt / .html /
            metadata.json are regenerated in the server's exact format; folders
            are re-locked read-only. Discards go to bin/ rather than /dev/null.

Plan schema (see SKILL.md for worked examples):
  {
    "backup_label": "premerge",            # optional; tag for the backup dir
    "operations": [
      {
        "dest": "2026-...-harvey-intro",   # folder NAME (created or overwritten)
        "bin": false,                       # true -> placed under transcripts/bin/
        "label": "Harvey_Intro",           # optional; defaults to dest's existing label
        "saved_at": "2026-06-23T10:19:21", # optional; defaults to first segment src's
        "segments": [
          {"src": "2026-...-chris-lovell", "start": 155, "end": 170},
          {"src": "2026-...-harvey-intro", "start": 0,   "end": 302}
        ]
      }
    ]
  }

Rules the caller MUST honour:
  * Every folder whose contents change must appear as a `dest` with its COMPLETE
    new segment list (the tool overwrites dests wholesale; unlisted folders are
    untouched).
  * `start`/`end` are Python slice bounds (end exclusive) into the SOURCE folder's
    original session.json.
  * Sources are always read from the backup taken at the start of `apply`, so a
    folder may safely be both a source and a destination.
"""
import argparse
import html
import json
import os
import re
import shutil
import sys
from datetime import datetime

ROOT = os.environ.get("TRANSCRIPTS_DIR")
if not ROOT:
    # default: <repo>/transcripts relative to this file (.claude/skills/<skill>/)
    here = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.abspath(os.path.join(here, "..", "..", "..", "transcripts"))

MARKER_RE = re.compile(
    r"\b(thank you|thanks|applause|next speaker|next talk|welcome back|"
    r"any question|questions|my name|name is|name's|i'm a|phd|introduce|"
    r"hand over|over to|please join|welcom|kick us off|take it away)\b", re.I)
INTRO_RE = re.compile(r"\b(my name(?:'s| is)|i'm a|hello everyone|hi everyone)\b", re.I)


def talk_dirs(base=ROOT):
    out = []
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        if name in ("bin",) or name.startswith("."):
            continue
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "metadata.json")):
            out.append(d)
    return out


def load_session(d):
    return json.load(open(os.path.join(d, "session.json")))


def load_meta(d):
    return json.load(open(os.path.join(d, "metadata.json")))


def _txt(x, n=90):
    return (x.get("text", "")[:n]) if x["type"] == "text" else "[SLIDE]"


# ── analyze ──────────────────────────────────────────────────────────────────
def analyze(base=ROOT, context=6):
    dirs = talk_dirs(base)
    print(f"# {len(dirs)} talks in {base}\n")
    print("## Timeline")
    for d in dirs:
        m = load_meta(d)
        s = load_session(d)
        nt = sum(1 for x in s if x["type"] == "text")
        ns = sum(1 for x in s if x["type"] == "slide")
        intros = sum(1 for x in s if x["type"] == "text" and INTRO_RE.search(x.get("text", "")))
        flag = "  <-- MULTIPLE INTROS, possible mega-folder" if intros >= 2 else ""
        print(f"  {os.path.basename(d)}")
        print(f"     {m.get('first_ts')}->{m.get('last_ts')}  text={nt} slides={ns} "
              f"self-intros={intros}{flag}")

    print("\n## Transition markers per talk (index, ts, text)")
    for d in dirs:
        s = load_session(d)
        print(f"\n### {os.path.basename(d)}")
        for i, x in enumerate(s):
            if x["type"] == "text" and MARKER_RE.search(x.get("text", "")):
                print(f"   [{i:4}] {x['ts']}  {x['text'][:96]}")

    print("\n## Boundary windows (tail of N -> head of N+1)")
    for a, b in zip(dirs, dirs[1:]):
        sa, sb = load_session(a), load_session(b)
        print(f"\n### {os.path.basename(a)}  ->  {os.path.basename(b)}")
        print("  TAIL:")
        for i in range(max(0, len(sa) - context), len(sa)):
            print(f"   [{i:4}] {sa[i]['ts']}  {_txt(sa[i])}")
        print("  HEAD:")
        for i in range(min(context, len(sb))):
            print(f"   [{i:4}] {sb[i]['ts']}  {_txt(sb[i])}")


# ── apply ────────────────────────────────────────────────────────────────────
def _render_html(label, m, session):
    title = html.escape(label or m.get("saved_at", ""))
    rows = []
    for it in session:
        if it["type"] == "slide":
            rows.append(f'<figure><img src="{it["file"]}" loading="lazy">'
                        f'<figcaption>{html.escape(it.get("ts", ""))}</figcaption></figure>')
        else:
            rows.append(f'<p>{html.escape(it["text"])}</p>')
    body = "\n".join(rows)
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title} — Transcript</title>
  <style>
    body {{ background:#0a0a0a; color:#ddd; font-family:sans-serif;
           max-width:820px; margin:0 auto; padding:48px 24px 96px; line-height:1.8; }}
    h1 {{ font-size:1.4em; font-weight:600; margin-bottom:0.2em; }}
    .meta {{ color:#555; font-size:0.8em; letter-spacing:1px; margin-bottom:2.5em; }}
    p {{ color:#bbb; margin:0.2em 0; }}
    figure {{ margin:2em 0; }}
    figure img {{ width:100%; border:1px solid #222; border-radius:6px; }}
    figcaption {{ color:#555; font-size:0.7em; letter-spacing:1px; margin-top:6px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">Saved {html.escape(m.get("saved_at", ""))} · {m.get("n_slides", 0)} slides · {m.get("n_text_chunks", 0)} text chunks</div>
  {body}
</body>
</html>
"""


def _unlock(d):
    if not os.path.exists(d):
        return
    os.chmod(d, 0o755)
    for r, ds, fs in os.walk(d):
        for x in ds + fs:
            p = os.path.join(r, x)
            os.chmod(p, 0o755 if os.path.isdir(p) else 0o644)


def _lock(d):
    for r, ds, fs in os.walk(d):
        for f in fs:
            try:
                os.chmod(os.path.join(r, f), 0o444)
            except OSError:
                pass
    for r, ds, fs in os.walk(d, topdown=False):
        for x in ds:
            try:
                os.chmod(os.path.join(r, x), 0o555)
            except OSError:
                pass
    os.chmod(d, 0o555)


def _build(dest, tagged, label, saved_at):
    """tagged: list of (item, src_dir). Rebuild dest fresh, copying+renumbering
    each slide from the item's own source folder."""
    _unlock(dest)
    # Carry over derived artefacts the tool does not regenerate (summaries,
    # questions) so a boundary re-cut never silently drops them. They may be
    # stale if the talk's content changed materially — run
    # `backfill_summaries.py --force <folder>` afterwards to refresh.
    carried = {}
    for fn in ("summary.json", "summary.md", "questions.json"):
        p = os.path.join(dest, fn)
        if os.path.exists(p):
            with open(p, "rb") as fh:
                carried[fn] = fh.read()
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    slides_dir = os.path.join(dest, "slides")
    new, n = [], 0
    for it, srcdir in tagged:
        if it["type"] == "slide":
            os.makedirs(slides_dir, exist_ok=True)
            n += 1
            fn = f"slide_{n:03d}.jpg"
            shutil.copyfile(os.path.join(srcdir, it["file"]), os.path.join(slides_dir, fn))
            new.append({"type": "slide", "ts": it.get("ts", ""), "file": f"slides/{fn}"})
        else:
            new.append({"type": "text", "ts": it.get("ts", ""), "text": it["text"]})
    texts = [x for x in new if x["type"] == "text"]
    m = {"label": label, "saved_at": saved_at, "n_slides": n, "n_text_chunks": len(texts),
         "first_ts": new[0].get("ts", "") if new else "",
         "last_ts": new[-1].get("ts", "") if new else ""}
    open(dest + "/session.json", "w").write(json.dumps(new, indent=2))
    open(dest + "/transcript.txt", "w").write(" ".join(x["text"] for x in texts).strip() + "\n")
    open(dest + "/metadata.json", "w").write(json.dumps(m, indent=2))
    open(dest + "/transcript.html", "w").write(_render_html(label, m, new))
    for fn, blob in carried.items():
        with open(os.path.join(dest, fn), "wb") as fh:
            fh.write(blob)
    _lock(dest)
    return len(texts), n, m["first_ts"], m["last_ts"]


def apply(plan_path, base=ROOT, date=None, dry_run=False):
    plan = json.load(open(plan_path))
    ops = plan["operations"]
    date = date or datetime.now().strftime("%Y-%m-%d")
    bk_root = os.path.join(base, ".backups", f"{date}_{plan.get('backup_label', 'recut')}")

    # 1. Back up every folder referenced as a source or an (existing) destination.
    refs = set()
    for op in ops:
        for seg in op["segments"]:
            refs.add(seg["src"])
        if not op.get("bin"):
            refs.add(op["dest"])
    print(f"backup -> {bk_root}")
    if not dry_run:
        os.makedirs(bk_root, exist_ok=True)
    for name in sorted(refs):
        src = os.path.join(base, name)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(bk_root, name)
        if dry_run:
            print(f"  would back up {name}")
            continue
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
            _unlock(dst)

    def src_path(name):  # always read sources from the immutable backup
        p = os.path.join(bk_root, name)
        return p if os.path.isdir(p) else os.path.join(base, name)

    # 2. Build each destination from its segments.
    for op in ops:
        dest_name = op["dest"]
        dest = os.path.join(base, "bin", dest_name) if op.get("bin") else os.path.join(base, dest_name)
        tagged = []
        default_label, default_saved = "", ""
        for k, seg in enumerate(op["segments"]):
            sp = src_path(seg["src"])
            sess = json.load(open(os.path.join(sp, "session.json")))
            if k == 0:
                try:
                    sm = json.load(open(os.path.join(sp, "metadata.json")))
                    default_label, default_saved = sm.get("label", ""), sm.get("saved_at", "")
                except OSError:
                    pass
            for it in sess[seg["start"]:seg["end"]]:
                tagged.append((it, sp))
        label = op.get("label", default_label)
        saved_at = op.get("saved_at", default_saved)
        tag = "bin/" if op.get("bin") else ""
        if dry_run:
            nt = sum(1 for it, _ in tagged if it["type"] == "text")
            ns = sum(1 for it, _ in tagged if it["type"] == "slide")
            print(f"  would build {tag}{dest_name}: {nt} text, {ns} slides")
            continue
        if op.get("bin"):
            os.makedirs(os.path.join(base, "bin"), exist_ok=True)
        nt, ns, ft, lt = _build(dest, tagged, label, saved_at)
        print(f"  built {tag}{dest_name}: {nt} text, {ns} slides, {ft}->{lt}")

    if not dry_run:
        _verify(base)


def _verify(base=ROOT):
    print("\nverify: slide references")
    bad = 0
    for d in talk_dirs(base) + _bin_dirs(base):
        s = load_session(d)
        for x in s:
            if x["type"] == "slide" and not os.path.exists(os.path.join(d, x["file"])):
                print(f"  MISSING {os.path.basename(d)} {x['file']}")
                bad += 1
    print("  all slide references resolve" if not bad else f"  {bad} missing slide(s)")


def _bin_dirs(base=ROOT):
    b = os.path.join(base, "bin")
    if not os.path.isdir(b):
        return []
    return [os.path.join(b, n) for n in sorted(os.listdir(b))
            if os.path.exists(os.path.join(b, n, "metadata.json"))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("analyze", help="report timeline, markers, boundary windows")
    pa.add_argument("--context", type=int, default=6)
    pp = sub.add_parser("apply", help="execute a recut plan (JSON)")
    pp.add_argument("plan")
    pp.add_argument("--dry-run", action="store_true")
    pp.add_argument("--date", help="override backup date stamp (YYYY-MM-DD)")
    args = ap.parse_args()
    if args.cmd == "analyze":
        analyze(context=args.context)
    elif args.cmd == "apply":
        apply(args.plan, date=args.date, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
