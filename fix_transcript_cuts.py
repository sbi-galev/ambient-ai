#!/usr/bin/env python3
"""Inspect and re-cut saved talk folders — entirely on-box, no privacy leak.

Talks are saved by transcript_server.py as a slice of one continuous recording,
cut by a manual "End talk & Save" click. Because the click is mistimed relative
to speech-to-text lag, content leaks across boundaries: a late click traps the
next speaker's opening at the END of the current folder; an early click spills
the current speaker's ending into the START of the next.

This tool fixes that without sending any transcript to an external service. It
has three subcommands:

  analyze   Print a chronological timeline, transition markers (chair handoffs /
            speaker self-intros) and the tail/head window of every consecutive
            pair of talks. Pure local inspection.

  propose   Ask the LOCAL summariser LLM (config.toml [llm], the same model that
            writes the summaries) to read each boundary window and decide where
            the next talk really begins, then emit a re-cut plan (JSON) you can
            review. This replaces the old Claude-driven skill: the transcripts
            only ever reach your own model, never a third party.

  apply     Execute a plan (JSON). Every folder that changes is rebuilt from
            scratch as an ordered concatenation of segments sliced out of
            backed-up copies of source folders; slides are copied and renumbered;
            transcript.txt / .html / metadata.json are regenerated in the
            server's exact format; folders are re-locked read-only. Originals are
            backed up under transcripts/.backups/ first.

Typical workflow (all on-box):

    python3 fix_transcript_cuts.py analyze
    python3 fix_transcript_cuts.py propose --out /tmp/plan.json
    # review /tmp/plan.json, then:
    python3 fix_transcript_cuts.py apply /tmp/plan.json --dry-run
    python3 fix_transcript_cuts.py apply /tmp/plan.json
    python3 backfill_summaries.py --force <changed-folder> [...]   # refresh summaries

`propose` only handles leakage between ADJACENT talks (the common case). Splitting
a "mega-folder" that bundles several talks or a break is still done with a
hand-written plan — `apply` supports `bin: true` dests and arbitrary segments for
that; see the plan schema below.

Plan schema:
  {
    "backup_label": "recut",                 # optional tag for the backup dir
    "operations": [
      {
        "dest": "2026-..._second-talk",      # folder NAME (created or overwritten)
        "bin": false,                         # true -> placed under transcripts/bin/
        "label": "Second talk",              # optional; defaults to first segment src's
        "saved_at": "2026-06-23T10:19:21",   # optional; defaults to first segment src's
        "segments": [
          {"src": "2026-..._first-talk", "start": 155, "end": 170},
          {"src": "2026-..._second-talk", "start": 0,   "end": 302}
        ]
      }
    ]
  }

Rules `apply` enforces:
  * Every folder whose contents change must appear as a `dest` with its COMPLETE
    new segment list (dests are overwritten wholesale; unlisted folders untouched).
  * start/end are Python slice bounds (end exclusive) into the SOURCE folder's
    original session.json. Use a large end (e.g. 9999) to mean "to the end".
  * Sources are read from the backup taken at the start of `apply`, so a folder
    may safely be both a source and a destination.
"""
import argparse
import html
import json
import os
import re
import shutil
import sys
from datetime import datetime

import transcript_server as ts   # single source of truth for paths + the local LLM

# Transcripts live where the server writes them; TRANSCRIPTS_DIR overrides.
ROOT = os.environ.get("TRANSCRIPTS_DIR") or str(ts.SAVE_DIR)

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


# ── propose (local LLM decides each boundary) ─────────────────────────────────
PROPOSE_SYSTEM = (
    "You are a careful editor finding the exact boundary between two consecutive "
    "conference talks in an automatic transcript. The 'Save' between them was "
    "mistimed, so a few chunks may sit in the wrong folder. Decide where the "
    "SECOND talk truly begins. A new talk starting is signalled by the chair "
    "thanking the previous speaker and introducing/welcoming the next ('thanks "
    "again', 'please join me in welcoming', 'over to you', 'next speaker'), or by "
    "a new speaker self-introducing ('hi, my name is', 'thanks for having me'). "
    "Expect ASR noise and missing punctuation. Respond with ONLY a JSON object."
)

PROPOSE_INSTRUCTIONS = (
    'Respond with ONLY this JSON (no prose, no code fence):\n'
    '{"decision": "ok" | "move_tail" | "move_head", "index": <int>, "reason": "<short>"}\n'
    '  - "ok": talk 2 already starts at its first shown chunk; nothing moves.\n'
    '  - "move_tail": talk 2 actually begins partway through talk 1\'s END. Set '
    '"index" to the [bracketed index] of the talk-1 chunk where talk 2 starts; '
    "those chunks (and any slides among them) move forward into talk 2.\n"
    '  - "move_head": talk 1 actually continues into talk 2\'s START. Set "index" '
    "to the [bracketed index] of the talk-2 chunk where talk 2's own content "
    "finally starts; the earlier talk-2 chunks move back into talk 1.\n"
    'Pick the single best index from the bracketed numbers shown. If unsure, "ok".'
)


def _parse_json(s):
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        s = s[i:j + 1]
    return json.loads(s)


def _decide_boundary(label_a, sa, label_b, sb, window, timeout):
    """Ask the local LLM where talk B truly begins. Returns (type, index, reason),
    with type in {ok, move_tail, move_head} and index validated into range."""
    tail_start = max(0, len(sa) - window)
    head_end = min(window, len(sb))
    lines_a = [f"  [{i}] {sa[i].get('ts','')}  {_txt(sa[i], 140)}"
               for i in range(tail_start, len(sa))]
    lines_b = [f"  [{i}] {sb[i].get('ts','')}  {_txt(sb[i], 140)}"
               for i in range(0, head_end)]
    user = (f'TALK 1 = "{label_a}", its END (chunks {tail_start}..{len(sa)-1}):\n'
            + "\n".join(lines_a)
            + f'\n\nTALK 2 = "{label_b}", its START (chunks 0..{head_end-1}):\n'
            + "\n".join(lines_b) + "\n\n" + PROPOSE_INSTRUCTIONS)
    msgs = [{"role": "system", "content": PROPOSE_SYSTEM},
            {"role": "user", "content": user}]
    raw = ts._llm_chat(msgs, max_tokens=300, temperature=0.0, timeout=timeout)
    try:
        obj = _parse_json(raw)
        dec = str(obj.get("decision", "ok")).strip()
        idx = int(obj.get("index", 0))
        reason = str(obj.get("reason", ""))[:200]
    except Exception as e:
        return "ok", 0, f"unparseable model reply, left unchanged ({e})"

    if dec == "move_tail":
        # talk 1's chunks [idx:] move into talk 2; idx must be a real tail index.
        if tail_start <= idx < len(sa) and idx > 0:
            return "move_tail", idx, reason
        return "ok", 0, f"move_tail index {idx} out of range, left unchanged"
    if dec == "move_head":
        # talk 2's chunks [:idx] move back into talk 1; idx must be a real head index.
        if 0 < idx <= head_end and idx < len(sb):
            return "move_head", idx, reason
        return "ok", 0, f"move_head index {idx} out of range, left unchanged"
    return "ok", 0, reason or "boundary already correct"


def propose(base=ROOT, window=14, timeout=120, out=None, backup_label="recut-llm"):
    dirs = talk_dirs(base)
    if len(dirs) < 2:
        print("Need at least two talks to check boundaries; nothing to do.", file=sys.stderr)
        return
    names = [os.path.basename(d) for d in dirs]
    sessions = [load_session(d) for d in dirs]
    metas = [load_meta(d) for d in dirs]

    for d, s, name in zip(dirs, sessions, names):
        intros = sum(1 for x in s if x["type"] == "text" and INTRO_RE.search(x.get("text", "")))
        if intros >= 2:
            print(f"! {name}: {intros} self-intros — possible mega-folder; propose only "
                  f"handles adjacent leakage, split this one with a hand-written plan.",
                  file=sys.stderr)

    print(f"Asking the local model ({ts.LLM_MODEL}) to check "
          f"{len(dirs)-1} boundaries …", file=sys.stderr)
    seams = []
    for i in range(len(dirs) - 1):
        typ, idx, reason = _decide_boundary(
            metas[i].get("label") or names[i], sessions[i],
            metas[i + 1].get("label") or names[i + 1], sessions[i + 1], window, timeout)
        seams.append({"type": typ, "index": idx})
        arrow = {"ok": "· ok", "move_tail": "← move_tail", "move_head": "move_head →"}[typ]
        suffix = f" @ {idx}" if typ != "ok" else ""
        print(f"  {names[i]}  ->  {names[i+1]}: {arrow}{suffix}  ({reason})", file=sys.stderr)

    # Translate seam decisions into per-folder segment lists. Each seam moves
    # chunks across one boundary only, so a folder's final content is its own
    # (possibly trimmed) chunks plus at most one neighbour fragment on each side.
    n = len(dirs)
    ops = []
    for m in range(n):
        left = seams[m - 1] if m > 0 else None
        right = seams[m] if m < n - 1 else None
        sh = left["index"] if (left and left["type"] == "move_head") else 0
        et = right["index"] if (right and right["type"] == "move_tail") else len(sessions[m])
        segments = []
        if left and left["type"] == "move_tail":      # gain prev talk's trapped tail
            segments.append({"src": names[m - 1], "start": left["index"],
                             "end": len(sessions[m - 1])})
        segments.append({"src": names[m], "start": sh, "end": et})
        if right and right["type"] == "move_head":     # gain next talk's spilled head
            segments.append({"src": names[m + 1], "start": 0, "end": right["index"]})

        changed = bool(segments[0]["src"] != names[m] or len(segments) > 1
                       or sh != 0 or et != len(sessions[m]))
        if not changed:
            continue
        # Guard: never emit an empty talk (can happen if both seams cut inward).
        kept = sum(max(0, min(s["end"], len(sessions[names.index(s["src"])])) - s["start"])
                   for s in segments)
        if kept <= 0:
            print(f"! {names[m]}: proposed cut would empty the folder — skipped.",
                  file=sys.stderr)
            continue
        ops.append({"dest": names[m],
                    "label": metas[m].get("label", ""),
                    "saved_at": metas[m].get("saved_at", ""),
                    "segments": segments})

    if not ops:
        print("\nNo boundary changes proposed — every adjacent split looks correct.",
              file=sys.stderr)
        return

    plan = {"backup_label": backup_label, "operations": ops}
    text = json.dumps(plan, indent=2)
    if out:
        with open(out, "w") as f:
            f.write(text + "\n")
        print(f"\nWrote plan with {len(ops)} changed folder(s) -> {out}", file=sys.stderr)
        print(f"Review it, then:\n"
              f"  python3 fix_transcript_cuts.py apply {out} --dry-run\n"
              f"  python3 fix_transcript_cuts.py apply {out}\n"
              f"  python3 backfill_summaries.py --force "
              + " ".join(op['dest'] for op in ops), file=sys.stderr)
    else:
        print(text)


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
    pr = sub.add_parser("propose", help="local LLM proposes a re-cut plan (JSON)")
    pr.add_argument("--out", help="write the plan here (default: stdout)")
    pr.add_argument("--window", type=int, default=14,
                    help="chunks of tail/head shown to the model per boundary (default 14)")
    pr.add_argument("--timeout", type=int, default=120, help="per-LLM-call timeout (s)")
    pr.add_argument("--backup-label", default="recut-llm")
    pp = sub.add_parser("apply", help="execute a recut plan (JSON)")
    pp.add_argument("plan")
    pp.add_argument("--dry-run", action="store_true")
    pp.add_argument("--date", help="override backup date stamp (YYYY-MM-DD)")
    args = ap.parse_args()
    if args.cmd == "analyze":
        analyze(context=args.context)
    elif args.cmd == "propose":
        propose(window=args.window, timeout=args.timeout, out=args.out,
                backup_label=args.backup_label)
    elif args.cmd == "apply":
        apply(args.plan, date=args.date, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
