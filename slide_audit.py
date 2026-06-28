#!/usr/bin/env python3
"""Per-slide audit: describe + triage every captured slide with the local model.

This is the SHARED CORE for slide cleanup, deliberately split from any one entry
point so the exact same judgement runs in three places:

  * on the fly   — transcript_server's /image handler can call audit_slide() the
                   moment a new slide is captured and stash desc/verdict on the
                   live history item (best-effort, async);
  * at save time — _save_session can persist whatever audit it has;
  * post-hoc     — this module's CLI scans saved talks and writes a central
                   sidecar, transcripts/slides_audit.json.

audit_slide() is stateless and per-slide (description + content-vs-junk +
confidence). Deciding which TALK a boundary slide belongs to needs the
neighbours and lives in fix_transcript_cuts.py (slides-propose). Nothing here
ever deletes a slide — it only describes and flags; removals are applied, with a
human gate, through fix_transcript_cuts.py slides-apply (reversible).

CLI:
    python3 slide_audit.py                 # audit all slides -> slides_audit.json
    python3 slide_audit.py --talk SLUG     # limit to one talk (repeatable)
    python3 slide_audit.py --force         # re-audit even if already recorded

The sidecar maps {folder: {slide_file: {desc, content, confidence, reason, ts}}}
and is updated incrementally, so re-running only audits new/forced slides.
"""
import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

import transcript_server as ts

AUDIT_PATH = ts.SAVE_DIR / "slides_audit.json"

SYSTEM = (
    "You caption and triage ONE screenshot captured automatically during a "
    "conference talk. Reply with ONLY a JSON object:\n"
    '{"desc": "<<=15-word factual description of what is visible>", '
    '"content": true|false, "confidence": <0.0-1.0>, "reason": "<short>"}\n'
    "content=true if it is a real presentation slide a speaker is showing (title "
    "slide, figure, plot, equation, bulleted text, code/demo output being "
    "presented). content=false if it is NON-content: the OS desktop/home screen, a "
    "screen-share or 'mirror/extend display' dialog, the presentation-software "
    "editor (PowerPoint/Keynote/Slides edit view), a video-call window (Zoom/Teams/"
    "Meet), a file picker or browser chrome or terminal opened for setup, a blank/"
    "black/white frame, or a photo of the room/stage during a hand-over. "
    "confidence is how sure you are of the content judgement (1.0 = certain)."
)


def _slug(name):
    return re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_", "", os.path.basename(name))


def talk_identity(d: Path) -> str:
    """'Speaker — Title' for a talk dir, to ground the description; falls back to
    the operator label / slug."""
    speaker = title = label = ""
    try:
        label = json.loads((d / "metadata.json").read_text()).get("label", "")
    except Exception:
        pass
    sp = d / "summary.json"
    if sp.exists():
        try:
            sm = json.loads(sp.read_text())
            speaker, title = sm.get("speaker") or "", sm.get("title") or ""
        except Exception:
            pass
    title = title or label or _slug(d.name)
    return (f"{speaker} — {title}" if speaker else title).strip(" —")


def audit_slide(b64: str, talk_ctx: str = "", timeout: int = 120) -> dict:
    """Describe + triage one slide (given as base64 JPEG). Returns
    {desc, content: bool, confidence: float, reason}. Raises on transport error;
    returns a low-confidence 'content' verdict if the reply can't be parsed (so a
    parse glitch never culls a slide)."""
    user = [{"type": "text",
             "text": (f"This frame is from the talk: {talk_ctx}\n" if talk_ctx else "")
                     + "Describe and triage it."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]
    raw = ts._llm_chat([{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user}],
                       max_tokens=220, temperature=0.0, timeout=timeout)
    try:
        s = raw.strip()
        i, j = s.find("{"), s.rfind("}")
        obj = json.loads(s[i:j + 1])
        conf = float(obj.get("confidence", 0.0))
        return {"desc": str(obj.get("desc", "")).strip()[:300],
                "content": bool(obj.get("content", True)),
                "confidence": max(0.0, min(1.0, conf)),
                "reason": str(obj.get("reason", "")).strip()[:200]}
    except Exception as e:
        return {"desc": "", "content": True, "confidence": 0.0,
                "reason": f"unparseable model reply ({e})"}


# ── sidecar load/save ────────────────────────────────────────────────────────

def load_audit(path: Path = AUDIT_PATH) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def _save_audit(data: dict, path: Path = AUDIT_PATH):
    ts._atomic_write_bytes(path, json.dumps(data, indent=2).encode())


# ── CLI: scan saved talks ────────────────────────────────────────────────────

def _talk_dirs(only=None):
    out = []
    for d in sorted(ts.SAVE_DIR.iterdir()):
        if not d.is_dir() or not (d / "metadata.json").exists():
            continue
        if only and not (d.name in only or _slug(d.name) in only):
            continue
        out.append(d)
    return out


def run(only=None, force=False, timeout=120):
    audit = load_audit()
    total = done = 0
    for d in _talk_dirs(only):
        sd = d / "slides"
        slides = sorted(sd.glob("slide_*.jpg")) if sd.is_dir() else []
        if not slides:
            continue
        rec = audit.setdefault(d.name, {})
        ctx = talk_identity(d)
        pending = [f for f in slides if force or f.name not in rec]
        if not pending:
            continue
        print(f"  {_slug(d.name)}: {len(pending)} slide(s) to audit", flush=True)
        for f in pending:
            total += 1
            b64 = base64.b64encode(f.read_bytes()).decode()
            r = audit_slide(b64, ctx, timeout)
            r["ts"] = ""  # filled from session.json below if available
            rec[f.name] = r
            done += 1
            tag = "JUNK" if not r["content"] else "ok"
            print(f"    {f.name} [{tag} {r['confidence']:.2f}] {r['desc'][:70]}", flush=True)
        # backfill timestamps from session.json (best-effort)
        sj = d / "session.json"
        if sj.exists():
            try:
                for it in json.loads(sj.read_text()):
                    if it.get("type") == "slide":
                        fn = os.path.basename(it.get("file", ""))
                        if fn in rec:
                            rec[fn]["ts"] = it.get("ts", "")
            except Exception:
                pass
        _save_audit(audit)  # checkpoint after each talk
    print(f"\naudited {done} slide(s) → {AUDIT_PATH.name}"
          if done else "nothing to audit (all cached; use --force to redo).")


def main():
    ap = argparse.ArgumentParser(description="Audit (describe + triage) saved slides.")
    ap.add_argument("--talk", action="append", metavar="SLUG",
                    help="limit to this talk (folder name or slug); repeatable")
    ap.add_argument("--force", action="store_true", help="re-audit cached slides")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()
    run(only=set(args.talk) if args.talk else None, force=args.force, timeout=args.timeout)


if __name__ == "__main__":
    sys.exit(main())
