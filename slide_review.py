#!/usr/bin/env python3
"""Turn the slide audit + boundary plan into a human review surface.

Reads:
  * transcripts/slides_audit.json   (from slide_audit.py: desc/content/confidence)
  * a boundary plan JSON            (from fix_transcript_cuts.py slides-propose:
                                     the slide->neighbour-talk reassignments)

Splits the proposed changes per the agreed policy:
  * AUTO-CULL   high-confidence junk (content=false, confidence >= threshold) is
                written to an auto-cull plan you can apply directly (reversible).
  * REVIEW      everything else that changes — borderline junk and every boundary
                reassignment — is rendered into a self-contained review.html with
                the slide image, its description and a keep / cull / move control,
                defaulting to the recommendation. You adjust, then click
                "Download approved plan" to get a slides-apply plan.

Apply (both through the reversible rebuild engine):
    python3 fix_transcript_cuts.py slides-apply auto_cull.json
    python3 fix_transcript_cuts.py slides-apply approved_slides.json   # your download

Usage:
    python3 slide_review.py [--boundary PLAN.json] [--threshold 0.85]
                            [--out site/review.html] [--autocull-out auto_cull.json]
"""
import argparse
import base64
import html
import json
import re
from pathlib import Path

import transcript_server as ts
from slide_audit import load_audit, AUDIT_PATH

DEFAULT_THRESHOLD = 0.85


def _slug(name):
    return re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_", "", name)


def _img_data_uri(folder, fn):
    p = ts.SAVE_DIR / folder / "slides" / fn
    if not p.is_file():
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()


def build(boundary_path=None, threshold=DEFAULT_THRESHOLD,
          out="site/review.html", autocull_out="auto_cull.json"):
    audit = load_audit()
    if not audit:
        raise SystemExit(f"no audit found at {AUDIT_PATH} — run slide_audit.py first.")

    moves = []
    if boundary_path and Path(boundary_path).exists():
        moves = [m for m in json.loads(Path(boundary_path).read_text()).get("moves", [])
                 if m.get("dest") != "CULL"]
    move_keys = {(m["src"], m["file"]) for m in moves}

    auto_cull, review = [], []
    for folder, files in audit.items():
        for fn, rec in files.items():
            file = f"slides/{fn}"
            if (folder, file) in move_keys:
                continue  # a reassignment owns this slide; reviewed as a move
            if not rec.get("content", True):
                item = {"folder": folder, "file": file, "fn": fn,
                        "ts": rec.get("ts", ""), "desc": rec.get("desc", ""),
                        "reason": rec.get("reason", ""),
                        "confidence": rec.get("confidence", 0.0)}
                if rec.get("confidence", 0.0) >= threshold:
                    auto_cull.append(item)
                else:
                    item["kind"] = "cull"
                    review.append(item)
    for m in moves:
        folder, fn = m["src"], m["file"].split("/")[-1]
        rec = audit.get(folder, {}).get(fn, {})
        review.append({"folder": folder, "file": m["file"], "fn": fn,
                       "ts": m.get("ts", "") or rec.get("ts", ""),
                       "desc": rec.get("desc", ""),
                       "reason": m.get("reason", ""),
                       "confidence": rec.get("confidence", 1.0),
                       "kind": "move", "dest": m["dest"],
                       "audit_junk": not rec.get("content", True)})

    # Auto-cull plan (applied directly).
    plan = {"type": "slide-reassign", "backup_label": "autocull",
            "moves": [{"src": i["folder"], "file": i["file"], "ts": i["ts"],
                       "dest": "CULL", "reason": i["reason"] or i["desc"]}
                      for i in auto_cull]}
    Path(autocull_out).write_text(json.dumps(plan, indent=2))

    _render(review, out, threshold, len(auto_cull))
    print(f"auto-cull: {len(auto_cull)} high-confidence junk slide(s) -> {autocull_out}")
    print(f"review:    {len(review)} slide(s) for the admin -> {out}")
    print(f"\n  apply auto-culls:  python3 fix_transcript_cuts.py slides-apply {autocull_out}")
    print(f"  review the rest:   open {out} (served by your local http server)")


def _render(items, out, threshold, n_autocull):
    esc = html.escape
    cards = []
    for k, it in enumerate(items):
        uri = _img_data_uri(it["folder"], it["fn"])
        talk = esc(_slug(it["folder"]))
        conf = it.get("confidence", 0.0)
        if it["kind"] == "move":
            dest = esc(_slug(it["dest"]))
            rec_label = f"reassign → {dest}"
            warn = ('<span class="warn">audit thinks this may be junk</span>'
                    if it.get("audit_junk") else "")
            controls = (
                f'<label><input type="radio" name="a{k}" value="keep"> keep in {talk}</label>'
                f'<label><input type="radio" name="a{k}" value="move" checked> move → {dest}</label>'
                f'<label><input type="radio" name="a{k}" value="cull"> cull</label>')
            data_dest = esc(it["dest"])
        else:
            rec_label = "cull (likely junk)"
            warn = ""
            controls = (
                f'<label><input type="radio" name="a{k}" value="keep"> keep in {talk}</label>'
                f'<label><input type="radio" name="a{k}" value="cull" checked> cull</label>')
            data_dest = ""
        cards.append(f"""
    <div class="rcard" data-k="{k}" data-src="{esc(it['folder'])}"
         data-file="{esc(it['file'])}" data-ts="{esc(it['ts'])}" data-dest="{data_dest}">
      <div class="rimg">{'<img src="' + uri + '">' if uri else '<div class=miss>image missing</div>'}</div>
      <div class="rbody">
        <div class="rtalk">{talk} · {esc(it['ts'])} · conf {conf:.2f} {warn}</div>
        <div class="rdesc">{esc(it['desc']) or '<i>no description</i>'}</div>
        <div class="rreason">{esc(it['reason'])}</div>
        <div class="rrec">recommended: <b>{rec_label}</b></div>
        <div class="rctrl">{controls}</div>
      </div>
    </div>""")

    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Slide review</title>
<style>
  body {{ background:#0a0a0a; color:#e6e6e6; font-family:sans-serif; margin:0;
    padding:24px 20px 120px; font-size:16px; }}
  .wrap {{ max-width:1000px; margin:0 auto; }}
  h1 {{ font-size:1.4em; color:#fff; }}
  .lede {{ color:#9a9a9a; margin:0.5em 0 1.5em; line-height:1.6; }}
  .lede b {{ color:#9ec9b0; }}
  .rcard {{ display:flex; gap:18px; padding:18px 0; border-top:1px solid #1a1a1a; }}
  .rimg {{ flex:0 0 300px; }}
  .rimg img {{ width:100%; border:1px solid #222; border-radius:6px; display:block; }}
  .rimg .miss {{ color:#a55; font-size:0.8em; }}
  .rbody {{ flex:1; min-width:0; }}
  .rtalk {{ color:#777; font-size:0.8em; letter-spacing:0.5px; margin-bottom:0.4em; }}
  .warn {{ color:#d2a24c; margin-left:8px; }}
  .rdesc {{ color:#ddd; line-height:1.5; margin-bottom:0.4em; }}
  .rreason {{ color:#8a8a8a; font-size:0.85em; line-height:1.5; margin-bottom:0.6em; }}
  .rrec {{ color:#9a9; font-size:0.85em; margin-bottom:0.7em; }}
  .rctrl label {{ margin-right:16px; font-size:0.9em; color:#cdd; cursor:pointer; }}
  .bar {{ position:fixed; left:0; right:0; bottom:0; background:#111; border-top:1px solid #2a2a2a;
    padding:14px 20px; display:flex; align-items:center; gap:18px; justify-content:center; }}
  .bar .count {{ color:#9a9a9a; font-size:0.9em; }}
  .bar button {{ background:#13321f; border:1px solid #3a5a48; color:#9ec9b0; font-size:1em;
    padding:10px 20px; border-radius:8px; cursor:pointer; }}
  .bar button:hover {{ background:#16401f; color:#fff; }}
</style></head>
<body><div class="wrap">
  <h1>Slide review</h1>
  <p class="lede">{n_autocull} high-confidence junk slide(s) were split into
    <b>auto_cull.json</b> (apply separately). Below are the <b>{len(items)}</b> slides
    that need your eye — borderline junk and every cross-talk reassignment.
    Adjust any you disagree with, then <b>Download approved plan</b> and run
    <code>slides-apply</code> on it. Nothing changes until you do.</p>
  {''.join(cards)}
</div>
<div class="bar">
  <span class="count" id="count"></span>
  <button onclick="download_plan()">Download approved plan</button>
</div>
<script>
function summarize() {{
  var keep=0, cull=0, move=0;
  document.querySelectorAll('.rcard').forEach(function(c) {{
    var v=(c.querySelector('input:checked')||{{}}).value;
    if(v==='cull') cull++; else if(v==='move') move++; else keep++;
  }});
  document.getElementById('count').textContent =
    move+' reassign · '+cull+' cull · '+keep+' keep';
}}
document.addEventListener('change', summarize); summarize();
function download_plan() {{
  var moves=[];
  document.querySelectorAll('.rcard').forEach(function(c) {{
    var v=(c.querySelector('input:checked')||{{}}).value;
    if(v==='keep'||!v) return;
    moves.push({{src:c.dataset.src, file:c.dataset.file, ts:c.dataset.ts,
      dest: v==='cull' ? 'CULL' : c.dataset.dest, reason:'admin-approved'}});
  }});
  var plan={{type:'slide-reassign', backup_label:'review', moves:moves}};
  var blob=new Blob([JSON.stringify(plan,null,2)],{{type:'application/json'}});
  var a=document.createElement('a');
  a.href=URL.createObjectURL(blob); a.download='approved_slides.json'; a.click();
}}
</script>
</body></html>"""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(page)


def main():
    ap = argparse.ArgumentParser(description="Build the slide review page + auto-cull plan.")
    ap.add_argument("--boundary", default="/tmp/slides.json",
                    help="boundary reassign plan from slides-propose (default /tmp/slides.json)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"auto-cull confidence cutoff (default {DEFAULT_THRESHOLD})")
    ap.add_argument("--out", default="site/review.html")
    ap.add_argument("--autocull-out", default="auto_cull.json")
    args = ap.parse_args()
    build(boundary_path=args.boundary, threshold=args.threshold,
          out=args.out, autocull_out=args.autocull_out)


if __name__ == "__main__":
    main()
