"""SBI4GALEV live transcript server.

Loads stt_en_fastconformer_transducer_xxlarge once on startup, then serves:
  POST /transcribe  — audio file → NeMo inference → pushes text to SSE
  POST /image       — JPEG screenshot → pushes to SSE if slide changed
  POST /push        — raw text → pushes to SSE (fallback/testing)
  POST /vote        — public: record a thumbs up/down on a speaker question
  POST /save        — bundles the current talk (transcript + slides) into a
                      dedicated, read-only folder, then resets for the next
                      speaker. Token-protected; not exposed on the public page.
  GET  /stream      — SSE stream for browsers
  GET  /            — public, view-only transcript viewer
  GET  /admin?token=… — operator viewer with the End-talk/Save control
  GET  /summaries   — AI summary of every saved talk (alan / local gemma)
  GET  /talks/…     — read-only static access to saved slides/summaries; the
                      full transcript files require the admin token (?token=…)

After a talk is saved, a background worker sends its transcript + slides to the
local model (alan's engine: gemma-4-31b via SGLang) and writes summary.json /
summary.md into the talk folder, then finalises the folder read-only.

There is deliberately no endpoint that deletes or mutates saved talks: once a
talk is finalised its folder is made read-only and cannot be removed via HTTP.
"""
import sys
import types

# torchaudio CUDA version check fails if torchaudio/PyTorch were built against
# different CUDA versions (e.g. after a PyTorch update). Stub the extension
# module to bypass the check — NeMo only needs torchaudio for transforms, not
# the C++ extension.
_fake_ext = types.ModuleType('torchaudio._extension')
_fake_ext._check_cuda_version = lambda: None
_fake_ext._IS_TORCHAUDIO_EXT_AVAILABLE = False
_fake_ext._IS_ALIGN_AVAILABLE = False
_fake_ext.fail_if_no_align = lambda f: f
sys.modules['torchaudio._extension'] = _fake_ext

import base64
import html
import io
import json
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote, quote
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SAVE_DIR = Path(__file__).parent / "transcripts"
SAVE_DIR.mkdir(exist_ok=True)
# Talk folders are read-only once finalised, so audience votes on speaker
# questions live in a separate writable store, one JSON per talk.
VOTES_DIR = Path(__file__).parent / "votes"
VOTES_DIR.mkdir(exist_ok=True)
# Per-day editorial overviews ("Day 1", "Day 2", …). Generated from the day's
# talk summaries and cached here (one JSON per calendar date), regenerated in the
# background when the set of talks for that day changes.
DAYS_DIR = Path(__file__).parent / "day_summaries"
DAYS_DIR.mkdir(exist_ok=True)
# Conference-wide "key topics": a single synthesis across every saved talk that
# clusters their topic tags into the meeting's main threads, links each back to
# the talks that raised it, and powers the /topics graph. Cached as one JSON,
# regenerated in the background when the set of talks (or their summaries) changes.
TOPICS_DIR = Path(__file__).parent / "topic_summaries"
TOPICS_DIR.mkdir(exist_ok=True)
TOPICS_PATH = TOPICS_DIR / "topics.json"
TOPICS_MAX = int(os.environ.get("TOPICS_MAX", "14"))

PORT = int(os.environ.get("TRANSCRIPT_PORT", "7103"))
TOKEN = os.environ.get("TRANSCRIPT_TOKEN", "sbi4galev")
DEVICE = os.environ.get("STT_DEVICE", "cuda:0")
MODEL_NAME = "stt_en_fastconformer_transducer_xxlarge"
SAMPLE_RATE = 16000

# "End talk & Save" is invariably clicked a beat late — by the time the chair has
# thanked the speaker and the next one starts, the last few seconds of audio (the
# next speaker's opening) have already landed in this talk's history. We hold back
# the most recent SAVE_OFFSET_SECONDS of chunks at save time, leaving them in the
# live history so they bundle with the *next* talk instead. Set to 0 to disable.
SAVE_OFFSET_SECONDS = int(os.environ.get("SAVE_OFFSET_SECONDS", "30"))

# Crash safety: the live transcript buffer lives in memory and is only written to
# disk when a talk is explicitly saved. To survive a crash/restart mid-talk we
# periodically snapshot the in-memory history to a hidden file under SAVE_DIR and
# reload it on startup, so the in-progress (unsaved) talk resumes instead of being
# lost. Set LIVE_FLUSH_SECONDS=0 to disable.
LIVE_STATE_PATH = SAVE_DIR / ".live.json"
LIVE_FLUSH_SECONDS = int(os.environ.get("LIVE_FLUSH_SECONDS", "10"))

# Local summarisation model — alan's engine: SGLang serving gemma-4-31b-it on an
# OpenAI-compatible endpoint. Override via env to repoint at another provider.
LLM_URL = os.environ.get("ALAN_LLM_URL", "http://127.0.0.1:30000/v1/chat/completions")
LLM_MODEL = os.environ.get("ALAN_LLM_MODEL", "google/gemma-4-31b-it")
SUMMARIES_ENABLED = os.environ.get("SUMMARIES", "1") != "0"
SUMMARY_MAX_SLIDES = int(os.environ.get("SUMMARY_MAX_SLIDES", "8"))

_subscribers = []
_lock = threading.Lock()
_infer_lock = threading.Lock()
_save_lock = threading.Lock()
_summary_lock = threading.Lock()
_votes_lock = threading.Lock()
_day_inflight = set()          # calendar dates whose overview is being (re)built
_day_inflight_lock = threading.Lock()
_topics_lock = threading.Lock()
_topics_inflight = False        # True while the key-topics synthesis is running
_topics_started_at = 0.0        # epoch the in-flight synthesis began (drives the timer)
_live_dirty = threading.Event()  # set when _history changes; drives the autosave
_history = []  # list of {"type": "text"|"image", "data": str, "ts": str}

# ── Page template ───────────────────────────────────────────────────────────
# Shared between the public viewer (/) and the operator viewer (/admin). The
# admin-only Save controls are injected via the /*ADMIN_*/ placeholders so the
# public page never carries the auth token or a way to end a talk.

PAGE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Live Transcript</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0a0a0a; color: #e0e0e0; font-family: sans-serif; overflow: hidden; }

    #container {
      height: 100vh;
      overflow-y: scroll;
      scroll-behavior: auto;
    }
    /* hide scrollbar */
    #container::-webkit-scrollbar { display: none; }
    #container { -ms-overflow-style: none; scrollbar-width: none; }

    .section {
      height: 100vh;
      display: grid;
      grid-template-columns: 58% 42%;
      opacity: 0.2;
      transition: opacity 0.5s ease;
      padding: 0;
    }
    .section.text-only {
      grid-template-columns: 1fr;
    }
    .section.active { opacity: 1; }

    .slide-pane {
      position: relative;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 48px 32px 48px 48px;
      border-right: 1px solid #1a1a1a;
    }
    .slide-pane img {
      width: 100%;
      max-height: calc(100vh - 120px);
      object-fit: contain;
      border-radius: 6px;
      border: 1px solid #222;
    }
    .slide-timestamp {
      margin-top: 10px;
      font-size: 0.65em;
      color: #444;
      letter-spacing: 1px;
      align-self: flex-start;
    }

    .text-pane {
      display: flex;
      flex-direction: column;
      justify-content: center;
      padding: 48px 48px 48px 32px;
      overflow: hidden;
    }
    .section.text-only .text-pane {
      max-width: 720px;
      margin: 0 auto;
      padding: 80px 48px;
      justify-content: flex-start;
      padding-top: 15vh;
    }
    .label {
      font-size: 0.65em;
      color: #333;
      letter-spacing: 2px;
      text-transform: uppercase;
      margin-bottom: 1.4em;
    }
    .text-pane p {
      line-height: 1.85;
      font-size: 1.05em;
      color: #999;
    }
    .chunk.new { color: #fff; }

    #status {
      position: fixed; top: 16px; right: 20px;
      font-size: 0.65em; color: #333; letter-spacing: 1px; z-index: 100;
    }
    #status.live { color: #4a4; }

    #nav {
      position: fixed; right: 20px; top: 50%; transform: translateY(-50%);
      display: flex; flex-direction: column; gap: 6px; z-index: 100;
    }
    .nav-dot {
      width: 5px; height: 5px; border-radius: 50%;
      background: #333; cursor: pointer; transition: background 0.3s;
    }
    .nav-dot.active { background: #888; }

    #toplinks {
      position: fixed; top: 16px; left: 20px; z-index: 100;
      display: flex; gap: 14px;
    }
    #toplinks a {
      font-size: 0.65em; color: #444; letter-spacing: 1px; text-decoration: none;
    }
    #toplinks a:hover { color: #aaa; }
    /*ADMIN_CSS*/
  </style>
</head>
<body>
  <div id="container"></div>
  <div id="status">connecting…</div>
  <div id="toplinks"><a href="/summaries">▤ summaries</a><a href="/topics">✦ topics</a></div>
  <!--ADMIN_HTML-->
  <div id="nav"></div>

  <script>
    const container = document.getElementById('container');
    const statusEl = document.getElementById('status');
    const navEl = document.getElementById('nav');

    let sections = [];       // DOM .section elements
    let currentIdx = 0;
    let animating = false;
    let atLatest = true;     // whether user is on the newest section

    // ── Section management ──────────────────────────────────────────────

    function getOrCreatePrologue() {
      if (sections.length === 0) addSection(true);
      if (sections[sections.length - 1].dataset.type === 'text-only') {
        return sections[sections.length - 1];
      }
      return null;
    }

    function addSection(textOnly) {
      const sec = document.createElement('div');
      sec.className = 'section' + (textOnly ? ' text-only' : '');
      sec.dataset.type = textOnly ? 'text-only' : 'slide';

      if (!textOnly) {
        const slidePane = document.createElement('div');
        slidePane.className = 'slide-pane';
        sec.appendChild(slidePane);
      }

      const textPane = document.createElement('div');
      textPane.className = 'text-pane';
      const label = document.createElement('div');
      label.className = 'label';
      label.textContent = textOnly ? 'SBI4GALEV — Live Transcript' : 'Transcript';
      const p = document.createElement('p');
      textPane.appendChild(label);
      textPane.appendChild(p);
      sec.appendChild(textPane);

      container.appendChild(sec);
      sections.push(sec);
      observer.observe(sec);
      updateNav();
      return sec;
    }

    function currentTextP() {
      const sec = sections[sections.length - 1];
      return sec.querySelector('.text-pane p');
    }

    function appendText(text, isNew) {
      if (sections.length === 0) addSection(true);
      const p = currentTextP();
      const span = document.createElement('span');
      span.className = 'chunk' + (isNew ? ' new' : '');
      span.textContent = text + ' ';
      p.appendChild(span);
      if (isNew && atLatest) snapTo(sections.length - 1);
    }

    function appendImage(b64, timestamp) {
      // new section for this slide
      const sec = addSection(false);
      const slidePane = sec.querySelector('.slide-pane');
      const img = document.createElement('img');
      img.src = 'data:image/jpeg;base64,' + b64;
      slidePane.appendChild(img);
      const ts = document.createElement('div');
      ts.className = 'slide-timestamp';
      ts.textContent = timestamp;
      slidePane.appendChild(ts);
      if (atLatest) snapTo(sections.length - 1);
    }

    function appendItem(item, isNew) {
      if (item.type === 'image') {
        appendImage(item.data, item.ts || '');
      } else {
        // dim previous "new" spans
        if (isNew) document.querySelectorAll('.chunk.new').forEach(el => el.classList.remove('new'));
        appendText(item.data, isNew);
      }
    }

    function resetView() {
      container.innerHTML = ''; sections = []; navEl.innerHTML = '';
      currentIdx = 0; atLatest = true;
    }

    // ── SSE ─────────────────────────────────────────────────────────────

    const es = new EventSource('/stream');
    es.onopen = () => { statusEl.textContent = '● live'; statusEl.className = 'live'; };
    es.onerror = () => { statusEl.textContent = 'reconnecting…'; statusEl.className = ''; };
    es.addEventListener('history', e => {
      resetView();
      JSON.parse(e.data).forEach(item => appendItem(item, false));
    });
    es.addEventListener('chunk', e => appendItem({type:'text', data: e.data}, true));
    es.addEventListener('image', e => appendItem(JSON.parse(e.data), true));
    es.addEventListener('reset', () => resetView());

    // ── Intersection Observer (dim/undim) ────────────────────────────────

    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        const idx = sections.indexOf(entry.target);
        entry.target.classList.toggle('active', entry.isIntersecting);
        if (entry.isIntersecting && idx !== -1) {
          currentIdx = idx;
          updateNav();
          atLatest = (idx === sections.length - 1);
        }
      });
    }, { root: container, threshold: 0.5 });

    // ── Scroll with ease-out-cubic ───────────────────────────────────────

    function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }

    function snapTo(idx) {
      if (idx < 0 || idx >= sections.length || animating) return;
      currentIdx = idx;
      animating = true;
      const start = container.scrollTop;
      const end = sections[idx].offsetTop;
      const dist = end - start;
      const dur = 650;
      let t0 = null;
      function step(ts) {
        if (!t0) t0 = ts;
        const p = Math.min((ts - t0) / dur, 1);
        container.scrollTop = start + dist * easeOutCubic(p);
        if (p < 1) requestAnimationFrame(step);
        else { animating = false; updateNav(); }
      }
      requestAnimationFrame(step);
    }

    container.addEventListener('wheel', e => {
      e.preventDefault();
      if (animating) return;
      if (e.deltaY > 20) snapTo(Math.min(currentIdx + 1, sections.length - 1));
      else if (e.deltaY < -20) snapTo(Math.max(currentIdx - 1, 0));
    }, { passive: false });

    document.addEventListener('keydown', e => {
      if (e.key === 'ArrowDown' || e.key === 'PageDown') { e.preventDefault(); snapTo(currentIdx + 1); }
      if (e.key === 'ArrowUp'   || e.key === 'PageUp')   { e.preventDefault(); snapTo(currentIdx - 1); }
    });

    // ── Nav dots ─────────────────────────────────────────────────────────

    function updateNav() {
      navEl.innerHTML = '';
      sections.forEach((_, i) => {
        const dot = document.createElement('div');
        dot.className = 'nav-dot' + (i === currentIdx ? ' active' : '');
        dot.onclick = () => snapTo(i);
        navEl.appendChild(dot);
      });
    }

    /*ADMIN_JS*/
  </script>
</body>
</html>"""

ADMIN_CSS = """
    #save-bar {
      position: fixed; bottom: 20px; right: 20px; z-index: 100;
      display: flex; gap: 8px; align-items: center;
    }
    #talk-label {
      background: #111; color: #ccc; border: 1px solid #222; border-radius: 4px;
      padding: 8px 10px; font-size: 0.75em; width: 220px;
    }
    #talk-label:focus { outline: none; border-color: #444; }
    #save-btn {
      background: #111; color: #777; border: 1px solid #222;
      padding: 8px 18px; border-radius: 4px; cursor: pointer;
      font-size: 0.75em; letter-spacing: 1px;
      transition: color 0.2s, border-color 0.2s;
    }
    #save-btn:hover { color: #fff; border-color: #444; }
    #save-btn.saved { color: #4a4; border-color: #4a4; }
    #save-btn.busy { opacity: 0.6; pointer-events: none; }"""

ADMIN_HTML = """<div id="save-bar">
    <input id="talk-label" placeholder="speaker / talk title (optional)" autocomplete="off">
    <button id="save-btn" onclick="saveTalk()">End talk &amp; Save</button>
  </div>"""

ADMIN_JS = """
    const TOKEN = __TOKEN__;
    function saveTalk() {
      const btn = document.getElementById('save-btn');
      const labelEl = document.getElementById('talk-label');
      const label = labelEl ? labelEl.value.trim() : '';
      if (!confirm('End this talk and save? This starts a new recording for the next speaker.')) return;
      btn.classList.add('busy'); btn.textContent = 'Saving…';
      fetch('/save', {method:'POST', headers:{'X-Token': TOKEN, 'Content-Type':'application/json'},
                      body: JSON.stringify({label})})
        .then(r => r.json())
        .then(res => {
          btn.classList.remove('busy');
          if (res && res.saved) {
            btn.textContent = 'Saved ✓ ' + res.saved; btn.className = 'saved';
            if (labelEl) labelEl.value = '';
          } else {
            btn.textContent = (res && res.reason) ? 'Nothing to save' : 'Saved ✓';
          }
          setTimeout(() => { btn.textContent = 'End talk & Save'; btn.className = ''; }, 4000);
        })
        .catch(() => { btn.classList.remove('busy'); btn.textContent = 'Error';
                       setTimeout(() => { btn.textContent = 'End talk & Save'; }, 2500); });
    }"""


def render_page(admin: bool) -> str:
    if admin:
        return (PAGE_TEMPLATE
                .replace("/*ADMIN_CSS*/", ADMIN_CSS)
                .replace("<!--ADMIN_HTML-->", ADMIN_HTML)
                .replace("/*ADMIN_JS*/", ADMIN_JS.replace("__TOKEN__", json.dumps(TOKEN))))
    return (PAGE_TEMPLATE
            .replace("/*ADMIN_CSS*/", "")
            .replace("<!--ADMIN_HTML-->", "")
            .replace("/*ADMIN_JS*/", ""))


# ── Live-buffer autosave (crash recovery) ────────────────────────────────────

def _flush_live_state():
    """Atomically persist the current in-memory history so a crash/restart can
    resume the in-progress, not-yet-saved talk. A no-op-safe best effort: any
    failure is logged but never interrupts recording."""
    with _lock:
        snapshot = list(_history)
    try:
        _atomic_write_bytes(LIVE_STATE_PATH, json.dumps(snapshot).encode())
    except Exception as e:
        print(f"[live autosave warn] {e}", flush=True)


def _live_autosave_worker():
    """Background: whenever the history changes, wait a short coalescing window
    (so a burst of chunks becomes one write) then snapshot to disk."""
    while True:
        _live_dirty.wait()
        _live_dirty.clear()
        if LIVE_FLUSH_SECONDS > 0:
            time.sleep(LIVE_FLUSH_SECONDS)
        _flush_live_state()


def _load_live_state():
    """On startup, restore an in-progress talk left behind by a previous run."""
    if not LIVE_STATE_PATH.exists():
        return
    try:
        items = json.loads(LIVE_STATE_PATH.read_text())
    except Exception as e:
        print(f"[live restore warn] {e}", flush=True)
        return
    if isinstance(items, list) and items:
        with _lock:
            _history.extend(items)
        n_txt = sum(1 for it in items if it.get("type") == "text")
        n_img = sum(1 for it in items if it.get("type") == "image")
        print(f"Restored in-progress talk from autosave: "
              f"{n_txt} text chunk(s), {n_img} slide(s)", flush=True)


# ── Saving a talk ────────────────────────────────────────────────────────────

def _slugify(label: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (label or "").strip().lower()).strip("-")
    return s[:40]


def _unique_dir(base: Path, name: str) -> Path:
    cand = base / name
    i = 2
    while cand.exists():
        cand = base / f"{name}-{i}"
        i += 1
    return cand


def _make_readonly(top: Path):
    """Drop write permission on a saved talk so it can't be modified or removed
    in place (a 0o555 directory refuses unlink/create of its entries)."""
    for root, dirs, files in os.walk(top):
        for f in files:
            try:
                os.chmod(os.path.join(root, f), 0o444)
            except OSError:
                pass
    for root, dirs, files in os.walk(top, topdown=False):
        for d in dirs:
            try:
                os.chmod(os.path.join(root, d), 0o555)
            except OSError:
                pass
    try:
        os.chmod(top, 0o555)
    except OSError:
        pass


def _render_archive_html(label: str, meta: dict, session: list) -> str:
    title = html.escape(label or meta.get("saved_at", "Transcript"))
    rows = []
    for item in session:
        if item["type"] == "slide":
            cap = html.escape(item.get("ts", ""))
            rows.append(
                f'<figure><img src="{item["file"]}" loading="lazy">'
                f'<figcaption>{cap}</figcaption></figure>')
        else:
            rows.append(f'<p>{html.escape(item["text"])}</p>')
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
  <div class="meta">Saved {html.escape(meta.get("saved_at", ""))} · {meta.get("n_slides", 0)} slides · {meta.get("n_text_chunks", 0)} text chunks</div>
  {body}
</body>
</html>
"""


def _ts_to_seconds(ts: str):
    """'HH:MM:SS' → seconds since midnight, or None if unparseable."""
    try:
        h, m, s = (int(p) for p in ts.split(":"))
        return h * 3600 + m * 60 + s
    except (ValueError, AttributeError):
        return None


def _split_for_save(history, offset_seconds):
    """Return (to_save, to_keep): the saved prefix is everything older than
    `offset_seconds` before the most recent chunk; the recent tail is kept in
    the live history so it bundles with the next talk. Falls back to saving
    everything when the offset would leave nothing to save (e.g. a very short
    talk) so we never silently drop a whole talk."""
    if offset_seconds <= 0 or not history:
        return list(history), []
    last = next((_ts_to_seconds(it.get("ts", "")) for it in reversed(history)
                 if _ts_to_seconds(it.get("ts", "")) is not None), None)
    if last is None:
        return list(history), []
    cutoff = last - offset_seconds
    # History is appended in time order, so the kept tail is a suffix; split at
    # the first chunk whose timestamp falls within the offset window.
    split = len(history)
    for i, it in enumerate(history):
        sec = _ts_to_seconds(it.get("ts", ""))
        if sec is not None and sec > cutoff:
            split = i
            break
    if split == 0:  # whole talk inside the window — don't drop it
        return list(history), []
    return list(history[:split]), list(history[split:])


def _save_session(label: str):
    """Bundle the current talk into a dedicated read-only folder, then reset the
    live history for the next speaker. Returns a result dict, or None if there
    was nothing recorded (in which case nothing is written and nothing reset)."""
    with _save_lock:
        with _lock:
            snapshot, _ = _split_for_save(_history, SAVE_OFFSET_SECONDS)

        texts = [it for it in snapshot if it["type"] == "text"]
        images = [it for it in snapshot if it["type"] == "image"]
        if not texts and not images:
            return None

        ts_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        slug = _slugify(label)
        folder = f"{ts_name}_{slug}" if slug else ts_name
        talk_dir = _unique_dir(SAVE_DIR, folder)
        slides_dir = talk_dir / "slides"
        slides_dir.mkdir(parents=True)

        # Slides → slides/slide_NNN.jpg, building the ordered session as we go.
        session = []
        n_slides = 0
        for it in snapshot:
            if it["type"] == "image":
                n_slides += 1
                fname = f"slide_{n_slides:03d}.jpg"
                (slides_dir / fname).write_bytes(base64.b64decode(it["data"]))
                session.append({"type": "slide", "ts": it.get("ts", ""),
                                "file": f"slides/{fname}"})
            else:
                session.append({"type": "text", "ts": it.get("ts", ""),
                                "text": it["data"]})

        # Plain-text transcript.
        text = " ".join(it["data"] for it in texts).strip() + "\n"
        (talk_dir / "transcript.txt").write_text(text)

        meta = {
            "label": label or "",
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "n_slides": n_slides,
            "n_text_chunks": len(texts),
            "first_ts": snapshot[0].get("ts", ""),
            "last_ts": snapshot[-1].get("ts", ""),
        }
        (talk_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        (talk_dir / "session.json").write_text(json.dumps(session, indent=2))
        (talk_dir / "transcript.html").write_text(_render_archive_html(label, meta, session))

        # Reset for the next speaker: drop exactly the saved prefix (anything
        # that arrived mid-save, plus the held-back offset tail, is preserved)
        # and rebase every viewer onto what remains so the carried-over opening
        # of the next talk stays on screen rather than vanishing.
        with _lock:
            del _history[:len(snapshot)]
            carryover = list(_history)
            for q in _subscribers:
                q.put({"type": "history", "data": carryover})
        # Persist the post-save buffer now, so a crash right after saving resumes
        # only the carried-over opening of the next talk, not the talk just saved.
        _flush_live_state()

        # Summarise off the response thread (transcript + slides → alan/gemma),
        # which finalises the folder read-only when done. If summaries are off,
        # lock the folder now instead.
        if SUMMARIES_ENABLED:
            threading.Thread(target=_summary_worker, args=(talk_dir, label),
                             daemon=True).start()
        else:
            _make_readonly(talk_dir)

        print(f"Saved talk: {talk_dir.name} ({n_slides} slides, {len(texts)} text chunks)",
              flush=True)
        return {"saved": talk_dir.name, "path": str(talk_dir),
                "n_slides": n_slides, "n_text_chunks": len(texts)}


# ── Summarisation (alan / local gemma) ───────────────────────────────────────

SUMMARY_SYSTEM = (
    "You are alan, summarising a recorded conference talk for a public archive. "
    "You are given the talk's transcript (from automatic speech recognition — expect "
    "transcription errors and unreliable punctuation) and a sample of its slides, in order. "
    "Write a faithful, concise summary that helps someone who missed the talk understand what "
    "it covered and what was shown. Do not invent results or claims unsupported by the "
    "transcript and slides. "
    "Also propose exactly three questions an audience member might ask the speaker. "
    "Make them direct, specific and substantive: real questions about the methods, design "
    "choices, results, assumptions, limitations, comparisons or implications the talk actually "
    "presented — the kind a well-prepared colleague would ask to understand or probe the work. "
    "Keep them courteous and constructive rather than hostile, combative or dismissive, but do "
    "not soften them with praise or pad them with warmth. Avoid vague, feel-good or motivational "
    "questions — in particular do not ask what excited the speaker, what first drew them to the "
    "topic, or simply what comes next. Ground every question in the talk's actual content. "
    "Respond with ONLY a JSON object (no markdown, no code fences) with keys: "
    '"title" (string), "speaker" (string, empty if unknown), "abstract" (1-3 sentences), '
    '"key_points" (array of 3-7 short strings), "topics" (array of 2-6 keyword strings), '
    '"questions" (array of exactly 3 kind, considerate question strings).')


def _atomic_write_bytes(path: Path, data: bytes):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _llm_chat(messages, max_tokens=900, temperature=0.2, timeout=240) -> str:
    payload = {"model": LLM_MODEL, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    req = urllib.request.Request(
        LLM_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"]


def _sample_slides(slides_dir: Path, limit: int):
    """Return ([(filename, jpeg_b64), …], total_count), evenly sampled across the talk."""
    files = sorted(slides_dir.glob("slide_*.jpg")) if slides_dir.is_dir() else []
    total = len(files)
    if total > limit > 1:
        step = (total - 1) / (limit - 1)
        idx = sorted({round(i * step) for i in range(limit)})
        files = [files[i] for i in idx]
    out = [(f.name, base64.b64encode(f.read_bytes()).decode()) for f in files]
    return out, total


def _parse_summary_json(raw: str, label: str) -> dict:
    txt = raw.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z]*\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt).strip()
    d = None
    candidates = [txt]
    if "{" in txt and "}" in txt:
        candidates.append(txt[txt.find("{"):txt.rfind("}") + 1])
    for c in candidates:
        try:
            d = json.loads(c)
            break
        except Exception:
            continue
    if not isinstance(d, dict):
        d = {"title": label or "Untitled talk", "abstract": raw.strip()[:600]}
    return {
        "title": (str(d.get("title") or "").strip() or label or "Untitled talk"),
        "speaker": str(d.get("speaker") or "").strip(),
        "abstract": str(d.get("abstract") or "").strip(),
        "key_points": [str(x).strip() for x in (d.get("key_points") or []) if str(x).strip()][:10],
        "topics": [str(x).strip() for x in (d.get("topics") or []) if str(x).strip()][:12],
        "questions": [str(x).strip() for x in (d.get("questions") or []) if str(x).strip()][:3],
    }


def _summary_markdown(d: dict) -> str:
    lines = [f"# {d['title']}", ""]
    if d.get("speaker"):
        who = d["speaker"]
        if d.get("affiliation"):
            who += f" — {d['affiliation']}"
        lines += [f"**Speaker:** {who}", ""]
    if d.get("abstract"):
        lines += [d["abstract"], ""]
    if d.get("official_abstract"):
        lines += ["## Abstract (conference schedule)", "", d["official_abstract"], ""]
    if d.get("key_points"):
        lines += ["## Key points", ""] + [f"- {p}" for p in d["key_points"]] + [""]
    if d.get("topics"):
        lines += ["**Topics:** " + ", ".join(d["topics"]), ""]
    lines += [f"_Summary by {d.get('model', '')} · {d.get('generated_at', '')}._", ""]
    return "\n".join(lines)


def _load_official_abstract(talk_dir: Path) -> dict | None:
    """Look up the scraped conference-schedule abstract for this talk, if any.

    Source of truth is the recut-surviving sidecar SAVE_DIR/official_abstracts.json,
    a {folder_name: {speaker, affiliation, title, time, abstract}} map."""
    f = SAVE_DIR / "official_abstracts.json"
    if not f.exists():
        return None
    try:
        entry = json.loads(f.read_text()).get(talk_dir.name)
    except Exception:
        return None
    return entry if isinstance(entry, dict) and entry.get("abstract") else None


def _generate_summary(talk_dir: Path, label: str) -> dict:
    tf = talk_dir / "transcript.txt"
    transcript = tf.read_text().strip() if tf.exists() else ""
    slides, total = _sample_slides(talk_dir / "slides", SUMMARY_MAX_SLIDES)
    official = _load_official_abstract(talk_dir)

    header = f"Operator label for this talk: {label}\n\n" if label else ""
    if official:
        who = official.get("speaker", "")
        if official.get("affiliation"):
            who = f"{who} ({official['affiliation']})" if who else official["affiliation"]
        header += (
            "Official abstract for this talk, from the conference schedule. It is "
            "authoritative for the title, speaker name and technical terms (use it to "
            "correct ASR garbling), and for framing — but summarise what the talk "
            "ACTUALLY covered from the transcript and slides; do not copy it verbatim "
            "or describe content the talk did not deliver.\n"
            f"Title: {official.get('title', '')}\n"
            f"Speaker: {who}\n"
            f"{official['abstract']}\n\n")
    header += (f"Slides: {total} total, {len(slides)} sampled below in order.\n\n"
               if total else "No slides were captured.\n\n")
    user = [{"type": "text",
             "text": f"{header}=== TRANSCRIPT ===\n{transcript[:200000] or '(empty)'}\n"
                     f"=== END TRANSCRIPT ==="}]
    for name, b64 in slides:
        user.append({"type": "text", "text": f"Slide {name}:"})
        user.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}})

    raw = _llm_chat([{"role": "system", "content": SUMMARY_SYSTEM},
                     {"role": "user", "content": user}])
    data = _parse_summary_json(raw, label)
    if official:
        # Prefer the authoritative title/speaker over the ASR-derived guesses.
        if official.get("title"):
            data["title"] = official["title"]
        if official.get("speaker"):
            data["speaker"] = official["speaker"]
        if official.get("affiliation"):
            data["affiliation"] = official["affiliation"]
        data["official_abstract"] = official["abstract"]
    data["generated_at"] = datetime.now().isoformat(timespec="seconds")
    data["model"] = LLM_MODEL
    data["n_slides"] = total
    return data


def _write_summary_outputs(talk_dir: Path, data: dict):
    """Write summary.json, summary.md and questions.json from a generated dict.
    Shared by the live worker and the backfill script."""
    questions = data.pop("questions", [])
    _atomic_write_bytes(talk_dir / "summary.json", json.dumps(data, indent=2).encode())
    (talk_dir / "summary.md").write_text(_summary_markdown(data))
    qobj = {"questions": [{"id": f"q{i + 1}", "text": t} for i, t in enumerate(questions[:3])],
            "generated_at": data.get("generated_at", ""), "model": data.get("model", "")}
    _atomic_write_bytes(talk_dir / "questions.json", json.dumps(qobj, indent=2).encode())


def _summary_worker(talk_dir: Path, label: str):
    """Background: generate a summary + questions, write them, then lock the folder."""
    with _summary_lock:
        try:
            data = _generate_summary(talk_dir, label)
            _write_summary_outputs(talk_dir, data)
            print(f"Summary ready: {talk_dir.name} — {data['title']!r}", flush=True)
        except Exception as e:
            print(f"[summary warn] {talk_dir.name}: {e}", flush=True)
            try:
                _atomic_write_bytes(talk_dir / "summary.json", json.dumps(
                    {"status": "failed", "error": str(e),
                     "title": label or talk_dir.name,
                     "generated_at": datetime.now().isoformat(timespec="seconds")},
                    indent=2).encode())
            except Exception:
                pass
        finally:
            _make_readonly(talk_dir)


# ── Saved-talk archive: static serving + summaries page ──────────────────────

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8", ".txt": "text/plain; charset=utf-8",
    ".json": "application/json", ".md": "text/plain; charset=utf-8",
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
}

# Saved files that contain the full transcript verbatim. These are served only
# with the admin token (?token=…); the public gets summaries + slides only.
PRIVATE_FILES = {"transcript.txt", "transcript.html", "session.json"}


def _safe_talk_path(rel: str):
    """Resolve rel under SAVE_DIR, refusing path traversal. Returns an existing
    file Path, or None."""
    base = SAVE_DIR.resolve()
    try:
        target = (base / rel).resolve()
    except Exception:
        return None
    if target != base and base not in target.parents:
        return None
    # Never serve hidden files (e.g. the .live.json autosave buffer) or anything
    # under a hidden directory (.backups/) via the public archive route.
    if any(part.startswith(".") for part in target.relative_to(base).parts):
        return None
    return target if target.is_file() else None


# ── Audience voting on speaker questions ─────────────────────────────────────

def _load_votes(folder: str) -> dict:
    p = VOTES_DIR / f"{folder}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _save_votes(folder: str, votes: dict):
    _atomic_write_bytes(VOTES_DIR / f"{folder}.json", json.dumps(votes).encode())


def _apply_vote(folder: str, qid: str, frm: str, to: str):
    """Move one vote frm→to (each of none/up/down) for a question. Returns the
    new {up, down} counts, or None if the request is invalid."""
    if frm not in ("none", "up", "down") or to not in ("none", "up", "down"):
        return None
    if not folder or "/" in folder or "\\" in folder or ".." in folder:
        return None
    tdir = SAVE_DIR / folder
    qfile = tdir / "questions.json"
    if not tdir.is_dir() or not qfile.exists():
        return None
    try:
        valid = {q.get("id") for q in json.loads(qfile.read_text()).get("questions", [])}
    except Exception:
        return None
    if qid not in valid:
        return None
    with _votes_lock:
        votes = _load_votes(folder)
        cur = votes.get(qid, {"up": 0, "down": 0})
        cur = {"up": int(cur.get("up", 0)), "down": int(cur.get("down", 0))}
        if frm in ("up", "down"):
            cur[frm] = max(0, cur[frm] - 1)
        if to in ("up", "down"):
            cur[to] += 1
        votes[qid] = cur
        _save_votes(folder, votes)
    return cur


# ── Related-talk similarity ──────────────────────────────────────────────────

_STOPWORDS = {"the", "a", "an", "of", "and", "for", "with", "to", "in", "on", "at", "by",
              "from", "into", "via", "using", "new", "towards", "toward", "talk", "about",
              "is", "are", "as", "an", "study", "analysis", "approach"}


def _words(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower())
            if len(w) > 2 and w not in _STOPWORDS}


def _related_talks(t: dict, talks: list, k: int = 3):
    """Up to k earlier talks most similar to t, by topic + title-word overlap."""
    cur_topics, cur_title = set(t["topics"]), _words(t["title"])
    scored = []
    for o in talks:
        if o["folder"] == t["folder"] or o["saved_at"] >= t["saved_at"]:
            continue  # only earlier ("previous") talks
        s = 0.0
        ot = set(o["topics"])
        if cur_topics and ot:
            s += len(cur_topics & ot) / len(cur_topics | ot)
        ow = _words(o["title"])
        if cur_title and ow:
            s += 0.3 * len(cur_title & ow) / len(cur_title | ow)
        if s > 0:
            scored.append((s, o["folder"], o["title"]))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [(f, ti) for _, f, ti in scored[:k]]


def _render_related_html(related) -> str:
    if not related:
        return ""
    esc = html.escape
    links = " · ".join(f'<a href="#{esc(f)}">{esc(ti)}</a>' for f, ti in related)
    return f'<div class="related">Related talks: {links}</div>'


def _render_questions_html(folder: str, questions: list, votes: dict) -> str:
    if not questions:
        return ""
    esc = html.escape
    rows = []
    for idx, q in enumerate(questions):
        qid = q.get("id") or f"q{idx + 1}"
        v = votes.get(qid, {})
        up, down = int(v.get("up", 0)), int(v.get("down", 0))
        rows.append((up - down, idx, qid, q.get("text", ""), up, down))
    rows.sort(key=lambda r: (-r[0], r[1]))  # most upvotes / least downvotes first
    items = []
    for score, idx, qid, text, up, down in rows:
        items.append(
            f'<li class="question" data-qid="{esc(qid)}" data-idx="{idx}" data-score="{score}">'
            f'<span class="qtext">{esc(text)}</span>'
            f'<span class="qvote">'
            f'<button class="vb up" onclick="vote(this,\'up\')" aria-label="thumbs up">'
            f'&#9650; <em>{up}</em></button>'
            f'<button class="vb down" onclick="vote(this,\'down\')" aria-label="thumbs down">'
            f'&#9660; <em>{down}</em></button>'
            f'</span></li>')
    return (f'<details class="qbox"><summary>Questions for the speaker ({len(questions)})</summary>'
            f'<ol class="questions" data-folder="{esc(folder)}">{"".join(items)}</ol></details>')


SUMMARIES_JS = """<script>
function applyActive(q, v) {
  q.querySelector('.vb.up').classList.toggle('on', v === 'up');
  q.querySelector('.vb.down').classList.toggle('on', v === 'down');
}
function resort(list) {
  Array.from(list.children)
    .sort((a, b) => (b.dataset.score - a.dataset.score) || (a.dataset.idx - b.dataset.idx))
    .forEach(i => list.appendChild(i));
}
function vote(btn, dir) {
  const q = btn.closest('.question'), list = btn.closest('.questions');
  const folder = list.dataset.folder, qid = q.dataset.qid;
  const key = 'vote:' + folder + ':' + qid;
  const cur = localStorage.getItem(key) || 'none';
  const to = (cur === dir) ? 'none' : dir;
  fetch('/vote', {method: 'POST', headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify({folder: folder, qid: qid, from: cur, to: to})})
    .then(r => r.json()).then(res => {
      if (!res || res.error) return;
      localStorage.setItem(key, to);
      q.querySelector('.vb.up em').textContent = res.up;
      q.querySelector('.vb.down em').textContent = res.down;
      q.dataset.score = res.up - res.down;
      applyActive(q, to);
      resort(list);
    }).catch(() => {});
}
document.querySelectorAll('.questions').forEach(function (list) {
  const folder = list.dataset.folder;
  list.querySelectorAll('.question').forEach(function (q) {
    applyActive(q, localStorage.getItem('vote:' + folder + ':' + q.dataset.qid) || 'none');
  });
});
</script>"""


# ── Per-day overviews ────────────────────────────────────────────────────────

DAY_SYSTEM = (
    "You are alan, writing a short editorial overview of a single day of a "
    "research conference (SBI4GALEV — simulation-based inference for galaxy "
    "evolution) for a public archive. You are given that day's talks in order, "
    "each with its title, speaker, abstract and key points (themselves AI "
    "summaries of automatic transcripts, so expect some noise). Write a faithful, "
    "concise overview of the day: what ground it covered, how the talks related, "
    "and the threads running through them. Do not invent talks, results or claims "
    "not present in the material. "
    "Respond with ONLY a JSON object (no markdown, no code fences) with keys: "
    '"overview" (a 3-5 sentence paragraph) and "themes" (array of 3-6 short '
    "cross-cutting theme strings).")


def _talk_day(t: dict) -> str:
    """Calendar-date key 'YYYY-MM-DD' for a talk, from saved_at or folder name."""
    sa = t.get("saved_at") or ""
    if len(sa) >= 10 and sa[4] == "-":
        return sa[:10]
    return t["folder"][:10]


def _group_by_day(talks: list):
    """Group talks by date → ordered list of (date, day_label, [talks]); both the
    days and the talks within each day are in chronological order."""
    days = {}
    for t in talks:
        days.setdefault(_talk_day(t), []).append(t)
    out = []
    for i, date in enumerate(sorted(days), start=1):
        day_talks = sorted(days[date], key=lambda x: x["saved_at"])
        out.append((date, f"Day {i}", day_talks))
    return out


def _format_day_date(date: str) -> str:
    try:
        return datetime.strptime(date, "%Y-%m-%d").strftime("%A %-d %B %Y")
    except ValueError:
        return date


def _day_signature(day_talks: list) -> str:
    """A fingerprint of the day's talks + their summary versions, so the overview
    is regenerated whenever a talk is added, removed or re-summarised."""
    parts = []
    for t in sorted(day_talks, key=lambda x: x["folder"]):
        gen = (t["summary"] or {}).get("generated_at", "") if t["summary"] else ""
        parts.append(f"{t['folder']}@{gen}")
    return "|".join(parts)


def _load_day_summary(date: str):
    p = DAYS_DIR / f"{date}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _generate_day_summary(date: str, day_label: str, day_talks: list) -> dict:
    blocks = []
    for t in day_talks:
        s = t["summary"] or {}
        title = s.get("title") or t["title"]
        line = f"### {title}"
        if s.get("speaker"):
            line += f" — {s['speaker']}"
        if s.get("abstract"):
            line += f"\n{s['abstract']}"
        for p in s.get("key_points", []):
            line += f"\n- {p}"
        blocks.append(line)
    user = (f"{day_label} ({_format_day_date(date)}) — {len(day_talks)} talk(s), "
            f"in order:\n\n" + "\n\n".join(blocks))
    raw = _llm_chat([{"role": "system", "content": DAY_SYSTEM},
                     {"role": "user", "content": user}], max_tokens=700)
    d = _parse_summary_json(raw, day_label)  # tolerant JSON extraction
    # _parse_summary_json keys off a talk schema; pull what we need from raw too.
    try:
        txt = raw.strip()
        if txt.startswith("```"):
            txt = re.sub(r"^```[a-zA-Z]*\s*", "", txt)
            txt = re.sub(r"\s*```$", "", txt).strip()
        obj = json.loads(txt[txt.find("{"):txt.rfind("}") + 1]) if "{" in txt else {}
    except Exception:
        obj = {}
    overview = str(obj.get("overview") or d.get("abstract") or "").strip()
    themes = [str(x).strip() for x in (obj.get("themes") or d.get("topics") or []) if str(x).strip()][:6]
    return {
        "date": date, "day_label": day_label, "n_talks": len(day_talks),
        "overview": overview, "themes": themes,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": LLM_MODEL,
    }


def _day_worker(date: str, day_label: str, day_talks: list, sig: str):
    try:
        data = _generate_day_summary(date, day_label, day_talks)
        data["signature"] = sig
        _atomic_write_bytes(DAYS_DIR / f"{date}.json", json.dumps(data, indent=2).encode())
        print(f"Day overview ready: {date} ({day_label})", flush=True)
    except Exception as e:
        print(f"[day overview warn] {date}: {e}", flush=True)
    finally:
        with _day_inflight_lock:
            _day_inflight.discard(date)


def _ensure_day_summary(date: str, day_label: str, day_talks: list):
    """Return the cached day overview if it is fresh for the current set of talks.
    Otherwise kick off a one-off background regeneration and return whatever we
    have (a stale overview, or None) so the page renders immediately."""
    cached = _load_day_summary(date)
    sig = _day_signature(day_talks)
    if cached and cached.get("signature") == sig and cached.get("overview"):
        return cached
    if SUMMARIES_ENABLED:
        with _day_inflight_lock:
            if date not in _day_inflight:
                _day_inflight.add(date)
                threading.Thread(target=_day_worker,
                                 args=(date, day_label, list(day_talks), sig),
                                 daemon=True).start()
    return cached


# ── Conference-wide key topics ───────────────────────────────────────────────
# A synthesis across every saved talk: the per-talk topic keywords are clustered
# (by the LLM, or by exact keyword when it is unavailable) into the conference's
# main threads. Each topic is grounded back to the real talks that tagged it, and
# topics that co-occur in a talk become linked nodes in the /topics graph.

TOPICS_SYSTEM = (
    "You are alan, distilling the key topics of a research conference (SBI4GALEV — "
    "simulation-based inference for galaxy evolution) for a public archive. You are "
    "given the list of talks so far and a vocabulary of topic keywords drawn from "
    "their summaries (themselves AI summaries of automatic transcripts, so expect "
    "noise, duplicates and spelling variants). Cluster the keywords into the "
    "conference's main topics, merging synonyms and variants (e.g. 'amortised' and "
    "'amortized inference', or 'SBI' and 'simulation-based inference') into a single "
    "topic. For each topic give a short, specific display name and a one-sentence "
    "description of what it covers, grounded only in the supplied material — do not "
    "invent topics or claims. Use only keywords from the supplied vocabulary, and "
    "assign each keyword to at most one topic; it is fine to leave marginal keywords "
    "out. Order topics from most to least central to the conference. "
    "Respond with ONLY a JSON object (no markdown, no code fences) with key "
    '"topics": an array of objects, each with keys "name" (string), "description" '
    '(one sentence string) and "keywords" (array of strings drawn only from the '
    "supplied vocabulary).")


def _topics_signature(talks: list) -> str:
    """Fingerprint of every talk + its summary version (same scheme as the per-day
    overviews) so the synthesis is rebuilt whenever a talk is added, removed or
    re-summarised."""
    return _day_signature(talks)


def _topic_vocab(talks: list):
    """Distinct topic keywords across all talks → (sorted vocab, {kw: set(folders)})."""
    kw_to_folders = {}
    for t in talks:
        for kw in t["topics"]:          # already lower-cased in _load_talks
            kw = kw.strip()
            if kw:
                kw_to_folders.setdefault(kw, set()).add(t["folder"])
    return sorted(kw_to_folders), kw_to_folders


def _parse_topics_json(raw: str) -> list:
    """Extract the clusters array from the model reply, tolerant of code fences and
    chatter. Returns a list (possibly empty)."""
    txt = raw.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z]*\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt).strip()
    candidates = [txt]
    if "{" in txt and "}" in txt:
        candidates.append(txt[txt.find("{"):txt.rfind("}") + 1])
    if "[" in txt and "]" in txt:
        candidates.append(txt[txt.find("["):txt.rfind("]") + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
        except Exception:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("topics"), list):
            return obj["topics"]
        if isinstance(obj, list):
            return obj
    return []


def _keyword_fallback_clusters(vocab: list, kw_to_folders: dict) -> list:
    """LLM-free fallback: each distinct keyword becomes a topic (ranked later by how
    many talks use it). Keeps /topics working when the model is unavailable."""
    return [{"name": kw, "description": "", "keywords": [kw]}
            for kw in sorted(vocab, key=lambda k: (-len(kw_to_folders[k]), k))]


def _resolve_topic_clusters(clusters: list, kw_to_folders: dict, talks: list):
    """Turn [{name, description, keywords}] into ranked topic dicts with grounded
    talk references and co-occurrence edges. A topic's talks are the union of the
    talks tagged with any of its keywords, so every reference is real. Drops topics
    that match no talk, de-duplicates by name, and caps at TOPICS_MAX."""
    by_folder = {t["folder"]: t for t in talks}
    topics, seen = [], set()
    for c in clusters:
        name = str((c or {}).get("name") or "").strip()
        if not name or name.lower() in seen:
            continue
        kws, kwseen = [], set()
        for k in (c.get("keywords") or []):
            k = str(k).strip().lower()
            if k in kw_to_folders and k not in kwseen:
                kws.append(k)
                kwseen.add(k)
        folders = set()
        for k in kws:
            folders |= kw_to_folders[k]
        if not folders:
            continue
        seen.add(name.lower())
        refs = sorted(
            ({"folder": f, "title": by_folder[f]["title"], "date": _talk_day(by_folder[f])}
             for f in folders),
            key=lambda r: r["title"].lower())
        topics.append({"name": name, "description": str(c.get("description") or "").strip(),
                       "keywords": kws, "talks": refs, "count": len(refs), "_folders": folders})
    topics.sort(key=lambda x: (-x["count"], x["name"].lower()))
    topics = topics[:TOPICS_MAX]
    edges = []
    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            w = len(topics[i]["_folders"] & topics[j]["_folders"])
            if w:
                edges.append({"source": i, "target": j, "weight": w})
    for tp in topics:
        del tp["_folders"]
    return topics, edges


def _generate_topics_summary(talks: list, use_llm: bool) -> dict:
    t0 = time.time()
    vocab, kw_to_folders = _topic_vocab(talks)
    clusters, source = [], "keywords"
    if use_llm and vocab:
        lines = "\n".join(f"{i + 1}. {t['title']}" for i, t in enumerate(talks))
        user = (f"{len(talks)} talks so far:\n{lines}\n\n"
                f"Topic keyword vocabulary ({len(vocab)} terms):\n{', '.join(vocab)}\n\n"
                f"Cluster these into at most {TOPICS_MAX} key topics of the conference, "
                "as instructed.")
        try:
            raw = _llm_chat([{"role": "system", "content": TOPICS_SYSTEM},
                             {"role": "user", "content": user}], max_tokens=1300)
            clusters = _parse_topics_json(raw)
            if clusters:
                source = "llm"
        except Exception as e:
            print(f"[topics warn] LLM synthesis failed: {e}", flush=True)
    if not clusters:
        clusters = _keyword_fallback_clusters(vocab, kw_to_folders)
    topics, edges = _resolve_topic_clusters(clusters, kw_to_folders, talks)
    return {
        "topics": topics, "edges": edges,
        "n_talks": len(talks), "n_keywords": len(vocab), "source": source,
        "duration_seconds": round(time.time() - t0, 1),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": LLM_MODEL if source == "llm" else "keyword aggregation",
    }


def _load_topics_summary():
    if TOPICS_PATH.exists():
        try:
            return json.loads(TOPICS_PATH.read_text())
        except Exception:
            return None
    return None


def _topics_worker(talks: list, sig: str, use_llm: bool):
    global _topics_inflight
    try:
        data = _generate_topics_summary(talks, use_llm)
        data["signature"] = sig
        _atomic_write_bytes(TOPICS_PATH, json.dumps(data, indent=2).encode())
        print(f"Key topics ready: {len(data['topics'])} topics, {len(data['edges'])} "
              f"links ({data['source']}, {data['duration_seconds']}s)", flush=True)
    except Exception as e:
        print(f"[topics warn] {e}", flush=True)
    finally:
        with _topics_lock:
            _topics_inflight = False


def _ensure_topics_summary(talks: list):
    """Return the cached synthesis if fresh for the current talks. Otherwise kick
    off a single background (re)build and return what we have (a stale synthesis,
    or None) so the page renders its timer immediately."""
    global _topics_inflight, _topics_started_at
    cached = _load_topics_summary()
    sig = _topics_signature(talks)
    if cached and cached.get("signature") == sig and "topics" in cached:
        return cached
    with _topics_lock:
        if not _topics_inflight:
            _topics_inflight = True
            _topics_started_at = time.time()
            threading.Thread(target=_topics_worker,
                             args=(list(talks), sig, SUMMARIES_ENABLED),
                             daemon=True).start()
    return cached


def _topics_status(talks: list) -> dict:
    """Lightweight JSON for the /topics poller: whether the synthesis is ready or
    still running (with elapsed seconds, to drive the on-page timer)."""
    cached = _load_topics_summary()
    sig = _topics_signature(talks)
    fresh = bool(cached and cached.get("signature") == sig and "topics" in cached)
    with _topics_lock:
        inflight, started = _topics_inflight, _topics_started_at
    if fresh and not inflight:
        return {"status": "ready"}
    if inflight:
        return {"status": "generating", "elapsed": round(max(0.0, time.time() - started), 1)}
    return {"status": "generating", "elapsed": 0.0}


# ── Saved-talk archive: page rendering ───────────────────────────────────────

def _load_talks() -> list:
    talks = []
    for d in SAVE_DIR.iterdir():
        if not d.is_dir() or not (d / "metadata.json").exists():
            continue
        try:
            meta = json.loads((d / "metadata.json").read_text())
        except Exception:
            meta = {}
        summary = None
        if (d / "summary.json").exists():
            try:
                summary = json.loads((d / "summary.json").read_text())
            except Exception:
                summary = None
        questions = []
        if (d / "questions.json").exists():
            try:
                questions = json.loads((d / "questions.json").read_text()).get("questions", [])
            except Exception:
                questions = []
        slides = sorted((d / "slides").glob("slide_*.jpg")) if (d / "slides").is_dir() else []
        talks.append({
            "folder": d.name, "meta": meta, "summary": summary, "questions": questions,
            "votes": _load_votes(d.name), "thumb": slides[0].name if slides else None,
            "saved_at": meta.get("saved_at") or d.name,
            "title": (summary or {}).get("title") or meta.get("label") or d.name,
            "topics": [str(x).lower() for x in (summary or {}).get("topics", [])],
        })

    talks.sort(key=lambda t: t["saved_at"], reverse=True)
    return talks


SUMMARIES_CSS = """
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background:#0a0a0a; color:#ddd; font-family:sans-serif; padding:48px 20px 96px; }
    .wrap { max-width:860px; margin:0 auto; }
    header.top { display:flex; align-items:baseline; justify-content:space-between; margin-bottom:2em; }
    header.top h1 { font-size:1.3em; font-weight:600; letter-spacing:0.5px; }
    header.top .navlinks a { color:#555; font-size:0.7em; letter-spacing:1px; text-decoration:none; margin-left:14px; }
    header.top .navlinks a:hover { color:#aaa; }
    .day { padding:28px 0; border-top:1px solid #1a1a1a; }
    .day-head { display:flex; align-items:baseline; gap:14px; margin-bottom:0.2em; }
    .day-head h2 { font-size:1.15em; font-weight:700; color:#fff; letter-spacing:0.5px; }
    .day-head .date { color:#777; font-size:0.8em; }
    .day-head .count { color:#555; font-size:0.72em; letter-spacing:1px; margin-left:auto; }
    .day-overview { color:#cfcfcf; line-height:1.7; margin:0.7em 0 0.9em; }
    .day-thumbs { display:flex; gap:8px; overflow:hidden; margin:0.6em 0 0.9em; }
    .day-thumbs img { height:74px; border:1px solid #222; border-radius:4px; display:block; }
    .day-talklist { list-style:none; margin:0.4em 0 0; padding:0; }
    .day-talklist li { padding:5px 0; border-top:1px solid #141414; font-size:0.92em; }
    .day-talklist a { color:#cdd; text-decoration:none; }
    .day-talklist a:hover { color:#fff; }
    .day-talklist .who { color:#777; font-size:0.85em; }
    .day-more { display:inline-block; margin-top:0.9em; color:#6a8; font-size:0.8em;
                text-decoration:none; letter-spacing:0.5px; }
    .day-more:hover { text-decoration:underline; }
    .day-page-overview { color:#cfcfcf; line-height:1.75; margin:0.4em 0 0.6em; font-size:1.02em; }
    .card { display:flex; gap:20px; padding:24px 0; border-top:1px solid #1a1a1a; }
    .card .thumb { flex:0 0 200px; }
    .card .thumb img { width:100%; border:1px solid #222; border-radius:6px; display:block; }
    .card-body { flex:1; min-width:0; }
    .card h2 { font-size:1.05em; font-weight:600; color:#fff; margin-bottom:0.2em; }
    .when { color:#555; font-size:0.7em; letter-spacing:1px; margin-bottom:0.9em; }
    .speaker { color:#9a9; font-size:0.85em; margin-bottom:0.6em; }
    .abstract { color:#bbb; line-height:1.65; margin-bottom:0.8em; }
    .official-abstract { margin:0 0 0.8em 0; }
    .official-abstract > summary { color:#7fa7d6; cursor:pointer; font-size:0.85em;
        text-transform:uppercase; letter-spacing:0.04em; }
    .official-abstract > p { color:#9a9a9a; line-height:1.65; margin:0.5em 0 0; }
    .card ul { margin:0 0 0.8em 1.1em; color:#9a9a9a; line-height:1.6; }
    .topics { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:0.8em; }
    .topics span { font-size:0.65em; letter-spacing:1px; color:#777; border:1px solid #262626;
                    border-radius:10px; padding:2px 9px; }
    .links { font-size:0.72em; color:#555; }
    .links a { color:#6a8; text-decoration:none; }
    .links a:hover { text-decoration:underline; }
    .pending { color:#666; font-style:italic; }
    .empty { color:#555; text-align:center; padding:60px 0; }
    .related { font-size:0.75em; color:#777; margin:0.2em 0 0.9em; }
    .related a { color:#6a8; text-decoration:none; }
    .related a:hover { text-decoration:underline; }
    details.qbox { margin:0.2em 0 0.9em; }
    details.qbox > summary { cursor:pointer; color:#9a9; font-size:0.8em; letter-spacing:0.5px;
        list-style:none; user-select:none; }
    details.qbox > summary::-webkit-details-marker { display:none; }
    details.qbox > summary::before { content:'▸ '; color:#555; }
    details.qbox[open] > summary::before { content:'▾ '; }
    ol.questions { list-style:none; margin:0.7em 0 0; padding:0; }
    li.question { display:flex; gap:12px; align-items:flex-start; padding:8px 0;
        border-top:1px solid #161616; }
    .qtext { flex:1; color:#ccc; line-height:1.5; font-size:0.92em; }
    .qvote { display:flex; gap:6px; flex:0 0 auto; }
    .vb { background:#111; border:1px solid #222; color:#888; border-radius:4px;
        padding:3px 9px; font-size:0.8em; cursor:pointer; white-space:nowrap; }
    .vb:hover { border-color:#444; color:#ccc; }
    .vb.up.on { color:#4a4; border-color:#4a4; }
    .vb.down.on { color:#b55; border-color:#b55; }
    .vb em { font-style:normal; }
    .topics-meta { color:#666; font-size:0.75em; letter-spacing:0.5px; margin:0.4em 0 1.6em; }
    .topics-meta .timer { color:#9a9; font-variant-numeric:tabular-nums; }
    #topic-graph-wrap { position:relative; margin:0.4em 0 2.2em; }
    #topic-graph { width:100%; height:460px; display:block; background:#0c0c0c;
        border:1px solid #1a1a1a; border-radius:8px; cursor:grab; touch-action:none; }
    #topic-graph.dragging { cursor:grabbing; }
    .graph-hint { position:absolute; top:10px; right:14px; color:#3d3d3d; font-size:0.66em;
        letter-spacing:1px; pointer-events:none; }
    .tcard { padding:18px 0; border-top:1px solid #1a1a1a; }
    .tcard.flash { animation:tflash 1.1s ease; }
    @keyframes tflash { from { background:#181818; } to { background:transparent; } }
    .tcard h2 { font-size:1.02em; font-weight:600; color:#fff; margin-bottom:0.25em;
        display:flex; align-items:baseline; gap:10px; }
    .tcard h2 .n { color:#555; font-size:0.62em; font-weight:400; letter-spacing:1px; }
    .tcard .tdesc { color:#bbb; line-height:1.65; margin-bottom:0.5em; }
    .tcard .tkw { color:#5f5f5f; font-size:0.72em; letter-spacing:0.5px; margin-bottom:0.5em; }
    .tcard .ttalks { list-style:none; margin:0; padding:0; }
    .tcard .ttalks li { padding:3px 0; font-size:0.9em; }
    .tcard .ttalks a { color:#cdd; text-decoration:none; }
    .tcard .ttalks a:hover { color:#fff; }
    .generating { text-align:center; color:#888; padding:72px 0; }
    .generating .timer { font-size:2.3em; color:#cdd; font-variant-numeric:tabular-nums;
        display:block; margin:0.35em 0; letter-spacing:1px; }
    .generating .sub { font-size:0.78em; color:#555; letter-spacing:1px; }
    .spinner { display:inline-block; width:13px; height:13px; border:2px solid #2c2c2c;
        border-top-color:#9a9; border-radius:50%; animation:spin 0.8s linear infinite;
        vertical-align:middle; margin-right:8px; }
    @keyframes spin { to { transform:rotate(360deg); } }"""


def _page_shell(title: str, heading: str, nav_html: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{SUMMARIES_CSS}</style>
</head>
<body>
  <div class="wrap">
    <header class="top"><h1>{heading}</h1><div class="navlinks">{nav_html}</div></header>
    {body}
  </div>
  {SUMMARIES_JS}
</body>
</html>"""


def _render_talk_card(t: dict, all_talks: list) -> str:
    esc = html.escape
    folder, fq, summary, thumb = t["folder"], quote(t["folder"]), t["summary"], t["thumb"]
    title = esc(t["title"])
    when = esc(t["meta"].get("saved_at", ""))
    n_slides = t["meta"].get("n_slides", 0)
    thumb_html = (f'<a class="thumb" href="/talks/{fq}/slides/{quote(thumb)}">'
                  f'<img src="/talks/{fq}/slides/{quote(thumb)}" loading="lazy"></a>'
                  if thumb else "")
    if summary and summary.get("status") == "failed":
        inner = '<p class="pending">Summary unavailable.</p>'
    elif summary:
        speaker = esc(summary.get("speaker") or "")
        if speaker and summary.get("affiliation"):
            speaker += f" — {esc(summary['affiliation'])}"
        sp = f'<div class="speaker">{speaker}</div>' if speaker else ""
        abstract = esc(summary.get("abstract") or "")
        official = esc(summary.get("official_abstract") or "")
        official_html = (f"<details class='official-abstract'><summary>Abstract "
                         f"(conference schedule)</summary><p>{official}</p></details>"
                         if official else "")
        pts = "".join(f"<li>{esc(p)}</li>" for p in summary.get("key_points", []))
        pts_html = f"<ul>{pts}</ul>" if pts else ""
        tps = summary.get("topics", [])
        tps_html = ("<div class='topics'>" +
                    "".join(f"<span>{esc(x)}</span>" for x in tps) + "</div>") if tps else ""
        inner = f"{sp}<p class='abstract'>{abstract}</p>{official_html}{pts_html}{tps_html}"
    else:
        inner = '<p class="pending">Summary generating…</p>'
    related_html = _render_related_html(_related_talks(t, all_talks))
    questions_html = _render_questions_html(folder, t["questions"], t["votes"])
    return (f'<article class="card" id="{esc(folder)}">{thumb_html}<div class="card-body">'
            f'<h2>{title}</h2><div class="when">{when}</div>{inner}'
            f'{related_html}{questions_html}'
            f'<div class="links">{n_slides} slides</div></div></article>')


def _render_day_overview(day_summary, n_talks: int, on_index: bool) -> str:
    """The editorial paragraph + theme chips for a day. `on_index` picks the
    slightly smaller styling used in the day list vs. the full day page."""
    cls = "day-overview" if on_index else "day-page-overview"
    if not day_summary or not day_summary.get("overview"):
        return f'<p class="pending {cls}">Day overview generating…</p>'
    esc = html.escape
    out = f'<p class="{cls}">{esc(day_summary["overview"])}</p>'
    themes = day_summary.get("themes", [])
    if themes:
        out += ("<div class='topics'>" +
                "".join(f"<span>{esc(x)}</span>" for x in themes) + "</div>")
    return out


def render_day_page(date: str) -> str:
    talks = _load_talks()
    grouped = _group_by_day(talks)
    match = next(((dt, label, dtalks) for dt, label, dtalks in grouped if dt == date), None)
    nav = '<a href="/summaries">← all days</a><a href="/topics">key topics</a><a href="/">live</a>'
    if match is None:
        return _page_shell("SBI4GALEV — Day not found", "SBI4GALEV", nav,
                           '<p class="empty">No talks recorded for this day.</p>')
    _, day_label, day_talks = match
    day_summary = _ensure_day_summary(date, day_label, day_talks)
    heading = f"{day_label} <span style='color:#555;font-weight:400;font-size:0.7em'>{html.escape(_format_day_date(date))}</span>"
    overview = _render_day_overview(day_summary, len(day_talks), on_index=False)
    # Cards in talk order for the day (oldest first → as the day unfolded).
    cards = "\n".join(_render_talk_card(t, talks) for t in day_talks)
    body = f'<div class="day-page-head">{overview}</div>{cards}'
    return _page_shell(f"SBI4GALEV — {day_label}", heading, nav, body)


def render_summaries_page() -> str:
    """Index of conference days: each day shows its editorial overview, a slide
    strip and the day's talk list, linking through to the full day page."""
    talks = _load_talks()
    grouped = _group_by_day(talks)
    esc = html.escape
    nav = '<a href="/topics">key topics</a><a href="/">← live</a>'
    if not grouped:
        return _page_shell("SBI4GALEV — Talk summaries", "SBI4GALEV — Talks", nav,
                           '<p class="empty">No talks saved yet.</p>')

    blocks = []
    for date, day_label, day_talks in reversed(grouped):  # most recent day first
        day_summary = _ensure_day_summary(date, day_label, day_talks)
        overview = _render_day_overview(day_summary, len(day_talks), on_index=True)
        thumbs = "".join(
            f'<img src="/talks/{quote(t["folder"])}/slides/{quote(t["thumb"])}" loading="lazy">'
            for t in day_talks if t["thumb"])
        thumbs_html = f'<div class="day-thumbs">{thumbs}</div>' if thumbs else ""
        items = []
        for t in day_talks:
            who = (t["summary"] or {}).get("speaker") or ""
            who_html = f' <span class="who">— {esc(who)}</span>' if who else ""
            items.append(
                f'<li><a href="/summaries/{quote(date)}#{esc(t["folder"])}">{esc(t["title"])}</a>'
                f'{who_html}</li>')
        n = len(day_talks)
        blocks.append(
            f'<section class="day">'
            f'<div class="day-head"><h2>{day_label}</h2>'
            f'<span class="date">{esc(_format_day_date(date))}</span>'
            f'<span class="count">{n} talk{"s" if n != 1 else ""}</span></div>'
            f'{overview}{thumbs_html}'
            f'<ul class="day-talklist">{"".join(items)}</ul>'
            f'<a class="day-more" href="/summaries/{quote(date)}">View {day_label} in full →</a>'
            f'</section>')

    return _page_shell("SBI4GALEV — Talk summaries", "SBI4GALEV — Talks", nav,
                       "\n".join(blocks))


# ── Key-topics page (graph + cards + generation timer) ───────────────────────

TOPIC_GRAPH_JS = """<script>
(function () {
  var canvas = document.getElementById('topic-graph');
  var holder = document.getElementById('graph-data');
  if (!canvas || !holder) return;
  var data; try { data = JSON.parse(holder.textContent); } catch (e) { return; }
  var nodes = data.nodes || [], links = data.links || [], N = nodes.length;
  if (N < 2) return;

  var maxC = 1;
  nodes.forEach(function (n) { maxC = Math.max(maxC, n.count || 1); });
  nodes.forEach(function (n, i) {
    n.r = 9 + 17 * Math.sqrt((n.count || 1) / maxC);
    n.hue = Math.round(360 * i / N);
  });

  var ctx = canvas.getContext('2d');
  var W = 0, H = 0, dpr = Math.min(window.devicePixelRatio || 1, 2);
  function resize() {
    W = canvas.clientWidth || 600; H = canvas.clientHeight || 460;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  nodes.forEach(function (n, i) {
    var a = 2 * Math.PI * i / N, R = Math.min(W, H) * 0.32;
    n.x = W / 2 + Math.cos(a) * R; n.y = H / 2 + Math.sin(a) * R; n.vx = 0; n.vy = 0;
  });

  var alpha = 1, hover = -1, drag = -1, dragMoved = false, downX = 0, downY = 0, rafId = null;

  function tick() {
    var i, j;
    for (i = 0; i < N; i++) {
      for (j = i + 1; j < N; j++) {
        var dx = nodes[j].x - nodes[i].x, dy = nodes[j].y - nodes[i].y;
        var d2 = dx * dx + dy * dy || 0.01, d = Math.sqrt(d2), f = 2800 / d2;
        var ux = dx / d, uy = dy / d;
        nodes[i].vx -= ux * f; nodes[i].vy -= uy * f;
        nodes[j].vx += ux * f; nodes[j].vy += uy * f;
      }
    }
    links.forEach(function (l) {
      var a = nodes[l.source], b = nodes[l.target];
      var dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) || 0.01;
      var k = 0.02 * (1 + Math.min(l.weight || 1, 4) * 0.4), f = (d - 96) * k;
      var ux = dx / d, uy = dy / d;
      a.vx += ux * f; a.vy += uy * f; b.vx -= ux * f; b.vy -= uy * f;
    });
    for (i = 0; i < N; i++) {
      var n = nodes[i];
      n.vx += (W / 2 - n.x) * 0.002; n.vy += (H / 2 - n.y) * 0.002;
      if (i === drag) { n.vx = 0; n.vy = 0; continue; }
      n.vx *= 0.86; n.vy *= 0.86;
      n.x += n.vx * alpha; n.y += n.vy * alpha;
      n.x = Math.max(n.r + 2, Math.min(W - n.r - 2, n.x));
      n.y = Math.max(n.r + 2, Math.min(H - n.r - 2, n.y));
    }
    if (alpha > 0.02) alpha *= 0.99;
  }

  function neighbours(i) {
    var s = {};
    links.forEach(function (l) {
      if (l.source === i) s[l.target] = 1;
      if (l.target === i) s[l.source] = 1;
    });
    return s;
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    var nb = hover >= 0 ? neighbours(hover) : null;
    links.forEach(function (l) {
      var a = nodes[l.source], b = nodes[l.target];
      var on = hover < 0 || l.source === hover || l.target === hover;
      ctx.strokeStyle = on ? 'rgba(150,170,150,' + Math.min(0.16 + (l.weight || 1) * 0.16, 0.7) + ')'
                           : 'rgba(130,130,130,0.05)';
      ctx.lineWidth = on ? Math.min(1 + (l.weight || 1), 4) : 1;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    });
    nodes.forEach(function (n, i) {
      var dim = hover >= 0 && i !== hover && !(nb && nb[i]);
      ctx.globalAlpha = dim ? 0.25 : 1;
      ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, 2 * Math.PI);
      ctx.fillStyle = 'hsl(' + n.hue + ',45%,' + (i === hover ? 62 : 48) + '%)';
      ctx.fill();
      ctx.lineWidth = 1.5; ctx.strokeStyle = 'rgba(0,0,0,0.45)'; ctx.stroke();
      ctx.fillStyle = i === hover ? '#fff' : '#cfcfcf';
      ctx.font = (i === hover ? '600 ' : '') + Math.max(11, Math.min(15, 10 + n.r * 0.13)) + 'px sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText(n.name, n.x, n.y + n.r + 3);
    });
    ctx.globalAlpha = 1;
  }

  function loop() {
    tick(); draw();
    if (alpha > 0.03 || drag >= 0 || hover >= 0) rafId = requestAnimationFrame(loop);
    else rafId = null;
  }
  function kick(a) { if (a) alpha = Math.max(alpha, a); if (!rafId) rafId = requestAnimationFrame(loop); }
  kick(1);

  function pos(e) { var r = canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
  function hit(p) {
    for (var i = N - 1; i >= 0; i--) {
      var dx = p.x - nodes[i].x, dy = p.y - nodes[i].y, rr = nodes[i].r + 4;
      if (dx * dx + dy * dy <= rr * rr) return i;
    }
    return -1;
  }
  canvas.addEventListener('pointermove', function (e) {
    var p = pos(e);
    if (drag >= 0) {
      nodes[drag].x = p.x; nodes[drag].y = p.y; nodes[drag].vx = 0; nodes[drag].vy = 0;
      if (Math.abs(p.x - downX) + Math.abs(p.y - downY) > 4) dragMoved = true;
      kick(0.3);
    } else {
      var h = hit(p);
      if (h !== hover) { hover = h; kick(0); }
      canvas.style.cursor = h >= 0 ? 'pointer' : 'grab';
    }
  });
  canvas.addEventListener('pointerdown', function (e) {
    var p = pos(e), h = hit(p);
    downX = p.x; downY = p.y; dragMoved = false;
    if (h >= 0) {
      drag = h; hover = h; canvas.classList.add('dragging');
      try { canvas.setPointerCapture(e.pointerId); } catch (x) {}
      kick(0.3);
    }
  });
  function release() {
    if (drag >= 0 && !dragMoved) {
      var el = document.getElementById('topic-' + drag);
      if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash'); }
    }
    drag = -1; canvas.classList.remove('dragging'); kick(0);
  }
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', function () { drag = -1; canvas.classList.remove('dragging'); kick(0); });
  canvas.addEventListener('pointerleave', function () { if (drag < 0 && hover >= 0) { hover = -1; kick(0); } });
  window.addEventListener('resize', function () {
    var ox = W, oy = H; resize();
    if (ox && oy) nodes.forEach(function (n) { n.x *= W / ox; n.y *= H / oy; });
    kick(0.4);
  });
})();
</script>"""

_TOPICS_GENERATING = """
    <div class="generating">
      <div><span class="spinner"></span>Synthesising the conference's key topics…</div>
      <span class="timer" id="gen-timer">__ELAPSED__s</span>
      <div class="sub">alan is reading every talk summary · this page refreshes automatically</div>
    </div>
    <script>
    (function () {
      var el = document.getElementById('gen-timer'), t = __ELAPSED__;
      var iv = setInterval(function () { t += 0.1; el.textContent = t.toFixed(1) + 's'; }, 100);
      function poll() {
        fetch('/topics.json', { cache: 'no-store' }).then(function (r) { return r.json(); })
          .then(function (d) {
            if (d.status === 'ready') { clearInterval(iv); location.reload(); return; }
            if (typeof d.elapsed === 'number') t = d.elapsed;
            setTimeout(poll, 1500);
          }).catch(function () { setTimeout(poll, 2500); });
      }
      setTimeout(poll, 1500);
    })();
    </script>"""

_TOPICS_REFRESH_POLL = """
<script>
(function () {
  function poll() {
    fetch('/topics.json', { cache: 'no-store' }).then(function (r) { return r.json(); })
      .then(function (d) { if (d.status === 'ready') location.reload(); else setTimeout(poll, 2500); })
      .catch(function () { setTimeout(poll, 4000); });
  }
  setTimeout(poll, 2500);
})();
</script>"""


def _render_topic_graph(data) -> str:
    topics = data.get("topics", [])
    if len(topics) < 2:
        return ""
    nodes = [{"name": tp.get("name", ""), "count": tp.get("count", 0)} for tp in topics]
    payload = json.dumps({"nodes": nodes, "links": data.get("edges", [])}).replace("</", "<\\/")
    return ('<div id="topic-graph-wrap"><canvas id="topic-graph"></canvas>'
            '<div class="graph-hint">drag to explore · click a topic</div></div>'
            f'<script id="graph-data" type="application/json">{payload}</script>'
            + TOPIC_GRAPH_JS)


def _render_topic_cards(data) -> str:
    esc = html.escape
    out = []
    for i, tp in enumerate(data.get("topics", [])):
        items = []
        for r in tp.get("talks", []):
            date, anchor = r.get("date") or "", esc(r["folder"])
            href = (f'/summaries/{quote(date)}#{anchor}'
                    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) else f'/summaries#{anchor}')
            items.append(f'<li><a href="{href}">{esc(r["title"])}</a></li>')
        desc = esc(tp.get("description") or "")
        desc_html = f'<p class="tdesc">{desc}</p>' if desc else ""
        kws = ", ".join(tp.get("keywords", []))
        kw_html = f'<div class="tkw">{esc(kws)}</div>' if kws else ""
        n = tp.get("count", len(items))
        out.append(
            f'<article class="tcard" id="topic-{i}">'
            f'<h2>{esc(tp.get("name", ""))} <span class="n">{n} talk{"" if n == 1 else "s"}</span></h2>'
            f'{desc_html}{kw_html}<ul class="ttalks">{"".join(items)}</ul></article>')
    return "\n".join(out)


def render_topics_page() -> str:
    talks = _load_talks()
    nav = '<a href="/summaries">summaries</a><a href="/">live</a>'
    heading, title = "Key topics", "SBI4GALEV — Key topics"
    if not talks:
        return _page_shell(title, heading, nav, '<p class="empty">No talks saved yet.</p>')

    data = _ensure_topics_summary(talks)
    sig = _topics_signature(talks)
    fresh = bool(data and data.get("signature") == sig)
    has_content = bool(data and data.get("topics"))

    if not has_content and not fresh:
        with _topics_lock:
            elapsed = max(0.0, time.time() - _topics_started_at) if _topics_inflight else 0.0
        body = _TOPICS_GENERATING.replace("__ELAPSED__", f"{elapsed:.1f}")
        return _page_shell(title, heading, nav, body)

    esc = html.escape
    topics = data.get("topics", [])
    src_label = "alan" if data.get("source") == "llm" else "keyword aggregation"
    parts = [f'{len(topics)} topic{"" if len(topics) == 1 else "s"} across '
             f'{data.get("n_talks", len(talks))} talks', f'by {esc(src_label)}']
    dur = data.get("duration_seconds")
    if isinstance(dur, (int, float)):
        parts.append(f'in <span class="timer">{dur:.1f}s</span>')
    if data.get("generated_at"):
        parts.append(esc(data["generated_at"]))
    meta_txt = " · ".join(parts)
    if not fresh:
        meta_txt += ' · <span class="timer">updating…</span>'

    intro = ('<p class="day-page-overview">The threads running through the conference so far — '
             'each topic links to the talks that raised it; drag the graph to explore.</p>')
    meta = f'<div class="topics-meta">{meta_txt}</div>'
    inner = (_render_topic_graph(data) + _render_topic_cards(data)
             if topics else '<p class="empty">No topics identified yet.</p>')
    poll = "" if fresh else _TOPICS_REFRESH_POLL
    return _page_shell(title, heading, nav, f'{intro}{meta}{inner}{poll}')


def _push(event_type, data):
    item = {"type": event_type, "data": data, "ts": datetime.now().strftime("%H:%M:%S")}
    with _lock:
        _history.append(item)
        for q in _subscribers:
            q.put(item)
    _live_dirty.set()


def _to_wav16k(raw):
    src = tempfile.NamedTemporaryFile(suffix=".in", delete=False)
    src.write(raw)
    src.close()
    dst = src.name + ".wav"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", src.name,
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "wav", dst],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.unlink(src.name)
    return dst


def transcribe(raw: bytes) -> str:
    wav = _to_wav16k(raw)
    try:
        with _infer_lock:
            result = _model.transcribe([wav], batch_size=1, verbose=False)
    finally:
        os.unlink(wav)
    item = result[0]
    return (item.text if hasattr(item, "text") else str(item)).strip()


class Handler(BaseHTTPRequestHandler):
    def _check_token(self):
        if self.headers.get("X-Token") != TOKEN:
            self.send_error(403)
            return False
        return True

    def _read_body(self):
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, body_str):
        body = body_str.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._html(render_page(admin=False))
        elif parsed.path == "/admin":
            qs = parse_qs(parsed.query)
            if qs.get("token", [""])[0] != TOKEN:
                self.send_error(403)
                return
            self._html(render_page(admin=True))
        elif parsed.path == "/summaries":
            self._html(render_summaries_page())
        elif parsed.path.startswith("/summaries/"):
            date = unquote(parsed.path[len("/summaries/"):]).strip("/")
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                self.send_error(404)
                return
            self._html(render_day_page(date))
        elif parsed.path == "/topics":
            self._html(render_topics_page())
        elif parsed.path == "/topics.json":
            talks = _load_talks()
            _ensure_topics_summary(talks)
            self._json(200, _topics_status(talks))
        elif parsed.path.startswith("/talks/"):
            target = _safe_talk_path(unquote(parsed.path[len("/talks/"):]))
            if target is None:
                self.send_error(404)
                return
            if target.name in PRIVATE_FILES and parse_qs(parsed.query).get("token", [""])[0] != TOKEN:
                self.send_error(403)
                return
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type",
                             _CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif parsed.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            q = queue.Queue()
            with _lock:
                history = list(_history)
                _subscribers.append(q)
            try:
                self.wfile.write(f"event: history\ndata: {json.dumps(history)}\n\n".encode())
                self.wfile.flush()
                while True:
                    item = q.get()
                    if item is None:
                        break
                    kind = item["type"]
                    if kind == "image":
                        self.wfile.write(f"event: image\ndata: {json.dumps(item)}\n\n".encode())
                    elif kind == "history":
                        self.wfile.write(f"event: history\ndata: {json.dumps(item['data'])}\n\n".encode())
                    elif kind == "reset":
                        self.wfile.write(b"event: reset\ndata: {}\n\n")
                    else:
                        self.wfile.write(f"event: chunk\ndata: {item['data']}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with _lock:
                    _subscribers.remove(q)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/vote":
            # Public: anyone in the audience may vote on speaker questions.
            raw = self._read_body()
            try:
                body = json.loads(raw or b"{}")
            except Exception:
                return self._json(400, {"error": "bad json"})
            res = _apply_vote(str(body.get("folder", "")), str(body.get("qid", "")),
                              str(body.get("from", "none")), str(body.get("to", "none")))
            if res is None:
                return self._json(400, {"error": "invalid vote"})
            return self._json(200, res)
        if not self._check_token():
            return
        if self.path == "/transcribe":
            raw = self._read_body()
            try:
                text = transcribe(raw)
            except Exception as e:
                return self._json(500, {"error": str(e)})
            if text:
                _push("text", text)
            self._json(200, {"text": text})
        elif self.path == "/image":
            raw = self._read_body()
            b64 = base64.b64encode(raw).decode()
            ts = datetime.now().strftime("%H:%M:%S")
            item = {"type": "image", "data": b64, "ts": ts}
            with _lock:
                _history.append(item)
                for q in _subscribers:
                    q.put(item)
            _live_dirty.set()
            self.send_response(204)
            self.end_headers()
        elif self.path == "/push":
            text = self._read_body().decode().strip()
            if text:
                _push("text", text)
            self.send_response(204)
            self.end_headers()
        elif self.path == "/save":
            raw = self._read_body()
            label = ""
            if raw:
                try:
                    label = str((json.loads(raw) or {}).get("label", "") or "").strip()
                except Exception:
                    label = ""
            try:
                result = _save_session(label[:80])
            except Exception as e:
                return self._json(500, {"error": str(e)})
            if result is None:
                return self._json(200, {"saved": None, "reason": "nothing to record"})
            self._json(200, result)
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


def main():
    global _model
    import nemo.collections.asr as nemo_asr
    print(f"Loading {MODEL_NAME} on {DEVICE}...", flush=True)
    _model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME, map_location=DEVICE)
    print(f"Model ready. Starting server on port {PORT}...", flush=True)
    # Resume an in-progress talk left over from a previous run, then start the
    # autosave loop that keeps the live buffer crash-recoverable.
    _load_live_state()
    threading.Thread(target=_live_autosave_worker, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Transcript server live at http://0.0.0.0:{PORT}", flush=True)
    print(f"  public viewer : http://0.0.0.0:{PORT}/", flush=True)
    print(f"  operator page : http://0.0.0.0:{PORT}/admin?token={TOKEN}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
