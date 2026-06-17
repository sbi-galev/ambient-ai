"""SBI4GALEV live transcript server.

Loads stt_en_fastconformer_transducer_xxlarge once on startup, then serves:
  POST /transcribe  — audio file → NeMo inference → pushes text to SSE
  POST /image       — JPEG screenshot → pushes to SSE if slide changed
  POST /push        — raw text → pushes to SSE (fallback/testing)
  POST /save        — saves current transcript to a timestamped file
  POST /clear       — saves transcript then clears it
  GET  /stream      — SSE stream for browsers
  GET  /            — transcript viewer HTML
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
import io
import json
import os
import queue
import subprocess
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SAVE_DIR = Path(__file__).parent / "transcripts"
SAVE_DIR.mkdir(exist_ok=True)

PORT = int(os.environ.get("TRANSCRIPT_PORT", "7103"))
TOKEN = os.environ.get("TRANSCRIPT_TOKEN", "sbi4galev")
DEVICE = os.environ.get("STT_DEVICE", "cuda:0")
MODEL_NAME = "stt_en_fastconformer_transducer_xxlarge"
SAMPLE_RATE = 16000

_subscribers = []
_lock = threading.Lock()
_infer_lock = threading.Lock()
_history = []  # list of {"type": "text"|"image", "data": str}

HTML = """<!DOCTYPE html>
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

    #save-btn {
      position: fixed; bottom: 20px; right: 20px; z-index: 100;
      background: #111; color: #555; border: 1px solid #222;
      padding: 8px 18px; border-radius: 4px; cursor: pointer;
      font-size: 0.75em; letter-spacing: 1px;
      transition: color 0.2s, border-color 0.2s;
    }
    #save-btn:hover { color: #fff; border-color: #444; }
    #save-btn.saved { color: #4a4; border-color: #4a4; }

    #nav {
      position: fixed; right: 20px; top: 50%; transform: translateY(-50%);
      display: flex; flex-direction: column; gap: 6px; z-index: 100;
    }
    .nav-dot {
      width: 5px; height: 5px; border-radius: 50%;
      background: #333; cursor: pointer; transition: background 0.3s;
    }
    .nav-dot.active { background: #888; }
  </style>
</head>
<body>
  <div id="container"></div>
  <div id="status">connecting…</div>
  <button id="save-btn" onclick="saveTranscript()">Save</button>
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

    // ── SSE ─────────────────────────────────────────────────────────────

    const es = new EventSource('/stream');
    es.onopen = () => { statusEl.textContent = '● live'; statusEl.className = 'live'; };
    es.onerror = () => { statusEl.textContent = 'reconnecting…'; statusEl.className = ''; };
    es.addEventListener('history', e => {
      container.innerHTML = ''; sections = []; navEl.innerHTML = '';
      JSON.parse(e.data).forEach(item => appendItem(item, false));
    });
    es.addEventListener('chunk', e => appendItem({type:'text', data: e.data}, true));
    es.addEventListener('image', e => appendItem(JSON.parse(e.data), true));

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

    // ── Save ─────────────────────────────────────────────────────────────

    function saveTranscript() {
      const btn = document.getElementById('save-btn');
      fetch('/save', {method:'POST', headers:{'X-Token':'sbi4galev'}})
        .then(r => r.json())
        .then(() => { btn.textContent = 'Saved ✓'; btn.className = 'saved';
                      setTimeout(() => { btn.textContent = 'Save'; btn.className = ''; }, 3000); })
        .catch(() => { btn.textContent = 'Error';
                       setTimeout(() => { btn.textContent = 'Save'; }, 2000); });
    }
  </script>
</body>
</html>"""


def _save_transcript() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = SAVE_DIR / f"{timestamp}.txt"
    with _lock:
        text = " ".join(item["data"] for item in _history if item["type"] == "text")
    path.write_text(text.strip() + "\n")
    print(f"Transcript saved: {path}", flush=True)
    return path


def _push(event_type, data):
    item = {"type": event_type, "data": data}
    with _lock:
        _history.append(item)
        for q in _subscribers:
            q.put(item)


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

    def do_GET(self):
        if self.path == "/":
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stream":
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
                    if item["type"] == "image":
                        self.wfile.write(f"event: image\ndata: {json.dumps(item)}\n\n".encode())
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
            self.send_response(204)
            self.end_headers()
        elif self.path == "/push":
            text = self._read_body().decode().strip()
            if text:
                _push("text", text)
            self.send_response(204)
            self.end_headers()
        elif self.path == "/save":
            path = _save_transcript()
            self._json(200, {"saved": str(path)})
        elif self.path == "/clear":
            _save_transcript()
            with _lock:
                _history.clear()
                for q in _subscribers:
                    q.put(None)
            self.send_response(204)
            self.end_headers()
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
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Transcript server live at http://0.0.0.0:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
