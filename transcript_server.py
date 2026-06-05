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
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live Transcript</title>
  <style>
    body { font-family: sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px;
           background: #0a0a0a; color: #e0e0e0; }
    h2 { color: #aaa; font-weight: 300; letter-spacing: 2px; text-transform: uppercase;
         font-size: 0.8em; margin-bottom: 2em; }
    #transcript { line-height: 1.8; font-size: 1.1em; }
    .chunk.new { color: #fff; }
    .chunk.old { color: #888; }
    .slide { display: block; max-width: 100%; margin: 1.2em 0 0.8em 0;
             border-radius: 6px; border: 1px solid #222; }
    #status { position: fixed; top: 10px; right: 15px; font-size: 0.7em;
              color: #444; letter-spacing: 1px; }
    #status.live { color: #4a4; }
    #save-btn { position: fixed; bottom: 20px; right: 20px;
                background: #1a1a1a; color: #888; border: 1px solid #333;
                padding: 8px 16px; border-radius: 4px; cursor: pointer;
                font-size: 0.8em; letter-spacing: 1px; }
    #save-btn:hover { color: #fff; border-color: #555; }
    #save-btn.saved { color: #4a4; border-color: #4a4; }
  </style>
</head>
<body>
  <h2>SBI4GALEV — Live Transcript</h2>
  <div id="transcript"></div>
  <div id="status">connecting…</div>
  <button id="save-btn" onclick="saveTranscript()">Save transcript</button>
  <script>
    const t = document.getElementById('transcript');
    const s = document.getElementById('status');
    const es = new EventSource('/stream');
    es.onopen = () => { s.textContent = '● live'; s.className = 'live'; };
    es.onerror = () => { s.textContent = 'reconnecting…'; s.className = ''; };
    es.addEventListener('history', e => {
      t.innerHTML = '';
      JSON.parse(e.data).forEach(item => appendItem(item, false));
    });
    es.addEventListener('chunk', e => {
      document.querySelectorAll('.chunk.new').forEach(el => el.className = 'chunk old');
      appendItem({type:'text', data: e.data}, true);
      window.scrollTo(0, document.body.scrollHeight);
    });
    es.addEventListener('image', e => {
      appendItem({type:'image', data: e.data}, true);
      window.scrollTo(0, document.body.scrollHeight);
    });
    function saveTranscript() {
      const btn = document.getElementById('save-btn');
      fetch('/save', {method:'POST', headers:{'X-Token':'sbi4galev'}})
        .then(r => r.json())
        .then(d => { btn.textContent = 'Saved ✓'; btn.className = 'saved';
                     setTimeout(() => { btn.textContent = 'Save transcript'; btn.className = ''; }, 3000); })
        .catch(() => { btn.textContent = 'Error'; setTimeout(() => { btn.textContent = 'Save transcript'; }, 2000); });
    }
    function appendItem(item, isNew) {
      if (item.type === 'image') {
        const img = document.createElement('img');
        img.src = 'data:image/jpeg;base64,' + item.data;
        img.className = 'slide';
        t.appendChild(img);
      } else {
        const span = document.createElement('span');
        span.className = 'chunk ' + (isNew ? 'new' : 'old');
        span.textContent = item.data + ' ';
        t.appendChild(span);
      }
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
                    event = item["type"] if item["type"] == "image" else "chunk"
                    self.wfile.write(f"event: {event}\ndata: {item['data']}\n\n".encode())
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
            _push("image", b64)
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
