# SBI4GALEV Live Transcript

Live transcription and AI tooling for the SBI4GALEV conference, hosted on the Turing GPU workstation.

## What it does

- Streams microphone audio to Turing's GPU for transcription using NeMo FastConformer (20x faster than real time)
- Displays a live transcript on a locally hosted website accessible to anyone on the UDN network
- Captures slide screenshots silently via PipeWire (no flash, no manual triggering) and displays them interleaved with the transcript
- Transcript is visible during talks and cleared between them

## Architecture

```
Laptop mic ──────────────────────────────────────────┐
                                                      ▼
PipeWire screen → live_transcribe.py → POST audio/image → Turing (172.24.17.90:7103)
                                                               ↓ NeMo STT
                                                          SSE push
                                                               ↓
                                            Attendee phones/laptops ← GET /stream
```

## Laptop setup

### System dependencies (apt)

```bash
sudo apt install \
  python3-gi python3-dbus \
  gstreamer1.0-pipewire gstreamer1.0-plugins-base \
  gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0
```

### Python dependencies (pip)

```bash
pip install -r requirements.txt
```

## Running

### On Turing (start first)

```bash
PYTHONPATH=/usr/lib/turing-voice python3 transcript_server.py
```

Model loading takes ~30s. Wait for "Transcript server live" before starting the laptop script.

### On your laptop

```bash
python3 live_transcribe.py
```

On first run, a one-time permission dialog appears for screen capture. After that it runs silently — screenshots are taken each chunk and sent only if the slide has changed significantly. The last sent slide is cached at `~/Pictures/SBI4GALEV/last_sent.jpg`.

### Attendees

Open `http://172.24.17.90:7103` on any device connected to the Cambridge UDN.

## Files

| File | Runs on | Purpose |
|------|---------|---------|
| `transcript_server.py` | Turing | NeMo STT + SSE website server |
| `live_transcribe.py` | Laptop | Mic capture + silent screen capture → POST to server |
| `screen_capture.py` | Laptop | PipeWire/ScreenCast portal screen capture class |
| `screenshot_watcher.py` | Laptop | Legacy: watches ~/Pictures/Screenshots for manual Print Screen |
| `test_transducer.py` | Turing | One-shot benchmark for the transducer model |

## Config

Scripts use token `sbi4galev` for authentication on `/transcribe`, `/push`, `/save`, `/clear`, `/image`. The viewer page (`GET /`) is open to anyone on the UDN.

Server binds to `0.0.0.0:7103` — accessible on UDN only (Turing is not internet-facing).

STT model: `stt_en_fastconformer_transducer_xxlarge`, cached at `~/.cache/torch/NeMo/` on Turing. First start after a reboot loads in ~30s.

## Turing-side dependencies

Already installed as part of the `turing-voice` system package:
- NeMo (`nemo_toolkit[asr]`)
- ffmpeg
- CUDA drivers
