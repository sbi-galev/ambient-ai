# SBI4GALEV Live Transcript

Live transcription and AI tooling for the SBI4GALEV conference, hosted on the Turing GPU workstation.

## What it does

- Streams microphone audio to Turing's GPU for transcription using NeMo FastConformer (20x faster than real time)
- Displays a live transcript on a locally hosted website accessible to anyone on the UDN network
- Captures slide screenshots (via Print Screen) and displays them interleaved with the transcript
- Transcript is cleared after each talk — not stored

## Architecture

```
Laptop mic → live_transcribe.py → POST audio → Turing (172.24.17.90:7103)
                                                    ↓ NeMo STT
Print Screen → screenshot_watcher.py → POST image → Turing
                                                    ↓ SSE push
                                         Attendee phones/laptops ← GET /stream
```

## Running

### On Turing (start first)

```bash
PYTHONPATH=/usr/lib/turing-voice python3 transcript_server.py
```

Model loading takes ~30s. Wait for "Transcript server live" before starting the laptop scripts.

### On your laptop (two terminals)

```bash
# Terminal 1 — transcription
/home/toby/.local/share/pipx/venvs/recordcli/bin/python live_transcribe.py

# Terminal 2 — slide screenshots (press Print Screen to capture a slide)
python3 screenshot_watcher.py
```

### Attendees

Open `http://172.24.17.90:7103` on any device connected to the Cambridge UDN.

## Files

| File | Runs on | Purpose |
|------|---------|---------|
| `transcript_server.py` | Turing | NeMo STT + SSE website server |
| `live_transcribe.py` | Laptop | Mic capture → POST to server |
| `screenshot_watcher.py` | Laptop | Watches ~/Pictures/Screenshots, POSTs new slides |
| `test_transducer.py` | Turing | One-shot benchmark for the transducer model |

## Config

Both scripts use token `sbi4galev` for authentication. The server binds to `0.0.0.0:7103` — accessible on UDN only (Turing is not internet-facing).

STT model: `stt_en_fastconformer_transducer_xxlarge`, cached at `~/.cache/torch/NeMo/` on Turing.
