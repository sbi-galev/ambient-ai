# SBI4GALEV Live Transcript

Live transcription and AI tooling for the SBI4GALEV conference, hosted on the Turing GPU workstation.

## What it does

- Streams microphone audio to Turing's GPU for transcription using NeMo FastConformer (20x faster than real time)
- Displays a live transcript on a locally hosted website accessible to anyone on the UDN network
- Captures slide screenshots silently via PipeWire (no flash, no manual triggering) and displays them interleaved with the transcript
- At the end of each talk the operator presses **Save** on a token-protected `/admin` page: the transcript and every captured slide are written together into a dedicated, read-only folder under `transcripts/`, then the live view resets for the next speaker
- Each saved talk is then summarised by the local LLM (alan's engine — gemma-4-31b via SGLang) from the transcript **and** its slides; summaries are browsable at `/summaries`
- Every summary also suggests three kind, considerate questions the audience can vote up/down, and links to up to three related earlier talks

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

Open `http://172.24.17.90:7103` on any device connected to the Cambridge UDN. This page is **view-only** — it carries no controls and no auth token.

### Operator (saving talks)

Open `http://172.24.17.90:7103/admin?token=sbi4galev` (replace the token with whatever `TRANSCRIPT_TOKEN` is set to). This is the same live view plus an **End talk & Save** button and an optional speaker/title field.

At the end of a talk, type the speaker/title (optional) and press **Save**. The server bundles everything recorded so far into a dedicated folder and resets the live view for the next speaker. Keep the `/admin` URL to yourself — anyone with it can end a talk.

## Saved talks

Each Save writes one self-contained, read-only folder:

```
transcripts/
  2026-06-23_09-15-00_jane-smith/
    transcript.txt      # plain-text transcript
    transcript.html     # transcript + slides interleaved, open in a browser to review
    session.json        # ordered text/slide items with timestamps
    metadata.json       # label, save time, slide/chunk counts, first/last timestamps
    summary.json        # LLM summary (title, speaker, abstract, key_points, topics)
    summary.md          # the same summary as readable Markdown
    questions.json      # three kind speaker questions (votes live in ../../votes/)
    slides/
      slide_001.jpg
      slide_002.jpg
```

The folder is named `<timestamp>[_<label-slug>]`. Once the summary has been generated the folder and its contents are made read-only (`0555`/`0444`), and **there is no HTTP endpoint that deletes or modifies saved talks** — they cannot be removed through the browser or a POST. (To delete one deliberately on the server you must first restore write permission, e.g. `chmod -R u+w <folder>`. For tamper-proof storage beyond accidental loss, run `chattr +i` on the folder as root or move it to WORM/backup storage.)

Saving requires nothing recorded to be discarded: if nothing has been captured since the last save, Save is a no-op.

## Talk summaries

After each Save, a background worker sends the talk's transcript plus a sample of its slides (up to `SUMMARY_MAX_SLIDES`, default 8, evenly spaced) to the local model and writes `summary.json` / `summary.md` / `questions.json` into the talk folder. This runs **off** the Save response, so the reset for the next speaker is instant; the folder is finalised read-only once the summary lands (typically a few seconds). Generation is serialised, so back-to-back saves queue rather than overload the GPU.

- **`GET /summaries`** — public page listing every saved talk newest-first: title, speaker, abstract, key points, topic tags, a slide thumbnail, three speaker questions (see below), and links to related talks. A talk with no summary yet shows "generating…"; a failed one shows "unavailable". It deliberately does **not** link to the raw transcript.
- **`GET /talks/<folder>/<file>`** — read-only static access to saved files (path-traversal guarded). Slides, `summary.json`/`summary.md`, and `metadata.json` are public; the **full transcript files** (`transcript.txt`, `transcript.html`, `session.json`) are served only with the admin token, e.g. `…/transcript.html?token=sbi4galev`.

### Speaker questions & voting

Each summary includes three **kind, considerate questions** for the speaker, generated in the same LLM call with a prompt that insists they be warm, curious and encouraging — never harsh, sceptical or confrontational. They appear in a "Questions for the speaker" dropdown on `/summaries`, each with thumbs up / down.

- **`POST /vote`** `{folder, qid, from, to}` — public and anonymous; records a vote and returns the new `{up, down}`. Questions are ordered by score (most upvotes / fewest downvotes first), re-sorting live as people vote.
- Because finalised talk folders are read-only, votes live in a separate writable `votes/<folder>.json`. Each browser gets one vote per question (tracked in `localStorage`, click again to change or undo) — lightweight, not authenticated.

### Related talks

Each summary links to up to **three related earlier talks**, chosen at render time by topic- and title-keyword overlap among talks saved before it. The links jump to those talks' cards on the same page.

### Transcript privacy

Public viewers get the live transcript (as it streams), the slides, and the AI summaries — but the **saved full transcript is not downloadable** by them. The verbatim record stays on disk and is reachable only with the token (the `/admin` operator, or `…?token=` on a `/talks` transcript URL). To regenerate or backfill summaries, run `python3 backfill_summaries.py`.

The model is **alan's engine**: `google/gemma-4-31b-it` served by SGLang at `127.0.0.1:30000` (OpenAI-compatible, multimodal). alan itself is an interactive TUI, so the server talks to the same underlying model directly. Override with env vars `ALAN_LLM_URL` and `ALAN_LLM_MODEL`, or disable summaries entirely with `SUMMARIES=0`. If the model endpoint is unreachable the talk is still saved — only the summary is marked unavailable.

## Files

| File | Runs on | Purpose |
|------|---------|---------|
| `transcript_server.py` | Turing | NeMo STT + SSE website server + summaries |
| `backfill_summaries.py` | Turing | Generate summaries for talks saved without one |
| `live_transcribe.py` | Laptop | Mic capture + silent screen capture → POST to server |
| `screen_capture.py` | Laptop | PipeWire/ScreenCast portal screen capture class |
| `screenshot_watcher.py` | Laptop | Legacy: watches ~/Pictures/Screenshots for manual Print Screen |
| `test_transducer.py` | Turing | One-shot benchmark for the transducer model |

## Config

Scripts use token `sbi4galev` (env `TRANSCRIPT_TOKEN`) for authentication on `/transcribe`, `/push`, `/image`, and `/save`. The public, read-only pages (`GET /`, `GET /summaries`, `GET /talks/…`) are open to anyone on the UDN and carry no token; the operator page (`GET /admin?token=…`) requires the token and is the only place the Save control is exposed. Set a stronger `TRANSCRIPT_TOKEN` for real events.

Server binds to `0.0.0.0:7103` — accessible on UDN only (Turing is not internet-facing).

STT model: `stt_en_fastconformer_transducer_xxlarge`, cached at `~/.cache/torch/NeMo/` on Turing. First start after a reboot loads in ~30s.

## Turing-side dependencies

Already installed as part of the `turing-voice` system package:
- NeMo (`nemo_toolkit[asr]`)
- ffmpeg
- CUDA drivers
