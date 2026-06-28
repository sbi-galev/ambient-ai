# Ambient AI for Conferences

A self-hosted toolkit that turns a scientific meeting into a durable, searchable record. A single GPU machine listens to the room, transcribes the talks, and captures the projected slides into a context store. This is then used by a local or external LLM agent to produce per-talk summaries, suggest audience questions, provide per-day overviews, a conference-wide topic graph, a publishable static archive, and a multi-page summary white paper.

With privacy in mind, the full stack can be run on local hardware. None of the context store is sent to an external service; the only network traffic is on your own LAN.

> This repository was built for and deployed at **SBI4GALEV 2026**. It is now a general template: all site- and hardware-specific values live in
> [`config.toml`](config.toml), which ships with the SBI4GALEV deployment's values as a worked example. **To run your own event, edit `config.toml`**.
> You can optionally add extra features as required or update the design by working with your own favourite local or external LLM.

---

## Contents

- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Configuration (`config.toml`)](#configuration-configtoml)
- [Running an event](#running-an-event)
- [The context store (data model)](#the-context-store-data-model)
- [AI features](#ai-features)
- [Publishing a public archive](#publishing-a-public-archive)
- [White papers](#white-papers)
- [Maintenance scripts](#maintenance-scripts)
- [Script reference](#script-reference)
- [Privacy & security](#privacy--security)

---

## What it does

While a talk is in progress the laptop streams microphone (or shared-system) audio
to the GPU, where NeMo FastConformer transcribes it at roughly 20× real time, and
quietly grabs the projected slides through the PipeWire screen-capture portal.
There is no flash and nothing to trigger by hand; a slide is only sent when the
picture actually changes. The running transcript and its slides appear on a plain
web page that anyone on the network can open, with no app and no login.

When a talk ends the operator presses **End talk & Save** on the token-protected
`/admin` page. Everything captured so far is written into the context store as one
read-only talk folder, and the live view resets for the next speaker.

From there the LLM agent works over the context store. Each talk gets a summary
drawn from both its transcript and its slides, three suggested audience questions
that attendees can vote on, and links to related earlier talks (browsable at
`/summaries`). Across talks the agent keeps per-day overviews and a
conference-wide key-topics graph (`/topics`) up to date. Two further outputs build
on the same store: `export_static.py` publishes a curated subset as a static
`site/` for GitHub Pages, gated per speaker by a fail-closed permissions allowlist,
and `whitepaper.py` writes a short LaTeX summary of the whole meeting, optionally
compared against last year.

## How it works

```
  Laptop (mic + screen)                         GPU server
  ─────────────────────                         ──────────
  microphone ─┐                          ┌─► NeMo STT ─┐
              ├─ live_transcribe.py ─POST─┤             ├─► SSE ─► attendee browsers (GET /)
  slides  ────┘   (audio + JPEG)         └─► slide store┘
                                                │
                          operator presses Save │
                                                ▼
                                  context store: transcripts/<talk>/  (read-only)
                                                │
                          local or external LLM agent (summaries, questions,
                          day overviews, key-topics graph)
                                                │
                ┌───────────────────────────────┼───────────────────────────────┐
                ▼                                ▼                                ▼
        GET /summaries, /topics        export_static.py ─► site/          whitepaper.py
        (live browsing)                ─► GitHub Pages                    ─► LaTeX / PDF
```

The capture client (laptop) and the server (GPU box) are separate machines that
share this repository, and therefore the same `config.toml`. The context store
lives on the server, which binds to the LAN and is never exposed to the internet;
only the curated static archive is published.

## Requirements

### Server (the GPU box)

- **Python 3.11+** (the config loader uses the standard-library `tomllib`).
- **A CUDA GPU** for real-time STT. CPU works for testing — set
  `[stt].device = "cpu"` — but is far slower.
- **NeMo ASR** and its dependencies:
  ```bash
  pip install "nemo_toolkit[asr]"      # also pulls in torch/torchaudio
  sudo apt install ffmpeg
  ```
  If NeMo is installed as a *system* package rather than via pip, add it to
  `PYTHONPATH` when launching the server (see [Running an event](#running-an-event)).
- **An OpenAI-compatible, multimodal chat endpoint** for the AI features. Run one
  locally for full privacy — e.g. [SGLang](https://github.com/sgl-project/sglang)
  or [vLLM](https://github.com/vllm-project/vllm) serving a vision-language model
  (the reference deployment used `google/gemma-4-31b-it`) — or point it at an
  external API. It is optional: set `[llm].enabled = false` and talks are still
  saved, just without summaries.

### Laptop (the capture client)

- **System packages** (Debian/Ubuntu/GNOME-Wayland) for silent screen capture:
  ```bash
  sudo apt install \
    python3-gi python3-dbus \
    gstreamer1.0-pipewire gstreamer1.0-plugins-base \
    gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0
  ```
- **Python packages:**
  ```bash
  pip install -r requirements.txt          # sounddevice, numpy, Pillow
  ```

Screen capture needs a desktop that supports the XDG ScreenCast portal (GNOME
Wayland). If yours does not, the client prints a warning and carries on —
transcription still works, slides just are not sent.

## Configuration (`config.toml`)

[`config.toml`](config.toml) is the **single place** to configure an event. Both
the server and the laptop client read it via [`config.py`](config.py). Every
value can also be overridden at runtime by an environment variable, so existing
deployment scripts keep working. Resolution order, first match wins:

> **environment variable → `config.toml` → built-in default**

| Section | Key | What it is | Env override |
|---|---|---|---|
| `[conference]` | `short_name` | Tag in page titles & prompts (e.g. `SBI4GALEV`) | `CONF_SHORT` |
| | `full_name` | Full descriptive name | `CONF_FULL` |
| | `year` | Used in the white-paper title | `CONF_YEAR` |
| | `assistant_name` | Persona the model writes as; byline on pages | `ASSISTANT_NAME` |
| `[server]` | `host` | Bind interface (`0.0.0.0` = all/LAN) | `TRANSCRIPT_HOST` |
| | `port` | Server port | `TRANSCRIPT_PORT` |
| | `token` | Shared secret for `/admin` & write endpoints | `TRANSCRIPT_TOKEN` |
| | `public_url` | How clients/attendees reach the server | `TRANSCRIPT_URL` |
| `[stt]` | `model` | NeMo ASR model name | `STT_MODEL` |
| | `device` | Torch device (`cuda:0`, `cpu`) | `STT_DEVICE` |
| | `sample_rate` | Audio sample rate (Hz) | `STT_SAMPLE_RATE` |
| `[llm]` | `url` | OpenAI-compatible chat-completions endpoint | `ALAN_LLM_URL` |
| | `model` | Model name to request | `ALAN_LLM_MODEL` |
| | `enabled` | `false` saves talks without AI summaries | `SUMMARIES` |
| | `max_slides` | Slides (evenly sampled) sent per summary | `SUMMARY_MAX_SLIDES` |
| `[tuning]` | `save_offset_seconds` | Hold back the last N s at save time | `SAVE_OFFSET_SECONDS` |
| | `live_flush_seconds` | Crash-recovery snapshot interval | `LIVE_FLUSH_SECONDS` |
| | `topics_max` | Cap on synthesised key topics | `TOPICS_MAX` |

**Minimum to change for your event:** `[conference]` names, `[server].token` (set
a strong one!) and `[server].public_url` (the server's LAN address, so the laptop
client and attendees can reach it), plus `[stt].device` and `[llm].url`/`model`
to match your hardware.

To run from a config elsewhere, point `CONFIG_FILE` at it:
`CONFIG_FILE=/path/to/other.toml python3 transcript_server.py`.

## Running an event

### 1 · Start the server (GPU box, start first)

```bash
python3 transcript_server.py
```

If your NeMo install is a system package, prefix with its path, e.g.
`PYTHONPATH=/usr/lib/<stt-package> python3 transcript_server.py`. Model loading
takes ~30 s after a reboot; wait for the **"transcript server live"** line before
starting capture. The STT model is cached under `~/.cache/torch/NeMo/`.

### 2 · Start capture (laptop)

Find your microphone's device index:

```bash
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```

Then run, replacing `N` with that index:

```bash
python3 live_transcribe.py --device N
```

On first run a one-time screen-capture permission dialog appears — approve it;
after that it runs silently. To capture **shared computer audio** instead of the
mic (e.g. a remote Zoom speaker), use `python3 live_transcribe.py --zoom`.

### 3 · Attendees

Open `[server].public_url` (e.g. `http://172.24.17.90:7103`) on any device on the
network. This page is **view-only** — no controls, no token.

### 4 · Operator (saving talks)

Open `<public_url>/admin?token=<your-token>`. This is the same live view plus an
**End talk & Save** button and an optional speaker/title field. At the end of a
talk, type the speaker/title and press **Save**; the server bundles everything
recorded so far and resets for the next speaker. Keep this URL private — anyone
with it can end a talk.

## The context store (data model)

The context store is the corpus the system captures and then reasons over: the
per-talk transcripts, slides and screenshots, their timing and metadata, and any
other context you add (for example the official schedule abstracts). It lives
under `transcripts/`, one self-contained folder per talk, named
`<timestamp>[_<speaker-slug>]`. Each save writes one such folder, and the LLM
agent later writes its summary back alongside the raw material:

```
transcripts/
  2026-06-23_09-15-00_jane-smith/
    transcript.txt      # plain-text transcript
    transcript.html     # transcript + slides interleaved (open in a browser)
    session.json        # ordered text/slide items with timestamps
    metadata.json       # label, save time, slide/chunk counts, timestamps
    summary.json        # LLM summary (title, speaker, abstract, key_points, topics)
    summary.md          # the same summary as Markdown
    questions.json      # three suggested speaker questions
    slides/
      slide_001.jpg …
```

Once its summary lands a talk folder is made read-only (`0555`/`0444`), and no HTTP
endpoint can delete or change a saved talk. The in-progress talk is held in memory
and snapshotted to `transcripts/.live.json` every `live_flush_seconds`, so it
survives a crash or restart. The store itself is git-ignored: the raw recordings
are never committed, and only the curated static archive is ever published.

## AI features

These need an LLM endpoint (`[llm].enabled = true`); everything below is the LLM
agent reading the context store. Generation runs off the Save response, so the
reset for the next speaker is instant, and is serialised so back-to-back saves
queue rather than overload the model.

Each talk is summarised from its transcript and slides into a title, speaker,
abstract, key points and topic tags, shown newest-first at `GET /summaries` with a
slide thumbnail. The summaries never link to the raw transcript. The
same pass suggests three constructive questions per talk, which attendees vote on
anonymously (`POST /vote`); because talk folders are read-only the votes live
separately under `votes/`. Each summary also links to up to three earlier talks
chosen by topic and title overlap.

Across the whole store the agent keeps two syntheses fresh. Per-day overviews
(cached in `day_summaries/`) are rebuilt whenever a day's set of talks changes. The
conference-wide key-topics page (`GET /topics`, cached in `topic_summaries/`)
clusters every talk's topic tags into the meeting's main threads and draws a graph
linking each topic back to the talks that raised it; if the model is unreachable it
falls back to plain keyword aggregation.

If the endpoint is down a talk is still saved, with its summary marked
unavailable — regenerate it later with `backfill_summaries.py`.

## Publishing a public archive

`export_static.py` renders a curated subset of the context store — `/summaries`,
the per-day pages and `/topics` — into a self-contained `site/` of plain HTML and
the slide thumbnails it references, with the server-only behaviour (live SSE,
voting, absolute links) stripped out. It reuses the server's own render functions,
so run it on the GPU box, where the context store lives (and the LLM, for any stale
day or topic caches).

### Permissions allowlist (fail-closed)

Publishing is **opt-in per speaker**, controlled by a `public_talks.txt`
allowlist: one approved talk per line (folder name or readable slug), anything
not listed is excluded, and a missing/empty file exports nothing. This file is
**git-ignored** — it names real speakers, so it stays out of the template. Copy
the tracked [`public_talks.example.txt`](public_talks.example.txt) to
`public_talks.txt` and fill in your event's approved talks; add a speaker once
they grant permission, then re-export.

### Build & deploy

The built `site/` is **not** committed to the main branch (it's git-ignored). It
lives on its own `gh-pages` branch, which GitHub Pages serves directly. The
[`publish_site.sh`](publish_site.sh) helper builds the archive and pushes it
there in one step:

```bash
./publish_site.sh            # = export_static.py --clean, then push site/ to gh-pages
```

It force-pushes a single fresh commit to `gh-pages` (the branch only ever holds
the latest generated site — no history to scrub). One-time setup: repo
**Settings → Pages → Source: Deploy from a branch → `gh-pages` / (root)**. No
Actions workflow or personal credentials are involved; GitHub auto-builds Pages
on each push to the branch. To build without publishing, run
`python3 export_static.py --clean` and inspect `site/` directly.

Useful flags: `--no-llm` (skip the model; topics use the keyword fallback),
`--all-slides`, `--strict` (treat missing summaries as errors), `-v`. Serve a
build locally with `python3 -m http.server -d site 8000`.

## White papers

**`whitepaper.py`** writes the meeting's summary white paper: a ≤3-page LaTeX document built by the LLM agent from the context store — the on-box summaries, day overviews and topic synthesis.
Pass last year's abstract booklet (or drop an `*abstract*booklet*.pdf` in the repo root) for a grounded year-on-year comparison.
Needs `pdftotext` or `pdfminer.six` to read a PDF booklet, and `latexmk`/`pdflatex` for `--pdf`.
```bash
python3 whitepaper.py --refresh --pdf
```

Generated `.tex`/`.pdf` are git-ignored (reproducible from the script).

## Maintenance scripts

All run on the GPU box and reuse `transcript_server`'s generators:

- **`backfill_summaries.py [--force] [folder …]`** — summarise talks saved
  without one (e.g. before summaries existed, or after a model outage).
- **`backfill_day_summaries.py [--force] [YYYY-MM-DD …]`** — warm/rebuild the
  per-day overviews.
- **`backfill_topics.py [--force] [--no-llm]`** — warm/rebuild the key-topics
  synthesis.

### Re-cutting talk boundaries

The "End talk & Save" click is mistimed relative to the speech-to-text lag, so a
talk's opening can be trapped at the end of the previous folder, or its ending can
spill into the next. `fix_transcript_cuts.py` repairs this entirely on-box:

```bash
python3 fix_transcript_cuts.py analyze                  # timeline, markers, boundary windows
python3 fix_transcript_cuts.py propose --out plan.json  # the local LLM decides the cuts
python3 fix_transcript_cuts.py apply plan.json --dry-run
python3 fix_transcript_cuts.py apply plan.json          # backs up, rebuilds, re-locks, verifies
python3 backfill_summaries.py --force <changed-folder> ...   # refresh the moved talks' summaries
```

`propose` sends each boundary window to the same local model as the summaries, so
no transcript leaves the LAN. It only writes a plan — you review it and `apply` it
yourself, with the originals backed up under `transcripts/.backups/`. Splitting a
folder that bundles several talks or a break is done with a hand-written plan (the
schema is documented at the top of the script).

### Slide cleanup

Capture is noisy — setup screens, the next speaker's title shown early, blank
frames, slides that bleed across hand-overs. All on-box and human-reviewed
(nothing is auto-deleted):

- **`slide_audit.py`** — describe + triage every slide (content vs junk + a
  confidence) → `transcripts/slides_audit.json` (descriptions double as captions).
- **`slide_review.py`** — split that into `auto_cull.json` (high-confidence junk)
  and a `site/review.html` to approve/adjust the rest and download a plan.
- **`fix_transcript_cuts.py slides-propose`** — flag boundary slides that belong
  to the neighbouring talk (timestamp + content) or are junk.
- **`cull_slides.py`** — propose whole-talk junk culls.

Apply any plan with **`fix_transcript_cuts.py slides-apply <plan>`**: backs up,
moves/renumbers slides in timestamp order, re-locks, verifies; culled slides go to
`slides/.culled/` (reversible). Refresh affected talks with
`backfill_summaries.py --force`.

### Methods & Tools indexes

**`concepts_tools.py build`** has the local model extract the methods/terms and
the software & simulation suites used across the approved talks, counts mentions
(transcripts + slide descriptions) and links each to the talks → `methods.html` /
`tools.html`. Tool URLs are model-proposed then verified by an on-box HTTP fetch
(dead links dropped). Re-run `concepts_tools.py count` (no model) after editing
`transcripts/concepts_tools.raw.json`.


## Script reference

| File | Runs on | Purpose |
|------|---------|---------|
| `config.toml` / `config.py` | both | Single source of truth for all settings |
| `transcript_server.py` | server | NeMo STT + SSE site + summaries/topics |
| `live_transcribe.py` | laptop | Mic/system-audio capture + silent slide capture → POST |
| `screen_capture.py` | laptop | PipeWire/ScreenCast portal capture (used by the above) |
| `export_static.py` | server | Render the archive to a static `site/` for Pages |
| `public_talks.example.txt` | server | Template for the per-speaker publish allowlist; copy to `public_talks.txt` (git-ignored, fail-closed) |
| `backfill_summaries.py` | server | Generate summaries for talks missing one |
| `backfill_day_summaries.py` | server | Generate/refresh per-day overviews |
| `backfill_topics.py` | server | Generate/refresh the key-topics synthesis |
| `fix_transcript_cuts.py` | server | Re-cut mis-bounded talks + reassign/cull boundary slides (local-LLM) |
| `slide_audit.py` | server | Describe + triage every slide (content vs junk) |
| `cull_slides.py` | server | Propose whole-talk junk-slide culls (local-LLM) |
| `slide_review.py` | server | Build the auto-cull plan + `review.html` |
| `concepts_tools.py` | server | Build the Methods & Tools indexes (`methods.html` / `tools.html`) |
| `whitepaper.py` | server | Conference **summary** white paper (via the LLM agent) |
| `whitepaper/ambient_ai_whitepaper.py` | server | Methods paper about the system |
| `test_transducer.py` | server | One-shot STT smoke test / benchmark |

## Privacy & security

Keeping the context store private is the main design goal. If you point `[llm].url`
at a model running on your own hardware, the whole pipeline stays on the LAN and no
transcript, slide or abstract ever leaves the building. Pointing it at an external
API is supported too, but then the slices of the context store sent for
summarisation do leave your network, so choose that endpoint accordingly.

The write endpoints (`/admin`, `/save`, `/transcribe`, `/image`, `/push`) require
`[server].token`; the public pages (`/`, `/summaries`, `/topics` and `/talks/…`
assets) carry none, so set a strong token for a real event. Public viewers see the
live stream, the slides and the AI summaries, but not the verbatim transcript: the
full transcript files are served only with the token (`…?token=…`).

Finalised talk folders are read-only and no endpoint can delete or modify them; for
stronger guarantees run `chattr +i` on them or move them to WORM/backup storage.
Keep the server on the LAN and publish only the curated static archive, through the
permissions allowlist.
