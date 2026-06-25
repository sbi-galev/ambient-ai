---
name: fix-transcript-cuts
description: Check and correct the boundaries of saved talk transcripts in transcripts/. Use when talks have leaked into each other (one talk's folder contains the start of the next speaker, or one folder bundles several talks / a break), or when asked to verify, re-cut, realign, or clean up the conference transcript folders. Moves misplaced text AND slides to the right talk, regenerates files in the server's format, and routes discards to a hidden bin/.
---

# Fix transcript cuts

Saved talks live in `transcripts/<timestamp>_<slug>/`, written by
`transcript_server.py`. Each is a slice of one continuous recording, cut by a
manual "End talk & Save" click. Because the click is mistimed relative to the
speech-to-text lag, content leaks across boundaries:

- **Save clicked late** (the common case): the next speaker has already started,
  so their opening is trapped at the **end** of the current folder, and that
  talk's own opening is missing (it's at the end of the *previous* folder).
- **Save clicked early**: the current speaker's ending leaks into the **start**
  of the next folder.
- **Save missed entirely**: one folder bundles several talks plus a break
  (a "mega-folder").

> A 30s save offset now exists in `transcript_server.py`
> (`SAVE_OFFSET_SECONDS`) to reduce late-click leakage going forward, but it is
> a fixed approximation — it does not fix early clicks or missed saves, so this
> skill is still needed for cleanup.

The STT never emits `[applause]`. The real boundaries are marked by **chair
phrases** ("thank X again", "next speaker", "welcome back", "please join me in
welcoming…") and **speaker self-intros** ("hi everyone, my name's…").

## Folder format (what the tool keeps consistent)

Each folder: `session.json` (ordered `text`/`slide` chunks with `ts`),
`transcript.txt`, `transcript.html`, `metadata.json`, `slides/slide_NNN.jpg`.
Folders are finalised **read-only** (files `0444`, dirs `0555`). The tool
unlocks, rewrites, and re-locks them; never edit by hand.

`transcripts/bin/` and `transcripts/.backups/` are **invisible to the web
interface** — the server lists `transcripts/` non-recursively and only shows
dirs that have a `metadata.json` directly. So `bin/` (no top-level metadata) and
its children never appear. Put cut/junk material in `bin/` instead of deleting.

## Workflow

All commands run from the repo root. The tool is
`.claude/skills/fix-transcript-cuts/cut_tool.py` (set `TRANSCRIPTS_DIR` to point
elsewhere; it defaults to the repo's `transcripts/`).

### 1. Analyze

```bash
python3 .claude/skills/fix-transcript-cuts/cut_tool.py analyze
```

This prints: a chronological timeline (with a `self-intros>=2` flag for likely
mega-folders), every transition marker per talk (with chunk index + ts), and the
tail→head window for each consecutive pair. Read it to decide, for each
boundary, the **exact chunk index** where the next speaker actually starts.

If a boundary window isn't conclusive, read more context directly, e.g.:

```bash
python3 - <<'PY'
import json
s=json.load(open("transcripts/<folder>/session.json"))
for i,x in enumerate(s[290:330], start=290):
    print(i, x["ts"], x.get("text","[SLIDE]") if x["type"]=="text" else "[SLIDE]")
PY
```

### 2. Decide the cuts

For each talk, find:
- its **true start** = the chunk where that speaker first talks (their self-intro
  or first sentence), and
- its **true end** = the last chunk before the chair's applause/handoff.

Leakage almost always means: a block at the **end of folder N** belongs to
**N+1** (move it forward), or a block at the **start of N+1** belongs to **N**
(move it back). Mega-folders split into several talks; break/test/lunch
remnants go to `bin/`.

Keep slides with their talk — if the moved range contains `[SLIDE]` chunks, those
slide files move too (the tool handles the renumbering).

### 3. Write a plan and apply

Express the desired final state as a plan. **Every folder that changes must be
listed as a `dest` with its complete new contents**, built from `segments` that
slice source folders (`start`/`end` are Python slice bounds into the source's
original `session.json`). Sources are read from an automatic backup, so a folder
can be both a source and a destination.

Minimal example — move the trapped tail of folder A into folder B (a late-click
boundary), and split a mega-folder C into a kept talk + a binned remainder:

```json
{
  "backup_label": "recut",
  "operations": [
    { "dest": "A", "segments": [ {"src": "A", "start": 0, "end": 155} ] },
    { "dest": "B", "segments": [
        {"src": "A", "start": 155, "end": 9999},
        {"src": "B", "start": 0,   "end": 9999} ] },

    { "dest": "C", "segments": [ {"src": "C", "start": 3, "end": 158} ] },
    { "dest": "2026-..._exoplanet-talk", "bin": true,
      "label": "Exoplanet talk (cut from C)",
      "segments": [ {"src": "C", "start": 158, "end": 719} ] },
    { "dest": "2026-..._lunch-tail", "bin": true,
      "label": "Lunch tail (cut)",
      "segments": [ {"src": "C", "start": 719, "end": 9999} ] }
  ]
}
```

Dry-run first, then apply:

```bash
python3 .claude/skills/fix-transcript-cuts/cut_tool.py apply /tmp/plan.json --dry-run
python3 .claude/skills/fix-transcript-cuts/cut_tool.py apply /tmp/plan.json
```

`apply` backs up every referenced folder to
`transcripts/.backups/<date>_<label>/`, rebuilds each dest (copying +
renumbering slides, regenerating txt/html/metadata, re-locking), routes `bin`
dests under `transcripts/bin/`, and finally verifies that every slide reference
resolves.

### 4. Verify

Re-run `analyze` and confirm each talk's head is the speaker's first words and
its tail is the applause/Q&A end. Sanity-check that total text + slide counts
across the affected folders still match the originals (nothing lost — moved or
binned, never dropped). The backup under `.backups/` is the safety net.

### 5. Refresh summaries

`summary.json`, `summary.md` and `questions.json` are LLM-generated, not
regenerated by this tool — `apply` carries the dest's existing copies across the
rebuild so they're never dropped, but they describe the *old* boundaries. After
a re-cut, regenerate them from the corrected transcripts:

```bash
python3 backfill_summaries.py --force <changed-folder> [<changed-folder> ...]
```

(Needs the summariser LLM up at `ALAN_LLM_URL`. Newly split-out / binned folders
have no summary at all until you run this. The backfill also clears stale
question votes.)

## Tips

- To move a folder wholesale into `bin/` (e.g. a coffee-break/test capture), it's
  simplest to `chmod -R u+w` it and `mv` it under `transcripts/bin/` with a
  descriptive suffix — no rebuild needed.
- For a speaker whose name the STT garbled, leave the slug off (timestamp-only
  folder) or confirm the spelling with the user before naming.
- Use large `end` bounds (e.g. `9999`) to mean "to the end" — Python slicing
  clamps safely.
