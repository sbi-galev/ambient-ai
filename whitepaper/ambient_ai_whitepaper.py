#!/usr/bin/env python3
"""Generate the "Ambient AI" white paper (LaTeX) describing this system itself.

Unlike whitepaper.py — which asks the *local* model (alan) to summarise the
conference so the meeting's content never leaves the building — this paper is
ABOUT the tool, not about any talk. It carries no transcript, abstract or slide
content, so there is no privacy constraint on who writes it. Its prose was
authored directly by Claude (Anthropic, "claude max"), which is why it can
describe the implementation, hardware and models with outside knowledge that the
on-box model does not have.

The prose below is fixed (hand-written). The script only fills in a handful of
*live* facts so the paper stays accurate as the archive grows — the number of
talks and days actually captured, the date range, and the exact model / config
constants — by importing transcript_server as the single source of truth. It
then writes ambient_ai_whitepaper.tex and (with --pdf) compiles it.

Usage:
    python3 ambient_ai_whitepaper.py [--out FILE.tex] [--pdf]
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
# transcript_server.py lives in the repo root (one level up); make it importable
# so we can read the live model/config constants and talk counts from it.
sys.path.insert(0, str(ROOT.parent))
import transcript_server as ts


# ── live facts (single source of truth: the running system's own constants) ───

def _latex_escape(s: str) -> str:
    repl = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
            "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}"}
    return "".join(repl.get(c, c) for c in str(s))


def gather_facts() -> dict:
    """Pull the numbers/names the paper cites from the live system so it never
    drifts from reality. Falls back gracefully if the archive is empty."""
    try:
        talks = [t for t in ts._load_talks() if t.get("summary")]
        days = ts._group_by_day(ts._load_talks())
    except Exception:
        talks, days = [], []
    n_talks = len(talks) if talks else 0
    n_days = len(days) if days else 0
    if days:
        date_range = f"{days[0][0]} to {days[-1][0]}"
    else:
        date_range = "the meeting"
    return {
        "CONF_SHORT": _latex_escape(ts.CONF_SHORT),
        "CONF_FULL": _latex_escape(ts.CONF_FULL),
        "ASSISTANT": _latex_escape(ts.ASSISTANT_NAME),
        "NTALKS": str(n_talks) if n_talks else "the",
        "NDAYS": str(n_days) if n_days else "several",
        "DATERANGE": _latex_escape(date_range),
        "STT_MODEL": _latex_escape(ts.MODEL_NAME),
        "SAMPLE_RATE": f"{ts.SAMPLE_RATE // 1000}~kHz",
        "DEVICE": _latex_escape(ts.DEVICE),
        "LLM_MODEL": _latex_escape(ts.LLM_MODEL),
        "LLM_URL": _latex_escape(ts.LLM_URL),
        "SAVE_OFFSET": str(ts.SAVE_OFFSET_SECONDS),
        "MAX_SLIDES": str(ts.SUMMARY_MAX_SLIDES),
        "FLUSH": str(ts.LIVE_FLUSH_SECONDS),
        "PORT": str(ts.PORT),
        "GENERATED": _latex_escape(datetime.now().strftime("%-d %B %Y")),
    }


# ── the paper (hand-written by Claude; live facts injected at @@TOKENS@@) ──────

TEX = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{parskip}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{hyperref}
\hypersetup{hidelinks}
\titlespacing*{\section}{0pt}{1.2em}{0.4em}
\titlespacing*{\subsection}{0pt}{0.8em}{0.3em}

\title{Ambient AI for Scientific Meetings:\\
       A Local, Privacy-Preserving Transcription, Summarisation\\
       and Synthesis Pipeline}
\author{Claude (Anthropic) --- written via Claude Code}
\date{@@GENERATED@@}

\begin{document}
\maketitle

\begin{abstract}
This paper describes an ``ambient AI'' system built to observe a scientific
conference unobtrusively and turn it into a durable, searchable record. A single
machine listens to the room, transcribes the spoken talks, captures the
projected slides, and --- without any human writing a word --- produces per-talk
summaries, audience questions, per-day overviews, a conference-wide topic graph,
and finally a multi-page summary white paper. Every model runs on local
hardware: no transcript, slide or abstract is ever sent to an external service.
The system was deployed at the @@CONF_FULL@@
(@@CONF_SHORT@@) meeting, where it processed @@NTALKS@@ talks across @@NDAYS@@ days
(@@DATERANGE@@). We describe the architecture, the implementation choices that
made it robust in a live setting, the hardware and models, and our reflections on
what worked and what did not. This document is deliberately the one artifact the
system does \emph{not} write about itself: because it contains no meeting content,
it was authored by a frontier model (Claude) rather than the on-box model that
writes everything else.
\end{abstract}

\section{Motivation}
Conferences produce an enormous amount of transient knowledge that is almost
entirely lost the moment a session ends. Slides may be shared; the spoken
narrative around them --- the caveats, the asides, the questions --- rarely is.
``Ambient AI'' is the idea that an unattended system can sit in the room and
quietly preserve that narrative, then distil it into something a participant (or
someone who could not attend) can actually use.

Two constraints shaped the design. First, the system had to be \textbf{ambient}:
low-friction, requiring no per-speaker setup and no behaviour change from
presenters. An operator clicks a single ``End talk'' button between speakers; the
rest is automatic. Second, and more importantly, it had to be
\textbf{privacy-preserving}. Unpublished results, work in progress and informal
remarks are exactly the material a conference contains, and exactly the material
that must not be uploaded to a third-party AI service. The hard rule of this
project is therefore that the meeting's content never leaves the local network:
every model --- speech recognition, vision-language summarisation, synthesis ---
runs on-premises.

\section{System Overview}
The pipeline is a single long-running Python service (\texttt{transcript\_server.py})
that exposes a small HTTP API and a live web view, plus a set of offline tools
that operate on what it has saved. The flow for one talk is:

\begin{enumerate}[leftmargin=1.4em,itemsep=0.15em]
  \item \textbf{Capture.} Audio chunks are POSTed to the server and run through a
        local speech-to-text model; projected slides are POSTed as JPEG frames
        and de-duplicated so only genuine slide \emph{changes} are kept. Both
        stream to any open browser over Server-Sent Events, giving a live,
        auto-scrolling transcript beside the current slide.
  \item \textbf{Save.} Between speakers the operator clicks ``End talk \& Save.''
        The server bundles that talk's ordered text and slides into a dedicated
        folder, writes a plain-text transcript and an HTML archive page, and then
        \emph{finalises the folder read-only} so the record cannot later be
        mutated in place.
  \item \textbf{Summarise.} A background worker sends the transcript plus an
        evenly-sampled set of slides to the local vision-language model, which
        returns a structured summary --- title, speaker, abstract, key points,
        topic tags --- and three substantive questions an audience member might
        ask. The authoritative title and speaker are corrected against the
        conference's official schedule abstract.
  \item \textbf{Synthesise.} As talks accumulate, the system regenerates a
        per-day editorial overview for each day and a conference-wide topic
        synthesis that clusters every talk's tags into the meeting's main
        threads and links each thread back to the talks that raised it.
  \item \textbf{Report.} A separate generator (\texttt{whitepaper.py}) assembles
        all of the above into a single context and asks the local model to write
        a multi-page summary white paper, with depth allocated in proportion to
        how much each topic was discussed.
\end{enumerate}

The public web view shows only summaries, slides and questions; the verbatim
transcripts are gated behind an operator token. Audience members can up- or
down-vote the machine-generated questions, surfacing the ones the room most wants
asked.

\section{Implementation Details}
Several design choices were driven by the realities of a live room rather than by
the happy path.

\paragraph{Boundary leakage and the save offset.}
The ``End talk'' click is invariably a beat late: by the time the chair has
thanked the speaker, the next speaker's opening words have already been
transcribed into the current talk. The server therefore holds back the most
recent @@SAVE_OFFSET@@~seconds of material at save time, leaving it in the live
buffer so it bundles with the \emph{next} talk instead. This is a fixed
approximation --- it cannot fix an \emph{early} click or a missed save --- so a
companion offline tool lets an editor re-cut boundaries after the fact by moving
slices of one talk's record into its neighbour (slides follow their text
automatically), routing breaks and lunch gaps into a hidden bin rather than
deleting anything. Re-cut talks are then re-summarised from the corrected
transcripts.

\paragraph{Crash safety.}
The live transcript lives in memory and is only written to disk on an explicit
save, so a crash mid-talk would lose the in-progress speaker. The server
periodically snapshots the in-memory history to a hidden file (every
@@FLUSH@@~seconds, coalescing bursts into one write) and restores it on startup,
so an interrupted talk resumes rather than vanishing.

\paragraph{Immutability of the record.}
Once a talk is summarised, its folder is made read-only (files \texttt{0444},
directories \texttt{0555}). There is deliberately no HTTP endpoint that deletes
or edits a saved talk: the archive is append-only by construction, and offline
tools that legitimately need to re-cut a talk must explicitly unlock, rewrite and
re-lock it, taking a backup first.

\paragraph{Grounding against the official schedule.}
Automatic speech recognition garbles proper nouns --- names, instruments,
method acronyms --- exactly the tokens a reader most needs correct. The
summariser is given the talk's official schedule abstract as an authoritative
reference for spelling and framing, but is instructed to summarise what the talk
\emph{actually delivered} from the transcript and slides rather than copy the
abstract. The authoritative title and speaker name then overwrite the
ASR-derived guesses.

\paragraph{Buying ``thinking time'' from a non-reasoning model.}
The local language model is not a native reasoning model, so the white-paper
generator spends a separate planning pass first: the model drafts an editorial
plan --- a complete topic inventory, weighted by how many talks engaged each
topic, with proportional space recommendations --- before it writes any prose.
This test-time-compute step is how comprehensive breadth (mention every topic)
and proportional depth (more space for more-discussed topics) are achieved
without a reasoning model.

\section{Hardware and Models}
The entire stack runs on a single GPU workstation; the only network traffic is
between the browser and the local server.

\begin{itemize}[leftmargin=1.4em,itemsep=0.2em]
  \item \textbf{Speech recognition.} NVIDIA NeMo's
        \texttt{@@STT_MODEL@@} --- a FastConformer transducer (the
        ``XXL'' configuration) --- run on the GPU (\texttt{@@DEVICE@@}) at
        @@SAMPLE_RATE@@ audio. It transcribes streamed audio chunks with low
        enough latency to drive the live view.
  \item \textbf{Vision-language summarisation and synthesis (``@@ASSISTANT@@'').} A
        \texttt{@@LLM_MODEL@@} instruction-tuned model, served locally through
        SGLang behind an OpenAI-compatible endpoint (\texttt{@@LLM_URL@@}). It is
        multimodal: each summary request carries up to @@MAX_SLIDES@@ slide
        images alongside the transcript, so the summary reflects what was
        \emph{shown}, not only what was said. The same endpoint produces the
        per-day overviews, the topic synthesis and the white paper.
  \item \textbf{Serving.} A small standard-library HTTP server (no web framework)
        on port~@@PORT@@, using Server-Sent Events for the live stream and a
        thread pool of background workers for summarisation so inference never
        blocks capture.
\end{itemize}

Choosing on-premises open models over a hosted API was the privacy requirement
made concrete: a 31B-parameter vision-language model is large enough to write
genuinely useful, well-grounded summaries, yet small enough to serve in real
time on a single workstation. That trade --- capability against locality --- is
the central engineering decision of the project.

\section{Reflections: How It Has Worked}
After a full multi-day deployment, our honest assessment is that the approach
works, with caveats that are themselves instructive.

\paragraph{What worked well.}
The ambient model of capture --- one button between talks, nothing asked of
speakers --- proved low enough friction to run for an entire conference. Grounding
summaries on slides as well as speech noticeably improved their faithfulness: the
model recovers structure and terminology from a slide that the ASR mangled in the
audio. Injecting the official abstracts all but eliminated the most jarring
class of error (wrong names and method acronyms). And the local-only constraint,
far from being merely a compliance checkbox, turned out to be a feature
participants cared about: the system can be pointed at unpublished work precisely
because nothing is uploaded.

\paragraph{What was hard.}
The dominant practical problem was \textbf{talk-boundary leakage}. Mistimed save
clicks, and occasionally a missed save that bundled several talks plus a lunch
break into one ``mega-folder,'' meant the raw cut never quite matched the true
boundaries. The fixed save offset reduces this but cannot eliminate it, so a
human-in-the-loop re-cut step remained necessary; building a tool that makes that
re-cut safe (backed up, count-conserving, lossless) mattered more than we
expected. The second recurring issue was that the language model is not a
reasoning model, which is why proportional, comprehensive coverage had to be
engineered through an explicit planning pass rather than assumed. Finally,
automatic transcription of dense technical speech remains imperfect; the system's
faithfulness comes less from perfect transcripts than from triangulating
transcript, slides and the official abstract.

\paragraph{What it changes.}
The result is that a talk a participant missed is no longer simply gone. Within
minutes of a speaker finishing, there is a grounded summary, a set of questions
worth asking, and --- by the end of the meeting --- a coherent narrative of how
the whole conference fit together, all produced without anyone taking notes and
without any of it leaving the room.

\section{Limitations and Future Work}
The system is an archivist, not an oracle: every summary is only as good as the
transcript, slides and abstract it was given, and it is explicitly constrained
not to invent results. Boundary detection is still partly manual; an obvious next
step is to detect chair hand-offs and speaker self-introductions automatically
and propose cuts, rather than relying on a well-timed click. Speaker
diarisation, richer cross-talk linking, and a retrieval interface over the whole
archive are natural extensions. Throughout, the constraint that keeps the design
honest is unchanged: whatever is added must run locally, so that the meeting's
content stays where the meeting happened.

\section{Conclusion}
Ambient AI for a scientific meeting is feasible today on a single machine and
without surrendering the room's privacy. The combination of a real-time local
transcriber, a multimodal local summariser grounded against the official
programme, an append-only immutable archive, and a synthesis stage that scales
from one talk to a whole-conference white paper produced a useful record of
@@CONF_SHORT@@ with minimal human effort. The remaining rough edges --- chiefly talk
boundaries and the limits of automatic transcription --- are tractable, and none
of them require giving up the property that matters most: it all runs at home.

\vspace{1em}
\noindent\rule{\linewidth}{0.4pt}\\
{\small This white paper describes the system but contains no meeting content,
so unlike every other artifact the system produces it was written by Claude
(Anthropic) rather than by the local on-box model. The conference summaries and
the conference white paper are, by design, written entirely on local hardware.}

\end{document}
"""


def render(facts: dict) -> str:
    header = (
        "% Ambient AI white paper — generated by ambient_ai_whitepaper.py\n"
        "% Prose authored by Claude (Anthropic); live facts injected from "
        "transcript_server.\n"
        f"% Generated {datetime.now().isoformat(timespec='seconds')}\n")
    body = TEX
    for k, v in facts.items():
        body = body.replace(f"@@{k}@@", v)
    return header + body


def compile_pdf(tex_path: Path) -> bool:
    workdir = tex_path.parent
    have = lambda c: subprocess.run(["which", c], capture_output=True).returncode == 0
    try:
        if have("latexmk"):
            subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode",
                            "-halt-on-error", tex_path.name],
                           cwd=workdir, capture_output=True, text=True, timeout=300)
        elif have("pdflatex"):
            for _ in range(2):
                subprocess.run(["pdflatex", "-interaction=nonstopmode",
                                "-halt-on-error", tex_path.name],
                               cwd=workdir, capture_output=True, text=True, timeout=300)
        else:
            print("  ! no latexmk/pdflatex on PATH — skipping PDF compile", file=sys.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("  ! LaTeX compile timed out", file=sys.stderr)
        return False
    return tex_path.with_suffix(".pdf").exists()


def main():
    ap = argparse.ArgumentParser(
        description="Generate the Ambient AI white paper (authored by Claude; "
                    "live facts from the running system).")
    ap.add_argument("--out", default=str(ROOT / "ambient_ai_whitepaper.tex"),
                    help="output .tex path (defaults beside this script)")
    ap.add_argument("--pdf", action="store_true", help="also compile to PDF")
    args = ap.parse_args()

    facts = gather_facts()
    print(f"Live facts: {facts['NTALKS']} talks, {facts['NDAYS']} days "
          f"({facts['DATERANGE']}); STT={facts['STT_MODEL']}; LLM={facts['LLM_MODEL']}")
    out = Path(args.out)
    out.write_text(render(facts))
    print(f"  ✓ wrote {out}  ({out.stat().st_size} bytes)")

    if args.pdf:
        print("Compiling PDF ...")
        if compile_pdf(out):
            print(f"  ✓ {out.with_suffix('.pdf')}")
        else:
            print(f"  ! compile failed; see {out.with_suffix('.log')}", file=sys.stderr)


if __name__ == "__main__":
    main()
