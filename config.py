"""Central configuration for the ambient-AI conference toolkit.

Single source of truth for every site- and hardware-specific value, shared by
the GPU server (transcript_server.py) and the laptop capture client
(live_transcribe.py). Values resolve in this order, first match wins:

    1. environment variable   — per-invocation override (names noted below)
    2. config.toml            — the file you edit for your event
    3. built-in default       — the SBI4GALEV reference values

Only the standard library is used here, so this module is safe to import on the
laptop client as well as on the GPU server. Point CONFIG_FILE at another path to
load a different config.toml.
"""
import os
import tomllib
from pathlib import Path

CONFIG_PATH = Path(os.environ.get("CONFIG_FILE", Path(__file__).parent / "config.toml"))

_data = {}
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "rb") as _f:
        _data = tomllib.load(_f)


def _cfg(section, key, default):
    return _data.get(section, {}).get(key, default)


def _s(env, section, key, default):
    v = os.environ.get(env)
    return v if v is not None else str(_cfg(section, key, default))


def _i(env, section, key, default):
    v = os.environ.get(env)
    return int(v) if v is not None else int(_cfg(section, key, default))


def _b(env, section, key, default):
    v = os.environ.get(env)
    if v is not None:
        return v.strip().lower() not in ("0", "false", "no", "off", "")
    return bool(_cfg(section, key, default))


# ── Conference identity & assistant persona ───────────────────────────────────
CONF_SHORT     = _s("CONF_SHORT", "conference", "short_name", "SBI4GALEV")
CONF_FULL      = _s("CONF_FULL", "conference", "full_name",
                    "Simulation-Based Inference for Galaxy Evolution")
CONF_YEAR      = _s("CONF_YEAR", "conference", "year", "2026")
ASSISTANT_NAME = _s("ASSISTANT_NAME", "conference", "assistant_name", "alan")

# ── Server ────────────────────────────────────────────────────────────────────
HOST       = _s("TRANSCRIPT_HOST", "server", "host", "0.0.0.0")
PORT       = _i("TRANSCRIPT_PORT", "server", "port", 7103)
TOKEN      = _s("TRANSCRIPT_TOKEN", "server", "token", "sbi4galev")
PUBLIC_URL = _s("TRANSCRIPT_URL", "server", "public_url", "http://127.0.0.1:7103")

# ── Speech-to-text ────────────────────────────────────────────────────────────
STT_MODEL   = _s("STT_MODEL", "stt", "model", "stt_en_fastconformer_transducer_xxlarge")
STT_DEVICE  = _s("STT_DEVICE", "stt", "device", "cuda:0")
SAMPLE_RATE = _i("STT_SAMPLE_RATE", "stt", "sample_rate", 16000)

# ── Local LLM (OpenAI-compatible, multimodal) ─────────────────────────────────
# The ALAN_LLM_* override names are kept for backward compatibility.
LLM_URL            = _s("ALAN_LLM_URL", "llm", "url",
                        "http://127.0.0.1:30000/v1/chat/completions")
LLM_MODEL          = _s("ALAN_LLM_MODEL", "llm", "model", "google/gemma-4-31b-it")
SUMMARIES_ENABLED  = _b("SUMMARIES", "llm", "enabled", True)
SUMMARY_MAX_SLIDES = _i("SUMMARY_MAX_SLIDES", "llm", "max_slides", 8)

# ── Tuning ────────────────────────────────────────────────────────────────────
SAVE_OFFSET_SECONDS = _i("SAVE_OFFSET_SECONDS", "tuning", "save_offset_seconds", 30)
LIVE_FLUSH_SECONDS  = _i("LIVE_FLUSH_SECONDS", "tuning", "live_flush_seconds", 10)
TOPICS_MAX          = _i("TOPICS_MAX", "tuning", "topics_max", 14)
