"""Test stt_en_fastconformer_transducer_xxlarge on a WAV chunk."""
import time, sys, wave, io
import numpy as np

MODEL = "stt_en_fastconformer_transducer_xxlarge"

import nemo.collections.asr as nemo_asr
print(f"Loading {MODEL}...", flush=True)
t0 = time.time()
model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL, map_location="cuda:0")
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

wav_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/turing_demo.wav"
t1 = time.time()
result = model.transcribe([wav_path], batch_size=1, verbose=False)
elapsed = time.time() - t1

with wave.open(wav_path) as w:
    duration = w.getnframes() / w.getframerate()

text = result[0].text if hasattr(result[0], "text") else str(result[0])
print(f"Text: {text}")
print(f"Audio: {duration:.1f}s  Inference: {elapsed:.2f}s  RTF: {elapsed/duration:.3f}x")
