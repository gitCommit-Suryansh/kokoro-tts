"""
Kokoro TTS microservice.

Two endpoints:
  POST /synthesize         -> returns a complete WAV file (simple, good for testing)
  POST /synthesize/stream  -> streams raw 16-bit PCM (24kHz, mono), one sentence at a
                               time, so the caller hears the first sentence while the
                               rest is still being generated (low time-to-first-audio)

Model loads once at startup and stays resident in memory - no per-request load cost.
"""
import asyncio
import io
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from kokoro_onnx import Kokoro

MODEL_PATH = os.getenv("KOKORO_MODEL_PATH", "/models/kokoro-v1.0.onnx")
VOICES_PATH = os.getenv("KOKORO_VOICES_PATH", "/models/voices-v1.0.bin")

# Streaming: split on sentence ends (. ! ? and Hindi danda ।)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+")
CLAUSE_SPLIT = re.compile(r"(?<=[,;:])\s+")
MAX_CHUNK_CHARS = 200          # very long sentences get broken at commas
PAUSE_SECONDS = 0.12           # small natural gap between streamed sentences

kokoro: Optional[Kokoro] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global kokoro
    t0 = time.time()
    kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
    print(f"Kokoro model loaded in {time.time() - t0:.2f}s from {MODEL_PATH}")
    yield


app = FastAPI(title="Kokoro TTS Service", lifespan=lifespan)


class SynthesizeRequest(BaseModel):
    text: str
    voice: str = "af_heart"   # af_heart/af_bella = EN female, am_michael = EN male
                                # hf_alpha/hf_beta = HI female, hm_omega/hm_psi = HI male
    lang: str = "en-us"        # "en-us", "en-gb", or "hi" for Hindi
    speed: float = 1.0


def split_text(text: str) -> list[str]:
    """Split text into sentence-sized chunks; break very long sentences at commas."""
    parts: list[str] = []
    for sentence in SENTENCE_SPLIT.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= MAX_CHUNK_CHARS:
            parts.append(sentence)
            continue
        buf = ""
        for piece in CLAUSE_SPLIT.split(sentence):
            if buf and len(buf) + len(piece) + 1 > MAX_CHUNK_CHARS:
                parts.append(buf)
                buf = piece
            else:
                buf = f"{buf} {piece}".strip()
        if buf:
            parts.append(buf)
    return parts


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": kokoro is not None}


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    """Non-streaming: returns one complete WAV file. Good for quick curl/Postman tests."""
    if kokoro is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")

    t0 = time.time()
    samples, sample_rate = kokoro.create(
        req.text, voice=req.voice, speed=req.speed, lang=req.lang
    )
    gen_time = time.time() - t0

    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={
            "X-Generation-Time": f"{gen_time:.3f}",
            "X-Audio-Duration": f"{len(samples) / sample_rate:.3f}",
        },
    )


@app.post("/synthesize/stream")
async def synthesize_stream(req: SynthesizeRequest):
    """
    Streaming: generates and sends one sentence at a time.
    Output format: raw PCM, 16-bit signed little-endian, mono, 24000 Hz.
    """
    if kokoro is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")

    chunks = split_text(req.text)
    if not chunks:
        raise HTTPException(status_code=400, detail="text is empty")

    async def pcm_chunks():
        for i, chunk in enumerate(chunks):
            samples, sample_rate = await asyncio.to_thread(
                kokoro.create, chunk, voice=req.voice, speed=req.speed, lang=req.lang
            )
            pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
            yield pcm16.tobytes()
            if i < len(chunks) - 1:
                yield np.zeros(int(PAUSE_SECONDS * sample_rate), dtype=np.int16).tobytes()

    return StreamingResponse(
        pcm_chunks(),
        media_type="application/octet-stream",
        headers={"X-Accel-Buffering": "no"},
    )