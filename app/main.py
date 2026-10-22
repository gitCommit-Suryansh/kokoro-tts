"""
Kokoro TTS microservice.

Two endpoints:
  POST /synthesize         -> returns a complete WAV file (simple, good for testing)
  POST /synthesize/stream  -> streams raw 16-bit PCM (24kHz, mono), one sentence at a
                               time, so the caller hears the first sentence while the
                               rest is still being generated (low time-to-first-audio)

Model loads once at startup and stays resident in memory - no per-request load cost.

CPU tuning:
  - ONNX Runtime threads are pinned to TTS_THREADS (match the container's CPU cap)
    and thread spinning is disabled, so idle threads don't burn CPU.
  - Generation runs one sentence at a time behind a FIFO lock. Concurrent calls take
    turns sentence-by-sentence, each getting the full CPU, instead of fighting over it.
"""
import asyncio
import io
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import onnxruntime as ort
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from kokoro_onnx import Kokoro

MODEL_PATH = os.getenv("KOKORO_MODEL_PATH", "/models/kokoro-v1.0.onnx")
VOICES_PATH = os.getenv("KOKORO_VOICES_PATH", "/models/voices-v1.0.bin")
TTS_THREADS = int(os.getenv("TTS_THREADS", "3"))   # keep equal to `cpus` in docker-compose

# Streaming: split on sentence ends (. ! ? and Hindi danda ।)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+")
CLAUSE_SPLIT = re.compile(r"(?<=[,;:])\s+")
MAX_CHUNK_CHARS = 200          # very long sentences get broken at commas
PAUSE_SECONDS = 0.12           # small natural gap between streamed sentences

kokoro: Optional[Kokoro] = None
tts_lock = asyncio.Lock()      # FIFO: concurrent requests take turns per sentence


def build_kokoro() -> Kokoro:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = TTS_THREADS
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session = ort.InferenceSession(
        MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"]
    )
    return Kokoro.from_session(session, VOICES_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global kokoro
    t0 = time.time()
    kokoro = build_kokoro()
    print(
        f"Kokoro model loaded in {time.time() - t0:.2f}s from {MODEL_PATH} "
        f"(threads={TTS_THREADS})"
    )
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


async def generate(text: str, voice: str, speed: float, lang: str):
    """Run one generation with the full CPU, waiting our turn if another is running."""
    async with tts_lock:
        return await asyncio.to_thread(
            kokoro.create, text, voice=voice, speed=speed, lang=lang
        )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": kokoro is not None}


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    """Non-streaming: returns one complete WAV file. Good for quick curl/Postman tests."""
    if kokoro is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")

    t0 = time.time()
    samples, sample_rate = await generate(req.text, req.voice, req.speed, req.lang)
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
            samples, sample_rate = await generate(chunk, req.voice, req.speed, req.lang)
            pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
            yield pcm16.tobytes()
            if i < len(chunks) - 1:
                yield np.zeros(int(PAUSE_SECONDS * sample_rate), dtype=np.int16).tobytes()

    return StreamingResponse(
        pcm_chunks(),
        media_type="applicaton/octet-stream",
        headers={"X-Accel-Buffering": "no"},
    )