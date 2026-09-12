"""
Kokoro TTS microservice.

Two endpoints:
  POST /synthesize         -> returns a complete WAV file (simple, good for testing)
  POST /synthesize/stream  -> streams raw 16-bit PCM (24kHz, mono) chunks as they're
                               generated, sentence by sentence, for low-latency IVR use

Model loads once at startup and stays resident in memory - no per-request load cost.
"""
import io
import os
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
    Streaming: yields raw 16-bit PCM mono audio chunks as each sentence/clause
    finishes generating, instead of waiting for the whole utterance. This is what
    you want wired into the live call leg for low time-to-first-audio.

    Output format: raw PCM, 16-bit signed little-endian, mono, 24000 Hz.
    Point your telephony layer at that directly (same idea as Piper's --output_raw).
    """
    if kokoro is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")

    async def pcm_chunks():
        async for samples, sample_rate in kokoro.create_stream(
            req.text, voice=req.voice, speed=req.speed, lang=req.lang
        ):
            pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
            yield pcm16.tobytes()

    return StreamingResponse(pcm_chunks(), media_type="application/octet-stream")
