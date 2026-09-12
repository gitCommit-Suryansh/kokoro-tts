# Deploying Kokoro TTS on srv1159835

Tested end-to-end in a sandbox already (model install, generation, FastAPI wrapper,
both endpoints) — this is the same code, just pointed at your VPS.

## 1. Copy this folder to the server

```bash
scp -r kokoro-deploy/ user@72.61.229.141:/opt/dockerapps/kokoro-tts
```

## 2. Build and start

```bash
ssh user@72.61.229.141
cd /opt/dockerapps/kokoro-tts
docker compose up -d --build
```

First build downloads the ~340MB of model weights, so give it a minute or two
depending on the VPS's outbound bandwidth.

## 3. Sanity check

```bash
curl http://127.0.0.1:8030/health
# {"status":"ok","model_loaded":true}
```

## 4. Listen-test both languages from the server itself

```bash
curl -s -X POST http://127.0.0.1:8030/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"Thank you for calling I Digital Studies.","voice":"af_heart","lang":"en-us"}' \
  -o test_en.wav

curl -s -X POST http://127.0.0.1:8030/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"नमस्ते, आई डिजिटल स्टूडियोज में आपका स्वागत है।","voice":"hf_alpha","lang":"hi"}' \
  -o test_hi.wav

scp user@72.61.229.141:/opt/dockerapps/kokoro-tts/test_*.wav .
```

## 5. Timing / RTF check on the actual VPS hardware

The sandbox test above ran on a single CPU core and still landed at RTF ~0.6-0.9
(faster than real-time). Your VPS has 4 vCPUs, so I'd expect this to come down
further — but confirm it, don't assume:

```bash
curl -s -X POST http://127.0.0.1:8030/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"This is a longer sentence to get a more reliable timing measurement for the real time factor calculation on this box.","voice":"af_heart","lang":"en-us"}' \
  -D - -o /dev/null | grep -i x-
```

Check the `X-Generation-Time` vs `X-Audio-Duration` headers — that ratio is your RTF.

## 6. Wire it into the voice agent

- **Non-streaming**: `POST /synthesize` — returns a complete WAV. Fine for testing,
  adds full-utterance latency in production.
- **Streaming**: `POST /synthesize/stream` — returns raw 16-bit PCM, mono, 24kHz,
  chunked sentence-by-sentence as it's generated (Kokoro synthesizes the next chunk
  while the current one is being sent — no need to wait for the whole reply).
  Point your telephony leg at this the same way you'd have piped Piper's
  `--output_raw` output.

## Voices reference

| Code | Language | Gender |
|---|---|---|
| `af_heart`, `af_bella` | English (US) | Female |
| `am_michael`, `am_adam` | English (US) | Male |
| `bf_emma` | English (UK) | Female |
| `hf_alpha`, `hf_beta` | Hindi | Female |
| `hm_omega`, `hm_psi` | Hindi | Male |

Set `lang` to `"en-us"` / `"en-gb"` / `"hi"` to match the voice you pick — the
phonemizer needs the correct language code independent of the voice name.

## Known issue

The int8-quantized model (`kokoro-v1.0.int8.onnx`) returned empty audio in testing
with `kokoro-onnx==0.6.1` — don't use it yet. `fp32` (default here) and `fp16` both
work correctly; fp16 is ~15-20% faster with no audible quality loss in testing, so
it's worth trying if you want more headroom once the LLM/RAG hops are also
running on this box.
