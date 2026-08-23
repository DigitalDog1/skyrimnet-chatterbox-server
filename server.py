"""SkyrimNet Chatterbox TTS server.

FastAPI server that emulates the Gradio queue protocol used by
SkyrimNet's `GradioTTSInterface` (ChatterboxInterface, HiggsInterface,
PocketTTSInterface, ...) so the Chatterbox Multilingual model can serve
as a local TTS engine for the Skyrim mod SkyrimNet.

Endpoints
---------
- POST  /gradio_api/upload
- POST  /gradio_api/call/generate_audio
- GET   /gradio_api/call/generate_audio/{eid}        (SSE)
- GET   /gradio_api/file=<abs_path>                  (audio/wav)
- GET   /health

Run
---
    pip install -r requirements.txt
    python server.py          # http://127.0.0.1:7861
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response

# --- configuration ---------------------------------------------------------
GRADIO_HOST: str = os.environ.get("GRADIO_HOST", "127.0.0.1")
GRADIO_PORT: int = int(os.environ.get("GRADIO_PORT", "7861"))
UPLOAD_DIR: Path = Path(os.environ.get(
    "GRADIO_UPLOAD_DIR",
    r"C:\Users\DigitalJesus\AppData\Local\Temp\gradio",
))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Default voice sample used when SkyrimNet sends no reference
# (e.g. placeholder "empty_100ms.wav").  Set to your own sample if you
# want a different default timbre.
DEFAULT_VOICE_SAMPLE: str = os.environ.get(
    "DEFAULT_VOICE_SAMPLE",
    str(Path(__file__).parent / "voice-sample.mp3"),
)

# Where to load the model from.  If unset, huggingface_hub will pull
# it to the default cache directory.
MODEL_REPO: str = os.environ.get("MODEL_REPO", "ResembleAI/chatterbox")
DEVICE: str = os.environ.get("DEVICE",
                             "cuda" if torch.cuda.is_available() else "cpu")

# Languages accepted by the multilingual model.  Anything else falls
# back to English, which is the safest default for a quick first run.
LANG_CODES = {"en", "ru", "zh", "de", "fr", "es", "it", "ja", "ko",
              "pt", "pl", "tr", "uk", "ar", "nl", "sv", "fi", "no",
              "he", "hi", "ms", "el", "da", "sw"}

# --- emotion presets -------------------------------------------------------
# SkyrimNet appends bracketed tags to dialogue text, e.g.
#   "Run, you fool! [shout]"
# Chatterbox is a pure acoustic model and would pronounce the tag
# literally.  Strip the tag and substitute (exaggeration, cfg_weight)
# values that produce the matching delivery.
EMOTION_PRESETS: dict[str, tuple[float, float]] = {
    # name(s)                : (exag, cfg)
    "whisper":               (0.25, 0.70),  # quiet, careful
    "whispering":            (0.25, 0.70),
    "quiet":                 (0.30, 0.65),
    "sad":                   (0.40, 0.70),  # low energy, slow
    "somber":                (0.40, 0.70),
    "fear":                  (0.70, 0.40),  # nervous, shaky
    "scared":                (0.70, 0.40),
    "terrified":             (0.85, 0.35),
    "angry":                 (0.75, 0.35),  # harsh, intense
    "dramatic":              (0.75, 0.35),  # theatrical, broad
    "shout":                 (0.85, 0.30),  # loud, free
    "yelling":               (0.85, 0.30),
    "scream":                (0.95, 0.25),
    "happy":                 (0.60, 0.45),  # light, upbeat
    "cheerful":              (0.60, 0.45),
    "laugh":                 (0.65, 0.40),
    "laughing":              (0.65, 0.40),
    "sarcastic":             (0.55, 0.55),  # controlled, dry
    "neutral":               (0.50, 0.50),
}

# --- model loading (single-shot) -------------------------------------------
_model: Any = None
_model_lock = threading.Lock()
_model_loaded = False
_model_error: str | None = None


def _load_model() -> Any:
    """Pull and load the multilingual Chatterbox model from HF Hub."""
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    print(f"[server] loading {MODEL_REPO} on {DEVICE} ...", flush=True)
    m = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
    print(f"[server] model loaded, sample rate = {m.sr}", flush=True)
    return m


def _ensure_model() -> Any:
    global _model, _model_loaded, _model_error
    if _model_loaded:
        return _model
    if _model_error:
        raise RuntimeError(f"model load failed: {_model_error}")
    with _model_lock:
        if not _model_loaded:
            try:
                _model = _load_model()
                _model_loaded = True
            except Exception as e:
                _model_error = str(e)
                print(f"[server] MODEL LOAD FAILED: {e}", flush=True)
                raise
    return _model


# --- job state -------------------------------------------------------------
_jobs: dict[str, dict[str, Any]] = {}
_job_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")


def _sniff_args(data: list) -> tuple[str, str, str | None, float, float, float]:
    """Extract (text, language, voice_path, exaggeration, cfg_weight,
    temperature) from SkyrimNet's 30-element data array.

    The array is type-sniffed because the position of each argument does
    not match the layout of either gradio_tts_app.py published by
    Resemble AI.
    """
    texts: list[str] = []
    refs: list[str] = []
    langs: list[str] = []
    floats: list[float] = []
    for a in data:
        if a is None or isinstance(a, bool):
            continue
        if isinstance(a, dict):
            p = a.get("path")
            if p:
                refs.append(p)
            continue
        if isinstance(a, (list, tuple)):
            if a:
                v = a[0]
                if isinstance(v, dict):
                    p = v.get("path")
                    if p:
                        refs.append(p)
                else:
                    refs.append(str(v))
            continue
        if isinstance(a, float):
            floats.append(a)
            continue
        if isinstance(a, int):
            continue
        if isinstance(a, str):
            s = a.strip()
            if not s:
                continue
            if len(s) == 2 and s.lower() in LANG_CODES:
                langs.append(s.lower())
            else:
                texts.append(s)
    text = max(texts, key=len) if texts else ""
    language = langs[0] if langs else "en"
    # Pick the first reference that is NOT the empty 100ms placeholder
    # that SkyrimNet always sends as a secondary ref.
    real_ref = None
    for r in refs:
        if "empty_100ms" not in Path(r).name:
            real_ref = r
            break
    if not real_ref and refs:
        real_ref = refs[0]
    exag = next((f for f in floats if 0.0 <= f <= 2.0), 0.5)
    cfg_w = next((f for f in floats if 0.0 < f <= 1.0), 0.5)
    temp = next((f for f in floats if 0.4 <= f <= 1.6), 0.8)
    return text, language, real_ref, exag, cfg_w, temp


def _apply_emotion_tag(text: str, base_exag: float,
                       base_cfg: float) -> tuple[str, float, float]:
    """Strip a trailing `[emotion]` tag from the text and substitute
    the matching (exaggeration, cfg_weight) preset.  If no tag is
    present, return the input values unchanged.
    """
    m = re.search(r"\s*\[([a-zA-Z_]+)\]\s*\.?\s*$", text)
    if not m:
        return text, base_exag, base_cfg
    tag = m.group(1).lower()
    cleaned = text[:m.start()].rstrip()
    if tag not in EMOTION_PRESETS:
        return text, base_exag, base_cfg
    ex, cf = EMOTION_PRESETS[tag]
    # Only override when SkyrimNet's defaults are flat (0.35/0.35);
    # if the user raised them globally, keep their values.
    if base_exag <= 0.45 and base_cfg <= 0.45:
        print(f"[server] emotion tag [{tag}] -> exag={ex:.2f} cfg={cf:.2f} "
              f"(overrode {base_exag:.2f}/{base_cfg:.2f})", flush=True)
        return cleaned, ex, cf
    print(f"[server] emotion tag [{tag}] detected, kept SkyrimNet "
          f"exag={base_exag:.2f} cfg={base_cfg:.2f} (user-tuned)", flush=True)
    return cleaned, base_exag, base_cfg


def _synthesize(model: Any, text: str, language: str, ref: str | None,
                exag: float, cfg_w: float, temp: float) -> tuple[int, np.ndarray]:
    """Run the model.  Must be called from a single thread; the
    alignment state is reset before each call because
    ChatterboxMultilingualTTS caches it across invocations.
    """
    voice_path = None
    if ref and Path(ref).is_file():
        voice_path = ref
    elif Path(DEFAULT_VOICE_SAMPLE).is_file():
        voice_path = DEFAULT_VOICE_SAMPLE
    print(f"[server] synth text={text!r} lang={language} ref={voice_path} "
          f"exag={exag:.2f} cfg={cfg_w:.2f} temp={temp:.2f}", flush=True)
    # Reset alignment state.  Without this, two consecutive generations
    # can fail with a tensor size mismatch inside
    # alignment_stream_analyzer.alignment.
    try:
        asa = getattr(getattr(model, "patched_model", None),
                      "alignment_stream_analyzer", None)
        if asa is not None and hasattr(asa, "alignment"):
            asa.alignment = None
    except Exception as e:
        print(f"[server] alignment reset warning: {e}", flush=True)
    with torch.no_grad():
        wav = model.generate(
            text=text,
            audio_prompt_path=voice_path,
            exaggeration=exag,
            cfg_weight=cfg_w,
            temperature=temp,
            language_id=language,
        )
    if hasattr(wav, "detach"):
        wav = wav.detach()
    if hasattr(wav, "cpu"):
        wav = wav.cpu()
    arr = wav.numpy().squeeze().astype(np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return int(model.sr), arr


def _write_wav(path: Path, sr: int, pcm: np.ndarray) -> None:
    pcm_int16 = (pcm * 32767.0).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm_int16.tobytes())


# --- HTTP layer ------------------------------------------------------------
app = FastAPI(title="SkyrimNet Chatterbox TTS Server")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model_loaded,
        "model_error": _model_error,
        "device": DEVICE,
        "upload_dir": str(UPLOAD_DIR),
        "host": GRADIO_HOST,
        "port": GRADIO_PORT,
    }


@app.post("/upload")
@app.post("/gradio_api/upload")
async def upload(
    files: list[UploadFile] = File(default_factory=list),
    audio: UploadFile | None = File(None),
):
    """Receive a voice sample.  SkyrimNet's
    `GradioTTSInterface::UploadBufferToGradio` expects a non-empty
    array of absolute path strings.
    """
    chosen = (files[0] if files else None) or audio
    if chosen is None:
        raise HTTPException(400, "no file in 'files' or 'audio'")
    raw = await chosen.read()
    h = hashlib.sha1(raw).hexdigest()[:16]
    p = UPLOAD_DIR / f"{h}_{chosen.filename or 'upload.wav'}"
    if not p.exists():
        p.write_bytes(raw)
    print(f"[upload] {chosen.filename} -> {p} ({len(raw)} bytes)", flush=True)
    return [str(p)]


@app.post("/gradio_api/call/generate_audio")
@app.post("/call/generate_audio")
async def call_generate(request: Request):
    """Accept a generation job.  Returns an event id; the actual
    synthesis runs in a background worker and the client polls the
    matching /call/generate_audio/{eid} endpoint.
    """
    body = await request.json()
    data = body.get("data", []) if isinstance(body, dict) else []
    eid = hashlib.sha1(repr(data).encode() + str(time.time_ns()).encode()).hexdigest()
    _jobs[eid] = {"status": "queued", "data": data}
    loop = asyncio.get_running_loop()
    loop.run_in_executor(_job_pool, _run_job, eid)
    return {"event_id": eid}


def _run_job(eid: str) -> None:
    _jobs[eid]["status"] = "processing"
    t0 = time.time()
    try:
        data = _jobs[eid]["data"]
        text, language, ref, exag, cfg_w, temp = _sniff_args(data)
        if not text:
            raise RuntimeError("empty text in 30-arg data array")
        text, exag, cfg_w = _apply_emotion_tag(text, exag, cfg_w)
        model = _ensure_model()
        sr, pcm = _synthesize(model, text, language, ref, exag, cfg_w, temp)
        out_path = UPLOAD_DIR / f"{eid[:16]}_audio.wav"
        _write_wav(out_path, sr, pcm)
        with open(out_path, "rb") as f:
            wav_bytes = f.read()
        b64 = base64.b64encode(wav_bytes).decode("ascii")
        _jobs[eid] = {
            "status": "done",
            "path": str(out_path),
            "url_path": f"/gradio_api/file={out_path}",
            "size": len(wav_bytes),
            "sr": sr,
            "samples": int(pcm.shape[0]),
            "duration": float(pcm.shape[0]) / sr,
            "b64": b64,
            "text": text,
            "language": language,
            "ref": ref,
        }
        print(f"[job] {eid[:8]} OK text={text!r} lang={language} "
              f"dur={pcm.shape[0]/sr:.2f}s wav={out_path.stat().st_size}B "
              f"total={time.time()-t0:.2f}s", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        _jobs[eid] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
        print(f"[job] {eid[:8]} FAIL: {e}", flush=True)


def _build_complete_payload(job: dict) -> dict:
    """Build the SSE `data:` payload.

    SkyrimNet's response parser looks for the keys `success`, `path`,
    and `bytes` at the TOP LEVEL of the JSON, not inside `data[0]`.
    Both layouts are exposed so the response is also usable by other
    Gradio clients.
    """
    if job.get("status") == "error":
        err = job.get("error", "unknown error")
        return {
            "msg": "process_completed", "success": False,
            "path": "", "bytes": "", "bytes_b64": "",
            "error": err, "output": {"error": err, "is_generating": False},
        }
    path = job["path"]
    url = f"http://{GRADIO_HOST}:{GRADIO_PORT}{job['url_path']}"
    b64 = job.get("b64", "")
    fd = {
        "path": path,
        "url": url,
        "result_url": url,
        "file_url": url,
        "name": Path(path).name,
        "orig_name": Path(path).name,
        "size": job["size"],
        "type": "file",
        "mime_type": "audio/wav",
        "is_file": True,
        "data": b64,
        "bytes": b64,
        "bytes_b64": b64,
        "content_type": "audio/wav",
        "meta": {"_type": "gradio.FileData"},
    }
    return {
        "msg": "process_completed", "success": True,
        "path": path, "bytes": b64, "bytes_b64": b64,
        "data": [fd],
        "output": {
            "data": [fd],
            "is_generating": False,
            "duration": round(job.get("duration", 0.0), 3),
            "average_duration": round(job.get("duration", 0.0), 3),
            "render_config": None,
        },
    }


@app.get("/gradio_api/call/generate_audio/{eid}")
@app.get("/call/generate_audio/{eid}")
async def poll_generate(eid: str):
    job = _jobs.get(eid)
    if not job:
        return Response(
            content='event: error\ndata: {"error":"job not found"}\n\n',
            status_code=404, media_type="text/event-stream",
        )
    t0 = time.time()
    while job.get("status") in ("queued", "processing"):
        if time.time() - t0 > 300:
            return Response(
                content='event: error\ndata: {"error":"timeout"}\n\n',
                status_code=504, media_type="text/event-stream",
            )
        await asyncio.sleep(0.1)
        job = _jobs.get(eid, job)
    payload = _build_complete_payload(job)
    body = "event: complete\ndata: " + json.dumps(payload) + "\n\n"
    return Response(
        content=body, media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Content-Length": str(len(body.encode("utf-8"))),
        },
    )


@app.api_route("/gradio_api/file={fpath:path}", methods=["GET", "HEAD"])
@app.api_route("/file={fpath:path}", methods=["GET", "HEAD"])
async def serve_file(fpath: str):
    """Serve a generated WAV.  SkyrimNet may also HEAD-check this
    endpoint before downloading.
    """
    p = Path(fpath)
    if not p.is_absolute():
        cand = (UPLOAD_DIR / fpath).resolve()
        if cand.is_file():
            p = cand
    if not p.is_file():
        raise HTTPException(404, f"not found: {fpath}")
    return FileResponse(str(p), media_type="audio/wav",
                        headers={"Accept-Ranges": "bytes"})


# --- entrypoint ------------------------------------------------------------
def main() -> int:
    import uvicorn
    print(f"[server] starting uvicorn on {GRADIO_HOST}:{GRADIO_PORT}", flush=True)
    uvicorn.run(app, host=GRADIO_HOST, port=GRADIO_PORT, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
