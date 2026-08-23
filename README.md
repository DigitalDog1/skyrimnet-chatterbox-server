**English** | [Русский](README.ru.md)

# SkyrimNet Chatterbox TTS Server

A self-contained FastAPI server that lets the [SkyrimNet](https://github.com/MinLL/SkyrimNet-GamePlugin)
SKSE plugin use [Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox)
as a local TTS engine with **voice cloning** and **emotion control**,
all on a single 8 GB GPU.

Chatterbox is not in the list of engines shipped with SkyrimNet
(Piper / PocketTTS / XTTS / ElevenLabs / Inworld / Zonos), and the
generic Chatterbox UIs on Hugging Face do not speak the
`GradioTTSInterface` wire protocol that SkyrimNet's C++ client uses.
This project bridges the two.

## Features

- Local inference.  No cloud API, no rate limits.
- 23 languages out of the box, including Russian, English, German, etc.
- Live voice cloning from any reference WAV SkyrimNet uploads.
- Emotion tags from SkyrimNet dialogue (`[shout]`, `[whisper]`,
  `[angry]`, `[sad]`, `[happy]`, `[dramatic]`, ...) are mapped to
  per-emotion (exaggeration, cfg_weight) values so they actually
  change the delivery, since Chatterbox is a pure acoustic model and
  would otherwise pronounce the tag literally.
- Single binary, no Gradio framework, no Node.js, no WSL.

## Requirements

- Python 3.10+
- An NVIDIA GPU with ~4 GB of VRAM and
  CUDA 11.8 or 12.x.  CPU mode also works, but a 30-second dialogue
  takes about 3 minutes on CPU.
- [SkyrimNet](https://github.com/MinLL/SkyrimNet-GamePlugin) installed
  in your Skyrim Special Edition `SKSE/Plugins/` folder (this project
  only provides the server; the mod itself is distributed separately).

## Install

```bash
git clone https://github.com/DigitalJesus/skyrimnet-chatterbox-server.git
cd skyrimnet-chatterbox-server
pip install -r requirements.txt
```

On first run the multilingual model weights (~2 GB) will be pulled
from Hugging Face into your local cache (`~/.cache/huggingface/`).

## Run

```bash
python server.py
# or on Windows:
start.bat
```

The server listens on `http://127.0.0.1:7861` by default.  Open
`http://127.0.0.1:7861/health` to confirm the model is loaded
(`"model_loaded": true`).

In SkyrimNet's TTS settings, point the Gradio server URL to
`http://127.0.0.1:7861` and pick any voice profile.  Voice samples
are uploaded by the mod itself and cached in
`%LOCALAPPDATA%\Temp\gradio` (or wherever `GRADIO_UPLOAD_DIR`
points).

## Configuration

All settings come from environment variables, with sensible defaults
for Windows + Skyrim.

| Variable                | Default                                                 | What it does |
|-------------------------|---------------------------------------------------------|--------------|
| `GRADIO_HOST`           | `127.0.0.1`                                             | Bind address |
| `GRADIO_PORT`           | `7861`                                                  | Bind port    |
| `GRADIO_UPLOAD_DIR`     | `%LOCALAPPDATA%\Temp\gradio`                            | Where voice samples and generated WAVs are cached |
| `DEFAULT_VOICE_SAMPLE`  | `voice-sample.mp3` next to the script                   | Fallback voice when SkyrimNet sends no reference |
| `MODEL_REPO`            | `ResembleAI/chatterbox`                                 | Hugging Face repo to load |
| `DEVICE`                | `cuda` if available, else `cpu`                         | Torch device |

## Emotion tags

SkyrimNet appends bracketed tags to dialogue text, e.g.

> "Run, you fool! [shout]"

This server recognises the following tags and replaces them with the
matching Chatterbox preset:

| Tag                       | exaggeration | cfg_weight | Feel |
|---------------------------|--------------|------------|------|
| `[whisper]` `[whispering]`| 0.25         | 0.70       | quiet, careful |
| `[quiet]`                 | 0.30         | 0.65       | soft |
| `[sad]` `[somber]`        | 0.40         | 0.70       | low energy |
| `[fear]` `[scared]`       | 0.70         | 0.40       | nervous |
| `[terrified]`             | 0.85         | 0.35       | panicked |
| `[angry]` `[dramatic]`    | 0.75         | 0.35       | intense, theatrical |
| `[shout]` `[yelling]`     | 0.85         | 0.30       | loud |
| `[scream]`                | 0.95         | 0.25       | screaming |
| `[happy]` `[cheerful]`    | 0.60         | 0.45       | upbeat |
| `[laugh]` `[laughing]`    | 0.65         | 0.40       | laughing |
| `[sarcastic]`             | 0.55         | 0.55       | dry |
| `[neutral]`               | 0.50         | 0.50       | flat |
| (no tag)                  | 0.35         | 0.35       | SkyrimNet's default |

If you have raised the per-engine `exaggeration` and `cfg_weight`
beyond 0.45 in SkyrimNet's UI, your values are kept and the tag
emotion is ignored (the global setting wins).

## Protocol

The server implements the small subset of the Gradio queue API that
SkyrimNet's `GradioTTSInterface` actually calls.  Three endpoints
plus one file-server route:

```
POST  /gradio_api/upload                multipart      -> ["abs/path.wav"]
POST  /gradio_api/call/generate_audio   {data:[...]}  -> {"event_id": "..."}
GET   /gradio_api/call/generate_audio/{eid}            -> SSE: "event: complete\ndata: {...}\n\n"
GET   /gradio_api/file=<abs_path>                     -> audio/wav bytes
```

The 30-element `data` array is type-sniffed rather than positionally
parsed, because the order in which SkyrimNet sends the arguments
does not match either of the two reference `gradio_tts_app.py`
files published by Resemble AI.  The parser picks the longest string
(text), a 2-letter language code, a wav path that is not the
`empty_100ms.wav` placeholder (voice reference), and the three
floats that fall in the `(exaggeration, cfg_weight, temperature)`
ranges respectively.

The poll endpoint returns a single-shot SSE response (not
`StreamingResponse` with chunked transfer) because some Gradio
client libraries mis-parse the latter as a prematurely closed
stream.  The audio WAV bytes are inlined as base64 in the
`bytes` / `bytes_b64` / `data` fields so the SkyrimNet client does
not need a follow-up GET on `/file=...`.

## Troubleshooting

- **`Model load failed: ChatterboxMultilingualTTS.from_pretrained() got an unexpected keyword argument 'cache_dir'`** —
  this is from an older fork.  Install the official package:
  `pip install -U chatterbox-tts`.

- **Long audio cuts off mid-sentence with `forcing EOS token`** —
  Chatterbox has an internal repetition detector that forces the end
  of the sequence when the decoder gets stuck.  Try lowering
  `cfg_weight` (e.g. 0.25) or simplifying the text.

- **SkyrimNet still shows `Failed to process TTS response`** — make
  sure the server URL in the mod's TTS settings is exactly
  `http://127.0.0.1:7861` (no trailing slash, no `localhost`).  The
  SSE poll response carries the audio inline as base64, so a
  follow-up `GET /file=...` is not required, but SkyrimNet does an
  `HEAD` against the file URL first; that endpoint must return 200.

- **You have 8 GB VRAM and a 3 GB model** — fine.  Chatterbox
  Multilingual is 500 M parameters; on a 4060 it loads in about
  30 s and synthesises a 3 s line in about 1 s.

## License

MIT.  See `LICENSE`.  This project does not bundle or distribute
SkyrimNet, Skyrim, or Chatterbox model weights; users must obtain
each from their respective sources under their own licenses.
