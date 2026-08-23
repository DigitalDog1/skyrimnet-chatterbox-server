#!/usr/bin/env bash
# Start the SkyrimNet Chatterbox TTS server (Linux/macOS).

export GRADIO_HOST="${GRADIO_HOST:-127.0.0.1}"
export GRADIO_PORT="${GRADIO_PORT:-7861}"
export GRADIO_UPLOAD_DIR="${GRADIO_UPLOAD_DIR:-/tmp/gradio}"

mkdir -p "$GRADIO_UPLOAD_DIR"

python3 server.py
