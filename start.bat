@echo off
REM Start the SkyrimNet Chatterbox TTS server.
REM Edit the variables below if you want a different port or upload dir.

set GRADIO_HOST=127.0.0.1
set GRADIO_PORT=7861
set GRADIO_UPLOAD_DIR=%LOCALAPPDATA%\Temp\gradio

python server.py
