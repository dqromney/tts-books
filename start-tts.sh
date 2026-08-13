#!/usr/bin/env bash
set -e

VENV="$HOME/chatterbox-venv"
APP="$HOME/bin/gradio_tts_turbo_app.py"

if [ ! -d "$VENV" ]; then
    echo "Error: venv not found at $VENV" >&2
    exit 1
fi

if [ ! -f "$APP" ]; then
    echo "Error: app not found at $APP" >&2
    exit 1
fi

source "$VENV/bin/activate"
echo "Starting Gradio TTS Turbo (CPU mode)..."
CUDA_VISIBLE_DEVICES="" python "$APP"
