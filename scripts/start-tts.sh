#!/usr/bin/env bash
# Thin launcher for the tts-book-web entry point (Gradio UI) installed via
# `pip install -e .` into the chatterbox venv.
#
# Override the venv location with TTS_BOOKS_VENV=/some/other/venv.

set -e

VENV="${TTS_BOOKS_VENV:-$HOME/chatterbox-venv}"

if [ ! -d "$VENV" ]; then
    echo "Error: venv not found at $VENV" >&2
    exit 1
fi

source "$VENV/bin/activate"

if ! command -v tts-book-web >/dev/null 2>&1; then
    echo "Error: 'tts-book-web' entry point not found in $VENV" >&2
    echo "Run: (cd \"$(dirname \"$0\")/..\" && pip install -e .)" >&2
    exit 1
fi

echo "Starting Gradio TTS Turbo (CPU mode)..."
CUDA_VISIBLE_DEVICES="" exec tts-book-web
