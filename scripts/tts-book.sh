#!/usr/bin/env bash
# Thin launcher for the tts-book entry point installed via `pip install -e .`
# into the chatterbox venv. Sets memory-management env vars, activates the
# venv, and exec's the installed console script.
#
# Override the venv location with TTS_BOOKS_VENV=/some/other/venv.

set -e

VENV="${TTS_BOOKS_VENV:-$HOME/chatterbox-venv}"

if [ ! -d "$VENV" ]; then
    echo "Error: venv not found at $VENV" >&2
    exit 1
fi

# ── Memory management: prevent glibc arena fragmentation during TTS ──
# Without these, ~300 chunks exhaust 62 GB RAM even with per-chunk
# gc.collect() + malloc_trim(0). Root cause: glibc's malloc arena
# fragmentation — freed blocks can't be coalesced with the heap top,
# so malloc_trim can't release them to the OS.

# Use mmap for allocations > 128 KB so each large tensor is freed
# individually rather than held in a thread-local arena.
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=65536

# Prefer jemalloc if installed — handles fragmentation vastly better
# than glibc. libjemalloc2 is the shared-library package; the -dev
# package is not needed.
if [ -f /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ]; then
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
    # Aggressive decay: return dirty pages to the OS immediately instead
    # of holding them in reserve. Critical for reloading the model without
    # doubling memory usage.
    export MALLOC_CONF="dirty_decay_ms:0,muzzy_decay_ms:0"
fi

source "$VENV/bin/activate"

if ! command -v tts-book >/dev/null 2>&1; then
    echo "Error: 'tts-book' entry point not found in $VENV" >&2
    echo "Run: (cd \"$(dirname \"$0\")/..\" && pip install -e .)" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="" exec tts-book "$@"
