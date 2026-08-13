#!/usr/bin/env bash
# Apply Chatterbox upstream patches to the active venv's site-packages.
#
# Usage:
#   ./patches/apply.sh [/path/to/venv]          # defaults to ~/chatterbox-venv
#
# Patches are against chatterbox-tts==0.1.7.  Run after:
#   pip install chatterbox-tts==0.1.7
# to re-apply after an upgrade or fresh install.
#
# Each patch is idempotent: `patch --dry-run` is checked first; if already
# applied the script skips rather than failing.
set -euo pipefail

VENV="${1:-$HOME/chatterbox-venv}"
SITE="$VENV/lib/python3.12/site-packages/chatterbox"

if [[ ! -d "$SITE" ]]; then
    echo "ERROR: chatterbox not found at $SITE" >&2
    echo "       Install it first: pip install chatterbox-tts==0.1.7" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

apply_patch() {
    local patch_file="$1"
    local label
    label="$(basename "$patch_file")"

    echo "--- $label ---"
    if patch --dry-run -f -p1 -d "$SITE/.." -s < "$patch_file" 2>/dev/null; then
        patch -f -p1 -d "$SITE/.." < "$patch_file"
        echo "  Applied."
    else
        echo "  Already applied or does not apply cleanly — skipping."
    fi
}

apply_patch "$SCRIPT_DIR/chatterbox-tts_turbo-max-gen-len-and-ngram-kwargs.patch"
apply_patch "$SCRIPT_DIR/chatterbox-t3-max-gen-len-and-ngram-guard.patch"

echo ""
echo "Done. Verify with: python -c \"from chatterbox.tts_turbo import ChatterboxTTS\""
