#!/usr/bin/env bash
# Convert WAV files to MP3 at 192k.
# Usage: wav2mp3.sh file.wav [file2.wav ...] ['glob' ...]
# Accepts literal filenames, shell-expanded globs, and quoted glob patterns.
# Examples:
#   wav2mp3.sh track.wav
#   wav2mp3.sh *.wav
#   wav2mp3.sh "chapters/*.wav"
set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: wav2mp3.sh file.wav [file2.wav ...] ['*.wav' ...]" >&2
    exit 1
fi

_convert() {
    local wav="$1"
    local mp3="${wav%.wav}.mp3"
    echo "$wav → $mp3"
    ffmpeg -y -i "$wav" -b:a 192k "$mp3"
}

count=0
for pattern in "$@"; do
    if [[ "$pattern" == *[\*\?\[]* ]]; then
        # Quoted glob pattern — expand it inside the script
        shopt -s nullglob
        matches=( $pattern )
        shopt -u nullglob
        if [ ${#matches[@]} -eq 0 ]; then
            echo "Warning: no files match '$pattern'" >&2
            continue
        fi
        for wav in "${matches[@]}"; do
            _convert "$wav"
            (( count++ )) || true
        done
    else
        # Literal filename (or already shell-expanded by the caller)
        if [ ! -f "$pattern" ]; then
            echo "Warning: '$pattern' not found, skipping." >&2
            continue
        fi
        _convert "$pattern"
        (( count++ )) || true
    fi
done

echo "$count file(s) converted."
