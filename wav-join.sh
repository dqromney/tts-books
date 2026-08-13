#!/usr/bin/env bash
# Join two or more WAV files into a single WAV file.
# Usage: wav-join.sh output.wav input1.wav input2.wav [input3.wav ...]
#   or:  wav-join.sh output.wav chunk_*.wav

set -e

if [ $# -lt 3 ]; then
    echo "Usage: wav-join.sh output.wav input1.wav input2.wav [...]" >&2
    exit 1
fi

OUTPUT="$1"
shift

# Build a temp filelist for ffmpeg's concat demuxer
LISTFILE="$(mktemp /tmp/wav-join-XXXXXX.txt)"
trap 'rm -f "$LISTFILE"' EXIT

for f in "$@"; do
    if [ ! -f "$f" ]; then
        echo "File not found: $f" >&2
        exit 1
    fi
    printf "file '%s'\n" "$(realpath "$f")" >> "$LISTFILE"
done

COUNT=$#
echo "Joining $COUNT files → $OUTPUT"

ffmpeg -y -f concat -safe 0 -i "$LISTFILE" -c copy "$OUTPUT"

echo "Done: $OUTPUT"
