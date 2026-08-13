#!/usr/bin/env bash
# Enhance TTS audio, embed metadata, and convert to MP3.
# Applies EQ, highpass filter, dynamic compression, and resamples to 44.1kHz.
#
# Usage: tts-enhance.sh input.wav output.mp3 [options]
#
# Options:
#   --title TITLE       Track title (ID3 TIT2)
#   --artist ARTIST     Artist / author (ID3 TPE1)
#   --album ALBUM       Album name (ID3 TALB)
#   --date DATE         Year or date (ID3 TDRC)
#   --genre GENRE       Genre (ID3 TCON), e.g. "Audiobook"
#   --comment COMMENT   Comment (ID3 COMM)
#   --track TRACK       Track number, e.g. "1/10" (ID3 TRCK)
#   --cover COVER       Path to cover image (JPEG/PNG, embedded as attached picture)
#
# Examples:
#   tts-enhance.sh speech.wav speech.mp3
#   tts-enhance.sh ch1.wav ch1.mp3 --title "Chapter 1" --artist "David Romney" --genre Audiobook
#   tts-enhance.sh book.wav book.mp3 --title "Abish" --artist "David Romney" --cover cover.jpg

set -e

# ── handle --help before positional arg check ──────────────────────
for arg in "$@"; do
    if [ "$arg" = "--help" ] || [ "$arg" = "-h" ]; then
        echo "Usage: tts-enhance.sh input.wav output.mp3 [options]"
        echo ""
        echo "Audio processing: EQ, highpass, compression, resample to 44.1kHz, MP3 @ 192k"
        echo ""
        echo "Metadata options (all optional):"
        echo "  --title TITLE       Track title"
        echo "  --artist ARTIST     Artist / author"
        echo "  --album ALBUM       Album name"
        echo "  --date DATE         Year or date"
        echo "  --genre GENRE       Genre (e.g. Audiobook)"
        echo "  --comment COMMENT   Comment text"
        echo "  --track TRACK       Track number (e.g. 1/10)"
        echo "  --cover COVER       Cover image path (JPEG/PNG)"
        exit 0
    fi
done

# ── parse required positional args ──────────────────────────────────
if [ $# -lt 2 ]; then
    echo "Usage: tts-enhance.sh input.wav output.mp3 [options]" >&2
    echo "       tts-enhance.sh --help    for full option list" >&2
    exit 1
fi

INPUT="$1"
OUTPUT="$2"
shift 2

# ── parse optional flags ────────────────────────────────────────────
TITLE=""
ARTIST=""
ALBUM=""
DATE=""
GENRE=""
COMMENT=""
TRACK=""
COVER=""

while [ $# -gt 0 ]; do
    case "$1" in
        --title)   TITLE="$2";   shift 2 ;;
        --artist)  ARTIST="$2";  shift 2 ;;
        --album)   ALBUM="$2";   shift 2 ;;
        --date)    DATE="$2";    shift 2 ;;
        --genre)   GENRE="$2";   shift 2 ;;
        --comment) COMMENT="$2"; shift 2 ;;
        --track)   TRACK="$2";   shift 2 ;;
        --cover)   COVER="$2";   shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── build ffmpeg command ─────────────────────────────────────────────

# Audio filter chain (unchanged)
AUDIO_FILTER="highpass=f=80, \
equalizer=f=150:t=q:w=1:g=3, \
equalizer=f=400:t=q:w=0.8:g=2, \
equalizer=f=3000:t=q:w=1:g=-1.5, \
equalizer=f=8000:t=q:w=1.2:g=2, \
compand=attacks=0.01:decays=0.2:points=-80/-80|-30/-15|0/-3|20/-3, \
aresample=44100"

# Base command: input file(s)
CMD=(-i "$INPUT")

# If cover art provided, add it as a second input
if [ -n "$COVER" ]; then
    if [ ! -f "$COVER" ]; then
        echo "ERROR: cover file not found: $COVER" >&2
        exit 1
    fi
    CMD+=(-i "$COVER")
    COVER_MAP=(-map 0:a -map 1:v -c:v copy -disposition:v attached_pic)
else
    COVER_MAP=()
fi

# Metadata tags
META=()
[ -n "$TITLE" ]   && META+=(-metadata title="$TITLE")
[ -n "$ARTIST" ]  && META+=(-metadata artist="$ARTIST")
[ -n "$ALBUM" ]   && META+=(-metadata album="$ALBUM")
[ -n "$DATE" ]    && META+=(-metadata date="$DATE")
[ -n "$GENRE" ]   && META+=(-metadata genre="$GENRE")
[ -n "$COMMENT" ] && META+=(-metadata comment="$COMMENT")
[ -n "$TRACK" ]   && META+=(-metadata track="$TRACK")

# Assemble and run
# -ac 2 forces stereo output.  The source TTS is mono; duplicating to both
# channels costs almost nothing at 192 kbps because libmp3lame uses joint
# stereo by default and compresses the redundant channel efficiently.
# The stereo file is broadly more compatible with player firmware and car
# stereos that misbehave on mono MP3.
ffmpeg \
    "${CMD[@]}" \
    -af "$AUDIO_FILTER" \
    -ac 2 \
    "${COVER_MAP[@]}" \
    "${META[@]}" \
    -codec:a libmp3lame -b:a 192k \
    "$OUTPUT"

echo "Done: $OUTPUT"
