# tts-books

CPU-only text-to-speech audiobook generator built on **Chatterbox Turbo TTS**, with a tkinter desktop GUI, batch queue, resume support, garbled-chunk detection with automatic retries, and post-processing helpers (crossfade stitching, DC-offset removal, ffmpeg EQ, MP3 conversion).

Runs entirely on CPU — no GPU required. Designed for machines with ample RAM (developed on a ThinkPad P51, Xeon E3-1505M v6, 62 GB RAM) where the GPU has too little VRAM to hold the model.

## Features

- **Chunked generation with resume** — text is split into ~200-char chunks, each saved individually with a state checkpoint. Cancelled or crashed jobs resume from the last completed chunk.
- **Batch queue** — persistent queue of jobs; add/remove/reorder items while a run is in progress.
- **Garbled-chunk detection** — ten-check heuristic (duration, Whisper word overlap, word-count ratio, unrecognized tail, tail-word mismatch, stutter, missing final content word, phrase repetition loop, anaphoric over-repetition, single-source phrase spoken twice) with up to 3 retry attempts per chunk using fresh random seeds.
- **Rework workflow** — jobs with unresolved garbled chunks are archived with a `rework.json`. The GUI's Rework button lists pending jobs and re-generates only the flagged chunks.
- **Crossfade stitching** — equal-power crossfades (default 50 ms) between chunks, eliminating audible clicks at chunk boundaries.
- **DC-offset removal** — 2nd-order high-pass at 15 Hz on the final output.
- **Pronunciation dictionary** — flat word→replacement JSON, editable from the GUI, applied before chunking.
- **Voice cloning** — reference WAV per job. Includes a VoxCeleb1 browser that downloads speaker clips on demand from HuggingFace Hub.
- **Auto MP3 conversion** — optional per-job ffmpeg 192k MP3 export.
- **Memory management** — three-tier strategy (glibc tuning, jemalloc preload, periodic model reload) to keep 300+ chunk runs from fragmenting RAM.

## Repository layout

```
tts_book_gui.py                    Main tkinter GUI (~170 KB, single file)
gradio_tts_turbo_app.py            Alternative Gradio web UI (simpler, no resume)
capture_mol.py                     Royal Road scraper — Mother of Learning
capture_zos.py                     Royal Road scraper — Zenith of Sorcery
tts-pronunciation.example.json     Sample pronunciation dictionary

scripts/
  tts-book.sh                      Launcher for the tkinter GUI (sets memory flags)
  start-tts.sh                     Launcher for the Gradio web UI
  tts-watchdog.sh                  Watchdog for long-running TTS jobs
  tts-enhance.sh                   ffmpeg EQ + compression + WAV→MP3 conversion
  wav2mp3.sh                       Bare WAV→MP3 at 192k
  wav-join.sh                      Concatenate WAV files via ffmpeg concat demuxer
  voice-sample                     Extract voice reference clips from YouTube via yt-dlp

docs/
  CLAUDE.md                        Detailed architecture / internals notes
  REFACTOR-PLAN.md                 Roadmap: convert to installable package

tests/                             (placeholder — no tests yet)

pyproject.toml                     Project metadata + tool config (black, ruff, pytest)
requirements.txt                   Runtime dependencies
```

## Setup

1. Create the Python virtual environment (expected at `~/chatterbox-venv/`):

   ```bash
   python3.12 -m venv ~/chatterbox-venv
   source ~/chatterbox-venv/bin/activate
   pip install -r requirements.txt
   ```

2. (Recommended) Install jemalloc for better memory behavior across long runs:

   ```bash
   sudo apt install libjemalloc2
   ```

3. Install system dependencies:

   ```bash
   sudo apt install ffmpeg yt-dlp
   ```

4. (Optional) Copy `tts-pronunciation.example.json` to `~/bin/tts-pronunciation.json` if you want to seed the pronunciation dictionary.

## Running

```bash
./scripts/tts-book.sh    # tkinter desktop GUI (primary tool)
./scripts/start-tts.sh   # Gradio web interface (alternative)
```

Both scripts activate the venv, force CPU-only inference (`CUDA_VISIBLE_DEVICES=""`), and launch the corresponding Python app.

Voice reference WAVs are expected in `~/voice-samples/`. Books (plain text, PDF, or EPUB) live in `~/books/`. Generated audio lands in `~/tts_output/`.

## Chatterbox library patches

`tts_book_gui.py` relies on two upstream patches to the `chatterbox` package in the venv. Details are documented in [docs/CLAUDE.md](docs/CLAUDE.md) under "Chatterbox library patches" — reapply these after upgrading the package:

- `tts_turbo.py::generate()` — cache conditionals across chunks; add `max_gen_len` and `no_repeat_ngram_size` kwargs
- `models/t3/t3.py::inference_turbo()` — raise `max_gen_len` default to 4000; add `NoRepeatNGramLogitsProcessor` (default n=6)

## License

Apache-2.0 — see [LICENSE](LICENSE).
