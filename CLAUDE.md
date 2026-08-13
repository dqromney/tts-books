# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Orientation

Personal toolbox for CPU-only text-to-speech audiobook generation using **Chatterbox Turbo TTS**. The project is currently a flat set of scripts — no `src/` package yet — with a large single-file tkinter GUI (`tts_book_gui.py`, ~170 KB, ~2000+ lines) as the primary tool.

**Read `docs/CLAUDE.md` before non-trivial edits to `tts_book_gui.py`.** It documents the thread model, chunk/resume system, batch queue persistence, garbled-chunk detection, archive/rework flow, memory-management tiers, settings sync, and the two Chatterbox library patches that must be reapplied after upgrades. Do not duplicate that content here — treat it as authoritative for internals.

`docs/REFACTOR-PLAN.md` is a *sketch*, not implemented. If asked to "refactor to `src/`", "convert to a package", or "add tests", follow that plan (Phase 1 → 5 → 2 → 3 → 4 → 6 → 7) rather than inventing structure.

## Environment assumptions

- Python 3.12 in a venv at `~/chatterbox-venv/` (not `.venv/` in the repo — the `.venv/` here is a JetBrains artifact). All Python invocations must be inside that venv.
- CPU-only inference: every launcher exports `CUDA_VISIBLE_DEVICES=""`. Do not add GPU code paths; the target machine's GPU has too little VRAM.
- System deps: `ffmpeg`, `yt-dlp`, and optionally `libjemalloc2` (memory-management tier 2 — the launcher `LD_PRELOAD`s it if present).
- Runtime state files (`tts-book.config`, `tts-book.queue.json`, `tts-pronunciation.json`) and voice references (`~/voice-samples/`), books (`~/books/`), and output (`~/tts_output/`) live in `$HOME`, not the repo. They are gitignored.

## Repo-vs-runtime path discrepancy (important)

`scripts/tts-book.sh` and `scripts/start-tts.sh` currently hardcode `$HOME/bin/tts_book_gui.py` and `$HOME/bin/gradio_tts_turbo_app.py` as the launch target — **not** the copies in this repo. In practice the user runs from `~/bin/` copies while editing here, then syncs. When editing the launcher scripts, preserve this until the Phase 1/5 refactor lands (which switches them to installed entry points).

## Common commands

Assume the venv is already activated (`source ~/chatterbox-venv/bin/activate`).

```bash
# Lint / format (config in pyproject.toml — line-length 100, ruff selects E,F,W,I,UP,B,SIM, ignores E501)
ruff check .
ruff check --fix .
black .

# Tests (tests/ is a placeholder — currently only .gitkeep)
pytest
pytest tests/test_chunking.py            # single file
pytest tests/test_chunking.py::test_name # single test
pytest -k "chunking and not slow"        # keyword filter
pytest -m "not slow"                     # skip slow-marked (Whisper-dependent) tests once they exist

# Launch (from repo — note the ~/bin path caveat above)
./scripts/tts-book.sh     # tkinter GUI
./scripts/start-tts.sh    # Gradio web UI
```

## Repository layout at a glance

- `tts_book_gui.py` — main tkinter GUI. Everything of substance lives here.
- `gradio_tts_turbo_app.py` — simpler web UI, no resume support.
- `capture_mol.py`, `capture_zos.py` — Royal Road chapter scrapers. Pattern: `fetch()` → `extract()` → follow `next_url`. Copy and edit `start_url`/`output_file` to add a new fiction.
- `scripts/` — bash wrappers. `tts-book.sh` sets the memory-management env vars (`MALLOC_MMAP_THRESHOLD_`, `MALLOC_TRIM_THRESHOLD_`, jemalloc `LD_PRELOAD`) — do not strip these.
- `tts-pronunciation.example.json` — sample seed for the runtime pronunciation dictionary.
- `docs/CLAUDE.md` — detailed internals (authoritative).
- `docs/REFACTOR-PLAN.md` — future package refactor (not started).

## Editing guidelines specific to this repo

- **Never call tkinter widgets from the generation thread.** All UI updates from `_do_generate` and callees must go through `root.after(0, lambda: ...)`. See `docs/CLAUDE.md` "Thread model".
- **When resuming a job, use `state["chunks"]` from disk; never re-split the text.** Re-splitting produces different boundaries and desyncs the `completed` set. See "Chunk/resume system".
- **Chatterbox patches live in the venv's `site-packages/chatterbox/`**, not in this repo. If a change to `tts_book_gui.py` assumes patched behavior (cached conditionals, `max_gen_len`/`no_repeat_ngram_size` kwargs, raised `max_gen_len` default, `NoRepeatNGramLogitsProcessor`), keep it consistent with what `docs/CLAUDE.md` "Chatterbox library patches" describes.
- Dependencies are currently declared in `requirements.txt`, not `pyproject.toml` (a deliberate deferral until the package refactor — do not migrate them piecemeal).
