# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Orientation

Personal toolbox for CPU-only text-to-speech audiobook generation using **Chatterbox Turbo TTS**. Installable Python package at `src/tts_books/`, with a large single-file tkinter GUI (`src/tts_books/gui.py`, ~170 KB, ~2000+ lines) as the primary tool. Split into non-GUI modules is planned but not started (Phase 3.1 of the refactor plan).

**Read `docs/CLAUDE.md` before non-trivial edits to `src/tts_books/gui.py`.** It documents the thread model, chunk/resume system, batch queue persistence, garbled-chunk detection, archive/rework flow, memory-management tiers, settings sync, and the two Chatterbox library patches that must be reapplied after upgrades. Do not duplicate that content here — treat it as authoritative for internals.

`docs/REFACTOR-PLAN.md` phases 1, 3.0, 5 are done. Remaining: 2 (XDG config paths), 3.1-3.3 (break up gui.py per the 13-step extraction order), 4 (test scaffolding), 6 (Chatterbox patch files), 7 (CI + release). Follow that plan rather than inventing structure.

## Environment assumptions

- Python 3.12 in a venv at `~/chatterbox-venv/` (not `.venv/` in the repo — the `.venv/` here is a JetBrains artifact). All Python invocations must be inside that venv.
- CPU-only inference: every launcher exports `CUDA_VISIBLE_DEVICES=""`. Do not add GPU code paths; the target machine's GPU has too little VRAM.
- System deps: `ffmpeg`, `yt-dlp`, and optionally `libjemalloc2` (memory-management tier 2 — the launcher `LD_PRELOAD`s it if present).
- Runtime state files (`tts-book.config`, `tts-book.queue.json`, `tts-pronunciation.json`) and voice references (`~/voice-samples/`), books (`~/books/`), and output (`~/tts_output/`) live in `$HOME`, not the repo. They are gitignored.

## Package install (required before first launch)

After cloning or pulling, the package must be installed editable into the chatterbox venv so the console entry points (`tts-book`, `tts-book-web`, `capture-mol`, `capture-zos`) exist:

```bash
~/chatterbox-venv/bin/pip install -e .
```

The launcher scripts (`scripts/*.sh`) fail with a helpful error if the entry point is missing.

## Common commands

Assume the venv is already activated (`source ~/chatterbox-venv/bin/activate`).

```bash
# Lint / format (config in pyproject.toml — line-length 100, ruff selects E,F,W,I,UP,B,SIM, ignores E501)
ruff check src/ tests/
ruff check --fix src/ tests/
black src/ tests/

# Tests (currently only tests/test_prep_refactors.py — 22 tests covering
# the three prep-refactor dataclasses; broader coverage is Phase 4)
pytest
pytest tests/test_prep_refactors.py::TestCancelToken            # single class
pytest tests/test_prep_refactors.py::TestCancelToken::test_reset_clears_cancel_and_resumes  # single test
pytest -k "cancel and not slow"                                 # keyword filter
pytest -m "not slow"                                            # skip slow-marked tests once they exist

# Launch
./scripts/tts-book.sh     # tkinter GUI (tts-book entry point)
./scripts/start-tts.sh    # Gradio web UI (tts-book-web entry point)

# Or bypass the launcher (skips MALLOC/jemalloc env prep -- fine for
# short-running tests, not for full audiobook generation)
tts-book --help
```

## Repository layout at a glance

- `src/tts_books/gui.py` — main tkinter GUI. Everything of substance lives here (~170 KB, ~2000+ lines).
- `src/tts_books/gradio_app.py` — simpler web UI, no resume support.
- `src/tts_books/scrapers/{mother_of_learning,zenith_of_sorcery}.py` — Royal Road chapter scrapers. Pattern: `fetch()` → `extract()` → follow `next_url`. Copy and edit `start_url`/`output_file` to add a new fiction.
- `scripts/` — thin bash wrappers around the installed entry points. `tts-book.sh` sets the memory-management env vars (`MALLOC_MMAP_THRESHOLD_`, `MALLOC_TRIM_THRESHOLD_`, jemalloc `LD_PRELOAD`) — do not strip these; they must be set before Python starts.
- `tts-pronunciation.example.json` — sample seed for the runtime pronunciation dictionary.
- `docs/CLAUDE.md` — detailed internals (authoritative).
- `docs/REFACTOR-PLAN.md` — remaining refactor phases (2, 3.1-3.3, 4, 6, 7).

## Editing guidelines specific to this repo

- **Never call tkinter widgets from the generation thread.** All UI updates from `_do_generate` and callees must go through `root.after(0, lambda: ...)`. See `docs/CLAUDE.md` "Thread model".
- **When resuming a job, use `state["chunks"]` from disk; never re-split the text.** Re-splitting produces different boundaries and desyncs the `completed` set. See "Chunk/resume system".
- **Chatterbox patches live in the venv's `site-packages/chatterbox/`**, not in this repo. If a change to `src/tts_books/gui.py` assumes patched behavior (cached conditionals, `max_gen_len`/`no_repeat_ngram_size` kwargs, raised `max_gen_len` default, `NoRepeatNGramLogitsProcessor`), keep it consistent with what `docs/CLAUDE.md` "Chatterbox library patches" describes.
- Dependencies are currently declared in `requirements.txt`, not `pyproject.toml` (a deliberate deferral until Phase 4/6 — do not migrate them piecemeal).
