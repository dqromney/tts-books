# Refactor Plan: convert tts-books to an installable package

**Status:** Not started. Sketch only — no code has been changed against this plan.

**Motivation:** The current layout is flat scripts with hardcoded `~/bin/` paths, a 2000+ line monolithic `tts_book_gui.py`, and no tests. This works for personal use but blocks: (a) `pip install -e .` in the venv, (b) PyCharm's package-aware refactor tools, (c) unit tests, (d) publishing to GitHub as something contributors can navigate.

**Non-goal:** Feature changes. This refactor is structural; behavior stays identical.

**Key constraint:** `~/bin/` remains the canonical runtime during and after the refactor. Either the project checkout becomes the runtime (via `pip install -e .` into `~/chatterbox-venv/`, and `~/bin/*.sh` become thin wrappers that just `exec` the entry points), or `~/bin/` is symlinked into the project. Decide this before starting Phase 1.

---

## Phase 1 — Move to `src/` layout

Create a proper Python package under `src/tts_books/`. No behavior change yet; just move and rename.

```
src/tts_books/
    __init__.py
    gui.py                  ← was tts_book_gui.py (still one big file for now)
    gradio_app.py           ← was gradio_tts_turbo_app.py
    scrapers/
        __init__.py
        mother_of_learning.py    ← was capture_mol.py
        zenith_of_sorcery.py     ← was capture_zos.py
```

Update `pyproject.toml` with a `[build-system]` block and package discovery:

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[project.scripts]
tts-book = "tts_books.gui:main"
tts-book-web = "tts_books.gradio_app:main"
capture-mol = "tts_books.scrapers.mother_of_learning:main"
capture-zos = "tts_books.scrapers.zenith_of_sorcery:main"
```

Each Python file gets a `main()` function that wraps the current top-level code. After `pip install -e .`, the commands `tts-book`, `tts-book-web`, etc. become available in the venv.

**Verification:** `pip install -e . && tts-book` launches the GUI identically to `./scripts/tts-book.sh`.

---

## Phase 2 — Config discovery (kill hardcoded `~/bin/` paths)

Today, `tts_book_gui.py` hardcodes:
- `PRON_DICT_PATH = ~/bin/tts-pronunciation.json`
- `CONFIG_PATH = ~/bin/tts-book.config`
- `QUEUE_PATH = ~/bin/tts-book.queue.json`
- `ARCHIVE_DIR = ~/tts_output/archive/`

Introduce `src/tts_books/paths.py`:

```python
from pathlib import Path
import os

def config_dir() -> Path:
    """XDG-compliant: $XDG_CONFIG_HOME/tts-books, fallback ~/.config/tts-books"""
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    d = Path(base) / "tts-books"
    d.mkdir(parents=True, exist_ok=True)
    return d

def data_dir() -> Path:
    """XDG-compliant: $XDG_DATA_HOME/tts-books, fallback ~/.local/share/tts-books"""
    base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    d = Path(base) / "tts-books"
    d.mkdir(parents=True, exist_ok=True)
    return d

PRON_DICT_PATH = config_dir() / "pronunciation.json"
APP_CONFIG_PATH = config_dir() / "app.json"
QUEUE_PATH = config_dir() / "queue.json"
JOBS_DIR = data_dir() / "jobs"
ARCHIVE_DIR = data_dir() / "archive"
```

**Migration:** on first launch after upgrade, detect legacy `~/bin/*.json` files and offer to move them. Or ship a one-shot `tts-books-migrate` command.

**Verification:** running under a fake `HOME` produces all files under the fake `HOME` and none under `~/bin/`.

---

## Phase 3 — Break up `tts_book_gui.py`

The single file is ~170 KB / 2000+ lines and mixes: tkinter UI, chunking, TTS invocation, garbled detection, archiving, memory management, VoxCeleb1 browsing. Proposed split (module boundaries, not the classes themselves — those come later):

```
src/tts_books/
    gui.py                  Tkinter shell — imports everything else, wires the app
    generation/
        __init__.py
        chunking.py         split_text(), _SEPARATOR_RE, pronunciation apply
        pipeline.py         chunk loop, retry, resume, checkpoint state.json
        stitching.py        _crossfade_chunks(), _remove_dc_offset()
        archiving.py        _archive_job(), rework.json read/write
    quality/
        __init__.py
        garbled.py          _is_chunk_garbled() + all six heuristics
        whisper_backend.py  faster-whisper load/unload/transcribe wrapper
    voices/
        __init__.py
        registry.py         voice combobox population
        voxceleb.py         VoxCeleb1 browser + HuggingFace downloads
    memory/
        __init__.py
        pruning.py          gc.collect + malloc_trim, free-RAM watcher
        model_reload.py     periodic reload logic
    io/
        __init__.py
        text_extract.py     PDF (fitz) + EPUB (ebooklib) + plain
        audio_io.py         WAV read/write, MP3 conversion via ffmpeg
    config.py               live JSON config load/save (uses paths.py)
    pronunciation.py        dict load/save/apply
    scrapers/               (as in Phase 1)
```

Do this in small commits — one module at a time, with tests where practical. `gui.py` stays as the tkinter integration surface but shrinks dramatically.

**Verification:** `pytest` passes on each extracted module; the GUI still launches and completes a small test job identically.

---

## Phase 4 — Test scaffolding

`tests/` currently has only `.gitkeep`. Populate incrementally:

```
tests/
    test_chunking.py        split_text edge cases, separator strip, pronunciation
    test_garbled.py         known-good and known-bad WAV fixtures
    test_stitching.py       crossfade math, DC-offset filter
    test_archiving.py       archive round-trip, rework.json format
    test_paths.py           XDG discovery under fake HOME
    fixtures/
        short.txt
        garbled_sample.wav
        clean_sample.wav
```

Use `pytest` (already configured in `pyproject.toml`). Aim to cover the pure-logic modules first (chunking, stitching, garbled heuristics 1/3/5/6 that don't need Whisper). Whisper-dependent tests can be marked `@pytest.mark.slow` and skipped in fast CI.

---

## Phase 5 — Shell script cleanup

After Phase 1, `scripts/tts-book.sh` becomes:

```bash
#!/usr/bin/env bash
set -e
VENV="${TTS_BOOKS_VENV:-$HOME/chatterbox-venv}"
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=65536
if [ -f /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ]; then
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
    export MALLOC_CONF="dirty_decay_ms:0,muzzy_decay_ms:0"
fi
source "$VENV/bin/activate"
CUDA_VISIBLE_DEVICES="" exec tts-book "$@"
```

No more `$HOME/bin/tts_book_gui.py` reference; the entry point comes from the installed package. The audio-processing scripts (`tts-enhance.sh`, `wav2mp3.sh`, `wav-join.sh`, `voice-sample`) can stay as-is — they're standalone ffmpeg/yt-dlp wrappers, not Python.

---

## Phase 6 — Chatterbox patches as a patch file

Currently the Chatterbox patches (documented in `docs/CLAUDE.md`) live inside the venv's `site-packages/`. Reapplying them after an upgrade is manual.

Ship them under `patches/`:
```
patches/
    chatterbox-tts_turbo-cache-conds-and-kwargs.patch
    chatterbox-t3-max-gen-len-and-ngram-guard.patch
    apply.sh                Auto-apply against the installed chatterbox package
```

`apply.sh` locates the installed `chatterbox` package (`python -c "import chatterbox; print(chatterbox.__path__[0])"`), then runs `patch -p1 --forward` against each file, verifying the checksum first so we don't re-apply blindly on an unfamiliar upstream version.

Consider upstreaming the two patches to the Chatterbox project — they're small, general-purpose improvements (allocation cap, n-gram guard).

---

## Phase 7 — CI + release

- GitHub Actions: run `ruff check`, `black --check`, `pytest` on push.
- Version tags → GitHub Releases with wheel + sdist attached (built from `pyproject.toml`).
- Optional: publish to PyPI as `tts-books`. Requires a distinct name check — `tts` prefix is crowded.

---

## Ordering and effort

| Phase | Effort | Blocker for later phases? |
|-------|--------|---------------------------|
| 1. src/ layout + entry points | ~2 hours | Yes — enables everything below |
| 2. Config discovery / XDG paths | ~2 hours | No, but nice before Phase 3 |
| 3. Break up gui.py | 1–2 days (spread over commits) | Enables Phase 4 |
| 4. Test scaffolding | ~1 day for meaningful coverage | No |
| 5. Shell script cleanup | ~30 min | Depends on Phase 1 |
| 6. Chatterbox patches as files | ~2 hours | No |
| 7. CI + release | ~2 hours | Depends on Phase 4 |

**Recommended order:** 1 → 5 → 2 → 3 → 4 → 6 → 7. Phases 1 and 5 together give the biggest single win (installable, no more `~/bin/` coupling) at the lowest risk.

---

## Open questions

1. **Runtime consolidation** — after Phase 1, does `~/bin/` become a set of thin wrappers pointing at the installed package, or do we keep the current `~/bin/` copies and treat this repo as a mirror?
2. **XDG migration** — auto-detect and move legacy `~/bin/*.json` on first run, or require a manual `tts-books-migrate` command?
3. **Whisper backend** — currently `faster-whisper tiny.en`. Any interest in supporting alternatives (whisper.cpp, distil-whisper) via a pluggable backend interface in `quality/whisper_backend.py`?
4. **PyPI publication** — do we want this discoverable, or stay a GitHub-only project?
