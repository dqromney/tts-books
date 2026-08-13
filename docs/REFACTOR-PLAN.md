# Refactor Plan: convert tts-books to an installable package

**Status:**
- ✅ Phase 1 — src/ layout + entry points (done)
- ✅ Phase 3.0 — prep refactors (dataclasses, Logger, CancelToken) done in-place
- ✅ Phase 5 — thin launcher scripts wrapping installed entry points (done)
- ⏳ Phase 2 — XDG config paths (not started)
- ⏳ Phase 3.1-3.3 — break up gui.py per the 13-step extraction order
- ⏳ Phase 4 — test scaffolding (partial: 22 tests for the three prep classes; broader coverage pending)
- ⏳ Phase 6 — Chatterbox patches as patch files
- ⏳ Phase 7 — CI + release

**Motivation:** The current layout was flat scripts with hardcoded `~/bin/` paths, a 2000+ line monolithic `tts_book_gui.py`, and no tests. Phases 1 and 5 resolved the flat-scripts and `~/bin/` coupling; the monolith is now `src/tts_books/gui.py` awaiting the 3.1 extraction.

**Non-goal:** Feature changes. This refactor is structural; behavior stays identical.

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

## Phase 3 — Break up `gui.py`

The single file is ~170 KB / 2000+ lines and mixes: tkinter UI, chunking, TTS invocation, garbled detection, archiving, memory management, VoxCeleb1 browsing.

This phase has three parts: **3.0 prep refactors that land in-place before any file moves**, **3.1 module boundaries** (revised from the original sketch after auditing the actual code), and **3.2 an extraction order** grounded in the dependency graph. Do all extractions in a worktree; do all commits one module at a time; verify the GUI launches and completes a small test job after each.

### 3.0 Prep refactors (in-place, must land before any file split)

Four small refactors that stay inside `tts_book_gui.py` but freeze the interfaces every extracted module will consume. Each is independently valuable — the code is cleaner even if Phase 3 never continues.

1. **`BatchItem` dataclass.** The queue item is currently an untyped `dict` built in three places (`_add_to_batch`, `_add_dir_to_batch`, `do_add_to_batch`) and mutated with string keys throughout `_batch_thread`, `_do_generate`, and `_process_rework`. Notably, `_job_hash` is set inside `_do_generate` and read by `_batch_thread` — a classic cross-module field. Fields: `id, source_path, file_name, text, ref_path, voice_name, settings, status, output_path, error, chunks_done, chunks_total, gen_time, _job_hash, _text_sidecar, try_resume` (plus a transient `_eta_str`).

2. **`Settings` dataclass.** `_snapshot_settings()` and `_apply_settings_to_ui()` are already silently drifting — `ref_path` is present in the snapshot but stored on the batch item separately. Same dict flows into `_do_generate`, `state.json`, the on-disk queue, and the batch editor. A dataclass with `.to_dict()`/`.from_dict()` freezes the schema. ~17 fields today.

3. **`Logger` protocol + explicit log buffer.** Today `self._log` both colors the tkinter widget *and* appends to `self._log_buffer` (which `_archive_job` reads to write `generation.log`). Non-GUI modules must not import tkinter, so introduce a small interface with `.info/.warn/.error/.success/.chunk` and a `.buffer()` accessor. Split the two responsibilities now, before archiving is extracted — otherwise the extracted `archiving.py` silently loses `generation.log`.

4. **`CancelToken`** wrapping `cancel_flag: bool` + `pause_event: threading.Event`. `_do_generate` reads both directly; passing an object lets `pipeline.py` avoid a back-reference to the App.

Alongside these four, apply **one code hygiene fix** while everything is still in one file: normalise the `ARCHIVE_DIR` duplication (module constant + `self._archive_dir` instance attribute derived independently) — pick one source of truth so Phase 3 doesn't inherit the drift.

### 3.1 Module boundaries (revised)

```
src/tts_books/
    gui.py                  Tkinter shell — TTSBookApp, all Toplevel dialogs,
                            _build_ui, _lock_ui/_unlock_ui, batch queue Treeview,
                            EVENT_TAGS (only consumed by _build_ui),
                            _refresh_rework_badge, _open_rework_dialog,
                            _update_mem_bar (all GUI-only despite living beside logic)
    batch.py                _load_batch_queue, _save_batch_queue, BatchItem dataclass
                            (persistence + reset-processing-on-startup semantics)
    generation/
        __init__.py
        chunking.py         split_text(), _SEPARATOR_RE (NOT pronunciation — see below)
        pipeline.py         _do_generate, chunk loop, retry orchestration
                            (owns MAX_CHUNK_RETRIES — the constant is only read here),
                            _save_state, _find_resumable_job, _reconcile_job_state,
                            _text_hash/_job_dir/_state_path,
                            periodic model-reload block (30 lines, not worth its own module)
        stitching.py        _crossfade_chunks(), _remove_dc_offset()
        archiving.py        _archive_job (reads Logger.buffer()), _scan_rework_jobs,
                            disk-manipulation portion of _process_rework
                            (rework.json read, tree copy, state.json patch);
                            the tail that calls _start_resume stays in gui.py
    quality/
        __init__.py
        garbled.py          is_chunk_garbled(wav, text, whisper, logger)
                            as a free function — 10 heuristics (not 6, not 4)
        whisper_backend.py  WhisperHandle class exposing load/unload/transcribe;
                            NOT a module-level global (see Risk #1)
    voices/
        __init__.py
        registry.py         voice-combo population as a helper returning {stem: path};
                            widget wiring stays in gui.py
        voxceleb.py         open_voxceleb_dialog(parent, dir, on_downloaded) factory
                            + _sanitize_name (only used here, despite living up top)
    memory/
        __init__.py
        pruning.py          mem_rss_mb(), manual_prune(), auto-prune helper.
                            NO separate model_reload.py — 30 inline lines belong in pipeline.
    io/
        __init__.py
        text_extract.py     load_text_file, _strip_markdown, _clean_inline
        audio_io.py         convert_to_mp3(), auto_convert(),
                            concat+silence-pad+DC-remove tail of _do_generate
    config.py               _load_config, _save_config, all path constants,
                            instance-count management. NO pronunciation constants here.
    pronunciation.py        PRON_DICT_PATH + load/save/apply
                            (constant lives here, not in config.py)
    scrapers/               (as in Phase 1)
```

**Changes from the original sketch:**
- `pronunciation apply` moved *out* of `chunking.py` (it applies to full text before splitting; the two responsibilities are unrelated).
- `EVENT_TAGS`, `_refresh_rework_badge`, `_open_rework_dialog`, `_update_mem_bar` explicitly assigned to `gui.py` (they were implicitly bundled with logic modules).
- `_sanitize_name` assigned to `voxceleb.py` (its only caller), not `chunking.py`.
- `MAX_CHUNK_RETRIES` stays with `pipeline.py` (retry orchestration), not `garbled.py`.
- `PRON_DICT_PATH` moves to `pronunciation.py`, not `config.py`.
- Added `batch.py` for queue persistence (no home in the original sketch).
- Dropped `memory/model_reload.py` (30 lines used once — fold into pipeline).
- `whisper_backend.py` becomes a class, not a module-level global (see Risk #1).

### 3.2 Extraction order

Do these in this order. Each is a single commit that keeps `pytest` green and the GUI launching. Start with pure functions (no App state, immediate test coverage), end with the risky pipeline extraction after every dependency is already outside the App.

| # | Module | Why here |
|---|--------|----------|
| 1 | `io/text_extract.py` | Pure. Zero App coupling. Immediate `pytest` win. |
| 2 | `pronunciation.py` | Already module-level functions. Zero App coupling. |
| 3 | `generation/chunking.py` | Pure. No App state. |
| 4 | `generation/stitching.py` | Pure functions taking tensors. |
| 5 | `config.py` | Small, well-scoped. Instance-count logic stays here. |
| 6 | `memory/pruning.py` | Pure once the Logger is decoupled (step 3.0.3). |
| 7 | `quality/whisper_backend.py` | Wrap the module singleton behind `WhisperHandle`. |
| 8 | `quality/garbled.py` | Free function `is_chunk_garbled(wav, text, whisper, logger)`. |
| 9 | `io/audio_io.py` | MP3 conversion + concat tail. |
| 10 | `batch.py` | Queue persistence. Pipeline should stay stateless about batch. |
| 11 | `generation/archiving.py` | Needs Logger.buffer() from prep step 3.0.3. Only the disk portion. |
| 12 | `generation/pipeline.py` | Biggest and riskiest. Do after every dependency is outside the App. |
| 13 | `voices/registry.py` + `voices/voxceleb.py` | Most GUI-tangled. Extract as factories/helpers; widget wiring stays in `gui.py`. |

`gui.py` is what remains after step 13.

### 3.3 Risk callouts (things a naive extraction breaks)

These are the invariants an "obvious cleanup" refactor gets wrong. Preserve them exactly.

1. **Whisper singleton mutation.** `_whisper_model` is a module-level `global` mutated by both `_load_whisper` *and* the periodic-unload block inside `_do_generate`. If `whisper_backend.py` and `pipeline.py` are separate modules that both reference `_whisper_model` by name, they get separate copies of `None` and the unload silently does nothing. Extract as a `WhisperHandle` class; both modules must reference the same instance.

2. **Resume path is state-machine-fragile.** Always use `state["chunks"]` from disk on resume — never re-split the text (docs/CLAUDE.md). `_reconcile_job_state` uses `<=` (not `<`) to intentionally cover "all chunks done, concat failed" — a "cleanup" refactor breaks resume. `_find_resumable_job` has a three-branch fallback (hint hash → text hash → raw pre-pronunciation text) that recovers from pronunciation-dict drift; combining branches into "one call" loses that recovery.

3. **Tkinter cross-thread rule.** `_do_generate` runs in a daemon thread and never touches widgets directly — all UI updates go through `self.root.after(0, ...)`. The extracted `pipeline.py` must not import tkinter. Replace the inline `self.root.after` closures (~15 of them for progress/status/label updates) with a `ProgressReporter` callback protocol during prep, not during extraction.

4. **Live tk.IntVar reads mid-loop.** `_do_generate` reads `self.reload_every.get()` and `self.prune_every.get()` inside the chunk loop, so the user can change these mid-run. Pipeline can't own tk vars — wrap each in `Callable[[], int]` before extraction.

5. **Chatterbox caching contract.** `prepare_conditionals` is called once per job before the chunk loop, and each chunk call passes `audio_prompt_path=None` to reuse the cached conditionals. Splitting `_do_generate` across pipeline.py and helpers must keep this invariant — one place calls prepare, another calls generate. Add an assertion or a clear comment at both sites.

6. **`_process_rework` order is load-bearing.** Copy archive → patch state.json → delete old archive → resume. If pipeline fails after the archive is gone but before rework chunks regenerate, the job is only in `jobs/`. Splitting this across `archiving.py` and `gui.py` risks reordering. Extract only the copy/patch prefix into archiving; keep the delete+resume tail in gui.py.

7. **Process-global side effects.** `_do_generate` sets `os.environ["OMP_NUM_THREADS"]`, `["MKL_NUM_THREADS"]`, and `th.set_num_threads`. Today `_gen_running` prevents concurrent runs. A cleanly extracted `pipeline.py` should document (or gate) the global mutation so a future caller doesn't spawn two concurrent pipelines and race on the env vars.

8. **`_snapshot_settings` must run on the main thread.** It reads tk vars. `_start_generate` snapshots on the main thread before spawning the worker; any extraction that moves the snapshot off gui.py must preserve the "snapshot on main first, hand dict to worker" ordering.

**Verification for each extraction:** `pytest` passes on the new module; `ruff check` and `black --check` pass; the GUI launches and completes a small test job (single short file, single chunk, and one 5-chunk file to exercise the loop). For the risky extractions (7, 11, 12) also run one resume test (kill after N chunks, restart, confirm it picks up from `state["chunks"]` correctly) and one garbled-retry test (a chunk that historically flagged as garbled — confirm the retry loop still runs).

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
| 3.0 Prep refactors in-place (dataclasses + Logger + CancelToken) | ~1 day | Yes — blocks 3.1-3.3 |
| 3.1-3.3 Break up gui.py | 1–2 days (spread over commits) | Enables Phase 4 |
| 4. Test scaffolding | ~1 day for meaningful coverage | No |
| 5. Shell script cleanup | ~30 min | Depends on Phase 1 |
| 6. Chatterbox patches as files | ~2 hours | No |
| 7. CI + release | ~2 hours | Depends on Phase 4 |

**Recommended order:** 1 → 5 → 2 → 3.0 → 3.1-3.3 → 4 → 6 → 7. Phases 1 and 5 together give the biggest single win (installable, no more `~/bin/` coupling) at the lowest risk. Phase 3.0 (prep) can land on `master` before opening the Phase 1 worktree — it's independently valuable and de-risks everything after.

---

## Open questions

1. **Runtime consolidation** — after Phase 1, does `~/bin/` become a set of thin wrappers pointing at the installed package, or do we keep the current `~/bin/` copies and treat this repo as a mirror?
2. **XDG migration** — auto-detect and move legacy `~/bin/*.json` on first run, or require a manual `tts-books-migrate` command?
3. **Whisper backend** — currently `faster-whisper tiny.en`. Any interest in supporting alternatives (whisper.cpp, distil-whisper) via a pluggable backend interface in `quality/whisper_backend.py`?
4. **PyPI publication** — do we want this discoverable, or stay a GitHub-only project?
