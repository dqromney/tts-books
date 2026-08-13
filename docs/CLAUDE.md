# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal toolbox for CPU-only text-to-speech audiobook generation using **Chatterbox Turbo TTS**. The machine is a ThinkPad P51 (Xeon E3-1505M v6, 62 GB RAM). The Quadro M2200 GPU has only 4 GB VRAM — too small for the model — so all inference runs on CPU. Every script forces `CUDA_VISIBLE_DEVICES=""`.

The virtual environment lives at `~/chatterbox-venv/`. All Python tools must be run inside it.

## Launching the apps

```bash
./scripts/tts-book.sh        # tkinter desktop GUI (primary tool)
./scripts/start-tts.sh       # Gradio web interface (alternative)
```

Both scripts activate the venv and exec the installed console entry points (`tts-book`, `tts-book-web`) — see Phase 1/5 of `REFACTOR-PLAN.md`. Package must be installed editable first: `pip install -e .` inside the venv.

## Key files

| File | Role |
|------|------|
| `src/tts_books/gui.py` | Main tkinter GUI — single Python file, ~2000+ lines |
| `src/tts_books/gradio_app.py` | Gradio web alternative — simpler, no resume support |
| `~/.config/tts-books/app.json` | JSON config (paths + instance count). Location resolved by `src/tts_books/paths.py`; honours `$XDG_CONFIG_HOME` |
| `~/.config/tts-books/queue.json` | Persisted batch queue |
| `~/.config/tts-books/pronunciation.json` | Pronunciation substitution dictionary (word→replacement); created on first Add |
| `scripts/voice-sample` | Bash script: extracts voice clips from YouTube via yt-dlp |
| `scripts/tts-enhance.sh` | ffmpeg EQ + compression + WAV→MP3 conversion |
| `scripts/wav2mp3.sh` | Bare WAV→MP3 at 192k (no processing) |
| `scripts/wav-join.sh` | Concatenates multiple WAV files via ffmpeg concat demuxer |
| `src/tts_books/scrapers/mother_of_learning.py` | Royal Road chapter scraper — Mother of Learning |
| `src/tts_books/scrapers/zenith_of_sorcery.py` | Royal Road chapter scraper — Zenith of Sorcery |

Voice reference WAVs live in `~/voice-samples/` (canonical) and in `~/bin/*.wav` (legacy copies).  
Books (plain text) live in `~/books/`.  
Generated audio lands in `~/tts_output/`.

## gui.py architecture

### Thread model
All TTS generation runs in a daemon thread (`_generate_thread` → `_do_generate`). The main thread is tkinter's event loop. Cross-thread UI updates **must** use `root.after(0, lambda: ...)` — never call tkinter widgets directly from the generation thread.

### Chunk/resume system
Text is split by `split_text()` into chunks (default 200 chars, configurable). Each chunk is generated and saved individually to `~/tts_output/jobs/<md5hash>/chunk_NNNNNN.wav`. A `state.json` checkpoint is written after each chunk. On completion, chunks are concatenated with 0.3 s silence between them and the job directory is **archived** to `~/tts_output/archive/<hash>/` (not deleted). A `generation.log` is saved to the archive. If any chunks remained garbled after all retries, a `rework.json` listing those chunks is also written.

Resume works by two-step lookup:
1. MD5 hash of current text → job directory (normal path)
2. `_startup_job` fallback (set at startup by `_scan_partial_jobs`) for cases where reconstructed text hashes differently than the original

**Critical**: when resuming, always use `state["chunks"]` from disk — never re-split the text. Re-splitting produces different chunk boundaries and misaligns the completed-set indices.

### Config system
`app.json` (path resolved by `src/tts_books/paths.py`; defaults to `~/.config/tts-books/app.json` or `$XDG_CONFIG_HOME/tts-books/app.json`) stores directory paths and an `instance_count`. The count is incremented at launch and decremented at clean exit via `WM_DELETE_WINDOW`. The Settings dialog reads the live count from disk (not memory) to detect concurrent instances and block changes when >1 is open. Pre-Phase-2 files at `~/bin/tts-book.config` are auto-copied to the XDG location on first launch by `paths.migrate_legacy()`; the `~/bin/` copies stay in place as a transition-period safety net.

### Batch queue persistence
`queue.json` (alongside `app.json` under `$XDG_CONFIG_HOME/tts-books/`) saves the batch queue to disk on every add/remove/clear/status-change. On startup, `_load_batch_queue()` restores the queue. Items stuck in "processing" state (from a crash) are reset to "pending". The `_batch_next_id` counter is persisted so IDs don't collide across restarts.

Add/Remove/Clear buttons (`batch_add_btn`, `batch_remove_btn`, `batch_clear_btn`) are stored as instance vars and remain **enabled during batch runs** to allow live queue management. `_remove_from_batch()` skips items with `status == "processing"`. `_clear_batch()` only removes non-processing items.

### Auto MP3 conversion
The Advanced Options panel has an "Auto-convert to MP3" checkbox (`self.auto_mp3`). When checked, every completed generation (single-file or batch item) runs `ffmpeg -b:a 192k` on the output WAV. The MP3 lands alongside the WAV with the same stem. Silent on failure — logs to the detail pane.

### UI locking
`self._lockable` is a list of all input widgets populated during `_build_ui()`. `_lock_ui()` disables everything in that list plus the voice combobox. `_unlock_ui()` re-enables them, restoring the combobox to `"readonly"` state. Lock/unlock is called at the start and end of generation (including cancel and error paths).

The **Play Voice button (`play_ref_btn`) is never disabled** during generation — it is not included in `_lockable` and `_lock_ui()` does not touch it. `_play_ref()` does not guard on `_gen_running`.

`self._gen_running` is initialized to `False` in `__init__` (required to avoid AttributeError on VoxCeleb1 download before any generation has started).

Voice selection uses the voice combobox and Browse VoxCeleb1 only. The former "Choose .wav" file-picker button has been removed.

### Text preprocessing
`split_text()` in the module: splits on sentence endings and (optionally) clause endings, groups sentences greedily up to `max_chars`, hard-splits oversized sentences at comma/semicolon/word boundaries. `_SEPARATOR_RE` strips decorative separator lines (`====`, `----`, `****`, etc.) from paragraphs before chunking.

### Pronunciation dictionary
`PRON_DICT_PATH` (imported from `src/tts_books/paths.py`; resolves to `~/.config/tts-books/pronunciation.json` by default) stores word-to-replacement mappings as a flat JSON object. Module-level functions:

- `_load_pron_dict()` — reads the file (returns `{}` if missing)
- `_save_pron_dict(d)` — writes the file atomically
- `apply_pronunciation(text, d)` — applies all substitutions (whole-word, case-insensitive)

`apply_pronunciation()` is called on the full text before `split_text()` in `_do_generate()`, but only for fresh (non-resume) generations.

The "Pronunciation..." button in the controls row opens `_open_pron_dict()` — a Toplevel dialog with a Treeview and Add/Update/Remove controls. This button is **not** in `_lockable`, so the dialog remains accessible during generation.

### Garbled chunk detection and retry
`MAX_CHUNK_RETRIES = 3` (module-level constant).

`_is_chunk_garbled(wav_path, text)` runs ten checks in order (the function's own docstring is stale — it says "four" but the body has ten):

1. **Duration heuristic** — flags if `duration < len(text) / 25.0` (cutoff/silence) or `duration > len(text) / 3.5` (repetition loop).
2. **Word overlap** (faster-whisper tiny.en) — fraction of expected words appearing in transcription; < 45% = garbled. Hyphenated words are excluded from the expected set because Whisper transcribes the *sound* of a pronunciation substitution ("ay-bish"), not its spelling.
3. **Word-count ratio** (fallback) — transcribed words / expected words < 0.5 = garbled.
4. **Unrecognized tail** — last transcribed word ends > 2.5 s before audio ends (whisper stopped, noise follows).
5. **Tail word check** — last 6 transcribed words < 35% match against expected word set (whisper transcribed garble as wrong words).
6. **Word stutter** — consecutive identical tokens in transcription not expected to repeat consecutively in the source.
7. **Final content word absent** — the last unhyphenated ≥5-char word in the source doesn't appear anywhere in the transcript, unless its vowel ratio suggests Whisper can't spell it (proper nouns, archaic terms).
8. **Phrase repetition loop** — a 5-gram in the transcription appears twice but never appears in the source (model looped a clause).
9. **Anaphoric over-repetition** — a 5-gram appears more times in the transcription than in the source (catches loops of a legitimately-repeating phrase like Holland-style anaphora that check 8 misses).
10. **Single-source phrase spoken twice** — a 6-gram appears exactly once in the source but two or more times in the transcription (classic "the road went through the dust, as all roads went through the dust…" loop that checks 8 and 9 both miss).

After each chunk is saved, a retry loop re-generates with a fresh randomized `torch.manual_seed()` if flagged as garbled, up to `MAX_CHUNK_RETRIES` times. **The initial generation is saved as a `.best` backup before retrying.** If a retry comes back clean it wins; if all retries are also garbled, the `.best` (first/most-confident) attempt is restored. Chunks that exhaust all retries are recorded in `garbled_chunks` and written to `rework.json` in the archive.

### Archive and rework system
`ARCHIVE_DIR = ~/tts_output/archive/` (module-level constant, also stored as `self._archive_dir`).

When a job completes successfully, `_archive_job(jd, garbled_chunks, output_path)` moves the job directory to `~/tts_output/archive/<hash>/`, writes `generation.log` (full plain-text log buffer), and writes `rework.json` if any chunks were garbled after all retries.

`rework.json` format:
```json
{
  "output_path": "/home/.../abish-v3.wav",
  "archived": "2026-07-01T09:00:00",
  "rework_chunks": [
    {"index": 7, "filename": "chunk_000007.wav", "reason": "STT: tail garbled (25%)", "text": "…"}
  ]
}
```

The **Rework… button** (Single File controls row) shows a pending count (`Rework (2)…`) when archived jobs have unresolved garbled chunks. Clicking it opens a dialog listing all such jobs. Selecting one and clicking **Process Rework** copies the archive back to `jobs/`, removes the garbled WAV files, patches `state.json` to remove their indices from `completed`, deletes the archive entry, and resumes generation — regenerating only the missing chunks then re-concatenating everything. On completion the job is re-archived cleanly.

`_log_buffer` (instance list) captures every log line as plain text so `_archive_job` can write `generation.log` without accessing the tkinter widget from a background thread.

### Settings sync
`_apply_settings_to_ui(settings)` is called in three places to keep the Advanced Options panel in sync:

- **Partial job restore** (`_restore_partial_job`) — Advanced Options update immediately when a partial job is found at startup or selected from the picker.
- **Resume confirmation** (`_start_generate`) — settings update before the UI locks when the user clicks Yes to resume a found partial job.
- **Batch item click** (`_on_batch_select`, bound to `<<TreeviewSelect>>` on the batch Treeview) — clicking any batch queue row reflects that item's settings and voice in the UI.
- **Batch processing** — already called at the start of each batch item (unchanged).

### Memory management

Three-tier strategy to prevent glibc malloc arena fragmentation from exhausting 62 GB RAM across hundreds of chunks:

**Tier 1 — glibc tuning (tts-book.sh):** `MALLOC_MMAP_THRESHOLD_=131072` makes glibc use `mmap` for allocations > 128 KB so each large tensor can be freed individually to the OS rather than held in a thread-local arena. `MALLOC_TRIM_THRESHOLD_=65536` makes `malloc_trim` more aggressive.

**Tier 2 — jemalloc (tts-book.sh):** If `libjemalloc2` is installed (`sudo apt install libjemalloc2`), the launch script `LD_PRELOAD`s jemalloc, which handles fragmentation vastly better than glibc. Falls back silently if not installed.

**Tier 3 — periodic model reload (GUI checkbox):** The Advanced Options panel has a "Reload model every N chunks" spinbox (0 = disabled). When set (e.g., 100), the model is destroyed and recreated every N chunks during generation. This dumps all model memory at once, giving the allocator a clean heap. Costs ~30s per reload. Best used as a last resort if tiers 1+2 aren't enough.

**Per-chunk (unchanged):** `gc.collect()` + `ctypes.CDLL("libc.so.6").malloc_trim(0)` after each chunk. Whisper model unloaded/reloaded every 25 chunks. Auto-prune when free RAM drops below 8 GB.

### Crossfade stitching

Module-level `_crossfade_chunks(wavs, sr, crossfade_ms=50)` replaces silence-gap concatenation with equal-power crossfades. Each junction overlaps `crossfade_ms` of audio, applying cos² fade-out to the outgoing chunk and sin² fade-in to the incoming chunk, then summing them. This eliminates the audible clicks/pops of hard silence-gap joins and produces seamless audiobook narration.

Controlled by the "Crossfade (ms, 0=off)" spinbox in Advanced Options (default 50 ms). Set to 0 to revert to the old 0.3s silence-gap concatenation.

### DC offset removal

`_remove_dc_offset(audio, sr)` applies a 2nd-order high-pass Butterworth filter at 15 Hz (zero-phase via `filtfilt`) to the final concatenated output before saving. This strips out any DC bias introduced per-chunk by the TTS model, which otherwise causes low-frequency thumps at chunk boundaries. Requires scipy (already in the venv); silently passes through if scipy is unavailable.

### Voice combobox + VoxCeleb1
`_populate_voice_combo()` scans `self._voice_samples_dir` for `*.wav` files and populates a `ttk.Combobox`. `_browse_dataset()` opens a Toplevel that downloads `metadata.csv` from `sdialog/voices-voxceleb1` on HuggingFace Hub (using `hf_hub_download` from `huggingface_hub` v1.19+; `datasets` is **not** installed). Downloads individual speaker WAVs on demand and copies to `voice_samples_dir`.

## Chatterbox library patches

These files under `~/chatterbox-venv/lib/python3.12/site-packages/chatterbox/` have been patched and differ from the upstream release. Do not overwrite them with a package upgrade without re-applying these changes.

Patch files live in `patches/` in this repo. After a `pip install --upgrade chatterbox-tts` run:
```bash
./patches/apply.sh            # defaults to ~/chatterbox-venv
./patches/apply.sh /path/to/other-venv
```
The script is idempotent — it skips hunks that are already applied.

### `tts_turbo.py` — `generate()`

- `prepare_conditionals()` is called **once per job** (before the chunk loop in `src/tts_books/gui.py`). Each subsequent chunk call passes `audio_prompt_path=None` so the cached `self.conds` is reused without re-loading the reference WAV on every chunk.
- `generate()` now accepts two new keyword arguments: `max_gen_len=None` (defaults internally to `max(2000, len(text)*25)`) and `no_repeat_ngram_size=6`. Both are forwarded to `inference_turbo()`. (Lowered from 8 to 6 on 2026-07-23 after observing 6-token phrase loops slipping past the 8-gram guard.)

### `models/t3/t3.py` — `inference_turbo()`

- Default value of `max_gen_len` raised from **1000 → 4000** to prevent truncated output on longer chunks.
- `NoRepeatNGramLogitsProcessor(no_repeat_ngram_size)` added to the logits processor list. This prevents speech-token n-gram repetitions that caused phrase echoing in longer passages. Default n=6 (was 8 pre-2026-07-23).

## Royal Road chapter capture pattern

`src/tts_books/scrapers/mother_of_learning.py` and `zenith_of_sorcery.py` share the same structure: `fetch()` → `extract()` → follow `next_url`. For ad-hoc capture of a new fiction, copy either file, change `start_url` and `output_file`, and run directly (or invoke via the installed entry points `capture-mol` / `capture-zos`). For sites that return 403 to simple requests, use `curl` with Firefox User-Agent headers piped to a BeautifulSoup parser (see the Unconquered Tower session for the exact headers).

Chapter separators in scraped text (`='*60`) are automatically skipped by `_SEPARATOR_RE` in `split_text()`.

## Running in WSL (Windows Subsystem for Linux)

Tkinter requires an X11 display server. Two approaches:

**WSLg (WSL2 + Windows 11 / recent Win10 — automatic):** WSLg is a built-in Wayland/X11 compositor; GUI apps and audio (`paplay`) work with no setup. Verify with `echo $DISPLAY` (should print `:0`) and `ls /mnt/wslg/`.

**Third-party X server (older WSL or WSL1):** Install VcXsrv, X410, or MobaXterm on Windows, then set in WSL:
```bash
export DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0
```
Add to `~/.bashrc` to persist. Audio requires a separate PulseAudio bridge to Windows (WSLg handles this automatically).

`CUDA_VISIBLE_DEVICES=""` in `tts-book.sh` is already correct for WSL.

## Dependencies (inside ~/chatterbox-venv)

- `chatterbox` — Chatterbox Turbo TTS model (patched — see above)
- `torch`, `torchaudio` — inference + audio I/O
- `fitz` (PyMuPDF) — PDF text extraction
- `ebooklib`, `beautifulsoup4` — EPUB extraction
- `huggingface_hub` — VoxCeleb1 dataset browser
- `gradio` — web UI alternative
- `ffmpeg` (system) — audio post-processing, WAV joining, MP3 conversion
- `paplay` (system, PipeWire/PulseAudio) — in-app audio playback
- `yt-dlp` (system) — voice sample extraction from YouTube
