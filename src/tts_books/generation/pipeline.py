"""Generation pipeline — pure disk state functions and do_generate().

do_generate() is the core TTS generation loop. It runs in a daemon thread.
UI updates use callback functions rather than tkinter imports; the caller
wraps each callback in root.after(0, ...) so widget access stays on the
main thread.

Model state is passed via model_box: list (a single-element list holding the
model or None). The pipeline mutates model_box[0] in place on lazy load and
periodic reload; the caller must sync self.model = model_box[0] after return.

Process-global side effects: sets os.environ OMP_NUM_THREADS, MKL_NUM_THREADS,
and calls torch.set_num_threads. Only one concurrent caller is safe — the App's
_gen_running flag enforces this.
"""

from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import os
import re
import shutil
import statistics
import time
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

import torch as th
import torchaudio
from chatterbox.tts_turbo import ChatterboxTurboTTS

from tts_books.config import _PRUNE_THRESHOLD_GB, MAX_CHUNK_RETRIES
from tts_books.generation.archiving import archive_job
from tts_books.generation.chunking import split_text
from tts_books.generation.stitching import crossfade_chunks, remove_dc_offset
from tts_books.memory.pruning import mem_rss_mb
from tts_books.pronunciation import apply_pronunciation, load_pron_dict
from tts_books.quality.garbled import is_chunk_garbled

if TYPE_CHECKING:
    from tts_books.gui import Logger
    from tts_books.quality.whisper_backend import WhisperHandle

DEVICE = "cpu"


# ── Pure state functions ──────────────────────────────────────────────────────


def text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def job_dir(text: str, jobs_dir: str) -> str:
    return os.path.join(jobs_dir, text_hash(text))


def state_path(text: str, jobs_dir: str) -> str:
    return os.path.join(job_dir(text, jobs_dir), "state.json")


def find_resumable_job(
    text: str,
    jobs_dir: str,
    logger: Logger,
    _hint_hash: str | None = None,
) -> tuple[str, dict] | tuple[None, None]:
    """Return (job_dir, state) for an incomplete job, or (None, None).

    _hint_hash: directory basename recorded at generation time — tried first
    so the lookup survives pronunciation-dict or source-file drift.
    """

    def _check_dir(jd: str) -> dict | None:
        sp = os.path.join(jd, "state.json")
        if not os.path.exists(sp):
            logger.info(f"  [resume] {jd}: no state.json")
            return None
        try:
            with open(sp) as f:
                state = json.load(f)
            completed = set(state.get("completed", []))
            total = state.get("total_chunks", 0)
            logger.info(
                f"  [resume] {os.path.basename(jd)}: "
                f"completed={len(completed)}/{total}"
            )
            # Include jobs where all chunks done but concat/archive didn't finish
            # (completed == total); <= intentionally catches that case.
            if total > 0 and len(completed) <= total:
                return state
        except Exception as _e:
            logger.warn(f"  [resume] {jd}: read error {_e}")
        return None

    # 1. Hint hash recorded at generation time (most reliable)
    if _hint_hash:
        jd = os.path.join(jobs_dir, _hint_hash)
        logger.info(f"  [resume] hint-hash lookup: {jd}")
        found = _check_dir(jd)
        if found is not None:
            return jd, found

    # 2. Hash of the text as provided (pronunciation already applied by caller)
    jd = job_dir(text, jobs_dir)
    logger.info(f"  [resume] text-hash lookup: {os.path.basename(jd)}")
    found = _check_dir(jd)
    if found is not None:
        return jd, found

    return None, None


def reconcile_job_state(
    job_dir_path: str,
    state: dict,
) -> tuple[dict, int, int, bool]:
    """Compare state.json completed set against chunk WAVs actually on disk.

    Returns (updated_state, disk_count, json_count, was_changed).
    If counts differ, state.json is rewritten to match disk before returning.
    """
    disk_indices: set[int] = set()
    try:
        for fname in os.listdir(job_dir_path):
            m = re.match(r"chunk_(\d+)\.wav$", fname)
            if m:
                disk_indices.add(int(m.group(1)))
    except OSError:
        return state, 0, len(state.get("completed", [])), False

    json_indices = set(state.get("completed", []))
    disk_count = len(disk_indices)
    json_count = len(json_indices)

    if disk_indices == json_indices:
        return state, disk_count, json_count, False

    # Reconcile: disk is truth
    updated = dict(state)
    updated["completed"] = sorted(disk_indices)
    try:
        sp = os.path.join(job_dir_path, "state.json")
        tmp = sp + ".tmp"
        with open(tmp, "w") as f:
            json.dump(updated, f)
        os.replace(tmp, sp)
    except Exception:
        pass
    return updated, disk_count, json_count, True


def save_state(
    text: str,
    chunks: list[str],
    completed: set[int],
    settings: dict,
    output_path: str,
    jobs_dir: str,
    job_dir_path: str | None = None,
) -> None:
    jd = job_dir_path or job_dir(text, jobs_dir)
    os.makedirs(jd, exist_ok=True)
    chunks_path = os.path.join(jd, "chunks.json")
    if not os.path.exists(chunks_path):
        tmp = chunks_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"chunks": chunks}, f)
        os.replace(tmp, chunks_path)
    state = {
        "text_hash": text_hash(text),
        "total_chunks": len(chunks),
        "completed": sorted(completed),
        "settings": settings,
        "output_path": output_path,
        "created": datetime.now().isoformat(),
    }
    tmp = os.path.join(jd, "state.json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, os.path.join(jd, "state.json"))


# ── Core generation pipeline ──────────────────────────────────────────────────


def do_generate(
    text: str,
    settings: dict,
    *,
    jobs_dir: str,
    output_dir: str,
    archive_dir: str,
    model_box: list,
    cancel_token: Any,
    prune_every_fn: Callable[[], int],
    reload_every_fn: Callable[[], int],
    logger: Logger,
    on_status: Callable[[str, str], None],
    on_out_label: Callable[[str, str], None],
    on_progress: Callable[[float, str], None],
    on_batch_update: Callable[[], None],
    on_archive_done: Callable[[], None],
    whisper_handle: WhisperHandle,
    ref_path: str | None = None,
    source_path: str | None = None,
    resume: bool = False,
    job_dir_arg: str | None = None,
    state: dict | None = None,
    batch_item: dict | None = None,
) -> str | None:
    """Core TTS generation loop. Must run in a daemon thread, never the UI thread.

    model_box is a single-element list [model_or_None]. If model_box[0] is None
    the model is loaded lazily. On periodic reload, model_box[0] is set to None
    then back to the fresh model so the caller can sync self.model afterward.

    Chatterbox caching contract: prepare_conditionals() is called exactly once
    before the chunk loop (when ref_path is provided). Every chunk call passes
    audio_prompt_path=None to reuse the cached conditionals without re-reading
    the reference WAV.
    """
    # Lazy load model
    if model_box[0] is None:
        on_status("Loading model…", "black")
        t0 = time.time()
        _rss_before = mem_rss_mb()[0]
        model_box[0] = ChatterboxTurboTTS.from_pretrained(DEVICE)
        _rss_after, _sys_free = mem_rss_mb()
        model_mem_mb = max(0.0, _rss_after - _rss_before)
        logger.success(
            f"Model loaded in {time.time()-t0:.1f}s"
            f" (+{model_mem_mb/1024:.1f} GB RSS, {_sys_free/1024:.1f} GB free)"
        )

    model = model_box[0]

    n_threads = str(settings.get("cpu_threads", 4))
    th.set_num_threads(int(n_threads))
    os.environ["OMP_NUM_THREADS"] = n_threads
    os.environ["MKL_NUM_THREADS"] = n_threads

    seed_val = settings.get("seed", 0)
    if seed_val != 0:
        th.manual_seed(seed_val)

    # Apply pronunciation substitutions only on fresh (non-resume) generations.
    # On resume the text is reconstructed from the saved chunks, so pronunciation
    # was already applied when the job was first created.
    if not resume:
        pron_dict = load_pron_dict()
        if pron_dict:
            text = apply_pronunciation(text, pron_dict)

    # Job directory — must be resolved before loading chunks so the path is
    # available throughout the rest of this function.
    jd = job_dir_arg if (resume and job_dir_arg) else job_dir(text, jobs_dir)
    os.makedirs(jd, exist_ok=True)
    # Record the exact job hash so resume can find it reliably even if the
    # pronunciation dict or source file changes later.
    if batch_item is not None:
        batch_item["_job_hash"] = os.path.basename(jd)

    # Chunks
    if resume and state:
        # Prefer the separate chunks.json (current format); fall back to inline
        # chunks in state (old format), then re-split as last resort.
        chunks_file = os.path.join(jd, "chunks.json")
        if os.path.exists(chunks_file):
            with open(chunks_file) as f:
                chunks = json.load(f)["chunks"]
        elif state.get("chunks"):
            chunks = state["chunks"]
        else:
            chunks = split_text(
                text,
                max_chars=settings.get("chunk_size", 200),
                split_clauses=settings.get("split_clauses", True),
            )
    else:
        chunks = split_text(
            text,
            max_chars=settings.get("chunk_size", 200),
            split_clauses=settings.get("split_clauses", True),
        )
    total = len(chunks)

    logger.info(f"Text split into {total} chunks (max {settings.get('chunk_size', 200)} chars each)")
    if batch_item is not None:
        if batch_item.get("chunks_total", 0) == 0:
            batch_item["chunks_total"] = total
        on_batch_update()

    # Output path
    if state and state.get("output_path"):
        output_path = state["output_path"]
    else:
        os.makedirs(output_dir, exist_ok=True)
        if source_path:
            stem = os.path.splitext(os.path.basename(source_path))[0]
            output_path = os.path.join(output_dir, f"{stem}.wav")
        elif batch_item and batch_item.get("source_path"):
            stem = os.path.splitext(os.path.basename(batch_item["source_path"]))[0]
            output_path = os.path.join(output_dir, f"{stem}.wav")
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"tts_{ts}.wav")

    if resume and state:
        completed: set[int] = set(state.get("completed", []))
        # Reconcile against WAV files actually on disk — an OOM kill can leave
        # state.json behind while more chunks were already written to disk.
        for fname in os.listdir(jd):
            m = re.match(r"chunk_(\d+)\.wav", fname)
            if m:
                completed.add(int(m.group(1)))
        on_out_label(f"Resuming: {len(completed)}/{total} chunks → {output_path}", "blue")
        logger.info(f"Resuming: {len(completed)}/{total} chunks already done")
    else:
        completed = set()
        save_state(text, chunks, completed, settings, output_path, jobs_dir, job_dir_path=jd)

    chunk_times: list[float] = []
    t_start = time.time()
    garbled_chunks: list[dict] = []

    # Cache voice conditionals once for all chunks.
    # Chatterbox caching contract: each chunk call passes audio_prompt_path=None
    # to reuse these cached conditionals without re-reading the reference WAV.
    if ref_path:
        model.prepare_conditionals(ref_path, norm_loudness=settings.get("norm_loudness", True))

    try:
        for i, chunk in enumerate(chunks):
            # Check cancel FIRST, before pause wait
            if cancel_token.cancelled:
                on_out_label(
                    f"Partial: {len(completed)}/{total} chunks saved in {jd}", "orange"
                )
                logger.warn(f"Cancelled at chunk {i+1}/{total}")
                return None

            # Wait if paused
            cancel_token.wait_if_paused()

            # Re-check cancel after un-pausing
            if cancel_token.cancelled:
                on_out_label(
                    f"Partial: {len(completed)}/{total} chunks saved in {jd}", "orange"
                )
                logger.warn(f"Cancelled at chunk {i+1}/{total}")
                return None

            if i in completed:
                continue

            pct = (len(completed) / total) * 100
            on_progress(pct, f"Chunk {i+1}/{total}")

            t_chunk = time.time()
            wav = model.generate(
                chunk,
                audio_prompt_path=None,  # reuses cached conditionals
                temperature=settings["temperature"],
                min_p=settings.get("min_p", 0.0),
                top_p=settings["top_p"],
                top_k=settings["top_k"],
                repetition_penalty=settings["repetition_penalty"],
                norm_loudness=settings.get("norm_loudness", True),
            )
            elapsed = time.time() - t_chunk
            chunk_times.append(elapsed)

            wav = wav.squeeze(0).cpu()
            fpath = os.path.join(jd, f"chunk_{i:06d}.wav")
            torchaudio.save(fpath, wav.unsqueeze(0), model.sr)
            del wav

            # Prune (gc.collect + malloc_trim) — cadence set by prune_every_fn().
            # 1 = after every chunk; 0 = disabled. Read live so mid-run changes
            # take effect immediately.
            prune_every = prune_every_fn()
            if prune_every > 0 and (i + 1) % prune_every == 0:
                gc.collect()
                ctypes.CDLL("libc.so.6").malloc_trim(0)

            # Periodically unload whisper to release ctranslate2 memory arena
            if i > 0 and i % 25 == 0 and whisper_handle.is_loaded:
                whisper_handle.unload()
                gc.collect()
                ctypes.CDLL("libc.so.6").malloc_trim(0)
                logger.info(f"  Whisper reloaded at chunk {i+1} to free ctranslate2 memory")

            # Warn (and prune) when system RAM drops below threshold
            try:
                import psutil

                _avail = psutil.virtual_memory().available
                if _avail < _PRUNE_THRESHOLD_GB * 1024**3:
                    whisper_handle.unload()
                    gc.collect()
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                    _proc_mb, _free_mb = mem_rss_mb()
                    logger.warn(
                        f"  ⚠ Low RAM auto-prune: {_free_mb/1024:.1f} GB free"
                        f", {_proc_mb/1024:.1f} GB RSS"
                    )
            except Exception:
                pass

            garble_reason = is_chunk_garbled(fpath, chunk, whisper_handle, logger)
            if garble_reason:
                # Save the initial attempt as the current best before retrying.
                # Each retry overwrites fpath; we restore best if all retries fail.
                best_fpath = fpath + ".best"
                shutil.copy2(fpath, best_fpath)
                best_reason = garble_reason

                for _retry in range(MAX_CHUNK_RETRIES):
                    logger.warn(
                        f"  ⚠ Chunk {i+1} garbled [{garble_reason}]"
                        f" (retry {_retry+1}/{MAX_CHUNK_RETRIES})…"
                    )
                    th.manual_seed(int(time.time() * 1e6) % (2**31))
                    retry_wav = model.generate(
                        chunk,
                        audio_prompt_path=None,
                        temperature=settings["temperature"],
                        min_p=settings.get("min_p", 0.0),
                        top_p=settings["top_p"],
                        top_k=settings["top_k"],
                        repetition_penalty=settings["repetition_penalty"],
                        norm_loudness=settings.get("norm_loudness", True),
                    )
                    retry_wav = retry_wav.squeeze(0).cpu()
                    torchaudio.save(fpath, retry_wav.unsqueeze(0), model.sr)
                    del retry_wav
                    gc.collect()
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                    garble_reason = is_chunk_garbled(fpath, chunk, whisper_handle, logger)
                    if not garble_reason:
                        # Clean retry — discard the saved best and keep this one
                        os.remove(best_fpath)
                        break
                    # This retry is also garbled; keep whichever attempt was first
                    # (initial generation is the model's most confident output)
                else:
                    # All retries garbled — restore the initial attempt as best
                    shutil.move(best_fpath, fpath)
                    logger.warn(
                        f"  ⚠ Chunk {i+1} still garbled [{best_reason}]"
                        f" after {MAX_CHUNK_RETRIES} retries — keeping first attempt"
                    )
                    garbled_chunks.append(
                        {
                            "index": i,
                            "filename": f"chunk_{i:06d}.wav",
                            "reason": best_reason,
                            "text": chunk,
                        }
                    )
                # Clean up .best file if somehow still present
                if os.path.exists(best_fpath):
                    os.remove(best_fpath)

            completed.add(i)
            save_state(text, chunks, completed, settings, output_path, jobs_dir, job_dir_path=jd)

            if batch_item is not None:
                avg_t = statistics.mean(chunk_times) if chunk_times else 0
                remaining = total - len(completed)
                batch_item["chunks_done"] = len(completed)
                batch_item["chunks_total"] = total
                batch_item["_eta_str"] = (
                    f"~{avg_t * remaining / 60:.0f}m"
                    if avg_t > 0 and remaining > 0
                    else ("done" if remaining == 0 else "")
                )
                on_batch_update()

            # Periodic model reload to defragment memory
            reload_every = reload_every_fn()  # read live, not from snapshot
            if reload_every > 0 and (i + 1) % reload_every == 0 and i + 1 < total:
                t_reload = time.time()
                logger.info(f"Reloading model at chunk {i+1} (every {reload_every})...")
                # Free the old model and whisper FIRST so malloc_trim can return
                # their pages to the OS before we measure free RAM or load fresh.
                model_box[0] = None
                model = None  # type: ignore[assignment]
                whisper_handle.unload()
                for _ in range(3):
                    gc.collect()
                ctypes.CDLL("libc.so.6").malloc_trim(0)
                time.sleep(0.5)  # let OS reclaim mmap'd pages
                _proc, free_mb = mem_rss_mb()
                if free_mb < 10 * 1024:  # < 10 GB free after full cleanup
                    logger.warn(
                        f"  ⚠ Only {free_mb/1024:.1f} GB free after cleanup — "
                        f"model freed but memory critically low; reloading anyway"
                    )
                model_box[0] = ChatterboxTurboTTS.from_pretrained(DEVICE)
                model = model_box[0]
                th.set_num_threads(int(settings.get("cpu_threads", 4)))
                if ref_path:
                    model.prepare_conditionals(
                        ref_path, norm_loudness=settings.get("norm_loudness", True)
                    )
                proc_mb, free_mb = mem_rss_mb()
                logger.info(
                    f"  Model reloaded in {time.time()-t_reload:.0f}s"
                    f" | {proc_mb/1024:.1f} GB RSS, {free_mb/1024:.1f} GB free"
                )

            # Log every 10th chunk or first/last
            if i % 10 == 0 or i == 0 or i == total - 1:
                avg_t = statistics.mean(chunk_times) if chunk_times else 0
                eta = avg_t * (total - len(completed))
                proc_mb, free_mb = mem_rss_mb()
                logger.chunk(
                    f"Chunk {i+1}/{total} ({elapsed:.1f}s, avg {avg_t:.1f}s,"
                    f" ETA {eta/60:.0f}m)"
                    f" | {proc_mb/1024:.1f} GB RSS, {free_mb/1024:.1f} GB free"
                )

        if cancel_token.cancelled:
            on_out_label(
                f"Partial: {len(completed)}/{total} chunks saved in {jd}", "orange"
            )
            logger.warn(f"Cancelled after {len(completed)}/{total} chunks")
            return None

        total_elapsed = time.time() - t_start
        logger.success(f"All {total} chunks generated in {total_elapsed/60:.1f}m")
        if batch_item is not None:
            h, m = divmod(int(total_elapsed / 60), 60)
            batch_item["gen_time"] = f"{h}h {m}m" if h else f"{m}m"
        on_progress(100, "Concatenating…")
        logger.info("Concatenating chunks…")

        chunk_files = [os.path.join(jd, f"chunk_{i:06d}.wav") for i in range(total)]
        sr = model.sr

        crossfade_ms = settings.get("crossfade_ms", 50)
        if crossfade_ms > 0 and total > 1:
            wavs = []
            for fpath in chunk_files:
                w, _ = torchaudio.load(fpath)
                wavs.append(w)
            final = crossfade_chunks(wavs, sr, crossfade_ms=crossfade_ms)
            logger.info(
                f"Crossfaded {total} chunks ({crossfade_ms} ms overlap)"
                f" → {final.shape[1] / sr:.1f}s"
            )
        elif total == 1:
            final, _ = torchaudio.load(chunk_files[0])
        else:
            between_sil = th.zeros(1, int(sr * 0.3))
            waveforms = []
            for fpath in chunk_files:
                w, _ = torchaudio.load(fpath)
                waveforms.append(w)
                waveforms.append(between_sil)
            final = th.cat(waveforms[:-1], dim=1)

        lead_s = settings.get("lead_silence", 0.0)
        trail_s = settings.get("trail_silence", 0.0)
        if lead_s > 0:
            final = th.cat([th.zeros(1, int(sr * lead_s)), final], dim=1)
        if trail_s > 0:
            final = th.cat([final, th.zeros(1, int(sr * trail_s))], dim=1)
        if lead_s > 0 or trail_s > 0:
            logger.info(f"Silence padding: {lead_s}s lead, {trail_s}s trail")

        # Remove DC offset before saving (prevents low-frequency thumps)
        final = remove_dc_offset(final, sr)
        torchaudio.save(output_path, final, sr)
        logger.success(f"Saved: {output_path}")

        archive_job(jd, garbled_chunks, output_path, archive_dir, logger)
        on_archive_done()

        return output_path

    except Exception:
        logger.error(f"Error during generation — job state preserved in {jd}")
        raise
