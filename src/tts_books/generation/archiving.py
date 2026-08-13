"""Job archiving and rework-job scanning.

archive_job() moves a completed job to the archive directory, saves the
generation log, and writes rework.json if any chunks were garbled.

scan_rework_jobs() returns a list of archive directories that have pending
rework.json files.

restore_for_rework() is the disk-manipulation half of the rework flow:
copy-archive → remove-garbled-wavs → patch-state.json. The GUI tail
(_startup_job assignment, text_area fill, _start_resume call) stays in gui.py
because it touches tkinter widgets.

IMPORTANT: restore_for_rework() does NOT delete the archive entry — that is
done in gui.py AFTER the function returns successfully, so the order is:
  1. restore_for_rework() — copy + patch (archive still intact)
  2. gui.py deletes the archive entry
  3. gui.py calls _start_resume()
If the pipeline fails after step 2 but before step 3 the job is still in
jobs/. Re-running Process Rework re-does step 1 safely.
"""

import json
import os
import shutil
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tts_books.gui import Logger


def archive_job(
    job_dir: str,
    garbled_chunks: list[dict],
    output_path: str,
    archive_dir: str,
    logger: "Logger",
) -> str:
    """Move completed job dir to archive_dir/, save log, write rework.json if needed.

    Returns the archive destination path.
    """
    os.makedirs(archive_dir, exist_ok=True)
    dest = os.path.join(archive_dir, os.path.basename(job_dir))
    if os.path.exists(dest):
        dest = dest + "_" + datetime.now().strftime("%Y%m%d%H%M%S")
    shutil.move(job_dir, dest)

    log_path = os.path.join(dest, "generation.log")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(logger.buffer)
    except Exception:
        pass

    if garbled_chunks:
        rework = {
            "output_path": output_path,
            "archived": datetime.now().isoformat(),
            "rework_chunks": garbled_chunks,
        }
        rework_path = os.path.join(dest, "rework.json")
        with open(rework_path, "w", encoding="utf-8") as f:
            json.dump(rework, f, indent=2, ensure_ascii=False)
        logger.warn(f"⚠ {len(garbled_chunks)} chunk(s) need rework → {rework_path}")

    logger.info(f"Job archived → {dest}")
    return dest


def scan_rework_jobs(archive_dir: str) -> list[tuple[str, dict]]:
    """Return [(archive_subdir, rework_info), …] for all pending rework jobs."""
    jobs = []
    if not os.path.isdir(archive_dir):
        return jobs
    for name in sorted(os.listdir(archive_dir)):
        rp = os.path.join(archive_dir, name, "rework.json")
        if os.path.exists(rp):
            try:
                with open(rp, encoding="utf-8") as f:
                    info = json.load(f)
                if info.get("rework_chunks"):
                    jobs.append((os.path.join(archive_dir, name), info))
            except Exception:
                pass
    return jobs


def restore_for_rework(arc_dir: str, jobs_dir: str) -> tuple[str, dict, str] | None:
    """Copy archive to jobs_dir, remove garbled WAVs, patch state.json.

    Returns (dest_job_dir, patched_state, reconstituted_text) on success, or
    raises RuntimeError with a user-facing message on failure.

    DOES NOT delete the archive — the caller must do that after a successful
    return, before calling _start_resume(), per Risk #6 ordering.
    """
    rp = os.path.join(arc_dir, "rework.json")
    try:
        with open(rp, encoding="utf-8") as f:
            rework_info = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Cannot read rework.json:\n{exc}") from exc

    job_id = os.path.basename(arc_dir)
    dest_jd = os.path.join(jobs_dir, job_id)
    bad_idxs = {c["index"] for c in rework_info.get("rework_chunks", [])}

    if os.path.exists(dest_jd):
        shutil.rmtree(dest_jd, ignore_errors=True)
    shutil.copytree(arc_dir, dest_jd)

    for idx in bad_idxs:
        bad_wav = os.path.join(dest_jd, f"chunk_{idx:06d}.wav")
        if os.path.exists(bad_wav):
            os.remove(bad_wav)

    for fname in ("rework.json", "generation.log"):
        p = os.path.join(dest_jd, fname)
        if os.path.exists(p):
            os.remove(p)

    state_path = os.path.join(dest_jd, "state.json")
    try:
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        state["completed"] = [i for i in state.get("completed", []) if i not in bad_idxs]
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as exc:
        shutil.rmtree(dest_jd, ignore_errors=True)
        raise RuntimeError(f"Cannot patch state.json:\n{exc}") from exc

    chunks_file = os.path.join(dest_jd, "chunks.json")
    try:
        with open(chunks_file, encoding="utf-8") as f:
            chunks_data = json.load(f)
        text = "\n".join(chunks_data["chunks"])
    except Exception as exc:
        raise RuntimeError(f"Cannot read chunks.json:\n{exc}") from exc

    return dest_jd, state, text
