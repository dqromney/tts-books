"""Tests for generation/archiving.py — archive_job(), scan_rework_jobs(), restore_for_rework().

All disk I/O uses tmp_path (pytest fixture); no real job directories touched.
"""

import json
import os

import pytest

from tts_books.generation.archiving import archive_job, restore_for_rework, scan_rework_jobs

# ── stub logger ──────────────────────────────────────────────────────────────

class StubLogger:
    def __init__(self):
        self._buf = []

    def info(self, msg):
        self._buf.append(f"[info] {msg}\n")

    def warn(self, msg):
        self._buf.append(f"[warn] {msg}\n")

    def error(self, msg):
        self._buf.append(f"[error] {msg}\n")

    def success(self, msg):
        self._buf.append(f"[ok] {msg}\n")

    def chunk(self, msg):
        self._buf.append(f"[chunk] {msg}\n")

    @property
    def buffer(self):
        return self._buf


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_job_dir(tmp_path, name="abc123", chunks=None, completed=None):
    """Create a minimal job directory under tmp_path."""
    jd = tmp_path / "jobs" / name
    jd.mkdir(parents=True)
    chunks = chunks or ["Hello world.", "Goodbye world."]
    completed = completed or list(range(len(chunks)))
    (jd / "chunks.json").write_text(json.dumps({"chunks": chunks}))
    (jd / "state.json").write_text(json.dumps({
        "text_hash": name,
        "total_chunks": len(chunks),
        "completed": completed,
        "settings": {},
        "output_path": "/tmp/out.wav",
    }))
    # Write dummy WAV placeholders (empty files — archiving doesn't read them)
    for i in completed:
        (jd / f"chunk_{i:06d}.wav").write_bytes(b"")
    return str(jd)


# ── archive_job ──────────────────────────────────────────────────────────────

class TestArchiveJob:
    def test_job_dir_moved_to_archive(self, tmp_path):
        jd = _make_job_dir(tmp_path)
        archive_dir = str(tmp_path / "archive")
        logger = StubLogger()

        dest = archive_job(jd, [], "/tmp/out.wav", archive_dir, logger)

        assert os.path.isdir(dest)
        assert not os.path.exists(jd)   # original gone

    def test_returns_archive_destination(self, tmp_path):
        jd = _make_job_dir(tmp_path)
        archive_dir = str(tmp_path / "archive")
        dest = archive_job(jd, [], "/tmp/out.wav", archive_dir, StubLogger())
        assert os.path.basename(dest) == os.path.basename(jd) or dest.startswith(archive_dir)

    def test_generation_log_written(self, tmp_path):
        logger = StubLogger()
        logger.info("chunk 1 done")
        jd = _make_job_dir(tmp_path)
        archive_dir = str(tmp_path / "archive")

        dest = archive_job(jd, [], "/tmp/out.wav", archive_dir, logger)

        log_path = os.path.join(dest, "generation.log")
        assert os.path.isfile(log_path)
        with open(log_path) as fh:
            content = fh.read()
        assert "chunk 1 done" in content

    def test_no_rework_json_when_no_garbled(self, tmp_path):
        jd = _make_job_dir(tmp_path)
        archive_dir = str(tmp_path / "archive")
        dest = archive_job(jd, [], "/tmp/out.wav", archive_dir, StubLogger())
        assert not os.path.exists(os.path.join(dest, "rework.json"))

    def test_rework_json_written_when_garbled(self, tmp_path):
        garbled = [{"index": 1, "filename": "chunk_000001.wav", "reason": "too short", "text": "Hello."}]
        jd = _make_job_dir(tmp_path)
        archive_dir = str(tmp_path / "archive")

        dest = archive_job(jd, garbled, "/tmp/out.wav", archive_dir, StubLogger())

        rp = os.path.join(dest, "rework.json")
        assert os.path.isfile(rp)
        with open(rp) as fh:
            data = json.loads(fh.read())
        assert data["output_path"] == "/tmp/out.wav"
        assert len(data["rework_chunks"]) == 1
        assert data["rework_chunks"][0]["index"] == 1

    def test_rework_json_has_archived_timestamp(self, tmp_path):
        garbled = [{"index": 0, "filename": "chunk_000000.wav", "reason": "too long", "text": "x"}]
        jd = _make_job_dir(tmp_path)
        archive_dir = str(tmp_path / "archive")
        dest = archive_job(jd, garbled, "/tmp/out.wav", archive_dir, StubLogger())
        with open(os.path.join(dest, "rework.json")) as fh:
            data = json.loads(fh.read())
        assert "archived" in data

    def test_existing_dest_gets_timestamp_suffix(self, tmp_path):
        jd1 = _make_job_dir(tmp_path, name="samehash")
        archive_dir = str(tmp_path / "archive")
        dest1 = archive_job(jd1, [], "/tmp/out1.wav", archive_dir, StubLogger())

        # Recreate a job with the same name and archive again
        jd2 = _make_job_dir(tmp_path, name="samehash")
        dest2 = archive_job(jd2, [], "/tmp/out2.wav", archive_dir, StubLogger())

        assert dest1 != dest2   # second gets a timestamp suffix
        assert os.path.isdir(dest1)
        assert os.path.isdir(dest2)

    def test_archive_dir_created_if_missing(self, tmp_path):
        jd = _make_job_dir(tmp_path)
        archive_dir = str(tmp_path / "deep" / "archive")
        assert not os.path.exists(archive_dir)
        archive_job(jd, [], "/tmp/out.wav", archive_dir, StubLogger())
        assert os.path.isdir(archive_dir)


# ── scan_rework_jobs ─────────────────────────────────────────────────────────

class TestScanReworkJobs:
    def test_missing_dir_returns_empty(self, tmp_path):
        result = scan_rework_jobs(str(tmp_path / "nonexistent"))
        assert result == []

    def test_no_rework_json_returns_empty(self, tmp_path):
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        (archive_dir / "job1").mkdir()
        # No rework.json in job1
        result = scan_rework_jobs(str(archive_dir))
        assert result == []

    def test_finds_pending_rework(self, tmp_path):
        archive_dir = tmp_path / "archive"
        job1 = archive_dir / "abc123"
        job1.mkdir(parents=True)
        rework = {
            "output_path": "/tmp/out.wav",
            "archived": "2026-01-01T00:00:00",
            "rework_chunks": [{"index": 0, "filename": "chunk_000000.wav", "reason": "too short", "text": "x"}],
        }
        (job1 / "rework.json").write_text(json.dumps(rework))

        result = scan_rework_jobs(str(archive_dir))
        assert len(result) == 1
        assert result[0][0] == str(job1)
        assert result[0][1]["rework_chunks"][0]["index"] == 0

    def test_ignores_empty_rework_chunks(self, tmp_path):
        archive_dir = tmp_path / "archive"
        job1 = archive_dir / "job1"
        job1.mkdir(parents=True)
        rework = {"output_path": "/tmp/out.wav", "rework_chunks": []}
        (job1 / "rework.json").write_text(json.dumps(rework))

        result = scan_rework_jobs(str(archive_dir))
        assert result == []

    def test_ignores_corrupt_rework_json(self, tmp_path):
        archive_dir = tmp_path / "archive"
        job1 = archive_dir / "job1"
        job1.mkdir(parents=True)
        (job1 / "rework.json").write_text("not valid json{{{")

        result = scan_rework_jobs(str(archive_dir))
        assert result == []

    def test_multiple_rework_jobs_sorted(self, tmp_path):
        archive_dir = tmp_path / "archive"
        for name in ("zzz", "aaa", "mmm"):
            d = archive_dir / name
            d.mkdir(parents=True)
            rework = {
                "output_path": "/tmp/out.wav",
                "rework_chunks": [{"index": 0, "filename": "chunk_000000.wav", "reason": "x", "text": "y"}],
            }
            (d / "rework.json").write_text(json.dumps(rework))

        result = scan_rework_jobs(str(archive_dir))
        names = [os.path.basename(r[0]) for r in result]
        assert names == sorted(names)


# ── restore_for_rework ───────────────────────────────────────────────────────

class TestRestoreForRework:
    def _make_archive(self, tmp_path, garbled_indices=(1,)):
        """Create a minimal archive directory with rework.json and state.json."""
        arc = tmp_path / "archive" / "abc123"
        arc.mkdir(parents=True)

        chunks = ["First chunk.", "Second chunk.", "Third chunk."]
        completed = list(range(len(chunks)))

        (arc / "chunks.json").write_text(json.dumps({"chunks": chunks}))
        (arc / "state.json").write_text(json.dumps({
            "text_hash": "abc123",
            "total_chunks": len(chunks),
            "completed": completed,
            "settings": {},
            "output_path": "/tmp/out.wav",
        }))
        (arc / "generation.log").write_text("some log\n")

        rework = {
            "output_path": "/tmp/out.wav",
            "archived": "2026-01-01T00:00:00",
            "rework_chunks": [
                {"index": i, "filename": f"chunk_{i:06d}.wav", "reason": "too short", "text": chunks[i]}
                for i in garbled_indices
            ],
        }
        (arc / "rework.json").write_text(json.dumps(rework))

        # Write dummy WAV files for all completed chunks
        for i in completed:
            (arc / f"chunk_{i:06d}.wav").write_bytes(b"RIFF")

        return str(arc)

    def test_returns_dest_state_text(self, tmp_path):
        arc = self._make_archive(tmp_path, garbled_indices=(1,))
        jobs_dir = str(tmp_path / "jobs")
        dest_jd, state, text = restore_for_rework(arc, jobs_dir)
        assert os.path.isdir(dest_jd)
        assert isinstance(state, dict)
        assert isinstance(text, str)

    def test_archive_still_intact_after_restore(self, tmp_path):
        arc = self._make_archive(tmp_path, garbled_indices=(1,))
        jobs_dir = str(tmp_path / "jobs")
        restore_for_rework(arc, jobs_dir)
        # Archive must NOT be deleted by restore_for_rework
        assert os.path.isdir(arc)

    def test_garbled_wav_removed_from_copy(self, tmp_path):
        arc = self._make_archive(tmp_path, garbled_indices=(1,))
        jobs_dir = str(tmp_path / "jobs")
        dest_jd, _, _ = restore_for_rework(arc, jobs_dir)
        # chunk_000001.wav must be gone from the copy
        assert not os.path.exists(os.path.join(dest_jd, "chunk_000001.wav"))
        # chunk_000000.wav and chunk_000002.wav must still be present
        assert os.path.exists(os.path.join(dest_jd, "chunk_000000.wav"))
        assert os.path.exists(os.path.join(dest_jd, "chunk_000002.wav"))

    def test_state_json_patched_to_remove_garbled(self, tmp_path):
        arc = self._make_archive(tmp_path, garbled_indices=(1,))
        jobs_dir = str(tmp_path / "jobs")
        dest_jd, state, _ = restore_for_rework(arc, jobs_dir)
        assert 1 not in state["completed"]
        assert 0 in state["completed"]
        assert 2 in state["completed"]

    def test_rework_json_removed_from_copy(self, tmp_path):
        arc = self._make_archive(tmp_path, garbled_indices=(1,))
        jobs_dir = str(tmp_path / "jobs")
        dest_jd, _, _ = restore_for_rework(arc, jobs_dir)
        assert not os.path.exists(os.path.join(dest_jd, "rework.json"))

    def test_generation_log_removed_from_copy(self, tmp_path):
        arc = self._make_archive(tmp_path, garbled_indices=(1,))
        jobs_dir = str(tmp_path / "jobs")
        dest_jd, _, _ = restore_for_rework(arc, jobs_dir)
        assert not os.path.exists(os.path.join(dest_jd, "generation.log"))

    def test_text_reconstituted_from_chunks(self, tmp_path):
        arc = self._make_archive(tmp_path, garbled_indices=(1,))
        jobs_dir = str(tmp_path / "jobs")
        _, _, text = restore_for_rework(arc, jobs_dir)
        assert "First chunk" in text
        assert "Second chunk" in text

    def test_raises_on_missing_rework_json(self, tmp_path):
        # Make a directory without rework.json
        arc = tmp_path / "archive" / "nojob"
        arc.mkdir(parents=True)
        with pytest.raises(RuntimeError, match="rework.json"):
            restore_for_rework(str(arc), str(tmp_path / "jobs"))

    def test_existing_dest_overwritten(self, tmp_path):
        arc = self._make_archive(tmp_path, garbled_indices=(1,))
        jobs_dir = str(tmp_path / "jobs")
        # First restore
        restore_for_rework(arc, jobs_dir)
        # Second restore should succeed (overwrites)
        dest_jd, _, _ = restore_for_rework(arc, jobs_dir)
        assert os.path.isdir(dest_jd)
