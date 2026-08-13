"""Behavior-preserving verification for Phase 3.0 prep refactors.

Covers CancelToken, Settings, and BatchItem -- introduced by commits
0ca1624, f4fae4a, b9a2e95. These classes are all pure Python (threading,
dataclasses); no tkinter needed. Runs against tts_books.gui for now
(they move to standalone modules in Phase 3.1).
"""

import threading
import time
from dataclasses import fields as dataclass_fields

from tts_books.gui import BatchItem, CancelToken, Settings

# ── CancelToken ───────────────────────────────────────────────────

class TestCancelToken:
    def test_initial_state_is_running(self):
        tok = CancelToken()
        assert tok.cancelled is False
        assert tok.is_paused() is False

    def test_cancel_sets_flag_and_unblocks_pause(self):
        tok = CancelToken()
        tok.pause()
        assert tok.is_paused() is True
        tok.cancel()
        assert tok.cancelled is True
        # cancel MUST unblock any pause so waiters can exit
        assert tok.is_paused() is False

    def test_reset_clears_cancel_and_resumes(self):
        tok = CancelToken()
        tok.cancel()
        tok.pause()  # simulate a stuck state
        tok.reset()
        assert tok.cancelled is False
        assert tok.is_paused() is False

    def test_pause_then_resume(self):
        tok = CancelToken()
        tok.pause()
        assert tok.is_paused() is True
        tok.resume()
        assert tok.is_paused() is False

    def test_wait_if_paused_returns_immediately_when_running(self):
        tok = CancelToken()
        start = time.monotonic()
        tok.wait_if_paused()
        assert time.monotonic() - start < 0.05

    def test_wait_if_paused_blocks_until_resume(self):
        tok = CancelToken()
        tok.pause()
        released = threading.Event()

        def waiter():
            tok.wait_if_paused()
            released.set()

        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        # Give the waiter time to enter wait state
        time.sleep(0.05)
        assert not released.is_set(), "waiter should still be blocked"
        tok.resume()
        assert released.wait(timeout=1.0), "waiter should be released"

    def test_wait_if_paused_unblocked_by_cancel(self):
        """Cancel must wake up a paused waiter -- otherwise cancel hangs."""
        tok = CancelToken()
        tok.pause()
        released = threading.Event()

        def waiter():
            tok.wait_if_paused()
            released.set()

        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        time.sleep(0.05)
        assert not released.is_set()
        tok.cancel()
        assert released.wait(timeout=1.0), "cancel should wake paused waiter"


# ── Settings ──────────────────────────────────────────────────────

class TestSettings:
    def test_default_construction_has_all_17_fields(self):
        names = {f.name for f in dataclass_fields(Settings)}
        assert len(names) == 17

    def test_to_dict_round_trip_preserves_values(self):
        s = Settings(temperature=0.5, seed=42, ref_path="/tmp/x.wav")
        d = s.to_dict()
        s2 = Settings.from_dict(d)
        assert s2 == s

    def test_from_dict_none_returns_defaults(self):
        s = Settings.from_dict(None)
        assert s.temperature == 0.8
        assert s.seed == 0
        assert s.ref_path is None

    def test_from_dict_empty_returns_defaults(self):
        assert Settings.from_dict({}) == Settings()

    def test_from_dict_partial_merges_with_defaults(self):
        s = Settings.from_dict({"temperature": 0.5, "seed": 7})
        assert s.temperature == 0.5
        assert s.seed == 7
        assert s.top_p == 0.95  # unspecified -> default

    def test_from_dict_ignores_unknown_keys(self):
        """Older queue/state files with removed fields must still load."""
        d = {"temperature": 0.5, "obsolete_field": "removed"}
        s = Settings.from_dict(d)
        assert s.temperature == 0.5

    def test_to_dict_returns_all_fields(self):
        d = Settings().to_dict()
        expected = {f.name for f in dataclass_fields(Settings)}
        assert set(d.keys()) == expected

    def test_ref_path_is_serialized(self):
        """ref_path is intentionally skipped by _apply_settings_to_ui but
        must survive the wire (batch item uses it directly)."""
        s = Settings(ref_path="/voices/narrator.wav")
        assert s.to_dict()["ref_path"] == "/voices/narrator.wav"


# ── BatchItem ─────────────────────────────────────────────────────

class TestBatchItem:
    def test_default_construction_has_all_17_fields(self):
        names = {f.name for f in dataclass_fields(BatchItem)}
        assert len(names) == 17

    def test_leading_underscore_fields_serialize(self):
        """_job_hash, _eta_str, _text_sidecar must round-trip through JSON."""
        item = BatchItem(id=1, _job_hash="abc123", _eta_str="2m 30s")
        d = item.to_dict()
        assert d["_job_hash"] == "abc123"
        assert d["_eta_str"] == "2m 30s"
        assert d["_text_sidecar"] is None

    def test_settings_default_is_independent_dict(self):
        """The mutable-default trap -- two BatchItems must not share a
        settings dict."""
        a = BatchItem(id=1)
        b = BatchItem(id=2)
        a.settings["temperature"] = 0.5
        assert "temperature" not in b.settings

    def test_from_dict_partial_merges_with_defaults(self):
        d = {"id": 5, "file_name": "book.txt", "status": "processing"}
        item = BatchItem.from_dict(d)
        assert item.id == 5
        assert item.file_name == "book.txt"
        assert item.status == "processing"
        assert item.chunks_done == 0  # default
        assert item.error == ""       # default

    def test_from_dict_ignores_unknown_keys(self):
        """Old queue.json files with removed fields must still load."""
        d = {"id": 1, "voice_path": "/legacy/path.wav"}
        item = BatchItem.from_dict(d)
        assert item.id == 1
        # legacy 'voice_path' silently dropped (only ref_path is known)
        assert not hasattr(item, "voice_path")

    def test_construction_defaults_match_dict_get_defaults(self):
        """The dataclass defaults must match what existing item.get(k, X)
        calls in the file use as fallback -- otherwise behavior changes
        for new items."""
        item = BatchItem(id=1)
        d = item.to_dict()
        # These correspond to the .get() calls surveyed in the refactor audit
        assert d["chunks_done"] == 0
        assert d["chunks_total"] == 0
        assert d["_eta_str"] == ""
        assert d["gen_time"] == ""
        assert d["status"] == "pending"
        assert d["output_path"] == ""
        assert d["error"] == ""

    def test_round_trip_all_fields(self):
        item = BatchItem(
            id=7,
            source_path="/books/x.txt",
            file_name="x.txt",
            text="Hello world.",
            ref_path="/voices/n.wav",
            voice_name="n",
            settings={"temperature": 0.5},
            status="processing",
            output_path="/out/x.wav",
            error="",
            chunks_done=3,
            chunks_total=10,
            gen_time="1m 5s",
            _job_hash="deadbeef",
            _eta_str="45s",
            _text_sidecar="/tmp/x.txt",
            try_resume=True,
        )
        d = item.to_dict()
        item2 = BatchItem.from_dict(d)
        assert item2 == item
