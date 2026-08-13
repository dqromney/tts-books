"""Tests for quality/garbled.py — is_chunk_garbled() heuristics.

Heuristic 1 (duration) runs without Whisper and is tested here using
synthetic WAV files written to tmp_path. Heuristics 2-10 are STT-based and
require faster-whisper; they are marked @pytest.mark.slow and skipped by
default (run with: pytest -m slow).
"""


import torch as th
import torchaudio

from tts_books.quality.garbled import is_chunk_garbled
from tts_books.quality.whisper_backend import WhisperHandle

SR = 24000


# ── stub logger / whisper ─────────────────────────────────────────────────────

class StubLogger:
    def info(self, msg): pass
    def warn(self, msg): pass
    def error(self, msg): pass
    def success(self, msg): pass
    def chunk(self, msg): pass

    @property
    def buffer(self): return []


class UnavailableWhisper:
    """WhisperHandle substitute that reports unavailable — skips STT checks."""
    is_loaded = False
    is_unavailable = True

    def load(self):
        return None

    def unload(self):
        pass


# ── helpers ──────────────────────────────────────────────────────────────────

def _write_wav(path, duration_s, sr=SR):
    """Write a silent WAV of the given duration to path."""
    samples = int(sr * duration_s)
    audio = th.zeros(1, samples)
    torchaudio.save(path, audio, sr)
    return path


# ── Duration heuristic (check 1) — no Whisper needed ──────────────────────────

class TestDurationHeuristic:
    """These tests exercise check 1 (duration) only; STT is bypassed via
    UnavailableWhisper so no GPU/model download is required."""

    def test_short_wav_flagged(self, tmp_path):
        # text = 100 chars → expected_min_duration = 100/25 = 4.0s
        # WAV is only 0.5s → "too short"
        text = "a" * 100
        wav_path = str(tmp_path / "short.wav")
        _write_wav(wav_path, 0.5)
        result = is_chunk_garbled(wav_path, text, UnavailableWhisper(), StubLogger())
        assert result == "too short"

    def test_long_wav_flagged(self, tmp_path):
        # text = 50 chars → expected_max_duration = 50/3.5 ≈ 14.3s
        # WAV is 60s → "too long"
        text = "a" * 50
        wav_path = str(tmp_path / "long.wav")
        _write_wav(wav_path, 60.0)
        result = is_chunk_garbled(wav_path, text, UnavailableWhisper(), StubLogger())
        assert result == "too long"

    def test_appropriate_duration_passes(self, tmp_path):
        # text = 100 chars → min=4.0s, max=28.6s
        # WAV at 10s should be fine (no garble from duration check)
        text = "a" * 100
        wav_path = str(tmp_path / "ok.wav")
        _write_wav(wav_path, 10.0)
        result = is_chunk_garbled(wav_path, text, UnavailableWhisper(), StubLogger())
        assert result is None

    def test_very_short_text_skips_duration_check(self, tmp_path):
        # text < 10 chars → check 1 returns None immediately
        text = "Hi."
        wav_path = str(tmp_path / "skip.wav")
        _write_wav(wav_path, 0.1)
        result = is_chunk_garbled(wav_path, text, UnavailableWhisper(), StubLogger())
        assert result is None

    def test_few_words_skips_stt_checks(self, tmp_path):
        # After duration check passes, n_words < 5 skips the rest.
        # text = 60 chars, 4 words → duration ok, STT skipped → None
        text = "Hello there good morning."  # 4 words exactly — just under threshold
        wav_path = str(tmp_path / "few_words.wav")
        _write_wav(wav_path, 5.0)   # duration ok for 30 chars
        result = is_chunk_garbled(wav_path, text, UnavailableWhisper(), StubLogger())
        assert result is None

    def test_missing_wav_returns_none(self, tmp_path):
        # torchaudio.info() will fail → exception caught → returns None
        result = is_chunk_garbled(
            str(tmp_path / "nonexistent.wav"),
            "Some text here.",
            UnavailableWhisper(),
            StubLogger(),
        )
        assert result is None

    def test_returns_string_or_none(self, tmp_path):
        text = "a" * 100
        wav_path = str(tmp_path / "check.wav")
        _write_wav(wav_path, 5.0)
        result = is_chunk_garbled(wav_path, text, UnavailableWhisper(), StubLogger())
        assert result is None or isinstance(result, str)


# ── WhisperHandle unit tests (no model load) ──────────────────────────────────

class TestWhisperHandle:
    def test_initial_state(self):
        wh = WhisperHandle()
        assert wh.is_loaded is False
        assert wh.is_unavailable is False

    def test_unload_noop_when_not_loaded(self):
        wh = WhisperHandle()
        wh.unload()  # must not raise
        assert wh.is_loaded is False

    def test_load_returns_none_when_unavailable(self, monkeypatch):
        wh = WhisperHandle()
        # Simulate faster-whisper not installed
        import builtins
        real_import = builtins.__import__
        def fake_import(name, *args, **kwargs):
            if name == "faster_whisper":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        result = wh.load()
        assert result is None
        assert wh.is_unavailable is True
