"""Tests for generation/stitching.py — crossfade_chunks() and remove_dc_offset().

All tests operate on synthetic torch tensors; no WAV files needed.
"""

import torch as th

from tts_books.generation.stitching import crossfade_chunks, remove_dc_offset

SR = 24000  # matches Chatterbox output rate


# ── helpers ──────────────────────────────────────────────────────────────────

def _sine(freq_hz: float, duration_s: float, sr: int = SR) -> th.Tensor:
    """Return a (1, samples) sine wave tensor."""
    t = th.linspace(0, duration_s, int(sr * duration_s))
    return th.sin(2 * th.pi * freq_hz * t).unsqueeze(0)


def _silence(duration_s: float, sr: int = SR) -> th.Tensor:
    return th.zeros(1, int(sr * duration_s))


def _dc(level: float, duration_s: float, sr: int = SR) -> th.Tensor:
    """Return a constant-value (DC) tensor."""
    return th.full((1, int(sr * duration_s)), level)


# ── crossfade_chunks ──────────────────────────────────────────────────────────

class TestCrossfadeChunks:
    def test_single_chunk_returns_unchanged(self):
        wav = _sine(440, 0.5)
        result = crossfade_chunks([wav], SR, crossfade_ms=50)
        assert result.shape == wav.shape
        assert th.allclose(result, wav, atol=1e-6)

    def test_zero_crossfade_is_plain_concat(self):
        a = _sine(440, 0.2)
        b = _sine(880, 0.2)
        result = crossfade_chunks([a, b], SR, crossfade_ms=0)
        expected = th.cat([a, b], dim=1)
        assert th.allclose(result, expected, atol=1e-6)

    def test_output_shorter_than_naive_concat(self):
        a = _sine(440, 0.5)
        b = _sine(880, 0.5)
        crossfade_ms = 50
        result = crossfade_chunks([a, b], SR, crossfade_ms=crossfade_ms)
        naive_len = a.shape[1] + b.shape[1]
        overlap = int(SR * crossfade_ms / 1000)
        assert result.shape[1] == naive_len - overlap

    def test_three_chunks_correct_length(self):
        a = _sine(440, 0.4)
        b = _sine(880, 0.4)
        c = _sine(1320, 0.4)
        crossfade_ms = 30
        result = crossfade_chunks([a, b, c], SR, crossfade_ms=crossfade_ms)
        overlap = int(SR * crossfade_ms / 1000)
        expected_len = a.shape[1] + b.shape[1] + c.shape[1] - 2 * overlap
        assert result.shape[1] == expected_len

    def test_output_is_2d_tensor(self):
        a = _sine(440, 0.3)
        b = _sine(880, 0.3)
        result = crossfade_chunks([a, b], SR, crossfade_ms=50)
        assert result.ndim == 2
        assert result.shape[0] == 1

    def test_crossfade_region_blends_not_clips(self):
        # Crossfade of two unit-amplitude sines should stay within [-√2, √2].
        a = _sine(440, 0.3)
        b = _sine(440, 0.3)
        result = crossfade_chunks([a, b], SR, crossfade_ms=50)
        assert result.abs().max().item() <= 1.5  # equal-power blend peaks < sqrt(2)

    def test_very_short_chunk_handled_gracefully(self):
        # A chunk shorter than the overlap region should not raise.
        crossfade_ms = 200
        short = _silence(0.05)   # only 50 ms — shorter than the overlap
        normal = _sine(440, 0.5)
        result = crossfade_chunks([normal, short], SR, crossfade_ms=crossfade_ms)
        assert result.shape[1] > 0

    def test_silence_chunk_produces_finite_output(self):
        a = _silence(0.3)
        b = _silence(0.3)
        result = crossfade_chunks([a, b], SR, crossfade_ms=50)
        assert th.isfinite(result).all()


# ── remove_dc_offset ─────────────────────────────────────────────────────────

class TestRemoveDcOffset:
    def test_constant_dc_is_removed(self):
        dc = _dc(0.5, 1.0)
        result = remove_dc_offset(dc, SR)
        # After HP filtering, mean of output should be near zero
        assert result.mean().abs().item() < 0.01

    def test_pure_tone_passes_through(self):
        # A 1 kHz sine is well above the 15 Hz cutoff — should be mostly preserved.
        tone = _sine(1000, 0.5)
        result = remove_dc_offset(tone, SR)
        rms_in = tone.pow(2).mean().sqrt().item()
        rms_out = result.pow(2).mean().sqrt().item()
        # Less than 5% energy loss at 1 kHz
        assert rms_out > 0.95 * rms_in

    def test_output_shape_preserved(self):
        audio = _sine(440, 0.5)
        result = remove_dc_offset(audio, SR)
        assert result.shape == audio.shape

    def test_output_is_float_tensor(self):
        audio = _sine(440, 0.2)
        result = remove_dc_offset(audio, SR)
        assert result.dtype == th.float32

    def test_zero_signal_stays_zero(self):
        silence = _silence(0.3)
        result = remove_dc_offset(silence, SR)
        assert result.abs().max().item() < 1e-6

    def test_dc_plus_tone_removes_dc_keeps_tone(self):
        tone = _sine(500, 1.0)
        dc_offset = 0.3
        dc = _dc(dc_offset, 1.0)
        combined = tone + dc
        result = remove_dc_offset(combined, SR)
        # Mean should drop close to zero
        assert result.mean().abs().item() < 0.05
        # The 500 Hz tone should still be present (RMS close to sine RMS ≈ 0.5/√2)
        rms = result.pow(2).mean().sqrt().item()
        expected_rms = tone.pow(2).mean().sqrt().item()
        assert rms > 0.85 * expected_rms
