"""Audio stitching utilities: crossfade concatenation and DC-offset removal.

These operate on torch tensors of shape (1, samples) — the native output
format of ChatterboxTurboTTS. No App state; safe to import anywhere.
"""

import numpy as np
import torch as th


def crossfade_chunks(wavs, sr, crossfade_ms=50):
    """Stitch a list of 2-D tensors [(1, samples), …] with equal-power crossfades.

    Replaces hard silence-gap joins with overlapping cosine-windowed crossfades
    that eliminate audible clicks between chunks. Each junction overlaps
    ``crossfade_ms`` of audio, fades the outgoing tail with cos² and the incoming
    head with sin² (equal-power), then sums them. Result length is shorter than
    simple concatenation by ``(n_chunks - 1) * crossfade_samples``.

    Falls back to plain concatenation when *crossfade_ms* is 0 or there is only
    one chunk.
    """
    fade_samples = int(sr * crossfade_ms / 1000)
    if fade_samples < 1 or len(wavs) <= 1:
        return th.cat(wavs, dim=1)

    # Equal-power curves — constant perceptual loudness through the transition.
    t = np.linspace(0, np.pi / 2, fade_samples, dtype=np.float32)
    curve_out = np.cos(t) ** 2  # 1 → 0
    curve_in = np.sin(t) ** 2   # 0 → 1

    # Start with the first chunk in numpy.
    accum = wavs[0].numpy()[0].copy()

    for i in range(1, len(wavs)):
        curr = wavs[i].numpy()[0]

        actual = min(fade_samples, len(accum), len(curr))
        if actual < 2:   # chunk too short to crossfade
            accum = np.concatenate([accum, curr])
            continue

        # Compute curves at the actual overlap length (truncated for short chunks).
        if actual == fade_samples:
            co = curve_out
            ci = curve_in
        else:
            t2 = np.linspace(0, np.pi / 2, actual, dtype=np.float32)
            co, ci = np.cos(t2) ** 2, np.sin(t2) ** 2

        tail = accum[-actual:].copy()
        head = curr[:actual].copy()

        # Crossfade: weighted sum of overlapping regions.
        overlap = tail * co + head * ci

        # Assemble: [accum without tail] + [crossfaded overlap] + [curr without head]
        accum = np.concatenate([
            accum[:-actual],
            overlap,
            curr[actual:],
        ])

    return th.from_numpy(accum).unsqueeze(0)


def remove_dc_offset(audio, sr, cutoff_hz=15.0):
    """High-pass filter to remove DC offset from audio.

    DC offset causes low-frequency thumps when concatenating chunks.
    Uses a 2nd-order Butterworth filter with zero-phase (filtfilt) so
    there is no phase distortion. Accepts a 2-D torch tensor (1, samples).

    Returns a torch tensor of the same shape. If scipy is unavailable,
    returns the input unchanged.
    """
    try:
        from scipy.signal import butter, filtfilt
    except ImportError:
        return audio

    nyq = sr / 2
    b, a = butter(2, cutoff_hz / nyq, btype="high")
    arr = audio.numpy()[0].astype(np.float64)
    filtered = filtfilt(b, a, arr).astype(np.float32)
    return th.from_numpy(filtered).unsqueeze(0)
