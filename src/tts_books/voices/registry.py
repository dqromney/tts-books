"""Voice registry — scan a directory for reference voice WAV files."""

from __future__ import annotations

import os


def scan_voice_dir(voice_samples_dir: str) -> dict[str, str]:
    """Return {stem: abs_path} for every .wav file in voice_samples_dir.

    The dict is ordered by stem name (alphabetical). Returns {} if the
    directory does not exist.
    """
    voice_map: dict[str, str] = {}
    if os.path.isdir(voice_samples_dir):
        for f in sorted(os.listdir(voice_samples_dir)):
            if f.lower().endswith(".wav"):
                stem = os.path.splitext(f)[0]
                voice_map[stem] = os.path.join(voice_samples_dir, f)
    return voice_map
