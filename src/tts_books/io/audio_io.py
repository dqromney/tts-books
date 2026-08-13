"""Audio I/O helpers — MP3 conversion via ffmpeg.

These functions are pure (no App state, no tkinter). The GUI wrapper methods
_convert_to_mp3() and _auto_convert() in gui.py call these and add status
bar updates / error dialogs.
"""

import subprocess


def wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "192k") -> str:
    """Convert a WAV file to MP3 using ffmpeg. Returns mp3_path on success.

    Raises subprocess.CalledProcessError on failure. stderr is available on
    the exception as .stderr (bytes).
    """
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-b:a", bitrate, mp3_path],
        check=True,
        capture_output=True,
    )
    return mp3_path
