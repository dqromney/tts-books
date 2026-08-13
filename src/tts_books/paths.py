"""XDG-compliant path resolution for tts-books state files.

Config, batch queue, and pronunciation-dictionary state files live under
``$XDG_CONFIG_HOME/tts-books/`` (default ``~/.config/tts-books/``).

Legacy state files at ``~/bin/*.json`` (from before Phase 1 landed) are
copied to the XDG location on first import, so upgrading is seamless.
The copies at ``~/bin/`` stay in place as a safety net during the
transition — a user running the pre-Phase-1 launcher script (which
points at ``~/bin/tts_book_gui.py``) will still find its state file.
Once confident in the new location, run:

    rm ~/bin/tts-book.config ~/bin/tts-book.queue.json ~/bin/tts-pronunciation.json

Note: user-configurable data paths (``JOBS_DIR``, ``OUTPUT_DIR``,
``VOICE_SAMPLES_DIR``) are NOT relocated here — they default to
``~/tts_output/`` and ``~/voice-samples/`` which already live outside
the source tree, and existing users have real data there.
"""

import os
import shutil
from pathlib import Path


def config_dir() -> Path:
    """``$XDG_CONFIG_HOME/tts-books``, defaulting to ``~/.config/tts-books``."""
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    d = root / "tts-books"
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir() -> Path:
    """``$XDG_DATA_HOME/tts-books``, defaulting to ``~/.local/share/tts-books``.

    Currently unused — reserved for a future phase that relocates the
    ``jobs/`` / ``archive/`` directories out of the user-configurable
    ``~/tts_output/`` default.
    """
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local" / "share"
    d = root / "tts-books"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Exposed as ``str`` so callers can do ``PATH + ".tmp"`` and
# ``open(PATH, ...)`` without pathlib fiddling.
APP_CONFIG_PATH = str(config_dir() / "app.json")
QUEUE_PATH = str(config_dir() / "queue.json")
PRON_DICT_PATH = str(config_dir() / "pronunciation.json")


_LEGACY_BIN = Path.home() / "bin"
_LEGACY = {
    APP_CONFIG_PATH: _LEGACY_BIN / "tts-book.config",
    QUEUE_PATH: _LEGACY_BIN / "tts-book.queue.json",
    PRON_DICT_PATH: _LEGACY_BIN / "tts-pronunciation.json",
}


def migrate_legacy() -> list[tuple[Path, str]]:
    """Copy ``~/bin/*.json`` files to the XDG location if not already there.

    Idempotent: if the XDG file already exists, skip. Uses copy (not
    move) so a pre-Phase-1 launcher pointing at ``~/bin/tts_book_gui.py``
    keeps working during the transition.

    Returns the list of ``(legacy_path, new_path)`` pairs that were
    actually copied, so the caller can log the migration.

    Call from the application entry point, not at import time — importing
    paths.py should have no filesystem side effects.
    """
    copied: list[tuple[Path, str]] = []
    for new, legacy in _LEGACY.items():
        if not Path(new).exists() and legacy.exists():
            Path(new).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy, new)
            copied.append((legacy, new))
    return copied
