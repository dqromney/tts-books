"""Pronunciation dictionary — load, save, and apply word substitutions.

The dictionary is stored at PRON_DICT_PATH (XDG-compliant via paths.py) as a
JSON object mapping words/phrases to substitution entries:
  {"word": {"replacement": "wurd", "enabled": true}, ...}

Legacy flat format {"word": "wurd"} is migrated to the richer format on first
read and re-saved automatically.
"""

import json
import os
import re

from tts_books.paths import PRON_DICT_PATH


def load_pron_dict():
    """Return {word: {"replacement": str, "enabled": bool}}.
    Migrates the old flat {word: str} format on first read."""
    if os.path.isfile(PRON_DICT_PATH):
        try:
            with open(PRON_DICT_PATH) as f:
                raw = json.load(f)
            migrated = {}
            dirty = False
            for k, v in raw.items():
                if isinstance(v, str):
                    migrated[k] = {"replacement": v, "enabled": True}
                    dirty = True
                else:
                    migrated[k] = v
            if dirty:
                save_pron_dict(migrated)
            return migrated
        except Exception:
            pass
    return {}


def save_pron_dict(d):
    tmp = PRON_DICT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2, sort_keys=True)
    os.replace(tmp, PRON_DICT_PATH)


def apply_pronunciation(text, pron_dict):
    """Replace words/phrases using the pronunciation dictionary (word-boundary, case-insensitive).
    Entries with enabled=False are silently skipped."""
    for original, entry in pron_dict.items():
        if isinstance(entry, dict):
            if not entry.get("enabled", True):
                continue
            replacement = entry["replacement"]
        else:
            replacement = entry  # legacy flat format
        pattern = r'\b' + re.escape(original) + r'\b'
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text
