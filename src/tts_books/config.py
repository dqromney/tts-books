"""App configuration: load/save app.json and expose default path constants.

The app.json file lives at CONFIG_PATH (XDG-compliant via paths.py). It stores
three user-configurable directory paths and an instance_count for concurrent-
launch detection.

Constants here are the *defaults* used when no config file is present. The
actual paths used at runtime come from load_config() which merges the file.
"""

import json
import os

from tts_books.paths import APP_CONFIG_PATH as CONFIG_PATH

# Default paths — overridden by config file if present. Unlike the XDG state
# files, these are user-configurable via the Settings dialog and default to
# well-known $HOME locations for backward compatibility.
VOICE_SAMPLES_DIR = os.path.expanduser("~/voice-samples")
JOBS_DIR = os.path.expanduser("~/tts_output/jobs")
OUTPUT_DIR = os.path.expanduser("~/tts_output")
MAX_CHUNK_RETRIES = 3
_PRUNE_THRESHOLD_GB = 12.0  # auto-prune trigger threshold (system free RAM)


def load_config():
    cfg = {
        "voice_samples_dir": VOICE_SAMPLES_DIR,
        "jobs_dir": JOBS_DIR,
        "output_dir": OUTPUT_DIR,
        "instance_count": 0,
    }
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                data = json.load(f)
            dir_keys = {"voice_samples_dir", "jobs_dir", "output_dir"}
            for k, v in data.items():
                if k in dir_keys and v:
                    cfg[k] = os.path.expanduser(v)
                elif k not in dir_keys:
                    cfg[k] = v
        except Exception:
            pass
    return cfg


def save_config(cfg):
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG_PATH)
