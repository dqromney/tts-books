"""VoxCeleb1 voice helpers — name sanitization, token lookup, metadata fetch, download.

These functions are pure (no tkinter). The _browse_dataset() dialog in gui.py
calls them from worker threads, keeping all widget access on the main thread.
"""

from __future__ import annotations

import os
import re


def sanitize_name(name: str) -> str:
    """Convert a speaker name to a safe filename stem (lowercase, hyphens only)."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def get_hf_token() -> str | None:
    """Return a HuggingFace token from env vars or the cached token file."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    cached = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.isfile(cached):
        try:
            with open(cached) as f:
                t = f.read().strip()
            if t:
                return t
        except Exception:
            pass
    return None


def load_voxceleb_metadata(
    token: str | None = None,
) -> list[tuple[str, str, str]]:
    """Fetch VoxCeleb1 metadata.csv from HuggingFace and return rows.

    Returns [(name, gender, speaker_id), ...]. Raises on network/parse errors.
    """
    import csv

    from huggingface_hub import hf_hub_download

    csv_path = hf_hub_download(
        repo_id="sdialog/voices-voxceleb1",
        filename="metadata.csv",
        repo_type="dataset",
        token=token,
    )
    rows: list[tuple[str, str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        name_col = (
            next((c for c in fieldnames if "name" in c.lower()), None)
            or (fieldnames[0] if fieldnames else "")
        )
        gender_col = next((c for c in fieldnames if "gender" in c.lower()), None) or ""
        id_col = next(
            (
                c
                for c in fieldnames
                if c.lower() in ("id", "voxceleb1 id", "voxceleb_id", "speaker_id")
            ),
            None,
        )
        if id_col is None:
            id_col = (
                next(
                    (c for c in fieldnames if "id" in c.lower() and c != name_col),
                    None,
                )
                or ""
            )
        for row in reader:
            name = row.get(name_col, "").strip()
            gender = row.get(gender_col, "").strip() if gender_col else ""
            sid = row.get(id_col, "").strip() if id_col else ""
            if name and sid:
                rows.append((name, gender, sid))
    return rows


def download_voxceleb_speaker(
    speaker_id: str,
    name: str,
    dest_dir: str,
    token: str | None = None,
) -> str:
    """Download a VoxCeleb1 speaker WAV to dest_dir. Returns the destination path.

    The filename is derived from the speaker name via sanitize_name(). Raises on
    network errors.
    """
    import shutil

    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(
        repo_id="sdialog/voices-voxceleb1",
        filename=f"audio/{speaker_id}.wav",
        repo_type="dataset",
        token=token,
    )
    dest_name = sanitize_name(name) + ".wav"
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, dest_name)
    shutil.copy2(cached, dest)
    return dest
