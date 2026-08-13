"""Batch queue persistence — serialize/deserialize the job queue to disk.

The queue is stored as JSON at BATCH_QUEUE_PATH (XDG-compliant via paths.py).
Text is stripped from the saved JSON (it may be large) and reloaded from
source_path or a sidecar .txt file on startup.

The BatchItem and Settings dataclasses remain importable from here as well
so callers can import from one place.
"""

import json
import os

from tts_books.paths import QUEUE_PATH as BATCH_QUEUE_PATH


def save_batch_queue(batch_queue: list[dict], next_id: int) -> None:
    """Persist the batch queue to BATCH_QUEUE_PATH.

    Text fields are excluded from the JSON. For pasted text (no source_path)
    a sidecar .txt is written alongside the queue file and its path is stored
    in _text_sidecar.
    """
    items_to_save = []
    queue_dir = os.path.dirname(BATCH_QUEUE_PATH)
    for item in batch_queue:
        entry = {k: v for k, v in item.items() if k not in ("text", "_eta_str")}
        if not item.get("source_path") and item.get("text"):
            sidecar = os.path.join(queue_dir, f"tts-book.queue-text-{item['id']}.txt")
            try:
                with open(sidecar, "w", encoding="utf-8") as f:
                    f.write(item["text"])
                entry["_text_sidecar"] = sidecar
            except Exception:
                pass
        items_to_save.append(entry)

    data = {"next_id": next_id, "items": items_to_save}
    try:
        with open(BATCH_QUEUE_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass  # disk full / permission — non-critical


def load_batch_queue(load_text_fn) -> tuple[int, list[dict]]:
    """Read the queue file and return (next_id, items) with text reloaded.

    Items that were in "processing" state (from a crash) are reset to "pending".
    Text is reloaded from source_path or a sidecar .txt file. Returns (0, [])
    if the file doesn't exist or is corrupt.

    Args:
        load_text_fn: Callable[[str], str] — e.g. tts_books.io.text_extract.load_text_file
    """
    if not os.path.isfile(BATCH_QUEUE_PATH):
        return 0, []
    try:
        with open(BATCH_QUEUE_PATH) as f:
            data = json.load(f)
        next_id = data.get("next_id", 0)
        items = data.get("items", [])
        for item in items:
            if item.get("status") == "processing":
                item["status"] = "pending"
                item["error"] = ""
            if "text" not in item:
                if item.get("source_path"):
                    try:
                        item["text"] = load_text_fn(item["source_path"])
                    except Exception as e:
                        item["text"] = ""
                        if item.get("status") != "done":
                            item["status"] = "error"
                            item["error"] = f"Cannot reload source: {e}"
                elif item.get("_text_sidecar") and os.path.isfile(item["_text_sidecar"]):
                    try:
                        with open(item["_text_sidecar"], encoding="utf-8") as f:
                            item["text"] = f.read()
                    except Exception:
                        item["text"] = ""
                else:
                    item["text"] = ""
        return next_id, items
    except Exception:
        return 0, []
