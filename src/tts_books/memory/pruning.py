"""Memory pruning helpers — RSS measurement and GC/malloc_trim.

All functions here are pure (no App state, no tkinter). The App wrapper
_manual_prune() stays in gui.py because it logs through self._log.
"""

import contextlib
import ctypes
import gc


def mem_rss_mb() -> tuple[float, float]:
    """Return (process_rss_mb, system_free_mb) via psutil, or (0, 0) if unavailable."""
    try:
        import psutil
        return (
            psutil.Process().memory_info().rss / (1024 * 1024),
            psutil.virtual_memory().available / (1024 * 1024),
        )
    except Exception:
        return 0.0, 0.0


def do_prune() -> tuple[float, float, float]:
    """Run gc.collect() + malloc_trim(). Return (before_rss_mb, after_rss_mb, free_mb)."""
    before_mb, _ = mem_rss_mb()
    gc.collect()
    with contextlib.suppress(Exception):
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    after_mb, free_mb = mem_rss_mb()
    return before_mb, after_mb, free_mb
