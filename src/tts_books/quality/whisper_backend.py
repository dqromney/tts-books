"""Whisper transcription backend for garbled-chunk detection.

The module-level `whisper` singleton is shared by both garbled.py (which
calls whisper.load() to get the model) and the pipeline (which calls
whisper.unload() to free ctranslate2 memory every 25 chunks). Both must
reference the SAME instance — importing the module-level object achieves
this because Python module objects are singletons per interpreter.
"""


class WhisperHandle:
    """Lazy-loading wrapper around faster-whisper WhisperModel.

    load() returns the model or None if faster-whisper is unavailable.
    unload() releases the model and frees its memory.
    """

    def __init__(self):
        self._model = None
        self._unavailable = False

    def load(self):
        """Return the WhisperModel, loading it on first call. Returns None if unavailable."""
        if self._model is not None:
            return self._model
        if self._unavailable:
            return None
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
            return self._model
        except Exception:
            self._unavailable = True
            return None

    def unload(self):
        """Release the model so ctranslate2's memory arena can be freed."""
        if self._model is not None:
            del self._model
            self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_unavailable(self) -> bool:
        return self._unavailable


# Module-level singleton — import this object, not the class.
whisper = WhisperHandle()
