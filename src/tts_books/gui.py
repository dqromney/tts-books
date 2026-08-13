#!/usr/bin/env python3
"""Standalone TTS Book GUI — Chatterbox Turbo on CPU.

Features:
- Single-file and batch-queue generation
- Pause/resume mid-generation (thread-safe threading.Event)
- Multi-job resume picker (choose which partial job to resume)
- Detail log panel (collapsible, timestamped, per-chunk timing)
- Redesigned Advanced Options panel
"""

import ctypes
import gc
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import threading
import time
import tkinter as tk
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
from typing import Protocol

import torch as th
import torchaudio
from chatterbox.tts_turbo import ChatterboxTurboTTS

from tts_books.io.text_extract import load_text_file
from tts_books.paths import APP_CONFIG_PATH as CONFIG_PATH, QUEUE_PATH as BATCH_QUEUE_PATH
from tts_books.pronunciation import apply_pronunciation, load_pron_dict, save_pron_dict

DEVICE = "cpu"
MAX_CHARS_PER_CHUNK = 200   # default; override via UI

# Default paths — overridden by config file if present. Unlike the state
# files above (which now live under $XDG_CONFIG_HOME/tts-books/ per
# paths.py), these are user-configurable via the Settings dialog and
# default to well-known $HOME locations for backward compatibility.
VOICE_SAMPLES_DIR = os.path.expanduser("~/voice-samples")
JOBS_DIR = os.path.expanduser("~/tts_output/jobs")
OUTPUT_DIR = os.path.expanduser("~/tts_output")
MAX_CHUNK_RETRIES = 3
_PRUNE_THRESHOLD_GB = 12.0  # auto-prune warning threshold (system free RAM)

_whisper_model = None   # lazy-loaded faster-whisper tiny.en, shared across chunks
_whisper_unavailable = False  # set True after a failed import so we don't retry

def _load_whisper():
    """Return a faster-whisper WhisperModel, or None if unavailable."""
    global _whisper_model, _whisper_unavailable
    if _whisper_model is not None:
        return _whisper_model
    if _whisper_unavailable:
        return None
    try:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        return _whisper_model
    except Exception:
        _whisper_unavailable = True
        return None


def _load_config():
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


def _save_config(cfg):
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG_PATH)


EVENT_TAGS = [
    "[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]",
    "[sniff]", "[gasp]", "[chuckle]", "[laugh]",
]


def _sanitize_name(name):
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')


_SEPARATOR_RE = re.compile(r'^([=\-*~_#])\1{3,}$')


def split_text(text, max_chars=200, split_clauses=True):
    """Split text into TTS-friendly chunks at natural boundaries."""
    sent_end = r'(?<=[.!?])[\"”\u2019]?\s+'
    clause_end = r'|(?<=[;:])\s+'
    boundary = re.compile(sent_end + (clause_end if split_clauses else ''))

    sentences = []
    for para in re.split(r'\n\s*\n', text.strip()):
        para = para.strip()
        if not para:
            continue
        lines = [l for l in para.splitlines() if not _SEPARATOR_RE.match(l.strip())]
        para = '\n'.join(lines).strip()
        if not para:
            continue
        para = re.sub(r'(?<!\n)\n(?!\n)', ' ', para)
        para = re.sub(r' {2,}', ' ', para)
        for s in boundary.split(para):
            s = s.strip()
            if s:
                sentences.append(s)

    def _opener(s):
        """Return the first-2-word tuple of a sentence (used for anaphora guard)."""
        ws = s.split()
        return tuple(ws[:2]) if len(ws) >= 2 else (tuple(ws) if ws else ())

    chunks = []
    current = ""
    current_openers = []   # first-2-word tuples of sentences already in current chunk

    for s in sentences:
        opener = _opener(s)
        if not current:
            current = s
            current_openers = [opener]
        elif len(current) + 1 + len(s) <= max_chars:
            # Anaphora guard: avoid 3+ consecutive sentences with the same
            # first-2-word opener in one chunk — the TTS tends to loop on them.
            if opener and current_openers.count(opener) >= 2:
                chunks.append(current)
                current = s
                current_openers = [opener]
            else:
                current = current + " " + s
                current_openers.append(opener)
        else:
            chunks.append(current)
            current = s
            current_openers = [opener]
        while len(current) > max_chars:
            semi_pos  = current.rfind(';', 0, max_chars)
            comma_pos = current.rfind(',', 0, max_chars)
            space_pos = current.rfind(' ', 0, max_chars)
            split_at  = max(semi_pos, comma_pos, space_pos)
            if split_at <= 0:
                split_at = max_chars
            chunks.append(current[:split_at].strip())
            rest = current[split_at:].strip()
            # Capitalize continuations split at a semicolon (Appendix list items
            # starting mid-sentence confuse the TTS without a sentence-start signal).
            if split_at == semi_pos and rest and rest[0].islower():
                rest = rest[0].upper() + rest[1:]
            current = rest
            current_openers = [_opener(current)]
    if current:
        chunks.append(current)
    return chunks if chunks else [text]


def _crossfade_chunks(wavs, sr, crossfade_ms=50):
    """Stitch a list of 2-D tensors [(1, samples), …] with equal-power crossfades.

    Replaces hard silence-gap joins with overlapping cosine-windowed crossfades
    that eliminate audible clicks between chunks.  Each junction overlaps
    ``crossfade_ms`` of audio, fades the outgoing tail with cos² and the incoming
    head with sin² (equal-power), then sums them.  Result length is shorter than
    simple concatenation by ``(n_chunks - 1) * crossfade_samples``.

    Falls back to plain concatenation when *crossfade_ms* is 0 or there is only
    one chunk.
    """
    import numpy as np

    fade_samples = int(sr * crossfade_ms / 1000)
    if fade_samples < 1 or len(wavs) <= 1:
        return th.cat(wavs, dim=1)

    # Equal-power curves — constant perceptual loudness through the transition.
    t = np.linspace(0, np.pi / 2, fade_samples, dtype=np.float32)
    curve_out = np.cos(t) ** 2  # 1 → 0
    curve_in  = np.sin(t) ** 2  # 0 → 1

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


def _remove_dc_offset(audio, sr, cutoff_hz=15.0):
    """High-pass filter to remove DC offset from audio.

    DC offset causes low-frequency thumps when concatenating chunks.
    Uses a 2nd-order Butterworth filter with zero-phase (filtfilt) so
    there is no phase distortion.  Accepts a 2-D torch tensor (1, samples).

    Returns a torch tensor of the same shape.  If scipy is unavailable,
    returns the input unchanged.
    """
    try:
        from scipy.signal import butter, filtfilt
    except ImportError:
        return audio

    import numpy as np
    nyq = sr / 2
    b, a = butter(2, cutoff_hz / nyq, btype="high")
    arr = audio.numpy()[0].astype(np.float64)
    filtered = filtfilt(b, a, arr).astype(np.float32)
    return th.from_numpy(filtered).unsqueeze(0)


@dataclass
class Settings:
    """User-configurable generation parameters snapshotted from the Advanced
    Options panel.

    Flows into the on-disk batch queue, state.json checkpoints, and
    _do_generate. Wire format is a plain dict via to_dict()/from_dict();
    the dataclass exists as the single source of truth for field names
    and defaults.

    Note: ref_path is included in the snapshot but intentionally NOT
    written back to the UI by _apply_settings_to_ui — voice/reference
    selection is per-batch-item, not per-settings block.
    """

    temperature: float = 0.8
    min_p: float = 0.0
    top_p: float = 0.95
    top_k: int = 1000
    repetition_penalty: float = 1.2
    norm_loudness: bool = True
    split_clauses: bool = True
    chunk_size: int = 200
    seed: int = 0
    ref_path: str | None = None
    cpu_threads: int = 4
    lead_silence: float = 0.0
    trail_silence: float = 0.0
    auto_mp3: bool = False
    reload_every: int = 0
    prune_every: int = 1
    crossfade_ms: int = 50

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Settings":
        """Build from a possibly-partial dict. Unknown keys are ignored so
        older state.json / queue files remain loadable after new fields
        are added. Missing keys fall back to dataclass defaults.
        """
        if not d:
            return cls()
        known = {f.name for f in dataclass_fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class BatchItem:
    """One entry in the batch queue.

    Wire format is a plain dict (via to_dict()/from_dict()) since the
    queue is persisted as JSON and mutated in-place across many sites.
    The dataclass exists as the single source of truth for field names
    and defaults so construction can't silently drift across the three
    call sites (_add_to_batch, _add_dir_to_batch, and the rework-add
    flow in _open_rework_dialog).

    Leading-underscore fields are runtime-computed, not user-supplied:
    _job_hash is set by _do_generate for later resume-hint lookup,
    _eta_str is refreshed by the progress reporter, and _text_sidecar
    is set by _load_batch_queue when a saved sidecar text file exists.
    """

    id: int = 0
    source_path: str | None = None
    file_name: str = ""
    text: str = ""
    ref_path: str | None = None
    voice_name: str = ""
    settings: dict = field(default_factory=dict)
    status: str = "pending"
    output_path: str = ""
    error: str = ""
    chunks_done: int = 0
    chunks_total: int = 0
    gen_time: str = ""
    _job_hash: str | None = None
    _eta_str: str = ""
    _text_sidecar: str | None = None
    try_resume: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "BatchItem":
        """Build from a possibly-partial dict. Unknown keys are ignored so
        older queue.json files remain loadable after new fields are added.
        Missing keys fall back to dataclass defaults.
        """
        if not d:
            return cls()
        known = {f.name for f in dataclass_fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class Logger(Protocol):
    """Log target for background jobs.

    The App implements this interface so non-GUI modules (pipeline,
    archiving) can accept ``logger: Logger`` without importing tkinter.
    ``buffer`` is the plain-text line list that ``_archive_job`` writes
    to ``generation.log``.
    """

    def info(self, msg: str) -> None: ...
    def warn(self, msg: str) -> None: ...
    def error(self, msg: str) -> None: ...
    def success(self, msg: str) -> None: ...

    @property
    def buffer(self) -> list[str]: ...


class CancelToken:
    """Thread-safe cancel + pause signal for background jobs.

    Consolidates cancel_flag (bool) and pause_event (Event, inverted
    "set means running") behind natural pause()/resume()/is_paused()
    calls. Passed to the generation pipeline so it doesn't need a
    back-reference to the tkinter App.
    """

    def __init__(self):
        self._cancelled = False
        self._not_paused = threading.Event()
        self._not_paused.set()

    def reset(self):
        """Clear cancel and resume. Called at the start of each job."""
        self._cancelled = False
        self._not_paused.set()

    def cancel(self):
        """Request cancellation. Also unblocks any pause so cancel proceeds."""
        self._cancelled = True
        self._not_paused.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def pause(self):
        self._not_paused.clear()

    def resume(self):
        self._not_paused.set()

    def is_paused(self) -> bool:
        return not self._not_paused.is_set()

    def wait_if_paused(self):
        self._not_paused.wait()


# ── GUI ───────────────────────────────────────────────────────────

class TTSBookApp:
    def __init__(self, root, auto_resume=False):
        self.root = root
        self.root.title("Chatterbox TTS Book")
        self.root.geometry("900x1150")
        self.root.minsize(750, 850)

        # Force ttk 'clam' theme with explicit black-on-white colours.
        # On Linux systems whose GTK theme is dark, the default ttk theme
        # inherits foreground/background from GTK and renders text invisible
        # (white on white or black on black).  'clam' is a portable ttk theme
        # whose colours we can override without fighting GTK.
        try:
            _style = ttk.Style(self.root)
            _style.theme_use('clam')
            # Explicit palette for every widget class the app uses.
            _COMMON = dict(background='white', foreground='black',
                           fieldbackground='white', selectbackground='#0078d4',
                           selectforeground='white')
            for _cls in ('TFrame', 'TLabelframe', 'TLabelframe.Label',
                         'TLabel', 'TEntry', 'TSpinbox', 'TCombobox',
                         'TCheckbutton', 'TRadiobutton', 'TButton',
                         'TNotebook', 'TNotebook.Tab', 'Treeview',
                         'Treeview.Heading', 'TScrollbar', 'TSeparator'):
                try:
                    _style.configure(_cls, **_COMMON)
                except tk.TclError:
                    pass
            # Treeview needs its row background set separately
            _style.configure('Treeview', rowheight=22)
            # Make hovering/active state legible too
            _style.map('TButton',
                       foreground=[('active', 'black'), ('disabled', '#888888')],
                       background=[('active', '#e8e8e8')])
            _style.map('TCheckbutton',
                       foreground=[('active', 'black'), ('disabled', '#888888')])
            _style.map('TCombobox',
                       fieldbackground=[('readonly', 'white')],
                       foreground=[('readonly', 'black')])
        except tk.TclError:
            pass  # If clam is unavailable for some reason, fall through

        # Fix Combobox dropdown list colours on Linux themes where the popup
        # listbox inherits white foreground from GTK, making text invisible.
        self.root.option_add('*TCombobox*Listbox.foreground', 'black')
        self.root.option_add('*TCombobox*Listbox.background', 'white')
        self.root.option_add('*TCombobox*Listbox.selectForeground', 'white')
        self.root.option_add('*TCombobox*Listbox.selectBackground', '#0078d4')

        # tk.* widgets (Toplevel dialogs, Listbox, etc.) inherit from the Tcl
        # palette rather than the ttk theme.  Force black-on-white there too.
        self.root.tk_setPalette(
            background='white', foreground='black',
            activeBackground='#e8e8e8', activeForeground='black',
            highlightBackground='white', highlightColor='black',
            selectBackground='#0078d4', selectForeground='white',
            insertBackground='black',
        )

        self.model = None
        self.cancel_token = CancelToken()
        self._source_path = None
        self._play_proc = None
        self._ref_proc = None
        self._voice_map = {}
        self.voice_combo_var = tk.StringVar()
        self._lockable = []
        self._source_lockable = []
        self._adv_lockable = []
        self._gen_running = False
        self._model_mem_mb = 0.0  # RSS delta measured when model first loads

        cfg = _load_config()
        self._voice_samples_dir = cfg["voice_samples_dir"]
        self._jobs_dir = cfg["jobs_dir"]
        self._output_dir = cfg["output_dir"]
        self._archive_dir = os.path.join(os.path.dirname(self._jobs_dir), "archive")
        self._log_buffer = []   # plain-text log lines for saving to archive

        cfg["instance_count"] = cfg.get("instance_count", 0) + 1
        _save_config(cfg)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Batch queue
        self._batch_queue = []       # list of queue-item dicts
        self._batch_next_id = 0
        self._batch_running = False

        # Auto MP3
        self.auto_mp3 = tk.BooleanVar(value=False)

        self._build_ui()
        self.root.after(100, self._scan_partial_jobs)
        self._load_batch_queue()  # restore queue from disk

        # Auto-resume: kick off batch processing if flag set and pending items exist
        if auto_resume:
            pending = [q for q in self._batch_queue if q.get("status") == "pending"]
            if pending:
                self.root.after(500, self._process_batch)
                self._log(f"Auto-resume: {len(pending)} pending items, starting batch...", "info")

    # ── UI layout ──────────────────────────────────────────────────

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        # Row 0: tools toolbar
        tools = ttk.LabelFrame(main, text="Tools", padding=(5, 2))
        tools.pack(fill="x", pady=(0, 5))

        _b = ttk.Button(tools, text="Analyze Text", command=self._analyze_text)
        _b.pack(side="left", padx=(0, 4))
        self._lockable.append(_b)
        ttk.Button(tools, text="Pronunciation…", command=self._open_pron_dict).pack(side="left", padx=(0, 4))
        _b = ttk.Button(tools, text="Settings…", command=self._open_settings)
        _b.pack(side="left", padx=(0, 4))
        self._lockable.append(_b)
        ttk.Button(tools, text="Prune", command=self._manual_prune).pack(side="left")
        # Memory bar — fills space between Prune and Exit
        _mem_frame = ttk.Frame(tools)
        _mem_frame.pack(side="left", fill="x", expand=True, padx=(10, 10))
        ttk.Label(_mem_frame, text="RAM:").pack(side="left")
        self._mem_bar = ttk.Progressbar(_mem_frame, mode="determinate", length=130, maximum=100)
        self._mem_bar.pack(side="left", padx=(4, 4))
        self._mem_bar_label = ttk.Label(_mem_frame, text="--/-- GB")
        self._mem_bar_label.pack(side="left")
        ttk.Button(tools, text="Exit", command=self._on_close).pack(side="right")

        # Row 1: text source
        src_frame = ttk.LabelFrame(main, text="Text Source", padding=5)
        src_frame.pack(fill="x", pady=(0, 5))

        btn_row = ttk.Frame(src_frame)
        btn_row.pack(fill="x")
        _b = ttk.Button(btn_row, text="Open .txt / .md / .pdf / .epub", command=self._open_file)
        _b.pack(side="left", padx=(0, 5))
        self._source_lockable.append(_b)
        _b = ttk.Button(btn_row, text="Clear", command=self._clear_text)
        _b.pack(side="left", padx=(0, 10))
        self._source_lockable.append(_b)
        self.file_label = ttk.Label(btn_row, text="No file loaded", foreground="gray")
        self.file_label.pack(side="left")

        text_frame = ttk.Frame(src_frame)
        text_frame.pack(fill="both", expand=True, pady=(5, 0))
        self.text_area = tk.Text(text_frame, height=7, wrap="word", font=("monospace", 10))
        text_vsb = ttk.Scrollbar(text_frame, orient="vertical", command=self.text_area.yview)
        self.text_area.configure(yscrollcommand=text_vsb.set)
        text_vsb.pack(side="right", fill="y")
        self.text_area.pack(side="left", fill="both", expand=True)
        self.text_area.insert("1.0", "Hello world. This is a test.")
        self._source_lockable.append(self.text_area)

        # Row 1: event tags
        tag_frame = ttk.LabelFrame(main, text="Event Tags", padding=5)
        tag_frame.pack(fill="x", pady=(0, 5))
        tag_row = ttk.Frame(tag_frame)
        tag_row.pack()
        for tag in EVENT_TAGS:
            _b = ttk.Button(tag_row, text=tag, command=lambda t=tag: self._insert_tag(t))
            _b.pack(side="left", padx=2)
            self._source_lockable.append(_b)

        # Row 2: reference audio
        ref_frame = ttk.LabelFrame(main, text="Reference Voice", padding=5)
        ref_frame.pack(fill="x", pady=(0, 5))
        ref_row = ttk.Frame(ref_frame)
        ref_row.pack(fill="x")
        self.ref_path = tk.StringVar(value="")
        _b = ttk.Button(ref_row, text="Browse VoxCeleb1", command=self._browse_dataset)
        _b.pack(side="left", padx=(0, 5))
        self._source_lockable.append(_b)
        self.voice_combo = ttk.Combobox(ref_row, textvariable=self.voice_combo_var,
                                        state="readonly", width=28)
        self.voice_combo.pack(side="left", padx=(0, 10))
        self.voice_combo.bind("<<ComboboxSelected>>", self._on_voice_selected)
        self.ref_label = ttk.Label(ref_row, text="(default prompt)", foreground="gray")
        self.ref_label.pack(side="left", padx=(0, 10))
        self.play_ref_btn = ttk.Button(ref_row, text="Play Voice", command=self._play_ref, state="disabled")
        self.play_ref_btn.pack(side="left")

        # Row 3: advanced options (redesigned — 4-column grid)
        adv = ttk.LabelFrame(main, text="Advanced Options", padding=5)
        adv.pack(fill="x", pady=(0, 5))

        self.temp = tk.DoubleVar(value=0.8)
        self.top_p = tk.DoubleVar(value=0.95)
        self.top_k = tk.IntVar(value=1000)
        self.rep_pen = tk.DoubleVar(value=1.2)
        self.min_p = tk.DoubleVar(value=0.0)
        self.norm = tk.BooleanVar(value=True)
        self.seed = tk.IntVar(value=0)
        self.cpu_threads = tk.IntVar(value=4)
        self.split_clauses = tk.BooleanVar(value=True)
        self.chunk_size = tk.IntVar(value=MAX_CHARS_PER_CHUNK)
        self.lead_silence = tk.DoubleVar(value=0.0)
        self.trail_silence = tk.DoubleVar(value=0.0)
        self.reload_every = tk.IntVar(value=0)   # periodic model reload (0=disabled)
        self.prune_every  = tk.IntVar(value=1)   # gc+malloc_trim cadence in chunks (1=every, 0=off)
        self.crossfade_ms = tk.IntVar(value=50)   # crossfade overlap between chunks

        # Row 0: sampling params
        opts_row0 = [
            ("Temperature", self.temp, 0.05, 2.0, 0.05),
            ("Top P", self.top_p, 0.0, 1.0, 0.01),
            ("Top K", self.top_k, 0, 1000, 10),
            ("Rep. Penalty", self.rep_pen, 1.0, 2.0, 0.05),
        ]
        grid = ttk.Frame(adv)
        grid.pack(fill="x")
        for col, (label, var, lo, hi, step) in enumerate(opts_row0):
            f = ttk.Frame(grid)
            f.grid(row=0, column=col, sticky="ew", padx=4, pady=2)
            ttk.Label(f, text=label, font=("", 8)).pack(anchor="w")
            sb = ttk.Spinbox(f, textvariable=var, from_=lo, to=hi, increment=step, width=10)
            sb.pack(fill="x")
            self._adv_lockable.append(sb)

        # Row 1: more params + checkboxes
        opts_row1 = [
            ("Min P", self.min_p, 0.0, 1.0, 0.01),
            ("Seed (0=random)", self.seed, 0, 999999, 1),
            ("Chunk size", self.chunk_size, 50, 500, 25),
            ("CPU threads", self.cpu_threads, 1, 8, 1),
        ]
        for col, (label, var, lo, hi, step) in enumerate(opts_row1):
            f = ttk.Frame(grid)
            f.grid(row=1, column=col, sticky="ew", padx=4, pady=2)
            ttk.Label(f, text=label, font=("", 8)).pack(anchor="w")
            sb = ttk.Spinbox(f, textvariable=var, from_=lo, to=hi, increment=step, width=10)
            sb.pack(fill="x")
            self._adv_lockable.append(sb)

        # Row 2: lead / trail silence padding + model reload
        opts_row2 = [
            ("Lead silence (s)", self.lead_silence, 0.0, 30.0, 0.5),
            ("Trail silence (s)", self.trail_silence, 0.0, 30.0, 0.5),
        ]
        for col, (label, var, lo, hi, step) in enumerate(opts_row2):
            f = ttk.Frame(grid)
            f.grid(row=2, column=col, sticky="ew", padx=4, pady=2)
            ttk.Label(f, text=label, font=("", 8)).pack(anchor="w")
            sb = ttk.Spinbox(f, textvariable=var, from_=lo, to=hi, increment=step, width=10)
            sb.pack(fill="x")
            self._adv_lockable.append(sb)

        # Reload model every N chunks (0 = disabled)
        rf = ttk.Frame(grid)
        rf.grid(row=2, column=2, sticky="ew", padx=4, pady=2)
        ttk.Label(rf, text="Reload model every N chunks", font=("", 8)).pack(anchor="w")
        rsb = ttk.Spinbox(rf, textvariable=self.reload_every, from_=0, to=500, increment=50, width=10)
        rsb.pack(fill="x")
        # NOT in _adv_lockable — stays editable during generation for live adjustment

        # Crossfade overlap (ms, 0 = plain concat with silence)
        cf = ttk.Frame(grid)
        cf.grid(row=2, column=3, sticky="ew", padx=4, pady=2)
        ttk.Label(cf, text="Crossfade (ms, 0=off)", font=("", 8)).pack(anchor="w")
        csb = ttk.Spinbox(cf, textvariable=self.crossfade_ms, from_=0, to=500, increment=10, width=10)
        csb.pack(fill="x")
        self._adv_lockable.append(csb)

        # Prune every N chunks (gc.collect + malloc_trim). 1 = every chunk, 0 = off.
        pf = ttk.Frame(grid)
        pf.grid(row=3, column=0, sticky="ew", padx=4, pady=2)
        ttk.Label(pf, text="Prune every N chunks (1=every, 0=off)", font=("", 8)).pack(anchor="w")
        psb = ttk.Spinbox(pf, textvariable=self.prune_every, from_=0, to=500, increment=1, width=10)
        psb.pack(fill="x")
        # NOT in _adv_lockable — stays editable during generation for live adjustment

        chk_row = ttk.Frame(adv)
        chk_row.pack(fill="x", pady=(4, 0))
        _ck = ttk.Checkbutton(chk_row, text="Normalize loudness (-27 LUFS)", variable=self.norm)
        _ck.pack(side="left", padx=4)
        self._adv_lockable.append(_ck)
        _ck = ttk.Checkbutton(chk_row, text="Split on clauses (; :)", variable=self.split_clauses)
        _ck.pack(side="left", padx=(16, 0))
        self._adv_lockable.append(_ck)
        _ck = ttk.Checkbutton(chk_row, text="Auto-convert to MP3", variable=self.auto_mp3)
        _ck.pack(side="left", padx=(16, 0))
        self._adv_lockable.append(_ck)

        # Row 4: batch queue (collapsible)
        self._build_batch_queue(main)

        # Row 5: generation controls (single-file)
        ctrl = ttk.LabelFrame(main, text="Single File", padding=(5, 2))
        ctrl.pack(fill="x", pady=(5, 2))

        self.gen_btn = ttk.Button(ctrl, text="Generate", command=self._start_generate)
        self.gen_btn.pack(side="left", padx=(0, 4))

        self.resume_btn = ttk.Button(ctrl, text="Resume", command=self._start_resume, state="disabled")
        self.resume_btn.pack(side="left", padx=(0, 4))

        self.pause_btn = ttk.Button(ctrl, text="Pause", command=self._toggle_pause, state="disabled")
        self.pause_btn.pack(side="left", padx=(0, 4))

        self.cancel_btn = ttk.Button(ctrl, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=(0, 16))

        self.play_btn = ttk.Button(ctrl, text="Play Last Output", command=self._play, state="disabled")
        self.play_btn.pack(side="left", padx=(0, 4))

        self.mp3_btn = ttk.Button(ctrl, text="Convert to MP3", command=self._convert_to_mp3, state="disabled")
        self.mp3_btn.pack(side="left", padx=(0, 16))

        self.rework_btn = ttk.Button(ctrl, text="Rework…", command=self._open_rework_dialog)
        self.rework_btn.pack(side="left")

        # Row 7: progress
        self.progress = ttk.Progressbar(main, mode="determinate")
        self.progress.pack(fill="x", pady=(0, 5))

        self.status = ttk.Label(main, text="Ready", foreground="gray")
        self.status.pack(anchor="w")

        self.out_label = ttk.Label(main, text="", foreground="blue")
        self.out_label.pack(anchor="w")

        # Row 7: detail log (collapsible)
        self._build_log_panel(main)

        self._populate_voice_combo()
        self.root.after(100, self._refresh_rework_badge)
        # Show panels that are on by default and start memory bar updates
        self._toggle_batch_visibility()
        self._toggle_log_visibility()
        self.root.after(500, self._update_mem_bar)

    def _build_batch_queue(self, parent):
        """Collapsible batch queue panel."""
        self.batch_frame = ttk.LabelFrame(parent, text="Batch Queue (0 items)", padding=5)

        # Treeview for queue items
        tree_frame = ttk.Frame(self.batch_frame)
        tree_frame.pack(fill="x", pady=(0, 4))
        cols = ("id", "file", "voice", "status", "chunks", "eta")
        self.batch_tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                                       selectmode="extended", height=4)
        self.batch_tree.heading("id", text="#")
        self.batch_tree.heading("file", text="File")
        self.batch_tree.heading("voice", text="Voice")
        self.batch_tree.heading("status", text="Status")
        self.batch_tree.heading("chunks", text="Chunks")
        self.batch_tree.heading("eta", text="ETA")
        self.batch_tree.column("id", width=35, stretch=False)
        self.batch_tree.column("file", width=240)
        self.batch_tree.column("voice", width=110)
        self.batch_tree.column("status", width=75, stretch=False)
        self.batch_tree.column("chunks", width=65, stretch=False)
        self.batch_tree.column("eta", width=55, stretch=False)
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.batch_tree.yview)
        self.batch_tree.configure(yscrollcommand=vsb.set)
        self.batch_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self._drag_item = None
        self.batch_tree.bind("<ButtonPress-1>", self._batch_drag_start)
        self.batch_tree.bind("<B1-Motion>", self._batch_drag_motion)
        self.batch_tree.bind("<ButtonRelease-1>", self._batch_drag_release)
        self.batch_tree.bind("<<TreeviewSelect>>", self._on_batch_select)
        self.batch_tree.bind("<Double-Button-1>", self._on_batch_double_click)

        btn_frame = ttk.Frame(self.batch_frame)
        btn_frame.pack(fill="x")
        self.batch_add_btn = ttk.Button(btn_frame, text="Add to Queue", command=self._add_to_batch)
        self.batch_add_btn.pack(side="left", padx=(0, 4))
        self.batch_add_dir_btn = ttk.Button(btn_frame, text="Add Directory…", command=self._add_dir_to_batch)
        self.batch_add_dir_btn.pack(side="left", padx=(0, 4))
        self.batch_remove_btn = ttk.Button(btn_frame, text="Remove Selected", command=self._remove_from_batch)
        self.batch_remove_btn.pack(side="left", padx=(0, 4))
        self.batch_update_btn = ttk.Button(btn_frame, text="Update Selected",
                                           command=self._update_selected_batch_item)
        self.batch_update_btn.pack(side="left", padx=(0, 4))
        self.batch_edit_btn = ttk.Button(btn_frame, text="Edit Selected…", command=self._edit_selected_batch_item)
        self.batch_edit_btn.pack(side="left", padx=(0, 4))
        self.batch_resume_btn = ttk.Button(btn_frame, text="Resume Selected", command=self._resume_batch_items)
        self.batch_resume_btn.pack(side="left", padx=(0, 4))
        self.batch_clear_btn = ttk.Button(btn_frame, text="Clear Queue", command=self._clear_batch)
        self.batch_clear_btn.pack(side="left", padx=(0, 4))
        self.batch_process_btn = ttk.Button(btn_frame, text="Process Queue", command=self._process_batch)
        self.batch_process_btn.pack(side="left", padx=(0, 4))
        self.batch_down_btn = ttk.Button(btn_frame, text="▼", width=3, command=self._batch_move_down)
        self.batch_down_btn.pack(side="right")
        self.batch_up_btn = ttk.Button(btn_frame, text="▲", width=3, command=self._batch_move_up)
        self.batch_up_btn.pack(side="right", padx=(0, 2))
        self.batch_stop_btn = ttk.Button(btn_frame, text="Stop", width=6,
                                         command=self._cancel, state="disabled")
        self.batch_stop_btn.pack(side="right", padx=(0, 4))
        self.batch_pause_btn = ttk.Button(btn_frame, text="Pause", width=7,
                                          command=self._toggle_pause, state="disabled")
        self.batch_pause_btn.pack(side="right", padx=(0, 2))

        # Collapse toggle
        self.batch_visible = tk.BooleanVar(value=True)
        self.batch_toggle_btn = ttk.Checkbutton(
            parent, text="Show Batch Queue", variable=self.batch_visible,
            command=self._toggle_batch_visibility)
        self.batch_toggle_btn.pack(anchor="w", pady=(0, 2))

    def _toggle_batch_visibility(self):
        if self.batch_visible.get():
            self.batch_frame.pack(fill="x", pady=(0, 5), before=self.batch_toggle_btn.master.winfo_children()[-2])
        else:
            self.batch_frame.pack_forget()

    def _build_log_panel(self, parent):
        """Collapsible detail log panel."""
        self.log_visible = tk.BooleanVar(value=True)
        self.log_toggle_btn = ttk.Checkbutton(
            parent, text="Show Detail Log", variable=self.log_visible,
            command=self._toggle_log_visibility)
        self.log_toggle_btn.pack(anchor="w", pady=(2, 0))

        self.log_frame = ttk.Frame(parent)
        log_text_frame = ttk.Frame(self.log_frame)
        log_text_frame.pack(fill="both", expand=True)
        self.log_area = tk.Text(log_text_frame, height=8, wrap="word",
                                font=("monospace", 9), state="disabled",
                                bg="#1e1e1e", fg="#d4d4d4")
        log_vsb = ttk.Scrollbar(log_text_frame, orient="vertical", command=self.log_area.yview)
        self.log_area.configure(yscrollcommand=log_vsb.set)
        log_vsb.pack(side="right", fill="y")
        self.log_area.pack(side="left", fill="both", expand=True)

        log_btn_row = ttk.Frame(self.log_frame)
        log_btn_row.pack(fill="x")
        ttk.Button(log_btn_row, text="Clear Log", command=self._clear_log).pack(side="left")

    def _toggle_log_visibility(self):
        if self.log_visible.get():
            self.log_frame.pack(fill="both", expand=True, pady=(2, 0))
        else:
            self.log_frame.pack_forget()

    # ── logging ────────────────────────────────────────────────────

    def _log(self, msg, level="info"):
        """Append a timestamped message to the detail log. Thread-safe."""
        levels = {"info": "#d4d4d4", "warn": "#e6b422", "error": "#e06c75",
                  "success": "#98c379", "chunk": "#61afef"}
        color = levels.get(level, "#d4d4d4")
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self._log_buffer.append(line)   # plain-text copy for archive

        def _append():
            self.log_area.configure(state="normal")
            # Tag for coloring
            tag = f"c_{ts.replace(':', '')}"
            self.log_area.insert("end", line, tag)
            self.log_area.tag_config(tag, foreground=color)
            self.log_area.see("end")
            self.log_area.configure(state="disabled")
        self.root.after(0, _append)

    # Logger protocol implementation. Non-GUI modules take `logger: Logger`
    # and call these instead of _log(msg, tag) directly.
    def info(self, msg: str) -> None:
        self._log(msg, "info")

    def warn(self, msg: str) -> None:
        self._log(msg, "warn")

    def error(self, msg: str) -> None:
        self._log(msg, "error")

    def success(self, msg: str) -> None:
        self._log(msg, "success")

    @property
    def buffer(self) -> list[str]:
        return self._log_buffer

    def _clear_log(self):
        self.log_area.configure(state="normal")
        self.log_area.delete("1.0", "end")
        self.log_area.configure(state="disabled")

    # ── batch queue ────────────────────────────────────────────────

    def _add_to_batch(self):
        text = self.text_area.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("Empty", "No text to add to queue.")
            return
        ref = self.ref_path.get() or None
        voice_name = os.path.splitext(os.path.basename(ref))[0] if ref else "(default)"

        item = BatchItem(
            id=self._batch_next_id,
            source_path=self._source_path,
            file_name=os.path.basename(self._source_path) if self._source_path else "(pasted text)",
            text=text,
            ref_path=ref,
            voice_name=voice_name,
            settings=self._snapshot_settings(),
        ).to_dict()
        self._batch_next_id += 1
        self._batch_queue.append(item)
        self._reflect_batch_change()
        self._log(f"Added to queue: {item['file_name']} ({voice_name})", "info")

    def _add_dir_to_batch(self):
        """Custom directory browser — avoids native GTK dialog so file names render
        correctly regardless of the system colour theme.  All widgets use explicit
        colours so they remain readable under dark system themes."""

        SUPPORTED = {'.txt', '.pdf', '.epub', '.md'}

        def _nat_key(s):
            return [int(c) if c.isdigit() else c.lower()
                    for c in re.split(r'(\d+)', s)]

        # Explicit colours + explicit font — immune to dark themes AND to
        # environments where the default font resolves to something invisible
        BG   = "white"
        FG   = "black"
        MUTE = "gray"
        FONT = ("TkFixedFont", 10)   # verified visible via the Listbox below

        # ── dialog shell ────────────────────────────────────────────
        dlg = tk.Toplevel(self.root)
        dlg.title("Add Directory to Queue")
        dlg.geometry("580x420")
        dlg.resizable(True, True)
        dlg.transient(self.root)
        # option_add sets defaults for all widgets in this dialog
        dlg.option_add("*font", "TkFixedFont 10")
        # tk_setPalette cascades colours to all descendants at the Tcl level
        dlg.tk_setPalette(
            background=BG, foreground=FG,
            activeBackground=BG, activeForeground=FG,
            highlightBackground=BG, highlightColor=FG,
            selectBackground="#0078d7", selectForeground="white",
            insertBackground=FG,
        )
        dlg.grab_set()
        dlg.configure(bg=BG)

        # ── directory path row ──────────────────────────────────────
        dir_row = tk.Frame(dlg, bg=BG)
        dir_row.pack(fill="x", padx=10, pady=(10, 4))
        tk.Label(dir_row, text="Directory:", bg=BG, fg=FG, font=FONT).pack(side="left")
        dir_var = tk.StringVar()
        dir_entry = tk.Entry(dir_row, textvariable=dir_var,
                             bg=BG, fg=FG, insertbackground=FG, font=FONT,
                             highlightthickness=1, highlightbackground="#888888",
                             relief="solid", bd=1)
        dir_entry.pack(side="left", fill="x", expand=True, padx=(4, 4))

        # ── file listbox (explicit colours — immune to theme) ───────
        list_frame = tk.Frame(dlg, bg=BG)
        list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 2))

        listbox = tk.Listbox(
            list_frame,
            selectmode="extended",
            bg=BG, fg=FG,
            selectbackground="#0078d7", selectforeground="white",
            activestyle="none",
            font=FONT,
        )
        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=listbox.yview)
        listbox.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        listbox.pack(side="left", fill="both", expand=True)

        count_lbl = tk.Label(dlg, text="No directory selected.",
                             bg=BG, fg=MUTE, anchor="w", font=FONT)
        count_lbl.pack(fill="x", padx=10, pady=(0, 2))

        _files = []   # filenames matching SUPPORTED, in sorted order

        def _populate(directory):
            listbox.delete(0, "end")
            _files.clear()
            if not directory:
                return
            if not os.path.isdir(directory):
                count_lbl.config(text="Directory not found.", fg="red")
                return
            try:
                found = [
                    f for f in os.listdir(directory)
                    if os.path.isfile(os.path.join(directory, f))
                    and os.path.splitext(f)[1].lower() in SUPPORTED
                ]
            except OSError as e:
                count_lbl.config(text=f"Cannot read directory: {e}", fg="red")
                return
            _files.extend(sorted(found, key=_nat_key))
            for f in _files:
                listbox.insert("end", f)
            listbox.select_set(0, "end")   # pre-select all
            n = len(_files)
            count_lbl.config(
                text=f"{n} file(s) — all selected." if n else "No .txt / .md / .pdf / .epub files found.",
                fg=MUTE if n else "#c76a00",
            )

        def _browse():
            chosen = filedialog.askdirectory(
                title="Select directory",
                initialdir=dir_var.get() or os.path.expanduser("~"),
                parent=dlg,
            )
            if chosen:
                dir_var.set(chosen)
                _populate(chosen)

        tk.Button(dir_row, text="Browse…",
                  bg=BG, fg=FG, activebackground=BG, activeforeground=FG,
                  font=FONT,
                  command=_browse).pack(side="left")
        dir_entry.bind("<Return>",   lambda _e: _populate(dir_var.get()))
        dir_entry.bind("<FocusOut>", lambda _e: _populate(dir_var.get()) if dir_var.get() else None)

        # ── select all / deselect all ────────────────────────────────
        sel_row = tk.Frame(dlg, bg=BG)
        sel_row.pack(fill="x", padx=10, pady=(0, 6))
        tk.Button(sel_row, text="Select All",
                  bg=BG, fg=FG, activebackground=BG, activeforeground=FG,
                  font=FONT,
                  command=lambda: (listbox.select_set(0, "end"),
                                   count_lbl.config(text=f"{len(_files)} file(s) — all selected.",
                                                    fg=MUTE))
                  ).pack(side="left", padx=(0, 4))
        tk.Button(sel_row, text="Deselect All",
                  bg=BG, fg=FG, activebackground=BG, activeforeground=FG,
                  font=FONT,
                  command=lambda: (listbox.select_clear(0, "end"),
                                   count_lbl.config(text=f"{len(_files)} file(s) — none selected.",
                                                    fg=MUTE))
                  ).pack(side="left")

        # ── OK / Cancel ──────────────────────────────────────────────
        def _do_add():
            sel_indices = listbox.curselection()
            if not sel_indices:
                messagebox.showwarning("Nothing selected",
                                       "Select at least one file.", parent=dlg)
                return
            directory = dir_var.get()
            ref        = self.ref_path.get() or None
            voice_name = (os.path.splitext(os.path.basename(ref))[0]
                          if ref else "(default)")
            settings   = self._snapshot_settings()

            added, errors = 0, []
            for i in sel_indices:
                fname = _files[i]
                path  = os.path.join(directory, fname)
                try:
                    text = load_text_file(path)
                    if not text.strip():
                        continue
                    self._batch_queue.append(BatchItem(
                        id=self._batch_next_id,
                        source_path=path,
                        file_name=fname,
                        text=text,
                        ref_path=ref,
                        voice_name=voice_name,
                        settings=settings,
                    ).to_dict())
                    self._batch_next_id += 1
                    added += 1
                except Exception as e:
                    errors.append(f"{fname}: {e}")

            if added:
                self._reflect_batch_change()
                summary = f"Added {added} file(s) from {os.path.basename(directory)}/."
                if errors:
                    summary += f"  ({len(errors)} skipped — see log)"
                    for msg in errors:
                        self._log(f"  Skipped: {msg}", "warn")
                self._log(summary, "info")
                dlg.destroy()
            else:
                messagebox.showerror("Nothing added",
                                     "No files could be loaded.\n\n" + "\n".join(errors[:8]),
                                     parent=dlg)

        bf = tk.Frame(dlg, bg=BG)
        bf.pack(fill="x", padx=10, pady=(0, 10))
        tk.Button(bf, text="Cancel",
                  bg=BG, fg=FG, activebackground=BG, activeforeground=FG,
                  font=FONT,
                  command=dlg.destroy).pack(side="right")
        tk.Button(bf, text="Add to Queue",
                  bg=BG, fg=FG, activebackground=BG, activeforeground=FG,
                  font=FONT,
                  command=_do_add).pack(side="right", padx=(0, 4))

        # Pre-populate from the directory of the currently loaded file
        initial = (os.path.dirname(self._source_path)
                   if getattr(self, "_source_path", None) else "")
        if initial and os.path.isdir(initial):
            dir_var.set(initial)
            _populate(initial)

        dir_entry.focus_set()

    def _remove_from_batch(self):
        sel = self.batch_tree.selection()
        if not sel:
            return
        ids = {int(self.batch_tree.item(s, "values")[0]) for s in sel}
        for q in self._batch_queue:
            if q["id"] in ids and q["status"] != "processing":
                self._cleanup_item_sidecar(q)
        self._batch_queue = [q for q in self._batch_queue if q["id"] not in ids or q["status"] == "processing"]
        self._reflect_batch_change()

    def _resume_batch_items(self):
        """Reset selected items to pending; resume from partial job if one exists."""
        sel = self.batch_tree.selection()
        if not sel:
            return
        ids = {int(self.batch_tree.item(s, "values")[0]) for s in sel}

        candidates = [q for q in self._batch_queue
                      if q["id"] in ids and q["status"] != "processing"]
        if not candidates:
            return

        # Check upfront (with pronunciation applied, matching _do_generate's hash)
        pron_dict = load_pron_dict()
        found_lines = []   # items with a partial job — show reconciliation result
        no_job = []        # items with no partial job on disk

        for q in candidates:
            resume_text = apply_pronunciation(q["text"], pron_dict) if pron_dict else q["text"]
            job_dir, state = self._find_resumable_job(resume_text, _hint_hash=q.get("_job_hash"))
            if not job_dir and pron_dict and resume_text != q["text"]:
                # Pron dict may have changed since this job was created — try raw text
                job_dir, state = self._find_resumable_job(q["text"], _hint_hash=q.get("_job_hash"))
            if not job_dir:
                no_job.append(q["file_name"])
                continue
            _, disk_n, json_n, changed = self._reconcile_job_state(job_dir, state)
            total_n = state.get("total_chunks", 0)
            if changed:
                found_lines.append(
                    f"• {q['file_name']}: state.json={json_n}, disk={disk_n}/{total_n} — corrected")
            else:
                found_lines.append(
                    f"• {q['file_name']}: {disk_n}/{total_n} chunks verified ✓")

        msg_parts = []
        if found_lines:
            msg_parts.append("Partial jobs found:\n" + "\n".join(found_lines))
        if no_job:
            msg_parts.append("No saved progress (will start fresh):\n• " + "\n• ".join(no_job))

        if not messagebox.askyesno("Resume Batch Items",
                                   "\n\n".join(msg_parts) + "\n\nContinue?"):
            return

        for q in candidates:
            q["status"] = "pending"
            q["error"] = ""
            q["output_path"] = ""
            q["try_resume"] = True

        self._reflect_batch_change()
        if not self._batch_running:
            self._process_batch()

    def _clear_batch(self):
        removable = [q for q in self._batch_queue if q["status"] != "processing"]
        if not removable:
            return
        if messagebox.askyesno("Clear Queue", f"Remove {len(removable)} non-active item(s)?"):
            for q in removable:
                self._cleanup_item_sidecar(q)
            self._batch_queue = [q for q in self._batch_queue if q["status"] == "processing"]
            if not self._batch_queue:
                self._batch_next_id = 0
            self._reflect_batch_change()

    def _refresh_batch_tree(self):
        self.batch_tree.delete(*self.batch_tree.get_children())
        for item in self._batch_queue:
            cd = item.get("chunks_done", 0)
            ct = item.get("chunks_total", 0)
            chunks_str = f"{cd}/{ct}" if ct > 0 else ""
            status = item.get("status", "")
            if status == "processing":
                eta_str = item.get("_eta_str", "")
            else:
                eta_str = item.get("gen_time", "")
            self.batch_tree.insert("", "end", values=(
                item["id"], item["file_name"], item["voice_name"], item["status"],
                chunks_str, eta_str))
        self.batch_frame.configure(text=f"Batch Queue ({len(self._batch_queue)} items)")

    def _on_batch_select(self, _event=None):
        """Reflect the selected batch item's settings in the Advanced Options panel."""
        sel = self.batch_tree.selection()
        if not sel:
            return
        row_ids = self.batch_tree.get_children()
        try:
            idx = row_ids.index(sel[0])
            item = self._batch_queue[idx]
        except (ValueError, IndexError):
            return
        if item.get("settings"):
            self._apply_settings_to_ui(item["settings"])
        ref = item.get("ref_path") or item.get("voice_path")
        if ref and os.path.exists(ref):
            self.ref_path.set(ref)
            self.ref_label.config(text=item.get("voice_name", ""), foreground="black")
            self.play_ref_btn.config(state="normal")
            stem = os.path.splitext(os.path.basename(ref))[0]
            if stem in self._voice_map:
                self.voice_combo_var.set(stem)

    def _on_batch_double_click(self, _event=None):
        """Double-click a batch row: focus the main panel, ready for edits.
        Change voice / advanced options in the main panel, then click
        'Update Selected' to save those changes back to the item."""
        # _on_batch_select has already applied the item's settings to the UI;
        # this is a no-op that keeps double-click responsive.
        return

    def _update_selected_batch_item(self):
        """Write the current main-panel voice + advanced options back to every
        selected batch item.  Use this after selecting one or more batch rows
        and modifying anything you want to change for those items.  Items in
        'processing' state are skipped."""
        sel = self.batch_tree.selection()
        if not sel:
            messagebox.showinfo("No selection",
                                "Select one or more batch items first, then "
                                "change the voice or Advanced Options in the "
                                "main panel, then click Update Selected.",
                                parent=self.root)
            return
        row_ids = self.batch_tree.get_children()

        new_ref = self.ref_path.get() or None
        new_voice_name = (os.path.splitext(os.path.basename(new_ref))[0]
                          if new_ref else "(default)")
        new_settings = self._snapshot_settings()

        updated_indices = []
        updated_items   = []
        skipped         = 0
        for row_id in sel:
            try:
                idx = row_ids.index(row_id)
                item = self._batch_queue[idx]
            except (ValueError, IndexError):
                continue
            if item.get("status") == "processing":
                skipped += 1
                continue
            item["settings"]   = new_settings
            item["ref_path"]   = new_ref
            item["voice_name"] = new_voice_name
            updated_indices.append(idx)
            updated_items.append(item)

        if not updated_items:
            messagebox.showinfo(
                "Nothing updated",
                "All selected items are currently processing. Wait for them "
                "to finish, or stop the batch, before updating them.",
                parent=self.root)
            return

        self._refresh_batch_tree()
        self._save_batch_queue()

        # Re-select the same rows so the UI stays in sync with the tree
        children = self.batch_tree.get_children()
        restore = [children[i] for i in updated_indices if i < len(children)]
        if restore:
            self.batch_tree.selection_set(restore)

        if len(updated_items) == 1:
            item = updated_items[0]
            self._log(
                f"Updated batch item #{item['id']} ({item['file_name']}) — "
                f"voice: {new_voice_name}", "info")
        else:
            ids = ", ".join(f"#{it['id']}" for it in updated_items)
            self._log(
                f"Updated {len(updated_items)} batch items ({ids}) — "
                f"voice: {new_voice_name}", "info")
        if skipped:
            self._log(
                f"Skipped {skipped} item(s) currently processing.", "warn")

    def _edit_selected_batch_item(self):
        """Open a modal editor for the selected batch item's voice + advanced options.
        Saves back to the queue item; items in 'processing' state are read-only."""
        sel = self.batch_tree.selection()
        if not sel:
            messagebox.showinfo("No selection", "Select a batch item to edit.", parent=self.root)
            return
        row_ids = self.batch_tree.get_children()
        try:
            idx = row_ids.index(sel[0])
            item = self._batch_queue[idx]
        except (ValueError, IndexError):
            return
        if item.get("status") == "processing":
            messagebox.showinfo(
                "Item is processing",
                "This item is currently being generated. Wait for it to finish, "
                "or stop the batch, before editing.", parent=self.root)
            return

        current = item.get("settings") or self._snapshot_settings()

        # Simplest possible dialog: no ttk, no LabelFrame, no tk_setPalette,
        # explicit font+colours on every widget.  Uses tk.OptionMenu for the
        # voice picker (a pure tk widget with no theme dependency).
        BG   = "white"
        FG   = "black"
        FONT = ("Helvetica", 11)  # a Tk font alias that always resolves

        dlg = tk.Toplevel(self.root)
        dlg.title(f"Edit Batch Item — {item['file_name']}")
        dlg.geometry("680x440")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.configure(bg=BG)

        def _label(parent, text, font=FONT, **kw):
            return tk.Label(parent, text=text, bg=BG, fg=FG, font=font, **kw)

        def _entry(parent, var):
            return tk.Entry(parent, textvariable=var, width=10,
                            bg=BG, fg=FG, insertbackground=FG, font=FONT,
                            highlightthickness=1, highlightbackground="#888888",
                            relief="solid", bd=1)

        def _check(parent, text, var):
            return tk.Checkbutton(parent, text=text, variable=var,
                                  bg=BG, fg=FG, font=FONT,
                                  activebackground=BG, activeforeground=FG,
                                  selectcolor=BG, anchor="w")

        def _button(parent, text, cmd):
            return tk.Button(parent, text=text, bg=BG, fg=FG, font=FONT,
                             activebackground=BG, activeforeground=FG,
                             command=cmd)

        # ── voice section ─────────────────────────────────────────
        _label(dlg, "── Reference Voice ──", font=("Helvetica", 11, "bold")
              ).pack(anchor="w", padx=12, pady=(6, 2))

        voice_row = tk.Frame(dlg, bg=BG)
        voice_row.pack(fill="x", padx=12, pady=(0, 8))

        voice_var = tk.StringVar()
        current_ref = item.get("ref_path") or ""
        current_stem = (os.path.splitext(os.path.basename(current_ref))[0]
                        if current_ref else "")
        voice_names = ["(default prompt)"] + sorted(self._voice_map.keys())
        if current_stem and current_stem in self._voice_map:
            voice_var.set(current_stem)
        else:
            voice_var.set("(default prompt)")

        # ttk.Combobox — has a scrollable dropdown; bound the popup to 15 rows.
        # Works cleanly now that the clam theme + explicit palette are in place.
        voice_menu = ttk.Combobox(voice_row, textvariable=voice_var,
                                  values=voice_names, state="readonly",
                                  width=30, height=15, font=FONT)
        voice_menu.pack(side="left")

        ref_label_var = tk.StringVar(
            value=(os.path.basename(current_ref) if current_ref else "(default prompt)"))
        _label(voice_row, "").config(textvariable=ref_label_var, fg="#555555")

        custom_ref_holder = {"path": None}
        def browse_ref():
            f = filedialog.askopenfilename(
                title="Select reference voice WAV",
                initialdir=self._voice_samples_dir,
                filetypes=[("WAV files", "*.wav"), ("All", "*.*")],
                parent=dlg)
            if f:
                stem = os.path.splitext(os.path.basename(f))[0]
                voice_var.set(stem if stem in self._voice_map else "(default prompt)")
                ref_label_var.set(os.path.basename(f))
                custom_ref_holder["path"] = f
        _button(voice_row, "Browse WAV…", browse_ref).pack(side="left", padx=(10, 0))

        # ── advanced options ──────────────────────────────────────
        _label(dlg, "── Advanced Options ──", font=("Helvetica", 11, "bold")
              ).pack(anchor="w", padx=12, pady=(6, 2))

        adv = tk.Frame(dlg, bg=BG)
        adv.pack(fill="both", expand=True, padx=12, pady=(0, 6))

        v = {
            "temperature":        tk.DoubleVar(value=current.get("temperature", 0.8)),
            "min_p":              tk.DoubleVar(value=current.get("min_p", 0.0)),
            "top_p":              tk.DoubleVar(value=current.get("top_p", 0.95)),
            "top_k":              tk.IntVar(value=int(current.get("top_k", 1000))),
            "repetition_penalty": tk.DoubleVar(value=current.get("repetition_penalty", 1.2)),
            "chunk_size":         tk.IntVar(value=int(current.get("chunk_size", 200))),
            "seed":               tk.IntVar(value=int(current.get("seed", 0))),
            "cpu_threads":        tk.IntVar(value=int(current.get("cpu_threads", 4))),
            "lead_silence":       tk.DoubleVar(value=current.get("lead_silence", 0.0)),
            "trail_silence":      tk.DoubleVar(value=current.get("trail_silence", 0.0)),
            "crossfade_ms":       tk.IntVar(value=int(current.get("crossfade_ms", 50))),
            "reload_every":       tk.IntVar(value=int(current.get("reload_every", 0))),
            "prune_every":        tk.IntVar(value=int(current.get("prune_every", 1))),
            "norm_loudness":      tk.BooleanVar(value=bool(current.get("norm_loudness", True))),
            "split_clauses":      tk.BooleanVar(value=bool(current.get("split_clauses", True))),
            "auto_mp3":           tk.BooleanVar(value=bool(current.get("auto_mp3", False))),
        }

        left = [
            ("Temperature",         "temperature"),
            ("Min-P",               "min_p"),
            ("Top-P",               "top_p"),
            ("Top-K",               "top_k"),
            ("Repetition penalty",  "repetition_penalty"),
            ("Chunk size (chars)",  "chunk_size"),
            ("Seed (0 = random)",   "seed"),
        ]
        right = [
            ("CPU threads",             "cpu_threads"),
            ("Lead silence (s)",        "lead_silence"),
            ("Trail silence (s)",       "trail_silence"),
            ("Crossfade (ms, 0=off)",   "crossfade_ms"),
            ("Reload every N (0=off)",  "reload_every"),
            ("Prune every N (1=every)", "prune_every"),
        ]
        for i, (label, key) in enumerate(left):
            _label(adv, label, anchor="e", width=22).grid(row=i, column=0, sticky="e", padx=(4, 4), pady=2)
            _entry(adv, v[key]).grid(row=i, column=1, sticky="w", padx=(0, 16), pady=2)
        for i, (label, key) in enumerate(right):
            _label(adv, label, anchor="e", width=22).grid(row=i, column=2, sticky="e", padx=(4, 4), pady=2)
            _entry(adv, v[key]).grid(row=i, column=3, sticky="w", padx=(0, 4), pady=2)

        bool_row = len(left)
        _check(adv, "Normalize loudness",         v["norm_loudness"]).grid(
            row=bool_row,     column=0, columnspan=2, sticky="w", padx=4, pady=(8, 2))
        _check(adv, "Split at clause boundaries", v["split_clauses"]).grid(
            row=bool_row + 1, column=0, columnspan=2, sticky="w", padx=4, pady=2)
        _check(adv, "Auto-convert to MP3",        v["auto_mp3"]).grid(
            row=bool_row + 2, column=0, columnspan=2, sticky="w", padx=4, pady=2)

        # ── save / cancel ─────────────────────────────────────────
        def save():
            new_ref = custom_ref_holder["path"]
            if new_ref:
                new_voice_name = os.path.splitext(os.path.basename(new_ref))[0]
            else:
                sel_name = voice_var.get()
                if sel_name and sel_name != "(default prompt)" and sel_name in self._voice_map:
                    new_ref = self._voice_map[sel_name]
                    new_voice_name = sel_name
                else:
                    new_voice_name = "(default)"

            new_settings = {k: var.get() for k, var in v.items()}
            new_settings["ref_path"] = new_ref

            item["settings"]   = new_settings
            item["ref_path"]   = new_ref
            item["voice_name"] = new_voice_name

            self._refresh_batch_tree()
            self._save_batch_queue()

            children = self.batch_tree.get_children()
            if idx < len(children):
                self.batch_tree.selection_set(children[idx])
                self._on_batch_select()

            self._log(f"Edited batch item #{item['id']}: voice={new_voice_name}", "info")
            dlg.destroy()

        btn_row = tk.Frame(dlg, bg=BG)
        btn_row.pack(fill="x", padx=12, pady=(0, 12))
        _button(btn_row, "Cancel", dlg.destroy).pack(side="right")
        _button(btn_row, "Save",   save).pack(side="right", padx=(6, 6))

    # ── batch reorder ──────────────────────────────────────────────

    def _batch_move_up(self):
        if self._batch_running:
            return
        sel = self.batch_tree.selection()
        if not sel:
            return
        row_ids = self.batch_tree.get_children()
        idx = row_ids.index(sel[0])
        if idx == 0:
            return
        if self._batch_queue[idx].get("status") == "processing" or \
                self._batch_queue[idx - 1].get("status") == "processing":
            return
        self._batch_queue[idx - 1], self._batch_queue[idx] = \
            self._batch_queue[idx], self._batch_queue[idx - 1]
        self._refresh_batch_tree()
        self.batch_tree.selection_set(self.batch_tree.get_children()[idx - 1])
        self._save_batch_queue()

    def _batch_move_down(self):
        if self._batch_running:
            return
        sel = self.batch_tree.selection()
        if not sel:
            return
        row_ids = self.batch_tree.get_children()
        idx = row_ids.index(sel[0])
        if idx >= len(self._batch_queue) - 1:
            return
        if self._batch_queue[idx].get("status") == "processing" or \
                self._batch_queue[idx + 1].get("status") == "processing":
            return
        self._batch_queue[idx], self._batch_queue[idx + 1] = \
            self._batch_queue[idx + 1], self._batch_queue[idx]
        self._refresh_batch_tree()
        self.batch_tree.selection_set(self.batch_tree.get_children()[idx + 1])
        self._save_batch_queue()

    def _batch_drag_start(self, event):
        item = self.batch_tree.identify_row(event.y)
        self._drag_item = item if item else None

    def _batch_drag_motion(self, event):
        if not self._drag_item or self._batch_running:
            return
        target = self.batch_tree.identify_row(event.y)
        if not target or target == self._drag_item:
            return
        row_ids = self.batch_tree.get_children()
        src = row_ids.index(self._drag_item)
        tgt = row_ids.index(target)
        if self._batch_queue[src].get("status") == "processing" or \
                self._batch_queue[tgt].get("status") == "processing":
            return
        self._batch_queue[src], self._batch_queue[tgt] = \
            self._batch_queue[tgt], self._batch_queue[src]
        self._refresh_batch_tree()
        self._drag_item = self.batch_tree.get_children()[tgt]
        self.batch_tree.selection_set(self._drag_item)

    def _batch_drag_release(self, event):
        if self._drag_item:
            self._save_batch_queue()
        self._drag_item = None

    def _save_batch_queue(self):
        """Persist the batch queue to disk so it survives restarts.

        Text is excluded from the JSON — it is reloaded from source_path on
        startup.  For pasted text (no source file) a sidecar .txt is written
        alongside the queue file.
        """
        items_to_save = []
        for item in self._batch_queue:
            entry = {k: v for k, v in item.items()
                     if k not in ("text", "_eta_str")}
            if not item.get("source_path") and item.get("text"):
                sidecar = os.path.join(
                    os.path.dirname(BATCH_QUEUE_PATH),
                    f"tts-book.queue-text-{item['id']}.txt",
                )
                try:
                    with open(sidecar, "w", encoding="utf-8") as f:
                        f.write(item["text"])
                    entry["_text_sidecar"] = sidecar
                except Exception:
                    pass
            items_to_save.append(entry)
        data = {"next_id": self._batch_next_id, "items": items_to_save}
        try:
            with open(BATCH_QUEUE_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # disk full / permission — non-critical

    def _load_batch_queue(self):
        """Restore the batch queue from disk on startup."""
        if not os.path.isfile(BATCH_QUEUE_PATH):
            return
        try:
            with open(BATCH_QUEUE_PATH) as f:
                data = json.load(f)
            self._batch_next_id = data.get("next_id", 0)
            self._batch_queue = data.get("items", [])
            for item in self._batch_queue:
                # Reset any "processing" items back to "pending" (crashed mid-run)
                if item.get("status") == "processing":
                    item["status"] = "pending"
                    item["error"] = ""
                # Re-load text that was stripped from the saved JSON
                if "text" not in item:
                    if item.get("source_path"):
                        try:
                            item["text"] = load_text_file(item["source_path"])
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
            self._refresh_batch_tree()
            self._log(f"Restored batch queue: {len(self._batch_queue)} items", "info")
        except Exception:
            pass

    def _cleanup_item_sidecar(self, item):
        sidecar = item.get("_text_sidecar")
        if sidecar and os.path.isfile(sidecar):
            try:
                os.remove(sidecar)
            except Exception:
                pass

    def _reflect_batch_change(self):
        """Refresh tree + persist after any queue modification."""
        self._refresh_batch_tree()
        self._save_batch_queue()

    def _process_batch(self):
        if self._batch_running:
            return
        if not self._batch_queue:
            messagebox.showwarning("Empty Queue", "Add items to the queue first.")
            return
        pending = [q for q in self._batch_queue if q["status"] == "pending"]
        if not pending:
            messagebox.showinfo("Queue Done", "All items already processed.")
            return

        self._log(f"=== Batch processing {len(pending)} items ===", "info")
        self._log(f"  Items: {', '.join(q['file_name'] for q in pending)}", "info")

        self.cancel_token.reset()
        self._batch_running = True
        self._lock_ui(batch=True)
        self.pause_btn.config(state="normal")
        self.cancel_btn.config(state="normal")
        self.batch_pause_btn.config(state="normal", text="Pause")
        self.batch_stop_btn.config(state="normal")
        self.gen_btn.config(state="disabled")
        self.resume_btn.config(state="disabled")
        self.batch_process_btn.config(state="disabled")

        t = threading.Thread(target=self._batch_thread, daemon=True)
        t.start()

    def _batch_thread(self):
        """Process batch queue items sequentially in background thread."""
        processed = 0

        while True:
            if self.cancel_token.cancelled:
                self.root.after(0, lambda: self._log("Batch cancelled by user.", "warn"))
                break

            # Honour pause between items (not just between chunks inside _do_generate)
            self.cancel_token.wait_if_paused()
            if self.cancel_token.cancelled:
                self.root.after(0, lambda: self._log("Batch cancelled by user.", "warn"))
                break

            # Pick next pending item dynamically so items added mid-run are included
            item = next((q for q in self._batch_queue if q["status"] == "pending"), None)
            if item is None:
                break

            processed += 1
            item["status"] = "processing"
            self._save_batch_queue()
            self.root.after(0, self._refresh_batch_tree)
            remaining = sum(1 for q in self._batch_queue if q["status"] == "pending")
            self.root.after(0, lambda fn=item["file_name"], r=remaining:
                            self.status.config(
                                text=f"Batch: {fn} ({r} remaining)" if r else f"Batch: {fn}",
                                foreground="black"))

            self._log(f"--- Batch item {processed}: {item['file_name']} ---", "info")
            self._log(f"  Voice: {item['voice_name']}", "info")
            self._log("  Chunks: estimating...", "info")
            try:
                resume_job_dir, resume_state = None, None
                item.pop("try_resume", None)  # consume flag; lookup always runs
                _pron = load_pron_dict()
                _rtxt = apply_pronunciation(item["text"], _pron) if _pron else item["text"]
                resume_job_dir, resume_state = self._find_resumable_job(
                    _rtxt, _hint_hash=item.get("_job_hash"))
                if not resume_job_dir and _pron and _rtxt != item["text"]:
                    resume_job_dir, resume_state = self._find_resumable_job(
                        item["text"], _hint_hash=item.get("_job_hash"))
                if resume_job_dir:
                    resume_state, disk_n, json_n, changed = self._reconcile_job_state(
                        resume_job_dir, resume_state)
                    total_n = resume_state.get("total_chunks", 0)
                    if changed:
                        self._log(
                            f"  Reconciled: state.json had {json_n}, "
                            f"disk has {disk_n}/{total_n} — updated", "warn")
                    self._log(f"  Resuming: {disk_n}/{total_n} chunks done", "info")
                    item["chunks_done"] = disk_n
                    item["chunks_total"] = total_n
                    self.root.after(0, self._refresh_batch_tree)
                else:
                    self._log("  Starting fresh (no partial job found)", "info")
                output_path = self._do_generate(
                    text=item["text"],
                    settings=item["settings"],
                    ref_path=item["ref_path"],
                    source_path=item["source_path"],
                    resume=bool(resume_job_dir),
                    job_dir=resume_job_dir,
                    state=resume_state,
                    batch_item=item,
                )
                if output_path:
                    item["status"] = "done"
                    item["output_path"] = output_path
                    ct = item.get("chunks_total", 0)
                    if ct > 0:
                        item["chunks_done"] = ct
                    self._last_output = output_path
                    self._log(f"  ✓ Done: {output_path}", "success")
                    self._save_batch_queue()
                    if self.auto_mp3.get():
                        self.root.after(0, lambda p=output_path: self._auto_convert(p))
                elif self.cancel_token.cancelled:
                    item["status"] = "pending"  # keep for resume
                    self._log("  ⏸ Cancelled (partial state saved)", "warn")
                    break
            except Exception as e:
                item["status"] = "error"
                item["error"] = str(e)
                self._log(f"  ✗ Error: {e}", "error")
                self._save_batch_queue()
                if self.cancel_token.cancelled:
                    break

            self.root.after(0, self._refresh_batch_tree)

        self._batch_running = False
        done_count = sum(1 for q in self._batch_queue if q["status"] == "done")
        self.root.after(0, lambda: self._log(
            f"=== Batch complete: {done_count}/{processed} processed ===",
            "success" if done_count == processed else "warn"))
        self.root.after(0, lambda: self._on_batch_done())

    def _on_batch_done(self):
        self._unlock_ui()
        self.progress.config(value=100)
        self.pause_btn.config(state="disabled", text="Pause")
        self.cancel_btn.config(state="disabled")
        self.batch_pause_btn.config(state="disabled", text="Pause")
        self.batch_stop_btn.config(state="disabled")
        self.batch_process_btn.config(state="normal")
        self.gen_btn.config(state="normal")
        if hasattr(self, "_last_output") and self._last_output:
            self.play_btn.config(state="normal")
            self.mp3_btn.config(state="normal")
        self._refresh_batch_tree()
        self._save_batch_queue()


    # ── actions ────────────────────────────────────────────────────

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open text file",
            filetypes=[("Text/MD/EPUB/PDF", "*.txt *.md *.epub *.pdf"), ("All files", "*")]
        )
        if not path:
            return
        try:
            text = load_text_file(path)
            self.text_area.delete("1.0", "end")
            self.text_area.insert("1.0", text)
            self._source_path = path
            self.file_label.config(text=os.path.basename(path), foreground="black")
            self._check_resumable()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")

    def _clear_text(self):
        self.text_area.delete("1.0", "end")
        self._source_path = None
        self.file_label.config(text="No file loaded", foreground="gray")
        self.resume_btn.config(state="disabled")
        self.out_label.config(text="")
        self._startup_job = None

    def _analyze_text(self):
        text = self.text_area.get("1.0", "end-1c").strip()
        if not text:
            self.status.config(text="No text to analyze.", foreground="orange")
            return
        sentences = split_text(text, max_chars=9999, split_clauses=self.split_clauses.get())
        if not sentences:
            self.status.config(text="No sentences found.", foreground="orange")
            return
        lengths = [len(s) for s in sentences]
        pct95 = int(statistics.quantiles(lengths, n=100)[94]) if len(lengths) >= 2 else lengths[0]
        rounded = max(50, min(500, ((pct95 + 24) // 25) * 25))
        self.chunk_size.set(rounded)
        self.status.config(
            text=f"Chunk size set to {rounded} (95th pct of {len(sentences)} sentences, max {max(lengths)} chars)",
            foreground="green")
        self._log(f"Analyzed: {len(sentences)} sentences, 95th pct={pct95}, chunk size→{rounded}", "info")

    def _check_resumable(self):
        text = self.text_area.get("1.0", "end-1c").strip()
        jd, st = self._find_resumable_job(text)
        if jd:
            done = len(st.get("completed", []))
            total = st.get("total_chunks", 0)
            self.resume_btn.config(state="normal")
            self.out_label.config(
                text=f"Resume available: {done}/{total} chunks done",
                foreground="blue")
        else:
            self.resume_btn.config(state="disabled")
            self.out_label.config(text="")

    def _scan_partial_jobs(self):
        """Scan all job dirs at startup. If exactly one partial job, auto-restore it.
        If multiple, show a picker dialog so the user can choose."""
        if not os.path.isdir(self._jobs_dir):
            return

        partials = []  # [(job_dir, state, mtime), ...]
        for name in os.listdir(self._jobs_dir):
            jd = os.path.join(self._jobs_dir, name)
            sp = os.path.join(jd, "state.json")
            if not os.path.isfile(sp):
                continue
            try:
                with open(sp) as f:
                    state = json.load(f)
                completed = set(state.get("completed", []))
                total = state.get("total_chunks", 0)
                if total > 0 and len(completed) < total:
                    mtime = os.path.getmtime(sp)
                    partials.append((jd, state, mtime))
            except Exception:
                pass

        if not partials:
            return

        if len(partials) == 1:
            # Single partial job — auto-restore (existing behavior)
            self._restore_partial_job(*partials[0])
        else:
            # Multiple — show picker
            self.root.after(50, lambda: self._show_resume_picker(partials))

    def _restore_partial_job(self, jd, state, _mtime):
        """Restore a partial job into the UI (called by scan or picker)."""
        chunks = state.get("chunks", [])
        if not chunks:
            # Chunks were moved to a separate file in newer versions
            chunks_file = os.path.join(jd, "chunks.json")
            if os.path.exists(chunks_file):
                try:
                    with open(chunks_file) as f:
                        chunks = json.load(f).get("chunks", [])
                except Exception:
                    pass
        if not chunks:
            return
        done = len(state.get("completed", []))
        total = state.get("total_chunks", 0)

        self._startup_job = (jd, state)

        # Restore reference voice
        ref = state.get("settings", {}).get("ref_path") or ""
        if ref and os.path.exists(ref):
            self.ref_path.set(ref)
            self.ref_label.config(text=os.path.basename(ref), foreground="black")
            self.play_ref_btn.config(state="normal")
            self._populate_voice_combo()
            stem = os.path.splitext(os.path.basename(ref))[0]
            if stem in self._voice_map:
                self.voice_combo_var.set(stem)

        # Reflect saved settings in Advanced Options
        if state.get("settings"):
            self._apply_settings_to_ui(state["settings"])

        # Reconstruct and show text
        text = " ".join(chunks)
        self.text_area.delete("1.0", "end")
        self.text_area.insert("1.0", text)
        self.file_label.config(text="(recovered from partial job)", foreground="blue")
        self.resume_btn.config(state="normal")
        self.out_label.config(
            text=f"Resume available: {done}/{total} chunks done",
            foreground="blue")
        self.status.config(text="Found partial job — click Resume to continue", foreground="blue")

    def _show_resume_picker(self, partials):
        """Dialog: pick which partial job to resume when multiple exist."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Select Job to Resume")
        dlg.geometry("680x420")
        dlg.transient(self.root)
        dlg.grab_set()

        ttk.Label(dlg, text=f"Found {len(partials)} partial jobs. Select one to resume:",
                  padding=10).pack()

        tree_frame = ttk.Frame(dlg)
        tree_frame.pack(fill="both", expand=True, padx=10, pady=(0, 5))
        cols = ("preview", "progress", "voice", "date")
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="browse")
        tree.heading("preview", text="Text Preview")
        tree.heading("progress", text="Progress")
        tree.heading("voice", text="Voice")
        tree.heading("date", text="Last Modified")
        tree.column("preview", width=280)
        tree.column("progress", width=100)
        tree.column("voice", width=120)
        tree.column("date", width=130)
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Sort by most recent first
        partials.sort(key=lambda x: x[2], reverse=True)
        for jd, state, mtime in partials:
            chunks = state.get("chunks", [])
            if not chunks:
                chunks_file = os.path.join(jd, "chunks.json")
                if os.path.exists(chunks_file):
                    try:
                        with open(chunks_file) as f:
                            chunks = json.load(f).get("chunks", [])
                    except Exception:
                        pass
            joined = " ".join(chunks)
            preview = joined[:80] + "…" if len(joined) > 80 else joined
            done = len(state.get("completed", []))
            total = state.get("total_chunks", 0)
            ref = (state.get("settings") or {}).get("ref_path", "")
            voice = os.path.splitext(os.path.basename(ref))[0] if ref else "(default)"
            date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            tree.insert("", "end", values=(preview, f"{done}/{total}", voice, date_str),
                        tags=(jd,))

        btn_frame = ttk.Frame(dlg, padding=10)
        btn_frame.pack(fill="x")
        selected_jd = [None]

        def on_select():
            sel = tree.selection()
            if sel:
                selected_jd[0] = tree.item(sel[0], "tags")[0]
                resume_btn.config(state="normal")
                delete_btn.config(state="normal")
            else:
                selected_jd[0] = None
                resume_btn.config(state="disabled")
                delete_btn.config(state="disabled")

        def do_resume():
            jd = selected_jd[0]
            if jd:
                for p in partials:
                    if p[0] == jd:
                        self._restore_partial_job(*p)
                        break
            dlg.destroy()

        def do_delete():
            jd = selected_jd[0]
            if jd and messagebox.askyesno("Delete", "Delete this partial job? Chunks are gone forever.", parent=dlg):
                shutil.rmtree(jd, ignore_errors=True)
                # Remove from tree
                for item in tree.selection():
                    tree.delete(item)
                # If none left, close dialog
                if not tree.get_children():
                    dlg.destroy()

        resume_btn = ttk.Button(btn_frame, text="Resume Selected", command=do_resume, state="disabled")
        resume_btn.pack(side="left", padx=(0, 5))
        delete_btn = ttk.Button(btn_frame, text="Delete Selected", command=do_delete, state="disabled")
        delete_btn.pack(side="left", padx=(0, 5))
        ttk.Button(btn_frame, text="Cancel", command=dlg.destroy).pack(side="left")

        tree.bind("<<TreeviewSelect>>", lambda e: on_select())
        # Double-click to resume
        tree.bind("<Double-1>", lambda e: do_resume() if selected_jd[0] else None)

    def _populate_voice_combo(self):
        self._voice_map = {}
        if os.path.isdir(self._voice_samples_dir):
            for f in sorted(os.listdir(self._voice_samples_dir)):
                if f.lower().endswith(".wav"):
                    stem = os.path.splitext(f)[0]
                    self._voice_map[stem] = os.path.join(self._voice_samples_dir, f)
        self.voice_combo["values"] = list(self._voice_map.keys())
        current = self.voice_combo_var.get()
        if current not in self._voice_map:
            self.voice_combo_var.set("")

    def _on_voice_selected(self, event=None):
        stem = self.voice_combo_var.get()
        path = self._voice_map.get(stem, "")
        if path:
            self.ref_path.set(path)
            self.ref_label.config(text=os.path.basename(path), foreground="black")
            self.play_ref_btn.config(state="normal")

    @staticmethod
    def _get_hf_token():
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            return token
        cached = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.isfile(cached):
            try:
                t = open(cached).read().strip()
                if t:
                    return t
            except Exception:
                pass
        return None

    def _snapshot_settings(self) -> dict:
        """Return dict of current UI settings for batch queue snapshot."""
        return Settings(
            temperature=self.temp.get(),
            min_p=self.min_p.get(),
            top_p=self.top_p.get(),
            top_k=int(self.top_k.get()),
            repetition_penalty=self.rep_pen.get(),
            norm_loudness=self.norm.get(),
            split_clauses=self.split_clauses.get(),
            chunk_size=self.chunk_size.get(),
            seed=self.seed.get(),
            ref_path=self.ref_path.get() or None,
            cpu_threads=self.cpu_threads.get(),
            lead_silence=self.lead_silence.get(),
            trail_silence=self.trail_silence.get(),
            auto_mp3=self.auto_mp3.get(),
            reload_every=self.reload_every.get(),
            prune_every=self.prune_every.get(),
            crossfade_ms=self.crossfade_ms.get(),
        ).to_dict()

    def _apply_settings_to_ui(self, settings):
        """Reflect a batch item's saved settings in the Advanced Options panel.

        ref_path is intentionally not applied; voice selection is separate.
        """
        s = Settings.from_dict(settings)
        self.temp.set(s.temperature)
        self.min_p.set(s.min_p)
        self.top_p.set(s.top_p)
        self.top_k.set(s.top_k)
        self.rep_pen.set(s.repetition_penalty)
        self.norm.set(s.norm_loudness)
        self.split_clauses.set(s.split_clauses)
        self.chunk_size.set(s.chunk_size)
        self.seed.set(s.seed)
        self.cpu_threads.set(s.cpu_threads)
        self.lead_silence.set(s.lead_silence)
        self.trail_silence.set(s.trail_silence)
        self.auto_mp3.set(s.auto_mp3)
        self.reload_every.set(s.reload_every)
        self.prune_every.set(s.prune_every)
        self.crossfade_ms.set(s.crossfade_ms)

    def _open_pron_dict(self):
        """Open the pronunciation substitution dictionary editor."""
        pron_dict = load_pron_dict()  # {word: {"replacement": str, "enabled": bool}}

        dlg = tk.Toplevel(self.root)
        dlg.title("Pronunciation Dictionary")
        dlg.geometry("640x440")
        dlg.transient(self.root)
        dlg.grab_set()

        ttk.Label(dlg,
                  text="Words replaced before TTS generation. Uncheck 'Enabled' to suspend without deleting.",
                  foreground="gray").pack(anchor="w", padx=10, pady=(8, 2))

        # Treeview — three columns: word / replacement / enabled
        tree_frame = ttk.Frame(dlg)
        tree_frame.pack(fill="both", expand=True, padx=10, pady=(0, 4))
        cols = ("word", "replacement", "enabled")
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=10)
        tree.heading("word", text="Original word / phrase")
        tree.heading("replacement", text="Phonetic replacement")
        tree.heading("enabled", text="On")
        tree.column("word", width=180)
        tree.column("replacement", width=310)
        tree.column("enabled", width=36, anchor="center", stretch=False)
        tree.tag_configure("disabled", foreground="gray")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        tree.pack(side="left", fill="both", expand=True)

        def _entry_enabled(entry):
            return entry.get("enabled", True) if isinstance(entry, dict) else True

        def _entry_repl(entry):
            return entry["replacement"] if isinstance(entry, dict) else entry

        def refresh_tree():
            sel_word = None
            sel = tree.selection()
            if sel:
                sel_word = tree.item(sel[0], "values")[0]
            tree.delete(*tree.get_children())
            reselect = None
            for word, entry in sorted(pron_dict.items(), key=lambda x: x[0].lower()):
                enabled = _entry_enabled(entry)
                iid = tree.insert("", "end",
                                  values=(word, _entry_repl(entry), "✓" if enabled else "✗"),
                                  tags=() if enabled else ("disabled",))
                if word == sel_word:
                    reselect = iid
            if reselect:
                tree.selection_set(reselect)
                tree.see(reselect)

        refresh_tree()

        # Edit form
        ef = ttk.LabelFrame(dlg, text="Add / Edit Entry", padding=5)
        ef.pack(fill="x", padx=10, pady=4)
        ttk.Label(ef, text="Original:").grid(row=0, column=0, sticky="w")
        word_var = tk.StringVar()
        word_entry = ttk.Entry(ef, textvariable=word_var, width=18)
        word_entry.grid(row=0, column=1, padx=(4, 12), sticky="ew")
        ttk.Label(ef, text="Replacement:").grid(row=0, column=2, sticky="w")
        repl_var = tk.StringVar()
        repl_entry = ttk.Entry(ef, textvariable=repl_var, width=24)
        repl_entry.grid(row=0, column=3, padx=(4, 12), sticky="ew")
        enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ef, text="Enabled", variable=enabled_var).grid(row=0, column=4, padx=(0, 4))
        ef.columnconfigure(1, weight=1)
        ef.columnconfigure(3, weight=2)

        def on_select(event=None):
            sel = tree.selection()
            if sel:
                vals = tree.item(sel[0], "values")
                word_var.set(vals[0])
                repl_var.set(vals[1])
                enabled_var.set(vals[2] == "✓")

        tree.bind("<<TreeviewSelect>>", on_select)

        def add_or_update():
            word = word_var.get().strip()
            repl = repl_var.get().strip()
            if not word or not repl:
                messagebox.showwarning("Missing fields", "Both fields are required.", parent=dlg)
                return
            pron_dict[word] = {"replacement": repl, "enabled": enabled_var.get()}
            save_pron_dict(pron_dict)
            refresh_tree()
            word_var.set("")
            repl_var.set("")
            enabled_var.set(True)
            word_entry.focus_set()

        def toggle_selected():
            """Flip enabled/disabled on selected rows without touching word or replacement."""
            sel = tree.selection()
            if not sel:
                return
            for item in sel:
                word = tree.item(item, "values")[0]
                if word in pron_dict:
                    entry = pron_dict[word]
                    cur = _entry_enabled(entry)
                    repl = _entry_repl(entry)
                    pron_dict[word] = {"replacement": repl, "enabled": not cur}
            save_pron_dict(pron_dict)
            refresh_tree()

        def remove_selected():
            sel = tree.selection()
            if not sel:
                return
            for item in sel:
                word = tree.item(item, "values")[0]
                pron_dict.pop(word, None)
            save_pron_dict(pron_dict)
            refresh_tree()
            word_var.set("")
            repl_var.set("")

        bf = ttk.Frame(dlg)
        bf.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(bf, text="Add / Update", command=add_or_update).pack(side="left", padx=(0, 4))
        ttk.Button(bf, text="Toggle Enable", command=toggle_selected).pack(side="left", padx=(0, 4))
        ttk.Button(bf, text="Remove Selected", command=remove_selected).pack(side="left", padx=(0, 4))
        ttk.Button(bf, text="Close", command=dlg.destroy).pack(side="right")

        word_entry.focus_set()

    def _open_settings(self):
        try:
            live = _load_config().get("instance_count", 1)
        except Exception:
            live = 1
        if live > 1:
            messagebox.showinfo(
                "Settings locked",
                f"{live} instances are running.\n"
                "Close all other instances before changing Settings.",
                parent=self.root,
            )
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("Settings")
        dlg.geometry("620x180")
        dlg.resizable(True, False)
        dlg.transient(self.root)
        dlg.grab_set()

        fields = [
            ("Voice samples dir:", "_voice_samples_dir"),
            ("Jobs (processing) dir:", "_jobs_dir"),
            ("Output dir:", "_output_dir"),
        ]
        vars_ = {}
        for row_i, (label, attr) in enumerate(fields):
            ttk.Label(dlg, text=label, anchor="e", width=22).grid(
                row=row_i, column=0, padx=(10, 5), pady=6, sticky="e")
            var = tk.StringVar(value=getattr(self, attr))
            vars_[attr] = var
            ttk.Entry(dlg, textvariable=var, width=48).grid(
                row=row_i, column=1, padx=(0, 5), sticky="ew")
            ttk.Button(
                dlg, text="Browse…",
                command=lambda v=var: v.set(
                    filedialog.askdirectory(initialdir=v.get() or os.path.expanduser("~")) or v.get()
                ),
            ).grid(row=row_i, column=2, padx=(0, 10))

        dlg.columnconfigure(1, weight=1)

        def save():
            for attr, var in vars_.items():
                val = var.get().strip()
                if not val:
                    messagebox.showwarning("Settings", f"{attr} cannot be empty.", parent=dlg)
                    return
                setattr(self, attr, os.path.expanduser(val))
            _save_config({
                "voice_samples_dir": self._voice_samples_dir,
                "jobs_dir": self._jobs_dir,
                "output_dir": self._output_dir,
            })
            self._populate_voice_combo()
            dlg.destroy()

        btn_row = ttk.Frame(dlg)
        btn_row.grid(row=len(fields), column=0, columnspan=3, pady=10)
        ttk.Button(btn_row, text="Save", command=save).pack(side="left", padx=5)
        ttk.Button(btn_row, text="Cancel", command=dlg.destroy).pack(side="left")

    def _browse_dataset(self):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            messagebox.showerror("Missing dependency", "huggingface_hub is not installed.")
            return
        import csv
        import shutil

        hf_token = self._get_hf_token()

        dlg = tk.Toplevel(self.root)
        dlg.title("Browse VoxCeleb1 Voices")
        dlg.geometry("660x520")
        dlg.transient(self.root)

        if not hf_token:
            warn_frame = tk.Frame(dlg, bg="#fff3cd", pady=4)
            warn_frame.pack(fill="x")
            tk.Label(
                warn_frame,
                text="No HuggingFace token — downloads will be slow (rate-limited).",
                bg="#fff3cd", fg="#856404", font=("", 9),
            ).pack(side="left", padx=8)
            tk.Label(
                warn_frame,
                text="Fix: run  hf auth login  in a terminal",
                bg="#fff3cd", fg="#856404", font=("", 9, "italic"),
            ).pack(side="left")

        search_frame = ttk.Frame(dlg, padding=5)
        search_frame.pack(fill="x")
        ttk.Label(search_frame, text="Search:").pack(side="left", padx=(0, 5))
        search_var = tk.StringVar()
        ttk.Entry(search_frame, textvariable=search_var, width=40).pack(side="left")

        dlg_status = ttk.Label(dlg, text="Loading metadata…", foreground="gray", padding=(5, 2))
        dlg_status.pack(fill="x")

        tree_frame = ttk.Frame(dlg)
        tree_frame.pack(fill="both", expand=True, padx=5, pady=2)
        cols = ("name", "gender", "status")
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="browse")
        tree.heading("name", text="Name")
        tree.heading("gender", text="Gender")
        tree.heading("status", text="Status")
        tree.column("name", width=320)
        tree.column("gender", width=80)
        tree.column("status", width=130)
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        bottom = ttk.Frame(dlg, padding=5)
        bottom.pack(fill="x")
        dl_btn = ttk.Button(bottom, text="Download & Use", state="disabled")
        dl_btn.pack(side="left")
        ttk.Button(bottom, text="Close", command=dlg.destroy).pack(side="right")

        all_rows = []

        def local_status(name):
            fname = _sanitize_name(name) + ".wav"
            return "✓ Local" if os.path.exists(os.path.join(self._voice_samples_dir, fname)) else ""

        def refresh_tree(filter_text=""):
            tree.delete(*tree.get_children())
            q = filter_text.lower().strip()
            for name, gender, sid in all_rows:
                if q and q not in name.lower():
                    continue
                tree.insert("", "end", values=(name, gender, local_status(name)), tags=(sid,))

        def on_select(event=None):
            dl_btn.config(state="normal" if tree.selection() else "disabled")

        def on_search(*args):
            refresh_tree(search_var.get())

        def on_downloaded(name, dest):
            refresh_tree(search_var.get())
            self._populate_voice_combo()
            stem = os.path.splitext(os.path.basename(dest))[0]
            if stem in self._voice_map:
                self.voice_combo_var.set(stem)
                self._on_voice_selected()
            dlg_status.config(text=f"Downloaded: {os.path.basename(dest)}", foreground="green")
            dl_btn.config(state="normal")

        def do_download():
            sel = tree.selection()
            if not sel:
                return
            values = tree.item(sel[0], "values")
            name = values[0]
            sid = tree.item(sel[0], "tags")[0]
            dl_btn.config(state="disabled")
            dlg_status.config(text=f"Downloading {name}…", foreground="black")

            def worker():
                try:
                    cached = hf_hub_download(
                        repo_id="sdialog/voices-voxceleb1",
                        filename=f"audio/{sid}.wav",
                        repo_type="dataset",
                        token=hf_token,
                    )
                    dest_name = _sanitize_name(name) + ".wav"
                    os.makedirs(self._voice_samples_dir, exist_ok=True)
                    dest = os.path.join(self._voice_samples_dir, dest_name)
                    shutil.copy2(cached, dest)
                    dlg.after(0, lambda: on_downloaded(name, dest))
                except Exception as e:
                    dlg.after(0, lambda err=e: dlg_status.config(
                        text=f"Download failed: {err}", foreground="red"))
                    dlg.after(0, lambda: dl_btn.config(state="normal"))

            threading.Thread(target=worker, daemon=True).start()

        def load_metadata():
            try:
                csv_path = hf_hub_download(
                    repo_id="sdialog/voices-voxceleb1",
                    filename="metadata.csv",
                    repo_type="dataset",
                    token=hf_token,
                )
                with open(csv_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames or []
                    name_col = next((c for c in fieldnames if "name" in c.lower()), None) or (fieldnames[0] if fieldnames else "")
                    gender_col = next((c for c in fieldnames if "gender" in c.lower()), None) or ""
                    id_col = next((c for c in fieldnames if c.lower() in ("id", "voxceleb1 id", "voxceleb_id", "speaker_id")), None)
                    if id_col is None:
                        id_col = next((c for c in fieldnames if "id" in c.lower() and c != name_col), None) or ""
                    for row in reader:
                        name = row.get(name_col, "").strip()
                        gender = row.get(gender_col, "").strip() if gender_col else ""
                        sid = row.get(id_col, "").strip() if id_col else ""
                        if name and sid:
                            all_rows.append((name, gender, sid))
                dlg.after(0, lambda: dlg_status.config(
                    text=f"{len(all_rows)} speakers available", foreground="gray"))
                dlg.after(0, lambda: refresh_tree())
            except Exception as e:
                dlg.after(0, lambda err=e: dlg_status.config(
                    text=f"Failed to load metadata: {err}", foreground="red"))

        tree.bind("<<TreeviewSelect>>", on_select)
        search_var.trace_add("write", on_search)
        dl_btn.config(command=do_download)

        threading.Thread(target=load_metadata, daemon=True).start()

    def _on_close(self):
        if self._batch_running or (hasattr(self, '_gen_running') and self._gen_running):
            if not messagebox.askyesno(
                    "Stop and Exit",
                    "A job is in progress.\n\n"
                    "Stop generation and exit?\n"
                    "(Partial progress is saved and can be resumed.)"):
                return
            self.cancel_token.cancel()
        try:
            cfg = _load_config()
            cfg["instance_count"] = max(0, cfg.get("instance_count", 1) - 1)
            _save_config(cfg)
        except Exception:
            pass
        self.root.destroy()

    def _lock_ui(self, batch=False):
        for w in self._lockable:
            w.config(state="disabled")
        if not batch:
            for w in self._source_lockable:
                w.config(state="disabled")
            for w in self._adv_lockable:
                w.config(state="disabled")
            self.voice_combo.config(state="disabled")
        self.play_btn.config(state="disabled")
        self.mp3_btn.config(state="disabled")
        self.batch_process_btn.config(state="disabled")
        self._gen_running = True

    def _unlock_ui(self):
        for w in self._lockable:
            w.config(state="normal")
        for w in self._source_lockable:
            w.config(state="normal")
        for w in self._adv_lockable:
            w.config(state="normal")
        self.voice_combo.config(state="readonly")
        if self.ref_path.get():
            self.play_ref_btn.config(state="normal")
        self.batch_process_btn.config(state="normal")
        self._gen_running = False

    def _insert_tag(self, tag):
        self.text_area.insert("insert", f" {tag} ")

    def _cancel(self):
        self.cancel_token.cancel()
        self.status.config(text="Cancelling…", foreground="orange")
        self._log("Cancel requested — finishing current chunk…", "warn")

    def _toggle_pause(self):
        if self.cancel_token.is_paused():
            self.cancel_token.resume()
            self.pause_btn.config(text="Pause")
            self.batch_pause_btn.config(text="Pause")
            self.status.config(text="Generating…", foreground="black")
            self._log("▶ Resumed", "info")
        else:
            self.cancel_token.pause()
            self.pause_btn.config(text="Resume")
            self.batch_pause_btn.config(text="Resume")
            self.status.config(text="Paused", foreground="orange")
            self._log("⏸ Paused", "warn")

    def _play(self):
        if self._play_proc and self._play_proc.poll() is None:
            self._play_proc.terminate()
            self._play_proc = None
            self.play_btn.config(text="Play Last Output")
            return
        outpath = getattr(self, "_last_output", None)
        if not outpath or not os.path.exists(outpath):
            messagebox.showwarning("No output", "Generate audio first.")
            return
        self._play_proc = subprocess.Popen(["paplay", outpath])
        self.play_btn.config(text="Stop Playback")
        self._poll_playback()

    def _poll_playback(self):
        if self._play_proc and self._play_proc.poll() is None:
            self.root.after(500, self._poll_playback)
        else:
            self._play_proc = None
            self.play_btn.config(text="Play Last Output")

    def _play_ref(self):
        if self._ref_proc and self._ref_proc.poll() is None:
            self._ref_proc.terminate()
            self._ref_proc = None
            self.play_ref_btn.config(text="Play Voice")
            return
        refpath = self.ref_path.get()
        if not refpath or not os.path.exists(refpath):
            messagebox.showwarning("No voice", "Choose a reference voice first.")
            return
        self._ref_proc = subprocess.Popen(["paplay", refpath])
        self.play_ref_btn.config(text="Stop Voice")
        self._poll_ref()

    def _poll_ref(self):
        if self._ref_proc and self._ref_proc.poll() is None:
            self.root.after(500, self._poll_ref)
        else:
            self._ref_proc = None
            self.play_ref_btn.config(text="Play Voice")

    def _convert_to_mp3(self):
        outpath = getattr(self, "_last_output", None)
        if not outpath or not os.path.exists(outpath):
            messagebox.showwarning("No output", "Generate audio first.")
            return
        mp3path = os.path.splitext(outpath)[0] + ".mp3"
        self.status.config(text="Converting to MP3…", foreground="black")
        self.mp3_btn.config(state="disabled")
        self.root.update_idletasks()
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", outpath, "-b:a", "192k", mp3path],
                check=True, capture_output=True
            )
            self.status.config(text=f"MP3 saved: {mp3path}", foreground="green")
            self._log(f"MP3: {mp3path}", "success")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Conversion failed", e.stderr.decode().strip()[-500:])
            self.status.config(text="MP3 conversion failed.", foreground="red")
            self._log(f"MP3 conversion failed: {e.stderr.decode().strip()[-200:]}", "error")
        finally:
            self.mp3_btn.config(state="normal")

    def _auto_convert(self, wav_path):
        """Silently convert a WAV to MP3 (used by auto-convert checkbox)."""
        if not wav_path or not os.path.exists(wav_path):
            return
        mp3path = os.path.splitext(wav_path)[0] + ".mp3"
        if os.path.exists(mp3path):
            return  # don't overwrite
        try:
            self.status.config(text="Auto-converting to MP3…", foreground="black")
            self.root.update_idletasks()
            subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path, "-b:a", "192k", mp3path],
                check=True, capture_output=True
            )
            self._log(f"Auto MP3: {mp3path}", "success")
        except subprocess.CalledProcessError as e:
            self._log(f"Auto MP3 failed: {e.stderr.decode().strip()[-200:]}", "error")

    # ── state / resume support ──────────────────────────────────────

    def _text_hash(self, text):
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def _job_dir(self, text):
        return os.path.join(self._jobs_dir, self._text_hash(text))

    def _state_path(self, text):
        return os.path.join(self._job_dir(text), "state.json")

    def _find_resumable_job(self, text, _hint_hash=None):
        """Return (job_dir, state) for an incomplete job, or (None, None).

        _hint_hash: directory basename recorded at generation time — tried first
        so the lookup survives pronunciation-dict or source-file drift.
        """
        def _check_dir(jd):
            sp = os.path.join(jd, "state.json")
            if not os.path.exists(sp):
                self._log(f"  [resume] {jd}: no state.json", "info")
                return None
            try:
                with open(sp) as f:
                    state = json.load(f)
                completed = set(state.get("completed", []))
                total = state.get("total_chunks", 0)
                self._log(
                    f"  [resume] {os.path.basename(jd)}: "
                    f"completed={len(completed)}/{total}",
                    "info")
                # Include jobs where all chunks are done but concat/archive
                # didn't finish (completed == total); <= catches that case.
                if total > 0 and len(completed) <= total:
                    return state
            except Exception as _e:
                self._log(f"  [resume] {jd}: read error {_e}", "warn")
            return None

        # 1. Hint hash recorded at generation time (most reliable)
        if _hint_hash:
            jd = os.path.join(self._jobs_dir, _hint_hash)
            self._log(f"  [resume] hint-hash lookup: {jd}", "info")
            state = _check_dir(jd)
            if state is not None:
                return jd, state

        # 2. Hash of the text as provided (pronunciation already applied by caller)
        jd = self._job_dir(text)
        self._log(f"  [resume] text-hash lookup: {os.path.basename(jd)}", "info")
        state = _check_dir(jd)
        if state is not None:
            return jd, state

        return None, None

    def _reconcile_job_state(self, job_dir, state):
        """Compare state.json completed set against chunk WAVs actually on disk.

        Returns (updated_state, disk_count, json_count, was_changed).
        If counts differ, state.json is rewritten to match disk before returning.
        """
        disk_indices = set()
        try:
            for fname in os.listdir(job_dir):
                m = re.match(r'chunk_(\d+)\.wav$', fname)
                if m:
                    disk_indices.add(int(m.group(1)))
        except OSError:
            return state, 0, len(state.get("completed", [])), False

        json_indices = set(state.get("completed", []))
        disk_count = len(disk_indices)
        json_count = len(json_indices)

        if disk_indices == json_indices:
            return state, disk_count, json_count, False

        # Reconcile: disk is truth
        updated = dict(state)
        updated["completed"] = sorted(disk_indices)
        try:
            sp = os.path.join(job_dir, "state.json")
            tmp = sp + ".tmp"
            with open(tmp, "w") as f:
                json.dump(updated, f)
            os.replace(tmp, sp)
        except Exception:
            pass
        return updated, disk_count, json_count, True

    def _save_state(self, text, chunks, completed, settings, output_path, job_dir=None):
        jd = job_dir or self._job_dir(text)
        os.makedirs(jd, exist_ok=True)
        chunks_path = os.path.join(jd, "chunks.json")
        if not os.path.exists(chunks_path):
            tmp = chunks_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"chunks": chunks}, f)
            os.replace(tmp, chunks_path)
        state = {
            "text_hash": self._text_hash(text),
            "total_chunks": len(chunks),
            "completed": sorted(completed),
            "settings": settings,
            "output_path": output_path,
            "created": datetime.now().isoformat(),
        }
        tmp = os.path.join(jd, "state.json.tmp")
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, os.path.join(jd, "state.json"))

    def _start_resume(self):
        self._start_generate(resume=True)

    # ── generation (runs in background thread) ─────────────────────

    def _start_generate(self, resume=False):
        text = self.text_area.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("Empty", "No text to synthesize.")
            return

        job_dir, state = self._find_resumable_job(text)
        if job_dir is None and resume and getattr(self, "_startup_job", None):
            job_dir, state = self._startup_job
            self._startup_job = None
        if job_dir and not resume:
            done = len(state.get("completed", []))
            total = state.get("total_chunks", 0)
            answer = messagebox.askyesno(
                "Resume?",
                f"Found a partial job: {done}/{total} chunks done.\n\n"
                f"Click Yes to resume, No to start fresh."
            )
            if answer:
                resume = True
            else:
                shutil.rmtree(job_dir, ignore_errors=True)
                job_dir, state = None, None

        if resume and state and state.get("settings"):
            # Preserve current silence settings — they're post-processing applied at
            # concatenation time, not chunk-generation params, so the user's current
            # UI values should always take precedence over the saved state.
            cur_lead  = self.lead_silence.get()
            cur_trail = self.trail_silence.get()
            self._apply_settings_to_ui(state["settings"])
            self.lead_silence.set(cur_lead)
            self.trail_silence.set(cur_trail)

        # Snapshot on the main thread so committed spinbox values are captured.
        settings_snap = self._snapshot_settings()

        self._lock_ui()
        self.gen_btn.config(state="disabled")
        self.resume_btn.config(state="disabled")
        self.pause_btn.config(state="normal")
        self.cancel_btn.config(state="normal")
        self.play_btn.config(state="disabled")
        self.mp3_btn.config(state="disabled")
        self.progress.config(value=0)
        self.status.config(text="Resuming…" if resume else "Generating…", foreground="black")

        self.cancel_token.reset()
        self._gen_running = True
        t = threading.Thread(target=self._generate_thread,
                             args=(text, resume, job_dir, state, settings_snap), daemon=True)
        t.start()

    def _generate_thread(self, text, resume, job_dir, state, settings_snap):
        try:
            output_path = self._do_generate(
                text=text,
                settings=settings_snap,
                ref_path=self.ref_path.get() or None,
                source_path=self._source_path,
                resume=resume,
                job_dir=job_dir,
                state=state,
            )
            if output_path:
                self.root.after(0, lambda: self._on_done(output_path))
            else:
                self.root.after(0, lambda: self._on_cancel())
        except Exception:
            self.root.after(0, lambda: self._on_error(str(e)))

    def _is_chunk_garbled(self, wav_path: str, text: str):
        """Return a reason string if the chunk is garbled, or None if it sounds clean.

        Four checks in order:
        1. Duration heuristic — catches cutoffs and severe repetition loops.
        2. Word overlap — what fraction of expected words appear in transcription.
           Catches word substitutions, mid-chunk phoneme glitches, and wrong words
           that the old word-count check missed.
        3. Word-count fallback — catches large-scale dropped speech (< 50% words).
        4. Tail check — catches garbage appended to an otherwise clean chunk.
        """
        import re as _re

        try:
            info = torchaudio.info(wav_path)
            duration = info.num_frames / info.sample_rate
            n_chars = len(text.strip())
            if n_chars < 10:
                return None
            if duration < (n_chars / 25.0):
                return "too short"
            if duration > (n_chars / 3.5):
                return "too long"
        except Exception:
            return None

        n_words = len(text.split())
        if n_words < 5:
            return None

        # Build expected word set once (used by checks 2 and 4).
        # Exclude hyphenated words (pronunciation substitutions like "ay-bish",
        # "Ah-mun", "La-moan-ee") — Whisper transcribes the audio sound, not the
        # synthetic spelling, so they never appear in the transcript and would
        # artificially deflate the overlap score for substitution-dense chunks.
        _tokenize = lambda w: _re.sub(r'[^a-z0-9]', '', w.lower())
        expected_set = {_tokenize(w) for w in text.split() if len(w) >= 3 and '-' not in w}

        try:
            first_load = (_whisper_model is None and not _whisper_unavailable)
            if first_load:
                self._log("  Loading faster-whisper tiny.en for garble detection…", "info")
            wm = _load_whisper()
            if wm is None:
                return None

            segments, _ = wm.transcribe(
                wav_path, language="en", beam_size=1, word_timestamps=True
            )
            segments = list(segments)
            if not segments:
                return "STT: no transcription"

            transcribed_text = ' '.join(s.text for s in segments)

            # Check 2: word overlap — catches wrong/substituted words
            # A clean chunk should have most of its expected words recognized.
            # Threshold of 45% is lenient enough for whisper errors on tiny.en
            # but low enough to flag phonetic garbage and word substitutions.
            transcribed_set = {_tokenize(w) for w in transcribed_text.split() if len(w) >= 3}
            if expected_set:
                overlap = len(transcribed_set & expected_set) / len(expected_set)
                if overlap < 0.45:
                    return f"STT: {overlap:.0%} word overlap ({len(transcribed_set & expected_set)}/{len(expected_set)} matched)"

            # Check 3: word-count ratio — fallback for massive dropouts
            spoken_words = sum(len(s.text.split()) for s in segments)
            if spoken_words / n_words < 0.5:
                return f"STT: {spoken_words}/{n_words} words ({spoken_words/n_words:.0%})"

            all_words = [w for s in segments for w in (s.words or [])]

            # Check 4: unrecognized tail — whisper stopped before audio ended.
            # The model sometimes appends a burst of noise/phonemes that whisper
            # can't parse at all, so it simply stops transcribing early.
            # Signal: last transcribed word ends > 2.5 s before audio ends.
            if all_words:
                last_word_end = all_words[-1].end
                unrecognized_tail = duration - last_word_end
                if unrecognized_tail > 2.5:
                    return (f"STT: {unrecognized_tail:.1f}s unrecognized tail"
                            f" (whisper stopped at {last_word_end:.1f}s/{duration:.1f}s)")

            # Check 5: garbled words whisper DID transcribe at the tail.
            # The model sometimes appends wrong but speech-like words at the end.
            # Signal: last 6 transcribed words mostly not in expected-text word set.
            if len(all_words) >= 6:
                end_words = all_words[-6:]
                end_tokens = [
                    _tokenize(w.word)
                    for w in end_words
                    if len(w.word.strip()) >= 3
                ]
                if len(end_tokens) >= 2:
                    match_frac = sum(1 for t in end_tokens if t in expected_set) / len(end_tokens)
                    if match_frac < 0.35:
                        return f"STT: tail garbled ({match_frac:.0%} end words recognized)"

            # Check 6: word stutter/repetition — consecutive identical tokens in
            # transcription that are NOT expected to repeat in the source text.
            # Catches the model stuttering mid-generation ("the the most unlikely…").
            word_tokens_all = [_tokenize(w.word) for w in all_words]
            # Build set of words that legitimately repeat consecutively in the source
            exp_tokens_seq = [_tokenize(w) for w in text.split()]
            expected_consec = {
                exp_tokens_seq[i] for i in range(1, len(exp_tokens_seq))
                if exp_tokens_seq[i] and exp_tokens_seq[i] == exp_tokens_seq[i - 1]
            }
            stutter_pairs = [
                word_tokens_all[i] for i in range(1, len(word_tokens_all))
                if (word_tokens_all[i]
                    and word_tokens_all[i] == word_tokens_all[i - 1]
                    and word_tokens_all[i] not in expected_consec
                    and not word_tokens_all[i].isdigit())
            ]
            if stutter_pairs:
                sample = stutter_pairs[0]
                extra = f", {len(stutter_pairs)} pairs" if len(stutter_pairs) > 1 else ""
                return f"STT: word stutter ('{sample}' repeated{extra})"

            # Check 7: final content word absent — the model stopped speaking before
            # finishing the chunk, or substituted garbage at the very end.
            # Only checks unhyphenated words (≥ 5 chars) to avoid pronunciation subs.
            last_content_raw = next(
                (w for w in reversed(text.split())
                 if '-' not in w and re.match(r'[a-zA-Z]{5}', w)),
                None
            )
            if last_content_raw:
                last_content = _tokenize(last_content_raw)
                # Skip check for words Whisper likely can't spell: proper nouns, place
                # names, archaic terms (tekoa, bashan, viols, etc.) identified by a
                # vowel ratio below 20% — standard English averages ~38%.
                vowel_count = sum(1 for c in last_content if c in 'aeiou')
                _whisper_knows = len(last_content) <= 3 or (vowel_count / len(last_content)) >= 0.2
                if _whisper_knows:
                    transcribed_tokens = {_tokenize(w) for w in transcribed_text.split()}
                    if last_content not in transcribed_tokens:
                        return f"STT: final content word '{last_content}' not spoken"

            # Check 8: phrase-level repetition loop — same 5-gram appears twice in
            # the transcription but not in the source text.  Catches the model looping
            # back and repeating a clause at the end of a chunk.
            _N = 5
            if len(word_tokens_all) >= _N * 2:
                exp_toks = [_tokenize(w) for w in text.split() if _tokenize(w)]
                exp_ngrams = {
                    tuple(exp_toks[i:i + _N])
                    for i in range(len(exp_toks) - _N + 1)
                }
                seen_ngrams: set = set()
                for i in range(len(word_tokens_all) - _N + 1):
                    gram = tuple(t for t in word_tokens_all[i:i + _N] if t)
                    if len(gram) < _N:
                        continue
                    if gram in seen_ngrams and gram not in exp_ngrams:
                        preview = ' '.join(gram[:4]) + '…'
                        return f"STT: phrase repetition loop ('{preview}')"
                    seen_ngrams.add(gram)

            # Check 9: anaphoric over-repetition — the TTS loops on a phrase that
            # already repeats 2+ times in the source (Holland-style anaphora like
            # "You will be called to…" ×3).  Check 8 misses these because the
            # phrase IS in exp_ngrams; this check catches one or more extra copies.
            if len(word_tokens_all) >= _N * 2:
                from collections import Counter as _Counter
                exp_ngram_counts = _Counter(
                    tuple(exp_toks[i:i + _N])
                    for i in range(len(exp_toks) - _N + 1)
                )
                trx_ngram_counts = _Counter(
                    tuple(t for t in word_tokens_all[i:i + _N] if t)
                    for i in range(len(word_tokens_all) - _N + 1)
                )
                for gram, trx_count in trx_ngram_counts.items():
                    src_count = exp_ngram_counts.get(gram, 0)
                    if src_count >= 2 and trx_count > src_count:
                        preview = ' '.join(gram[:4]) + '…'
                        return (f"STT: anaphoric loop ('{preview}' "
                                f"{trx_count}x spoken, {src_count}x in source)")

            # Check 10: single-source phrase spoken twice — catches the case
            # where a 5+ token phrase appears exactly ONCE in the source but
            # TWO OR MORE times in the transcription.  Check 8 misses this
            # because the phrase IS in exp_ngrams; Check 9 misses it because
            # src_count is 1 (not ≥2).  This is the classic "the road went
            # through the dust, as all roads went through the dust, as all
            # roads went through the dust" loop.
            _N6 = 6  # slightly longer than Check 8's N to avoid false-positives
                     # on legitimately similar clauses
            if len(word_tokens_all) >= _N6 * 2 and len(exp_toks) >= _N6:
                from collections import Counter as _Counter6
                exp_ngram_counts6 = _Counter6(
                    tuple(exp_toks[i:i + _N6])
                    for i in range(len(exp_toks) - _N6 + 1)
                )
                trx_ngram_counts6 = _Counter6(
                    tuple(t for t in word_tokens_all[i:i + _N6] if t)
                    for i in range(len(word_tokens_all) - _N6 + 1)
                )
                for gram, trx_count in trx_ngram_counts6.items():
                    if len(gram) < _N6:
                        continue
                    src_count = exp_ngram_counts6.get(gram, 0)
                    if src_count == 1 and trx_count >= 2:
                        preview = ' '.join(gram[:5]) + '…'
                        return (f"STT: phrase spoken twice ('{preview}' "
                                f"{trx_count}x spoken, 1x in source)")

        except Exception:
            pass
        return None

    def _mem_rss_mb(self):
        """Return (process_rss_mb, system_free_mb) via psutil, or (0, 0) if unavailable."""
        try:
            import psutil
            return (psutil.Process().memory_info().rss / (1024 * 1024),
                    psutil.virtual_memory().available / (1024 * 1024))
        except Exception:
            return 0.0, 0.0

    def _update_mem_bar(self):
        """Refresh the toolbar RAM progress bar with current system memory usage."""
        try:
            import psutil
            vm = psutil.virtual_memory()
            used_gb = vm.used / (1024 ** 3)
            total_gb = vm.total / (1024 ** 3)
            self._mem_bar["value"] = vm.percent
            self._mem_bar_label.configure(text=f"{used_gb:.1f}/{total_gb:.0f} GB")
        except Exception:
            pass
        self.root.after(3000, self._update_mem_bar)

    def _manual_prune(self):
        """Trigger GC + malloc_trim and log the RSS before/after."""
        before_mb, _ = self._mem_rss_mb()
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        after_mb, free_mb = self._mem_rss_mb()
        freed = before_mb - after_mb
        self._log(
            f"Prune: {before_mb/1024:.2f} GB → {after_mb/1024:.2f} GB RSS"
            f" (freed {max(0.0, freed):.0f} MB, {free_mb/1024:.1f} GB system free)",
            "success" if freed > 50 else "info")

    def _do_generate(self, text, settings, ref_path=None, source_path=None,
                     resume=False, job_dir=None, state=None, batch_item=None):
        """Core generation pipeline. Can be called for single-file or batch items.

        Args:
            text: Full text to synthesize
            settings: Dict of generation parameters
            ref_path: Path to reference voice .wav (or None)
            source_path: Original source file path (for output naming)
            resume: Whether to resume an existing partial job
            job_dir: Pre-computed job directory (for resume, avoids hash mismatch)
            state: Pre-loaded state dict (for resume)
            batch_item: Batch queue item dict (for status updates), or None
        """
        # Lazy load model
        if self.model is None:
            self.root.after(0, lambda: self.status.config(text="Loading model…", foreground="black"))
            t0 = time.time()
            _rss_before = self._mem_rss_mb()[0]
            self.model = ChatterboxTurboTTS.from_pretrained(DEVICE)
            _rss_after, _sys_free = self._mem_rss_mb()
            self._model_mem_mb = max(0.0, _rss_after - _rss_before)
            self._log(
                f"Model loaded in {time.time()-t0:.1f}s"
                f" (+{self._model_mem_mb/1024:.1f} GB RSS, {_sys_free/1024:.1f} GB free)",
                "success")

        n_threads = str(settings.get("cpu_threads", 4))
        th.set_num_threads(int(n_threads))
        os.environ["OMP_NUM_THREADS"] = n_threads
        os.environ["MKL_NUM_THREADS"] = n_threads

        seed_val = settings.get("seed", 0)
        if seed_val != 0:
            th.manual_seed(seed_val)

        # Apply pronunciation substitutions only on fresh (non-resume) generations.
        # On resume the text is reconstructed from the saved chunks, so pronunciation
        # was already applied when the job was first created.
        if not resume:
            pron_dict = load_pron_dict()
            if pron_dict:
                text = apply_pronunciation(text, pron_dict)

        # Job directory — must be resolved before loading chunks so the path is
        # available throughout the rest of this function.
        jd = job_dir if (resume and job_dir) else self._job_dir(text)
        os.makedirs(jd, exist_ok=True)
        # Record the exact job hash in the batch item so resume can find it
        # reliably even if the pronunciation dict or source file changes later.
        if batch_item is not None:
            batch_item["_job_hash"] = os.path.basename(jd)

        # Chunks
        if resume and state:
            # Prefer the separate chunks.json (current format); fall back to inline
            # chunks in state (old format), then re-split as last resort.
            chunks_file = os.path.join(jd, "chunks.json")
            if os.path.exists(chunks_file):
                with open(chunks_file) as f:
                    chunks = json.load(f)["chunks"]
            elif state.get("chunks"):
                chunks = state["chunks"]
            else:
                chunks = split_text(text, max_chars=settings.get("chunk_size", 200),
                                    split_clauses=settings.get("split_clauses", True))
        else:
            chunks = split_text(text, max_chars=settings.get("chunk_size", 200),
                                split_clauses=settings.get("split_clauses", True))
        total = len(chunks)

        self._log(f"Text split into {total} chunks (max {settings.get('chunk_size', 200)} chars each)", "info")
        if batch_item is not None:
            if batch_item.get("chunks_total", 0) == 0:
                batch_item["chunks_total"] = total
            self.root.after(0, self._refresh_batch_tree)

        # Output path
        if state and state.get("output_path"):
            output_path = state["output_path"]
        else:
            outdir = self._output_dir
            os.makedirs(outdir, exist_ok=True)
            if source_path:
                stem = os.path.splitext(os.path.basename(source_path))[0]
                output_path = os.path.join(outdir, f"{stem}.wav")
            elif batch_item and batch_item.get("source_path"):
                stem = os.path.splitext(os.path.basename(batch_item["source_path"]))[0]
                output_path = os.path.join(outdir, f"{stem}.wav")
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(outdir, f"tts_{ts}.wav")

        if resume and state:
            completed = set(state.get("completed", []))
            # Reconcile against WAV files actually on disk — an OOM kill can leave
            # state.json behind while more chunks were already written to disk.
            for fname in os.listdir(jd):
                m = re.match(r'chunk_(\d+)\.wav', fname)
                if m:
                    completed.add(int(m.group(1)))
            self.root.after(0, lambda: self.out_label.config(
                text=f"Resuming: {len(completed)}/{total} chunks → {output_path}",
                foreground="blue"))
            self._log(f"Resuming: {len(completed)}/{total} chunks already done", "info")
        else:
            completed = set()
            self._save_state(text, chunks, completed, settings, output_path, job_dir=jd)

        chunk_times = []
        t_start = time.time()
        garbled_chunks = []   # chunks that remained garbled after all retries

        # Cache voice conditionals once for all chunks
        if ref_path:
            self.model.prepare_conditionals(ref_path, norm_loudness=settings.get("norm_loudness", True))

        try:
            for i, chunk in enumerate(chunks):
                # Check cancel FIRST, before pause wait
                if self.cancel_token.cancelled:
                    self.root.after(0, lambda: self.out_label.config(
                        text=f"Partial: {len(completed)}/{total} chunks saved in {jd}", foreground="orange"))
                    self._log(f"Cancelled at chunk {i+1}/{total}", "warn")
                    return None

                # Wait if paused
                self.cancel_token.wait_if_paused()

                # Re-check cancel after un-pausing
                if self.cancel_token.cancelled:
                    self.root.after(0, lambda: self.out_label.config(
                        text=f"Partial: {len(completed)}/{total} chunks saved in {jd}", foreground="orange"))
                    self._log(f"Cancelled at chunk {i+1}/{total}", "warn")
                    return None

                if i in completed:
                    continue

                pct = (len(completed) / total) * 100
                desc = f"Chunk {i+1}/{total}"
                self.root.after(0, lambda p=pct, d=desc: self._update_progress(p, d))

                t_chunk = time.time()
                wav = self.model.generate(
                    chunk,
                    audio_prompt_path=None,
                    temperature=settings["temperature"],
                    min_p=settings.get("min_p", 0.0),
                    top_p=settings["top_p"],
                    top_k=settings["top_k"],
                    repetition_penalty=settings["repetition_penalty"],
                    norm_loudness=settings.get("norm_loudness", True),
                )
                elapsed = time.time() - t_chunk
                chunk_times.append(elapsed)

                wav = wav.squeeze(0).cpu()
                fpath = os.path.join(jd, f"chunk_{i:06d}.wav")
                torchaudio.save(fpath, wav.unsqueeze(0), self.model.sr)
                del wav

                # Prune (gc.collect + malloc_trim) — cadence set by the
                # "Prune every N chunks" spinbox in Advanced Options.
                # 1 = after every chunk (original behavior); 0 = disabled.
                # Read live so mid-run changes take effect immediately.
                prune_every = self.prune_every.get()
                if prune_every > 0 and (i + 1) % prune_every == 0:
                    gc.collect()
                    ctypes.CDLL("libc.so.6").malloc_trim(0)

                # Periodically unload whisper to release ctranslate2 memory arena
                global _whisper_model
                if i > 0 and i % 25 == 0 and _whisper_model is not None:
                    del _whisper_model
                    _whisper_model = None
                    gc.collect()
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                    self._log(f"  Whisper reloaded at chunk {i+1} to free ctranslate2 memory", "info")

                # Warn (and prune) when system RAM drops below threshold
                try:
                    import psutil
                    _avail = psutil.virtual_memory().available
                    if _avail < _PRUNE_THRESHOLD_GB * 1024 ** 3:
                        if _whisper_model is not None:
                            del _whisper_model
                            _whisper_model = None
                        gc.collect()
                        ctypes.CDLL("libc.so.6").malloc_trim(0)
                        _proc_mb, _free_mb = self._mem_rss_mb()
                        self._log(
                            f"  ⚠ Low RAM auto-prune: {_free_mb/1024:.1f} GB free"
                            f", {_proc_mb/1024:.1f} GB RSS",
                            "warn")
                except Exception:
                    pass

                garble_reason = self._is_chunk_garbled(fpath, chunk)
                if garble_reason:
                    # Save the initial attempt as the current best before retrying.
                    # Each retry overwrites fpath; we restore best if all retries fail.
                    import shutil as _shutil
                    best_fpath = fpath + ".best"
                    _shutil.copy2(fpath, best_fpath)
                    best_reason = garble_reason

                    for _retry in range(MAX_CHUNK_RETRIES):
                        self._log(
                            f"  ⚠ Chunk {i+1} garbled [{garble_reason}]"
                            f" (retry {_retry+1}/{MAX_CHUNK_RETRIES})…",
                            "warn"
                        )
                        th.manual_seed(int(time.time() * 1e6) % (2 ** 31))
                        retry_wav = self.model.generate(
                            chunk,
                            audio_prompt_path=None,
                            temperature=settings["temperature"],
                            min_p=settings.get("min_p", 0.0),
                            top_p=settings["top_p"],
                            top_k=settings["top_k"],
                            repetition_penalty=settings["repetition_penalty"],
                            norm_loudness=settings.get("norm_loudness", True),
                        )
                        retry_wav = retry_wav.squeeze(0).cpu()
                        torchaudio.save(fpath, retry_wav.unsqueeze(0), self.model.sr)
                        del retry_wav
                        gc.collect()
                        ctypes.CDLL("libc.so.6").malloc_trim(0)
                        garble_reason = self._is_chunk_garbled(fpath, chunk)
                        if not garble_reason:
                            # Clean retry — discard the saved best and keep this one
                            os.remove(best_fpath)
                            break
                        # This retry is also garbled; keep whichever attempt was first
                        # (initial generation is the model's most confident output)
                    else:
                        # All retries garbled — restore the initial attempt as best
                        _shutil.move(best_fpath, fpath)
                        self._log(
                            f"  ⚠ Chunk {i+1} still garbled [{best_reason}]"
                            f" after {MAX_CHUNK_RETRIES} retries — keeping first attempt",
                            "warn"
                        )
                        garbled_chunks.append({
                            "index": i,
                            "filename": f"chunk_{i:06d}.wav",
                            "reason": best_reason,
                            "text": chunk,
                        })
                    # Clean up .best file if somehow still present
                    if os.path.exists(best_fpath):
                        os.remove(best_fpath)

                completed.add(i)
                self._save_state(text, chunks, completed, settings, output_path, job_dir=jd)

                if batch_item is not None:
                    avg_t = statistics.mean(chunk_times) if chunk_times else 0
                    remaining = total - len(completed)
                    batch_item["chunks_done"] = len(completed)
                    batch_item["chunks_total"] = total
                    batch_item["_eta_str"] = (
                        f"~{avg_t * remaining / 60:.0f}m" if avg_t > 0 and remaining > 0
                        else ("done" if remaining == 0 else ""))
                    self.root.after(0, self._refresh_batch_tree)

                # Periodic model reload to defragment memory
                reload_every = self.reload_every.get()  # read live, not from snapshot
                if reload_every > 0 and (i + 1) % reload_every == 0 and i + 1 < total:
                    t_reload = time.time()
                    self._log(f"Reloading model at chunk {i+1} (every {reload_every})...", "info")
                    # Free the old model and whisper FIRST so malloc_trim can return
                    # their pages to the OS before we measure free RAM or load fresh.
                    self.model = None
                    if _whisper_model is not None:
                        del _whisper_model
                        _whisper_model = None
                    for _ in range(3):
                        gc.collect()
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                    time.sleep(0.5)   # let OS reclaim mmap'd pages
                    _proc, free_mb = self._mem_rss_mb()
                    if free_mb < 10 * 1024:   # < 10 GB free after full cleanup
                        self._log(
                            f"  ⚠ Only {free_mb/1024:.1f} GB free after cleanup — "
                            f"model freed but memory critically low; reloading anyway",
                            "warn")
                    self.model = ChatterboxTurboTTS.from_pretrained(DEVICE)
                    th.set_num_threads(int(settings.get("cpu_threads", 4)))
                    if ref_path:
                        self.model.prepare_conditionals(ref_path,
                            norm_loudness=settings.get("norm_loudness", True))
                    proc_mb, free_mb = self._mem_rss_mb()
                    self._log(
                        f"  Model reloaded in {time.time()-t_reload:.0f}s"
                        f" | {proc_mb/1024:.1f} GB RSS, {free_mb/1024:.1f} GB free",
                        "success"
                    )

                # Log every 10th chunk or first/last
                if i % 10 == 0 or i == 0 or i == total - 1:
                    avg_t = statistics.mean(chunk_times) if chunk_times else 0
                    eta = avg_t * (total - len(completed))
                    proc_mb, free_mb = self._mem_rss_mb()
                    self._log(
                        f"Chunk {i+1}/{total} ({elapsed:.1f}s, avg {avg_t:.1f}s, ETA {eta/60:.0f}m)"
                        f" | {proc_mb/1024:.1f} GB RSS, {free_mb/1024:.1f} GB free",
                        "chunk")

            if self.cancel_token.cancelled:
                self.root.after(0, lambda: self.out_label.config(
                    text=f"Partial: {len(completed)}/{total} chunks saved in {jd}", foreground="orange"))
                self._log(f"Cancelled after {len(completed)}/{total} chunks", "warn")
                return None

            total_elapsed = time.time() - t_start
            self._log(f"All {total} chunks generated in {total_elapsed/60:.1f}m", "success")
            if batch_item is not None:
                h, m = divmod(int(total_elapsed / 60), 60)
                batch_item["gen_time"] = f"{h}h {m}m" if h else f"{m}m"
            self.root.after(0, lambda: self._update_progress(100, "Concatenating…"))
            self._log("Concatenating chunks…", "info")

            chunk_files = [os.path.join(jd, f"chunk_{i:06d}.wav") for i in range(total)]
            sr = self.model.sr

            crossfade_ms = settings.get("crossfade_ms", 50)
            if crossfade_ms > 0 and total > 1:
                # Load all chunks and crossfade-stitch them.
                wavs = []
                for fpath in chunk_files:
                    w, _ = torchaudio.load(fpath)
                    wavs.append(w)
                final = _crossfade_chunks(wavs, sr, crossfade_ms=crossfade_ms)
                self._log(
                    f"Crossfaded {total} chunks ({crossfade_ms} ms overlap) "
                    f"→ {final.shape[1] / sr:.1f}s", "info")
            elif total == 1:
                final, _ = torchaudio.load(chunk_files[0])
            else:
                between_sil = th.zeros(1, int(sr * 0.3))
                waveforms = []
                for fpath in chunk_files:
                    w, _ = torchaudio.load(fpath)
                    waveforms.append(w)
                    waveforms.append(between_sil)
                final = th.cat(waveforms[:-1], dim=1)

            lead_s = settings.get("lead_silence", 0.0)
            trail_s = settings.get("trail_silence", 0.0)
            if lead_s > 0:
                final = th.cat([th.zeros(1, int(sr * lead_s)), final], dim=1)
            if trail_s > 0:
                final = th.cat([final, th.zeros(1, int(sr * trail_s))], dim=1)
            if lead_s > 0 or trail_s > 0:
                self._log(f"Silence padding: {lead_s}s lead, {trail_s}s trail", "info")

            # Remove DC offset before saving (prevents low-frequency thumps)
            final = _remove_dc_offset(final, sr)
            torchaudio.save(output_path, final, sr)
            self._log(f"Saved: {output_path}", "success")

            self._archive_job(jd, garbled_chunks, output_path)

            return output_path

        except Exception:
            self._log(f"Error during generation — job state preserved in {jd}", "error")
            raise

    # ── Archive & rework ──────────────────────────────────────────

    def _archive_job(self, jd, garbled_chunks, output_path):
        """Move completed job dir to archive/, save log, write rework.json if needed."""
        os.makedirs(self._archive_dir, exist_ok=True)
        dest = os.path.join(self._archive_dir, os.path.basename(jd))
        if os.path.exists(dest):
            dest = dest + "_" + datetime.now().strftime("%Y%m%d%H%M%S")
        shutil.move(jd, dest)

        # Save plain-text log snapshot
        log_path = os.path.join(dest, "generation.log")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.writelines(self.buffer)
        except Exception:
            pass

        if garbled_chunks:
            rework = {
                "output_path": output_path,
                "archived": datetime.now().isoformat(),
                "rework_chunks": garbled_chunks,
            }
            rework_path = os.path.join(dest, "rework.json")
            with open(rework_path, "w", encoding="utf-8") as f:
                json.dump(rework, f, indent=2, ensure_ascii=False)
            self._log(
                f"⚠ {len(garbled_chunks)} chunk(s) need rework → {rework_path}", "warn")

        self._log(f"Job archived → {dest}", "info")
        self.root.after(0, self._refresh_rework_badge)

    def _scan_rework_jobs(self):
        """Return list of (archive_dir, rework_info) for all pending rework jobs."""
        jobs = []
        if not os.path.isdir(self._archive_dir):
            return jobs
        for name in sorted(os.listdir(self._archive_dir)):
            rp = os.path.join(self._archive_dir, name, "rework.json")
            if os.path.exists(rp):
                try:
                    with open(rp, encoding="utf-8") as f:
                        info = json.load(f)
                    if info.get("rework_chunks"):
                        jobs.append((os.path.join(self._archive_dir, name), info))
                except Exception:
                    pass
        return jobs

    def _refresh_rework_badge(self):
        """Update Rework button to show pending count."""
        n = len(self._scan_rework_jobs())
        label = f"Rework ({n})…" if n else "Rework…"
        self.rework_btn.config(text=label)

    def _open_rework_dialog(self):
        """Show a dialog listing archived jobs that have chunks needing rework."""
        jobs = self._scan_rework_jobs()
        dlg = tk.Toplevel(self.root)
        dlg.title("Rework — Garbled Chunks")
        dlg.geometry("780x420")
        dlg.transient(self.root)
        dlg.grab_set()

        ttk.Label(dlg, text="Archived jobs with garbled chunks:", padding=(8, 6)).pack(anchor="w")

        tree_frame = ttk.Frame(dlg)
        tree_frame.pack(fill="both", expand=True, padx=8)
        cols = ("job", "output", "chunks", "archived")
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="browse", height=10)
        tree.heading("job",      text="Job ID")
        tree.heading("output",   text="Output file")
        tree.heading("chunks",   text="#")
        tree.heading("archived", text="Archived")
        tree.column("job",      width=140, stretch=False)
        tree.column("output",   width=320)
        tree.column("chunks",   width=35, stretch=False)
        tree.column("archived", width=160, stretch=False)
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        selected_job = [None]
        sort_state   = {"col": None, "asc": True}  # tracks current sort column/direction

        def _populate_tree():
            """(Re-)fill the tree from the jobs list, preserving selection."""
            prev = selected_job[0]
            tree.delete(*tree.get_children())
            for arc_dir, info in jobs:
                job_id  = os.path.basename(arc_dir)
                outfile = os.path.basename(info.get("output_path", ""))
                n_bad   = len(info.get("rework_chunks", []))
                ts      = info.get("archived", "")[:19].replace("T", " ")
                iid = tree.insert("", "end", values=(job_id, outfile, n_bad, ts),
                                  tags=(arc_dir,))
                if arc_dir == prev:
                    tree.selection_set(iid)
                    tree.see(iid)

        def _sort_by(col):
            """Sort jobs list in-place by col, toggle asc/desc, refresh tree."""
            col_idx = {"output": 1, "archived": 3}[col]
            asc = not sort_state["asc"] if sort_state["col"] == col else True
            sort_state["col"] = col
            sort_state["asc"] = asc
            jobs.sort(key=lambda item: (item[1].get("output_path", "") if col == "output"
                                        else item[1].get("archived", "")).lower(),
                      reverse=not asc)
            arrow = " ▲" if asc else " ▼"
            tree.heading("output",   text="Output file" + (arrow if col == "output"   else ""))
            tree.heading("archived", text="Archived"    + (arrow if col == "archived" else ""))
            _populate_tree()

        tree.heading("output",   command=lambda: _sort_by("output"))
        tree.heading("archived", command=lambda: _sort_by("archived"))

        _populate_tree()

        def on_select(evt):
            sel = tree.selection()
            selected_job[0] = tree.item(sel[0], "tags")[0] if sel else None
            s = "normal" if selected_job[0] else "disabled"
            process_btn.config(state=s)
            detail_btn.config(state=s)
            add_batch_btn.config(state=s)
            remove_btn.config(state=s)

        tree.bind("<<TreeviewSelect>>", on_select)

        # Detail pane — show chunk texts for selected job
        detail_frame = ttk.LabelFrame(dlg, text="Garbled chunks", padding=5)
        detail_frame.pack(fill="x", padx=8, pady=(4, 0))
        detail_text = tk.Text(detail_frame, height=5, wrap="word", state="disabled",
                              font=("Courier", 9))
        detail_text.pack(fill="x")

        def show_detail(evt=None):
            if not selected_job[0]:
                return
            rp = os.path.join(selected_job[0], "rework.json")
            try:
                with open(rp, encoding="utf-8") as f:
                    info = json.load(f)
            except Exception:
                return
            detail_text.configure(state="normal")
            detail_text.delete("1.0", "end")
            for c in info.get("rework_chunks", []):
                detail_text.insert("end",
                    f"chunk_{c['index']:06d}.wav  [{c['reason']}]\n"
                    f"  {c['text'][:120]}{'…' if len(c['text'])>120 else ''}\n\n")
            detail_text.configure(state="disabled")

        tree.bind("<<TreeviewSelect>>", lambda e: (on_select(e), show_detail()))

        btn_row = ttk.Frame(dlg)
        btn_row.pack(fill="x", padx=8, pady=6)

        def do_process():
            arc = selected_job[0]
            if not arc:
                return
            if not messagebox.askyesno(
                    "Reprocess rework",
                    f"Restore job from archive and regenerate "
                    f"{len(jobs[[d for d,_ in jobs].index(arc)][1]['rework_chunks'])} "
                    f"garbled chunk(s)?",
                    parent=dlg):
                return
            dlg.destroy()
            self._process_rework(arc)

        def do_add_to_batch():
            arc = selected_job[0]
            if not arc:
                return
            state_path  = os.path.join(arc, "state.json")
            chunks_path = os.path.join(arc, "chunks.json")
            try:
                with open(state_path, encoding="utf-8") as f:
                    state = json.load(f)
                with open(chunks_path, encoding="utf-8") as f:
                    chunk_data = json.load(f)
            except Exception as exc:
                messagebox.showerror("Error", f"Cannot read job data:\n{exc}", parent=dlg)
                return
            settings    = state.get("settings", {})
            output_path = state.get("output_path", "")
            text        = " ".join(chunk_data.get("chunks", []))
            ref         = settings.get("ref_path") or None
            voice_name  = os.path.splitext(os.path.basename(ref))[0] if ref else "(default)"
            file_name   = os.path.basename(output_path) if output_path else os.path.basename(arc)
            item = BatchItem(
                id=self._batch_next_id,
                source_path=output_path or None,
                file_name=file_name,
                text=text,
                ref_path=ref,
                voice_name=voice_name,
                settings=settings,
            ).to_dict()
            self._batch_next_id += 1
            self._batch_queue.append(item)
            self._reflect_batch_change()
            self._log(f"Added rework job to batch queue: {file_name} ({voice_name})", "info")
            messagebox.showinfo("Added to queue", f"Added to batch queue:\n{file_name}", parent=dlg)

        def do_remove():
            arc = selected_job[0]
            if not arc:
                return
            job_id = os.path.basename(arc)
            if not messagebox.askyesno(
                    "Remove rework entry",
                    f"Remove '{job_id}' from the rework list?\n\n"
                    f"The archive directory is kept; only the rework flag is cleared.",
                    parent=dlg):
                return
            rp = os.path.join(arc, "rework.json")
            try:
                os.remove(rp)
            except Exception as exc:
                messagebox.showerror("Error", f"Could not remove rework.json:\n{exc}", parent=dlg)
                return
            # Remove from in-memory list and refresh
            idx = next((i for i, (d, _) in enumerate(jobs) if d == arc), None)
            if idx is not None:
                jobs.pop(idx)
            selected_job[0] = None
            detail_text.configure(state="normal")
            detail_text.delete("1.0", "end")
            detail_text.configure(state="disabled")
            _populate_tree()
            self._refresh_rework_badge()
            for btn in (process_btn, detail_btn, add_batch_btn, remove_btn):
                btn.config(state="disabled")

        process_btn = ttk.Button(btn_row, text="Process Rework", command=do_process, state="disabled")
        process_btn.pack(side="left", padx=(0, 6))

        detail_btn = ttk.Button(btn_row, text="Show Chunks", command=show_detail, state="disabled")
        detail_btn.pack(side="left", padx=(0, 6))

        add_batch_btn = ttk.Button(btn_row, text="Add to Batch Queue",
                                   command=do_add_to_batch, state="disabled")
        add_batch_btn.pack(side="left", padx=(0, 6))

        remove_btn = ttk.Button(btn_row, text="Remove", command=do_remove, state="disabled")
        remove_btn.pack(side="left", padx=(0, 6))

        ttk.Button(btn_row, text="Close", command=dlg.destroy).pack(side="right")

        if not jobs:
            ttk.Label(dlg, text="No pending rework jobs found.", foreground="gray",
                      padding=(8, 4)).pack()
            for btn in (process_btn, detail_btn, add_batch_btn, remove_btn):
                btn.config(state="disabled")

    def _process_rework(self, arc_dir):
        """Copy archive back to jobs/, remove garbled WAVs, resume generation."""
        rp = os.path.join(arc_dir, "rework.json")
        try:
            with open(rp, encoding="utf-8") as f:
                rework_info = json.load(f)
        except Exception as exc:
            messagebox.showerror("Rework error", f"Cannot read rework.json:\n{exc}")
            return

        job_id   = os.path.basename(arc_dir)
        dest_jd  = os.path.join(self._jobs_dir, job_id)
        bad_idxs = {c["index"] for c in rework_info.get("rework_chunks", [])}

        # Copy archive → jobs dir (archive stays intact until rework completes)
        if os.path.exists(dest_jd):
            shutil.rmtree(dest_jd, ignore_errors=True)
        shutil.copytree(arc_dir, dest_jd)

        # Remove the garbled WAV files so the resume loop regenerates them
        for idx in bad_idxs:
            bad_wav = os.path.join(dest_jd, f"chunk_{idx:06d}.wav")
            if os.path.exists(bad_wav):
                os.remove(bad_wav)

        # Also remove rework.json from the copy (it belongs in the archive)
        for fname in ("rework.json", "generation.log"):
            p = os.path.join(dest_jd, fname)
            if os.path.exists(p):
                os.remove(p)

        # Patch state.json — remove garbled indices from completed set
        state_path = os.path.join(dest_jd, "state.json")
        try:
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
            state["completed"] = [i for i in state.get("completed", [])
                                  if i not in bad_idxs]
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as exc:
            messagebox.showerror("Rework error", f"Cannot patch state.json:\n{exc}")
            shutil.rmtree(dest_jd, ignore_errors=True)
            return

        # Remove the old archive entry — it will be replaced when rework completes
        shutil.rmtree(arc_dir, ignore_errors=True)
        self.root.after(0, self._refresh_rework_badge)

        self._log(f"Rework: restoring {job_id} — {len(bad_idxs)} chunk(s) to regenerate", "info")

        # Load chunks and resume via the normal generate path
        chunks_file = os.path.join(dest_jd, "chunks.json")
        try:
            with open(chunks_file, encoding="utf-8") as f:
                chunks_data = json.load(f)
            text = "\n".join(chunks_data["chunks"])
        except Exception as exc:
            messagebox.showerror("Rework error", f"Cannot read chunks.json:\n{exc}")
            return

        self._startup_job = (dest_jd, state)
        self.text_area.delete("1.0", "end")
        self.text_area.insert("1.0", text)
        self._start_resume()

    # ── UI updates (main thread) ───────────────────────────────────

    def _update_progress(self, pct, desc):
        self.progress.config(value=pct)
        self.status.config(text=desc, foreground="black")

    def _on_done(self, path):
        self._unlock_ui()
        self._last_output = path
        self.progress.config(value=100)
        self.status.config(text="Done!", foreground="green")
        self.out_label.config(text=f"Saved: {path}")
        self.play_btn.config(state="normal")
        self.mp3_btn.config(state="normal")
        self.gen_btn.config(state="normal")
        self.resume_btn.config(state="disabled")
        self.pause_btn.config(state="disabled")
        self.cancel_btn.config(state="disabled")
        if self.auto_mp3.get():
            self._auto_convert(path)

    def _on_cancel(self):
        self._unlock_ui()
        self.progress.config(value=0)
        self.status.config(text="Cancelled", foreground="orange")
        self.gen_btn.config(state="normal")
        self.resume_btn.config(state="normal")
        self.pause_btn.config(state="disabled")
        self.cancel_btn.config(state="disabled")

    def _on_error(self, msg):
        self._unlock_ui()
        self.progress.config(value=0)
        self.status.config(text=f"Error: {msg}", foreground="red")
        self.gen_btn.config(state="normal")
        self.resume_btn.config(state="normal")
        self.pause_btn.config(state="disabled")
        self.cancel_btn.config(state="disabled")
        self._log(f"ERROR: {msg}", "error")
        messagebox.showerror("Generation Error", msg)


# ── entry point ────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-resume", action="store_true",
                        help="Auto-start batch processing if pending items exist")
    args = parser.parse_args()

    # Copy pre-Phase-2 state files from ~/bin/ to $XDG_CONFIG_HOME/tts-books/
    # on first launch. No-op after that.
    from tts_books.paths import migrate_legacy
    for legacy, new in migrate_legacy():
        print(f"Migrated {legacy} -> {new}")

    root = tk.Tk()
    app = TTSBookApp(root, auto_resume=args.auto_resume)
    root.mainloop()


if __name__ == "__main__":
    main()
