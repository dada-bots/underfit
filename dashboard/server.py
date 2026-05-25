#!/usr/bin/env python3
"""Training dashboard server for LoRA finetuning runs."""

import json
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import time
import uuid
from datetime import datetime, timezone
from email.utils import formatdate
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

import numpy as np
import soundfile as sf
import torch
from PIL import Image

BASE_DIR = Path(__file__).parent.parent
DASHBOARD_DIR = Path(__file__).parent

# Writable user state — runs.json/datasets.json, per-run outputs, generated
# audio, checkpoint symlinks, gradio logs. Defaults to <repo>/state/ for a
# clean separation from the code in dashboard/. Override with UNDERFIT_STATE_DIR
# to relocate (e.g. ~/.underfit, persistent volume in Colab).
STATE_DIR = Path(os.environ.get("UNDERFIT_STATE_DIR", BASE_DIR / "state")).expanduser()
STATE_DIR.mkdir(parents=True, exist_ok=True)

# Tracked, ship-with-the-repo paths
MODELS_SHIPPED_DIR = DASHBOARD_DIR / "models"   # per-model {registry.json, training_template.json}
PRE_DIR = BASE_DIR / "dataset_processing"       # autotagger, pre_encode, metadata helpers

# Per-instance runtime paths (under STATE_DIR)
RUNS_DIR = STATE_DIR / "runs"                   # per-run checkpoints, configs, demos — durable, on Drive in Colab
# Seed LoRA uploads land here briefly before _handle_new_finetune copies them
# into the run dir. Temporary staging — no need for Drive (slow writes mean
# the upload spinner just hangs for 5-30 s per file). Override-able via env.
SEED_LORAS_DIR = Path(os.environ.get("UNDERFIT_SEED_LORAS_DIR",
                                      str(STATE_DIR / "seed_loras"))).expanduser()
# Live training logs live on local SSD on Colab (fast reads for the dashboard)
# and get rsync'd to Drive every ~60s by a separate thread in the notebook.
# Outside Colab, defaults to RUNS_DIR so single-machine setups behave as before.
LOGS_DIR = Path(os.environ.get("UNDERFIT_LOGS_DIR", str(RUNS_DIR))).expanduser()
LOGS_DIR.mkdir(parents=True, exist_ok=True)
# The dashboard's own stdout/stderr is teed here at startup so we can serve
# it via /api/server_log. Truncated on every dashboard start.
SERVER_LOG_FILE = LOGS_DIR / "dashboard_server.log"
# Small state files (runs.json, datasets.json, gradio_*.json) live here.
# On Colab → /content/underfit-state (local SSD) for instant dashboard reads.
# Outside Colab → defaults to STATE_DIR so non-Colab setups are unchanged.
STATE_FILES_DIR = Path(os.environ.get("UNDERFIT_STATE_FILES_DIR", str(STATE_DIR))).expanduser()
STATE_FILES_DIR.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path, data):
    """Write JSON via tmp+rename so a concurrent reader (e.g. the Drive sync
    thread in the notebook kernel) never sees a partial file. POSIX guarantees
    os.replace is atomic."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))
AUDIO_DIR = STATE_DIR / "audio"                 # generated demo MP3s + spectrogram JPGs

# Base-model files (SA3 RF + ARC, T5Gemma) — defaults to STATE_DIR/models,
# but UNDERFIT_MODELS_DIR lets you put them somewhere fast on machines where
# STATE_DIR lives on slow storage (e.g. Colab: STATE_DIR on Drive +
# MODELS_DIR on /content/ SSD). Distinct from per-run LoRA *training*
# checkpoints (which live in RUNS_DIR and use the field name "checkpoints_dir").
MODELS_DIR = Path(os.environ.get(
    "UNDERFIT_MODELS_DIR", STATE_DIR / "models"
)).expanduser()
RUNS_FILE = STATE_FILES_DIR / "runs.json"
PORT = int(os.environ.get("UNDERFIT_DASHBOARD_PORT", 8787))
DEMO_STEPS = 50
DEMO_CFG_SCALES = [7]

# Cap CPU thread pools (OMP/MKL/BLAS/NumExpr/Rayon) on each launched gradio
# AND training process. Without caps, every Python process spins up nproc-
# sized pools, so a handful of concurrent processes can blow through
# ulimit -u and cause tokenizer/Rayon thread-spawn failures. Set False to
# disable for both.
GRADIO_THREAD_CAP = True

# MODEL_INFO is populated entirely from dashboard/models/*/registry.json at startup
# (see _load_models_from_json below). To add a new model: drop a JSON file
# in that directory and restart. No Python edits required.
MODEL_INFO: dict = {}

def _get_model_info(name):
    """Return MODEL_INFO[name], or the first registered model when name is
    None / unknown. Returns None only when no models are registered at all —
    callers handle that as a hard error."""
    if name and name in MODEL_INFO:
        return MODEL_INFO[name]
    return next(iter(MODEL_INFO.values()), None)

# ── Model JSON loader (spike) ────────────────────────────────────────────
# Per-model registry files in dashboard/models/<key>/registry.json. Each
# uses the {models_dir} placeholder so the registry stays portable —
# only STATE_DIR has to be set correctly for paths to resolve. Adding a new
# model = mkdir dashboard/models/<new-key>/ + drop in registry.json and
# training_template.json. No Python edits, no JS edits.
#
# Each file also exposes a "ui" block with VRAM, lora aggregate, sequence
# info, and module structure — surfaced via /api/models for the frontend
# to consume in place of its current hardcoded tables.
import json as _json

# Mapping from each model's encoder_id to its canonical key (drives SHARED_ENCODERS).
_ENCODER_GROUPS: dict = {}  # encoder_id -> {"canonical": key, "members": [keys]}

# Frontend-only payload, served by /api/models. One entry per model with the
# "ui" block plus a minimal {label, description, encoder_id, backend, arc_type}.
MODELS_UI_PAYLOAD: dict = {}

def _resolve_paths(obj, substitutions):
    """Recursively substitute placeholders in string values."""
    if isinstance(obj, dict):
        return {k: _resolve_paths(v, substitutions) for k, v in obj.items() if not k.startswith("_comment")}
    if isinstance(obj, list):
        return [_resolve_paths(v, substitutions) for v in obj]
    if isinstance(obj, str):
        for placeholder, real in substitutions.items():
            obj = obj.replace("{" + placeholder + "}", real)
        return obj
    return obj

def _load_models_from_json():
    """Walk dashboard/models/*/registry.json and merge into MODEL_INFO / ENCODING_MODELS / SHARED_ENCODERS."""
    if not MODELS_SHIPPED_DIR.is_dir():
        return
    subs = {"models_dir": str(MODELS_DIR)}
    for registry_path in sorted(MODELS_SHIPPED_DIR.glob("*/registry.json")):
        with open(registry_path) as f:
            raw = _json.load(f)
        m = _resolve_paths(raw, subs)
        key = m["key"]
        if key in MODEL_INFO:
            raise ValueError(f"{registry_path}: key '{key}' already exists in MODEL_INFO")

        # Build MODEL_INFO entry (matches the shape used by hardcoded entries).
        # Template path follows convention: training_template.json next to registry.json.
        paths = m["paths"]
        # Normalize backends: accept either the new 'backends': ['sa3', 'sat_dev']
        # list OR the legacy 'backend': 'sa3' scalar. Both materialize to a list
        # in MODEL_INFO so downstream code only has to handle one shape.
        if "backends" in m:
            backends = list(m["backends"])
        elif "backend" in m:
            backends = [m["backend"]]
        else:
            backends = []
        entry = {
            "backends":            backends,
            "base_config":         paths["base_config"],
            "base_ckpt":           paths["base_ckpt"],
            "template":            str(registry_path.parent / "training_template.json"),
            "clip_duration":       m["training"]["clip_duration"],
            "latent_crop_length":  m["training"]["latent_crop_length"],
            "seconds_total":       m["training"]["seconds_total"],
            "diffusion_objective": m["diffusion_objective"],
        }
        if "svd_bases" in paths:
            entry["svd_bases"] = paths["svd_bases"]
        if "arc" in m:
            entry["arc_type"]   = m["arc"]["type"]
            entry["arc_config"] = paths["arc_config"]
            entry["arc_ckpt"]   = paths["arc_ckpt"]
        MODEL_INFO[key] = entry

        # Per-model symlink dir for tools that prefer a stable local path
        # (e.g. pre_encode.py reads MODELS_DIR/<key>/{config,ckpt}).
        _model_dir = MODELS_DIR / key
        _model_dir.mkdir(parents=True, exist_ok=True)
        _link_specs = [
            ("config",     paths.get("base_config")),
            ("ckpt",       paths.get("base_ckpt")),
            ("arc_config", paths.get("arc_config")),
            ("arc_ckpt",   paths.get("arc_ckpt")),
        ]
        for _sym, _real in _link_specs:
            if not _real:
                continue
            _sympath = _model_dir / _sym
            if _sympath.is_symlink() or _sympath.exists():
                continue
            try:
                _sympath.symlink_to(_real)
            except OSError as _e:
                print(f"[models] couldn't link {_sympath} -> {_real}: {_e}")

        # ENCODING_MODELS description
        if "description" in m:
            ENCODING_MODELS[key] = m["description"]

        # Encoder grouping → SHARED_ENCODERS
        enc_id = m.get("encoder_id")
        canon = m.get("encoder_canonical_model", key)
        if enc_id:
            grp = _ENCODER_GROUPS.setdefault(enc_id, {"canonical": canon, "members": []})
            if key not in grp["members"]:
                grp["members"].append(key)

        # Frontend payload
        ui = m.get("ui", {})
        MODELS_UI_PAYLOAD[key] = {
            "key":                  key,
            "label":                m.get("label", key),
            "description":          m.get("description", ""),
            "backends":             backends,
            "diffusion_objective":  m["diffusion_objective"],
            "arc_type":             m.get("arc", {}).get("type"),
            "encoder_id":           enc_id,
            "compatible_encoders":  m.get("compatible_encoders", [enc_id] if enc_id else []),
            "show_in_finetune_dropdown": ui.get("show_in_finetune_dropdown", True),
            "show_in_dataset_dropdown":  ui.get("show_in_dataset_dropdown", True),
            "vram":             ui.get("vram"),
            "lora_aggregate":   ui.get("lora_aggregate"),
            "sequence":         ui.get("sequence"),
            "module_structure": ui.get("module_structure"),
            "lora_layer_template": ui.get("lora_layer_template"),
        }

# Loader is invoked further down, AFTER ENCODING_MODELS and SHARED_ENCODERS
# are defined (otherwise the loader's references to those globals fail).


def estimate_training_vram_mb(base_model="sa3", batch_size=8, lora_rank=16, precision="16-mixed"):
    """Calibrated VRAM estimate for LoRA training (benchmarked on A100 80GB, 16-mixed).

    Formula: base + lora_overhead + activation_per_sample × batch_size
    - Base (11,500 MB): fp32 model weights + CUDA context + gradient buffers
    - LoRA overhead: 24 bytes/param (fp32 weights + fp16 copy + Adam m,v + grads + buffers)
    - Activation: ~1,100 MB per sample (measured)
    """
    base_mb = 11500

    # Rough LoRA param estimate: rank × ~1M params (varies by adapter type/filtering)
    # 24 bytes per param / 1M ≈ rank × 25 MB for standard LoRA
    lora_mb = lora_rank * 25

    act_per_sample_mb = 1100
    activation_mb = act_per_sample_mb * batch_size

    return int(base_mb + lora_mb + activation_mb)


def _backend_env_for_model(model_key):
    """Return an env-var fragment to set UNDERFIT_BACKEND for a given base model.

    Each MODEL_INFO entry declares a list of supported backends (e.g. SA3
    models support both 'sa3' and 'sat_dev' since the codepath is compatible).
    Pick the first one whose Python package is importable in the venv — that
    way 'pip install -e <sa3>' alone is enough to run an SA3 model even when
    the registry also lists sat_dev as compatible. Returns "" if none of the
    declared backends are importable (falls back to autodetect downstream).
    """
    import importlib.util
    info = MODEL_INFO.get(model_key) or {}
    backends = info.get("backends") or []
    # Skip backends whose Python package isn't installed.
    mod_for = {"sa3": "stable_audio_3", "sat_dev": "stable_audio_tools"}
    for b in backends:
        modname = mod_for.get(b)
        if modname and importlib.util.find_spec(modname) is not None:
            return f"UNDERFIT_BACKEND={b} "
    return ""


# Gradio launch constants. VENV_ACTIVATE is autodetected from the running
# Python's parent dir (so `source <bin>/activate` works for whichever venv
# the dashboard was launched from). Override with UNDERFIT_VENV_ACTIVATE.
VENV_ACTIVATE = os.environ.get(
    "UNDERFIT_VENV_ACTIVATE",
    str(Path(sys.executable).parent / "activate"),
)
# Path to the stable-audio-tools checkout (used by the sat_dev backend to
# read its defaults.ini). Defaults to a sibling clone next to underfit —
# the location where `underfit-setup --backend sat_dev` clones to. Override
# with UNDERFIT_SAT_DEV_DIR if your checkout lives elsewhere.
SA_TOOLS_DIR = Path(os.environ.get("UNDERFIT_SAT_DEV_DIR", BASE_DIR.parent / "stable-audio-tools")).expanduser()
# Use Underfit's backend-agnostic launcher; selects sat_dev or sa3 backend
# from --backend / UNDERFIT_BACKEND. Falls back to auto-detect.
RUN_GRADIO_SCRIPT = str(BASE_DIR / "run_gradio.py")
GRADIO_PORT_BASE = 7860
GRADIO_LOG_DIR = STATE_DIR / "gradio_logs"
GRADIO_LOG_DIR.mkdir(exist_ok=True)

def _slugify(name):
    """Sanitize a name to a lowercase slug [a-z0-9._-]."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9._\-]+", "-", s)  # replace non-slug chars with dash
    s = re.sub(r"-{2,}", "-", s)  # collapse multiple dashes
    s = s.strip("-")  # remove leading/trailing dashes
    return s or "unnamed"


def _dataset_for_step(dataset_history, step):
    """Return the dataset_history entry active at the given effective step."""
    if not dataset_history:
        return None
    result = dataset_history[0]
    for seg in dataset_history:
        if seg["from_step"] <= step:
            result = seg
        else:
            break
    return result


def _detect_process_state(pid):
    """Return 'running', 'paused', or 'dead'.

    Zombies (state 'Z') are treated as dead — they linger in the process
    table until reaped, but they're not doing anything. os.kill(pid, 0)
    succeeds against zombies, so we have to check /proc/<pid>/status to
    distinguish.
    """
    if pid is None:
        return "dead"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "dead"
    except PermissionError:
        pass  # process exists but we can't signal — it's alive
    # Check /proc/{pid}/status for stopped/zombie state
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("State:"):
                    state_char = line.split()[1]
                    if state_char in ("T", "t"):
                        return "paused"
                    if state_char in ("Z", "X"):
                        return "dead"
                    return "running"
    except Exception:
        pass
    return "running"


def _kill_process_group(pid, paused=False):
    """Kill an entire process group by PID. Handles orphaned children.

    Since we launch with os.setsid(), the stored PID is the session/group leader.
    Even if the bash wrapper is dead, orphaned children keep the same PGID,
    so we can signal the group directly via os.killpg(pid, ...) since pid == pgid.
    Uses SIGKILL directly — PyTorch/Lightning catches SIGTERM and delays exit.
    """
    # Try to get the actual PGID (works if leader is still alive)
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        # Leader dead — but pid == pgid because we used os.setsid()
        pgid = pid
    try:
        if paused:
            os.killpg(pgid, signal.SIGCONT)
            time.sleep(0.1)
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return  # entire group is already dead
    except Exception as e:
        print(f"[control] SIGKILL to PGID {pgid} failed: {e}")


def _free_gpu_memory(gpu):
    """Run cuda.empty_cache() on the specified GPU to free VRAM."""
    if gpu is None:
        return
    try:
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu} python3 -c "
            "'import torch; torch.cuda.empty_cache(); print(\"VRAM freed on GPU\", {gpu})'"
        )
        subprocess.Popen(
            ["bash", "-c", f"source {VENV_ACTIVATE} && {cmd}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"[cleanup] cuda.empty_cache failed for GPU {gpu}: {e}")


def _diagnose_quick_kill(run: dict) -> str | None:
    """Return a one-line hint about why a run died, or None if no signal.

    Checks in priority order:
      1. `<log>.exit` — written by lora_train.py's global try/except. Has the
         actual Python exception + traceback, robust to pipe/buffering issues.
      2. `<log>.bash.err` — bash's own stderr (source-failures, shell syntax,
         etc.). Rare but extremely useful when it fires.
      3. Elapsed <30 s + tiny log → SIGKILL (OOM-killer) heuristic.
    """
    log_path = run.get("log_path") or ""

    # 1. Python exit sidecar (highest signal)
    exit_path = log_path + ".exit"
    if os.path.exists(exit_path):
        try:
            with open(exit_path, "rb") as f:
                exit_txt = f.read(4096).decode("utf-8", errors="replace").strip()
            first = exit_txt.splitlines()[0][:250] if exit_txt else ""
            if first:
                return first

        except OSError:
            pass

    # 2. Bash-level error
    bash_err = log_path + ".bash.err"
    if os.path.exists(bash_err):
        try:
            with open(bash_err, "rb") as f:
                bash_err_txt = f.read(2048).decode("utf-8", errors="replace").strip()
            if bash_err_txt:
                first_line = bash_err_txt.splitlines()[0][:200]
                return f"Shell error before training started: {first_line}"
        except OSError:
            pass

    # 3. Did lora_train.py even reach __main__?
    started_path = log_path + ".started"
    reached_python = os.path.exists(started_path)
    if log_path and not reached_python:
        # No marker means bash died before python ran. Most likely: bad venv path,
        # source failure, missing executable, or shell-quoting bug. Should already
        # have shown up in .bash.err above, but if not, this is the next-best clue.
        return ("Process never reached python (no '.started' marker) — bash / "
                "venv / source failed silently. Inspect <log>.bash.err if it "
                "exists, or try the run's restart_cmd manually.")

    # 4. Quick death with empty log → OOM-killer heuristic
    elapsed = None
    try:
        ts = run.get("created_at", "")
        if ts:
            elapsed = time.time() - datetime.fromisoformat(ts).timestamp()
    except Exception:
        pass
    log_size = 0
    if log_path:
        try:
            log_size = os.path.getsize(log_path)
        except OSError:
            pass
    if elapsed is not None and elapsed < 30 and log_size < 200:
        return ("Process died in <30 s with no output — almost certainly SIGKILL "
                "from the kernel OOM-killer. Check `dmesg | tail` for "
                "'Killed process …', or try a smaller batch size / model.")
    return None


def _parse_latest_step(run):
    """Parse the latest global_step from a run's log file. Returns (step, max_steps) or (None, max_steps)."""
    log_path = Path(run.get("log_path", ""))
    max_steps = run.get("max_steps", 20000)
    step_offset = run.get("step_offset", 0)
    if not log_path.exists():
        return None, max_steps
    progress_re = re.compile(
        r"Epoch (\d+):\s+\d+%.*?(\d+)/(\d+)"
    )
    matched_line = None
    with open(log_path, "rb") as f:
        file_size = f.seek(0, 2)
        for chunk_size in [65536, 262144, 1048576]:
            f.seek(max(0, file_size - chunk_size))
            tail = f.read().decode("utf-8", errors="replace")
            lines = tail.strip().splitlines()
            m = None
            for line in reversed(lines):
                m = progress_re.search(line)
                if m:
                    matched_line = line
                    break
            if m:
                break
    if m:
        # New format publishes "Step N, Epoch N: ...": trust the in-band global step.
        step_prefix_m = re.search(r"Step (\d+),", matched_line) if matched_line else None
        if step_prefix_m:
            return int(step_prefix_m.group(1)), max_steps
        # Legacy "Epoch N: ..." only: derive from epoch + step_offset.
        epoch = int(m.group(1))
        steps_in_epoch = int(m.group(2))
        total_in_epoch = int(m.group(3))
        raw_step = max(0, epoch * total_in_epoch + steps_in_epoch - 1)
        return raw_step + step_offset, max_steps
    return None, max_steps


# Extract the "task signature" from a tqdm progress bar: prefix text + total count
# e.g. "Epoch 5490:  17%|... | 1/6"            -> ("Epoch:", "6")
#       "Step 518, Epoch 99: 20%|... | 1/5"    -> ("Epoch:", "5")
#       " 70%|... | 35/50"                     -> ("", "50")
_PBAR_SIG_RE = re.compile(r'^(.*?)\d+%\|[^|]*\|\s*\d+/(\d+)')

# Strip both legacy "Epoch N" and the new "Step N, Epoch N" prefixes so
# consecutive lines from different steps/epochs share the same signature
# and get collapsed into one row by _collapse_progress_lines.
_STEP_EPOCH_NUM_RE = re.compile(r'(?:Step \d+,\s*)?Epoch \d+')

def _pbar_signature(line):
    """Return a grouping key for a progress bar line, or None."""
    m = _PBAR_SIG_RE.match(line.rstrip())
    if not m:
        return None
    prefix = _STEP_EPOCH_NUM_RE.sub('Epoch', m.group(1).strip())
    return (prefix, m.group(2))

def _collapse_progress_lines(lines):
    """Collapse consecutive tqdm progress-bar lines with the same task signature.

    Keeps only the last line per consecutive group of bars that share the same
    prefix + total (e.g. all " N/50 [A" demo steps collapse, and consecutive
    "Epoch 5490: N/6" lines collapse, but different epochs are kept).
    """
    collapsed = []
    prev_sig = None
    for ln in lines:
        sig = _pbar_signature(ln)
        if sig and sig == prev_sig:
            collapsed[-1] = ln  # replace previous progress line in same group
        else:
            collapsed.append(ln)
        prev_sig = sig
    return collapsed


_COMPLETED_PBAR_RE = re.compile(r'^\s*100%\|')

def _process_log_lines(raw):
    """Shared log processing: handle \\r overwrites, collapse progress bars,
    strip completed (100%) bars, then re-collapse (stripping may expose
    new adjacent progress bars)."""
    lines = []
    for chunk in raw.split(b"\n"):
        if b"\r" in chunk:
            chunk = chunk.rsplit(b"\r", 1)[-1]
        lines.append(chunk.decode("utf-8", errors="replace"))
    lines = _collapse_progress_lines(lines)
    lines = [l for l in lines if not _COMPLETED_PBAR_RE.match(l)]
    # Strip empty lines then re-collapse: removing 100% bars and blank lines
    # may have made previously-separated progress bars adjacent
    lines = [l for l in lines if l.strip()]
    lines = _collapse_progress_lines(lines)
    return lines

def _read_log_tail(log_path, max_bytes=8192):
    """Read the last ~max_bytes of a log file, processed identically to the
    full log viewer (\\r handling, progress bar collapsing, 100% bar stripping)."""
    try:
        with open(log_path, "rb") as f:
            file_size = f.seek(0, 2)
            if file_size == 0:
                return ""
            offset = max(0, file_size - max_bytes)
            f.seek(offset)
            raw = f.read()
            # Drop first partial line if we didn't read from start
            if offset > 0:
                nl = raw.find(b"\n")
                if nl >= 0:
                    raw = raw[nl + 1:]
            return "\n".join(_process_log_lines(raw))
    except Exception:
        return ""

def _read_log_compressed(log_path):
    """Read an entire log file with full compression."""
    try:
        with open(log_path, "rb") as f:
            raw = f.read()
        if not raw:
            return ""
        return "\n".join(_process_log_lines(raw))
    except Exception:
        return ""


class RunsRegistry:
    """Thread-safe registry for training runs, backed by runs.json."""

    def __init__(self, path=RUNS_FILE):
        self._path = path
        self._lock = threading.Lock()
        self._runs = []
        self._load()

    def _load(self):
        if self._path.exists():
            with open(self._path) as f:
                self._runs = json.load(f)
        else:
            self._runs = []
        self._migrate()

    def _migrate(self):
        """Infer status for old runs and mark stale active runs as killed on startup."""
        changed = False
        for r in self._runs:
            if "step_offset" not in r:
                r["step_offset"] = 0
                changed = True
            pid = r.get("pid")
            status = r.get("status")
            # Check runs with no status or that claim to be active. Include
            # loading/resuming here so a crash during model load doesn't get
            # stuck displaying "Loading..." forever.
            if status is None or status in ("training", "paused", "loading", "resuming"):
                proc_state = _detect_process_state(pid)
                if proc_state in ("running", "paused"):
                    new_status = "training" if proc_state == "running" else "paused"
                else:
                    step, max_steps = _parse_latest_step(r)
                    if step is not None and step >= max_steps - 1:
                        new_status = "completed"
                    elif step is None or status in ("loading", "resuming"):
                        # No step ever logged (or still in init state) →
                        # crashed before training began. Show as error so
                        # the user sees there's a problem rather than
                        # mistaking it for an intentional kill.
                        new_status = "error"
                    else:
                        new_status = "killed"
                if status != new_status:
                    if status is not None:
                        print(f"[startup] Run '{r.get('display_name', r['id'])}' was {status}, PID {pid} dead → {new_status}")
                    r["status"] = new_status
                    if new_status in ("killed", "error") and not r.get("kill_hint"):
                        hint = _diagnose_quick_kill(r)
                        if hint:
                            r["kill_hint"] = hint
                    changed = True
        if changed:
            self._save()

    def _save(self):
        _atomic_write_json(self._path, self._runs)

    def list_runs(self):
        with self._lock:
            return list(self._runs)

    def get_run(self, run_id):
        with self._lock:
            for r in self._runs:
                if r["id"] == run_id:
                    return dict(r)
        return None

    def get_active_run_id(self):
        with self._lock:
            for r in reversed(self._runs):
                if r.get("active"):
                    return r["id"]
            if self._runs:
                return self._runs[-1]["id"]
        return None

    def add_run(self, run):
        with self._lock:
            # Deactivate all existing runs
            for r in self._runs:
                r["active"] = False
            self._runs.append(run)
            self._save()

    def reorder(self, id_list):
        """Reorder runs to match the given list of IDs. IDs not in the list keep their relative order at the end."""
        with self._lock:
            by_id = {r["id"]: r for r in self._runs}
            reordered = [by_id[rid] for rid in id_list if rid in by_id]
            remaining = [r for r in self._runs if r["id"] not in {rid for rid in id_list}]
            self._runs = reordered + remaining
            self._save()

    def remove_run(self, run_id):
        """Remove a run from the registry. Returns the removed run dict or None."""
        with self._lock:
            for i, r in enumerate(self._runs):
                if r["id"] == run_id:
                    removed = self._runs.pop(i)
                    self._save()
                    return removed
        return None

    def update_run(self, run_id, **fields):
        """General-purpose update method. Thread-safe, saves to disk."""
        with self._lock:
            for r in self._runs:
                if r["id"] == run_id:
                    r.update(fields)
                    self._save()
                    return True
        return False


registry = RunsRegistry()


# ---------------------------------------------------------------------------
# Dataset pre-encoding registry
# ---------------------------------------------------------------------------

DATASETS_FILE = STATE_FILES_DIR / "datasets.json"
DATASETS_DIR = STATE_DIR / "datasets"   # latents + tag caches per dataset


def _tag_cache_path(ds_id):
    """Return the path for a dataset's cached tag metadata."""
    return DATASETS_DIR / f"{ds_id}_tags.json"


_TAG_CACHE_VERSION = 2  # bump to invalidate old caches (v2: added duration)

def _save_tag_cache(ds_id, file_info, total_files, files_with_tags, files_with_json):
    """Write scanned tag data to a JSON cache file."""
    cache = {
        "version": _TAG_CACHE_VERSION,
        "files": file_info,
        "total_files": total_files,
        "files_with_tags": files_with_tags,
        "files_with_json": files_with_json,
    }
    try:
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        with open(_tag_cache_path(ds_id), "w") as f:
            json.dump(cache, f)
        print(f"[datasets] Cached {total_files} file tags for {ds_id}")
    except Exception as e:
        print(f"[datasets] Warning: failed to write tag cache for {ds_id}: {e}")


def _load_tag_cache(ds_id):
    """Load cached tag data. Returns (file_info, total, with_tags, with_json) or None."""
    cp = _tag_cache_path(ds_id)
    if not cp.exists():
        return None
    try:
        with open(cp) as f:
            cache = json.load(f)
        if cache.get("version") != _TAG_CACHE_VERSION:
            return None  # stale cache, re-scan
        return (cache["files"], cache["total_files"],
                cache["files_with_tags"], cache["files_with_json"])
    except Exception:
        return None


# Populated by _load_models_from_json() from dashboard/models/*/registry.json.
# Mirror of MODELS dict from pre_encode.py (model_key → description).
ENCODING_MODELS: dict = {}

# Populated by _load_models_from_json() — derived from each JSON's encoder_id
# and encoder_canonical_model fields. Models sharing an encoder_id are grouped
# under the canonical model's key. Datasets pre-encoded for any member are
# reusable across all members.
SHARED_ENCODERS: dict = {}

# Load model registry files now that the dicts they populate exist.
_load_models_from_json()
for _enc_id, _grp in _ENCODER_GROUPS.items():
    _canon = _grp["canonical"]
    _existing = SHARED_ENCODERS.setdefault(_canon, [_canon])
    for _m in _grp["members"]:
        if _m not in _existing:
            _existing.append(_m)

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".aiff", ".aif", ".m4a"}
_MIN_AUDIO_SIZE = 4096  # bytes — skip files smaller than this (resource forks, corrupt)


def _is_audio_file(path):
    """Return True if path looks like a real audio file (not a macOS resource fork, etc.)."""
    p = Path(path)
    if p.name.startswith("._"):
        return False
    if p.suffix.lower() not in AUDIO_EXTS:
        return False
    try:
        if p.stat().st_size < _MIN_AUDIO_SIZE:
            return False
    except OSError:
        return False
    return True


# MP4 atom → our canonical key name
_MP4_TAG_MAP = {
    "\xa9nam": "title", "\xa9ART": "artist", "\xa9alb": "album",
    "\xa9gen": "genre", "\xa9day": "date", "\xa9wrt": "composer",
    "tmpo": "bpm", "aART": "artist",
}
_TAG_KEYS = ("title", "artist", "album", "genre", "label", "date", "composer", "bpm")
# ID3 frame → canonical key (for mutagen fallback on MP3/FLAC/OGG)
_ID3_TAG_MAP = {
    "TIT2": "title", "TPE1": "artist", "TALB": "album",
    "TCON": "genre", "TDRC": "date", "TDAT": "date",
    "TCOM": "composer", "TBPM": "bpm", "TPUB": "label",
}


try:
    import audio_metadata as _am
except ImportError:
    _am = None
try:
    from mutagen import File as _MutagenFile
except ImportError:
    _MutagenFile = None


_MUTAGEN_FIRST_EXTS = {".m4a", ".mp4", ".aac", ".alac"}


def _find_json_sidecar(fpath, json_map=None):
    """Find JSON sidecar for an audio file.

    Search order:
      1. Same directory: track.json next to track.wav
      2. Cross-directory: use json_map to find matching stems in sister/nested dirs.
         Prefer closest match (fewest path component differences).
    Returns the Path to the sidecar, or None.
    """
    fp = Path(fpath)
    # 1. Same directory (fast path)
    same_dir = fp.with_suffix(".json")
    if same_dir.exists():
        return same_dir
    # 2. Cross-directory lookup via pre-built map
    if json_map:
        stem = fp.stem
        candidates = json_map.get(stem)
        if candidates:
            if len(candidates) == 1:
                return candidates[0]
            # Rank by path similarity: count shared path components from the root
            audio_parts = fp.parent.parts
            best = None
            best_score = -1
            for jpath in candidates:
                jp = jpath.parent.parts
                shared = 0
                for a, b in zip(audio_parts, jp):
                    if a == b:
                        shared += 1
                    else:
                        break
                if shared > best_score:
                    best_score = shared
                    best = jpath
            return best
    return None


def _read_audio_tags(fpath, json_map=None):
    """Read metadata tags from an audio file.

    Priority: JSON sidecar ({stem}.json) > embedded tags.
    The sidecar is created by autotagger.py for files without embedded tags.
    For embedded: MP4 uses mutagen directly (fast), ID3 uses audio_metadata first.
    """
    # Check for JSON sidecar first (same dir or cross-directory)
    sidecar = _find_json_sidecar(fpath, json_map)
    if sidecar:
        try:
            with open(sidecar) as f:
                sc = json.load(f)
            sc_tags = {}
            for key, val in sc.items():
                if val and isinstance(val, (str, int, float)):
                    sc_tags[key] = str(val)
            if sc_tags:
                return sc_tags
        except Exception:
            pass

    ext = Path(fpath).suffix.lower()
    tags = {}

    # MP4/M4A/AAC — mutagen is fast and correct, audio_metadata is very slow
    if ext in _MUTAGEN_FIRST_EXTS:
        if _MutagenFile is not None:
            try:
                mf = _MutagenFile(str(fpath))
                if mf:
                    for atom, canonical in _MP4_TAG_MAP.items():
                        val = mf.get(atom)
                        if val:
                            val = str(val[0]) if isinstance(val, (list, tuple)) else str(val)
                            if val and canonical not in tags:
                                tags[canonical] = val
            except Exception:
                pass
        return tags

    # ID3 formats — try audio_metadata first (richer tag parsing)
    if _am is not None:
        try:
            track_md = _am.load(str(fpath))
            raw_tags = track_md.get("tags", {})
            for key in _TAG_KEYS:
                if key in raw_tags:
                    val = raw_tags[key]
                    if isinstance(val, (list, tuple)) and len(val) > 0:
                        val = str(val[0])
                    else:
                        val = str(val)
                    if val:
                        tags[key] = val
            if tags:
                return tags
        except Exception:
            pass

    # Fallback to mutagen for anything else (tries both ID3 and MP4 keys)
    if _MutagenFile is not None:
        try:
            mf = _MutagenFile(str(fpath))
            if mf:
                for frame, canonical in _ID3_TAG_MAP.items():
                    val = mf.get(frame)
                    if val:
                        val = str(val[0]) if isinstance(val, (list, tuple)) else str(val)
                        if val and canonical not in tags:
                            tags[canonical] = val
                for atom, canonical in _MP4_TAG_MAP.items():
                    val = mf.get(atom)
                    if val:
                        val = str(val[0]) if isinstance(val, (list, tuple)) else str(val)
                        if val and canonical not in tags:
                            tags[canonical] = val
        except Exception:
            pass
    return tags


class DatasetsRegistry:
    """Thread-safe registry for pre-encoded datasets, backed by datasets.json."""

    def __init__(self, path=DATASETS_FILE):
        self._path = path
        self._lock = threading.Lock()
        self._datasets = []
        self._mtime = 0
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                self._mtime = self._path.stat().st_mtime
                with open(self._path) as f:
                    self._datasets = json.load(f)
            except Exception:
                self._datasets = []
        else:
            self._datasets = []

    def _maybe_reload(self):
        """Reload from disk if the file has been modified externally."""
        try:
            if self._path.exists() and self._path.stat().st_mtime != self._mtime:
                self._load()
        except OSError:
            pass

    def _save(self):
        _atomic_write_json(self._path, self._datasets)

    def list_datasets(self):
        with self._lock:
            self._maybe_reload()
            return list(self._datasets)

    def get_dataset(self, ds_id):
        with self._lock:
            self._maybe_reload()
            for d in self._datasets:
                if d["id"] == ds_id:
                    return dict(d)
        return None

    def add_dataset(self, ds):
        with self._lock:
            self._datasets.append(ds)
            self._save()

    def update_dataset(self, ds_id, **fields):
        with self._lock:
            for d in self._datasets:
                if d["id"] == ds_id:
                    d.update(fields)
                    self._save()
                    return True
        return False

    def remove_dataset(self, ds_id):
        with self._lock:
            for i, d in enumerate(self._datasets):
                if d["id"] == ds_id:
                    removed = self._datasets.pop(i)
                    self._save()
                    return removed
        return None


datasets_registry = DatasetsRegistry()


def _generate_dataset_ground_truth(dataset_files, dataset_name, model="sa3-medium", num_tracks=4):
    """Pick random files from dataset_files, convert to MP3 clips.

    Returns (ground_truth_list, demo_prompts_list) or (None, None) on failure.
    Ground truth MP3s are saved to audio/ground_truth/{dataset_name}/.
    """
    if not dataset_files:
        return None, None
    picks = random.sample(dataset_files, min(num_tracks, len(dataset_files)))

    gt_dir = AUDIO_DIR / "ground_truth" / dataset_name
    gt_dir.mkdir(parents=True, exist_ok=True)

    gt_list = []
    prompts = []
    for i, entry in enumerate(picks):
        fpath = entry.get("source_path", "")
        if not fpath or not os.path.isfile(fpath):
            continue
        out_mp3 = gt_dir / f"track_{i}.mp3"
        if not out_mp3.exists():
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", fpath,
                     "-map", "0:a", "-codec:a", "libmp3lame", "-q:a", "2",
                     "-loglevel", "error", str(out_mp3)],
                    check=True, timeout=300,
                )
            except Exception as e:
                print(f"[gt] Failed to convert {fpath}: {e}")
                continue
        title = entry.get("title", Path(fpath).stem)
        url = f"/audio/ground_truth/{dataset_name}/track_{i}.mp3"
        gt_entry = {"title": title, "url": url}
        for k in ("album", "year", "bpm", "genre"):
            if entry.get(k):
                gt_entry[k] = entry[k]
        gt_list.append(gt_entry)
        prompts.append(entry.get("prompt", f"Title: {title}"))
        # Generate spectrogram
        jpg = out_mp3.with_suffix(".jpg")
        if not jpg.exists():
            _spec_pool.submit(generate_spectrogram, out_mp3, jpg)

    if not gt_list:
        return None, None
    # Pad to num_tracks if we got fewer
    while len(gt_list) < num_tracks and gt_list:
        gt_list.append(gt_list[-1])
        prompts.append(prompts[-1])
    print(f"[gt] Generated {len(gt_list)} ground truth tracks for '{dataset_name}'")
    return gt_list, prompts


def _build_dataset_files(input_dir, exclude_set=None):
    """Scan audio files in input_dir and build a list of {prompt, source_path, title, ...}.

    This is the source of truth for which files are in the dataset.
    Used to select demo prompts and ground truth tracks per run.
    Reads both embedded tags and JSON sidecars (same dir or cross-directory).
    All sidecar key-value pairs are stored in the pool entry for matching.
    """
    src = Path(input_dir)
    if not src.is_dir():
        return []
    audio_files = []
    json_files = {}  # stem -> list of Paths (for cross-directory sidecar lookup)
    for dirpath_, _, filenames in os.walk(src):
        dp = Path(dirpath_)
        for fn in filenames:
            fp = dp / fn
            if _is_audio_file(fp):
                if exclude_set and str(fp.relative_to(src)) in exclude_set:
                    continue
                audio_files.append(fp)
            elif fn.lower().endswith(".json"):
                json_files.setdefault(Path(fn).stem, []).append(fp)
    pool = []
    for fpath in audio_files:
        tags = _read_audio_tags(str(fpath), json_map=json_files)
        title = tags.get("title", fpath.stem)
        album = tags.get("album", "")
        artist = tags.get("artist", "")
        genre = tags.get("genre", "")
        year = tags.get("date", "")
        bpm = tags.get("bpm", "")
        # Build GT-style prompt string
        parts = []
        if artist: parts.append(f"Artist: {artist}")
        parts.append(f"Title: {title}")
        if year: parts.append(f"Year: {year}")
        if bpm: parts.append(f"BPM: {bpm}")
        if genre: parts.append(f"Genre: {genre}")
        if album: parts.append(f"Album: {album}")
        prompt = ", ".join(parts) if parts else f"Track: {fpath.stem}"
        entry = {"prompt": prompt, "source_path": str(fpath), "title": title}
        if album: entry["album"] = album
        if artist: entry["artist"] = artist
        if year: entry["year"] = year
        if bpm: entry["bpm"] = bpm
        if genre: entry["genre"] = genre
        # Store all extra sidecar tags for matching (e.g. 'id', 'prompt' from autotagger)
        for k, v in tags.items():
            if k not in ("title", "artist", "album", "genre", "date", "bpm", "label", "composer"):
                # Avoid colliding with our GT-style 'prompt' key
                store_key = "sidecar_prompt" if k == "prompt" else k
                if store_key not in entry:
                    entry[store_key] = v
        pool.append(entry)
    print(f"[dataset_files] Built {len(pool)} file entries from '{input_dir}'")
    return pool







GRADIO_STATE_FILE = STATE_FILES_DIR / "gradio_instances.json"


class GradioManager:
    """Manages Gradio inference instances — multiple per GPU if VRAM allows."""

    def __init__(self):
        self._lock = threading.Lock()
        self._instances = {}      # id -> instance dict
        self._cleanup_stale_on_startup()

    def _cleanup_stale_on_startup(self):
        """On startup, check persisted gradio instances and mark dead ones as stopped."""
        if not GRADIO_STATE_FILE.exists():
            return
        try:
            with open(GRADIO_STATE_FILE) as f:
                data = json.load(f)
            alive = []
            for item in data:
                pid = item.get("pid")
                if pid and _detect_process_state(pid) != "dead":
                    alive.append(item)
                else:
                    print(f"[startup] Gradio instance PID {pid} ({item.get('title', '?')}) is dead — removing")
            _atomic_write_json(GRADIO_STATE_FILE, alive)
        except Exception as e:
            print(f"[startup] Failed to clean stale gradio instances: {e}")

    def _persist(self):
        """Save instance state to disk so share URLs survive restarts."""
        try:
            data = []
            for inst in self._instances.values():
                if inst["status"] in ("ready", "starting"):
                    data.append({
                        "pid": inst["pid"],
                        "gpu": inst["gpu"],
                        "checkpoint_path": inst["checkpoint_path"],
                        "checkpoint_name": inst["checkpoint_name"],
                        "share_url": inst["share_url"],
                        "run_id": inst["run_id"],
                        "title": inst["title"],
                        "log_path": inst.get("log_path"),
                        "started_at": inst.get("started_at"),
                    })
            _atomic_write_json(GRADIO_STATE_FILE, data)
        except Exception as e:
            print(f"[gradio] Failed to persist state: {e}")

    @staticmethod
    def load_persisted():
        """Load persisted instance state for orphan recovery.

        Returns dict of pid -> {"share_url": ..., "log_path": ...}.
        """
        if not GRADIO_STATE_FILE.exists():
            return {}
        try:
            with open(GRADIO_STATE_FILE) as f:
                data = json.load(f)
            return {
                item["pid"]: {
                    "share_url": item.get("share_url"),
                    "log_path": item.get("log_path"),
                    "started_at": item.get("started_at"),
                }
                for item in data if item.get("pid")
            }
        except Exception:
            return {}

    def _find_available_port(self):
        """Find next available port starting from GRADIO_PORT_BASE."""
        used_ports = {inst["port"] for inst in self._instances.values()
                      if inst["status"] in ("starting", "ready")}
        port = GRADIO_PORT_BASE
        while port in used_ports:
            port += 1
        return port

    def launch(self, checkpoint_path, gpu, run_id=None, checkpoint_name=None, title=None, model_variant=None, verbose=False):
        with self._lock:
            instance_id = uuid.uuid4().hex[:12]
            port = self._find_available_port()
            if not title:
                title = checkpoint_name or Path(checkpoint_path).name
            # Resolve base model from run record
            base_model = "sa3-medium"
            if run_id:
                run = registry.get_run(run_id)
                if run:
                    base_model = run.get("base_model", "sa3-medium")
            gmi = _get_model_info(base_model)
            # Choose ARC vs RF base checkpoint
            use_arc_full = (model_variant == "arc" and gmi.get("arc_type") == "full_model")
            use_arc_lora = (model_variant == "arc" and gmi.get("arc_type") == "lora")
            arc_lora_path = None
            if use_arc_full:
                ckpt_path_model = gmi["arc_ckpt"]
                config_path_model = gmi["arc_config"]
            elif use_arc_lora:
                ckpt_path_model = gmi["base_ckpt"]
                config_path_model = gmi["arc_config"]
                arc_lora_path = gmi["arc_ckpt"]
            else:
                ckpt_path_model = gmi["base_ckpt"]
                # Prefer run's model config (has demo_cond prompts), fall back to base
                config_path_model = gmi['base_config']
                if run_id:
                    runs_dir = RUNS_DIR
                    resume_cfg = runs_dir / f"{run_id}_model_resume.json"
                    orig_cfg = runs_dir / f"{run_id}_model.json"
                    if resume_cfg.exists():
                        config_path_model = str(resume_cfg)
                    elif orig_cfg.exists():
                        config_path_model = str(orig_cfg)
            # Resolve default prompt from LoRA's training config
            default_prompt_arg = ""
            if run_id:
                runs_dir_p = RUNS_DIR
                for cfg_name in [f"{run_id}_model_resume.json", f"{run_id}_model.json"]:
                    cfg_p = runs_dir_p / cfg_name
                    if cfg_p.exists():
                        try:
                            with open(cfg_p) as _f:
                                _cfg = json.load(_f)
                            demo_cond = _cfg.get("training", {}).get("demo", {}).get("demo_cond", [])
                            prompts = [e.get("prompt", "") for e in demo_cond if e.get("prompt")]
                            if prompts:
                                default_prompt_arg = f" --default-prompt {shlex.quote(random.choice(prompts))}"
                        except Exception:
                            pass
                        break

            lora_args = f"--lora-ckpt-path {arc_lora_path} {checkpoint_path}" if arc_lora_path else f"--lora-ckpt-path {checkpoint_path}"
            verbose_arg = " --verbose" if verbose else ""
            thread_env = (
                "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "
                "NUMEXPR_NUM_THREADS=4 RAYON_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false "
            ) if GRADIO_THREAD_CAP else ""
            backend_env = _backend_env_for_model(base_model)
            cmd = (
                f"source {VENV_ACTIVATE} && "
                f"{backend_env}CUDA_VISIBLE_DEVICES={gpu} GRADIO_SERVER_PORT={port} PYTHONUNBUFFERED=1 "
                f"{thread_env}"
                f"python3 {RUN_GRADIO_SCRIPT} "
                f"--model-config {config_path_model} "
                f"--ckpt-path {ckpt_path_model} "
                f"{lora_args} "
                f"--model-half "
                f"--title {shlex.quote(title)}"
                f"{default_prompt_arg}"
                f"{verbose_arg}"
            )
            log_path = str(GRADIO_LOG_DIR / f"{instance_id}.log")
            log_file = open(log_path, "w")
            try:
                proc = subprocess.Popen(
                    ["bash", "-c", cmd],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,
                )
            except Exception as e:
                log_file.close()
                return None, str(e)
            finally:
                log_file.close()  # child has its own fd via fork

            instance = {
                "id": instance_id,
                "run_id": run_id,
                "checkpoint_path": checkpoint_path,
                "checkpoint_name": checkpoint_name or Path(checkpoint_path).name,
                "gpu": gpu,
                "port": port,
                "pid": proc.pid,
                "status": "starting",
                "share_url": None,
                "local_url": f"http://localhost:{port}",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "error": None,
                "title": title,
                "log_path": log_path,
            }
            self._instances[instance_id] = instance

        # Record VRAM baseline before model loads
        _record_gradio_baseline(instance_id, gpu)

        # Start log tailer in background
        t = threading.Thread(target=self._log_tailer, args=(instance_id, proc, log_path), daemon=True)
        t.start()
        return instance_id, None

    def stop(self, instance_id):
        with self._lock:
            inst = self._instances.get(instance_id)
            if not inst:
                return False
            pid = inst["pid"]
            gpu = inst["gpu"]

        # Kill the process group
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception:
                pass

        with self._lock:
            inst = self._instances.get(instance_id)
            if inst:
                inst["status"] = "stopped"
            self._persist()
        return True

    def stop_by_gpu(self, gpu):
        """Stop ALL Gradio instances on a given GPU."""
        ids_to_stop = []
        with self._lock:
            for iid, inst in self._instances.items():
                if inst["gpu"] == gpu and inst["status"] in ("starting", "ready"):
                    ids_to_stop.append(iid)
        for iid in ids_to_stop:
            self.stop(iid)

    def register_existing(self, pid, gpu, checkpoint_path, checkpoint_name=None,
                          title=None, run_id=None, share_url=None, log_path=None,
                          started_at=None):
        """Register an already-running Gradio instance (orphan recovery).

        Unlike launch(), allows multiple instances on the same GPU since
        orphaned processes may have been started outside our control.
        """
        with self._lock:
            # Don't double-register
            for inst in self._instances.values():
                if inst["pid"] == pid:
                    return None
            instance_id = uuid.uuid4().hex[:12]
            # Detect actual listening port; fall back to gpu-based estimate
            port = _detect_listening_port(pid) or (GRADIO_PORT_BASE + gpu)
            if not checkpoint_name:
                checkpoint_name = Path(checkpoint_path).name
            instance = {
                "id": instance_id,
                "run_id": run_id,
                "checkpoint_path": checkpoint_path,
                "checkpoint_name": checkpoint_name,
                "gpu": gpu,
                "port": port,
                "pid": pid,
                "status": "ready",
                "share_url": share_url,
                "local_url": f"http://localhost:{port}",
                "started_at": started_at,
                "error": None,
                "title": title or checkpoint_name,
                "log_path": log_path,
            }
            self._instances[instance_id] = instance
            self._persist()

        # Start log tailer if we have a log file
        if log_path and Path(log_path).exists():
            t = threading.Thread(
                target=self._log_tailer, args=(instance_id, None, log_path), daemon=True)
            t.start()

        return instance_id

    def remove_instance(self, instance_id):
        """Remove a dead instance from tracking entirely."""
        with self._lock:
            inst = self._instances.pop(instance_id, None)
            if inst:
                self._persist()
            return inst is not None

    def list_instances(self):
        with self._lock:
            return list(self._instances.values())

    def _log_tailer(self, instance_id, proc, log_path):
        """Tail a Gradio instance's log file, parsing status lines.

        Works for both dashboard-launched instances (proc is a Popen object)
        and orphan-recovered instances (proc is None, uses PID from instance).
        """
        share_re = re.compile(r"Running on public URL:\s*(https://\S+)")
        local_re = re.compile(r"Running on local URL:\s*https?://[^:]+:(\d+)")

        def _process_line(line):
            line = line.strip()
            if not line:
                return
            print(f"[gradio:{instance_id}] {line}")
            lm = local_re.search(line)
            if lm:
                actual_port = int(lm.group(1))
                with self._lock:
                    inst = self._instances.get(instance_id)
                    if inst and inst["port"] != actual_port:
                        print(f"[gradio:{instance_id}] Port reassigned: {inst['port']} -> {actual_port}")
                        inst["port"] = actual_port
                        inst["local_url"] = f"http://localhost:{actual_port}"
                        self._persist()
            m = share_re.search(line)
            if m:
                with self._lock:
                    inst = self._instances.get(instance_id)
                    if inst:
                        inst["share_url"] = m.group(1)
                        inst["status"] = "ready"
                    self._persist()

        def _is_alive():
            if proc is not None:
                return proc.poll() is None
            with self._lock:
                inst = self._instances.get(instance_id)
                if not inst:
                    return False
                pid = inst["pid"]
            try:
                os.kill(pid, 0)
                return True
            except (ProcessLookupError, PermissionError):
                return False

        try:
            with open(log_path, "r") as f:
                while True:
                    line = f.readline()
                    if line:
                        _process_line(line)
                    elif not _is_alive():
                        # Drain remaining lines after process exit
                        for line in f:
                            _process_line(line)
                        break
                    else:
                        time.sleep(0.5)
        except Exception:
            pass

        # Handle process exit
        if proc is not None:
            proc.wait()
            rc = proc.returncode
        else:
            rc = -1  # orphan — we just detected it died
        with self._lock:
            inst = self._instances.get(instance_id)
            if inst and inst["status"] not in ("stopped",):
                if rc == 137:
                    inst["error"] = "OOM (killed by kernel)"
                elif rc == 139:
                    inst["error"] = "Segfault"
                elif rc != 0:
                    inst["error"] = f"Exit code {rc}"
                else:
                    inst["error"] = "Process exited"
                inst["status"] = "error"
            self._persist()


gradio_manager = GradioManager()


def _discover_share_url_from_pipe(gradio_pid):
    """Try to read frpc share URL from the stdout pipe buffer of a Gradio process.

    When frpc reconnects to the tunnel server, it prints 'start proxy success: URL'
    to stdout. If the Gradio process isn't consuming the pipe, messages accumulate
    in the kernel buffer and can be read here.
    """
    try:
        # Find frpc child processes and their stdout pipe inodes
        frpc_pipes = {}  # pipe_inode -> frpc_pid
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            child_pid = int(entry.name)
            try:
                status = (entry / "status").read_text()
                if f"PPid:\t{gradio_pid}" not in status:
                    continue
                cmdline = (entry / "cmdline").read_bytes().decode("utf-8", errors="replace")
                if "frpc" not in cmdline:
                    continue
                # Get stdout pipe inode
                link = os.readlink(f"/proc/{child_pid}/fd/1")
                if link.startswith("pipe:["):
                    inode = link.split("[")[1].rstrip("]")
                    frpc_pipes[inode] = child_pid
            except Exception:
                continue

        if not frpc_pipes:
            return None

        # Find the read end of the pipe in the Gradio process
        fd_dir = f"/proc/{gradio_pid}/fd"
        for inode in frpc_pipes:
            for fd_name in os.listdir(fd_dir):
                try:
                    link = os.readlink(f"{fd_dir}/{fd_name}")
                    if link == f"pipe:[{inode}]":
                        # Read all available data (non-blocking)
                        fd = os.open(f"{fd_dir}/{fd_name}", os.O_RDONLY | os.O_NONBLOCK)
                        try:
                            data = b""
                            while True:
                                try:
                                    chunk = os.read(fd, 65536)
                                    if not chunk:
                                        break
                                    data += chunk
                                except OSError:
                                    break
                        finally:
                            os.close(fd)
                        if data:
                            text = data.decode("utf-8", errors="replace")
                            m = re.search(r"start proxy success:\s*(\S+)", text)
                            if m:
                                return m.group(1)
                except Exception:
                    continue
    except Exception:
        pass
    return None


def _discover_share_url_via_frpc(gradio_pid):
    """Discover share URL by finding the frpc child's token and probing the tunnel server.

    The share URL subdomain equals the frpc proxy name. We start a temporary
    frpc with the same share_token — the server rejects the duplicate but the
    proxy name in the error log reveals the URL.
    """
    try:
        # Find frpc child and extract share_token from its cmdline
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                status = (entry / "status").read_text()
                if f"PPid:\t{gradio_pid}" not in status:
                    continue
                cmdline = (entry / "cmdline").read_bytes().decode("utf-8", errors="replace").split("\0")
                if not any("frpc" in arg for arg in cmdline):
                    continue
                # Extract -n token and --server_addr
                token = None
                server_addr = None
                for i, arg in enumerate(cmdline):
                    if arg == "-n" and i + 1 < len(cmdline):
                        token = cmdline[i + 1]
                    elif arg == "--server_addr" and i + 1 < len(cmdline):
                        server_addr = cmdline[i + 1]
                if not token or not server_addr:
                    continue

                # Find certificate file
                cwd = os.readlink(f"/proc/{entry.name}/cwd")
                cert = os.path.join(cwd, ".gradio", "certificate.pem")
                if not os.path.exists(cert):
                    continue

                # Find frpc binary
                binary = os.readlink(f"/proc/{entry.name}/exe")

                # Start temporary frpc with same token to get proxy name
                cmd = [
                    binary, "http",
                    "-n", token,
                    "-l", "19999",  # dummy port
                    "-i", "127.0.0.1",
                    "--uc", "--sd", "random", "--ue",
                    "--server_addr", server_addr,
                    "--disable_log_color",
                    "--tls_enable", "--tls_trusted_ca_file", cert,
                ]
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                try:
                    # Read output for up to 10 seconds
                    import select
                    proxy_name = None
                    deadline = time.time() + 10
                    while time.time() < deadline:
                        ready, _, _ = select.select(
                            [proc.stdout, proc.stderr], [], [], 1.0,
                        )
                        for fd in ready:
                            line = fd.readline().decode("utf-8", errors="replace")
                            # Proxy name appears in: "proxy added: [NAME]"
                            m = re.search(r"proxy added: \[(\S+)\]", line)
                            if m:
                                proxy_name = m.group(1)
                            # Also check for "start proxy success: URL" (unlikely but possible)
                            m2 = re.search(r"start proxy success:\s*(\S+)", line)
                            if m2:
                                return m2.group(1)
                        if proxy_name:
                            break
                        if proc.poll() is not None:
                            break
                finally:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except Exception:
                        proc.kill()

                if proxy_name:
                    return f"https://{proxy_name}.gradio.live"
            except Exception:
                continue
    except Exception:
        pass
    return None


def _discover_missing_share_urls():
    """Try to discover share URLs for instances that don't have them."""
    updated = False
    for inst in gradio_manager.list_instances():
        if inst["status"] == "ready" and not inst["share_url"]:
            # Try pipe buffer first (fast, no subprocess)
            url = _discover_share_url_from_pipe(inst["pid"])
            # Fall back to frpc probe (starts a temporary process)
            if not url:
                url = _discover_share_url_via_frpc(inst["pid"])
            if url:
                with gradio_manager._lock:
                    live_inst = gradio_manager._instances.get(inst["id"])
                    if live_inst:
                        live_inst["share_url"] = url
                        updated = True
                print(f"[discover] Found share URL for PID {inst['pid']}: {url}")
    if updated:
        with gradio_manager._lock:
            gradio_manager._persist()


def _detect_listening_port(pid):
    """Detect the TCP port a process is listening on (in Gradio port range).

    Inspects /proc/{pid}/fd for socket inodes, then cross-references with
    /proc/net/tcp{,6} to find LISTEN sockets in the GRADIO_PORT_BASE+ range.
    """
    try:
        fd_dir = f"/proc/{pid}/fd"
        socket_inodes = set()
        for fd_name in os.listdir(fd_dir):
            try:
                link = os.readlink(f"{fd_dir}/{fd_name}")
                if link.startswith("socket:["):
                    inode = link.split("[")[1].rstrip("]")
                    socket_inodes.add(inode)
            except Exception:
                continue
        if not socket_inodes:
            return None
        for proto in ("tcp", "tcp6"):
            try:
                with open(f"/proc/{pid}/net/{proto}") as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) < 10:
                            continue
                        # State 0A = LISTEN
                        if parts[3] != "0A":
                            continue
                        if parts[9] in socket_inodes:
                            port = int(parts[1].split(":")[1], 16)
                            if port >= GRADIO_PORT_BASE:
                                return port
            except Exception:
                continue
    except Exception:
        pass
    return None


def _audit_gradio_ports():
    """Check registered Gradio instances for port mismatches and fix them."""
    updated = False
    for inst in gradio_manager.list_instances():
        if inst["status"] not in ("ready", "starting"):
            continue
        pid = inst["pid"]
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError, TypeError):
            continue
        actual_port = _detect_listening_port(pid)
        if actual_port and actual_port != inst["port"]:
            with gradio_manager._lock:
                live = gradio_manager._instances.get(inst["id"])
                if live:
                    print(f"[port-audit] {inst['id']} PID {pid}: port {live['port']} -> {actual_port}")
                    live["port"] = actual_port
                    live["local_url"] = f"http://localhost:{actual_port}"
                    updated = True
    if updated:
        with gradio_manager._lock:
            gradio_manager._persist()


def _recover_orphaned_gradios():
    """Scan for run_gradio.py processes not tracked by GradioManager and register them."""
    # Load persisted state from previous dashboard session
    persisted = GradioManager.load_persisted()  # pid -> {share_url, log_path}

    # Map PID -> GPU index via nvidia-smi
    pid_to_gpu = {}
    try:
        # Get GPU UUID -> index mapping
        gpu_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,gpu_uuid", "--format=csv,noheader"],
            timeout=5, stderr=subprocess.DEVNULL,
        ).decode().strip()
        uuid_to_idx = {}
        for line in gpu_out.split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                uuid_to_idx[parts[1]] = int(parts[0])
        # Get PID -> GPU UUID mapping
        app_out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
            timeout=5, stderr=subprocess.DEVNULL,
        ).decode().strip()
        for line in app_out.split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    pid_to_gpu[int(parts[0])] = uuid_to_idx.get(parts[1])
                except ValueError:
                    pass
    except Exception:
        return 0

    # Already-tracked PIDs
    tracked_pids = {inst["pid"] for inst in gradio_manager.list_instances()}

    # Build run_id lookup from checkpoint paths
    runs = registry.list_runs()

    recovered = 0
    # Scan /proc for run_gradio.py processes
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in tracked_pids:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().decode("utf-8", errors="replace").split("\0")
        except Exception:
            continue
        # Check if this is a run_gradio.py process
        if not any("run_gradio.py" in arg for arg in cmdline):
            continue
        # Parse --lora-ckpt-path (may have multiple values: ARC LoRA + finetune LoRA) and --title
        lora_ckpts = []
        title = None
        i = 0
        while i < len(cmdline):
            if cmdline[i] == "--lora-ckpt-path":
                i += 1
                # Collect all positional values until next --flag or end
                while i < len(cmdline) and not cmdline[i].startswith("--"):
                    if cmdline[i]:
                        lora_ckpts.append(cmdline[i])
                    i += 1
            elif cmdline[i] == "--title" and i + 1 < len(cmdline):
                title = cmdline[i + 1]
                i += 2
            else:
                i += 1
        if not lora_ckpts:
            continue
        # Last LoRA path is the finetune checkpoint (first may be ARC base LoRA)
        lora_ckpt = lora_ckpts[-1]
        gpu = pid_to_gpu.get(pid)
        if gpu is None:
            continue
        # Try to match a run_id from any of the LoRA checkpoint paths
        run_id = None
        for ckpt in lora_ckpts:
            for r in runs:
                if r["id"] in ckpt:
                    run_id = r["id"]
                    break
            if run_id:
                break
        # Only adopt PIDs we know about (in our persisted state) or whose
        # LoRA checkpoint maps to one of our tracked runs. Skip foreign
        # run_gradio.py processes from other dashboards / shells / users.
        if pid not in persisted and run_id is None:
            continue
        info = persisted.get(pid, {})
        share_url = info.get("share_url")
        log_path = info.get("log_path")
        started_at = info.get("started_at")
        iid = gradio_manager.register_existing(
            pid=pid, gpu=gpu, checkpoint_path=lora_ckpt,
            checkpoint_name=Path(lora_ckpt).name, title=title, run_id=run_id,
            share_url=share_url, log_path=log_path, started_at=started_at,
        )
        if iid:
            print(f"[recover] Registered orphaned Gradio PID {pid} on GPU {gpu}: {title or Path(lora_ckpt).name} share_url={share_url} log={log_path}")
            recovered += 1
    return recovered


class TrainingMonitor:
    """Monitors training processes and restarts on crash."""

    MAX_RESTART_ATTEMPTS = 3
    RESTART_COOLDOWN = 120  # seconds — don't restart if last restart was < this long ago

    def __init__(self, registry, gradio_mgr):
        self._registry = registry
        self._gradio_mgr = gradio_mgr
        self._restart_times = {}  # run_id -> [timestamp, ...]

    def _can_restart(self, run_id):
        """Check if we've restarted too many times recently."""
        now = time.time()
        times = self._restart_times.get(run_id, [])
        # Prune old entries
        times = [t for t in times if now - t < self.RESTART_COOLDOWN * self.MAX_RESTART_ATTEMPTS]
        self._restart_times[run_id] = times
        # Check cooldown on most recent restart
        if times and now - times[-1] < self.RESTART_COOLDOWN:
            return False
        # Check max attempts in window
        if len(times) >= self.MAX_RESTART_ATTEMPTS:
            return False
        return True

    def _record_restart(self, run_id):
        self._restart_times.setdefault(run_id, []).append(time.time())

    def monitor_loop(self):
        while True:
            time.sleep(15)
            try:
                # Iterate ALL runs with active status
                for run in self._registry.list_runs():
                    run_status = run.get("status")
                    if run_status not in ("training", "paused"):
                        continue
                    pid = run.get("pid")
                    restart_cmd = run.get("restart_cmd")
                    if not pid:
                        continue
                    proc_state = _detect_process_state(pid)
                    if proc_state != "dead":
                        continue
                    # Re-read from registry to avoid acting on stale data
                    run_id = run["id"]
                    fresh_run = self._registry.get_run(run_id)
                    if not fresh_run or fresh_run.get("status") not in ("training", "paused"):
                        continue
                    # Paused runs that died — just mark killed, don't restart
                    if fresh_run.get("status") == "paused":
                        hint = _diagnose_quick_kill(fresh_run)
                        if hint:
                            self._registry.update_run(run_id, status="killed", kill_hint=hint)
                        else:
                            self._registry.update_run(run_id, status="killed")
                        _free_gpu_memory(fresh_run.get("gpu"))
                        print(f"[monitor] Paused run {run_id} PID {pid} died — marking killed"
                              + (f" ({hint})" if hint else ""))
                        continue
                    # PID died — check if completed
                    effective_step, max_steps = _parse_latest_step(fresh_run)
                    gpu = fresh_run.get("gpu")
                    if effective_step is not None and effective_step >= max_steps - 1:
                        self._registry.update_run(run_id, status="completed")
                        _free_gpu_memory(gpu)
                        print(f"[monitor] Run {run_id} completed (step {effective_step}/{max_steps})")
                        continue
                    # Check log tail for OOM before attempting restart
                    is_oom = False
                    log_path = fresh_run.get("log_path", "")
                    if log_path:
                        try:
                            with open(log_path, "rb") as f:
                                f.seek(max(0, f.seek(0, 2) - 8192))
                                tail = f.read().decode("utf-8", errors="replace")
                            if "OutOfMemoryError" in tail or "CUDA out of memory" in tail:
                                is_oom = True
                        except Exception:
                            pass

                    if is_oom:
                        self._registry.update_run(run_id, status="error",
                                                  error="OOM — not enough GPU memory")
                        _free_gpu_memory(gpu)
                        print(f"[monitor] Run {run_id} OOM on GPU {gpu} — not restarting")
                        continue

                    # Crashed — attempt restart if we have a restart_cmd
                    if restart_cmd is None:
                        hint = _diagnose_quick_kill(fresh_run)
                        if hint:
                            self._registry.update_run(run_id, status="killed", kill_hint=hint)
                        else:
                            self._registry.update_run(run_id, status="killed")
                        _free_gpu_memory(gpu)
                        print(f"[monitor] Run {run_id} PID {pid} died, no restart_cmd — marking killed"
                              + (f" ({hint})" if hint else ""))
                        continue
                    # Cooldown: don't restart-loop if process keeps crashing
                    if not self._can_restart(run_id):
                        self._registry.update_run(run_id, status="error",
                                                  error=f"Crashed {self.MAX_RESTART_ATTEMPTS} times")
                        _free_gpu_memory(gpu)
                        print(f"[monitor] Run {run_id} crashed too many times — marking error")
                        continue
                    restart_count = len(self._restart_times.get(run_id, [])) + 1
                    gpu = run.get("gpu")
                    print(f"[monitor] Training PID {pid} died — restarting run {run_id} (attempt {restart_count}/{self.MAX_RESTART_ATTEMPTS})")
                    self._registry.update_run(run_id, restart_count=restart_count)
                    # Free the GPU if Gradio is on it
                    if gpu is not None:
                        self._gradio_mgr.stop_by_gpu(int(gpu))
                        time.sleep(2)
                    # Ensure gradient clipping is present (older runs may lack it)
                    if "--gradient-clip-val" not in restart_cmd:
                        restart_cmd += "     --gradient-clip-val 1.0"
                    # Restart training with tee to log file
                    gpu_env = f"CUDA_VISIBLE_DEVICES={gpu} " if gpu is not None else ""
                    backend_env = _backend_env_for_model(fresh_run.get("base_model"))
                    demo_dir = fresh_run.get("demo_source_dir", str(RUNS_DIR))
                    os.makedirs(demo_dir, exist_ok=True)
                    log_env = f"UNDERFIT_LOG_PATH={shlex.quote(log_path)} "
                    bash_err_path = log_path + ".bash.err"
                    try:
                        bash_err_f = open(bash_err_path, "wb")
                    except OSError:
                        bash_err_f = subprocess.DEVNULL
                    launch_cmd = f"source {VENV_ACTIVATE} && cd {shlex.quote(demo_dir)} && PYTHONUNBUFFERED=1 {log_env}{backend_env}{gpu_env}{restart_cmd} 2>&1 | tee -a {shlex.quote(log_path)}"
                    proc = subprocess.Popen(
                        ["bash", "-c", launch_cmd],
                        stdout=subprocess.DEVNULL,
                        stderr=bash_err_f,
                        preexec_fn=os.setsid,
                    )
                    self._registry.update_run(run_id, pid=proc.pid)
                    self._record_restart(run_id)
                    print(f"[monitor] Restarted run {run_id} as PID {proc.pid}")
            except Exception as e:
                print(f"[monitor] Error: {e}")


training_monitor = TrainingMonitor(registry, gradio_manager)


class EncodingMonitor:
    """Monitors encoding processes and updates dataset status."""

    def __init__(self, ds_registry):
        self._registry = ds_registry

    def monitor_loop(self):
        while True:
            time.sleep(5)
            try:
                for ds in self._registry.list_datasets():
                    if ds["status"] != "encoding":
                        continue
                    pid = ds.get("encoding_pid")
                    ds_id = ds["id"]
                    proc_state = _detect_process_state(pid)

                    if proc_state != "dead":
                        # Still running — parse log for progress
                        log_path = ds.get("log_path", "")
                        if log_path:
                            tail = _read_log_tail(log_path, max_bytes=4096)
                            if tail:
                                progress_re = re.compile(r"\[(\d+)/(\d+)\]")
                                matches = progress_re.findall(tail)
                                if matches:
                                    total = int(matches[-1][1])
                                    encoded = len(set(int(n) for n, _ in matches))
                                    self._registry.update_dataset(ds_id,
                                        encoding_progress={"encoded": encoded, "skipped": 0, "errors": 0, "total": total})
                    else:
                        # Process dead — check if completed
                        latent_dir = Path(ds.get("latent_dir", ""))
                        details_path = latent_dir / "details.json"
                        if details_path.exists():
                            try:
                                with open(details_path) as f:
                                    details = json.load(f)
                                # 1 source audio file → 1 latent .npy. Gate on num_files > 0:
                                # encoder crashed mid-run still writes details.json sometimes,
                                # so we need an actual-output check before saying "ready".
                                num_files = sum(1 for _ in latent_dir.rglob("*.npy"))
                                if num_files == 0:
                                    self._registry.remove_dataset(ds_id)
                                    print(f"[encoding_monitor] Dataset '{ds_id}' removed: "
                                          f"encoding finished but produced 0 latents — check the encode log.")
                                else:
                                    total = ds.get("encoding_progress", {}).get("total", num_files)
                                    self._registry.update_dataset(ds_id,
                                        status="ready",
                                        encoding_pid=None,
                                        num_files=num_files,
                                        latent_dim=details.get("latent_dim", 64),
                                        sample_rate=details.get("sample_rate", 44100),
                                        details=details,
                                        encoding_progress={"encoded": num_files, "skipped": 0, "errors": 0, "total": total},
                                    )
                                    print(f"[encoding_monitor] Dataset '{ds_id}' completed: {num_files} files")
                                    # Regenerate GT if missing (normally already created at dataset creation)
                                    if not ds.get("ground_truth"):
                                        _ds_files = ds.get("dataset_files", [])
                                        if _ds_files:
                                            gt_list, gt_prompts = _generate_dataset_ground_truth(_ds_files, ds.get("name", ds_id), model=ds.get("model", "sa3-medium"))
                                            if gt_list:
                                                self._registry.update_dataset(ds_id, ground_truth=gt_list, demo_prompts=gt_prompts)
                            except Exception as e:
                                # Failed mid-encode → drop the record so the name slot is freed.
                                self._registry.remove_dataset(ds_id)
                                print(f"[encoding_monitor] Dataset '{ds_id}' removed (error reading details: {e})")
                        else:
                            # Encoder crashed before producing details.json → drop the record.
                            self._registry.remove_dataset(ds_id)
                            print(f"[encoding_monitor] Dataset '{ds_id}' removed (encoding crashed before completion)")
                        # Free GPU memory
                        for gpu in ds.get("encoding_gpus", []):
                            _free_gpu_memory(gpu)
            except Exception as e:
                print(f"[encoding_monitor] Error: {e}")


encoding_monitor = EncodingMonitor(datasets_registry)


def _validate_datasets_on_startup():
    """Background validation: check ready datasets still have valid latent dirs."""
    for ds in datasets_registry.list_datasets():
        if ds["status"] == "ready":
            latent_dir = Path(ds.get("latent_dir", ""))
            details_path = latent_dir / "details.json"
            if not latent_dir.exists() or not details_path.exists():
                datasets_registry.update_dataset(ds["id"], status="error")
                print(f"[startup] Dataset '{ds['id']}' latent dir missing — marked error")
            else:
                # Re-count .npy files on disk. A "ready" dataset with 0 files
                # is a zombie (failed encode that got marked ready by old code) —
                # mark it error so it's clearly broken and gets filtered out.
                num_files = sum(1 for _ in latent_dir.rglob("*.npy"))
                if num_files == 0:
                    datasets_registry.update_dataset(ds["id"], status="error",
                                                     num_files=0,
                                                     error="encoding produced 0 latents")
                    print(f"[startup] Dataset '{ds['id']}' has 0 .npy files — marked error")
                elif num_files != ds.get("num_files", 0):
                    datasets_registry.update_dataset(ds["id"], num_files=num_files)
        elif ds["status"] == "encoding":
            # Check if encoding PID is still alive
            pid = ds.get("encoding_pid")
            if _detect_process_state(pid) == "dead":
                latent_dir = Path(ds.get("latent_dir", ""))
                details_path = latent_dir / "details.json"
                if details_path.exists():
                    num_files = sum(1 for _ in latent_dir.rglob("*.npy"))
                    if num_files == 0:
                        datasets_registry.update_dataset(ds["id"], status="error",
                                                         encoding_pid=None, num_files=0,
                                                         error="encoding produced 0 latents")
                        print(f"[startup] Dataset '{ds['id']}' encoding crashed (0 latents) — marked error")
                    else:
                        datasets_registry.update_dataset(ds["id"], status="ready", encoding_pid=None, num_files=num_files)
                        print(f"[startup] Dataset '{ds['id']}' encoding finished (PID dead, details exists, {num_files} files)")
                else:
                    datasets_registry.update_dataset(ds["id"], status="error", encoding_pid=None)
                    print(f"[startup] Dataset '{ds['id']}' encoding PID dead — marked error")


# Per-run state: current global step from log parsing
_run_steps = {}
_run_steps_lock = threading.Lock()

# Per-run VRAM history: run_id -> [{step, used_mb}]
_vram_history = {}
_vram_history_lock = threading.Lock()
VRAM_HISTORY_MAX = 500

# --- Caches for /api/status performance ---
# Log history cache: run_id -> {log_sizes, byte_offsets, loss, grad_norm, lora_mag, lr, cumulative_offset}
_log_history_cache = {}
_log_history_cache_lock = threading.Lock()

# Hyperparams cache: run_id -> result (or False for "no hyperparams")
_hyperparams_cache = {}
_hyperparams_cache_lock = threading.Lock()

_HISTORY_RE = re.compile(r"Epoch (\d+):\s+\d+%.*?(\d+)/(\d+).*?train/loss=([\d.eE+-]+|nan[\d.]*|inf[\d.]*)")
_LR_RE = re.compile(r"train/lr=([\d.eE+-]+|nan[\d.]*|inf[\d.]*)")
_GN_RE = re.compile(r"train/grad_norm=([\d.eE+-]+|nan[\d.]*|inf[\d.]*)")
_LM_RE = re.compile(r"train/lora_magnitude=([\d.eE+-]+|nan[\d.]*|inf[\d.]*)")
# New tqdm format includes the authoritative global step as a prefix.
_STEP_PREFIX_RE = re.compile(r"Step (\d+),")


def _safe_float(s):
    """Parse a metric value string, handling nan.0, inf.0, etc. Returns None for NaN/inf."""
    if s.startswith("nan"):
        return None
    if s.startswith("inf"):
        return None
    v = float(s)
    if v != v:  # NaN check
        return None
    return v


def _parse_log_lines(text, effective_offset, prev_loss=None):
    """Parse log text for history data points. Returns (loss, gn, lm, lr, max_raw_step)."""
    file_loss, file_gn, file_lm, file_lr = [], [], [], []
    file_max_raw = 0
    for line in text.splitlines():
        m = _HISTORY_RE.search(line)
        if not m:
            continue
        ep, s, t = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if s == 0:
            continue
        # Trust the explicit "Step N," prefix when present (new format);
        # otherwise derive step from epoch * total + step_in_epoch + offset
        # for legacy logs.
        step_m = _STEP_PREFIX_RE.search(line)
        if step_m:
            step = int(step_m.group(1))
            raw_step = step - effective_offset
        else:
            raw_step = ep * t + s - 1
            step = raw_step + effective_offset
        file_max_raw = max(file_max_raw, raw_step)
        # Detect run restart within this file (step goes backward)
        last_loss = prev_loss if not file_loss else file_loss
        if last_loss and raw_step < (last_loss[-1]["step"] - effective_offset) - t:
            file_loss, file_gn, file_lm, file_lr = [], [], [], []
        file_loss.append({"step": step, "loss": _safe_float(m.group(4))})
        lr_m = _LR_RE.search(line)
        if lr_m:
            file_lr.append({"step": step, "value": _safe_float(lr_m.group(1))})
        gn = _GN_RE.search(line)
        if gn:
            file_gn.append({"step": step, "value": _safe_float(gn.group(1))})
        lm = _LM_RE.search(line)
        if lm:
            file_lm.append({"step": step, "value": _safe_float(lm.group(1))})
    return file_loss, file_gn, file_lm, file_lr, file_max_raw


def _parse_log_history_cached(run_id, all_logs, step_offset):
    """Parse log files for history with caching. Returns (loss, grad_norm, lora_mag, lr) lists."""
    # Build current file signature: [(path_str, size), ...]
    current_sig = []
    for lf in all_logs:
        try:
            current_sig.append((str(lf), lf.stat().st_size))
        except OSError:
            current_sig.append((str(lf), -1))

    with _log_history_cache_lock:
        cached = _log_history_cache.get(run_id)

    if cached and cached["step_offset"] == step_offset:
        cached_sig = cached["sig"]
        # Check if all files are unchanged (fast path for completed/killed runs)
        if cached_sig == current_sig:
            return cached["loss"], cached["grad_norm"], cached["lora_mag"], cached["lr"]

        # Check if only the last file grew (active training incremental parse)
        if (len(cached_sig) == len(current_sig) and
                all(cached_sig[i] == current_sig[i] for i in range(len(current_sig) - 1)) and
                cached_sig[-1][0] == current_sig[-1][0] and
                current_sig[-1][1] > cached_sig[-1][1]):
            # Incremental: read only new bytes from last file
            last_file = all_logs[-1]
            old_size = cached_sig[-1][1]
            with open(last_file, "rb") as f:
                f.seek(old_size)
                new_bytes = f.read()
            # Find line boundary — the saved offset may be mid-line
            nl = new_bytes.find(b"\n")
            if nl >= 0:
                new_text = new_bytes[nl + 1:].decode("utf-8", errors="replace")
            else:
                new_text = new_bytes.decode("utf-8", errors="replace")
            effective_offset = step_offset  # last file uses step_offset
            new_loss, new_gn, new_lm, new_lr, new_max_raw = _parse_log_lines(
                new_text, effective_offset, prev_loss=cached["loss"])
            loss = cached["loss"] + new_loss
            grad_norm = cached["grad_norm"] + new_gn
            lora_mag = cached["lora_mag"] + new_lm
            lr = cached["lr"] + new_lr
            with _log_history_cache_lock:
                _log_history_cache[run_id] = {
                    "sig": current_sig, "step_offset": step_offset,
                    "loss": loss, "grad_norm": grad_norm, "lora_mag": lora_mag, "lr": lr,
                }
            return loss, grad_norm, lora_mag, lr

    # Full parse (cold start or signature mismatch)
    loss_history, grad_norm_history, lora_mag_history, lr_history = [], [], [], []
    cumulative_offset = 0
    for log_idx, log_file in enumerate(all_logs):
        is_last = (log_idx == len(all_logs) - 1)
        effective_offset = step_offset if is_last else cumulative_offset
        if is_last and step_offset < cumulative_offset:
            loss_history = [p for p in loss_history if p["step"] < step_offset]
            grad_norm_history = [p for p in grad_norm_history if p["step"] < step_offset]
            lora_mag_history = [p for p in lora_mag_history if p["step"] < step_offset]
            lr_history = [p for p in lr_history if p["step"] < step_offset]
        with open(log_file, "r", errors="replace") as f:
            text = f.read()
        fl, fg, fm, flr, fmax = _parse_log_lines(text, effective_offset, prev_loss=loss_history)
        loss_history.extend(fl)
        grad_norm_history.extend(fg)
        lora_mag_history.extend(fm)
        lr_history.extend(flr)
        if fmax > 0:
            cumulative_offset += fmax

    with _log_history_cache_lock:
        _log_history_cache[run_id] = {
            "sig": current_sig, "step_offset": step_offset,
            "loss": loss_history, "grad_norm": grad_norm_history,
            "lora_mag": lora_mag_history, "lr": lr_history,
        }
    return loss_history, grad_norm_history, lora_mag_history, lr_history


# Per-GPU VRAM history: gpu_index -> [{t: epoch_seconds, value: used_mb}]
_gpu_vram_history = {}
_gpu_vram_history_lock = threading.Lock()
GPU_VRAM_HISTORY_SECS = 3600  # 1 hour

# Per-GPU compute capability (e.g. "7.5" Turing, "7.0" Volta, "8.0" Ampere).
# Cached at module level: never changes during a session, so we query nvidia-smi
# once on first request and reuse the result. Frontend uses this to pick a
# sensible demo preset (fewer demos on older GPUs that lack flex_attention).
_gpu_compute_caps: dict | None = None


def _query_gpu_compute_caps() -> dict:
    global _gpu_compute_caps
    if _gpu_compute_caps is not None:
        return _gpu_compute_caps
    caps = {}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,compute_cap",
             "--format=csv,noheader,nounits"],
            timeout=10, stderr=subprocess.DEVNULL,
        ).decode().strip()
        for line in out.split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    caps[int(parts[0])] = parts[1]
                except ValueError:
                    pass
    except Exception:
        pass
    _gpu_compute_caps = caps
    return caps

# Gradio VRAM estimation
GRADIO_VRAM_FILE = STATE_FILES_DIR / "gradio_vram_estimate.json"
_gradio_vram_lock = threading.Lock()
_gradio_vram = {"load_mb": 10000, "peak_mb": 12000, "n_samples": 0}
_gradio_vram_baselines = {}  # instance_id -> {"gpu": int, "before_mb": int, "measured": bool}


def _load_gradio_vram_estimate():
    global _gradio_vram
    if GRADIO_VRAM_FILE.exists():
        try:
            with open(GRADIO_VRAM_FILE) as f:
                _gradio_vram.update(json.load(f))
        except Exception:
            pass


def _save_gradio_vram_estimate():
    try:
        _atomic_write_json(GRADIO_VRAM_FILE, _gradio_vram)
    except Exception:
        pass


def _record_gradio_baseline(instance_id, gpu):
    """Record VRAM baseline before Gradio launch for estimation."""
    gpu_mem = _query_gpu_mem()
    if gpu in gpu_mem:
        with _gradio_vram_lock:
            _gradio_vram_baselines[instance_id] = {
                "gpu": gpu, "before_mb": gpu_mem[gpu], "measured": False,
            }


def _update_gradio_vram_estimates():
    """Check ready Gradio instances, measure VRAM delta, update estimates."""
    gpu_mem = _query_gpu_mem()
    if not gpu_mem:
        return
    instances = {i["id"]: i for i in gradio_manager.list_instances()}
    with _gradio_vram_lock:
        to_remove = []
        for iid, bl in _gradio_vram_baselines.items():
            inst = instances.get(iid)
            if not inst:
                to_remove.append(iid)
                continue
            gpu = bl["gpu"]
            if gpu not in gpu_mem:
                continue
            current_used = gpu_mem[gpu]
            delta = current_used - bl["before_mb"]
            # Measure model-load cost when instance first becomes ready
            if inst["status"] == "ready" and not bl["measured"]:
                if delta > 1000:  # Sanity check: model should use >1GB
                    n = _gradio_vram["n_samples"]
                    _gradio_vram["load_mb"] = int(
                        (_gradio_vram["load_mb"] * n + delta) / (n + 1)
                    )
                    _gradio_vram["n_samples"] = n + 1
                    bl["measured"] = True
                    _save_gradio_vram_estimate()
            # Track peak VRAM while running (captures inference spikes)
            if inst["status"] == "ready" and delta > 0:
                if delta > _gradio_vram["peak_mb"] or _gradio_vram["n_samples"] <= 1:
                    _gradio_vram["peak_mb"] = max(_gradio_vram["peak_mb"], delta)
                    _save_gradio_vram_estimate()
            # Clean up stopped/errored instances
            if inst["status"] in ("stopped", "error"):
                to_remove.append(iid)
        for iid in to_remove:
            _gradio_vram_baselines.pop(iid, None)


def _get_gpu_count(force_refresh=False):
    """Return number of CUDA GPUs from nvidia-smi (cached after the first
    *successful* call). Returns 0 if nvidia-smi is missing or fails — the UI
    surfaces this as a warning in the GPU panel rather than pretending GPUs
    exist. The 15 s timeout is generous because the first nvidia-smi call on
    a fresh Colab VM can take 5–10 s while the driver initializes."""
    if force_refresh or not hasattr(_get_gpu_count, "_cache") or _get_gpu_count._cache == 0:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                timeout=15, stderr=subprocess.DEVNULL,
            ).decode().strip()
            _get_gpu_count._cache = len([l for l in out.split("\n") if l.strip()])
        except Exception:
            _get_gpu_count._cache = 0
    return _get_gpu_count._cache


def _query_gpu_mem():
    """Return dict of gpu_index -> used_mb from nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            timeout=5, stderr=subprocess.DEVNULL,
        ).decode().strip()
        result = {}
        for line in out.split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                result[int(parts[0])] = int(parts[1])
        return result
    except Exception:
        return {}


def vram_sampler():
    """Background thread: sample VRAM every 10s for each GPU and active training run."""
    _cycle = 0
    while True:
        time.sleep(10)
        _cycle += 1
        try:
            # Recover orphaned Gradio instances every ~60s
            if _cycle % 6 == 0:
                _recover_orphaned_gradios()
            # Audit Gradio ports for reassignment every ~60s
            if _cycle % 6 == 3:
                _audit_gradio_ports()
            # Try to discover share URLs for instances missing them every ~30s
            if _cycle % 3 == 0:
                _discover_missing_share_urls()
            gpu_mem = _query_gpu_mem()
            if not gpu_mem:
                continue

            # Per-GPU VRAM history (all GPUs, time-based)
            now = time.time()
            cutoff = now - GPU_VRAM_HISTORY_SECS
            with _gpu_vram_history_lock:
                for gpu_idx, used_mb in gpu_mem.items():
                    hist = _gpu_vram_history.setdefault(gpu_idx, [])
                    hist.append({"t": now, "value": used_mb})
                    # Trim entries older than 1 hour
                    while hist and hist[0]["t"] < cutoff:
                        hist.pop(0)

            # Per-run VRAM history (training runs only, step-based)
            for r in registry.list_runs():
                gpu = r.get("gpu")
                if gpu is None:
                    continue
                gpu = int(gpu)
                if gpu not in gpu_mem:
                    continue
                pid = r.get("pid")
                if not pid:
                    continue
                # Check process alive
                try:
                    os.kill(pid, 0)
                except (ProcessLookupError, PermissionError, TypeError):
                    continue
                run_id = r["id"]
                with _run_steps_lock:
                    step = _run_steps.get(run_id, 0)
                if step <= 0:
                    continue
                used_mb = gpu_mem[gpu]
                with _vram_history_lock:
                    hist = _vram_history.setdefault(run_id, [])
                    # Avoid duplicate steps
                    if not hist or hist[-1]["step"] != step:
                        hist.append({"step": step, "value": used_mb})
                        if len(hist) > VRAM_HISTORY_MAX:
                            _vram_history[run_id] = hist[-VRAM_HISTORY_MAX:]
            # Update Gradio VRAM estimates
            _update_gradio_vram_estimates()
        except Exception as e:
            print(f"[vram_sampler] Error: {e}")


def _extract_hyperparams_cached(run):
    """Cached wrapper — hyperparams never change for a given run."""
    run_id = run.get("id", "")
    with _hyperparams_cache_lock:
        if run_id in _hyperparams_cache:
            cached = _hyperparams_cache[run_id]
            return cached if cached else None
    result = _extract_hyperparams(run)
    with _hyperparams_cache_lock:
        _hyperparams_cache[run_id] = result if result else False
    return result


def _extract_hyperparams(run):
    """Extract training hyperparameters from model config and restart_cmd."""
    result = {}
    # Base model — stored directly on run record
    if run.get("base_model"):
        result["base_model"] = run["base_model"]
    restart_cmd = run.get("restart_cmd") or ""

    # Parse --model-config path from restart_cmd
    m = re.search(r"--model-config\s+(\S+)", restart_cmd)
    model_config_path = m.group(1) if m else None

    # Fallback: look for _model_resume.json or _model.json in runs dir
    if not model_config_path:
        run_id = run.get("id", "")
        if run_id:
            runs_dir = RUNS_DIR
            resume_path = runs_dir / f"{run_id}_model_resume.json"
            orig_path = runs_dir / f"{run_id}_model.json"
            if resume_path.exists():
                model_config_path = str(resume_path)
            elif orig_path.exists():
                model_config_path = str(orig_path)

    # Parse --batch-size from restart_cmd
    m = re.search(r"--batch-size\s+(\d+)", restart_cmd)
    if m:
        result["batch_size"] = int(m.group(1))

    # Parse --precision from restart_cmd
    m = re.search(r"--precision\s+(\S+)", restart_cmd)
    if m:
        result["precision"] = m.group(1)

    # Parse --checkpoint-every from restart_cmd
    m = re.search(r"--checkpoint-every\s+(\d+)", restart_cmd)
    if m:
        result["checkpoint_every"] = int(m.group(1))

    # Read model config if available
    if model_config_path:
        config_path = Path(model_config_path)
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                training = cfg.get("training", {})
                lora_cfg = training.get("lora_config", {})
                result["lora_rank"] = lora_cfg.get("rank")
                result["lora_alpha"] = lora_cfg.get("alpha", lora_cfg.get("rank"))
                result["lora_type"] = lora_cfg.get("adapter_type", "lora")
                lora_include = lora_cfg.get("include")
                lora_exclude = lora_cfg.get("exclude")
                if lora_include:
                    result["lora_include"] = lora_include
                if lora_exclude:
                    result["lora_exclude"] = lora_exclude

                demo_config = training.get("demo", {})
                de = demo_config.get("demo_every")
                if de is not None:
                    result["demo_every"] = de

                opt = training.get("optimizer_configs", {}).get("diffusion", {}).get("optimizer", {})
                lr = opt.get("config", {}).get("lr")
                if lr is not None:
                    result["lr"] = lr

                result["model_type"] = cfg.get("model_type", "")
                result["timestep_sampler"] = training.get("timestep_sampler", "")
                result["pre_encoded"] = training.get("pre_encoded", False)
                if training.get("base_precision"):
                    result["base_precision"] = training["base_precision"]
            except Exception:
                pass

    # Read crop mode from dataset config — prefer _dataset_resume.json so the
    # display reflects the most recent resume's overrides.
    run_id = run.get("id", "")
    if run_id:
        for ds_cfg_name in [f"{run_id}_dataset_resume.json", f"{run_id}_dataset.json"]:
            ds_cfg_path = RUNS_DIR / ds_cfg_name
            if ds_cfg_path.exists():
                try:
                    with open(ds_cfg_path) as f:
                        ds_cfg = json.load(f)
                    result["crop_mode"] = "random" if ds_cfg.get("random_crop", False) else "start"
                    lcl = ds_cfg.get("latent_crop_length")
                    if lcl:
                        result["latent_crop_length"] = lcl
                except Exception:
                    pass
                break

    # Dataset name — prefer the run's dataset_id (authoritative after swaps)
    ds_id = run.get("dataset_id")
    if ds_id:
        result["dataset_id"] = ds_id
        ds_rec = datasets_registry.get_dataset(ds_id)
        if ds_rec:
            result["dataset_name"] = ds_rec["name"]
        else:
            result["dataset_name"] = ds_id  # fallback to id

    # Try to count dataset files from the current --dataset-config in restart_cmd
    ds_config_match = re.search(r"--dataset-config\s+(\S+)", restart_cmd)
    ds_config_path = ds_config_match.group(1) if ds_config_match else None
    if not ds_config_path and model_config_path:
        # Legacy fallback: derive from model config path
        base_config_path = re.sub(r"_resume\.json$", ".json", model_config_path)
        ds_config_path = base_config_path.replace("_model.json", "_dataset.json")
    if ds_config_path:
        dp = Path(ds_config_path)
        if dp.exists():
            try:
                with open(dp) as f:
                    dcfg = json.load(f)
                datasets = dcfg.get("datasets", [])
                if not ds_id and datasets:
                    result["dataset_name"] = datasets[0].get("id", "")
                total_files = 0
                for ds in datasets:
                    ds_path = Path(ds.get("path", ""))
                    if ds_path.exists():
                        total_files += sum(1 for _ in ds_path.rglob("*.npy"))
                if total_files > 0:
                    result["dataset_size"] = total_files
                # Extract prompt_config for display
                pc = dcfg.get("prompt_config")
                if pc:
                    result["prompt_config"] = pc
            except Exception:
                pass

    return result if result else None


SPEC_BANDS = [
    (0, 200, (1.0, 0.0, 0.0)),      # Bass -> Red
    (200, 1500, (0.0, 1.0, 0.0)),   # Mid  -> Green
    (1500, 16000, (0.0, 0.0, 1.0)), # High -> Blue
]
SPEC_W, SPEC_H = 300, 60
_spec_pool = ThreadPoolExecutor(max_workers=4)

# Pre-compute band colors as (3, 3) array for vectorized multiply
_BAND_COLORS = np.array([c for _, _, c in SPEC_BANDS], dtype=np.float32)


# Slaney mel scale (linear < 1 kHz, log above) — matches librosa default.
_F_SP = 200.0 / 3
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = np.log(6.4) / 27.0


def _hz_to_mel(hz):
    hz = np.asarray(hz, dtype=np.float64)
    # np.where evaluates both branches — mask the log input so the linear-region
    # values don't trigger log(0) warnings (they're discarded anyway).
    log_term = np.log(np.maximum(hz, _MIN_LOG_HZ) / _MIN_LOG_HZ) / _LOGSTEP
    return np.where(hz >= _MIN_LOG_HZ, _MIN_LOG_MEL + log_term, hz / _F_SP)


def _mel_to_hz(mels):
    mels = np.asarray(mels, dtype=np.float64)
    return np.where(mels >= _MIN_LOG_MEL,
                    _MIN_LOG_HZ * np.exp(_LOGSTEP * (mels - _MIN_LOG_MEL)),
                    _F_SP * mels)


def _mel_frequencies(n_mels, fmax, fmin=0.0):
    return _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels))


@lru_cache(maxsize=8)
def _mel_filterbank(n_mels, n_fft, sr, fmax):
    pts = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(fmax), n_mels + 2))
    fft_f = np.linspace(0, sr / 2, n_fft // 2 + 1)
    filt = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        lo, ce, hi = pts[i], pts[i + 1], pts[i + 2]
        left = (fft_f - lo) / max(ce - lo, 1e-10)
        right = (hi - fft_f) / max(hi - ce, 1e-10)
        filt[i] = np.maximum(0, np.minimum(left, right))
    # Slaney area-normalization (matches librosa norm='slaney' default)
    enorm = (2.0 / (pts[2:n_mels + 2] - pts[0:n_mels])).astype(np.float32)
    filt *= enorm[:, None]
    return torch.from_numpy(filt)


def _melspectrogram(y_ch, sr, n_mels=30, fmax=16000, hop_length=2048, n_fft=2048):
    y_t = torch.from_numpy(np.ascontiguousarray(y_ch)).float()
    win = torch.hann_window(n_fft)
    spec = torch.stft(y_t, n_fft=n_fft, hop_length=hop_length, window=win,
                      center=True, return_complex=True, pad_mode='reflect')
    return (_mel_filterbank(n_mels, n_fft, sr, fmax) @ spec.abs().square()).numpy()


def _power_to_db(S, top_db=80.0):
    log_spec = 10.0 * np.log10(np.maximum(S, 1e-10))
    return np.maximum(log_spec - log_spec.max(), -top_db)


def _load_audio(path, target_sr=32000):
    """Load + resample to target_sr. Returns (channels, samples) like librosa(mono=False)."""
    y, sr = sf.read(str(path), dtype='float32', always_2d=False)
    if y.ndim == 2:
        y = y.T  # soundfile gives (n, c); we want (c, n)
    if sr != target_sr:
        y_t = torch.from_numpy(np.ascontiguousarray(y))
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(0)
        new_len = int(round(y_t.shape[-1] * target_sr / sr))
        y_t = torch.nn.functional.interpolate(
            y_t.unsqueeze(0), size=new_len, mode='linear', align_corners=False).squeeze(0)
        y = y_t.numpy()
    return y, target_sr


def _mel_channel(y_ch, sr, n_mels=30):
    """Compute dB-scaled mel + band-tinted RGB for one channel.

    Single mel spectrogram (no redundant STFT), hop=2048 for ~4x fewer frames.
    """
    S = _melspectrogram(y_ch, sr, n_mels=n_mels, fmax=16000, hop_length=2048)

    # dB-scale with gamma for visual contrast
    S_db = _power_to_db(S)
    np.clip(S_db, -60, 0, out=S_db)
    S_db += 60.0
    S_db /= 60.0
    np.power(S_db, 0.6, out=S_db)

    # Band colors from mel bin frequencies (no separate STFT needed)
    mel_f = _mel_frequencies(n_mels, fmax=16000)
    n_frames = S.shape[1]
    # Compute per-band normalized energy, then mix into RGB
    band_norms = np.empty((3, n_frames), dtype=np.float32)
    for i, (flo, fhi, _) in enumerate(SPEC_BANDS):
        mask = (mel_f >= flo) & (mel_f < fhi)
        if mask.any():
            power = np.sum(S[mask], axis=0)
            db = 10.0 * np.log10(power + 1e-10)
            np.clip(db, -20, None, out=db)
            db -= -20
            mx = db.max()
            if mx > 0:
                db /= mx
            band_norms[i] = db
        else:
            band_norms[i] = 0.0

    # (n_frames, 3) = (n_frames, 3_bands) @ (3_bands, 3_rgb)
    rgb = band_norms.T @ _BAND_COLORS
    for c in range(3):
        mx = rgb[:, c].max()
        if mx > 0:
            rgb[:, c] /= mx

    return S_db, rgb


def generate_spectrogram(mp3_path, jpg_path):
    """Generate a 300x60 3-band tinted stereo mel spectrogram."""
    try:
        y, sr = _load_audio(mp3_path, target_sr=32000)
        if y.ndim == 1:
            y = np.stack([y, y])

        S_L, rgb_L = _mel_channel(y[0], sr)
        S_R, rgb_R = _mel_channel(y[1], sr)

        nf = min(S_L.shape[1], S_R.shape[1])
        S_L, S_R = S_L[:, :nf], S_R[:, :nf]
        rgb_L, rgb_R = rgb_L[:nf], rgb_R[:nf]
        nm = S_L.shape[0]

        S_L = S_L[::-1]  # L: flip so bass at bottom

        # Vectorized compositing — no Python loop over frames
        img = np.empty((nm * 2, nf, 3), dtype=np.float32)
        img[:nm] = S_L[:, :, np.newaxis] * rgb_L[np.newaxis, :, :]
        img[nm:] = S_R[:, :, np.newaxis] * rgb_R[np.newaxis, :, :]

        np.clip(img, 0, 1, out=img)
        img *= 255
        Image.fromarray(img.astype(np.uint8)).resize(
            (SPEC_W, SPEC_H), Image.LANCZOS).save(
            str(jpg_path), quality=60, optimize=True)
    except Exception as e:
        print(f"[spectrogram] Failed for {mp3_path}: {e}")


def _process_run_demos(run):
    """Process demos for a single run: copy from source dir, generate spectrograms."""
    demo_source = Path(run.get("demo_source_dir", ""))
    if not demo_source.exists():
        return
    run_id = run["id"]
    run_mi = _get_model_info(run.get("base_model"))
    out_base = AUDIO_DIR / "runs" / run_id
    out_base.mkdir(parents=True, exist_ok=True)

    # Only process demo files created after the run started (avoid
    # picking up stale demos from previous runs sharing the same dir)
    run_created = run.get("created_at", "")
    run_created_ts = 0.0
    if run_created:
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(run_created)
            run_created_ts = dt.timestamp()
        except Exception:
            pass

    # Copy individual demo files (demo_{i}_{step}.mp3 + .json sidecars)
    spec_futs = []
    for src in sorted(demo_source.glob("demo_*_*.mp3")):
        if run_created_ts and src.stat().st_mtime < run_created_ts:
            continue  # Pre-dates this run
        m = re.match(r"demo_(\d+)_(\d+)\.mp3", src.name)
        if not m:
            continue
        idx = int(m.group(1))
        step = int(m.group(2))
        step_dir = out_base / f"step_{step:08d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        dest = step_dir / f"demo_{idx}.mp3"
        src_size = src.stat().st_size
        dest_size = dest.stat().st_size if dest.exists() else 0
        if dest_size == 0 or dest_size != src_size:
            shutil.copy2(src, dest)
        # Generate spectrogram for copied clip
        jpg_dest = step_dir / f"demo_{idx}.jpg"
        if not jpg_dest.exists() or jpg_dest.stat().st_size == 0:
            spec_futs.append(_spec_pool.submit(generate_spectrogram, dest, jpg_dest))
        # Copy JSON sidecar if present
        json_src = src.with_suffix(".json")
        json_dest = step_dir / f"demo_{idx}.json"
        if json_src.exists() and not json_dest.exists():
            shutil.copy2(json_src, json_dest)
    # Named extra demos (e.g. demo_arc_00001000.mp3)
    for src in sorted(demo_source.glob("demo_arc_*.mp3")):
        if run_created_ts and src.stat().st_mtime < run_created_ts:
            continue
        m = re.match(r"demo_arc_(\d+)\.mp3", src.name)
        if not m:
            continue
        step = int(m.group(1))
        step_dir = out_base / f"step_{step:08d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        dest = step_dir / "demo_arc.mp3"
        if not dest.exists() or dest.stat().st_size == 0:
            shutil.copy2(src, dest)
        jpg_dest = step_dir / "demo_arc.jpg"
        if not jpg_dest.exists():
            spec_futs.append(_spec_pool.submit(generate_spectrogram, dest, jpg_dest))
        json_src = src.with_suffix(".json")
        json_dest = step_dir / "demo_arc.json"
        if json_src.exists() and not json_dest.exists():
            shutil.copy2(json_src, json_dest)

    for f in spec_futs:
        f.result()


def process_all_demos():
    """Iterate over all registered runs, split demos into per-run dirs."""
    # Generate spectrograms for ground truth clips (top-level and per-run subdirs)
    gt_dir = AUDIO_DIR / "ground_truth"
    if gt_dir.exists():
        for mp3 in gt_dir.rglob("*.mp3"):
            jpg = mp3.with_suffix(".jpg")
            if not jpg.exists():
                generate_spectrogram(mp3, jpg)

    for run in registry.list_runs():
        _process_run_demos(run)


def demo_watcher():
    while True:
        process_all_demos()
        time.sleep(30)


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)
        run_id = params.get("run_id", [None])[0]

        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(self._serve_index())
        elif path == "/favicon.ico":
            ico_path = Path(__file__).parent / "favicon.ico"
            if ico_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/x-icon")
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(ico_path.read_bytes())
            else:
                self.send_response(404)
                self.end_headers()
        elif path == "/api/models":
            self._json_response({"models": MODELS_UI_PAYLOAD})
        elif path == "/api/runs":
            self._json_response(self._get_runs())
        elif path == "/api/status":
            self._json_response(self._get_status(run_id))
        elif path == "/api/checkpoints":
            self._json_response(self._get_checkpoints(run_id))
        elif path == "/api/log_tail":
            client_file_size = int(params.get("file_size", [0])[0])
            self._json_response(self._get_log_tail(run_id, file_size=client_file_size))
        elif path == "/api/server_log":
            try:
                p = SERVER_LOG_FILE
                if not p.exists():
                    self._json_response({"log": "", "mtime": 0, "size": 0})
                    return
                # Tail the last ~128 KB to keep the response light; dashboard
                # logs can grow over a long-running session.
                MAX = 128 * 1024
                with open(p, "rb") as f:
                    f.seek(0, 2); size = f.tell()
                    f.seek(max(0, size - MAX))
                    raw = f.read()
                # Drop first partial line if we read from the middle
                if size > MAX:
                    nl = raw.find(b"\n")
                    if nl >= 0:
                        raw = raw[nl + 1:]
                text = raw.decode("utf-8", errors="replace")
                # Hide '[gradio: ...' lines — they have their own tab in the
                # Logs modal, no need to also show them in SERVER.
                text = "\n".join(
                    ln for ln in text.splitlines()
                    if not ln.lstrip().startswith("[gradio:")
                )
                self._json_response({
                    "log": text,
                    "mtime": p.stat().st_mtime,
                    "size": size,
                })
            except Exception as e:
                self._json_response({"error": str(e)}, status=500)
        elif path == "/api/clone_settings":
            self._json_response(self._get_clone_settings(run_id))
        elif path == "/api/loss_by_timestep":
            self._json_response(self._get_loss_by_timestep(run_id))
        elif path == "/api/demos":
            run = registry.get_run(run_id) if run_id else None
            # Process demos on-demand: explicit refresh OR active training run (throttled to 10s)
            force = "nocache" in params
            auto = run and run.get("status") in ("training", "demos")
            if auto and run_id:
                last = DashboardHandler._demo_process_times.get(run_id, 0)
                if time.time() - last < 10:
                    auto = False  # Too soon, skip
            if (force or auto) and run:
                DashboardHandler._demo_process_times[run_id] = time.time()
                try:
                    t = threading.Thread(target=_process_run_demos, args=(run,), daemon=True)
                    t.start()
                    t.join(timeout=30)
                    if t.is_alive():
                        print(f"[refresh-demos] Timeout processing demos for {run_id}")
                except Exception as e:
                    print(f"[refresh-demos] Error processing demos for {run_id}: {e}")
                with DashboardHandler._demo_cache_lock:
                    DashboardHandler._demo_cache.pop(run_id, None)
            elif "nocache" in params:
                with DashboardHandler._demo_cache_lock:
                    DashboardHandler._demo_cache.pop(run_id, None)
            self._json_response(self._get_demos(run_id))
        elif path == "/api/gradio":
            instances = gradio_manager.list_instances()
            for inst in instances:
                lp = inst.get("log_path")
                inst["log_mtime"] = os.path.getmtime(lp) if lp and os.path.exists(lp) else None
            self._json_response(instances)
        elif path.startswith("/api/gradio/") and path.endswith("/log"):
            # /api/gradio/{id}/log?tail=500
            instance_id = path.split("/")[3]
            tail = int(params.get("tail", [500])[0])
            inst = None
            for i in gradio_manager.list_instances():
                if i["id"] == instance_id:
                    inst = i
                    break
            if not inst:
                self._json_response({"error": "instance not found"}, status=404)
                return
            lp = inst.get("log_path")
            if not lp or not os.path.exists(lp):
                self._json_response({"content": "", "mtime": None})
                return
            try:
                with open(lp, "rb") as f:
                    raw = f.read()
                # Collapse \r (inline tqdm) and consecutive progress-bar lines
                lines = []
                for chunk in raw.split(b"\n"):
                    if b"\r" in chunk:
                        chunk = chunk.rsplit(b"\r", 1)[-1]
                    line = chunk.decode("utf-8", errors="replace")
                    if line:
                        lines.append(line)
                lines = _collapse_progress_lines(lines)
                content = "\n".join(lines[-tail:])
                mtime = os.path.getmtime(lp)
                self._json_response({"content": content, "mtime": mtime})
            except Exception as e:
                self._json_response({"error": str(e)}, status=500)
        elif path == "/api/gpu":
            self._json_response(self._get_gpu_info())
        elif path.startswith("/api/gpu/") and path.endswith("/history"):
            # /api/gpu/3/history
            gpu_idx = path.split("/")[3]
            try:
                gpu_idx = int(gpu_idx)
            except (ValueError, IndexError):
                self.send_error(400, "Invalid GPU index")
                return
            with _gpu_vram_history_lock:
                hist = list(_gpu_vram_history.get(gpu_idx, []))
            self._json_response(hist)
        elif path.startswith("/api/gpu/") and path.endswith("/processes"):
            # /api/gpu/3/processes
            gpu_idx = path.split("/")[3]
            try:
                gpu_idx = int(gpu_idx)
            except (ValueError, IndexError):
                self.send_error(400, "Invalid GPU index")
                return
            self._json_response(self._get_gpu_processes(gpu_idx))
        elif path == "/api/estimate_vram":
            params = parse_qs(parsed.query)
            base_model = params.get("model", ["sa3"])[0]
            batch_size = int(params.get("batch_size", ["8"])[0])
            lora_rank = int(params.get("rank", ["16"])[0])
            precision = params.get("precision", ["16-mixed"])[0]
            est_mb = estimate_training_vram_mb(base_model, batch_size, lora_rank, precision)
            self._json_response({"estimated_mb": est_mb})
        elif path == "/api/rare_tokens":
            token_file = DASHBOARD_DIR / "rare_tokens.json"
            if token_file.exists():
                with open(token_file) as f:
                    self._json_response(json.load(f))
            else:
                self._json_response([])
        elif path == "/api/datasets":
            # Strip dataset_files from response — it's only needed server-side
            # and can be 100s of KB per dataset.  Copy dicts so we don't
            # mutate the registry's internal data.
            ds_list = [dict(d) for d in datasets_registry.list_datasets()]
            for ds in ds_list:
                ds.pop("dataset_files", None)
            ds_list.sort(key=lambda d: d.get("created_at", ""), reverse=True)
            self._json_response(ds_list)
        elif path.startswith("/api/datasets/") and path.endswith("/progress"):
            ds_id = unquote(path.split("/")[3])
            self._json_response(self._get_encoding_progress(ds_id))
        elif path.startswith("/api/datasets/") and path.endswith("/files"):
            ds_id = unquote(path.split("/")[3])
            ds = datasets_registry.get_dataset(ds_id)
            if not ds:
                self._json_response({"error": "dataset not found"}, status=404)
                return
            # Try cached tag data first; invalidate if sidecars were added since
            cached = _load_tag_cache(ds_id)
            if cached:
                file_info, total_files, files_with_tags, files_with_json = cached
                # If cache shows no JSON sidecars, check if any exist now (autotagger ran)
                if files_with_json == 0 and file_info:
                    sample_path = Path(ds.get("input_dir", "")) / file_info[0]["relpath"]
                    if sample_path.with_suffix(".json").exists():
                        cached = None  # invalidate — re-scan below
            if not cached:
                input_dir = ds.get("input_dir", "")
                if input_dir and Path(input_dir).is_dir():
                    file_info, total_files, files_with_tags, files_with_json = self._scan_audio_tags(
                        input_dir, sample_size=None, tag_sample_size=None)
                    _save_tag_cache(ds_id, file_info, total_files, files_with_tags, files_with_json)
                else:
                    file_info, total_files, files_with_tags, files_with_json = [], 0, 0, 0
            self._json_response({
                "dataset": ds,
                "files": file_info,
                "total_files": total_files,
                "files_with_tags": files_with_tags,
                "files_with_json": files_with_json,
            })
        elif path.startswith("/api/datasets/") and "/audio/" in path:
            # Serve audio file from dataset input_dir: /api/datasets/{id}/audio/{relpath}
            parts = path.split("/audio/", 1)
            ds_id = unquote(parts[0].split("/")[3])
            rel = unquote(parts[1]) if len(parts) > 1 else ""
            ds = datasets_registry.get_dataset(ds_id)
            if not ds or not rel:
                self.send_error(404)
                return
            fpath = Path(ds.get("input_dir", "")) / rel
            if not fpath.exists() or not fpath.is_file():
                self.send_error(404)
                return
            content_type = "audio/mpeg" if fpath.suffix.lower() in (".mp3",) else "audio/wav" if fpath.suffix.lower() in (".wav",) else "audio/flac"
            file_size = fpath.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            # Cache based on file size — immutable once non-empty, but don't cache 0-byte files
            if file_size > 0:
                self.send_header("Cache-Control", "public, max-age=86400")
            else:
                self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            with open(fpath, "rb") as f:
                self.wfile.write(f.read())
        elif path.startswith("/api/scan-audio/"):
            # Serve audio preview from an absolute path (for scan modal before dataset exists)
            abs_path = unquote(path[len("/api/scan-audio/"):])
            fpath = Path(abs_path)
            if not fpath.exists() or not fpath.is_file():
                self.send_error(404)
                return
            ext = fpath.suffix.lower()
            ct = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac",
                  ".ogg": "audio/ogg", ".aif": "audio/aiff", ".aiff": "audio/aiff",
                  ".m4a": "audio/mp4", ".aac": "audio/aac"}.get(ext, "audio/wav")
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(fpath.stat().st_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(fpath, "rb") as f:
                self.wfile.write(f.read())
        elif path.startswith("/audio/"):
            dl_name = params.get("dl", [None])[0]
            self._serve_audio(path[7:], dl_name=dl_name)
        elif path == "/api/audio_slice":
            self._serve_audio_slice(params)
        elif path == "/api/download":
            ckpt_path = params.get("path", [None])[0]
            self._serve_checkpoint_download(ckpt_path)
        elif path == "/gradio-clone":
            clone_path = DASHBOARD_DIR.parent / "gradio" / "index.html"
            if clone_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(clone_path.read_bytes())
            else:
                self.send_error(404, "gradio clone not found")
        elif not path.startswith("/api/"):
            # Serve index.html for /{run_name} deep-link URLs (SPA routing)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(self._serve_index())
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/runs":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            # Validate required fields
            run_id = body.get("id")
            if not run_id:
                self._json_response({"error": "missing id"}, status=400)
                return
            # Check for duplicate
            if registry.get_run(run_id):
                self._json_response({"error": "run already exists", "id": run_id}, status=409)
                return
            run = {
                "id": run_id,
                "display_name": body.get("display_name", run_id),
                "log_path": body.get("log_path", ""),
                "demo_source_dir": body.get("demo_source_dir", ""),
                "checkpoints_dir": body.get("checkpoints_dir", ""),
                "max_steps": body.get("max_steps", 20000),
                "active": True,
                "status": "training",
                "step_offset": 0,
                "created_at": body.get("created_at", datetime.now(timezone.utc).isoformat()),
                "pid": body.get("pid"),
                "gpu": body.get("gpu"),
                "restart_cmd": body.get("restart_cmd"),
            }
            registry.add_run(run)
            # Create run's demo output directory
            (AUDIO_DIR / "runs" / run_id).mkdir(parents=True, exist_ok=True)
            self._json_response(run, status=201)
        elif parsed.path == "/api/lora/validate_seed":
            self._handle_validate_seed_lora(parsed)
            return
        elif parsed.path == "/api/gradio":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            ckpt_path = body.get("checkpoint_path")
            gpu = body.get("gpu")
            if ckpt_path is None or gpu is None:
                self._json_response({"error": "missing checkpoint_path or gpu"}, status=400)
                return
            gpu = int(gpu)
            if gpu < 0 or gpu >= _get_gpu_count():
                self._json_response({"error": f"gpu must be 0-{_get_gpu_count()-1}"}, status=400)
                return
            instance_id, err = gradio_manager.launch(
                checkpoint_path=ckpt_path,
                gpu=gpu,
                run_id=body.get("run_id"),
                checkpoint_name=body.get("checkpoint_name"),
                title=body.get("title"),
                model_variant=body.get("model_variant"),
                verbose=bool(body.get("verbose", False)),
            )
            if err:
                self._json_response({"error": err}, status=409)
            else:
                self._json_response({"id": instance_id}, status=201)
        elif parsed.path == "/api/runs/new":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._handle_new_finetune(body)
        elif parsed.path == "/api/runs/adopt":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._handle_adopt(body)
        elif parsed.path == "/api/save_checkpoint":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._handle_save_checkpoint(body)
        elif parsed.path == "/api/kill_pid":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._handle_kill_pid(body)
        elif parsed.path == "/api/datasets/scan":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._handle_datasets_scan(body)
        elif parsed.path == "/api/datasets/encode":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._handle_datasets_encode(body)
        else:
            # Route: POST /api/datasets/{id}/{action}
            ds_m = re.match(r"^/api/datasets/([^/]+)/(stop|delete)$", parsed.path)
            if ds_m:
                ds_id = unquote(ds_m.group(1))
                action = ds_m.group(2)
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else {}
                if action == "stop":
                    self._handle_datasets_stop(ds_id)
                elif action == "delete":
                    self._handle_datasets_delete(ds_id, body)
                return
            # Route: POST /api/runs/{id}/{action}
            m = re.match(r"^/api/runs/([^/]+)/(pause|continue|kill|resume|delete)$", parsed.path)
            if m:
                run_id = unquote(m.group(1))
                action = m.group(2)
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else {}
                if action == "pause":
                    self._handle_pause(run_id)
                elif action == "continue":
                    self._handle_continue(run_id)
                elif action == "kill":
                    self._handle_kill(run_id)
                elif action == "resume":
                    self._handle_resume(run_id, body)
                elif action == "delete":
                    self._handle_delete(run_id, body)
            else:
                self.send_error(404)

    def _handle_validate_seed_lora(self, parsed):
        """Receive a raw .safetensors body, content-address-store it under
        SEED_LORAS_DIR, validate, return extracted config OR error+partial info.

        Frontend uploads as `application/octet-stream`; original filename comes
        through as a `filename` query param. Body is read in chunks to avoid
        loading the whole file into memory.
        """
        import hashlib
        from urllib.parse import parse_qs
        from underfit.utils.lora_validate import validate_lora_safetensors

        qs = parse_qs(parsed.query)
        orig_name = (qs.get("filename") or ["uploaded.safetensors"])[0]
        if not orig_name.lower().endswith(".safetensors"):
            self._json_response({
                "ok": False,
                "error": "only .safetensors files are accepted (no .ckpt / .pt — pickle is a security risk).",
            }, status=400)
            return
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            self._json_response({"ok": False, "error": "empty upload"}, status=400)
            return
        MAX = 500 * 1024 * 1024  # 500 MB cap — generous; typical LoRA <100 MB
        if length > MAX:
            self._json_response({
                "ok": False,
                "error": f"upload too large ({length / 1e6:.1f} MB > {MAX / 1e6:.0f} MB cap)",
            }, status=413)
            return

        SEED_LORAS_DIR.mkdir(parents=True, exist_ok=True)
        # Stream into a temp file while hashing; rename to content-addressed final.
        tmp = SEED_LORAS_DIR / f".upload-{os.getpid()}-{int(time.time()*1000)}.partial"
        h = hashlib.sha256()
        remaining = length
        try:
            with open(tmp, "wb") as f:
                while remaining > 0:
                    chunk = self.rfile.read(min(1 << 20, remaining))  # 1 MB blocks
                    if not chunk:
                        break
                    f.write(chunk)
                    h.update(chunk)
                    remaining -= len(chunk)
        except OSError as e:
            tmp.unlink(missing_ok=True)
            self._json_response({"ok": False, "error": f"upload write failed: {e}"}, status=500)
            return

        sha = h.hexdigest()[:16]
        final = SEED_LORAS_DIR / f"{sha}.safetensors"
        # Also keep the original name for display, side-by-side
        meta_path = SEED_LORAS_DIR / f"{sha}.json"
        try:
            if final.exists():
                tmp.unlink(missing_ok=True)
            else:
                tmp.rename(final)
        except OSError as e:
            tmp.unlink(missing_ok=True)
            self._json_response({"ok": False, "error": f"could not finalize file: {e}"}, status=500)
            return

        result = validate_lora_safetensors(final)
        # Always store the original filename next to the content-addressed file
        try:
            meta_path.write_text(json.dumps({"original_filename": orig_name, "sha": sha}, indent=2))
        except OSError:
            pass

        # ── Base-model resolution ────────────────────────────────────────
        # The seed's metadata may record which base model it was trained on
        # (e.g. base_model='sa3-medium'). Three outcomes:
        #   1. base_model present + known to this dashboard → return as-is.
        #   2. base_model present but NOT known → flag as 'unknown_known_name'.
        #   3. base_model missing → heuristic: layer name+dims must be a
        #      subset of the candidate model's lora_layer_template.
        seed_config = result.get("config") or {}
        seed_base = seed_config.get("base_model")
        known_models = list(MODELS_UI_PAYLOAD.keys())
        inferred_models = []
        if not seed_base:
            # Build full per-model layer sets and check subset match.
            # IMPORTANT: only compare 'model.*' keys. SA3 LoRAs may also
            # include 'conditioner.*' entries (text encoder, etc.) which
            # aren't enumerated in the registry's lora_layer_template — and
            # that's expected. The diffusion-model layer set is the
            # discriminative signal.
            seed_layers = result.get("layers") or []
            seed_fingerprint = {
                (l["name"], l["fan_in"], l["fan_out"])
                for l in seed_layers
                if l["name"].startswith("model.")
            }
            for mkey, minfo in MODELS_UI_PAYLOAD.items():
                tmpl = minfo.get("lora_layer_template") or {}
                ms = minfo.get("module_structure") or {}
                n_blocks = ms.get("n_blocks") or tmpl.get("n_blocks") or 0
                model_set = set()
                for entry in tmpl.get("prefix", []) or []:
                    model_set.add((entry["name"], entry["fi"], entry["fo"]))
                for entry in tmpl.get("suffix", []) or []:
                    model_set.add((entry["name"], entry["fi"], entry["fo"]))
                pb_prefix = tmpl.get("per_block_prefix", "")
                for i in range(int(n_blocks)):
                    for entry in tmpl.get("per_block", []) or []:
                        name = pb_prefix.replace("{i}", str(i)) + entry["suffix"]
                        model_set.add((name, entry["fi"], entry["fo"]))
                # LoRA fingerprint must be a SUBSET of the model's full layer set.
                if seed_fingerprint and seed_fingerprint.issubset(model_set):
                    inferred_models.append(mkey)
            # Also surface the seed's model.* layers for the popup (helps
            # debug when neither metadata nor heuristic give a match).
            result.setdefault("partial", {})["model_layer_count"] = len(seed_fingerprint)

        # Response shape
        response = {
            "ok": result["ok"],
            "path": str(final) if result["ok"] else None,
            "filename": orig_name,
            "config": seed_config,
            "partial": result["partial"],
            "error": result["error"],
            # base-model verdict — frontend uses these to decide which popup to show:
            #   base_model_in_seed:       what the seed's metadata claims (str or None)
            #   base_model_known:         True iff that name is in this dashboard's registry
            #   inferred_base_models:     candidate matches when no metadata (may be empty)
            "base_model_in_seed":    seed_base,
            "base_model_known":      bool(seed_base) and seed_base in known_models,
            "inferred_base_models":  inferred_models,
        }
        if not result["ok"]:
            response["path"] = None
        self._json_response(response, status=200)

    def _handle_new_finetune(self, body):
        gpu = body.get("gpu")
        if gpu is None:
            self._json_response({"error": "gpu is required"}, status=400)
            return
        gpu = int(gpu)
        raw_name = body.get("name", "").strip()
        name = _slugify(raw_name)
        if not name:
            self._json_response({"error": "name is required"}, status=400)
            return
        # Check for duplicate slug
        for r in registry.list_runs():
            existing_slug = _slugify(r.get("display_name", r["id"]))
            if existing_slug == name:
                self._json_response({"error": f"A run with name '{name}' already exists"}, status=409)
                return
        max_steps = int(body.get("max_steps", 20000))
        batch_size = int(body.get("batch_size", 8))
        base_model = body.get("base_model", "sa3-medium")
        lora_type = body.get("lora_type", "lora")  # lora, dora, bora, lora-xs
        rank = int(body.get("rank", 16))
        checkpoint_every = int(body.get("checkpoint_every", 1000))
        demo_every = int(body.get("demo_every", 1000))
        alpha = body.get("alpha")
        lora_include = body.get("lora_include", "").strip()
        lora_exclude = body.get("lora_exclude", "").strip()

        # Optional seed: user-uploaded .safetensors LoRA to start from. When
        # set, it overrides the lora_type/rank/alpha/include/exclude knobs
        # with the values baked into the seed's metadata, and we point the
        # training loop at the file via lora_ckpt_path. Step counter resets
        # to 0 — running the seed's old step into the new training would be
        # confusing (it's not a real resume of the same dataset/optimizer).
        seed_lora_path = body.get("seed_lora_path") or None
        if seed_lora_path:
            sp = Path(seed_lora_path)
            if not sp.exists() or not sp.is_file():
                self._json_response(
                    {"error": f"seed_lora_path not found: {seed_lora_path}"},
                    status=400,
                )
                return
            # Re-validate server-side — never trust path coming back from client alone.
            from underfit.utils.lora_validate import validate_lora_safetensors
            seed_check = validate_lora_safetensors(sp)
            if not seed_check["ok"]:
                self._json_response(
                    {"error": f"seed LoRA failed re-validation: {seed_check['error']}"},
                    status=400,
                )
                return
            sc = seed_check["config"]
            seed_base = sc.get("base_model")
            if seed_base and seed_base != base_model:
                self._json_response(
                    {"error": f"Seed LoRA was trained from '{seed_base}', "
                              f"but selected base model is '{base_model}'. Switch the "
                              f"base model in the form, or clear the seed."},
                    status=400,
                )
                return
            lora_type = sc.get("adapter_type", lora_type)
            rank = int(sc.get("rank", rank))
            alpha = sc.get("alpha", alpha)
            inc = sc.get("include")
            exc = sc.get("exclude")
            lora_include = ",".join(inc) if isinstance(inc, list) else (inc or lora_include)
            lora_exclude = ",".join(exc) if isinstance(exc, list) else (exc or lora_exclude)
        lr_raw = body.get("lr", "")
        base_precision = body.get("base_precision")  # null, "bf16", "fp16"

        dataset_id = body.get("dataset_id")
        if not dataset_id:
            self._json_response({"error": "dataset_id is required"}, status=400)
            return
        prompt_config = body.get("prompt_config")
        custom_demo_cond = body.get("demo_cond")  # list of {prompt, cfg, steps, arc?, fixed_prompt?}

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        run_id = f"{name}-{timestamp}"
        save_dir = str(RUNS_DIR)
        mi = _get_model_info(base_model)
        base_model_config = mi["template"]
        # Log lives on local SSD when UNDERFIT_LOGS_DIR is set (= Colab),
        # otherwise falls back to RUNS_DIR (single-machine setups).
        log_path = str(LOGS_DIR / f"{run_id}.log")

        # Generate per-run dataset config
        ds = datasets_registry.get_dataset(dataset_id)
        if not ds:
            self._json_response({"error": f"dataset '{dataset_id}' not found"}, status=400)
            return
        if ds["status"] != "ready":
            self._json_response({"error": f"dataset '{dataset_id}' is not ready (status: {ds['status']})"}, status=400)
            return
        run_dataset_config = f"{save_dir}/{run_id}_dataset.json"
        ds_cfg = {
            "dataset_type": "pre_encoded",
            "datasets": [{
                "id": ds["name"],
                "path": ds["latent_dir"],
                "custom_metadata_module": ds.get("custom_metadata_module", str(PRE_DIR / "prompt_templates.py")),
            }],
            "latent_crop_length": body.get("latent_crop_length", mi["latent_crop_length"]),
            "random_crop": body.get("random_crop", True),
        }
        if prompt_config:
            ds_cfg["prompt_config"] = prompt_config
            ds_cfg["datasets"][0]["custom_metadata_module"] = str(PRE_DIR / "prompt_templates.py")
        try:
            os.makedirs(save_dir, exist_ok=True)
            with open(run_dataset_config, "w") as f:
                json.dump(ds_cfg, f, indent=2)
                f.write("\n")
            dataset_config = run_dataset_config
        except Exception as e:
            self._json_response({"error": f"failed to create dataset config: {e}"}, status=500)
            return

        # Create per-run model config with user's settings
        try:
            with open(base_model_config) as f:
                cfg = json.load(f)
            # Stash the dashboard's base-model key so lora_train can embed it
            # in saved checkpoint metadata. Used by 'Start from a previous
            # LoRA' uploads to verify shape compatibility.
            cfg["base_model"] = base_model
            cfg.setdefault("training", {}).setdefault("lora_config", {})["rank"] = rank
            cfg["training"]["lora_config"]["adapter_type"] = lora_type
            if alpha is not None:
                cfg["training"]["lora_config"]["alpha"] = float(alpha)
            if lora_include:
                cfg["training"]["lora_config"]["include"] = [s.strip() for s in lora_include.split(",") if s.strip()]
            if lora_exclude:
                cfg["training"]["lora_config"]["exclude"] = [s.strip() for s in lora_exclude.split(",") if s.strip()]
            # Seed: copy the uploaded .safetensors into the run dir + point training at it.
            # We keep the seed file with the run so it survives even if the
            # original upload gets garbage-collected from SEED_LORAS_DIR.
            if seed_lora_path:
                import shutil
                run_seed_dir = Path(save_dir) / run_id
                run_seed_dir.mkdir(parents=True, exist_ok=True)
                run_seed_dst = run_seed_dir / "seed_lora.safetensors"
                try:
                    shutil.copy2(seed_lora_path, run_seed_dst)
                except OSError as e:
                    self._json_response(
                        {"error": f"failed to copy seed LoRA into run dir: {e}"},
                        status=500,
                    )
                    return
                cfg["training"]["lora_ckpt_path"] = str(run_seed_dst)
                # Explicitly zero the step/epoch counters. Without these, the
                # training loop's _resolve_offsets falls through to the seed
                # safetensors' metadata (e.g. step=15000) — which is wrong for
                # a fresh run that just initializes from those weights.
                cfg["training"]["step_offset"] = 0
                cfg["training"]["epoch_offset"] = 0
            if mi.get("svd_bases"):
                cfg["svd_bases_path"] = mi["svd_bases"]
            if base_precision:
                cfg["training"]["base_precision"] = base_precision
            cfg["training"].setdefault("demo", {})["demo_every"] = demo_every
            cfg["training"]["demo"]["demo_mode"] = "lora_dashboard"
            cfg["training"]["demo"]["latent_crop_length"] = body.get("latent_crop_length", mi["latent_crop_length"])
            # Apply custom demo_cond from frontend if provided
            if custom_demo_cond:
                # Compute seconds_total from the actual latent crop length
                crop_len = body.get("latent_crop_length", mi["latent_crop_length"])
                seconds_total = max(1, int(crop_len * mi["clip_duration"] / mi["latent_crop_length"]))
                new_demo_cond = []
                for d in custom_demo_cond:
                    # Per-demo duration overrides global seconds_total
                    demo_dur = d.get("duration")
                    demo_sec = max(1, int(demo_dur)) if demo_dur else seconds_total
                    entry = {
                        "prompt": d.get("prompt", ""),
                        "seconds_total": demo_sec,
                        "cfg": d.get("cfg", 7),
                    }
                    if demo_dur:
                        entry["duration"] = demo_dur
                    # SAO needs seconds_start conditioning
                    if base_model == "sao":
                        entry["seconds_start"] = 0
                    if d.get("arc"):
                        entry["arc"] = True
                        entry["steps"] = d.get("steps", 8)
                    else:
                        entry["steps"] = d.get("steps", 50)
                    if d.get("fixed_prompt"):
                        entry["fixed_prompt"] = True
                    if d.get("seed") is not None:
                        entry["seed"] = d["seed"]
                    new_demo_cond.append(entry)
                cfg["training"].setdefault("demo", {})["demo_cond"] = new_demo_cond
            # Ground truth and demo prompts
            run_demo_prompts = None
            _frontend_gt = body.get("ground_truth")  # sent by frontend with source files

            # If the frontend didn't send GT (likely because the dataset's GT
            # was still being generated when the user clicked "Launch"),
            # synthesize a slot list matched to the demo_cond length so the
            # fill-from-dataset logic below can populate them from the
            # dataset's known dataset_files.
            _demo_cond_for_gt = cfg.get("training", {}).get("demo", {}).get("demo_cond", [])
            if not _frontend_gt and _demo_cond_for_gt and dataset_id:
                _frontend_gt = [{} for _ in _demo_cond_for_gt]

            # Fill missing sourceFiles with random dataset tracks so every demo
            # gets a GT to play. No repeats within this list — once a track is
            # claimed (either by an explicit attachment or a random pick), it
            # won't be reused. If the dataset has fewer files than missing
            # slots, the remainder simply stay empty.
            if _frontend_gt and isinstance(_frontend_gt, list) and dataset_id:
                _ds_for_fill = datasets_registry.get_dataset(dataset_id)
                _fill_input_dir = _ds_for_fill.get("input_dir", "") if _ds_for_fill else ""
                _fill_files = (_ds_for_fill.get("dataset_files") or []) if _ds_for_fill else []
                if _fill_files and _fill_input_dir:
                    _used_rels = {g.get("relpath") for g in _frontend_gt if g and g.get("relpath")}
                    _pool = []
                    for _df in _fill_files:
                        _sp = _df.get("source_path")
                        if not _sp:
                            continue
                        try:
                            _rel = os.path.relpath(_sp, _fill_input_dir)
                        except Exception:
                            continue
                        if _rel.startswith("..") or _rel in _used_rels:
                            continue
                        _pool.append({
                            "relpath": _rel,
                            "title": _df.get("title", ""),
                            "gt_prompt": _df.get("prompt") or _df.get("sidecar_prompt") or _df.get("title", ""),
                        })
                    import random as _fill_rng
                    _fill_rng.shuffle(_pool)
                    for _gte in _frontend_gt:
                        if not _gte:
                            continue
                        if not _gte.get("relpath") and _pool:
                            _pick = _pool.pop()
                            _gte["relpath"] = _pick["relpath"]
                            if not _gte.get("title"):
                                _gte["title"] = _pick["title"]
                            if not _gte.get("gt_prompt"):
                                _gte["gt_prompt"] = _pick["gt_prompt"]

            demo_cond = cfg["training"].get("demo", {}).get("demo_cond", [])
            if custom_demo_cond:
                # Frontend provided everything — just extract non-fixed prompts for display
                run_demo_prompts = [e.get("prompt", "") for e in demo_cond
                                    if not e.get("fixed_prompt")]
            # Assign random seeds 10-100 to demos that don't already have one
            import random as _rng
            demo_cond = cfg.get("training", {}).get("demo", {}).get("demo_cond", [])
            for entry in demo_cond:
                if entry.get("seed") is None:
                    entry["seed"] = _rng.randint(10, 100)
            # Embed source-file relpath into demo_cond so future clones can
            # rehydrate ground truth from the model config alone — no dependency
            # on runs.json. Each demo_cond entry pairs 1:1 with the frontend's
            # ground_truth payload by index.
            if _frontend_gt and isinstance(_frontend_gt, list):
                for i, entry in enumerate(demo_cond):
                    if i < len(_frontend_gt):
                        gte = _frontend_gt[i] or {}
                        if gte.get("relpath"):
                            entry["source_relpath"] = gte["relpath"]
                            if gte.get("title"):
                                entry["source_title"] = gte["title"]
                            if gte.get("gt_prompt"):
                                entry["source_gt_prompt"] = gte["gt_prompt"]
            if lr_raw:
                lr_val = float(lr_raw)
                cfg["training"].setdefault("optimizer_configs", {}).setdefault("diffusion", {}).setdefault("optimizer", {}).setdefault("config", {})["lr"] = lr_val
            # Inject ARC path for demos during training.
            if mi.get("arc_ckpt"):
                demo_config = cfg["training"].setdefault("demo", {})
                if mi.get("arc_type") == "lora":
                    demo_config["arc_lora_path"] = mi["arc_ckpt"]
                elif mi.get("arc_type") == "full_model":
                    demo_config["arc_full_model_path"] = mi["arc_ckpt"]
                    demo_config["arc_full_model_config"] = mi["arc_config"]
            run_config_path = f"{save_dir}/{run_id}_model.json"
            with open(run_config_path, "w") as f:
                json.dump(cfg, f, indent=2)
                f.write("\n")
        except Exception as e:
            self._json_response({"error": f"failed to create run config: {e}"}, status=500)
            return

        demo_dir = str(RUNS_DIR / run_id / "demos")
        os.makedirs(demo_dir, exist_ok=True)

        _q = shlex.quote
        restart_cmd = (
            f"python3 {BASE_DIR / 'lora_train.py'}"
            f"     --name {_q(run_id)}"
            f"     --config-file {BASE_DIR / 'defaults.ini'}"
            f"     --save-dir {_q(save_dir)}"
            f"     --model-config {_q(run_config_path)}"
            f"     --dataset-config {_q(dataset_config)}"
            f"     --val-dataset-config ''"
            f"     --pretrained-ckpt-path {_q(mi['base_ckpt'])}"
            f"     --pretransform-ckpt-path ''"
            f"     --ckpt-path ''"
            f"     --num-nodes 1"
            f"     --num-workers 8"
            f"     --precision 16-mixed"
            f"     --batch-size {batch_size}"
            f"     --checkpoint-every {checkpoint_every}"
            f"     --max-steps {max_steps}"
            f"     --gradient-clip-val 1.0"
            f"     --logger ''"
        )

        gpu_env = f"CUDA_VISIBLE_DEVICES={gpu} "
        # Cap CPU thread pools (same rationale as gradio launches: nproc-sized
        # default pools can exhaust ulimit -u when multiple training procs are
        # alive). Tied to GRADIO_THREAD_CAP for consistency.
        thread_env = (
            "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "
            "NUMEXPR_NUM_THREADS=4 RAYON_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false "
        ) if GRADIO_THREAD_CAP else ""
        backend_env = _backend_env_for_model(base_model)
        # UNDERFIT_LOG_PATH lets lora_train.py write a sidecar <log>.exit on
        # any unhandled exception — robust to pipe/buffering issues that can
        # cause the normal tee'd log to come out empty.
        log_env = f"UNDERFIT_LOG_PATH={_q(log_path)} "
        launch_cmd = f"source {VENV_ACTIVATE} && cd {_q(demo_dir)} && PYTHONUNBUFFERED=1 {log_env}{backend_env}{thread_env}{gpu_env}{restart_cmd} 2>&1 | tee {_q(log_path)}"
        # Capture bash's own stderr (shell errors, source failures, etc.) — used by
        # the run-monitor's diagnose helper to surface a hint when a run dies fast
        # with an empty log.
        bash_err_path = log_path + ".bash.err"
        try:
            bash_err_f = open(bash_err_path, "wb")
        except OSError:
            bash_err_f = subprocess.DEVNULL
        try:
            proc = subprocess.Popen(
                ["bash", "-c", launch_cmd],
                stdout=subprocess.DEVNULL,
                stderr=bash_err_f,
                preexec_fn=os.setsid,
            )
        except Exception as e:
            self._json_response({"error": f"failed to launch: {e}"}, status=500)
            return

        run = {
            "id": run_id,
            "display_name": name,
            "log_path": log_path,
            "demo_source_dir": demo_dir,
            "checkpoints_dir": save_dir,
            "max_steps": max_steps,
            "active": True,
            "status": "training",
            "step_offset": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pid": proc.pid,
            "gpu": gpu,
            "restart_cmd": restart_cmd,
            "base_model": base_model,
            "dataset_id": dataset_id,
            "dataset_history": [{
                "dataset_id": dataset_id,
                "dataset_name": ds["name"],
                "from_step": 0,
            }] if dataset_id else [],
        }
        if run_demo_prompts:
            run["demo_prompts"] = run_demo_prompts
        if seed_lora_path:
            run["seed_lora"] = {
                "filename": body.get("seed_lora_filename") or os.path.basename(seed_lora_path),
                "adapter_type": lora_type,
                "rank": rank,
            }
        registry.add_run(run)
        (AUDIO_DIR / "runs" / run_id).mkdir(parents=True, exist_ok=True)
        print(f"[control] New finetune {run_id} on GPU {gpu}, PID {proc.pid}, max_steps {max_steps}")
        self._json_response({"ok": True, "id": run_id, "pid": proc.pid, "display_name": name, "status": "loading"}, status=201)

        # Generate ground truth clips in background from frontend-provided GT
        if _frontend_gt and dataset_id:
            ds = datasets_registry.get_dataset(dataset_id)
            _input_dir = ds.get("input_dir", "") if ds else ""
            def _bg_gt_from_frontend(gt_entries, rid, input_dir):
                try:
                    gt_list = []
                    gt_prompts = []
                    gt_dir = AUDIO_DIR / "ground_truth" / rid
                    gt_dir.mkdir(parents=True, exist_ok=True)
                    for i, gte in enumerate(gt_entries):
                        relpath = gte.get("relpath", "")
                        src = os.path.join(input_dir, relpath) if relpath and input_dir else ""
                        # Skip entries with no source file — don't pollute runs.json
                        # with broken URLs (they would render as blank audio elements).
                        if not (src and os.path.isfile(src)):
                            continue
                        out = gt_dir / f"track_{i}.mp3"
                        title = gte.get("title", "")
                        gt_prompt = gte.get("gt_prompt", title)
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", src, "-map", "0:a",
                             "-codec:a", "libmp3lame", "-q:a", "2",
                             "-loglevel", "error", str(out)],
                            capture_output=True, timeout=600)
                        # Drop the entry if ffmpeg silently failed to write
                        if not out.exists() or out.stat().st_size == 0:
                            continue
                        entry = {"title": title,
                                 "url": f"/audio/ground_truth/{rid}/track_{i}.mp3",
                                 "source_path": src}
                        # Copy optional metadata fields
                        for k in ("album", "year", "genre"):
                            if gte.get(k):
                                entry[k] = gte[k]
                        gt_list.append(entry)
                        gt_prompts.append(gt_prompt)
                    # Generate spectrograms for the GT clips
                    for mp3 in gt_dir.glob("*.mp3"):
                        jpg = mp3.with_suffix(".jpg")
                        if not jpg.exists():
                            try:
                                generate_spectrogram(mp3, jpg)
                            except Exception:
                                pass
                    registry.update_run(rid, ground_truth=gt_list, gt_prompts=gt_prompts)
                    print(f"[control] Ground truth ready for {rid}: {len(gt_list)} tracks")
                except Exception as e:
                    print(f"[control] GT generation failed for {rid}: {e}")
            threading.Thread(
                target=_bg_gt_from_frontend,
                args=(_frontend_gt, run_id, _input_dir),
                daemon=True).start()

    def _handle_pause(self, run_id):
        run = registry.get_run(run_id)
        if not run:
            self._json_response({"error": "run not found"}, status=404)
            return
        if run.get("status") != "training":
            self._json_response({"error": f"cannot pause run in state '{run.get('status')}'"}, status=400)
            return
        pid = run.get("pid")
        if not pid or _detect_process_state(pid) == "dead":
            self._json_response({"error": "process is not running"}, status=400)
            return
        try:
            pgid = os.getpgid(pid) if _detect_process_state(pid) != "dead" else pid
            os.killpg(pgid, signal.SIGSTOP)
        except Exception as e:
            self._json_response({"error": f"SIGSTOP failed: {e}"}, status=500)
            return
        registry.update_run(run_id, status="paused")
        print(f"[control] Paused run {run_id} (PID {pid})")
        self._json_response({"ok": True, "status": "paused"})

    def _handle_continue(self, run_id):
        run = registry.get_run(run_id)
        if not run:
            self._json_response({"error": "run not found"}, status=404)
            return
        if run.get("status") != "paused":
            self._json_response({"error": f"cannot continue run in state '{run.get('status')}'"}, status=400)
            return
        pid = run.get("pid")
        if not pid:
            self._json_response({"error": "no PID"}, status=400)
            return
        try:
            pgid = os.getpgid(pid) if _detect_process_state(pid) != "dead" else pid
            os.killpg(pgid, signal.SIGCONT)
        except Exception as e:
            self._json_response({"error": f"SIGCONT failed: {e}"}, status=500)
            return
        registry.update_run(run_id, status="training")
        print(f"[control] Continued run {run_id} (PID {pid})")
        self._json_response({"ok": True, "status": "training"})

    def _handle_kill(self, run_id):
        run = registry.get_run(run_id)
        if not run:
            self._json_response({"error": "run not found"}, status=404)
            return
        if run.get("status") not in ("training", "paused", "loading", "resuming"):
            self._json_response({"error": f"cannot stop run in state '{run.get('status')}'"}, status=400)
            return
        prev_status = run.get("status")
        pid = run.get("pid")
        gpu = run.get("gpu")
        # Set status to killed immediately to prevent monitor auto-restart
        registry.update_run(run_id, status="killed")
        if pid:
            _kill_process_group(pid, paused=(prev_status == "paused"))
        print(f"[control] Stopped run {run_id} (PID {pid})")
        self._json_response({"ok": True, "status": "killed"})

    def _handle_resume(self, run_id, body):
        run = registry.get_run(run_id)
        if not run:
            self._json_response({"error": "run not found"}, status=404)
            return
        if run.get("status") in ("training", "paused"):
            self._json_response({"error": "run is still active"}, status=400)
            return
        restart_cmd = run.get("restart_cmd")
        if not restart_cmd:
            self._json_response({"error": "this run has no restart_cmd — cannot resume"}, status=400)
            return
        new_max_steps = body.get("max_steps")
        if not new_max_steps or not isinstance(new_max_steps, int):
            self._json_response({"error": "max_steps (int) required"}, status=400)
            return
        batch_size = body.get("batch_size")
        checkpoint_every = body.get("checkpoint_every")
        demo_every = body.get("demo_every")
        lr_raw = body.get("lr", "")

        # Find checkpoint to resume from
        latest_ckpt = None
        user_ckpt = body.get("checkpoint_path")
        if user_ckpt:
            p = Path(user_ckpt)
            if p.exists():
                latest_ckpt = p
            else:
                self._json_response({"error": f"checkpoint not found: {user_ckpt}"}, status=400)
                return
        else:
            ckpts_dir = Path(run.get("checkpoints_dir", ""))
            if ckpts_dir.exists():
                # Prefer .safetensors over .ckpt
                ckpts = list(ckpts_dir.rglob("*.safetensors")) + list(ckpts_dir.rglob("*.ckpt"))
                ckpts = [c for c in ckpts if run_id in str(c)]
                if ckpts:
                    ckpts.sort(key=lambda c: c.stat().st_mtime, reverse=True)
                    latest_ckpt = ckpts[0]

        # Extract step offset from checkpoint filename (step=N-epoch=M.safetensors).
        # The checkpoint's step is the *only* meaningful floor for max_steps —
        # the run's old max_steps and its latest logged step don't matter, since
        # we're rewinding training to the checkpoint.
        effective_offset = 0
        if latest_ckpt:
            step_match = re.search(r"step=(\d+)", latest_ckpt.name)
            if step_match:
                effective_offset = int(step_match.group(1))

        if new_max_steps <= effective_offset:
            self._json_response(
                {"error": f"max_steps ({new_max_steps}) must be > checkpoint step ({effective_offset})"},
                status=400,
            )
            return

        # Copy model config to run dir
        m = re.search(r"--model-config\s+(\S+)", restart_cmd)
        if not m:
            self._json_response({"error": "cannot parse --model-config from restart_cmd"}, status=500)
            return
        orig_model_config = Path(m.group(1))
        run_dir = orig_model_config.parent
        # Strip existing _resume suffix so successive resumes don't nest names
        base_stem = re.sub(r"_resume$", "", orig_model_config.stem)
        resume_config_path = run_dir / f"{base_stem}_resume.json"
        try:
            with open(orig_model_config) as f:
                cfg = json.load(f)
            if latest_ckpt:
                cfg.setdefault("training", {})["lora_ckpt_path"] = str(latest_ckpt)
            else:
                # No checkpoint — remove any stale lora_ckpt_path, start fresh from base model
                cfg.get("training", {}).pop("lora_ckpt_path", None)
            # Set step_offset so StepOffsetCallback sets global_step correctly
            cfg.setdefault("training", {})["step_offset"] = effective_offset
            # Apply user-specified overrides
            cfg.setdefault("training", {}).setdefault("demo", {})["demo_mode"] = "lora_dashboard"
            mi = _get_model_info(run.get("base_model"))
            if mi.get("svd_bases") and "svd_bases_path" not in cfg:
                cfg["svd_bases_path"] = mi["svd_bases"]
            if demo_every:
                cfg["training"]["demo"]["demo_every"] = int(demo_every)
            if lr_raw:
                lr_val = float(lr_raw)
                cfg.setdefault("training", {}).setdefault("optimizer_configs", {}).setdefault("diffusion", {}).setdefault("optimizer", {}).setdefault("config", {})["lr"] = lr_val
            # Assign fresh random seeds 10-100 to each demo on resume
            import random as _rng
            for entry in cfg.get("training", {}).get("demo", {}).get("demo_cond", []):
                entry["seed"] = _rng.randint(10, 100)
            with open(resume_config_path, "w") as f:
                json.dump(cfg, f, indent=2)
                f.write("\n")
        except Exception as e:
            self._json_response({"error": f"failed to create resume config: {e}"}, status=500)
            return

        # Optionally rewrite dataset config with new latent_crop_length /
        # random_crop. Both fields live in the dataset JSON, not the model
        # JSON, so we write a sibling _dataset_resume.json the same way we
        # write _model_resume.json.
        new_crop_len = body.get("latent_crop_length")
        new_random_crop = body.get("random_crop")
        resume_ds_path = None
        if new_crop_len is not None or new_random_crop is not None:
            ds_match = re.search(r"--dataset-config\s+(\S+)", restart_cmd)
            if ds_match:
                orig_ds_path = Path(ds_match.group(1).strip("'\""))
                if orig_ds_path.exists():
                    try:
                        with open(orig_ds_path) as f:
                            ds_cfg = json.load(f)
                        if new_crop_len is not None:
                            ds_cfg["latent_crop_length"] = int(new_crop_len)
                        if new_random_crop is not None:
                            ds_cfg["random_crop"] = bool(new_random_crop)
                        base_ds_stem = re.sub(r"_resume$", "", orig_ds_path.stem)
                        resume_ds_path = orig_ds_path.parent / f"{base_ds_stem}_resume.json"
                        with open(resume_ds_path, "w") as f:
                            json.dump(ds_cfg, f, indent=2)
                            f.write("\n")
                    except Exception:
                        resume_ds_path = None

        # Build new restart_cmd — max_steps is absolute since StepOffsetCallback offsets global_step
        new_cmd = re.sub(r"--model-config\s+\S+", f"--model-config {resume_config_path}", restart_cmd)
        if resume_ds_path:
            new_cmd = re.sub(r"--dataset-config\s+\S+", f"--dataset-config {resume_ds_path}", new_cmd)
        new_cmd = re.sub(r"--max-steps\s+\d+", f"--max-steps {new_max_steps}", new_cmd)
        if batch_size:
            new_cmd = re.sub(r"--batch-size\s+\d+", f"--batch-size {int(batch_size)}", new_cmd)
        if checkpoint_every:
            new_cmd = re.sub(r"--checkpoint-every\s+\d+", f"--checkpoint-every {int(checkpoint_every)}", new_cmd)
        # Ensure gradient clipping is present (older runs may lack it)
        if "--gradient-clip-val" not in new_cmd:
            new_cmd += "     --gradient-clip-val 1.0"

        # Create new log file
        orig_log = Path(run.get("log_path", ""))
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # Strip existing _resume_YYYYMMDDHHMMSS suffix to avoid nesting
        base_log_stem = re.sub(r"_resume_\d+$", "", orig_log.stem)
        # Always put resume logs in LOGS_DIR (local SSD on Colab) so resuming
        # an old run that originally wrote to Drive still gets fast logs.
        new_log = LOGS_DIR / f"{base_log_stem}_resume_{timestamp}.log"

        # Launch — use GPU from request body if provided, else fall back to run's previous GPU
        gpu = body.get("gpu", run.get("gpu"))
        gpu_env = f"CUDA_VISIBLE_DEVICES={gpu} " if gpu is not None else ""
        backend_env = _backend_env_for_model(run.get("base_model"))
        demo_dir = run.get("demo_source_dir", str(RUNS_DIR))
        os.makedirs(demo_dir, exist_ok=True)
        log_env = f"UNDERFIT_LOG_PATH={shlex.quote(str(new_log))} "
        bash_err_path = str(new_log) + ".bash.err"
        try:
            bash_err_f = open(bash_err_path, "wb")
        except OSError:
            bash_err_f = subprocess.DEVNULL
        launch_cmd = f"source {VENV_ACTIVATE} && cd {shlex.quote(demo_dir)} && PYTHONUNBUFFERED=1 {log_env}{backend_env}{gpu_env}{new_cmd} 2>&1 | tee {shlex.quote(str(new_log))}"
        try:
            proc = subprocess.Popen(
                ["bash", "-c", launch_cmd],
                stdout=subprocess.DEVNULL,
                stderr=bash_err_f,
                preexec_fn=os.setsid,
            )
        except Exception as e:
            self._json_response({"error": f"failed to launch: {e}"}, status=500)
            return

        # Update run record
        update_kwargs = dict(
            status="training",
            max_steps=new_max_steps,
            step_offset=effective_offset,
            log_path=str(new_log),
            restart_cmd=new_cmd,
            pid=proc.pid,
            gpu=gpu,
        )
        registry.update_run(run_id, **update_kwargs)
        # Invalidate hyperparams cache so dashboard picks up new dataset name
        with _hyperparams_cache_lock:
            _hyperparams_cache.pop(run_id, None)
        ckpt_info = f" from checkpoint" if latest_ckpt else " from base model (no checkpoint)"
        print(f"[control] Resumed run {run_id} as PID {proc.pid}, steps {effective_offset} -> {new_max_steps}{ckpt_info}")
        self._json_response({"ok": True, "status": "training", "pid": proc.pid, "new_max_steps": new_max_steps})

    def _handle_delete(self, run_id, body):
        run = registry.get_run(run_id)
        if not run:
            self._json_response({"error": "run not found"}, status=404)
            return
        delete_files = body.get("delete_files", False)

        # Kill process if still running
        status = run.get("status")
        pid = run.get("pid")
        gpu = run.get("gpu")
        if status in ("training", "paused", "loading", "resuming", "stopping") and pid:
            registry.update_run(run_id, status="killed")
            _kill_process_group(pid, paused=(status == "paused"))
            if gpu is not None:
                _free_gpu_memory(int(gpu))
            print(f"[delete] Killed process PID {pid} for run {run_id}")

        # Stop any Gradio instances using this run's checkpoints
        for inst in gradio_manager.list_instances():
            if inst.get("run_id") == run_id and inst["status"] in ("starting", "ready"):
                gradio_manager.stop(inst["id"])
                print(f"[delete] Stopped Gradio instance {inst['id']} for run {run_id}")

        # Remove from registry
        registry.remove_run(run_id)
        print(f"[delete] Removed run {run_id} from registry")

        # Clean up in-memory state
        with _run_steps_lock:
            _run_steps.pop(run_id, None)
        with _vram_history_lock:
            _vram_history.pop(run_id, None)

        # Delete files from disk if requested
        if delete_files:
            # Run directory (checkpoints, wandb, demos)
            run_dir = RUNS_DIR / run_id
            if run_dir.exists():
                try:
                    shutil.rmtree(run_dir)
                    print(f"[delete] Deleted run dir: {run_dir}")
                except Exception as e:
                    print(f"[delete] Failed to delete {run_dir}: {e}")

            # Log files: train/runs/{run_id}*.log
            log_dir = RUNS_DIR
            for log_file in log_dir.glob(f"{run_id}*.log"):
                try:
                    log_file.unlink()
                    print(f"[delete] Deleted log: {log_file}")
                except Exception as e:
                    print(f"[delete] Failed to delete {log_file}: {e}")

            # Processed audio
            audio_dir = AUDIO_DIR / "runs" / run_id
            if audio_dir.exists():
                try:
                    shutil.rmtree(audio_dir)
                    print(f"[delete] Deleted audio dir: {audio_dir}")
                except Exception as e:
                    print(f"[delete] Failed to delete {audio_dir}: {e}")

            # Resume config files (live in RUNS_DIR alongside the per-run configs)
            config_dir = RUNS_DIR
            for cfg_file in config_dir.glob("*_resume.json"):
                try:
                    with open(cfg_file) as f:
                        cfg = json.load(f)
                    lora_path = cfg.get("training", {}).get("lora_ckpt_path", "")
                    if run_id in lora_path:
                        cfg_file.unlink()
                        print(f"[delete] Deleted resume config: {cfg_file}")
                except Exception as e:
                    print(f"[delete] Failed to check/delete {cfg_file}: {e}")

        # Clear demo cache for deleted run
        with DashboardHandler._demo_cache_lock:
            DashboardHandler._demo_cache.pop(run_id, None)

        self._json_response({"ok": True})

    def do_PUT(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/runs/reorder":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            order = body.get("order")
            if not isinstance(order, list):
                self._json_response({"error": "missing order array"}, status=400)
                return
            registry.reorder(order)
            self._json_response({"ok": True})
        elif re.match(r"^/api/datasets/[^/]+/default_prompt$", parsed.path):
            ds_id = parsed.path.split("/")[3]
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            prompt = body.get("default_prompt", "").strip()
            ds = datasets_registry.get_dataset(ds_id)
            if not ds:
                self._json_response({"error": "dataset not found"}, status=404)
                return
            if prompt:
                datasets_registry.update_dataset(ds_id, default_prompt=prompt)
            else:
                datasets_registry.update_dataset(ds_id, default_prompt="")
            self._json_response({"ok": True})
        else:
            self.send_error(404)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        m = re.match(r"^/api/gradio/([^/]+)/log$", parsed.path)
        if m:
            instance_id = m.group(1)
            inst = None
            for i in gradio_manager.list_instances():
                if i["id"] == instance_id:
                    inst = i
                    break
            if not inst:
                self._json_response({"error": "instance not found"}, status=404)
                return
            lp = inst.get("log_path")
            if lp and os.path.exists(lp):
                try:
                    os.remove(lp)
                except Exception as e:
                    self._json_response({"error": str(e)}, status=500)
                    return
            # Remove the dead instance from tracking
            gradio_manager.remove_instance(instance_id)
            self._json_response({"ok": True})
            return
        m = re.match(r"^/api/gradio/([^/]+)$", parsed.path)
        if m:
            instance_id = m.group(1)
            if gradio_manager.stop(instance_id):
                self._json_response({"ok": True})
            else:
                self._json_response({"error": "instance not found"}, status=404)
        else:
            self.send_error(404)

    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _serve_audio(self, rel_path, dl_name=None):
        fpath = AUDIO_DIR / rel_path
        if not fpath.exists() or not fpath.is_file():
            self.send_error(404)
            return
        _mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
        content_type = _mime.get(fpath.suffix, "audio/mpeg")
        # Always revalidate against on-disk mtime so regenerated GT/demos/spectrograms
        # are picked up immediately instead of being served from browser cache.
        st = fpath.stat()
        last_modified = formatdate(st.st_mtime, usegmt=True)
        if_mod_since = self.headers.get("If-Modified-Since")
        if if_mod_since and if_mod_since == last_modified:
            self.send_response(304)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Last-Modified", last_modified)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(st.st_size))
        self.send_header("Last-Modified", last_modified)
        self.send_header("Cache-Control", "no-cache")
        if content_type.startswith("audio/"):
            self.send_header("Accept-Ranges", "bytes")
        if dl_name:
            self.send_header("Content-Disposition", f'inline; filename="{dl_name}"')
        self.end_headers()
        # Stream in chunks so we don't load the whole MP3 into memory.
        # ConnectionResetError / BrokenPipeError on a partial-load abort
        # is swallowed by ThreadedHTTPServer.handle_error.
        with open(fpath, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def _serve_audio_slice(self, params):
        """Slice an audio file by time range and serve for download."""
        audio_rel = params.get("path", [None])[0]
        start_s = params.get("start", [None])[0]
        end_s = params.get("end", [None])[0]
        filename = params.get("filename", [None])[0]
        if not audio_rel or start_s is None or end_s is None:
            self._json_response({"error": "missing path, start, or end"}, status=400)
            return
        try:
            start = float(start_s)
            end = float(end_s)
        except ValueError:
            self._json_response({"error": "invalid start/end"}, status=400)
            return
        if end <= start:
            self._json_response({"error": "end must be > start"}, status=400)
            return
        fpath = (AUDIO_DIR / audio_rel).resolve()
        if not str(fpath).startswith(str(AUDIO_DIR.resolve())):
            self.send_error(403)
            return
        if not fpath.exists() or not fpath.is_file():
            self.send_error(404)
            return
        duration = end - start
        if not filename:
            stem = fpath.stem
            filename = f"{stem}[{start:.1f}-{end:.1f}].mp3"
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(fpath), "-ss", str(start), "-t", str(duration),
                 "-c:a", "libmp3lame", "-q:a", "2", tmp_path],
                capture_output=True, timeout=30
            )
            if result.returncode != 0:
                self._json_response({"error": f"ffmpeg failed: {result.stderr.decode()[-200:]}"}, status=500)
                return
            size = os.path.getsize(tmp_path)
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(size))
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            with open(tmp_path, "rb") as f:
                self.wfile.write(f.read())
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _serve_checkpoint_download(self, ckpt_path):
        if not ckpt_path:
            self.send_error(400, "Missing path parameter")
            return
        fpath = Path(ckpt_path)
        if not fpath.exists() or not fpath.is_file() or fpath.suffix not in (".ckpt", ".safetensors"):
            self.send_error(404)
            return
        size = fpath.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="{fpath.name}"')
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with open(fpath, "rb") as f:
            while chunk := f.read(1024 * 1024):
                self.wfile.write(chunk)

    def _resolve_run_id(self, run_id):
        """Resolve run_id param: use provided, or fall back to active run."""
        if run_id:
            return run_id
        return registry.get_active_run_id()

    def _get_gpu_info(self):
        """Query nvidia-smi and annotate GPUs with training/gradio labels.
        Generous timeout because the first nvidia-smi call on a fresh Colab
        VM can take 5–10 s while the driver initializes."""
        gpus = []
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,memory.free,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                timeout=15, stderr=subprocess.DEVNULL,
            ).decode().strip()
            for line in out.split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 5:
                    continue
                gpus.append({
                    "gpu": int(parts[0]),
                    "used_mb": int(parts[1]),
                    "total_mb": int(parts[2]),
                    "free_mb": int(parts[3]),
                    "util_pct": int(parts[4]),
                    "labels": [],
                })
        except Exception:
            return {"gpus": [], "gradio_estimate": _gradio_vram}

        caps = _query_gpu_compute_caps()
        for g in gpus:
            g["compute_cap"] = caps.get(g["gpu"])

        # Build lookup: gpu -> used_mb for checking occupancy
        gpu_mem = {g["gpu"]: g["used_mb"] for g in gpus}

        # Build lookup: gpu -> list of labels
        gpu_labels = {}  # gpu -> [label_str, ...]
        # Training runs — only show active runs (not completed/killed)
        for r in registry.list_runs():
            run_status = r.get("status", "killed")
            if run_status in ("completed", "killed", "error"):
                continue
            gpu = r.get("gpu")
            if gpu is not None:
                gpu = int(gpu)
                # Determine label prefix
                if run_status == "training":
                    # Detect loading/resuming/demos sub-state
                    log_p = Path(r.get("log_path", ""))
                    is_loading = False
                    if log_p.exists() and log_p.stat().st_size > 0:
                        try:
                            with open(log_p, "rb") as f:
                                f.seek(max(0, f.seek(0, 2) - 65536))
                                tail = f.read().decode("utf-8", errors="replace")
                            has_progress = bool(re.search(r"Epoch \d+:", tail))
                            _demo_re = re.compile(r"Generating (?:(?:prompt|inpaint) demos for cfg scale|demo \d+)")
                            has_demo_marker = bool(_demo_re.search(tail))
                            if has_demo_marker:
                                tail_lines = tail.strip().splitlines()
                                last_prog_idx = -1
                                last_demo_idx = -1
                                for i, ln in enumerate(tail_lines):
                                    if re.search(r"Epoch \d+:", ln):
                                        last_prog_idx = i
                                    if _demo_re.search(ln):
                                        last_demo_idx = i
                                if last_demo_idx > last_prog_idx:
                                    run_status = "demos"
                                elif not has_progress:
                                    run_status = "demos"
                            elif not has_progress:
                                is_loading = True
                        except Exception:
                            pass
                    else:
                        is_loading = True
                    if is_loading and run_status == "training":
                        run_status = "resuming" if r.get("step_offset", 0) > 0 else "loading"
                prefix_map = {"training": "Training", "loading": "Loading",
                              "resuming": "Resuming", "paused": "Paused",
                              "demos": "Demos"}
                prefix = prefix_map.get(run_status, "Training")
                gpu_labels.setdefault(gpu, []).append(
                    f"{prefix}: {r.get('display_name', r['id'])}"
                )

        # Gradio instances
        for inst in gradio_manager.list_instances():
            if inst["status"] in ("starting", "ready"):
                gpu_labels.setdefault(inst["gpu"], []).append(
                    f"Gradio: {inst.get('title') or inst.get('checkpoint_name', '?')}"
                )

        # Encoding datasets
        for ds in datasets_registry.list_datasets():
            if ds["status"] == "encoding":
                for gpu in ds.get("encoding_gpus", []):
                    gpu_labels.setdefault(int(gpu), []).append(
                        f"Encoding: {ds['name']}"
                    )

        for g in gpus:
            g["labels"] = gpu_labels.get(g["gpu"], [])

        # Expose ARC availability per model for frontend
        arc_info = {}
        for k, v in MODEL_INFO.items():
            arc_info[k] = {"arc_type": v.get("arc_type"), "diffusion_objective": v.get("diffusion_objective", "v")}

        # Expose CUDA_VISIBLE_DEVICES so frontend can warn about invisible GPUs
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is not None:
            try:
                visible_gpus = [int(x.strip()) for x in cvd.split(",") if x.strip()]
            except ValueError:
                visible_gpus = None  # non-numeric (e.g. UUIDs) — skip
        else:
            visible_gpus = None  # not set = all visible

        resp = {"gpus": gpus, "gradio_estimate": _gradio_vram, "arc_info": arc_info}
        if visible_gpus is not None:
            resp["cuda_visible_devices"] = visible_gpus
        return resp

    def _get_gpu_processes(self, gpu_idx):
        """Return classified processes running on a specific GPU."""
        # Query nvidia-smi for compute processes
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "-i", str(gpu_idx),
                 "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                timeout=5, stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            return []

        if not out:
            return []

        # Build lookup of managed PIDs
        managed_train_pids = {}  # pid -> run record
        for r in registry.list_runs():
            if r.get("status") in ("completed", "killed"):
                continue
            pid = r.get("pid")
            if pid:
                # The dashboard tracks the bash wrapper PID; find its children too
                managed_train_pids[pid] = r
                try:
                    children = subprocess.check_output(
                        ["pgrep", "-P", str(pid)], stderr=subprocess.DEVNULL
                    ).decode().strip().split()
                    for cpid_s in children:
                        cpid = int(cpid_s)
                        managed_train_pids[cpid] = r
                        # Also check grandchildren (bash -> tee/python)
                        try:
                            gchildren = subprocess.check_output(
                                ["pgrep", "-P", cpid_s], stderr=subprocess.DEVNULL
                            ).decode().strip().split()
                            for gpid_s in gchildren:
                                managed_train_pids[int(gpid_s)] = r
                        except Exception:
                            pass
                except Exception:
                    pass

        managed_gradio_pids = {}  # pid -> instance
        for inst in gradio_manager.list_instances():
            if inst["status"] in ("starting", "ready"):
                pid = inst.get("pid")
                if pid:
                    managed_gradio_pids[pid] = inst
                    try:
                        children = subprocess.check_output(
                            ["pgrep", "-P", str(pid)], stderr=subprocess.DEVNULL
                        ).decode().strip().split()
                        for cpid_s in children:
                            managed_gradio_pids[int(cpid_s)] = inst
                    except Exception:
                        pass

        managed_encoding_pids = {}  # pid -> dataset record
        for ds in datasets_registry.list_datasets():
            if ds["status"] == "encoding":
                pid = ds.get("encoding_pid")
                if pid:
                    managed_encoding_pids[pid] = ds
                    try:
                        children = subprocess.check_output(
                            ["pgrep", "-P", str(pid)], stderr=subprocess.DEVNULL
                        ).decode().strip().split()
                        for cpid_s in children:
                            cpid = int(cpid_s)
                            managed_encoding_pids[cpid] = ds
                            try:
                                gchildren = subprocess.check_output(
                                    ["pgrep", "-P", cpid_s], stderr=subprocess.DEVNULL
                                ).decode().strip().split()
                                for gpid_s in gchildren:
                                    managed_encoding_pids[int(gpid_s)] = ds
                            except Exception:
                                pass
                    except Exception:
                        pass

        my_uid = os.getuid()
        processes = []
        for line in out.split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            pid = int(parts[0])
            used_mb = int(parts[1])

            # Read cmdline
            try:
                cmdline = Path(f"/proc/{pid}/cmdline").read_text().replace("\0", " ").strip()
            except Exception:
                cmdline = ""

            # Skip processes owned by other users
            try:
                proc_uid = Path(f"/proc/{pid}").stat().st_uid
            except Exception:
                proc_uid = -1
            if proc_uid != my_uid:
                short = cmdline.split()[0].split("/")[-1] if cmdline else f"pid-{pid}"
                processes.append({
                    "pid": pid, "used_mb": used_mb, "type": "other_user",
                    "name": short, "cmdline": cmdline[:200],
                })
                continue

            # Classify
            if pid in managed_encoding_pids:
                ds = managed_encoding_pids[pid]
                processes.append({
                    "pid": pid, "used_mb": used_mb, "type": "encoding",
                    "name": ds["name"], "dataset_id": ds["id"],
                    "cmdline": cmdline[:200],
                })
            elif pid in managed_train_pids:
                r = managed_train_pids[pid]
                processes.append({
                    "pid": pid, "used_mb": used_mb, "type": "training",
                    "name": r.get("display_name", r["id"]),
                    "run_id": r["id"], "cmdline": cmdline[:200],
                })
            elif pid in managed_gradio_pids:
                inst = managed_gradio_pids[pid]
                processes.append({
                    "pid": pid, "used_mb": used_mb, "type": "gradio",
                    "name": inst.get("title") or inst.get("checkpoint_name", "?"),
                    "run_id": inst.get("run_id"),
                    "checkpoint_path": inst.get("checkpoint_path"),
                    "instance_id": inst["id"], "cmdline": cmdline[:200],
                })
            elif "train.py" in cmdline or "lora_train.py" in cmdline:
                # Orphaned training process — parse run name
                name_m = re.search(r"--name\s+(\S+)", cmdline)
                run_name = name_m.group(1) if name_m else f"pid-{pid}"
                processes.append({
                    "pid": pid, "used_mb": used_mb, "type": "orphan_training",
                    "name": run_name, "cmdline": cmdline[:200],
                })
            elif "gradio" in cmdline.lower() or "run_gradio" in cmdline:
                processes.append({
                    "pid": pid, "used_mb": used_mb, "type": "unmanaged_gradio",
                    "name": "gradio", "cmdline": cmdline[:200],
                })
            else:
                # Extract a short name from the command
                short = cmdline.split()[0].split("/")[-1] if cmdline else f"pid-{pid}"
                processes.append({
                    "pid": pid, "used_mb": used_mb, "type": "other",
                    "name": short, "cmdline": cmdline[:200],
                })

        return processes

    def _handle_adopt(self, body):
        """Adopt an orphaned training process into the registry."""
        pid = body.get("pid")
        if not pid:
            self._json_response({"error": "pid required"}, status=400)
            return
        pid = int(pid)

        # Read cmdline
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_text().replace("\0", " ").strip()
        except Exception:
            self._json_response({"error": f"cannot read /proc/{pid}/cmdline"}, status=400)
            return

        if "train.py" not in cmdline and "lora_train.py" not in cmdline:
            self._json_response({"error": "not a training process"}, status=400)
            return

        # Parse args from cmdline
        name_m = re.search(r"--name\s+(\S+)", cmdline)
        save_dir_m = re.search(r"--save-dir\s+(\S+)", cmdline)
        model_config_m = re.search(r"--model-config\s+(\S+)", cmdline)
        max_steps_m = re.search(r"--max-steps\s+(\d+)", cmdline)
        batch_m = re.search(r"--batch-size\s+(\d+)", cmdline)
        ckpt_every_m = re.search(r"--checkpoint-every\s+(\d+)", cmdline)

        run_name = name_m.group(1) if name_m else f"adopted-{pid}"
        save_dir = save_dir_m.group(1) if save_dir_m else str(RUNS_DIR)
        max_steps = int(max_steps_m.group(1)) if max_steps_m else 20000

        # Check not already registered
        if registry.get_run(run_name):
            self._json_response({"error": f"run '{run_name}' already in registry"}, status=409)
            return

        # Determine GPU from nvidia-smi
        gpu = body.get("gpu")

        # Find CWD (demo dir) — strip " (deleted)" suffix from /proc symlink
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
            cwd = re.sub(r"\s*\(deleted\)$", "", cwd)
        except Exception:
            cwd = f"{save_dir}/{run_name}/demos"
        os.makedirs(cwd, exist_ok=True)

        # Find log file — look for newest matching log
        log_path = f"{save_dir}/{run_name}.log"
        log_candidates = sorted(
            Path(save_dir).glob(f"{run_name}*.log"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        ) if Path(save_dir).exists() else []
        if log_candidates:
            log_path = str(log_candidates[0])

        # Reconstruct restart_cmd from cmdline
        # Strip leading path to python3, restore ~ for home dir, re-quote empty args
        restart_cmd = re.sub(r"^.*?(python3\s)", r"python3 ", cmdline)
        home = os.path.expanduser("~")
        restart_cmd = restart_cmd.replace(home, "~")
        # Restore empty-string args that lost their quotes (e.g. --val-dataset-config '')
        for flag in ("--val-dataset-config", "--pretransform-ckpt-path", "--ckpt-path"):
            restart_cmd = re.sub(rf"({flag})\s+(--|\Z)", rf"\1 '' \2", restart_cmd)

        # Detect step_offset from log filename
        step_offset = 0
        if "_resume_" in log_path:
            # Try to get offset from log content
            step, _ = _parse_latest_step({"log_path": log_path, "step_offset": 0, "max_steps": max_steps})
            if step and step > 0:
                # This is the raw internal step; the offset should be figured from checkpoint
                pass  # leave at 0 for safety

        run = {
            "id": run_name,
            "display_name": run_name.rsplit("-", 1)[0] if re.search(r"-\d{14}$", run_name) else run_name,
            "log_path": log_path,
            "demo_source_dir": cwd,
            "checkpoints_dir": save_dir,
            "max_steps": max_steps,
            "active": True,
            "status": "training",
            "step_offset": step_offset,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pid": pid,
            "gpu": int(gpu) if gpu is not None else None,
            "restart_cmd": restart_cmd,
        }
        registry.add_run(run)
        (AUDIO_DIR / "runs" / run_name).mkdir(parents=True, exist_ok=True)
        print(f"[adopt] Adopted orphan PID {pid} as run '{run_name}' on GPU {gpu}")
        self._json_response({"ok": True, "id": run_name}, status=201)

    def _handle_kill_pid(self, body):
        """Kill an arbitrary process by PID (for orphans/unmanaged processes)."""
        pid = body.get("pid")
        if not pid:
            self._json_response({"error": "pid required"}, status=400)
            return
        pid = int(pid)
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            # Fall back to just killing the PID
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        # If this PID belongs to a managed run, mark it killed
        for r in registry.list_runs():
            if r.get("pid") == pid and r.get("status") in ("training", "paused"):
                registry.update_run(r["id"], status="killed")
                print(f"[kill_pid] Marked managed run {r['id']} as killed")
                break
        print(f"[kill_pid] Killed PID {pid}")
        self._json_response({"ok": True})

    def _handle_save_checkpoint(self, body):
        """Send SIGUSR1 to training process to trigger a manual checkpoint save."""
        run_id = body.get("run_id")
        if not run_id:
            self._json_response({"error": "run_id required"}, status=400)
            return
        run = registry.get_run(run_id)
        if not run:
            self._json_response({"error": "run not found"}, status=404)
            return
        if run.get("status") != "training":
            self._json_response({"error": f"run is not training (status: {run.get('status')})"}, status=400)
            return
        pid = run.get("pid")
        if not pid or _detect_process_state(pid) == "dead":
            self._json_response({"error": "training process is not running"}, status=400)
            return
        # The stored PID is the bash wrapper. Find the actual python3 child process.
        try:
            result = subprocess.run(
                ["pgrep", "-P", str(pid)],
                capture_output=True, text=True, timeout=5
            )
            child_pids = [int(p) for p in result.stdout.strip().split('\n') if p.strip()]
        except Exception:
            child_pids = []
        # Send SIGUSR1 to the direct python3 child of the bash wrapper (not deeper descendants
        # which may be zombie dataloader workers)
        target_pid = pid
        for cpid in child_pids:
            try:
                with open(f"/proc/{cpid}/comm", "r") as f:
                    comm = f.read().strip()
                if comm in ("python3", "python"):
                    target_pid = cpid
                    break
            except Exception:
                continue
        try:
            os.kill(target_pid, signal.SIGUSR1)
            print(f"[save_checkpoint] Sent SIGUSR1 to PID {target_pid} (run {run_id}, stored PID {pid})")
            self._json_response({"ok": True, "pid": target_pid})
        except Exception as e:
            self._json_response({"error": f"SIGUSR1 failed: {e}"}, status=500)

    # ------------------------------------------------------------------
    # Dataset endpoints
    # ------------------------------------------------------------------

    @staticmethod
    def _scan_audio_tags(dir_path, sample_size=50, tag_sample_size=50,
                         progress_fn=None):
        """Walk dir for audio files, read ID3 tags on a sample.

        Lists ALL audio files but only reads ID3 tags on the first
        ``tag_sample_size`` files (expensive I/O).  ``sample_size`` controls
        how many files are *returned* — ``None`` means return all.
        ``progress_fn(phase, current, total)`` is called periodically.
        Returns (file_info_list, total_files, files_with_tags, files_with_json).
        """
        p = Path(dir_path).expanduser().resolve()
        audio_files = []
        # Collect ALL .json files for cross-directory sidecar matching
        json_files = {}  # stem -> list of absolute Paths
        if progress_fn:
            progress_fn("walking", 0, 0)
        for dirpath_, _, filenames in os.walk(p):
            dp = Path(dirpath_)
            for fn in filenames:
                fp = dp / fn
                if _is_audio_file(fp):
                    audio_files.append(fp)
                elif fn.lower().endswith(".json"):
                    stem = Path(fn).stem
                    json_files.setdefault(stem, []).append(fp)
            if progress_fn and len(audio_files) % 100 < len(filenames):
                progress_fn("walking", len(audio_files), 0)
        audio_files.sort()
        total = len(audio_files)
        if progress_fn:
            progress_fn("walking_done", total, total)

        returned = audio_files if sample_size is None else audio_files[:sample_size]
        tag_limit = len(returned) if tag_sample_size is None else tag_sample_size

        # Read tags in parallel (I/O-bound) — big speedup on large dirs
        tag_results = {}  # index -> tags dict
        if tag_limit > 0:
            to_tag = returned[:tag_limit]
            from concurrent.futures import as_completed
            done_count = 0
            with ThreadPoolExecutor(max_workers=16) as pool:
                futures = {pool.submit(_read_audio_tags, fp, json_files): i
                           for i, fp in enumerate(to_tag)}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        tag_results[idx] = fut.result()
                    except Exception:
                        tag_results[idx] = {}
                    done_count += 1
                    if progress_fn and (done_count % 40 == 0
                                        or done_count == len(futures)):
                        progress_fn("tags", done_count, len(futures))

        file_info = []
        files_with_tags = 0
        files_with_json = 0
        for i, fpath in enumerate(returned):
            rel = str(fpath.relative_to(p))
            has_json = _find_json_sidecar(fpath, json_files) is not None
            if has_json:
                files_with_json += 1
            tags = tag_results.get(i, {})
            # has_tags = embedded audio tags (ID3/Vorbis/MP4), separate from sidecar.
            # _read_audio_tags prefers sidecar, so if both exist, tags came from sidecar.
            # Mark has_tags only when tags exist but didn't come from a sidecar.
            has_embedded_tags = bool(tags) and not has_json
            if has_embedded_tags:
                files_with_tags += 1
            # Get duration from mutagen (reads header only, very fast)
            dur = None
            if _MutagenFile is not None:
                try:
                    mf = _MutagenFile(str(fpath))
                    if mf is not None and mf.info is not None and hasattr(mf.info, 'length'):
                        dur = round(mf.info.length, 1)
                except Exception:
                    pass
            file_info.append({
                "relpath": rel,
                "has_tags": has_embedded_tags,
                "tags": tags,
                "has_json": has_json,
                "duration": dur,
            })
        return file_info, total, files_with_tags, files_with_json

    def _handle_datasets_scan(self, body):
        """Scan a directory for audio files, check tags & pre-encoded data.

        Streams NDJSON progress lines, final line is the full result or error.
        """
        dir_path = body.get("path", "").strip()
        if not dir_path:
            self._json_response({"error": "path is required"}, status=400)
            return
        p = Path(dir_path).expanduser().resolve()
        if not p.is_dir():
            self._json_response({"error": f"Not a directory: {p}"}, status=400)
            return

        # Stream NDJSON progress
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def send_progress(phase, current, total):
            try:
                line = json.dumps({"progress": True, "phase": phase,
                                   "current": current, "total": total})
                self.wfile.write((line + "\n").encode())
                self.wfile.flush()
            except Exception:
                pass

        file_info, total_files, files_with_tags, files_with_json = \
            self._scan_audio_tags(str(p), sample_size=None,
                                  tag_sample_size=None,
                                  progress_fn=send_progress)

        if total_files == 0:
            self.wfile.write(json.dumps(
                {"error": "No audio files found"}).encode() + b"\n")
            self.wfile.flush()
            return

        result = json.dumps({
            "path": str(p),
            "audio_files": file_info,
            "total_files": total_files,
            "files_with_tags": files_with_tags,
            "files_with_json": files_with_json,
        })
        self.wfile.write(result.encode() + b"\n")
        self.wfile.flush()

    def _handle_datasets_encode(self, body):
        """Launch pre-encoding on multiple GPUs."""
        raw_name = body.get("name", "").strip()
        name = _slugify(raw_name)
        input_dir = body.get("input_dir", "").strip()
        model = body.get("model", "sa3-medium")
        gpus = body.get("gpus", [])
        half = body.get("half", True)
        default_prompt = body.get("default_prompt", "").strip()
        exclude_list = body.get("exclude", [])
        if not name or not input_dir:
            self._json_response({"error": "name and input_dir required"}, status=400)
            return
        # Ensure dataset name (slug) is unique — but if a same-name dataset
        # exists in a non-recoverable state (status == 'error', or 'ready'
        # with 0 latents), silently evict it so the user can retry with the
        # same name. Active or healthy datasets still block the new one.
        for existing in list(datasets_registry.list_datasets()):
            if _slugify(existing.get("name", "")) != name:
                continue
            status = existing.get("status", "")
            num_files = existing.get("num_files", 0) or existing.get("num_chunks", 0)
            is_zombie = (status == "error") or (status == "ready" and num_files == 0)
            if is_zombie:
                datasets_registry.remove_dataset(existing["id"])
                print(f"[datasets] Replacing zombie dataset '{existing['id']}' "
                      f"(status={status}, num_files={num_files}) — same name '{name}'")
                continue
            self._json_response({
                "error": f"Dataset name '{name}' already exists (id={existing['id']}, "
                         f"status={status}). Delete it first."
            }, status=409)
            return
        if not gpus or not isinstance(gpus, list):
            self._json_response({"error": "gpus array required"}, status=400)
            return
        input_path = Path(input_dir).expanduser().resolve()
        if not input_path.is_dir():
            self._json_response({"error": f"Not a directory: {input_path}"}, status=400)
            return
        if model not in ENCODING_MODELS:
            self._json_response({"error": f"Unknown model: {model}. Choose from: {list(ENCODING_MODELS.keys())}"}, status=400)
            return

        # Count audio files (skip macOS resource forks, tiny files, etc.)
        exclude_set = set(exclude_list) if exclude_list else set()
        num_files = 0
        for dirpath, _, filenames in os.walk(input_path):
            for fn in filenames:
                fp = Path(dirpath) / fn
                if _is_audio_file(fp):
                    relpath = str(fp.relative_to(input_path))
                    if relpath not in exclude_set:
                        num_files += 1

        output_dir = DATASETS_DIR / name
        latent_dir = output_dir / "latents" / model
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d")
        ds_id = f"{name}-{model}-{timestamp}"
        # Ensure unique ID
        if datasets_registry.get_dataset(ds_id):
            ds_id += f"-{datetime.now().strftime('%H%M%S')}"

        # Cache tag metadata at creation time (scan all files, read all tags)
        file_info, _, files_with_tags, files_with_json = self._scan_audio_tags(
            str(input_path), sample_size=None, tag_sample_size=None)
        # Filter out excluded files from tag cache
        if exclude_set:
            file_info = [f for f in file_info if f.get("relpath") not in exclude_set]
        _save_tag_cache(ds_id, file_info, num_files, files_with_tags, files_with_json)

        gpu_str = ",".join(str(g) for g in gpus)
        log_path = DATASETS_DIR / f"encode_{ds_id}.log"
        half_flag = " --half" if half else ""

        # Write exclude file if any files were unchecked
        exclude_flag = ""
        if exclude_set:
            exclude_file = output_dir / "exclude.txt"
            exclude_file.write_text("\n".join(sorted(exclude_set)) + "\n")
            exclude_flag = f" --exclude-file {shlex.quote(str(exclude_file))}"

        _q = shlex.quote
        cmd = (
            f"source {VENV_ACTIVATE} && "
            f"CUDA_VISIBLE_DEVICES={gpu_str} PYTHONUNBUFFERED=1 "
            f"python3 {PRE_DIR / 'pre_encode.py'} "
            f"--input-dir {_q(str(input_path))} "
            f"--model {_q(model)} "
            f"--output-dir {_q(str(output_dir))} "
            f"--num-gpus {len(gpus)}"
            f"{half_flag}{exclude_flag} "
            f"2>&1 | tee {_q(str(log_path))}"
        )
        try:
            proc = subprocess.Popen(
                ["bash", "-c", cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
        except Exception as e:
            self._json_response({"error": f"Failed to launch: {e}"}, status=500)
            return

        # Build dataset file list (source of truth for which files are in this dataset)
        dataset_files = _build_dataset_files(str(input_path), exclude_set=exclude_set or None)
        # Generate ground truth tracks from the dataset files
        gt_list, gt_prompts = _generate_dataset_ground_truth(dataset_files, name, model=model)

        ds = {
            "id": ds_id,
            "name": name,
            "input_dir": str(input_path),
            "latent_dir": str(latent_dir),
            "model": model,
            "latent_dim": 64,  # will be updated from details.json on completion
            "num_files": num_files,
            "sample_rate": 44100,
            "status": "encoding",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "encoding_pid": proc.pid,
            "encoding_gpus": [int(g) for g in gpus],
            "encoding_progress": {"encoded": 0, "skipped": 0, "errors": 0, "total": num_files},
            "log_path": str(log_path),
            "custom_metadata_module": str(PRE_DIR / "prompt_templates.py"),
            "details": {},
            "ground_truth": gt_list,
            "demo_prompts": gt_prompts,
            "dataset_files": dataset_files,
        }
        if default_prompt:
            ds["default_prompt"] = default_prompt
        datasets_registry.add_dataset(ds)
        print(f"[datasets] Encoding '{ds_id}' on GPUs [{gpu_str}], PID {proc.pid}, {num_files} files")
        self._json_response({"ok": True, "id": ds_id, "pid": proc.pid}, status=201)

    def _get_encoding_progress(self, ds_id):
        """Get encoding progress for a dataset."""
        ds = datasets_registry.get_dataset(ds_id)
        if not ds:
            return {"error": "dataset not found"}
        log_path = ds.get("log_path", "")
        progress = dict(ds.get("encoding_progress", {}))
        total = progress.get("total", 0)
        log_tail = ""
        if log_path:
            log_tail = _read_log_tail(log_path, max_bytes=8192)

            encoded = 0
            skipped = 0
            errors = 0

            # Check for the final "Done" summary line (most authoritative)
            done_re = re.compile(r"Done in .+?: (\d+) encoded, (\d+) skipped, (\d+) errors")
            done_m = done_re.search(log_tail)
            if done_m:
                encoded = int(done_m.group(1))
                skipped = int(done_m.group(2))
                errors = int(done_m.group(3))
            else:
                # Count actual .npy files in the latent directory (reliable)
                latent_dir = ds.get("latent_dir", "")
                if latent_dir:
                    lp = Path(latent_dir)
                    if lp.is_dir():
                        try:
                            encoded = sum(1 for _ in lp.rglob("*.npy"))
                        except Exception:
                            pass

                # Fall back to log parsing if no latent dir
                if encoded == 0:
                    shard_re = re.compile(
                        r"shard done: (\d+) encoded, (\d+) skipped, (\d+) errors")
                    for m in shard_re.finditer(log_tail):
                        encoded += int(m.group(1))
                        skipped += int(m.group(2))
                        errors += int(m.group(3))

                if encoded == 0:
                    progress_re = re.compile(r"\[(\d+)/(\d+)\]")
                    matches = progress_re.findall(log_tail)
                    if matches:
                        encoded = len(set(int(n) for n, _ in matches))

            progress = {"encoded": encoded, "skipped": skipped,
                        "errors": errors, "total": total}
        return {
            "status": ds["status"],
            "progress": progress,
            "log_tail": log_tail,
        }

    def _handle_datasets_stop(self, ds_id):
        """Stop an encoding process."""
        ds = datasets_registry.get_dataset(ds_id)
        if not ds:
            self._json_response({"error": "dataset not found"}, status=404)
            return
        if ds["status"] != "encoding":
            self._json_response({"error": f"dataset is not encoding (status: {ds['status']})"}, status=400)
            return
        pid = ds.get("encoding_pid")
        if pid:
            _kill_process_group(pid)
            for gpu in ds.get("encoding_gpus", []):
                _free_gpu_memory(gpu)
        datasets_registry.update_dataset(ds_id, status="error", encoding_pid=None)
        print(f"[datasets] Stopped encoding '{ds_id}' (PID {pid})")
        self._json_response({"ok": True})

    def _handle_datasets_delete(self, ds_id, body=None):
        """Remove a dataset from the registry, optionally deleting files."""
        ds = datasets_registry.get_dataset(ds_id)
        if not ds:
            self._json_response({"error": "dataset not found"}, status=404)
            return
        # Check if ANY managed run references this dataset
        using_runs = []
        for r in registry.list_runs():
            if r.get("dataset_id") == ds_id:
                using_runs.append(r.get("display_name", r["id"]))
        if using_runs:
            names = ", ".join(using_runs)
            self._json_response(
                {"error": f"Cannot delete: dataset is in use by training run(s): {names}"},
                status=409)
            return
        # Dry-run mode: just check if deletion is allowed without actually deleting
        if body and body.get("dry_run"):
            self._json_response({"ok": True, "can_delete": True})
            return
        # Stop encoding if still running
        if ds["status"] == "encoding":
            pid = ds.get("encoding_pid")
            if pid:
                _kill_process_group(pid)
        # Optionally remove files from disk
        if body and body.get("delete_files"):
            if ds.get("latent_dir"):
                latent_p = Path(ds["latent_dir"])
                if latent_p.exists():
                    shutil.rmtree(latent_p, ignore_errors=True)
                    print(f"[datasets] Removed latent dir: {latent_p}")
            gt_dir = AUDIO_DIR / "ground_truth" / ds.get("name", "")
            if gt_dir.exists():
                shutil.rmtree(gt_dir, ignore_errors=True)
                print(f"[datasets] Removed GT dir: {gt_dir}")
            if ds.get("log_path") and Path(ds["log_path"]).exists():
                Path(ds["log_path"]).unlink(missing_ok=True)
                print(f"[datasets] Removed log: {ds['log_path']}")
        datasets_registry.remove_dataset(ds_id)
        print(f"[datasets] Deleted '{ds_id}' from registry")
        self._json_response({"ok": True})

    def _get_runs(self):
        runs = registry.list_runs()
        result = []
        for r in runs:
            status = r.get("status", "killed")
            # Detect "loading"/"resuming"/"demos" sub-state for training runs
            if status == "training":
                is_loading = False
                log_p = Path(r.get("log_path", ""))
                if log_p.exists() and log_p.stat().st_size > 0:
                    with open(log_p, "rb") as f:
                        f.seek(max(0, f.seek(0, 2) - 16384))
                        tail = f.read().decode("utf-8", errors="replace")
                    has_progress = bool(re.search(r"Epoch \d+:", tail))
                    _demo_re = re.compile(r"Generating (?:(?:prompt|inpaint) demos for cfg scale|demo \d+)")
                    has_demo_marker = bool(_demo_re.search(tail))
                    if has_demo_marker:
                        # Check if demo marker is AFTER last progress line
                        tail_lines = tail.strip().splitlines()
                        last_prog_idx = -1
                        last_demo_idx = -1
                        for i, ln in enumerate(tail_lines):
                            if re.search(r"Epoch \d+:", ln):
                                last_prog_idx = i
                            if _demo_re.search(ln):
                                last_demo_idx = i
                        if last_demo_idx > last_prog_idx:
                            status = "demos"
                        elif not has_progress:
                            status = "demos"  # demos during loading (before any progress)
                        else:
                            is_loading = False
                    elif not has_progress:
                        is_loading = True
                else:
                    is_loading = True
                if is_loading and status == "training":
                    status = "resuming" if r.get("step_offset", 0) > 0 else "loading"
            # Retroactive OOM detection for killed runs — check once, persist
            if status == "killed" and not r.get("error"):
                log_p = Path(r.get("log_path", ""))
                if log_p.exists():
                    try:
                        with open(log_p, "rb") as f:
                            f.seek(max(0, f.seek(0, 2) - 8192))
                            oom_tail = f.read().decode("utf-8", errors="replace")
                        if "OutOfMemoryError" in oom_tail or "CUDA out of memory" in oom_tail:
                            status = "error"
                            registry.update_run(r["id"], status="error",
                                                error="OOM — not enough GPU memory")
                    except Exception:
                        pass
            elif status == "error":
                pass  # already detected
            log_mtime = None
            log_p2 = Path(r.get("log_path", ""))
            if log_p2.exists():
                try:
                    log_mtime = log_p2.stat().st_mtime
                except Exception:
                    pass
            result.append({
                "id": r["id"],
                "display_name": r.get("display_name", r["id"]),
                "active": r.get("active", False),
                "status": status,
                "max_steps": r.get("max_steps", 20000),
                "created_at": r.get("created_at", ""),
                "gpu": r.get("gpu"),
                "dataset_id": r.get("dataset_id"),
                "base_model": r.get("base_model"),
                "log_mtime": log_mtime,
            })
        return result

    # Checkpoint cache: dir_str -> (scan_time, {path_str: (stat_result, ...)})
    _ckpt_scan_cache = {}
    _ckpt_scan_lock = threading.Lock()
    _CKPT_CACHE_TTL = 30  # seconds — rescan directory every 30s

    # Demo cache: run_id -> {"preamble": {...}, "steps": {step_int: step_entry}, "preamble_mtime": float}
    # Steps are append-only — once written they never change.
    # Preamble (GT, prompts, seed) is cached until model config mtime changes.
    _demo_cache = {}
    _demo_cache_lock = threading.Lock()
    _demo_process_times = {}  # run_id -> last time _process_run_demos was triggered

    @staticmethod
    def _find_checkpoints(ckpts_dir, run_id, dataset_history=None):
        """Find checkpoints for a run, computing effective steps across wandb sessions."""
        ckpts_dir = Path(ckpts_dir)
        if not ckpts_dir.exists():
            return []

        # Targeted scan: only look in this run's subdirectory
        # Structure: ckpts_dir/{run_id}/{wandb_session}/checkpoints/*.safetensors
        run_dir = ckpts_dir / run_id
        if not run_dir.exists():
            return []

        cache_key = str(run_dir)
        now = time.time()

        with DashboardHandler._ckpt_scan_lock:
            cached = DashboardHandler._ckpt_scan_cache.get(cache_key)
            if cached and (now - cached[0]) < DashboardHandler._CKPT_CACHE_TTL:
                all_ckpts = cached[1]
            else:
                all_ckpts = {}
                try:
                    for session in os.scandir(run_dir):
                        if not session.is_dir():
                            continue
                        ckpt_subdir = os.path.join(session.path, "checkpoints")
                        if not os.path.isdir(ckpt_subdir):
                            continue
                        for f in os.scandir(ckpt_subdir):
                            if f.name.endswith(".safetensors"):
                                all_ckpts[f.path] = (Path(f.path), f.stat())
                            elif f.name.endswith(".ckpt"):
                                st_sibling = f.path.replace(".ckpt", ".safetensors")
                                if st_sibling not in all_ckpts:
                                    all_ckpts[f.path] = (Path(f.path), f.stat())
                except OSError:
                    pass
                DashboardHandler._ckpt_scan_cache[cache_key] = (now, all_ckpts)

        ckpts = [(c, st) for path_str, (c, st) in all_ckpts.items()]
        if not ckpts:
            return []
        step_re = re.compile(r"step=(\d+)")
        sessions = {}
        for c, st in ckpts:
            session_dir = str(c.parent.parent)
            sm = step_re.search(c.name)
            internal_step = int(sm.group(1)) if sm else 0
            sessions.setdefault(session_dir, []).append((c, st, internal_step))
        # Sort sessions by earliest checkpoint mtime, compute chained offsets
        sorted_sessions = sorted(sessions.items(),
            key=lambda kv: min(st.st_mtime for _, st, _ in kv[1]))
        session_offsets = {}
        cumulative = 0
        for i, (sdir, items) in enumerate(sorted_sessions):
            session_offsets[sdir] = cumulative
            if i < len(sorted_sessions) - 1:
                cumulative += max(s for _, _, s in items)
        ckpt_list = []
        for c, st in ckpts:
            session_dir = str(c.parent.parent)
            sm = step_re.search(c.name)
            internal_step = int(sm.group(1)) if sm else 0
            effective = internal_step + session_offsets.get(session_dir, 0)
            ckpt_entry = {
                "name": c.name,
                "display_name": f"Step {effective:,}",
                "session": c.parent.parent.name,
                "step": effective,
                "size_mb": round(st.st_size / 1e6, 1),
                "path": str(c),
                "mtime": st.st_mtime,
            }
            if dataset_history and len(dataset_history) > 1:
                seg = _dataset_for_step(dataset_history, effective)
                if seg:
                    ckpt_entry["dataset_name"] = seg["dataset_name"]
            ckpt_list.append(ckpt_entry)
        ckpt_list.sort(key=lambda x: x["mtime"], reverse=True)
        return ckpt_list

    def _get_status(self, run_id=None):
        run_id = self._resolve_run_id(run_id)
        if not run_id:
            return {"running": False, "status": "killed", "message": "No runs registered"}

        run = registry.get_run(run_id)
        if not run:
            return {"running": False, "status": "killed", "message": f"Run {run_id} not found"}

        status = run.get("status", "training")
        step_offset = run.get("step_offset", 0)
        log_path = Path(run.get("log_path", ""))
        if not log_path.exists():
            resp = {"running": False, "status": status, "run_name": run.get("display_name", run_id),
                    "run_id": run_id, "message": "Log file not found",
                    "has_restart_cmd": run.get("restart_cmd") is not None}
            if run.get("error"):
                resp["error"] = run["error"]
            return resp

        max_steps = run.get("max_steps", 20000)

        # Read from end in growing chunks until we find a progress line
        # (wandb/profiler output at end of completed runs can be >500KB)
        latest = {}
        # Match progress and metrics separately — metric order varies between configs
        progress_re = re.compile(r"Epoch (\d+):\s+(\d+)%.*?(\d+)/(\d+)")
        loss_re = re.compile(r"train/loss=([\d.]+)")
        lr_re = re.compile(r"train/lr=([\d.e+-]+)")
        matched_line = None
        m = None
        with open(log_path, "rb") as f:
            file_size = f.seek(0, 2)
            for chunk_size in [65536, 262144, 1048576]:
                f.seek(max(0, file_size - chunk_size))
                tail = f.read().decode("utf-8", errors="replace")
                lines = tail.strip().splitlines()
                for line in reversed(lines):
                    m = progress_re.search(line)
                    if m:
                        matched_line = line
                        break
                if matched_line:
                    break

        # Check if demo generation is happening — look for markers after the last progress line
        _demo_marker_re = re.compile(r"Generating (?:(?:prompt|inpaint) demos for cfg scale|demo \d+)")
        _generating_demos = False
        if matched_line and lines:
            # Find index of matched_line in lines, check if demo markers appear after it
            try:
                match_idx = len(lines) - 1 - list(reversed(lines)).index(matched_line)
                after_lines = lines[match_idx + 1:]
                _generating_demos = any(_demo_marker_re.search(l) for l in after_lines)
            except ValueError:
                pass
        elif not matched_line and lines:
            # No progress yet (loading) — check tail for demo markers
            _generating_demos = any(_demo_marker_re.search(l) for l in lines[-50:])

        if m and matched_line:
            epoch = int(m.group(1))
            steps_in_epoch = int(m.group(3))
            total_in_epoch = int(m.group(4))
            # Prefer the explicit "Step N," prefix the loop publishes — it's
            # the authoritative global_step. Fallback to the legacy
            # (epoch * total + step) + step_offset derivation for old logs
            # that only have "Epoch N:".
            step_prefix_m = re.search(r"Step (\d+),", matched_line)
            if step_prefix_m:
                effective_step = int(step_prefix_m.group(1))
            else:
                # Progress bar shows completed count (1-indexed), global_step is 0-indexed
                raw_step = max(0, epoch * total_in_epoch + steps_in_epoch - 1)
                effective_step = raw_step + step_offset

            loss_m = loss_re.search(matched_line)
            lr_m = lr_re.search(matched_line)

            # Auto-detect completion or death: PID dead → update status immediately
            if status in ("training", "paused"):
                proc_state = _detect_process_state(run.get("pid"))
                if proc_state == "dead" and effective_step >= max_steps - 1:
                    status = "completed"
                    registry.update_run(run_id, status="completed")
                elif proc_state == "dead":
                    status = "killed"
                    registry.update_run(run_id, status="killed")
                    print(f"[status] Run {run_id} PID {run.get('pid')} is dead — marking killed")

            # Detect demo generation sub-state
            if _generating_demos and status == "training":
                status = "demos"

            is_active = status in ("training", "paused", "demos")
            # Trust the in-band epoch value parsed from tqdm; the loop is the
            # source of truth for global epoch (seeded from checkpoint metadata
            # at resume, incremented per dataloader exhaustion thereafter).
            # Re-deriving as effective_step // total_in_epoch double-counts on
            # resume and breaks when batch_size or dataset_size changes.
            effective_epoch = epoch
            # max_epochs: the global epoch we'll reach at max_steps. Accounts
            # for resume offsets (using observed current_epoch) and the
            # current pace (current total_in_epoch). Robust to mid-run
            # batch_size or dataset_size changes since both inputs are live.
            if total_in_epoch > 0:
                remaining_steps = max(0, max_steps - effective_step)
                max_epochs = effective_epoch + remaining_steps // total_in_epoch
            else:
                max_epochs = None
            latest = {
                "running": is_active,
                "status": status,
                "run_name": run.get("display_name", run_id),
                "run_id": run_id,
                "epoch": effective_epoch,
                "max_epochs": max_epochs,
                "global_step": effective_step,
                "max_steps": max_steps,
                "progress_pct": round(min(effective_step / max_steps * 100, 100), 2),
                "loss": float(loss_m.group(1)) if loss_m else 0.0,
                "lr": float(lr_m.group(1)) if lr_m else 0.0,
                "steps_in_epoch": steps_in_epoch,
                "total_in_epoch": total_in_epoch,
                "has_restart_cmd": run.get("restart_cmd") is not None,
                "step_offset": step_offset,
            }
            if run.get("restart_count"):
                latest["restart_count"] = run["restart_count"]
            if run.get("error"):
                latest["error"] = run["error"]
            # Retroactive OOM detection for killed runs without stored error
            if status in ("killed", "error") and not latest.get("error"):
                try:
                    with open(log_path, "rb") as f:
                        f.seek(max(0, f.seek(0, 2) - 8192))
                        oom_tail = f.read().decode("utf-8", errors="replace")
                    if "OutOfMemoryError" in oom_tail or "CUDA out of memory" in oom_tail:
                        latest["error"] = "OOM — not enough GPU memory"
                        latest["status"] = "error"
                except Exception:
                    pass

        if not latest:
            # No progress found yet — detect sub-state. Include loading/resuming
            # so a crashed-during-init run transitions to "error" instead of
            # being stuck on the "Loading..." label.
            if status in ("training", "paused", "loading", "resuming"):
                proc_state = _detect_process_state(run.get("pid"))
                if proc_state == "dead":
                    # No step has been logged → never made it past init.
                    new_status = "error" if status in ("loading", "resuming") else "killed"
                    status = new_status
                    registry.update_run(run_id, status=new_status)
                    print(f"[status] Run {run_id} PID {run.get('pid')} is dead (no progress) — marking {new_status}")
                elif status == "training" and _generating_demos:
                    status = "demos"
                elif status == "training":
                    status = "resuming" if step_offset > 0 else "loading"
            resp = {"running": status in ("loading", "resuming", "training", "demos"),
                    "status": status,
                    "run_name": run.get("display_name", run_id),
                    "run_id": run_id, "message": "Parsing...",
                    "max_steps": max_steps,
                    "has_restart_cmd": run.get("restart_cmd") is not None,
                    "step_offset": step_offset}
            if run.get("restart_count"):
                resp["restart_count"] = run["restart_count"]
            if run.get("error"):
                resp["error"] = run["error"]
            # Retroactive OOM detection
            if status in ("killed", "error") and not resp.get("error"):
                try:
                    with open(log_path, "rb") as f:
                        f.seek(max(0, f.seek(0, 2) - 8192))
                        oom_tail = f.read().decode("utf-8", errors="replace")
                    if "OutOfMemoryError" in oom_tail or "CUDA out of memory" in oom_tail:
                        resp["error"] = "OOM — not enough GPU memory"
                        resp["status"] = "error"
                except Exception:
                    pass
            hyperparams = _extract_hyperparams_cached(run)
            if hyperparams:
                resp["hyperparams"] = hyperparams
            tail = _read_log_tail(log_path)
            if tail:
                resp["log_tail"] = tail
            return resp

        # Update per-run step
        with _run_steps_lock:
            _run_steps[run_id] = latest["global_step"]

        # Find all log files for this run (original + resume logs) to show full history
        log_dir = log_path.parent
        all_logs = sorted(
            [p for p in log_dir.glob(f"{run_id}*.log") if p.suffix == ".log"],
            key=lambda p: p.stat().st_mtime,
        )
        if not all_logs:
            all_logs = [log_path]

        loss_history, grad_norm_history, lora_mag_history, lr_history = \
            _parse_log_history_cached(run_id, all_logs, step_offset)

        def _downsample(arr, n=200):
            if len(arr) > n:
                stride = len(arr) // n
                return arr[::stride]
            return arr

        latest["loss_history"] = _downsample(loss_history)
        latest["grad_norm_history"] = _downsample(grad_norm_history)
        latest["lora_mag_history"] = _downsample(lora_mag_history)
        latest["lr_history"] = _downsample(lr_history)

        # GPU info
        latest["gpu"] = run.get("gpu")

        # Hyperparameters
        hyperparams = _extract_hyperparams_cached(run)
        if hyperparams:
            latest["hyperparams"] = hyperparams

        # VRAM history for this run's GPU
        with _vram_history_lock:
            vram_hist = list(_vram_history.get(run_id, []))
        if len(vram_hist) > 200:
            stride = len(vram_hist) // 200
            vram_hist = vram_hist[::stride]
        latest["vram_history"] = vram_hist

        # Log tail — last ~8KB of raw console output
        tail = _read_log_tail(log_path)
        if tail:
            latest["log_tail"] = tail

        return latest

    def _get_log_tail(self, run_id=None, file_size=0):
        """Return compressed log for the log viewer, with incremental support.

        file_size: the raw log file size the client last saw. If 0, returns the
        full compressed log. If equal to current size, returns nothing (no change).
        If less than current, reads only new bytes and returns compressed delta.
        """
        run_id = self._resolve_run_id(run_id)
        if not run_id:
            return {}
        run = registry.get_run(run_id)
        if not run:
            return {}
        log_path = Path(run.get("log_path", ""))
        if not log_path.exists():
            return {}
        try:
            st = log_path.stat()
            current_size = st.st_size
            mtime = st.st_mtime
        except Exception:
            return {}

        base = {"log_mtime": mtime, "file_size": current_size}

        if file_size >= current_size:
            return base  # no change

        if file_size == 0:
            # Full read + compress — include every log file for this run
            # (original + each resume_<timestamp>.log) so the viewer shows the
            # full training history, not just the most recent session. Files
            # are concatenated in chronological order; the "incremental"
            # branch below still tracks only the latest log's size, since
            # only that file changes during a live run.
            log_dir = log_path.parent
            all_logs = sorted(
                [p for p in log_dir.glob(f"{run_id}*.log") if p.suffix == ".log"],
                key=lambda p: p.stat().st_mtime,
            )
            if not all_logs:
                all_logs = [log_path]
            chunks = []
            for lf in all_logs:
                txt = _read_log_compressed(lf)
                if not txt:
                    continue
                if len(all_logs) > 1:
                    # Separator so the user can see the resume boundaries
                    chunks.append(f"\n────── {lf.name} ──────\n")
                chunks.append(txt)
            return {**base, "log_tail": "".join(chunks).lstrip()}

        # Incremental: always re-read from the last \n before file_size so the
        # boundary line is included.  This lets _collapse_progress_lines merge
        # the client's last progress-bar line with new ones that follow it.
        try:
            with open(log_path, "rb") as f:
                # Look back up to 4KB for the previous newline
                search_start = max(0, file_size - 4096)
                f.seek(search_start)
                lookback = f.read(file_size - search_start)
                last_nl = lookback.rfind(b"\n")
                if last_nl >= 0:
                    read_from = search_start + last_nl + 1
                else:
                    read_from = max(0, file_size - 4096)  # no newline — grab more context
                f.seek(read_from)
                new_raw = f.read()
        except Exception:
            return base

        # Same processing pipeline as full log
        new_text = "\n".join(_process_log_lines(new_raw))

        # Always replace from the client's last line since we re-read the boundary
        return {**base, "log_tail": new_text, "replace_last_line": True}

    def _get_clone_settings(self, run_id=None):
        """Return full config for cloning a run's settings into a new finetune."""
        run_id = self._resolve_run_id(run_id)
        if not run_id:
            return {"error": "no run_id"}
        run = registry.get_run(run_id)
        if not run:
            return {"error": "run not found"}

        result = {
            "base_model": run.get("base_model"),
            "dataset_id": run.get("dataset_id"),
        }

        # Read model config
        for suffix in [f"{run_id}_model_resume.json", f"{run_id}_model.json"]:
            cfg_path = RUNS_DIR / suffix
            if cfg_path.exists():
                try:
                    cfg = json.load(open(cfg_path))
                    training = cfg.get("training", {})
                    lora = training.get("lora_config", {})
                    demo = training.get("demo", {})
                    opt = training.get("optimizer_configs", {}).get("diffusion", {}).get("optimizer", {})

                    result["rank"] = lora.get("rank")
                    result["alpha"] = lora.get("alpha")
                    result["lora_type"] = lora.get("adapter_type")
                    result["lora_include"] = ", ".join(lora.get("include", []))
                    result["lora_exclude"] = ", ".join(lora.get("exclude", []))
                    result["base_precision"] = training.get("base_precision", "")
                    result["lr"] = opt.get("config", {}).get("lr")
                    result["demo_every"] = demo.get("demo_every")
                    result["demo_cond"] = demo.get("demo_cond", [])
                    result["latent_crop_length"] = demo.get("latent_crop_length")
                except Exception:
                    pass
                break

        # Read dataset config
        ds_cfg_path = RUNS_DIR / f"{run_id}_dataset.json"
        if ds_cfg_path.exists():
            try:
                ds_cfg = json.load(open(ds_cfg_path))
                result["random_crop"] = ds_cfg.get("random_crop", False)
                result["latent_crop_length"] = ds_cfg.get("latent_crop_length") or result.get("latent_crop_length")
                # Extract prompt config if present
                result["prompt_config"] = ds_cfg.get("prompt_config")
            except Exception:
                pass

        # Parse batch_size, max_steps, checkpoint_every from restart_cmd
        restart_cmd = run.get("restart_cmd", "")
        import re as _re
        for flag, key in [("--batch-size", "batch_size"), ("--max-steps", "max_steps"), ("--checkpoint-every", "checkpoint_every")]:
            m = _re.search(rf"{flag}\s+(\d+)", restart_cmd)
            if m:
                result[key] = int(m.group(1))

        # Source run's ground_truth — fallback for cloning legacy runs whose
        # demo_cond predates source_relpath embedding. Each entry gets relpath
        # filled in (computed from source_path - input_dir) so the frontend
        # can rehydrate sourceFile uniformly.
        gt_entries = run.get("ground_truth", []) or []
        ds_id = run.get("dataset_id")
        input_dir = ""
        if ds_id:
            ds = datasets_registry.get_dataset(ds_id)
            if ds:
                input_dir = ds.get("input_dir", "") or ""
        out_gt = []
        for g in gt_entries:
            entry = dict(g)
            if not entry.get("relpath") and entry.get("source_path") and input_dir:
                try:
                    rel = os.path.relpath(entry["source_path"], input_dir)
                    if not rel.startswith(".."):
                        entry["relpath"] = rel
                except Exception:
                    pass
            out_gt.append(entry)
        result["ground_truth"] = out_gt

        return result

    # Cache for loss_by_timestep: run_id -> {file_size, bins}
    _lbt_cache = {}
    _NUM_SIGMA_BINS = 5

    def _get_loss_by_timestep(self, run_id=None):
        """Read loss_by_timestep.bin incrementally, bin into sigma buckets with EMA."""
        import struct
        run_id = self._resolve_run_id(run_id)
        if not run_id:
            return {}
        run = registry.get_run(run_id)
        if not run:
            return {}

        # Find the binary file in the run's demo directory
        demos_dir = RUNS_DIR / run_id / "demos"
        lbt_path = demos_dir / "loss_by_timestep.bin"
        if not lbt_path.exists():
            return {}

        try:
            file_size = lbt_path.stat().st_size
        except Exception:
            return {}

        entry_size = 12  # uint32 + float32 + float32
        n_entries = file_size // entry_size
        if n_entries == 0:
            return {}

        # Check cache
        cached = self._lbt_cache.get(run_id)
        if cached and cached["file_size"] == file_size:
            return cached["result"]

        # Read entire file and process
        n_bins = self._NUM_SIGMA_BINS
        bin_width = 1.0 / n_bins
        smoothing = "ema"  # "ema" or "sliding"
        ema_alpha = 0.02
        sliding_window = 50

        # Collect raw (step, loss) per bin
        bin_raw = [[] for _ in range(n_bins)]

        try:
            with open(lbt_path, "rb") as f:
                data = f.read()
            for i in range(n_entries):
                offset = i * entry_size
                step, t_val, loss_val = struct.unpack_from("Iff", data, offset)
                bin_idx = min(int(t_val / bin_width), n_bins - 1)
                if bin_idx < 0:
                    bin_idx = 0
                bin_raw[bin_idx].append((step, loss_val))
        except Exception:
            return {}

        all_steps = [s for br in bin_raw for s, _ in br]
        if not all_steps:
            return {}
        min_step, max_step = min(all_steps), max(all_steps)

        bin_curves = []
        bin_counts = []

        if smoothing == "ema":
            # EMA: process data points in order, emit sampled curve points
            for bi in range(n_bins):
                raw = bin_raw[bi]
                bin_counts.append(len(raw))
                if not raw:
                    bin_curves.append([])
                    continue
                raw.sort(key=lambda x: x[0])
                ema = raw[0][1]
                curve = []
                sample_every = max(1, len(raw) // 500)
                warmup = int(1 / ema_alpha)  # skip initial points before EMA converges
                for j, (step, loss) in enumerate(raw):
                    ema = ema_alpha * loss + (1 - ema_alpha) * ema
                    if j >= warmup and j % sample_every == 0:
                        curve.append([step, round(ema, 6)])
                # Always include the last point
                if curve and curve[-1][0] != raw[-1][0]:
                    curve.append([raw[-1][0], round(ema, 6)])
                bin_curves.append(curve)

        else:  # sliding window
            half = sliding_window // 2
            for bi in range(n_bins):
                raw = bin_raw[bi]
                bin_counts.append(len(raw))
                if not raw:
                    bin_curves.append([])
                    continue
                raw.sort(key=lambda x: x[0])
                curve = []
                n_windows = max(1, (max_step - min_step) // max(1, half))
                sample_every = max(1, n_windows // 500)
                ri = 0
                for wi, wc in enumerate(range(min_step + half, max_step + 1, max(1, half))):
                    while ri < len(raw) and raw[ri][0] < wc - half:
                        ri += 1
                    total = count = 0
                    for j in range(ri, len(raw)):
                        if raw[j][0] > wc + half:
                            break
                        total += raw[j][1]
                        count += 1
                    if count > 0 and wi % sample_every == 0:
                        curve.append([wc, round(total / count, 6)])
                bin_curves.append(curve)

        result = {
            "bins": [
                {
                    "range": [round(i * bin_width, 1), round((i + 1) * bin_width, 1)],
                    "count": bin_counts[i],
                    "curve": bin_curves[i],
                }
                for i in range(n_bins)
            ],
            "total_entries": n_entries,
        }

        self._lbt_cache[run_id] = {"file_size": file_size, "result": result}
        return result

    def _get_checkpoints(self, run_id=None):
        """Lightweight endpoint: just checkpoints list."""
        run_id = self._resolve_run_id(run_id)
        if not run_id:
            return {"checkpoints": []}
        run = registry.get_run(run_id)
        if not run:
            return {"checkpoints": []}
        ckpts = self._find_checkpoints(run.get("checkpoints_dir", ""), run_id, run.get("dataset_history"))
        return {"checkpoints": ckpts, "checkpoint_every": _extract_hyperparams_cached(run).get("checkpoint_every") if _extract_hyperparams_cached(run) else None, "step_offset": run.get("step_offset", 0)}

    def _get_demos(self, run_id=None):
        run_id = self._resolve_run_id(run_id)

        # Ground truth and demo prompts come from the selected run only.
        # No run / no GT generated yet → empty (UI shows "no ground truth").
        gt_source: list = []
        prompts_source: list = []
        run = registry.get_run(run_id) if run_id else None
        if run and run.get("ground_truth"):
            gt_source = run["ground_truth"]
        if run and run.get("demo_prompts"):
            prompts_source = run["demo_prompts"]

        # Read actual config files for this run to get correct prompts
        if run_id:
            run_base = RUNS_DIR / run_id
            # Read demo_cond prompts from model config (prefer resume config)
            run_model_cfg_resume = run_base.with_name(run_id + "_model_resume.json")
            run_model_cfg = run_model_cfg_resume if run_model_cfg_resume.exists() else run_base.with_name(run_id + "_model.json")
            if run_model_cfg.exists():
                try:
                    with open(run_model_cfg) as f:
                        mcfg = json.load(f)
                    demo_cond = mcfg.get("training", {}).get("demo", {}).get("demo_cond", [])
                    cfg_prompts = [d.get("prompt", "") for d in demo_cond if d.get("prompt")]
                    if cfg_prompts:
                        prompts_source = cfg_prompts
                except Exception:
                    pass

        # Enrich ground truth with spectrogram URLs. The audio/ tree lives
        # under STATE_DIR (per-instance writable state), not DASHBOARD_DIR
        # (read-only code dir), so resolve relative to STATE_DIR.
        gt_with_spec = []
        for gt in gt_source:
            entry = dict(gt)
            mp3_rel = gt["url"].lstrip("/")
            jpg_fs = STATE_DIR / mp3_rel.replace(".mp3", ".jpg")
            if jpg_fs.exists():
                entry["spectrogram_url"] = gt["url"].replace(".mp3", ".jpg")
            gt_with_spec.append(entry)
        # GT prompts: full tags from dataset, not generated/shuffled prompts
        gt_prompts_source = prompts_source
        if run and run.get("gt_prompts"):
            gt_prompts_source = run["gt_prompts"]
        elif run and run.get("dataset_id"):
            ds = datasets_registry.get_dataset(run["dataset_id"])
            if ds and ds.get("demo_prompts"):
                gt_prompts_source = ds["demo_prompts"]
        result = {
            "prompts": prompts_source,
            "gt_prompts": gt_prompts_source,
            "ground_truth": gt_with_spec,
            "demo_steps": DEMO_STEPS,
            "demo_cfg_scales": DEMO_CFG_SCALES,
            "run_id": run_id,
            "steps": [],
            "seed": None,
        }
        # Multi-dataset tracking: build ground_truth_groups when history has >1 entry
        dataset_history = run.get("dataset_history", []) if run else []
        if not dataset_history and run and run.get("dataset_id"):
            # Reconstruct history for legacy runs by comparing original dataset config
            # with current dataset_id
            current_ds_id = run["dataset_id"]
            original_ds_name = None
            if run_id:
                orig_ds_cfg_path = RUNS_DIR / f"{run_id}_dataset.json"
                if orig_ds_cfg_path.exists():
                    try:
                        with open(orig_ds_cfg_path) as f:
                            orig_cfg = json.load(f)
                        original_ds_name = orig_cfg.get("datasets", [{}])[0].get("id", "")
                    except Exception:
                        pass
            current_ds = datasets_registry.get_dataset(current_ds_id)
            current_name = current_ds["name"] if current_ds else current_ds_id
            if original_ds_name and original_ds_name != current_name:
                # Find original dataset by name match
                original_ds_id = None
                for ds_entry in datasets_registry.list_datasets():
                    if ds_entry["name"] == original_ds_name:
                        original_ds_id = ds_entry["id"]
                        break
                step_offset = run.get("step_offset", 0)
                dataset_history = [
                    {"dataset_id": original_ds_id or original_ds_name,
                     "dataset_name": original_ds_name, "from_step": 0},
                    {"dataset_id": current_ds_id,
                     "dataset_name": current_name, "from_step": step_offset},
                ]
            else:
                dataset_history = [{"dataset_id": current_ds_id,
                                    "dataset_name": current_name, "from_step": 0}]

        if len(dataset_history) > 1:
            gt_groups = []
            for si, seg in enumerate(dataset_history):
                # For first segment, use per-run GT if available
                if si == 0 and run and run.get("ground_truth"):
                    seg_gt_raw = run["ground_truth"]
                    seg_prompts = run.get("demo_prompts", [])
                    seg_gt_prompts = run.get("gt_prompts", seg_prompts)
                else:
                    seg_ds = datasets_registry.get_dataset(seg["dataset_id"])
                    seg_gt_raw = seg_ds.get("ground_truth", []) if seg_ds else []
                    seg_prompts = seg_ds.get("demo_prompts", []) if seg_ds else []
                    seg_gt_prompts = seg_prompts  # dataset-level GT always has full tags
                if seg_gt_raw:
                    seg_gt = []
                    for gt in seg_gt_raw:
                        entry = dict(gt)
                        mp3_rel = gt["url"].lstrip("/")
                        jpg_fs = STATE_DIR / mp3_rel.replace(".mp3", ".jpg")
                        if jpg_fs.exists():
                            entry["spectrogram_url"] = gt["url"].replace(".mp3", ".jpg")
                        seg_gt.append(entry)
                    gt_groups.append({
                        "dataset_name": seg["dataset_name"],
                        "from_step": seg["from_step"],
                        "ground_truth": seg_gt,
                        "prompts": seg_prompts,
                        "gt_prompts": seg_gt_prompts,
                    })
            result["ground_truth_groups"] = gt_groups

        result["multi_dataset"] = len(dataset_history) > 1

        if not run_id:
            return result

        # Parse seed from log
        if run:
            log_path = Path(run.get("log_path", ""))
            # Check all log files for this run (seed is in the first one)
            log_dir = log_path.parent
            all_logs = sorted(
                [p for p in log_dir.glob(f"{run_id}*.log") if p.suffix == ".log"],
                key=lambda p: p.stat().st_mtime,
            )
            first_log = all_logs[0] if all_logs else log_path
            if first_log.exists():
                try:
                    with open(first_log, "rb") as f:
                        head = f.read(4096).decode("utf-8", errors="replace")
                    sm = re.search(r"Seed set to (\d+)", head)
                    if sm:
                        result["seed"] = int(sm.group(1))
                except Exception:
                    pass

        run_audio_dir = AUDIO_DIR / "runs" / run_id
        if not run_audio_dir.exists():
            return result

        # Use cached step data — steps are append-only, only scan new dirs
        with DashboardHandler._demo_cache_lock:
            run_cache = DashboardHandler._demo_cache.get(run_id)
            if run_cache is None:
                run_cache = {"steps": {}}
                DashboardHandler._demo_cache[run_id] = run_cache
            cached_steps = run_cache["steps"]

        # List step dirs (single readdir, very cheap)
        try:
            dir_entries = sorted(run_audio_dir.iterdir())
        except OSError:
            return result

        for step_dir in dir_entries:
            if not step_dir.name.startswith("step_"):
                continue
            m = re.match(r"step_(\d+)", step_dir.name)
            if not m:
                continue
            step = int(m.group(1))  # raw-PT loop saves with the authoritative global_step

            if step in cached_steps:
                result["steps"].append(cached_steps[step])
                continue

            # Not cached — scan this step dir (only happens once per step)
            if not step_dir.is_dir():
                continue
            clips = []
            for mp3 in sorted(step_dir.glob("demo_*.mp3")):
                dm = re.match(r"demo_(\d+)\.mp3", mp3.name)
                if dm:
                    clip = {
                        "index": int(dm.group(1)),
                        "url": f"/audio/runs/{run_id}/{step_dir.name}/{mp3.name}",
                        "size_mb": round(mp3.stat().st_size / 1e6, 1),
                    }
                    # Add spectrogram URL if JPG exists
                    jpg_path = mp3.with_suffix(".jpg")
                    if jpg_path.exists():
                        clip["spectrogram_url"] = f"/audio/runs/{run_id}/{step_dir.name}/{jpg_path.name}"
                    # Read JSON sidecar for per-clip metadata
                    json_path = mp3.with_suffix(".json")
                    if json_path.exists():
                        try:
                            with open(json_path) as jf:
                                clip["meta"] = json.load(jf)
                        except Exception:
                            pass
                    clips.append(clip)
            # ARC demo clip
            arc_path = step_dir / "demo_arc.mp3"
            if arc_path.exists():
                arc_clip = {
                    "index": "arc",
                    "url": f"/audio/runs/{run_id}/{step_dir.name}/demo_arc.mp3",
                    "size_mb": round(arc_path.stat().st_size / 1e6, 1),
                }
                arc_jpg = step_dir / "demo_arc.jpg"
                if arc_jpg.exists():
                    arc_clip["spectrogram_url"] = f"/audio/runs/{run_id}/{step_dir.name}/demo_arc.jpg"
                arc_json = step_dir / "demo_arc.json"
                if arc_json.exists():
                    try:
                        with open(arc_json) as jf:
                            arc_clip["meta"] = json.load(jf)
                    except Exception:
                        pass
                clips.append(arc_clip)
            step_entry = {"step": step, "clips": clips}
            seg = _dataset_for_step(dataset_history, step)
            step_entry["dataset_name"] = seg["dataset_name"] if seg else None

            # Only cache if the step has clips (in-progress demo gen may have empty dir)
            if clips:
                with DashboardHandler._demo_cache_lock:
                    cached_steps[step] = step_entry

            result["steps"].append(step_entry)
        return result

    def _serve_index(self):
        return (Path(__file__).parent / "index.html").read_bytes()

    def log_message(self, format, *args):
        pass


def _resolve_backend_name():
    """Return what underfit.backends.get_backend() will resolve to for child procs.
    Mirrors the logic in underfit.backends.__init__._autodetect / get_backend.

    When a backend's checkout is on disk but the *package* isn't importable
    (typical after `uv sync` wipes a wizard-installed editable), point the
    user at the install command — telling them to `set UNDERFIT_BACKEND=sa3`
    in that situation is actively misleading because the env var won't make
    sa3 importable, the run will just crash deeper in the call stack.
    """
    import importlib.util

    explicit = os.environ.get("UNDERFIT_BACKEND", "").strip()
    if explicit and explicit != "auto":
        # User picked one — still warn if the package isn't actually importable.
        modname = {"sa3": "stable_audio_3", "sat_dev": "stable_audio_tools"}.get(explicit)
        if modname and importlib.util.find_spec(modname) is None:
            checkout_dir = {
                "sa3":     str(BASE_DIR.parent / "stable-audio-3"),
                "sat_dev": str(BASE_DIR.parent / "stable-audio-tools"),
            }.get(explicit, "")
            extras = {"sa3": "[lora,ui]", "sat_dev": "[train,ui]"}.get(explicit, "")
            hint = (f" — {modname!r} is not installed. "
                    f"Run: .venv/bin/uv pip install -e {checkout_dir}{extras}")
            return f"{explicit}  ⚠ {hint}"
        return explicit

    # Auto: prefer sa3 if importable, then sat_dev.
    if importlib.util.find_spec("stable_audio_3") is not None:
        return "sa3"
    if importlib.util.find_spec("stable_audio_tools") is not None:
        return "sat_dev"

    # Neither importable. Walk the candidate checkouts to give an install hint.
    candidates = [
        ("sa3",     "stable_audio_3",     BASE_DIR.parent / "stable-audio-3",          "[lora,ui]"),
        ("sat_dev", "stable_audio_tools", BASE_DIR.parent / "stable-audio-tools",      "[train,ui]"),
    ]
    on_disk = [(b, mod, p, ex) for b, mod, p, ex in candidates
               if (p / mod).is_dir()]
    if on_disk:
        lines = ["NONE_IMPORTABLE  ⚠ no backend package is installed in this venv."]
        for b, mod, p, ex in on_disk:
            lines.append(f"   {b}: checkout at {p} — install with:")
            lines.append(f"       .venv/bin/uv pip install -e {p}{ex}")
        lines.append("   (or re-run: .venv/bin/underfit-setup)")
        return "\n  ".join(lines)
    return ("NONE_IMPORTABLE  ⚠ no backend package or checkout found. "
            "Run .venv/bin/underfit-setup to install one.")


class _Tee:
    """Write to two streams (terminal + log file). Line-buffered enough that
    the file is readable while the dashboard is still running."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def _tee_stdio_to_server_log():
    """Mirror stdout + stderr into SERVER_LOG_FILE so the dashboard can serve
    its own log via /api/server_log. Truncates the file at startup so each
    fresh dashboard launch starts clean (the file is diagnostic, not durable)."""
    try:
        SERVER_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        f = open(SERVER_LOG_FILE, "w", buffering=1)  # line-buffered
    except OSError as e:
        print(f"(could not open server log {SERVER_LOG_FILE}: {e})", flush=True)
        return
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)


if __name__ == "__main__":
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIO_DIR / "runs").mkdir(parents=True, exist_ok=True)
    _tee_stdio_to_server_log()
    print(f"Backend: {_resolve_backend_name()}", flush=True)
    # Warm up nvidia-smi BEFORE the HTTP server starts accepting requests.
    # First nvidia-smi call on a fresh Colab VM can take 5–10 s while the
    # driver initializes; without this the frontend would show "No GPUs"
    # on the first /api/gpu poll and only update on the next one.
    _n_gpu = _get_gpu_count()
    print(f"Detected {_n_gpu} CUDA GPU(s) via nvidia-smi", flush=True)
    _load_gradio_vram_estimate()
    print(f"Gradio VRAM estimate: {_gradio_vram}")
    n = _recover_orphaned_gradios()
    if n:
        print(f"Recovered {n} orphaned Gradio instance(s)")
        _discover_missing_share_urls()
    print("Starting watcher thread (will process demos in background)...")
    t = threading.Thread(target=demo_watcher, daemon=True)
    t.start()
    tm = threading.Thread(target=training_monitor.monitor_loop, daemon=True)
    tm.start()
    print("Training monitor started.")
    em = threading.Thread(target=encoding_monitor.monitor_loop, daemon=True)
    em.start()
    print("Encoding monitor started.")
    threading.Thread(target=_validate_datasets_on_startup, daemon=True).start()
    print(f"Datasets: {len(datasets_registry.list_datasets())} registered")
    vt = threading.Thread(target=vram_sampler, daemon=True)
    vt.start()
    print("VRAM sampler started.")
    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            # Browsers routinely abort partial audio / image loads when the
            # user clicks away — those raise ConnectionResetError or
            # BrokenPipeError inside the handler. The default behavior prints
            # a multi-line traceback to stderr (and thus into our server log).
            # Swallow those specifically; let everything else fall through.
            exc = sys.exc_info()[1]
            if isinstance(exc, (ConnectionResetError, BrokenPipeError)):
                return
            super().handle_error(request, client_address)
    # If the requested port is taken, walk forward to find the next free one.
    # 50 ports is more than enough for any reasonable contention; bigger
    # than that and there's clearly something else wrong on the box.
    server = None
    bind_port = PORT
    for offset in range(50):
        try:
            server = ThreadedHTTPServer(("0.0.0.0", bind_port + offset), DashboardHandler)
            if offset:
                print(f"Port {PORT} taken — using {bind_port + offset} instead.")
            break
        except OSError as e:
            if e.errno != 98:  # EADDRINUSE
                raise
    if server is None:
        raise RuntimeError(f"Could not find a free port in {PORT}..{PORT + 49}")
    actual_port = server.server_address[1]
    print(f"Dashboard running on http://0.0.0.0:{actual_port}")
    server.serve_forever()
