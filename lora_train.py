#!/usr/bin/env python3
"""LoRA training launcher for the Underfit dashboard.

Thin wrapper around underfit.training.run_training. The training loop is
backend-agnostic (sat or sa3) and lives in underfit/training/loop.py.
"""
import os
import sys
import time
import traceback


def _dump_exit_reason(exc):
    """Write the unhandled exception's traceback to <log>.exit so the dashboard
    can surface it as a kill_hint even when the stdout/tee pipe got truncated
    (segfault, OOM-kill of a child, etc.). UNDERFIT_LOG_PATH is set by the
    dashboard launcher; falls back to a cwd-relative file otherwise."""
    log_path = os.environ.get("UNDERFIT_LOG_PATH") or "lora_train.log"
    exit_path = log_path + ".exit"
    try:
        with open(exit_path, "w") as f:
            f.write(f"lora_train.py exited with {type(exc).__name__}: {exc}\n\n")
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)
    except Exception:
        pass
    # Also re-print to stderr in case the pipe is still alive.
    try:
        sys.stderr.write(f"\n=== lora_train.py exited with {type(exc).__name__}: {exc} ===\n")
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        sys.stderr.flush()
    except Exception:
        pass


# Install excepthook BEFORE the rest of the imports so module-level failures
# (e.g. `from underfit.backends import get_backend` blowing up) still trigger
# the .exit sidecar dump. Without this, an ImportError during the imports
# below would bypass the try/except in __main__.
def _excepthook(exc_type, exc, tb):
    _dump_exit_reason(exc)
    # Let the default hook also print to stderr (best-effort).
    try:
        sys.__excepthook__(exc_type, exc, tb)
    except Exception:
        pass


sys.excepthook = _excepthook

# Write a 'got to python' marker so the diagnose helper can tell whether
# the failure was before python even ran (bash / venv / source issue) or
# after (python-side: ImportError, CUDA, etc.). Also records the torch+CUDA
# build so we can confirm the venv has cu128 wheels (sm_120 support).
_log_for_marker = os.environ.get("UNDERFIT_LOG_PATH") or "lora_train.log"
try:
    with open(_log_for_marker + ".started", "w") as _f:
        _f.write(f"lora_train.py reached __main__ at {time.time()}\n")
        _f.write(f"cwd: {os.getcwd()}\n")
        _f.write(f"python: {sys.executable}\n")
        try:
            import torch
            _f.write(f"torch:  {torch.__version__}  (CUDA {torch.version.cuda})\n")
            _f.write(f"archs:  {torch.cuda.get_arch_list()}\n")
            if torch.cuda.is_available():
                _f.write(f"device: {torch.cuda.get_device_name(0)} "
                         f"(sm{''.join(map(str, torch.cuda.get_device_capability(0)))})\n")
            elif torch.backends.mps.is_available():
                _f.write("device: mps (Apple Silicon)\n")
        except Exception as _e:
            _f.write(f"torch import failed: {type(_e).__name__}: {_e}\n")
except Exception:
    pass

print(f"Starting {os.path.basename(__file__)}...", flush=True)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
import configparser

from underfit.backends import get_backend
from underfit.training import run_training


def get_all_args(defaults_file="defaults.ini"):
    """Read [DEFAULTS] from a config file, expose them as argparse flags.

    Lightweight replacement for prefigure.get_all_args (no wandb dep).
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config-file", default=defaults_file)
    config_file = pre.parse_known_args()[0].config_file
    defaults = {}
    if os.path.isfile(config_file):
        cp = configparser.ConfigParser()
        cp.read(config_file)
        if cp.sections():
            defaults = dict(cp[cp.sections()[0]])
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default=defaults_file)
    p.add_argument("--wandb-config", default=None)
    p.add_argument("--backend", default=None,
                  help="sat | sa3 (default: env UNDERFIT_BACKEND or auto)")
    for key, value in defaults.items():
        arg_name = f"--{key.replace('_', '-')}"
        try:
            p.add_argument(arg_name, default=value)
        except argparse.ArgumentError:
            pass
    args, _ = p.parse_known_args()
    for key, val in vars(args).items():
        if isinstance(val, str):
            val = val.strip("'\"")
            # Bool first — int("True") would silently fall through as a
            # ValueError but "1"/"0" would coerce to int. Handle bool
            # literals explicitly before the numeric path.
            if val.lower() in ("true", "false"):
                setattr(args, key, val.lower() == "true")
                continue
            setattr(args, key, val)
            try:
                setattr(args, key, int(val))
            except ValueError:
                try:
                    setattr(args, key, float(val))
                except ValueError:
                    pass
    return args


def main():
    args = get_all_args()
    if os.environ.get("SLURM_PROCID") is not None:
        args.seed = (args.seed or 0) + int(os.environ["SLURM_PROCID"])
    # Pre-warn (and quiet torch's noisy autotune warnings) on pre-Ampere GPUs.
    from underfit.utils import check_attention_compute_capability, check_attention_backends
    check_attention_compute_capability()
    check_attention_backends()
    backend = get_backend(args.backend)
    run_training(args, backend)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as _exc:
        _dump_exit_reason(_exc)
        sys.exit(1)
