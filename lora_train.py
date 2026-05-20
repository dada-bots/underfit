#!/usr/bin/env python3
"""LoRA training launcher for the Underfit dashboard.

Thin wrapper around underfit.training.run_training. The training loop is
backend-agnostic (sat_dev or sa3) and lives in underfit/training/loop.py.
"""
import os
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
                  help="sat_dev | sa3 (default: env UNDERFIT_BACKEND or auto)")
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
    from underfit.utils import check_attention_compute_capability
    check_attention_compute_capability()
    backend = get_backend(args.backend)
    run_training(args, backend)


if __name__ == "__main__":
    main()
