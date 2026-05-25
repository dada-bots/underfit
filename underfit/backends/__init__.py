"""Underfit backend abstraction.

Selects between stable-audio-tools (sat) and stable-audio-3 (sa3).

Each backend module exposes the same set of free functions:
    load_model(config_path, ckpt_path, device, half) -> (model, model_config)
    apply_loras(model, lora_paths, model_type, svd_bases_path=None) -> list[str]
    sample(model, noise, cond_inputs, **kwargs) -> Tensor
    encode_conditioning(model, conditioning, device) -> dict
    get_conditioning_inputs(model, conditioning_tensors, negative=False) -> dict
    create_model(model_config) -> model           # for training-time construction without ckpt
    load_state_into(model, state_dict)            # in-place load with shape-matching
    create_dataloader(dataset_config, **kwargs)   # backend's own dataset helper

Selection priority:
    1. explicit `name` argument (e.g. CLI flag)
    2. UNDERFIT_BACKEND env var
    3. auto-detect: prefer sa3 if importable, else sat
"""
import importlib
import importlib.util
import os
from types import ModuleType


VALID_NAMES = ("sat", "sa3")


def _autodetect() -> str:
    if importlib.util.find_spec("stable_audio_3") is not None:
        return "sa3"
    return "sat"


def get_backend(name: str | None = None) -> ModuleType:
    """Return the chosen backend module."""
    if name is None:
        name = os.environ.get("UNDERFIT_BACKEND", "auto")
    if name == "auto":
        name = _autodetect()
    if name not in VALID_NAMES:
        raise ValueError(f"Unknown backend '{name}'. Valid: {VALID_NAMES}")
    return importlib.import_module(f"underfit.backends.{name}")
