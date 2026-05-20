"""State-dict helpers vendored from stable-audio-tools.

Pure PyTorch utilities — backend-agnostic. Used by both the SAT-dev and SA3
adapters, and by analysis scripts that need to load checkpoints directly.
"""
import torch
from safetensors.torch import load_file
from torch.nn.utils import remove_weight_norm


def copy_state_dict(model, state_dict):
    """Load state_dict into model for keys matching exactly in name and shape."""
    model_state_dict = model.state_dict()
    for key in state_dict:
        if key in model_state_dict and state_dict[key].shape == model_state_dict[key].shape:
            if isinstance(state_dict[key], torch.nn.Parameter):
                state_dict[key] = state_dict[key].data
            model_state_dict[key] = state_dict[key]
        else:
            print(f"Key {key} not found in target state_dict or shape mismatch. Skipping.")
    model.load_state_dict(model_state_dict, strict=False)


def load_ckpt_state_dict(ckpt_path):
    if ckpt_path.endswith(".safetensors"):
        return load_file(ckpt_path)
    return torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]


def remove_weight_norm_from_model(model):
    for module in model.modules():
        if hasattr(module, "weight"):
            print(f"Removing weight norm from {module}")
            remove_weight_norm(module)
    return model


WRAPPER_PREFIXES = {
    "diffusion_uncond": "diffusion.",
    "diffusion_cond": "diffusion.",
    "diffusion_cond_inpaint": "diffusion.",
    "diffusion_autoencoder": "diffusion.",
    "autoencoder": "autoencoder.",
    "lm": "lm.",
    "clap": "clap.",
    "captioner": "model.",
}


def unwrap_state_dict(state_dict, model_type):
    """Detect and strip Lightning training-wrapper prefixes from a state_dict.

    Wrapped checkpoints have keys like 'diffusion.model.xxx' or
    'diffusion_ema.ema_model.xxx'. Returns the unwrapped dict (or original if
    already unwrapped / unknown model_type).
    """
    prefix = WRAPPER_PREFIXES.get(model_type)
    if prefix is None:
        return state_dict

    has_wrapper_prefix = any(k.startswith(prefix) for k in state_dict.keys())
    if not has_wrapper_prefix:
        return state_dict

    ema_prefix = prefix.replace(".", "_ema.ema_model.")
    has_ema = any(k.startswith(ema_prefix) for k in state_dict.keys())
    ema_wraps_whole_model = model_type in ("autoencoder",)

    unwrapped = {}
    if has_ema:
        for k, v in state_dict.items():
            if k.startswith(ema_prefix):
                suffix = k[len(ema_prefix):]
                new_key = suffix if ema_wraps_whole_model else "model." + suffix
                unwrapped[new_key] = v
        if not ema_wraps_whole_model:
            conditioner_prefix = prefix + "conditioner."
            pretransform_prefix = prefix + "pretransform."
            for k, v in state_dict.items():
                if k.startswith(conditioner_prefix) or k.startswith(pretransform_prefix):
                    unwrapped[k[len(prefix):]] = v
    else:
        for k, v in state_dict.items():
            if k.startswith(prefix):
                unwrapped[k[len(prefix):]] = v
    return unwrapped
