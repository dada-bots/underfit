"""Audio output helpers vendored from stable-audio-tools.

Used by demo generation to trim varlen output back to its `seconds_total`
budget before concatenation. Backend-agnostic.
"""
import torch


def compute_per_elem_trim(conditioning, sample_rate, margin_seconds=5.0):
    """Compute per-element trim lengths from seconds_total in conditioning dicts.

    Returns a list of trim lengths (in audio samples) or None if no
    seconds_total found in any element.
    """
    if not any("seconds_total" in c for c in conditioning):
        return None
    margin_samples = int(margin_seconds * sample_rate)
    per_elem_trim = []
    for c in conditioning:
        if "seconds_total" in c:
            per_elem_trim.append(int(c["seconds_total"] * sample_rate) + margin_samples)
        else:
            per_elem_trim.append(None)
    return per_elem_trim


def trim_and_concat(x, per_elem_trim):
    """Per-element trim along time axis, then concatenate.

    Trims each batch element to its own length (from seconds_total + margin),
    removing trailing padding silence before concatenation.

    Args:
        x: (b, d, n) tensor or list of (d, n) tensors
        per_elem_trim: list of trim lengths per element, or None for no trimming
    """
    items = [x[i] for i in range(x.shape[0])] if isinstance(x, torch.Tensor) and x.dim() == 3 else x
    if per_elem_trim is None:
        return torch.cat(items, dim=-1)
    parts = []
    for i, elem in enumerate(items):
        if per_elem_trim[i] is not None:
            parts.append(elem[..., :min(per_elem_trim[i], elem.shape[-1])])
        else:
            parts.append(elem)
    return torch.cat(parts, dim=-1)
