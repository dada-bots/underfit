"""Detect GPU capability quirks and pre-warn the user instead of letting torch
spew autotune warnings mid-run.

Currently catches: pre-Ampere GPUs (compute capability < 8.0) hitting
flex_attention's Triton kernel, which requires SM80+. The kernel falls back
to a slow eager path and emits W0522-style logs that look terrifying but
are harmless. We replace that wall with a single readable warning."""
from __future__ import annotations


def _silence_compile_noise() -> None:
    """Silence torch's autotune / dynamo / inductor log channels.

    These emit W/E-level messages when torch.compile falls back to eager —
    which is a graceful degradation, NOT a real failure. The fallback runs
    the eager-mode path and training/encoding completes normally. The
    messages look terrifying (multi-screen stack traces) but they're noise.

    We silence the entire torch._dynamo and torch._inductor logger trees.
    Real runtime errors come through different loggers and still surface."""
    import logging
    for mod in (
        "torch._dynamo",
        "torch._inductor",
        "torch._inductor.select_algorithm",
        "torch._dynamo.convert_frame",
    ):
        logging.getLogger(mod).setLevel(logging.CRITICAL)


def check_attention_compute_capability() -> bool | None:
    """Return True if the GPU supports flex_attention's compiled path
    (compute capability >= 8.0), False if older, None if no GPU.

    Always silences torch.compile/autotune log noise (the W0522/E0522 walls).
    Prints a friendly heads-up if the GPU is pre-Ampere — those users will
    hit the eager fallback path more often than modern-GPU users would.
    """
    _silence_compile_noise()

    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    compatible = major >= 8

    if compatible:
        print(f"GPU: {name} (sm{major}{minor}, compute capability {major}.{minor})", flush=True)
        return True

    print(
        f"⚠️  Older GPU detected: {name} (sm{major}{minor}). "
        f"flex_attention / flash_attention may not work; expect slower runs than L4 / A100 / H100.",
        flush=True,
    )
    return False
