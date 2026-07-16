"""Device-neutral helpers: CUDA > MPS > CPU.

The raw-PT training loop and demo sampler route device selection and AMP
(autocast / GradScaler) construction through here instead of hardcoding
"cuda". Backend-agnostic on purpose — works for both the sa3 and sat
backends without importing either.

Empirically verified on torch 2.13.0 + macOS 15 (Apple Silicon):
  - torch.amp.autocast("mps", dtype=torch.float16) works
  - torch.amp.autocast("mps", dtype=torch.bfloat16) works (macOS 14+ required)
  - torch.amp.GradScaler("mps") works (scale/unscale_/step/update)
  - DataLoader(pin_memory=True) is not supported on MPS (torch warns and
    ignores it) — resolve_pin_memory() turns it off outside CUDA.
"""
import functools

import torch


def resolve_device(preference=None) -> str:
    """Best available device string: explicit preference > cuda > mps > cpu."""
    if preference:
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_type_of(device) -> str:
    """'cuda:0' -> 'cuda', torch.device -> its .type, None -> resolved."""
    return torch.device(device or resolve_device()).type


@functools.lru_cache(maxsize=None)
def mps_bf16_supported() -> bool:
    """bf16 on MPS needs macOS 14+; probe once with a tiny op."""
    if not torch.backends.mps.is_available():
        return False
    try:
        x = torch.ones(2, 2, device="mps", dtype=torch.bfloat16)
        (x @ x).sum().item()
        return True
    except Exception:
        return False


def autocast_context(device=None, dtype=None, enabled=True):
    """torch.amp.autocast pinned to the device the model actually lives on.

    bf16 on MPS is downgraded to fp16 (with a printed note) when the OS
    doesn't support it.
    """
    dt = device_type_of(device)
    if dt == "mps" and dtype is torch.bfloat16 and not mps_bf16_supported():
        print(
            "[device] bf16 autocast unsupported on this macOS/MPS build; "
            "falling back to fp16 autocast",
            flush=True,
        )
        dtype = torch.float16
    if dt == "cpu":
        # Historically these contexts were autocast("cuda"), a silent no-op on
        # CPU-only hosts. Keep CPU runs in fp32 rather than newly enabling CPU
        # fp16/bf16 autocast (slow and numerically different).
        enabled = False
    return torch.amp.autocast(dt, dtype=dtype, enabled=enabled)


class NoOpGradScaler:
    """Transparent stand-in when torch.amp.GradScaler doesn't support a backend."""

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer, *args, **kwargs):
        return optimizer.step(*args, **kwargs)

    def update(self, new_scale=None):
        pass

    def get_scale(self):
        return 1.0

    def is_enabled(self):
        return False

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


def make_grad_scaler(device=None, enabled=True):
    """GradScaler for the given backend.

    torch 2.13 supports GradScaler("mps") natively (verified). On CPU the
    scaler is constructed disabled (matches the old GradScaler("cuda")
    auto-disable-when-no-CUDA behavior). If construction fails on an older
    torch, returns a no-op shim with a printed note.
    """
    dt = device_type_of(device)
    if dt == "cpu":
        enabled = False
    try:
        return torch.amp.GradScaler(dt, enabled=enabled)
    except Exception as e:
        if enabled:
            print(
                f"[device] torch.amp.GradScaler({dt!r}) unsupported "
                f"({type(e).__name__}: {e}); using no-op scaler — fp16 loss "
                "scaling disabled, watch for gradient underflow",
                flush=True,
            )
        return NoOpGradScaler()


def resolve_pin_memory(requested: bool, device=None) -> bool:
    """pin_memory is a CUDA-only DataLoader optimization; MPS warns and
    ignores it, direct .pin_memory() calls raise. Auto-off outside CUDA."""
    return bool(requested) and device_type_of(device) == "cuda"


def empty_device_cache(device=None):
    """Release cached allocator memory on the active accelerator, if any."""
    dt = device_type_of(device)
    if dt == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif dt == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
