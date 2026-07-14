"""Stable-Audio-3 backend adapter.

Mirrors the SAT-dev adapter's free-function surface. Imports are deferred so
this module can be inspected without stable_audio_3 installed.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

from underfit.utils import copy_state_dict, unwrap_state_dict


NAME = "sa3"

# Auto-add a local stable-audio-3 checkout to sys.path if not already
# importable. Defaults to a sibling clone next to underfit (where
# `underfit-setup --backend sa3` clones to). Users with the package
# installed in the venv get hit by the importlib check below and skip this.
_APP_ROOT = Path(__file__).resolve().parent.parent.parent
_SA3_LOCAL = str(Path(__file__).resolve().parent.parent.parent.parent / "stable-audio-3")


def _resolve_app_relative_path(value):
    if not value:
        return value
    p = Path(value)
    if p.is_absolute():
        return str(p)
    cwd_path = Path.cwd() / p
    app_path = _APP_ROOT / p
    if cwd_path.exists() and not app_path.exists():
        return str(cwd_path)
    return str(app_path)


def _require_sa3():
    if importlib.util.find_spec("stable_audio_3") is None:
        if os.path.isdir(os.path.join(_SA3_LOCAL, "stable_audio_3")) and _SA3_LOCAL not in sys.path:
            sys.path.insert(0, _SA3_LOCAL)
        if importlib.util.find_spec("stable_audio_3") is None:
            raise ImportError(
                "stable_audio_3 not importable. Either pip install -e "
                f"{_SA3_LOCAL!r} or set UNDERFIT_BACKEND=sat."
            )
    _patch_sa3_compatibility()


# SA3's T5GemmaConditioner allowlist defaults to ("google/t5gemma-b-b-ul2",)
# in the current main; SAT-dev-authored configs commonly specify
# "stabilityai/t5gemma-b-b-ul2". Rather than rewriting on-disk configs,
# extend the allowlist at import time.
_SA3_T5GEMMA_EXTRAS = [
    ("stabilityai/t5gemma-b-b-ul2", 768),
]
_sa3_patched = False


def _patch_sa3_compatibility():
    """Extend SA3's T5Gemma model-name allowlist so SAT-dev-authored configs
    pass the new conditioner's assertion. Idempotent.

    Used to also patch StableAudioPipeline._encode_audio_input to cast audio
    to the pretransform's dtype; the new model.StableAudioModel does that
    itself (model.py:449), so the dtype patch is no longer needed.
    """
    global _sa3_patched
    if _sa3_patched:
        return
    from stable_audio_3.models import conditioners as _c
    cls = getattr(_c, "T5GemmaConditioner", None)
    if cls is not None:
        for name, dim in _SA3_T5GEMMA_EXTRAS:
            if name not in cls.T5GEMMA_MODELS:
                cls.T5GEMMA_MODELS.append(name)
            cls.T5GEMMA_MODEL_DIMS.setdefault(name, dim)
    _sa3_patched = True


def _load_ckpt_state_dict(ckpt_path):
    """Vendored equivalent of the old stable_audio_3.loading_utils.load_ckpt_state_dict.

    The new SA3 codebase only loads .safetensors via safetensors.torch.load_file
    (model.py uses load_file directly); old SAT-dev .ckpt files still need
    torch.load. Handle both here so SAT-dev-trained checkpoints work.
    """
    s = str(ckpt_path)
    if s.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(s)
    sd = torch.load(s, map_location="cpu", weights_only=True)
    return sd.get("state_dict", sd)


def _normalize_for_sa3(cfg):
    """Rewrite SAT-dev-style fields in a parsed model_config dict to forms SA3
    accepts. Currently a no-op (compatibility handled via _patch_sa3_compatibility);
    kept as the place to put any future deltas that can't be monkey-patched."""
    return cfg


def create_model(model_config):
    _require_sa3()
    from stable_audio_3.factory import create_diffusion_cond_from_config
    return create_diffusion_cond_from_config(_normalize_for_sa3(model_config))


def load_state_into(model, state_dict, model_type=None):
    """Load a state dict into a parent model. Strips Lightning training-wrapper
    prefixes (unwrap_state_dict) and remaps keys with extra nesting like
    pretransform.model.* -> pretransform.* (SAT-dev wraps the autoencoder one
    level deeper than SA3 does)."""
    _require_sa3()
    from stable_audio_3.loading_utils import remap_state_dict_keys
    if model_type is not None:
        state_dict = unwrap_state_dict(state_dict, model_type)
    state_dict = remap_state_dict_keys(state_dict, model.state_dict())
    copy_state_dict(model, state_dict)


def load_model(config_path, ckpt_path, device=None, half=False):
    """Load model + parsed config from disk paths.

    Composes SA3's public primitives directly (rather than using
    StableAudioModel.from_pretrained, which is registry-only) so we can
    accept arbitrary checkpoint paths.

    For safetensors checkpoints, weights are streamed tensor-by-tensor into
    the model (peak CPU RAM ~one tensor instead of the full state_dict) —
    important on memory-constrained hosts like Colab T4 (13 GB RAM)."""
    _require_sa3()
    from stable_audio_3.factory import create_diffusion_cond_from_config
    from stable_audio_3.loading_utils import remap_state_dict_keys
    from underfit.utils import stream_checkpoint_into_model
    from underfit.utils.device import device_type_of, resolve_device
    with open(config_path) as f:
        model_config = json.load(f)
    _normalize_for_sa3(model_config)
    # Requesting "cuda" on a CUDA-less host falls back to the best available
    # device (mps > cpu) rather than erroring.
    if device_type_of(device) == "cuda" and not torch.cuda.is_available():
        device = None
    device = resolve_device(device)
    if device_type_of(device) == "cpu":
        # fp16 inference on CPU is slow and numerically fragile; keep fp32.
        half = False
    model = create_diffusion_cond_from_config(model_config)

    # Stream safetensors weights directly to the target device; fall back to
    # bulk-load for .ckpt / .pt format (no mmap available there).
    target_device = device
    target_dtype = torch.float16 if half else None
    result = stream_checkpoint_into_model(
        model, ckpt_path, device=target_device, dtype=target_dtype,
    )
    if result is None:
        state_dict = _load_ckpt_state_dict(ckpt_path)
        state_dict = unwrap_state_dict(state_dict, model_config.get("model_type"))
        state_dict = remap_state_dict_keys(state_dict, model.state_dict())
        copy_state_dict(model, state_dict)
    model.to(device).eval().requires_grad_(False)
    if half:
        model.to(torch.float16)
    model.use_lora = False
    model.lora_names = []
    return model, model_config


def apply_loras(model, lora_paths, model_type, svd_bases_path=None):
    _require_sa3()
    from stable_audio_3.models.lora import load_and_apply_loras
    return load_and_apply_loras(model, lora_paths, model_type, svd_bases_path=svd_bases_path)


def encode_conditioning(model, conditioning, device):
    return model.conditioner(conditioning, device)


def get_conditioning_inputs(model, conditioning_tensors, negative=False):
    return model.get_conditioning_inputs(conditioning_tensors, negative=negative)


def sample(model, noise, cond_inputs, **kwargs):
    _require_sa3()
    from stable_audio_3.inference.sampling import sample_diffusion
    return sample_diffusion(
        model=model.model,
        noise=noise,
        cond_inputs=cond_inputs,
        pretransform=model.pretransform,
        **kwargs,
    )


def demo_sample(model, model_config, cond_list, *, steps, cfg_scale,
                seed=None, dist_shift=None, diffusion_objective_override=None,
                duration_latents=None, sample_rate=None):
    # duration_latents / sample_rate are accepted (and ignored) for surface
    # parity with sat.demo_sample. The pipeline derives its own latent
    # length from cond_list's seconds_total + chunk alignment, so passing a
    # fixed length isn't useful here.
    """Pipeline-style demo sampling. Routes through StableAudioModel.generate
    so chunk-alignment, padding masks, and effective-length scheduling match
    inference (gradio).

    Why this exists: the manual sample_diffusion path in demo_step.py generates
    noise at exactly seconds_total*sr/ds_ratio latents, which is below the
    model's min_length (256) for short clips and not chunk-aligned for the
    SAME encoder/decoder. The model wrapper rounds up to the next chunk-aligned
    length >= min_length, masks attention past the valid region, and truncates
    the decoded audio back to seconds_total. We delegate to it.

    `diffusion_objective_override` lets the caller force "rf_denoiser" when
    ARC weights have been swapped into a wrapper whose stored objective is
    "rectified_flow" (we mutate the attr just for this call, then restore).

    Returns float audio of shape [B, C, T] truncated to seconds_total*sr.
    """
    _require_sa3()
    from stable_audio_3.model import StableAudioModel
    saved_obj = None
    if diffusion_objective_override is not None:
        saved_obj = model.diffusion_objective
        model.diffusion_objective = diffusion_objective_override
    try:
        device = str(next(model.parameters()).device)
        pipe = StableAudioModel(model, model_config, device, False)
        # We want the demo to be exactly chunk-aligned to the prompted
        # seconds_total, with no extra padding. Pipeline aligns
        # `(seconds_total + duration_padding_sec) * sr` UP to a multiple of
        # `ds_ratio * latent_align` (= 65536 audio samples = 16 latents).
        # Setting padding=0 makes seconds_total=23 produce 256 latents and
        # seconds_total=380 produce 4096 latents (the model's native length).
        # truncate_output_to_duration=False keeps the chunk-aligned tail
        # rather than cutting at the (non-aligned) seconds_total boundary.
        #
        # Pass the model's actual sample_size as the cap; pipeline's default
        # is 5_292_032 (~120s) which would silently truncate any longer
        # demo. Read from model_config so this works for any model variant.
        cap_sample_size = (model_config or {}).get("sample_size") or 16_777_216
        return pipe.generate(
            conditioning=cond_list,
            sample_size=cap_sample_size,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed if seed is not None else -1,
            dist_shift=dist_shift,
            batch_size=1,
            duration_padding_sec=0.0,
            truncate_output_to_duration=False,
        )
    finally:
        if saved_obj is not None:
            model.diffusion_objective = saved_obj


def create_dataloader(dataset_config, batch_size, sample_size, sample_rate,
                     audio_channels=2, num_workers=4, shuffle=True,
                     tokenizers=None, pad=True,
                     pin_memory=True, persistent_workers=True):
    """Construct a DataLoader for the dataset types Underfit uses.

    Currently supports: pre_encoded. SAT-dev's factory supports more types
    (audio_dir, etc.); add them here as needed.
    """
    _require_sa3()
    from stable_audio_3.data.dataset import (
        LatentDatasetConfig,
        LocalDatasetConfig,
        PreEncodedDataset,
        SampleDataset,
        collation_fn,
    )

    dataset_type = dataset_config.get("dataset_type")

    if dataset_type == "pre_encoded":
        configs = []
        for ds in dataset_config["datasets"]:
            cmf = _load_custom_metadata_fn(ds.get("custom_metadata_module"))
            configs.append(LatentDatasetConfig(
                id=ds["id"],
                path=_resolve_app_relative_path(ds["path"]),
                custom_metadata_fn=cmf,
                weight=ds.get("weight", 1.0),
                filelist_path=_resolve_app_relative_path(ds.get("filelist_path")),
            ))
        ds = PreEncodedDataset(
            configs,
            latent_crop_length=dataset_config.get("latent_crop_length"),
            min_length_sec=dataset_config.get("min_length_sec"),
            max_length_sec=dataset_config.get("max_length_sec"),
            random_crop=dataset_config.get("random_crop", False),
            tokenizers=tokenizers,
        )

    elif dataset_type == "audio_dir":
        force_channels = "mono" if audio_channels == 1 else "stereo"
        configs = []
        for ds in dataset_config["datasets"]:
            cmf = _load_custom_metadata_fn(ds.get("custom_metadata_module"))
            configs.append(LocalDatasetConfig(
                id=ds["id"],
                path=_resolve_app_relative_path(ds["path"]),
                custom_metadata_fn=cmf,
                keywords=ds.get("keywords"),
                filelist_path=_resolve_app_relative_path(ds.get("filelist_path")),
                weight=ds.get("weight", 1.0),
            ))
        ds = SampleDataset(
            configs,
            sample_size=sample_size,
            sample_rate=sample_rate,
            random_crop=dataset_config.get("random_crop", True),
            force_channels=force_channels,
            volume_norm=dataset_config.get("volume_norm", False),
            volume_norm_param=dataset_config.get("volume_norm_param", (-16, 2)),
            strip_silence=dataset_config.get("strip_silence", False),
            pad=pad,
        )
    else:
        raise NotImplementedError(
            f"sa3 backend does not support dataset_type={dataset_type!r} yet"
        )

    # Oversample with replacement when the dataset is smaller than the
    # requested batch_size. Without this, a 1-file / 8-batch run would
    # yield one short batch per epoch (effective batch size = 1). With
    # replacement sampling and random_crop=True, each draw is a fresh
    # random latent-window from the same source file, so every batch is
    # the full requested size and we get real augmentation from the
    # tiny source set. Standard pattern for overfitting tiny datasets,
    # which is the entire point of underfit.
    n_ds = len(ds)
    sampler = None
    loader_shuffle = shuffle
    if n_ds < batch_size:
        # Aim for ~100 iters per epoch so logging/LR-schedule pacing
        # don't go wild on a 1-file dataset.
        samples_per_epoch = batch_size * 100
        sampler = torch.utils.data.RandomSampler(
            ds, replacement=True, num_samples=samples_per_epoch,
        )
        loader_shuffle = False  # PyTorch: shuffle and sampler are mutually exclusive
        rc = dataset_config.get("random_crop", False)
        print(
            f"Dataset has {n_ds} file{'s' if n_ds != 1 else ''} < batch_size {batch_size}; "
            f"oversampling with replacement → {samples_per_epoch} samples/epoch "
            f"({samples_per_epoch // batch_size} iters/epoch)"
            + ("" if rc else "  (tip: enable random_crop for real augmentation)"),
            flush=True,
        )

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=loader_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collation_fn,
        # drop_last=False so a small final batch isn't silently dropped.
        # With oversampling above, every batch is already the full size
        # anyway; this matters only when len(ds) % batch_size != 0 in the
        # normal (no-replacement) regime.
        drop_last=False,
        # PyTorch raises if persistent_workers=True is paired with
        # num_workers=0; guard at the call site.
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )


def _load_custom_metadata_fn(module_path):
    if module_path is None:
        return None
    module_path = _resolve_app_relative_path(module_path)
    import importlib.util
    spec = importlib.util.spec_from_file_location("metadata_module", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_custom_metadata


def create_gradio_ui(*, model_config_path=None, ckpt_path=None, pretrained_name=None,
                     pretransform_ckpt_path=None, model_half=False, gradio_title=None,
                     lora_ckpt_paths=None, default_prompt=None):
    """Build the gradio interface for SA3.

    SA3's UI takes a StableAudioModel; we construct one from explicit paths
    when the registry isn't being used.
    """
    _require_sa3()
    from stable_audio_3.interface.diffusion_cond import create_diffusion_cond_ui
    from stable_audio_3.model import StableAudioModel

    if pretrained_name:
        pipe = StableAudioModel.from_pretrained(
            pretrained_name,
            model_half=model_half,
        )
    elif model_config_path and ckpt_path:
        # Same load + remap path as load_model() above so SAT-dev-trained
        # checkpoints with pretransform.model.* keys work.
        from underfit.utils.device import resolve_device
        device = resolve_device()
        if device == "cpu":
            model_half = False
        model, model_config = load_model(model_config_path, ckpt_path, device=device, half=model_half)
        pipe = StableAudioModel(model, model_config, device, model_half)
    else:
        raise ValueError(
            "sa3 create_gradio_ui needs either pretrained_name or both "
            "model_config_path and ckpt_path"
        )

    if lora_ckpt_paths:
        pipe.load_lora(lora_ckpt_paths)

    return create_diffusion_cond_ui(
        pipe,
        gradio_title=gradio_title or "Stable Audio",
        default_prompt=default_prompt,
    )


def create_training_wrapper(model_config, model):
    """SA3 has no Lightning wrapper. underfit.training.loop bypasses it."""
    raise NotImplementedError(
        "sa3 backend uses a raw-PyTorch training loop; "
        "call sites should not invoke create_training_wrapper for SA3"
    )


def lora_module():
    """Return the backend's models.lora module."""
    _require_sa3()
    import stable_audio_3.models.lora as m
    return m


def random_inpaint_mask(*args, **kwargs):
    """Use the vendored inpainting helper since SA3 doesn't ship one."""
    from underfit.utils.inpainting import random_inpaint_mask as _f
    return _f(*args, **kwargs)


def inference_sampling_module():
    """Return the backend's inference.sampling module (for build_schedule, etc.)."""
    _require_sa3()
    import stable_audio_3.inference.sampling as m
    return m


def build_pretransform(pretransform_config, sample_rate):
    """Construct a pretransform (VAE / autoencoder wrapper) from its config block.
    Used by `pre_encode.py` to materialize the encoder for dataset latent generation.

    `pretransform_config` is the `model.pretransform` block of model_config.json.
    SA3's factory expects the full model_config with a "pretransform" key, so we
    wrap accordingly.
    """
    _require_sa3()
    from stable_audio_3.factory import create_pretransform_from_config
    return create_pretransform_from_config(
        {"pretransform": pretransform_config}, sample_rate
    )
