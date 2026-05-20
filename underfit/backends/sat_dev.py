"""SAT-dev (stable-audio-tools) backend adapter.

Thin wrappers around the existing imports. All Underfit code calls these
through `underfit.backends.get_backend()` so the SA3 adapter can mirror them.
"""
import json

import torch

from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.lora import load_and_apply_loras
from stable_audio_tools.inference.sampling import sample_diffusion
from stable_audio_tools.data.dataset import create_dataloader_from_config

from underfit.utils import copy_state_dict, load_ckpt_state_dict, unwrap_state_dict


NAME = "sat_dev"


def create_model(model_config):
    """Construct an empty model from a parsed config dict (no weights loaded)."""
    return create_model_from_config(model_config)


def load_state_into(model, state_dict, model_type=None):
    """Unwrap any training-wrapper prefixes and copy weights into model."""
    if model_type is not None:
        state_dict = unwrap_state_dict(state_dict, model_type)
    copy_state_dict(model, state_dict)


def load_model(config_path, ckpt_path, device="cuda", half=False):
    """Load model + parsed config from disk paths.

    Returns (model, model_config). Model is moved to device, set to eval, with
    requires_grad disabled — caller must re-enable grads for training params.
    """
    with open(config_path) as f:
        model_config = json.load(f)
    model = create_model(model_config)
    state_dict = load_ckpt_state_dict(ckpt_path)
    load_state_into(model, state_dict, model_type=model_config.get("model_type"))
    model.to(device).eval().requires_grad_(False)
    if half:
        model.to(torch.float16)
    return model, model_config


def apply_loras(model, lora_paths, model_type, svd_bases_path=None):
    return load_and_apply_loras(model, lora_paths, model_type, svd_bases_path=svd_bases_path)


def encode_conditioning(model, conditioning, device):
    return model.conditioner(conditioning, device)


def get_conditioning_inputs(model, conditioning_tensors, negative=False):
    return model.get_conditioning_inputs(conditioning_tensors, negative=negative)


def sample(model, noise, cond_inputs, **kwargs):
    """Run the diffusion sampler. `model` is the parent DiffusionCond object;
    we extract `.model` (the inner DiT) and `.pretransform` here."""
    return sample_diffusion(
        model=model.model,
        noise=noise,
        cond_inputs=cond_inputs,
        pretransform=model.pretransform,
        **kwargs,
    )


def demo_sample(model, model_config, cond_list, *, steps, cfg_scale,
                seed=None, dist_shift=None, diffusion_objective_override=None,
                duration_latents=None, sample_rate=44100):
    """Demo sampling for SAT-dev. Mirrors the existing sample_diffusion-based
    behavior since SAT-dev demos historically work without the SA3
    pipeline-style chunk alignment. Kept as a separate function so the
    backends present a unified surface for demo_step.

    `duration_latents` is the desired latent length (no chunk alignment
    applied — SAT-dev's encoders don't require it).
    """
    import torch
    io_channels = model.io_channels
    device = next(model.parameters()).device
    if seed is not None:
        gen = torch.Generator(device=device); gen.manual_seed(seed)
        noise = torch.randn(1, io_channels, duration_latents, device=device, generator=gen)
    else:
        noise = torch.randn(1, io_channels, duration_latents, device=device)
    noise = noise.to(next(model.parameters()).dtype)

    saved_obj = None
    if diffusion_objective_override is not None:
        saved_obj = model.diffusion_objective
        model.diffusion_objective = diffusion_objective_override
    try:
        cond_tensors = encode_conditioning(model, cond_list, device)
        # Inject zero inpaint placeholders for models that declare them in
        # local_add_cond_ids (sa3-small / sa3-sm-* etc). The conditioner doesn't
        # produce these — the inference pipeline normally does, but demo_step
        # bypasses the pipeline. Without this the DiT.forward raises
        # KeyError: 'inpaint_mask' inside the local_add_cond cat.
        _local_ids = getattr(model, "local_add_cond_ids", []) or []
        if "inpaint_mask" in _local_ids and "inpaint_mask" not in cond_tensors:
            cond_tensors["inpaint_mask"] = [torch.zeros((1, 1, duration_latents), device=device, dtype=noise.dtype)]
        if "inpaint_masked_input" in _local_ids and "inpaint_masked_input" not in cond_tensors:
            cond_tensors["inpaint_masked_input"] = [torch.zeros((1, io_channels, duration_latents), device=device, dtype=noise.dtype)]
        cond_inputs = get_conditioning_inputs(model, cond_tensors)
        return sample(
            model, noise=noise, cond_inputs=cond_inputs,
            diffusion_objective=diffusion_objective_override or model.diffusion_objective,
            steps=steps, cfg_scale=cfg_scale, conditioning=cond_list,
            sample_rate=sample_rate,
            mask_padding_attention=getattr(model, "mask_padding_attention", False),
            use_effective_length_for_schedule=getattr(model, "use_effective_length_for_schedule", False),
            headroom_seconds=5.0,
            dist_shift=dist_shift if dist_shift is not None else model.dist_shift,
            batch_cfg=True, decode=True,
        )
    finally:
        if saved_obj is not None:
            model.diffusion_objective = saved_obj


def create_dataloader(dataset_config, **kwargs):
    return create_dataloader_from_config(dataset_config, **kwargs)


def create_gradio_ui(*, model_config_path=None, ckpt_path=None, pretrained_name=None,
                     pretransform_ckpt_path=None, model_half=False, gradio_title=None,
                     lora_ckpt_paths=None, default_prompt=None):
    """Build the gradio interface (model loading + UI construction) for SAT-dev."""
    from stable_audio_tools.interface.gradio import create_ui
    return create_ui(
        model_config_path=model_config_path,
        ckpt_path=ckpt_path,
        pretrained_name=pretrained_name,
        pretransform_ckpt_path=pretransform_ckpt_path,
        model_half=model_half,
        gradio_title=gradio_title,
        lora_ckpt_paths=lora_ckpt_paths,
        default_prompt=default_prompt,
    )


def create_training_wrapper(model_config, model):
    """Lightning training wrapper. Kept for legacy callers; the raw-PyTorch
    loop in underfit.training.loop bypasses this entirely."""
    from stable_audio_tools.training import create_training_wrapper_from_config
    return create_training_wrapper_from_config(model_config, model)


def lora_module():
    """Return the backend's models.lora module, exposing add_lora,
    LoRAParametrization, get_lora_params, get_lora_state_dict,
    save_lora_safetensors, resolve_adapter_type, cast_base_to_precision,
    prepare_dora_state_dict, get_lora_layers, etc."""
    import stable_audio_tools.models.lora as m
    return m


def random_inpaint_mask(*args, **kwargs):
    """Generate a random inpainting mask + masked input for a batch."""
    from stable_audio_tools.models.inpainting import random_inpaint_mask as _f
    return _f(*args, **kwargs)


def inference_sampling_module():
    """Return the backend's inference.sampling module (for build_schedule, etc.)."""
    import stable_audio_tools.inference.sampling as m
    return m


def build_pretransform(pretransform_config, sample_rate):
    """Construct a pretransform from its config block. Used by pre_encode.py.

    Stubs out stable_audio_tools.models.diffusion before importing autoencoders
    — that module has a top-level `from .diffusion import …` that segfaults
    with certain package versions, and we don't actually need diffusion classes
    for pre-encoding."""
    import sys
    import types as _types

    pt_type = pretransform_config.get("type")

    if pt_type == "patched":
        from stable_audio_tools.models.pretransforms import PatchedPretransform
        return PatchedPretransform(**pretransform_config["config"])

    if pt_type != "autoencoder":
        from stable_audio_tools.models.factory import create_pretransform_from_config
        return create_pretransform_from_config(pretransform_config, sample_rate)

    _blocked = "stable_audio_tools.models.diffusion"
    _was_blocked = _blocked in sys.modules
    if not _was_blocked:
        _stub = _types.ModuleType(_blocked)
        for _name in ("ConditionedDiffusionModel", "DAU1DCondWrapper",
                      "UNet1DCondWrapper", "DiTWrapper"):
            setattr(_stub, _name, None)
        sys.modules[_blocked] = _stub

    from stable_audio_tools.models.autoencoders import (
        OobleckEncoder, OobleckDecoder,
        SAMEEncoder, SAMEDecoder,
        AudioAutoencoder,
    )
    from stable_audio_tools.models.pretransforms import AutoencoderPretransform
    from stable_audio_tools.models.factory import create_bottleneck_from_config

    if not _was_blocked:
        del sys.modules[_blocked]

    _ENCODERS = {"oobleck": OobleckEncoder, "same": SAMEEncoder, "taae_v2": SAMEEncoder}
    _DECODERS = {"oobleck": OobleckDecoder, "same": SAMEDecoder, "taae_v2": SAMEDecoder}

    ae_config = pretransform_config["config"]

    enc_cfg = ae_config["encoder"]
    encoder = _ENCODERS[enc_cfg["type"]](**enc_cfg["config"])
    if not enc_cfg.get("requires_grad", True):
        for p in encoder.parameters():
            p.requires_grad = False

    dec_cfg = ae_config["decoder"]
    decoder = _DECODERS[dec_cfg["type"]](**dec_cfg["config"])
    if not dec_cfg.get("requires_grad", True):
        for p in decoder.parameters():
            p.requires_grad = False

    bottleneck = ae_config.get("bottleneck")
    if bottleneck is not None:
        bottleneck = create_bottleneck_from_config(bottleneck)

    inner_pretransform = ae_config.get("pretransform")
    if inner_pretransform is not None:
        inner_pretransform = build_pretransform(inner_pretransform, sample_rate)

    autoencoder = AudioAutoencoder(
        encoder, decoder,
        io_channels=ae_config["io_channels"],
        latent_dim=ae_config["latent_dim"],
        downsampling_ratio=ae_config["downsampling_ratio"],
        sample_rate=sample_rate,
        bottleneck=bottleneck,
        pretransform=inner_pretransform,
        in_channels=ae_config.get("in_channels"),
        out_channels=ae_config.get("out_channels"),
        soft_clip=dec_cfg.get("soft_clip", False),
    )

    return AutoencoderPretransform(
        autoencoder,
        scale=pretransform_config.get("scale", 1.0),
        model_half=pretransform_config.get("model_half", False),
        iterate_batch=pretransform_config.get("iterate_batch", False),
        chunked=pretransform_config.get("chunked", False),
    )


