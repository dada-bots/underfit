"""LoRA setup for the raw-PyTorch training loop.

Adapter-agnostic: takes a backend module so the same code path works for
both sat and sa3.
"""
from functools import partial
from pathlib import Path

import torch


def apply_lora_from_config(backend, model, lora_config, lora_state_dict=None,
                          base_precision=None, svd_bases_path=None):
    """Mirror DiffusionCondTrainingWrapper's LoRA init in raw PyTorch.

    Returns the list of LoRA params (for optimizer construction) and a config
    dict suitable for save_lora_safetensors.
    """
    lora_mod = backend.lora_module()
    LoRAParametrization = lora_mod.LoRAParametrization
    add_lora = lora_mod.add_lora
    get_lora_params = lora_mod.get_lora_params
    get_lora_layers = lora_mod.get_lora_layers
    resolve_adapter_type = lora_mod.resolve_adapter_type
    prepare_dora_state_dict = lora_mod.prepare_dora_state_dict
    cast_base_to_precision = lora_mod.cast_base_to_precision

    # Freeze base
    model.model.eval().requires_grad_(False)
    model.conditioner.eval().requires_grad_(False)

    rank = lora_config.get("rank", 8)
    lora_alpha = lora_config.get("alpha", rank)
    adapter_type = lora_config.get("adapter_type", "lora")
    include = lora_config.get("include")
    exclude = lora_config.get("exclude")
    adapter_type = resolve_adapter_type(adapter_type, lora_state_dict)

    print(f"LoRA config: rank={rank}, alpha={lora_alpha}, adapter_type={adapter_type}")
    if include:
        print(f"  include: {include}")
    if exclude:
        print(f"  exclude: {exclude}")

    svd_bases = None
    if adapter_type.endswith("-xs"):
        if svd_bases_path is not None:
            print(f"Loading SVD bases from {svd_bases_path}")
            svd_bases = torch.load(svd_bases_path, map_location="cpu", weights_only=True)
        else:
            print("WARNING: -XS adapter without svd_bases_path; SVD will be computed per layer")

    layer_config = {
        torch.nn.Linear: {
            "weight": partial(LoRAParametrization.from_linear, rank=rank,
                              lora_alpha=lora_alpha, adapter_type=adapter_type),
        },
        torch.nn.Conv1d: {
            "weight": partial(LoRAParametrization.from_conv1d, rank=rank,
                              lora_alpha=lora_alpha, adapter_type=adapter_type),
        },
    }
    add_lora(model.model, layer_config, include=include, exclude=exclude, svd_bases=svd_bases)
    add_lora(model.conditioner, layer_config, include=include, exclude=exclude, svd_bases=svd_bases)
    print("lora layers:", len(get_lora_layers(model)))

    if lora_state_dict is not None:
        prepare_dora_state_dict(lora_state_dict)
        model.model.load_state_dict(lora_state_dict, strict=False)
        model.conditioner.load_state_dict(lora_state_dict, strict=False)

    if base_precision:
        cast_base_to_precision(model.model, base_precision)
        cast_base_to_precision(model.conditioner, base_precision)
        if model.pretransform is not None:
            dtype = torch.bfloat16 if base_precision in ("bf16", "bfloat16") else torch.float16
            model.pretransform.to(dtype)

    lora_params = list(get_lora_params(model.model)) + list(get_lora_params(model.conditioner))
    saved_config = {
        "rank": rank,
        "alpha": lora_alpha,
        "adapter_type": adapter_type,
        "include": include,
        "exclude": exclude,
    }
    return lora_params, saved_config


def save_lora_step(backend, model, lora_save_config, out_path,
                   *, step=None, epoch=None, base_model=None):
    """Save LoRA weights to out_path as a .safetensors file with config metadata.

    `step` and `epoch` are folded into the saved metadata (under the "step"
    and "epoch" keys inside the lora_config metadata blob), so resumes can
    recover them without parsing the filename.

    `base_model` (e.g. "sa3-medium") goes into metadata too — used by the
    dashboard's "Start from a previous LoRA" upload flow to verify the seed
    is shape-compatible with the user's selected base model.
    """
    lora_mod = backend.lora_module()
    state_dict = {
        **lora_mod.get_lora_state_dict(model.model),
        **lora_mod.get_lora_state_dict(model.conditioner),
    }
    enriched_cfg = dict(lora_save_config)
    if step is not None:
        enriched_cfg["step"] = int(step)
    if epoch is not None:
        enriched_cfg["epoch"] = int(epoch)
    if base_model:
        enriched_cfg["base_model"] = str(base_model)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    lora_mod.save_lora_safetensors(state_dict, enriched_cfg, out_path)


def load_lora_resume(backend, ckpt_path):
    """Read a .safetensors LoRA checkpoint and return (state_dict, config_dict)."""
    lora_mod = backend.lora_module()
    return lora_mod.load_lora_checkpoint(ckpt_path)
