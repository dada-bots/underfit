"""Demo generation for the raw-PT training loop.

Replaces lora_dashboard_demos.LoRADashboardDemoCallback. Same on-disk output
the dashboard already ingests: demo_<i>_<step:08d>.mp3 + .json sidecar in
cwd (the dashboard launches with cwd=<run>/demos/).

Backend-agnostic: model + sampler + LoRA primitives all flow through
underfit.backends.*. Per-prompt cfg/seed/lora_strength/sigma-interval
support and ARC stacking carry over from the original Lightning callback.
"""
import gc
import json
import os
import subprocess
import sys
import typing as tp
from functools import partial

import torch
from tqdm import tqdm

# Note: torchaudio is imported lazily inside _save_demo_file. Eager-importing
# torchaudio here triggered a C-extension load-order segfault when pytorch_
# lightning was imported afterwards (via stable_audio_tools.models.lora's
# callbacks.py), in this venv. Lazy import resolves the order at call time.

from underfit.utils import compute_per_elem_trim, trim_and_concat




def _get_demo_latent_length(model, sample_size, latent_crop_length=None):
    """Latent length for demos: per-demo override > latent_crop_length > full sample_size."""
    if latent_crop_length is not None:
        return latent_crop_length
    if model.pretransform is not None:
        return sample_size // model.pretransform.downsampling_ratio
    return sample_size


def _save_demo_file(audio_int16, demo_index, step, sample_rate, meta=None):
    """Save mp3 + JSON sidecar atomically (temp + rename) so the dashboard
    never copies a half-written file."""
    import torchaudio  # lazy: see module-level note
    final_mp3 = f"demo_{demo_index}_{step:08d}.mp3"
    if os.path.exists(final_mp3):
        return
    tmp_wav = f".tmp_demo_{demo_index}_{step:08d}.wav"
    tmp_mp3 = f".tmp_demo_{demo_index}_{step:08d}.mp3"
    torchaudio.save(tmp_wav, audio_int16, sample_rate)
    subprocess.run(
        ["ffmpeg", "-i", tmp_wav, "-q:a", "0", "-y", tmp_mp3, "-loglevel", "error"],
        check=True,
    )
    os.rename(tmp_mp3, final_mp3)
    os.remove(tmp_wav)
    if meta is not None:
        with open(f"demo_{demo_index}_{step:08d}.json", "w") as f:
            json.dump(meta, f)


def _build_inpaint_zeros(model, demo_samples, device, dtype):
    """Provide all-zero inpaint conditioning when the model expects it.
    For demos this means 'unconditional from noise + prompt-only conditioning'."""
    needs_inpaint = (
        "inpaint_mask" in (getattr(model, "local_add_cond_ids", []) or [])
        or "inpaint_masked_input" in (getattr(model, "local_add_cond_ids", []) or [])
    )
    if not needs_inpaint:
        return None
    io_channels = model.io_channels
    return {
        "inpaint_mask": [torch.zeros(1, 1, demo_samples, device=device, dtype=dtype)],
        "inpaint_masked_input": [torch.zeros(1, io_channels, demo_samples, device=device, dtype=dtype)],
    }


def _generate_single_sample(
    model, backend, cond_list, cfg_scale, sample_rate,
    *, demo_steps=50, seed=None, duration_latents=None,
    diffusion_objective_override=None, dist_shift_override=None,
    model_config=None,
):
    """Generate one decoded audio clip. Returns CPU int16 [C, T]."""
    per_elem_trim = compute_per_elem_trim(cond_list, sample_rate)

    with torch.amp.autocast("cuda"):
        # Delegate to backend.demo_sample so each backend can apply its own
        # alignment / padding-mask conventions. SA3 routes through the
        # StableAudioPipeline (chunk-aligns latent length, applies
        # mask_padding_attention, truncates output) so demos match gradio
        # output. SAT-dev preserves its existing sample_diffusion path.
        fakes = backend.demo_sample(
            model,
            model_config,
            cond_list,
            steps=demo_steps,
            cfg_scale=cfg_scale,
            seed=seed,
            dist_shift=dist_shift_override if dist_shift_override is not None else model.dist_shift,
            diffusion_objective_override=diffusion_objective_override,
            duration_latents=duration_latents,
            sample_rate=sample_rate,
        )

    fakes = trim_and_concat(fakes, per_elem_trim)
    fakes = fakes.to(torch.float32)
    fakes = fakes.div(torch.max(torch.abs(fakes)).clamp(min=1e-8)).mul(32767).to(torch.int16).cpu()
    return fakes


def _generate_with_lora_strength(
    model, backend, cond_list, cfg_scale, sample_rate,
    *, demo_steps=50, lora_strength=None, seed=None, duration_latents=None,
    model_config=None,
):
    """Wrapper that temporarily overrides global LoRA strength then restores it."""
    if lora_strength is None:
        return _generate_single_sample(
            model, backend, cond_list, cfg_scale, sample_rate,
            demo_steps=demo_steps, seed=seed, duration_latents=duration_latents,
            model_config=model_config,
        )
    set_lora_strength = backend.lora_module().set_lora_strength
    try:
        set_lora_strength(model.model, lora_strength)
        set_lora_strength(model.conditioner, lora_strength)
        return _generate_single_sample(
            model, backend, cond_list, cfg_scale, sample_rate,
            demo_steps=demo_steps, seed=seed, duration_latents=duration_latents,
            model_config=model_config,
        )
    finally:
        set_lora_strength(model.model, 1.0)
        set_lora_strength(model.conditioner, 1.0)


def _generate_split_lora_sample(
    model, backend, cond_list, cfg_scale, sample_rate,
    *, demo_steps=50, lora_interval_max=None, arc_lora_interval_max=None,
    lora_strength=1.0, arc_lora_strength=1.0, seed=None,
    diffusion_objective_override=None, duration_latents=None,
):
    """Sigma-interval LoRA sampling: change LoRA strength at each timestep
    based on whether sigma is below the configured interval. Mirrors the
    old _generate_split_lora_sample from lora_dashboard_demos.py."""
    device = next(model.parameters()).device
    io_channels = model.io_channels
    sampling_mod = backend.inference_sampling_module()
    set_lora_strength = backend.lora_module().set_lora_strength

    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        noise = torch.randn(1, io_channels, duration_latents, device=device, generator=gen)
    else:
        noise = torch.randn(1, io_channels, duration_latents, device=device)
    model_dtype = next(model.parameters()).dtype
    noise = noise.to(model_dtype)

    set_lora_strength(model.model, lora_strength, lora_index=0)

    with torch.amp.autocast("cuda"):
        conditioning = backend.encode_conditioning(model, cond_list, device)
        inpaint_zeros = _build_inpaint_zeros(model, duration_latents, device, model_dtype)
        if inpaint_zeros is not None:
            conditioning.update(inpaint_zeros)
        cond_inputs = backend.get_conditioning_inputs(model, conditioning)

        sampler_args = {
            **cond_inputs,
            "cfg_scale": cfg_scale,
            "batch_cfg": True,
            "rescale_cfg": False,
            "padding_mask": None,
            "apg_scale": 1.0,
        }
        sigmas = sampling_mod.build_schedule(
            steps=demo_steps, sigma_max=1.0,
            dist_shift=model.dist_shift,
            effective_seq_len=None, fallback_seq_len=duration_latents,
            include_endpoint=True, device=device,
        )
        x = noise
        for step_i in range(demo_steps):
            t_curr = sigmas[step_i]
            t_next = sigmas[step_i + 1]
            sigma = t_curr.item()
            if lora_interval_max is not None:
                set_lora_strength(model.model, lora_strength if sigma <= lora_interval_max else 0.0, lora_index=0)
            if arc_lora_interval_max is not None:
                set_lora_strength(model.model, arc_lora_strength if sigma <= arc_lora_interval_max else 0.0, lora_index=1)
            dt = t_next - t_curr
            t_tensor = t_curr * torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)
            v = model.model(x, t_tensor, **sampler_args)
            x = x + dt * v
        # Restore default strengths
        set_lora_strength(model.model, lora_strength, lora_index=0)
        if arc_lora_interval_max is not None:
            set_lora_strength(model.model, arc_lora_strength, lora_index=1)

    if model.pretransform is not None:
        x = x.to(next(model.pretransform.parameters()).dtype)
        x = model.pretransform.decode(x)

    per_elem_trim = compute_per_elem_trim(cond_list, sample_rate)
    x = trim_and_concat(x, per_elem_trim)
    x = x.to(torch.float32).div(torch.max(torch.abs(x)).clamp(min=1e-8)).mul(32767).to(torch.int16).cpu()
    return x


def _stack_arc_lora(model, backend, arc_lora_path):
    """Add an ARC LoRA at lora_index=1 on top of the existing finetune LoRA.
    Returns nothing — caller must call _unstack_arc_lora to reverse."""
    lora = backend.lora_module()
    arc_sd, arc_cfg = lora.load_lora_checkpoint(arc_lora_path)
    arc_rank = arc_cfg.get("rank", 64)
    arc_alpha = arc_cfg.get("alpha", arc_rank)
    arc_type = arc_cfg.get("adapter_type", "dora-cols")
    layer_config = {
        torch.nn.Linear: {
            "weight": partial(lora.LoRAParametrization.from_linear,
                              rank=arc_rank, lora_alpha=arc_alpha,
                              adapter_type=arc_type, lora_index=1),
        },
        torch.nn.Conv1d: {
            "weight": partial(lora.LoRAParametrization.from_conv1d,
                              rank=arc_rank, lora_alpha=arc_alpha,
                              adapter_type=arc_type, lora_index=1),
        },
    }
    lora.add_lora(model.model, layer_config)
    lora.add_lora(model.conditioner, layer_config)
    lora.prepare_dora_state_dict(arc_sd)
    remapped = lora.remap_lora_state_dict(arc_sd, 1)
    model.model.load_state_dict(remapped, strict=False)
    model.conditioner.load_state_dict(remapped, strict=False)


def _unstack_arc_lora(model, backend):
    lora = backend.lora_module()
    lora.remove_lora_by_index(model.model, lora_index=1)
    lora.remove_lora_by_index(model.conditioner, lora_index=1)


def _swap_full_arc_weights(model, backend, arc_full_model_path, arc_full_model_config):
    """Swap base weights with an ARC full-model checkpoint. Returns saved
    originals (CPU tensors) so they can be restored after demos."""
    from underfit.utils import load_ckpt_state_dict, unwrap_state_dict
    arc_sd = load_ckpt_state_dict(arc_full_model_path)
    mt = None
    if arc_full_model_config:
        try:
            with open(arc_full_model_config) as f:
                mt = json.load(f).get("model_type")
        except Exception:
            pass
    arc_sd = unwrap_state_dict(arc_sd, mt)

    # Detect each section's prefix in the checkpoint by trying to match its
    # keys against the live target submodule's state_dict. Different sources
    # use different layouts:
    #   HF SA3 safetensors: model.model.*, conditioner.*, pretransform.model.*
    #   Lightning-wrapped:  model.model.model.*, model.conditioner.*, model.pretransform.*
    #   Bare modules:       (no prefix at all)
    # Strategy: for each known target key, try matching against keys that END
    # with that target key. The longest common prefix across matches IS the
    # source prefix. Robust to any wrapper depth.
    def _target_keys_loose(target_sd):
        """Target keys, plus their `.parametrizations.weight.original` ↔ `.weight`
        equivalents (LoRA wraps Linear weights as parametrizations)."""
        out = set()
        for k in target_sd.keys():
            out.add(k)
            if k.endswith(".parametrizations.weight.original"):
                out.add(k.replace(".parametrizations.weight.original", ".weight"))
        return out

    def _detect_prefix(arc_sd, target_sd):
        target_keys = _target_keys_loose(target_sd)
        # Find prefixes that, when stripped, give a target key. The prefix
        # with the most matches wins.
        prefix_counts = {}
        for k in arc_sd:
            for tk in target_keys:
                if k.endswith(tk) and (len(k) == len(tk) or k[-len(tk)-1] == "."):
                    prefix = k[: len(k) - len(tk)]
                    prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
                    break  # one match per source key is enough
        if not prefix_counts:
            return None
        return max(prefix_counts.items(), key=lambda kv: kv[1])[0]

    model_prefix = _detect_prefix(arc_sd, model.model.state_dict())
    cond_prefix  = _detect_prefix(arc_sd, model.conditioner.state_dict())

    def _slice(sd_in, prefix, target_sd):
        if prefix is None:
            return {}
        target_keys = _target_keys_loose(target_sd)
        out = {}
        for k, v in sd_in.items():
            if not k.startswith(prefix):
                continue
            local = k[len(prefix):]
            if local in target_keys:
                out[local] = v
        return out

    arc_model_sd = _slice(arc_sd, model_prefix, model.model.state_dict())
    arc_cond_sd  = _slice(arc_sd, cond_prefix, model.conditioner.state_dict())
    tqdm.write(
        f"  diffusion prefix={model_prefix!r} ({len(arc_model_sd)} keys), "
        f"conditioner prefix={cond_prefix!r} ({len(arc_cond_sd)} keys)",
        file=sys.stdout,
    )
    del arc_sd

    def _get_param(net, dotted_key):
        parts = dotted_key.split(".")
        obj = net
        for p in parts[:-1]:
            obj = getattr(obj, p)
        return getattr(obj, parts[-1])

    def _swap(net, new_sd):
        saved = {}
        sd_keys = set(net.state_dict().keys())
        for key, new_val in new_sd.items():
            target = key if key in sd_keys else None
            if target is None and key.endswith(".weight"):
                pkey = key.replace(".weight", ".parametrizations.weight.original")
                if pkey in sd_keys:
                    target = pkey
            if target is None:
                continue
            param = _get_param(net, target)
            # Same-size swap only — if the ARC ckpt was distilled from a
            # differently-sized base, the weights aren't shape-compatible and
            # need a separate model instantiation path. Bail with a clear
            # signal instead of silently mis-loading or crashing mid-iteration.
            if tuple(param.shape) != tuple(new_val.shape):
                raise _ArcShapeMismatch(
                    f"ARC weight shape {tuple(new_val.shape)} doesn't match base "
                    f"{tuple(param.shape)} for {target!r} — base and ARC architectures differ"
                )
            saved[target] = param.data.cpu().clone()
            param.data.copy_(new_val)
        return saved

    saved_model = _swap(model.model, arc_model_sd)
    saved_cond = _swap(model.conditioner, arc_cond_sd) if arc_cond_sd else {}
    return saved_model, saved_cond


class _ArcShapeMismatch(RuntimeError):
    """Raised when ARC ckpt shapes don't match the base — caller should skip
    ARC demos rather than try to swap mismatched tensors."""
    pass


def _restore_swapped_weights(model, saved_model, saved_cond):
    def _get_param(net, dotted_key):
        parts = dotted_key.split(".")
        obj = net
        for p in parts[:-1]:
            obj = getattr(obj, p)
        return getattr(obj, parts[-1])
    for key, cpu_val in saved_model.items():
        _get_param(model.model, key).data.copy_(cpu_val)
    for key, cpu_val in saved_cond.items():
        _get_param(model.conditioner, key).data.copy_(cpu_val)


def run_demo_step(model, backend, demo_cfg, step, sample_size, sample_rate, device, model_config=None):
    """Generate the per-prompt demo set for one step.

    Args:
        model: the diffusion model wrapper (has .model, .conditioner, .pretransform).
        backend: underfit.backends.<sat_dev|sa3> module.
        demo_cfg: dict from model_config.training.demo. Keys we honor:
            demo_cond, demo_cfg_scales, demo_steps,
            arc_lora_path, arc_full_model_path, arc_full_model_config,
            latent_crop_length.
        step: global training step (used in filenames).
        sample_size: model_config.sample_size.
        sample_rate: model_config.sample_rate.
        device: torch device string.

    Writes demo_<i>_<step:08d>.mp3 + .json into cwd. Skips entries that
    already exist on disk (idempotent for resumes).
    """
    demo_cond = demo_cfg.get("demo_cond") or []
    if not demo_cond:
        return
    if os.path.exists(f"demo_0_{step:08d}.mp3"):
        tqdm.write(f"Demos already exist for step {step}, skipping generation", file=sys.stdout)
        return

    demo_cfg_scales = demo_cfg.get("demo_cfg_scales", [7])
    demo_steps = demo_cfg.get("demo_steps", 50)
    latent_crop_length = demo_cfg.get("latent_crop_length")
    arc_lora_path = demo_cfg.get("arc_lora_path")
    arc_full_model_path = demo_cfg.get("arc_full_model_path")
    arc_full_model_config = demo_cfg.get("arc_full_model_config")
    default_cfg = demo_cfg_scales[0] if demo_cfg_scales else 7

    default_dur_latents = _get_demo_latent_length(model, sample_size, latent_crop_length)
    ds_ratio = model.pretransform.downsampling_ratio if model.pretransform is not None else 1

    rf_entries = [(i, c) for i, c in enumerate(demo_cond) if not c.get("arc")]
    arc_entries = [(i, c) for i, c in enumerate(demo_cond) if c.get("arc")]

    base_label = "V" if model.diffusion_objective == "v" else "RF"

    model.eval()
    try:
        # --- RF/V demos: one print per demo (no folding) ---
        for i, entry in rf_entries:
            cfg = entry.get("cfg", default_cfg)
            seed = entry.get("seed")
            lora_strength = entry.get("lora_strength")
            lora_interval_max = entry.get("lora_interval_max")
            steps_override = entry.get("steps")
            dur_sec = entry.get("duration")
            dur_latents = max(1, int(dur_sec * sample_rate / ds_ratio)) if dur_sec else default_dur_latents
            prompt = entry.get("prompt", "")
            tqdm.write(
                f"Generating demo {i} ({base_label}): cfg={cfg}, seed={seed}, "
                f"lora_strength={lora_strength}, lora_interval_max={lora_interval_max}, "
                f"prompt={prompt!r}",
                file=sys.stdout,
            )
            try:
                if lora_interval_max is not None:
                    audio = _generate_split_lora_sample(
                        model, backend, [entry], cfg, sample_rate,
                        demo_steps=steps_override or demo_steps,
                        lora_interval_max=lora_interval_max,
                        lora_strength=lora_strength if lora_strength is not None else 1.0,
                        seed=seed, duration_latents=dur_latents,
                    )
                else:
                    audio = _generate_with_lora_strength(
                        model, backend, [entry], cfg, sample_rate,
                        model_config=model_config,
                        demo_steps=steps_override or demo_steps,
                        lora_strength=lora_strength,
                        seed=seed, duration_latents=dur_latents,
                    )
                meta = {"prompt": prompt, "cfg": cfg}
                if seed is not None:
                    meta["seed"] = seed
                if lora_strength is not None:
                    meta["lora_strength"] = lora_strength
                if lora_interval_max is not None:
                    meta["lora_skip"] = lora_interval_max
                _save_demo_file(audio, i, step, sample_rate, meta)
            except Exception as e:
                tqdm.write(f"Error generating demo {i}: {e}", file=sys.stdout)
                import traceback
                traceback.print_exc()

        # --- ARC demos: stacked LoRA path ---
        if arc_entries and arc_lora_path:
            _stack_arc_lora(model, backend, arc_lora_path)
            try:
                for i, entry in arc_entries:
                    _run_one_arc_entry(model, backend, entry, i, step, sample_rate, ds_ratio,
                                      default_dur_latents, demo_steps, has_lora_stack=True,
                                      model_config=model_config)
            finally:
                _unstack_arc_lora(model, backend)
        # --- ARC demos: full-model swap path ---
        elif arc_entries and arc_full_model_path:
            tqdm.write(f"Loading ARC full model from {arc_full_model_path}", file=sys.stdout)
            try:
                saved_model, saved_cond = _swap_full_arc_weights(
                    model, backend, arc_full_model_path, arc_full_model_config,
                )
            except _ArcShapeMismatch as e:
                tqdm.write(
                    f"Skipping {len(arc_entries)} ARC demos: {e}. "
                    f"Base and ARC have different architectures — same-size weight swap "
                    f"isn't possible. To enable, the ARC would need to load as a separate "
                    f"model instance (not yet implemented).",
                    file=sys.stdout,
                )
            else:
                try:
                    for i, entry in arc_entries:
                        _run_one_arc_entry(model, backend, entry, i, step, sample_rate, ds_ratio,
                                          default_dur_latents, demo_steps, has_lora_stack=False,
                                          model_config=model_config)
                finally:
                    _restore_swapped_weights(model, saved_model, saved_cond)
                    del saved_model, saved_cond
                    gc.collect()
                    tqdm.write("Restored base model weights after ARC demos", file=sys.stdout)
        elif arc_entries:
            tqdm.write(f"Skipping {len(arc_entries)} ARC demos: no ARC path configured", file=sys.stdout)
    finally:
        model.train()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_one_arc_entry(model, backend, entry, i, step, sample_rate, ds_ratio,
                       default_dur_latents, demo_steps, has_lora_stack, model_config=None):
    """Generate a single ARC demo (used by both stacked and full-swap paths)."""
    cfg = entry.get("cfg", 1)
    seed = entry.get("seed")
    steps = entry.get("steps", 8)
    lora_interval_max = entry.get("lora_interval_max")
    arc_lora_strength = entry.get("arc_lora_strength")
    arc_lora_interval_max = entry.get("arc_lora_interval_max")
    dur_sec = entry.get("duration")
    dur_latents = max(1, int(dur_sec * sample_rate / ds_ratio)) if dur_sec else default_dur_latents
    prompt = entry.get("prompt", "")
    tqdm.write(
        f"Generating demo {i} (ARC): cfg={cfg}, seed={seed}, steps={steps}, "
        f"lora_interval_max={lora_interval_max}, arc_lora_strength={arc_lora_strength}, "
        f"arc_lora_interval_max={arc_lora_interval_max}, prompt={prompt!r}",
        file=sys.stdout,
    )
    try:
        arc_dist_shift = getattr(model, "sampling_dist_shift", model.dist_shift)
        has_interval = lora_interval_max is not None or arc_lora_interval_max is not None
        if has_interval and has_lora_stack:
            audio = _generate_split_lora_sample(
                model, backend, [entry], cfg, sample_rate,
                demo_steps=steps,
                lora_interval_max=lora_interval_max,
                arc_lora_interval_max=arc_lora_interval_max,
                arc_lora_strength=arc_lora_strength if arc_lora_strength is not None else 1.0,
                seed=seed, duration_latents=dur_latents,
                diffusion_objective_override="rf_denoiser",
            )
        else:
            audio = _generate_single_sample(
                model, backend, [entry], cfg, sample_rate,
                demo_steps=steps, seed=seed, duration_latents=dur_latents,
                diffusion_objective_override="rf_denoiser",
                dist_shift_override=arc_dist_shift,
                model_config=model_config,
            )
        meta = {"prompt": prompt, "cfg": cfg, "arc": True, "steps": steps}
        if seed is not None:
            meta["seed"] = seed
        if lora_interval_max is not None:
            meta["lora_interval_max"] = lora_interval_max
        if arc_lora_strength is not None:
            meta["arc_lora_strength"] = arc_lora_strength
        if arc_lora_interval_max is not None:
            meta["arc_lora_interval_max"] = arc_lora_interval_max
        _save_demo_file(audio, i, step, sample_rate, meta)
    except Exception as e:
        tqdm.write(f"Error generating ARC demo {i}: {e}", file=sys.stdout)
        import traceback
        traceback.print_exc()
