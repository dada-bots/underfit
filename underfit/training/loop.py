"""Raw-PyTorch training loop for Underfit LoRA finetuning.

Replaces the Lightning training wrapper + callbacks. Same dashboard log
format (the regex in dashboard/server.py is satisfied), same on-disk
artifacts (.safetensors LoRA checkpoints, loss_by_timestep.bin demo files),
plus inline manual save via SIGUSR1.

Backend-agnostic: takes a backend module argument and routes all
model/dataset/conditioner calls through it.
"""
import json
import math
import os
import re
import signal
import struct
import sys
import time
import uuid
from pathlib import Path

import torch
from tqdm import tqdm

from underfit.training.demo_step import run_demo_step
from underfit.training.lora import apply_lora_from_config, load_lora_resume, save_lora_step
from underfit.training.loss import compute_masked_loss, compute_normalized_mse
from underfit.training.optim import create_optimizer_from_config, create_scheduler_from_config
from underfit.training.timestep import sample_t
from underfit.utils import copy_state_dict, load_ckpt_state_dict, remove_weight_norm_from_model


class _NullCtx:
    """Trivial null context manager for the no-AMP path."""
    def __enter__(self):
        return None
    def __exit__(self, *_):
        return False


def _resolve_amp(precision):
    """Map a Lightning-style precision string to (autocast_dtype, use_grad_scaler).

    "16-mixed" / "16"  -> fp16 autocast + GradScaler
    "bf16-mixed" / "bf16" -> bf16 autocast (no scaler)
    "32" / None / "" -> no autocast
    """
    if not precision or precision in ("32", "32-true"):
        return None, False
    p = str(precision).lower()
    if "bf16" in p:
        return torch.bfloat16, False
    if "16" in p:
        return torch.float16, True
    return None, False


def _resize_padding_mask(padding_mask, target_length):
    """Resize a (B, T) boolean mask to (B, target_length), preserving valid lengths
    via ceiling-based length scaling. Vendored from SAT-dev/training/utils.py."""
    valid_lengths = padding_mask.sum(dim=-1)
    source_length = padding_mask.shape[-1]
    new_valid = torch.ceil(valid_lengths.float() * target_length / source_length).long().clamp(max=target_length)
    positions = torch.arange(target_length, device=padding_mask.device).unsqueeze(0)
    return positions < new_valid.unsqueeze(1)


def _resolve_resume_path(args, training_cfg):
    """The .safetensors path we're resuming from, if any. CLI arg takes
    precedence over training_cfg.lora_ckpt_path."""
    return getattr(args, "lora_ckpt_path", None) or training_cfg.get("lora_ckpt_path")


def _parse_filename_offsets(lora_path):
    """Parse step= / epoch= from a checkpoint filename. Returns (step, epoch),
    either may be None if the filename doesn't contain that token."""
    base = os.path.basename(lora_path)
    s = re.search(r"step=(\d+)", base)
    e = re.search(r"epoch=(\d+)", base)
    return (int(s.group(1)) if s else None,
            int(e.group(1)) if e else None)


def _resolve_offsets(args, model_config, resume_metadata, steps_per_epoch=None):
    """Recover (step_offset, epoch_offset) for a resume.

    Resolution order, per requirement:
      1. training_cfg.step_offset / training_cfg.epoch_offset (explicit override)
      2. safetensors metadata (resume_metadata dict from load_lora_checkpoint)
      3. step= / epoch= tokens parsed from the resume filename
      4. step // steps_per_epoch as a last-resort estimate for epoch
    """
    training_cfg = model_config.get("training", {})
    cfg_step = int(training_cfg.get("step_offset", 0) or 0)
    cfg_epoch = int(training_cfg.get("epoch_offset", 0) or 0)

    lora_path = _resolve_resume_path(args, training_cfg)

    meta_step = meta_epoch = None
    if resume_metadata:
        if "step" in resume_metadata:
            try:
                meta_step = int(resume_metadata["step"])
            except (TypeError, ValueError):
                pass
        if "epoch" in resume_metadata:
            try:
                meta_epoch = int(resume_metadata["epoch"])
            except (TypeError, ValueError):
                pass

    file_step = file_epoch = None
    if lora_path:
        file_step, file_epoch = _parse_filename_offsets(lora_path)

    step_offset = cfg_step or meta_step or file_step or 0
    epoch_offset = cfg_epoch or meta_epoch or file_epoch
    if epoch_offset is None:
        if step_offset and steps_per_epoch:
            epoch_offset = step_offset // steps_per_epoch
        else:
            epoch_offset = 0
    return step_offset, epoch_offset


def _resolve_checkpoint_dir(args):
    if args.save_dir and args.name:
        session_id = uuid.uuid4().hex[:8]
        return os.path.join(args.save_dir, args.name, session_id, "checkpoints")
    return args.save_dir


def _ckpt_filename(run_label, step, epoch):
    if run_label:
        return f"{run_label}-step={step}-epoch={epoch}.safetensors"
    return f"step={step}-epoch={epoch}.safetensors"


class _LossByTimestepLog:
    """Append (step, t_mean, loss_mean) triples to a binary file the dashboard reads."""
    def __init__(self, path):
        self.path = path
        self._f = None

    def write(self, step, t, loss):
        if self._f is None:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self._f = open(self.path, "ab")
        self._f.write(struct.pack("Iff", step, float(t), float(loss)))
        if step % 10 == 0:
            self._f.flush()

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None


def _explain_model_load_error(exc, model_config):
    """Print friendly help for known model-load failure modes before the
    traceback bubbles up. Catches gated/missing HuggingFace repos (typically
    T5Gemma) and OOM-on-init."""
    msg = str(exc)
    name = type(exc).__name__
    is_hf_404 = (name == "RepositoryNotFoundError" or "404" in msg) and "huggingface.co" in msg
    is_hf_auth = name in ("GatedRepoError", "HfHubHTTPError") or "401" in msg or "gated" in msg.lower()
    is_oom = "out of memory" in msg.lower() or "OutOfMemoryError" in name

    if not (is_hf_404 or is_hf_auth or is_oom):
        return  # let the original traceback speak for itself

    print()
    print("=" * 72)
    print(" ✗ Model load failed before training could start.")
    print("=" * 72)

    if is_hf_404 or is_hf_auth:
        # Try to recover the HF repo the loader tripped on.
        bad_repo = None
        for part in msg.split():
            if "huggingface.co/api/models/" in part:
                bad_repo = part.split("huggingface.co/api/models/", 1)[1]
                bad_repo = bad_repo.split("/tree/")[0].split("/resolve/")[0]
                break
        print()
        print(" Cause: HuggingFace returned a 404 or 401 for the model below.")
        if bad_repo:
            print(f"   {bad_repo!r}")
        print()
        print(" Most common reasons:")
        print("   • You haven't run the underfit installer to download the SA3 model packs.")
        print("   • You're not logged in to HuggingFace on this machine.")
        print("   • Your account hasn't been granted access to the gated repo yet.")
        print()
        print(" Fix:")
        print("   1. Make sure you're logged in:        hf auth login")
        print("   2. Re-run the underfit setup wizard:  underfit-setup --backend sa3")
        print("      (It will detect what's already installed and offer to download")
        print("       any missing model packs into the dashboard's state dir.)")
        print()
        print(" The stabilityai SA3 release packs (base + ARC) all bundle their own")
        print(" T5Gemma tokenizer, so once the SA3 packs are downloaded, the")
        print(" T5Gemma 404 you saw resolves automatically.")
    elif is_oom:
        print()
        print(" Cause: GPU ran out of memory while building the model.")
        print()
        print(" Try one of:")
        print("   • Pick a smaller variant (sa3-sm-music / sa3-sm-sfx instead of sa3-medium)")
        print("   • Lower batch_size in the run's _model.json")
        print("   • Reduce lora_rank")
        print("   • Use --precision 16-mixed if you weren't already")

    print()
    print("=" * 72)
    print()


def _compute_grad_and_lora_norms(lora_params):
    """Post-clip grad norm and LoRA magnitude across LoRA params."""
    grad_sq = 0.0
    grad_count = 0
    lora_sq = 0.0
    lora_count = 0
    for p in lora_params:
        if p.grad is not None:
            gn = p.grad.data.float().norm(2).item()
            if gn == gn:
                grad_sq += gn ** 2
                grad_count += 1
        m = p.data.float().norm(2).item()
        lora_sq += m ** 2
        lora_count += 1
    grad_norm = math.sqrt(grad_sq) if grad_count > 0 else None
    lora_mag = math.sqrt(lora_sq) if lora_count > 0 else None
    return grad_norm, lora_mag


def run_training(args, backend):
    """Main entry point. Mirrors the contract of lora_train.py."""
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)

    print(f"Using backend: {backend.NAME}", flush=True)

    with open(args.model_config) as f:
        model_config = json.load(f)
    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    training_cfg = model_config.get("training", {})
    lora_config = training_cfg.get("lora_config")
    if lora_config:
        print("LoRA config:", lora_config)

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    audio_channels = model_config.get("audio_channels", 2)
    pre_encoded = bool(training_cfg.get("pre_encoded", False))

    # --- Build model and load pretrained weights ---
    print("[startup] Building model from config …", flush=True)
    try:
        model = backend.create_model(model_config)
    except Exception as e:
        _explain_model_load_error(e, model_config)
        raise
    if args.pretrained_ckpt_path:
        print(f"[startup] Loading base weights from {args.pretrained_ckpt_path} …", flush=True)
        # Route through backend.load_state_into so SA3's remap_state_dict_keys
        # gets applied (pretransform.model.* -> pretransform.* nesting fix).
        backend.load_state_into(
            model,
            load_ckpt_state_dict(args.pretrained_ckpt_path),
            model_type=model_config.get("model_type"),
        )
    if args.remove_pretransform_weight_norm == "pre_load":
        remove_weight_norm_from_model(model.pretransform)
    if args.pretransform_ckpt_path:
        print(f"[startup] Loading pretransform from {args.pretransform_ckpt_path}", flush=True)
        model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))
    if args.remove_pretransform_weight_norm == "post_load":
        remove_weight_norm_from_model(model.pretransform)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[startup] Moving model to {device} …", flush=True)
    model.to(device)

    # --- Resume state for LoRA ---
    # CLI arg takes precedence; otherwise fall back to training_config.lora_ckpt_path
    # (the dashboard writes this into the _model_resume.json on resume).
    lora_state_dict = None
    resume_metadata = None
    lora_resume_path = getattr(args, "lora_ckpt_path", None) or training_cfg.get("lora_ckpt_path")
    if lora_resume_path:
        print(f"Loading LoRA resume from {lora_resume_path}")
        lora_state_dict, resume_metadata = load_lora_resume(backend, lora_resume_path)

    if lora_config is None:
        raise ValueError("Underfit raw-PT loop requires training.lora_config in model config")
    base_precision = training_cfg.get("base_precision")
    svd_bases_path = model_config.get("svd_bases_path")
    print("[startup] Applying LoRA adapters …", flush=True)
    lora_params, saved_lora_cfg = apply_lora_from_config(
        backend, model, lora_config,
        lora_state_dict=lora_state_dict,
        base_precision=base_precision,
        svd_bases_path=svd_bases_path,
    )
    # LoRA params train in fp32; base stays in base_precision
    for p in lora_params:
        p.data = p.data.float()
        p.requires_grad_(True)
    print(f"Trainable LoRA params: {sum(p.numel() for p in lora_params):,}")

    # --- Dataset / DataLoader ---
    tokenizers = {}
    if hasattr(model, "conditioner"):
        for key, cond in model.conditioner.conditioners.items():
            if hasattr(cond, "tokenizer") and hasattr(cond, "max_length"):
                tokenizers[key] = (cond.tokenizer, cond.max_length)

    print("[startup] Building dataloader …", flush=True)
    train_dl = backend.create_dataloader(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=sample_rate,
        sample_size=sample_size,
        audio_channels=audio_channels,
        tokenizers=tokenizers if tokenizers else None,
    )

    # --- Optimizer and scheduler ---
    optimizer_configs = training_cfg.get("optimizer_configs")
    if optimizer_configs is None:
        if not getattr(args, "lr", None):
            raise ValueError("Need optimizer_configs in training config or --lr CLI arg")
        opt_cfg = {"diffusion": {"optimizer": {"type": "AdamW", "config": {"lr": float(args.lr)}}}}
    else:
        opt_cfg = optimizer_configs

    optimizer = create_optimizer_from_config(opt_cfg["diffusion"]["optimizer"], lora_params)
    scheduler = None
    if "scheduler" in opt_cfg["diffusion"]:
        scheduler = create_scheduler_from_config(opt_cfg["diffusion"]["scheduler"], optimizer)

    # --- Mixed precision setup ---
    # Lightning's --precision flag controls autocast dtype + grad scaling.
    # Without it, the conditioner forward sees fp16 inputs hitting fp32 LoRA-
    # parametrized weights and crashes with a Half/Float dtype mismatch.
    autocast_dtype, use_grad_scaler = _resolve_amp(getattr(args, "precision", None))
    grad_scaler = torch.amp.GradScaler("cuda") if use_grad_scaler else None
    if autocast_dtype is not None:
        print(f"AMP: autocast={autocast_dtype}, grad_scaler={grad_scaler is not None}", flush=True)

    # --- Step + epoch offsets for resume ---
    # Resolution order: training_cfg.{step,epoch}_offset > safetensors metadata >
    # filename tokens > step//steps_per_epoch (epoch only).
    steps_per_epoch = len(train_dl) if hasattr(train_dl, "__len__") else None
    step_offset, epoch_offset = _resolve_offsets(
        args, model_config, resume_metadata, steps_per_epoch=steps_per_epoch,
    )
    if step_offset:
        print(f"Resuming from step offset {step_offset}, epoch offset {epoch_offset}")

    # --- Filenames / dirs ---
    checkpoint_dir = _resolve_checkpoint_dir(args)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    run_label = re.sub(r"-\d{14}$", "", args.name) if args.name else None

    # --- Loss-by-timestep log ---
    lbt_log = _LossByTimestepLog(os.path.join(os.getcwd(), "loss_by_timestep.bin"))

    # --- SIGUSR1 manual save ---
    manual_save_requested = [False]
    def _request_save(*_):
        manual_save_requested[0] = True
        print("\n[SIGUSR1] Checkpoint save requested — will save after current step", flush=True)
    signal.signal(signal.SIGUSR1, _request_save)

    # --- Training loop ---
    diffusion_objective = model.diffusion_objective
    cfg_dropout_prob = float(training_cfg.get("cfg_dropout_prob", 0.1))
    timestep_sampler = training_cfg.get("timestep_sampler", "uniform")
    timestep_options = training_cfg.get("timestep_sampler_options", {})
    mask_loss_weight = float(training_cfg.get("mask_loss_weight", 0.0))
    mask_padding_attention = bool(getattr(model, "mask_padding_attention", False))
    use_effective_length_for_schedule = bool(getattr(model, "use_effective_length_for_schedule", False))
    loss_normalization = training_cfg.get("loss_normalization", "none")
    loss_norm_eps = float(training_cfg.get("loss_norm_eps", 1e-6))
    grad_clip = args.gradient_clip_val if args.gradient_clip_val else None
    # SAT-dev factory reads training.inpainting; SA3 train_lora always supplies
    # zeros for inpaint conditioning. We follow SAT-dev's key naming.
    inpainting_config = training_cfg.get("inpainting") or training_cfg.get("inpainting_config")
    inpaint_mask_kwargs = (inpainting_config or {}).get("mask_kwargs", {})
    needs_inpaint_cond = (
        "inpaint_mask" in (getattr(model, "local_add_cond_ids", []) or [])
        or "inpaint_masked_input" in (getattr(model, "local_add_cond_ids", []) or [])
    )
    pretransform_scale = getattr(model.pretransform, "scale", 1.0) if model.pretransform is not None else 1.0
    downsampling_ratio = model.pretransform.downsampling_ratio if model.pretransform is not None else 1

    max_steps = int(args.max_steps) if args.max_steps else 10**9
    save_every = int(args.checkpoint_every) if args.checkpoint_every else 1000
    demo_cfg = training_cfg.get("demo", {}) or {}
    demo_every = int(demo_cfg.get("demo_every", 0))
    last_demo_step = -1
    raw_step = 0
    epoch = epoch_offset

    print(f"Training for up to {max_steps} steps; save every {save_every}", flush=True)

    try:
        # Baseline demo at step 0 (fresh runs only). Captures the model output
        # before any LoRA updates — at init the LoRA delta is identity, so this
        # is effectively the base model. Skip on resume since prior demos were
        # already emitted by the original run.
        if demo_every > 0 and step_offset == 0:
            try:
                with torch.no_grad():
                    run_demo_step(
                        model, backend, demo_cfg, 0,
                        sample_size=sample_size,
                        sample_rate=sample_rate,
                        device=device,
                        model_config=model_config,
                    )
                last_demo_step = 0
            except Exception as e:
                print(f"Demo step error (step 0): {type(e).__name__}: {e}", flush=True)
                import traceback
                traceback.print_exc()

        while raw_step < max_steps:
            pbar = tqdm(
                train_dl,
                desc=f"Step {raw_step + step_offset}, Epoch {epoch}",
                mininterval=0,
                miniters=1,
                file=sys.stdout,
            )
            for batch_idx, batch in enumerate(pbar):
                if raw_step >= max_steps:
                    break
                global_step = raw_step + step_offset

                reals, metadata = batch
                if reals.ndim == 4 and reals.shape[0] == 1:
                    reals = reals[0]
                reals = reals.to(device)

                amp_ctx = (
                    torch.amp.autocast("cuda", dtype=autocast_dtype)
                    if autocast_dtype is not None
                    else _NullCtx()
                )
                with amp_ctx:
                    conditioning = backend.encode_conditioning(model, list(metadata), device)

                    if all("padding_mask" in md for md in metadata):
                        padding_masks = torch.stack(
                            [md["padding_mask"][0] for md in metadata], dim=0
                        ).to(device)
                    else:
                        padding_masks = torch.ones(reals.shape[0], reals.shape[-1], dtype=torch.bool, device=device)

                    if pre_encoded:
                        diffusion_input = reals
                        if pretransform_scale != 1.0:
                            diffusion_input = diffusion_input / pretransform_scale
                    else:
                        diffusion_input = model.pretransform.encode(reals)
                    if padding_masks.shape[-1] != diffusion_input.shape[-1]:
                        padding_masks = _resize_padding_mask(padding_masks, diffusion_input.shape[-1])

                    B = diffusion_input.shape[0]
                    t = sample_t(timestep_sampler, B, device, options=timestep_options)
                    if model.dist_shift is not None:
                        if use_effective_length_for_schedule and all("seconds_total" in md for md in metadata):
                            effective_seq_len = torch.tensor(
                                [int(math.ceil(int(md["seconds_total"] * sample_rate) / downsampling_ratio)) for md in metadata],
                                device=device,
                            )
                        else:
                            effective_seq_len = diffusion_input.shape[2]
                        t = model.dist_shift.shift(t, effective_seq_len)

                    if diffusion_objective in ("rectified_flow", "rf_denoiser"):
                        alphas, sigmas = 1 - t, t
                    elif diffusion_objective == "v":
                        alphas = torch.cos(t * math.pi / 2)
                        sigmas = torch.sin(t * math.pi / 2)
                    else:
                        raise ValueError(f"Unsupported diffusion_objective={diffusion_objective}")
                    alphas = alphas[:, None, None]
                    sigmas = sigmas[:, None, None]

                    noise = torch.randn_like(diffusion_input)
                    noised = diffusion_input * alphas + noise * sigmas
                    if diffusion_objective in ("rectified_flow", "rf_denoiser"):
                        target = noise - diffusion_input
                    else:
                        target = noise * alphas - diffusion_input * sigmas

                    loss_mask = padding_masks.to(torch.bool)
                    extra_args = {}
                    if mask_padding_attention:
                        extra_args["padding_mask"] = padding_masks

                    # Provide inpaint conditioning when the model expects it.
                    # If training.inpainting is configured, generate random masks;
                    # otherwise feed all-ones mask + zero context (= pure generation).
                    if needs_inpaint_cond:
                        if inpainting_config is not None:
                            inpaint_masked_input, inpaint_mask = backend.random_inpaint_mask(
                                diffusion_input, padding_masks=padding_masks,
                                mask_padding=mask_padding_attention, **inpaint_mask_kwargs,
                            )
                            # Restrict loss to the masked (predicted) region.
                            loss_mask = loss_mask & ~inpaint_mask.squeeze(1).to(torch.bool)
                        else:
                            inpaint_mask = torch.ones(
                                diffusion_input.shape[0], 1, diffusion_input.shape[2],
                                device=device, dtype=diffusion_input.dtype,
                            )
                            inpaint_masked_input = torch.zeros_like(diffusion_input)
                        conditioning["inpaint_mask"] = [inpaint_mask]
                        conditioning["inpaint_masked_input"] = [inpaint_masked_input]

                    output = model(noised, t, cond=conditioning,
                                  cfg_dropout_prob=cfg_dropout_prob, **extra_args)

                    mse_full = compute_normalized_mse(
                        output, target, loss_mask,
                        loss_normalization=loss_normalization,
                        loss_norm_eps=loss_norm_eps,
                    )
                    loss, signal_mean, padding_mean = compute_masked_loss(
                        mse_full, loss_mask, mask_padding_attention, mask_loss_weight=mask_loss_weight,
                    )

                optimizer.zero_grad()
                if grad_scaler is not None:
                    grad_scaler.scale(loss).backward()
                    if grad_clip is not None:
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
                    grad_norm, lora_mag = _compute_grad_and_lora_norms(lora_params)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
                    grad_norm, lora_mag = _compute_grad_and_lora_norms(lora_params)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                # --- Log metrics on the tqdm postfix (the dashboard parses this format) ---
                lr = optimizer.param_groups[0]["lr"]
                metrics = {
                    "train/loss": f"{loss.item():.6f}",
                    "train/lr": f"{lr:.3e}",
                }
                if grad_norm is not None:
                    metrics["train/grad_norm"] = f"{grad_norm:.6f}"
                if lora_mag is not None:
                    metrics["train/lora_magnitude"] = f"{lora_mag:.6f}"
                pbar.set_postfix(metrics)

                # --- Loss-by-timestep ---
                lbt_log.write(global_step, t.detach().float().mean().item(), loss.item())

                raw_step += 1
                global_step = raw_step + step_offset
                # Update progress bar prefix with the just-completed global step.
                # Dashboard regex still matches "Epoch (\d+):" via re.search.
                pbar.set_description(f"Step {global_step}, Epoch {epoch}")

                # --- Checkpoint save (before demos so we don't lose work if
                # demo generation crashes or stalls). Aligned with global_step
                # so saves happen at clean multiples of save_every regardless
                # of step_offset from a resume.
                save_now = manual_save_requested[0] or (
                    global_step > 0 and global_step % save_every == 0
                )
                demo_will_fire = (
                    demo_every > 0
                    and global_step > 0
                    and global_step % demo_every == 0
                    and last_demo_step != global_step
                )

                if save_now or demo_will_fire:
                    # Disable + clear the training pbar so save + demo prints
                    # land cleanly on their own lines and the inner sampler
                    # tqdm doesn't fight the parent's mininterval=0 redraws.
                    pbar.clear()
                    pbar.disable = True
                    try:
                        if save_now and checkpoint_dir:
                            if manual_save_requested[0]:
                                manual_save_requested[0] = False
                            out = os.path.join(checkpoint_dir, _ckpt_filename(run_label, global_step, epoch))
                            save_lora_step(backend, model, saved_lora_cfg, out, step=global_step, epoch=epoch)
                            print(f"✓ Saved checkpoint -- {os.path.basename(out)}", flush=True)

                        if demo_will_fire:
                            last_demo_step = global_step
                            try:
                                with torch.no_grad():
                                    run_demo_step(
                                        model, backend, demo_cfg, global_step,
                                        sample_size=sample_size,
                                        sample_rate=sample_rate,
                                        device=device,
                                        model_config=model_config,
                                    )
                            except Exception as e:
                                print(f"Demo step error: {type(e).__name__}: {e}", flush=True)
                                import traceback
                                traceback.print_exc()
                    finally:
                        pbar.disable = False
                        pbar.refresh()

                if raw_step >= max_steps:
                    break
            epoch += 1

        # Final save (skip if the last regular save already covered this step).
        global_step = raw_step + step_offset
        if checkpoint_dir and global_step > 0 and global_step % save_every != 0:
            out = os.path.join(checkpoint_dir, _ckpt_filename(run_label, global_step, epoch))
            save_lora_step(backend, model, saved_lora_cfg, out, step=global_step, epoch=epoch)
            print(f"✓ Saved checkpoint -- {os.path.basename(out)} (final)", flush=True)
    finally:
        lbt_log.close()
        print("Training done", flush=True)
