"""MLX engine adapter for Underfit.

Apple-Silicon-only alternate training/inference engine. Instead of running the
in-process torch training loop (underfit.training.loop), the MLX engine shells
out to a *separate* MLX trainer/gradio that lives in a sibling stable-audio-3
checkout (`<sa3>/optimized/mlx`). Everything here is a thin launcher: it maps
Underfit's args/config onto the MLX scripts' locked CLIs, streams the trainer's
stdout back in the dashboard's expected format, and propagates exit codes.

This module never imports torch or stable_audio_3 — it only reads the run's
JSON configs and spawns the MLX venv python. It is only reached when
`--engine mlx` (default is torch, and the torch path is untouched).

Environment overrides:
  UNDERFIT_MLX_ROOT          MLX code root (default <sa3-sibling>/optimized/mlx)
  UNDERFIT_MLX_PYTHON        MLX venv python (default <mlx_root>/.venv/bin/python)
  UNDERFIT_MLX_BASE_WEIGHTS  base DiT weights npz (default
                             <mlx_root>/models/mlx/dit_<model>-base_f16.npz)
"""
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path


# Mirror the sibling-checkout resolution in backends/sa3.py without importing
# it (sa3.py pulls in torch at import time; the MLX path must stay torch-free).
_HERE = Path(__file__).resolve()
_APP_ROOT = _HERE.parent.parent.parent                       # underfit repo root
_SA3_LOCAL = str(_HERE.parent.parent.parent.parent / "stable-audio-3")

_SUPPORTED_DITS = ("sm-music", "sm-sfx", "medium")

# Underfit's tqdm-postfix step line the MLX trainer emits, per its locked
# contract:  "step {N}  train/loss {L:.6f}  train/lr {lr:.3e}  epoch {E}  (…)"
_STEP_LINE_RE = re.compile(
    r"^step (\d+)\s+train/loss (\S+)\s+train/lr (\S+)\s+epoch (\d+)"
)

_STEP_TOKEN_RE = re.compile(r"step=(\d+)")
_EPOCH_TOKEN_RE = re.compile(r"epoch=(\d+)")


# --------------------------------------------------------------------------- #
# Path / name resolution
# --------------------------------------------------------------------------- #
def resolve_mlx_paths():
    """Return (mlx_root, mlx_python).

    Uses UNDERFIT_MLX_ROOT / UNDERFIT_MLX_PYTHON when set, else the sibling
    stable-audio-3 checkout defaults. Raises FileNotFoundError (mentioning the
    env vars) if the trainer script or venv python is missing.
    """
    mlx_root = os.environ.get("UNDERFIT_MLX_ROOT") or os.path.join(
        _SA3_LOCAL, "optimized", "mlx"
    )
    mlx_root = os.path.abspath(mlx_root)
    mlx_python = os.environ.get("UNDERFIT_MLX_PYTHON") or os.path.join(
        mlx_root, ".venv", "bin", "python"
    )
    trainer = os.path.join(mlx_root, "scripts", "lora_train_mlx.py")
    if not os.path.isfile(trainer):
        raise FileNotFoundError(
            f"MLX trainer script not found at {trainer!r}. Point "
            f"UNDERFIT_MLX_ROOT at the stable-audio-3/optimized/mlx checkout "
            f"(resolved root: {mlx_root!r})."
        )
    if not (os.path.isfile(mlx_python) or shutil.which(mlx_python)):
        raise FileNotFoundError(
            f"MLX venv python not found at {mlx_python!r}. Set "
            f"UNDERFIT_MLX_PYTHON to the MLX venv interpreter (default is "
            f"<UNDERFIT_MLX_ROOT>/.venv/bin/python)."
        )
    return mlx_root, mlx_python


def map_model_name(base_model):
    """Map an Underfit base-model key -> the MLX trainer's --dit value.

    Accepts both the dashboard form ("sa3-medium") and the already-stripped
    form ("medium"). Raises ValueError for unsupported models.
    """
    if not base_model:
        raise ValueError(
            "engine=mlx: the model config has no 'base_model' — cannot pick "
            "the MLX --dit value."
        )
    name = str(base_model).strip()
    if name.startswith("sa3-"):
        name = name[len("sa3-"):]
    if name not in _SUPPORTED_DITS:
        raise ValueError(
            f"engine=mlx does not support base_model={base_model!r}. "
            f"Supported: sa3-sm-music, sa3-sm-sfx, sa3-medium."
        )
    return name


def resolve_base_weights(dit_model):
    """Path to the MLX BASE-checkpoint DiT weights npz for a --dit value.

    UNDERFIT_MLX_BASE_WEIGHTS overrides; default is
    <mlx_root>/models/mlx/dit_<model>-base_f16.npz.
    """
    override = os.environ.get("UNDERFIT_MLX_BASE_WEIGHTS")
    if override:
        return os.path.abspath(override)
    mlx_root, _ = resolve_mlx_paths()
    return os.path.join(mlx_root, "models", "mlx", f"dit_{dit_model}-base_f16.npz")


def resolve_dit_model(model_config_path=None, pretrained_name=None):
    """Derive the MLX --dit value for gradio from a model-config path or a
    pretrained name. The dashboard passes the run's _model.json, which carries
    a 'base_model' key."""
    if model_config_path:
        try:
            mc = _load_json(model_config_path)
        except (OSError, ValueError):
            mc = {}
        base = mc.get("base_model")
        if base:
            return map_model_name(base)
    if pretrained_name:
        return map_model_name(pretrained_name)
    raise ValueError(
        "engine=mlx gradio: could not determine the --dit model. Pass a "
        "--model-config whose JSON has 'base_model', or --pretrained-name."
    )


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _resolve_app_relative_path(value):
    """Match backends/sa3._resolve_app_relative_path so the MLX trainer reads
    the same latents dir the torch dataloader would."""
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


def _fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _add(cmd, flag, value):
    """Append `flag value` unless value is None or an empty/whitespace string."""
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    cmd.extend([flag, _fmt(value)])


def _csv(value):
    """Normalize an include/exclude field (list or string) to a CSV string, or
    None when empty."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        items = [str(x).strip() for x in value if str(x).strip()]
        return ",".join(items) if items else None
    s = str(value).strip()
    return s or None


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _dit_model_from_config(model_config):
    return map_model_name(model_config.get("base_model"))


def _latents_dir_from_dataset(dataset_config):
    datasets = dataset_config.get("datasets") or []
    if not datasets:
        raise ValueError(
            "engine=mlx: dataset config has no 'datasets' entry to read "
            "--latents-dir from."
        )
    path = datasets[0].get("path")
    if not path:
        raise ValueError(
            "engine=mlx: dataset config datasets[0] has no 'path' for "
            "--latents-dir."
        )
    return _resolve_app_relative_path(path)


def _resolve_lr(training, args):
    lr = None
    opt = training.get("optimizer_configs")
    if isinstance(opt, dict):
        lr = (
            opt.get("diffusion", {})
            .get("optimizer", {})
            .get("config", {})
            .get("lr")
        )
    if lr is None:
        lr = getattr(args, "lr", None)
    if lr is None:
        raise ValueError(
            "engine=mlx: could not resolve the learning rate. Set "
            "training.optimizer_configs.diffusion.optimizer.config.lr in the "
            "model config (the dashboard writes this from the LR field)."
        )
    return lr


def _dist_shift_from_config(model_config):
    diffusion = (model_config.get("model", {}) or {}).get("diffusion", {}) or {}
    opts = diffusion.get("distribution_shift_options") or {}
    return opts.get("type")


def _use_effective_length(model_config):
    """model.diffusion.use_effective_length_for_schedule (True in SA3 templates):
    shift timesteps by ceil(int(seconds_total*sr)/4096) not the crop length."""
    diffusion = (model_config.get("model", {}) or {}).get("diffusion", {}) or {}
    return bool(diffusion.get("use_effective_length_for_schedule", False))


def _optimizer_config(training):
    """training.optimizer_configs.diffusion.optimizer.config (AdamW betas/eps/wd)."""
    opt = training.get("optimizer_configs") or {}
    return ((opt.get("diffusion", {}) or {}).get("optimizer", {}) or {}).get("config", {}) or {}


def _scheduler_config(training):
    """training.optimizer_configs.diffusion.scheduler (type + config), or None."""
    opt = training.get("optimizer_configs") or {}
    return (opt.get("diffusion", {}) or {}).get("scheduler")


def _parse_filename_offsets(path):
    """(step, epoch) parsed from a checkpoint filename; either may be None.
    Mirrors underfit.training.loop._parse_filename_offsets."""
    base = os.path.basename(str(path))
    s = _STEP_TOKEN_RE.search(base)
    e = _EPOCH_TOKEN_RE.search(base)
    return (int(s.group(1)) if s else None,
            int(e.group(1)) if e else None)


def _resolve_offsets(training, resume_path):
    """(step_offset, epoch_offset) mirroring the torch loop's resolution:
    explicit config keys win (even at 0), else parse from the resume filename."""
    has_step = "step_offset" in training
    has_epoch = "epoch_offset" in training
    step = int(training.get("step_offset", 0) or 0) if has_step else None
    epoch = int(training.get("epoch_offset", 0) or 0) if has_epoch else None
    if (step is None or epoch is None) and resume_path:
        f_step, f_epoch = _parse_filename_offsets(resume_path)
        if step is None:
            step = f_step
        if epoch is None:
            epoch = f_epoch
    return step, epoch


def _build_demo_entries(demo):
    """Map underfit's `training.demo` block onto the MLX trainer's --demo-config,
    which is a flat LIST of entries {prompt, cfg, seed, steps, lora_strength?,
    lora_interval_max?}. ARC entries are skipped: the MLX trainer finetunes the
    BASE model, and ARC demos need a weight-swap/second-model it doesn't do."""
    cfg_scales = demo.get("demo_cfg_scales") or [7]
    default_cfg = cfg_scales[0] if cfg_scales else 7
    default_steps = demo.get("demo_steps", 50)
    entries = []
    for e in demo.get("demo_cond") or []:
        if e.get("arc"):
            continue
        entry = {"prompt": e.get("prompt", ""),
                 "cfg": e.get("cfg", default_cfg),
                 "steps": e.get("steps", default_steps)}
        for k in ("seed", "lora_strength", "lora_interval_max"):
            if e.get(k) is not None:
                entry[k] = e[k]
        entries.append(entry)
    return entries


def _write_demo_config(entries):
    """Write the MLX --demo-config entry list to a temp json and return its path."""
    fd, path = tempfile.mkstemp(prefix="underfit_mlx_demo_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(entries, f)
    return path


# --------------------------------------------------------------------------- #
# Command builders
# --------------------------------------------------------------------------- #
def build_trainer_cmd(args, base_weights):
    """Map Underfit's args + run configs onto the locked lora_train_mlx.py CLI.

    Reads args.model_config / args.dataset_config (paths). base_weights is the
    resolved --dit-weights npz path (see resolve_base_weights).
    """
    mlx_root, mlx_python = resolve_mlx_paths()
    trainer = os.path.join(mlx_root, "scripts", "lora_train_mlx.py")

    model_config = _load_json(args.model_config)
    dataset_config = _load_json(args.dataset_config)
    training = model_config.get("training", {}) or {}
    lora_cfg = training.get("lora_config", {}) or {}

    dit = _dit_model_from_config(model_config)
    latents_dir = _latents_dir_from_dataset(dataset_config)
    lr = _resolve_lr(training, args)

    cmd = [
        mlx_python, trainer,
        "--dit", dit,
        "--dit-weights", str(base_weights),
        "--latents-dir", latents_dir,
        "--lr", _fmt(lr),
        "--name", str(args.name),
        "--save-dir", str(args.save_dir),
    ]

    _add(cmd, "--batch-size", getattr(args, "batch_size", None))
    _add(cmd, "--seed", getattr(args, "seed", None))
    _add(cmd, "--max-steps", getattr(args, "max_steps", None))
    _add(cmd, "--checkpoint-every", getattr(args, "checkpoint_every", None))

    # adapter_type: underfit's config-layer default is "lora" when absent
    # (lora_config.get("adapter_type", "lora")); the MLX trainer defaults to
    # dora-rows, so pass "lora" explicitly to match a template without one.
    _add(cmd, "--adapter-type", lora_cfg.get("adapter_type", "lora"))
    _add(cmd, "--rank", lora_cfg.get("rank"))
    _add(cmd, "--alpha", lora_cfg.get("alpha"))
    _add(cmd, "--include", _csv(lora_cfg.get("include")))
    _add(cmd, "--exclude", _csv(lora_cfg.get("exclude")))

    _add(cmd, "--latent-crop-length", dataset_config.get("latent_crop_length"))
    _add(cmd, "--timestep-sampler", training.get("timestep_sampler"))
    _add(cmd, "--dist-shift", _dist_shift_from_config(model_config))
    if _use_effective_length(model_config):
        cmd.append("--use-effective-length")
    _add(cmd, "--cfg-dropout-prob", training.get("cfg_dropout_prob"))

    # optimizer: AdamW betas/eps/weight_decay (torch decoupled-wd == MLX AdamW)
    opt_cfg = _optimizer_config(training)
    betas = opt_cfg.get("betas")
    if isinstance(betas, (list, tuple)) and len(betas) == 2:
        _add(cmd, "--beta1", betas[0])
        _add(cmd, "--beta2", betas[1])
    _add(cmd, "--eps", opt_cfg.get("eps"))
    _add(cmd, "--weight-decay", opt_cfg.get("weight_decay"))

    # LR scheduler: only InverseLR is supported (the SA3 templates' scheduler)
    sched = _scheduler_config(training)
    if isinstance(sched, dict) and sched.get("type") == "InverseLR":
        sc = sched.get("config", {}) or {}
        cmd.extend(["--lr-scheduler", "inverse"])
        _add(cmd, "--inv-gamma", sc.get("inv_gamma"))
        _add(cmd, "--lr-power", sc.get("power"))
        _add(cmd, "--lr-warmup", sc.get("warmup"))
        _add(cmd, "--lr-final", sc.get("final_lr"))

    # --- Resume: CLI arg wins over training_config.lora_ckpt_path (matches
    # underfit.training.loop) ---
    resume_path = getattr(args, "lora_ckpt_path", None) or training.get("lora_ckpt_path")
    if resume_path and str(resume_path).strip():
        resume_path = str(resume_path).strip()
        _add(cmd, "--lora-ckpt-path", resume_path)
    else:
        resume_path = None
    step_offset, epoch_offset = _resolve_offsets(training, resume_path)
    _add(cmd, "--step-offset", step_offset)
    _add(cmd, "--epoch-offset", epoch_offset)

    # --- Demos: only when the run actually has a demo config ---
    demo = training.get("demo") or {}
    demo_every = demo.get("demo_every")
    if demo.get("demo_cond") and demo_every:
        entries = _build_demo_entries(demo)
        if entries:
            _add(cmd, "--demo-config", _write_demo_config(entries))
            _add(cmd, "--demo-every", demo_every)

    return cmd


def build_gradio_cmd(model, lora_ckpt_paths, share=False, port=None):
    """Map onto the locked sa3_gradio.py CLI:
    --dit <model> --lora <path>… [--share] [--port N].

    `model` may be an Underfit base-model key ("sa3-medium") or a --dit value
    ("medium"); it is normalized via map_model_name.
    """
    mlx_root, mlx_python = resolve_mlx_paths()
    gradio_script = os.path.join(mlx_root, "scripts", "sa3_gradio.py")
    if not os.path.isfile(gradio_script):
        raise FileNotFoundError(
            f"MLX gradio script not found at {gradio_script!r}. Point "
            f"UNDERFIT_MLX_ROOT at the stable-audio-3/optimized/mlx checkout."
        )
    cmd = [mlx_python, gradio_script, "--dit", map_model_name(model)]
    for p in _as_list(lora_ckpt_paths):
        if p and str(p).strip():
            cmd.extend(["--lora", str(p)])
    if share:
        cmd.append("--share")
    if port:
        cmd.extend(["--port", str(int(port))])
    return cmd


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
def run_mlx_training(args):
    """Launch the MLX trainer, stream its stdout, and return its exit code.

    cwd is the current process cwd (the dashboard already cd's into the run's
    demo dir before invoking lora_train.py), so loss_by_timestep.bin and demo
    files land where the dashboard expects. Every trainer step line is echoed
    verbatim and ALSO re-emitted in the dashboard's tqdm-progress format so the
    step/loss/lr parsers pick it up.
    """
    model_config = _load_json(args.model_config)
    dit = _dit_model_from_config(model_config)
    base_weights = resolve_base_weights(dit)
    if not os.path.isfile(base_weights):
        raise FileNotFoundError(
            f"MLX base weights not found at {base_weights!r}. Set "
            f"UNDERFIT_MLX_BASE_WEIGHTS or place the npz at "
            f"<UNDERFIT_MLX_ROOT>/models/mlx/dit_{dit}-base_f16.npz."
        )

    cmd = build_trainer_cmd(args, base_weights)
    print("[mlx-engine] launching MLX trainer:", flush=True)
    print("  " + " ".join(shlex.quote(c) for c in cmd), flush=True)

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        cmd,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(line, flush=True)
            m = _STEP_LINE_RE.match(line)
            if m:
                n, loss, lr, ep = m.group(1), m.group(2), m.group(3), m.group(4)
                # Format satisfies dashboard/server.py:
                #   _parse_latest_step  (Epoch N: … a/b  + "Step N," prefix)
                #   _HISTORY_RE / _LR_RE (train/loss= and train/lr=)
                print(
                    f"Step {n}, Epoch {ep}: 100%|##########| 1/1 "
                    f"train/loss={loss} train/lr={lr}",
                    flush=True,
                )
    finally:
        if proc.stdout:
            proc.stdout.close()
    code = proc.wait()
    print(f"[mlx-engine] MLX trainer exited with code {code}", flush=True)
    return code


def run_mlx_gradio(model, lora_ckpt_paths, share=False, port=None):
    """Launch the MLX gradio server (inherits stdout/stderr so the dashboard's
    log redirect captures it) and return its exit code."""
    cmd = build_gradio_cmd(model, lora_ckpt_paths, share=share, port=port)
    print("[mlx-engine] launching MLX gradio:", flush=True)
    print("  " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(cmd, cwd=os.getcwd(), env=env)
    return proc.wait()
