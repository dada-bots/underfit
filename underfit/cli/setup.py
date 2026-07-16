"""Interactive setup wizard for underfit.

Two phases:

  1. Backend pick — choose stable-audio-tools (sat) or stable-audio-3 (sa3),
     install via uv pip / pip if missing.
  2. Model download — pick which Stable Audio 3 model packs to pull from the
     stabilityai HF org and stage into the dashboard's per-instance model
     registry (STATE_DIR/models/). Each repo bundles its own T5Gemma tokenizer
     as a subfolder, so no separate t5gemma download is required.

Selection logic (backend phase):
  - Default to whichever backend is already installed
  - If both installed: default to sat
  - If neither installed: default to sat
  - User can override with arrow / number keys

Selection logic (model phase):
  - Detect which SA3 model packs are already staged in STATE_DIR/models/<key>.json
  - For each missing pack, ask Y/n (defaults to Y).
  - At least one pack must end up installed.

Re-running the wizard is safe: already-installed models are reported and
skipped; the wizard only offers to download the ones you're missing.

Non-interactive use:
    underfit-setup --backend sat --no-install   # backend phase only, verify
    underfit-setup --backend sa3 --skip-models      # backend phase only, no HF
    underfit-setup --backend sa3 --models sa3-medium,sa3-sm-music  # explicit set
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


# ── BACKEND PHASE ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Backend:
    key: str             # underfit-internal id, also accepted by UNDERFIT_BACKEND
    label: str           # human-readable display name
    module: str          # python import name (for find_spec / import probe)
    clone_url: str       # github URL to clone if user picks "clone fresh"
    sibling_dirname: str # default directory name when cloned next to underfit
    description: str     # one-line description
    extras: tuple[str, ...] = ()  # `pip install -e <path>[extra1,extra2]` extras to enable


BACKENDS: list[Backend] = [
    Backend(
        key="sa3",
        label="stable-audio-3",
        module="stable_audio_3",
        clone_url="https://github.com/Stability-AI/stable-audio-3.git",
        sibling_dirname="stable-audio-3",
        description="Stability-AI/stable-audio-3 — simplified SA3 codebase",
        extras=("lora", "ui"),  # lora: dill/PL for dataloader; ui: gradio for run_gradio.py
    ),
    # sat is opt-in only — passing --backend sat picks it; the wizard never
    # prompts and the default is BACKENDS[0] (sa3).
    Backend(
        key="sat",
        label="stable-audio-tools",
        module="stable_audio_tools",
        clone_url="https://github.com/Stability-AI/stable-audio-tools.git",
        sibling_dirname="stable-audio-tools",
        description="Stability-AI/stable-audio-tools (opt-in via --backend sat)",
        extras=("train", "ui"),  # train: PL/prefigure/etc; ui: gradio for run_gradio.py
    ),
]


# ── SA3 MODEL PACKS ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SA3Pack:
    """A Stable Audio 3 model release.

    Each pack pairs a base (rectified-flow) repo with an ARC (adversarial
    consistency) repo — underfit uses the base for finetuning + as the
    teacher, and ARC for fast inference. Both ship as gated HF repos with
    `model.safetensors` + `model_config.json` and an embedded T5Gemma tokenizer
    in the `t5gemma-b-b-ul2/` subfolder.
    """
    key: str             # dashboard registry key (matches STATE_DIR/models/<key>.json)
    base_repo: str       # base / RF release
    arc_repo: str        # adversarial / fast-inference release
    label: str           # short description for the wizard
    ckpt_filename: str = "model.safetensors"
    config_filename: str = "model_config.json"


SA3_PACKS: list[SA3Pack] = [
    SA3Pack(
        key="sa3-medium",
        base_repo="stabilityai/stable-audio-3-medium-base",
        arc_repo="stabilityai/stable-audio-3-medium",
        label="sa3-medium",
    ),
    SA3Pack(
        key="sa3-sm-music",
        base_repo="stabilityai/stable-audio-3-small-music-base",
        arc_repo="stabilityai/stable-audio-3-small-music",
        label="sa3-small-music",
    ),
    SA3Pack(
        key="sa3-sm-sfx",
        base_repo="stabilityai/stable-audio-3-small-sfx-base",
        arc_repo="stabilityai/stable-audio-3-small-sfx",
        label="sa3-small-sfx",
    ),
]


# ── PATHS / STATE ────────────────────────────────────────────────────────────


def _repo_root() -> Path:
    """Underfit repo root (parent of this file's package)."""
    return Path(__file__).resolve().parent.parent.parent


def state_dir() -> Path:
    """Where the dashboard keeps writable state (runs, datasets, audio, checkpoints).
    Defaults to <repo>/state/. Override with UNDERFIT_STATE_DIR."""
    return Path(os.environ.get("UNDERFIT_STATE_DIR", _repo_root() / "state")).expanduser()


def shipped_models_dir() -> Path:
    """Where the per-model dirs (registry.json + training_template.json) live."""
    return _repo_root() / "dashboard" / "models"


def pack_dir(key: str) -> Path:
    """Per-model dir at dashboard/models/<key>/ — both registry.json and
    training_template.json live here. Created by the wizard if missing."""
    return shipped_models_dir() / key


# ── BACKEND HELPERS ──────────────────────────────────────────────────────────


def backend_installed(backend: Backend) -> bool:
    return importlib.util.find_spec(backend.module) is not None


def installed_backend_keys() -> list[str]:
    return [b.key for b in BACKENDS if backend_installed(b)]


def _arrow_select(title: str, lines_per_option: list[list[str]], default_idx: int) -> int:
    """Up/down arrow single-select with Enter to confirm.

    Each option can render across multiple lines (e.g. label + description);
    only the *first* line of each option is highlighted with the cursor.

    Falls back to a numeric prompt if stdin isn't a TTY (CI / piped input).
    """
    n = len(lines_per_option)
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        # Non-interactive fallback: print menu + accept a typed digit.
        # Sub-descriptions indent under the label text (past the [N] marker).
        print(title)
        for i, opt in enumerate(lines_per_option, 1):
            print(f"  [{i}] {opt[0]}")
            for extra in opt[1:]:
                print(f"        {extra}")  # 8 spaces — clearly indented past '  [1] '
        try:
            ans = input(f"Choice [1-{n}, ↵ for {default_idx + 1}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit(1)
        if not ans:
            return default_idx
        if ans.isdigit() and 1 <= int(ans) <= n:
            return int(ans) - 1
        return default_idx

    import termios, tty
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    selected = default_idx

    # ANSI codes for dim text on description lines (visually separates label
    # from sub-description so the menu reads as N options not 2N options).
    DIM = "\x1b[2;90m"  # faint + bright black
    RESET = "\x1b[0m"
    SUB_INDENT = "      "  # 6 spaces — clearly indented past the cursor marker

    def _render(first=False):
        if not first:
            total_lines = sum(len(opt) for opt in lines_per_option)
            sys.stdout.write(f"\x1b[{total_lines}A")  # move cursor up
        for i, opt in enumerate(lines_per_option):
            head, *rest = opt
            marker = " ▶ " if i == selected else "   "
            sys.stdout.write("\x1b[2K\r")  # clear line
            sys.stdout.write(f"{marker}{head}\n")
            for line in rest:
                sys.stdout.write("\x1b[2K\r")
                sys.stdout.write(f"{SUB_INDENT}{DIM}{line}{RESET}\n")
        sys.stdout.flush()

    print(title)
    print("(↑/↓ to move, Enter to select, q to quit)")
    print()
    _render(first=True)
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                if sys.stdin.read(1) == "[":
                    code = sys.stdin.read(1)
                    if code == "A":
                        selected = (selected - 1) % n
                    elif code == "B":
                        selected = (selected + 1) % n
                    _render()
            elif ch in ("\r", "\n"):
                return selected
            elif ch in ("q", "\x03"):  # q or Ctrl+C
                print()
                sys.exit(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _install_command() -> list[str]:
    """Prefer `uv pip install` when uv is available; fall back to pip.
    Pins the install to *this* Python (sys.executable) so uv doesn't pick a
    different interpreter from PATH — important on Colab where the system
    Python sits next to the project venv."""
    if shutil.which("uv"):
        return ["uv", "pip", "install", "--python", sys.executable]
    return [sys.executable, "-m", "pip", "install"]


def _underfit_root() -> Path:
    """The underfit repo root (parent of underfit/cli/setup.py)."""
    return Path(__file__).resolve().parent.parent.parent


def _is_valid_backend_checkout(backend: Backend, path: Path) -> tuple[bool, str]:
    """Return (ok, reason) — quick structural check that `path` looks like a
    `backend` source tree (has pyproject.toml + the expected package dir)."""
    if not path.exists():
        return False, "path does not exist"
    if not path.is_dir():
        return False, "path is not a directory"
    if not (path / "pyproject.toml").exists():
        return False, f"no pyproject.toml in {path}"
    if not (path / backend.module / "__init__.py").exists():
        return False, f"no {backend.module}/__init__.py — doesn't look like a {backend.label} checkout"
    return True, "OK"


def _discover_backend(backend: Backend) -> dict:
    """Return {installed_module_path, sibling_checkout} — both may be None.

    - installed_module_path: if `import <backend.module>` resolves, the resolved file path
    - sibling_checkout: nearby directory with the expected layout (e.g. underfit's
      parent dir contains a stable-audio-3/ folder with the right pyproject)
    """
    found = {"installed_module_path": None, "sibling_checkout": None}
    spec = importlib.util.find_spec(backend.module)
    if spec and spec.origin:
        found["installed_module_path"] = Path(spec.origin).resolve()
    parent = _underfit_root().parent
    for candidate in (parent / backend.sibling_dirname, parent / backend.module):
        ok, _ = _is_valid_backend_checkout(backend, candidate)
        if ok:
            found["sibling_checkout"] = candidate.resolve()
            break
    return found


def _editable_install(path: Path, extras: tuple[str, ...] = ()) -> int:
    """`uv pip install -e <path>[extras]` (or pip fallback).

    Extras matter: SA3's `lora` extra brings `dill` (required by its dataloader)
    and pytorch_lightning; sat's `train` extra brings pytorch_lightning,
    prefigure, etc. Without them, training fails at import time."""
    target = f"{path}[{','.join(extras)}]" if extras else str(path)
    cmd = [*_install_command(), "-e", target]
    print(f"\n→ {' '.join(cmd)}\n", flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        return rc

    # ── Backend-pin overrides ──────────────────────────────────────────
    # sat's [train] extra and its transitive deps include several
    # numpy-1.x-era packages (pandas==2.0.2, PyWavelets 1.4.x). We pin
    # numpy>=2.2 in our own pyproject, which leaves the venv in a broken
    # state where any import path through those C-extension packages
    # crashes with:
    #     numpy.dtype size changed, may indicate binary incompatibility.
    # Force-upgrade the known offenders to numpy-2.x-compatible releases
    # after sat's editable install settles. (sa3 doesn't pull these in,
    # so this is a no-op there.)
    if "train" in extras:
        print(f"\n→ overriding numpy-1.x-era pins (pandas, PyWavelets) ...\n",
              flush=True)
        subprocess.run([*_install_command(), "--upgrade",
                        "pandas>=2.2.3", "PyWavelets>=1.6"])
    return rc


def _verify_backend_import(backend: Backend) -> bool:
    """Verify the backend module imports cleanly in a *fresh* subprocess so
    a .pth file added by `uv pip install -e` is picked up reliably."""
    importlib.invalidate_caches()
    if backend_installed(backend):
        return True
    rc = subprocess.run(
        [sys.executable, "-c", f"import {backend.module}"],
        capture_output=True,
    ).returncode
    return rc == 0


def _clone_backend(backend: Backend, target: Path) -> bool:
    """git clone <backend.clone_url> <target>. Returns success."""
    if target.exists():
        print(f"  ✗ refusing to clone over existing path: {target}")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n→ git clone {backend.clone_url} {target}\n", flush=True)
    rc = subprocess.run(["git", "clone", backend.clone_url, str(target)]).returncode
    if rc != 0:
        print(f"  ✗ git clone failed (exit {rc})")
        return False
    return True


# ── HF AUTH ──────────────────────────────────────────────────────────────────


def hf_whoami() -> str | None:
    """Returns the authenticated HF username, or None if not logged in."""
    try:
        from huggingface_hub import whoami
    except ImportError:
        return None
    try:
        info = whoami()
        return info.get("name") or info.get("email")
    except Exception:
        return None


def print_hf_login_help() -> None:
    print()
    print("⚠ Not logged in to HuggingFace.")
    print("  Underfit downloads Stable Audio 3 model packs (and their bundled")
    print("  T5Gemma tokenizer) from gated HuggingFace repos in the stabilityai")
    print("  org. You need a HuggingFace account, access to each repo, and a")
    print("  read token on this machine.")
    print()
    print("  1. Create / sign in: https://huggingface.co/join")
    print("  2. Request access to each repo (base + ARC):")
    for p in SA3_PACKS:
        print(f"     https://huggingface.co/{p.base_repo}")
        print(f"     https://huggingface.co/{p.arc_repo}")
    print("  3. Authenticate locally: `hf auth login`")
    print("     (or set the HF_TOKEN env var to a read token).")


# ── MODEL DOWNLOAD + STAGING ─────────────────────────────────────────────────


def _models_dir() -> Path:
    """Where to download SA3 model packs. Defaults to STATE_DIR/models —
    on Colab, point UNDERFIT_MODELS_DIR at /content/models so model files
    live on local SSD (fast reads) instead of Drive (slow + FUSE-flaky).
    Distinct from per-run LoRA *training* checkpoints (those go in RUNS_DIR)."""
    env = os.environ.get("UNDERFIT_MODELS_DIR")
    if env:
        return Path(env).expanduser()
    return state_dir() / "models"


def staged_pack_keys() -> set[str]:
    """Set of SA3 pack keys with ALL required files present.

    Strict check — every pack needs base ckpt + base config + ARC ckpt +
    ARC config. Previously this only looked for base/model.safetensors,
    which let a partially-failed install (e.g. base downloaded, ARC
    blocked by an unaccepted HF license) appear "fully staged" on the
    next ./install.sh run — the wizard would say "nothing to do" and the
    user would be stuck with an unusable install.
    """
    d = _models_dir()
    if not d.is_dir():
        return set()
    required = ("base/model.safetensors", "base/model_config.json",
                "arc/model.safetensors",  "arc/model_config.json")
    ok = set()
    for pack_dir in d.iterdir():
        if not pack_dir.is_dir():
            continue
        if all((pack_dir / r).exists() for r in required):
            ok.add(pack_dir.name)
    return ok


def _print_license_help() -> None:
    """Big visible block telling the user how to accept the SA3 ARC license."""
    print()
    print("━" * 78)
    print(" ⚠️   Stable Audio 3 ARC models require accepting an HF license")
    print("━" * 78)
    print()
    print("  The ARC checkpoints (the inference-fast distilled models) are")
    print("  gated on HuggingFace. Approval is instant once you click")
    print("  Agree, and ONE click unlocks all three SA3 ARC repos at once.")
    print()
    print("    👉  https://huggingface.co/stabilityai/stable-audio-3-medium")
    print()
    print("  After clicking 'Agree and access repository' on that page,")
    print("  re-run ./install.sh and the wizard will resume where it left off.")
    print()
    print("  (The BASE models are not gated; if those already downloaded,")
    print("   the rerun will skip them and only fetch the ARC files.)")
    print()
    print("━" * 78)
    print()


def _check_arc_access(repo_id: str) -> tuple[bool, str | None]:
    """Probe one ARC repo for read access without downloading.

    Returns (ok, reason). ok=False means we're blocked by the gated-repo
    license (most common) or by a network/auth issue (rarer); the caller
    surfaces the license-help block either way since that's the only
    fixable scenario from the user's side.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return True, None  # no hub → defer to actual download attempt
    try:
        HfApi().model_info(repo_id)
        return True, None
    except Exception as e:
        name = type(e).__name__
        msg = str(e).splitlines()[0][:200]
        if "GatedRepoError" in name or "403" in msg or "Cannot access gated repo" in msg:
            return False, f"{name}: {msg}"
        # Anything else — let the real download attempt try.
        return True, None


def _pack_in_hf_cache(pack: SA3Pack) -> bool:
    """True if BOTH the base and ARC repos' main files are in the HF cache.
    A cache hit means snapshot_download will copy locally without going to
    the network — useful signal to print before the user picks packs."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    for repo in (pack.base_repo, pack.arc_repo):
        try:
            if try_to_load_from_cache(repo, pack.ckpt_filename) is None:
                return False
            if try_to_load_from_cache(repo, pack.config_filename) is None:
                return False
        except Exception:
            return False
    return True


def _print_pack_inventory(installed: set[str]) -> dict[str, str]:
    """Print a summary of pack status: which are fully staged, which are
    in HF cache (would copy fast), which are missing. Returns the per-pack
    state dict for downstream use."""
    state: dict[str, str] = {}
    print()
    print("Checking local cache for Stable Audio 3 model packs…")
    print()
    for p in SA3_PACKS:
        if p.key in installed:
            state[p.key] = "staged"
            print(f"  ✓ {p.label:<18s} already staged at {_pack_dir(p)}")
        elif _pack_in_hf_cache(p):
            state[p.key] = "cached"
            print(f"  ↺ {p.label:<18s} in HF cache (would symlink, no download)")
        else:
            state[p.key] = "missing"
            print(f"  · {p.label:<18s} not found")
    print()
    return state


def _ask_packs(missing: list[SA3Pack], installed: set[str], states: dict[str, str] | None = None) -> list[SA3Pack]:
    """Prompt the user once per missing pack; defaults to Y. Annotates the
    prompt with whether the pack is already in the HF cache (cached → fast
    copy) or actually needs a network download."""
    if not missing:
        return []
    states = states or {}
    print("Pick which packs to add (each is multi-GB):")
    print()
    selected: list[SA3Pack] = []
    for p in missing:
        st = states.get(p.key, "missing")
        tag = "  (in HF cache — instant symlink)" if st == "cached" else "  (will download from HF)"
        try:
            ans = input(f"  [Y/n] {p.label}{tag}: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(1)
        if ans in ("", "y", "yes"):
            selected.append(p)
    return selected


def _download_repo(repo_id: str, target_dir: Path) -> Path | None:
    """Download a HF repo into the standard $HF_HUB_CACHE and symlink
    `target_dir` to the snapshot.

    Big checkpoints (8.6 GB SA3-medium etc.) live ONCE in
    ~/.cache/huggingface/hub/, content-addressed. Each underfit clone has
    its own state/models/<key>/{base,arc}/ which is a symlink into that
    cache. Multiple clones share a single on-disk copy; cleaning a clone
    doesn't waste the big download.

    Compatibility paths:
      - target_dir already a symlink → unlinked + relinked (refreshes sha if
        the cache moved on).
      - target_dir already a regular dir with the files (older installs
        from before this change) → left alone, treated as already-staged.
      - target_dir doesn't exist or is empty → create the symlink.
    """
    import shutil
    from huggingface_hub import snapshot_download

    # ── Compatibility: legacy installs put the files DIRECTLY in target_dir
    # (via the old snapshot_download(local_dir=...) path). Don't touch
    # those — just confirm the canonical files are present.
    if (not target_dir.is_symlink()
            and target_dir.is_dir()
            and (target_dir / "model.safetensors").exists()
            and (target_dir / "model_config.json").exists()):
        print(f"  ⤷ already present at {target_dir} (legacy direct copy, leaving alone)")
        return target_dir

    # If it's already a working symlink into the cache, verify the snapshot
    # still exists; if so, no-op.
    if target_dir.is_symlink():
        resolved = target_dir.resolve()
        if (resolved / "model.safetensors").exists() and (resolved / "model_config.json").exists():
            print(f"  ⤷ already symlinked: {target_dir} → {resolved}")
            return resolved
        # Stale symlink — unlink and re-resolve.
        target_dir.unlink()

    print(f"\n→ fetching {repo_id} into HF cache ...", flush=True)
    try:
        # No local_dir → files stay in $HF_HUB_CACHE/models--<org>--<repo>/
        # snapshots/<sha>/, with content-addressed blobs/. Reused across
        # any future call (and any other underfit clone) for free.
        snapshot_path = Path(snapshot_download(repo_id=repo_id))
    except Exception as e:
        # Re-raise gated-repo errors so the orchestrator can show the
        # license-help block and abort the whole install — there's no
        # point continuing to the next pack since the same ARC license
        # gates all of them.
        if (type(e).__name__ == "GatedRepoError"
                or "Cannot access gated repo" in str(e)):
            raise
        print(f"  ✗ {type(e).__name__}: {e}")
        return None

    # Symlink target_dir to the snapshot.
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists() or target_dir.is_symlink():
        # Should only hit here if it was an empty dir created upstream.
        if target_dir.is_dir() and not target_dir.is_symlink() and not any(target_dir.iterdir()):
            target_dir.rmdir()
        elif target_dir.is_symlink():
            target_dir.unlink()
        else:
            print(f"  ✗ {target_dir} exists and is non-empty; refusing to clobber.")
            return None
    target_dir.symlink_to(snapshot_path)
    print(f"  ✓ ready at {target_dir} → {snapshot_path}")
    return snapshot_path


def _pack_dir(p: SA3Pack) -> Path:
    """Where the wizard downloads model files for pack <p.key>.
    Resolves to {models_dir} in the shipped registry."""
    return _models_dir() / p.key


def install_pack(p: SA3Pack) -> bool:
    """Download both the base (RF) and ARC repos into MODELS_DIR/<key>/
    {base,arc}/ (default STATE_DIR/models, overridable via UNDERFIT_MODELS_DIR).
    The shipped registry's {models_dir} placeholder resolves to that path
    at dashboard startup."""
    print(f"\n══ {p.label} ══")
    pack_dir = _pack_dir(p)
    if _download_repo(p.base_repo, pack_dir / "base") is None:
        return False
    if _download_repo(p.arc_repo, pack_dir / "arc") is None:
        return False
    print(f"  ✓ installed at {pack_dir}")
    return True


# ── PHASE ORCHESTRATION ──────────────────────────────────────────────────────


def _provision_backend(backend: Backend, args) -> bool:
    """After backend is picked, locate or install it.
    Flow: discover → present options → act → validate. Returns success."""
    # Non-interactive bypass: --backend-path PATH installs from that path and skips the menu.
    if getattr(args, "backend_path", None):
        path = Path(args.backend_path).expanduser().resolve()
        ok, reason = _is_valid_backend_checkout(backend, path)
        if not ok:
            print(f"\n✗ --backend-path {path}: {reason}")
            return False
        print(f"\n→ Using {backend.label} from {path}")
        if args.no_install:
            print("  (--no-install passed, skipping editable install)")
            return True
        rc = _editable_install(path, backend.extras)
        if rc != 0:
            print(f"\n✗ uv pip install -e {path} failed (exit {rc}).")
            return False
        if not _verify_backend_import(backend):
            print(f"\n✗ install finished but `import {backend.module}` still fails.")
            return False
        print(f"\n✓ {backend.label} installed from {path} and importable.")
        return True

    found = _discover_backend(backend)
    installed_at = found["installed_module_path"]
    sibling = found["sibling_checkout"]

    print()
    if installed_at:
        print(f"  ✓ already installed in this venv: {installed_at}")
    if sibling:
        print(f"  ✓ found sibling checkout: {sibling}")
    if not installed_at and not sibling:
        print(f"  no existing {backend.label} found nearby")

    if installed_at and args.no_install:
        return True  # nothing to do
    # If already installed and not forced to reinstall, accept it.
    if installed_at and not getattr(args, "force_reinstall", False):
        return True

    if args.no_install:
        print(f"\n✗ {backend.label} is not installed and --no-install was passed.")
        return False

    # No menu — wizard auto-picks. If a sibling checkout already exists, reuse
    # it; otherwise clone fresh from GitHub. To override (e.g. install from a
    # custom path), pass `--backend-path PATH`.
    sibling_dest = _underfit_root().parent / backend.sibling_dirname
    if sibling:
        print(f"\n→ using existing sibling checkout: {sibling}")
        checkout_path = sibling
    else:
        print(f"\n→ cloning {backend.label} into {sibling_dest}")
        if not _clone_backend(backend, sibling_dest):
            return False
        checkout_path = sibling_dest

    # Editable install + verify.
    rc = _editable_install(checkout_path, backend.extras)
    if rc != 0:
        print(f"\n✗ uv pip install -e failed (exit {rc}).")
        return False
    if not _verify_backend_import(backend):
        print(f"\n✗ install finished but `import {backend.module}` still fails.")
        return False
    print(f"\n✓ {backend.label} installed from {checkout_path} and importable.")
    return True


def run_backend_phase(args) -> Backend | None:
    """Returns the chosen Backend, or None if provisioning failed.

    The wizard never prompts. The default is BACKENDS[0] (sa3); pass
    `--backend sat` to opt into stable-audio-tools instead.
    """
    chosen_key = args.backend or BACKENDS[0].key
    chosen = next(b for b in BACKENDS if b.key == chosen_key)
    installed = installed_backend_keys()
    print(f"backend: {chosen.label} ({chosen.key}) — "
          f"{'already installed' if chosen.key in installed else 'will be installed'}")

    if not _provision_backend(chosen, args):
        return None
    return chosen


def run_model_phase(args, backend: Backend) -> int:
    """Download SA3 model packs from HuggingFace."""
    user = hf_whoami()
    if not user:
        print_hf_login_help()
        return 1
    print(f"\nHuggingFace: logged in as {user!r}")

    installed = staged_pack_keys()
    # Per-pack status report (staged | cached | missing) — shown BEFORE the
    # menu so the user knows what already exists and what would actually be
    # downloaded.
    pack_states = _print_pack_inventory(installed)

    # If everything's staged, skip the menu entirely.
    if all(s == "staged" for s in pack_states.values()):
        print("✓ All model packs are already staged — nothing to do.")
        return 0

    missing = [p for p in SA3_PACKS if p.key not in installed]

    if args.models is not None:
        # Explicit list — override the prompt.
        wanted = {s.strip() for s in args.models.split(",") if s.strip()}
        unknown = wanted - {p.key for p in SA3_PACKS}
        if unknown:
            print(f"✗ unknown model packs: {', '.join(unknown)}")
            return 1
        selected = [p for p in SA3_PACKS if p.key in wanted and p.key not in installed]
    else:
        selected = _ask_packs(missing, installed, pack_states)

    if not selected:
        if installed:
            print("\n✓ nothing to download (all selected packs already staged).")
            return 0
        print("\n✗ no packs selected and none are staged — at least one is required.")
        return 1

    # ── Up-front heads-up on the gated ARC license ─
    # Show this BEFORE we kick off any downloads so the user has the URL
    # in front of them while the first base model is fetching. If they
    # haven't accepted yet, the preflight below catches it and aborts
    # before we waste bandwidth.
    print()
    print("┌─ Heads-up ──────────────────────────────────────────────────────────────┐")
    print("│ The ARC checkpoints are gated by an HF license. ONE click on any of     │")
    print("│ the three SA3 ARC repos approves all three. If you haven't done this,   │")
    print("│ accept now:                                                             │")
    print("│   https://huggingface.co/stabilityai/stable-audio-3-medium              │")
    print("└─────────────────────────────────────────────────────────────────────────┘")

    # ── Preflight: probe ONE ARC repo before the first 10 GB base download.
    # Catches the most common install failure (gated-repo license not
    # accepted) in <1 s instead of after a 10-minute base download.
    print(f"\n→ verifying HuggingFace access to ARC repo {selected[0].arc_repo} …",
          flush=True)
    ok, reason = _check_arc_access(selected[0].arc_repo)
    if not ok:
        print(f"  ✗ blocked: {reason}")
        _print_license_help()
        return 1
    print("  ✓ ARC access OK")

    failures = 0
    try:
        for p in selected:
            if not install_pack(p):
                failures += 1
    except Exception as e:
        # Most likely a GatedRepoError that slipped past the preflight
        # (e.g. user accepted only one of two repo families, future
        # license changes, transient HF outage). Either way, abandon the
        # rest of the install — the same license gates every ARC repo so
        # continuing is futile — and show the help block.
        if (type(e).__name__ == "GatedRepoError"
                or "Cannot access gated repo" in str(e)):
            print(f"\n  ✗ blocked: {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:200]}")
            _print_license_help()
            return 1
        raise

    after = staged_pack_keys()
    print()
    print(f"staged packs after install: {', '.join(sorted(after)) or '(none)'}")
    if not after:
        print("✗ no packs staged — at least one is required.")
        return 1
    if failures:
        print(f"⚠ {failures} download(s) failed; see messages above.")
        return 2
    print("\n✓ Setup complete.")
    # UNDERFIT_STATE_DIR is the *parent* directory (durable, often on Drive in
    # Colab — holds runs/, datasets/, audio/, seed_loras/). Several subdirs can
    # be overridden to local SSD for live performance, with async sync back to
    # Drive. Print the active overrides so the user knows what's where.
    print(f"  UNDERFIT_STATE_DIR     = {state_dir()}    (parent, durable)")
    for env_var, default_suffix in (
        ("UNDERFIT_LOGS_DIR",        "runs"),
        ("UNDERFIT_STATE_FILES_DIR", ""),
        ("UNDERFIT_MODELS_DIR",      "models"),
        ("HF_HOME",                  ""),
    ):
        val = os.environ.get(env_var)
        if val:
            note = "(local SSD override)" if not val.startswith(str(state_dir())) else ""
            print(f"  {env_var:<22s} = {val}    {note}")
    print(f"\n  Re-launch the dashboard to pick up the current env.")
    return 0


# ── ENTRY POINT ──────────────────────────────────────────────────────────────


def maybe_download_mlx_packs(args) -> int:
    """Apple-only: pre-download the MLX weight packs (base + ARC + SAME codec +
    encoder + shared t5gemma) into the existing MLX checkout so the first MLX
    finetune / dataset-encode doesn't stall. Reuses underfit.backends.mlx_engine
    — the same download the dashboard's on-demand button uses (item 5). Degrades
    gracefully: if the MLX checkout / venv isn't set up, the weights just
    download lazily on first use instead."""
    import subprocess as _sp
    try:
        from underfit.backends import mlx_engine
    except Exception as e:
        print(f"\n(could not import the MLX engine — {e}. MLX weights will "
              f"download on first use.)")
        return 0
    root = mlx_engine.mlx_root_or_none()
    if not root:
        print("\n▸ Apple Silicon detected, but the MLX checkout wasn't found.\n"
              "  Set UNDERFIT_MLX_ROOT (or clone stable-audio-3 as a sibling and\n"
              "  create its .venv). MLX weights will download on first use.")
        return 0
    all_keys = [p.key for p in SA3_PACKS]
    if args.models:
        wanted = {s.strip() for s in args.models.split(",") if s.strip()}
        keys = [k for k in all_keys if k in wanted]
    else:
        keys = all_keys
    print(f"\n▸ MLX weight packs → {os.path.join(root, 'models', 'mlx')}")
    rc = 0
    for key in keys:
        try:
            dit = mlx_engine.map_model_name(key)
        except Exception:
            continue
        if mlx_engine.mlx_model_available(dit):
            print(f"  ✓ {dit}: already present")
            continue
        _missing, gb = mlx_engine.mlx_missing_pack(dit)
        print(f"  ↓ {dit}: downloading ~{gb} GB …", flush=True)
        try:
            cmd = mlx_engine.build_mlx_download_cmd(dit)
        except FileNotFoundError as e:
            print(f"    skipped — {e}")
            rc = 1
            continue
        try:
            _sp.run(cmd, cwd=root, check=True)
            print(f"  ✓ {dit}: done")
        except Exception as e:
            print(f"    download failed ({e}) — will retry on first use")
            rc = 1
    return rc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="underfit-setup",
        description="Install a Stable Audio backend (sa3 by default, sat is opt-in) "
                    "and download SA3 model packs for underfit.",
    )
    p.add_argument(
        "--backend",
        choices=[b.key for b in BACKENDS],
        default=None,
        help=f"Which backend to install (default: {BACKENDS[0].key}).",
    )
    p.add_argument(
        "--backend-path",
        metavar="PATH",
        default=None,
        help=("Path to a local backend source tree (skips the auto clone-or-reuse step). "
              "Must contain pyproject.toml + the backend's package directory. "
              "Useful for Colab and other headless setups."),
    )
    p.add_argument(
        "--no-install",
        action="store_true",
        help="Verify which backend is installed but do not run pip install.",
    )
    p.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip the SA3 model-download phase entirely.",
    )
    p.add_argument(
        "--models",
        metavar="key1,key2,...",
        default=None,
        help=("Comma-separated list of SA3 pack keys to install non-interactively "
              "(e.g. sa3-medium,sa3-sm-music). Already-installed packs are skipped."),
    )
    args = p.parse_args(argv)

    backend = run_backend_phase(args)
    if backend is None:
        return 1

    if args.skip_models:
        print("\n(skipping model phase as requested)")
        return 0

    import platform as _pf
    if _pf.system() == "Darwin" and _pf.machine() == "arm64":
        # Apple Silicon: the MLX engine (Metal GPU) is the primary path, so
        # download the MLX weight packs. The torch packs (run_model_phase) are
        # only needed for the torch/MPS engine — skip them here to avoid pulling
        # ~10 GB of torch weights an MLX-only Mac won't use.
        return maybe_download_mlx_packs(args)

    return run_model_phase(args, backend)


if __name__ == "__main__":
    raise SystemExit(main())
