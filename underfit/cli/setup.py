"""Interactive setup wizard for underfit.

Two phases:

  1. Backend pick — choose stable-audio-tools (sat_dev) or stable-audio-3 (sa3),
     install via uv pip / pip if missing.
  2. Model download — pick which Stable Audio 3 model packs to pull from the
     stabilityai HF org and stage into the dashboard's per-instance model
     registry (STATE_DIR/models/). Each repo bundles its own T5Gemma tokenizer
     as a subfolder, so no separate t5gemma download is required.

Selection logic (backend phase):
  - Default to whichever backend is already installed
  - If both installed: default to sat_dev
  - If neither installed: default to sat_dev
  - User can override with arrow / number keys

Selection logic (model phase):
  - Detect which SA3 model packs are already staged in STATE_DIR/models/<key>.json
  - For each missing pack, ask Y/n (defaults to Y).
  - At least one pack must end up installed.

Re-running the wizard is safe: already-installed models are reported and
skipped; the wizard only offers to download the ones you're missing.

Non-interactive use:
    underfit-setup --backend sat_dev --no-install   # backend phase only, verify
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
        key="sat_dev",
        label="stable-audio-tools",
        module="stable_audio_tools",
        clone_url="https://github.com/Stability-AI/stable-audio-tools.git",
        sibling_dirname="stable-audio-tools",
        description="Stability-AI/stable-audio-tools",
        extras=("train",),  # pytorch_lightning, prefigure, torchmetrics, etc.
    ),
    Backend(
        key="sa3",
        label="stable-audio-3",
        module="stable_audio_3",
        clone_url="https://github.com/Stability-AI/stable-audio-3.git",
        sibling_dirname="stable-audio-3",
        description="Stability-AI/stable-audio-3 — simplified SA3 codebase",
        extras=("lora",),   # dill, pytorch_lightning — required by SA3's dataloader
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


def _by_key(key: str) -> Backend:
    for b in BACKENDS:
        if b.key == key:
            return b
    raise KeyError(key)


def backend_installed(backend: Backend) -> bool:
    return importlib.util.find_spec(backend.module) is not None


def installed_backend_keys() -> list[str]:
    return [b.key for b in BACKENDS if backend_installed(b)]


def choose_default_backend(installed: list[str]) -> str:
    for b in BACKENDS:
        if b.key in installed:
            return b.key
    return BACKENDS[0].key


def _arrow_select(title: str, lines_per_option: list[list[str]], default_idx: int) -> int:
    """Up/down arrow single-select with Enter to confirm.

    Each option can render across multiple lines (e.g. label + description);
    only the *first* line of each option is highlighted with the cursor.

    Falls back to a numeric prompt if stdin isn't a TTY (CI / piped input).
    """
    n = len(lines_per_option)
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        # Non-interactive fallback: print menu + accept a typed digit.
        print(title)
        for i, opt in enumerate(lines_per_option, 1):
            print(f"  [{i}] {opt[0]}")
            for extra in opt[1:]:
                print(f"      {extra}")
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
                sys.stdout.write(f"   {line}\n")
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


def prompt_backend(default_key: str, installed: list[str]) -> str:
    default_idx = next(i for i, b in enumerate(BACKENDS) if b.key == default_key)
    lines = []
    for b in BACKENDS:
        tag = " [installed]" if b.key in installed else ""
        lines.append([f"{b.label}{tag}", b.description])
    idx = _arrow_select("Which Stable Audio backend should underfit use?", lines, default_idx)
    return BACKENDS[idx].key


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
    and pytorch_lightning; sat_dev's `train` extra brings pytorch_lightning,
    prefigure, etc. Without them, training fails at import time."""
    target = f"{path}[{','.join(extras)}]" if extras else str(path)
    cmd = [*_install_command(), "-e", target]
    print(f"\n→ {' '.join(cmd)}\n", flush=True)
    return subprocess.run(cmd).returncode


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


def _prompt_path(backend: Backend) -> Path | None:
    """Prompt the user for a backend checkout path, validating each entry.
    Returns the resolved Path on success, or None if the user gave up."""
    while True:
        try:
            raw = input(f"\nPath to existing {backend.label} checkout (blank to cancel): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not raw:
            return None
        p = Path(raw).expanduser().resolve()
        ok, reason = _is_valid_backend_checkout(backend, p)
        if ok:
            return p
        print(f"  ✗ {reason}. Try again.")


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


def _checkpoints_dir() -> Path:
    """Where to download SA3 model packs. Defaults to STATE_DIR/checkpoints —
    on Colab, point UNDERFIT_CHECKPOINTS_DIR at /content/checkpoints so model
    files live on local SSD (fast reads) instead of Drive (slow + FUSE-flaky)."""
    env = os.environ.get("UNDERFIT_CHECKPOINTS_DIR")
    if env:
        return Path(env).expanduser()
    return state_dir() / "checkpoints"


def staged_pack_keys() -> set[str]:
    """Set of SA3 pack keys with downloaded model files on disk."""
    d = _checkpoints_dir()
    if not d.is_dir():
        return set()
    return {p.parent.parent.name for p in d.glob("*/base/model.safetensors") if p.is_file()}


def _ask_packs(missing: list[SA3Pack], installed: set[str]) -> list[SA3Pack]:
    """Prompt the user once per missing pack; defaults to Y."""
    if not missing:
        return []
    print()
    print("Which Stable Audio 3 model packs would you like to download?")
    print("(downloads from the stabilityai HF org; each pack is multi-GB)")
    print()
    selected: list[SA3Pack] = []
    for p in missing:
        try:
            ans = input(f"  [Y/n] {p.label}: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(1)
        if ans in ("", "y", "yes"):
            selected.append(p)
    return selected


def _download_repo(repo_id: str, target_dir: Path) -> Path | None:
    """Download a repo into target_dir under the dashboard's state tree.

    huggingface_hub.snapshot_download with local_dir=...  uses the HF cache as
    a backing store: if the repo is already cached locally, the files land in
    target_dir as symlinks pointing at the cache (no re-download). If a fresh
    download is needed, it goes through the cache and is symlinked into
    target_dir. So "already downloaded somewhere" (= in HF cache) is honored
    for free.
    """
    from huggingface_hub import snapshot_download
    target_dir.mkdir(parents=True, exist_ok=True)
    if any(target_dir.iterdir()):
        # Heuristic: if the model files we need are already present (e.g. from
        # a prior run), skip the HF roundtrip. snapshot_download is cheap when
        # nothing changed, but skipping eliminates a network call entirely.
        if (target_dir / "model.safetensors").exists() and (target_dir / "model_config.json").exists():
            print(f"  ⤷ already present at {target_dir}, skipping download")
            return target_dir
    print(f"\n→ downloading {repo_id} → {target_dir} ...", flush=True)
    try:
        local = snapshot_download(repo_id=repo_id, local_dir=str(target_dir))
    except Exception as e:
        print(f"  ✗ {type(e).__name__}: {e}")
        return None
    print(f"  ✓ ready at {local}")
    return Path(local)


def _pack_dir(p: SA3Pack) -> Path:
    """Where the wizard downloads model files for pack <p.key>.
    Resolves to {checkpoints_dir} in the shipped registry."""
    return _checkpoints_dir() / p.key


def install_pack(p: SA3Pack) -> bool:
    """Download both the base (RF) and ARC repos into STATE_DIR/checkpoints/<key>/
    {base,arc}/. The shipped registry's {checkpoints_dir} placeholder will
    resolve to these paths at dashboard startup."""
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

    # Build the menu.
    sibling_dest = _underfit_root().parent / backend.sibling_dirname
    options: list[tuple[str, str, str]] = []  # (action, head, sub)
    if sibling:
        options.append(("use_sibling", f"Use existing sibling checkout", f"{sibling}"))
    options.append(("clone", f"Clone fresh from GitHub", f"→ {sibling_dest}"))
    options.append(("paste_path", "Paste a path to an existing checkout", "you'll be asked next"))

    idx = _arrow_select(
        f"How would you like to provide {backend.label}?",
        [[head, sub] for (_, head, sub) in options],
        default_idx=0,
    )
    action = options[idx][0]

    # Resolve to a checkout path.
    checkout_path: Path | None = None
    if action == "use_sibling":
        checkout_path = sibling
    elif action == "clone":
        if not _clone_backend(backend, sibling_dest):
            return False
        checkout_path = sibling_dest
    elif action == "paste_path":
        checkout_path = _prompt_path(backend)
        if checkout_path is None:
            print("\n✗ no path provided, aborting.")
            return False

    assert checkout_path is not None

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
    """Returns the chosen Backend, or None if provisioning failed."""
    installed = installed_backend_keys()
    print(f"detected installed backends: {', '.join(installed) if installed else 'none'}")

    if args.backend:
        chosen_key = args.backend
        print(f"using --backend {chosen_key}")
    else:
        chosen_key = prompt_backend(choose_default_backend(installed), installed)
    chosen = _by_key(chosen_key)

    if not _provision_backend(chosen, args):
        return None
    return chosen


def run_model_phase(args, backend: Backend) -> int:
    """Returns process exit code. Only runs for sa3 backend."""
    if backend.key != "sa3":
        return 0  # nothing to do for sat_dev (it sources its own checkpoints)

    user = hf_whoami()
    if not user:
        print_hf_login_help()
        return 1
    print(f"\nHuggingFace: logged in as {user!r}")

    installed = staged_pack_keys()
    if installed:
        print(f"already staged: {', '.join(sorted(installed))}")

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
        selected = _ask_packs(missing, installed)

    if not selected:
        if installed:
            print("\n✓ nothing to download (all selected packs already staged).")
            return 0
        print("\n✗ no packs selected and none are staged — at least one is required.")
        return 1

    failures = 0
    for p in selected:
        if not install_pack(p):
            failures += 1

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
    print(f"  Dashboard state dir: {state_dir()}")
    print(f"  Launch the dashboard with UNDERFIT_STATE_DIR={state_dir()}")
    return 0


# ── ENTRY POINT ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="underfit-setup",
        description="Pick a Stable Audio backend and download SA3 model packs for underfit.",
    )
    p.add_argument(
        "--backend",
        choices=[b.key for b in BACKENDS],
        help="Skip the interactive backend prompt and select this backend.",
    )
    p.add_argument(
        "--backend-path",
        metavar="PATH",
        default=None,
        help=("Path to a local backend source tree (skips the discover/clone/paste menu). "
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

    return run_model_phase(args, backend)


if __name__ == "__main__":
    raise SystemExit(main())
