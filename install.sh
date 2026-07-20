#!/usr/bin/env bash
# Underfit installer
#
# Usage:
#     ./install.sh                  # full flow: install uv (if missing) + uv sync + underfit-setup
#     ./install.sh --no-setup       # stop after `uv sync`, skip the underfit-setup wizard
#     ./install.sh --backend sat    # opt into stable-audio-tools (default is sa3)
#
# Idempotent: re-running upgrades anything missing and leaves the rest alone.

set -euo pipefail

UNDERFIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$UNDERFIT_DIR"

SKIP_SETUP=0
BACKEND=""
while [ $# -gt 0 ]; do
    case "$1" in
        --no-setup) SKIP_SETUP=1; shift ;;
        --backend)
            if [ $# -lt 2 ]; then
                echo "✗ --backend requires a value (sa3 | sat)"; exit 1
            fi
            BACKEND="$2"; shift 2 ;;
        --backend=*) BACKEND="${1#--backend=}"; shift ;;
        -h|--help)
            sed -n '2,10p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "unknown flag: $1"
            echo "use --help for usage"
            exit 1
            ;;
    esac
done

say() { printf "\033[1;36m▸\033[0m %s\n" "$*"; }
err() { printf "\033[1;31m✗\033[0m %s\n" "$*" >&2; }

# ── 1. uv ──────────────────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    say "uv not found, installing via official Astral installer …"
    if ! command -v curl >/dev/null 2>&1; then
        err "curl is required to bootstrap uv. Install curl (or install uv manually from https://docs.astral.sh/uv/) and re-run."
        exit 1
    fi
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The Astral installer drops `uv` into one of these. Add to PATH for this
    # shell so the next steps can call it.
    for candidate in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
        if [ -x "$candidate/uv" ]; then
            export PATH="$candidate:$PATH"
            break
        fi
    done
    if ! command -v uv >/dev/null 2>&1; then
        err "uv installed, but not on PATH. Open a new shell (or 'source ~/.bashrc') and re-run."
        exit 1
    fi
fi
say "uv $(uv --version | awk '{print $2}') ready"

# ── 2. deps ────────────────────────────────────────────────────────────────
say "syncing dependencies (uv sync --inexact) …"
# --inexact: keep packages uv sync didn't put there. The setup wizard
# installs the diffusion backend (stable-audio-3 / stable-audio-tools) via
# `uv pip install -e <path>[lora,ui]` after this step. Plain `uv sync`
# would treat that backend install as "extraneous" on the next run and
# remove it — leaving the dashboard with NONE_IMPORTABLE on startup until
# the wizard is re-run. --inexact skips that prune.
uv sync --inexact

# ── 2b. ffmpeg (optional but recommended) ───────────────────────────────────
# The dashboard uses ffmpeg for ground-truth preview clips + clip downloads,
# and the trainers (torch + MLX demos) use it to write MP3 demos. Missing
# ffmpeg is NOT fatal — GT previews are skipped and demos fall back to WAV —
# but MP3 is smaller and previews are nicer, so recommend installing it.
if command -v ffmpeg >/dev/null 2>&1; then
    say "ffmpeg found ($(command -v ffmpeg)) — MP3 demos + audio previews enabled"
else
    say "ffmpeg NOT found (optional): dashboard previews are skipped and training"
    say "  demos fall back to WAV. To enable MP3 (fast, prebuilt on Apple Silicon):"
    printf '    \033[1mbrew install ffmpeg\033[0m\n'
fi

# ── 3. wizard ──────────────────────────────────────────────────────────────
if [ "$SKIP_SETUP" -eq 1 ]; then
    say "skipping underfit-setup (--no-setup passed)"
    say "done — now run ./run.sh to start the dashboard."
    exit 0
fi
say "launching underfit-setup …"
if [ -n "$BACKEND" ]; then
    uv run python -m underfit.cli.setup --backend "$BACKEND"
else
    uv run python -m underfit.cli.setup
fi

echo
say "all done — now run ./run.sh to start the dashboard."
