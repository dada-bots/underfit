#!/usr/bin/env bash
# Underfit installer
#
# Usage:
#     ./install.sh             # full flow: install uv (if missing) + uv sync + underfit-setup
#     ./install.sh --no-setup  # stop after `uv sync`, skip the underfit-setup wizard
#
# Idempotent: re-running upgrades anything missing and leaves the rest alone.

set -euo pipefail

UNDERFIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$UNDERFIT_DIR"

SKIP_SETUP=0
for arg in "$@"; do
    case "$arg" in
        --no-setup) SKIP_SETUP=1 ;;
        -h|--help)
            sed -n '2,9p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "unknown flag: $arg"
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
say "syncing dependencies (uv sync) …"
uv sync

# ── 3. wizard ──────────────────────────────────────────────────────────────
if [ "$SKIP_SETUP" -eq 1 ]; then
    say "skipping underfit-setup (--no-setup passed)"
    say "done. Activate the venv with 'source .venv/bin/activate' or use 'uv run …'."
    exit 0
fi
say "launching underfit-setup …"
uv run python -m underfit.cli.setup
