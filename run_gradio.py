"""Underfit gradio launcher.

Backend-agnostic wrapper. Selects between the sat and sa3 backends and
delegates UI construction to the backend's create_gradio_ui().

CLI is the same as SAT-dev's run_gradio.py so dashboard launches don't need
to change beyond pointing at this script.
"""
import sys
import warnings

# Always-on suppression of two torchaudio UserWarnings that fire on every
# inference call (the SA3 pretransform's mel-spec is reconstructed per call,
# so the warnings re-emit each generation). These two are specifically known
# noise; the broader filterwarnings("ignore") below covers the rest unless
# --verbose was passed. Done as targeted filters so they survive any library
# that calls warnings.resetwarnings() during init.
warnings.filterwarnings(
    "ignore",
    message=r".*'onesided' has been deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*At least one mel filterbank has all zero values.*",
    category=UserWarning,
)

if "--verbose" not in sys.argv:
    import os as _os
    _os.environ.setdefault("PYTHONWARNINGS", "ignore")
    warnings.filterwarnings("ignore")

import argparse
import os

import torch

from underfit.backends import get_backend


def main(args):
    # --- MLX engine: delegate to the sibling stable-audio-3 MLX gradio. ---
    # Purely additive; the torch path below is unchanged for engine=torch.
    engine = (getattr(args, "engine", None) or os.environ.get("UNDERFIT_ENGINE") or "torch").strip().lower()
    if engine == "mlx":
        from underfit.backends import mlx_engine
        model = mlx_engine.resolve_dit_model(args.model_config, args.pretrained_name)
        port_env = os.environ.get("GRADIO_SERVER_PORT")
        port = int(port_env) if port_env and port_env.isdigit() else None
        sys.exit(mlx_engine.run_mlx_gradio(model, args.lora_ckpt_path, share=True, port=port))

    backend = get_backend(args.backend)
    print(f"Using backend: {backend.NAME}", flush=True)

    try:
        from stable_audio_tools.verbose import set_verbose
        set_verbose(args.verbose)
    except ImportError:
        pass

    torch.manual_seed(42)

    interface = backend.create_gradio_ui(
        model_config_path=args.model_config,
        ckpt_path=args.ckpt_path,
        pretrained_name=args.pretrained_name,
        pretransform_ckpt_path=args.pretransform_ckpt_path,
        model_half=args.model_half,
        gradio_title=args.title,
        lora_ckpt_paths=args.lora_ckpt_path,
        default_prompt=args.default_prompt,
    )
    interface.queue()
    interface.launch(
        share=True,
        auth=(args.username, args.password) if args.username is not None else None,
        js=getattr(interface, "_sao_js", None),
        theme=getattr(interface, "_sao_theme", None),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gradio interface (Underfit)")
    parser.add_argument("--backend", default=None, help="sat | sa3 (default: env UNDERFIT_BACKEND or auto)")
    parser.add_argument("--engine", default=None, choices=["torch", "mlx"],
                        help="torch | mlx (default: env UNDERFIT_ENGINE, else torch). "
                             "mlx launches the Apple-Silicon MLX gradio in the sibling "
                             "stable-audio-3 checkout.")
    parser.add_argument("--pretrained-name", type=str, required=False)
    parser.add_argument("--model-config", type=str, required=False)
    parser.add_argument("--ckpt-path", type=str, required=False)
    parser.add_argument("--pretransform-ckpt-path", type=str, required=False)
    parser.add_argument("--username", type=str, required=False)
    parser.add_argument("--password", type=str, required=False)
    parser.add_argument("--model-half", action="store_true", default=True)
    parser.add_argument("--title", type=str, required=False)
    parser.add_argument("--lora-ckpt-path", type=str, nargs="*", required=False)
    parser.add_argument("--default-prompt", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
