"""Compute SVD bases directly from a checkpoint's state_dict tensors.

Avoids constructing the full model (which can segfault on certain package
versions). Iterates every tensor whose shape implies a Linear/Conv1d weight
(2-D or 3-D) under the model.* and conditioner.* prefixes, SVD's it on CPU
in float32 with deterministic sign canonicalization, saves U/V/S keyed by
the original state_dict key.

Usage:
    python -m underfit.utils.compute_svd \
        --ckpt /path/to/base.ckpt \
        --output /path/to/svd_bases.pt
"""
import argparse
import torch
from .state_dict import load_ckpt_state_dict, unwrap_state_dict


def canonicalize_signs(U, Vh):
    max_abs_idx = U.abs().argmax(dim=0)
    signs = U[max_abs_idx, torch.arange(U.shape[1])].sign()
    signs[signs == 0] = 1
    U  = U * signs.unsqueeze(0)
    Vh = Vh * signs.unsqueeze(1)
    return U, Vh


def is_weight_key(k):
    # Linear/Conv1d weights end with ".weight" and live under model.* or
    # conditioner.* prefixes. Skip biases and layernorm scales only.
    # We do NOT skip "embedding" substrings — the conditioner's seconds_total
    # embedder is a Linear nested under `embedder.embedding.1.weight`, and
    # excluding it would diverge from the old factory-based SVD output.
    if not k.endswith(".weight"):
        return False
    for skip in (".norm.", ".ln.", ".rms."):
        if skip in k:
            return False
    return k.startswith("model.") or k.startswith("conditioner.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-type", default=None,
                   help="passed to unwrap_state_dict (e.g. diffusion_cond)")
    args = p.parse_args()

    print(f"Loading {args.ckpt} ...", flush=True)
    sd = load_ckpt_state_dict(args.ckpt)
    sd = unwrap_state_dict(sd, args.model_type)
    print(f"  {len(sd)} tensors", flush=True)

    bases = {}
    skipped = 0
    for i, (k, w) in enumerate(sd.items()):
        if not is_weight_key(k):
            continue
        if w.dim() < 2:
            continue
        # Reshape conv weights (out, in, k) → 2-D (out, in*k)
        W2 = w.view(w.shape[0], -1).float()
        m, n = W2.shape
        if min(m, n) < 2:
            skipped += 1
            continue
        try:
            U, S, Vh = torch.linalg.svd(W2, full_matrices=False)
        except Exception as e:
            print(f"  SVD failed on {k} (shape {tuple(w.shape)}): {e}", flush=True)
            skipped += 1
            continue
        U, Vh = canonicalize_signs(U, Vh)
        V = Vh.T
        bases[k] = {
            "U": U.to(torch.float16),
            "V": V.to(torch.float16),
            "S": S.to(torch.float32),
            "shape": list(w.shape),
        }
        if (len(bases) % 25) == 0:
            print(f"  {len(bases)} layers SVD'd...", flush=True)

    print(f"\n{len(bases)} bases, {skipped} skipped", flush=True)
    total_params = sum(d["U"].numel() + d["V"].numel() + d["S"].numel() for d in bases.values())
    print(f"Total stored params: {total_params:,} (~{total_params*2/1024/1024:.0f} MB)", flush=True)

    print(f"Saving to {args.output} ...", flush=True)
    torch.save(bases, args.output)
    print("Done.", flush=True)

    # Spot-check: reconstruct 3 layers
    print("\nSpot-check reconstruction:", flush=True)
    checked = 0
    for k, data in bases.items():
        if checked >= 3: break
        U = data["U"].float()
        V = data["V"].float()
        S = data["S"]
        W_recon = U @ torch.diag(S) @ V.T
        W_orig = sd[k].view(sd[k].shape[0], -1).float()
        err = (W_recon - W_orig).abs().max().item()
        print(f"  {k}: U={list(data['U'].shape)} V={list(data['V'].shape)} max_err={err:.6f}", flush=True)
        checked += 1


if __name__ == "__main__":
    main()
