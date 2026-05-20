"""Loss helpers for diffusion training.

Vendored from stable-audio-tools/training/utils.py.
"""
import torch


def compute_normalized_mse(pred, target, loss_mask, loss_normalization="none", loss_norm_eps=1e-6):
    """MSE normalized by detached target magnitude (per timestep / sample / channel).

    `loss_normalization`: "none" | "timestep" | "sample" | "sample_channel"
    Returns per-element MSE [B, C, T], normalized for signal positions only.
    """
    mse = (pred - target) ** 2
    if loss_normalization == "none":
        return mse

    if loss_mask is None:
        mask_expanded = torch.ones(pred.shape[0], 1, pred.shape[2], device=pred.device, dtype=torch.bool)
    else:
        mask_expanded = loss_mask.unsqueeze(1)

    with torch.no_grad():
        if loss_normalization == "timestep":
            mag_sq = torch.mean((target - torch.mean(target, dim=1, keepdim=True)) ** 2, dim=1, keepdim=True) + loss_norm_eps
        else:
            masked_targets = torch.where(mask_expanded, target, float("nan"))
            if loss_normalization == "sample":
                m = torch.nanmean(masked_targets, dim=(1, 2), keepdim=True)
                mag_sq = torch.nanmean((masked_targets - m) ** 2, dim=(1, 2), keepdim=True) + loss_norm_eps
            elif loss_normalization == "sample_channel":
                m = torch.nanmean(masked_targets, dim=2, keepdim=True)
                mag_sq = torch.nanmean((masked_targets - m) ** 2, dim=2, keepdim=True) + loss_norm_eps
            else:
                raise ValueError(f"Unknown loss normalization mode: {loss_normalization}")
            mag_sq = torch.where(torch.isnan(mag_sq), torch.ones_like(mag_sq), mag_sq)

    normalized_mse = mse / mag_sq
    return torch.where(mask_expanded, normalized_mse, mse)


def compute_masked_loss(loss_full, loss_mask, mask_padding_attention, mask_loss_weight=0.0):
    """Combine signal and padding loss according to attention/mask settings.

    Returns (scalar_loss, signal_mean, padding_mean) — last two detached for logging.
    """
    signal = torch.where(loss_mask.unsqueeze(1), loss_full, 0.0)
    signal_sum = signal.sum(dim=(1, 2))
    n_channels = loss_full.shape[1]
    signal_count = loss_mask.sum(dim=1) * n_channels

    padding = torch.where(~loss_mask.unsqueeze(1), loss_full, 0.0)
    padding_sum = padding.sum(dim=(1, 2))
    padding_count = (~loss_mask).sum(dim=1) * n_channels

    if mask_padding_attention:
        per_sample_loss = signal_sum / (signal_count + 1e-8)
        loss = per_sample_loss.mean()
    else:
        w = mask_loss_weight
        denom = signal_count + w * padding_count + 1e-8
        per_sample_loss = (signal_sum + w * padding_sum) / denom
        loss = per_sample_loss.mean()

    signal_mean = signal.sum() / (signal_count.sum() + 1e-8)
    padding_mean = padding.sum() / (padding_count.sum() + 1e-8)
    return loss, signal_mean.detach(), padding_mean.detach()
