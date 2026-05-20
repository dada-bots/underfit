"""Timestep samplers for diffusion training.

Vendored from stable-audio-tools/inference/sampling.py — these definitions
are identical in both backends, so vendoring lets the loop be backend-free.
"""
import torch
import torch.distributions as dist


def sample_timesteps_logsnr(batch_size, mean_logsnr=-1.2, std_logsnr=2.0):
    """Sample t from a Gaussian on logSNR (Eq. logsnr = ln((1-t)/t)) → t = sigmoid(-logsnr)."""
    logsnr = torch.randn(batch_size) * std_logsnr + mean_logsnr
    return torch.sigmoid(-logsnr).clamp(1e-4, 1 - 1e-4)


def sample_timesteps_logsnr_uniform(batch_size, min_logsnr=-6.0, max_logsnr=5.0):
    """Sample t from a uniform on logSNR."""
    logsnr = torch.rand(batch_size) * (max_logsnr - min_logsnr) + min_logsnr
    return torch.sigmoid(-logsnr).clamp(1e-4, 1 - 1e-4)


def truncated_logistic_normal_rescaled(shape, left_trunc=0.075, right_trunc=1.0):
    """Truncated logistic-normal, rescaled to [0, 1)."""
    logits = torch.randn(shape)
    normal_dist = dist.Normal(0, 1)
    cdf_values = normal_dist.cdf(logits)
    lower_bound = normal_dist.cdf(torch.logit(torch.tensor(left_trunc)))
    upper_bound = normal_dist.cdf(torch.logit(torch.tensor(right_trunc)))
    truncated_cdf_values = lower_bound + (upper_bound - lower_bound) * cdf_values
    truncated_samples = torch.sigmoid(normal_dist.icdf(truncated_cdf_values))
    return (truncated_samples - left_trunc) / (right_trunc - left_trunc)


def sample_t(timestep_sampler, batch_size, device, options=None):
    """Dispatch to the configured timestep sampler.

    `timestep_sampler` is one of: "uniform", "logit_normal", "trunc_logit_normal",
    "log_snr", "log_snr_uniform". `options` is an optional dict (currently used
    only by log_snr* variants for {mean,std}_logsnr / {min,max}_logsnr).
    """
    options = options or {}
    if timestep_sampler == "uniform":
        t = torch.rand(batch_size, device=device)
    elif timestep_sampler == "logit_normal":
        t = torch.sigmoid(torch.randn(batch_size, device=device))
    elif timestep_sampler == "trunc_logit_normal":
        # trunc + flip to match SAT-dev: t = 1 - truncated_logistic_normal_rescaled(...)
        t = (1 - truncated_logistic_normal_rescaled(batch_size)).to(device)
    elif timestep_sampler == "log_snr":
        t = sample_timesteps_logsnr(
            batch_size,
            mean_logsnr=options.get("mean_logsnr", -1.2),
            std_logsnr=options.get("std_logsnr", 2.0),
        ).to(device)
    elif timestep_sampler == "log_snr_uniform":
        t = sample_timesteps_logsnr_uniform(
            batch_size,
            min_logsnr=options.get("min_logsnr", -6.0),
            max_logsnr=options.get("max_logsnr", 5.0),
        ).to(device)
    else:
        raise ValueError(f"Invalid timestep_sampler: {timestep_sampler}")
    return t
