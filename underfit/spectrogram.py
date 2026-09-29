"""Spectrogram rendering, shared by the dashboard and the training loop.

Two entry points over one implementation, so both produce identical images:

  render_to_jpg(y, sr, jpg_path)   from audio already in memory
  render_file(path, jpg_path, ...) decode a file first, then render

The training loop uses the first: it holds each demo as a tensor when it writes
the mp3, so re-decoding that mp3 later just to draw a 300x60 image is wasted
work — and on some platforms the decode is where libsndfile's non-thread-safe
MPEG path bites. The dashboard uses the second for audio it did not produce
(ground truth, imports) and skips it entirely when a .jpg already exists.

No module-level side effects: safe to import from a trainer, a worker process,
or the dashboard.

torch is optional. The MLX trainer's venv has none, and loads this file by path
(UNDERFIT_SPECTROGRAM_MODULE) to render its demos; without torch the STFT and
resample run in numpy and produce the same image to within JPEG noise. With
torch present the torch path is used, so dashboard output is unchanged.
"""

import numpy as np
from functools import lru_cache
from PIL import Image

try:
    import torch
except ImportError:
    torch = None


SPEC_BANDS = [
    (0, 200, (1.0, 0.0, 0.0)),      # Bass -> Red
    (200, 1500, (0.0, 1.0, 0.0)),   # Mid  -> Green
    (1500, 16000, (0.0, 0.0, 1.0)), # High -> Blue
]
SPEC_W, SPEC_H = 300, 60

_BAND_COLORS = np.array([c for _, _, c in SPEC_BANDS], dtype=np.float32)

_F_SP = 200.0 / 3
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = np.log(6.4) / 27.0

def _hz_to_mel(hz):
    hz = np.asarray(hz, dtype=np.float64)
    # np.where evaluates both branches — mask the log input so the linear-region
    # values don't trigger log(0) warnings (they're discarded anyway).
    log_term = np.log(np.maximum(hz, _MIN_LOG_HZ) / _MIN_LOG_HZ) / _LOGSTEP
    return np.where(hz >= _MIN_LOG_HZ, _MIN_LOG_MEL + log_term, hz / _F_SP)


def _mel_to_hz(mels):
    mels = np.asarray(mels, dtype=np.float64)
    return np.where(mels >= _MIN_LOG_MEL,
                    _MIN_LOG_HZ * np.exp(_LOGSTEP * (mels - _MIN_LOG_MEL)),
                    _F_SP * mels)


def _mel_frequencies(n_mels, fmax, fmin=0.0):
    return _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels))

@lru_cache(maxsize=8)
def _mel_filterbank(n_mels, n_fft, sr, fmax):
    pts = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(fmax), n_mels + 2))
    fft_f = np.linspace(0, sr / 2, n_fft // 2 + 1)
    filt = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        lo, ce, hi = pts[i], pts[i + 1], pts[i + 2]
        left = (fft_f - lo) / max(ce - lo, 1e-10)
        right = (hi - fft_f) / max(hi - ce, 1e-10)
        filt[i] = np.maximum(0, np.minimum(left, right))
    # Slaney area-normalization (matches librosa norm='slaney' default)
    enorm = (2.0 / (pts[2:n_mels + 2] - pts[0:n_mels])).astype(np.float32)
    filt *= enorm[:, None]
    return filt

def _stft_power_np(y_ch, n_fft, hop_length):
    """numpy twin of torch.stft(center=True, pad_mode='reflect', periodic hann)
    followed by |.|^2, shaped (n_fft // 2 + 1, n_frames)."""
    y = np.pad(np.asarray(y_ch, dtype=np.float32), n_fft // 2, mode="reflect")
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop_length] * win
    spec = np.fft.rfft(frames, axis=-1)
    return (spec.real ** 2 + spec.imag ** 2).astype(np.float32).T

def _melspectrogram(y_ch, sr, n_mels=30, fmax=16000, hop_length=2048, n_fft=2048):
    fb = _mel_filterbank(n_mels, n_fft, sr, fmax)
    if torch is None:
        return fb @ _stft_power_np(y_ch, n_fft, hop_length)
    y_t = torch.from_numpy(np.ascontiguousarray(y_ch)).float()
    win = torch.hann_window(n_fft)
    spec = torch.stft(y_t, n_fft=n_fft, hop_length=hop_length, window=win,
                      center=True, return_complex=True, pad_mode='reflect')
    return (torch.from_numpy(fb) @ spec.abs().square()).numpy()


def resample_linear(y, sr, target_sr):
    """Linear resample matching the dashboard's
    F.interpolate(mode='linear', align_corners=False), in numpy, so a torch-free
    caller feeds render_to_jpg the same samples the dashboard would. `y` is
    (channels, samples) or (samples,); returns float32 of the same rank."""
    y = np.asarray(y, dtype=np.float32)
    if sr == target_sr:
        return y
    n_in = y.shape[-1]
    n_out = int(round(n_in * target_sr / sr))
    # Source positions in float32 like torch's kernel: over millions of samples
    # float64 positions drift into different neighbour pairs.
    f32 = np.float32
    src = (np.arange(n_out, dtype=f32) + f32(0.5)) * f32(n_in / n_out) - f32(0.5)
    np.maximum(src, 0, out=src)
    i0 = src.astype(np.int64)
    i1 = np.minimum(i0 + 1, n_in - 1)
    lam = src - i0.astype(f32)
    return (1 - lam) * y[..., i0] + lam * y[..., i1]

def _power_to_db(S, top_db=80.0):
    log_spec = 10.0 * np.log10(np.maximum(S, 1e-10))
    return np.maximum(log_spec - log_spec.max(), -top_db)

def _mel_channel(y_ch, sr, n_mels=30):
    """Compute dB-scaled mel + band-tinted RGB for one channel.

    Single mel spectrogram (no redundant STFT), hop=2048 for ~4x fewer frames.
    """
    S = _melspectrogram(y_ch, sr, n_mels=n_mels, fmax=16000, hop_length=2048)

    # dB-scale with gamma for visual contrast
    S_db = _power_to_db(S)
    np.clip(S_db, -60, 0, out=S_db)
    S_db += 60.0
    S_db /= 60.0
    np.power(S_db, 0.6, out=S_db)

    # Band colors from mel bin frequencies (no separate STFT needed)
    mel_f = _mel_frequencies(n_mels, fmax=16000)
    n_frames = S.shape[1]
    # Compute per-band normalized energy, then mix into RGB
    band_norms = np.empty((3, n_frames), dtype=np.float32)
    for i, (flo, fhi, _) in enumerate(SPEC_BANDS):
        mask = (mel_f >= flo) & (mel_f < fhi)
        if mask.any():
            power = np.sum(S[mask], axis=0)
            db = 10.0 * np.log10(power + 1e-10)
            np.clip(db, -20, None, out=db)
            db -= -20
            mx = db.max()
            if mx > 0:
                db /= mx
            band_norms[i] = db
        else:
            band_norms[i] = 0.0

    # (n_frames, 3) = (n_frames, 3_bands) @ (3_bands, 3_rgb)
    rgb = band_norms.T @ _BAND_COLORS
    for c in range(3):
        mx = rgb[:, c].max()
        if mx > 0:
            rgb[:, c] /= mx

    return S_db, rgb


def render_to_jpg(y, sr, jpg_path):
    """Render a 300x60 3-band tinted stereo mel spectrogram from an array.

    `y` is (channels, samples) or (samples,). int16 (what the trainer holds) is
    scaled to float; float input is used as-is. Mono is duplicated to stereo.
    Raises on failure so the caller decides whether that is fatal.
    """
    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.floating):
        y = y.astype(np.float32) / 32768.0

    # Force stereo. _load_audio returns 1D for mono-no-resample but 2D
    # (1, N) for mono-after-resample (the interpolate branch unsqueezes
    # mono inputs and never re-squeezes). Handle both.
    if y.ndim == 1:
        y = np.stack([y, y])
    elif y.shape[0] == 1:
        y = np.repeat(y, 2, axis=0)

    S_L, rgb_L = _mel_channel(y[0], sr)
    S_R, rgb_R = _mel_channel(y[1], sr)

    nf = min(S_L.shape[1], S_R.shape[1])
    S_L, S_R = S_L[:, :nf], S_R[:, :nf]
    rgb_L, rgb_R = rgb_L[:nf], rgb_R[:nf]
    nm = S_L.shape[0]

    S_L = S_L[::-1]  # L: flip so bass at bottom

    # Vectorized compositing — no Python loop over frames
    img = np.empty((nm * 2, nf, 3), dtype=np.float32)
    img[:nm] = S_L[:, :, np.newaxis] * rgb_L[np.newaxis, :, :]
    img[nm:] = S_R[:, :, np.newaxis] * rgb_R[np.newaxis, :, :]

    np.clip(img, 0, 1, out=img)
    img *= 255
    Image.fromarray(img.astype(np.uint8)).resize(
        (SPEC_W, SPEC_H), Image.LANCZOS).save(
        str(jpg_path), quality=60, optimize=True)
