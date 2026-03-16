import torch
from torch import Tensor
import torchaudio.transforms as T
from typing import Union, Tuple


def energy_decay_curve(h: Tensor, return_db: bool = False) -> Tensor:
    """
    Schroeder’s full-band EDC (https://ccrma.stanford.edu/~jos/pasp/Energy_Decay_Curve.html)

    Args:
        h: (T,) or (B, T)
    Returns:
        EDC of same shape as input.
    """
    h_pow   = h ** 2
    cum_sum = torch.cumsum(h_pow, dim=-1)                  # (..., T)
    edc     = h_pow + cum_sum[..., -1:] - cum_sum          # (..., T)
    return 10 * torch.log10(edc.clamp(min=1e-12)) if return_db else edc


def mel_energy_decay_relief(h: Tensor, sr: int, return_db: bool = True) -> Tensor:
    """
    Mel-frequency EDR.
    https://ccrma.stanford.edu/~jos/pasp/Energy_Decay_Relief.html

    Args:
        h: (T,) or (B, T)
    Returns:
        EDR of shape (n_mels, L) or (B, n_mels, L).
    """
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=512,
        win_length=320,
        hop_length=160,
        f_min=0.0,
        f_max=sr/2,
        n_mels=64,
        power=2.0,
        window_fn=torch.hann_window,
        normalized=False,
    ).to(h.device)

    spec    = mel_spectrogram(h)                           # (..., n_mels, L)
    cum_sum = torch.cumsum(spec, dim=-1)                   # (..., n_mels, L)
    edr     = spec + cum_sum[..., -1:] - cum_sum           # (..., n_mels, L)
    return T.AmplitudeToDB(top_db=80)(edr) if return_db else edr


def echo_density_profile(h: Tensor, sr: int, win_duration: float = 0.02, differentiable: bool = False,
                         kappa: Union[float, Tuple[float, float]] = (1e2, 1e5)) -> Tensor:
    """
    Echo Density Profile (EDP) — batched implementation via unfold.

    Reference:
        J. S. Abel & P. Huang, "A Simple, Robust Measure of Reverberation Echo Density,"
        121st AES Convention, 2006.

    Args:
        h:              (T,) or (B, T)
        sr:             sampling rate
        win_duration:   analysis window duration in seconds
        differentiable: use sigmoid approximation (SoftEDP) if True
        kappa:          sigmoid sharpness — scalar or (start, end) for linearly increasing schedule
    Returns:
        profile: (n_frames,) or (B, n_frames)  where n_frames = T - win_len
    """
    squeeze_output = h.ndim == 1
    if h.ndim == 1:
        h = h.unsqueeze(0)                                 # (1, T) — add batch dim
    B, T = h.shape

    # Instantiate the odd-length symmetric Hann window
    win_len = int(win_duration * sr)
    win_len = win_len + 1 if win_len % 2 == 0 else win_len
    win = torch.hann_window(win_len, periodic=False, device=h.device)
    win = win / win.sum()                                  # (win_len,) — sums to 1

    n_frames = T - win_len                                 # number of output frames

    # Materialize all frames at once: (B, T - win_len + 1, win_len)
    # Drop the last frame to match the original loop range(T - win_len)
    frames = h.unfold(-1, win_len, 1)[:, :-1, :].abs()    # (B, n_frames, win_len)

    # Per-frame RMS energy — shape (B, n_frames)
    sigma = torch.sqrt((win * frames ** 2).sum(-1))        # (B, n_frames)

    # Build sigmoid sharpness schedule
    if isinstance(kappa, tuple):
        kappa_t = torch.linspace(kappa[0], kappa[1], n_frames, device=h.device).view(1, n_frames, 1) # (1, n_frames, 1)
    else:
        kappa_t = torch.tensor(kappa, device=h.device).view(1, 1, 1)

    if differentiable:
        # SoftEDP: sigmoid approximation of the indicator function
        indicator = torch.sigmoid(kappa_t * (frames - sigma.unsqueeze(-1))) # (B, n_frames, win_len)
    else:
        indicator = (frames > sigma.unsqueeze(-1)).float() # (B, n_frames, win_len)

    # Weighted sum over window -> profile
    profile = (win * indicator).sum(-1)                    # (B, n_frames)

    # 1/erfc(1 /√2) ≈ 3.15148718753
    profile = 3.15148718753 * profile

    return profile.squeeze(0) if squeeze_output else profile