import torch
from torch import Tensor, nn
from curves import energy_decay_curve, echo_density_profile, mel_energy_decay_relief


def lp_error_fn(pred: Tensor, target: Tensor, power: float = 1.0,
                normalize: bool = False) -> Tensor:
    """L^p loss with optional normalization by the target norm."""
    loss = torch.mean(torch.abs(target - pred) ** power)
    if normalize:
        norm = torch.mean(torch.abs(target) ** power)
        return loss / norm
    return loss

def _to_batch(x: Tensor) -> Tensor:
    """Ensure tensor is 2-D (B, T). Adds a batch dimension if input is 1-D."""
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected 1-D or 2-D input tensor, got shape {tuple(x.shape)}.")

class EDCLoss(nn.Module):
    def __init__(self, power: float = 2.0, **kwargs) -> None:
        """Normalized MSE between linear-amplitude full-band EDC curves."""
        super().__init__(**kwargs)
        self.power = power

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred:   (T,) or (B, T)
            target: (T,) or (B, T)
        Returns:
            Scalar loss.
        """
        pred   = _to_batch(pred)    # (B, T)
        target = _to_batch(target)  # (B, T)

        edc_pred = energy_decay_curve(pred)  # (B, T) — natively batched
        edc_target = energy_decay_curve(target)  # (B, T)

        return lp_error_fn(edc_pred, edc_target, self.power, normalize=False)


class MelEDRLogLoss(nn.Module):
    def __init__(self, sr: int, power: float = 1.0, **kwargs) -> None:
        """Normalized MAE between log-amplitude mel-frequency EDRs."""
        super().__init__(**kwargs)
        self.sr    = sr
        self.power = power

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred:   (T,) or (B, T)
            target: (T,) or (B, T)
        Returns:
            Scalar loss.
        """
        pred   = _to_batch(pred)    # (B, T)
        target = _to_batch(target)  # (B, T)

        edr_pred = mel_energy_decay_relief(pred, self.sr, return_db=True)  # (B, n_mels, L)
        edr_target = mel_energy_decay_relief(target, self.sr, return_db=True)  # (B, n_mels, L)

        return lp_error_fn(edr_pred, edr_target, self.power, normalize=False)


class EDPLoss(nn.Module):
    def __init__(self, sr: int, win_duration: float = 0.02, power: float = 2.0, **kwargs) -> None:
        """MSE between SoftEDP curves. Reference: https://doi.org/10.1186/s13636-024-00371-5"""
        super().__init__(**kwargs)
        self.sr           = sr
        self.win_duration = win_duration
        self.power        = power

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred:   (T,) or (B, T)
            target: (T,) or (B, T)
        Returns:
            Scalar loss.
        """
        pred   = _to_batch(pred)    # (B, T)
        target = _to_batch(target)  # (B, T)

        profile_pred = echo_density_profile(
            pred, sr=self.sr, win_duration=self.win_duration, differentiable=True
        )  # (B, n_frames) — natively batched
        profile_target = echo_density_profile(
            target, sr=self.sr, win_duration=self.win_duration, differentiable=True
        )  # (B, n_frames)

        return lp_error_fn(profile_pred, profile_target, self.power, normalize=False)