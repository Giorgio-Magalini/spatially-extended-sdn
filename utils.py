import torch
import random
import numpy as np
import torchaudio.functional as F
import pyroomacoustics as pra
from scipy.io import wavfile


def seed_everything(seed):
    """Set the random seed across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_homula_rir(rir_path, ula_index: int, sr: int, trim: bool=False):
    """
    Load and preprocess a single Room Impulse Response (RIR) from HOMULA-RIR.
    https://doi.org/10.1109/ICASSPW62465.2024.10626753

    This function reads a multichannel RIR WAV file recorded with a Uniform Linear Array (ULA),
    selects a specific microphone channel, converts it to a tensor, resamples it to the target
    sampling rate, crops it to the effective reverberation time, and normalizes it to unit norm.
    """
    # Read WAV file
    orig_sr, ula_rir = wavfile.read(rir_path)

    # Select the target microphone from the ULA
    rir = ula_rir[:, ula_index]

    # Convert to tensor
    rir = torch.tensor(rir[None, :])

    # Resample RIR
    rir = F.resample(rir, orig_sr, sr)

    # Crop RIR to the reverberation time
    if trim:
        reverb_time = pra.experimental.rt60.measure_rt60(rir[0], fs=sr, decay_db=30)
        rir = rir[:, :int(reverb_time * sr)]

    # Normalize RIR to unit norm
    rir /= rir.norm()

    return rir

def load_homula_rirs(rirs_paths, sr: int, trim: bool = False, flip: bool = False):
    """
    Load and preprocess Room Impulse Responses (RIRs) from HOMULA-RIR
    for multiple sources.

    Parameters:
    rirs_paths : list or tuple of str
        Paths to multichannel RIR WAV files, one per source.
    sr : int
        Target sampling rate.
    trim : bool
        If True, crop each RIR to its estimated reverberation time.

    Returns:
    torch.Tensor
        Tensor of shape (num_sources, num_channels, num_samples)
    """

    if not isinstance(rirs_paths, (list, tuple)):
        raise ValueError("rirs_paths must be a list or tuple of file paths")

    processed_rirs = []

    for rir_path in rirs_paths:

        # --- Read WAV file ---
        orig_sr, ula_rir = wavfile.read(rir_path)

        # Convert to float tensor (channels, samples)
        rirs = torch.tensor(ula_rir.T)

        if flip:
            # Reorder channels to match CSV mic ordering (reverse X axis)
            rirs = torch.flip(rirs, dims=[0])

        # Resample to target sampling rate
        rirs = F.resample(rirs, orig_sr, sr)

        # Optional trimming to estimated reverberation time
        if trim:
            reverb_time = pra.experimental.rt60.measure_rt60(
                rirs[0], fs=sr, decay_db=30
            )
            max_samples = int(reverb_time * sr)
            rirs = rirs[:, :max_samples]

        processed_rirs.append(rirs)

    # Ensure equal length across sources (truncate to the shortest if needed)
    min_length = min(r.shape[1] for r in processed_rirs)
    processed_rirs = [r[:, :min_length] for r in processed_rirs]

    # Stack into (num_sources, num_channels, num_samples)
    stacked_rirs = torch.stack(processed_rirs, dim=0)

    return stacked_rirs


def load_rirs(rirs_paths, sr: int, trim_factor: float, flip: bool = False):
    """
    Load and preprocess multichannel RIRs (one WAV per source), then crop them to a
    fraction of their length.

    Pipeline:
        1. Read, optionally flip channel order, and resample each source to `sr`.
        2. Equalize all sources to a common length (truncate to the shortest), so every
           returned RIR has the same number of samples.
        3. Apply a fractional trim AFTER equalization: keep the first
           `trim_factor` portion of that common length.

    Parameters
    ----------
    rirs_paths  : list or tuple of str
        Paths to multichannel RIR WAV files, one per source.
    sr          : int
        Target sampling rate.
    trim_factor : float
        Fraction in (0, 1] of the length-equalized RIR to keep.
    flip        : bool
        If True, reverse the channel order to match the CSV mic ordering.

    Returns
    -------
    torch.Tensor of shape (num_sources, num_channels, num_samples)
    """
    if not isinstance(rirs_paths, (list, tuple)):
        raise ValueError("rirs_paths must be a list or tuple of file paths")
    if not (0.0 < trim_factor <= 1.0):
        raise ValueError(f"trim_factor must be in (0, 1], got {trim_factor}")

    processed_rirs = []
    for rir_path in rirs_paths:
        orig_sr, ula_rir = wavfile.read(rir_path)
        rirs = torch.tensor(ula_rir.T)
        if flip:
            rirs = torch.flip(rirs, dims=[0])
        rirs = F.resample(rirs, orig_sr, sr)
        processed_rirs.append(rirs)

    # Equalize all sources to the shortest length (same duration for all)
    min_length = min(r.shape[1] for r in processed_rirs)
    processed_rirs = [r[:, :min_length] for r in processed_rirs]

    # Fractional trim applied AFTER equalization
    keep = max(1, int(round(trim_factor * min_length)))
    processed_rirs = [r[:, :keep] for r in processed_rirs]

    return torch.stack(processed_rirs, dim=0)


def load_positions(csv_path):
    """
    Load microphone positions for a ULA from a CSV file.

    Returns:
        torch.Tensor of shape (N, 3)
    """
    mic_positions = []

    with open(csv_path, 'r') as f:
        f.readline()  # skip header
        for line in f:
            if not line.strip():
                continue
            x, y, z = map(float, line.strip().split(','))
            mic_positions.append([x, y, z])

    return torch.tensor(mic_positions, dtype=torch.float32)