import yaml
import torch
import pickle
import argparse
from pathlib import Path

import numpy
from tqdm import trange, tqdm as tqdm_
import math

from utils import seed_everything, load_homula_rirs, load_positions
from sdn import SDN
import losses
from calibration import load_and_calibration_pipeline


SPLIT_MODE_CHOICES = [
    'even', 'first_half', 'second_half',
    'center_line_x', 'center_line_y', 'concentric_square',
    'checkerboard', 'border_train',
]

def main(args):
    # Set seeds for reproducibility
    seed_everything(42)

    # Load room configuration
    with open(args.room) as stream:
        try:
            room = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Load training configuration
    with open(args.config) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Validate split_mode from config
    # Valid choices: 'even' (alternating mics) | 'first_half' (sequential split)
    split_mode = config.get('training', {}).get('split_mode', 'even') # not present -> defaults to 'even'
    if split_mode not in SPLIT_MODE_CHOICES:
        raise ValueError(
            f"Invalid split_mode '{split_mode}' in config under 'training'. "
            f"Must be one of: {SPLIT_MODE_CHOICES}."
        )

    # Create save directory
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Settings
    dtype = torch.float32
    device = args.device
    sr = config['sr']
    c = config['c']
    n_epochs = config['training']['n_epochs']
    batch_size = config['training']['batch_size']
    accumulation_factor = config['training']['accumulation_factor']
    dataset_type = config.get('dataset_type', 'real')
    val_accumulation_factor = config['training']['val_accumulation_factor']

    # Factory kwargs used to move tensors/models to device and dtype consistently
    factory_kwargs = {"device": device, "dtype": dtype}

    if dataset_type == 'simulated':
        rirs = load_homula_rirs([room['rir_path']], sr=sr)  # (1, n_mics, T)
        true_rirs = rirs[0]
        mic_positions = load_positions(room['mic_pos_path'])
        src_pos = torch.tensor(room['src_pos'])

    elif dataset_type == 'real':
        true_rirs_both, src_positions, mic_positions = load_and_calibration_pipeline(
            rirs_path=[room['rir']['path_s1'], room['rir']['path_s2']],
            mic_pos_path=room['mic_pos_path'],
            src_pos_path=room['src_pos_path'],
            c=c,
            sr=sr,
            flip=True
        )
        source_index = config['source_index']
        src_pos = src_positions[source_index]
        true_rirs = true_rirs_both[source_index]

    else:
        raise ValueError(f"Unknown dataset_type '{dataset_type}'. Must be 'real' or 'simulated'.")

    true_rirs = true_rirs.to(**factory_kwargs)
    mic_positions = mic_positions.to(**factory_kwargs)
    src_pos = src_pos.to(**factory_kwargs)

    # Build train/validation microphone index split
    n_mics = mic_positions.shape[0]
    if n_mics % 2 != 0:
        raise ValueError("Number of microphones must be even for this split.")

    train_indices = numpy.arange(n_mics)  # default fallback
    if split_mode in ('center_line_x', 'center_line_y', 'concentric_square',
                      'checkerboard', 'border_train'):
        if dataset_type != 'simulated':
            raise ValueError(f"split_mode '{split_mode}' requires dataset_type 'simulated'.")

        n_rows, n_cols = room['mic_array_grid']  # e.g. [20, 20]

        if split_mode == 'center_line_x':
            # Horizontal line at center row: fixed y, all x
            center_row = n_rows // 2
            train_indices = torch.tensor(
                [center_row * n_cols + col for col in range(n_cols)], device=device
            )

        elif split_mode == 'center_line_y':
            # Vertical line at center column: fixed x, all y
            center_col = n_cols // 2
            train_indices = torch.tensor(
                [r * n_cols + center_col for r in range(n_rows)], device=device
            )

        elif split_mode == 'concentric_square':
            # Inner square of side train_square_size, centered on the array
            sq = config['training']['train_square_size']
            row_start = (n_rows - sq) // 2
            col_start = (n_cols - sq) // 2
            train_indices = torch.tensor(
                [(row_start + dr) * n_cols + (col_start + dc)
                 for dr in range(sq) for dc in range(sq)],
                device=device
            )

        elif split_mode == 'checkerboard':
            # (r + c) % 2 == 0 → train, else → val (perfect 50/50 interleave)
            train_indices = torch.tensor(
                [r * n_cols + c
                 for r in range(n_rows) for c in range(n_cols)
                 if (r + c) % 2 == 0],
                device=device
            )

        elif split_mode == 'border_train':
            # Outer border of given thickness → train, inner square → val
            t = config['training'].get('border_thickness', 2)
            train_indices = torch.tensor(
                [r * n_cols + c
                 for r in range(n_rows) for c in range(n_cols)
                 if r < t or r >= n_rows - t or c < t or c >= n_cols - t],
                device=device
            )

        train_mask = torch.zeros(n_mics, dtype=torch.bool, device=device)
        train_mask[train_indices] = True
        val_indices = torch.arange(n_mics, device=device)[~train_mask]

    elif split_mode == 'even':
        train_indices = torch.arange(0, n_mics, 2, device=device)
        val_indices = torch.arange(1, n_mics, 2, device=device)
    elif split_mode == 'first_half':
        half = n_mics // 2
        train_indices = torch.arange(0, half, device=device)
        val_indices = torch.arange(half, n_mics, device=device)
    elif split_mode == 'second_half':
        half = n_mics // 2
        train_indices = torch.arange(half, n_mics, device=device)
        val_indices = torch.arange(0, half, device=device)
    else:
        raise ValueError("Invalid split_mode.")

    # Single-mic baseline: restrict training to one microphone drawn from train_indices.
    # val_indices is unaffected — evaluation is identical to the full-dataset run.
    single_mic_baseline = config['training'].get('single_mic_baseline', False)
    if single_mic_baseline:
        local_idx = config['training'].get('single_mic_local_idx', 0)
        train_indices = train_indices[local_idx: local_idx + 1]

    val_batch_size = math.ceil(len(val_indices) / val_accumulation_factor)

    # Instantiate the input unit pulse
    x = torch.zeros(true_rirs.shape[-1]).to(**factory_kwargs)
    x[0] = 1.

    # Instantiate the SDN model
    sdn_cfg = config['sdn']
    sdn = SDN(room_dim=room['room_dim'],
              N=sdn_cfg['N'],
              sr=sr,
              c=c,
              junction_type=sdn_cfg['junction_type'],
              use_mlp=sdn_cfg.get('use_mlp', False),
              geom_mode=sdn_cfg.get('geom_mode', 'dist_full'),
              junction_hidden_dims=sdn_cfg.get('junction_hidden_dims', None),
              fir_order=sdn_cfg['fir_order'],
              alpha=sdn_cfg['alpha'],
              **factory_kwargs)

    # Define (weighted) losses
    sdn_loss_functions = [  # (loss_name, loss_fn, lambda),
        ('EDC', losses.EDCLoss(), config['training']['lambda_edc']),
        ('EDR', losses.MelEDRLogLoss(sr=sr), config['training']['lambda_edr']),
        ('EDP', losses.EDPLoss(sr=sr), config['training']['lambda_edp']),
    ]

    # Define the optimizer
    optimizer = torch.optim.Adam(sdn.parameters(), lr=config['training']['learning_rate'])

    # Instantiate the data structure for storing loss values
    loss_history = {k[0]: [] for k in sdn_loss_functions}

    # Start training
    for epoch in trange(n_epochs, desc="Epochs", leave=False):
        # print(f"Epoch {epoch + 1}/{n_epochs}")

        sdn.train()

        # Shuffle microphone indices for this epoch to improve spatial generalization
        perm = train_indices[torch.randperm(len(train_indices), device=device)]

        # Accumulate loss terms over the entire epoch (one entry per loss type)
        epoch_loss_terms = {k[0]: 0.0 for k in sdn_loss_functions}
        n_loss_terms = 0  # Number of individual microphone evaluations

        # Reset gradients
        optimizer.zero_grad()
        # Loop over microphone batches
        for step, i in enumerate(pbar := trange(
            0, len(train_indices), batch_size,
            desc=f"Epoch {epoch + 1} batches", leave=False
            )):

            idx = perm[i:i + batch_size]
            mic_batch = mic_positions[idx]  # (B, 3)

            # Forward pass over the entire batch
            pbar.set_postfix(status="forward pass")
            pred_rirs = sdn(x, src_pos, mic_batch)  # (B, T)

            # Expand true_rir to match batch dimension
            true_rirs_batch = true_rirs[idx]  # (B, T)

            pbar.set_postfix(status="computing losses")
            batch_loss = 0.0
            for (loss_name, loss_fn, lmbda) in sdn_loss_functions:
                loss_term = loss_fn(pred_rirs, true_rirs_batch)
                epoch_loss_terms[loss_name] += loss_term.item()
                batch_loss += lmbda * loss_term

            n_loss_terms += 1

            pbar.set_postfix(status="backpropagating")
            # Backpropagate and update
            batch_loss.backward()

            pbar.set_postfix(status="accumulating")

            is_last_batch = (i + batch_size) >= len(train_indices)
            if (step + 1) % accumulation_factor == 0 or is_last_batch:
                optimizer.step()
                optimizer.zero_grad()

        # Validation phase (no gradient tracking)
        sdn.eval()
        with torch.no_grad():
            val_loss_terms = {k[0]: 0.0 for k in sdn_loss_functions}
            n_val_batches = 0

            for j, val_j in enumerate(trange(
                0, len(val_indices), val_batch_size,
                desc=f"Epoch {epoch + 1} validation", leave=False
            )):
                val_idx = val_indices[val_j:val_j + val_batch_size]
                pred_rirs_val = sdn(x, src_pos, mic_positions[val_idx])
                true_rir_val = true_rirs[val_idx]

                for (loss_name, loss_fn, _) in sdn_loss_functions:
                    val_loss_terms[loss_name] += loss_fn(pred_rirs_val, true_rir_val).item()
                n_val_batches += 1

            val_loss_terms = {k: v / n_val_batches for k, v in val_loss_terms.items()}
            val_str = "  ".join(f"{k}: {v:.4e}" for k, v in val_loss_terms.items())
            tqdm_.write(f"[Epoch {epoch + 1:>4}]  {val_str}")

        # Store epoch-averaged loss values
        for k in epoch_loss_terms:
            loss_history[k].append(epoch_loss_terms[k] / n_loss_terms)

        # Save model checkpoint at the end of the epoch
        torch.save(sdn.state_dict(), save_dir / f"sdn_epoch_{epoch}.pth")

        # Save loss history to disk for later analysis
        with open(save_dir / "loss_history.pickle", "wb") as f:
            pickle.dump(loss_history, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save model checkpoint for this epoch
    torch.save(sdn, save_dir.joinpath(f'sdn_epoch_{n_epochs}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-r', '--room', default='config/rooms/schiavoni_room.yaml')
    parser.add_argument('-d', '--device', default='cuda:0')
    args = parser.parse_args()
    main(args)
