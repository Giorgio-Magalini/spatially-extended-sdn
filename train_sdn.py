import yaml
import torch
import pickle
import argparse
from pathlib import Path

from utils import seed_everything
from sdn import SDN
import losses
from calibration import load_and_calibration_pipeline


SPLIT_MODE_CHOICES = ['even', 'first_half']

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
    source_index = config['source_index']

    # Factory kwargs used to move tensors/models to device and dtype consistently
    factory_kwargs = {"device": device, "dtype": dtype}

    # Load RIRs (trimmed by estimated system delay), source position (estimated) and mic position (original)
    true_rirs_both_channels, src_positions, mic_positions = load_and_calibration_pipeline(
        rirs_path=[room['rir']['path_s1'], room['rir']['path_s2']],
        mic_pos_path=room['mic_pos_path'],
        src_pos_path=room['src_pos_path'],
        c=c,
        sr=sr
    )

    # Select source and its corresponding RIRs
    src_pos = src_positions[source_index].to(**factory_kwargs)
    true_rirs = true_rirs_both_channels[source_index].to(**factory_kwargs)
    mic_positions = mic_positions.to(**factory_kwargs)

    # Build train/validation microphone index split
    n_mics = mic_positions.shape[0]
    if n_mics % 2 != 0:
        raise ValueError("Number of microphones must be even for this split.")

    if split_mode == 'even':
        train_indices = torch.arange(0, n_mics, 2, device=device)
        val_indices = torch.arange(1, n_mics, 2, device=device)
    elif split_mode == 'first_half':
        half = n_mics // 2
        train_indices = torch.arange(0, half, device=device)
        val_indices = torch.arange(half, n_mics, device=device)
    else:
        raise ValueError("Invalid split_mode.")

    # Instantiate the input unit pulse
    x = torch.zeros(true_rirs.shape[-1]).to(**factory_kwargs)
    x[0] = 1.

    # Instantiate the SDN model
    sdn = SDN(room_dim=room['room_dim'],
              N=config['sdn']['N'],
              sr=sr,
              c=c,
              junction_type=config['sdn']['junction_type'],
              fir_order=config['sdn']['fir_order'],
              alpha=config['sdn']['alpha'],
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
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")

        sdn.train()

        # Shuffle microphone indices for this epoch to improve spatial generalization
        perm = train_indices[torch.randperm(len(train_indices), device=device)]

        # Accumulate loss terms over the entire epoch (one entry per loss type)
        epoch_loss_terms = {k[0]: 0.0 for k in sdn_loss_functions}
        n_loss_terms = 0  # Number of individual microphone evaluations

        # Reset gradients
        optimizer.zero_grad()
        # Loop over microphone batches
        for step, i in enumerate(range(0, len(train_indices), batch_size)):

            idx = perm[i:i + batch_size]
            mic_batch = mic_positions[idx]  # (B, 3)

            # Forward pass over the entire batch
            pred_rirs = sdn(x, src_pos, mic_batch)  # (B, T)

            # Normalize each prediction to unit norm
            pred_rirs = pred_rirs / (pred_rirs.norm(dim=-1, keepdim=True) + 1e-8)

            # Expand true_rir to match batch dimension
            true_rirs_batch = true_rirs[idx]  # (B, T)

            batch_loss = 0.0
            for (loss_name, loss_fn, lmbda) in sdn_loss_functions:
                loss_term = loss_fn(pred_rirs, true_rirs_batch)
                epoch_loss_terms[loss_name] += loss_term.item()
                batch_loss += lmbda * loss_term

            n_loss_terms += 1

            # Backpropagate and update
            batch_loss.backward()

            is_last_batch = (i + batch_size) >= len(train_indices)
            if (step + 1) % accumulation_factor == 0 or is_last_batch:
                optimizer.step()
                optimizer.zero_grad()

        # Validation phase (no gradient tracking)
        sdn.eval()
        with torch.no_grad():
            val_loss_terms = {k[0]: 0.0 for k in sdn_loss_functions}

            # Run validation in a single batched forward pass
            pred_rirs_val = sdn(x, src_pos, mic_positions[val_indices])  # (B_val, T)
            pred_rirs_val = pred_rirs_val / (pred_rirs_val.norm(dim=-1, keepdim=True) + 1e-8)
            true_rir_val = true_rirs[val_indices]

            for (loss_name, loss_fn, lmbda) in sdn_loss_functions:
                loss_term = loss_fn(pred_rirs_val, true_rir_val)
                val_loss_terms[loss_name] += loss_term.item()

            print("Validation losses:", val_loss_terms)

        # Store epoch-averaged loss values
        for k in epoch_loss_terms:
            loss_history[k].append(epoch_loss_terms[k] / n_loss_terms)

        # Save model checkpoint at the end of the epoch
        torch.save(sdn.state_dict(), save_dir / f"sdn_epoch_{epoch}.pth")

        # Save loss history to disk for later analysis
        with open(save_dir / "loss_history.pickle", "wb") as f:
            pickle.dump(loss_history, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Model saved.')

    # Save model checkpoint for this epoch
    torch.save(sdn, save_dir.joinpath(f'sdn_epoch_{n_epochs}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-r', '--room', default='config/rooms/schiavoni_room.yaml')
    parser.add_argument('-d', '--device', default='cuda:0')
    args = parser.parse_args()
    main(args)
