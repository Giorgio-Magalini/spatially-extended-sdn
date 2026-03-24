import math
import torch
from torch import Tensor, nn
from tqdm import trange
from junction import HouseholderScatteringJunction
from integer_delay import IntegerDelayLines
from position_utils import get_distances


class SDN(nn.Module):
    '''Differential Scattering Delay Network'''
    def __init__(self,
                 room_dim,
                 N: int = 6,
                 sr: int = 16000,
                 c: float = 343.,
                 junction_type='householder',
                 fir_order: int = 7,
                 alpha: float = 0.02,
                 device=None,
                 dtype=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.device = device
        self.dtype = dtype
        # Factory kwargs used to move tensors to device and dtype consistently
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.N = N
        self.Nm1 = N - 1
        self.n_lines = self.N * self.Nm1
        self.fir_order = fir_order
        self.G = c / sr  # Helper constant

        self.room_dim = room_dim

        # Initialize junction filters (inverse sigmoid reparameterization for the scalar case)
        init_beta = -math.log((1 / math.sqrt(1 - alpha)) - 1) if self.fir_order == 0 else math.sqrt(1 - alpha)
        init_filt = torch.zeros(self.N, self.fir_order + 1, **self.factory_kwargs)
        init_filt[:, -1] = init_beta
        self.junction_filters = nn.Parameter(init_filt)

        # Initialize junctions
        if junction_type == 'householder':
            self.junctions = nn.ModuleList(
                [HouseholderScatteringJunction(self.N, j, **self.factory_kwargs) for j in range(self.N)]
            )
        else:
            raise ValueError(f'Junction type {junction_type} not recognized.')

        # Initialize permutation matrix P
        self.permutation_matrix = torch.zeros(self.n_lines, self.n_lines, **self.factory_kwargs)
        for i in range(self.n_lines):
            f = (6 * (i + 1) - (i % self.N) - 1) % self.n_lines + 1
            self.permutation_matrix[i, f - 1] = 1

        # Initialize pressure extraction weights
        self.pressure_weights = nn.Parameter(
            torch.full((self.n_lines,), 2 / self.Nm1, **self.factory_kwargs)
        )

    def forward(self, x: Tensor, src_pos: Tensor, mic_pos: Tensor) -> Tensor:
        """
        Forward pass with dynamic microphone position.

        Args:
            x: Input signal tensor of shape (T,) or (T, 1)
            src_pos: Source position as tensor of shape (3,)
            mic_pos: Microphone position as tensor of shape (3,) or (B, 3)

        Returns:
            y: Output signal tensor of shape (B, T)
        """
        # Fix input shape
        if x.ndim == 1:
            x = x.unsqueeze(1)
        T = x.shape[0]

        if mic_pos.ndim == 1:
            mic_pos = mic_pos.unsqueeze(0)
        B = mic_pos.shape[0]

        # Convert positions to list for get_distances function
        if isinstance(src_pos, Tensor):
            src_pos_list = src_pos.cpu().tolist()
        else:
            src_pos_list = src_pos

        dist_src_nodes_list: list[list[float]] = []
        dist_nodes_list: list[list[float]] = []
        dist_nodes_mic_list: list[list[float]] = []
        dist_src_mic_list: list[list[float]] = []

        for b in range(B):
            mic_pos_b: list[float] = mic_pos[b].cpu().tolist()
            dsn, dn, dnm, dsm = get_distances(
                self.N, self.room_dim, src_pos_list, mic_pos_b
            )
            dist_src_nodes_list.append(dsn)
            dist_nodes_list.append(dn)
            dist_nodes_mic_list.append(dnm)
            dist_src_mic_list.append(dsm)

        # Convert to tensors — all carry a leading batch dimension
        dist_src_nodes = torch.tensor(dist_src_nodes_list, **self.factory_kwargs)  # (B, N)
        dist_nodes = torch.tensor(dist_nodes_list, **self.factory_kwargs)  # (B, N*(N-1))
        dist_nodes_mic = torch.tensor(dist_nodes_mic_list, **self.factory_kwargs)  # (B, N)
        dist_src_mic = torch.tensor(dist_src_mic_list, **self.factory_kwargs)  # (B, 1)

        # Compute the attenuation coefficients from distances (spherical spreading law)
        source_gains = 0.5 * (self.G / dist_src_nodes).unsqueeze(-1)           # (N,)
        mic_gains = 1 / (1 + dist_nodes_mic / dist_src_nodes).unsqueeze(-1)    # (B, N)
        direct_gain  = 1 / dist_src_mic                                        # (B, 1)

        # Compute delays (in samples) from distances - use integer delays
        delay_src_nodes = torch.round(dist_src_nodes / self.G).long()  # (N,)
        delay_nodes     = torch.round(dist_nodes     / self.G).long()  # (n_lines,)
        delay_nodes_mic = torch.round(dist_nodes_mic / self.G).long()  # (B, N)
        delay_src_mic   = torch.round(dist_src_mic   / self.G).long()  # (B, 1)

        # Compute per-object maximum delays (+ 1 to account for zero-based indexing)
        max_delay_src_mic = int(delay_src_mic.max().item()) + 1
        max_delay_src_nodes = int(delay_src_nodes.max().item()) + 1
        max_delay_nodes = int(delay_nodes.max().item()) + 1
        max_delay_nodes_mic = int(delay_nodes_mic.max().item()) + 1

        # Instantiate batch-aware integer delay lines with tight buffer lengths
        src_to_mic = IntegerDelayLines(1, max_delay_src_mic, batch_size=B, **self.factory_kwargs)
        src_to_nodes = IntegerDelayLines(self.N, max_delay_src_nodes, batch_size=B, **self.factory_kwargs)
        nodes_to_nodes = IntegerDelayLines(self.n_lines, max_delay_nodes, batch_size=B, **self.factory_kwargs)
        nodes_to_mic = IntegerDelayLines(self.N, max_delay_nodes_mic, batch_size=B, **self.factory_kwargs)

        # Instantiate absorption/reflection filters (sigmoid reparameterization for the scalar case)
        junction_filters = torch.sigmoid(self.junction_filters) if self.fir_order == 0 else self.junction_filters
        junction_filters_nodes = junction_filters.repeat_interleave(self.Nm1, dim=0)

        # ========= Main simulation loop =========
        y = torch.zeros(B, T, **self.factory_kwargs)
        pp = torch.zeros(B, self.n_lines, 1, **self.factory_kwargs)
        for k in trange(len(x)):
            xk = x[k].view(1, 1, 1).expand(B, 1, 1)  # (B, 1, 1) — batch-aware scalar

            # Add the (delayed) source signal to the global incident waves
            pp += src_to_nodes(source_gains * xk, delay_src_nodes).repeat_interleave(self.Nm1, dim=1)

            # Compute the global reflected (outgoing) waves after local scattering
            pm = sum([junction(pp) for junction in self.junctions])

            # Compute the microphone signal (contribution from the nodes)
            y[:, k] = nodes_to_mic(
                mic_gains * (self.pressure_weights * pm).reshape(B, self.N, -1).sum(2, keepdim=True),
                delay_nodes_mic, junction_filters
            ).squeeze(-1).sum(dim=1)

            # Add the (delayed) source signal to the microphone signal
            y[:, k] += (direct_gain * src_to_mic(xk, delay_src_mic).squeeze(-1)).squeeze(-1)

            # Compute the next global incident waves by delaying and permuting the global reflected waves
            pp = self.permutation_matrix @ nodes_to_nodes(pm, delay_nodes, junction_filters_nodes)

        return y


