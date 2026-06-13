import torch
import torch.nn as nn
from torch import Tensor
from typing import Literal


class CayleyTransform(nn.Module):
    """
    Batched Cayley transform: K skew-symmetric parameters → (mat_size × mat_size) orthogonal matrix.

    Assembles a skew-symmetric S from the upper-triangle parameters, then computes
    Q = (I - S)(I + S)^{-1}.

    Since S is skew-symmetric, (I-S) and (I+S) commute, so Q = solve(I+S, I-S).
    K = mat_size * (mat_size - 1) // 2
    """

    def __init__(self, mat_size: int, device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.mat_size = mat_size
        rows, cols = torch.triu_indices(mat_size, mat_size, offset=1)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)
        self.register_buffer("eye", torch.eye(mat_size, **factory_kwargs))

    def forward(self, params: Tensor) -> Tensor:
        """
        Args:
            params : (B, K)
        Returns:
            Q      : (B, mat_size, mat_size)  — orthogonal matrices
        """
        B, N = params.shape[0], self.mat_size
        S = torch.zeros(B, N, N, dtype=params.dtype, device=params.device)
        S[:, self.rows, self.cols] = params
        S = S - S.mT                                    # skew-symmetric
        I = self.eye.expand(B, -1, -1)
        return torch.linalg.solve(I + S, I - S)         # (I+S)^{-1}(I-S) == (I-S)(I+S)^{-1}


GeomMode = Literal['mic_pos', 'dist_nodes_mic', 'dist_full']


class CayleyScatteringJunctions(nn.Module):
    """
    MLP + Cayley transform: dynamic replacement for the static HouseholderScatteringJunction list.

    Maps batched geometric features to n_junctions orthogonal scattering matrices,
    then applies R_j^T Q_j R_j to the global incident-wave state and sums over junctions.

    K = mat_size * (mat_size - 1) // 2  (independent Cayley parameters per matrix)
    MLP output dim = n_junctions * K

    Geometric input modes (geom_mode):
        'mic_pos'        (A) — mic coordinates,       input_dim = 3
                              geom_input = mic_pos                          # (B, 3)

        'dist_nodes_mic' (B) — node-to-mic distances, input_dim = N
                              geom_input = dist_nodes_mic                   # (B, N)

        'dist_full'      (C) — all distances,          input_dim = 2*N + 1
                              geom_input = cat([dist_src_nodes,
                                                dist_nodes_mic,
                                                dist_src_mic], dim=-1)     # (B, 2N+1)

    Efficient usage in sdn.py (MLP runs once per forward, not once per timestep):

        # __init__:
        self.junctions = CayleyScatteringJunctions(
            mat_size    = self.Nm1,
            n_junctions = self.N,
            hidden_dims = [64, 128, 64],
            geom_mode   = 'dist_full',
            **self.factory_kwargs,
        )

        # forward, BEFORE the loop — MLP runs once:
        geom_input = torch.cat([dist_src_nodes, dist_nodes_mic, dist_src_mic], dim=-1)
        Q = self.junctions.get_matrices(geom_input)   # (B, J, M, M)

        # INSIDE the loop — only matrix multiplications:
        pm = self.junctions(pp, Q)
    """

    def __init__(self,
                 mat_size: int,
                 n_junctions: int,
                 hidden_dims: list[int],
                 geom_mode: GeomMode = 'dist_full',
                 device=None,
                 dtype=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}

        self.mat_size    = mat_size
        self.n_junctions = n_junctions
        self.K           = mat_size * (mat_size - 1) // 2
        self.geom_mode   = geom_mode
        n_lines          = n_junctions * mat_size

        if geom_mode == 'mic_pos':
            input_dim = 3
        elif geom_mode == 'dist_nodes_mic':
            input_dim = n_junctions
        elif geom_mode == 'dist_full':
            input_dim = 2 * n_junctions + 1
        else:
            raise ValueError(f"geom_mode must be 'mic_pos', 'dist_nodes_mic' or 'dist_full', got {geom_mode!r}")

        # MLP: (B, input_dim) → (B, n_junctions * K)
        dims   = [input_dim] + hidden_dims + [n_junctions * self.K]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1], **factory_kwargs), nn.Tanh()]
        layers.append(nn.Linear(dims[-2], dims[-1], **factory_kwargs))
        self.mlp = nn.Sequential(*layers)

        # Shared Cayley block (applied in one fused batch over B * n_junctions items)
        self.cayley = CayleyTransform(mat_size, **factory_kwargs)

        # Selection matrices R_j ∈ ℝ^{mat_size × n_lines}, one per junction — non-trainable.
        # torch.roll without dims flattens first; this shifts the unit diagonal by j*mat_size
        # columns, extracting the j-th local block from the global state.
        Rs = [torch.roll(torch.eye(mat_size, n_lines), j * mat_size)
              for j in range(n_junctions)]
        self.register_buffer("R", torch.stack(Rs).to(**factory_kwargs))  # (J, M, n_lines)

    def get_matrices(self, geom_input: Tensor) -> Tensor:
        """
        Run the MLP once and return all orthogonal scattering matrices.
        Call this BEFORE the simulation loop; pass the result to forward().

        Args:
            geom_input : (B, input_dim)
        Returns:
            Q          : (B, n_junctions, mat_size, mat_size)
        """
        B, J, M = geom_input.shape[0], self.n_junctions, self.mat_size
        params = self.mlp(geom_input).view(B * J, self.K)   # (B*J, K)
        return self.cayley(params).view(B, J, M, M)         # (B, J, M, M)

    def forward(self, pp: Tensor, Q: Tensor) -> Tensor:
        """
        Apply precomputed scattering matrices to the incident-wave state.
        Call this INSIDE the simulation loop.

        Args:
            pp : (B, n_lines, 1)             — global incident-wave state
            Q  : (B, n_junctions, M, M)      — precomputed orthogonal matrices from get_matrices()
        Returns:
            pm : (B, n_lines, 1)             — global scattered-wave state
        """
        # Local projection: R_j @ pp → (B, J, M, 1)
        local = self.R.unsqueeze(0) @ pp.unsqueeze(1)       # (1,J,M,L) @ (B,1,L,1) → (B,J,M,1)

        # Scatter: sum_j R_j^T Q_j R_j pp → (B, n_lines, 1)
        return (self.R.mT @ (Q @ local)).sum(dim=1)
