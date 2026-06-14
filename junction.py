import torch
import torch.nn.functional as F
from torch import Tensor, nn


class HouseholderScatteringJunction(nn.Module):
    def __init__(self,
                 N: int,  # Scattering matrix size
                 index: int,  # junction progressive number, 0 to 5
                 device=None,
                 dtype=None,
                 trainable: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.N = N
        self.Nm1 = N - 1
        # Instantiate selection (routing) matrix
        self.R = torch.roll(torch.eye(self.Nm1, self.N * self.Nm1), index * self.Nm1).to(**factory_kwargs)
        # Instantiate an (N-1) identity matrix
        self.eye = torch.eye(self.Nm1).to(**factory_kwargs)

        if trainable:
            # Trainable admittance vector
            self.weight = nn.Parameter(
                torch.empty((1, self.Nm1), **factory_kwargs)
            )
        else:
            # Constant admittance vector -> Standard Householder matrix
            self.register_buffer('weight', torch.ones((self.Nm1, 1), **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    @property
    def S(self):
        # Ensure weights (admittances) are strictly positive
        weight = self.weight.abs() + 1e-12
        # Eq. (9) from the paper
        return (2 /weight.sum()) * weight.repeat(self.Nm1, 1) - self.eye

    def forward(self, x: Tensor) -> Tensor:
        # From right to left: global-to-local => scattering => local-to-global
        return  self.R.t() @ self.S @ self.R @ x


class HouseholderScatteringJunctions(nn.Module):
    """
    MLP + Householder: dynamic replacement for the static HouseholderScatteringJunction list.

    Maps batched geometric features to n_junctions Householder reflection matrices,
    then applies sum_j R_j^T A_j R_j to the global incident-wave state.

    MLP output dim = n_junctions * mat_size  (one unit vector v per junction)
    A_j = 2 * v_j * v_j^T - I

    Geometric input modes mirror CayleyScatteringJunctions (geom_mode).
    """

    def __init__(self,
                 mat_size: int,
                 n_junctions: int,
                 hidden_dims: list[int],
                 geom_mode: str = 'dist_full',
                 device=None,
                 dtype=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}

        self.mat_size    = mat_size
        self.n_junctions = n_junctions
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

        # MLP: (B, input_dim) → (B, n_junctions * mat_size)
        dims   = [input_dim] + hidden_dims + [n_junctions * mat_size]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1], **factory_kwargs), nn.Tanh()]
        layers.append(nn.Linear(dims[-2], dims[-1], **factory_kwargs))
        self.mlp = nn.Sequential(*layers)

        # Selection matrices R_j — same construction as CayleyScatteringJunctions
        Rs = [torch.roll(torch.eye(mat_size, n_lines), j * mat_size)
              for j in range(n_junctions)]
        self.register_buffer("R", torch.stack(Rs).to(**factory_kwargs))  # (J, M, n_lines)
        self.register_buffer("eye", torch.eye(mat_size, **factory_kwargs))

    def get_matrices(self, geom_input: Tensor) -> Tensor:
        """
        Run the MLP once and return all Householder matrices.
        Call this BEFORE the simulation loop; pass the result to forward().

        Args:
            geom_input : (B, input_dim)
        Returns:
            Q          : (B, n_junctions, mat_size, mat_size)
        """
        B, J, M = geom_input.shape[0], self.n_junctions, self.mat_size
        v = self.mlp(geom_input).view(B, J, M)          # (B, J, M)
        v = F.normalize(v, dim=-1)                       # unit norm
        # A_j = 2 * v_j * v_j^T - I
        Q = 2 * v.unsqueeze(-1) * v.unsqueeze(-2) - self.eye.unsqueeze(0).unsqueeze(0)
        return Q                                         # (B, J, M, M)

    def forward(self, pp: Tensor, Q: Tensor) -> Tensor:
        """
        Apply precomputed Householder matrices to the incident-wave state.
        Identical interface to CayleyScatteringJunctions.forward().

        Args:
            pp : (B, n_lines, 1)
            Q  : (B, n_junctions, M, M)
        Returns:
            pm : (B, n_lines, 1)
        """
        local = self.R.unsqueeze(0) @ pp.unsqueeze(1)   # (B, J, M, 1)
        return (self.R.mT @ (Q @ local)).sum(dim=1)      # (B, n_lines, 1)
