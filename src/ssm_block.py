"""Pure-PyTorch bidirectional selective state-space block.

Implements a Mamba-style selective SSM in plain PyTorch (no custom CUDA), so it
runs on any GPU including sm_120 (RTX 5090 / Blackwell) where the upstream
mamba-ssm CUDA kernels are not yet shipped. The trade is throughput: a Python
for-loop scan over the sequence length. For short token sequences (e.g. 256
tokens from a 16x16 ResNet-18 feat map at 512x512 input) this is acceptable
overhead (a few ms per batch).

Architecturally:
    x ∈ ℝ^(B, L, D)
        --> input-dependent (Δ, B, C) ← Linear(x)
        --> discretise: Â = exp(Δ ⊗ A), B̂ = Δ ⊗ B
        --> scan forward:  h_t = Â h_{t-1} + B̂ x_t,  y_t = C h_t
        --> scan backward: same on reversed sequence
        --> y = (y_fwd + y_bwd) / 2
        --> output projection + residual + RMSNorm

This is the bidirectional pure-PyTorch analogue of Mamba's selective scan
(Gu & Dao 2023). The bidirectional choice is appropriate for fundus images
where the token order (raster scan over the 16x16 feature map) has no natural
direction — left-to-right and right-to-left are equally valid scans of the
retinal field.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        n = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * n).type_as(x) * self.weight


class SelectiveSSMScan(nn.Module):
    """Single-direction selective-SSM scan (forward only).

    Args:
        d_model: feature dimension.
        d_state: SSM hidden state dimension N (typically 16).
        dt_rank: rank of the input-dependent Δ projection (typically d_model//16).
    """
    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, d_model // 16)

        # x_proj: linear that produces (Δ_input, B, C) per token, all input-dependent
        self.x_proj = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)
        # dt_proj: project Δ_input -> Δ (d_model)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        # initialise dt_proj.bias so initial Δ is small-positive (Mamba init)
        with torch.no_grad():
            dt_init_std = self.dt_rank ** -0.5
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
            dt = torch.exp(torch.rand(d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.copy_(inv_dt)

        # learned negative-real A (per channel × per state slot)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))   # so A = -exp(A_log) ≤ 0
        self.D = nn.Parameter(torch.ones(d_model))  # skip-connection weight per channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D). Returns (B, L, D)."""
        B, L, D = x.shape
        N = self.d_state

        proj = self.x_proj(x)                                              # (B, L, dt_rank + 2N)
        dt_in, B_, C_ = torch.split(proj, [self.dt_rank, N, N], dim=-1)
        delta = F.softplus(self.dt_proj(dt_in))                            # (B, L, D), positive
        A = -torch.exp(self.A_log.float())                                 # (D, N), negative-real

        # Discretisation (broadcast over B and L)
        # Â: (B, L, D, N), B̂: (B, L, D, N)
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaB = delta.unsqueeze(-1) * B_.unsqueeze(2)                     # (B, L, D, N)
        deltaB_x = deltaB * x.unsqueeze(-1)                                # (B, L, D, N)

        # Sequential scan
        h = x.new_zeros(B, D, N)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB_x[:, t]                          # (B, D, N)
            y_t = (h * C_[:, t].unsqueeze(1)).sum(-1)                      # (B, D)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                                         # (B, L, D)

        # Skip connection (parameterised)
        return y + x * self.D.view(1, 1, -1)


class BiSelectiveSSMBlock(nn.Module):
    """A residual block that wraps a forward + reverse selective scan, an
    output projection, dropout, and an RMSNorm. Drop-in for a single Mamba-style
    layer.
    """
    def __init__(self, d_model: int, d_state: int = 16,
                 expand: int = 2, dropout: float = 0.1,
                 bidirectional: bool = True):
        super().__init__()
        self.bidirectional = bidirectional
        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, expand * d_model)
        self.scan_fwd = SelectiveSSMScan(expand * d_model, d_state=d_state)
        if bidirectional:
            self.scan_bwd = SelectiveSSMScan(expand * d_model, d_state=d_state)
        self.gate = nn.Linear(d_model, expand * d_model)
        self.out_proj = nn.Linear(expand * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D). Returns (B, L, D) with residual."""
        h = self.norm(x)
        u = self.in_proj(h)                       # (B, L, 2D)
        g = F.silu(self.gate(h))                  # (B, L, 2D)
        y_fwd = self.scan_fwd(u)
        if self.bidirectional:
            y_bwd = self.scan_bwd(torch.flip(u, dims=[1]))
            y_bwd = torch.flip(y_bwd, dims=[1])
            y = (y_fwd + y_bwd) * 0.5 * g
        else:
            y = y_fwd * g
        y = self.out_proj(y)
        return x + self.dropout(y)


class MLPBottleneckBlock(nn.Module):
    """Param-matched control: a residual MLP block with the same d_model, the
    same residual+RMSNorm structure as BiSelectiveSSMBlock --- but no selective
    scan, no input-dependent state-space dynamics, no token-to-token routing.
    Used as a control to isolate whether the *selective scan* is what matters
    versus just additional bottleneck capacity.
    """
    def __init__(self, d_model: int, expand: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.fc1 = nn.Linear(d_model, expand * d_model)
        self.fc2 = nn.Linear(expand * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        y = self.fc2(F.gelu(self.fc1(h)))
        return x + self.dropout(y)
