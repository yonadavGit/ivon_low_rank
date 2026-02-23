import torch
from typing import Optional


def orthonormal_matrix(d: int, r: int, *, dtype=torch.float64, device: str = "cpu") -> torch.Tensor:
    if r == 0:
        return torch.zeros(d, 0, dtype=dtype, device=device)
    q, _ = torch.linalg.qr(torch.randn(d, r, dtype=dtype, device=device), mode="reduced")
    return q


def sample_reparam(
    mu: torch.Tensor,
    D: torch.Tensor,
    U: Optional[torch.Tensor],
    s: Optional[torch.Tensor],
    n: int,
) -> torch.Tensor:
    d = mu.numel()
    eps = torch.randn(n, d, dtype=mu.dtype, device=mu.device)
    y = eps

    if U is not None and s is not None and U.numel() > 0:
        shrink = 1.0 - (1.0 + s).rsqrt()
        y = eps - (eps @ U * shrink) @ U.t()

    return mu + y * D.rsqrt()


def implied_covariance(
    D: torch.Tensor, U: Optional[torch.Tensor], s: Optional[torch.Tensor]
) -> torch.Tensor:
    d = D.numel()
    B = torch.eye(d, dtype=D.dtype, device=D.device)
    if U is not None and s is not None and U.numel() > 0:
        shrink = 1.0 - (1.0 + s).rsqrt()
        B = B - U @ torch.diag(shrink) @ U.t()

    Dinv_sqrt = torch.diag(D.rsqrt())
    sigma = Dinv_sqrt @ B @ B.t() @ Dinv_sqrt
    return 0.5 * (sigma + sigma.t())


def empirical_cov(samples: torch.Tensor) -> torch.Tensor:
    centered = samples - samples.mean(dim=0)
    return centered.t() @ centered / samples.shape[0]


def frob(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x, ord="fro")
