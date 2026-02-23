import torch
import math

def orthonormal_U(d, r, device="cpu", dtype=torch.float64):
    A = torch.randn(d, r, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(A, mode="reduced")
    return Q

def sampler_reparam(mu, D, U, s, n):
    """
    mu: (d,)
    D:  (d,) positive (precision-like), used as D^{-1/2}
    U:  (d,r) orthonormal
    s:  (r,) nonnegative
    """
    d = mu.numel()
    eps = torch.randn(n, d, device=mu.device, dtype=mu.dtype)

    Dinv_sqrt = D.rsqrt()  # (d,)

    if U.numel() > 0:
        shrink = 1.0 - (1.0 + s).rsqrt()          # (r,)
        # Apply y = eps - U @ (shrink * (U^T eps))
        Ut_eps = eps @ U                           # (n,r)
        y = eps - (Ut_eps * shrink) @ U.t()        # (n,d)
    else:
        y = eps

    noise = y * Dinv_sqrt                          # (n,d), elementwise
    return mu + noise

def implied_cov(D, U, s):
    """Return Sigma = D^{-1/2} B B^T D^{-1/2} as a (d,d) matrix."""
    d = D.numel()
    Dinv_sqrt = torch.diag(D.rsqrt())              # (d,d)

    if U.numel() > 0:
        shrink = 1.0 - (1.0 + s).rsqrt()           # (r,)
        B = torch.eye(d, device=D.device, dtype=D.dtype) - U @ torch.diag(shrink) @ U.t()
    else:
        B = torch.eye(d, device=D.device, dtype=D.dtype)

    Sigma = Dinv_sqrt @ (B @ B.t()) @ Dinv_sqrt
    # Symmetrize to kill numerical asymmetry:
    Sigma = 0.5 * (Sigma + Sigma.t())
    return Sigma

def sampler_cholesky(mu, Sigma, n):
    """Traditional MVN sampling: mu + L z, z~N(0,I), with Sigma = L L^T."""
    d = mu.numel()
    # Add tiny jitter for numerical stability (should be unnecessary if PSD well-conditioned)
    jitter = 1e-12 * torch.eye(d, device=Sigma.device, dtype=Sigma.dtype)
    L = torch.linalg.cholesky(Sigma + jitter)
    z = torch.randn(n, d, device=mu.device, dtype=mu.dtype)
    return mu + z @ L.t()

def emp_mean_cov(X):
    """X: (n,d). Returns mean (d,) and covariance (d,d) (unbiased=False)."""
    n = X.shape[0]
    m = X.mean(dim=0)
    Xc = X - m
    C = (Xc.t() @ Xc) / n
    return m, C

def ks_1d(x, y):
    """
    Two-sample KS statistic (no p-value), x,y are 1D tensors.
    """
    x = torch.sort(x).values
    y = torch.sort(y).values
    n = x.numel()
    m = y.numel()
    # Merge grid
    grid = torch.sort(torch.cat([x, y])).values
    # Empirical CDFs on grid
    # searchsorted returns counts <= each grid point
    Fx = torch.searchsorted(x, grid, right=True).to(torch.float64) / n
    Fy = torch.searchsorted(y, grid, right=True).to(torch.float64) / m
    return torch.max(torch.abs(Fx - Fy)).item()

def run_test(seed=0, d=50, r=8, n=200_000, device="cpu"):
    torch.manual_seed(seed)
    dtype = torch.float64

    mu = torch.randn(d, device=device, dtype=dtype)

    # Make a positive diagonal D (your "precision"): bigger D => smaller variance
    # (You can model D = ess*(h+wd); here just random-ish but positive)
    D = torch.exp(torch.randn(d, device=device, dtype=dtype)) + 0.5

    U = orthonormal_U(d, r, device=device, dtype=dtype)
    s = torch.exp(torch.randn(r, device=device, dtype=dtype))  # positive

    Sigma = implied_cov(D, U, s)

    X1 = sampler_reparam(mu, D, U, s, n)
    X2 = sampler_cholesky(mu, Sigma, n)

    m1, C1 = emp_mean_cov(X1)
    m2, C2 = emp_mean_cov(X2)

    # Mean errors
    mean_err_12 = torch.norm(m1 - m2).item()
    mean_err_1t = torch.norm(m1 - mu).item()
    mean_err_2t = torch.norm(m2 - mu).item()

    # Covariance errors (relative Frobenius)
    def rel_frob(A, B):
        return (torch.norm(A - B) / torch.norm(B)).item()

    cov_err_12 = rel_frob(C1, C2)
    cov_err_1t = rel_frob(C1, Sigma)
    cov_err_2t = rel_frob(C2, Sigma)

    # Distribution check: KS on random projections
    num_proj = 20
    ks_stats = []
    for _ in range(num_proj):
        v = torch.randn(d, device=device, dtype=dtype)
        v = v / (torch.norm(v) + 1e-12)
        p1 = (X1 @ v)
        p2 = (X2 @ v)
        ks_stats.append(ks_1d(p1, p2))

    return {
        "mean_err_between_samples_L2": mean_err_12,
        "mean_err_reparam_vs_mu_L2": mean_err_1t,
        "mean_err_chol_vs_mu_L2": mean_err_2t,
        "cov_rel_frob_between_samples": cov_err_12,
        "cov_rel_frob_reparam_vs_theory": cov_err_1t,
        "cov_rel_frob_chol_vs_theory": cov_err_2t,
        "ks_stats_random_projections": ks_stats,
        "ks_max": max(ks_stats),
        "ks_mean": sum(ks_stats)/len(ks_stats),
    }

# Example run:
out = run_test(device="cpu")
for k, v in out.items():
    if k.startswith("ks_stats"):
        continue
    print(f"{k}: {v}")
print("ks_stats_random_projections:", out["ks_stats_random_projections"])