"""
Posterior Approximation Test: Small dim with strong off-diagonal covariance.
"""
import torch
from torch import nn

from ivon._ivon import IVON, IVONLR
from ivon.fixed_ivonlr import IVONLR as IVONLRFixed
from ivon.cov_ivonlr import IVONLRCov


def make_data(n_samples=500, n_dims=3, noise_std=0.01, seed=42):
    torch.manual_seed(seed)
    x1 = torch.randn(n_samples, 1)
    x2 = x1 + 0.01 * torch.randn(n_samples, 1)
    x3 = torch.randn(n_samples, 1)
    x = torch.cat([x1, x2, x3], dim=1)
    w_true = torch.randn(n_dims, 1)
    y = x @ w_true + noise_std * torch.randn(n_samples, 1)
    return x, y


class SimpleModel(nn.Module):
    def __init__(self, n_dims):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_dims, 1) * 0.1)

    def forward(self, x):
        return x @ self.weight


def train(model, x, y, optimizer, epochs=1500):
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        for _ in range(3):
            with optimizer.sampled_params(train=True):
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
        optimizer.step()


@torch.no_grad()
def sample_posterior(model, optimizer, n_samples=3000):
    samples = []
    for _ in range(n_samples):
        with optimizer.sampled_params():
            w = model.weight.flatten().cpu().clone()
            samples.append(w)
    return torch.stack(samples, dim=0)


def covariance(samples):
    centered = samples - samples.mean(dim=0, keepdim=True)
    return centered.t() @ centered / (samples.shape[0] - 1)


def main():
    n_dims = 3
    noise_std = 0.01
    weight_decay = 1e-6

    print("\n" + "=" * 70)
    print("POSTERIOR APPROX TEST: High Off-Diagonal")
    print("=" * 70)
    print(f"\nDims: {n_dims}, noise_std: {noise_std}, weight_decay: {weight_decay}")

    x, y = make_data(n_samples=500, n_dims=n_dims, noise_std=noise_std)

    xtx = x.t() @ x
    sigma2 = noise_std ** 2
    precision = xtx / sigma2 + weight_decay * torch.eye(n_dims)
    cov_true = torch.linalg.inv(precision)

    print("\nTraining IVON...")
    model_ivon = SimpleModel(n_dims)
    opt_ivon = IVON(model_ivon.parameters(), lr=0.1, ess=len(x), weight_decay=weight_decay)
    train(model_ivon, x, y, opt_ivon)
    samples_ivon = sample_posterior(model_ivon, opt_ivon)
    print("  ✓ Done")

    print("Training old IVONLR...")
    model_lr = SimpleModel(n_dims)
    opt_lr = IVONLR(model_lr.parameters(), lr=0.1, ess=len(x), rank=2, weight_decay=weight_decay)
    train(model_lr, x, y, opt_lr)
    samples_lr = sample_posterior(model_lr, opt_lr)
    print("  ✓ Done")

    print("Training fixed IVONLR...")
    model_fixed = SimpleModel(n_dims)
    opt_fixed = IVONLRFixed(model_fixed.parameters(), lr=0.1, ess=len(x), rank=2, weight_decay=weight_decay)
    train(model_fixed, x, y, opt_fixed)
    samples_fixed = sample_posterior(model_fixed, opt_fixed)
    print("  ✓ Done")

    print("Training IVONLRCov...")
    model_cov = SimpleModel(n_dims)
    opt_cov = IVONLRCov(model_cov.parameters(), lr=0.1, ess=len(x), rank=2, weight_decay=weight_decay)
    train(model_cov, x, y, opt_cov)
    samples_cov = sample_posterior(model_cov, opt_cov)
    print("  ✓ Done")

    cov_ivon = covariance(samples_ivon)
    cov_lr = covariance(samples_lr)
    cov_fixed = covariance(samples_fixed)
    cov_cov = covariance(samples_cov)

    def rel_frob(a, b):
        return (a - b).norm() / b.norm()

    print("\nCovariance error (relative Frobenius):")
    print(f"  IVON:   {rel_frob(cov_ivon, cov_true):.4f}")
    print(f"  IVONLR: {rel_frob(cov_lr, cov_true):.4f}")
    print(f"  Fixed:  {rel_frob(cov_fixed, cov_true):.4f}")
    print(f"  Cov:    {rel_frob(cov_cov, cov_true):.4f}")

    def max_offdiag(cov):
        off = cov - torch.diag(torch.diag(cov))
        return off.abs().max().item()

    def max_corr(cov):
        std = cov.diag().sqrt()
        denom = std[:, None] * std[None, :]
        corr = cov / denom
        corr = corr - torch.diag(torch.diag(corr))
        return corr.abs().max().item()

    print("\nMax |offdiag|:")
    print(f"  True:   {max_offdiag(cov_true):.6f}")
    print(f"  IVON:   {max_offdiag(cov_ivon):.6f}")
    print(f"  IVONLR: {max_offdiag(cov_lr):.6f}")
    print(f"  Fixed:  {max_offdiag(cov_fixed):.6f}")
    print(f"  Cov:    {max_offdiag(cov_cov):.6f}")

    print("\nMax |corr|:")
    print(f"  True:   {max_corr(cov_true):.4f}")
    print(f"  IVON:   {max_corr(cov_ivon):.4f}")
    print(f"  IVONLR: {max_corr(cov_lr):.4f}")
    print(f"  Fixed:  {max_corr(cov_fixed):.4f}")
    print(f"  Cov:    {max_corr(cov_cov):.4f}")


if __name__ == "__main__":
    main()
