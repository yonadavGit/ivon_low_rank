"""
Generate a suite of posterior heatmaps for multiple data constructions.
Each case saves a 3-panel heatmap: True vs IVON vs IVONLR (fixed).
"""
import torch
from torch import nn
import matplotlib.pyplot as plt

from ivon._ivon import IVON
from ivon.fixed_ivonlr import IVONLR


def make_toeplitz(n_samples=1000, n_dims=20, rho=0.8, noise_std=0.2, seed=42):
    torch.manual_seed(seed)
    idx = torch.arange(n_dims)
    cov = rho ** (idx[:, None] - idx[None, :]).abs().float()
    L = torch.linalg.cholesky(cov)
    x = torch.randn(n_samples, n_dims) @ L.t()
    w_true = torch.randn(n_dims, 1)
    y = x @ w_true + noise_std * torch.randn(n_samples, 1)
    return x, y


def make_block(n_samples=1000, n_dims=20, rho=0.85, noise_std=0.2, seed=42):
    torch.manual_seed(seed)
    n1 = n_dims // 2
    n2 = n_dims - n1
    z1 = torch.randn(n_samples, 1)
    z2 = torch.randn(n_samples, 1)
    b1 = rho * z1 + (1.0 - rho) * torch.randn(n_samples, n1)
    b2 = rho * z2 + (1.0 - rho) * torch.randn(n_samples, n2)
    x = torch.cat([b1, b2], dim=1)
    w_true = torch.randn(n_dims, 1)
    y = x @ w_true + noise_std * torch.randn(n_samples, 1)
    return x, y


def make_low_rank(n_samples=1000, n_dims=20, rank=4, noise_std=0.2, seed=42):
    torch.manual_seed(seed)
    z = torch.randn(n_samples, rank)
    b = torch.randn(rank, n_dims)
    x = z @ b + 0.05 * torch.randn(n_samples, n_dims)
    w_true = torch.randn(n_dims, 1)
    y = x @ w_true + noise_std * torch.randn(n_samples, 1)
    return x, y


def make_collinear(n_samples=1000, n_dims=20, noise_std=0.2, seed=42):
    torch.manual_seed(seed)
    z = torch.randn(n_samples, 1)
    x = z + 0.05 * torch.randn(n_samples, n_dims)
    w_true = torch.randn(n_dims, 1)
    y = x @ w_true + noise_std * torch.randn(n_samples, 1)
    return x, y


class SimpleModel(nn.Module):
    def __init__(self, n_dims):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_dims, 1) * 0.1)

    def forward(self, x):
        return x @ self.weight


def train(model, x, y, optimizer, epochs=1200):
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        for _ in range(3):
            with optimizer.sampled_params(train=True):
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
        optimizer.step()


@torch.no_grad()
def sample_posterior(model, optimizer, n_samples=2500):
    samples = []
    for _ in range(n_samples):
        with optimizer.sampled_params():
            w = model.weight.flatten().cpu().clone()
            samples.append(w)
    return torch.stack(samples, dim=0)


def covariance(samples):
    centered = samples - samples.mean(dim=0, keepdim=True)
    return centered.t() @ centered / (samples.shape[0] - 1)


def run_case(name, make_fn, n_dims=20, noise_std=0.2, weight_decay=1e-3, rank=10):
    x, y = make_fn(n_samples=1000, n_dims=n_dims, noise_std=noise_std)
    xtx = x.t() @ x
    sigma2 = noise_std ** 2
    precision = xtx / sigma2 + weight_decay * torch.eye(n_dims)
    cov_true = torch.linalg.inv(precision)

    model_ivon = SimpleModel(n_dims)
    opt_ivon = IVON(model_ivon.parameters(), lr=0.1, ess=len(x), weight_decay=weight_decay)
    train(model_ivon, x, y, opt_ivon)
    samples_ivon = sample_posterior(model_ivon, opt_ivon)

    model_lr = SimpleModel(n_dims)
    opt_lr = IVONLR(model_lr.parameters(), lr=0.1, ess=len(x), rank=rank, weight_decay=weight_decay)
    train(model_lr, x, y, opt_lr)
    samples_lr = sample_posterior(model_lr, opt_lr)

    cov_ivon = covariance(samples_ivon)
    cov_lr = covariance(samples_lr)

    cov_true = torch.nan_to_num(cov_true)
    cov_ivon = torch.nan_to_num(cov_ivon)
    cov_lr = torch.nan_to_num(cov_lr)
    all_vals = torch.cat([cov_true.flatten(), cov_ivon.flatten(), cov_lr.flatten()])
    vmax = all_vals.abs().quantile(0.99).item()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, mat, title in zip(
        axes,
        [cov_true, cov_ivon, cov_lr],
        ["True Posterior", "IVON", "IVONLR (fixed)"],
    ):
        im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    fig.suptitle(f"Case: {name}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = f"plots/cov_heatmap_{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"✓ Saved {out}")


def main():
    print("\n" + "=" * 70)
    print("COVARIANCE HEATMAP SUITE")
    print("=" * 70)

    cases = [
        ("toeplitz", make_toeplitz),
        ("block", make_block),
        ("low_rank", make_low_rank),
        ("collinear", make_collinear),
    ]

    for name, fn in cases:
        print(f"\nRunning case: {name}")
        run_case(name, fn)


if __name__ == "__main__":
    main()
