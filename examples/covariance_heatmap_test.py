"""
Covariance heatmap test: compare true vs IVON vs IVONLR.
"""
import torch
from torch import nn
import matplotlib.pyplot as plt

from ivon._ivon import IVON
from ivon._ivon import IVONLR



def make_data(n_samples=1000, n_dims=20, rho=0.8, noise_std=0.2, seed=42):
    torch.manual_seed(seed)
    idx = torch.arange(n_dims)
    cov = rho ** (idx[:, None] - idx[None, :]).abs().float()
    L = torch.linalg.cholesky(cov)
    x = torch.randn(n_samples, n_dims) @ L.t()
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
    n_dims = 20
    noise_std = 0.2
    weight_decay = 1e-3
    rho = 0.8

    print("\n" + "=" * 70)
    print("COVARIANCE HEATMAP TEST")
    print("=" * 70)
    print(f"\nDims: {n_dims}, noise_std: {noise_std}, weight_decay: {weight_decay}, rho: {rho}")

    x, y = make_data(n_samples=1000, n_dims=n_dims, rho=rho, noise_std=noise_std)

    xtx = x.t() @ x
    sigma2 = noise_std ** 2
    precision = xtx / sigma2 + weight_decay * torch.eye(n_dims)
    precision = precision + 1e-6 * torch.eye(n_dims)
    cov_true = torch.linalg.inv(precision)

    print("Training IVON...")
    model_ivon = SimpleModel(n_dims)
    opt_ivon = IVON(model_ivon.parameters(), lr=0.1, ess=len(x), weight_decay=weight_decay)
    train(model_ivon, x, y, opt_ivon)
    samples_ivon = sample_posterior(model_ivon, opt_ivon)
    print("  ✓ Done")

    print("Training IVONLR (fixed)...")
    model_lr = SimpleModel(n_dims)
    opt_lr = IVONLR(model_lr.parameters(), lr=0.1, ess=len(x), rank=10, weight_decay=weight_decay)
    train(model_lr, x, y, opt_lr)
    samples_lr = sample_posterior(model_lr, opt_lr)
    print("  ✓ Done")

    cov_ivon = covariance(samples_ivon)
    cov_lr = covariance(samples_lr)

    def _stats(name, mat):
        finite = torch.isfinite(mat)
        if not finite.all():
            print(f"{name}: non-finite entries present")
        vals = mat[finite]
        print(f"{name}: min={vals.min().item():.3e}, max={vals.max().item():.3e}")

    _stats("cov_true", cov_true)
    _stats("cov_ivon", cov_ivon)
    _stats("cov_lr", cov_lr)

    cov_true = torch.nan_to_num(cov_true)
    cov_ivon = torch.nan_to_num(cov_ivon)
    cov_lr = torch.nan_to_num(cov_lr)
    all_vals = torch.cat([cov_true.flatten(), cov_ivon.flatten(), cov_lr.flatten()])
    vmax = all_vals.abs().quantile(0.99).item()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, mat, title in zip(
        axes,
        [cov_true, cov_ivon, cov_lr],
        ["True Posterior", "IVON", "IVONLR"],
    ):
        im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig("plots/covariance_heatmap.pdf", bbox_inches="tight")
    print("✓ Saved to plots/covariance_heatmap.pdf")


if __name__ == "__main__":
    main()
