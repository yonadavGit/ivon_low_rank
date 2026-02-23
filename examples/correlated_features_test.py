"""
Better test: Create data with KNOWN correlation structure in the posterior,
then check if IVONLR can recover it better than IVON.
"""
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import ivon
import matplotlib.pyplot as plt


@dataclass
class Config:
    seed: int = 42
    n_samples: int = 100  # Reduced to create more uncertainty
    n_features: int = 40
    noise_std: float = 0.5  # Increased noise
    epochs: int = 2000
    lr: float = 0.15
    train_samples: int = 3
    test_samples: int = 1000
    rank: int = 8
    out_path: Path = Path("plots/ivon_vs_ivonlr_correlated.png")


def make_correlated_data(cfg: Config, device: torch.device):
    """Create data where parameters should have correlation structure"""
    torch.manual_seed(cfg.seed)

    # Create correlated features: first half and second half are related
    n_half = cfg.n_features // 2
    x_base = torch.randn(cfg.n_samples, n_half, device=device)
    # Second half correlated with first half
    x = torch.cat([
        x_base,
        x_base + 0.3 * torch.randn(cfg.n_samples, n_half, device=device)
    ], dim=1)

    # True weights: opposite signs in correlated pairs
    w_true = torch.randn(cfg.n_features, 1, device=device)
    # Make paired weights have opposite contributions
    w_true[n_half:] = -w_true[:n_half] + 0.1 * torch.randn(n_half, 1, device=device)

    y = x @ w_true + cfg.noise_std * torch.randn(cfg.n_samples, 1, device=device)
    return x, y, n_half


def train_optimizer(model, x, y, optimizer, cfg: Config):
    loss_fn = nn.MSELoss()
    for _ in range(cfg.epochs):
        for _ in range(cfg.train_samples):
            with optimizer.sampled_params(train=True):
                optimizer.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
        optimizer.step()


@torch.no_grad()
def sample_weights(model, optimizer, cfg: Config):
    samples = []
    for _ in range(cfg.test_samples):
        with optimizer.sampled_params():
            w = torch.cat([p.flatten() for p in model.parameters()], dim=0)
            samples.append(w.cpu())
    return torch.stack(samples, dim=0)


def covariance_from_samples(samples: torch.Tensor):
    centered = samples - samples.mean(dim=0, keepdim=True)
    return centered.T @ centered / (samples.shape[0] - 1)


def analyze_correlation_structure(cov: torch.Tensor, n_half: int):
    """Analyze the cross-correlation between first and second half of parameters"""
    # Extract cross-correlation block
    cross_block = cov[:n_half, n_half:2*n_half]

    # Diagonal of cross-block shows correlation between paired features
    paired_corr = torch.diag(cross_block)

    # Average magnitude
    avg_cross = cross_block.abs().mean().item()
    avg_paired = paired_corr.abs().mean().item()

    return avg_cross, avg_paired, cross_block


def plot_covariances(out_path: Path, cov_ivon, cov_ivonlr, n_half: int):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    vmax = max(cov_ivon.abs().max().item(), cov_ivonlr.abs().max().item())

    # Full covariances
    for ax, title, cov in zip(axes[0, :2], ["IVON", "IVONLR"], [cov_ivon, cov_ivonlr]):
        im = ax.imshow(cov, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{title} weight covariance")
        ax.axhline(n_half - 0.5, color='black', linewidth=1, linestyle='--')
        ax.axvline(n_half - 0.5, color='black', linewidth=1, linestyle='--')
        fig.colorbar(im, ax=ax)

    # Difference
    diff = cov_ivonlr - cov_ivon
    im = axes[0, 2].imshow(diff, cmap="coolwarm", vmin=-vmax/2, vmax=vmax/2)
    axes[0, 2].set_title("IVONLR - IVON")
    axes[0, 2].axhline(n_half - 0.5, color='black', linewidth=1, linestyle='--')
    axes[0, 2].axvline(n_half - 0.5, color='black', linewidth=1, linestyle='--')
    fig.colorbar(im, ax=axes[0, 2])

    # Cross-correlation blocks
    for ax, title, cov in zip(axes[1, :2], ["IVON cross-correlation", "IVONLR cross-correlation"],
                               [cov_ivon, cov_ivonlr]):
        cross_block = cov[:n_half, n_half:2*n_half]
        im = ax.imshow(cross_block, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Second half features")
        ax.set_ylabel("First half features")
        fig.colorbar(im, ax=ax)

    # Paired correlations
    ivon_cross = cov_ivon[:n_half, n_half:2*n_half]
    ivonlr_cross = cov_ivonlr[:n_half, n_half:2*n_half]
    ivon_paired = torch.diag(ivon_cross)
    ivonlr_paired = torch.diag(ivonlr_cross)

    axes[1, 2].plot(ivon_paired.numpy(), 'o-', label='IVON', alpha=0.7)
    axes[1, 2].plot(ivonlr_paired.numpy(), 's-', label='IVONLR', alpha=0.7)
    axes[1, 2].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[1, 2].set_xlabel("Feature pair index")
    axes[1, 2].set_ylabel("Covariance")
    axes[1, 2].set_title("Paired feature covariances")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y, n_half = make_correlated_data(cfg, device)

    print("Training IVON...")
    model_ivon = nn.Linear(cfg.n_features, 1, bias=False).to(device)
    optimizer_ivon = ivon.IVON(model_ivon.parameters(), lr=cfg.lr, ess=len(x))
    train_optimizer(model_ivon, x, y, optimizer_ivon, cfg)
    samples_ivon = sample_weights(model_ivon, optimizer_ivon, cfg)
    cov_ivon = covariance_from_samples(samples_ivon)

    print("Training IVONLR...")
    model_ivonlr = nn.Linear(cfg.n_features, 1, bias=False).to(device)
    optimizer_ivonlr = ivon.IVONLR(
        model_ivonlr.parameters(),
        lr=cfg.lr,
        ess=len(x),
        rank=cfg.rank,
    )
    train_optimizer(model_ivonlr, x, y, optimizer_ivonlr, cfg)
    samples_ivonlr = sample_weights(model_ivonlr, optimizer_ivonlr, cfg)
    cov_ivonlr = covariance_from_samples(samples_ivonlr)

    # Analyze correlation structure
    avg_cross_ivon, avg_paired_ivon, _ = analyze_correlation_structure(cov_ivon, n_half)
    avg_cross_ivonlr, avg_paired_ivonlr, _ = analyze_correlation_structure(cov_ivonlr, n_half)

    print(f"\nCross-correlation analysis (first half vs second half):")
    print(f"  IVON   - Average cross-block magnitude: {avg_cross_ivon:.6f}")
    print(f"  IVON   - Average paired correlation:    {avg_paired_ivon:.6f}")
    print(f"  IVONLR - Average cross-block magnitude: {avg_cross_ivonlr:.6f}")
    print(f"  IVONLR - Average paired correlation:    {avg_paired_ivonlr:.6f}")
    print(f"\nImprovement: {(avg_cross_ivonlr / avg_cross_ivon):.2f}x")

    plot_covariances(cfg.out_path, cov_ivon, cov_ivonlr, n_half)
    print(f"\nPlot saved to {cfg.out_path}")


if __name__ == "__main__":
    main()

