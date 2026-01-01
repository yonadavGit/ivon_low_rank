"""
Visual test: Banana-shaped posterior (Rosenbrock-like)

This creates a scenario where parameters have strong non-axis-aligned correlations.
IVONLR should capture the banana shape while IVON can only represent axis-aligned uncertainty.
"""
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import ivon
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Config:
    seed: int = 42
    n_samples: int = 50  # Small dataset = more uncertainty
    noise_std: float = 0.1
    epochs: int = 3000
    lr: float = 0.1
    train_samples: int = 3
    test_samples: int = 2000
    rank: int = 4
    out_path: Path = Path("plots/banana_posterior.png")


def make_banana_data(cfg: Config, device: torch.device):
    """
    Create data where the posterior has a banana/curved shape.
    We'll use a model: y = w1*x + w2*x^2 + noise
    With constraints that create correlation between w1 and w2.
    """
    torch.manual_seed(cfg.seed)

    # Input range
    x = torch.linspace(-2, 2, cfg.n_samples, device=device).unsqueeze(1)

    # True parameters (centered so both matter)
    w1_true = 1.5
    w2_true = -0.5

    # Generate data: y = w1*x + w2*x^2
    x_features = torch.cat([x, x**2], dim=1)  # [n, 2]
    w_true = torch.tensor([[w1_true], [w2_true]], device=device)
    y = x_features @ w_true + cfg.noise_std * torch.randn(cfg.n_samples, 1, device=device)

    return x_features, y, w_true


class SimpleModel(nn.Module):
    """Simple 2-parameter linear model (no bias)"""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 1))

    def forward(self, x):
        return x @ self.weight


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
    """Sample from posterior"""
    samples = []
    for _ in range(cfg.test_samples):
        with optimizer.sampled_params():
            w = model.weight.flatten().cpu()
            samples.append(w)
    return torch.stack(samples, dim=0)


def plot_comparison(cfg: Config, samples_ivon, samples_ivonlr, w_true):
    """Create comprehensive visualization"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    w1_true, w2_true = w_true[0, 0].item(), w_true[1, 0].item()

    # Color maps
    ivon_color = '#FF6B6B'
    ivonlr_color = '#4ECDC4'
    true_color = '#FFD93D'

    # 1. Joint distribution scatter plot (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(samples_ivon[:, 0], samples_ivon[:, 1],
               alpha=0.3, s=1, c=ivon_color, label='IVON')
    ax1.scatter(w1_true, w2_true, c=true_color, s=200, marker='*',
               edgecolor='black', linewidth=2, label='True', zorder=100)
    ax1.set_xlabel('w₁ (linear term)', fontsize=11)
    ax1.set_ylabel('w₂ (quadratic term)', fontsize=11)
    ax1.set_title('IVON Posterior Samples', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Joint distribution scatter plot (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(samples_ivonlr[:, 0], samples_ivonlr[:, 1],
               alpha=0.3, s=1, c=ivonlr_color, label='IVONLR')
    ax2.scatter(w1_true, w2_true, c=true_color, s=200, marker='*',
               edgecolor='black', linewidth=2, label='True', zorder=100)
    ax2.set_xlabel('w₁ (linear term)', fontsize=11)
    ax2.set_ylabel('w₂ (quadratic term)', fontsize=11)
    ax2.set_title('IVONLR Posterior Samples', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Overlay (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(samples_ivon[:, 0], samples_ivon[:, 1],
               alpha=0.15, s=1, c=ivon_color, label='IVON')
    ax3.scatter(samples_ivonlr[:, 0], samples_ivonlr[:, 1],
               alpha=0.15, s=1, c=ivonlr_color, label='IVONLR')
    ax3.scatter(w1_true, w2_true, c=true_color, s=200, marker='*',
               edgecolor='black', linewidth=2, label='True', zorder=100)
    ax3.set_xlabel('w₁ (linear term)', fontsize=11)
    ax3.set_ylabel('w₂ (quadratic term)', fontsize=11)
    ax3.set_title('Overlay Comparison', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 2D histograms (middle row)
    ax4 = fig.add_subplot(gs[1, 0])
    h1 = ax4.hist2d(samples_ivon[:, 0].numpy(), samples_ivon[:, 1].numpy(),
                    bins=40, cmap='Reds', alpha=0.7)
    ax4.scatter(w1_true, w2_true, c=true_color, s=200, marker='*',
               edgecolor='black', linewidth=2, zorder=100)
    ax4.set_xlabel('w₁', fontsize=11)
    ax4.set_ylabel('w₂', fontsize=11)
    ax4.set_title('IVON Density', fontsize=12, fontweight='bold')
    plt.colorbar(h1[3], ax=ax4, label='Count')

    ax5 = fig.add_subplot(gs[1, 1])
    h2 = ax5.hist2d(samples_ivonlr[:, 0].numpy(), samples_ivonlr[:, 1].numpy(),
                    bins=40, cmap='Blues', alpha=0.7)
    ax5.scatter(w1_true, w2_true, c=true_color, s=200, marker='*',
               edgecolor='black', linewidth=2, zorder=100)
    ax5.set_xlabel('w₁', fontsize=11)
    ax5.set_ylabel('w₂', fontsize=11)
    ax5.set_title('IVONLR Density', fontsize=12, fontweight='bold')
    plt.colorbar(h2[3], ax=ax5, label='Count')

    # 5. Covariance ellipses (middle right)
    ax6 = fig.add_subplot(gs[1, 2])

    # Compute covariances
    cov_ivon = torch.cov(samples_ivon.T)
    cov_ivonlr = torch.cov(samples_ivonlr.T)

    mean_ivon = samples_ivon.mean(dim=0)
    mean_ivonlr = samples_ivonlr.mean(dim=0)

    # Plot ellipses
    def plot_cov_ellipse(ax, mean, cov, color, label, n_std=2):
        from matplotlib.patches import Ellipse
        eigvals, eigvecs = torch.linalg.eigh(cov)
        angle = torch.atan2(eigvecs[1, 1], eigvecs[0, 1]) * 180 / np.pi
        width, height = 2 * n_std * torch.sqrt(eigvals)
        ellipse = Ellipse(mean.numpy(), width.item(), height.item(),
                         angle=angle.item(), facecolor=color, alpha=0.3,
                         edgecolor=color, linewidth=2, label=label)
        ax.add_patch(ellipse)

    plot_cov_ellipse(ax6, mean_ivon, cov_ivon, ivon_color, 'IVON (2σ)')
    plot_cov_ellipse(ax6, mean_ivonlr, cov_ivonlr, ivonlr_color, 'IVONLR (2σ)')
    ax6.scatter(w1_true, w2_true, c=true_color, s=200, marker='*',
               edgecolor='black', linewidth=2, label='True', zorder=100)
    ax6.scatter(mean_ivon[0], mean_ivon[1], c=ivon_color, s=100, marker='o',
               edgecolor='black', linewidth=1.5, zorder=99)
    ax6.scatter(mean_ivonlr[0], mean_ivonlr[1], c=ivonlr_color, s=100, marker='o',
               edgecolor='black', linewidth=1.5, zorder=99)
    ax6.set_xlabel('w₁', fontsize=11)
    ax6.set_ylabel('w₂', fontsize=11)
    ax6.set_title('2σ Uncertainty Ellipses', fontsize=12, fontweight='bold')
    ax6.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)
    ax6.axis('equal')

    # Set same limits for fair comparison
    all_samples = torch.cat([samples_ivon, samples_ivonlr], dim=0)
    xlim = [all_samples[:, 0].min() - 0.1, all_samples[:, 0].max() + 0.1]
    ylim = [all_samples[:, 1].min() - 0.1, all_samples[:, 1].max() + 0.1]
    for ax in [ax6]:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # 6. Marginal distributions (bottom row)
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(samples_ivon[:, 0].numpy(), bins=50, alpha=0.5, color=ivon_color,
            label='IVON', density=True)
    ax7.hist(samples_ivonlr[:, 0].numpy(), bins=50, alpha=0.5, color=ivonlr_color,
            label='IVONLR', density=True)
    ax7.axvline(w1_true, color=true_color, linewidth=3, label='True', linestyle='--')
    ax7.set_xlabel('w₁ (linear term)', fontsize=11)
    ax7.set_ylabel('Density', fontsize=11)
    ax7.set_title('Marginal: w₁', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(samples_ivon[:, 1].numpy(), bins=50, alpha=0.5, color=ivon_color,
            label='IVON', density=True)
    ax8.hist(samples_ivonlr[:, 1].numpy(), bins=50, alpha=0.5, color=ivonlr_color,
            label='IVONLR', density=True)
    ax8.axvline(w2_true, color=true_color, linewidth=3, label='True', linestyle='--')
    ax8.set_xlabel('w₂ (quadratic term)', fontsize=11)
    ax8.set_ylabel('Density', fontsize=11)
    ax8.set_title('Marginal: w₂', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')

    # 7. Correlation comparison (bottom right)
    ax9 = fig.add_subplot(gs[2, 2])

    corr_ivon = torch.corrcoef(samples_ivon.T)[0, 1].item()
    corr_ivonlr = torch.corrcoef(samples_ivonlr.T)[0, 1].item()

    # Compute off-diagonal covariance magnitude
    offdiag_ivon = abs(cov_ivon[0, 1].item())
    offdiag_ivonlr = abs(cov_ivonlr[0, 1].item())

    metrics = ['Correlation\n(w₁, w₂)', 'Off-diagonal\nCovariance']
    ivon_vals = [abs(corr_ivon), offdiag_ivon]
    ivonlr_vals = [abs(corr_ivonlr), offdiag_ivonlr]

    x_pos = np.arange(len(metrics))
    width = 0.35

    bars1 = ax9.bar(x_pos - width/2, ivon_vals, width, label='IVON',
                   color=ivon_color, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax9.bar(x_pos + width/2, ivonlr_vals, width, label='IVONLR',
                   color=ivonlr_color, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax9.set_ylabel('Magnitude', fontsize=11)
    ax9.set_title('Correlation Metrics', fontsize=12, fontweight='bold')
    ax9.set_xticks(x_pos)
    ax9.set_xticklabels(metrics, fontsize=10)
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')

    # Overall title
    fig.suptitle('IVON vs IVONLR: Banana-Shaped Posterior (2D Polynomial Regression)',
                fontsize=16, fontweight='bold', y=0.995)

    return fig


def print_statistics(samples_ivon, samples_ivonlr, w_true):
    """Print detailed statistics"""
    print("\n" + "="*70)
    print("POSTERIOR STATISTICS")
    print("="*70)

    w1_true, w2_true = w_true[0, 0].item(), w_true[1, 0].item()

    # Means
    mean_ivon = samples_ivon.mean(dim=0)
    mean_ivonlr = samples_ivonlr.mean(dim=0)

    print(f"\nTrue parameters: w₁={w1_true:.4f}, w₂={w2_true:.4f}")
    print(f"\nPosterior means:")
    print(f"  IVON:   w₁={mean_ivon[0]:.4f}, w₂={mean_ivon[1]:.4f}")
    print(f"  IVONLR: w₁={mean_ivonlr[0]:.4f}, w₂={mean_ivonlr[1]:.4f}")

    # Errors
    error_ivon = (mean_ivon - torch.tensor([w1_true, w2_true])).norm().item()
    error_ivonlr = (mean_ivonlr - torch.tensor([w1_true, w2_true])).norm().item()
    print(f"\nDistance from truth:")
    print(f"  IVON:   {error_ivon:.4f}")
    print(f"  IVONLR: {error_ivonlr:.4f}")

    # Covariances
    cov_ivon = torch.cov(samples_ivon.T)
    cov_ivonlr = torch.cov(samples_ivonlr.T)

    print(f"\nCovariance matrices:")
    print(f"  IVON:")
    print(f"    {cov_ivon.numpy()}")
    print(f"  IVONLR:")
    print(f"    {cov_ivonlr.numpy()}")

    # Correlations
    corr_ivon = torch.corrcoef(samples_ivon.T)[0, 1].item()
    corr_ivonlr = torch.corrcoef(samples_ivonlr.T)[0, 1].item()

    print(f"\nCorrelation coefficients:")
    print(f"  IVON:   {corr_ivon:.6f}")
    print(f"  IVONLR: {corr_ivonlr:.6f}")
    print(f"  Ratio:  {abs(corr_ivonlr/corr_ivon):.2f}x")

    # Variances
    var_ivon = samples_ivon.var(dim=0)
    var_ivonlr = samples_ivonlr.var(dim=0)

    print(f"\nMarginal variances:")
    print(f"  IVON:   w₁={var_ivon[0]:.6f}, w₂={var_ivon[1]:.6f}")
    print(f"  IVONLR: w₁={var_ivonlr[0]:.6f}, w₂={var_ivonlr[1]:.6f}")

    # Determinant (volume)
    det_ivon = torch.det(cov_ivon).item()
    det_ivonlr = torch.det(cov_ivonlr).item()

    print(f"\nCovariance determinant (uncertainty volume):")
    print(f"  IVON:   {det_ivon:.8f}")
    print(f"  IVONLR: {det_ivonlr:.8f}")
    print(f"  Ratio:  {det_ivonlr/det_ivon:.4f}x")

    print("\n" + "="*70)


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Generating banana-shaped posterior data...")
    x, y, w_true = make_banana_data(cfg, device)

    print(f"Training with {cfg.n_samples} samples (small dataset = high uncertainty)")
    print(f"True parameters: w1={w_true[0, 0]:.4f}, w2={w_true[1, 0]:.4f}\n")

    # Train IVON
    print("Training IVON...")
    model_ivon = SimpleModel().to(device)
    optimizer_ivon = ivon.IVON(model_ivon.parameters(), lr=cfg.lr, ess=len(x))
    train_optimizer(model_ivon, x, y, optimizer_ivon, cfg)
    samples_ivon = sample_weights(model_ivon, optimizer_ivon, cfg)
    print(f"  Sampled {cfg.test_samples} posterior samples")

    # Train IVONLR
    print("Training IVONLR...")
    model_ivonlr = SimpleModel().to(device)
    optimizer_ivonlr = ivon.IVONLR(
        model_ivonlr.parameters(),
        lr=cfg.lr,
        ess=len(x),
        rank=cfg.rank,
    )
    train_optimizer(model_ivonlr, x, y, optimizer_ivonlr, cfg)
    samples_ivonlr = sample_weights(model_ivonlr, optimizer_ivonlr, cfg)
    print(f"  Sampled {cfg.test_samples} posterior samples")

    # Statistics
    print_statistics(samples_ivon, samples_ivonlr, w_true)

    # Plot
    print(f"\nGenerating visualization...")
    fig = plot_comparison(cfg, samples_ivon, samples_ivonlr, w_true)
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.out_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {cfg.out_path}")

    print("\n✓ Done! IVONLR should show strong correlation between w₁ and w₂,")
    print("  while IVON is constrained to axis-aligned (diagonal) uncertainty.")


if __name__ == "__main__":
    main()

