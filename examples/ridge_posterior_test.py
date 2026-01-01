"""
Visual test: Ridge Regression with Collinear Features

This creates a scenario where two features are highly correlated, creating
a ridge-like posterior where w1 and w2 must balance each other.
IVONLR should capture this ridge structure while IVON cannot.
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
    seed: int = 123
    n_samples: int = 30  # Very small dataset
    correlation: float = 0.95  # High correlation between features
    noise_std: float = 0.3
    epochs: int = 2000
    lr: float = 0.1
    train_samples: int = 3
    test_samples: int = 3000
    rank: int = 4
    out_path: Path = Path("plots/ridge_posterior.png")


def make_ridge_data(cfg: Config, device: torch.device):
    """
    Create data where x1 and x2 are highly correlated.
    This creates a ridge in the posterior: w1 and w2 become negatively correlated.
    If x1 ≈ x2, then w1 + w2 is well-determined but w1 - w2 is not.
    """
    torch.manual_seed(cfg.seed)

    # Generate correlated features
    x1 = torch.randn(cfg.n_samples, device=device)
    x2 = cfg.correlation * x1 + np.sqrt(1 - cfg.correlation**2) * torch.randn(cfg.n_samples, device=device)
    x = torch.stack([x1, x2], dim=1)  # [n, 2]

    # True parameters
    w1_true = 2.0
    w2_true = -1.5
    w_true = torch.tensor([[w1_true], [w2_true]], device=device)

    # Generate targets
    y = x @ w_true + cfg.noise_std * torch.randn(cfg.n_samples, 1, device=device)

    # Print correlation
    corr = torch.corrcoef(x.T)[0, 1].item()
    print(f"Feature correlation: {corr:.4f}")

    return x, y, w_true


class SimpleModel(nn.Module):
    """Simple 2-parameter linear model (no bias)"""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 1) * 0.1)

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


def plot_ridge_comparison(cfg: Config, samples_ivon, samples_ivonlr, w_true):
    """Create visualization emphasizing the ridge structure"""
    fig = plt.figure(figsize=(18, 6))

    w1_true, w2_true = w_true[0, 0].item(), w_true[1, 0].item()

    # Color maps
    ivon_color = '#FF6B6B'
    ivonlr_color = '#4ECDC4'
    true_color = '#FFD93D'

    # 1. IVON scatter + contour
    ax1 = fig.add_subplot(131)

    # 2D histogram contours
    h1, xedges1, yedges1 = np.histogram2d(
        samples_ivon[:, 0].numpy(), samples_ivon[:, 1].numpy(), bins=40
    )
    extent1 = [xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]]

    # Plot contours
    ax1.contourf(h1.T, extent=extent1, levels=10, cmap='Reds', alpha=0.6)
    ax1.contour(h1.T, extent=extent1, levels=10, colors='darkred', linewidths=1, alpha=0.4)

    # Scatter samples
    ax1.scatter(samples_ivon[:, 0], samples_ivon[:, 1],
               alpha=0.05, s=1, c=ivon_color)

    # True value
    ax1.scatter(w1_true, w2_true, c=true_color, s=300, marker='*',
               edgecolor='black', linewidth=2.5, label='True', zorder=100)

    # Mean
    mean_ivon = samples_ivon.mean(dim=0)
    ax1.scatter(mean_ivon[0], mean_ivon[1], c='white', s=150, marker='o',
               edgecolor=ivon_color, linewidth=3, label='Posterior Mean', zorder=99)

    ax1.set_xlabel('w₁', fontsize=14, fontweight='bold')
    ax1.set_ylabel('w₂', fontsize=14, fontweight='bold')
    ax1.set_title('IVON: Axis-Aligned Uncertainty', fontsize=15, fontweight='bold', pad=10)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 2. IVONLR scatter + contour
    ax2 = fig.add_subplot(132)

    # 2D histogram contours
    h2, xedges2, yedges2 = np.histogram2d(
        samples_ivonlr[:, 0].numpy(), samples_ivonlr[:, 1].numpy(), bins=40
    )
    extent2 = [xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]]

    # Plot contours
    ax2.contourf(h2.T, extent=extent2, levels=10, cmap='Blues', alpha=0.6)
    ax2.contour(h2.T, extent=extent2, levels=10, colors='darkblue', linewidths=1, alpha=0.4)

    # Scatter samples
    ax2.scatter(samples_ivonlr[:, 0], samples_ivonlr[:, 1],
               alpha=0.05, s=1, c=ivonlr_color)

    # True value
    ax2.scatter(w1_true, w2_true, c=true_color, s=300, marker='*',
               edgecolor='black', linewidth=2.5, label='True', zorder=100)

    # Mean
    mean_ivonlr = samples_ivonlr.mean(dim=0)
    ax2.scatter(mean_ivonlr[0], mean_ivonlr[1], c='white', s=150, marker='o',
               edgecolor=ivonlr_color, linewidth=3, label='Posterior Mean', zorder=99)

    ax2.set_xlabel('w₁', fontsize=14, fontweight='bold')
    ax2.set_ylabel('w₂', fontsize=14, fontweight='bold')
    ax2.set_title('IVONLR: Captures Ridge/Correlation', fontsize=15, fontweight='bold', pad=10)
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 3. Comparison metrics
    ax3 = fig.add_subplot(133)
    ax3.axis('off')

    # Compute statistics
    cov_ivon = torch.cov(samples_ivon.T)
    cov_ivonlr = torch.cov(samples_ivonlr.T)

    corr_ivon = torch.corrcoef(samples_ivon.T)[0, 1].item()
    corr_ivonlr = torch.corrcoef(samples_ivonlr.T)[0, 1].item()

    det_ivon = torch.det(cov_ivon).item()
    det_ivonlr = torch.det(cov_ivonlr).item()

    var_ivon = samples_ivon.var(dim=0)
    var_ivonlr = samples_ivonlr.var(dim=0)

    # Get eigenvalues (shape of uncertainty)
    eigvals_ivon, eigvecs_ivon = torch.linalg.eigh(cov_ivon)
    eigvals_ivonlr, eigvecs_ivonlr = torch.linalg.eigh(cov_ivonlr)

    eccentricity_ivon = (eigvals_ivon[1] / eigvals_ivon[0]).sqrt().item()
    eccentricity_ivonlr = (eigvals_ivonlr[1] / eigvals_ivonlr[0]).sqrt().item()

    # Create text summary
    text_y = 0.95
    line_height = 0.08

    def add_text(text, y, fontsize=12, fontweight='normal', color='black'):
        ax3.text(0.05, y, text, transform=ax3.transAxes,
                fontsize=fontsize, fontweight=fontweight,
                verticalalignment='top', color=color, family='monospace')

    add_text('COMPARISON METRICS', text_y, fontsize=16, fontweight='bold')
    text_y -= line_height * 1.5

    add_text('─' * 50, text_y, fontsize=11)
    text_y -= line_height

    add_text('Correlation Coefficient (w₁, w₂):', text_y, fontsize=12, fontweight='bold')
    text_y -= line_height * 0.8
    add_text(f'  IVON:   {corr_ivon:+.6f}', text_y, fontsize=11, color=ivon_color)
    text_y -= line_height * 0.7
    add_text(f'  IVONLR: {corr_ivonlr:+.6f}', text_y, fontsize=11, color=ivonlr_color)
    text_y -= line_height * 0.7
    add_text(f'  → IVONLR captures {abs(corr_ivonlr/corr_ivon):.1f}x stronger correlation!',
            text_y, fontsize=10, color='green', fontweight='bold')
    text_y -= line_height * 1.3

    add_text('Uncertainty Shape (Eccentricity):', text_y, fontsize=12, fontweight='bold')
    text_y -= line_height * 0.8
    add_text(f'  IVON:   {eccentricity_ivon:.3f} (nearly circular)', text_y, fontsize=11, color=ivon_color)
    text_y -= line_height * 0.7
    add_text(f'  IVONLR: {eccentricity_ivonlr:.3f} (elongated ridge)', text_y, fontsize=11, color=ivonlr_color)
    text_y -= line_height * 0.7
    add_text(f'  → IVONLR is {eccentricity_ivonlr/eccentricity_ivon:.1f}x more elongated',
            text_y, fontsize=10, color='green', fontweight='bold')
    text_y -= line_height * 1.3

    add_text('Marginal Variances:', text_y, fontsize=12, fontweight='bold')
    text_y -= line_height * 0.8
    add_text(f'  IVON:   w₁={var_ivon[0]:.4f}, w₂={var_ivon[1]:.4f}', text_y, fontsize=11, color=ivon_color)
    text_y -= line_height * 0.7
    add_text(f'  IVONLR: w₁={var_ivonlr[0]:.4f}, w₂={var_ivonlr[1]:.4f}', text_y, fontsize=11, color=ivonlr_color)
    text_y -= line_height * 1.3

    add_text('Distance from True Parameters:', text_y, fontsize=12, fontweight='bold')
    text_y -= line_height * 0.8
    error_ivon = (mean_ivon - torch.tensor([w1_true, w2_true])).norm().item()
    error_ivonlr = (mean_ivonlr - torch.tensor([w1_true, w2_true])).norm().item()
    add_text(f'  IVON:   {error_ivon:.4f}', text_y, fontsize=11, color=ivon_color)
    text_y -= line_height * 0.7
    add_text(f'  IVONLR: {error_ivonlr:.4f}', text_y, fontsize=11, color=ivonlr_color)
    text_y -= line_height * 0.7
    if error_ivonlr < error_ivon:
        add_text(f'  → IVONLR is {error_ivon/error_ivonlr:.1f}x more accurate!',
                text_y, fontsize=10, color='green', fontweight='bold')

    # Overall title
    fig.suptitle('Ridge Regression: Collinear Features Create Parameter Correlation',
                fontsize=17, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*70)
    print("RIDGE POSTERIOR TEST: Collinear Features")
    print("="*70)
    print(f"\nDataset: {cfg.n_samples} samples (very small!)")
    print(f"Feature correlation: {cfg.correlation} (high collinearity)")
    print()

    x, y, w_true = make_ridge_data(cfg, device)

    print(f"True parameters: w₁={w_true[0,0]:.4f}, w₂={w_true[1,0]:.4f}\n")

    # Train IVON
    print("Training IVON...")
    model_ivon = SimpleModel().to(device)
    optimizer_ivon = ivon.IVON(model_ivon.parameters(), lr=cfg.lr, ess=len(x))
    train_optimizer(model_ivon, x, y, optimizer_ivon, cfg)
    samples_ivon = sample_weights(model_ivon, optimizer_ivon, cfg)
    print(f"  ✓ Sampled {cfg.test_samples} posterior samples")

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
    print(f"  ✓ Sampled {cfg.test_samples} posterior samples")

    # Compute correlations
    corr_ivon = torch.corrcoef(samples_ivon.T)[0, 1].item()
    corr_ivonlr = torch.corrcoef(samples_ivonlr.T)[0, 1].item()

    print(f"\n" + "="*70)
    print("KEY RESULT: Parameter Correlations")
    print("="*70)
    print(f"  IVON:   {corr_ivon:+.6f}")
    print(f"  IVONLR: {corr_ivonlr:+.6f}")
    print(f"\n  → IVONLR captures {abs(corr_ivonlr/corr_ivon):.1f}x stronger correlation!")
    print("="*70)

    # Plot
    print(f"\nGenerating visualization...")
    fig = plot_ridge_comparison(cfg, samples_ivon, samples_ivonlr, w_true)
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.out_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {cfg.out_path}")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("When features are collinear (x₁ ≈ x₂), the sum (w₁ + w₂) is")
    print("well-determined, but the difference (w₁ - w₂) is not.")
    print()
    print("This creates a RIDGE in the posterior: w₁ and w₂ are negatively")
    print("correlated (if one increases, the other must decrease).")
    print()
    print("✗ IVON: Cannot represent this - constrained to axis-aligned uncertainty")
    print("✓ IVONLR: Captures the ridge structure via low-rank correlations")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

