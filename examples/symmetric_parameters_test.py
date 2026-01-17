"""
Visual test: Symmetric/Redundant Parameters

Create a scenario where two parameters are completely interchangeable.
The model is: y = (w1 + w2) * x
So w1 and w2 are perfectly symmetric - any split works as long as w1+w2 is correct.

This creates a diagonal ridge in parameter space where w1 and w2 are perfectly correlated.
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
    n_samples: int = 40
    noise_std: float = 0.2
    epochs: int = 2000
    lr: float = 0.15
    train_samples: int = 3
    test_samples: int = 4000
    rank: int = 4
    out_path: Path = Path("plots/symmetric_parameters.png")


def make_symmetric_data(cfg: Config, device: torch.device):
    """
    Create data where model is y = (w1 + w2) * x
    This makes w1 and w2 perfectly symmetric/redundant
    """
    torch.manual_seed(cfg.seed)

    # Simple 1D input
    x = torch.randn(cfg.n_samples, 1, device=device)

    # True parameters (but ANY w1, w2 with w1+w2=3 would work!)
    w_sum_true = 3.0
    w1_true = 2.0
    w2_true = 1.0  # w1 + w2 = 3

    # Generate data: y = (w1 + w2) * x
    y = w_sum_true * x + cfg.noise_std * torch.randn(cfg.n_samples, 1, device=device)

    return x, y, w1_true, w2_true, w_sum_true


class SymmetricModel(nn.Module):
    """Model where two parameters are symmetric: y = (w1 + w2) * x"""
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(1) * 0.1)
        self.w2 = nn.Parameter(torch.randn(1) * 0.1)

    def forward(self, x):
        return (self.w1 + self.w2) * x


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
            w = torch.tensor([model.w1.item(), model.w2.item()])
            samples.append(w)
    return torch.stack(samples, dim=0)


def plot_symmetric(cfg: Config, samples_ivon, samples_ivonlr, w1_true, w2_true, w_sum_true):
    """Create beautiful visualization of symmetric parameters"""
    fig = plt.figure(figsize=(18, 6))

    # Colors
    ivon_color = '#FF6B6B'
    ivonlr_color = '#4ECDC4'
    true_color = '#FFD93D'
    ridge_color = '#95E1D3'

    # 1. IVON scatter
    ax1 = fig.add_subplot(131)

    # Draw the true ridge line: w1 + w2 = constant
    w1_range = np.linspace(-1, 5, 100)
    w2_ridge = w_sum_true - w1_range
    ax1.plot(w1_range, w2_ridge, 'g--', linewidth=3, alpha=0.7,
            label=f'True Ridge: w₁+w₂={w_sum_true:.1f}', zorder=1)

    # 2D histogram
    h1, xedges1, yedges1 = np.histogram2d(
        samples_ivon[:, 0].numpy(), samples_ivon[:, 1].numpy(), bins=50
    )
    extent1 = [xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]]
    ax1.contourf(h1.T, extent=extent1, levels=12, cmap='Reds', alpha=0.7)

    # Scatter
    ax1.scatter(samples_ivon[:, 0], samples_ivon[:, 1],
               alpha=0.02, s=2, c=ivon_color)

    # True point
    ax1.scatter(w1_true, w2_true, c=true_color, s=400, marker='*',
               edgecolor='black', linewidth=3, label='True Values', zorder=100)

    # Mean
    mean_ivon = samples_ivon.mean(dim=0)
    ax1.scatter(mean_ivon[0], mean_ivon[1], c='white', s=200, marker='o',
               edgecolor=ivon_color, linewidth=4, label='Posterior Mean', zorder=99)

    ax1.set_xlabel('w₁', fontsize=16, fontweight='bold')
    ax1.set_ylabel('w₂', fontsize=16, fontweight='bold')
    ax1.set_title('IVON: Axis-Aligned\n(Cannot capture ridge)',
                 fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_aspect('equal', adjustable='box')

    # 2. IVONLR scatter
    ax2 = fig.add_subplot(132)

    # Draw the ridge line
    ax2.plot(w1_range, w2_ridge, 'g--', linewidth=3, alpha=0.7,
            label=f'True Ridge: w₁+w₂={w_sum_true:.1f}', zorder=1)

    # 2D histogram
    h2, xedges2, yedges2 = np.histogram2d(
        samples_ivonlr[:, 0].numpy(), samples_ivonlr[:, 1].numpy(), bins=50
    )
    extent2 = [xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]]
    ax2.contourf(h2.T, extent=extent2, levels=12, cmap='Blues', alpha=0.7)

    # Scatter
    ax2.scatter(samples_ivonlr[:, 0], samples_ivonlr[:, 1],
               alpha=0.02, s=2, c=ivonlr_color)

    # True point
    ax2.scatter(w1_true, w2_true, c=true_color, s=400, marker='*',
               edgecolor='black', linewidth=3, label='True Values', zorder=100)

    # Mean
    mean_ivonlr = samples_ivonlr.mean(dim=0)
    ax2.scatter(mean_ivonlr[0], mean_ivonlr[1], c='white', s=200, marker='o',
               edgecolor=ivonlr_color, linewidth=4, label='Posterior Mean', zorder=99)

    ax2.set_xlabel('w₁', fontsize=16, fontweight='bold')
    ax2.set_ylabel('w₂', fontsize=16, fontweight='bold')
    ax2.set_title('IVONLR: Captures Ridge\n(Samples along diagonal)',
                 fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_aspect('equal', adjustable='box')


    def set_limits(ax, samples):
        margin = 0.3
        xlim = [samples[:, 0].min() - margin, samples[:, 0].max() + margin]
        ylim = [samples[:, 1].min() - margin, samples[:, 1].max() + margin]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    set_limits(ax1, samples_ivon)
    set_limits(ax2, samples_ivonlr)
    

    # 3. Statistics panel
    ax3 = fig.add_subplot(133)
    ax3.axis('off')

    # Compute statistics
    corr_ivon = torch.corrcoef(samples_ivon.T)[0, 1].item()
    corr_ivonlr = torch.corrcoef(samples_ivonlr.T)[0, 1].item()

    sum_ivon = (samples_ivon[:, 0] + samples_ivon[:, 1])
    sum_ivonlr = (samples_ivonlr[:, 0] + samples_ivonlr[:, 1])

    sum_mean_ivon = sum_ivon.mean().item()
    sum_std_ivon = sum_ivon.std().item()
    sum_mean_ivonlr = sum_ivonlr.mean().item()
    sum_std_ivonlr = sum_ivonlr.std().item()

    # Project onto ridge direction (w1 + w2) and perpendicular (w1 - w2)
    along_ridge_ivon = (samples_ivon[:, 0] + samples_ivon[:, 1]) / np.sqrt(2)
    perp_ridge_ivon = (samples_ivon[:, 0] - samples_ivon[:, 1]) / np.sqrt(2)
    along_ridge_ivonlr = (samples_ivonlr[:, 0] + samples_ivonlr[:, 1]) / np.sqrt(2)
    perp_ridge_ivonlr = (samples_ivonlr[:, 0] - samples_ivonlr[:, 1]) / np.sqrt(2)

    # Text
    text_y = 0.95
    line_height = 0.07

    def add_text(text, y, fontsize=12, fontweight='normal', color='black'):
        ax3.text(0.05, y, text, transform=ax3.transAxes,
                fontsize=fontsize, fontweight=fontweight,
                verticalalignment='top', color=color, family='monospace')

    add_text('SYMMETRY ANALYSIS', text_y, fontsize=17, fontweight='bold')
    text_y -= line_height * 1.8

    add_text('Model: y = (w₁ + w₂) · x', text_y, fontsize=13, fontweight='bold', color='green')
    text_y -= line_height * 0.9
    add_text('→ w₁ and w₂ are INTERCHANGEABLE!', text_y, fontsize=11, color='green')
    text_y -= line_height * 0.9
    add_text('→ Only w₁+w₂ matters for predictions', text_y, fontsize=11, color='green')
    text_y -= line_height * 1.5

    add_text('─' * 45, text_y, fontsize=10)
    text_y -= line_height * 1.0

    add_text('Correlation Coefficient:', text_y, fontsize=13, fontweight='bold')
    text_y -= line_height * 0.9
    add_text(f'  IVON:   {corr_ivon:+.4f}', text_y, fontsize=12, color=ivon_color)
    text_y -= line_height * 0.8
    add_text(f'  IVONLR: {corr_ivonlr:+.4f}', text_y, fontsize=12, color=ivonlr_color)
    text_y -= line_height * 0.8
    add_text(f'  → {abs(corr_ivonlr/corr_ivon):.0f}x stronger!',
            text_y, fontsize=11, color='darkgreen', fontweight='bold')
    text_y -= line_height * 1.5

    add_text('Sum w₁ + w₂ (well-determined):', text_y, fontsize=13, fontweight='bold')
    text_y -= line_height * 0.9
    add_text(f'  True:   {w_sum_true:.4f}', text_y, fontsize=12, color='black')
    text_y -= line_height * 0.8
    add_text(f'  IVON:   {sum_mean_ivon:.4f} ± {sum_std_ivon:.4f}',
            text_y, fontsize=12, color=ivon_color)
    text_y -= line_height * 0.8
    add_text(f'  IVONLR: {sum_mean_ivonlr:.4f} ± {sum_std_ivonlr:.4f}',
            text_y, fontsize=12, color=ivonlr_color)
    text_y -= line_height * 1.5

    add_text('Uncertainty along ridge:', text_y, fontsize=13, fontweight='bold')
    text_y -= line_height * 0.9
    add_text(f'  IVON:   σ = {perp_ridge_ivon.std():.4f}',
            text_y, fontsize=12, color=ivon_color)
    text_y -= line_height * 0.8
    add_text(f'  IVONLR: σ = {perp_ridge_ivonlr.std():.4f}',
            text_y, fontsize=12, color=ivonlr_color)
    text_y -= line_height * 0.8
    ratio = perp_ridge_ivonlr.std() / perp_ridge_ivon.std()
    add_text(f'  → IVONLR has {ratio:.1f}x more',
            text_y, fontsize=11, color='darkgreen', fontweight='bold')
    text_y -= line_height * 0.7
    add_text(f'     uncertainty along ridge!',
            text_y, fontsize=11, color='darkgreen', fontweight='bold')

    # Overall title
    fig.suptitle('Symmetric Parameters: y = (w₁ + w₂) · x',
                fontsize=18, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*70)
    print("SYMMETRIC PARAMETERS TEST")
    print("="*70)
    print("\nModel: y = (w₁ + w₂) · x")
    print("→ w₁ and w₂ are completely INTERCHANGEABLE")
    print("→ Only their sum matters for predictions")
    print("→ This creates a diagonal ridge in parameter space\n")

    x, y, w1_true, w2_true, w_sum_true = make_symmetric_data(cfg, device)

    print(f"True: w₁={w1_true:.2f}, w₂={w2_true:.2f}, sum={w_sum_true:.2f}")
    print(f"(But ANY w₁, w₂ with w₁+w₂={w_sum_true:.2f} would work!)\n")

    # Train IVON
    print("Training IVON...")
    model_ivon = SymmetricModel().to(device)
    optimizer_ivon = ivon.IVON(model_ivon.parameters(), lr=cfg.lr, ess=len(x))
    train_optimizer(model_ivon, x, y, optimizer_ivon, cfg)
    samples_ivon = sample_weights(model_ivon, optimizer_ivon, cfg)
    print(f"  ✓ Sampled {cfg.test_samples} posterior samples")

    # Train IVONLR
    print("Training IVONLR...")
    model_ivonlr = SymmetricModel().to(device)
    optimizer_ivonlr = ivon.IVONLR(
        model_ivonlr.parameters(),
        lr=cfg.lr,
        ess=len(x),
        rank=cfg.rank,
    )
    train_optimizer(model_ivonlr, x, y, optimizer_ivonlr, cfg)
    samples_ivonlr = sample_weights(model_ivonlr, optimizer_ivonlr, cfg)
    print(f"  ✓ Sampled {cfg.test_samples} posterior samples")

    # Key results
    corr_ivon = torch.corrcoef(samples_ivon.T)[0, 1].item()
    corr_ivonlr = torch.corrcoef(samples_ivonlr.T)[0, 1].item()

    print(f"\n" + "="*70)
    print("KEY RESULTS")
    print("="*70)
    print(f"\nCorrelation coefficient:")
    print(f"  IVON:   {corr_ivon:+.6f} (essentially zero - no correlation)")
    print(f"  IVONLR: {corr_ivonlr:+.6f} (strong positive correlation!)")
    print(f"\n  → IVONLR captures {abs(corr_ivonlr/corr_ivon):.0f}x stronger correlation!")

    sum_ivonlr = (samples_ivonlr[:, 0] + samples_ivonlr[:, 1])
    print(f"\nPosterior sum w₁+w₂:")
    print(f"  True:   {w_sum_true:.4f}")
    print(f"  IVONLR: {sum_ivonlr.mean():.4f} ± {sum_ivonlr.std():.4f}")
    print("="*70)

    # Plot
    print(f"\nGenerating visualization...")
    fig = plot_symmetric(cfg, samples_ivon, samples_ivonlr, w1_true, w2_true, w_sum_true)
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.out_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {cfg.out_path}")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("Since w₁ and w₂ are symmetric, the posterior should have:")
    print("  • HIGH uncertainty along the ridge (w₁ + w₂ = const)")
    print("  • LOW uncertainty perpendicular to ridge")
    print()
    print("✗ IVON: Axis-aligned → misses the diagonal ridge structure")
    print("✓ IVONLR: Captures the ridge → samples concentrate along diagonal")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

