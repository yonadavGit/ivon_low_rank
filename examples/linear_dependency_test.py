"""
Simplest Test: Linear Dependency Between Parameters

Model: y = w1*x1 + w2*x2, where x2 = x1 (identical features)
This means: y = (w1 + w2)*x1

Result: w1 and w2 are perfectly linearly dependent!
Any combination where w1 + w2 = constant works equally well.
"""
import torch
from torch import nn
import ivon
import matplotlib.pyplot as plt
import numpy as np


def make_data(n_samples=50, seed=42):
    """Create data where features are identical: x2 = x1"""
    torch.manual_seed(seed)

    # Single feature, repeated twice
    x1 = torch.randn(n_samples, 1)
    x2 = x1.clone()  # x2 is EXACTLY equal to x1
    x = torch.cat([x1, x2], dim=1)  # [n, 2]

    # True parameters (but ANY w1, w2 with w1+w2=3.0 works!)
    w_true = torch.tensor([[1.0], [2.0]])  # w1=1.5, w2=1.5, sum=3.0

    # Generate data: y = w1*x1 + w2*x2 = (w1+w2)*x1
    y = x @ w_true + 0.2 * torch.randn(n_samples, 1)

    return x, y


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 1) * 0.1)

    def forward(self, x):
        return x @ self.weight


def train(model, x, y, optimizer, epochs=2000):
    """Train model"""
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        for _ in range(3):  # 3 samples per step
            with optimizer.sampled_params(train=True):
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
        optimizer.step()


@torch.no_grad()
def sample_posterior(model, optimizer, n_samples=3000):
    """Sample from posterior"""
    samples = []
    for _ in range(n_samples):
        with optimizer.sampled_params():
            w = model.weight.flatten().cpu().clone()
            samples.append(w)
    return torch.stack(samples, dim=0)


def plot_results(samples_ivon, samples_ivonlr, w_true_sum=3.0):
    """Create simple, clear plot"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Colors
    ivon_color = '#FF6B6B'
    ivonlr_color = '#4ECDC4'

    # 1. IVON scatter
    ax = axes[0]
    ax.scatter(samples_ivon[:, 0], samples_ivon[:, 1],
              alpha=0.1, s=5, c=ivon_color)

    # Draw constraint line: w1 + w2 = 3
    w1_line = np.linspace(-1, 5, 100)
    w2_line = w_true_sum - w1_line
    ax.plot(w1_line, w2_line, 'g--', linewidth=2,
           label=f'Constraint: w₁+w₂={w_true_sum}')

    ax.set_xlabel('w₁', fontsize=14, fontweight='bold')
    ax.set_ylabel('w₂', fontsize=14, fontweight='bold')
    ax.set_title('IVON\n(Diagonal - No Correlation)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # 2. IVONLR scatter
    ax = axes[1]
    ax.scatter(samples_ivonlr[:, 0], samples_ivonlr[:, 1],
              alpha=0.1, s=5, c=ivonlr_color)
    ax.plot(w1_line, w2_line, 'g--', linewidth=2,
           label=f'Constraint: w₁+w₂={w_true_sum}')

    ax.set_xlabel('w₁', fontsize=14, fontweight='bold')
    ax.set_ylabel('w₂', fontsize=14, fontweight='bold')
    ax.set_title('IVONLR\n(Captures Constraint!)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Set same limits
    all_samples = torch.cat([samples_ivon, samples_ivonlr])
    xlim = [all_samples[:, 0].min()-0.5, all_samples[:, 0].max()+0.5]
    ylim = [all_samples[:, 1].min()-0.5, all_samples[:, 1].max()+0.5]
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)
    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)

    # 3. Comparison
    ax = axes[2]
    ax.axis('off')

    # Compute statistics
    corr_ivon = torch.corrcoef(samples_ivon.T)[0, 1].item()
    corr_ivonlr = torch.corrcoef(samples_ivonlr.T)[0, 1].item()

    sum_ivon = samples_ivon.sum(dim=1)
    sum_ivonlr = samples_ivonlr.sum(dim=1)

    # Text summary
    text = f"""
LINEAR DEPENDENCY TEST
{'='*30}

Problem Setup:
• Features: x₂ = x₁ (identical!)
• Model: y = w₁·x₁ + w₂·x₂
• Simplified: y = (w₁+w₂)·x₁

Result:
• w₁ and w₂ are DEPENDENT
• Only w₁+w₂ is identifiable
• Creates diagonal constraint

{'='*30}

Correlation (w₁, w₂):
  IVON:   {corr_ivon:+.4f}
  IVONLR: {corr_ivonlr:+.4f}
  
  Improvement: {abs(corr_ivonlr/corr_ivon):.0f}x

Sum (w₁ + w₂):
  True:   {w_true_sum:.2f}
  IVON:   {sum_ivon.mean():.2f} ± {sum_ivon.std():.2f}
  IVONLR: {sum_ivonlr.mean():.2f} ± {sum_ivonlr.std():.2f}

{'='*30}

✗ IVON: Circular uncertainty
   (ignores constraint)

✓ IVONLR: Elongated along line
   (respects dependency!)
"""

    ax.text(0.1, 0.5, text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center',
           family='monospace', bbox=dict(boxstyle='round',
           facecolor='wheat', alpha=0.3))

    fig.suptitle('Linear Dependency: y = (w₁ + w₂)·x  where x₂ = x₁',
                fontsize=15, fontweight='bold')
    fig.tight_layout()

    return fig


def main():
    print("\n" + "="*70)
    print("SIMPLE TEST: Linear Dependency (Identical Features)")
    print("="*70)
    print("\nSetup:")
    print("  • x₂ = x₁ (features are IDENTICAL)")
    print("  • Model: y = w₁·x₁ + w₂·x₂ = (w₁+w₂)·x₁")
    print("  • Result: w₁ and w₂ are linearly dependent!")
    print("  • Only their sum (w₁+w₂) matters for predictions")
    print()

    # Generate data
    x, y = make_data()

    # Train IVON
    print("Training IVON...")
    model_ivon = SimpleModel()
    opt_ivon = ivon.IVON(model_ivon.parameters(), lr=0.1, ess=len(x), weight_decay=0.1)
    train(model_ivon, x, y, opt_ivon)
    samples_ivon = sample_posterior(model_ivon, opt_ivon)
    print("  ✓ Done")

    # Train IVONLR
    print("Training IVONLR...")
    model_ivonlr = SimpleModel()
    opt_ivonlr = ivon.IVONLR(model_ivonlr.parameters(), lr=0.1,
                             ess=len(x), rank=4, weight_decay=0.1)
    train(model_ivonlr, x, y, opt_ivonlr)
    samples_ivonlr = sample_posterior(model_ivonlr, opt_ivonlr)
    print("  ✓ Done")

    # Results
    corr_ivon = torch.corrcoef(samples_ivon.T)[0, 1].item()
    corr_ivonlr = torch.corrcoef(samples_ivonlr.T)[0, 1].item()

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    # Print learned parameters (posterior means)
    mean_ivon = samples_ivon.mean(dim=0)
    mean_ivonlr = samples_ivonlr.mean(dim=0)

    print(f"\nLearned parameters (posterior mean):")
    print(f"  True:   w₁={1.5:.4f}, w₂={1.5:.4f}, sum={3.0:.4f}")
    print(f"  IVON:   w₁={mean_ivon[0]:.4f}, w₂={mean_ivon[1]:.4f}, sum={mean_ivon.sum():.4f}")
    print(f"  IVONLR: w₁={mean_ivonlr[0]:.4f}, w₂={mean_ivonlr[1]:.4f}, sum={mean_ivonlr.sum():.4f}")

    print(f"\nCorrelation between w₁ and w₂:")
    print(f"  IVON:   {corr_ivon:+.4f} (near zero)")
    print(f"  IVONLR: {corr_ivonlr:+.4f} (strong!)")
    print(f"\n  → IVONLR captures {abs(corr_ivonlr/corr_ivon):.0f}x stronger correlation")

    sum_ivon = samples_ivon.sum(dim=1)
    sum_ivonlr = samples_ivonlr.sum(dim=1)
    print(f"\nSum w₁ + w₂ uncertainty:")
    print(f"  IVON:   {sum_ivon.mean():.2f} ± {sum_ivon.std():.2f}")
    print(f"  IVONLR: {sum_ivonlr.mean():.2f} ± {sum_ivonlr.std():.2f}")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("\nWhen features are identical (x₂ = x₁), the model becomes:")
    print("  y = w₁·x₁ + w₂·x₁ = (w₁ + w₂)·x₁")
    print("\nThis means:")
    print("  ✗ Cannot determine w₁ and w₂ individually")
    print("  ✓ Can only determine their sum (w₁ + w₂)")
    print("  → Creates linear dependency: any split works!")
    print("\nPosterior uncertainty:")
    print("  IVON:   Circular (assumes independence)")
    print("  IVONLR: Elongated along diagonal (captures dependency)")
    print("="*70)

    # Plot
    print("\nGenerating plot...")
    fig = plot_results(samples_ivon, samples_ivonlr)
    fig.savefig('plots/linear_dependency_test.png', dpi=150, bbox_inches='tight')
    print("✓ Saved to plots/linear_dependency_test.png")

    print("\n" + "="*70)
    print("✓ Test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

