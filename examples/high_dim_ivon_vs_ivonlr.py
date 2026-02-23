from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

import ivon


@dataclass
class Config:
    seed: int = 17
    n_samples: int = 512
    n_features: int = 80
    noise_std: float = 0.2
    epochs: int = 2000
    lr: float = 0.15
    train_samples: int = 3
    test_samples: int = 400
    rank: int = 8
    plot_dims: int = 40
    out_path: Path = Path("plots/ivon_vs_ivonlr_cov.png")


def make_data(cfg: Config, device: torch.device):
    torch.manual_seed(cfg.seed)
    base = torch.randn(cfg.n_features, cfg.n_features, device=device)
    cov = base @ base.T / cfg.n_features
    chol = torch.linalg.cholesky(cov + 1e-3 * torch.eye(cfg.n_features, device=device))
    x = torch.randn(cfg.n_samples, cfg.n_features, device=device) @ chol.T

    w_true = torch.randn(cfg.n_features, 1, device=device)
    y = x @ w_true + cfg.noise_std * torch.randn(cfg.n_samples, 1, device=device)
    return x, y


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


def plot_covariances(out_path: Path, cov_ivon, cov_ivonlr, cfg: Config):
    import matplotlib.pyplot as plt

    dims = min(cfg.plot_dims, cov_ivon.shape[0])
    cov_ivon = cov_ivon[:dims, :dims]
    cov_ivonlr = cov_ivonlr[:dims, :dims]

    vmax = max(cov_ivon.abs().max().item(), cov_ivonlr.abs().max().item())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for ax, title, cov in [
        (axes[0], "IVON weight covariance", cov_ivon),
        (axes[1], "IVONLR weight covariance", cov_ivonlr),
    ]:
        im = ax.imshow(cov, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("weight index")
        ax.set_ylabel("weight index")

    fig.suptitle("High-D Regression: Diagonal vs Low-Rank Covariance")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def summarize_offdiag(cov: torch.Tensor):
    offdiag = cov - torch.diag(torch.diag(cov))
    return offdiag.abs().mean().item()


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = make_data(cfg, device)

    model_ivon = nn.Linear(cfg.n_features, 1, bias=False).to(device)
    optimizer_ivon = ivon.IVON(model_ivon.parameters(), lr=cfg.lr, ess=len(x))
    train_optimizer(model_ivon, x, y, optimizer_ivon, cfg)
    samples_ivon = sample_weights(model_ivon, optimizer_ivon, cfg)
    cov_ivon = covariance_from_samples(samples_ivon)

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

    off_ivon = summarize_offdiag(cov_ivon)
    off_ivonlr = summarize_offdiag(cov_ivonlr)
    print(f"Mean |off-diagonal| covariance: IVON={off_ivon:.6f}, IVONLR={off_ivonlr:.6f}")

    plot_covariances(cfg.out_path, cov_ivon, cov_ivonlr, cfg)


if __name__ == "__main__":
    main()
