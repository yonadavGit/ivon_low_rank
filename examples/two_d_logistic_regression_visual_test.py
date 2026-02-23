import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

import ivon


@dataclass
class TrainConfig:
    epochs: int = 2000
    lr: float = 0.2
    train_samples: int = 3
    test_samples: int = 200
    seed: int = 11


def make_data(n_per_class: int, device: torch.device):
    mean_pos = torch.tensor([2.0, 2.0], device=device)
    mean_neg = torch.tensor([-2.0, -1.5], device=device)
    cov = torch.tensor([[1.2, 0.6], [0.6, 1.0]], device=device)
    chol = torch.linalg.cholesky(cov)

    pos = mean_pos + torch.randn(n_per_class, 2, device=device) @ chol.T
    neg = mean_neg + torch.randn(n_per_class, 2, device=device) @ chol.T
    x = torch.cat([pos, neg], dim=0)
    y = torch.cat(
        [
            torch.ones(n_per_class, 1, device=device),
            torch.zeros(n_per_class, 1, device=device),
        ],
        dim=0,
    )
    return x, y


def build_model():
    return nn.Linear(2, 1)


def train_ivon(model, x, y, cfg: TrainConfig, use_low_rank: bool):
    Optimizer = ivon.IVONLR if use_low_rank else ivon.IVON
    optimizer = Optimizer(
        model.parameters(),
        lr=cfg.lr,
        ess=len(x),
    )
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(cfg.epochs):
        for _ in range(cfg.train_samples):
            with optimizer.sampled_params(train=True):
                optimizer.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
        optimizer.step()

    return optimizer


def train_sgd(model, x, y, cfg: TrainConfig):
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(cfg.epochs):
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def predict_ivon(model, optimizer, grid_x, cfg: TrainConfig):
    probs = []
    for _ in range(cfg.test_samples):
        with optimizer.sampled_params():
            logits = model(grid_x)
            probs.append(torch.sigmoid(logits).squeeze(1).cpu())
    probs = torch.stack(probs, dim=0)
    return probs.mean(dim=0), probs.std(dim=0)


@torch.no_grad()
def predict_deterministic(model, grid_x):
    logits = model(grid_x)
    return torch.sigmoid(logits).squeeze(1).cpu()


def plot_results(
    out_path: Path,
    x_train,
    y_train,
    grid_x,
    ivon_mean,
    ivon_std,
    sgd_prob,
):
    import matplotlib.pyplot as plt

    x_min, x_max = grid_x[:, 0].min().item(), grid_x[:, 0].max().item()
    y_min, y_max = grid_x[:, 1].min().item(), grid_x[:, 1].max().item()
    grid_n = int((grid_x.shape[0]) ** 0.5)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)

    for ax, title, values, cmap in [
        (axes[0], "IVON mean prob", ivon_mean, "RdYlGn"),
        (axes[1], "IVON uncertainty", ivon_std, "viridis"),
        (axes[2], "SGD mean prob", sgd_prob, "RdYlGn"),
    ]:
        im = ax.imshow(
            values.reshape(grid_n, grid_n),
            origin="lower",
            extent=(x_min, x_max, y_min, y_max),
            cmap=cmap,
            aspect="auto",
            alpha=0.95,
        )
        ax.scatter(
            x_train[:, 0].cpu(),
            x_train[:, 1].cpu(),
            c=y_train.squeeze(1).cpu(),
            cmap="bwr",
            edgecolor="white",
            linewidth=0.6,
            s=35,
        )
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    fig.suptitle("2-D Logistic Regression: IVON vs SGD")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    out_path = Path("plots/ivon_2d_logreg.png")
    use_low_rank = True
    epochs = 2000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(11)

    cfg = TrainConfig(epochs=epochs)
    x_train, y_train = make_data(n_per_class=120, device=device)

    model_ivon = build_model().to(device)
    optimizer_ivon = train_ivon(
        model_ivon, x_train, y_train, cfg, use_low_rank
    )

    model_sgd = build_model().to(device)
    train_sgd(model_sgd, x_train, y_train, cfg)

    grid_n = 140
    grid_x1 = torch.linspace(-6.0, 6.0, grid_n, device=device)
    grid_x2 = torch.linspace(-6.0, 6.0, grid_n, device=device)
    grid_x1v, grid_x2v = torch.meshgrid(grid_x1, grid_x2, indexing="xy")
    grid_x = torch.stack([grid_x1v.flatten(), grid_x2v.flatten()], dim=1)

    ivon_mean, ivon_std = predict_ivon(
        model_ivon, optimizer_ivon, grid_x, cfg
    )
    sgd_prob = predict_deterministic(model_sgd, grid_x)

    plot_results(
        out_path,
        x_train,
        y_train,
        grid_x,
        ivon_mean,
        ivon_std,
        sgd_prob,
    )


if __name__ == "__main__":
    main()
