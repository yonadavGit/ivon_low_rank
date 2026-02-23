import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

import ivon


@dataclass
class TrainConfig:
    epochs: int = 3000
    lr: float = 0.08
    train_samples: int = 3
    test_samples: int = 200
    hidden: int = 64
    seed: int = 7


def make_data(n_per_side: int, noise_std: float, device: torch.device):
    left = torch.linspace(-4.0, -2.0, n_per_side, device=device)
    right = torch.linspace(2.0, 4.0, n_per_side, device=device)
    x = torch.cat([left, right]).unsqueeze(1)
    y = (x ** 3) / 16.0 + 0.3 * torch.sin(3.0 * x)
    y = y + noise_std * torch.randn_like(y)
    return x, y


def build_model(hidden: int):
    return nn.Sequential(
        nn.Linear(1, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, 1),
    )


def train_ivon(model, x, y, cfg: TrainConfig):
    optimizer = ivon.IVON(model.parameters(), lr=cfg.lr, ess=len(x))
    loss_fn = nn.MSELoss()

    for _ in range(cfg.epochs):
        for _ in range(cfg.train_samples):
            with optimizer.sampled_params(train=True):
                optimizer.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
        optimizer.step()

    return optimizer


def train_adamw(model, x, y, cfg: TrainConfig):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    for _ in range(cfg.epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def predict_ivon(model, optimizer, grid_x, cfg: TrainConfig):
    preds = []
    for _ in range(cfg.test_samples):
        with optimizer.sampled_params():
            preds.append(model(grid_x).squeeze(1).cpu())
    preds = torch.stack(preds, dim=0)
    mean = preds.mean(dim=0)
    std = preds.std(dim=0)
    return mean, std


@torch.no_grad()
def predict_deterministic(model, grid_x):
    return model(grid_x).squeeze(1).cpu()


def plot_results(
    out_path: Path,
    x_train,
    y_train,
    grid_x,
    true_y,
    ivon_mean,
    ivon_std,
    adam_pred,
):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    plt.scatter(
        x_train.cpu(),
        y_train.cpu(),
        s=35,
        color="#2A6F97",
        alpha=0.85,
        label="Data",
        zorder=5,
    )
    plt.plot(
        grid_x.cpu(),
        true_y.cpu(),
        color="#111111",
        linewidth=1.2,
        label="Ground truth",
    )
    plt.plot(
        grid_x.cpu(),
        adam_pred,
        color="#9A031E",
        linewidth=1.6,
        label="AdamW mean",
    )
    plt.plot(
        grid_x.cpu(),
        ivon_mean,
        color="#386641",
        linewidth=2.2,
        label="IVON mean",
    )
    plt.fill_between(
        grid_x.cpu().squeeze(1),
        (ivon_mean - 2.0 * ivon_std),
        (ivon_mean + 2.0 * ivon_std),
        color="#6A994E",
        alpha=0.25,
        label="IVON ±2 std",
    )
    plt.title("1-D Regression: IVON Uncertainty vs AdamW")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visual 1-D regression test comparing IVON and AdamW."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("plots/ivon_1d_regression.png"),
        help="Output path for the plot.",
    )
    parser.add_argument("--epochs", type=int, default=3000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)

    cfg = TrainConfig(epochs=args.epochs)
    x_train, y_train = make_data(n_per_side=24, noise_std=0.1, device=device)

    model_ivon = build_model(cfg.hidden).to(device)
    optimizer_ivon = train_ivon(model_ivon, x_train, y_train, cfg)

    model_adam = build_model(cfg.hidden).to(device)
    train_adamw(model_adam, x_train, y_train, cfg)

    grid_x = torch.linspace(-6.0, 6.0, 240, device=device).unsqueeze(1)
    true_y = (grid_x ** 3) / 16.0 + 0.3 * torch.sin(3.0 * grid_x)

    ivon_mean, ivon_std = predict_ivon(
        model_ivon, optimizer_ivon, grid_x, cfg
    )
    adam_pred = predict_deterministic(model_adam, grid_x)

    plot_results(
        args.out,
        x_train,
        y_train,
        grid_x,
        true_y.squeeze(1),
        ivon_mean,
        ivon_std,
        adam_pred,
    )


if __name__ == "__main__":
    main()
