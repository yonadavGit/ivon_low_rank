import torch

from ivon._ivon import IVONLR
from tests._test_utils import orthonormal_matrix


def test_oja_learns_dominant_direction():
    torch.manual_seed(0)
    d, r = 20, 3

    true_dir = torch.randn(d, dtype=torch.float64)
    true_dir /= true_dir.norm()

    param = torch.nn.Parameter(torch.zeros(d, dtype=torch.float64))
    opt = IVONLR(
        [param],
        lr=1e-2,
        ess=100.0,
        rank=r,
        orth_every=0,
        eta_u=0.2,
        beta3=0.9,
    )

    group = opt.param_groups[0]
    group["U"] = orthonormal_matrix(d, r)

    for _ in range(200):
        g = true_dir + 0.05 * torch.randn(d, dtype=torch.float64)
        opt._update_low_rank_sample(group, g_w=g, alpha=torch.tensor(1.0, dtype=torch.float64))

    alignment = torch.abs(group["U"].t() @ true_dir).max().item()
    assert alignment > 0.8
