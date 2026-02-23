import torch

from ivon._ivon import _welford_mean


def test_welford_matches_batch_mean():
    torch.manual_seed(0)
    xs = [torch.randn(10, dtype=torch.float64) for _ in range(100)]

    avg = None
    for i, x in enumerate(xs, 1):
        avg = _welford_mean(avg, x, i)

    batch = torch.stack(xs).mean(0)
    assert torch.allclose(avg, batch, atol=1e-12, rtol=0.0)
