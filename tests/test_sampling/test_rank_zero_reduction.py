import torch

from tests._test_utils import empirical_cov, sample_reparam


def test_rank_zero_equals_diagonal_sampler():
    torch.manual_seed(0)
    d, n = 20, 50_000

    mu = torch.zeros(d, dtype=torch.float64)
    D = torch.exp(torch.randn(d, dtype=torch.float64)) + 0.5

    samples_lr = sample_reparam(mu, D, U=None, s=None, n=n)
    samples_diag = mu + torch.randn(n, d, dtype=torch.float64) * D.rsqrt()

    cov_lr = empirical_cov(samples_lr)
    cov_diag = empirical_cov(samples_diag)

    # Monte Carlo tolerance on covariance estimates.
    assert torch.allclose(cov_lr, cov_diag, atol=0.05, rtol=0.0)
