import torch

from tests._test_utils import empirical_cov, frob, implied_covariance, orthonormal_matrix, sample_reparam


def test_covariance_matches_theory():
    torch.manual_seed(0)
    d, r, n = 10, 3, 50_000

    mu = torch.zeros(d, dtype=torch.float64)
    D = torch.exp(torch.randn(d, dtype=torch.float64)) + 0.5
    U = orthonormal_matrix(d, r)
    s = torch.rand(r, dtype=torch.float64)

    sigma = implied_covariance(D, U, s)
    samples = sample_reparam(mu, D, U, s, n)

    emp_cov = empirical_cov(samples)
    rel_err = (frob(emp_cov - sigma) / frob(sigma)).item()

    assert rel_err < 0.05
