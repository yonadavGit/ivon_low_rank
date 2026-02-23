import torch

from tests._test_utils import orthonormal_matrix


def test_shrink_subspace_scaling():
    torch.manual_seed(0)
    d, r = 20, 5
    U = orthonormal_matrix(d, r)
    s = torch.rand(r, dtype=U.dtype)

    shrink = 1.0 - (1.0 + s).rsqrt()
    B = torch.eye(d, dtype=U.dtype) - U @ torch.diag(shrink) @ U.t()

    c = torch.randn(r, dtype=U.dtype)
    z = U @ c

    out = B @ z
    expected = U @ (c * (1.0 + s).rsqrt())

    assert torch.allclose(out, expected, atol=1e-8, rtol=1e-6)


def test_shrink_orthogonal_complement_unchanged():
    torch.manual_seed(1)
    d, r = 20, 5
    U = orthonormal_matrix(d, r)
    s = torch.rand(r, dtype=U.dtype)

    shrink = 1.0 - (1.0 + s).rsqrt()
    B = torch.eye(d, dtype=U.dtype) - U @ torch.diag(shrink) @ U.t()

    x = torch.randn(d, dtype=U.dtype)
    # Project onto orthogonal complement of span(U).
    z = x - U @ (U.t() @ x)

    out = B @ z
    assert torch.allclose(out, z, atol=1e-8, rtol=1e-6)
