"""Tests for reward_model normalizer components."""

import torch

from reward_model import MinMaxNormalizer, ProprioNormalizer


def test_minmax_roundtrip() -> None:
    norm = MinMaxNormalizer(min_val=0.0, max_val=6.0)
    raw = torch.tensor([0.0, 3.0, 6.0])
    encoded = norm.normalize(raw)
    decoded = norm.unnormalize(encoded)
    assert torch.allclose(raw, decoded, atol=1e-5)


def test_minmax_range() -> None:
    norm = MinMaxNormalizer(min_val=-1.0, max_val=1.0)
    raw = torch.tensor([-1.0, 0.0, 1.0])
    encoded = norm.normalize(raw)
    assert encoded.min() >= -1.0
    assert encoded.max() <= 1.0


def test_minmax_clamping() -> None:
    norm = MinMaxNormalizer(min_val=0.0, max_val=1.0)
    out_of_range = torch.tensor([-5.0, 10.0])
    encoded = norm.normalize(out_of_range)
    assert encoded[0] == -1.0
    assert encoded[1] == 1.0


def test_minmax_equal_min_max() -> None:
    """When min_val == max_val, normalize should not crash (division guarded by eps)."""
    norm = MinMaxNormalizer(min_val=5.0, max_val=5.0)
    x = torch.tensor([5.0, 6.0, 4.0])
    encoded = norm.normalize(x)
    assert torch.all(torch.isfinite(encoded))
    assert encoded.min() >= -1.0
    assert encoded.max() <= 1.0


def test_proprio_normalizer() -> None:
    norm = ProprioNormalizer(dim=3)
    mean = torch.tensor([1.0, 2.0, 3.0])
    std = torch.tensor([0.5, 0.5, 0.5])
    norm.set_stats(mean, std)

    x = torch.tensor([1.0, 2.0, 3.0])
    out = norm(x)
    assert torch.allclose(out, torch.zeros(3), atol=1e-5)


def test_proprio_normalizer_nonzero() -> None:
    norm = ProprioNormalizer(dim=2)
    mean = torch.tensor([0.0, 0.0])
    std = torch.tensor([2.0, 4.0])
    norm.set_stats(mean, std)

    x = torch.tensor([2.0, 8.0])
    out = norm(x)
    assert torch.allclose(out, torch.tensor([1.0, 2.0]), atol=1e-5)


def test_proprio_normalizer_zero_std() -> None:
    """Zero std should not produce NaN/inf due to eps guard."""
    norm = ProprioNormalizer(dim=3)
    mean = torch.tensor([1.0, 2.0, 3.0])
    std = torch.tensor([0.0, 0.0, 0.0])
    norm.set_stats(mean, std)

    x = torch.tensor([1.0, 2.0, 3.0])
    out = norm(x)
    assert torch.all(torch.isfinite(out))
