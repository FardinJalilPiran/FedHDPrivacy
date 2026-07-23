"""Tests for the differential privacy accounting."""

from __future__ import annotations

import math

import pytest
import torch

from fedhdprivacy.privacy import (
    GaussianNoiseMechanism,
    additional_noise_variance,
    cumulative_noise_variance,
    required_noise_variance,
)

BASE = dict(dimensions=10000, epsilon=10.0, n_clients=8, samples_per_round=1000)


def test_first_round_matches_theorem_2():
    expected = (2 * 10000 / 100.0) * math.log(1.25 * 1000)
    assert required_noise_variance(**BASE, round_index=1) == pytest.approx(expected)


def test_no_noise_inherited_in_first_round():
    assert cumulative_noise_variance(**BASE, round_index=1) == 0.0
    # With nothing to inherit, the whole requirement must be injected.
    assert additional_noise_variance(**BASE, round_index=1) == pytest.approx(
        required_noise_variance(**BASE, round_index=1)
    )


def test_second_round_matches_theorem_4():
    d, eps, k, ell = 10000, 10.0, 8, 1000
    required = (2 * d / eps**2) * math.log(1.25 * (k * ell) + 1.25 * ell)
    inherited = (2 * d / (k * eps**2)) * math.log(1.25 * ell)
    assert required_noise_variance(**BASE, round_index=2) == pytest.approx(required)
    assert cumulative_noise_variance(**BASE, round_index=2) == pytest.approx(inherited)
    assert additional_noise_variance(**BASE, round_index=2) == pytest.approx(required - inherited)


@pytest.mark.parametrize("round_index", range(1, 51))
def test_added_noise_never_exceeds_requirement(round_index):
    """The adaptive mechanism must only ever inject less than the naive bound."""
    added = additional_noise_variance(**BASE, round_index=round_index)
    required = required_noise_variance(**BASE, round_index=round_index)
    assert 0.0 <= added <= required


@pytest.mark.parametrize("round_index", range(2, 51))
def test_adaptive_mechanism_saves_noise(round_index):
    """From round 2 on there is inherited noise, so strictly less is added."""
    added = additional_noise_variance(**BASE, round_index=round_index)
    required = required_noise_variance(**BASE, round_index=round_index)
    assert added < required


def test_required_noise_grows_with_rounds():
    values = [required_noise_variance(**BASE, round_index=r) for r in range(1, 20)]
    assert values == sorted(values)


def test_noise_scales_inversely_with_epsilon():
    tight = required_noise_variance(
        dimensions=1000, epsilon=1.0, n_clients=4, samples_per_round=100, round_index=1
    )
    loose = required_noise_variance(
        dimensions=1000, epsilon=10.0, n_clients=4, samples_per_round=100, round_index=1
    )
    assert tight == pytest.approx(loose * 100)


def test_noise_scales_linearly_with_dimensions():
    small = required_noise_variance(
        dimensions=1000, epsilon=5.0, n_clients=4, samples_per_round=100, round_index=1
    )
    large = required_noise_variance(
        dimensions=2000, epsilon=5.0, n_clients=4, samples_per_round=100, round_index=1
    )
    assert large == pytest.approx(small * 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(dimensions=0, epsilon=1.0, n_clients=2, samples_per_round=10),
        dict(dimensions=10, epsilon=0.0, n_clients=2, samples_per_round=10),
        dict(dimensions=10, epsilon=1.0, n_clients=0, samples_per_round=10),
        dict(dimensions=10, epsilon=1.0, n_clients=2, samples_per_round=0),
    ],
)
def test_invalid_arguments_rejected(kwargs):
    with pytest.raises(ValueError):
        required_noise_variance(**kwargs, round_index=1)


def test_round_index_is_one_based():
    with pytest.raises(ValueError):
        required_noise_variance(**BASE, round_index=0)


def test_perturb_changes_the_model_and_preserves_shape():
    mechanism = GaussianNoiseMechanism(**BASE)
    model = torch.zeros((3, BASE["dimensions"]))
    noisy = mechanism.perturb(model, round_index=1)
    assert noisy.shape == model.shape
    assert not torch.allclose(noisy, model)


def test_perturb_draws_the_prescribed_variance():
    mechanism = GaussianNoiseMechanism(**BASE)
    model = torch.zeros((4, BASE["dimensions"]), dtype=torch.float64)
    noisy = mechanism.perturb(model, round_index=3)
    expected = additional_noise_variance(**BASE, round_index=3)
    # Sampling error over 40k draws is small; 5% tolerance is comfortable.
    assert noisy.var().item() == pytest.approx(expected, rel=0.05)


def test_disabled_mechanism_is_a_no_op():
    mechanism = GaussianNoiseMechanism(**BASE, enabled=False)
    model = torch.randn((3, 128))
    assert torch.equal(mechanism.perturb(model, round_index=1), model)
    assert mechanism.variances(1) == {"required": 0.0, "cumulative": 0.0, "added": 0.0}


def test_perturb_is_reproducible_with_a_seeded_generator():
    mechanism = GaussianNoiseMechanism(**BASE)
    model = torch.zeros((2, 512))
    first = mechanism.perturb(model, 2, generator=torch.Generator().manual_seed(0))
    second = mechanism.perturb(model, 2, generator=torch.Generator().manual_seed(0))
    assert torch.equal(first, second)


# ------------------------------------------------------- device / generator


def test_randn_is_never_called_with_a_mismatched_generator(monkeypatch):
    """Regression test for the CUDA crash.

    PyTorch rejects ``torch.randn(device=X, generator=<generator on Y>)`` when
    X != Y. That only fires on a machine with a GPU, so instead of requiring
    one, put the model on the ``meta`` device: it differs from the generator's
    CPU device without needing real hardware. The buggy implementation sampled
    on the *model's* device and would fail this; the fix samples on the
    generator's device and moves the result.
    """
    seen = []
    real_randn = torch.randn

    def spy(*args, **kwargs):
        generator = kwargs.get("generator")
        if generator is not None:
            seen.append((kwargs.get("device"), generator.device))
        return real_randn(*args, **kwargs)

    monkeypatch.setattr(torch, "randn", spy)

    mechanism = GaussianNoiseMechanism(**BASE)
    model = torch.zeros((3, 128), device="meta")
    noisy = mechanism.perturb(model, round_index=2, generator=torch.Generator().manual_seed(0))

    assert seen, "perturb did not sample any noise"
    for device, generator_device in seen:
        assert torch.device(device) == generator_device
    # and the caller still gets its tensor back on the device it asked for
    assert noisy.device.type == "meta"


def test_perturb_result_lands_on_the_model_device():
    mechanism = GaussianNoiseMechanism(**BASE)
    model = torch.zeros((3, 128))
    noisy = mechanism.perturb(model, 2, generator=torch.Generator().manual_seed(0))
    assert noisy.device == model.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_perturb_works_on_cuda_with_a_cpu_generator():
    mechanism = GaussianNoiseMechanism(**BASE)
    model = torch.zeros((3, 128), device="cuda")
    noisy = mechanism.perturb(model, 2, generator=torch.Generator().manual_seed(0))
    assert noisy.device.type == "cuda"
    assert not torch.allclose(noisy, model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_cpu_and_cuda_draw_identical_noise_for_the_same_seed():
    mechanism = GaussianNoiseMechanism(**BASE)
    on_cpu = mechanism.perturb(torch.zeros((3, 128)), 2, generator=torch.Generator().manual_seed(7))
    on_gpu = mechanism.perturb(
        torch.zeros((3, 128), device="cuda"), 2, generator=torch.Generator().manual_seed(7)
    )
    assert torch.allclose(on_cpu, on_gpu.cpu())
