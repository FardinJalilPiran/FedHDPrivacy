"""Device-placement regression tests.

The bug these guard against: PyTorch requires a random generator to live on the
same device as the tensor it fills. A seeded CPU generator paired with a CUDA
model raised

    RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'

The CUDA cases are skipped on machines without a GPU, so run this suite on the
GPU box too - a CPU-only run cannot prove the fix.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fedhdprivacy import ExperimentConfig, load_dataset, run_federated_training
from fedhdprivacy.hdc import HDClassifier, build_encoder
from fedhdprivacy.privacy import GaussianNoiseMechanism
from fedhdprivacy.utils import resolve_device

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")

MECHANISM = dict(dimensions=512, epsilon=10.0, n_clients=4, samples_per_round=50)


# ------------------------------------------------------- generator vs. device


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
def test_perturb_accepts_a_cpu_generator_on_any_device(device):
    """A host-side seeded generator must work against a device-side model."""
    mechanism = GaussianNoiseMechanism(**MECHANISM)
    model = torch.zeros((3, MECHANISM["dimensions"]), device=device)
    noisy = mechanism.perturb(model, round_index=2, generator=torch.Generator().manual_seed(0))
    assert noisy.device.type == torch.device(device).type
    assert not torch.allclose(noisy, model)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
def test_perturb_without_a_generator_stays_on_device(device):
    mechanism = GaussianNoiseMechanism(**MECHANISM)
    model = torch.zeros((3, MECHANISM["dimensions"]), device=device)
    assert mechanism.perturb(model, round_index=1).device.type == torch.device(device).type


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
def test_retrain_accepts_a_cpu_generator_on_any_device(device):
    encoded = torch.sign(torch.randn(24, 128)).to(device)
    labels = torch.randint(0, 3, (24,)).to(device)
    model = HDClassifier.zeros(3, 128, device=device)
    model.bundle(encoded, labels)
    model.retrain(encoded, labels, n_epochs=2, generator=torch.Generator().manual_seed(0))
    assert model.class_hypervectors.device.type == torch.device(device).type


@requires_cuda
def test_seeded_noise_is_identical_on_cpu_and_cuda():
    """Drawing on the generator's device keeps seeded runs comparable."""
    mechanism = GaussianNoiseMechanism(**MECHANISM)
    shape = (3, MECHANISM["dimensions"])
    on_cpu = mechanism.perturb(
        torch.zeros(shape), round_index=2, generator=torch.Generator().manual_seed(7)
    )
    on_gpu = mechanism.perturb(
        torch.zeros(shape, device="cuda"), round_index=2, generator=torch.Generator().manual_seed(7)
    )
    assert torch.allclose(on_cpu, on_gpu.cpu(), atol=1e-5)


# ---------------------------------------------------------------- encoder


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
@pytest.mark.parametrize("backend", ["projection", "density"])
def test_encoder_returns_tensors_on_the_requested_device(device, backend):
    encoder = build_encoder(n_features=16, dimensions=256, backend=backend, device=device, seed=0)
    encoded = encoder.encode(torch.rand(20, 16))
    assert encoded.device.type == torch.device(device).type
    assert set(encoded.unique().cpu().tolist()) <= {-1.0, 1.0}


# ------------------------------------------------------------------ end to end


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
def test_full_run_on_each_device(device):
    """The failure the user hit was only reachable through a full run."""
    dataset = load_dataset("synthetic", n_clients=4, partition="iid", seed=0)
    config = ExperimentConfig(
        dataset="synthetic",
        n_clients=4,
        rounds=3,
        dimensions=1000,
        local_epochs=2,
        epsilon=10.0,
        device=device,
        seed=0,
    )
    history = run_federated_training(dataset, config, progress=False)
    assert len(history.rounds) == 3
    assert np.isfinite(history.final_report.accuracy)


def test_resolve_device_falls_back_when_cuda_is_absent():
    resolved = resolve_device("cuda")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolved.type == expected


def test_resolve_device_auto():
    assert resolve_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")
