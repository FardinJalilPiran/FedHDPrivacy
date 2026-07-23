"""Tests for the hyperdimensional model and the encoder."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fedhdprivacy.hdc import Encoder, HDClassifier, average, binarize


def test_binarize_produces_only_plus_minus_one():
    x = torch.tensor([-2.0, -0.1, 0.0, 0.1, 3.0])
    out = binarize(x)
    assert set(out.unique().tolist()) <= {-1.0, 1.0}
    # Exact zeros must not stay zero, or the hypervector loses a dimension.
    assert out[2].item() == 1.0


def test_encoder_output_shape_and_values():
    encoder = Encoder(n_features=16, dimensions=256, seed=0)
    x = torch.rand(32, 16)
    encoded = encoder(x)
    assert encoded.shape == (32, 256)
    assert set(encoded.unique().tolist()) <= {-1.0, 1.0}


def test_encoder_accepts_a_single_sample():
    encoder = Encoder(n_features=8, dimensions=128, seed=0)
    assert encoder(torch.rand(8)).shape == (1, 128)


def test_encoder_is_deterministic_for_the_same_input():
    encoder = Encoder(n_features=8, dimensions=128, seed=1)
    x = torch.rand(5, 8)
    assert torch.equal(encoder(x), encoder(x))


def test_batched_encoding_matches_single_pass():
    encoder = Encoder(n_features=8, dimensions=128, seed=2)
    x = torch.rand(50, 8)
    assert torch.equal(encoder.encode(x, batch_size=7), encoder(x))


def test_similar_inputs_encode_to_similar_hypervectors():
    """Density encoding should preserve locality, which is what makes HD work."""
    encoder = Encoder(n_features=32, dimensions=4096, seed=3)
    anchor = torch.full((1, 32), 0.5)
    near = anchor + 0.01
    far = torch.full((1, 32), 0.99)

    encoded = encoder(torch.cat([anchor, near, far]))
    similarity_near = torch.dot(encoded[0], encoded[1]) / 4096
    similarity_far = torch.dot(encoded[0], encoded[2]) / 4096
    assert similarity_near > similarity_far


def test_bundle_accumulates_into_the_right_class():
    model = HDClassifier.zeros(n_classes=3, dimensions=8)
    encoded = torch.ones((4, 8))
    labels = torch.tensor([0, 0, 1, 2])
    model.bundle(encoded, labels)
    assert model.class_hypervectors[0].sum().item() == 16  # two samples of eight ones
    assert model.class_hypervectors[1].sum().item() == 8
    assert model.class_hypervectors[2].sum().item() == 8


def test_model_learns_a_separable_problem():
    torch.manual_seed(0)
    encoder = Encoder(n_features=10, dimensions=2048, seed=0)
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.uniform(0.0, 0.3, (60, 10)), rng.uniform(0.7, 1.0, (60, 10))])
    y = np.array([0] * 60 + [1] * 60)

    encoded = encoder(torch.from_numpy(x.astype(np.float32)))
    labels = torch.from_numpy(y).long()

    model = HDClassifier.zeros(2, 2048)
    model.bundle(encoded, labels)
    assert model.accuracy(encoded, labels) > 0.9


def test_retraining_does_not_hurt_a_learnable_problem():
    torch.manual_seed(0)
    encoder = Encoder(n_features=12, dimensions=2048, seed=1)
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.uniform(0.0, 0.45, (80, 12)), rng.uniform(0.55, 1.0, (80, 12))])
    y = np.array([0] * 80 + [1] * 80)

    encoded = encoder(torch.from_numpy(x.astype(np.float32)))
    labels = torch.from_numpy(y).long()

    model = HDClassifier.zeros(2, 2048)
    model.bundle(encoded, labels)
    before = model.accuracy(encoded, labels)
    model.retrain(encoded, labels, n_epochs=5, generator=torch.Generator().manual_seed(0))
    assert model.accuracy(encoded, labels) >= before - 1e-9


def test_predict_returns_one_label_per_sample():
    model = HDClassifier.zeros(4, 64)
    model.class_hypervectors[2] += 1.0
    predictions = model.predict(torch.ones((7, 64)))
    assert predictions.shape == (7,)
    assert torch.all(predictions == 2)


def test_average_is_the_elementwise_mean():
    a = HDClassifier(torch.zeros((2, 4)))
    b = HDClassifier(torch.ones((2, 4)) * 4)
    assert torch.allclose(average([a, b]).class_hypervectors, torch.full((2, 4), 2.0))


def test_average_rejects_an_empty_list():
    with pytest.raises(ValueError):
        average([])


def test_clone_is_independent():
    model = HDClassifier.zeros(2, 8)
    copy = model.clone()
    copy.class_hypervectors += 1
    assert model.class_hypervectors.sum().item() == 0


# ------------------------------------------------------------------ backends


def test_unknown_encoder_backend_is_rejected():
    with pytest.raises(ValueError):
        Encoder(n_features=4, dimensions=32, backend="nonsense")


@pytest.mark.parametrize("backend", ["projection", "density"])
def test_both_backends_produce_bipolar_hypervectors(backend):
    encoder = Encoder(n_features=8, dimensions=256, backend=backend, seed=0)
    encoded = encoder(torch.rand(10, 8))
    assert encoded.shape == (10, 256)
    assert set(encoded.unique().tolist()) <= {-1.0, 1.0}


@pytest.mark.parametrize("backend", ["projection", "density"])
def test_both_backends_preserve_locality(backend):
    encoder = Encoder(n_features=16, dimensions=4096, backend=backend, seed=0)
    anchor = torch.full((1, 16), 0.5)
    encoded = encoder(torch.cat([anchor, anchor + 0.01, torch.full((1, 16), 0.99)]))
    near = torch.dot(encoded[0], encoded[1])
    far = torch.dot(encoded[0], encoded[2])
    assert near > far


def test_density_backend_caps_its_batch_size():
    """High-dimensional inputs must not blow up memory in the density encoder."""
    encoder = Encoder(n_features=512, dimensions=10000, backend="density", seed=0)
    if encoder.backend == "density":  # skip silently if torchhd is absent
        assert encoder._max_batch < 512


def test_projection_backend_is_not_batch_limited():
    encoder = Encoder(n_features=512, dimensions=10000, backend="projection", seed=0)
    assert encoder._max_batch >= 1024
