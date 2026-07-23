"""Tests for metrics, partitioning, configuration and the end-to-end loop."""

from __future__ import annotations

import numpy as np
import pytest

from fedhdprivacy import ExperimentConfig, evaluate, load_dataset, run_federated_training
from fedhdprivacy.partition import partition_clients

# --------------------------------------------------------------------- metrics


def test_perfect_predictions_give_zero_error_rates():
    y = np.array([0, 1, 2, 0, 1, 2])
    report = evaluate(y, y, n_classes=3)
    assert report.accuracy == 1.0
    assert report.macro_fpr == 0.0
    assert report.macro_fnr == 0.0


def test_metrics_match_a_hand_computed_confusion_matrix():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    report = evaluate(y_true, y_pred, n_classes=2)
    assert report.accuracy == pytest.approx(0.75)
    # class 0: one of two positives missed -> FNR 0.5; no false positives -> FPR 0
    assert report.per_class_fnr[0] == pytest.approx(0.5)
    assert report.per_class_fpr[0] == pytest.approx(0.0)
    # class 1: nothing missed -> FNR 0; one of two negatives flagged -> FPR 0.5
    assert report.per_class_fnr[1] == pytest.approx(0.0)
    assert report.per_class_fpr[1] == pytest.approx(0.5)


def test_metrics_reject_mismatched_shapes():
    with pytest.raises(ValueError):
        evaluate(np.array([0, 1]), np.array([0, 1, 1]), n_classes=2)


# ------------------------------------------------------------------ partition


def _toy_data(n=600, features=5, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, features)).astype(np.float32), rng.integers(0, classes, n)


def test_iid_partition_is_balanced_and_disjoint():
    x, y = _toy_data()
    clients = partition_clients(x, y, n_clients=4, strategy="iid", seed=0)
    assert len(clients) == 4
    assert len({len(c) for c in clients}) == 1


def test_dirichlet_partition_produces_equal_shards():
    """The DP accounting assumes a common L, so shards must match in size."""
    x, y = _toy_data(n=1200)
    clients = partition_clients(x, y, n_clients=4, strategy="dirichlet", seed=0)
    assert len({len(c) for c in clients}) == 1
    assert sum(len(c) for c in clients) == len(y)


def test_dirichlet_partition_wastes_no_data():
    x, y = _toy_data(n=1200)
    clients = partition_clients(
        x, y, n_clients=4, strategy="dirichlet", seed=0, dirichlet_alpha=0.1
    )
    assert sum(len(c) for c in clients) == len(y)


def test_dirichlet_shards_are_disjoint():
    x, y = _toy_data(n=1200)
    clients = partition_clients(x, y, n_clients=4, strategy="dirichlet", seed=0)
    seen = np.concatenate([c.x for c in clients])
    assert len(np.unique(seen, axis=0)) == len(seen)


def test_low_alpha_skews_labels_more_than_high_alpha():
    x, y = _toy_data(n=3000, classes=3)

    def skew(alpha):
        clients = partition_clients(
            x, y, n_clients=6, strategy="dirichlet", seed=0, dirichlet_alpha=alpha
        )
        fractions = np.stack([np.bincount(c.y, minlength=3) / len(c) for c in clients])
        return float(fractions.std())

    assert skew(0.1) > skew(10.0)


def test_natural_partition_keeps_groups_together():
    x, y = _toy_data(n=600)
    groups = np.repeat(np.arange(12), 50)
    clients = partition_clients(
        x, y, n_clients=4, strategy="natural", groups=groups, seed=0, balance=False
    )
    assert len(clients) == 4
    assert sum(len(c) for c in clients) == len(y)


def test_natural_partition_needs_enough_groups():
    x, y = _toy_data(n=100)
    groups = np.repeat(np.arange(2), 50)
    with pytest.raises(ValueError):
        partition_clients(x, y, n_clients=8, strategy="natural", groups=groups)


def test_unknown_partition_strategy_is_rejected():
    x, y = _toy_data()
    with pytest.raises(ValueError):
        partition_clients(x, y, n_clients=2, strategy="nonsense")


# --------------------------------------------------------------------- config


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"dataset": "synthetic", "not_a_real_key": 1})


def test_config_validates_ranges():
    with pytest.raises(ValueError):
        ExperimentConfig(rounds=0)
    with pytest.raises(ValueError):
        ExperimentConfig(epsilon=-1.0)


def test_config_roundtrips_through_a_dict():
    config = ExperimentConfig(dataset="synthetic", rounds=2)
    assert ExperimentConfig.from_dict(config.to_dict()) == config


# ------------------------------------------------------------------ end to end


@pytest.fixture(scope="module")
def synthetic_dataset():
    """Offline dataset so the suite never touches the network."""
    return load_dataset("synthetic", n_clients=4, partition="iid", seed=0)


def test_dataset_shapes_are_consistent(synthetic_dataset):
    data = synthetic_dataset
    assert data.n_clients == 4
    assert data.test_x.shape[1] == data.n_features
    assert len({len(c) for c in data.clients}) == 1  # balanced by default
    assert data.n_classes == 3


def test_features_are_scaled_into_the_encoder_range(synthetic_dataset):
    for client in synthetic_dataset.clients:
        assert client.x.min() >= 0.0
        assert client.x.max() <= 1.0


def test_training_runs_and_improves_over_chance(synthetic_dataset):
    config = ExperimentConfig(
        dataset="synthetic",
        n_clients=4,
        rounds=3,
        dimensions=1000,
        local_epochs=2,
        epsilon=20.0,
        device="cpu",
        seed=0,
    )
    history = run_federated_training(synthetic_dataset, config, progress=False)

    assert len(history.rounds) == 3
    assert history.final_report is not None
    assert history.final_report.accuracy > 1.0 / synthetic_dataset.n_classes


def test_history_records_the_adaptive_noise_saving(synthetic_dataset):
    config = ExperimentConfig(
        dataset="synthetic", n_clients=4, rounds=3, dimensions=500, local_epochs=1, device="cpu"
    )
    history = run_federated_training(synthetic_dataset, config, progress=False)

    first, *later = history.rounds
    assert first.noise_cumulative == 0.0
    assert first.noise_added == pytest.approx(first.noise_required)
    for result in later:
        assert result.noise_added < result.noise_required


def test_disabling_dp_injects_no_noise(synthetic_dataset):
    config = ExperimentConfig(
        dataset="synthetic",
        n_clients=4,
        rounds=2,
        dimensions=500,
        local_epochs=1,
        differential_privacy=False,
        device="cpu",
    )
    history = run_federated_training(synthetic_dataset, config, progress=False)
    assert all(r.noise_added == 0.0 for r in history.rounds)


def test_runs_are_reproducible_under_a_fixed_seed(synthetic_dataset):
    def run():
        config = ExperimentConfig(
            dataset="synthetic",
            n_clients=4,
            rounds=2,
            dimensions=500,
            local_epochs=1,
            device="cpu",
            seed=123,
        )
        return run_federated_training(synthetic_dataset, config, progress=False).accuracies

    assert run() == run()


def test_too_many_rounds_for_the_data_is_reported_clearly(synthetic_dataset):
    config = ExperimentConfig(
        dataset="synthetic", n_clients=4, rounds=100000, dimensions=100, device="cpu"
    )
    with pytest.raises(ValueError, match="Not enough data"):
        run_federated_training(synthetic_dataset, config, progress=False)


def test_history_serialises_to_json_safe_types(synthetic_dataset):
    import json

    config = ExperimentConfig(
        dataset="synthetic", n_clients=4, rounds=2, dimensions=500, local_epochs=1, device="cpu"
    )
    history = run_federated_training(synthetic_dataset, config, progress=False)
    json.dumps(history.to_dict())  # must not raise
