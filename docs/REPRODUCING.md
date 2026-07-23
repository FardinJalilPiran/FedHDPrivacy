# Reproducing the paper

## What this repository can and cannot reproduce

The experiments in the paper use proprietary CNC machining telemetry that cannot be
redistributed. Anything that depends on that specific dataset — the accuracy
numbers in Figure 12, the FPR/FNR values in Table 3, the benchmark comparison
against FedAvg/FedNova/FedProx/FedOpt — cannot be reproduced here.

What **is** fully reproducible, because it depends only on the mechanism and not
on the data:

| Result | How | Depends on the private data? |
| --- | --- | --- |
| Noise accounting (Theorems 2–5) | `pytest tests/test_privacy.py` | No |
| Adaptive noise saving (Figure 8b) | `scripts/plot_noise_schedule.py` | No |
| Required noise vs. `K` and `L` (Figure 9) | `scripts/plot_noise_schedule.py` | No |
| Accuracy vs. `D` and `ε` (Figure 10, *shape*) | `scripts/sweep_privacy_budget.py` | Only for absolute values |
| Accuracy rising across rounds (Figure 12, *trend*) | any full run | Only for absolute values |
| Absolute accuracy / FPR / FNR (Table 3) | — | **Yes** |
| Comparison against FedAvg etc. | — | **Yes** |

## The original experimental setup

For reference, the paper's configuration:

| Parameter | Value |
| --- | --- |
| Clients `K` | 8 (one LASERTEC 65 DED hybrid CNC machine each) |
| Rounds `R` | 10 |
| Hypervector size `D` | 10,000 for the benchmark comparison; swept from 1K to 60K |
| Privacy budget `ε` | 10 for the benchmark comparison; swept from 4 to 30 |
| Classes | 3 — nominal / under / over drilling, from hole-diameter Z-scores |
| Features | 90 process signals (15 signal types × 5 axes + spindle) at 500 Hz |
| Windowing | sliding window of length 10 |
| Split | 80% train / 20% test, test sets pooled across clients |
| Reported accuracy | 70.73% after 10 rounds at `ε = 10`, `D = 10,000` |

To run the closest available equivalent:

```bash
fedhdprivacy --config configs/default.yaml
```

This uses UCI HAR with 8 clients, 10 rounds, `D = 10,000` and `ε = 10` — the same
hyperparameters against different data.

## Substituting the original data

If you have access to comparable machining data, the framework accepts any
federated split whose features are scaled to `[0, 1]`:

```python
import numpy as np
from fedhdprivacy import ExperimentConfig, run_federated_training
from fedhdprivacy.data import ClientData, FederatedDataset

# One ClientData per machine. x: (n_samples, n_features) in [0, 1]; y: int labels.
clients = [
    ClientData(client_id=f"machine_{i}", x=x_i.astype(np.float32), y=y_i.astype(np.int64))
    for i, (x_i, y_i) in enumerate(per_machine_shards)
]

dataset = FederatedDataset(
    name="machining",
    clients=clients,
    test_x=test_x.astype(np.float32),
    test_y=test_y.astype(np.int64),
    n_classes=3,
)

config = ExperimentConfig(dataset="machining", n_clients=len(clients), rounds=10,
                          dimensions=10000, epsilon=10.0)
history = run_federated_training(dataset, config)
```

Two requirements the DP accounting relies on:

1. **Every client holds the same number of samples.** The derivation assumes a
   common `L`. `load_dataset(..., balance_clients=True)` enforces this; if you
   build a `FederatedDataset` by hand, truncate the shards yourself.
2. **Features are in `[0, 1]`.** The encoder's input range is fixed rather than
   fitted, because fitting a scaler on pooled client data would leak across the
   federation.

## Interpreting a run

A round line looks like this:

```
round  7/10  acc=0.6790  FPR=0.1858  FNR=0.3693  noise saved= 23.0%  (0.6s)
```

`noise saved` is `1 − added/required`: the fraction of the naive noise budget the
adaptive mechanism avoided by accounting for what the downloaded global model
already carried. It starts at 0% in round 1 (nothing to inherit) and grows as
rounds accumulate. This is the quantity Figure 8b plots, and the reason accuracy
keeps climbing instead of degrading under repeated noising.

## Expected behaviour

- **Round 1 accuracy is near chance.** The model has seen `L` samples and carries
  the full `ln(1.25L)` noise burden with no inherited noise to offset it. Round 2
  usually jumps sharply.
- **The accuracy curve is not monotone.** Noise is redrawn each round, so
  individual rounds fluctuate. The trend over 10 rounds is what matters.
- **Larger `D` needs more noise.** Variance grows linearly with `D`, which partly
  offsets the extra capacity. Accuracy versus `D` flattens rather than climbing
  indefinitely — visible in Figure 10, where 60K beats 10K by only a few points.
- **Small `ε` is expensive.** Variance grows as `1/ε²`.

## Determinism

Runs are reproducible given a fixed `--seed`, on the same device. CPU and GPU
results can differ slightly through floating-point reduction order. The seed
fixes the data partition, the random projection, the retraining sample order and
the noise draws.
