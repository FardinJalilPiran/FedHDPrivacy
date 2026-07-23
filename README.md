# FedHDPrivacy

Reference implementation of **FedHDPrivacy** — an explainable federated learning framework that combines hyperdimensional computing with differential privacy for IoT environments.

[![Paper](https://img.shields.io/badge/paper-10.1016%2Fj.compeleceng.2025.110261-blue)](https://doi.org/10.1016/j.compeleceng.2025.110261)
[![arXiv](https://img.shields.io/badge/arXiv-2411.01140-b31b1b)](https://arxiv.org/abs/2411.01140)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

> Fardin Jalil Piran, Zhiling Chen, Mohsen Imani, Farhad Imani.
> *Privacy-Preserving Federated Learning with Differentially Private Hyperdimensional Computing.*
> Computers and Electrical Engineering, 123:110261, 2025.

---

## The idea in one paragraph

Federated learning keeps raw data on-device, but the model updates it exchanges still leak information — model inversion attacks can reconstruct training signals, and membership inference attacks can tell whether a specific sample was used. Differential privacy fixes this by adding calibrated noise, but in a **continual-learning** setting the noise compounds: re-noising the model every round eventually drowns the signal. FedHDPrivacy exploits the fact that hyperdimensional models are transparent enough to *account for* noise exactly. A client can compute how much noise the global model it just downloaded already carries, work out how much privacy requires at this round, and inject only the difference. The server adds nothing at all — averaging `K` independently noised local models is already private.

<p align="center">
  <em>Required noise grows with the data seen; injected noise stays well below it.</em>
</p>

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/FardinJalilPiran/FedHDPrivacy.git
cd FedHDPrivacy

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[all]"
```

`pip install -e .` installs only the core (torch, numpy, scikit-learn, PyYAML). The extras are optional:

| Extra | Adds | Needed for |
| --- | --- | --- |
| `hdc` | `torch-hd` | the `density` encoder from the original notebook |
| `vision` | `torchvision` | the MNIST / Fashion-MNIST datasets |
| `plots` | `matplotlib` | the figures produced by `scripts/` |
| `dev` | `pytest`, `ruff` | running the tests and linter |

> **Note** — the PyPI package is `torch-hd` but the import name is `torchhd`. Omitting it is fine: the encoder falls back to an equivalent pure-PyTorch implementation.

### Check it works

```bash
make smoke        # 3 rounds on synthetic data, no network access, < 30s on CPU
pytest            # full test suite
```

## Quick start

```bash
# The paper's configuration, on a public dataset (downloads UCI HAR on first run)
fedhdprivacy --config configs/default.yaml

# Explore the privacy/accuracy trade-off
fedhdprivacy --dataset uci_har --epsilon 4 --dimensions 10000

# Non-private upper bound, to measure what privacy costs you
fedhdprivacy --config configs/no_privacy.yaml
```

Or from Python:

```python
from fedhdprivacy import ExperimentConfig, load_dataset, run_federated_training

config = ExperimentConfig(dataset="uci_har", n_clients=8, rounds=10, epsilon=10.0)
dataset = load_dataset(config.dataset, n_clients=config.n_clients)
history = run_federated_training(dataset, config)

print(history.final_report)          # accuracy, macro FPR, macro FNR
print(history.accuracies)            # accuracy after each round
```

Every run writes a JSON record to `results/` containing the configuration, the per-round accuracy and error rates, and the noise variances that were required, inherited and actually injected.

## About the data

**The dataset in the paper is not in this repository.** The experiments used proprietary CNC machining telemetry (a LASERTEC 65 DED hybrid machine, 90 process signals at 500 Hz, hole-diameter Z-scores as labels) collected with the Connecticut Center for Advanced Technology, which cannot be redistributed.

The framework instead ships with public substitutes that download themselves on first use:

| `--dataset` | Source | Why it's here |
| --- | --- | --- |
| `uci_har` *(default)* | [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) — 561 features, 6 classes, 30 subjects | The closest public analogue: real IoT sensor streams with a **natural** non-IID split, one human subject per client. |
| `mnist`, `fashion_mnist` | torchvision | Standard FL benchmarks, for comparison with the wider literature. |
| `synthetic` | scikit-learn | Runs offline. Used by the test suite and `make smoke`. |

Numbers produced on these datasets will not match the paper's tables — different data, different task. What *is* reproducible is the mechanism: the noise accounting, the per-round accuracy trend, and the gap between required and injected noise.

### Using your own data

Build a `FederatedDataset` directly; anything scaled to `[0, 1]` works:

```python
import numpy as np
from fedhdprivacy.data import ClientData, FederatedDataset

clients = [ClientData(client_id=f"machine_{i}", x=x_i, y=y_i) for i, (x_i, y_i) in enumerate(shards)]
dataset = FederatedDataset(name="my_process", clients=clients,
                           test_x=test_x, test_y=test_y, n_classes=3)
```

## How the noise accounting works

For hypervector size `D`, privacy budget `ε`, `K` clients and `L` new samples per client per round, at round `r`:

| Quantity | Variance |
| --- | --- |
| Required by round `r` (Thm. 2, 4) | `(2D/ε²)·ln(1.25·[(r−1)KL + L])` |
| Already inherited from the global model | `(2D/Kε²)·ln(1.25·[(r−2)KL + L])` |
| **Actually injected** | the difference between the two |

The server adds nothing: Theorems 3 and 5 show the averaged model is already private, because averaging `K` independently noised models divides the noise variance by `K` while the sensitivity of the aggregate falls by `K` as well.

Two consequences worth knowing before you tune anything:

- **Noise scales linearly with `D`.** A bigger hyperspace is more expressive *and* needs proportionally more noise; the two effects partly cancel, which is why accuracy versus `D` flattens out rather than climbing forever.
- **Noise scales with `1/ε²`.** Halving `ε` quadruples the injected variance. This is the knob that hurts.

Inspect the schedule without training anything:

```bash
python scripts/plot_noise_schedule.py --rounds 50 --clients 8 --samples 1000
```

## Repository layout

```
FedHDPrivacy/
├── src/fedhdprivacy/
│   ├── config.py       ExperimentConfig: every knob, validated
│   ├── data.py         dataset download, scaling, FederatedDataset
│   ├── partition.py    natural / IID / Dirichlet client splits
│   ├── hdc.py          encoders, class hypervectors, train/retrain/predict
│   ├── privacy.py      the DP accounting (Theorems 2-5)
│   ├── federated.py    the communication-round loop
│   ├── metrics.py      accuracy, macro FPR/FNR
│   ├── utils.py        seeding, device selection, result I/O
│   └── cli.py          the `fedhdprivacy` command
├── configs/            default, no_privacy, quick, mnist
├── scripts/            privacy-budget sweep, noise-schedule plot
├── notebooks/          walkthrough of the framework
├── tests/              pytest suite
└── docs/              reproducing the paper, algorithm notes
```

## Configuration

Any config file field can be overridden on the command line. Run `fedhdprivacy --help` for the full list; the ones that matter:

| Option | Default | Notes |
| --- | --- | --- |
| `--dataset` | `uci_har` | see the table above |
| `--partition` | `natural` | `natural` / `iid` / `dirichlet` |
| `--clients` | `8` | `K` |
| `--rounds` | `10` | `R`; each client's data is split into `R` shards, one per round |
| `--dimensions` | `10000` | `D`; accuracy, memory, time and energy all scale with this |
| `--epsilon` | `10.0` | lower is more private and less accurate |
| `--encoder` | `projection` | `projection` is fast; `density` matches the original notebook |
| `--no-dp` | off | non-private baseline |
| `--device` | `auto` | CUDA if available |

### Which encoder?

`projection` implements the random vector functional link mapping the paper describes, as a single matmul: memory is `O(n_features × D)` regardless of batch size. `density` uses `torchhd.embeddings.Density`, exactly as in the released notebook, but it materialises a `(batch × n_features × D)` intermediate, so it is roughly 25× slower and memory-bound at large `D`. Accuracy is comparable. Use `density` when you need bit-level parity with the original code, `projection` otherwise.

## Reproducing the figures

```bash
# Figure 8b: required vs. injected noise across rounds
python scripts/plot_noise_schedule.py --rounds 50 --clients 8 --samples 1000

# Figure 10: accuracy vs. hypervector size and privacy budget
python scripts/sweep_privacy_budget.py --epsilons 4 6 8 10 --dimensions 1000 5000 10000
```

See [`docs/REPRODUCING.md`](docs/REPRODUCING.md) for what does and does not carry over from the paper, and [`CHANGELOG.md`](CHANGELOG.md) for what changed from the original notebook.

## Development

```bash
make dev          # editable install with all extras
make test         # pytest
make lint         # ruff check + format check
make format       # auto-fix
```

## Citation

```bibtex
@article{piran2025privacy,
  title   = {Privacy-Preserving Federated Learning with Differentially Private Hyperdimensional Computing},
  author  = {Piran, Fardin Jalil and Chen, Zhiling and Imani, Mohsen and Imani, Farhad},
  journal = {Computers and Electrical Engineering},
  volume  = {123},
  pages   = {110261},
  year    = {2025},
  doi     = {10.1016/j.compeleceng.2025.110261},
  publisher = {Elsevier}
}
```

## Acknowledgments

Supported by the National Science Foundation (2127780, 2312517), the Semiconductor Research Corporation, the Office of Naval Research (N00014-21-1-2225, N00014-22-1-2067), the Air Force Office of Scientific Research (FA9550-22-1-0253), UConn Startup Funding, and gifts from Xilinx and Cisco. We thank the Connecticut Center for Advanced Technology, and Nasir Mannan in particular, for sharing data for this research.

## License

MIT — see [LICENSE](LICENSE).
