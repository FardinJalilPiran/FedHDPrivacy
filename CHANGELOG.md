# Changelog

## 1.0.1

### Fixed
- **CUDA runs crashed at the first noise injection** with
  `RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'`.
  The seeded generator is created on the host, but PyTorch requires a generator
  to live on the same device as the tensor it fills. Both `privacy.perturb` and
  `HDClassifier.retrain` now sample on the generator's own device and move the
  result, which also makes a seeded run produce identical noise on CPU and GPU.

### Changed
- The retraining shuffle order is pulled to the host as plain integers. Indexing
  one row at a time with a device-resident index tensor forced a transfer on
  every access.

### Added
- `tests/test_device.py`, covering generator/device pairing, encoder placement
  and a full run on each device. The CUDA cases skip when no GPU is present, so
  run the suite on a GPU machine to exercise them.

## 1.0.0

First packaged release. Replaces the original single-notebook implementation.

### Added
- Installable `fedhdprivacy` package with a `fedhdprivacy` command-line entry point.
- Public datasets that download on first use: UCI HAR (default), MNIST,
  Fashion-MNIST, and an offline synthetic generator.
- Client partitioning strategies: `natural` (by subject/device), `iid`, and a
  size-preserving `dirichlet` split.
- YAML configurations with command-line overrides (`configs/`).
- Test suite covering the DP accounting, the encoders, partitioning and the
  end-to-end loop.
- Scripts reproducing the noise schedule and the privacy-budget sweep.
- `docs/ALGORITHM.md` and `docs/REPRODUCING.md`.

### Changed
- The proprietary CNC machining dataset (`data_hole_diameter_10.pkl`) has been
  removed; it cannot be redistributed. See `docs/REPRODUCING.md` for how to
  substitute your own data.
- Encoding, training and inference now operate on batches rather than one
  sample at a time.
- Added a `projection` encoder as the default. The original `density` encoder
  (torchhd) materialises a `(batch x n_features x D)` intermediate and becomes
  memory-bound well before `D = 10000`; it remains available via
  `--encoder density`.
- The code now runs on CPU as well as GPU; device selection is automatic.

### Fixed
- Balancing a Dirichlet split to the smallest shard discarded most of the
  training data. Clients now receive equal-sized shards with a skewed label
  mix, which also satisfies the common `L` the DP derivation assumes.
- `torch.sign` mapped exact zeros to zero, silently dropping a dimension from
  affected hypervectors. Zeros now map to +1.
