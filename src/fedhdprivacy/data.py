"""Public datasets for the FedHDPrivacy experiments.

The dataset used in the paper is proprietary CNC machining telemetry collected
at the Connecticut Center for Advanced Technology and cannot be redistributed.
This module provides publicly downloadable substitutes so that the framework is
reproducible end to end.

Available datasets
------------------
``uci_har``
    UCI Human Activity Recognition Using Smartphones.  561 hand-engineered
    features from accelerometer and gyroscope streams, 6 activity classes,
    30 human subjects.  This is the closest public analogue to the paper's
    setting: real IoT sensor signals with a *natural*, genuinely non-IID
    client partition (one subject = one device).  Downloaded from the UCI
    Machine Learning Repository on first use.

``mnist`` / ``fashion_mnist``
    Standard federated-learning benchmarks, downloaded via torchvision.
    Useful as a sanity check and for comparison with the wider FL literature.

``synthetic``
    Generated locally with scikit-learn.  Requires no network access, so the
    test suite and a quick smoke run work offline.

Feature scaling
---------------
The encoder expects inputs in ``[0, 1]``.  Rather than fitting a scaler on the
pooled training data - which would leak information across clients and defeat
the point of a federated setup - each dataset declares a fixed, a-priori known
input range that is applied identically everywhere.
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

__all__ = ["ClientData", "FederatedDataset", "load_dataset", "AVAILABLE_DATASETS"]

AVAILABLE_DATASETS = ("uci_har", "mnist", "fashion_mnist", "synthetic")

_UCI_HAR_URLS = (
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
)


# --------------------------------------------------------------------------- #
# containers
# --------------------------------------------------------------------------- #


@dataclass
class ClientData:
    """Training data held locally by one client."""

    client_id: str
    x: np.ndarray  # (n_samples, n_features), scaled to [0, 1]
    y: np.ndarray  # (n_samples,), int64

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class FederatedDataset:
    """A federated split plus a held-out global test set."""

    name: str
    clients: list[ClientData]
    test_x: np.ndarray
    test_y: np.ndarray
    n_classes: int
    metadata: dict = field(default_factory=dict)

    @property
    def n_clients(self) -> int:
        return len(self.clients)

    @property
    def n_features(self) -> int:
        return self.clients[0].x.shape[1]

    def min_client_samples(self) -> int:
        return min(len(c) for c in self.clients)

    def summary(self) -> str:
        sizes = [len(c) for c in self.clients]
        return (
            f"{self.name}: {self.n_clients} clients, {self.n_features} features, "
            f"{self.n_classes} classes | train sizes min={min(sizes)} "
            f"max={max(sizes)} total={sum(sizes)} | test={len(self.test_y)}"
        )


# --------------------------------------------------------------------------- #
# download helpers
# --------------------------------------------------------------------------- #


def _download(url: str, destination: Path) -> None:
    LOGGER.info("Downloading %s", url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _extract_zip_recursively(archive: Path, target: Path) -> None:
    """Extract ``archive`` into ``target``, then extract any nested zips.

    The UCI HAR download is a zip containing another zip, so one pass is not
    enough.
    """
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    for nested in list(target.rglob("*.zip")):
        with zipfile.ZipFile(nested) as zf:
            zf.extractall(nested.parent)
        nested.unlink()


def _ensure_uci_har(root: Path) -> Path:
    """Return the directory containing ``train/X_train.txt``, downloading if needed."""
    root = Path(root) / "uci_har"
    marker = next(root.rglob("train/X_train.txt"), None) if root.exists() else None
    if marker is not None:
        return marker.parent.parent

    archive = root / "uci_har.zip"
    last_error: Exception | None = None
    for url in _UCI_HAR_URLS:
        try:
            _download(url, archive)
            _extract_zip_recursively(archive, root)
            archive.unlink(missing_ok=True)
            break
        except Exception as exc:  # noqa: BLE001 - report every mirror that failed
            last_error = exc
            LOGGER.warning("Could not fetch %s (%s)", url, exc)
    else:
        raise RuntimeError(
            "Failed to download the UCI HAR dataset from any known mirror.\n"
            f"Last error: {last_error}\n\n"
            "You can download it manually and unzip it so that the following "
            f"path exists:\n  {root}/UCI HAR Dataset/train/X_train.txt\n"
            "Source: https://archive.ics.uci.edu/dataset/240/"
            "human+activity+recognition+using+smartphones"
        )

    marker = next(root.rglob("train/X_train.txt"), None)
    if marker is None:
        raise RuntimeError(f"UCI HAR archive extracted to {root} but X_train.txt was not found.")
    return marker.parent.parent


# --------------------------------------------------------------------------- #
# raw loaders -> (train_x, train_y, train_group, test_x, test_y)
# --------------------------------------------------------------------------- #


def _load_uci_har(root: Path):
    base = _ensure_uci_har(root)

    def read(split: str):
        x = np.loadtxt(base / split / f"X_{split}.txt", dtype=np.float32)
        y = np.loadtxt(base / split / f"y_{split}.txt", dtype=np.int64) - 1  # 1..6 -> 0..5
        subjects = np.loadtxt(base / split / f"subject_{split}.txt", dtype=np.int64)
        return x, y, subjects

    train_x, train_y, train_subjects = read("train")
    test_x, test_y, _ = read("test")

    # Features are documented as already normalised to [-1, 1].
    train_x = (np.clip(train_x, -1.0, 1.0) + 1.0) / 2.0
    test_x = (np.clip(test_x, -1.0, 1.0) + 1.0) / 2.0
    return train_x, train_y, train_subjects, test_x, test_y


def _load_torchvision(name: str, root: Path):
    try:
        from torchvision import datasets  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"The '{name}' dataset requires torchvision. Install it with "
            "`pip install torchvision`, or use --dataset uci_har / synthetic."
        ) from exc

    cls = {"mnist": datasets.MNIST, "fashion_mnist": datasets.FashionMNIST}[name]
    root = Path(root) / name
    train = cls(root=str(root), train=True, download=True)
    test = cls(root=str(root), train=False, download=True)

    def to_arrays(ds):
        x = ds.data.numpy().reshape(len(ds.data), -1).astype(np.float32) / 255.0
        y = ds.targets.numpy().astype(np.int64)
        return x, y

    train_x, train_y = to_arrays(train)
    test_x, test_y = to_arrays(test)
    return train_x, train_y, None, test_x, test_y


def _load_synthetic(
    n_samples: int = 12000, n_features: int = 64, n_classes: int = 3, seed: int = 0
):
    from sklearn.datasets import make_classification  # noqa: PLC0415

    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(8, n_features // 4),
        n_redundant=4,
        n_classes=n_classes,
        n_clusters_per_class=2,
        class_sep=1.5,
        random_state=seed,
    )
    x = x.astype(np.float32)
    lo, hi = x.min(axis=0, keepdims=True), x.max(axis=0, keepdims=True)
    x = (x - lo) / np.maximum(hi - lo, 1e-8)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    x, y = x[order], y[order].astype(np.int64)
    split = int(0.8 * len(x))
    return x[:split], y[:split], None, x[split:], y[split:]


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #


def load_dataset(
    name: str = "uci_har",
    n_clients: int = 8,
    partition: str = "natural",
    root: str | Path = "data",
    seed: int = 42,
    dirichlet_alpha: float = 0.5,
    balance_clients: bool = True,
) -> FederatedDataset:
    """Load a dataset and split it across ``n_clients``.

    Parameters
    ----------
    name:
        One of :data:`AVAILABLE_DATASETS`.
    n_clients:
        Number of simulated clients ``K``.
    partition:
        ``"natural"`` groups by the dataset's own device/subject identifier when
        one exists (UCI HAR) and falls back to ``"dirichlet"`` otherwise.
        ``"iid"`` shuffles uniformly; ``"dirichlet"`` produces label skew.
    root:
        Directory used to cache downloads.
    balance_clients:
        Truncate every client to the same number of samples.  The DP accounting
        assumes a common ``L``, so this is on by default.
    """
    from .partition import partition_clients  # local import avoids a cycle

    name = name.lower()
    if name not in AVAILABLE_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {AVAILABLE_DATASETS}.")

    root = Path(root)
    if name == "uci_har":
        train_x, train_y, groups, test_x, test_y = _load_uci_har(root)
    elif name in ("mnist", "fashion_mnist"):
        train_x, train_y, groups, test_x, test_y = _load_torchvision(name, root)
    else:
        train_x, train_y, groups, test_x, test_y = _load_synthetic(seed=seed)

    n_classes = int(max(train_y.max(), test_y.max()) + 1)
    clients = partition_clients(
        train_x,
        train_y,
        n_clients=n_clients,
        strategy=partition,
        groups=groups,
        seed=seed,
        dirichlet_alpha=dirichlet_alpha,
        balance=balance_clients,
    )

    return FederatedDataset(
        name=name,
        clients=clients,
        test_x=test_x,
        test_y=test_y,
        n_classes=n_classes,
        metadata={"partition": partition, "seed": seed},
    )
