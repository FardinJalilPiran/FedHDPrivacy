"""Strategies for splitting a centralised dataset across simulated clients.

The DP accounting in :mod:`fedhdprivacy.privacy` assumes every client
contributes the same number of samples per round, so all strategies here aim to
produce equal-sized shards.  For the Dirichlet strategy that means allocating a
fixed budget per client and skewing *which* labels fill it, rather than
generating wildly uneven shards and then throwing most of the data away.
"""

from __future__ import annotations

import logging

import numpy as np

from .data import ClientData

LOGGER = logging.getLogger(__name__)

__all__ = ["partition_clients"]

VALID_STRATEGIES = ("natural", "iid", "dirichlet")


def partition_clients(
    x: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    strategy: str = "natural",
    groups: np.ndarray | None = None,
    seed: int = 42,
    dirichlet_alpha: float = 0.5,
    balance: bool = True,
) -> list[ClientData]:
    """Split ``(x, y)`` into ``n_clients`` shards.

    ``natural``
        Group by the dataset's own subject/device identifier.  The most
        realistic heterogeneity, because the skew comes from who generated the
        data.  Falls back to ``dirichlet`` when no identifier exists.
    ``iid``
        Uniform random split; every client sees the same distribution.
    ``dirichlet``
        Every client gets the same number of samples, but its label mix is
        drawn from ``Dir(alpha)``.  Lower ``alpha`` means stronger skew.
    """
    if n_clients < 1:
        raise ValueError(f"n_clients must be >= 1, got {n_clients}")
    strategy = strategy.lower()
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown partition strategy '{strategy}'. Choose from {VALID_STRATEGIES}."
        )

    rng = np.random.default_rng(seed)

    if strategy == "natural" and groups is None:
        LOGGER.info("No natural client identifier in this dataset; using a Dirichlet split.")
        strategy = "dirichlet"

    if strategy == "natural":
        indices = _split_by_group(groups, n_clients, rng)
    elif strategy == "iid":
        indices = _split_iid(len(y), n_clients, rng)
    else:
        indices = _split_dirichlet(y, n_clients, dirichlet_alpha, rng)

    if balance:
        indices = _balance(indices, rng)

    return [
        ClientData(client_id=f"client_{i:02d}", x=x[idx], y=y[idx]) for i, idx in enumerate(indices)
    ]


def _balance(indices: list[np.ndarray], rng) -> list[np.ndarray]:
    """Truncate every shard to the smallest one, warning if that is wasteful."""
    sizes = [len(idx) for idx in indices]
    smallest = min(sizes)
    if smallest == 0:
        raise ValueError(
            "A client received zero samples. Reduce n_clients or increase dirichlet_alpha."
        )
    discarded = sum(sizes) - smallest * len(sizes)
    if discarded > 0.25 * sum(sizes):
        LOGGER.warning(
            "Balancing clients to %d samples each discards %.0f%% of the training data "
            "because the split is very uneven (sizes %d-%d). Consider a larger "
            "dirichlet_alpha, or balance_clients=False if your setup tolerates unequal L.",
            smallest,
            100 * discarded / sum(sizes),
            smallest,
            max(sizes),
        )
    return [rng.permutation(idx)[:smallest] for idx in indices]


def _split_by_group(groups: np.ndarray, n_clients: int, rng) -> list[np.ndarray]:
    """Assign whole groups (subjects) to clients, greedily balancing sizes."""
    unique = np.unique(groups)
    if len(unique) < n_clients:
        raise ValueError(
            f"Dataset has only {len(unique)} natural groups but {n_clients} clients were "
            "requested. Use --partition dirichlet, or lower --clients."
        )

    counts = {g: int((groups == g).sum()) for g in unique}
    buckets: list[list] = [[] for _ in range(n_clients)]
    loads = np.zeros(n_clients, dtype=np.int64)

    # Largest group first, always onto the lightest client: a standard greedy
    # approximation that keeps shard sizes close without splitting a subject.
    for group in sorted(unique, key=lambda g: -counts[g]):
        target = int(loads.argmin())
        buckets[target].append(group)
        loads[target] += counts[group]

    return [rng.permutation(np.where(np.isin(groups, bucket))[0]) for bucket in buckets]


def _split_iid(n_samples: int, n_clients: int, rng) -> list[np.ndarray]:
    return list(np.array_split(rng.permutation(n_samples), n_clients))


def _split_dirichlet(y: np.ndarray, n_clients: int, alpha: float, rng) -> list[np.ndarray]:
    """Equal-sized shards with Dirichlet-skewed label composition.

    Each client is given a budget of ``len(y) // n_clients`` samples and a label
    mix drawn from ``Dir(alpha)``.  It is filled greedily from the per-class
    pools; when a preferred class runs dry the shortfall is taken from whichever
    class still has the most samples left.  This keeps the label skew that makes
    federated aggregation interesting while preserving the equal ``L`` the DP
    accounting assumes.
    """
    classes = np.unique(y)
    pools = {c: list(rng.permutation(np.where(y == c)[0])) for c in classes}
    budget = len(y) // n_clients

    shards: list[np.ndarray] = []
    for _ in range(n_clients):
        proportions = rng.dirichlet(np.repeat(alpha, len(classes)))
        wanted = np.floor(proportions * budget).astype(int)

        taken: list[int] = []
        for cls, count in zip(classes, wanted):
            available = min(count, len(pools[cls]))
            for _ in range(available):
                taken.append(pools[cls].pop())

        # Top up from the largest remaining pool until the budget is met.
        while len(taken) < budget:
            largest = max(pools, key=lambda c: len(pools[c]))
            if not pools[largest]:
                break
            taken.append(pools[largest].pop())

        shards.append(rng.permutation(np.array(taken, dtype=np.int64)))

    return shards
