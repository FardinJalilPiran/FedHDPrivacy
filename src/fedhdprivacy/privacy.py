"""Differential privacy mechanisms for FedHDPrivacy.

This module implements the adaptive Gaussian mechanism described in the paper.
The central idea is that a client does not re-noise its model from scratch every
round: the global model it downloads already carries the averaged noise of the
previous round.  A client therefore only injects the *difference* between the
noise required at round ``r`` and the noise already accumulated.

Notation (matching the paper)
-----------------------------
``D``       hypervector dimensionality
``K``       number of clients
``L``       number of training samples used by one client per round
``eps``     privacy budget (epsilon)
``r``       communication round, 1-indexed
``xi``      required noise variance at round r
``psi``     cumulative noise variance already present in the model
``gamma``   additional noise variance actually injected (xi - psi)

References
----------
Theorem 2 (round 1), Theorem 4 (round >= 2) of

    Piran, Chen, Imani & Imani, "Privacy-Preserving Federated Learning with
    Differentially Private Hyperdimensional Computing",
    Computers and Electrical Engineering 123:110261, 2025.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "required_noise_variance",
    "cumulative_noise_variance",
    "additional_noise_variance",
    "GaussianNoiseMechanism",
]

# The 1.25 factor comes from the Gaussian mechanism bound
#   sigma^2 > 2 ln(1.25 / delta) * (Delta f)^2 / eps^2
_GAUSSIAN_CONST = 1.25


def _check_args(dimensions: int, epsilon: float, n_clients: int, samples_per_round: int) -> None:
    if dimensions <= 0:
        raise ValueError(f"dimensions must be positive, got {dimensions}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if n_clients < 1:
        raise ValueError(f"n_clients must be >= 1, got {n_clients}")
    if samples_per_round < 1:
        raise ValueError(f"samples_per_round must be >= 1, got {samples_per_round}")


def required_noise_variance(
    dimensions: int,
    epsilon: float,
    n_clients: int,
    samples_per_round: int,
    round_index: int,
) -> float:
    """Variance of the noise a local model must carry at ``round_index``.

    Round 1 (Theorem 2)::

        Var = (2 D / eps^2) * ln(1.25 L)

    Round r >= 2 (Theorem 4, first term)::

        Var = (2 D / eps^2) * ln(1.25 (r - 1) K L + 1.25 L)

    The argument of the logarithm is the total number of training samples that
    have contributed to the model so far, since ``delta`` is set to the inverse
    of that count.
    """
    _check_args(dimensions, epsilon, n_clients, samples_per_round)
    if round_index < 1:
        raise ValueError(f"round_index is 1-indexed, got {round_index}")

    coefficient = (2.0 * dimensions) / (epsilon**2)
    if round_index == 1:
        return coefficient * math.log(_GAUSSIAN_CONST * samples_per_round)

    seen = (round_index - 1) * n_clients * samples_per_round + samples_per_round
    return coefficient * math.log(_GAUSSIAN_CONST * seen)


def cumulative_noise_variance(
    dimensions: int,
    epsilon: float,
    n_clients: int,
    samples_per_round: int,
    round_index: int,
) -> float:
    """Variance of the noise already present in the downloaded global model.

    The server performs a plain average of ``K`` independently noised local
    models, so the variance of the aggregate is the per-client variance divided
    by ``K`` (Proof 2, eq. 16).  At round 1 there is nothing to inherit.
    """
    _check_args(dimensions, epsilon, n_clients, samples_per_round)
    if round_index < 1:
        raise ValueError(f"round_index is 1-indexed, got {round_index}")
    if round_index == 1:
        return 0.0

    coefficient = (2.0 * dimensions) / (n_clients * epsilon**2)
    seen = (round_index - 2) * n_clients * samples_per_round + samples_per_round
    return coefficient * math.log(_GAUSSIAN_CONST * seen)


def additional_noise_variance(
    dimensions: int,
    epsilon: float,
    n_clients: int,
    samples_per_round: int,
    round_index: int,
) -> float:
    """Variance of the noise that is actually injected this round.

    This is ``xi_r - psi_{r-1}``: the whole point of FedHDPrivacy.  Naively
    re-applying ``required_noise_variance`` every round would over-noise the
    model and destroy accuracy in a long-running deployment.
    """
    required = required_noise_variance(
        dimensions, epsilon, n_clients, samples_per_round, round_index
    )
    cumulative = cumulative_noise_variance(
        dimensions, epsilon, n_clients, samples_per_round, round_index
    )
    delta = required - cumulative
    if delta < 0.0:
        # Should not happen for K, L, r >= 2 (Lemma 1 and Lemma 2), but a
        # degenerate configuration (e.g. K = 1) can make the difference
        # negative.  Falling back to zero keeps the guarantee conservative
        # rather than silently taking sqrt of a negative number.
        return 0.0
    return delta


@dataclass
class GaussianNoiseMechanism:
    """Stateful helper that noises class hypervectors round by round.

    Parameters
    ----------
    dimensions:
        Hypervector size ``D``.
    epsilon:
        Privacy budget.  Smaller means more privacy and more noise.
    n_clients:
        Number of participating clients ``K``.
    samples_per_round:
        Number of new samples each client trains on per round ``L``.
    enabled:
        When ``False`` the mechanism is a no-op.  Useful for measuring the
        accuracy cost of privacy against a non-private baseline.
    """

    dimensions: int
    epsilon: float
    n_clients: int
    samples_per_round: int
    enabled: bool = True

    def variances(self, round_index: int) -> dict[str, float]:
        """Return required / cumulative / added variance for bookkeeping."""
        if not self.enabled:
            return {"required": 0.0, "cumulative": 0.0, "added": 0.0}
        args = (self.dimensions, self.epsilon, self.n_clients, self.samples_per_round, round_index)
        required = required_noise_variance(*args)
        cumulative = cumulative_noise_variance(*args)
        return {
            "required": required,
            "cumulative": cumulative,
            "added": max(required - cumulative, 0.0),
        }

    def perturb(
        self,
        class_hypervectors: torch.Tensor,
        round_index: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return a noised copy of ``class_hypervectors``.

        Parameters
        ----------
        class_hypervectors:
            Tensor of shape ``(n_classes, dimensions)``.
        round_index:
            1-indexed communication round.
        generator:
            Optional RNG for reproducible noise draws.
        """
        if not self.enabled:
            return class_hypervectors.clone()

        variance = additional_noise_variance(
            self.dimensions,
            self.epsilon,
            self.n_clients,
            self.samples_per_round,
            round_index,
        )
        if variance <= 0.0:
            return class_hypervectors.clone()

        std = math.sqrt(variance)
        target_device = class_hypervectors.device

        # PyTorch requires a generator to live on the same device as the tensor
        # being sampled. Rather than forcing the caller to match devices, draw
        # on the generator's own device and move the result. The noise tensor is
        # only (n_classes, D), so the transfer is negligible - and it makes a
        # seeded run produce identical noise on CPU and GPU.
        sample_device = generator.device if generator is not None else target_device
        noise = torch.randn(
            class_hypervectors.shape,
            device=sample_device,
            dtype=class_hypervectors.dtype,
            generator=generator,
        )
        return class_hypervectors + std * noise.to(target_device)
