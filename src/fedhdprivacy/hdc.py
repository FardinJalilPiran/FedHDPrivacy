"""Hyperdimensional computing primitives.

A model is a bundle of ``n_classes`` class hypervectors.  Training bundles the
encoded samples of each class, inference is a nearest-class search, and
retraining nudges the two class hypervectors involved in a misclassification.

Encoders
--------
The paper describes encoding as a random vector functional link mapping: a
random projection into a high-dimensional space followed by a non-linearity and
binarisation.  Two backends implement this:

``projection`` (default)
    Pure PyTorch.  One ``(n_features, D)`` projection matrix and a single
    matmul per batch, so memory is ``O(n_features * D)`` and independent of the
    batch size.  This is what makes ``D = 60000`` practical.

``density``
    :class:`torchhd.embeddings.Density`, the encoder used in the original
    notebook.  Faithful to the released code, but it materialises a
    ``(batch, n_features, D)`` intermediate and an ``O(D^2)`` level codebook,
    so memory becomes the binding constraint well before ``D = 10000``.  Batch
    sizes are capped automatically; expect it to be slow on high-dimensional
    inputs.

Everything operates on batches so the encoder is called once per round rather
than once per sample per epoch, which is where the original notebook spent most
of its time.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch

LOGGER = logging.getLogger(__name__)

__all__ = [
    "AVAILABLE_ENCODERS",
    "Encoder",
    "HDClassifier",
    "average",
    "binarize",
    "build_encoder",
]

AVAILABLE_ENCODERS = ("projection", "density")

# Soft ceiling on the temporary allocation made while encoding one batch.
_ENCODE_MEMORY_BUDGET_BYTES = 256 * 1024 * 1024


def binarize(x: torch.Tensor) -> torch.Tensor:
    """Map a real tensor to {-1, +1}.

    ``torch.sign`` sends exact zeros to zero, which would silently drop a
    dimension from the hypervector.  Zeros go to +1 instead.
    """
    return torch.where(x >= 0, 1.0, -1.0).to(x.dtype)


class Encoder:
    """Maps feature vectors in ``[0, 1]`` to bipolar hypervectors.

    Parameters
    ----------
    n_features:
        Input dimensionality.
    dimensions:
        Hypervector size ``D``.
    backend:
        ``"projection"`` or ``"density"``; see the module docstring.
    device, dtype:
        Where the encoded hypervectors live.
    seed:
        Fixes the random projection so runs are reproducible.
    """

    def __init__(
        self,
        n_features: int,
        dimensions: int,
        backend: str = "projection",
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int | None = None,
    ) -> None:
        if backend not in AVAILABLE_ENCODERS:
            raise ValueError(f"Unknown encoder '{backend}'. Choose from {AVAILABLE_ENCODERS}.")
        self.n_features = n_features
        self.dimensions = dimensions
        self.backend = backend
        self.device = torch.device(device)
        self.dtype = dtype
        self._impl = self._build(seed)
        self._max_batch = self._safe_batch_size()

    # ------------------------------------------------------------- internals

    def _build(self, seed: int | None):
        if self.backend == "density":
            try:
                from torchhd import embeddings  # noqa: PLC0415 - optional dependency
            except ImportError:
                LOGGER.warning(
                    "torchhd is not installed (pip install torch-hd); "
                    "falling back to the 'projection' encoder."
                )
                self.backend = "projection"
            else:
                if seed is not None:
                    torch.manual_seed(seed)
                module = embeddings.Density(self.n_features, self.dimensions, low=0.0, high=1.0)
                return module.to(device=self.device)

        return _RandomProjectionEncoder(self.n_features, self.dimensions, self.device, seed)

    def _safe_batch_size(self) -> int:
        """Largest batch that keeps the encoder's scratch space bounded.

        The density backend allocates one hypervector per input feature per
        sample before bundling them, so its footprint grows with the batch.
        The projection backend does not, and can take whatever it is given.
        """
        if self.backend != "density":
            return 8192
        per_sample = self.n_features * self.dimensions * 4  # float32
        batch = max(1, _ENCODE_MEMORY_BUDGET_BYTES // max(per_sample, 1))
        if batch < 32:
            LOGGER.warning(
                "The 'density' encoder needs ~%.0f MB per sample at D=%d with %d features, "
                "so batches are capped at %d and encoding will be slow. "
                "Consider --encoder projection.",
                per_sample / 1e6,
                self.dimensions,
                self.n_features,
                batch,
            )
        return int(batch)

    # ---------------------------------------------------------------- public

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of shape ``(n_samples, n_features)`` to ``(n_samples, D)``."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            encoded = torch.as_tensor(self._impl(x))
        return binarize(encoded).to(self.dtype)

    def encode(self, x: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
        """Encode a large tensor in memory-safe chunks."""
        batch_size = min(batch_size or self._max_batch, self._max_batch)
        if len(x) <= batch_size:
            return self(x)
        chunks = [self(x[i : i + batch_size]) for i in range(0, len(x), batch_size)]
        return torch.cat(chunks, dim=0)

    # Alias kept so external scripts written against the first release still work.
    encode_in_batches = encode


class _RandomProjectionEncoder:
    """Random vector functional link encoder.

    Projects the input through a fixed Gaussian matrix and applies a cosine
    non-linearity with a random phase; binarisation is left to the caller.
    Nearby inputs map to nearby hypervectors, which is the locality property
    hyperdimensional classification relies on.
    """

    def __init__(self, n_features: int, dimensions: int, device, seed: int | None) -> None:
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        # Scaling by 1/sqrt(n_features) keeps the pre-activation magnitude
        # independent of the input dimensionality, so the cosine does not
        # oscillate wildly and destroy locality.
        scale = 1.0 / math.sqrt(n_features)
        self.projection = (
            torch.randn((n_features, dimensions), generator=generator, dtype=torch.float32) * scale
        ).to(device)
        self.bias = (
            torch.rand((dimensions,), generator=generator, dtype=torch.float32) * 2 * math.pi
        ).to(device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(x @ self.projection + self.bias)


def build_encoder(
    n_features: int,
    dimensions: int,
    backend: str = "projection",
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> Encoder:
    """Convenience factory mirroring the config field names."""
    return Encoder(
        n_features=n_features,
        dimensions=dimensions,
        backend=backend,
        device=device,
        seed=seed,
    )


@dataclass
class HDClassifier:
    """A bundle of class hypervectors with train / retrain / predict."""

    class_hypervectors: torch.Tensor  # (n_classes, dimensions)

    @classmethod
    def zeros(
        cls,
        n_classes: int,
        dimensions: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> HDClassifier:
        return cls(torch.zeros((n_classes, dimensions), device=device, dtype=dtype))

    @property
    def n_classes(self) -> int:
        return self.class_hypervectors.shape[0]

    @property
    def dimensions(self) -> int:
        return self.class_hypervectors.shape[1]

    def clone(self) -> HDClassifier:
        return HDClassifier(self.class_hypervectors.clone())

    def to(self, device: torch.device | str) -> HDClassifier:
        return HDClassifier(self.class_hypervectors.to(device))

    # ------------------------------------------------------------------ train

    def bundle(self, encoded: torch.Tensor, labels: torch.Tensor) -> HDClassifier:
        """One-shot training: add every encoded sample to its class hypervector."""
        device = self.class_hypervectors.device
        labels = labels.to(device).long()
        encoded = encoded.to(device, dtype=self.class_hypervectors.dtype)
        self.class_hypervectors.index_add_(0, labels, encoded)
        return self

    def retrain(
        self,
        encoded: torch.Tensor,
        labels: torch.Tensor,
        n_epochs: int = 10,
        generator: torch.Generator | None = None,
    ) -> HDClassifier:
        """Perceptron-style error-driven refinement.

        Each misclassified sample is added to its true class hypervector and
        subtracted from the predicted one.  Iteration stops early once an epoch
        makes no mistakes.  Updates are sequential by construction: a correction
        changes the decision boundary seen by the next sample.
        """
        device = self.class_hypervectors.device
        encoded = encoded.to(device, dtype=self.class_hypervectors.dtype)
        labels = labels.to(device).long()
        n_samples = len(encoded)

        for _ in range(n_epochs):
            # Same device rule as in privacy.perturb: sample where the generator
            # lives. The order is then pulled to the host as plain ints - the
            # loop indexes one row at a time, and a device-resident index tensor
            # would force a transfer on every single access.
            sample_device = generator.device if generator is not None else device
            order = torch.randperm(n_samples, generator=generator, device=sample_device).tolist()
            mistakes = 0
            for index in order:
                sample = encoded[index]
                prediction = torch.matmul(self.class_hypervectors, sample).argmax()
                target = labels[index]
                if prediction != target:
                    self.class_hypervectors[target] += sample
                    self.class_hypervectors[prediction] -= sample
                    mistakes += 1
            if mistakes == 0:
                break
        return self

    # ---------------------------------------------------------------- predict

    def predict(self, encoded: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
        """Return predicted class indices for a batch of encoded samples."""
        device = self.class_hypervectors.device
        outputs = []
        for start in range(0, len(encoded), batch_size):
            chunk = encoded[start : start + batch_size].to(
                device, dtype=self.class_hypervectors.dtype
            )
            outputs.append((chunk @ self.class_hypervectors.T).argmax(dim=1))
        return torch.cat(outputs) if outputs else torch.empty(0, dtype=torch.long, device=device)

    def accuracy(self, encoded: torch.Tensor, labels: torch.Tensor) -> float:
        predictions = self.predict(encoded)
        return (predictions == labels.to(predictions.device).long()).float().mean().item()


def average(models: list[HDClassifier]) -> HDClassifier:
    """Element-wise mean of a list of models: the FedAvg aggregation rule."""
    if not models:
        raise ValueError("cannot average an empty list of models")
    stacked = torch.stack([m.class_hypervectors for m in models], dim=0)
    return HDClassifier(stacked.mean(dim=0))
