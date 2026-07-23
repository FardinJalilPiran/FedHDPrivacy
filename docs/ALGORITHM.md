# Algorithm reference

Notation follows the paper.

| Symbol | Meaning | Code |
| --- | --- | --- |
| `D` | hypervector size | `config.dimensions` |
| `K` | number of clients | `config.n_clients` |
| `L` | new samples per client per round | `history.samples_per_round` |
| `R` | communication rounds | `config.rounds` |
| `S` | number of classes | `dataset.n_classes` |
| `ε` | privacy budget | `config.epsilon` |
| `δ` | privacy loss threshold | set to 1 / (samples seen) |
| `ξ` | noise required at this round | `required_noise_variance` |
| `Ψ` | noise inherited from the global model | `cumulative_noise_variance` |
| `Γ` | noise actually injected (`ξ − Ψ`) | `additional_noise_variance` |

## Hyperdimensional computing

**Encoding.** A sample `x` is mapped to a bipolar hypervector via a random
projection with a non-linearity, then binarised to `{−1, +1}`
(`hdc.Encoder`). Nearby inputs map to nearby hypervectors — the locality
property everything else depends on.

**Training.** Class hypervector `s` is the sum of every encoded sample labelled
`s` (`HDClassifier.bundle`). One pass, no gradients.

**Inference.** Score the query against every class hypervector and take the
argmax (`HDClassifier.predict`). The paper writes this as cosine similarity;
since all hypervectors have the same norm, the dot product gives identical
rankings, so that is what the code computes.

**Retraining.** On a misclassification, add the sample's hypervector to its true
class and subtract it from the predicted one (`HDClassifier.retrain`). This is
what makes the model continual: new data refines the existing model instead of
requiring a retrain from scratch. Updates are sequential — each correction
changes the boundary the next sample sees.

## Differential privacy

The Gaussian mechanism satisfies `(ε, δ)`-DP when

```
σ² > 2·ln(1.25/δ)·Δf²/ε²
```

**Sensitivity.** Two neighbouring datasets differ by one sample, so two class
hypervectors differ by one hypervector. Hypervectors are binary with `D`
components, so `Δf = √D` for a local model. For the aggregate, averaging over
`K` clients gives `Δf = √D / K`.

**δ.** Set to the inverse of the number of samples the model has seen, following
Saifullah et al. Worst case, all samples belong to one class, so the same `δ`
is applied to every class.

### Round 1 (Theorems 2 and 3)

A client has seen `L` samples, so `δ = 1/L`:

```
ξ¹ₖ ~ N(0, (2D/ε²)·ln(1.25L))
Γ¹ₖ = ξ¹ₖ                        (nothing to inherit)
```

The server averages `K` noisy models. The aggregate noise variance is
`(2D/Kε²)·ln(1.25L)`, while the requirement given sensitivity `√D/K` and
`δ = 1/(KL)` is `(2D/K²ε²)·ln(1.25KL)`. Their ratio

```
γ = K·ln(1.25L) / ln(1.25KL)
```

is increasing in both `K` and `L` with minimum `2·ln(2.5)/ln(5) ≈ 1.13 > 1`
(Lemma 1). The aggregate is already private — **the server adds nothing**.

### Round r ≥ 2 (Theorems 4 and 5)

A client downloads the global model (containing `(r−1)KL` samples) and retrains
on `L` fresh ones, so `δ = 1/((r−1)KL + L)`:

```
ξʳₖ ~ N(0, (2D/ε²)·ln(1.25[(r−1)KL + L]))
Ψʳ⁻¹ₖ ~ N(0, (2D/Kε²)·ln(1.25[(r−2)KL + L]))
Γʳₖ = ξʳₖ − Ψʳ⁻¹ₖ
```

Lemma 2 shows the server again needs to add nothing, with
`γ = K·ln(1.25[(r−1)KL + L]) / ln(1.25KLr) > 1` for `K, L, r ≥ 2`.

**This difference is the contribution.** Naively re-noising every round would
inject `ξʳₖ` each time; accounting for `Ψʳ⁻¹ₖ` injects strictly less from round 2
onward, and the saving grows with `r`. Over 50 rounds the paper reports the
injected noise falling to 80.03% of the requirement with 5 clients and 90% with
10 — roughly a 20% and 10% reduction respectively.

## One communication round

```
for each client k in parallel:
    receive the global model
    encode this round's L new samples
    r == 1 → bundle into fresh class hypervectors
    r  > 1 → retrain the downloaded global model
    perturb with N(0, Γʳₖ)
    send

server:
    global model ← element-wise mean of the K noisy local models
    (adds no noise of its own)
```

Implemented in `federated.run_federated_training`.

## Threat model

Noise is applied to local models *before* transmission, which covers all four
attack surfaces in Figure 1 of the paper:

| Attack | Why it fails |
| --- | --- |
| Eavesdropping | intercepted models are noisy |
| Malicious participant | the global model it receives is noisy |
| Server compromise | the global model at rest is noisy |
| Untrusted server | local models arrive already noised |

Out of scope, and named as future work in the paper: model poisoning,
free-riding, and collusion among clients.

## Complexity

Per client per round, with `L` samples, `S` classes, `E` retraining epochs:

| Stage | Cost |
| --- | --- |
| Encoding | `O(L · n_features · D)` |
| Bundling | `O(L · D)` |
| Retraining | `O(E · L · S · D)` |
| Perturbation | `O(S · D)` |
| Communication | `O(S · D)` per direction |

Everything is linear in `D`, which is why the accuracy/energy trade-off in
Figure 11 is close to linear in the hypervector size.
