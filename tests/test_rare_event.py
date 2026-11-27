"""Soundness and correctness tests for the score-based rare-event
baselines (importance sampling, cross-entropy, sequential Monte Carlo)."""

from __future__ import annotations

import numpy as np
import pytest

from dise.baselines.rare_event import (
    CrossEntropy,
    ImportanceSampling,
    RareEventResult,
    SequentialMonteCarlo,
    _defensive_weight,
    _tilt_from,
)
from dise.distributions import Geometric


def _geom_tail(p: float, T: int) -> float:
    """Exact P(X >= T) for X ~ Geometric(p) on {1, 2, ...}."""
    return (1.0 - p) ** (T - 1)


def _single_var_setup(p: float, T: int):
    dist = {"a": Geometric(p)}
    score = lambda x: float(x["a"])  # noqa: E731
    return dist, score, float(T)


# ---------------------------------------------------------------------------
# Importance sampling
# ---------------------------------------------------------------------------


def test_is_covers_analytic_truth() -> None:
    p, T = 0.3, 12
    truth = _geom_tail(p, T)
    dist, score, thr = _single_var_setup(p, T)
    est = ImportanceSampling(p_tilt=0.05, alpha=0.5)
    # Coverage should hold across seeds (a (1 - delta) interval may miss
    # occasionally, so require the large majority to cover).
    covered = 0
    for seed in range(20):
        r = est.run(dist, score, thr, budget=4000, delta=0.05, seed=seed)
        lo, hi = r.interval
        covered += lo <= truth <= hi
    assert covered >= 18  # >= 90% empirical coverage at delta = 0.05


def test_is_weight_bound_respected() -> None:
    p, alpha = 0.3, 0.5
    dist = {"a": Geometric(p), "b": Geometric(p)}
    tilt = _tilt_from(dist, 0.05)
    bound = 1.0 / (1.0 - alpha)
    rng = np.random.default_rng(0)
    for _ in range(500):
        x = {"a": int(rng.integers(1, 200)), "b": int(rng.integers(1, 200))}
        w = _defensive_weight(dist, tilt, alpha, x)
        assert 0.0 <= w <= bound + 1e-12


def test_is_point_estimate_roughly_unbiased() -> None:
    p, T = 0.3, 10
    truth = _geom_tail(p, T)
    dist, score, thr = _single_var_setup(p, T)
    est = ImportanceSampling(p_tilt=0.05, alpha=0.5)
    mus = [
        est.run(dist, score, thr, budget=6000, delta=0.05, seed=s).mu_hat
        for s in range(15)
    ]
    assert abs(float(np.mean(mus)) - truth) < 0.25 * truth


def test_is_certified_flag() -> None:
    dist, score, thr = _single_var_setup(0.3, 8)
    r = ImportanceSampling().run(dist, score, thr, 2000, 0.05, 0)
    assert isinstance(r, RareEventResult)
    assert r.certified is True


# ---------------------------------------------------------------------------
# Cross-entropy
# ---------------------------------------------------------------------------


def test_ce_covers_and_charges_all_samples() -> None:
    p, T = 0.3, 14
    truth = _geom_tail(p, T)
    dist, score, thr = _single_var_setup(p, T)
    est = CrossEntropy(alpha=0.5, n_round=300)
    covered = 0
    for seed in range(20):
        r = est.run(dist, score, thr, budget=6000, delta=0.05, seed=seed)
        lo, hi = r.interval
        covered += lo <= truth <= hi
        # samples_used includes adaptation plus the final certified pass.
        assert r.extras["adapt_samples"] + r.extras["final_samples"] == r.samples_used
    assert covered >= 18


def test_ce_adapts_toward_the_tail() -> None:
    # On a rare right-tail event the tuned tilt should have a smaller p than
    # a naive match (heavier tail => more mass on large values).
    p, T = 0.2, 30
    dist, score, thr = _single_var_setup(p, T)
    r = CrossEntropy(alpha=0.5, n_round=400).run(dist, score, thr, 8000, 0.05, 0)
    tuned = r.extras["tuned_p_tilt"]["a"]
    assert tuned < p  # heavier tail than the base distribution


# ---------------------------------------------------------------------------
# Sequential Monte Carlo
# ---------------------------------------------------------------------------


def test_smc_is_marked_asymptotic() -> None:
    dist, score, thr = _single_var_setup(0.3, 12)
    r = SequentialMonteCarlo(n_particles=800).run(dist, score, thr, 0, 0.05, 0)
    assert r.certified is False
    assert "not a finite-sample" in r.extras["note"]


def test_smc_estimate_in_right_ballpark() -> None:
    p, T = 0.3, 14
    truth = _geom_tail(p, T)
    dist, score, thr = _single_var_setup(p, T)
    mus = [
        SequentialMonteCarlo(n_particles=1000, p0=0.1)
        .run(dist, score, thr, 0, 0.05, seed=s)
        .mu_hat
        for s in range(8)
    ]
    med = float(np.median(mus))
    # Subset simulation is consistent but biased at finite N; require the
    # median within a factor of ~2.5 of the truth.
    assert 0.4 * truth <= med <= 2.5 * truth


# ---------------------------------------------------------------------------
# event_fn threading (event distinct from the steering score)
# ---------------------------------------------------------------------------


def test_event_fn_overrides_threshold() -> None:
    # Event is stricter than score >= threshold: also require a even.
    p, T = 0.3, 8
    dist, score, thr = _single_var_setup(p, T)
    event = lambda x: x["a"] >= T and x["a"] % 2 == 0  # noqa: E731
    truth = sum(
        Geometric(p).pmf(k) for k in range(T, 5000) if k % 2 == 0
    )
    est = ImportanceSampling(p_tilt=0.05, alpha=0.5)
    covered = 0
    for seed in range(20):
        r = est.run(dist, score, thr, 5000, 0.05, seed, event_fn=event)
        lo, hi = r.interval
        covered += lo <= truth <= hi
    assert covered >= 17


@pytest.mark.parametrize("bad", [-0.1, 0.0, 1.0, 1.5])
def test_is_rejects_bad_params(bad: float) -> None:
    with pytest.raises(ValueError):
        ImportanceSampling(p_tilt=bad)
    with pytest.raises(ValueError):
        ImportanceSampling(alpha=bad)
