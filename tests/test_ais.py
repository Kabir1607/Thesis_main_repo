"""Tests for certified Adaptive Importance Sampling (``dise.ais``).

Three groups:

1. Proposal math -- exact PMFs, likelihood-ratio identities, sampler
   correctness, CE bisection.
2. Off-policy betting CS -- numerical stability, running intersection,
   conservative compression, and a multi-seed *anytime* coverage audit
   under adaptive CE proposals and data-dependent stopping.
3. End-to-end DiSE integration on an unclosable-tail program.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dise import BoundedGeometric, Categorical, Uniform, estimate, failure_probability
from dise.ais import (
    OffPolicyBettingCS,
    TwistedBGProposal,
    solve_eta_for_mean,
    trunc_geom_log_z,
    trunc_geom_logpmf,
    trunc_geom_mean,
    trunc_geom_sample,
)
from dise.benchmarks.unclosable import make_monitor_nonlinear
from dise.distributions import ProductDistribution
from dise.regions import AxisAlignedBox, Status
from dise.scheduler import ASIPScheduler, SchedulerConfig
from dise.smt import default_backend

# ---------------------------------------------------------------------------
# 1. Proposal math
# ---------------------------------------------------------------------------

ETAS = [-3.0, -0.05, -1e-13, 0.0, 1e-9, 0.05, 2.0]


@pytest.mark.parametrize("M", [1, 2, 7, 300, 10_000])
@pytest.mark.parametrize("eta", ETAS)
def test_trunc_geom_pmf_sums_to_one(eta: float, M: int) -> None:
    lp = trunc_geom_logpmf(np.arange(M), eta, M)
    assert np.all(np.isfinite(lp))
    assert abs(np.exp(lp).sum() - 1.0) < 1e-12


@pytest.mark.parametrize("eta", [-60.0, -40.0, 40.0, 60.0])
def test_extreme_tilt_no_overflow_large_box(eta: float) -> None:
    """``r = e^{eta}`` far from 1 with M = 10^4: r^M would overflow."""
    M = 10_000
    assert math.isfinite(trunc_geom_log_z(eta, M))
    lp = trunc_geom_logpmf(np.arange(M), eta, M)
    assert np.all(np.isfinite(lp))
    assert abs(np.exp(lp).sum() - 1.0) < 1e-12
    j = trunc_geom_sample(np.random.default_rng(0), eta, M, 1000)
    assert j.min() >= 0 and j.max() <= M - 1
    expected_end = 0 if eta < 0 else M - 1
    assert np.all(j == expected_end)


@pytest.mark.parametrize("eta", ETAS)
@pytest.mark.parametrize("M", [2, 13, 5_000])
def test_trunc_geom_mean_exact(eta: float, M: int) -> None:
    j = np.arange(M)
    brute = float(np.dot(np.exp(trunc_geom_logpmf(j, eta, M)), j))
    assert abs(trunc_geom_mean(eta, M) - brute) < 1e-9 * max(1.0, M)


@pytest.mark.parametrize("M", [2, 17, 1_000, 10_000])
def test_ce_bisection_recovers_target_mean(M: int) -> None:
    for frac in [0.001, 0.1, 0.37, 0.5, 0.81, 0.999]:
        target = 0.05 + frac * (M - 1 - 0.1)
        eta = solve_eta_for_mean(target, M)
        assert abs(trunc_geom_mean(eta, M) - target) < 1e-6


def _proposal_cases():
    yield ProductDistribution({"x": BoundedGeometric(0.1, 100)}), {"x": (50, 100)}
    yield ProductDistribution({"x": BoundedGeometric(0.02, 10_000)}), {"x": (1, 10_000)}
    yield (
        ProductDistribution({"a": BoundedGeometric(0.3, 12), "b": Uniform(0, 9)}),
        {"a": (2, 11), "b": (3, 9)},
    )
    yield (
        ProductDistribution(
            {"c": Categorical((0.1, 0.0, 0.2, 0.3, 0.4)), "u": Uniform(1, 4)}
        ),
        {"c": (0, 4), "u": (1, 4)},
    )


def _box_points(bounds: dict[str, tuple[int, int]]) -> np.ndarray:
    grids = np.meshgrid(*[np.arange(lo, hi + 1) for lo, hi in bounds.values()], indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("case", list(range(4)))
def test_likelihood_ratio_identity_and_cap(case: int, alpha: float) -> None:
    """sum_x q(x) W(x) = 1 exactly, and 0 <= W <= 1/alpha everywhere."""
    dist, bounds = list(_proposal_cases())[case]
    rng = np.random.default_rng(case)
    eta = {v: float(rng.normal(0, 0.3)) for v in bounds}
    prop = TwistedBGProposal(dist, bounds, alpha=alpha, eta=eta)
    X = _box_points(bounds)
    q = np.exp(prop.log_mixture(X))
    W = prop.weights(X)
    assert abs(q.sum() - 1.0) < 1e-10
    assert abs((q * W).sum() - 1.0) < 1e-10
    assert np.all(W >= 0.0)
    assert np.all(W <= 1.0 / alpha + 1e-12)
    # D|_R itself: sum exp(log_base) = 1 and matches the distribution.
    base = np.exp(prop.log_base(X))
    assert abs(base.sum() - 1.0) < 1e-10
    region_mass = 1.0
    for v, (lo, hi) in bounds.items():
        region_mass *= dist.factors[v].mass(lo, hi)
    direct = np.array(
        [dist.pmf(dict(zip(bounds, row, strict=True))) for row in X]
    ) / region_mass
    assert np.allclose(base, direct, atol=1e-12)


@pytest.mark.parametrize("eta", [-0.15, 0.0, 0.15])
def test_sampler_matches_exact_pmf(eta: float) -> None:
    """50,000 mixture draws match the exact mixture PMF."""
    dist = ProductDistribution({"x": BoundedGeometric(0.2, 30)})
    bounds = {"x": (5, 24)}
    prop = TwistedBGProposal(dist, bounds, alpha=0.3, eta={"x": eta})
    n = 50_000
    draw = prop.sample(np.random.default_rng(42), n)
    X = _box_points(bounds)
    exact = np.exp(prop.log_mixture(X))
    counts = np.bincount(draw.x[:, 0] - 5, minlength=20) / n
    tol = 5.0 * np.sqrt(exact * (1 - exact) / n) + 1e-4
    assert np.all(np.abs(counts - exact) <= tol)
    # Weights reported by the sampler equal the exact ratio.
    assert np.allclose(draw.weights, prop.weights(draw.x))
    # Unbiasedness: E_q[W] = 1.
    assert abs(draw.weights.mean() - 1.0) < 0.02


def test_ce_update_moves_proposal_toward_failures() -> None:
    dist = ProductDistribution({"x": BoundedGeometric(0.05, 200)})
    prop = TwistedBGProposal(dist, {"x": (1, 200)}, alpha=0.5)
    x_fail = np.array([[150.0], [170.0], [190.0]])
    w_fail = np.array([0.2, 0.5, 0.3])
    tgt = prop.ce_target_eta(x_fail, w_fail)
    assert tgt is not None
    jbar = float(np.dot(w_fail, x_fail[:, 0] - 1) / w_fail.sum())
    assert abs(trunc_geom_mean(float(tgt[0]), 200) - jbar) < 1e-6
    before = trunc_geom_mean(float(prop.eta[0]), 200)
    assert prop.update_ce(x_fail, w_fail, rho=1.0)
    assert trunc_geom_mean(float(prop.eta[0]), 200) > before
    # No failures -> no update.
    assert not prop.update_ce(np.empty((0, 1)), np.empty(0))


# ---------------------------------------------------------------------------
# 2. Off-policy betting CS
# ---------------------------------------------------------------------------


def _tail_leaf(p_geo: float = 0.05, N: int = 200, c: int = 91):
    """BG(p, N) leaf on {1..N}; failure iff x >= c.  Returns (dist, bounds,
    c, exact failure probability)."""
    d = BoundedGeometric(p_geo, N)
    p_fail = d.mass(c, N)
    return ProductDistribution({"x": d}), {"x": (1, N)}, c, p_fail


def test_cs_stability_100k_samples() -> None:
    dist, bounds, c, p = _tail_leaf()
    rng = np.random.default_rng(0)
    prop = TwistedBGProposal(dist, bounds, alpha=0.2, eta={"x": 0.01})
    cs = OffPolicyBettingCS(0.05)
    for _ in range(20):
        d = prop.sample(rng, 5_000)
        cs.update(d.weights * (d.x[:, 0] >= c), d.alpha)
        L, U = cs.bounds()
        assert math.isfinite(L) and math.isfinite(U)
        assert 0.0 <= L <= U <= 1.0
    assert cs.n == 100_000
    for m in [1e-12, 1e-6, p, 0.5, 1 - 1e-12]:
        assert math.isfinite(cs.log_wealth_lower(m))
        assert math.isfinite(cs.log_wealth_upper(m))
    assert L <= p <= U


def test_running_intersection_monotone() -> None:
    dist, bounds, c, _ = _tail_leaf()
    rng = np.random.default_rng(1)
    prop = TwistedBGProposal(dist, bounds, alpha=0.5)
    cs = OffPolicyBettingCS(0.05)
    prev_L, prev_U = 0.0, 1.0
    for _ in range(40):
        d = prop.sample(rng, 100)
        cs.update(d.weights * (d.x[:, 0] >= c), d.alpha)
        L, U = cs.bounds()
        assert L >= prev_L - 1e-15
        assert U <= prev_U + 1e-15
        prev_L, prev_U = L, U


def test_wealth_monotone_in_m() -> None:
    dist, bounds, c, _ = _tail_leaf()
    rng = np.random.default_rng(2)
    prop = TwistedBGProposal(dist, bounds, alpha=0.3)
    cs = OffPolicyBettingCS(0.05)
    d = prop.sample(rng, 2_000)
    cs.update(d.weights * (d.x[:, 0] >= c), d.alpha)
    ms = np.exp(np.linspace(math.log(1e-8), math.log(0.99), 60))
    lw = [cs.log_wealth_lower(m) for m in ms]
    uw = [cs.log_wealth_upper(m) for m in ms]
    assert all(a >= b - 1e-9 for a, b in zip(lw, lw[1:], strict=False))
    assert all(a <= b + 1e-9 for a, b in zip(uw, uw[1:], strict=False))


def test_binned_wealth_is_conservative() -> None:
    """Histogram compression only ever lowers the wealth (sound)."""
    dist, bounds, c, p = _tail_leaf()
    rng = np.random.default_rng(3)
    prop = TwistedBGProposal(dist, bounds, alpha=0.25, eta={"x": 0.02})
    ms_lo = [0.3 * p, 0.5 * p, 0.9 * p]
    ms_hi = [1.2 * p, 2.0 * p, 5.0 * p]
    cs = OffPolicyBettingCS(0.05, track_lower=ms_lo, track_upper=ms_hi)
    for _ in range(10):
        d = prop.sample(rng, 500)
        cs.update(d.weights * (d.x[:, 0] >= c), d.alpha)
    exact_lo = cs.exact_log_wealth_lower()
    exact_hi = cs.exact_log_wealth_upper()
    # Rounding Z by at most a factor (1 + bin_rel) changes each lower-bound
    # log-factor by <= log(1 + bin_rel) and each upper-bound log-factor by
    # <= gamma_max / (1 - gamma_max) * bin_rel * W_max / (W_max - m).
    h = math.log1p(cs.bin_rel)
    w_max = 1.0 / 0.25
    for i, m in enumerate(ms_lo):
        binned = cs.log_wealth_lower(m)
        assert binned <= exact_lo[i] + 1e-9
        assert binned >= exact_lo[i] - cs.n_positive * h - 1e-9
    for i, m in enumerate(ms_hi):
        binned = cs.log_wealth_upper(m)
        slack = cs.n_positive * (0.95 / 0.05) * cs.bin_rel * w_max / (w_max - m)
        assert binned <= exact_hi[i] + 1e-9
        assert binned >= exact_hi[i] - slack - 1e-9


def test_one_sided_modes() -> None:
    dist, bounds, c, p = _tail_leaf()
    rng = np.random.default_rng(4)
    prop = TwistedBGProposal(dist, bounds, alpha=0.5)
    d = prop.sample(rng, 3_000)
    z = d.weights * (d.x[:, 0] >= c)
    fal = OffPolicyBettingCS(0.05, mode="falsify")
    ver = OffPolicyBettingCS(0.05, mode="verify")
    two = OffPolicyBettingCS(0.05, mode="two_sided")
    for cs in (fal, ver, two):
        cs.update(z, d.alpha)
    assert fal.bounds()[1] == 1.0 and fal.bounds()[0] > 0.0
    assert ver.bounds()[0] == 0.0 and ver.bounds()[1] < 1.0
    # One-sided bounds spend the whole budget on their side: never looser.
    assert fal.bounds()[0] >= two.bounds()[0]
    assert ver.bounds()[1] <= two.bounds()[1]


def test_bernoulli_special_case_alpha_one() -> None:
    """alpha = 1 (W = 1): a valid betting CS for plain Bernoulli data."""
    rng = np.random.default_rng(5)
    p = 0.2
    cs = OffPolicyBettingCS(0.05)
    for _ in range(20):
        cs.update((rng.random(200) < p).astype(float), 1.0)
    L, U = cs.bounds()
    assert L <= p <= U
    assert U - L < 0.1


def test_update_rejects_out_of_range() -> None:
    cs = OffPolicyBettingCS(0.05)
    with pytest.raises(ValueError):
        cs.update([3.0], alpha=0.5)  # exceeds W_max = 2
    with pytest.raises(ValueError):
        cs.update([-0.1], alpha=0.5)


def test_anytime_coverage_adaptive_ce_data_dependent_stopping() -> None:
    """CRITICAL soundness audit.

    200 seeds at delta = 0.05.  Each run adapts the CE proposal between
    batches and stops *as soon as* the running interval excludes 0.5 p or
    2 p (or at the budget cap).  Empirical miscoverage -- both at the
    stopping time and at *any* checkpoint along the path -- must stay
    within binomial tolerance of delta, with no catastrophic violation.
    """
    dist, bounds, c, p = _tail_leaf()
    delta = 0.05
    n_seeds = 200
    budget = 3_000
    batch = 50
    alphas = [0.1, 0.5, 0.9]
    miss_at_stop = 0
    miss_any_time = 0
    catastrophic = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(10_000 + seed)
        prop = TwistedBGProposal(dist, bounds, alpha=alphas[seed % 3])
        cs = OffPolicyBettingCS(delta)
        fx: list[np.ndarray] = []
        fw: list[float] = []
        ever_missed = False
        while cs.n < budget:
            if fx:
                prop.update_ce(np.vstack(fx), np.asarray(fw))  # predictable
            d = prop.sample(rng, batch)
            r = d.x[:, 0] >= c
            cs.update(d.weights * r, d.alpha)
            fx.extend(d.x[r].astype(float))
            fw.extend(d.weights[r].tolist())
            L, U = cs.bounds()
            if not (L <= p <= U):
                ever_missed = True
            if L > 2 * p or U < 0.5 * p:
                catastrophic += 1
                break
            if not (L <= 0.5 * p <= U) or not (L <= 2 * p <= U):
                break
        L, U = cs.bounds()
        miss_at_stop += int(not (L <= p <= U))
        miss_any_time += int(ever_missed)
    assert catastrophic == 0
    assert miss_at_stop / n_seeds <= 0.08
    assert miss_any_time / n_seeds <= 0.08


# ---------------------------------------------------------------------------
# 3. End-to-end DiSE integration
# ---------------------------------------------------------------------------


def _run_monitor(enable_ais: bool, seed: int = 0, **kw):
    bench = make_monitor_nonlinear()
    cfg = SchedulerConfig(
        epsilon=1e-6,
        delta=0.05,
        budget_samples=3_000,
        method="betting",
        enable_ais=enable_ais,
        **kw,
    )
    sched = ASIPScheduler(
        program=bench.program,
        distribution=ProductDistribution(bench.distribution),
        property_fn=bench.property_fn,
        smt=default_backend(),
        config=cfg,
        rng=np.random.default_rng(seed),
    )
    return bench, sched.run()


def _leaf_for_box(frontier, lo: int, hi: int):
    for leaf in frontier.leaves():
        if isinstance(leaf.region, AxisAlignedBox) and leaf.region.bounds["x"] == (lo, hi):
            return leaf
    raise AssertionError(f"no leaf with box [{lo}, {hi}]")


def test_monitor_ais_only_in_unclosable_leaf() -> None:
    bench, res = _run_monitor(enable_ais=True, ais_alpha=0.5)
    safe = _leaf_for_box(res.frontier, 1, 49)
    tail = _leaf_for_box(res.frontier, 50, 100)
    # x < 50 is closed symbolically by Z3 -- no AIS spent there.
    assert safe.status == Status.CLOSED_TRUE
    assert safe.n_ais_samples == 0
    assert safe.ais_proposal is None
    # AIS ran only in the unclosable x >= 50 leaf.
    assert tail.status == Status.OPEN
    assert tail.is_unclosable and tail.has_concretized_obs
    assert tail.n_ais_samples > 0
    assert res.ais_samples_used == tail.n_ais_samples
    # Certified interval contains the exact ground truth.
    lo, hi = res.final_estimator.interval
    assert lo <= bench.mu <= hi
    assert res.final_estimator.method == "ais_betting"
    # Asymmetric bounds come straight from the CS.
    L_pi, U_pi = tail.ais_cs.reliability_bounds()
    assert abs(hi - (safe.w_hat + tail.w_hat * U_pi)) < 1e-12
    assert abs(lo - (safe.w_hat + tail.w_hat * L_pi)) < 1e-12


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_monitor_ais_covers_truth_multiple_seeds(seed: int) -> None:
    bench, res = _run_monitor(enable_ais=True, seed=seed, ais_alpha=0.5)
    lo, hi = res.final_estimator.interval
    assert lo <= bench.mu <= hi


def test_baseline_untouched_without_opt_in() -> None:
    """Default config: no AIS state anywhere, baseline estimator method."""
    _, res = _run_monitor(enable_ais=False)
    assert res.ais_samples_used == 0
    assert res.final_estimator.method == "betting"
    assert res.decision is None
    for node in res.frontier.all_nodes():
        assert node.ais_cs is None
        assert node.ais_proposal is None
        assert not node.is_unclosable


def test_threshold_falsification_end_to_end() -> None:
    bench = make_monitor_nonlinear()
    res = estimate(
        bench.program,
        bench.distribution,
        bench.property_fn,
        epsilon=1e-9,
        budget=5_000,
        seed=0,
        enable_ais=True,
        ais_alpha=0.1,
        ais_mode="falsify",
        threshold_tau=bench.tau,
    )
    assert res.decision == "FALSIFIED"
    assert res.terminated_reason == "threshold_falsified"
    lo, hi = res.interval
    assert lo <= bench.mu <= hi
    assert 1.0 - hi > bench.tau


def test_method_ais_betting_enables_ais() -> None:
    bench = make_monitor_nonlinear()
    res = estimate(
        bench.program, bench.distribution, bench.property_fn,
        epsilon=1e-6, budget=1_500, seed=0, method="ais_betting",
    )
    assert res.ais_samples_used > 0


def test_failure_probability_sign_convention_with_ais() -> None:
    """failure_probability's mu is the FAILURE probability; with AIS on it
    must still be returned in failure space and cover the truth."""

    def prog(x):
        y = x | 0
        assert not (y >= 40 and y % 3 == 0)
        return y

    dist = {"x": BoundedGeometric(0.1, 60)}
    truth = sum(dist["x"].pmf(k) for k in range(40, 61) if k % 3 == 0)
    res = failure_probability(
        prog, dist, epsilon=1e-6, budget=2_000, seed=0, enable_ais=True,
    )
    lo, hi = res.interval
    assert lo <= truth <= hi
    assert hi < 0.5  # failure space, not reliability space


def test_betting_strict_covers_and_is_opt_in() -> None:
    bench = make_monitor_nonlinear()
    res = estimate(
        bench.program, bench.distribution, bench.property_fn,
        epsilon=1e-6, budget=1_000, seed=0, method="betting_strict",
    )
    lo, hi = res.interval
    assert lo <= bench.mu <= hi
    assert res.ais_samples_used == 0
