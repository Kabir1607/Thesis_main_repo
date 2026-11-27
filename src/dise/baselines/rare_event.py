"""Score-based rare-event estimators: importance sampling, the
cross-entropy method, and sequential Monte Carlo (subset simulation).

Each estimates ``mu = Pr_{X ~ D}[score(X) >= T]`` for a rare threshold
``T`` over an integer distribution ``D`` with possibly unbounded support.
Unlike the black-box Boolean :class:`~dise.baselines.Baseline` protocol,
they consume the real-valued ``score`` and ``D`` directly to steer
sampling toward the event; they are the statistical competitors to DiSE's
certified importance-sampling instantiation in the ``beta > 0`` regime,
compared in ``experiments/run_nonenumerable.py``.

Interval guarantees differ by method:

* :class:`ImportanceSampling` and :class:`CrossEntropy` use a defensive
  mixture ``q = (1 - alpha) D + alpha G``. Since ``q >= (1 - alpha) D``,
  the weight ``w = D / q`` is bounded by ``1 / (1 - alpha)`` (Hesterberg
  1995), so a Maurer-Pontil empirical-Bernstein half-width on
  ``w * 1[event]`` is a genuine finite-sample ``(1 - delta)`` certificate.
* :class:`SequentialMonteCarlo` (Au-Beck 2001) reports the standard
  coefficient-of-variation interval, which is asymptotic (CLT), not a
  finite-sample certificate. ``certified`` is ``False`` for its results.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..distributions import Distribution, Geometric
from ..estimator import empirical_bernstein_halfwidth_mp

ScoreFn = Callable[[Mapping[str, int]], float]
EventFn = Callable[[Mapping[str, int]], bool]


def _event_from(
    score_fn: ScoreFn, threshold: float, event_fn: EventFn | None
) -> EventFn:
    """Resolve the hit indicator. When ``event_fn`` is given it defines the
    rare event directly and ``score`` is used only for *steering* (the IS
    tilt direction and the SMC intermediate levels). Otherwise the event is
    the pure threshold crossing ``score(x) >= threshold``.
    """
    if event_fn is not None:
        return event_fn
    return lambda x: score_fn(x) >= threshold


@dataclass
class RareEventResult:
    """Return type for the score-based rare-event estimators.

    ``certified`` records whether ``interval`` is a finite-sample
    ``(1 - delta)`` bound (True for IS/CE) or an asymptotic CLT interval
    (False for SMC). Downstream code must not treat an asymptotic interval
    as a certificate.
    """

    name: str
    mu_hat: float
    interval: tuple[float, float]
    samples_used: int
    wall_clock_s: float
    delta: float
    certified: bool
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def half_width(self) -> float:
        lo, hi = self.interval
        return (hi - lo) / 2.0

    def __repr__(self) -> str:
        lo, hi = self.interval
        tag = "cert" if self.certified else "asymp"
        return (
            f"RareEventResult(name={self.name!r}, mu_hat={self.mu_hat:.3e}, "
            f"interval=[{lo:.3e}, {hi:.3e}], {tag}, samples={self.samples_used})"
        )


def _factors(distribution: Mapping[str, Distribution]) -> dict[str, Distribution]:
    return dict(distribution)


def _pmf_product(
    factors: Mapping[str, Distribution], x: Mapping[str, int]
) -> float:
    p = 1.0
    for k, d in factors.items():
        p *= d.pmf(int(x[k]))
    return p


# ---------------------------------------------------------------------------
# Geometric tilt: the proposal family shared by IS and CE.
# ---------------------------------------------------------------------------


def _tilt_from(base: Mapping[str, Distribution], p_tilt: float) -> dict[str, Geometric]:
    """A per-coordinate :class:`Geometric` proposal with success
    probability ``p_tilt``. A smaller ``p_tilt`` has a heavier tail, so it
    places more mass on large coordinate values and hence on large scores.

    Only the shape matters for the defensive-mixture weight bound; the tilt
    need not match the base distribution's family.
    """
    return {k: Geometric(p=p_tilt) for k in base}


def _draw_defensive(
    base_factors: Mapping[str, Distribution],
    tilt_factors: Mapping[str, Geometric],
    alpha: float,
    n: int,
    rng: np.random.Generator,
) -> list[dict[str, int]]:
    """Draw ``n`` samples from the mixture ``(1 - alpha) * D + alpha * G``.

    Each sample independently comes from ``G`` with probability ``alpha``
    and from ``D`` otherwise, coordinate-wise.
    """
    keys = list(base_factors.keys())
    use_tilt = rng.random(n) < alpha
    base_draw = {k: base_factors[k].sample(rng, n) for k in keys}
    tilt_draw = {k: tilt_factors[k].sample(rng, n) for k in keys}
    out: list[dict[str, int]] = []
    for i in range(n):
        if use_tilt[i]:
            out.append({k: int(tilt_draw[k][i]) for k in keys})
        else:
            out.append({k: int(base_draw[k][i]) for k in keys})
    return out


def _defensive_weight(
    base_factors: Mapping[str, Distribution],
    tilt_factors: Mapping[str, Geometric],
    alpha: float,
    x: Mapping[str, int],
) -> float:
    """Importance weight ``w(x) = D(x) / q(x)`` for the defensive mixture
    ``q = (1 - alpha) * D + alpha * G``. Bounded above by ``1 / (1 - alpha)``.
    """
    d = _pmf_product(base_factors, x)
    g = _pmf_product(tilt_factors, x)
    q = (1.0 - alpha) * d + alpha * g
    if q <= 0.0:
        return 0.0
    return d / q


def _certified_is(
    base: Mapping[str, Distribution],
    tilt: Mapping[str, Geometric],
    alpha: float,
    is_hit: EventFn,
    n: int,
    delta: float,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    """One defensive-mixture IS pass. Returns ``(mu_hat, half_width,
    n_hits)`` with a Maurer-Pontil empirical-Bernstein ``(1 - delta)``
    half-width on the bounded estimand ``w * 1[event]``.
    """
    range_bound = 1.0 / (1.0 - alpha)
    ys = np.empty(n, dtype=np.float64)
    for i, x in enumerate(_draw_defensive(base, tilt, alpha, n, rng)):
        ys[i] = _defensive_weight(base, tilt, alpha, x) if is_hit(x) else 0.0
    mu_hat = float(ys.mean()) if n > 0 else 0.0
    var = float(ys.var(ddof=1)) if n > 1 else 0.0
    half = empirical_bernstein_halfwidth_mp(var, n, delta, range_bound)
    return mu_hat, half, int((ys > 0).sum())


# ---------------------------------------------------------------------------
# Importance sampling with a defensive mixture proposal.
# ---------------------------------------------------------------------------


class ImportanceSampling:
    """Fixed-proposal importance sampling with a defensive-mixture weight
    bound and a Maurer-Pontil empirical-Bernstein certified interval.

    Parameters
    ----------
    p_tilt : float
        Success probability of the per-coordinate geometric tilt. Smaller
        means a heavier tail (more mass on large scores). Should be well
        below the base ``p`` to help on right-tail rare events.
    alpha : float
        Mixture weight on the tilt. The importance weight is bounded by
        ``1 / (1 - alpha)``; larger ``alpha`` steers harder but widens the
        weight range (and hence the empirical-Bernstein constant).
    """

    name = "importance_sampling"

    def __init__(self, p_tilt: float = 0.01, alpha: float = 0.5) -> None:
        if not (0.0 < p_tilt < 1.0):
            raise ValueError("p_tilt must be in (0, 1)")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.p_tilt = p_tilt
        self.alpha = alpha

    def run(
        self,
        distribution: Mapping[str, Distribution],
        score_fn: ScoreFn,
        threshold: float,
        budget: int,
        delta: float,
        seed: int,
        event_fn: EventFn | None = None,
    ) -> RareEventResult:
        rng = np.random.default_rng(seed)
        base = _factors(distribution)
        is_hit = _event_from(score_fn, threshold, event_fn)
        tilt = _tilt_from(base, self.p_tilt)
        t0 = time.perf_counter()
        mu_hat, half, n_hits = _certified_is(
            base, tilt, self.alpha, is_hit, budget, delta, rng
        )
        wall = time.perf_counter() - t0
        return RareEventResult(
            name=self.name,
            mu_hat=mu_hat,
            interval=(max(0.0, mu_hat - half), min(1.0, mu_hat + half)),
            samples_used=budget,
            wall_clock_s=wall,
            delta=delta,
            certified=True,
            extras={
                "p_tilt": self.p_tilt,
                "alpha": self.alpha,
                "weight_bound": 1.0 / (1.0 - self.alpha),
                "n_hits": n_hits,
                "half_width": half,
            },
        )


# ---------------------------------------------------------------------------
# Cross-entropy method: adapt the tilt, then a certified final IS pass.
# ---------------------------------------------------------------------------


class CrossEntropy:
    """The cross-entropy method (Rubinstein-Kroese) for the right-tail
    event ``score(X) >= T``.

    Adaptation rounds tune a per-coordinate geometric tilt toward the
    elite (top-``rho``) score region, raising an intermediate level
    ``gamma`` toward ``T``. The certified estimate is a *final* defensive-
    mixture importance-sampling pass with the tuned tilt, independent of
    the adaptation samples, so the empirical-Bernstein interval is a valid
    finite-sample ``(1 - delta)`` bound. All samples (adaptation plus the
    final pass) are charged to ``samples_used``.
    """

    name = "cross_entropy"

    def __init__(
        self,
        rho: float = 0.1,
        alpha: float = 0.5,
        n_round: int = 400,
        max_rounds: int = 12,
        p_tilt_floor: float = 1e-4,
    ) -> None:
        if not (0.0 < rho < 1.0):
            raise ValueError("rho must be in (0, 1)")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.rho = rho
        self.alpha = alpha
        self.n_round = n_round
        self.max_rounds = max_rounds
        self.p_tilt_floor = p_tilt_floor

    def _fit_tilt(
        self,
        base: Mapping[str, Distribution],
        score_fn: ScoreFn,
        threshold: float,
        rng: np.random.Generator,
    ) -> tuple[dict[str, Geometric], int, int]:
        """Return ``(tuned_tilt, samples_spent, rounds)``.

        Starts from the base distribution's own mean as the initial tilt and
        walks the elite quantile toward ``T``. Falls back to the last tilt
        if the level stops advancing.
        """
        keys = list(base.keys())
        # Initial tilt: match the base per-coordinate mean where finite,
        # else a mild heavy tail.
        tilt = _tilt_from(base, min(0.5, max(self.p_tilt_floor, 0.2)))
        spent = 0
        rounds = 0
        for _ in range(self.max_rounds):
            rounds += 1
            draws = _draw_defensive(base, tilt, self.alpha, self.n_round, rng)
            spent += self.n_round
            scores = np.array([score_fn(x) for x in draws], dtype=np.float64)
            gamma = float(np.quantile(scores, 1.0 - self.rho))
            gamma = min(gamma, threshold)
            elite_mask = scores >= gamma
            if elite_mask.sum() < 2:
                break
            elite = [draws[i] for i in range(len(draws)) if elite_mask[i]]
            # Geometric MLE per coordinate on the elite set: p = 1 / mean(x).
            new_tilt: dict[str, Geometric] = {}
            for k in keys:
                vals = np.array([e[k] for e in elite], dtype=np.float64)
                mean_v = max(1.0, float(vals.mean()))
                p_new = min(0.5, max(self.p_tilt_floor, 1.0 / mean_v))
                new_tilt[k] = Geometric(p=p_new)
            tilt = new_tilt
            if gamma >= threshold:
                break
        return tilt, spent, rounds

    def run(
        self,
        distribution: Mapping[str, Distribution],
        score_fn: ScoreFn,
        threshold: float,
        budget: int,
        delta: float,
        seed: int,
        event_fn: EventFn | None = None,
    ) -> RareEventResult:
        rng = np.random.default_rng(seed)
        base = _factors(distribution)
        is_hit = _event_from(score_fn, threshold, event_fn)
        t0 = time.perf_counter()
        tilt, adapt_spent, rounds = self._fit_tilt(base, score_fn, threshold, rng)
        final_n = max(1, budget - adapt_spent)
        mu_hat, half, n_hits = _certified_is(
            base, tilt, self.alpha, is_hit, final_n, delta, rng
        )
        wall = time.perf_counter() - t0
        return RareEventResult(
            name=self.name,
            mu_hat=mu_hat,
            interval=(max(0.0, mu_hat - half), min(1.0, mu_hat + half)),
            samples_used=adapt_spent + final_n,
            wall_clock_s=wall,
            delta=delta,
            certified=True,
            extras={
                "alpha": self.alpha,
                "adapt_samples": adapt_spent,
                "final_samples": final_n,
                "ce_rounds": rounds,
                "tuned_p_tilt": {k: g.p for k, g in tilt.items()},
                "weight_bound": 1.0 / (1.0 - self.alpha),
                "n_hits": n_hits,
                "half_width": half,
            },
        )


# ---------------------------------------------------------------------------
# Sequential Monte Carlo / subset simulation (asymptotic interval).
# ---------------------------------------------------------------------------


class SequentialMonteCarlo:
    """Subset simulation (Au-Beck 2001): estimate the rare-event
    probability as a product of conditional level-crossing probabilities
    along an adaptively chosen sequence of intermediate thresholds.

    Between levels, seeds are propagated by a component-wise geometric
    random-walk Metropolis move conditioned on staying above the current
    intermediate level. The reported interval is the standard Au-Beck
    coefficient-of-variation interval, which is **asymptotic** (not a
    finite-sample ``(1 - delta)`` certificate); ``certified`` is ``False``.
    """

    name = "sequential_mc"

    def __init__(
        self,
        n_particles: int = 1000,
        p0: float = 0.1,
        mcmc_steps: int = 8,
        step_scale: float = 0.5,
        max_levels: int = 40,
    ) -> None:
        if n_particles < 10:
            raise ValueError("n_particles must be >= 10")
        if not (0.0 < p0 < 1.0):
            raise ValueError("p0 must be in (0, 1)")
        self.n_particles = n_particles
        self.p0 = p0
        self.mcmc_steps = mcmc_steps
        self.step_scale = step_scale
        self.max_levels = max_levels

    def _log_pmf(self, base: Mapping[str, Distribution], x: Mapping[str, int]) -> float:
        s = 0.0
        for k, d in base.items():
            pv = d.pmf(int(x[k]))
            s += math.log(pv) if pv > 0.0 else -math.inf
        return s

    def _mcmc_move(
        self,
        base: Mapping[str, Distribution],
        score_fn: ScoreFn,
        level: float,
        x: dict[str, int],
        keys: list[str],
        sigmas: dict[str, float],
        rng: np.random.Generator,
    ) -> dict[str, int]:
        """Component-wise integer random-walk Metropolis targeting
        ``D`` restricted to ``score >= level``. Unbounded, symmetric
        proposal; out-of-support or below-level proposals are rejected
        (reject-and-stay), which keeps the proposal symmetric.
        """
        cur = dict(x)
        cur_logp = self._log_pmf(base, cur)
        for _ in range(self.mcmc_steps):
            prop = dict(cur)
            for k in keys:
                step = int(round(float(rng.normal(0.0, sigmas[k]))))
                prop[k] = cur[k] + step
            prop_logp = self._log_pmf(base, prop)
            if prop_logp == -math.inf:
                continue
            if score_fn(prop) < level:
                continue
            if prop_logp >= cur_logp or math.log(rng.random()) < prop_logp - cur_logp:
                cur, cur_logp = prop, prop_logp
        return cur

    def run(
        self,
        distribution: Mapping[str, Distribution],
        score_fn: ScoreFn,
        threshold: float,
        budget: int,
        delta: float,
        seed: int,
        event_fn: EventFn | None = None,
    ) -> RareEventResult:
        rng = np.random.default_rng(seed)
        base = _factors(distribution)
        is_hit = _event_from(score_fn, threshold, event_fn)
        keys = list(base.keys())
        n = self.n_particles
        # Proposal spread per coordinate from the base support scale.
        sigmas: dict[str, float] = {}
        for k, d in base.items():
            b_lo, b_hi = d.support_bounds(1e-6)
            sigmas[k] = max(1.0, self.step_scale * (b_hi - b_lo) / 10.0)
        t0 = time.perf_counter()
        # Level 0: iid from D.
        particles = [
            {k: int(base[k].sample(rng, 1)[0]) for k in keys} for _ in range(n)
        ]
        samples_used = n
        scores = np.array([score_fn(x) for x in particles], dtype=np.float64)
        log_mu = 0.0
        cond_probs: list[float] = []
        levels: list[float] = []
        reached = False
        for _ in range(self.max_levels):
            if float(np.max(scores)) >= threshold:
                # Final level: conditional fraction satisfying the actual
                # event (which may be stricter than the steering threshold,
                # e.g. a bitwise sub-condition on top of score >= T).
                pk = float(np.mean([1.0 if is_hit(x) else 0.0 for x in particles]))
                if pk <= 0.0:
                    break
                cond_probs.append(pk)
                levels.append(threshold)
                log_mu += math.log(pk)
                reached = True
                break
            # Intermediate level: the (1 - p0) quantile of current scores.
            level = float(np.quantile(scores, 1.0 - self.p0))
            pk = float(np.mean(scores >= level))
            if pk <= 0.0:
                break
            cond_probs.append(pk)
            levels.append(level)
            log_mu += math.log(pk)
            # Seeds above the level; resample to n by MCMC propagation.
            seed_idx = [i for i in range(n) if scores[i] >= level]
            if not seed_idx:
                break
            new_particles: list[dict[str, int]] = []
            j = 0
            while len(new_particles) < n:
                start = particles[seed_idx[j % len(seed_idx)]]
                moved = self._mcmc_move(
                    base, score_fn, level, dict(start), keys, sigmas, rng
                )
                new_particles.append(moved)
                samples_used += self.mcmc_steps
                j += 1
            particles = new_particles
            scores = np.array([score_fn(x) for x in particles], dtype=np.float64)
        wall = time.perf_counter() - t0
        mu_hat: float = math.exp(log_mu) if (reached and cond_probs) else 0.0
        # Au-Beck asymptotic c.o.v.: delta_k^2 ~ (1 - p_k)/(p_k n), summed.
        cov2 = 0.0
        for pk in cond_probs:
            if pk > 0.0:
                cov2 += (1.0 - pk) / (pk * n)
        cov = math.sqrt(cov2)
        # Two-sided normal interval at 1 - delta (asymptotic, not certified).
        from ..estimator import _phi_inv_one_sided

        z = _phi_inv_one_sided(1.0 - delta / 2.0)
        half = min(mu_hat, z * cov * mu_hat)
        lo = max(0.0, mu_hat - half)
        hi = min(1.0, mu_hat + half)
        return RareEventResult(
            name=self.name,
            mu_hat=mu_hat,
            interval=(lo, hi),
            samples_used=samples_used,
            wall_clock_s=wall,
            delta=delta,
            certified=False,
            extras={
                "n_particles": n,
                "levels": levels,
                "cond_probs": cond_probs,
                "reached_threshold": reached,
                "cov": cov,
                "half_width": half,
                "note": "asymptotic CLT interval; not a finite-sample (1-delta) certificate",
            },
        )


__all__ = [
    "CrossEntropy",
    "ImportanceSampling",
    "RareEventResult",
    "SequentialMonteCarlo",
]
