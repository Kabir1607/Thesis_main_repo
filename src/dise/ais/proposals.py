"""Exponentially twisted, defensively mixed proposals on axis-aligned boxes.

The proposal family used by certified AIS inside an unclosable ``OPEN``
leaf :math:`R_\\pi = \\prod_i \\{\\ell_i, \\dots, u_i\\}`.

Tilted marginal
---------------
Per coordinate we use the *truncated geometric* (discrete exponential)
family on the shifted index :math:`j = x_i - \\ell_i \\in \\{0, \\dots,
M_i - 1\\}`:

.. math::

    \\tilde q_{\\eta_i}(\\ell_i + j) = \\frac{e^{\\eta_i j}}{\\sum_{t=0}^{M_i-1} e^{\\eta_i t}},
    \\qquad r_i = e^{\\eta_i}.

For a :class:`~dise.distributions.BoundedGeometric` (or
:class:`~dise.distributions.Geometric`) marginal with
:math:`\\mathcal D_i(k) \\propto (1-p_i)^k`, the exponential tilt
:math:`\\mathcal D_i(k) e^{\\theta_i k}` *is* this family with natural
parameter :math:`\\eta_i = \\log(1-p_i) + \\theta_i` (so the context's
ratio :math:`r_i = (1-p_i)e^{\\theta_i}`).  For a
:class:`~dise.distributions.Uniform` marginal, :math:`\\eta_i = \\theta_i`.
For any other finite discrete marginal (Categorical, Poisson, ...) the
same family is still a valid proposal; the base :math:`\\mathcal D|_R`
then uses an exact normalized log-PMF array lookup on the box.

Defensive mixture
-----------------
.. math::

    q(x) = \\alpha\\, \\mathcal D|_{R}(x) + (1-\\alpha)\\, \\tilde q_{\\boldsymbol\\eta}(x),
    \\qquad
    W(x) = \\frac{\\mathcal D|_R(x)}{q(x)}
         = \\frac{1}{\\alpha + (1-\\alpha) \\exp(\\ell_{\\tilde q} - \\ell_{\\mathcal D})}
         \\in [0, 1/\\alpha].

Sampling is rejection-free: the mixture component is chosen per draw and
each coordinate is drawn by exact inverse CDF.

All arithmetic is in log space; the ``r > 1`` case is handled by the
flip :math:`j' = M - 1 - j`, :math:`s = 1/r < 1`, so :math:`r^M` is
never formed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..distributions import (
    BoundedGeometric,
    Distribution,
    Geometric,
    ProductDistribution,
    Uniform,
)

#: ``|eta| < UNIFORM_TOL`` is treated as exactly uniform (``r == 1``).
UNIFORM_TOL = 1e-12
#: Below this value of ``a * M`` the truncated-geometric mean uses its
#: second-order Taylor expansion (avoids catastrophic cancellation).
_TAYLOR_AM = 1e-5
#: Largest box side for which a non-geometric marginal is tabulated.
MAX_TABULATED_SIDE = 2_000_000
#: Bisection bracket for the natural parameter in CE updates.
_ETA_BRACKET = 60.0


# ---------------------------------------------------------------------------
# Truncated geometric on {0, ..., M-1}: log-normalizer, log-PMF, sampling,
# mean, and the inverse-mean (CE) solver.  ``eta`` is the natural parameter
# (log r).  Every routine is overflow-safe for any M >= 1 and any finite eta.
# ---------------------------------------------------------------------------


def _log_z_decreasing(a: float, M: int) -> float:
    """``log sum_{t<M} e^{-a t}`` for ``a > 0`` (ratio ``s = e^{-a} < 1``).

    ``= log(1 - s^M) - log(1 - s)``, evaluated with ``expm1``.
    """
    return math.log(-math.expm1(-a * M)) - math.log(-math.expm1(-a))


def trunc_geom_log_z(eta: float, M: int) -> float:
    """Log-normalizer :math:`\\log \\sum_{t=0}^{M-1} e^{\\eta t}`."""
    if M <= 0:
        raise ValueError("M must be >= 1")
    if M == 1:
        return 0.0
    if abs(eta) < UNIFORM_TOL:
        return math.log(M)
    if eta < 0.0:
        return _log_z_decreasing(-eta, M)
    # eta > 0: sum e^{eta t} = e^{eta (M-1)} * sum e^{-eta t'}; never form r^M.
    return eta * (M - 1) + _log_z_decreasing(eta, M)


def trunc_geom_logpmf(j: np.ndarray, eta: float, M: int) -> np.ndarray:
    """Exact log-PMF of the truncated geometric at shifted indices ``j``.

    For ``eta > 0`` uses the flipped form ``j' log s - log((1 - s^M)/(1 - s))``
    with ``s = e^{-eta}``, so no term exceeds ``O(eta * M)`` in magnitude
    and nothing overflows.
    """
    j = np.asarray(j, dtype=np.float64)
    if M == 1:
        return np.zeros_like(j)
    if abs(eta) < UNIFORM_TOL:
        return np.full_like(j, -math.log(M))
    if eta < 0.0:
        a = -eta
        return -a * j - _log_z_decreasing(a, M)
    jp = (M - 1) - j
    return -eta * jp - _log_z_decreasing(eta, M)


def _sample_decreasing(u: np.ndarray, a: float, M: int) -> np.ndarray:
    """Exact inverse CDF for ratio ``s = e^{-a} < 1``:
    ``j = floor(log(1 - u (1 - s^M)) / log s)``.
    """
    # 1 - u (1 - s^M) = 1 + u * expm1(-a M);  log s = -a.
    val = np.log1p(u * math.expm1(-a * M)) / (-a)
    j = np.floor(val).astype(np.int64)
    np.clip(j, 0, M - 1, out=j)
    return j


def trunc_geom_sample(
    rng: np.random.Generator, eta: float, M: int, n: int
) -> np.ndarray:
    """Draw ``n`` shifted indices ``j in {0, ..., M-1}`` exactly."""
    if M == 1:
        return np.zeros(n, dtype=np.int64)
    u = rng.random(n)
    if abs(eta) < UNIFORM_TOL:
        j = np.floor(u * M).astype(np.int64)
        np.clip(j, 0, M - 1, out=j)
        return j
    if eta < 0.0:
        return _sample_decreasing(u, -eta, M)
    return (M - 1) - _sample_decreasing(u, eta, M)


def _mean_decreasing(a: float, M: int) -> float:
    """Mean of ``j`` under ratio ``e^{-a}``, ``a > 0``:
    ``1/expm1(a) - M/expm1(a M)``.
    """
    aM = a * M
    if aM < _TAYLOR_AM:
        return 0.5 * (M - 1) - a * (M * M - 1) / 12.0
    first = 1.0 / math.expm1(a) if a < 700.0 else 0.0
    second = M / math.expm1(aM) if aM < 700.0 else 0.0
    return max(0.0, min(float(M - 1), first - second))


def trunc_geom_mean(eta: float, M: int) -> float:
    """Exact mean of the shifted index under :math:`\\tilde q_\\eta`."""
    if M == 1:
        return 0.0
    if abs(eta) < UNIFORM_TOL:
        return 0.5 * (M - 1)
    if eta < 0.0:
        return _mean_decreasing(-eta, M)
    return (M - 1) - _mean_decreasing(eta, M)


def solve_eta_for_mean(target: float, M: int, tol: float = 1e-13) -> float:
    """Solve :math:`\\mathbb E_{\\tilde q_\\eta}[j] = \\text{target}` for ``eta``.

    The mean of an exponential family is strictly increasing in its
    natural parameter (its derivative is the variance), so the root is
    unique; we find it by bisection on ``[-60, 60]``.  ``target`` is
    clipped to the open interval ``(0, M - 1)`` first.
    """
    if M <= 1:
        return 0.0
    lo_t, hi_t = 1e-9, (M - 1) - 1e-9
    target = min(max(target, lo_t), hi_t)
    lo, hi = -_ETA_BRACKET, _ETA_BRACKET
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if trunc_geom_mean(mid, M) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, abs(mid)):
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Base marginal D_i restricted to [lo, hi]
# ---------------------------------------------------------------------------


@dataclass
class _BoxMarginal:
    """One coordinate of ``D|_R``: either a truncated geometric with
    natural parameter ``eta0`` (closed form) or an exact log-PMF table."""

    lo: int
    hi: int
    eta0: float | None  # closed-form family member, or None if tabulated
    table_logpmf: np.ndarray | None = None
    table_cdf: np.ndarray | None = None

    @property
    def M(self) -> int:
        return self.hi - self.lo + 1

    def logpmf(self, j: np.ndarray) -> np.ndarray:
        if self.eta0 is not None:
            return trunc_geom_logpmf(j, self.eta0, self.M)
        assert self.table_logpmf is not None
        return self.table_logpmf[np.asarray(j, dtype=np.int64)]

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.eta0 is not None:
            return trunc_geom_sample(rng, self.eta0, self.M, n)
        assert self.table_cdf is not None
        u = rng.random(n)
        j = np.searchsorted(self.table_cdf, u, side="right").astype(np.int64)
        np.clip(j, 0, self.M - 1, out=j)
        return j

    def mean(self) -> float:
        if self.eta0 is not None:
            return trunc_geom_mean(self.eta0, self.M)
        assert self.table_logpmf is not None
        return float(np.sum(np.exp(self.table_logpmf) * np.arange(self.M)))


def _make_marginal(dist: Distribution, lo: int, hi: int) -> _BoxMarginal:
    if lo > hi:
        raise ValueError(f"empty box side [{lo}, {hi}]")
    if isinstance(dist, (BoundedGeometric, Geometric)):
        # D(k) ∝ (1-p)^k on its support; restricted to [lo, hi] this is a
        # truncated geometric with natural parameter log(1 - p).
        lo_eff = max(lo, 1)
        hi_eff = min(hi, dist.N) if isinstance(dist, BoundedGeometric) else hi
        if lo_eff != lo or hi_eff != hi:
            raise ValueError(
                f"box side [{lo}, {hi}] exceeds support of {dist!r}"
            )
        return _BoxMarginal(lo=lo, hi=hi, eta0=math.log1p(-dist.p))
    if isinstance(dist, Uniform):
        if lo < dist.lo or hi > dist.hi:
            raise ValueError(f"box side [{lo}, {hi}] exceeds support of {dist!r}")
        return _BoxMarginal(lo=lo, hi=hi, eta0=0.0)
    M = hi - lo + 1
    if M > MAX_TABULATED_SIDE:
        raise ValueError(f"box side of {M} cells too large to tabulate for {dist!r}")
    pmf = np.array([dist.pmf(k) for k in range(lo, hi + 1)], dtype=np.float64)
    Z = pmf.sum()
    if Z <= 0.0:
        raise ValueError(f"zero-mass box side [{lo}, {hi}] for {dist!r}")
    pmf /= Z
    with np.errstate(divide="ignore"):
        logpmf = np.log(pmf)
    cdf = np.cumsum(pmf)
    cdf[-1] = 1.0
    return _BoxMarginal(lo=lo, hi=hi, eta0=None, table_logpmf=logpmf, table_cdf=cdf)


# ---------------------------------------------------------------------------
# TwistedBGProposal
# ---------------------------------------------------------------------------


@dataclass
class ProposalDraw:
    """A batch drawn from the defensive mixture.

    ``x`` has shape ``(n, d)`` (columns in ``variables`` order);
    ``weights`` are the exact likelihood ratios ``D|_R(x) / q(x)``;
    ``alpha`` is the (predictable) mixture weight used for the batch.
    """

    variables: tuple[str, ...]
    x: np.ndarray
    weights: np.ndarray
    alpha: float
    from_base: np.ndarray

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    def assignments(self):
        for i in range(self.n):
            yield {v: int(self.x[i, k]) for k, v in enumerate(self.variables)}


class TwistedBGProposal:
    """Defensive mixture of ``D|_R`` and a product of truncated geometrics.

    Parameters
    ----------
    distribution:
        The product input distribution ``D``.
    bounds:
        Axis-aligned box ``{var: (lo, hi)}`` (inclusive) defining ``R``.
    alpha:
        Defensive mixture weight on ``D|_R`` (``alpha in (0, 1]``).
        ``alpha = 1`` is plain sampling (``W == 1``).
    eta:
        Initial natural parameters per variable.  Default: ``0`` for every
        coordinate, i.e. *uniform on the box* -- the exploratory bootstrap
        tilt (``r_i = 1  <=>  theta_i = -log(1 - p_i)`` for BG marginals),
        which pushes mass toward the low-probability tail.
    """

    def __init__(
        self,
        distribution: ProductDistribution,
        bounds: Mapping[str, tuple[int, int]],
        alpha: float = 0.5,
        eta: Mapping[str, float] | None = None,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.variables: tuple[str, ...] = tuple(distribution.variables)
        self.alpha = float(alpha)
        self.marginals: list[_BoxMarginal] = []
        for v in self.variables:
            lo, hi = bounds[v]
            self.marginals.append(_make_marginal(distribution.factors[v], int(lo), int(hi)))
        self.lo = np.array([m.lo for m in self.marginals], dtype=np.int64)
        self.M = np.array([m.M for m in self.marginals], dtype=np.int64)
        init = dict(eta or {})
        self.eta = np.array([float(init.get(v, 0.0)) for v in self.variables])
        self.n_ce_updates = 0

    # ---- parameters ----

    @property
    def d(self) -> int:
        return len(self.variables)

    @property
    def theta(self) -> np.ndarray:
        """Tilt relative to the base family: ``theta_i = eta_i - eta0_i``
        (``nan`` for tabulated marginals, which have no base eta)."""
        return np.array(
            [
                self.eta[i] - m.eta0 if m.eta0 is not None else float("nan")
                for i, m in enumerate(self.marginals)
            ]
        )

    def base_eta(self) -> np.ndarray:
        """Natural parameters of ``D|_R`` (``nan`` where tabulated)."""
        return np.array(
            [m.eta0 if m.eta0 is not None else float("nan") for m in self.marginals]
        )

    def set_eta(self, eta: Sequence[float]) -> None:
        eta = np.asarray(eta, dtype=np.float64)
        if eta.shape != (self.d,) or not np.all(np.isfinite(eta)):
            raise ValueError("eta must be a finite vector of length d")
        self.eta = eta.copy()

    # ---- exact log-densities ----

    def _shift(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.int64))
        j = x - self.lo[None, :]
        if np.any(j < 0) or np.any(j >= self.M[None, :]):
            raise ValueError("point outside the proposal box")
        return j

    def log_base(self, x: np.ndarray) -> np.ndarray:
        """``log D|_R(x)`` for rows of ``x`` (shape ``(n, d)``)."""
        j = self._shift(x)
        out = np.zeros(j.shape[0])
        for i, m in enumerate(self.marginals):
            out += m.logpmf(j[:, i])
        return out

    def log_tilt(self, x: np.ndarray) -> np.ndarray:
        """``log q~_eta(x)`` for rows of ``x``."""
        j = self._shift(x)
        out = np.zeros(j.shape[0])
        for i in range(self.d):
            out += trunc_geom_logpmf(j[:, i], float(self.eta[i]), int(self.M[i]))
        return out

    def log_mixture(self, x: np.ndarray) -> np.ndarray:
        """``log q(x)`` of the defensive mixture."""
        lb = self.log_base(x)
        lt = self.log_tilt(x)
        a = self.alpha
        if a >= 1.0:
            return lb
        return np.logaddexp(math.log(a) + lb, math.log1p(-a) + lt)

    def weights(self, x: np.ndarray) -> np.ndarray:
        """Exact likelihood ratio ``W = D|_R(x) / q(x) in [0, 1/alpha]``.

        ``W = 1 / (alpha + (1 - alpha) exp(l_q~ - l_D))`` with the guard
        ``l_q~ - l_D > 700 => W = 0`` (the ratio underflows; ``D|_R(x)`` is
        negligible relative to ``q(x)``).
        """
        a = self.alpha
        if a >= 1.0:
            return np.ones(np.atleast_2d(x).shape[0])
        diff = self.log_tilt(x) - self.log_base(x)
        out = np.zeros_like(diff)
        ok = diff <= 700.0
        out[ok] = 1.0 / (a + (1.0 - a) * np.exp(diff[ok]))
        return out

    @property
    def w_max(self) -> float:
        """Predictable deterministic cap on the weight, ``1 / alpha``."""
        return 1.0 / self.alpha

    # ---- sampling ----

    def sample(self, rng: np.random.Generator, n: int) -> ProposalDraw:
        """Draw ``n`` points from the defensive mixture with exact weights.

        Rejection-free: with probability ``alpha`` a draw comes from
        ``D|_R`` (exact inverse CDF per coordinate), otherwise from
        ``q~_eta``.
        """
        from_base = rng.random(n) < self.alpha
        j = np.empty((n, self.d), dtype=np.int64)
        nb = int(from_base.sum())
        nt = n - nb
        for i, m in enumerate(self.marginals):
            col = np.empty(n, dtype=np.int64)
            if nb:
                col[from_base] = m.sample(rng, nb)
            if nt:
                col[~from_base] = trunc_geom_sample(
                    rng, float(self.eta[i]), int(self.M[i]), nt
                )
            j[:, i] = col
        x = j + self.lo[None, :]
        return ProposalDraw(
            variables=self.variables,
            x=x,
            weights=self.weights(x),
            alpha=self.alpha,
            from_base=from_base,
        )

    # ---- CE update ----

    def ce_target_eta(
        self,
        x_fail: np.ndarray,
        w_fail: np.ndarray,
        clip_margin: float = 0.05,
    ) -> np.ndarray | None:
        """Cross-entropy optimum within the family for the weighted failing
        sample: per coordinate, match the importance-weighted mean of the
        failing inputs (exact CE / moment matching for an exponential
        family).  The target mean is clipped to
        ``[lo + clip_margin, hi - clip_margin]`` so ``eta`` stays finite.

        Returns ``None`` when there is no usable failing mass.
        """
        x_fail = np.atleast_2d(np.asarray(x_fail, dtype=np.float64))
        w_fail = np.asarray(w_fail, dtype=np.float64)
        if x_fail.shape[0] == 0 or w_fail.sum() <= 0.0:
            return None
        sw = w_fail.sum()
        target = np.empty(self.d)
        for i in range(self.d):
            M = int(self.M[i])
            if M <= 1:
                target[i] = self.eta[i]
                continue
            jbar = float(np.dot(w_fail, x_fail[:, i] - self.lo[i]) / sw)
            jbar = min(max(jbar, clip_margin), (M - 1) - clip_margin)
            target[i] = solve_eta_for_mean(jbar, M)
        return target

    def update_ce(
        self,
        x_fail: np.ndarray,
        w_fail: np.ndarray,
        rho: float = 0.7,
        clip_margin: float = 0.05,
    ) -> bool:
        """Smoothed CE step ``eta <- (1 - rho) eta + rho eta_CE``.

        Must be called *between* batches with failures from earlier
        batches only (predictability).  Returns whether ``eta`` changed.
        """
        tgt = self.ce_target_eta(x_fail, w_fail, clip_margin=clip_margin)
        if tgt is None:
            return False
        self.eta = (1.0 - rho) * self.eta + rho * tgt
        self.n_ce_updates += 1
        return True

    # ---- exact enumeration helpers (tests / oracle) ----

    def marginal_tilt_pmf(self, i: int) -> np.ndarray:
        """Exact PMF vector of the tilted marginal ``i`` on its side."""
        M = int(self.M[i])
        return np.exp(trunc_geom_logpmf(np.arange(M), float(self.eta[i]), M))

    def marginal_base_pmf(self, i: int) -> np.ndarray:
        """Exact PMF vector of ``D_i|_[lo_i, hi_i]``."""
        return np.exp(self.marginals[i].logpmf(np.arange(int(self.M[i]))))

    def __repr__(self) -> str:
        sides = ", ".join(
            f"{v}:[{m.lo},{m.hi}]" for v, m in zip(self.variables, self.marginals, strict=True)
        )
        return f"TwistedBGProposal({sides}, alpha={self.alpha}, eta={self.eta.round(4).tolist()})"


def box_bounds_for_region(region, distribution: ProductDistribution) -> dict[str, tuple[int, int]] | None:
    """Axis-aligned box for an AIS-eligible region, or ``None``.

    Eligible: :class:`~dise.regions.AxisAlignedBox` and
    :class:`~dise.regions.UnconstrainedRegion` (whose box is the
    distribution's support bounds).  :class:`~dise.regions.GeneralRegion`
    is *not* eligible -- sampling ``D|_R`` exactly would need rejection.
    """
    from ..regions import AxisAlignedBox, UnconstrainedRegion

    if isinstance(region, AxisAlignedBox):
        return {v: tuple(region.bounds[v]) for v in distribution.variables}
    if isinstance(region, UnconstrainedRegion):
        return {v: tuple(b) for v, b in distribution.support_bounds().items()}
    return None


__all__ = [
    "ProposalDraw",
    "TwistedBGProposal",
    "box_bounds_for_region",
    "solve_eta_for_mean",
    "trunc_geom_log_z",
    "trunc_geom_logpmf",
    "trunc_geom_mean",
    "trunc_geom_sample",
]
