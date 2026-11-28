"""Anytime-valid off-policy betting confidence sequence for a failure rate.

Observations are :math:`Z_i = W_i R_i \\ge 0`, where :math:`R_i \\in \\{0,1\\}`
is the failure indicator and :math:`W_i \\in [0, W_{\\max,i}]` the
importance weight with *predictable* cap :math:`W_{\\max,i} = 1/\\alpha_i`.
Under the sampling law, :math:`\\mathbb E[Z_i \\mid \\mathcal F_{i-1}] =
p_{\\text{fail}}` for every ``i``.

For a grid of bet fractions :math:`\\gamma_1, \\dots, \\gamma_G` (uniform
prior), the two mixture wealth processes

.. math::

    K_t^+(m) &= \\tfrac1G \\textstyle\\sum_j \\prod_{i \\le t}
              \\big(1 - \\gamma_j + \\gamma_j Z_i / m\\big), \\\\
    K_t^-(m) &= \\tfrac1G \\textstyle\\sum_j \\prod_{i \\le t}
              \\Big(1 - \\gamma_j + \\gamma_j
              \\frac{W_{\\max,i} - Z_i}{W_{\\max,i} - m}\\Big)

are non-negative supermartingales under :math:`H_0^+: p_{\\text{fail}} \\le m`
and :math:`H_0^-: p_{\\text{fail}} \\ge m` respectively, so by Ville's
inequality the sets
:math:`\\{m : K_t^+(m) < 1/\\delta_{\\text{lo}}\\}` and
:math:`\\{m : K_t^-(m) < 1/\\delta_{\\text{up}}\\}` are time-uniform
confidence sets.  Monotonicity in ``m`` makes both one-dimensional
binary searches.

Storage
-------
To keep evaluation cost independent of ``t``, observations are kept as
a histogram: zeros are counted per weight cap, and positive ``Z`` values
are binned on a geometric grid of relative step ``bin_rel`` (default
``1e-3``).  The lower-bound wealth uses each bin's *lower* edge and the
upper-bound wealth each bin's *upper* edge (clipped to ``W_max``).  Each
rounded factor is pointwise no larger than the exact factor, so the
rounded wealth is itself a non-negative supermartingale under the same
null: the compression is conservative (it can only delay rejection, never
cause a false one).  Exact incremental wealth at user-chosen fixed
thresholds can additionally be tracked (``track_lower`` /
``track_upper``), which the experiments use for exact
samples-to-certify measurements.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Mode = Literal["two_sided", "falsify", "verify"]

M_MIN = 1e-12
M_MAX = 1.0 - 1e-12
_N_BISECT = 64


def _logsumexp_cols(a: np.ndarray) -> float:
    """``log sum_j exp(a_j)`` for a 1-D array, stable."""
    mx = float(np.max(a))
    if not math.isfinite(mx):
        return mx
    return mx + math.log(float(np.sum(np.exp(a - mx))))


def default_gamma_grid(n: int = 30, lo: float = 1e-4, hi: float = 0.95) -> np.ndarray:
    """``n`` log-spaced bet fractions in ``[lo, hi]``."""
    return np.exp(np.linspace(math.log(lo), math.log(hi), n))


@dataclass
class _Group:
    """All observations sharing one weight cap ``w_max``."""

    w_max: float
    n_zero: int = 0
    bins: dict[int, int] = field(default_factory=dict)


class OffPolicyBettingCS:
    """Stateful per-leaf confidence sequence for :math:`p_{\\text{fail}}`.

    Parameters
    ----------
    delta:
        The leaf's fixed a-priori budget :math:`\\delta_\\pi` (e.g.
        ``delta / K_max``).  Never changes after construction.
    mode:
        ``"two_sided"`` splits ``delta`` evenly between the bounds;
        ``"falsify"`` spends it all on the lower bound (``U_fail == 1``);
        ``"verify"`` spends it all on the upper bound (``L_fail == 0``).
    gammas:
        Bet-fraction grid (default: 30 log-spaced values in
        ``[1e-4, 0.95]``), uniform prior.
    bin_rel:
        Relative width of the conservative ``Z`` histogram bins.
    track_lower, track_upper:
        Optional fixed thresholds ``m`` at which the *exact* (un-binned)
        log-wealth is maintained incrementally.
    """

    def __init__(
        self,
        delta: float,
        mode: Mode = "two_sided",
        gammas: Sequence[float] | None = None,
        bin_rel: float = 1e-3,
        track_lower: Sequence[float] = (),
        track_upper: Sequence[float] = (),
    ) -> None:
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if mode not in ("two_sided", "falsify", "verify"):
            raise ValueError(f"unknown mode {mode!r}")
        self.delta = float(delta)
        self.mode: Mode = mode
        if mode == "two_sided":
            self.delta_lower: float | None = delta / 2.0
            self.delta_upper: float | None = delta / 2.0
        elif mode == "falsify":
            self.delta_lower, self.delta_upper = delta, None
        else:
            self.delta_lower, self.delta_upper = None, delta
        self.gammas = np.asarray(
            gammas if gammas is not None else default_gamma_grid(), dtype=np.float64
        )
        if np.any(self.gammas <= 0.0) or np.any(self.gammas >= 1.0):
            raise ValueError("bet fractions must lie in (0, 1)")
        self.log_G = math.log(len(self.gammas))
        self._log1m_g = np.log1p(-self.gammas)
        self.bin_rel = float(bin_rel)
        self._h = math.log1p(self.bin_rel)
        self._groups: dict[float, _Group] = {}
        self.n = 0
        self.sum_z = 0.0
        self.n_positive = 0
        # Running intersection state.
        self._L = 0.0
        self._U = 1.0
        self._dirty = False
        self._cache: tuple[np.ndarray, ...] | None = None
        self.crossed = False
        self.n_evaluations = 0
        # Exact trackers at fixed thresholds.
        self.track_lower = np.asarray(track_lower, dtype=np.float64)
        self.track_upper = np.asarray(track_upper, dtype=np.float64)
        self._tl = np.zeros((len(self.track_lower), len(self.gammas)))
        self._tu = np.zeros((len(self.track_upper), len(self.gammas)))

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def update(self, z: Sequence[float] | np.ndarray, alpha: float) -> None:
        """Record a batch ``Z_1..Z_k`` drawn with mixture weight ``alpha``.

        ``alpha`` (hence ``W_max = 1/alpha``) must have been fixed before
        the batch was drawn.  Values are validated to lie in
        ``[0, W_max]``.
        """
        z = np.asarray(z, dtype=np.float64).ravel()
        if z.size == 0:
            return
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        w_max = 1.0 / alpha
        if np.any(~np.isfinite(z)) or np.any(z < 0.0) or np.any(z > w_max * (1 + 1e-12)):
            raise ValueError("observations must satisfy 0 <= Z <= 1/alpha")
        z = np.minimum(z, w_max)
        key = float(w_max)
        g = self._groups.get(key)
        if g is None:
            g = _Group(w_max=w_max)
            self._groups[key] = g
        pos = z[z > 0.0]
        g.n_zero += int(z.size - pos.size)
        if pos.size:
            idx = np.floor(np.log(pos) / self._h).astype(np.int64)
            uniq, cnt = np.unique(idx, return_counts=True)
            for k, c in zip(uniq.tolist(), cnt.tolist(), strict=True):
                g.bins[k] = g.bins.get(k, 0) + c
        self.n += int(z.size)
        self.sum_z += float(z.sum())
        self.n_positive += int(pos.size)
        # Exact trackers (O(k * n_thresholds * G)).
        if self.track_lower.size:
            ratio = z[None, :, None] / self.track_lower[:, None, None]
            self._tl += np.log1p(self.gammas[None, None, :] * (ratio - 1.0)).sum(axis=1)
        if self.track_upper.size:
            m = self.track_upper[:, None, None]
            ratio = (m - z[None, :, None]) / (w_max - m)
            self._tu += np.log1p(self.gammas[None, None, :] * ratio).sum(axis=1)
        self._dirty = True
        self._cache = None

    # ------------------------------------------------------------------
    # Wealth
    # ------------------------------------------------------------------

    def _arrays(self) -> tuple[np.ndarray, ...]:
        """Flattened histogram: lower-edge values / counts, and upper-edge
        values / caps / counts (zeros included with value 0)."""
        if self._cache is not None:
            return self._cache
        zl, cl, zu, wu, cu = [], [], [], [], []
        n_zero_total = 0
        for g in self._groups.values():
            n_zero_total += g.n_zero
            if g.n_zero:
                zu.append(0.0)
                wu.append(g.w_max)
                cu.append(g.n_zero)
            for k, c in g.bins.items():
                lo_edge = math.exp(k * self._h) * (1.0 - 1e-12)
                hi_edge = min(math.exp((k + 1) * self._h) * (1.0 + 1e-12), g.w_max)
                zl.append(lo_edge)
                cl.append(c)
                zu.append(hi_edge)
                wu.append(g.w_max)
                cu.append(c)
        self._cache = (
            np.asarray(zl, dtype=np.float64),
            np.asarray(cl, dtype=np.float64),
            np.asarray(zu, dtype=np.float64),
            np.asarray(wu, dtype=np.float64),
            np.asarray(cu, dtype=np.float64),
            np.asarray([n_zero_total], dtype=np.float64),
        )
        return self._cache

    def log_wealth_lower(self, m: float) -> float:
        """``log K_t^+(m)`` (conservatively binned) for ``H0: p <= m``."""
        zl, cl, _, _, _, nz = self._arrays()
        per_j = nz[0] * self._log1m_g
        if zl.size:
            terms = np.log1p(self.gammas[None, :] * (zl[:, None] / m - 1.0))
            per_j = per_j + cl @ terms
        return _logsumexp_cols(per_j) - self.log_G

    def log_wealth_upper(self, m: float) -> float:
        """``log K_t^-(m)`` (conservatively binned) for ``H0: p >= m``."""
        _, _, zu, wu, cu, _ = self._arrays()
        if zu.size == 0:
            return 0.0  # empty product: wealth 1 for every gamma
        ratio = (m - zu) / (wu - m)
        terms = np.log1p(self.gammas[None, :] * ratio[:, None])
        per_j = cu @ terms
        return _logsumexp_cols(per_j) - self.log_G

    def exact_log_wealth_lower(self) -> np.ndarray:
        """Exact mixture log-wealth at each ``track_lower`` threshold."""
        out = np.empty(len(self.track_lower))
        for i in range(len(out)):
            out[i] = _logsumexp_cols(self._tl[i]) - self.log_G
        return out

    def exact_log_wealth_upper(self) -> np.ndarray:
        """Exact mixture log-wealth at each ``track_upper`` threshold."""
        out = np.empty(len(self.track_upper))
        for i in range(len(out)):
            out[i] = _logsumexp_cols(self._tu[i]) - self.log_G
        return out

    def certified_lower_at(self) -> np.ndarray:
        """Boolean per ``track_lower`` threshold ``m``: is ``p_fail > m``
        certified now (exact wealth crossed ``1/delta_lower``)?"""
        if self.delta_lower is None:
            return np.zeros(len(self.track_lower), dtype=bool)
        return self.exact_log_wealth_lower() >= math.log(1.0 / self.delta_lower)

    def certified_upper_at(self) -> np.ndarray:
        """Boolean per ``track_upper`` threshold ``m``: is ``p_fail < m``
        certified now?"""
        if self.delta_upper is None:
            return np.zeros(len(self.track_upper), dtype=bool)
        return self.exact_log_wealth_upper() >= math.log(1.0 / self.delta_upper)

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------

    def _lower_inst(self) -> float:
        assert self.delta_lower is not None
        thr = math.log(1.0 / self.delta_lower)
        if self.n_positive == 0 or self.log_wealth_lower(M_MIN) < thr:
            return 0.0
        if self.log_wealth_lower(M_MAX) >= thr:
            return M_MAX
        lo, hi = math.log(M_MIN), math.log(M_MAX)  # lo rejected, hi not
        for _ in range(_N_BISECT):
            mid = 0.5 * (lo + hi)
            if self.log_wealth_lower(math.exp(mid)) >= thr:
                lo = mid
            else:
                hi = mid
        return math.exp(lo)  # a rejected value: conservative

    def _upper_inst(self) -> float:
        assert self.delta_upper is not None
        thr = math.log(1.0 / self.delta_upper)
        if self.n == 0 or self.log_wealth_upper(M_MAX) < thr:
            return 1.0
        if self.log_wealth_upper(M_MIN) >= thr:
            return M_MIN
        lo, hi = math.log(M_MIN), math.log(M_MAX)  # hi rejected, lo not
        for _ in range(_N_BISECT):
            mid = 0.5 * (lo + hi)
            if self.log_wealth_upper(math.exp(mid)) >= thr:
                hi = mid
            else:
                lo = mid
        return math.exp(hi)

    def bounds(self) -> tuple[float, float]:
        """Running-intersection interval ``[L_fail, U_fail]`` at the current
        evaluation step (recomputed only if new data arrived)."""
        if self._dirty:
            self.n_evaluations += 1
            if self.delta_lower is not None:
                self._L = max(self._L, self._lower_inst())
            if self.delta_upper is not None:
                self._U = min(self._U, self._upper_inst())
            if self._L > self._U:
                # Consistency guard: a genuine crossing is itself a
                # (probability <= delta) miscoverage event; collapse the
                # interval to its midpoint and flag it.
                if self._L - self._U > 1e-12:
                    self.crossed = True
                mid = 0.5 * (self._L + self._U)
                self._L = self._U = mid
            self._L = min(max(self._L, 0.0), 1.0)
            self._U = min(max(self._U, 0.0), 1.0)
            self._dirty = False
        return self._L, self._U

    @property
    def point_estimate(self) -> float:
        """``clip(mean Z, 0, 1)`` (unbiased before clipping)."""
        if self.n == 0:
            return 0.5
        return min(1.0, max(0.0, self.sum_z / self.n))

    def reliability_bounds(self) -> tuple[float, float]:
        """``[L_pi, U_pi] = [1 - U_fail, 1 - L_fail]`` (sign convention:
        ``phi = 1`` is PASS)."""
        L, U = self.bounds()
        return 1.0 - U, 1.0 - L

    @property
    def n_bins(self) -> int:
        return sum(len(g.bins) + (1 if g.n_zero else 0) for g in self._groups.values())

    def __repr__(self) -> str:
        L, U = self._L, self._U
        return (
            f"OffPolicyBettingCS(n={self.n}, p_hat={self.point_estimate:.3e}, "
            f"[L_fail, U_fail]=[{L:.3e}, {U:.3e}], mode={self.mode!r}, "
            f"delta={self.delta:.3g})"
        )


__all__ = ["OffPolicyBettingCS", "default_gamma_grid"]
