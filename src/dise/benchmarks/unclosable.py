"""Benchmarks whose failure mass sits in OPEN leaves Z3 cannot close.

Used by the certified-AIS tests and ``scripts/run_ais_experiments.py``.
Each program returns ``1`` on FAULT and ``0`` otherwise; the DiSE property
is ``phi(out) = (out == 0)`` (PASS), so ``mu`` is the reliability and the
failure rate is ``1 - mu``.

Unclosability comes from *concretization*: ``v | 0`` (and other bitwise
ops) makes the concolic tracer drop the symbolic view, so every later
comparison records a ground ``const op const`` clause and
:meth:`~dise.regions.Frontier.try_close` refuses to certify the leaf.

Exact ground truth is computed by enumeration / convolution, so coverage
checks are exact.  These benchmarks are deliberately *not* added to the
global registry (``dise list`` output is unchanged).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..distributions import BoundedGeometric, Distribution


@dataclass(frozen=True)
class UnclosableBenchmark:
    name: str
    description: str
    program: Callable[..., Any]
    distribution: dict[str, Distribution]
    #: SLA threshold on the failure rate; the program violates it
    #: (true failure rate > tau), so the correct decision is FALSIFIED.
    tau: float
    #: Exact failure rate 1 - mu.
    p_fail: float
    #: Exact mass of the unclosable leaf (the region holding all faults).
    unclosable_mass: float

    @property
    def mu(self) -> float:
        return 1.0 - self.p_fail

    @staticmethod
    def property_fn(out: Any) -> bool:
        return out == 0


# ---------------------------------------------------------------------------
# 1. monitor_nonlinear
# ---------------------------------------------------------------------------


def monitor_nonlinear(x):
    """Nominal region ``x < 50`` is trivially safe (Z3 closes it).  Above
    it, a checksum-style bit operation concretizes the trace and a rare
    nonlinear/modular fault fires in the far tail."""
    if x < 50:
        return 0
    y = x | 0  # bitwise -> concretized: the leaf becomes unclosable
    if y >= 75 and (y * y + 3 * y) % 11 == 0:
        return 1
    return 0


# ---------------------------------------------------------------------------
# 2. sum_threshold_loop
# ---------------------------------------------------------------------------

SUM_DIM = 4
SUM_T = 40


def sum_threshold_loop(x1, x2, x3, x4):
    """Accumulate four BG inputs through a bit-packed (concretizing) add and
    fault when the total crosses ``T``: the classical exponential-twisting
    regime (Juneja-Shahabuddin).  Every trace is concretized, so the whole
    input space is one unclosable leaf."""
    s = 0
    for v in (x1, x2, x3, x4):
        s = s + (v | 0)
    if s >= SUM_T:
        return 1
    return 0


# ---------------------------------------------------------------------------
# 3. sla_violation_check
# ---------------------------------------------------------------------------


def sla_violation_check(size, retries):
    """Request latency model.  ``retries <= 2`` takes the fast path (safe,
    closable).  The slow path mixes a bit-twiddled jitter into the latency
    (concretized) and violates the 400-tick SLA for large payloads."""
    if retries <= 2:
        return 0
    s = size | 0
    r = retries | 0
    jitter = (s ^ (s >> 2)) & 31
    latency = 2 * s + 40 * r + jitter
    if latency > 400:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Exact ground truth
# ---------------------------------------------------------------------------


def _support(d: Distribution) -> range:
    lo, hi = d.support_bounds()
    return range(lo, hi + 1)


def _enumerate_fail(program, dist: dict[str, Distribution], leaf_pred) -> tuple[float, float]:
    names = list(dist)
    pmfs = {k: {v: dist[k].pmf(v) for v in _support(dist[k])} for k in names}
    p_fail = 0.0
    leaf = 0.0
    for combo in itertools.product(*(list(pmfs[k]) for k in names)):
        x = dict(zip(names, combo, strict=True))
        pr = 1.0
        for k in names:
            pr *= pmfs[k][x[k]]
        if leaf_pred(x):
            leaf += pr
        if program(**x):
            p_fail += pr
    return p_fail, leaf


def _sum_tail(d: BoundedGeometric, dim: int, T: int) -> float:
    v = np.array([d.pmf(k) for k in range(1, d.N + 1)])
    conv = np.array([1.0])
    for _ in range(dim):
        conv = np.convolve(conv, v)
    sums = np.arange(dim, dim + len(conv))
    return float(conv[sums >= T].sum())


def make_monitor_nonlinear() -> UnclosableBenchmark:
    dist = {"x": BoundedGeometric(0.1, 100)}
    p_fail, leaf = _enumerate_fail(monitor_nonlinear, dist, lambda x: x["x"] >= 50)
    return UnclosableBenchmark(
        name="monitor_nonlinear",
        description="x<50 closable SAFE; x>=50 concretized, rare nonlinear tail fault",
        program=monitor_nonlinear,
        distribution=dist,
        tau=3e-5,
        p_fail=p_fail,
        unclosable_mass=leaf,
    )


def make_sum_threshold_loop() -> UnclosableBenchmark:
    d = BoundedGeometric(0.3, 40)
    dist = {f"x{i + 1}": d for i in range(SUM_DIM)}
    return UnclosableBenchmark(
        name="sum_threshold_loop",
        description="sum of 4 BG(0.3,40) through a concretized loop crosses T=40",
        program=sum_threshold_loop,
        distribution=dist,
        tau=4e-4,
        p_fail=_sum_tail(d, SUM_DIM, SUM_T),
        unclosable_mass=1.0,
    )


def make_sla_violation_check() -> UnclosableBenchmark:
    dist = {"size": BoundedGeometric(0.05, 200), "retries": BoundedGeometric(0.5, 10)}
    p_fail, leaf = _enumerate_fail(
        sla_violation_check, dist, lambda x: x["retries"] >= 3
    )
    return UnclosableBenchmark(
        name="sla_violation_check",
        description="slow path (retries>=3) concretized latency exceeds the SLA",
        program=sla_violation_check,
        distribution=dist,
        tau=2e-3,
        p_fail=p_fail,
        unclosable_mass=leaf,
    )


def all_unclosable_benchmarks() -> list[UnclosableBenchmark]:
    return [make_monitor_nonlinear(), make_sum_threshold_loop(), make_sla_violation_check()]


__all__ = [
    "UnclosableBenchmark",
    "all_unclosable_benchmarks",
    "make_monitor_nonlinear",
    "make_sla_violation_check",
    "make_sum_threshold_loop",
    "monitor_nonlinear",
    "sla_violation_check",
    "sum_threshold_loop",
]
