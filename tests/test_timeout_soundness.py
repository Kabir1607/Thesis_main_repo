"""Regression test for a soundness bug surfaced by the experiment harness.

``experiments/run_full_study.py`` enforces a per-cell wall-clock cap with an
asynchronous ``SIGALRM`` that raises ``_CellTimeout``. If that exception (or
any other control-flow ``BaseException``) is swallowed by a broad
``except Exception`` deep inside the estimator -- e.g. the region-sampling
fallback in ``Frontier._proportional_split_mass`` -- the scheduler keeps
running on a half-mutated frontier and can return an **unsound** certified
interval (observed: gcd certified to ``[0, 0.15]`` when the true value is
0.989).

The contract this test pins: a ``BaseException`` raised inside the program
under analysis must propagate out of :func:`dise.estimate` unchanged. It must
never be swallowed and turned into a (possibly unsound) result. The harness
relies on this by making ``_CellTimeout`` a ``BaseException``.
"""

from __future__ import annotations

import pytest

from dise import estimate
from dise.distributions import Uniform


class _Boom(BaseException):
    """Stand-in for an async timeout (``BaseException``, like ``_CellTimeout``)."""


def test_baseexception_propagates_through_estimate() -> None:
    calls = {"n": 0}

    def program(x: int) -> int:
        # Fire once the run is under way (past the first bootstrap draws),
        # mimicking an alarm landing mid-computation.
        calls["n"] += 1
        if calls["n"] == 25:
            raise _Boom()
        return x

    with pytest.raises(_Boom):
        estimate(
            program=program,
            distribution={"x": Uniform(1, 100)},
            property_fn=lambda y: y < 50,
            epsilon=0.01,
            delta=0.05,
            budget=5000,
            seed=0,
        )


def test_ordinary_exception_still_handled_or_propagates() -> None:
    # A plain Exception from the program should not be silently turned into a
    # confident wrong answer either: it must propagate (the estimator has no
    # basis to fabricate a result for a crashing program).
    def program(x: int) -> int:
        raise ValueError("boom")

    with pytest.raises(ValueError):
        estimate(
            program=program,
            distribution={"x": Uniform(1, 100)},
            property_fn=lambda y: y < 50,
            epsilon=0.01,
            delta=0.05,
            budget=500,
            seed=0,
        )
