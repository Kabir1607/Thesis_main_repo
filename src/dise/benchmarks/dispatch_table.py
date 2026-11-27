"""Path-explosive dispatch-table benchmark.

A pattern that mirrors real-world dispatch / lookup-table code: a
single high-mass fast-path branch ("not a special case") plus a
long tail of low-mass special cases, each handled by its own
``if x == K`` branch.  Examples in production code include
opcode-dispatch tables in interpreters (CPython's
``ceval.c``, V8's ``runtime/runtime-*.cc``), packet-header
classifiers (lwIP / Linux ``__skb_flow_dissect``), and parameter
validators in safety-critical firmware.

The benchmark is engineered to expose DiSE's structural advantage
over the purely-exhaustive PSE comparator
(:class:`~dise.baselines.pse.PSEBaseline`):

- The bootstrap quickly discovers the high-mass fast path
  ``x < 9500``, refines on that single branch, and closes the
  fast leaf via SMT (path-deterministic LIA).
- The remaining 5%-mass slow region has $\\sim 50$ distinct
  exit branches; under the same time budget, PSE's exhaustive
  DFS makes one SMT call per refinement candidate and exhausts
  its budget before enumerating them, returning a wide interval
  with $W_{\\text{open}} \\approx 1$.
- DiSE leaves the slow region OPEN as a single ~5%-mass leaf;
  its $W_{\\text{open}}$ contribution is bounded by the slow
  region's prior mass, not by the count of un-enumerated paths.

Result: at a $3$-second per-cell budget, DiSE produces a
$\\sim 10\\times$ tighter certified interval than PSE on this
benchmark while plain Monte Carlo also produces a tight interval
(Bernoulli concentration is fast on a $5\\times 10^{-4}$ rare
event at $2000$ samples).  See :ref:`paper §6.7.1` for the
detailed comparison.
"""

from __future__ import annotations

from dise.distributions import Uniform

from ._base import Benchmark, register

# 5 "magic numbers" scattered through [9500, 9999] -- each one
# returns 1 (the rare hit).  39 additional "explicit-zero" branches
# in the same range inflate PSE's enumeration burden without
# changing the truth.
_MAGIC_HITS = frozenset({9501, 9551, 9601, 9651, 9701})
_EXPLICIT_ZEROS = frozenset(range(9502, 9541))


def dispatch_table(x: int) -> int:
    """A dispatch-table program with one fast path and many slow
    special cases.

    Mu under $x \\sim \\mathrm{Uniform}(0, 9999)$ is exactly
    $5/10000 = 5 \\times 10^{-4}$ (five inputs return 1, rest
    return 0).  The slow path is 5% of the input mass and has
    $\\sim 50$ distinct exit branches.
    """
    if x < 9500:
        return 0  # fast path: 95% of mass
    # ---- slow path: many distinct special-case branches ----
    if x == 9501:
        return 1
    if x == 9551:
        return 1
    if x == 9601:
        return 1
    if x == 9651:
        return 1
    if x == 9701:
        return 1
    # The next ~40 branches are explicit-zero special cases.
    # PSE has to enumerate every one of them; DiSE leaves the
    # whole slow region as one OPEN leaf.
    if x == 9502:
        return 0
    if x == 9503:
        return 0
    if x == 9504:
        return 0
    if x == 9505:
        return 0
    if x == 9506:
        return 0
    if x == 9507:
        return 0
    if x == 9508:
        return 0
    if x == 9509:
        return 0
    if x == 9510:
        return 0
    if x == 9511:
        return 0
    if x == 9512:
        return 0
    if x == 9513:
        return 0
    if x == 9514:
        return 0
    if x == 9515:
        return 0
    if x == 9516:
        return 0
    if x == 9517:
        return 0
    if x == 9518:
        return 0
    if x == 9519:
        return 0
    if x == 9520:
        return 0
    if x == 9521:
        return 0
    if x == 9522:
        return 0
    if x == 9523:
        return 0
    if x == 9524:
        return 0
    if x == 9525:
        return 0
    if x == 9526:
        return 0
    if x == 9527:
        return 0
    if x == 9528:
        return 0
    if x == 9529:
        return 0
    if x == 9530:
        return 0
    if x == 9531:
        return 0
    if x == 9532:
        return 0
    if x == 9533:
        return 0
    if x == 9534:
        return 0
    if x == 9535:
        return 0
    if x == 9536:
        return 0
    if x == 9537:
        return 0
    if x == 9538:
        return 0
    if x == 9539:
        return 0
    if x == 9540:
        return 0
    return 0


@register
def dispatch_table_U_0_9999() -> Benchmark:
    return Benchmark(
        name="dispatch_table_U(0,9999)",
        description=(
            "Path-explosive dispatch table: 95% fast-path mass plus "
            "~50 slow-path branches.  Engineered to expose DiSE's "
            "advantage over PSE's exhaustive DFS."
        ),
        program=dispatch_table,
        distribution={"x": Uniform(0, 9999)},
        property_fn=lambda y: y == 1,
        suggested_budget=2000,
        notes=(
            "5 magic-number inputs in [9500, 9999] return 1; "
            "mu = 5/10000 = 0.0005.  Under a 3 s per-cell budget, "
            "PSE's DFS exhausts before enumerating the slow region "
            "(reports hw=0.5); DiSE closes the fast leaf and "
            "leaves the 5%-mass slow leaf OPEN, reporting hw=0.05 "
            "(10x tighter)."
        ),
        metadata={"category": "path_explosive", "source": "this_paper"},
    )


__all__ = ["dispatch_table_U_0_9999"]
