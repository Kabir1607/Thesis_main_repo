"""Certified Adaptive Importance Sampling (AIS) for unclosable OPEN leaves.

Opt-in extension of DiSE (``SchedulerConfig(enable_ais=True)`` or
``method="ais_betting"``).  Two building blocks:

* :class:`~dise.ais.proposals.TwistedBGProposal` -- defensive mixture of
  ``D|_R`` and an exponentially twisted product of truncated geometrics on
  the leaf's box, with exact log-PMFs, overflow-safe inverse-CDF sampling,
  exact likelihood ratios ``W in [0, 1/alpha]``, and a bisection
  cross-entropy (moment-matching) updater.
* :class:`~dise.ais.betting_cs.OffPolicyBettingCS` -- grid-mixture betting
  confidence sequence on ``Z = W * R`` with a fixed a-priori leaf budget
  ``delta_pi``, giving explicit asymmetric, anytime-valid bounds
  ``[L_fail, U_fail]`` with running intersection.

Sign convention (matches DiSE): ``phi = 1`` is PASS; ``R = 1 - phi`` is the
failure indicator; a leaf's certified reliability interval is
``[1 - U_fail, 1 - L_fail]``.

Asymmetry (see ``docs`` / ``changes.md``): AIS can make certified
*falsification* (a lower bound on the failure rate) dramatically cheaper,
but cannot remove the ``Omega(1/p)`` floor for certified *verification*
(an upper bound) inside an unclosable leaf -- the upper-bound bet is capped
by ``W_max = 1/alpha``, so small ``alpha`` hurts verification.
"""

from .betting_cs import OffPolicyBettingCS, default_gamma_grid
from .proposals import (
    ProposalDraw,
    TwistedBGProposal,
    box_bounds_for_region,
    solve_eta_for_mean,
    trunc_geom_log_z,
    trunc_geom_logpmf,
    trunc_geom_mean,
    trunc_geom_sample,
)

__all__ = [
    "OffPolicyBettingCS",
    "ProposalDraw",
    "TwistedBGProposal",
    "box_bounds_for_region",
    "default_gamma_grid",
    "solve_eta_for_mean",
    "trunc_geom_log_z",
    "trunc_geom_logpmf",
    "trunc_geom_mean",
    "trunc_geom_sample",
]
