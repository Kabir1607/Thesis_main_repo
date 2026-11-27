"""Probabilistic symbolic execution (PSE) baseline.

A purely symbolic competitor to DiSE: enumerate program paths via
SMT-driven concrete-input selection, compute the closed-form
probability mass of each path under the (D1) input distribution, and
sum the masses of paths whose property value is ``1``.

This is the algorithm of Geldenhuys, Dwyer, and Visser
(ISSTA 2012)~\\cite{geldenhuys2012pse} and its descendants (Filieri
et al.\\ TACAS 2015; Pasareanu 2022 synthesis lecture), implemented
here against the same concolic + (D1)-mass machinery as
:class:`~dise.baselines.DiSEBaseline`.  The difference from DiSE is
the *scheduler*: PSE refines exhaustively (DFS over the frontier with
no per-leaf sampling) instead of trading off sample-and-refine under
a gain-per-cost rule.

Where it dominates DiSE:
  - On programs whose paths are bounded and (D1)-mass is closed-form
    on every leaf, PSE returns an exact $\\mu$ with no sampling noise.
  - PSE has no statistical bound (because it does not sample inside
    a closed leaf), so the certified interval is $\\{\\mu_{\\text{exact}}\\}$
    on every fully-enumerated program.

Where it fails:
  - Programs with exponentially many paths (path explosion).  PSE
    times out; DiSE still returns a certified interval.
  - Programs with non-LIA regions whose mass cannot be computed
    in closed form.  PSE either falls back to importance-sampling
    (matching DiSE's mass-conservative refinement, but without
    DiSE's adaptive variance reduction) or refuses to close.

Implementation notes:
  - We bootstrap with ``n_bootstrap`` concolic samples drawn from the
    operational distribution to seed the path-clause vocabulary.
  - Then DFS over open leaves: for each, draw one concrete input from
    the region (via rejection sampling), run concolic to capture its
    path, refine on the first observed branch clause that diverges
    from the leaf's region formula, then attempt closure.
  - Termination: all leaves closed; or max depth reached; or
    ``time_limit_s`` exceeded; or ``max_paths`` open paths in the
    enumeration stack.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ..concolic import run_concolic
from ..distributions import Distribution, ProductDistribution
from ..estimator import _phi_inv_one_sided
from ..regions import Frontier, FrontierNode, Status
from ..sampler import RejectionSampler
from ..smt import SMTBackend, default_backend
from . import Baseline, BaselineResult


class PSEBaseline(Baseline):
    """Exhaustive path-enumeration PSE baseline.

    Drives :class:`~dise.regions.Frontier` to maximum refinement
    via DFS over open leaves, computing exact :math:`\\mu` as the
    sum of CLOSED_TRUE leaf masses.  Falls back to a sampled-mass
    estimate (with its variance recorded) on non-axis-aligned closed
    leaves, matching DiSE's mass-conservative refinement.

    Parameters
    ----------
    n_bootstrap : int, default 200
        Bootstrap concolic samples used to seed the path-clause
        vocabulary at the root.
    max_depth : int, default 50
        Maximum frontier depth; deeper enumeration is cut off and
        the leaf contributes its current mass to ``W_open``.
    max_paths : int, default 10_000
        Hard cap on the DFS frontier size.  A safety net for
        path-explosive programs.
    time_limit_s : float, default 30.0
        Wall-clock cap.  When exceeded, the enumeration stops and
        the current frontier is reported.
    backend : SMTBackend, optional
        SMT backend; defaults to :func:`dise.smt.default_backend`.
    """

    name = "pse_exact"

    def __init__(
        self,
        n_bootstrap: int = 200,
        max_depth: int = 50,
        max_paths: int = 10_000,
        time_limit_s: float = 30.0,
        backend: SMTBackend | None = None,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.max_depth = max_depth
        self.max_paths = max_paths
        self.time_limit_s = time_limit_s
        self.backend = backend

    # ------------------------------------------------------------------
    # Baseline.run
    # ------------------------------------------------------------------

    def run(
        self,
        program: Callable[..., Any],
        distribution: Mapping[str, Distribution],
        property_fn: Callable[[Any], bool],
        budget: int,
        delta: float,
        seed: int,
    ) -> BaselineResult:
        backend = self.backend or default_backend()
        rng = np.random.default_rng(seed)
        dist = ProductDistribution(factors=dict(distribution))
        frontier = Frontier(dist, backend)
        sampler = RejectionSampler()

        samples_used = 0
        t0 = time.perf_counter()

        # ---- Phase 1: bootstrap concolic samples at the root ----
        bootstrap_n = min(self.n_bootstrap, budget)
        batch = dist.sample(rng, bootstrap_n)
        keys = list(distribution.keys())
        for i in range(bootstrap_n):
            x = {k: int(batch[k][i]) for k in keys}
            cr = run_concolic(program, x, property_fn, backend)
            samples_used += 1
            leaf = frontier.find_leaf_for(x)
            if cr.terminated:
                canonical = tuple(
                    backend.repr_expr(b.clause_taken) for b in cr.path_condition
                )
                frontier.add_observation(
                    leaf,
                    canonical,
                    int(cr.phi_value),
                    path_clauses=tuple(b.clause_taken for b in cr.path_condition),
                    concretized=cr.concretized,
                )

        # Try to close root if everything agreed.
        frontier.try_close(frontier.root, min_samples=1)

        # ---- Phase 2: DFS-refine open leaves until budget hits ----
        stack: list[FrontierNode] = list(frontier.open_leaves())
        enumerated_paths = 0
        while stack:
            if time.perf_counter() - t0 > self.time_limit_s:
                break
            if enumerated_paths >= self.max_paths:
                break
            if samples_used >= budget:
                break

            leaf = stack.pop()
            if leaf.status != Status.OPEN:
                continue
            if leaf.depth >= self.max_depth:
                continue

            # Draw one concrete input from this leaf's region.
            sb = sampler.sample(leaf.region, dist, backend, rng, 1)
            if sb.n == 0:
                continue  # Leaf is empty under rejection (rare).
            x = {k: int(sb.inputs[k][0]) for k in keys}
            cr = run_concolic(program, x, property_fn, backend)
            samples_used += 1
            if cr.terminated:
                canonical = tuple(
                    backend.repr_expr(b.clause_taken) for b in cr.path_condition
                )
                frontier.add_observation(
                    leaf,
                    canonical,
                    int(cr.phi_value),
                    path_clauses=tuple(b.clause_taken for b in cr.path_condition),
                    concretized=cr.concretized,
                )

            # Pick the first observed branch clause that's not already
            # part of the leaf's region formula.  If we can find one,
            # refine on it; otherwise, attempt closure.
            clause = self._pick_refinement_clause(leaf, cr, backend)
            if clause is None:
                # Cannot refine further — either path-deterministic or
                # exhausted observed clauses.  Attempt closure.
                if frontier.try_close(leaf, min_samples=1):
                    enumerated_paths += 1
                continue

            try:
                frontier.refine(leaf, clause, rng)
            except Exception:
                # Refinement failed (e.g., importance-sample budget
                # exhausted on a rare region).  Leave the leaf open.
                continue

            # After refinement, push the new children onto the stack
            # and attempt closure on each.
            for child in leaf.children:
                if child.status == Status.OPEN:
                    stack.append(child)
                frontier.try_close(child, min_samples=1)

        wall = time.perf_counter() - t0

        # ---- Phase 3: compute the certified interval ----
        mu_hat = 0.0
        w_open = 0.0
        leaves = list(frontier.leaves())
        mass_leaves = [
            leaf
            for leaf in leaves
            if leaf.status == Status.CLOSED_TRUE and leaf.w_var > 0.0
        ]
        # Bonferroni-split delta over the mass-carrying closed-true leaves
        # and derive z from it (not a hardcoded 1.96), matching DiSE.
        k_bounds = max(len(mass_leaves), 1)
        z = _phi_inv_one_sided(1.0 - (delta / k_bounds) / 2.0)
        eps_mass = 0.0
        for leaf in leaves:
            if leaf.status == Status.CLOSED_TRUE:
                mu_hat += leaf.w_hat
                if leaf.w_var > 0.0:
                    eps_mass += z * (leaf.w_var ** 0.5)
            elif leaf.status == Status.OPEN:
                w_open += leaf.w_hat
        eps_mass = min(eps_mass, 1.0)

        lo = max(0.0, mu_hat - w_open - eps_mass)
        hi = min(1.0, mu_hat + w_open + eps_mass)

        return BaselineResult(
            name=self.name,
            mu_hat=mu_hat,
            interval=(lo, hi),
            samples_used=samples_used,
            wall_clock_s=wall,
            delta=delta,
            extras={
                "n_leaves": frontier.n_leaves(),
                "n_open_leaves": sum(
                    1 for lf in frontier.leaves() if lf.status == Status.OPEN
                ),
                "n_closed_true": sum(
                    1 for lf in frontier.leaves() if lf.status == Status.CLOSED_TRUE
                ),
                "n_closed_false": sum(
                    1 for lf in frontier.leaves() if lf.status == Status.CLOSED_FALSE
                ),
                "enumerated_paths": enumerated_paths,
                "W_open": w_open,
                "eps_mass": eps_mass,
                "hit_time_limit": wall >= self.time_limit_s,
                "hit_path_limit": enumerated_paths >= self.max_paths,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_refinement_clause(
        leaf: FrontierNode, cr, backend: SMTBackend
    ):
        """Pick the first observed branch clause that is NOT already
        implied by the leaf's region formula.

        Returns ``None`` if every observed clause is already subsumed
        (the leaf is effectively path-deterministic on the observed
        prefix) or if the trace has no branches.
        """
        if not cr.path_condition:
            return None
        F = leaf.region.formula
        for b in cr.path_condition:
            clause = b.clause_taken
            # Is the clause already implied by F?  Check: F AND NOT clause
            # is unsat means clause is implied.
            try:
                check = backend.conjunction(F, backend.negation(clause))
                if backend.is_satisfiable(check) == "unsat":
                    continue  # Already implied; not useful for refinement.
            except Exception:
                # Backend can't decide; skip cautiously.
                continue
            return clause
        return None


__all__ = ["PSEBaseline"]
