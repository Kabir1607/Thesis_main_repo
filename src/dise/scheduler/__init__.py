"""ASIP (Adaptive Symbolic Importance Partitioning) scheduler.

The :class:`ASIPScheduler` ties together concolic execution, sampling,
refinement, and the certified estimator. At each iteration it picks one
action — allocate more samples to an open leaf, or refine an open leaf
on a divergent branch — based on expected variance reduction per unit
cost.

Termination is reached when:

* ``samples_used >= budget_samples``  → ``terminated_reason='budget_exhausted'``
* ``eps_stat + W_open <= epsilon``    → ``terminated_reason='epsilon_reached'``
* No action with positive gain exists → ``terminated_reason='no_actions_available'``
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..concolic import BranchRecord, ConcolicResult, run_concolic
from ..distributions import ProductDistribution
from ..estimator import EstimatorState, compute_estimator_state
from ..regions import Frontier, FrontierNode, Status
from ..sampler import RejectionSampler, Sampler
from ..smt import SMTBackend, SMTExpr

# -----------------------------------------------------------------------------
# Config / logging / results
# -----------------------------------------------------------------------------


@dataclass
class SchedulerConfig:
    """Configuration for :class:`ASIPScheduler`.

    Termination is governed by four orthogonal knobs. The algorithm
    halts at the **first** condition that fires:

    * ``epsilon`` (always active) — stop when
      :math:`\\varepsilon_{\\text{stat}} + W_{\\text{open}} \\le \\varepsilon`.
    * ``budget_samples`` — optional cap on concolic runs.
      ``None`` disables the cap (recommended for soundness-mode runs;
      see :class:`SchedulerResult.terminated_reason`).
    * ``budget_seconds`` — optional wall-clock cap. ``None`` disables.
    * ``min_gain_per_cost`` — diminishing-returns floor; the algorithm
      declares ``"no_actions_available"`` when the best candidate
      action's expected gain-per-cost falls below this threshold.

    Setting *all* of ``budget_samples``, ``budget_seconds``, and
    ``min_gain_per_cost`` to permissive values yields a "soundness-
    only" run: the algorithm runs until ``epsilon_reached``. For
    pathologically hard targets this may not terminate; configure at
    least one cap unless you trust the target is reachable.
    """

    epsilon: float = 0.05
    delta: float = 0.05
    # ``None`` disables the cap; the algorithm runs until the target
    # ``epsilon`` is reached or another cap fires.
    budget_samples: int | None = 10_000
    # ``None`` disables the wall-clock cap.
    budget_seconds: float | None = None
    # The algorithm stops when the best action's expected
    # gain-per-cost falls below this threshold. Default 0 keeps
    # backward-compatible behavior.
    min_gain_per_cost: float = 0.0
    # Certified half-width method (see ``compute_estimator_state``).
    # ``"wilson"`` is tightest at fixed n. ``"anytime"`` is sound under
    # the scheduler's adaptive stopping rule. ``"betting"`` is the WSR
    # 2024 PrPl-EB variant — same adaptive-schedule soundness as
    # ``"anytime"`` but tighter in low-variance regimes; the
    # recommended setting for ATVA-style certificates.
    method: Literal[
        "wilson",
        "anytime",
        "betting",
        "betting_strict",
        "ais_betting",
        "bernstein",
        "empirical-bernstein",
    ] = "wilson"
    bootstrap_samples: int = 200
    batch_size: int = 50
    refinement_cost_in_samples: float = 1.0
    max_refinement_depth: int = 50
    n_mass_samples: int = 1000
    smt_timeout_ms: int = 5000
    closure_min_samples: int = 5
    max_concolic_branches: int = 10_000
    verbose: bool = False

    # ---- Certified AIS (opt-in; see dise.ais) ----
    # ``enable_ais=True`` or ``method="ais_betting"`` switches on AIS
    # inside unclosable OPEN leaves *and* the Definition-6 asymmetric
    # estimator.  With both off, behaviour is exactly the baseline.
    enable_ais: bool = False
    # Defensive mixture weight on D|_R (W <= 1 / ais_alpha).  Suggested:
    # 0.5 two-sided, 0.1 falsification, 0.9 verification.
    ais_alpha: float = 0.5
    # Fixed a-priori number of per-leaf confidence sequences; each gets
    # delta_pi = delta / ais_k_max.  Leaves beyond the budget get [0, 1].
    ais_k_max: int = 1024
    # Error-budget split of each leaf CS: two-sided, all on the lower
    # failure bound ("falsify"), or all on the upper ("verify").
    ais_mode: Literal["two_sided", "falsify", "verify"] = "two_sided"
    # Optional SLA threshold on the failure rate 1 - mu.  Stops early on
    # VERIFIED (1 - lo <= tau) or FALSIFIED (1 - hi > tau).  Applies to
    # every method (default None: disabled).
    threshold_tau: float | None = None
    # CE smoothing: eta <- (1 - rho) eta + rho eta_CE.
    ais_ce_rho: float = 0.7
    # Samples per AIS batch (proposal updated between batches);
    # ``None`` -> ``batch_size``.
    ais_batch_size: int | None = None
    # Mark a leaf unclosable after this many ``unknown`` SMT closure answers.
    ais_unknown_threshold: int = 2

    @property
    def ais_enabled(self) -> bool:
        return self.enable_ais or self.method == "ais_betting"

    @property
    def estimator_method(self) -> str:
        return "ais_betting" if self.ais_enabled else self.method


@dataclass
class IterationLog:
    iter_idx: int
    action_kind: str
    leaf_depth: int
    samples_used_after: int
    mu_hat: float
    eps_stat: float
    W_open: float
    interval: tuple[float, float]
    n_leaves: int
    n_open_leaves: int


@dataclass
class SchedulerResult:
    final_estimator: EstimatorState
    iterations: list[IterationLog]
    samples_used: int
    refinements_done: int
    smt_calls: int
    terminated_reason: str
    frontier: Frontier
    # Concolic runs drawn from an AIS proposal (subset of samples_used).
    ais_samples_used: int = 0
    # Final SLA decision when ``threshold_tau`` was configured.
    decision: str | None = None

    def __repr__(self) -> str:
        lo, hi = self.final_estimator.interval
        return (
            f"SchedulerResult(mu_hat={self.final_estimator.mu_hat:.4f}, "
            f"interval=[{lo:.4f}, {hi:.4f}], "
            f"samples={self.samples_used}, refinements={self.refinements_done}, "
            f"terminated={self.terminated_reason!r})"
        )


# -----------------------------------------------------------------------------
# Internal action representation
# -----------------------------------------------------------------------------


@dataclass
class _Action:
    kind: Literal["allocate", "refine", "ais_allocate"]
    leaf: FrontierNode
    expected_gain: float
    cost: float
    k: int = 0
    clause: SMTExpr | None = None

    @property
    def gain_per_cost(self) -> float:
        if self.cost <= 0:
            return float("inf") if self.expected_gain > 0 else 0.0
        return self.expected_gain / self.cost


# -----------------------------------------------------------------------------
# Scheduler
# -----------------------------------------------------------------------------


class ASIPScheduler:
    """The main DiSE driver."""

    def __init__(
        self,
        program: Callable[..., Any],
        distribution: ProductDistribution,
        property_fn: Callable[[Any], bool],
        smt: SMTBackend,
        config: SchedulerConfig,
        rng: np.random.Generator,
        sampler: Sampler | None = None,
    ) -> None:
        self.program = program
        self.distribution = distribution
        self.property_fn = property_fn
        self.smt = smt
        self.config = config
        self.rng = rng
        self.sampler: Sampler = sampler if sampler is not None else RejectionSampler()
        self.frontier = Frontier(distribution, smt, n_mc_for_mass=config.n_mass_samples)
        self.samples_used = 0
        self.refinements_done = 0
        self.smt_calls = 0
        self.iterations: list[IterationLog] = []
        self.terminated_reason = "not_terminated"
        # Wall-clock timer, set at `run` time.
        self._start_time: float | None = None
        # Per-leaf raw path conditions and phi values (used to choose
        # refinement clauses). Scheduler-local; not part of Frontier state.
        self._leaf_paths: dict[int, list[list[BranchRecord]]] = {}
        self._leaf_path_phis: dict[int, list[int]] = {}
        # ---- AIS bookkeeping (inactive unless config.ais_enabled) ----
        self._ais = config.ais_enabled
        # Per-leaf CSs issued so far; capped at ais_k_max so the union
        # bound sum_pi delta_pi <= delta holds over every leaf ever bounded.
        self._n_cs_issued = 0
        self.ais_samples_used = 0
        # Cache: canonical clause key -> clause has free variables.
        self._clause_symbolic: dict[object, bool] = {}
        # Per-leaf number of failures already used by the last CE fit.
        self._ce_fail_used: dict[int, int] = {}
        if self._ais:
            if not 0.0 < config.ais_alpha <= 1.0:
                raise ValueError("ais_alpha must be in (0, 1]")
            if config.ais_k_max < 1:
                raise ValueError("ais_k_max must be >= 1")

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _run_concolic(self, x: dict[str, int]) -> ConcolicResult:
        return run_concolic(
            self.program,
            x,
            self.property_fn,
            self.smt,
            max_branches=self.config.max_concolic_branches,
        )

    def _leaf_key(self, leaf: FrontierNode) -> int:
        return id(leaf)

    def _record_observation(
        self,
        leaf: FrontierNode,
        result: ConcolicResult,
        weight: float = 1.0,
        alpha: float = 1.0,
    ) -> None:
        if self._ais:
            # Feed the leaf's confidence sequence *before* the
            # termination filter: dropping diverged runs would bias the
            # CS, so a run that exceeded the branch budget counts as a
            # failure (conservative for the reliability bound).
            r = 0 if (result.terminated and result.phi_value) else 1
            self._cs_observe(leaf, weight * r, alpha)
        if not result.terminated:
            return
        canonical = tuple(
            self.smt.canonical_key(br.clause_taken) for br in result.path_condition
        )
        path_clauses = tuple(br.clause_taken for br in result.path_condition)
        self.frontier.add_observation(
            leaf, canonical, result.phi_value,
            path_clauses=path_clauses,
            concretized=result.concretized,
        )
        key = self._leaf_key(leaf)
        self._leaf_paths.setdefault(key, []).append(list(result.path_condition))
        self._leaf_path_phis.setdefault(key, []).append(int(result.phi_value))

    def _budget_remaining(self) -> int | None:
        """Concolic-run budget left before the sample cap fires, or
        ``None`` if no sample cap is configured."""
        if self.config.budget_samples is None:
            return None
        return self.config.budget_samples - self.samples_used

    def _time_exhausted(self) -> bool:
        if self.config.budget_seconds is None or self._start_time is None:
            return False
        return (time.perf_counter() - self._start_time) >= self.config.budget_seconds

    def _allocate_one_batch(self, leaf: FrontierNode, k: int) -> int:
        """Draw ``k`` samples from ``leaf.region``, run concolic on each,
        record observations. Returns the number of concolic runs actually
        performed (may be < k if the sample cap or time cap fires).
        """
        if leaf.status != Status.OPEN:
            return 0
        remaining = self._budget_remaining()
        if remaining is not None:
            if remaining <= 0:
                return 0
            k = min(k, remaining)
        batch = self.sampler.sample(
            leaf.region, self.distribution, self.smt, self.rng, k
        )
        n_run = 0
        for x in batch.iter_assignments():
            remaining = self._budget_remaining()
            if remaining is not None and remaining <= 0:
                break
            if self._time_exhausted():
                break
            result = self._run_concolic(x)
            self.samples_used += 1
            n_run += 1
            target = self.frontier.find_leaf_for(x)
            self._record_observation(target, result)
        return n_run

    # ------------------------------------------------------------------
    # Certified AIS (only reached when config.ais_enabled)
    # ------------------------------------------------------------------

    def _cs_observe(self, leaf: FrontierNode, z: float, alpha: float) -> None:
        """Append one ``Z = W * R`` (cap ``1/alpha``) to ``leaf.ais_cs``.

        The CS is created on the leaf's first observation with the fixed
        a-priori budget ``delta / K_max``.  Once ``K_max`` CSs have been
        issued, further leaves stay unbounded (``[0, 1]`` in Def. 6).
        Plain ``D|_R`` observations enter with ``alpha = 1`` (``W = 1``),
        for which the CS reduces to a Bernoulli betting CS.
        """
        if leaf.ais_cs is None:
            if self._n_cs_issued >= self.config.ais_k_max:
                return
            from ..ais import OffPolicyBettingCS

            leaf.ais_cs = OffPolicyBettingCS(
                delta=self.config.delta / self.config.ais_k_max,
                mode=self.config.ais_mode,
            )
            self._n_cs_issued += 1
        leaf.ais_cs.update([z], alpha)

    def _is_symbolic_clause(self, clause: SMTExpr) -> bool:
        """True iff ``clause`` mentions an input variable.  Ground clauses
        (``const op const``, emitted after a concretization) never make
        refinement progress: one child is EMPTY, the other is the parent."""
        key = self.smt.canonical_key(clause)
        hit = self._clause_symbolic.get(key)
        if hit is None:
            hit = len(self.smt.free_vars(clause)) > 0
            self._clause_symbolic[key] = hit
        return hit

    def _ais_box(self, leaf: FrontierNode) -> dict[str, tuple[int, int]] | None:
        from ..ais import box_bounds_for_region

        return box_bounds_for_region(leaf.region, self.distribution)

    def _is_unclosable(self, leaf: FrontierNode, has_refine: bool) -> bool:
        """Sticky detection of leaves Z3 cannot close.

        A leaf is unclosable once any of these holds:

        * an observation crossed a concretization point (``try_close``
          will refuse forever);
        * the SMT backend answered ``unknown`` on closure
          ``ais_unknown_threshold`` times;
        * it sits at ``max_refinement_depth``;
        * it has ``closure_min_samples`` observations, is still OPEN after
          the closure attempt, and no symbolic clause is left to refine on.
        """
        if leaf.is_unclosable:
            return True
        if leaf.n_samples == 0:
            return False
        cfg = self.config
        if (
            leaf.has_concretized_obs
            or leaf.closure_unknown_count >= cfg.ais_unknown_threshold
            or leaf.depth >= cfg.max_refinement_depth
            or (leaf.n_samples >= cfg.closure_min_samples and not has_refine)
        ):
            leaf.is_unclosable = True
        return leaf.is_unclosable

    def _allocate_ais_batch(self, leaf: FrontierNode, k: int) -> int:
        """One AIS batch in an unclosable leaf.

        1. Fit the proposal (CE) *strictly from earlier batches*.
        2. Draw ``k`` points from ``q_t = alpha D|_R + (1-alpha) q~_eta``.
        3. Run ``P`` concolically; record ``Z = W R`` (cap ``1/alpha``)
           in ``leaf.ais_cs`` and the usual FrontierNode statistics.
        Returns the number of concolic runs performed.
        """
        if leaf.status != Status.OPEN:
            return 0
        remaining = self._budget_remaining()
        if remaining is not None:
            if remaining <= 0:
                return 0
            k = min(k, remaining)
        box = self._ais_box(leaf)
        if box is None:
            return self._allocate_one_batch(leaf, k)
        from ..ais import TwistedBGProposal

        key = self._leaf_key(leaf)
        if leaf.ais_proposal is None:
            # Bootstrap tilt: uniform on the box (r_i = 1), inside the
            # defensive mixture -- explores the low-probability tail.
            leaf.ais_proposal = TwistedBGProposal(
                self.distribution, box, alpha=self.config.ais_alpha
            )
            self._ce_fail_used[key] = 0
        prop = leaf.ais_proposal
        # Predictable CE update: uses only failures from batches 1..b-1.
        n_fail = len(leaf.ais_fail_w)
        if prop.alpha < 1.0 and n_fail > self._ce_fail_used.get(key, 0):
            prop.update_ce(
                np.vstack(leaf.ais_fail_x),
                np.asarray(leaf.ais_fail_w),
                rho=self.config.ais_ce_rho,
            )
            self._ce_fail_used[key] = n_fail
        draw = prop.sample(self.rng, k)
        alpha = draw.alpha
        n_run = 0
        new_fail_x: list[np.ndarray] = []
        new_fail_w: list[float] = []
        for i, x in enumerate(draw.assignments()):
            remaining = self._budget_remaining()
            if remaining is not None and remaining <= 0:
                break
            if self._time_exhausted():
                break
            result = self._run_concolic(x)
            self.samples_used += 1
            self.ais_samples_used += 1
            n_run += 1
            w = float(draw.weights[i])
            self._record_observation(leaf, result, weight=w, alpha=alpha)
            failed = not (result.terminated and result.phi_value)
            if failed:
                new_fail_x.append(draw.x[i].astype(np.float64))
                new_fail_w.append(w)
        # Failures become visible to the CE fit only after the batch.
        leaf.ais_fail_x.extend(new_fail_x)
        leaf.ais_fail_w.extend(new_fail_w)
        leaf.n_ais_samples += n_run
        return n_run

    # ------------------------------------------------------------------
    # Closure
    # ------------------------------------------------------------------

    def _try_close_all(self) -> int:
        n_closed = 0
        for leaf in self.frontier.open_leaves():
            if self.frontier.try_close(leaf, self.config.closure_min_samples):
                n_closed += 1
                # Drop the leaf's local paths once closed.
                key = self._leaf_key(leaf)
                self._leaf_paths.pop(key, None)
                self._leaf_path_phis.pop(key, None)
        return n_closed

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def _best_refinement_clause(
        self,
        leaf: FrontierNode,
        paths: list[list[BranchRecord]],
        path_phis: list[int],
    ) -> tuple[SMTExpr | None, float]:
        """Pick the best refinement clause for ``leaf``.

        Candidate clauses are the observed branch clauses not already in
        ``F_pi`` (the refining clauses of the leaf and its ancestors).

        When the observed samples take *divergent* paths, each candidate
        ``b`` is scored by the informational gain

            G(b) = V_pi - V_{pi ^ b} - V_{pi ^ ~b}

        (Wilson-smoothed per-side variance), and the maximiser is
        returned.

        When no candidate separates the observed samples -- a single
        observed path, or several that agree -- the variance score is
        degenerate, but refinement is still valid: a clause and its
        negation always partition the leaf.  We then refine on the
        earliest observed branch clause, mirroring PSE's prefix-wise
        exploration.  This is what lets the adaptive schedule make
        progress when the input distribution concentrates mass on a
        single path: it no longer waits for sampled paths to diverge.
        The gain-per-cost scheduler still governs whether the leaf's
        mass justifies the refinement.

        Returns ``(None, 0.0)`` only when there is genuinely no branch
        clause past ``leaf.depth`` to split on.
        """
        if not paths:
            return None, 0.0
        # Clauses already conjoined into F_pi: the refining clauses of
        # this leaf and its ancestors. A candidate equal to one of them
        # is a tautology given F_pi -- refining on it yields an empty
        # sibling and no progress -- so they are excluded.
        in_formula: set[object] = set()
        node: FrontierNode | None = leaf
        while node is not None:
            if node.refining_clause is not None:
                in_formula.add(self.smt.canonical_key(node.refining_clause))
            node = node.parent
        # Canonicalise every branch once, then reuse the keys both for
        # the candidate set and for the per-sample membership test below.
        canonical_paths = [
            [self.smt.canonical_key(br.clause_taken) for br in path] for path in paths
        ]
        # Candidate clauses: every observed branch clause not already in
        # F_pi. Tree depth does not align with path position -- a
        # refinement may split on any branch, not a prefix -- so the
        # candidate set is the whole observed path minus the F_pi
        # clauses, not a depth-indexed suffix.
        candidates: dict[object, SMTExpr] = {}
        for path, can in zip(paths, canonical_paths, strict=True):
            for k in range(len(path)):
                if can[k] in in_formula:
                    continue
                # AIS mode only: skip ground clauses (no free variables),
                # which can never split a region.  Baseline behaviour is
                # left untouched.
                if self._ais and not self._is_symbolic_clause(path[k].clause_taken):
                    continue
                candidates.setdefault(can[k], path[k].clause_taken)
        if not candidates:
            return None, 0.0
        w_pi = leaf.w_hat
        v_pi = leaf.variance_contribution

        best_clause: SMTExpr | None = None
        best_gain = 0.0

        for key, clause_expr in candidates.items():
            n_true = 0
            h_true = 0
            n_false = 0
            h_false = 0
            for _path, phi, can in zip(paths, path_phis, canonical_paths, strict=True):
                # A sample's path "takes" the clause iff `key` appears in its
                # canonical sequence.
                if key in can:
                    n_true += 1
                    if phi:
                        h_true += 1
                else:
                    n_false += 1
                    if phi:
                        h_false += 1
            n_total = n_true + n_false
            if n_total == 0 or n_true == 0 or n_false == 0:
                # Doesn't separate the samples; not informative.
                continue
            f_b = n_true / n_total
            # Wilson-smoothed per-side mu_var
            p_t = (h_true + 1) / (n_true + 2)
            p_f = (h_false + 1) / (n_false + 2)
            mu_var_t = p_t * (1.0 - p_t)
            mu_var_f = p_f * (1.0 - p_f)
            w_t = w_pi * f_b
            w_f = w_pi * (1.0 - f_b)
            v_b = w_t * w_t * mu_var_t / max(n_true, 1)
            v_nb = w_f * w_f * mu_var_f / max(n_false, 1)
            gain = v_pi - (v_b + v_nb)
            if gain > best_gain:
                best_gain = gain
                best_clause = clause_expr

        if best_clause is not None:
            return best_clause, best_gain
        # No observed clause separated the samples (a single observed
        # path, or several that agree). Refine on the earliest observed
        # branch clause: a valid, progress-making split, prioritised
        # across leaves by the gain-per-cost schedule.
        first_clause = next(iter(candidates.values()))
        return first_clause, max(v_pi, 1e-12)

    def _execute_refine(self, leaf: FrontierNode, clause: SMTExpr) -> None:
        if leaf.depth >= self.config.max_refinement_depth:
            return
        try:
            children = self.frontier.refine(leaf, clause, self.rng)
        except ValueError:
            return
        self.smt_calls += 2  # two satisfiability checks per refinement
        self.refinements_done += 1
        # Drop scheduler-side paths for the parent; children start fresh.
        parent_key = self._leaf_key(leaf)
        self._leaf_paths.pop(parent_key, None)
        self._leaf_path_phis.pop(parent_key, None)
        # Seed each non-empty child with at least a few samples so closure
        # can be evaluated.
        seed_n = min(self.config.batch_size, self.config.closure_min_samples * 2)
        for c in children:
            if c.status == Status.OPEN:
                self._allocate_one_batch(c, seed_n)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _leaf_hw_proxy(self, leaf: FrontierNode, n: int) -> float:
        """Monotone proxy in [0, 1] for a leaf's per-leaf half-width
        ``W_pi`` after ``n`` samples: 1 when unsampled, shrinking as
        ``~sqrt(var / n)`` once sampled. Used only to *rank* candidate
        actions; the certified interval uses the real anytime bound.
        """
        if n <= 0:
            return 1.0
        return min(1.0, 2.0 * (leaf.mu_var / n) ** 0.5)

    def _candidate_actions(self) -> list[_Action]:
        """Enumerate the candidate actions on the current frontier.

        Both gains are expressed as reductions of the certified
        half-width ``eps_stat + W_open`` (Section 5.4), so allocate and
        refine are directly comparable:

        * \\textsc{Allocate} reduces ``eps_stat``: the gain is the
          shrinkage of the leaf's contribution ``w_pi * W_pi`` from
          ``k`` further samples.
        * \\textsc{Refine} reduces ``W_open``: refining a leaf toward
          symbolic closure removes up to its whole mass ``w_pi`` from
          the certified half-width, so the gain is ``w_pi``. A pure
          variance metric undervalues this -- a path-deterministic leaf
          has near-zero sample variance yet its mass can be discharged
          only by closure, never by sampling.
        """
        actions: list[_Action] = []
        k = self.config.batch_size
        for leaf in self.frontier.open_leaves():
            n = leaf.n_samples
            w = leaf.w_hat
            if w <= 0.0:
                continue

            # Refine: reduce W_open. Available whenever an observed path
            # supplies a branch clause past leaf.depth to split on, and
            # the depth cap allows. Refinement no longer waits for
            # sampled paths to diverge -- a single observed path's
            # clauses already partition the leaf, the common case once
            # the input distribution concentrates mass on one path.
            refine_action: _Action | None = None
            if leaf.depth < self.config.max_refinement_depth:
                paths = self._leaf_paths.get(self._leaf_key(leaf), [])
                phis = self._leaf_path_phis.get(self._leaf_key(leaf), [])
                if paths:
                    clause, _ = self._best_refinement_clause(leaf, paths, phis)
                    if clause is not None:
                        refine_action = _Action(
                            kind="refine",
                            leaf=leaf,
                            expected_gain=w,
                            cost=self.config.refinement_cost_in_samples,
                            clause=clause,
                        )

            ais_action: _Action | None = None
            if (
                self._ais
                and self._is_unclosable(leaf, has_refine=refine_action is not None)
                and leaf.ais_cs is not None
                and self._ais_box(leaf) is not None
            ):
                # AIS allocate: priced by the predicted drop in the leaf's
                # certified contribution w_pi * (U_pi - L_pi), using the
                # ~1/sqrt(n) shrinkage heuristic (ranking only).
                k_ais = self.config.ais_batch_size or k
                cs = leaf.ais_cs
                lo_f, hi_f = cs.bounds()
                width = max(hi_f - lo_f, 0.0)
                n_cs = cs.n
                shrink = 1.0 - (n_cs / (n_cs + k_ais)) ** 0.5 if n_cs > 0 else 1.0
                ais_action = _Action(
                    kind="ais_allocate",
                    leaf=leaf,
                    expected_gain=w * width * shrink,
                    cost=float(k_ais),
                    k=k_ais,
                )

            if ais_action is not None:
                actions.append(ais_action)
            else:
                # Allocate: reduce eps_stat.
                hw_now = self._leaf_hw_proxy(leaf, n)
                hw_next = self._leaf_hw_proxy(leaf, n + k)
                alloc_gain = w * max(0.0, hw_now - hw_next)
                actions.append(
                    _Action(
                        kind="allocate",
                        leaf=leaf,
                        expected_gain=alloc_gain,
                        cost=float(k),
                        k=k,
                    )
                )
            if refine_action is not None:
                actions.append(refine_action)
        return actions

    def _pick_best_action(self, actions: list[_Action]) -> _Action | None:
        if not actions:
            return None
        # Sort by gain_per_cost desc; tie-break preferring "refine" (structural).
        def key(a: _Action) -> tuple[float, int]:
            return (a.gain_per_cost, 1 if a.kind == "refine" else 0)
        actions.sort(key=key, reverse=True)
        best = actions[0]
        if best.gain_per_cost <= self.config.min_gain_per_cost:
            return None
        return best

    # ------------------------------------------------------------------
    # Bootstrap + main loop
    # ------------------------------------------------------------------

    def bootstrap(self) -> None:
        """Initial sampling pass: draw ``bootstrap_samples`` from ``D``
        and attribute each to the root."""
        n = self.config.bootstrap_samples
        if self.config.budget_samples is not None:
            n = min(n, self.config.budget_samples)
        self._allocate_one_batch(self.frontier.root, n)

    def _estimate(self) -> EstimatorState:
        return compute_estimator_state(
            self.frontier,
            self.config.delta,
            method=self.config.estimator_method,  # type: ignore[arg-type]
            k_max=self.config.ais_k_max,
            threshold_tau=self.config.threshold_tau,
        )

    def _should_terminate(self) -> tuple[bool, EstimatorState]:
        state = self._estimate()
        if (
            self.config.budget_samples is not None
            and self.samples_used >= self.config.budget_samples
        ):
            self.terminated_reason = "budget_exhausted"
            return True, state
        if self._time_exhausted():
            self.terminated_reason = "time_exhausted"
            return True, state
        if state.decision == "VERIFIED":
            self.terminated_reason = "threshold_verified"
            return True, state
        if state.decision == "FALSIFIED":
            self.terminated_reason = "threshold_falsified"
            return True, state
        if self._ais:
            # Definition-6 stopping rule: (U - L) / 2 <= epsilon.
            if state.half_width <= self.config.epsilon:
                self.terminated_reason = "epsilon_reached"
                return True, state
        elif state.eps_stat + state.W_open <= self.config.epsilon:
            self.terminated_reason = "epsilon_reached"
            return True, state
        return False, state

    def _log_iteration(
        self, iter_idx: int, action: _Action, state: EstimatorState
    ) -> None:
        self.iterations.append(
            IterationLog(
                iter_idx=iter_idx,
                action_kind=action.kind,
                leaf_depth=action.leaf.depth,
                samples_used_after=self.samples_used,
                mu_hat=state.mu_hat,
                eps_stat=state.eps_stat,
                W_open=state.W_open,
                interval=state.interval,
                n_leaves=state.n_leaves,
                n_open_leaves=state.n_open_leaves,
            )
        )

    def run(self) -> SchedulerResult:
        self._start_time = time.perf_counter()
        self.bootstrap()
        self._try_close_all()

        # Safety upper bound on iterations. We use:
        #   - the sample budget (each iteration spends at least 1 sample
        #     in the allocate branch), plus
        #   - a generous slack for refinement-only iterations, plus
        #   - a fallback of 10^7 when there is no sample budget.
        if self.config.budget_samples is not None:
            max_iters = self.config.budget_samples + 1000
        else:
            max_iters = 10_000_000

        iter_idx = 0
        for iter_idx in range(max_iters):
            done, state = self._should_terminate()
            if done:
                break
            actions = self._candidate_actions()
            best = self._pick_best_action(actions)
            if best is None:
                self.terminated_reason = "no_actions_available"
                break
            if best.kind == "allocate":
                self._allocate_one_batch(best.leaf, best.k)
            elif best.kind == "ais_allocate":
                self._allocate_ais_batch(best.leaf, best.k)
            else:
                assert best.clause is not None
                self._execute_refine(best.leaf, best.clause)
            self._try_close_all()
            self._log_iteration(iter_idx, best, state)

        final_state = self._estimate()
        return SchedulerResult(
            final_estimator=final_state,
            iterations=self.iterations,
            samples_used=self.samples_used,
            refinements_done=self.refinements_done,
            smt_calls=self.smt_calls,
            terminated_reason=self.terminated_reason,
            frontier=self.frontier,
            ais_samples_used=self.ais_samples_used,
            decision=final_state.decision,
        )


__all__ = [
    "ASIPScheduler",
    "IterationLog",
    "SchedulerConfig",
    "SchedulerResult",
]
