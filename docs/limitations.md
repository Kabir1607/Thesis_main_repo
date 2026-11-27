# Limitations and open extensions

DiSE is a research prototype. The implementation covers class **(D1)** —
product-form discrete distributions over deterministic integer /
bitvector programs — and leaves several axes for future work. This
document is exhaustive: every regret is listed here.

## 1. Data types and programs

* **Strings, floats, byte buffers.** Integer / bitvector inputs only.
  Floats would require an SMT theory for floating-point (Z3 has
  `FloatingPoint`, but the concolic wrapper does not).
* **Bitwise operators outside the LIA fragment.** Our SMT backend
  speaks linear integer arithmetic (LIA).  Bitwise `|`, `^`, `<<`,
  `>>`, and `&` with a non-power-of-two-minus-1 mask on a symbolic
  operand cannot be modelled in LIA, so the concolic interpreter
  falls back to *concretization*: the symbolic expression becomes a
  constant per-sample.  A trace that crosses any such fallback is
  tagged `concretized` (see `Tracer.concretized` in
  `dise.concolic`), and `Frontier.try_close` refuses to issue an
  SMT closure certificate against tagged observations — the leaf
  stays OPEN and its mass moves into the `W_open` cushion.  This
  preserves soundness at the cost of wider intervals on
  bitwise-heavy code (e.g. CRC / SHA / AES kernels, transpiled
  SV-COMP programs with unsigned masking).  A bitvector-aware
  backend (Z3's `QF_BV` logic) is the natural extension.
* **Symbolic memory.** Heap or array accesses indexed by symbolic
  values are not modeled. Programs that use lists / dicts in a way
  the concolic wrapper doesn't see (no comparisons) still run, but
  the path condition won't capture them.
* **Non-deterministic programs.** DiSE assumes $P$ is a deterministic
  function. If `program(...)` uses external randomness the closure
  rule's path-determinism is violated and the certified-interval
  guarantees break. Wrap or seed any internal RNG.
* **Concurrent programs.** Out of scope.

## 2. Distribution classes

* **Class (D2) Bayes-net-structured discrete distributions** — not
  supported. Would require ancestor-conditioned constrained samplers
  and mass estimators that handle dependence.
* **Class (D3) general discrete** — out of scope.
* **Continuous distributions** of any kind — out of scope.

## 3. Algorithmic gaps

* **Rare-event regions.** When acceptance under the envelope drops
  below ~1 %, `RejectionSampler` returns partial batches. The
  `IntegerLatticeMHSampler` (Metropolis-Hastings on integers) is
  shipped as a working alternative, but its mixing is heuristic and
  not theoretically tied to a $\delta$-certified bound on a burn-in /
  thin schedule. Future work: replace with a perfect-sampling scheme
  or a verified-mixing one.
* **`MockBackend` path-determinism shortcut.** The closure rule's
  symbolic shortcut returns `"unknown"` for non-axis-aligned
  arithmetic under `MockBackend`.  Since the **strict closure rule**
  (only certifies a leaf when SMT proves path-determinism), a
  `"unknown"` result simply means the leaf stays open and contributes
  to `W_open` — the certified interval widens but remains sound.  No
  sample-based fallback; no `strict_unknown` knob (it became the
  default and was retired).  Under `Z3Backend`, most LIA closure
  queries resolve to `"unsat"` and the leaf closes cleanly.
* **Mass conservation on `GeneralRegion`s** — resolved. The
  proportional-split refinement (see [`algorithm.md`](algorithm.md) §5)
  guarantees children's masses sum to the parent's mass exactly,
  modulo Wilson-smoothed IS noise from the split-proportion estimator.
  Root mass is exactly $1$ by construction. Earlier prototype builds
  had a systematic mass drift of ~30 % on `popcount`; that's fixed.
* **Refinement clause selection.** The current heuristic
  (`_best_refinement_clause`) scores a *finite* set of candidate
  clauses harvested from observed paths. It does not invent new
  clauses (e.g. via Craig interpolation), so it cannot refine on
  predicates the program never branches on.
* **Concolic divergence.** A run exceeding `max_concolic_branches` is
  dropped (not aggregated). Programs with unbounded loops thus
  quietly waste sample budget. The estimator state's
  `terminated_reason` and the scheduler's iteration log are the
  diagnostic.

## 4. Termination

Termination is governed by four orthogonal stop conditions
(see [`algorithm.md`](algorithm.md) §9.2):

1. **`epsilon_reached`** — the primary, **always-active** condition.
2. **`budget_exhausted`** — optional sample cap. Disabled by
   `budget=None` (the recommended setting for soundness-mode runs).
3. **`time_exhausted`** — optional wall-clock cap (`budget_seconds`).
   Disabled by default.
4. **`no_actions_available`** — every candidate's gain/cost falls
   below `min_gain_per_cost` (default $0$; legacy behavior).

If *all three* of `budget_samples`, `budget_seconds`, and
`min_gain_per_cost > 0` are disabled and the target $\varepsilon$ is
unreachable, the algorithm runs forever. Configure at least one
practical cap unless you are willing to trust the target.

## 5. Tooling gaps

* **CLI:** `dise run` accepts only registered benchmarks; ad-hoc
  programs go through the Python `estimate()` / `failure_probability()`
  API.
* **Tighter Bernstein bounds.** We implement classical Bernstein and
  Maurer-Pontil empirical-Bernstein. The Audibert-Munos-Szepesvári
  bound and the Howard-Ramdas-McAuliffe time-uniform bound would be
  natural additions.
* **Parallel execution.** All concolic runs are serial. Threading the
  outer loop is straightforward (each run is independent) but not yet
  implemented.

## 6. Verification gaps

* DiSE produces **certified intervals** under the soundness of its SMT
  oracle and the closure rule. We do not produce *machine-checked
  proofs* of those certificates (no Lean / Coq export).
* Theorem 2 assumes mass and sample-mean estimators are
  *independent* per leaf. The proportional-split refinement
  introduces *implicit* coupling (the split proportion is a function
  of the IS batch which uses the same RNG stream). In practice this
  coupling is benign — the split is fixed *before* per-child sampling
  begins — but a formal independence argument would benefit from a
  stronger RNG-partition scheme.

## 7. Open extensions, ranked by impact

1. **Wall-clock-bound regime on non-LIA programs.**  The SV-COMP
   sweep (271 programs) shows DiSE times out on ~50 % of the corpus
   when path conditions defeat Z3 within the per-cell budget.  Two
   engineering paths: parallel SMT (per-leaf closure checks are
   independent), or delegating non-LIA region mass to a certified
   approximate model counter (ApproxMC6 / the formally certified
   variant of Kiesl-Reiter et al., CAV 2024).
2. **Nondets inside loop bodies.**  The current SV-COMP frontend
   explicitly rejects programs whose `__VERIFIER_nondet_*` calls
   appear inside loop bodies (each call is semantically a fresh iid
   draw).  Lifting this would require an iid-stream model and a
   different mass-conservation argument.
3. **Bayes-net (D2) distributions.** A whole class of structured
   joints without sacrificing closed-form mass.
4. **Craig-interpolation refinement.** Augment the candidate-clause
   set with predicates the SMT solver invents.
5. **Counter-example-guided MCMC.** Use SMT to find feasible starts
   for the integer-lattice MH chain; tightens the acceptance rate on
   very rare events.
6. **Machine-checked certificates.** Export $(F_\pi, n_\pi, h_\pi,
   \mathrm{path}_\pi, \hat\mu, \varepsilon_{\text{total}})$ tuples in
   a format consumable by a proof assistant.
7. **C frontend extensions.** Pointer-free arrays-of-arrays, common
   helper-function inlining for the 158 SV-COMP programs the
   transpiler currently rejects, opt-in handling of nondets in loops
   (see item 2 above).
8. **Continuous extensions.** A discretized continuous distribution
   with a discrete envelope; an interesting bridge to probabilistic
   verification on numerical programs.

*Resolved (see § 3):*

- Strict handling of `"unknown"` — closure now requires SMT-`unsat`
  unconditionally (no sample-based fallback).  The `strict_unknown`
  flag has been retired (it was the default).
- Closed-leaf mass uncertainty — the certified interval now includes
  a $z \cdot \sqrt{w_{\text{var}}}$ term for every `CLOSED_TRUE` leaf
  with non-zero mass variance, fixing a soundness gap discovered on
  the `assertion_overflow` benchmark.  Restricting to leaves where
  $\mu_\pi = 1$ avoids spurious widening on CLOSED_FALSE leaves.
- C frontend — `dise.frontends.svcomp_c` translates a tractable
  subset of C (integers/Bools, control flow, simple arrays, helper
  functions, the SV-COMP ``__VERIFIER_*`` idioms).  Roughly
  three-quarters of the 601-program SV-COMP sample transpile
  cleanly; the exact per-run count is reported by
  `experiments/run_svcomp.py`.
