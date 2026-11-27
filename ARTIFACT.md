# Artifact evaluation guide

This document maps the DiSE artifact onto the standard
**Functional / Reusable / Available** criteria used by the ATVA artifact-evaluation committee.

## Summary

| Badge        | Status | Where to verify                                                  |
|--------------|--------|------------------------------------------------------------------|
| **Available**  | YES | Public GitHub repository (MIT-licensed), Docker image, archived release. |
| **Functional** | YES | `pytest` (326 tests) + the `dise` CLI sanity commands pass.   |
| **Reusable**   | YES | Public Python API (`estimate`, `failure_probability`), benchmark protocol, pluggable baselines (incl. importance sampling / cross-entropy / SMC), JSON I/O. |

## 1. Available

* **Source**: <https://github.com/aalok-thakkar/discrete-symbolic-estimation>
* **License**: MIT (see [`LICENSE`](LICENSE)).
* **Long-term archive**: tag `v0.1.0` (and any subsequent releases).
* **Container**: `docker build -t dise .` (see [`Dockerfile`](Dockerfile)).

## 2. Functional

Reproduce the artifact's core claims in a clean environment:

```bash
# 1. Install
uv sync
uv pip install -e ".[dev,plot]"

# 2. Type-check + lint
uv run mypy src/dise
uv run ruff check src/dise tests/

# 3. Full test suite
uv run pytest

# 4. CLI smoke tests
uv run dise version
uv run dise list                # 17 registered benchmarks
uv run dise compare 'integer_sqrt_correct_U(1,1023)' \
    --budget 400 --n-seeds 2 --mc-samples 1000 --no-pse
```

Each command should exit with status 0. The `pytest` run prints
`326 passed` (with the `[plot]` extra installed; otherwise one
`matplotlib` test is skipped). The `dise compare … --no-pse`
invocation prints a three-row table (`plain_mc`,
`adaptive_stratified`, `dise_betting`), each with 100% coverage;
it takes ~1–2 min because `dise_betting` runs Z3 closure on the
`isqrt` loop.

Dropping `--no-pse` adds a fourth row, `pse_exact`, the internal
PSE-style ablation. On path-heavy benchmarks like `integer_sqrt`
its exhaustive DFS is slow — keep `--no-pse` for a quick smoke, or
point `pse_exact` at an enumerable benchmark (e.g.
`assertion_overflow_mul_w=8_U(1,31)`).

The **SV-COMP corpus sweep** (§ 4) uses a three-method slate
(`plain_mc`, `pse_exact`, `dise_betting`); `pse_exact` there is an
internal PSE-style ablation, not a claim of a full PSE implementation.

For the headline result from the brief — the GCD step-count
benchmark — a single run is:

```bash
uv run dise run 'gcd_steps_le_5_BG(p=0.1,N=100)' \
    --epsilon 0.05 --budget 5000 --cache-smt --backend z3
```

Expected: `terminated_reason='epsilon_reached'` and the certified
interval contains the MC ground truth (≈ 0.989).

For the assertion-violation framing:

```bash
uv run dise run 'assertion_overflow_mul_w=8_U(1,31)' \
    --epsilon 0.05 --budget 5000 --backend z3 --cache-smt
```

Expected: `mu_hat ≈ 0.40` (the 8-bit overflow probability for
$a, b \sim \mathrm{Uniform}(1, 31)$), `terminated_reason='epsilon_reached'`,
with an exact zero-width interval (the overflow region closes under Z3).

Note: this benchmark needs the `z3` backend to *close* — the `mock`
backend is conservative (it cannot discharge the nonlinear overflow
guard), so leaves stay open, `W_open` stays near 1, and an
`--no-budget` run would not reach `epsilon`. Use `--backend z3` here.

## 3. Reusable

The artifact is structured as a Python package; reuse takes three forms.

### 3.1 Library API

```python
from dise import estimate, failure_probability, Uniform, BoundedGeometric

# Output-property framing
result = estimate(
    program=my_program,
    distribution={"x": Uniform(1, 100)},
    property_fn=lambda y: y < 50,
    epsilon=0.05, delta=0.05,
    budget=None,   # run until epsilon_reached
)

# Assertion-violation framing
result = failure_probability(
    program=my_program_with_asserts,
    distribution={"a": BoundedGeometric(p=0.1, N=100)},
    epsilon=0.05, delta=0.05,
)
```

Every exported name is documented in [`docs/architecture.md`](docs/architecture.md).

### 3.2 Benchmark protocol

New benchmarks register themselves via a decorator:

```python
# benchmarks/my_kernel.py
from dise.benchmarks._base import Benchmark, register
from dise.distributions import Uniform

@register
def my_kernel() -> Benchmark:
    return Benchmark(
        name="my_kernel_u100",
        description="...",
        program=lambda x: ...,
        distribution={"x": Uniform(1, 100)},
        property_fn=lambda y: y > 0,
        suggested_budget=5000,
    )
```

After import, the benchmark appears in `dise list` and is runnable
via `dise run my_kernel_u100`.

### 3.3 Baseline / experiment infrastructure

```python
from dise.experiment import run_experiment, default_methods

report = run_experiment(
    benchmark_name="my_kernel_u100",
    description="...",
    program=my_program,
    distribution=my_distribution,
    property_fn=my_property,
    methods=default_methods(budget=5000),
    seeds=range(5),
)
print(report.aggregates)  # one MethodAggregate per comparator
```

JSON-serialize with `dise.experiment.save_report`; plot with
`dise plot --report ... --kind compare`.

## 4. Reproducing the paper's experiments

See [`EXPERIMENTS.md`](EXPERIMENTS.md). The headline (SV-COMP corpus
sweep) is reproducible in three commands:

```bash
uv run python experiments/fetch_svcomp.py        # ~601 .c files, ~30 s (idempotent)
uv run python experiments/run_svcomp.py          # full corpus, ~1 hr; MAX_PROGRAMS=200 for a shorter run
uv run python experiments/plot_results.py --rare-event-only
```

Outputs (three-method slate `plain_mc`, `pse_exact`, `dise_betting`):

* `experiments/results/svcomp_runs.jsonl` — one row per (program, method) cell
* `experiments/results/svcomp_summary.json` — aggregates and per-comparator
  head-to-head; the exact `n_programs` for a run is recorded here (it depends
  on `PER_PROGRAM_TIMEOUT` and `MAX_PROGRAMS`)
* `experiments/figures/08_svcomp_aggregate.{pdf,png}` — the SV-COMP aggregate figure (DiSE vs plain-MC scatter + per-method ECDF)

For the internal suite of 17 benchmarks:

```bash
scripts/reproduce.sh                # canonical: 3 seeds, budget 5000
QUICK=1 scripts/reproduce.sh        # fast: 2 seeds, budget 500
```

Output: per-benchmark JSON reports + a `summary.json` in `results/`,
plus PNG plots in `fig/` (when `matplotlib` is installed).

### Non-enumerable rare-event experiment (IS / CE / SMC)

The `beta > 0` fallback and the rare-event statistical baselines are
exercised by a dedicated experiment on unbounded-support integer inputs:

```bash
uv run python experiments/run_nonenumerable.py           # 5 seeds
QUICK=1 uv run python experiments/run_nonenumerable.py   # 2 seeds, faster MC truth
```

It compares DiSE against importance sampling, the cross-entropy method,
and sequential Monte Carlo (subset simulation) at a matched budget, and
reports the fair metric — samples to a certified interval — plus coverage
against ground truth. Two instances map where each method wins:

* `linear_tail` (`a + b >= T`, `beta = 0`): DiSE closes the half-plane in
  closed form — an exact, zero-width certificate in ~220 program runs —
  while IS/CE need the full budget for a half-width near `3e-3`.
* `bitwise_tail` (`a + b >= L` and `(a ^ b) & M == 0`, `beta > 0`): the
  XOR concretizes, so DiSE cannot close the tail leaf and falls back to
  the certified importance-sampling instantiation. It stays sound
  (coverage 1.0) but pays `W_open`, so the specialized IS/CE estimators
  are tighter here. SMC's interval is asymptotic (CLT), not a finite-sample
  `(1 - delta)` certificate — flagged as such in the output.

Outputs `experiments/results/nonenumerable.{jsonl,json}`.

## 5. Hardware and software requirements

* **CPU**: any x86-64 / ARM64 with ≥ 1 GiB RAM. All experiments are
  single-threaded; wall-clock numbers in the paper are from a
  consumer laptop (M-series Mac).
* **OS**: Linux / macOS / Windows.
* **Python**: 3.10, 3.11, or 3.12 (tested in CI).
* **Memory**: ≤ 500 MiB for any single benchmark.
* **External dependencies**: `numpy`, `scipy`, `z3-solver`,
  `pycparser` (C frontend), `matplotlib` (optional, for plotting).
  All pip-installable; pinned in `uv.lock`.

## 6. Known limitations

See [`docs/limitations.md`](docs/limitations.md). Of particular note:

* The closure rule is **strict (SMT-only)**.  Programs whose path
  conditions defeat Z3 within the per-cell budget cause leaves to
  stay open, widening the certified interval to (in the worst case)
  $[0, 1]$.  On the SV-COMP corpus this affects a large fraction of
  programs (see
  [`experiments/EXPERIMENTAL_STUDY.md`](experiments/EXPERIMENTAL_STUDY.md)).
* The C frontend rejects programs with `__VERIFIER_nondet_*` calls
  inside loop bodies (each call is semantically a fresh iid draw,
  requiring a stream model out of our current scope).
* **Non-axis-aligned / non-enumerable regions ($\beta > 0$).** These are
  handled by a certified importance-sampling instantiation: the mass
  $\hat w_\pi$ of an open leaf is estimated with a bounded-weight
  importance sampler, and its statistical error enters the certified
  interval through the empirical-Bernstein / betting bound (the
  `eps_mass` term), so the $(1 - \delta)$ guarantee is preserved. The
  `experiments/run_nonenumerable.py` experiment exercises this path and
  compares it against importance sampling, cross-entropy, and sequential
  Monte Carlo. The `IntegerLatticeMHSampler` used inside general regions
  draws from a symmetric, reject-and-stay proposal (no boundary clipping),
  so its stationary distribution is the target $D \cdot \mathbf{1}[R_\pi]$;
  chain **mixing** remains a heuristic knob, but the reported interval does
  not depend on mixing quality for its soundness.
* The concolic front-end traces integer arithmetic (`+`, `-`, `*`, `//`,
  `%`) symbolically and routes bitwise / shift operators through a
  **concretization guard**: the trace is marked opaque and the affected
  leaf is refused symbolic closure (it stays open and contributes to
  `W_open`), rather than emitting an unsound `const == const` clause. This
  keeps intervals sound on bitwise-heavy code at the cost of a wider
  `W_open`; a QF_BV backend would close such leaves at higher SMT cost.


