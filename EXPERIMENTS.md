# Reproducing the paper's experiments

> All commands below assume you have followed [`INSTALL.md`](INSTALL.md)
> and `dise list` succeeds. Wall-clock estimates are for a laptop;
> Z3-backed runs are roughly 2× to 4× slower than `MockBackend`.
>
> The **headline experiment** is the SV-COMP corpus sweep
> (§ "SV-COMP corpus sweep" below); see also the dedicated
> [`experiments/EXPERIMENTAL_STUDY.md`](experiments/EXPERIMENTAL_STUDY.md)
> for the full study.

## SV-COMP corpus sweep (headline)

```bash
# 1. Fetch ~600 SV-COMP programs (idempotent; skips already-fetched).
uv run python experiments/fetch_svcomp.py

# 2. Run the sweep: subprocess-isolated, 15 s wall-clock per program.
uv run python experiments/run_svcomp.py

# 3. Regenerate the aggregate plot (scatter + ECDF).
uv run python experiments/plot_results.py --rare-event-only
```

Outputs:
* `experiments/results/svcomp_runs.jsonl` — one row per (program, method) cell
* `experiments/results/svcomp_summary.json` — per-method aggregates and head-to-head
* `experiments/figures/08_svcomp_aggregate.{pdf,png}` — the headline figure

The sweep uses a three-method slate (`plain_mc`, `pse_exact`,
`dise_betting`); `pse_exact` is an internal PSE-style ablation, not a
full PSE implementation. Of the 601 programs in the corpus, the
transpiler accepts a subset, and those that fit the per-program budget
(`PER_PROGRAM_TIMEOUT=20s`) are evaluated end-to-end; the exact counts
for a given run are recorded in `svcomp_summary.json` under
`per_method[*].n_programs`. Wall-clock is roughly an hour on a laptop
(many nested-loop programs hit the per-program timeout and are skipped).
Pass `PER_PROGRAM_TIMEOUT=40` for a higher-coverage sweep (slower), or
`MAX_PROGRAMS=50` for a quick smoke.

Override the per-program input distribution by placing a sidecar at
`experiments/svcomp_samples/<name>.dist.json` with e.g.
`{"varname": {"type": "BoundedGeometric", "p": 0.01, "N": 10000}}`.

## Decision-curve experiment (operational anchor)

```bash
uv run python experiments/run_decision_curve.py
```

Sweeps the SLA threshold $\tau$ around the truth $\mu \approx 0.0067$
on a load-classifier benchmark under `BoundedGeometric(0.001, 10000)`,
measuring samples-to-$(1-\delta)$-decision per method.

Outputs `experiments/results/decision.{jsonl,json}` and the
`07_decision_curve.{pdf,png}` figure.

## Convergence curve and rare-event scaling

```bash
uv run python experiments/run_convergence_curve.py
uv run python experiments/run_rare_event_scaling.py
```

Two complementary illustrations: half-width vs sample budget on a
single-conditional threshold (DiSE cliff to exact zero) and
samples-to-$\varepsilon$ vs $\mu$ on a parametric rare-event program.

## Non-enumerable rare-event experiment ($\beta > 0$; IS / CE / SMC)

```bash
uv run python experiments/run_nonenumerable.py           # 5 seeds
QUICK=1 uv run python experiments/run_nonenumerable.py   # 2 seeds
```

Compares DiSE against importance sampling, the cross-entropy method, and
sequential Monte Carlo (subset simulation) on unbounded-support integer
inputs, at a matched budget, reporting samples-to-a-certified-interval and
coverage. Two instances: `linear_tail` (`a+b >= T`), which DiSE closes in
closed form ($\beta = 0$, exact zero-width certificate); and `bitwise_tail`
(`a+b >= L` and `(a^b) & M == 0`), where the XOR concretizes and DiSE falls
back to its certified importance-sampling instantiation ($\beta > 0$). IS
and CE carry finite-sample empirical-Bernstein certificates; SMC's interval
is asymptotic (CLT) and flagged as such. Outputs
`experiments/results/nonenumerable.{jsonl,json}`.

## Internal benchmark suite (legacy)

## Table 1 — Comparison vs. baselines (single benchmark)

```bash
dise compare gcd_steps_le_5_BG\(p=0.1,N=100\) \
    --budget 5000 --n-seeds 5 --mc-samples 20000 \
    --json-out results/gcd_le_5.json
```

Replace `gcd_steps_le_5_BG(p=0.1,N=100)` with any name from
`dise list`. The resulting table prints `mu_hat ± median half-width`
per `(method, benchmark)`; the JSON includes per-seed rows.

## Table 2 — Full suite

```bash
dise experiment --budget 5000 --n-seeds 3 --mc-samples 20000 \
    --out-dir results/
```

This iterates over every registered benchmark and writes
`results/<benchmark>.json` plus `results/summary.json`. Console output
is a Markdown-friendly table.

To restrict to a subset:

```bash
dise experiment --benchmarks \
    gcd_steps_le_5_BG\(p=0.1,N=100\) \
    integer_sqrt_correct_U\(1,1023\) \
    miller_rabin_w=2_BG\(p=0.05,N=200\) \
    --budget 5000 --n-seeds 3
```

## Figures 1–N — `dise plot`

```bash
dise plot --report results/gcd_le_5.json --out fig/gcd_le_5.png --kind compare
dise plot --report results/gcd_le_5.json --out fig/convergence.png --kind convergence
```

`--kind compare` produces a 3-panel bar chart (half-width, samples,
coverage) per method. `--kind convergence` plots per-seed
``(samples, half_width)`` curves.

## End-to-end reproduction

The shell wrapper bundles everything:

```bash
scripts/reproduce.sh
```

It runs the experiment suite with the canonical defaults (budget 5000,
3 seeds, full benchmark list), then emits `results/*.png` if
`matplotlib` is installed. Total wall-clock on a laptop is ≈ 30–60
minutes (Z3 backend) or ≈ 5–10 minutes (`--backend mock`, less
accurate but fast).

For the three-method full study used in the paper's
`tab:suite` (`plain_mc`, `pse_exact`, `dise_betting`), prefer:

```bash
QUICK=1 uv run python experiments/run_full_study.py  # ≈ 6 min, 17 benchmarks × 3 methods × 2 seeds
uv run python experiments/run_full_study.py          # ≈ 30 min, 3 seeds × 2 budgets
```

This produces `experiments/results/runs.jsonl` and
`experiments/results/summary.json` with one row per
`(benchmark, method, budget, seed)` cell.

## Hyperparameter ablations

To vary one knob at a time:

```bash
for budget in 1000 2000 5000 10000; do
    dise compare gcd_steps_le_5_BG\(p=0.1,N=100\) \
        --budget "$budget" --n-seeds 5 \
        --json-out "results/budget_${budget}.json"
done
```

The JSON reports include `samples_used` per seed, enabling
*budget-vs-half-width* curves to be drawn from the same data.

## Reproducing the headline acceptance commands from the brief

```bash
# Brief acceptance check 1: converges with terminated_reason='epsilon_reached'
dise run gcd_steps_le_5_BG\(p=0.1,N=100\) --epsilon 0.05 --budget 2000 \
    --backend z3 --cache-smt

# Brief acceptance check 2: half-width < 0.05 at budget 5000
dise run gcd_steps_le_5_BG\(p=0.1,N=100\) --epsilon 0.05 --budget 5000 \
    --backend z3 --cache-smt
```

Both produce an `EstimationResult` line on stdout that includes
`half_width`, `terminated_reason`, and `samples_used`. Pass
``--json-out result.json`` to keep the structured record.

## Seeding policy

* `--seed S` controls DiSE's RNG.
* The MC ground truth uses a fixed independent seed (12 345) so it is
  reproducible.
* In `dise compare` and `dise experiment`, seeds enumerate
  `range(n_seeds)`. If you need a different range, the Python API
  (`dise.experiment.run_experiment`) accepts any iterable.

## Wall-clock and SMT-cache

Use `--cache-smt` (single-run CLI) to enable the `CachedBackend`
wrapper. Cache statistics are available programmatically:

```python
from dise.smt import CachedBackend, Z3Backend
backend = CachedBackend(Z3Backend())
# ... run DiSE with `backend=...` ...
print(backend.stats.hit_rate)
```

On the GCD running example with budget 5000, the cache hit rate is
typically 40–60 %.
