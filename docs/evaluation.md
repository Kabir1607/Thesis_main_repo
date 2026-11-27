# Evaluation methodology

We evaluate DiSE in two complementary settings: a **headline corpus
sweep** on $271$ SV-COMP integer programs (the full picture; see
[`experiments/EXPERIMENTAL_STUDY.md`](../experiments/EXPERIMENTAL_STUDY.md)
for the detailed write-up) and an **internal suite** of $12$
hand-crafted integer benchmarks drawn from established sources
(Hacker's Delight, CLRS, CERT C, ...).  This document spells out
how those experiments are constructed; the full reproduction
commands are in [`EXPERIMENTS.md`](../EXPERIMENTS.md).

## 1. Comparators

The paper's experiment runners (`experiments/run_svcomp.py`,
`experiments/run_full_study.py`) use a compact **three-method** slate —
`plain_mc`, `pse_exact`, `dise_betting`. The CLI `dise compare` /
`dise experiment` (via `dise.experiment.default_methods`) adds a fourth
pure-sampling comparator, `adaptive_stratified`. All four share the
[`Baseline`](../src/dise/baselines/__init__.py) protocol:

| Method                  | Class                                                  | Role |
|-------------------------|--------------------------------------------------------|------|
| `plain_mc`              | `dise.baselines.PlainMonteCarlo`                       | Vanilla MC with a Wilson-score certified interval; the statistical floor. |
| `adaptive_stratified`   | `dise.baselines.AdaptiveStratifiedMC`                  | Two-pass Neyman allocation across $16$ random hash buckets (Carpentier–Munos 2011 spirit); the strongest pure-sampling stratifier (no symbolic reasoning). |
| `pse_exact`             | `dise.baselines.pse.PSEBaseline`                       | Probabilistic symbolic execution (Geldenhuys–Dwyer–Visser 2012 + Filieri et al.\ 2015 lineage); DFS over the frontier with exact mass on closed leaves. |
| `dise_betting`          | `dise.baselines.DiSEBaseline` with `method="betting"`  | DiSE / ASIP with WSR PrPl-EB intervals; our adaptive symbolic-importance partitioner. |

These rows occupy distinct corners of the design space: vanilla
sampling (`plain_mc`), adaptive sampling (`adaptive_stratified`, CLI
only), purely symbolic (`pse_exact`), and adaptive symbolic
(`dise_betting`).  The paper runs compare the three that isolate the
structural contrast — statistical, symbolic, and hybrid — with
`adaptive_stratified` available from the CLI as the strongest
pure-sampling point of reference.  For the non-enumerable ($\beta > 0$)
regime, `dise.baselines.rare_event` adds importance sampling, the
cross-entropy method, and sequential Monte Carlo (see
`experiments/run_nonenumerable.py`).  The tier-2 baselines that lived in
`dise.baselines.tier2` (`PlainMonteCarloHoeffding`,
`PlainMonteCarloEmpiricalBernstein`, `PlainMonteCarloBetting`,
`QuasiMonteCarloSobol`, `StratifiedRandomMC`) remain importable for
ablation studies but are no longer part of the headline experiment
set — they isolate \emph{parametric} interval-construction
differences rather than the \emph{structural} contrasts the four
methods above expose.

## 2. Benchmarks

The benchmark suite is partitioned into **three non-overlapping
categories**, each defined by an external curation criterion that
exists independently of DiSE.  This addresses the standard
reviewer concern that benchmark selection is the single largest
lever authors have over a paper's measured numbers.

### Category A — Introductory-CS / textbook integer kernels

Twelve hand-coded integer-arithmetic kernels of the kind students
write in CS-101 and standard-textbook exercises.  They isolate
DiSE's algorithmic behaviour from any C-frontend artefacts and
provide the cleanest demonstration of the $O(1)$-sample exact-
certificate regime.

| Name                                          | Source        | Description                                                  |
|-----------------------------------------------|---------------|--------------------------------------------------------------|
| **`coin_machine_U(1,9999)`**                  | pedagogical   | **Three-region branching toy with a rare bug; ASIP's intro example.** |
| `gcd_steps_le_5_BG(p=0.1,N=100)`              | Knuth TAoCP §4.5.2 | Euclidean GCD step count $\le k$.                       |
| `modpow_fits_in_4b_m=37`                      | CLRS Ch. 31   | Modular exponentiation fits in $w$ bits.                     |
| `miller_rabin_w=2_BG(p=0.05,N=200)`           | Rabin 1980    | Miller-Rabin (witness 2) accepts $n$.                        |
| `popcount_w6`, `parity_w6`, `log2_w6`         | Hacker's Delight 2e | Bit-vector tautologies; property always true ($\mu = 1$). |
| `collatz_le_30_BG(p=0.05,N=200)`              | Lagarias 2010 | Collatz trajectory $\le k$ (irregular control flow).         |
| `sieve_primality_U(2,200)`                    | classical     | Trial-division primality.                                    |
| `integer_sqrt_correct_U(1,1023)`              | CLRS          | $\texttt{isqrt}(n)^2 \le n < (\texttt{isqrt}(n)+1)^2$.       |
| `sparse_trie_depth_le_3_U(0,63)`              | TAoCP §6.3    | Max-depth of a 4-ary trie after two insertions.              |
| **`assertion_overflow_mul_w=8_U(1,31)`**      | CERT C INT32-C | **Integer overflow on `a*b`; failure-probability target.** |

### Category B — SV-COMP (verification community's standard corpus)

$601$ C programs drawn unchanged from the
[Software Verification Competition](https://sv-comp.sosy-lab.org/)
corpus, the field's de-facto reference suite.  We did not author
these programs or filter them by expected algorithmic behaviour;
we ran our C frontend on the entire ReachSafety-Loops category and
report aggregates across every program that transpiled cleanly
into our integer fragment.  See
[`experiments/EXPERIMENTAL_STUDY.md`](../experiments/EXPERIMENTAL_STUDY.md)
for the headline result: on the programs that fit the three-method
per-program budget, `pse_exact` delivers exact (zero-width)
certificates on a substantial fraction of the enumerable-LIA
programs, and `dise_betting` holds its nominal $1-\delta$ coverage.
The exact per-run figures — `n_programs`, `fraction_hw_zero`,
`coverage` per method — are recorded in
[`experiments/results/svcomp_summary.json`](../experiments/results/svcomp_summary.json)
and depend on `PER_PROGRAM_TIMEOUT` / `MAX_PROGRAMS`.

### Category C — Real-world systems-code kernels

Four integer-arithmetic functions extracted verbatim from
production codebases used in deployed software, chosen as
canonical examples of safety-critical embedded software
(FreeRTOS, lwIP) and ubiquitous foundational libraries (zlib,
SQLite, in every iPhone and major browser).  Each is the
smallest semantically-meaningful unit on which the listed
operational question is asked by the codebase's maintainers.

| Name                                  | Source kernel                          | Operational question                  |
|---------------------------------------|----------------------------------------|---------------------------------------|
| `freertos_queue_wrap_BG(p=0.1,N=256)` | FreeRTOS V11.0.1 `queue.c`             | wrap-branch hit rate in queue index   |
| `lwip_chksum_zero_U(0,131071)`        | lwIP V2.2.0 `inet_chksum.c`            | fraction folding to zero (RFC-1071 special) |
| `zlib_crc32_low_bit_U(0,255)`         | zlib V1.3.1 `crc32.c`                  | low bit of CRC-32 of a byte           |
| `sqlite_varint_fastpath_BG(p=0.05,N=255)` | SQLite 3.45 `src/util.c` getVarint32 | 1-byte fast-path coverage             |

Honest scope: we do not pretend these four kernels constitute a
*whole-codebase* analysis of FreeRTOS, lwIP, zlib, or SQLite.
Most of those codebases sits outside our integer-only fragment
(pointers, concurrency, floating point, dynamic memory).
Category C demonstrates DiSE runs on the integer-arithmetic core
of real production software; scaling to whole codebases is
honest future work, not a hidden gap.

`dise list` is the always-up-to-date roster across all three
categories.

## 3. Metrics

For every $(\text{method}, \text{benchmark}, \text{seed})$ triple,
the experiment runner records:

* **`mu_hat`** — point estimate.
* **`interval = [lo, hi]`** — certified $(1 - \delta)$-coverage interval.
* **`half_width = (hi - lo) / 2`** — *the* headline metric.
* **`samples_used`** — actual concolic / MC calls.
* **`wall_clock_s`** — perf_counter measurement.
* **`interval_contains_truth`** — whether the MC ground truth lies in
  the certified interval. Aggregated across seeds, this is the
  empirical *coverage*; for sound methods at confidence $1 - \delta$
  we expect $\text{coverage} \ge 1 - \delta$.

Aggregated per `(method, benchmark)`:

* **`median_mu_hat`**, **`median_half_width`**, **`median_samples`**,
  **`median_wall_clock_s`** — robust to outliers.
* **`iqr_half_width`**, **`iqr_samples`** — variability across seeds.
* **`coverage`** — empirical, see above.

## 4. MC ground truth

For each benchmark, the experiment runner computes a high-budget plain
Monte-Carlo estimate (`mc_samples = 20 000` by default) using a fixed
independent seed (`12 345`), so the reference is reproducible and the
same across methods.

This MC estimate has its own standard error
$\mathrm{se}_{\mathrm{MC}} = \sqrt{\hat\mu_{\mathrm{MC}} (1 - \hat\mu_{\mathrm{MC}}) / n_{\mathrm{MC}}}$.
We do *not* report this as a certified interval — it is a regression
check.

## 5. Comparison style

Two question forms guide our reporting:

* **At fixed budget, who gets the tightest certified interval?**
  Compares `median_half_width` across methods at matched `budget`.
* **At what budget does each method reach a target $\varepsilon$?**
  Compares `median_samples` across methods constrained by
  `terminated_reason == "epsilon_reached"`.

DiSE's structural-variance-reduction wins are most visible in benchmarks
where the operational distribution concentrates on a path-deterministic
region with weight $\approx 1$ (e.g. `gcd_steps_le_10_*`,
`integer_sqrt_correct_*`). On flat distributions over many program
paths (e.g. `popcount_w6` under uniform), the benefit narrows because
each leaf demands its own constrained-sample budget.

For the assertion-violation benchmark, the comparison is over the
**failure probability** estimate: DiSE certifies a tight interval on
$\mu_{\text{fail}}$ where plain MC needs $\Theta(1/\varepsilon^2)$
samples for the same half-width.

## 6. Soundness verification

For every method × seed, we record `interval_contains_truth`. Over $n$
seeds × $b$ benchmarks, the empirical coverage rate should be
$\ge 1 - \delta$ asymptotically. A single seed × benchmark giving the
wrong answer is *expected* (at $\delta = 0.05$ we anticipate ~5 % miss
rate); the aggregate column shows the rate.

When DiSE's coverage rate falls below the target, the most likely
explanations (in decreasing order of probability) are:

1. The MC reference itself is noisy (only $1000$ samples by
   default in the SVCOMP runner) — coverage can dip below
   $1-\delta$ on programs with $\mutrue < 5 / \text{n\_ref}$.
2. The SMT backend is `MockBackend`, which returns `"unknown"` on
   most non-axis-aligned arithmetic; under the strict (SMT-only)
   closure rule this just keeps the leaf open (sound) but it can
   hide a real bug if it's misconfigured.
3. `closure_min_samples` is too small for the rare-minority-path
   regime of a particular benchmark; raise it.
4. The benchmark's program raises before tracing terminates (rare).

## 7. Reproducibility

Two routes:

1. **Single command** — see [`scripts/reproduce.sh`](../scripts/reproduce.sh).
2. **Manual** — see [`EXPERIMENTS.md`](../EXPERIMENTS.md). The
   experiment runner writes per-benchmark JSON reports plus a
   `summary.json`; tables in the paper come from these files via
   `dise plot`.

All seeds are derived from a single `--seed` flag; the experiment
runner enumerates `range(n_seeds)`. Wall-clock measurements are
machine-dependent and reported here for indication only.

### 7.1 Soundness-mode runs

For tight $\varepsilon$ targets, pass `--no-budget` (the algorithm
runs until `epsilon_reached` with no sample cap). Combined with
`--budget-seconds 600` (10 minute wall-clock cap) you get a
"converge as far as you can in 10 minutes" semantics:

```bash
dise run gcd_steps_le_5_BG\(p=0.1,N=100\) \
    --no-budget --budget-seconds 600 --epsilon 0.001
```
