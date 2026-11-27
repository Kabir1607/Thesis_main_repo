#!/usr/bin/env python3
"""Non-enumerable rare-event experiment: DiSE vs importance sampling, the
cross-entropy method, and sequential Monte Carlo.

An evaluation on a non-enumerable domain (unbounded-support integer inputs)
that (i) exercises DiSE's ``beta > 0`` importance-sampling fallback,
(ii) supplies the rare-event statistical baselines (IS, CE, SMC), and
(iii) reports samples-to-a-certified-interval rather than raw estimator
variance.

Two benchmark instances, chosen to map where each method wins:

* ``linear_tail``  : event {a + b >= T} over a, b ~ Geometric(p).
  The satisfying region is a half-plane that DiSE closes in *closed form*
  (beta = 0): exact mass, zero-width certified interval, ~200 program runs.
  IS/CE/SMC must sample. This is the symbolic-closure win.

* ``bitwise_tail`` : event {a + b >= L} AND {(a XOR b) & M == 0}.
  The XOR concretizes in the concolic trace, so DiSE cannot close the tail
  leaf symbolically and falls back to the certified importance-sampling
  instantiation on a non-axis-aligned region (beta > 0). Here the specialized
  rare-event methods are competitive; DiSE stays sound but pays W_open.

For each (benchmark, method, seed) we record mu_hat, the certified interval,
its half-width, samples used, and whether the interval covers the ground
truth. ``linear_tail`` uses an exact analytic truth; ``bitwise_tail`` uses a
high-budget Monte Carlo truth with a reported standard error.

Usage
-----
    uv run python experiments/run_nonenumerable.py           # full: 5 seeds
    QUICK=1 uv run python experiments/run_nonenumerable.py   # smoke: 2 seeds

Outputs ``experiments/results/nonenumerable.{jsonl,json}`` and prints a table.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dise import estimate  # noqa: E402
from dise.baselines.rare_event import (  # noqa: E402
    CrossEntropy,
    ImportanceSampling,
    SequentialMonteCarlo,
)
from dise.distributions import Geometric  # noqa: E402

QUICK = os.environ.get("QUICK", "0") not in ("0", "", "false", "False")

DELTA = 0.05
EPSILON = 0.01
BUDGET = int(os.environ.get("BUDGET", "8000"))
N_SEEDS = 2 if QUICK else 5
MC_TRUTH_N = 2_000_000 if QUICK else 8_000_000
MC_TRUTH_SEED = 12345

RESULTS_DIR = ROOT / "experiments" / "results"


# --------------------------------------------------------------------------
# Benchmark definitions
# --------------------------------------------------------------------------


def _geom_tail_analytic(p: float, T: int, kmax: int = 200_000) -> float:
    """Exact P(a + b >= T) for a, b iid Geometric(p) on {1, 2, ...}.

    a + b has pmf P(a+b=k) = (k-1) p^2 (1-p)^(k-2) for k >= 2 (negative
    binomial). Summed to ``kmax``, whose truncation error is below machine
    noise for the (p, T) used here.
    """
    q = 1.0 - p
    # Tail = sum_{k=T}^inf (k-1) p^2 q^(k-2).
    total = 0.0
    for k in range(max(2, T), kmax + 1):
        term = (k - 1) * p * p * q ** (k - 2)
        total += term
        if term < 1e-18 and k > T + 10:
            break
    return total


BENCHMARKS = {
    "linear_tail": {
        "p": 0.3,
        "T": 22,
        "distribution": lambda p: {"a": Geometric(p), "b": Geometric(p)},
        "program": lambda a, b: 1 if (a + b) >= 22 else 0,
        "property_fn": lambda out: out >= 1,
        "score": lambda x: float(x["a"] + x["b"]),
        "threshold": 22.0,
        "event": lambda x: (x["a"] + x["b"]) >= 22,
        "beta_positive": False,
        "truth_kind": "analytic",
    },
    "bitwise_tail": {
        "p": 0.3,
        "L": 12,
        "M": 7,
        "distribution": lambda p: {"a": Geometric(p), "b": Geometric(p)},
        "program": lambda a, b: 1 if ((a + b) >= 12 and ((a ^ b) & 7) == 0) else 0,
        "property_fn": lambda out: out >= 1,
        "score": lambda x: float(x["a"] + x["b"]),
        "threshold": 12.0,
        "event": lambda x: (x["a"] + x["b"]) >= 12 and ((x["a"] ^ x["b"]) & 7) == 0,
        "beta_positive": True,
        "truth_kind": "mc",
    },
}


def ground_truth(bench_name: str, spec: dict) -> tuple[float, float]:
    """Return ``(mu_truth, standard_error)``. Analytic truth has se = 0."""
    if spec["truth_kind"] == "analytic":
        mu = _geom_tail_analytic(spec["p"], spec["T"])
        return mu, 0.0
    rng = np.random.default_rng(MC_TRUTH_SEED)
    p = spec["p"]
    a = rng.geometric(p, MC_TRUTH_N)
    b = rng.geometric(p, MC_TRUTH_N)
    # Vectorized event for the known MC-truth benchmark.
    if bench_name == "bitwise_tail":
        L, M = spec["L"], spec["M"]
        mask = (a + b >= L) & (((a ^ b) & M) == 0)
    else:  # pragma: no cover - defensive
        ev = spec["event"]
        mask = np.array(
            [ev({"a": int(a[i]), "b": int(b[i])}) for i in range(MC_TRUTH_N)]
        )
    mu = float(mask.mean())
    se = float(np.sqrt(mu * (1.0 - mu) / MC_TRUTH_N))
    return mu, se


# --------------------------------------------------------------------------
# Method runners -> uniform dict of results
# --------------------------------------------------------------------------


def run_dise(spec: dict, seed: int) -> dict:
    dist = spec["distribution"](spec["p"])
    t0 = time.perf_counter()
    res = estimate(
        program=spec["program"],
        distribution=dist,
        property_fn=spec["property_fn"],
        epsilon=EPSILON,
        delta=DELTA,
        budget=BUDGET,
        seed=seed,
    )
    wall = time.perf_counter() - t0
    lo, hi = res.interval
    return {
        "method": "dise",
        "mu_hat": res.mu_hat,
        "lo": lo,
        "hi": hi,
        "half_width": (hi - lo) / 2.0,
        "samples_used": res.samples_used,
        "wall_clock_s": wall,
        "certified": True,
        "extras": {
            "n_leaves": res.n_leaves,
            "n_open_leaves": res.n_open_leaves,
            "W_open": res.W_open,
            "eps_stat": res.eps_stat,
            "terminated_reason": res.terminated_reason,
        },
    }


def run_rare_event(est, spec: dict, seed: int) -> dict:
    dist = spec["distribution"](spec["p"])
    r = est.run(
        distribution=dist,
        score_fn=spec["score"],
        threshold=spec["threshold"],
        budget=BUDGET,
        delta=DELTA,
        seed=seed,
        event_fn=spec["event"],
    )
    lo, hi = r.interval
    return {
        "method": r.name,
        "mu_hat": r.mu_hat,
        "lo": lo,
        "hi": hi,
        "half_width": r.half_width,
        "samples_used": r.samples_used,
        "wall_clock_s": r.wall_clock_s,
        "certified": r.certified,
        "extras": r.extras,
    }


METHODS = [
    ("dise", None),
    ("importance_sampling", ImportanceSampling(p_tilt=0.05, alpha=0.5)),
    ("cross_entropy", CrossEntropy(alpha=0.5, n_round=300 if not QUICK else 150)),
    ("sequential_mc", SequentialMonteCarlo(n_particles=1000 if not QUICK else 400)),
]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    truths: dict[str, tuple[float, float]] = {}

    print(f"# non-enumerable experiment (QUICK={QUICK})")
    print(f"# budget={BUDGET}, delta={DELTA}, seeds={N_SEEDS}, mc_truth_n={MC_TRUTH_N}")

    for bench_name, spec in BENCHMARKS.items():
        mu_truth, se = ground_truth(bench_name, spec)
        truths[bench_name] = (mu_truth, se)
        # Coverage slack: 3 standard errors of the MC truth, plus a small
        # relative/absolute floor so an *exact* zero-width certificate (DiSE
        # closing a beta=0 leaf) counts as covering when it matches the
        # analytic truth to numerical precision rather than bit-for-bit.
        tol = 3.0 * se + 1e-6 * abs(mu_truth) + 1e-9
        print(
            f"\n## {bench_name}  (beta>0={spec['beta_positive']}, "
            f"truth={mu_truth:.4e}"
            + (f" +/- {se:.1e}" if se > 0 else " exact")
            + ")"
        )
        for method_name, est in METHODS:
            for seed in range(N_SEEDS):
                if method_name == "dise":
                    row = run_dise(spec, seed)
                else:
                    row = run_rare_event(est, spec, seed)
                covers = row["lo"] - tol <= mu_truth <= row["hi"] + tol
                row.update(
                    benchmark=bench_name,
                    seed=seed,
                    mu_truth=mu_truth,
                    truth_se=se,
                    covers_truth=bool(covers),
                    beta_positive=spec["beta_positive"],
                )
                rows.append(row)
            # Per-method aggregate line.
            sub = [
                r
                for r in rows
                if r["benchmark"] == bench_name and r["method"] == method_name
            ]
            med_hw = statistics.median(r["half_width"] for r in sub)
            med_n = statistics.median(r["samples_used"] for r in sub)
            cov = sum(r["covers_truth"] for r in sub) / len(sub)
            cert = sub[0]["certified"]
            tag = "cert " if cert else "asymp"
            print(
                f"  {method_name:20s} [{tag}] median_hw={med_hw:.3e} "
                f"median_samples={med_n:8.0f} coverage={cov:.2f}"
            )

    summary = {
        "metadata": {
            "budget": BUDGET,
            "delta": DELTA,
            "epsilon": EPSILON,
            "n_seeds": N_SEEDS,
            "mc_truth_n": MC_TRUTH_N,
            "quick": QUICK,
            "methods": [m for m, _ in METHODS],
        },
        "truths": {k: {"mu": v[0], "se": v[1]} for k, v in truths.items()},
        "per_cell": {},
    }
    for bench_name in BENCHMARKS:
        summary["per_cell"][bench_name] = {}
        for method_name, _ in METHODS:
            sub = [
                r
                for r in rows
                if r["benchmark"] == bench_name and r["method"] == method_name
            ]
            summary["per_cell"][bench_name][method_name] = {
                "median_half_width": statistics.median(r["half_width"] for r in sub),
                "median_samples": statistics.median(r["samples_used"] for r in sub),
                "coverage": sum(r["covers_truth"] for r in sub) / len(sub),
                "certified": sub[0]["certified"],
            }

    runs_path = RESULTS_DIR / "nonenumerable.jsonl"
    summary_path = RESULTS_DIR / "nonenumerable.json"
    with runs_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n# wrote {runs_path}")
    print(f"# wrote {summary_path}")


if __name__ == "__main__":
    main()
