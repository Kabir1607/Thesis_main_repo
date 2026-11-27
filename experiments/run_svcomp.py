"""SV-COMP sweep with subprocess-level isolation.

For each transpilable SV-COMP program with non-trivial nondet inputs,
this driver:

  1. Spawns a child process to run all the per-program work
     (transpile, ground-truth MC, plain-MC + DiSE methods).
  2. Imposes a hard per-program wall-clock deadline; if the child
     exceeds it (e.g. the program has a non-terminating loop), the
     child is terminated and the program is skipped with a
     ``timeout`` row.
  3. Streams per-cell rows to ``experiments/results/svcomp_runs.jsonl``.

The subprocess isolation is essential because many SV-COMP programs
have divergent loops on adversarial inputs (e.g., ``while (i != 0):
i -= 2`` for odd ``i``).  Python's signal-based timeouts can be
delayed by C-extension calls (notably Z3); a subprocess kill cannot
be.

Distribution policy: ``BoundedGeometric(p=0.1, N=100)`` by default --
a skewed operational model in which small integer inputs are
exponentially more common, the skew production traffic typically
shows (the same model as the Section 3 running example).  Per-program
override via ``svcomp_samples/<name>.dist.json``.

Usage::

    uv run python experiments/run_svcomp.py
    MAX_PROGRAMS=50 uv run python experiments/run_svcomp.py  # smoke
"""

from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "experiments" / "svcomp_samples"
RESULTS_DIR = ROOT / "experiments" / "results"
RUNS_PATH = RESULTS_DIR / "svcomp_runs.jsonl"
SUMMARY_PATH = RESULTS_DIR / "svcomp_summary.json"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BUDGET = int(os.environ.get("BUDGET", 2000))
# Ground truth is computed exactly by enumerating the finite input
# domain when that domain has at most TRUTH_ENUM_MAX points and the
# enumeration completes within TRUTH_ENUM_SECONDS; otherwise a Monte
# Carlo estimate of MC_TRUTH_BUDGET samples is used.
MC_TRUTH_BUDGET = int(os.environ.get("MC_TRUTH_BUDGET", 10000))
TRUTH_ENUM_MAX = int(os.environ.get("TRUTH_ENUM_MAX", 2_000_000))
TRUTH_ENUM_SECONDS = float(os.environ.get("TRUTH_ENUM_SECONDS", 6.0))
DELTA = 0.05
PER_CELL_SECONDS = float(os.environ.get("PER_CELL_SECONDS", 3.0))
PER_PROGRAM_TIMEOUT = float(os.environ.get("PER_PROGRAM_TIMEOUT", 20.0))
SEED = int(os.environ.get("SEED", 0))
MAX_PROGRAMS = int(os.environ.get("MAX_PROGRAMS", 10_000))


# ---------------------------------------------------------------------------
# Worker that runs *inside* a subprocess
# ---------------------------------------------------------------------------


def _worker(prog_path_str: str, conn: "mp.connection.Connection") -> None:  # type: ignore[name-defined]
    """Process one program. Send the resulting list-of-cells back over
    ``conn``. Any exception is caught and sent as a single error row."""
    # Re-import inside the worker (so the parent doesn't pay startup
    # cost for every program).
    import os
    import time as _time
    import numpy as np
    sys.path.insert(0, str(ROOT / "src"))

    from dise.baselines import (
        DiSEBaseline,
        PlainMonteCarlo,
    )
    from dise.baselines.pse import PSEBaseline
    from dise.distributions import (
        BoundedGeometric, ProductDistribution, Uniform,
    )
    from dise.estimator.api import estimate
    from dise.frontends import Untranslatable, transpile_c_program

    prog_path = Path(prog_path_str)
    name = prog_path.name

    def _build_distribution(nondets):
        sidecar = prog_path.with_suffix(".dist.json")
        override = {}
        if sidecar.exists():
            try:
                override = json.loads(sidecar.read_text())
            except Exception:
                pass
        out = {}
        for n in nondets:
            if n.name in override:
                spec = override[n.name]
                kind = spec.get("type", "Uniform")
                if kind == "BoundedGeometric":
                    out[n.name] = BoundedGeometric(p=spec["p"], N=spec["N"])
                else:
                    out[n.name] = Uniform(spec.get("lo", 0), spec.get("hi", 100))
            else:
                # Skewed operational model: small integer inputs are
                # exponentially more common (Section 3's running-example
                # distribution). Support {1,...,100} keeps the per-region
                # mass exact by enumeration, as under the former Uniform.
                out[n.name] = BoundedGeometric(p=0.1, N=100)
        return out

    def _safe_call(program, kwargs):
        """Call ``program(**kwargs)``; return 1 on AssertionError,
        0 on normal return, None on any other exception (transpiler
        artefact)."""
        try:
            program(**kwargs)
            return 0
        except AssertionError:
            return 1
        except Exception:
            return None

    try:
        # 1. Transpile.
        tr = transpile_c_program(str(prog_path))
        if len(tr.nondets) == 0:
            conn.send({"skip": "no_nondets", "name": name})
            return

        ns: dict[str, Any] = {}
        exec(compile(tr.python_source, "<svcomp>", "exec"), ns)
        program = ns[tr.function_name]

        distribution = _build_distribution(tr.nondets)

        # 2. Ground truth. The input domain is finite, so mu is
        # computed *exactly* by enumerating it, which makes the coverage
        # check reference-noise-free (Section 7). Enumeration is
        # time-bounded; programs whose domain is too large or too slow
        # to enumerate within TRUTH_ENUM_SECONDS fall back to a Monte
        # Carlo estimate. The parent process enforces the overall cap.
        dist_inst = ProductDistribution(factors=dict(distribution))
        keys = list(distribution.keys())
        supports = {k: distribution[k].support_bounds() for k in keys}
        domain_size = 1
        for lo_s, hi_s in supports.values():
            domain_size *= (hi_s - lo_s + 1)
        mu_truth: float | None = None
        truth_mode = "mc"
        if domain_size <= TRUTH_ENUM_MAX:
            ranges = [range(supports[k][0], supports[k][1] + 1) for k in keys]
            acc = 0.0
            done = True
            t_enum = _time.perf_counter()
            for j, combo in enumerate(itertools.product(*ranges)):
                if (j & 0xFFFF) == 0 and (
                    _time.perf_counter() - t_enum > TRUTH_ENUM_SECONDS
                ):
                    done = False
                    break
                x = dict(zip(keys, combo))
                if _safe_call(program, x) == 1:
                    acc += dist_inst.pmf(x)
            if done:
                mu_truth = acc
                truth_mode = "exact"
        if mu_truth is None:
            rng = np.random.default_rng(12345)
            batch = dist_inst.sample(rng, MC_TRUTH_BUDGET)
            hits = 0
            for i in range(MC_TRUTH_BUDGET):
                x = {k: int(batch[k][i]) for k in keys}
                if _safe_call(program, x) == 1:
                    hits += 1
            mu_truth = hits / MC_TRUTH_BUDGET

        # 3. Methods. Wrap the program so AssertionError -> v=1.
        #
        # IMPORTANT: we deliberately do NOT swallow non-AssertionError
        # exceptions here.  An earlier version caught ``Exception`` and
        # returned 0, which silently corrupted DiSE / PSE certified
        # intervals on programs whose concolic trace hit a TypeError
        # (e.g. unsupported bitwise op): every "sample" became a zero
        # with no recorded branches, the root leaf closed trivially as
        # CLOSED_FALSE, and the estimator returned mu_hat=0 with hw=0
        # on a program whose true mass was ~0.5.  Bubbling the
        # exception up causes the parent process to record the cell as
        # a `skip: error` row instead, which is the correct outcome.
        def _wrapped(**kw):
            try:
                program(**kw)
                return 0
            except AssertionError:
                return 1

        cells = []

        # Method 1: Plain Monte Carlo + Wilson interval.  The vanilla
        # statistical floor every paper should beat.
        t0 = _time.perf_counter()
        pmc = PlainMonteCarlo()
        r = pmc.run(program=_wrapped, distribution=distribution,
                    property_fn=lambda y: y == 1,
                    budget=BUDGET, delta=DELTA, seed=SEED)
        lo, hi = r.interval
        cells.append(dict(method="plain_mc", mu_hat=r.mu_hat,
                          interval=[lo, hi], half_width=r.half_width,
                          samples=r.samples_used,
                          wall_s=_time.perf_counter() - t0,
                          extras=dict(r.extras)))

        # Method 2: PSE — purely symbolic baseline (Geldenhuys-Dwyer-
        # Visser 2012 lineage).  Exhaustive DFS over the frontier with
        # exact closed-form mass on closed leaves.  Wins on programs
        # whose paths fully enumerate within budget; falls back to
        # importance-sample mass on the few non-LIA leaves.
        t0 = _time.perf_counter()
        pse = PSEBaseline(n_bootstrap=300, time_limit_s=PER_CELL_SECONDS)
        r = pse.run(program=_wrapped, distribution=distribution,
                    property_fn=lambda y: y == 1,
                    budget=BUDGET, delta=DELTA, seed=SEED)
        lo, hi = r.interval
        cells.append(dict(method="pse_exact", mu_hat=r.mu_hat,
                          interval=[lo, hi], half_width=r.half_width,
                          samples=r.samples_used,
                          wall_s=_time.perf_counter() - t0,
                          extras={k: v for k, v in r.extras.items()
                                  if k in ("n_leaves", "n_open_leaves",
                                           "n_closed_true",
                                           "n_closed_false",
                                           "enumerated_paths", "W_open",
                                           "hit_time_limit",
                                           "hit_path_limit")}))

        # Method 3: DiSE — our adaptive symbolic-importance partition
        # estimator with WSR PrPl-EB anytime-valid intervals.  The
        # algorithm the paper is about.
        t0 = _time.perf_counter()
        result = estimate(
            program=_wrapped,
            distribution=distribution,
            property_fn=lambda y: y == 1,
            epsilon=1e-6, delta=DELTA, budget=BUDGET,
            budget_seconds=PER_CELL_SECONDS,
            method="betting", bootstrap=300, batch_size=100, seed=SEED,
        )
        lo, hi = result.interval
        cells.append(dict(method="dise_betting", mu_hat=result.mu_hat,
                          interval=[lo, hi],
                          half_width=(hi - lo) / 2.0,
                          samples=result.samples_used,
                          wall_s=_time.perf_counter() - t0,
                          extras={"terminated": result.terminated_reason,
                                  "n_leaves": result.n_leaves,
                                  "n_open": result.n_open_leaves}))

        conn.send({
            "ok": True, "name": name,
            "n_nondets": len(tr.nondets),
            "mu_truth": mu_truth,
            "truth_mode": truth_mode,
            "cells": cells,
        })
    except Untranslatable as e:
        conn.send({"skip": "untranslatable", "name": name, "reason": str(e)[:120]})
    except Exception as e:
        conn.send({"skip": "error", "name": name,
                   "reason": f"{type(e).__name__}: {e}"[:200]})
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Parent: orchestrate
# ---------------------------------------------------------------------------


def _process_with_timeout(prog_path: Path, timeout: float) -> dict:
    parent_conn, child_conn = mp.Pipe(duplex=False)
    p = mp.Process(target=_worker, args=(str(prog_path), child_conn))
    p.start()
    child_conn.close()
    t0 = time.perf_counter()
    result: dict | None = None
    try:
        if parent_conn.poll(timeout):
            result = parent_conn.recv()
        else:
            result = {"skip": "timeout", "name": prog_path.name,
                       "reason": f"exceeded {timeout}s"}
    except EOFError:
        result = {"skip": "worker_died", "name": prog_path.name}
    p.join(timeout=2.0)
    if p.is_alive():
        p.terminate()
        p.join(timeout=1.0)
        if p.is_alive():
            p.kill()
            p.join()
    parent_conn.close()
    return result or {"skip": "worker_died", "name": prog_path.name}


def main() -> int:
    mp.set_start_method("fork", force=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_PATH.write_text("")

    programs = sorted(CORPUS_DIR.glob("*.c"))[:MAX_PROGRAMS]
    print(f"# SV-COMP sweep (subprocess-isolated)")
    print(f"# {len(programs)} programs to consider")
    print(f"# budget={BUDGET}, mc_truth_budget={MC_TRUTH_BUDGET}")
    print(f"# per_cell_seconds={PER_CELL_SECONDS}  "
          f"per_program_timeout={PER_PROGRAM_TIMEOUT}")
    print(f"# seed={SEED}")
    print()

    n_done = 0
    n_kept = 0
    skip_counts: dict[str, int] = {}
    t_start = time.perf_counter()

    with open(RUNS_PATH, "a") as fh:
        for prog_path in programs:
            n_done += 1
            t0 = time.perf_counter()
            res = _process_with_timeout(prog_path, PER_PROGRAM_TIMEOUT)
            elapsed = time.perf_counter() - t0
            if "skip" in res:
                tag = res["skip"]
                skip_counts[tag] = skip_counts.get(tag, 0) + 1
                if n_done % 25 == 0 or tag in ("timeout", "worker_died"):
                    print(f"  [{n_done:4d}/{len(programs)}] "
                          f"{prog_path.name[:35]:35s} SKIP {tag} ({elapsed:5.1f}s)",
                          flush=True)
                continue
            n_kept += 1
            for cell in res["cells"]:
                lo, hi = cell["interval"]
                row = {
                    "program": res["name"],
                    "n_nondets": res["n_nondets"],
                    "mu_truth": res["mu_truth"],
                    "truth_mode": res.get("truth_mode", "mc"),
                    "budget": BUDGET, "delta": DELTA, "seed": SEED,
                    "interval_contains_truth": (lo <= res["mu_truth"] <= hi),
                    **cell,
                }
                fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            line = f"  [{n_done:4d}/{len(programs)}] {res['name'][:30]:30s} mu*={res['mu_truth']:.3f}"
            short_map = {"plain_mc": "mc",
                         "pse_exact": "pse",
                         "dise_betting": "dise"}
            for c in res["cells"]:
                short = short_map.get(c["method"], c["method"][:4])
                line += f"  {short}={c['half_width']:.3f}"
            line += f"  ({elapsed:4.1f}s)"
            print(line, flush=True)

    wall = time.perf_counter() - t_start
    print(f"\n# total wall-clock: {wall:.1f}s")
    print(f"# kept: {n_kept}/{len(programs)}")
    print(f"# skipped:")
    for tag, c in sorted(skip_counts.items(), key=lambda x: -x[1]):
        print(f"  {c:4d}  {tag}")

    _build_summary()
    return 0


def _build_summary() -> None:
    if not RUNS_PATH.exists():
        return
    rows = [json.loads(l) for l in open(RUNS_PATH) if l.strip()]
    from collections import defaultdict
    bymethod: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        bymethod[r["method"]].append(r)
    per_method: dict[str, dict] = {}
    for method, rs in bymethod.items():
        hws = sorted([r["half_width"] for r in rs])
        samples = sorted([r["samples"] for r in rs])
        cov = [r["interval_contains_truth"] for r in rs
               if r.get("interval_contains_truth") is not None]
        per_method[method] = {
            "n_programs": len(rs),
            "median_half_width": hws[len(hws)//2] if hws else None,
            "p90_half_width": hws[int(0.9*len(hws))] if hws else None,
            "median_samples": samples[len(samples)//2] if samples else None,
            "fraction_hw_zero": sum(1 for h in hws if h == 0) / max(len(hws), 1),
            "coverage": sum(1 for c in cov if c) / max(len(cov), 1) if cov else None,
        }
    by_prog: dict[str, dict] = defaultdict(dict)
    for r in rows:
        by_prog[r["program"]][r["method"]] = r

    # Pair-wise comparisons of DiSE against each non-DiSE comparator.
    # For each pair: win = strictly tighter half-width; tie = equal up
    # to 1e-9; loss = strictly wider.
    pairwise: dict[str, dict] = {}
    for opp in ("plain_mc", "pse_exact"):
        wins = ties = losses = exact = n_paired = 0
        for cells in by_prog.values():
            if "dise_betting" not in cells or opp not in cells:
                continue
            d_hw = cells["dise_betting"]["half_width"]
            o_hw = cells[opp]["half_width"]
            n_paired += 1
            if d_hw == 0:
                exact += 1
            if d_hw < o_hw - 1e-9:
                wins += 1
            elif d_hw > o_hw + 1e-9:
                losses += 1
            else:
                ties += 1
        pairwise[f"dise_vs_{opp}"] = {
            "n_paired": n_paired,
            "dise_wins_hw": wins,
            "tie_hw": ties,
            "opponent_wins_hw": losses,
            "dise_exact_certificate": exact,
        }

    summary = {
        "metadata": {"budget": BUDGET, "delta": DELTA, "seed": SEED,
                     "per_cell_seconds": PER_CELL_SECONDS,
                     "per_program_timeout": PER_PROGRAM_TIMEOUT,
                     "mc_truth_budget": MC_TRUTH_BUDGET,
                     "methods": ["plain_mc", "pse_exact", "dise_betting"]},
        "per_method": per_method,
        "pairwise_dise_vs_each": pairwise,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"# wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    raise SystemExit(main())
