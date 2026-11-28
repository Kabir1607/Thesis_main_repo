# Agent Context: Adding Certified Adaptive Importance Sampling (AIS) to DiSE

## 0. Mission & Ground Rules

You are extending **DiSE** (Distribution-Aware Reliability Estimation) to support **certified Adaptive Importance Sampling (AIS)** inside unclosable `OPEN` leaves, verifying when and how it improves efficiency, generating publication-ready plots, and documenting every modification in `changes.md`.

### Hard Constraints
1. **Do not break existing baselines or tests.** `plain_mc`, `pse_exact`, and baseline `dise` (`method="betting"` / `method="wilson"`) must continue to pass existing tests and produce comparable baseline behavior unless an explicit opt-in flag (e.g., `use_ais=True` or `method="ais_betting"`) is enabled.
2. **Preserve DiSE's soundness contract (Theorem 1):**
   - **I1 (Mass conservation):** Leaf masses remain exact on axis-aligned boxes ($\beta = 0$); AIS does **not** alter leaf mass formulas.
   - **I2 (Closure soundness):** SMT closure via Z3 is untouched. AIS runs **only** inside `OPEN` leaves that Z3 cannot close (e.g., solver returns `unknown`, trace is concretized via bitwise/nonlinear ops, or observations disagree and closure fails).
   - **I3 (Anytime validity):** Every per-leaf bound must hold simultaneously at **all** sample counts $n \ge 1$ under adaptive stopping and predictable proposal updates, using a **fixed a-priori** leaf confidence budget $\delta_\pi = \delta / K_{\max}$ (never retroactively changing past bets $\lambda_i$ when the frontier changes).
3. **Be scientifically honest about the verification/falsification asymmetry:**
   - AIS makes **certified falsification** (proving failure rate exceeds a threshold $\tau$, i.e., a certified *lower* bound on failure probability) dramatically faster—often orders of magnitude fewer runs.
   - AIS **cannot** eliminate the $\Omega(1/p)$ cost floor for **certified verification** (proving failure rate $\le \tau$, i.e., a certified *upper* bound on failure probability) inside an unclosable leaf. Aggressive failure-focused proposals ($\alpha \to 0$) actually *worsen* upper bounds, while large defensive mixtures ($\alpha \approx 0.8\text{–}0.9$) give a constant-factor speedup on tight relative precision.
   - Your tests and plots **must demonstrate both sides clearly**.
4. **Compute awareness:** Target machine has 16 GB RAM and an AMD Ryzen 7000 HS CPU (Arch Linux, Python `.venv`). Use NumPy vectorization so multi-seed experiments and plotting finish in a few minutes rather than hours.
5. **Mandatory `changes.md` logging:** `changes.md` already exists in the repo root. Append/write a complete, structured changelog there covering every file created or edited, every mathematical/engineering choice, exact commands run, test outcomes, and paths to all generated plots and data tables.

---

## 1. Codebase Map & Mandatory Pre-Flight Audit

Before writing code, inspect these exact files (do not guess interfaces):
- `src/dise/estimator/__init__.py` — `prpl_eb_halfwidth_anytime` (:180–272), `prpl_eb_center` (:275–309), `compute_estimator_state`, `EstimatorState`, `K_bounds` (:451–455), global `[lo, hi]` assembly (:513–514).
- `src/dise/regions/_frontier.py` — `FrontierNode` (:20–57), `add_observation` (:487–511), `Frontier.refine` (:303–364), `reset_observations` (:116–122).
- `src/dise/scheduler/__init__.py` — `SchedulerConfig`, `_leaf_hw_proxy` (:416–424), `_execute_refine` (:401–410), `_allocate_one_batch`, action scoring (:444–498), stopping rule (:523).
- `src/dise/sampler/__init__.py` and distribution definitions (verify exact `BG(p, N)` support convention: 0-indexed `{0, ..., N-1}` vs 1-indexed `{1, ..., N}`).
- `src/dise/baselines/rare_event.py` — inspect existing defensive IS / subset simulation baselines so you reuse conventions and avoid duplicating logic.
- `tests/test_betting.py` — existing confidence-sequence unit tests.

### Known Soundness Pitfalls in Existing Code (Do NOT Copy into AIS)
1. `compute_estimator_state` currently computes symmetric scalar half-widths centered on `leaf.mu_hat = n_hits / n_samples` instead of `prpl_eb_center`. **For AIS, you must return explicit asymmetric bounds `[L_pi, U_pi]` directly from the betting confidence sequence**—do not approximate them as `mu_hat +/- W`.
2. `K_bounds` is currently dynamic (`len(open_leaves) + len(closed_true_with_var)`) and gets fed back into `prpl_eb_halfwidth_anytime`, retroactively recomputing past bets $\lambda_i$. **For AIS, use a fixed a-priori `K_max` in `SchedulerConfig` (e.g., `delta_per_leaf = delta / K_max`)** so $\delta_\pi$ is fixed before sampling begins. Log whether you also add an opt-in fix for baseline `PrPl-EB` or leave baseline untouched for clean A/B comparison (document your choice in `changes.md`).
3. Verify whether `W_open` in `estimator/__init__.py:428–435` sums over *all* open leaves or only *unresolved* (`n_samples == 0`) open leaves. Follow DiSE Definition 6: sampled open leaves contribute via their per-leaf bounds `[L_pi, U_pi]`, while unresolved (`n_samples == 0`) open leaves contribute `[0, 1]` (i.e., mass `w_hat` to `W_open`).

---

## 2. Mathematical Specification (Follow Strictly)

### 2.1 Sign Convention (CRITICAL — Do Not Flip!)
In DiSE:
- $\varphi(P(x)) = 1$ means **SAFE / PASS** (reliability $\mu = \Pr[\varphi(P(x)) = 1]$).
- $\varphi(P(x)) = 0$ means **FAIL** (failure rate $p_{\text{fail}} = 1 - \mu = \Pr[\varphi(P(x)) = 0]$).

Inside an `OPEN` leaf $\pi$ with region $R_\pi$, define:
- $R_t = 1 - \varphi(P(x_t)) \in \{0, 1\}$ (the **failure indicator**: 1 on failure, 0 on pass).
- True conditional failure rate: $p_{\text{fail}, \pi} = \Pr_{x \sim \mathcal{D}|_{R_\pi}}[R = 1]$.
- True conditional reliability: $p_\pi = 1 - p_{\text{fail}, \pi}$.
- If the AIS betting CS certifies $p_{\text{fail}, \pi} \in [L_{\text{fail}, \pi}, U_{\text{fail}, \pi}]$, then the certified reliability interval for leaf $\pi$ is:
  $$L_\pi = 1 - U_{\text{fail}, \pi}, \qquad U_\pi = 1 - L_{\text{fail}, \pi}$$
- **Falsification** ($1 - \mu > \tau$, program is too buggy) $\iff$ certified **lower bound** $L_{\text{fail}} > \tau$ ($\iff U_{\text{global}} < 1 - \tau$).
- **Verification** ($1 - \mu \le \tau$, program is safe enough) $\iff$ certified **upper bound** $U_{\text{fail}} \le \tau$ ($\iff L_{\text{global}} \ge 1 - \tau$).

### 2.2 Discrete Proposal Family: Exponentially Twisted $BG(p, N)$ + Defensive Mixture
For an axis-aligned leaf box $R_\pi = \prod_{i=1}^d \{\ell_i, \dots, u_i\}$ under independent bounded geometric marginals $\mathcal{D}_i = BG(p_i, N_i)$ (where $\mathcal{D}_i(k) \propto (1 - p_i)^k$):

1. **Exponentially tilted marginal $\tilde{q}_{\theta_i}$ on $\{\ell_i, \dots, u_i\}$:**
   Multiplying $(1 - p_i)^k$ by $e^{\theta_i k}$ yields a truncated geometric distribution with ratio $r_i = (1 - p_i) e^{\theta_i} > 0$.
   Shift to $j = k - \ell_i \in \{0, \dots, M_i - 1\}$ where $M_i = u_i - \ell_i + 1$:
   $$\tilde{q}_{\theta_i}(\ell_i + j) = \frac{r_i^j}{\sum_{t=0}^{M_i - 1} r_i^t}$$
   **Numerical stability (mandatory):**
   - If $|r_i - 1| < 10^{-12}$: uniform on $\{0, \dots, M_i - 1\}$; $\log \tilde{q} = -\log M_i$; inverse-CDF is $j = \lfloor u \cdot M_i \rfloor$.
   - If $r_i < 1$: $\log Z = \log(1 - r_i^{M_i}) - \log(1 - r_i)$ (use `math.expm1` / `log1p` where appropriate); exact inverse-CDF is $j = \lfloor \log(1 - u(1 - r_i^{M_i})) / \log r_i \rfloor$, clamped to $[0, M_i - 1]$.
   - If $r_i > 1$: **never compute $r_i^{M_i}$ directly** (overflows for large $M_i$). Substitute $s_i = 1 / r_i < 1$ and flipped index $j' = (M_i - 1) - j$. Sample $j'$ using the $s_i < 1$ formula and return $j = (M_i - 1) - j'$. Compute $\log \tilde{q}(\ell_i + j) = j' \log s_i - \log((1 - s_i^{M_i})/(1 - s_i))$.
   - Extend cleanly to uniform or general finite discrete marginals (if any benchmarks use non-BG marginals, exponential tilting $p_i(k) e^{\theta_i k} / Z$ works by exact normalized array lookup on small discrete supports).

2. **Defensive Mixture Proposal ($q_t$):**
   Fix $\alpha_t \in (0, 1]$ (default configurable, e.g., $\alpha = 0.5$ for general two-sided / relative precision, $\alpha = 0.1$ for falsification mode, $\alpha = 0.9$ for verification/upper-bound mode):
   $$q_t(x) = \alpha_t \, \mathcal{D}|_{R_\pi}(x) + (1 - \alpha_t) \, \tilde{q}_{\boldsymbol{\theta}_t}(x)$$
   - **Sampling:** With probability $\alpha_t$, draw $x \sim \mathcal{D}|_{R_\pi}$; with probability $1 - \alpha_t$, draw $x \sim \tilde{q}_{\boldsymbol{\theta}_t}$. Zero rejection!
   - **Numerically stable exact importance weight:** Compute $\ell_{\mathcal{D}} = \log \mathcal{D}|_{R_\pi}(x)$ and $\ell_{\tilde{q}} = \log \tilde{q}_{\boldsymbol{\theta}_t}(x)$ by summing coordinate log-PMFs. Then:
     $$W_t = \frac{\mathcal{D}|_{R_\pi}(x)}{q_t(x)} = \frac{1}{\alpha_t + (1 - \alpha_t) \exp(\ell_{\tilde{q}} - \ell_{\mathcal{D}})}$$
     (Guard $\ell_{\tilde{q}} - \ell_{\mathcal{D}} > 700 \implies W_t = 0.0$ safely.) This guarantees $0 \le W_t \le 1/\alpha_t$ everywhere on $R_\pi$, and $\mathbb{E}_{x_t \sim q_t}[W_t R_t \mid \mathcal{F}_{t-1}] = p_{\text{fail}, \pi}$.

3. **Predictable Proposal Adaptation (Between Batches Only!):**
   - **CRITICAL:** Parameters $\boldsymbol{\theta}_t$ and $\alpha_t$ for batch $b$ must be determined **strictly before** drawing batch $b$, using only observations from batches $1, \dots, b-1$.
   - **Bootstrap (before any failure is observed in leaf $\pi$):** Start with an exploratory tilt pushing mass toward the low-probability tail of $BG(p_i, N_i)$ (e.g., uniform on the box, $r_i = 1 \iff \theta_i = -\log(1 - p_i)$) inside the defensive mixture. (If the trace exposes numeric branch distances to a failing branch, you may optionally use branch-distance elite selection as an enhancement—keep it cleanly optional and documented.)
   - **Cross-Entropy (CE) update (once $\ge 1$ failure is observed):** For each coordinate $i$, compute the importance-weighted empirical mean of failing inputs:
     $$\bar{x}_i = \frac{\sum_{s: R_s = 1} W_s \, x_{s,i}}{\sum_{s: R_s = 1} W_s}$$
     Apply mild Laplace/shrinkage smoothing toward the box midpoint (or clip to $[\ell_i + 0.05, u_i - 0.05]$ if $M_i > 1$) so $\theta_i$ stays finite. Solve $\mathbb{E}_{\tilde{q}_{\theta_i}}[X_i] = \bar{x}_i$ for $\log r_i$ via 1D bisection (guaranteed unique solution because the mean of an exponential family is strictly increasing in its natural parameter). Optionally smooth updates via $\boldsymbol{\theta}_{\text{new}} = (1 - \rho)\boldsymbol{\theta}_{\text{old}} + \rho \boldsymbol{\theta}_{\text{CE}}$ (e.g., $\rho = 0.7$).

### 2.3 Anytime-Valid Off-Policy Betting Confidence Sequence (Slot 3 / I3)
Let $Z_i = W_i R_i \ge 0$ for $i = 1, \dots, t$, with predictable defensive weight caps $W_{\max, i} = 1 / \alpha_i$ (so $0 \le Z_i \le W_{\max, i}$).
Split the leaf's fixed error budget $\delta_\pi = \delta_{\text{lower}} + \delta_{\text{upper}}$ (default $\delta_\pi / 2$ each for two-sided intervals; allow one-sided modes $\delta_{\text{lower}} = \delta_\pi$ for pure falsification or $\delta_{\text{upper}} = \delta_\pi$ for pure verification).

Use a **grid-mixture betting test supermartingale** over $G$ fixed log-spaced bet fractions $\gamma_1, \dots, \gamma_G \in [10^{-4}, 0.95]$ (e.g., $G = 30$, uniform prior $1/G$):

1. **Certified Lower Bound on Failure $L_{\text{fail}, t}$ (No Weight Cap Needed!):**
   To test $H_0: p_{\text{fail}} \le m$ for $m \in (0, 1)$, define for each grid point $\gamma_j$:
   $$\log K_{t, j}^+(m) = \sum_{i=1}^t \log\!\left((1 - \gamma_j) + \gamma_j \frac{Z_i}{m}\right)$$
   - Because $Z_i \ge 0$ and $\gamma_j \in (0, 1)$, every factor is strictly positive even if $W_i$ is unbounded, and $\mathbb{E}[\text{factor} \mid \mathcal{F}_{i-1}] = 1 + \gamma_j (p_{\text{fail}} / m - 1) \le 1$ under $H_0$.
   - Combined log-wealth: $\log K_t^+(m) = \text{logsumexp}_j(\log K_{t,j}^+(m)) - \log G$.
   - Reject $m$ if $\log K_t^+(m) \ge \log(1 / \delta_{\text{lower}})$.
   - **Monotonicity:** $\log K_t^+(m)$ is strictly non-increasing in $m$. Find the instantaneous lower bound $L_{\text{fail}, t}^{\text{inst}} = \sup \{m \in (0, 1) : \log K_t^+(m) \ge \log(1/\delta_{\text{lower}})\}$ via **1D binary search** (returns `0.0` if even $m = 10^{-12}$ is not rejected).
   - **Running intersection:** Maintain $L_{\text{fail}, t} = \max(L_{\text{fail}, t-1}, L_{\text{fail}, t}^{\text{inst}})$ across evaluation steps.

2. **Certified Upper Bound on Failure $U_{\text{fail}, t}$ (Uses Defensive Cap $W_{\max, i} = 1/\alpha_i$):**
   To test $H_0: p_{\text{fail}} \ge m$ for $m \in (0, 1)$, define for each grid point $\gamma_j$:
   $$\log K_{t, j}^-(m) = \sum_{i=1}^t \log\!\left((1 - \gamma_j) + \gamma_j \frac{W_{\max, i} - Z_i}{W_{\max, i} - m}\right)$$
   - Because $0 \le Z_i \le W_{\max, i}$ and $m < 1 \le W_{\max, i}$, every factor is strictly positive, and $\mathbb{E}[\text{factor} \mid \mathcal{F}_{i-1}] = 1 + \gamma_j \frac{m - p_{\text{fail}}}{W_{\max, i} - m} \le 1$ under $H_0$.
   - Combined log-wealth: $\log K_t^-(m) = \text{logsumexp}_j(\log K_{t,j}^-(m)) - \log G$.
   - Reject $m$ if $\log K_t^-(m) \ge \log(1 / \delta_{\text{upper}})$.
   - **Monotonicity:** $\log K_t^-(m)$ is strictly non-decreasing in $m$. Find $U_{\text{fail}, t}^{\text{inst}} = \inf \{m \in (0, 1) : \log K_t^-(m) \ge \log(1/\delta_{\text{upper}})\}$ via **1D binary search** (returns `1.0` if $m = 1 - 10^{-12}$ is not rejected).
   - **Running intersection:** Maintain $U_{\text{fail}, t} = \min(U_{\text{fail}, t-1}, U_{\text{fail}, t}^{\text{inst}})$.

3. **Consistency guard:** Always ensure $0 \le L_{\text{fail}, t} \le U_{\text{fail}, t} \le 1$ (if running intersection ever numerically crosses by $10^{-12}$, clamp cleanly). Notice that when $\alpha_i \equiv 1$ ($W_i \equiv 1$, plain sampling), these exact same formulas recover a valid finite-sample betting CS for Bernoulli data!

### 2.4 Global Reliability Interval & Stopping Rule
Convert per-leaf failure intervals $[L_{\text{fail}, \pi}, U_{\text{fail}, \pi}]$ to reliability intervals $[L_\pi, U_\pi] = [1 - U_{\text{fail}, \pi}, 1 - L_{\text{fail}, \pi}]$. For non-AIS leaves, use their standard `[L_pi, U_pi]`. Then:
$$L_{\text{global}} = \max\!\left(0,\; \sum_{\pi \in \text{CLOSED-T}} \hat{w}_\pi + \sum_{\pi \in \text{OPEN, sampled}} \hat{w}_\pi L_\pi - \beta\right)$$
$$U_{\text{global}} = \min\!\left(1,\; \sum_{\pi \in \text{CLOSED-T}} \hat{w}_\pi + \sum_{\pi \in \text{OPEN, sampled}} \hat{w}_\pi U_\pi + \sum_{\pi \in \text{OPEN, unresolved}} \hat{w}_\pi + \beta\right)$$
Point estimate for AIS leaves: $\hat{p}_{\text{fail}, \pi} = \text{clip}\!\left(\frac{1}{n_\pi}\sum_{i=1}^{n_\pi} Z_i, 0, 1\right)$, $\hat{p}_\pi = 1 - \hat{p}_{\text{fail}, \pi}$.
- **Estimation stopping:** Stop when $(U_{\text{global}} - L_{\text{global}}) / 2 \le \epsilon$.
- **Optional threshold stopping (SLA mode, `threshold_tau`):** Stop early as soon as $1 - L_{\text{global}} \le \tau$ (**VERIFIED**: failure rate $\le \tau$) or $1 - U_{\text{global}} > \tau$ (**FALSIFIED**: failure rate $> \tau$). This directly showcases AIS's falsification speedup!

---

## 3. Architecture & Implementation Plan

Keep changes modular and idiomatic to the existing codebase:

1. **`src/dise/ais/` (new subpackage):**
   - `proposals.py`: `TwistedBGProposal` (exact log-PMF, overflow-safe truncated inverse-CDF sampling, exact mean computation + bisection CE updater, defensive mixture sampling and exact weight computation).
   - `betting_cs.py`: `OffPolicyBettingCS` stateful per-leaf tracker storing `(Z_i, alpha_i)` arrays (or incremental update), log-space vectorized lower/upper binary search, running intersection `[L_fail, U_fail]`, fixed `delta_pi`.
2. **`src/dise/regions/_frontier.py`:**
   - Extend `FrontierNode` cleanly (default `None` so existing code is unaffected) to optionally hold AIS state (`ais_cs: OffPolicyBettingCS | None`, `ais_proposal: TwistedBGProposal | None`, `is_unclosable: bool = False`, plus weighted failure sample storage needed for CE updates).
   - Ensure `reset_observations()` clears AIS state too when a leaf refines (preserving post-refinement freshness).
3. **`src/dise/estimator/__init__.py`:**
   - Add support for asymmetric per-leaf intervals (`[L_pi, U_pi]`) when `method="ais_betting"` (and/or when a leaf has an active `ais_cs`), using a fixed a-priori `K_max` for `delta_per_leaf = delta / K_max`.
   - Expose `lo`, `hi`, and optional threshold decision (`VERIFIED` / `FALSIFIED` / `UNDECIDED`) cleanly on `EstimatorState`.
4. **`src/dise/scheduler/__init__.py`:**
   - Add opt-in config fields to `SchedulerConfig`: `enable_ais: bool = False`, `ais_alpha: float = 0.5`, `ais_k_max: int = 1024`, `ais_mode: Literal["two_sided", "falsify", "verify"] = "two_sided"`, `threshold_tau: float | None = None`.
   - Detect unclosable leaves (e.g., `leaf.has_concretized_obs` or solver returned `unknown` / failed to refine further after repeated attempts) and offer an `ais_allocate` action priced by predicted drop in `w_hat * (U_pi - L_pi)` (or drop in relevant one-sided distance to `threshold_tau`).
   - Inside `_allocate_ais_batch(leaf, k)`: update proposal $\boldsymbol{\theta}_t$ strictly from prior batches, draw $k$ samples from $q_t$, run program $P$, record $(Z_i, \alpha_i)$ into `leaf.ais_cs`, and update `FrontierNode` stats.

---

## 4. Required Tests & Experiments

### 4.1 Unit & Soundness Test Suite (`tests/test_ais.py`, runnable via `pytest`)
1. **Proposal math tests:**
   - Verify PMF sums to $1 \pm 10^{-12}$ on arbitrary boxes $[\ell, u]$ for $r < 1$, $r = 1$, $r > 1$, and large $N = 10^4$ (no overflow).
   - Verify $\sum_{x \in R} q(x) W(x) = 1$ (exact expectation of likelihood ratio) and $W(x) \le 1/\alpha$ for all $x$.
   - Verify empirical samples from `TwistedBGProposal` match exact theoretical PMF (e.g., 50,000 draws within tolerance).
   - Verify CE bisection solver recovers target mean $\bar{x}$ to $< 10^{-6}$.
2. **Off-policy Betting CS soundness tests:**
   - Log-space numerical stability up to $t = 100{,}000$ samples (no `NaN`/`Inf`).
   - Running intersection property: $L_{\text{fail}, t}$ non-decreasing, $U_{\text{fail}, t}$ non-increasing.
   - **Multi-seed anytime empirical coverage test (CRITICAL):** Run $\ge 200$ independent seeds at $\delta = 0.05$ with **adaptively updated** CE proposals and **data-dependent stopping** (stopping as soon as the interval excludes $0.5p$ or $2p$, or at budget cap). Assert empirical miscoverage rate is $\le \delta$ (allowing standard binomial test tolerance, e.g., $\le 0.08$ at 200 seeds, and zero catastrophic violations).
3. **End-to-end DiSE integration tests:**
   - Unclosable tail benchmark (e.g., `monitor`-style program where $x < 50$ closes symbolically as `SAFE`, and $x \ge 50$ has an unclosable nonlinear/bitwise fault condition). Verify:
     - `x < 50` closes via Z3 (0 AIS samples spent there).
     - AIS activates only in the unclosable `x >= 50` leaf.
     - Reported global `[lo, hi]` contains exact ground-truth $\mu$.
     - Existing tests in `tests/` still pass without regression.

### 4.2 Efficiency Benchmark Script (`scripts/run_ais_experiments.py`)
Run **multi-seed** experiments ($\ge 20$ seeds; report **medians and IQR [25th–75th percentile]**) and save raw JSON/CSV results to `results/ais/`:

1. **Experiment 1 — Verification vs. Falsification Scaling (Leaf-Level):**
   - Sweep true leaf failure probability $p_{\text{fail}} \in \{10^{-1}, 10^{-2}, 10^{-3}, 10^{-4}\}$ on a discrete $BG(p, N)$ unclosable leaf.
   - Compare:
     1. Plain sampling + DiSE `PrPl-EB` (and Plain sampling + Betting CS as control).
     2. AIS with **oracle/ideal** tilted proposal ($\alpha \in \{0.1, 0.5, 0.9\}$).
     3. AIS with **learned adaptive CE** proposal ($\alpha \in \{0.1, 0.5, 0.9\}$).
   - Measure samples needed to certify:
     - **Falsification (Lower Bounds):** $L_{\text{fail}} \ge 0.5 p_{\text{fail}}$ and $L_{\text{fail}} \ge 0.9 p_{\text{fail}}$.
     - **Verification (Upper Bounds):** $U_{\text{fail}} \le 2.0 p_{\text{fail}}$ and $U_{\text{fail}} \le 1.1 p_{\text{fail}}$.
   - Cap budget realistically (e.g., $2 \times 10^5$ or $5 \times 10^5$ per seed so all 20 seeds finish quickly; record right-censored caps clearly).
2. **Experiment 2 — Defensive Mixture $\alpha$ Trade-off:**
   - At fixed $p_{\text{fail}} = 10^{-3}$, sweep $\alpha \in \{0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0\}$ ($\alpha = 1.0$ = plain sampling) across $\ge 20$ seeds.
   - Measure median & IQR samples to certify lower bound ($\ge 0.5 p$), loose upper bound ($\le 2p$), and tight upper bound ($\le 1.2p$).
3. **Experiment 3 — End-to-End DiSE vs. AIS-DiSE vs. Plain MC on Unclosable Programs:**
   - Include at least 2–3 concrete programs with known exact ground-truth $\mu$:
     1. **`monitor_nonlinear`**: dominant safe region ($x < 50$ under $BG(0.1, 100)$) closable by Z3 + rare unclosable quadratic/modulo/bitwise fault in $x \ge 50$.
     2. **`sum_threshold_loop`**: multi-variable $BG$ sum crossing a threshold inside an unclosable/concretized loop body (directly matching Juneja's exponential-twisting regime).
     3. **`sla_violation_check`**: a program violating an SLA threshold $\tau$ where AIS-DiSE falsifies $1 - \mu > \tau$ in tens/hundreds of evaluations while Plain DiSE and Plain MC take thousands+.
   - Compare total program evaluations to reach certified decision (falsification & estimation), final interval width at matched budget, and empirical coverage across $\ge 20$ seeds.

---

## 5. Required Plots (`scripts/plot_ais_results.py` -> `results/ais/plots/`)

Generate clean, high-DPI (300 DPI) matplotlib figures (both `.png` and `.pdf`) with clear titles, axis labels, legends, log scales where appropriate, and shaded IQR bands (or error bars) across seeds:

1. **`fig1_asymmetry_scaling.png`**: Two-panel log-log plot of **Samples to Certify vs. True Failure Rate $p_{\text{fail}}$**:
   - *Left panel:* **Falsification** (certify $p_{\text{fail}} \ge 0.5 p_{\text{true}}$) — shows AIS (both oracle and adaptive CE) staying orders of magnitude below Plain sampling and nearly flat in $1/p$.
   - *Right panel:* **Verification** (certify $p_{\text{fail}} \le 2 p_{\text{true}}$) — shows all methods scaling as $\Omega(1/p)$, with $\alpha = 0.1$ worse than plain and $\alpha = 0.9$ achieving a constant-factor improvement.
2. **`fig2_alpha_tradeoff.png`**: **Samples to Certify vs. Defensive Mixture $\alpha$** at $p_{\text{fail}} = 10^{-3}$:
   - Clearly highlights the opposite optima: small $\alpha$ dominates falsification (lower bounds), whereas large $\alpha \approx 0.8\text{–}0.9$ dominates verification / tight relative-precision upper bounds (and explains why $\alpha = 0.1$ hurts upper bounds).
3. **`fig3_end_to_end_dise.png`**: **End-to-End DiSE Benchmark Comparison**:
   - *Left panel:* Certified failure-rate interval $[1 - U_t, 1 - L_t]$ vs. number of program evaluations on an unclosable-leaf benchmark (Plain MC vs. Baseline DiSE vs. AIS-DiSE), with horizontal dashed lines for exact failure rate $1 - \mu$ and SLA threshold $\tau$.
   - *Right panel:* Bar/box chart comparing median evaluations to certify SLA falsification across the end-to-end benchmarks (log scale).
4. **`fig4_soundness_coverage.png`**: **Empirical Coverage Audit**:
   - Shows empirical miscoverage rate (with binomial confidence bars) across seeds under adaptive stopping for Plain DiSE and AIS-DiSE ($\alpha \in \{0.1, 0.5, 0.9\}$) against the nominal $\delta = 0.05$ dashed threshold—visually certifying zero soundness regressions.

---

## 6. Verification Checklist & Mandatory `changes.md` Report

Before finishing, run the full verification sequence:
1. Run `pytest` on both existing tests and `tests/test_ais.py` — all must pass.
2. Run `scripts/run_ais_experiments.py` and `scripts/plot_ais_results.py` — verify all 4 figures render cleanly and no `NaN`/`Inf` values appear in the CSV/JSON tables.
3. Write a thorough, honest report into `changes.md` containing:
   - **Summary of Architecture & Files Changed:** exact file paths, new classes/functions, and modified lines.
   - **Discrepancies / Bugs Found in Existing Repo:** e.g., exact `BG` support convention found, what `W_open` actually summed, how `K_max` was handled vs. `K_bounds`, and anything reused from `baselines/rare_event.py`.
   - **Mathematical & Engineering Safeguards Implemented:** confirm predictability of CE updates, overflow protection for $r > 1$, log-space wealth stability, and sign convention verification.
   - **Commands Executed & Test Results:** exact CLI commands and pass/fail counts.
   - **Experimental Summary Tables:** copy key median/IQR numbers from Experiments 1–3 and empirical coverage numbers directly into Markdown tables.
   - **Generated Plot Index:** list every figure path with a 1–2 sentence takeaway of what it shows (including any unexpected results or limitations observed).