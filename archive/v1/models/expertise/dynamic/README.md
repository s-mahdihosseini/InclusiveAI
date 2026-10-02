# Dynamic Occupational Choice Model — Code README

**Paper:** Hosseini & Lichtinger (2026), "Generative AI and Occupational Entry Barriers"  
**Section:** 5 (Dynamic Extension: Occupational Switching Costs and Transition Dynamics)

---

## Quick start

```bash
cd "Dynamic model/2. Dynamic Model"
python reproduce_figures.py          # regenerates both paper figures from saved results
```

To rerun the full pipeline from scratch (takes ~10 minutes total):

```bash
python calibrate_sdm_smm.py         # Step 1
python compare_kappa_specs.py        # Step 2
python ai_counterfactual_ss.py       # Step 3
python ai_transition_path.py         # Step 4
python reproduce_figures.py          # Step 5
```

---

## Minimal file set to reproduce the paper results

To reproduce everything in Section 5 from scratch (calibration → solving → figures), you need exactly these 9 code files plus the data:

```
REQUIRED CODE (run in this order)
─────────────────────────────────
paths_override.py            ← path redirect (import only, not run directly)
simple_dynamic_model.py      ← core model engine (import only)
flow_data.py                 ← worker flow data loader (import only)
sdm_kappa_vec.py             ← vector-κ extension + calibrators (import only)

1. calibrate_sdm_smm.py     ← calibrates scalar κ and τ  →  calibrated_parameters_sdm_smm.json
2. compare_kappa_specs.py    ← calibrates twoway κ_out, κ_in  →  kappa_heterogeneity_results.json
3. ai_counterfactual_ss.py   ← pre/post AI steady states  →  ai_counterfactual_ss.json
4. ai_transition_path.py     ← transition dynamics T=100  →  ai_transition_results.json
5. reproduce_figures.py      ← paper figures  →  sdm_inequality_evolution.pdf, top30_inflows_outflows.pdf

REQUIRED DATA (read by simple_dynamic_model.py via paths_override.py)
─────────────────────────────────────────────────────────────────────
calibrated_parameters.json                        β, σ from OLG model
national_M2024_dl.xlsx                            BLS OES wages
occ2occ_retraining_merged_without_ai.csv          retraining distances (no AI)
occ2occ_retraining_merged_with_ai.csv             retraining distances (with AI)
expertise_by_soc3.csv                             education data
college_share_by_soc3.csv                         college shares
Final_Occupation_Dataset.xlsx                     LLM occupation characteristics
occupation_transitions_public_data_set.dta        worker flows (Schubert/Stansbury/Taska 2024)

NOT NEEDED for reproduction (exploration / robustness / legacy)
──────────────────────────────────────────────────────────────
analyze_transition.py          exploratory CSVs and heatmaps
run_option_b.py                robustness check (alternative moment definition)
plot_inequality_evolution.py   superseded by reproduce_figures.py
plot_top30_inflows_outflows.py superseded by reproduce_figures.py
```

---

## File-by-file descriptions

### 1. `simple_dynamic_model.py` — Core model engine

The foundation everything else builds on. Implements the infinite-horizon discrete occupational choice model at the SOC3 level (J=94 occupations).

**Model:** Each period, a worker in occupation s draws iid Type-I Extreme Value taste shocks for all J occupations and chooses the one maximizing:

    log w_o + a_o − κ · d_{so} · 1{o ≠ s} + β V(o) + τ ε_o

The value function satisfies the closed-form Bellman equation:

    V(s) = τ · log Σ_o exp[(log w_o + a_o − κ d_{so} 1{o≠s} + β V(o)) / τ]

Wages clear via CES inverse labor demand: w_o = B_o · L_o^{−1/σ}.

**Key functions:**

- `build_model_data()` — Loads and merges all input datasets (BLS wages, LLM-based retraining distances d_so with and without AI, employment, education shares). Returns a dict with everything needed to run the model.
- `SdmParams(data, tier3)` — Parameter container. `tier3` is a dict with `kappa` and `tau`. Pulls β=0.95, σ=5.0 from `calibrated_parameters.json`.
- `solve_vf(wages, p, kappa, d)` — Iterates the Bellman equation to convergence. Returns value function V (J,).
- `invert_amenities(wages, p, kappa, d, data_emp)` — BLP-style contraction mapping. Finds the amenity vector a_o such that the model's stationary employment distribution matches data. Uses the (1−β) dynamic correction: a_o ← a_o + (1−β) · (log μ_data − log μ_model).
- `invert_B(wages, L_eff, sigma)` — Recovers productivity shifters B_o from observed wages and employment via B_o = w_o · L_o^{1/σ}.
- `solve_steady_state(p, kappa, d, ...)` — Full steady-state solver: inverts amenities, inverts B, then iterates wages to a fixed point.
- `compute_stationary_distribution(V, p, d, ...)` — Given V, computes the choice probabilities π(o|s) and finds the stationary distribution μ satisfying μ = π' μ.

**Data files it reads** (paths resolved by `paths_override.py`):

- `calibrated_parameters.json` — β, σ from the OLG model
- `national_M2024_dl.xlsx` — BLS OES wages
- `occ2occ_retraining_merged_{without,with}_ai.csv` — Retraining distance matrices (d_so before and after AI)
- `expertise_by_soc3.csv`, `college_share_by_soc3.csv` — Education data
- `Final_Occupation_Dataset.xlsx` — LLM occupation characteristics

---

### 2. `paths_override.py` — Path redirect

Monkey-patches the hardcoded file paths in `simple_dynamic_model.py` to match the current workspace mount. **Must be imported before** `simple_dynamic_model` in every script.

**⚠ Known issue:** Line 16 contains a session-specific path. Edit it to match your workspace mount before running on a new machine.

---

### 3. `flow_data.py` — Worker flow data loader

Loads the Schubert/Stansbury/Taska (2024) dataset of occupation-to-occupation worker transitions and computes the calibration targets for the vector-κ specifications.

**Input:** `occupation_transitions_public_data_set.dta` — Stata file with 6-digit SOC origin-destination pairs, each with `transition_share` (conditional on switching) and `total_obs` (switchers from that origin).

**Processing:**

1. Aggregates 6-digit SOC → SOC3 (first 4 characters, e.g., "29-1141" → "29-1")
2. Drops within-SOC3 flows (~8% of switch mass — these are "staying" at the 3-digit level)
3. Builds the J×J count matrix N[s,o] and normalizes rows to get π(o|s,switch)

**Outputs** (returned as a dict):

- `outflow_rate_data[s]` (J,) — Fraction of workers leaving origin s per year. Rescaled so the employment-weighted mean = 0.12 (CPS aggregate). One occupation (19-5, OSH Specialists) has zero observed switchers → set to NaN.
- `inflow_share_switch_data[o]` (J,) — Destination o's share of aggregate switching, computed as an employment-weighted average of conditional destination probabilities.
- `pi_switch_data[s,o]` (J×J) — Full conditional-on-switching transition matrix.
- `valid_origin_mask[s]` (J,) — False for origins with zero switch mass.

---

### 4. `calibrate_sdm_smm.py` — Baseline calibration (scalar κ, τ)

Calibrates the two scalar parameters (κ, τ) via Simulated Method of Moments.

**Targets (2 moments, 2 parameters — just identified):**

1. Var(log occupational wage) = 0.208 (BLS OEWS)
2. Aggregate 1-year occupational mobility rate = 0.12 (CPS)

**Method:** Nelder-Mead simplex minimization of weighted squared percentage deviations. Each objective evaluation: inverts amenities → inverts B → solves steady state → computes model moments. ~150 function evaluations.

**Intuition:**
- τ controls the scale of taste shocks — higher τ means more switching (more randomness in choices), which raises mobility and compresses wage variance.
- κ controls the level of switching costs — higher κ means less switching and more wage dispersion (workers get stuck in bad occupations).

**Output:** `sdm_output/calibrated_parameters_sdm_smm.json` with the calibrated (κ, τ).

---

### 5. `sdm_kappa_vec.py` — Vector switching cost extension

The main model extension. Generalizes switching costs from a single scalar to occupation-specific vectors. Three specifications:

| Spec | Cost formula | Parameters | Moments matched |
|------|-------------|------------|-----------------|
| scalar | c_{so} = κ · d_{so} | 1 (κ) | Var(log w) + aggregate mobility |
| dest | c_{so} = κ_in[o] · d_{so} | J (κ_in) | Destination inflow shares |
| twoway | c_{so} = (κ_out[s] + κ_in[o]) · d_{so} | 2J (κ_out + κ_in) | Outflow rates + inflow shares |

**Key classes and functions:**

- `KappaSpec(kind, J, ...)` — Container. `kind` ∈ {"scalar", "dest", "twoway"}. The `.multiplier()` method returns the J×J matrix K where cost[s,o] = K[s,o] · d[s,o].

- `solve_vf_vec(wages, p, kspec, d)` — Same Bellman iteration as in the base model but with the full J×J cost matrix instead of scalar κ.

- `invert_amenities_vec(wages, p, kspec, d, data_emp)` — Same BLP contraction as base model, adapted for vector κ.

- `solve_steady_state_vec(p, kspec, d, data_emp, ...)` — Full steady-state solver: amenity inversion → B inversion → wage fixed point. Used by all calibrators and counterfactual scripts.

- `compute_flow_moments(policy, mu, data_emp)` — From the choice probability matrix π(o|s) and stationary distribution μ, computes the model counterparts of the data flow moments: outflow rate per origin (= 1 − π(s|s)), destination inflow share (employment-weighted average of π(o|s,switch)).

- `calibrate_twoway_stable(p, d, data_emp, data_flow, ...)` — The main calibrator for the paper's specification. Calibrates 2J = 188 parameters to match 2J moments. Algorithm:
  1. Initialize κ_out = κ_in = scalar_κ / 2
  2. For each iteration:
     - **Gauss-Seidel half-step 1:** Solve steady state, compute model inflow shares, update κ_in via multiplicative ratio rule: κ_in[o] ← κ_in[o] · (model_inflow[o] / data_inflow[o])
     - **Re-solve** steady state with updated κ_in
     - **Gauss-Seidel half-step 2:** Compute model outflow rates, update κ_out: κ_out[s] ← κ_out[s] · (model_outflow[s] / data_outflow[s])
     - **EMA smoothing:** If >30% of coordinates flipped sign relative to previous update, average over last 4 iterates (Krasnoselskii-Mann averaging — kills period-2 oscillation)
     - **Adaptive step shrinking:** If total deviation hasn't improved in 2 iterations, multiply step sizes by 0.7
  3. Convergence when max |log(model/data)| < 0.005 for both moments

---

### 6. `compare_kappa_specs.py` — Run and compare all three κ specs

Runs the scalar, destination-only, and two-way calibrations side by side and saves comprehensive results.

**What it does:**
1. Loads scalar (κ, τ) from `calibrated_parameters_sdm_smm.json`
2. Runs `calibrate_dest()` for destination-only κ_in
3. Runs `calibrate_twoway_stable()` for the full two-way specification
4. For each spec: solves steady state, computes flow moments, compares to data

**Output:** `sdm_output/kappa_heterogeneity_results.json` containing:
- All κ vectors (κ_out, κ_in for each spec)
- Calibration convergence history
- Flow moment fit (correlations, RMSE)
- Steady-state wages, employment shares

**Key finding:** Scalar κ cannot match flow heterogeneity (outflow rate correlation ≈ −0.11). Two-way κ matches both moments well (correlations > 0.99).

---

### 7. `ai_counterfactual_ss.py` — AI steady-state counterfactual

Computes the long-run (new steady state) effect of AI on wages and employment.

**The AI shock:** Retraining distances change from d_so to d_so_AI (AI makes retraining cheaper between most occupation pairs). Structural parameters (κ vectors, τ, amenities a_o, productivity B_o) are **held fixed** — the assumption is that AI changes retraining technology, not the occupation-specific barriers themselves.

**Algorithm:**
1. Solve the pre-AI steady state with full amenity + B inversion (reproduces data)
2. Freeze a_o and B_o at pre-AI values
3. Solve a new steady state under d_post with the same κ, but let wages and employment adjust
4. Compare pre vs. post: Var(log w), Gini, percentile gaps, output

**Output:** `sdm_output/ai_counterfactual_ss.json` with pre/post wages, employment shares, and inequality statistics for all three κ specs.

---

### 8. `ai_transition_path.py` — Forward-looking transition dynamics

Solves the transition path from the pre-AI to post-AI steady state over T=100 periods. This is the most computationally intensive script (~2–5 minutes).

**Why forward-looking matters:** A myopic solver treats each period's wages as permanent and iterates the value function to convergence at each t. This ignores the fact that workers know wages will keep changing. The proper solution has workers discount future wage paths through the Bellman equation.

**Algorithm (outer loop iterates until wage path converges):**

1. **Initialize** wage path by linear interpolation: w_pre → w_post over first 50 periods
2. **Backward induction** (t = T → 1): Given the wage path, solve V_t backward from the terminal V_post:
   - v_all[s,o] = (log w_{o,t} + a_o − cost[s,o] + β V_{t+1}(o)) / τ
   - V_t(s) = τ · logsumexp_o v_all[s,o]
   - π_t(o|s) = softmax_o v_all[s,o]
3. **Forward sweep** (t = 1 → T): Starting from μ_pre, apply π_t to get μ_{t+1} = π_t' μ_t
4. **Wage update:** From μ_t, compute implied wages w_t = B · L_t^{−1/σ}
5. **Damped update** of wage path (adaptive damping: shrink if diverging, expand if converging)
6. Repeat until max |Δw/w| < 5×10⁻⁴

**Output:** `sdm_output/ai_transition_results.json` (3.6 MB) containing:
- Full transition paths: wages, employment shares, Var(log w), Gini, mobility, output — at each of 100 periods
- Pre/post steady-state values and policies
- Summary statistics (half-lives, percentile gaps)

---

### 9. `reproduce_figures.py` — Regenerate paper figures

**The script to run if you just want the figures.** Reads the saved `ai_transition_results.json` and model data, produces both paper figures. No model solving, no calibration — pure plotting. Takes ~20 seconds.

**Figure 1 — Inequality IRF** (`sdm_inequality_evolution.pdf`):
Var(log w) over 30 years after the AI shock, with markers at 50% adjustment (year 5) and 80% adjustment (year 9). Pre/post steady-state horizontal lines.

**Figure 2 — Top-30 inflows/outflows** (`top30_inflows_outflows.pdf`):
Two-panel bar chart. Left: 30 occupations gaining the most workers (green). Right: 30 losing the most (red). Each bar annotated with the Δlog w (% wage change).

---

### 10. `analyze_transition.py` — Detailed transition analysis

Produces detailed CSVs and exploratory figures for each κ spec. Output goes to `sdm_output/transition_analysis/{scalar,dest,twoway}/`.

**Outputs per spec:**
- `top50_inflows.csv`, `top50_outflows.csv` — ranked occupations by ΔL
- `top_pair_flows_post.csv` — largest occupation-to-occupation flow pairs after AI
- `top_pair_flow_changes.csv` — pairs with biggest flow changes
- `heatmap_flows_pre/post/change.png` — J×J flow heatmaps
- Various bar chart PNGs
- `summary.md` — text summary

---

### 11. `run_option_b.py` — Robustness check (alternative moment definition)

Tests sensitivity of results to the definition of the inflow-share calibration moment.

**Option A (baseline):** Inflow share = employment-weighted average of π(o|s,switch). Asks: "if a random worker switched, where would they go?"

**Option B (robustness):** Inflow share = switcher-mass-weighted (μ_s × outflow_rate_s). Asks: "of workers who actually switch, what fraction land in o?"

Recalibrates κ under Option B, reruns the full counterfactual and transition, and generates comparison figures. All output in `sdm_output/option_b/`. Nothing in the main pipeline is overwritten.

**Result:** Qualitatively robust. Top-30 lists overlap 26–27/30, inequality path same shape (−8.7% vs −11.0%), κ_in correlation 0.90.

---

### 12. `plot_inequality_evolution.py`, `plot_top30_inflows_outflows.py` — Legacy standalone plotters

Older standalone versions of the two figures. `reproduce_figures.py` supersedes both. Kept for backward compatibility.

---

## Output directory: `sdm_output/`

| File | Produced by | Used by |
|------|-------------|---------|
| `calibrated_parameters_sdm_smm.json` | `calibrate_sdm_smm.py` | All downstream (provides scalar κ, τ) |
| `kappa_heterogeneity_results.json` | `compare_kappa_specs.py` | `ai_counterfactual_ss.py`, `ai_transition_path.py` |
| `ai_counterfactual_ss.json` | `ai_counterfactual_ss.py` | Paper table |
| `ai_transition_results.json` | `ai_transition_path.py` | `reproduce_figures.py`, `analyze_transition.py` |
| `sdm_inequality_evolution.pdf` | `reproduce_figures.py` | Paper figure |
| `top30_inflows_outflows.pdf` | `reproduce_figures.py` | Paper figure |
| `transition_analysis/` | `analyze_transition.py` | Exploration |
| `option_b/` | `run_option_b.py` | Robustness |

---

## Key parameters

| Symbol | Value | Source |
|--------|-------|--------|
| β | 0.95 | `calibrated_parameters.json` |
| σ | 5.0 | `calibrated_parameters.json` |
| τ | SMM-calibrated | `calibrate_sdm_smm.py` |
| J | 94 | SOC 3-digit |
| T | 100 periods | `ai_transition_path.py` |
| Aggregate mobility target | 0.12 | CPS |

---

## Data dependencies

| File | Source |
|------|--------|
| `occupation_transitions_public_data_set.dta` | Schubert, Stansbury & Taska (2024) |
| `calibrated_parameters.json` | OLG model calibration (Archive/) |
| `national_M2024_dl.xlsx` | BLS OES 2024 |
| `occ2occ_retraining_merged_{without,with}_ai.csv` | LLM-based retraining distance estimates |
| `expertise_by_soc3.csv`, `college_share_by_soc3.csv` | Education data |
| `Final_Occupation_Dataset.xlsx` | LLM occupation characteristics |

All data paths are resolved via `paths_override.py`.

---

## Known issues

1. **`paths_override.py` line 16** is hardcoded to one workspace session. Edit it to match your mount point before running on a new machine.
2. **Stale comment** in `simple_dynamic_model.py` line 213: says `# 4.0` next to sigma, but the loaded value is 5.0 (from JSON). The comment is wrong; the code is correct.
3. **`ai_transition_results.json` `mob_pre` field** reports mobility at t=1 of the transition (after the AI shock hits), not the true pre-AI steady-state mobility. Does not affect any paper figure or table.
