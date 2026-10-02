# Dynamic Occupational Choice Model — Codebase Guide

**Project:** "Generative AI and Occupational Entry Barriers" (Hosseini & Lichtinger, 2026)

**Purpose:** This document describes every code file, data dependency, and output artifact for the dynamic model extension (Section 5 of the paper). It is written so that another AI or collaborator can understand the full pipeline without reading the original conversation.

---

## 1. Overview

The dynamic model is an infinite-horizon discrete occupational choice model with J=94 occupations (SOC 3-digit). Workers choose occupations each period, facing switching costs and idiosyncratic Type-I Extreme Value taste shocks. AI is modeled as a change in occupation-to-occupation retraining distances d_so, while structural switching cost parameters κ are held fixed.

The key extension over the baseline (scalar κ) is **occupation-specific two-way switching costs**: the cost of moving from occupation s to occupation o is decomposed as:

    c_{so} = (κ_out[s] + κ_in[o]) · d_{so}

where κ_out[s] captures origin-specific barriers to leaving and κ_in[o] captures destination-specific entry barriers. This gives 2J = 188 parameters, calibrated to match 2J moments from observed worker flows (origin outflow rates + destination inflow shares).

---

## 2. Directory Structure

All code lives in:
```
/Dynamic model/2. Dynamic Model/
```

Key subdirectories:
```
sdm_output/                          # All model outputs
├── ai_transition_results.json       # Main results file (transition paths, all specs)
├── ai_counterfactual_ss.json        # Steady-state counterfactual results
├── kappa_heterogeneity_results.json # κ calibration comparison across specs
├── calibrated_parameters_sdm_smm.json # SMM-calibrated baseline (scalar κ, τ)
├── model_results.npz                # Pre-AI model state (wages, distributions)
├── sdm_inequality_evolution.pdf/png # Publication figure: Var(log w) transition
├── top30_inflows_outflows.pdf/png   # Publication figure: top-30 gainers/losers
└── transition_analysis/
    ├── scalar/                      # Analysis outputs for scalar κ spec
    ├── dest/                        # Analysis outputs for destination-only κ spec
    └── twoway/                      # Analysis outputs for two-way κ spec
        ├── top50_inflows.csv
        ├── top50_outflows.csv
        ├── top_pair_flows_post.csv
        ├── top_pair_flow_changes.csv
        ├── heatmap_flows_pre/post/change.png
        ├── top50_inflows_outflows.png
        ├── top_pair_flows.png
        └── summary.md
```

The LaTeX output and paper figures are also copied to:
```
/AI and Expertise/Claude Folder/Dynamic model/
├── dynamic_section_REPLACEMENT.tex  # Full replacement for Section 5
├── main_updated.tex                 # Full paper with Section 5 replaced
├── top30_inflows_outflows.pdf/png
├── sdm_inequality_evolution.pdf/png
```

---

## 3. Code Files — Detailed Descriptions

### 3.1 `simple_dynamic_model.py` — Core Model Engine

**What it does:** Implements the baseline infinite-horizon occupational choice model at the SOC3 level (J=94). This is the foundation that all extensions build on.

**Key functions:**
- `build_model_data(verbose=True)` — Loads and merges all input data (BLS wages, retraining distances, AI exposure, expertise, employment). Returns a dict with fields: `soc3_codes`, `soc3_names`, `wages`, `employment`, `d_matrix` (J×J retraining distances without AI), `d_matrix_ai` (with AI), `college_share`, etc.
- `solve_value_function(wages, p, kappa, d)` — Bellman iteration for scalar κ. Returns value function V (J,) and choice probabilities π (J×J).
- `invert_amenities(wages, p, kappa, d, data_emp)` — BLP-style contraction mapping to recover amenity vector a_o that rationalizes observed employment shares.
- `solve_steady_state(p, kappa, d, data_emp)` — Full steady-state solver: inverts amenities, solves VF, computes stationary distribution.
- `compute_ai_counterfactual(...)` — Computes new steady state after AI shock (d → d_ai).

**Parameters dict `p`:** `{"beta": 0.96, "sigma": 5.0, "tau": <calibrated>}`
- β: discount factor
- σ: CES elasticity of substitution across occupations (σ=5)
- τ: scale of taste shocks (Type-I EV)

**Data dependencies:** Reads from multiple files in the Archive directory. See `paths_override.py` for exact paths.

---

### 3.2 `paths_override.py` — Path Monkey-Patch

**What it does:** Must be imported *before* `simple_dynamic_model` to redirect its hardcoded file paths to the correct locations within the mounted workspace. It patches module-level constants like `sdm._CALIBRATED_JSON`, `sdm._BLS_OES_XLSX`, etc.

**Why it exists:** The original `simple_dynamic_model.py` has paths relative to a different directory structure. Since the workspace mounts at a different location, this override bridges the gap.

**Usage pattern:**
```python
import paths_override  # noqa: F401  — must come first
from simple_dynamic_model import build_model_data
```

---

### 3.3 `flow_data.py` — Occupation-to-Occupation Flow Data

**What it does:** Loads the Schubert/Stansbury/Taska (2024) dataset of occupation-to-occupation worker transitions, aggregates from 6-digit SOC to 3-digit SOC (J=94), and computes the calibration targets.

**Input:** `occupation_transitions_public_data_set.dta` (Stata file, located in the Claude Folder)

**Key function:** `load_flow_data(soc3_codes, employment)` → returns dict with:
- `outflow_rate_data` (J,) — fraction of workers leaving each origin each period. Employment-weighted mean normalized to 0.12 (matching CPS aggregate switching rate). One occupation (19-5, Occupational Health and Safety Specialists) has zero observed switchers → set to NaN.
- `inflow_share_switch_data` (J,) — each destination's share of total switching inflows
- `pi_switch_data` (J×J) — full transition matrix (conditional on switching)
- `valid_origin_mask` (J,) — boolean, False for origins with zero switch mass (only 19-5)

**Aggregation details:**
- Drops within-SOC3 flows (workers moving between 6-digit occupations within the same 3-digit group, ~8% of switch mass)
- Maps 6-digit SOC codes to 3-digit by taking first 4 characters (e.g., "29-1141" → "29-1")

---

### 3.4 `sdm_kappa_vec.py` — Extended Model with Vector κ

**What it does:** This is the main extension module. Generalizes the model to support three κ specifications: scalar, destination-only, and two-way. Contains all the machinery for calibration, steady-state solving, and flow moment computation with heterogeneous switching costs.

**Key classes:**
- `KappaSpec(kind, J, kappa=None, kappa_out=None, kappa_in=None)` — Container for κ parameters. `kind` ∈ {"scalar", "dest", "twoway"}.

**Key functions:**

- `build_cost_matrix(kspec, d)` → J×J matrix C where:
  - scalar: C[s,o] = κ · d[s,o]
  - dest: C[s,o] = κ_in[o] · d[s,o]
  - twoway: C[s,o] = (κ_out[s] + κ_in[o]) · d[s,o]

- `solve_vf_vec(wages, p, kspec, d, ...)` — Bellman iteration with vector κ. Returns V (J,), policy π (J×J).

- `invert_amenities_vec(wages, p, kspec, d, data_emp, ...)` — BLP contraction to recover amenities a_o, with the (1−β) dynamic correction for the infinite-horizon setting.

- `solve_steady_state_vec(p, kspec, d, data_emp, ...)` — Full steady-state solver for any κ spec.

- `compute_flow_moments(policy, mu, data_emp)` — From policy π and distribution μ, computes model-implied outflow rates and inflow shares. These are the moments matched in calibration.

- `calibrate_dest(p, d, data_emp, data_flow, ...)` — Calibrates destination-only κ_in[o] to match inflow shares. Just-identified (J params, J moments). Simple fixed-point iteration.

- `calibrate_twoway_stable(p, d, data_emp, data_flow, ...)` — Calibrates two-way κ = (κ_out, κ_in) to jointly match outflow rates AND inflow shares. Just-identified (2J params, 2J moments). Uses:
  - **Gauss-Seidel alternation:** Update κ_in (holding κ_out fixed), re-solve, then update κ_out (holding κ_in fixed)
  - **Adaptive EMA averaging:** Exponential moving average kicks in when flip fraction > 30%, to kill period-2 oscillation
  - **Adaptive step shrinking:** Reduces step size if max deviation stops improving

**Calibration logic (two-way):**
1. Initialize κ_out = κ_in = scalar κ / 2
2. Outer loop (up to 120 iterations):
   a. Solve steady state with current (κ_out, κ_in)
   b. Compute model flow moments
   c. Update κ_in: κ_in_new[o] = κ_in[o] · (target_inflow_share[o] / model_inflow_share[o])
   d. Re-solve steady state
   e. Update κ_out: κ_out_new[s] = κ_out[s] · (target_outflow_rate[s] / model_outflow_rate[s])
   f. Apply EMA if oscillation detected
   g. Check convergence (max_dev < 0.005)

---

### 3.5 `calibrate_sdm_smm.py` — SMM Baseline Calibration

**What it does:** Simulated Method of Moments estimation of the two baseline parameters (scalar κ, τ) to match: (1) variance of log wages, and (2) aggregate occupational mobility rate (12%). This provides the starting point for the heterogeneous-κ extensions.

**Output:** `sdm_output/calibrated_parameters_sdm_smm.json`

---

### 3.6 `compare_kappa_specs.py` — Compare All Three κ Specifications

**What it does:** Runs the scalar, destination-only, and two-way calibrations side by side and saves comprehensive comparison results.

**Output:** `sdm_output/kappa_heterogeneity_results.json` containing:
- All κ vectors (κ_out, κ_in for each spec)
- Calibration convergence history
- Flow moment fit statistics (correlations, RMSE between model and data moments)
- Steady-state wages, employment shares, amenities for each spec

**Key finding:** Scalar κ cannot match flow heterogeneity at all (outflow rate correlation ≈ −0.11). Two-way κ matches both outflow rates and inflow shares well (correlations > 0.99).

---

### 3.7 `ai_counterfactual_ss.py` — AI Steady-State Counterfactual

**What it does:** Computes the post-AI long-run steady state for all three κ specs. The AI shock changes retraining distances d_so → d_so_AI (from LLM-based estimates of how AI reduces retraining costs). Structural parameters (κ, τ, amenities a_o, productivity B_o) are held fixed.

**Mechanism:** Lower d means lower switching costs → more mobility → labor reallocation → wages adjust via inverse labor demand w_o = B_o · L_o^{−1/σ}.

**Output:** `sdm_output/ai_counterfactual_ss.json`

---

### 3.8 `ai_transition_path.py` — Forward-Looking Transition Dynamics

**What it does:** Solves the full forward-looking transition path from pre-AI to post-AI steady state over T=100 periods. This is NOT a myopic solver — it uses backward induction on value functions.

**Algorithm (outer loop iterates until wage path converges):**
1. **Backward sweep (t = T → 1):** Given wage path {w_t}, solve Bellman equations backward from terminal V_post to get V_t and policy π_t at each period.
2. **Forward sweep (t = 1 → T):** Starting from μ_pre, apply policy π_t to get μ_{t+1} at each period.
3. **Wage update:** From μ_t, compute w_t = B · L_t^{−1/σ} via inverse labor demand. Dampen update.
4. Repeat until wage path converges.

**Output:** `sdm_output/ai_transition_results.json` (3.6 MB) containing:
- `by_spec.{scalar,dest,twoway}`: Full transition data per spec
  - `mu_pre`, `mu_post` — pre/post steady-state employment shares
  - `wages_pre`, `wages_post` — pre/post wages
  - `var_log_w_path` — Var(log w) at each period along the transition
  - `policy_pre`, `policy_post` — J×J choice probability matrices
  - `p90_p50_pre/post`, `p90_p10_pre/post` — wage percentile gaps
- `summary` — list of dicts with key statistics per spec

---

### 3.9 `analyze_transition.py` — Detailed Transition Analysis

**What it does:** Post-processes transition results to produce detailed CSVs and exploratory figures for each κ spec.

**Outputs (per spec, in `sdm_output/transition_analysis/{spec}/`):**
- `top50_inflows.csv` — 50 occupations with largest employment gains (ΔL, Δlog w)
- `top50_outflows.csv` — 50 occupations with largest employment losses
- `top_pair_flows_post.csv` — largest occupation-to-occupation flow pairs post-AI
- `top_pair_flow_changes.csv` — pairs with biggest flow changes (post − pre)
- `heatmap_flows_pre/post/change.png` — J×J flow heatmaps
- `top50_inflows_outflows.png` — bar chart of top-50 gainers/losers
- `top_pair_flows.png` — bar chart of top pair flows
- `summary.md` — text summary of key results

---

### 3.10 `plot_top30_inflows_outflows.py` — Publication Figure: Winners & Losers

**What it does:** Generates a publication-quality two-panel figure showing the top 30 occupations with the largest net employment inflows (left, green) and outflows (right, red). Each bar is labeled with SOC3 code + occupation name, and annotated with the corresponding Δlog w (wage change in %).

**Style:** LaTeX-like serif fonts (DejaVu Serif + Computer Modern mathtext), clean spines, thin gridlines.

**Output:** `sdm_output/top30_inflows_outflows.pdf` and `.png`

**Referenced in paper as:** Figure showing which occupations gain and lose workers due to AI-induced barrier reductions.

---

### 3.11 `plot_inequality_evolution.py` — Publication Figure: Inequality Transition

**What it does:** Plots the transition path of Var(log w) (between-occupation wage inequality) over 30 years after the AI shock. Includes horizontal lines for pre/post steady states, vertical markers for 50% and 80% adjustment points, and an annotation box with the total long-run change.

**Style:** LaTeX-like serif fonts matching the top-30 figure.

**Output:** `sdm_output/sdm_inequality_evolution.pdf` and `.png`

**Key results shown:** Half-life ≈ 5 years, 80% adjustment by year 9. Total Var(log w) falls by 8.7% (from 0.208 to 0.190).

---

## 4. Data Dependencies

| File | Source | Used by |
|------|--------|---------|
| `occupation_transitions_public_data_set.dta` | Schubert, Stansbury & Taska (2024) | `flow_data.py` |
| `calibrated_parameters.json` | OLG model calibration (Archive) | `simple_dynamic_model.py` |
| `national_M2024_dl.xlsx` | BLS OES wage data | `simple_dynamic_model.py` |
| `expertise_by_soc3.csv` | Education/expertise data | `simple_dynamic_model.py` |
| `Final_Occupation_Dataset.xlsx` | LLM occupation characteristics | `simple_dynamic_model.py` |
| `occ2occ_retraining_merged_without_ai.csv` | Retraining distances (no AI) | `simple_dynamic_model.py` |
| `occ2occ_retraining_merged_with_ai.csv` | Retraining distances (with AI) | `simple_dynamic_model.py` |
| `college_share_by_soc3.csv` | College share by occupation | `simple_dynamic_model.py` |

---

## 5. Execution Order (Full Pipeline)

To reproduce all results from scratch:

```bash
cd "Dynamic model/2. Dynamic Model"

# Step 1: Baseline calibration (scalar κ, τ)
python calibrate_sdm_smm.py

# Step 2: Calibrate all three κ specs and compare
python compare_kappa_specs.py

# Step 3: AI steady-state counterfactual
python ai_counterfactual_ss.py

# Step 4: Forward-looking transition dynamics (T=100)
python ai_transition_path.py

# Step 5: Detailed analysis (CSVs, exploratory figures)
python analyze_transition.py

# Step 6: Publication figures
python plot_top30_inflows_outflows.py
python plot_inequality_evolution.py
```

**Runtime notes:**
- Steps 2–4 each take 1–5 minutes depending on iteration counts
- Step 4 (transition path) is the most computationally intensive
- All scripts require `paths_override.py` to be importable (same directory)

---

## 6. LaTeX Integration

The paper section is in `dynamic_section_REPLACEMENT.tex`. It replaces the entire Section 5 of the paper. Key LaTeX elements:

- **Figures referenced:**
  - `top30_inflows_outflows.pdf` — Top 30 inflows/outflows bar chart
  - `sdm_inequality_evolution.pdf` — Var(log w) transition path

- **Tables:** One table ("Dynamic AI Counterfactual") with Var(log w), p90−p50, and p90−p10 gaps, pre vs. post AI.

- **BibTeX entry needed:** `schubert2024entrylevel` for the flow data citation.

---

## 7. Key Model Parameters and Results

| Parameter | Value | Source |
|-----------|-------|--------|
| β (discount factor) | 0.96 | Standard |
| σ (CES elasticity) | 5.0 | Literature |
| τ (taste shock scale) | SMM-calibrated | `calibrate_sdm_smm.py` |
| J (occupations) | 94 | SOC 3-digit |
| T (transition horizon) | 100 periods | `ai_transition_path.py` |
| Aggregate mobility rate | 12% | CPS target |

**Headline results (two-way κ):**
- Var(log w): 0.208 → 0.190 (−8.7%)
- Half-life of inequality adjustment: ~5 years
- 80% adjustment: ~9 years
- Top gainers: healthcare support, personal care, food service
- Top losers: legal, financial, management occupations
