"""
==============================================================================
Simple Dynamic Model — Infinite-Horizon Occupational Choice (No Ability)
==============================================================================
Dynamic extension of the static occupational choice model.  Workers are
infinitely lived, forward-looking agents choosing among J = 94 occupations
(3-digit SOC).  Each period an incumbent in occupation s draws i.i.d. T1EV
taste shocks and may switch to a new occupation o, paying switching cost
κ · d_{s,o}.  There is NO entry/exit — the stationary distribution is the
ergodic distribution of the incumbent Markov chain.

Key feature: NO worker heterogeneity (no ability θ).  All workers in
occupation o earn w_o.  Inequality is purely between-occupation, matching
the static model framework.  This makes the dynamic model a clean
extension of the static model, adding only switching frictions and dynamics.

State space:  occupation s  —  J-dimensional
Value function: V(s) is a 1D array

Authors: Seyed M Hosseini, Guy Lichtinger
Date:    March 2026

Usage:   python simple_dynamic_model.py
Deps:    numpy, pandas, openpyxl, matplotlib
==============================================================================
"""

import numpy as np
import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.dirname(THIS_DIR)   # "OLG - External Calibration"

OUTPUT_DIR = os.path.join(THIS_DIR, "sdm_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data file locations (all relative to OLG - External Calibration)
_OLG_MODEL_DIR   = os.path.join(BASE_DIR, "OLG Model")
_CALIBRATED_JSON = os.path.join(_OLG_MODEL_DIR, "calibrated_parameters.json")
_BLS_OES_XLSX    = os.path.join(_OLG_MODEL_DIR, "national_M2024_dl.xlsx")
_EXPERTISE_CSV   = os.path.join(BASE_DIR, "prepare education data",
                                "expertise_by_soc3.csv")
_LLM_DIR         = os.path.join(BASE_DIR, "LLM Output and Old Datasets")
_FINAL_OCC_XLSX  = os.path.join(_LLM_DIR, "Final_Occupation_Dataset.xlsx")
_RETRAIN_NO_AI   = os.path.join(_LLM_DIR,
                                "occ2occ_retraining_merged_without_ai.csv")
_RETRAIN_WITH_AI = os.path.join(_LLM_DIR,
                                "occ2occ_retraining_merged_with_ai.csv")
_COLLEGE_CSV     = os.path.join(BASE_DIR,
                                "Create education by occupation data",
                                "college_share_by_soc3.csv")


# ===========================================================================
# SECTION 0 — SELF-CONTAINED DATA LOADING
# ===========================================================================

def _load_retraining_matrix_soc3(csv_path, soc3_codes):
    """
    Load the LLM-generated occ-to-occ retraining CSV and aggregate to a
    J×J matrix at the 3-digit SOC level (simple mean across 6-digit pairs).
    Returns months (not years).  Diagonal set to 0.
    """
    import pandas as pd
    df = pd.read_csv(csv_path, usecols=["origin_code", "target_code",
                                         "retraining_months"])
    df["orig_soc3"] = (df["origin_code"].astype(str)
                       .str.replace(".00", "", regex=False).str[:4])
    df["targ_soc3"] = (df["target_code"].astype(str)
                       .str.replace(".00", "", regex=False).str[:4])

    agg = (df.groupby(["orig_soc3", "targ_soc3"])["retraining_months"]
             .mean().reset_index())

    code2idx = {c: i for i, c in enumerate(soc3_codes)}
    J = len(soc3_codes)
    mat = np.full((J, J), np.nan)

    for _, row in agg.iterrows():
        i = code2idx.get(row["orig_soc3"])
        j = code2idx.get(row["targ_soc3"])
        if i is not None and j is not None:
            mat[i, j] = row["retraining_months"]

    # Fill remaining NaN with row mean (rare edge cases)
    for i in range(J):
        row_vals = mat[i, :]
        valid = row_vals[~np.isnan(row_vals)]
        if len(valid) > 0:
            mat[i, np.isnan(mat[i, :])] = valid.mean()
        else:
            mat[i, :] = 0.0

    np.fill_diagonal(mat, 0.0)
    return mat   # in months


def build_model_data(verbose=True):
    """
    Self-contained data loader for the Simple Dynamic Model.

    Loads all data from files in the OLG - External Calibration tree,
    with no dependency on olg_model_3dig.py.

    Returns dict with keys:
        soc3_codes, soc3_names, J,
        mean_wage, employment,
        e_o, e_o_AI,
        college_share,
        d_so, d_so_AI,
        ppg,
        beta, sigma
    """
    import pandas as pd
    t0 = time.time()
    if verbose:
        print("[DATA] Loading all data sources...")

    # ---- Calibrated parameters (Tier 1 + SOC3 code list) ----
    with open(_CALIBRATED_JSON) as f:
        cal = json.load(f)
    soc3_codes = cal["soc3"]["codes"]
    soc3_names = cal["soc3"]["names"]
    J = len(soc3_codes)
    code2idx = {c: i for i, c in enumerate(soc3_codes)}

    # ---- BLS OEWS wages & employment ----
    df = pd.read_excel(_BLS_OES_XLSX)
    detailed = df[df["O_GROUP"] == "detailed"].copy()
    detailed["soc3"] = detailed["OCC_CODE"].str[:4]
    detailed["TOT_EMP"] = pd.to_numeric(detailed["TOT_EMP"], errors="coerce")
    detailed["A_MEAN"]  = pd.to_numeric(detailed["A_MEAN"],  errors="coerce")
    detailed = detailed.dropna(subset=["TOT_EMP", "A_MEAN"])

    bls = detailed.groupby("soc3").apply(
        lambda g: pd.Series({
            "tot_emp":   g["TOT_EMP"].sum(),
            "mean_wage": (g["A_MEAN"] * g["TOT_EMP"]).sum() / g["TOT_EMP"].sum(),
        }),
        include_groups=False,
    ).reset_index()

    mean_wage  = np.zeros(J)
    employment = np.zeros(J)
    for _, row in bls.iterrows():
        idx = code2idx.get(row["soc3"])
        if idx is not None:
            mean_wage[idx]  = row["mean_wage"]
            employment[idx] = row["tot_emp"]

    # Fill zero-employment / zero-wage placeholders
    zero_emp = employment == 0
    if zero_emp.any():
        min_emp = employment[employment > 0].min()
        employment[zero_emp] = min_emp * 0.01
    zero_wage = mean_wage == 0
    if zero_wage.any():
        med_wage = np.median(mean_wage[mean_wage > 0])
        mean_wage[zero_wage] = med_wage

    # ---- LLM expertise: e_o and e_o^AI (years) ----
    exp_df = pd.read_csv(_EXPERTISE_CSV)
    exp_map_eo   = dict(zip(exp_df["soc3"], exp_df["e_o_years"]))
    exp_map_eoAI = dict(zip(exp_df["soc3"], exp_df["e_o_AI_years"]))

    e_o_json = cal["soc3"]["e_o"]
    e_o    = np.array([exp_map_eo.get(c, e_o_json[i])
                       for i, c in enumerate(soc3_codes)])
    e_o_AI = np.array([exp_map_eoAI.get(c, e_o[i] * 0.7)
                       for i, c in enumerate(soc3_codes)])
    e_o_AI = np.minimum(e_o_AI, e_o)

    # ---- College share ----
    college_share = np.full(J, 0.5)
    if os.path.exists(_COLLEGE_CSV):
        cs_df = pd.read_csv(_COLLEGE_CSV)
        cs_map = dict(zip(cs_df["soc3"], cs_df["college_share"]))
        for i, c in enumerate(soc3_codes):
            college_share[i] = cs_map.get(c, 0.5)

    # ---- Retraining matrices (LLM, months -> years) ----
    if verbose:
        print("  Loading retraining matrix (without AI)...")
    d_so = _load_retraining_matrix_soc3(_RETRAIN_NO_AI, soc3_codes) / 12.0
    if verbose:
        print("  Loading retraining matrix (with AI)...")
    d_so_AI = _load_retraining_matrix_soc3(_RETRAIN_WITH_AI, soc3_codes) / 12.0
    d_so_AI = np.minimum(d_so_AI, d_so)

    # ---- PPG (productivity gains from AI) ----
    ppg_df = pd.read_excel(_FINAL_OCC_XLSX)
    ppg_df["soc3"] = ppg_df["soc_code_6"].str[:4]
    ppg_df = ppg_df.dropna(subset=["total_employment_2024",
                                    "pred_productivity_effect"])
    ppg_agg = ppg_df.groupby("soc3").apply(
        lambda g: np.average(g["pred_productivity_effect"],
                             weights=g["total_employment_2024"]),
        include_groups=False,
    )
    ppg_map = ppg_agg.to_dict()
    ppg = np.array([ppg_map.get(c, 0.10) for c in soc3_codes])

    # ---- Tier 1 scalars ----
    beta     = cal["preferences"]["beta"]      # 0.95
    sigma    = cal["production"]["sigma"]       # 4.0

    if verbose:
        print(f"  Loaded {J} 3-digit SOC occupations in {time.time()-t0:.1f}s")
        print(f"  d_so range: {d_so[d_so>0].min():.2f} – {d_so.max():.2f} years")
        print(f"  PPG range: {ppg.min():.3f} – {ppg.max():.3f}")
        print(f"  Mean wage range: ${mean_wage.min():,.0f} – "
              f"${mean_wage.max():,.0f}")
        print()

    return {
        "soc3_codes": soc3_codes,
        "soc3_names": soc3_names,
        "J": J,
        "mean_wage": mean_wage,
        "employment": employment,
        "e_o": e_o,
        "e_o_AI": e_o_AI,
        "college_share": college_share,
        "d_so": d_so,
        "d_so_AI": d_so_AI,
        "ppg": ppg,
        "beta": beta,
        "sigma": sigma,
    }


# ===========================================================================
# SECTION 1 — PARAMETERS & GRIDS
# ===========================================================================

def logsumexp(a, axis=None):
    """Numerically stable log-sum-exp."""
    a_max = a.max(axis=axis, keepdims=True)
    out = a_max + np.log(np.exp(a - a_max).sum(axis=axis, keepdims=True))
    if axis is not None:
        return out.squeeze(axis=axis)
    return out.squeeze()


class SdmParams:
    """
    Container for all model parameters.

    State space: occupation s   (J-dimensional, no ability heterogeneity)
    Value function: V(s)        (J,) array

    Workers are homogeneous (no θ).  All workers in occupation o earn w_o.
    Inequality is purely between-occupation, consistent with the static model.
    """
    def __init__(self, data, tier3):
        # Occupation count
        self.J = data["J"]
        J = self.J

        # Tier 1: externally fixed
        self.beta    = data["beta"]          # annual discount factor
        self.sigma   = data["sigma"]         # CES elasticity

        # No exit rate — infinitely-lived workers
        self.beta_eff = self.beta

        # Tier 3: SMM-estimated (2 parameters)
        self.kappa = tier3["kappa"]
        self.tau   = tier3["tau"]

        # Occupation-specific amenities (J,)
        # Inverted from data employment shares in pre-AI steady state
        self.a_o = np.zeros(J)

        # Store data reference
        self.data = data


# ===========================================================================
# SECTION 2 — VALUE FUNCTION (fixed-point iteration)
# ===========================================================================

def solve_vf(wages, p, d, tol=1e-6, max_iter=500, return_policy=False,
             V_init=None):
    """
    Solve the incumbent Bellman equation by value iteration.

    V(s) = τ · logsumexp_o [ (log w_o - κ·d_{s,o}·1_{o≠s}
                              + a_o + β · V(o)) / τ ]

    State space is just occupation s (no ability θ).

    Returns:
        V: (J,) value function
        policy: (J, J) transition probabilities π(o|s) if return_policy=True
    """
    J     = p.J
    tau   = p.tau
    kappa = p.kappa
    beta  = p.beta_eff

    log_w = np.log(np.maximum(wages, 1e-10))   # (J,)

    # Switching cost matrix: cost[s, o] = κ · d[s, o] for o ≠ s, 0 for o = s
    d_copy = d.copy()
    np.fill_diagonal(d_copy, 0.0)
    cost = kappa * np.maximum(d_copy, 0.0)   # (J, J)

    a_o = p.a_o   # (J,)

    # Initialize V
    V = V_init.copy() if V_init is not None else log_w.copy()   # (J,)

    for it in range(max_iter):
        # v_all[s, o] = (log w_o + a_o - cost[s,o] + β·V(o)) / τ
        v_all = (log_w[None, :] + a_o[None, :] - cost + beta * V[None, :]) / tau
        # v_all shape: (J, J) — rows = origin s, cols = destination o

        V_new = tau * logsumexp(v_all, axis=1)   # (J,)

        diff = np.max(np.abs(V_new - V))
        V = V_new

        if diff < tol:
            break

    if return_policy:
        v_all = (log_w[None, :] + a_o[None, :] - cost + beta * V[None, :]) / tau
        v_max = v_all.max(axis=1, keepdims=True)
        exp_v = np.exp(v_all - v_max)
        policy = exp_v / exp_v.sum(axis=1, keepdims=True)   # (J, J)
        return V, policy

    return V


# ===========================================================================
# SECTION 3 — STATIONARY DISTRIBUTION (ergodic, no entry)
# ===========================================================================

def compute_stationary_distribution(V, p, d, wages=None, policy=None,
                                     max_iter=2000, tol=1e-8):
    """
    Compute the stationary (ergodic) distribution μ(o) over occupations.

    Since workers are infinitely lived with no entry or exit and no
    heterogeneity, the stationary distribution satisfies:

        μ(o) = Σ_s π(o|s) · μ(s)

    This is the left eigenvector of the Markov transition matrix.
    Returns μ as (J,) array summing to 1.
    """
    J = p.J

    if policy is None:
        _, policy = solve_vf(wages, p, d, return_policy=True)

    # Initialize μ uniformly
    mu = np.ones(J) / J

    for it in range(max_iter):
        mu_new = policy.T @ mu   # (J,) — π(o|s)^T · μ(s)
        mu_new = mu_new / mu_new.sum()

        diff = np.max(np.abs(mu_new - mu))
        mu = mu_new

        if diff < tol:
            break

    return mu


# ===========================================================================
# SECTION 4 — EFFECTIVE LABOR & WAGES
# ===========================================================================

def compute_L_eff(mu, p):
    """
    Compute effective labor per occupation.

    With no ability heterogeneity: L_o = μ(o)  (just employment share).
    We scale by total employment to get levels.
    """
    return mu.copy()


def compute_wages(B, L_eff, sigma):
    """CES inverse labor demand: w_o = B_o · L_o^{-1/σ}."""
    return B * np.maximum(L_eff, 1e-10) ** (-1.0 / sigma)


def invert_B(wages, L_eff, sigma):
    """Model inversion: B_o = w_o · L_o^{1/σ}."""
    return wages * np.maximum(L_eff, 1e-10) ** (1.0 / sigma)


# ===========================================================================
# SECTION 5a — AMENITY INVERSION (BLP-style contraction)
# ===========================================================================

def invert_amenities(wages, p, d, data_emp, max_iter=80, tol=1e-4,
                     verbose=True):
    """
    Invert occupation-specific amenities a_o from data employment shares.

    Uses a BLP-style contraction mapping (Berry 1994), adjusted for the
    dynamic model.  In a static logit, a_o enters choice probabilities
    directly, so the standard BLP update is:
        a_o += log(s_data) - log(s_model)

    In our infinite-horizon model, a_o enters flow utility and gets amplified
    through the Bellman recursion by roughly 1/(1 - β).  The corrected
    contraction is:
        a_o += (1 - β) · (log(s_data) - log(s_model))
    """
    J = p.J

    # Normalize data employment to shares
    emp_data = data_emp.astype(float).copy()
    emp_data = emp_data / emp_data.sum()
    emp_data = np.maximum(emp_data, 1e-8)
    log_emp_data = np.log(emp_data)

    # Initialize a_o = 0
    p.a_o = np.zeros(J)

    # Dynamic correction: flow amenity is amplified by 1/(1-β) in V
    dynamic_step = (1.0 - p.beta)   # = 0.05 (for β = 0.95)
    damping = 0.7

    if verbose:
        print(f"  [Amenity inversion] Starting BLP contraction "
              f"(dynamic_step={dynamic_step:.4f}, damping={damping})...")

    prev_max_dev = None

    for it in range(max_iter):
        # Solve VF and get stationary distribution with current a_o
        V, policy = solve_vf(wages, p, d, return_policy=True, tol=1e-6)
        mu = compute_stationary_distribution(V, p, d, policy=policy, tol=1e-9)

        # Model employment shares
        emp_model = mu.copy()
        emp_model = emp_model / np.maximum(emp_model.sum(), 1e-15)
        emp_model = np.maximum(emp_model, 1e-8)

        # Convergence check
        max_dev = np.max(np.abs(emp_model - emp_data))
        corr = np.corrcoef(emp_model, emp_data)[0, 1]

        if verbose and (it % 5 == 0 or max_dev < tol):
            print(f"    Iter {it:3d}: max |emp_dev| = {max_dev:.6f}  "
                  f"corr = {corr:.4f}")

        if max_dev < tol:
            if verbose:
                print(f"    Amenity inversion converged in {it} iterations. "
                      f"Max dev = {max_dev:.6f}")
            break

        # Adaptive damping: reduce if diverging
        if prev_max_dev is not None and max_dev > prev_max_dev * 1.1:
            damping *= 0.8
        prev_max_dev = max_dev

        # BLP contraction update with dynamic correction
        log_emp_model = np.log(emp_model)
        delta = dynamic_step * damping * (log_emp_data - log_emp_model)
        p.a_o = p.a_o + delta

        # Normalize amenities (subtract mean for identification)
        p.a_o = p.a_o - p.a_o.mean()

    else:
        if verbose:
            print(f"    WARNING: Amenity inversion did not converge after "
                  f"{max_iter} iterations. Max dev = {max_dev:.6f}")

    return p


# ===========================================================================
# SECTION 5b — STEADY STATE SOLVER
# ===========================================================================

def solve_steady_state(p, d, max_iter=200, tol=1e-3, damping=0.15,
                       verbose=True, B_ext=None, wages_init=None,
                       invert_amenities_flag=False, data_emp=None,
                       amenity_tol=1e-4, amenity_max_iter=80):
    """
    Solve for the stationary equilibrium by iterating on wages.

    B_ext: (J,) externally provided productivity parameters.
    wages_init: (J,) initial wage guess. If None, uses data wages.
    invert_amenities_flag: if True, invert a_o from data employment shares.
    data_emp: (J,) data employment levels, required if invert_amenities_flag.
    """
    J     = p.J
    sigma = p.sigma

    if wages_init is not None:
        wages = wages_init.copy()
    else:
        wages = p.data["mean_wage"].copy()

    # Step 1: Invert amenities (if requested — pre-AI calibration only)
    if invert_amenities_flag:
        if data_emp is None:
            raise ValueError("data_emp required when invert_amenities_flag=True")
        invert_amenities(wages, p, d, data_emp,
                         max_iter=amenity_max_iter, tol=amenity_tol,
                         verbose=verbose)

    # Step 2: Invert B (or use externally provided B)
    if B_ext is not None:
        B = B_ext.copy()
    else:
        # Pre-AI case: invert B from data wages and model-implied L_eff
        V, policy = solve_vf(wages, p, d, return_policy=True)
        mu = compute_stationary_distribution(V, p, d, policy=policy)
        L_eff = compute_L_eff(mu, p)
        B = invert_B(wages, L_eff, sigma)

    prev_dw = None

    for it in range(max_iter):
        V, policy = solve_vf(wages, p, d, return_policy=True)
        mu = compute_stationary_distribution(V, p, d, policy=policy)
        L_eff = compute_L_eff(mu, p)
        wages_new = compute_wages(B, L_eff, sigma)

        dw = np.max(np.abs(wages_new - wages) / np.maximum(wages, 1e-10))

        if verbose and (it % 25 == 0 or it < 3 or dw < tol):
            print(f"    Iter {it:3d}: max Δw = {dw:.6f}  (damp={damping:.3f})")

        if dw < tol:
            if verbose:
                print(f"    Converged in {it} iterations.")
            break

        if prev_dw is not None and dw > prev_dw * 1.05:
            damping = max(damping * 0.85, 0.02)
        prev_dw = dw

        wages = wages + damping * (wages_new - wages)

    return wages, B, V, mu, L_eff


# ===========================================================================
# SECTION 6 — MOMENT COMPUTATIONS
# ===========================================================================

def compute_var_log_occ_wage(wages, mu):
    """
    Variance of log occupational wages, weighted by employment shares.

    This is the ONLY inequality measure in the no-ability model,
    consistent with the static model framework.
    """
    emp_share = mu.copy()
    emp_share = emp_share / np.maximum(emp_share.sum(), 1e-15)
    log_wages = np.log(np.maximum(wages, 1e-10))
    mean_lw = (emp_share * log_wages).sum()
    var_log_occ_wage = (emp_share * (log_wages - mean_lw) ** 2).sum()
    return var_log_occ_wage


def compute_var_log_occ_wage_dataweights(wages, data_emp):
    """
    Variance of log occupational wages using DATA employment weights.
    """
    emp_share = data_emp.astype(float).copy()
    emp_share = emp_share / np.maximum(emp_share.sum(), 1e-15)
    log_wages = np.log(np.maximum(wages, 1e-10))
    mean_lw = (emp_share * log_wages).sum()
    var_log_occ_wage = (emp_share * (log_wages - mean_lw) ** 2).sum()
    return var_log_occ_wage


def compute_gini_occ_wages(wages, mu):
    """Gini coefficient of occupational wages (weighted by employment)."""
    emp_share = mu.copy()
    emp_share = emp_share / np.maximum(emp_share.sum(), 1e-15)

    total_we = (emp_share * wages).sum()
    if total_we < 1e-15:
        return 0.0

    order = np.argsort(wages)
    w_sort = emp_share[order]
    e_sort = wages[order]

    cum_pop    = np.cumsum(w_sort)
    cum_income = np.cumsum(w_sort * e_sort) / total_we

    cum_pop    = np.concatenate([[0.0], cum_pop])
    cum_income = np.concatenate([[0.0], cum_income])

    dp = np.diff(cum_pop)
    avg_L = 0.5 * (cum_income[:-1] + cum_income[1:])
    area_under_lorenz = (dp * avg_L).sum()

    gini = 1.0 - 2.0 * area_under_lorenz
    return max(0.0, min(1.0, gini))


def compute_model_moments(wages, mu, p, d, V=None):
    """
    Compute model-implied moments for SMM estimation.

    Targets (2 moments for 2 parameters):
        var_log_occ_wage, mobility_rate
    """
    J = p.J

    # 1. Var(log occ wage) — between-occupation
    var_log_occ_wage = compute_var_log_occ_wage(wages, mu)

    # 2. Mobility rate (reuse V if provided)
    _, policy = solve_vf(wages, p, d, return_policy=True, V_init=V)
    s_idx = np.arange(J)
    stay_prob = policy[s_idx, s_idx]   # diagonal: π(s|s)
    switch_prob = 1.0 - stay_prob
    mobility_rate = (mu * switch_prob).sum() / np.maximum(mu.sum(), 1e-15)

    return {
        "var_log_occ_wage":   var_log_occ_wage,
        "mobility_rate":      mobility_rate,
    }


# ===========================================================================
# SECTION 7 — STEADY STATE VERIFICATION
# ===========================================================================

def verify_steady_state(wages, mu, p, data, label="Pre-AI"):
    """Print detailed diagnostics comparing model vs data steady state."""
    J = p.J
    names = data["soc3_names"]

    # --- Employment comparison ---
    data_emp = data["employment"].astype(float)
    data_share = data_emp / data_emp.sum()

    model_share = mu.copy()
    model_share = model_share / np.maximum(model_share.sum(), 1e-15)

    corr = np.corrcoef(model_share, data_share)[0, 1]
    rmse = np.sqrt(np.mean((model_share - data_share) ** 2))
    max_dev_idx = np.argmax(np.abs(model_share - data_share))
    max_dev = np.abs(model_share - data_share).max()

    print(f"\n  === {label} Steady State Verification ===")
    print(f"\n  Employment Distribution:")
    print(f"    Corr(model, data):  {corr:.6f}")
    print(f"    RMSE (shares):      {rmse:.6f}")
    print(f"    Max deviation:      {max_dev:.6f}  "
          f"({names[max_dev_idx][:35]})")

    # Top 10 deviations
    dev = model_share - data_share
    abs_dev = np.abs(dev)
    top10 = np.argsort(abs_dev)[::-1][:10]
    print(f"\n    Top-10 employment share deviations:")
    print(f"    {'Occupation':<35s} {'Data':>8s} {'Model':>8s} {'Diff':>8s}")
    print(f"    {'-'*63}")
    for i in top10:
        print(f"    {names[i][:35]:<35s} {data_share[i]:8.4f} "
              f"{model_share[i]:8.4f} {dev[i]:+8.4f}")

    # --- Wage comparison ---
    data_wages = data["mean_wage"]
    wage_dev = np.abs(wages - data_wages) / np.maximum(data_wages, 1e-10)
    max_wage_dev = wage_dev.max()
    max_wage_idx = np.argmax(wage_dev)
    mean_wage_dev = wage_dev.mean()

    print(f"\n  Wage Match:")
    print(f"    Max % deviation:    {100*max_wage_dev:.4f}%  "
          f"({names[max_wage_idx][:35]})")
    print(f"    Mean % deviation:   {100*mean_wage_dev:.4f}%")

    # --- Amenity summary ---
    a_o = p.a_o
    print(f"\n  Amenity Distribution (a_o):")
    print(f"    Mean: {a_o.mean():.4f}  Std: {a_o.std():.4f}  "
          f"Min: {a_o.min():.4f}  Max: {a_o.max():.4f}")

    top5 = np.argsort(a_o)[::-1][:5]
    bot5 = np.argsort(a_o)[:5]
    print(f"\n    Top-5 amenity occupations:")
    for i in top5:
        print(f"      {names[i][:40]:<40s}  a_o = {a_o[i]:+.4f}  "
              f"emp_share = {data_share[i]:.4f}")
    print(f"    Bottom-5 amenity occupations:")
    for i in bot5:
        print(f"      {names[i][:40]:<40s}  a_o = {a_o[i]:+.4f}  "
              f"emp_share = {data_share[i]:.4f}")

    print()
    return {
        "emp_corr": corr,
        "emp_rmse": rmse,
        "emp_max_dev": max_dev,
        "wage_max_pct_dev": max_wage_dev,
        "wage_mean_pct_dev": mean_wage_dev,
        "amenity_std": a_o.std(),
    }


# ===========================================================================
# SECTION 8 — AI SHOCK
# ===========================================================================

def apply_ai_shock(p, B_pre, data):
    """
    Apply the full AI shock (productivity + contestability).
    Amenities a_o are structural and FIXED from pre-AI inversion.
    """
    ppg = data["ppg"]
    sig = p.sigma

    prod_boost = ((1.0 - np.clip(ppg, 0.0, 0.99)) ** (-1)) ** ((sig - 1) / sig)
    B_post = B_pre * prod_boost

    data_post = data.copy()
    data_post["d_so"] = data["d_so_AI"].copy()

    p_post = SdmParams.__new__(SdmParams)
    p_post.__dict__.update(p.__dict__)
    p_post.a_o = p.a_o.copy()
    p_post.data = data_post

    d_post = data["d_so_AI"].copy()

    return p_post, B_post, d_post


def apply_ai_shock_supply_only(p, B_pre, data):
    """
    Apply supply-side only AI shock (retraining changes, no PPG).
    Amenities a_o are structural and FIXED from pre-AI inversion.
    """
    data_post = data.copy()
    data_post["d_so"] = data["d_so_AI"].copy()

    p_post = SdmParams.__new__(SdmParams)
    p_post.__dict__.update(p.__dict__)
    p_post.a_o = p.a_o.copy()
    p_post.data = data_post

    d_post = data["d_so_AI"].copy()

    return p_post, B_pre.copy(), d_post


# ===========================================================================
# SECTION 9 — TRANSITION PATH
# ===========================================================================

def solve_transition(p_pre, p_post, B_pre, B_post, d_pre, d_post,
                     wages_pre, wages_post, mu_pre, data_emp,
                     T_trans=80, n_outer=30, tol=5e-3, damping=0.05,
                     verbose=True):
    """
    Solve the transition path after an unexpected permanent AI shock.

    Workers are infinitely lived — no entry flows at any period.
    The distribution evolves purely through incumbent switching.
    """
    J     = p_post.J
    sigma = p_post.sigma

    wage_path = np.zeros((T_trans, J))
    for t in range(T_trans):
        frac = min(t / max(T_trans // 2, 1), 1.0)
        wage_path[t] = (1 - frac) * wages_pre + frac * wages_post

    prev_max_dw = None

    for outer in range(n_outer):
        mu_t = mu_pre.copy()
        L_path = np.zeros((T_trans, J))
        var_occ_path = np.zeros(T_trans)
        var_occ_dw_path = np.zeros(T_trans)
        wage_path_new = np.zeros_like(wage_path)

        for t in range(T_trans):
            w_t = wage_path[t]

            V_t, policy_t = solve_vf(w_t, p_post, d_post, return_policy=True,
                                     tol=1e-5, max_iter=200)

            # Forward-iterate distribution (no entry flow)
            mu_new = policy_t.T @ mu_t   # (J,)
            mu_new = mu_new / mu_new.sum()

            L_eff = compute_L_eff(mu_new, p_post)
            wage_path_new[t] = compute_wages(B_post, L_eff, sigma)

            var_occ_path[t] = compute_var_log_occ_wage(w_t, mu_new)
            var_occ_dw_path[t] = compute_var_log_occ_wage_dataweights(
                w_t, data_emp)

            L_path[t] = L_eff
            mu_t = mu_new

        max_dw = np.max(np.abs(wage_path_new - wage_path) /
                        np.maximum(wage_path, 1e-10))

        # Adaptive damping
        if prev_max_dw is not None:
            if max_dw > prev_max_dw * 1.05:
                damping = max(damping * 0.7, 0.01)
            elif max_dw < prev_max_dw * 0.9:
                damping = min(damping * 1.1, 0.3)
        prev_max_dw = max_dw

        if verbose:
            print(f"    Outer {outer+1}/{n_outer}: max Δw = {max_dw:.5f}  "
                  f"(damp={damping:.3f})")

        if max_dw < tol:
            if verbose:
                print(f"    Transition converged in {outer+1} outer iterations.")
            break

        # Damped update
        update = damping * (wage_path_new - wage_path)
        max_step = 0.5 * wage_path
        update = np.clip(update, -max_step, max_step)
        wage_path = wage_path + update

    return wage_path, var_occ_path, var_occ_dw_path


# ===========================================================================
# SECTION 10 — PLOTTING
# ===========================================================================

def plot_results(var_occ_dw_pre, var_occ_dw_post, var_occ_dw_path,
                 wages_pre, wages_post, data, save_dir=None):
    """Generate standard output plots using data employment weights."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    J = len(wages_pre)

    # --- Plot 1: Inequality evolution (data-weighted) ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    T_trans = len(var_occ_dw_path)
    ax.axhline(var_occ_dw_pre, color="gray", ls="--", alpha=0.7,
               label=f"Pre-AI SS = {var_occ_dw_pre:.4f}")
    ax.axhline(var_occ_dw_post, color="gray", ls=":", alpha=0.7,
               label=f"Post-AI SS = {var_occ_dw_post:.4f}")
    ax.plot(range(T_trans), var_occ_dw_path, "b-", lw=2.5, label="Transition")

    pct_chg = 100 * (var_occ_dw_post - var_occ_dw_pre) / abs(var_occ_dw_pre)
    ax.set_xlabel("Years after AI shock", fontsize=12)
    ax.set_ylabel("Var(log occupational wage)", fontsize=12)
    ax.set_title(f"Wage Inequality After AI Reduces Expertise Barriers\n"
                 f"Supply-Side GE: {pct_chg:+.1f}% "
                 f"(Data Employment Weights)",
                 fontsize=13)
    ax.set_xlim(0, 20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, "inequality_evolution.png"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: Wage changes by occupation ---
    pct_chg_w = 100 * (wages_post - wages_pre) / np.maximum(wages_pre, 1e-10)
    order = np.argsort(pct_chg_w)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    names = data["soc3_names"]
    colors = ["green" if x > 0 else "red" for x in pct_chg_w[order]]
    y_pos = np.arange(J)
    ax.barh(y_pos, pct_chg_w[order], color=colors, alpha=0.7)
    ax.set_yticks(y_pos[::3])
    ax.set_yticklabels([names[i][:35] for i in order[::3]], fontsize=7)
    ax.set_xlabel("% Wage Change")
    ax.set_title("Long-Run Wage Changes by Occupation\n"
                 "(Supply-Side Only, No Ability Heterogeneity)")
    ax.axvline(0, color="black", lw=0.5)
    ax.grid(True, alpha=0.3, axis="x")
    if save_dir:
        fig.savefig(os.path.join(save_dir, "wage_changes.png"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 3: Retraining distance reduction ---
    d_pre  = data["d_so"]
    d_post = data["d_so_AI"]
    mask = ~np.eye(J, dtype=bool)
    mean_pre  = d_pre[mask].mean()
    mean_post = d_post[mask].mean()
    pct_red = 100 * (mean_pre - mean_post) / mean_pre

    d_red_by_occ = np.zeros(J)
    for s in range(J):
        orig = d_pre[s, mask[s]].mean()
        post = d_post[s, mask[s]].mean()
        d_red_by_occ[s] = 100 * (orig - post) / max(orig, 1e-10)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    order_d = np.argsort(d_red_by_occ)
    ax.barh(range(J), d_red_by_occ[order_d], color="steelblue", alpha=0.7)
    ax.set_yticks(range(0, J, 5))
    ax.set_yticklabels([data["soc3_names"][i][:30] for i in order_d[::5]],
                       fontsize=6)
    ax.set_xlabel("Mean retraining distance reduction (%)")
    ax.set_title(f"AI Impact on Retraining Distances by Origin Occupation\n"
                 f"(Overall mean reduction: {pct_red:.1f}%)")
    ax.grid(True, alpha=0.3, axis="x")
    if save_dir:
        fig.savefig(os.path.join(save_dir, "retraining_reduction.png"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# SECTION 11 — MAIN
# ===========================================================================

def main():
    t_start = time.time()

    print("=" * 72)
    print("  SIMPLE DYNAMIC MODEL — Infinite-Horizon Occupational Choice")
    print("  No Ability Heterogeneity (Between-Occupation Inequality Only)")
    print("  AI, Expertise Barriers, and Wage Inequality")
    print("  Hosseini & Lichtinger (2026)")
    print("=" * 72)
    print()

    # ---- 1. Load data ----
    print("[1/6] Loading model data...")
    data = build_model_data(verbose=False)
    J = data["J"]
    d_pre = data["d_so"]
    print(f"  Loaded {J} occupations")

    # ---- 2. Load calibrated parameters ----
    smm_cache = os.path.join(OUTPUT_DIR, "calibrated_parameters_sdm_smm.json")
    if not os.path.exists(smm_cache):
        raise FileNotFoundError(
            f"Calibrated parameters not found at:\n  {smm_cache}\n"
            "Run calibrate_sdm_smm.py first."
        )
    print("[2/6] Loading calibrated parameters...")
    with open(smm_cache) as f:
        saved = json.load(f)
    tier3 = saved["tier3_smm"]
    print(f"  Parameters: {tier3}")

    # ---- 3. Initialize model ----
    print("[3/6] Building model parameters...")
    p = SdmParams(data, tier3)

    print(f"\n  SIMPLE DYNAMIC MODEL PARAMETERS (NO ABILITY):")
    print(f"    β     = {p.beta}")
    print(f"    β_eff = {p.beta_eff:.4f}  (= β, no exit rate)")
    print(f"    σ     = {p.sigma}")
    print(f"    κ     = {p.kappa:.4f}")
    print(f"    τ     = {p.tau:.4f}")
    print()

    # ---- 4. Pre-AI steady state (with amenity inversion) ----
    print("[4/6] Solving pre-AI steady state...")
    print("  Step 1: Inverting occupation-specific amenities from data employment")
    print("  Step 2: Inverting B from data wages")
    t0 = time.time()
    data_emp = data["employment"]
    wages_pre, B_pre, V_pre, mu_pre, L_pre = solve_steady_state(
        p, d_pre, max_iter=200, tol=1e-3, damping=0.15, verbose=True,
        invert_amenities_flag=True, data_emp=data_emp)
    var_occ_pre = compute_var_log_occ_wage(wages_pre, mu_pre)
    var_occ_dw_pre = compute_var_log_occ_wage_dataweights(wages_pre, data_emp)
    gini_pre = compute_gini_occ_wages(wages_pre, mu_pre)
    print(f"  Var(log occ wage, model w) = {var_occ_pre:.4f}")
    print(f"  Var(log occ wage, data w)  = {var_occ_dw_pre:.4f}")
    print(f"  Gini (occ wages)           = {gini_pre:.4f}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Verify pre-AI steady state
    verification = verify_steady_state(wages_pre, mu_pre, p, data, label="Pre-AI")
    print()

    # ---- 5. Supply-side only AI shock ----
    print("=" * 72)
    print("  SUPPLY-SIDE ONLY SCENARIO")
    print("=" * 72)

    p_sup, B_sup, d_sup = apply_ai_shock_supply_only(p, B_pre, data)

    d_pre_offdiag = d_pre[~np.eye(J, dtype=bool)].mean()
    d_post_offdiag = d_sup[~np.eye(J, dtype=bool)].mean()
    d_reduction = (d_pre_offdiag - d_post_offdiag) / d_pre_offdiag

    print(f"  Retraining reduction (mean off-diag): {100*d_reduction:.1f}%")
    print()

    print("[5/6] Solving post-AI steady state (supply-only)...")
    print("  Using pre-AI B (supply-only: no productivity change)")
    print("  Starting from pre-AI wages as initial guess")
    t0 = time.time()
    wages_sup, B_sup2, V_sup, mu_sup, L_sup = solve_steady_state(
        p_sup, d_sup, max_iter=200, tol=1e-3, damping=0.15, verbose=True,
        B_ext=B_sup, wages_init=wages_pre)
    var_occ_sup = compute_var_log_occ_wage(wages_sup, mu_sup)
    var_occ_dw_sup = compute_var_log_occ_wage_dataweights(wages_sup, data_emp)
    gini_sup = compute_gini_occ_wages(wages_sup, mu_sup)
    print(f"  Var(log occ wage, model w) = {var_occ_sup:.4f}")
    print(f"  Var(log occ wage, data w)  = {var_occ_dw_sup:.4f}")
    print(f"  Gini (occ wages)           = {gini_sup:.4f}")
    print(f"  Time: {time.time()-t0:.1f}s")

    pct_change_occ = 100 * (var_occ_sup - var_occ_pre) / max(abs(var_occ_pre), 1e-10)
    pct_change_dw  = 100 * (var_occ_dw_sup - var_occ_dw_pre) / max(abs(var_occ_dw_pre), 1e-10)
    print(f"  Long-run Δ Var(log occ wage, model):{pct_change_occ:+.2f}%")
    print(f"  Long-run Δ Var(log occ wage, data): {pct_change_dw:+.2f}%")
    print()

    # ---- 6. Transition path ----
    print("[6/6] Solving transition path...")
    t0 = time.time()
    wage_path, var_occ_path, var_occ_dw_path = solve_transition(
        p, p_sup, B_pre, B_sup, d_pre, d_sup,
        wages_pre, wages_sup, mu_pre, data_emp,
        T_trans=80, n_outer=100, tol=1e-2, damping=0.10, verbose=True)
    print(f"  Transition solve time: {time.time()-t0:.1f}s")

    # ---- Results ----
    print()
    print("=" * 72)
    print("  RESULTS SUMMARY (SUPPLY-SIDE ONLY, NO ABILITY)")
    print("=" * 72)
    print()
    print(f"  Pre-AI Var(log occ wage, model):  {var_occ_pre:.4f}  (Gini={gini_pre:.4f})")
    print(f"  Post-AI Var(log occ wage, model): {var_occ_sup:.4f}  (Gini={gini_sup:.4f})")
    print(f"  Δ Var(log occ wage, model):       {var_occ_sup-var_occ_pre:+.4f} ({pct_change_occ:+.1f}%)")
    print()
    print(f"  Pre-AI Var(log occ w, data wt):   {var_occ_dw_pre:.4f}")
    print(f"  Post-AI Var(log occ w, data wt):  {var_occ_dw_sup:.4f}")
    print(f"  Δ Var(log occ wage, data wt):     {var_occ_dw_sup-var_occ_dw_pre:+.4f} ({pct_change_dw:+.1f}%)")

    # Save plots
    plot_results(var_occ_dw_pre, var_occ_dw_sup, var_occ_dw_path,
                 wages_pre, wages_sup, data, save_dir=OUTPUT_DIR)

    # Save results
    np.savez(os.path.join(OUTPUT_DIR, "model_results.npz"),
             wages_pre=wages_pre, wages_post=wages_sup,
             var_occ_path=var_occ_path,
             var_occ_dw_path=var_occ_dw_path,
             wage_path=wage_path,
             mu_pre=mu_pre, mu_post=mu_sup,
             amenities=p.a_o)

    results = {
        "tier3_smm": tier3,
        "amenities": {
            "a_o": p.a_o.tolist(),
            "mean": float(p.a_o.mean()),
            "std": float(p.a_o.std()),
        },
        "verification": verification,
        "results": {
            "var_occ_pre": float(var_occ_pre),
            "var_occ_post": float(var_occ_sup),
            "var_occ_dw_pre": float(var_occ_dw_pre),
            "var_occ_dw_post": float(var_occ_dw_sup),
            "gini_pre": float(gini_pre),
            "gini_post": float(gini_sup),
            "pct_change_occ": float(pct_change_occ),
            "pct_change_dw": float(pct_change_dw),
        }
    }
    with open(os.path.join(OUTPUT_DIR, "model_output_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"  Total wall time: {time.time()-t_start:.1f}s")
    print(f"\n  Output saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
