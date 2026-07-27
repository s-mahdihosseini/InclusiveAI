"""
==============================================================================
Calibration Module: SMM Estimation — Simple Dynamic Model (No Ability)
==============================================================================
Estimates Tier 3 parameters: κ, τ

2 parameters, 2 target moments:
  1. Var(log occ wage)             = 0.208  (BLS OEWS)
  2. Mobility rate                 = 0.12   (CPS, 1-year switching)

Each SMM evaluation inverts occupation-specific amenities (a_o) from data
employment shares before B inversion, ensuring correct employment distribution.

Authors: Seyed M Hosseini, Guy Lichtinger
Date:    March 2026
==============================================================================
"""

import numpy as np
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

from simple_dynamic_model import (
    SdmParams,
    solve_steady_state,
    compute_model_moments,
    solve_vf,
    compute_stationary_distribution,
    compute_L_eff,
    compute_wages,
    invert_B,
    invert_amenities,
    logsumexp,
    build_model_data,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(THIS_DIR, "sdm_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# SECTION 1 — CALIBRATION TARGETS
# ===========================================================================

DATA_MOMENTS = {
    "var_log_occ_wage":   0.208,   # Between-occ Var(log wage), BLS
    "mobility_rate":      0.12,    # 1-yr occupational switching rate, CPS
}

MOMENT_WEIGHTS = {
    "var_log_occ_wage":   5.0,
    "mobility_rate":      8.0,
}


# ===========================================================================
# SECTION 2 — SMM OBJECTIVE FUNCTION
# ===========================================================================

def smm_objective(theta_vec, data, d_so, data_emp, verbose=False):
    """
    SMM objective function for the Simple Dynamic Model (no ability).

    Each evaluation inverts amenities from data employment shares (coarse
    tolerance for speed), then inverts B from data wages.
    """
    kappa, tau = theta_vec

    # Bounds enforcement
    if (kappa < 0.10 or kappa > 20.0 or
        tau < 0.05 or tau > 3.0):
        return 1e6

    tier3 = {
        "kappa": kappa,
        "tau": tau,
    }

    try:
        p = SdmParams(data, tier3)
        wages, B, V, mu, L_eff = solve_steady_state(
            p, d_so, max_iter=80, tol=3e-3, damping=0.15, verbose=False,
            invert_amenities_flag=True, data_emp=data_emp,
            amenity_tol=1e-2, amenity_max_iter=30)
        model_mom = compute_model_moments(wages, mu, p, d_so, V=V)
    except Exception as e:
        if verbose:
            print(f"    [SMM] Exception: {e}")
        return 1e6

    # Weighted squared percentage deviations
    obj = 0.0
    for key in DATA_MOMENTS:
        data_m  = DATA_MOMENTS[key]
        model_m = model_mom[key]
        weight  = MOMENT_WEIGHTS[key]
        pct_dev = (model_m - data_m) / max(abs(data_m), 1e-10)
        obj += weight * pct_dev ** 2

    if verbose:
        print(f"    [SMM] obj={obj:.4f} | ", end="")
        for key in DATA_MOMENTS:
            print(f"{key}={model_mom[key]:.4f}({DATA_MOMENTS[key]:.4f}) ", end="")
        print()

    return obj


# ===========================================================================
# SECTION 3 — NELDER-MEAD OPTIMIZER
# ===========================================================================

def _nelder_mead(func, x0, args=(), max_fev=150, xatol=1e-3, fatol=1e-4,
                 verbose=False):
    """Nelder-Mead simplex optimizer (no scipy dependency)."""
    n = len(x0)
    alpha_nm, gamma_nm, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5

    simplex = np.zeros((n + 1, n))
    simplex[0] = x0.copy()
    for i in range(n):
        simplex[i + 1] = x0.copy()
        step = max(abs(x0[i]) * 0.15, 0.05)
        simplex[i + 1, i] += step

    f_vals = np.array([func(simplex[i], *args) for i in range(n + 1)])
    n_fev = n + 1
    best_f = f_vals.min()
    best_x = simplex[np.argmin(f_vals)].copy()

    if verbose:
        print(f"    NM init: best obj = {best_f:.4f}")

    while n_fev < max_fev:
        order = np.argsort(f_vals)
        simplex = simplex[order]
        f_vals  = f_vals[order]

        if f_vals[0] < best_f:
            best_f = f_vals[0]
            best_x = simplex[0].copy()

        f_range = f_vals[-1] - f_vals[0]
        x_range = np.max(np.abs(simplex[-1] - simplex[0]))
        if f_range < fatol and x_range < xatol:
            break

        if verbose and n_fev % 20 == 0:
            print(f"    NM eval {n_fev:3d}: best obj = {best_f:.6f}  "
                  f"spread = {f_range:.4f}")

        centroid = simplex[:-1].mean(axis=0)

        xr = centroid + alpha_nm * (centroid - simplex[-1])
        fr = func(xr, *args); n_fev += 1

        if f_vals[0] <= fr < f_vals[-2]:
            simplex[-1] = xr; f_vals[-1] = fr
            continue

        if fr < f_vals[0]:
            xe = centroid + gamma_nm * (xr - centroid)
            fe = func(xe, *args); n_fev += 1
            if fe < fr:
                simplex[-1] = xe; f_vals[-1] = fe
            else:
                simplex[-1] = xr; f_vals[-1] = fr
            continue

        if fr < f_vals[-1]:
            xc = centroid + rho * (xr - centroid)
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
        fc = func(xc, *args); n_fev += 1

        if fc < min(fr, f_vals[-1]):
            simplex[-1] = xc; f_vals[-1] = fc
            continue

        for i in range(1, n + 1):
            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
            f_vals[i] = func(simplex[i], *args); n_fev += 1

    return best_x, best_f, n_fev


# ===========================================================================
# SECTION 4 — SMM RUNNER
# ===========================================================================

def run_smm(data, d_so, data_emp, max_iter=100, verbose=True):
    """Run SMM estimation for the Simple Dynamic Model (no ability)."""
    if verbose:
        print("[SMM] Starting estimation for Simple Dynamic Model...")
        print(f"  Target moments: {list(DATA_MOMENTS.keys())}")
        print(f"  Parameters: kappa, tau")
        print(f"  (with amenity inversion per evaluation)")
        print()

    x0 = np.array([
        5.0,      # kappa
        0.45,     # tau
    ])
    param_names = ["kappa", "tau"]

    t0 = time.time()

    best_x, best_f, n_fev = _nelder_mead(
        smm_objective, x0, args=(data, d_so, data_emp, False),
        max_fev=max_iter, xatol=1e-3, fatol=1e-4, verbose=verbose)

    elapsed = time.time() - t0

    tier3 = {k: float(v) for k, v in zip(param_names, best_x)}

    if verbose:
        print(f"\n  SMM completed in {elapsed:.1f}s "
              f"({n_fev} function evaluations)")
        print(f"  Final objective: {best_f:.6f}")
        print(f"\n  Estimated parameters:")
        for name, val in tier3.items():
            print(f"    {name:15s} = {val:.4f}")

        # Final moments at full resolution
        print(f"\n  Computing final moments at full resolution...")
        p_final = SdmParams(data, tier3)
        wages_f, B_f, V_f, mu_f, L_f = solve_steady_state(
            p_final, d_so, max_iter=200, tol=1e-3, damping=0.15, verbose=False,
            invert_amenities_flag=True, data_emp=data_emp,
            amenity_tol=1e-4, amenity_max_iter=100)
        final_mom = compute_model_moments(wages_f, mu_f, p_final, d_so, V=V_f)

        print(f"\n  {'Moment':<25s} {'Data':>10s} {'Model':>10s} {'% Dev':>10s}")
        print(f"  {'-'*55}")
        for key in DATA_MOMENTS:
            d_val = DATA_MOMENTS[key]
            m_val = final_mom[key]
            pct = 100 * (m_val - d_val) / max(abs(d_val), 1e-10)
            print(f"  {key:<25s} {d_val:10.4f} {m_val:10.4f} {pct:+10.2f}%")
        print()

    smm_results = {
        "tier3": tier3,
        "objective": float(best_f),
        "n_fev": n_fev,
        "elapsed": elapsed,
    }
    return tier3, smm_results


# ===========================================================================
# SECTION 5 — MAIN
# ===========================================================================

def main():
    t_start = time.time()

    print("=" * 72)
    print("  SMM CALIBRATION — Simple Dynamic Model (No Ability)")
    print("  AI, Expertise Barriers, and Inequality")
    print("  Hosseini & Lichtinger (2026)")
    print("=" * 72)
    print()

    # ---- 1. Load data ----
    print("[1/2] Loading model data...")
    data = build_model_data(verbose=False)
    J = data["J"]
    d_so = data["d_so"]
    data_emp = data["employment"]
    print(f"  Loaded {J} occupations")
    print()

    # ---- 2. Run SMM estimation ----
    print("[2/2] Running SMM estimation (with amenity inversion per eval)...")
    print()
    tier3, smm_results = run_smm(data, d_so, data_emp, max_iter=80, verbose=True)

    # ---- 3. Save results ----
    smm_cache = os.path.join(OUTPUT_DIR, "calibrated_parameters_sdm_smm.json")
    output_dict = {
        "tier3_smm": tier3,
        "smm_diagnostics": {
            "objective": smm_results["objective"],
            "n_evaluations": smm_results["n_fev"],
            "elapsed_seconds": smm_results["elapsed"],
        },
        "data_moments": DATA_MOMENTS,
        "moment_weights": MOMENT_WEIGHTS,
    }

    with open(smm_cache, "w") as f:
        json.dump(output_dict, f, indent=2)
    print(f"\n[COMPLETE] Results saved to:")
    print(f"  {smm_cache}")
    print()

    elapsed_total = time.time() - t_start
    print(f"Total execution time: {elapsed_total:.1f}s")


if __name__ == "__main__":
    main()
