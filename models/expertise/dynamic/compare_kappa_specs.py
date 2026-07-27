"""
Run the three kappa parameterizations side by side and write a
comparison JSON + console table.

Specs
-----
(A) scalar     — reload κ, τ from calibrated_parameters_sdm_smm.json
                 (the existing SMM benchmark).
(B) destination — κ_in[o] (J=94 params), τ fixed at (A)'s value;
                   targets destination inflow shares among switchers.
(C) two-way    — κ_out[s] + κ_in[o] (2J=188 params), τ fixed at (A);
                  targets origin outflow + destination inflow.

All three specs use the same pre-AI amenity / B inversion machinery.
For (B) and (C) we re-invert amenities periodically so that occupation
employment shares remain matched to data throughout kappa calibration.
"""

import os
import json
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import paths_override  # noqa: F401   (must come before simple_dynamic_model)

from simple_dynamic_model import (
    build_model_data,
    SdmParams,
    compute_var_log_occ_wage,
    compute_gini_occ_wages,
    solve_vf,
    compute_stationary_distribution,
    solve_steady_state,
)
from flow_data import load_flow_moments
from sdm_kappa_vec import (
    KappaSpec,
    solve_steady_state_vec,
    compute_flow_moments,
    calibrate_dest,
    calibrate_twoway,
    calibrate_twoway_stable,
    build_cost_matrix,
)


THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(THIS_DIR, "sdm_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
SMM_JSON   = os.path.join(OUTPUT_DIR, "calibrated_parameters_sdm_smm.json")


# ---------------------------------------------------------------------------

def _flow_fit_stats(f_mod, f_data, m_mod, m_data, valid_mask=None):
    """Concise fit metrics. Outflow stats computed on VALID origins only."""
    f_mod  = np.asarray(f_mod);  f_data = np.asarray(f_data)
    m_mod  = np.asarray(m_mod);  m_data = np.asarray(m_data)
    eps = 1e-10
    log_dev_f = np.log(f_mod + eps) - np.log(f_data + eps)
    corr_f = float(np.corrcoef(f_mod, f_data)[0, 1])
    rmse_f = float(np.sqrt(np.mean((f_mod - f_data) ** 2)))
    max_lf = float(np.max(np.abs(log_dev_f)))
    if valid_mask is None:
        valid_mask = np.ones_like(m_mod, dtype=bool)
    mm = m_mod[valid_mask]
    md = m_data[valid_mask]
    md = np.where(np.isnan(md), 0.0, md)
    log_dev_m = np.log(mm + eps) - np.log(md + eps)
    corr_m = float(np.corrcoef(mm, md)[0, 1]) if mm.size > 1 else float("nan")
    rmse_m = float(np.sqrt(np.mean((mm - md) ** 2)))
    max_lm = float(np.max(np.abs(log_dev_m)))
    return {
        "inflow_corr":        corr_f,
        "inflow_rmse":        rmse_f,
        "inflow_max_log_dev": max_lf,
        "outflow_corr":       corr_m,
        "outflow_rmse":       rmse_m,
        "outflow_max_log_dev": max_lm,
        "n_valid_origins":    int(valid_mask.sum()),
    }


def _summary(tag, wages, mu, policy, data_emp, data_flow):
    fm = compute_flow_moments(policy, mu, data_emp)
    stats = _flow_fit_stats(
        fm["inflow_share_switch_model"], data_flow["inflow_share_switch_data"],
        fm["outflow_rate_model"],        data_flow["outflow_rate_data"],
        valid_mask=data_flow.get("valid_origin_mask"),
    )
    var_lw = compute_var_log_occ_wage(wages, mu)
    gini   = compute_gini_occ_wages(wages, mu)
    out = {
        "spec":             tag,
        "var_log_occ_wage": float(var_lw),
        "gini":             float(gini),
        "mobility_rate":    float(fm["mobility_rate"]),
        **stats,
    }
    return out, fm


# ---------------------------------------------------------------------------

def main(max_iter_dest=25, max_iter_twoway=35):

    t0 = time.time()
    print("=" * 72)
    print("  Kappa-heterogeneity comparison — scalar vs dest vs two-way")
    print("=" * 72)

    # --- data -------------------------------------------------------------
    print("\n[1/5] Loading model data...")
    data = build_model_data(verbose=False)
    J = data["J"]
    d_so     = data["d_so"]
    data_emp = data["employment"]
    print(f"   J={J}  soc3 example={data['soc3_codes'][:3]}")

    # --- flow moments -----------------------------------------------------
    print("\n[2/5] Loading & aggregating flow data (6-digit -> SOC3)...")
    data_flow = load_flow_moments(data["soc3_codes"], data_emp, verbose=True)

    # --- benchmark scalar kappa ------------------------------------------
    if not os.path.exists(SMM_JSON):
        raise FileNotFoundError(
            f"Benchmark SMM parameters not found at {SMM_JSON}. "
            "Run calibrate_sdm_smm.py first.")
    with open(SMM_JSON) as f:
        smm = json.load(f)
    kappa_scalar = smm["tier3_smm"]["kappa"]
    tau_scalar   = smm["tier3_smm"]["tau"]
    print(f"\n[3/5] Scalar-κ benchmark (from SMM):  "
          f"κ={kappa_scalar:.4f}  τ={tau_scalar:.4f}")

    tier3 = {"kappa": kappa_scalar, "tau": tau_scalar}
    p_bench = SdmParams(data, tier3)
    kspec_s = KappaSpec("scalar", J, kappa=kappa_scalar)

    t1 = time.time()
    wages_s, B_s, V_s, mu_s, L_s, pol_s = solve_steady_state_vec(
        p_bench, kspec_s, d_so, data_emp,
        invert_amenities_flag=True,
        amenity_tol=1e-4, amenity_max_iter=80,
        max_iter=150, tol=1e-3, verbose=False)
    print(f"   scalar steady state: {time.time()-t1:.1f}s")
    sum_s, fm_s = _summary("scalar", wages_s, mu_s, pol_s,
                            data_emp, data_flow)
    print(f"   var(log w)={sum_s['var_log_occ_wage']:.4f}  "
          f"mob={sum_s['mobility_rate']:.4f}  "
          f"inflow corr={sum_s['inflow_corr']:+.3f}  "
          f"outflow corr={sum_s['outflow_corr']:+.3f}")

    # --- destination-specific κ_in ---------------------------------------
    print("\n[4/5] Calibrating destination-specific κ_in[o]...")
    p_dest = SdmParams(data, tier3)
    t1 = time.time()
    kspec_d, hist_d, state_d = calibrate_dest(
        p_dest, d_so, data_emp, data_flow,
        kappa_init=kappa_scalar,
        step=0.6, max_iter=max_iter_dest, tol=1e-2,
        reinvert_amenity_every=4,
        amenity_tol=2e-3, amenity_max_iter=30,
        verbose=True,
    )
    wages_d, B_d, V_d, mu_d, L_d, pol_d = state_d
    print(f"   dest calibration: {time.time()-t1:.1f}s "
          f"over {len(hist_d)} outer iters")
    sum_d, fm_d = _summary("dest", wages_d, mu_d, pol_d,
                            data_emp, data_flow)
    print(f"   var(log w)={sum_d['var_log_occ_wage']:.4f}  "
          f"mob={sum_d['mobility_rate']:.4f}  "
          f"inflow corr={sum_d['inflow_corr']:+.3f}  "
          f"outflow corr={sum_d['outflow_corr']:+.3f}")

    # --- two-way κ_out + κ_in (stabilized) ------------------------------
    print("\n[5/5] Calibrating two-way κ_out[s] + κ_in[o] (stabilized)...")
    p_two = SdmParams(data, tier3)
    t1 = time.time()
    kspec_t, hist_t, state_t = calibrate_twoway_stable(
        p_two, d_so, data_emp, data_flow,
        kappa_out_init=kappa_scalar / 2.0,
        kappa_in_init =kappa_scalar / 2.0,
        step_out=0.35, step_in=0.35,
        max_iter=max_iter_twoway, tol=5e-3,
        reinvert_amenity_every=4,
        amenity_tol=2e-3, amenity_max_iter=30,
        ema_window=4, adapt_step=True, gauss_seidel=True,
        verbose=True,
    )
    wages_t, B_t, V_t, mu_t, L_t, pol_t = state_t
    print(f"   twoway calibration: {time.time()-t1:.1f}s "
          f"over {len(hist_t)} outer iters")
    sum_t, fm_t = _summary("twoway", wages_t, mu_t, pol_t,
                            data_emp, data_flow)
    print(f"   var(log w)={sum_t['var_log_occ_wage']:.4f}  "
          f"mob={sum_t['mobility_rate']:.4f}  "
          f"inflow corr={sum_t['inflow_corr']:+.3f}  "
          f"outflow corr={sum_t['outflow_corr']:+.3f}")

    # --- comparison table ------------------------------------------------
    print("\n" + "=" * 72)
    print("  COMPARISON  (higher corr / lower rmse, log-dev = better fit)")
    print("=" * 72)
    cols = ["spec", "n_params", "var_log_occ_wage", "mobility_rate",
            "inflow_corr", "inflow_rmse", "inflow_max_log_dev",
            "outflow_corr", "outflow_rmse", "outflow_max_log_dev"]
    rows = []
    rows.append({**sum_s, "n_params": 1})
    rows.append({**sum_d, "n_params": J})
    rows.append({**sum_t, "n_params": 2 * J})

    header = (f"{'spec':<8} {'#p':>4}  {'Var(lnw)':>9} {'mob':>6}  "
              f"{'corrIn':>7} {'rmseIn':>8} {'|Δlog|In':>9}  "
              f"{'corrOut':>8} {'rmseOut':>8} {'|Δlog|Out':>10}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['spec']:<8} {r['n_params']:>4}  "
              f"{r['var_log_occ_wage']:>9.4f} {r['mobility_rate']:>6.3f}  "
              f"{r['inflow_corr']:>+7.3f} {r['inflow_rmse']:>8.4f} "
              f"{r['inflow_max_log_dev']:>9.3f}  "
              f"{r['outflow_corr']:>+8.3f} {r['outflow_rmse']:>8.4f} "
              f"{r['outflow_max_log_dev']:>10.3f}")
    print()

    # --- persist ---------------------------------------------------------
    out = {
        "benchmark_scalar": {
            "kappa": float(kappa_scalar),
            "tau":   float(tau_scalar),
            "summary": sum_s,
        },
        "dest": {
            "tau":      float(tau_scalar),
            "kappa_in": kspec_d.kappa_in.tolist(),
            "history":  hist_d,
            "summary":  sum_d,
        },
        "twoway": {
            "tau":       float(tau_scalar),
            "kappa_out": kspec_t.kappa_out.tolist(),
            "kappa_in":  kspec_t.kappa_in.tolist(),
            "history":   hist_t,
            "summary":   sum_t,
        },
        "soc3_codes": data["soc3_codes"],
        "data_moments": {
            "inflow_share_switch_data":
                data_flow["inflow_share_switch_data"].tolist(),
            "outflow_rate_data":
                data_flow["outflow_rate_data"].tolist(),
        },
        "elapsed_seconds": time.time() - t0,
    }
    out_path = os.path.join(OUTPUT_DIR, "kappa_heterogeneity_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] Wrote {out_path}")
    print(f"[done] Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    import sys
    # Allow short demo runs: python compare_kappa_specs.py --quick
    quick = ("--quick" in sys.argv)
    main(max_iter_dest=(6 if quick else 25),
         max_iter_twoway=(6 if quick else 30))
