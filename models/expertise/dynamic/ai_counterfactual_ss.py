"""
AI counterfactual: post-AI long-run steady state.

Treatment protocol (per model description, Section 'AI Shock'):
    • Pre-AI retraining distances   d_so      are replaced by d_so_AI.
    • Productivity shifters B_o                remain at their pre-AI values.
    • Amenities a_o                            remain at their pre-AI values.
    • κ vectors (κ_out, κ_in, or scalar κ)    remain at their pre-AI values
                                               (κ is structural by assumption).
    • τ                                        remains at the pre-AI value.
    • The new long-run equilibrium is the fixed point of
        w_o = B_o · L_o(w)^{-1/σ}              under d_so_AI.

We run the counterfactual for THREE kappa specifications (same three we
calibrated) so the scalar/dest/twoway comparison carries through to the
counterfactual:
    (A) scalar      κ           (benchmark from SMM)
    (B) destination κ_in[o]    (from calibrate_dest)
    (C) two-way    κ_out[s] + κ_in[o]  (from calibrate_twoway_stable)

This script loads all three calibrated specs from
    sdm_output/kappa_heterogeneity_results.json
re-creates the pre-AI steady state for each (to get a_o and B_o under
each spec), and then re-solves the post-AI steady state.

Reported quantities, per spec:
    • Δ log w_o     — percent wage change by occupation
    • Δ L_o / L_o   — percent employment change by occupation
    • Var(log w)    pre vs post
    • Gini          pre vs post
    • Mobility rate pre vs post
    • Dispersion of wage changes (p10, p50, p90 of Δ log w)
    • Aggregate output change

Transition-path computation is deferred to a follow-up script.
"""

import os
import json
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import paths_override  # noqa: F401

from simple_dynamic_model import (
    build_model_data,
    SdmParams,
    compute_var_log_occ_wage,
    compute_gini_occ_wages,
    compute_L_eff,
    compute_wages,
    compute_stationary_distribution,
    invert_B,
)
from flow_data import load_flow_moments
from sdm_kappa_vec import (
    KappaSpec,
    solve_steady_state_vec,
    solve_vf_vec,
    compute_flow_moments,
)


THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(THIS_DIR, "sdm_output")
KAPPA_JSON = os.path.join(OUTPUT_DIR, "kappa_heterogeneity_results.json")


def solve_pre_and_post(p, kspec, d_pre, d_post, data_emp, verbose=False):
    """
    Solve pre-AI (with amenity & B inversion) and then post-AI steady state
    (with amenities and B held fixed at pre-AI values).

    Returns a dict with pre- and post-state + key aggregates.
    """
    # --- pre-AI ---
    wages_pre, B_pre, V_pre, mu_pre, L_pre, pol_pre = solve_steady_state_vec(
        p, kspec, d_pre, data_emp,
        invert_amenities_flag=True,
        amenity_tol=1e-4, amenity_max_iter=80,
        max_iter=200, tol=1e-4, verbose=False,
    )
    a_o_pre = p.a_o.copy()

    # --- post-AI: freeze a_o and B, re-solve wages under d_post ---
    p.a_o = a_o_pre.copy()
    wages_post, B_post, V_post, mu_post, L_post, pol_post = \
        solve_steady_state_vec(
            p, kspec, d_post, data_emp,
            invert_amenities_flag=False,
            B_ext=B_pre,
            wages_init=wages_pre,
            max_iter=500, tol=1e-5, damping=0.2, verbose=False,
        )

    # --- moments ---
    fm_pre  = compute_flow_moments(pol_pre,  mu_pre,  data_emp)
    fm_post = compute_flow_moments(pol_post, mu_post, data_emp)

    return {
        "a_o":        a_o_pre,
        "B":          B_pre,
        "pre": {
            "wages":  wages_pre, "mu": mu_pre,  "policy": pol_pre,
            "V":      V_pre,     "L_eff": L_pre,
            "var_log_occ_wage": compute_var_log_occ_wage(wages_pre, mu_pre),
            "gini":             compute_gini_occ_wages(wages_pre, mu_pre),
            "mobility_rate":    fm_pre["mobility_rate"],
        },
        "post": {
            "wages":  wages_post, "mu": mu_post, "policy": pol_post,
            "V":      V_post,    "L_eff": L_post,
            "var_log_occ_wage": compute_var_log_occ_wage(wages_post, mu_post),
            "gini":             compute_gini_occ_wages(wages_post, mu_post),
            "mobility_rate":    fm_post["mobility_rate"],
        },
    }


def _summary_row(tag, res, data_emp, sigma):
    pre, post = res["pre"], res["post"]
    # Percent changes
    dlog_w = np.log(post["wages"]) - np.log(pre["wages"])
    dL_pct = (post["L_eff"] - pre["L_eff"]) / np.maximum(pre["L_eff"], 1e-10)

    # Aggregate output: Y = [Σ B_o^{1/σ} L_o^{(σ-1)/σ}]^{σ/(σ-1)}
    def _Y(B, L):
        return (np.sum(B**(1.0 / sigma)
                       * np.maximum(L, 1e-10)**((sigma - 1.0) / sigma))
                ** (sigma / (sigma - 1.0)))
    Y_pre  = _Y(res["B"], pre["L_eff"])
    Y_post = _Y(res["B"], post["L_eff"])
    dlogY  = float(np.log(Y_post) - np.log(Y_pre))

    # Employment-weighted mean wage change (welfare-relevant)
    w_share_pre = data_emp / data_emp.sum()
    mean_dlog_w = float((w_share_pre * dlog_w).sum())

    return {
        "spec": tag,
        "var_log_w_pre":  float(pre["var_log_occ_wage"]),
        "var_log_w_post": float(post["var_log_occ_wage"]),
        "d_var_log_w":    float(post["var_log_occ_wage"]
                                - pre["var_log_occ_wage"]),
        "gini_pre":       float(pre["gini"]),
        "gini_post":      float(post["gini"]),
        "d_gini":         float(post["gini"] - pre["gini"]),
        "mob_pre":        float(pre["mobility_rate"]),
        "mob_post":       float(post["mobility_rate"]),
        "d_mob":          float(post["mobility_rate"] - pre["mobility_rate"]),
        "d_logY":         dlogY,
        "dlog_w_mean_empwt": mean_dlog_w,
        "dlog_w_p10":     float(np.percentile(dlog_w, 10)),
        "dlog_w_p50":     float(np.percentile(dlog_w, 50)),
        "dlog_w_p90":     float(np.percentile(dlog_w, 90)),
        "dlog_w_max":     float(dlog_w.max()),
        "dlog_w_min":     float(dlog_w.min()),
        "dL_pct_p10":     float(np.percentile(dL_pct, 10)),
        "dL_pct_p50":     float(np.percentile(dL_pct, 50)),
        "dL_pct_p90":     float(np.percentile(dL_pct, 90)),
        "dL_pct_max":     float(dL_pct.max()),
        "dL_pct_min":     float(dL_pct.min()),
    }


def main():
    t0 = time.time()
    print("=" * 72)
    print("  AI counterfactual (long-run steady state)")
    print("  d_so -> d_so_AI;  κ, τ, a_o, B  all fixed at pre-AI values")
    print("=" * 72)

    # --- load data ---
    data = build_model_data(verbose=False)
    J = data["J"]
    d_pre  = data["d_so"]
    d_post = data["d_so_AI"]
    sigma  = data["sigma"]
    data_emp = data["employment"]
    print(f"\nJ={J}  σ={sigma}")
    print(f"d_pre  range: {d_pre[d_pre>0].min():.2f} – {d_pre.max():.2f} yrs  "
          f"mean: {d_pre[d_pre>0].mean():.2f}")
    print(f"d_post range: {d_post[d_post>0].min():.2f} – {d_post.max():.2f} yrs  "
          f"mean: {d_post[d_post>0].mean():.2f}")
    print(f"AI reduction in d (off-diag mean): "
          f"{(d_pre - d_post)[d_pre > 0].mean():.2f} yrs  "
          f"({100 * (d_pre - d_post)[d_pre>0].mean() / d_pre[d_pre>0].mean():.1f}%)")

    # --- load calibrated κ specs ---
    if not os.path.exists(KAPPA_JSON):
        raise FileNotFoundError(
            f"{KAPPA_JSON} not found.  Run compare_kappa_specs.py first.")
    with open(KAPPA_JSON) as f:
        cal = json.load(f)

    kappa_scalar = cal["benchmark_scalar"]["kappa"]
    tau_scalar   = cal["benchmark_scalar"]["tau"]
    kappa_in_d   = np.array(cal["dest"]["kappa_in"])
    kappa_out_t  = np.array(cal["twoway"]["kappa_out"])
    kappa_in_t   = np.array(cal["twoway"]["kappa_in"])
    print(f"\nCalibrated params loaded from {os.path.basename(KAPPA_JSON)}:")
    print(f"  scalar:  κ={kappa_scalar:.4f}  τ={tau_scalar:.4f}")
    print(f"  dest:    κ_in ∈ [{kappa_in_d.min():.3f}, "
          f"{kappa_in_d.max():.3f}]  (J={len(kappa_in_d)})")
    print(f"  twoway:  κ_out ∈ [{kappa_out_t.min():.3f}, "
          f"{kappa_out_t.max():.3f}]  κ_in ∈ [{kappa_in_t.min():.3f}, "
          f"{kappa_in_t.max():.3f}]")

    # --- run counterfactual for each spec ---
    rows = []
    state_by_spec = {}
    tier3 = {"kappa": kappa_scalar, "tau": tau_scalar}

    for tag, kspec in [
        ("scalar", KappaSpec("scalar", J, kappa=kappa_scalar)),
        ("dest",   KappaSpec("dest",   J, kappa_in=kappa_in_d)),
        ("twoway", KappaSpec("twoway", J, kappa_out=kappa_out_t,
                              kappa_in=kappa_in_t)),
    ]:
        print(f"\n--- spec: {tag} ---")
        t1 = time.time()
        p = SdmParams(data, tier3)
        res = solve_pre_and_post(p, kspec, d_pre, d_post, data_emp)
        row = _summary_row(tag, res, data_emp, sigma)
        rows.append(row)
        state_by_spec[tag] = res
        dt = time.time() - t1
        print(f"  solved pre & post in {dt:.1f}s")
        print(f"  Var(log w): {row['var_log_w_pre']:.4f} → "
              f"{row['var_log_w_post']:.4f}  (Δ={row['d_var_log_w']:+.4f})")
        print(f"  Gini:       {row['gini_pre']:.4f} → "
              f"{row['gini_post']:.4f}  (Δ={row['d_gini']:+.4f})")
        print(f"  Mobility:   {row['mob_pre']:.4f} → "
              f"{row['mob_post']:.4f}  (Δ={row['d_mob']:+.4f})")
        print(f"  Δ log Y:    {row['d_logY']:+.4f}  "
              f"(= {100*row['d_logY']:+.2f}% output change)")
        print(f"  Δ log w  empwt-mean: {100*row['dlog_w_mean_empwt']:+.2f}%  "
              f"p10/p50/p90: {100*row['dlog_w_p10']:+.2f}% / "
              f"{100*row['dlog_w_p50']:+.2f}% / "
              f"{100*row['dlog_w_p90']:+.2f}%")
        print(f"  Δ L    : p10/p50/p90: {100*row['dL_pct_p10']:+.2f}% / "
              f"{100*row['dL_pct_p50']:+.2f}% / "
              f"{100*row['dL_pct_p90']:+.2f}%")

    # --- comparison table ---
    print("\n" + "=" * 72)
    print("  SUMMARY — AI steady-state counterfactual by κ specification")
    print("=" * 72)
    header = (f"{'spec':<8}  {'Var(lnw) pre→post (Δ)':<28}  "
              f"{'Gini pre→post (Δ)':<24}  "
              f"{'mob pre→post':<18}  {'ΔlogY':>7}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['spec']:<8}  "
              f"{r['var_log_w_pre']:.4f}→{r['var_log_w_post']:.4f} "
              f"({r['d_var_log_w']:+.4f})    "
              f"{r['gini_pre']:.4f}→{r['gini_post']:.4f} "
              f"({r['d_gini']:+.4f})  "
              f"{r['mob_pre']:.3f}→{r['mob_post']:.3f} "
              f"({r['d_mob']:+.3f})  "
              f"{r['d_logY']:+.4f}")
    print()

    # --- persist ---
    out = {
        "rows": rows,
        "soc3_codes": data["soc3_codes"],
        "soc3_names": data["soc3_names"],
        "d_pre_mean":  float(d_pre[d_pre > 0].mean()),
        "d_post_mean": float(d_post[d_post > 0].mean()),
        "elapsed_seconds": time.time() - t0,
        # per-spec per-occ arrays for downstream plots
        "by_spec": {},
    }
    for tag, res in state_by_spec.items():
        dlog_w = (np.log(res["post"]["wages"])
                  - np.log(res["pre"]["wages"])).tolist()
        dL = ((res["post"]["L_eff"] - res["pre"]["L_eff"])
              / np.maximum(res["pre"]["L_eff"], 1e-12)).tolist()
        out["by_spec"][tag] = {
            "wages_pre":  res["pre"]["wages"].tolist(),
            "wages_post": res["post"]["wages"].tolist(),
            "mu_pre":     res["pre"]["mu"].tolist(),
            "mu_post":    res["post"]["mu"].tolist(),
            "dlog_w":     dlog_w,
            "dL_pct":     dL,
        }

    out_path = os.path.join(OUTPUT_DIR, "ai_counterfactual_ss.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] Wrote {out_path}")
    print(f"[done] Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
