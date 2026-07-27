"""
AI transition path — dynamic adjustment between pre- and post-AI steady
states, with κ held structural (same κ vectors in pre and post).

Runs the transition for all three κ specifications:
    scalar / dest / twoway.

Algorithm (proper forward-looking transition, vectorized)
---------------------------------------------------------
For a given wage path {w_t}_{t=0..T-1} and terminal value V_T = V_post:

    (1) Backward induction of V_t, t = T-1, ..., 0:
            v_all[s, o] = (log w_{o,t} + a_o - κ·d_{so}·I{o≠s}
                           + β V_{t+1}(o)) / τ
            V_t(s)      = τ · logsumexp_o v_all[s, o]
            π_t(o | s)  = softmax_o v_all[s, o]

    (2) Forward sweep of μ_t from μ_pre:
            μ_{t+1}     = π_t^T @ μ_t

    (3) Implied wages from CES demand:
            w_t^{new}   = B · μ_{t+1}^{-1/σ}        (matches timing of
                                                     existing solve_transition)

    (4) Damped update of wage path:
            w_t ← w_t + ω · (w_t^{new} - w_t)

The old `solve_transition` in simple_dynamic_model.py calls `solve_vf`
(iterated to convergence) at every period t.  That treats each w_t as if
it were a permanent steady state — myopic agents, no forward-looking
discounting of future wage changes.  Our backward induction is:
    (i)  cheaper: 1 Bellman step per period per outer iter, versus ~20
         iterations per period,
    (ii) correct: V_t depends on the FUTURE path of w.

Vectorized inner loop: all J×J tensor ops are numpy broadcasts.  The
transition is ~1 s per outer iter at T_trans=80, J=94 on a laptop.
"""

import os
import json
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import paths_override  # noqa: F401

from simple_dynamic_model import (
    build_model_data, SdmParams,
    compute_var_log_occ_wage, compute_gini_occ_wages,
    compute_L_eff, compute_wages,
    compute_stationary_distribution,
    invert_B, logsumexp,
)
from sdm_kappa_vec import (
    KappaSpec, solve_steady_state_vec, solve_vf_vec, build_cost_matrix,
    compute_flow_moments,
)


THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(THIS_DIR, "sdm_output")
KAPPA_JSON = os.path.join(OUTPUT_DIR, "kappa_heterogeneity_results.json")


# ---------------------------------------------------------------------------
# Core transition solver (generic κ spec via cost matrix)
# ---------------------------------------------------------------------------

def solve_transition_vec(p, kspec, d_post, a_o, B,
                         wages_pre, wages_post, mu_pre, V_post,
                         T_trans=80, max_outer=60, tol=5e-4,
                         damping=0.20, verbose=True):
    """
    Forward-looking transition path with vector κ.

    Parameters
    ----------
    p           : SdmParams  (only τ, β, σ, J, beta_eff used)
    kspec       : KappaSpec  (scalar / dest / twoway — same pre and post)
    d_post      : (J, J)     post-AI retraining distances (in years)
    a_o         : (J,)       amenities (held at pre-AI values)
    B           : (J,)       productivity shifters (held at pre-AI values)
    wages_pre   : (J,)       pre-AI steady-state wages
    wages_post  : (J,)       post-AI steady-state wages  (terminal boundary)
    mu_pre      : (J,)       pre-AI steady-state distribution (initial)
    V_post      : (J,)       post-AI steady-state value (terminal boundary)

    Returns a dict with the full converged paths.
    """
    J     = p.J
    tau   = p.tau
    beta  = p.beta_eff
    sigma = p.sigma

    # Set amenities on p so subsequent logic is consistent (not strictly
    # needed for pure transition, but useful).
    p.a_o = a_o.copy()

    # Precompute cost matrix (doesn't change over time since κ and d_post
    # are fixed along the transition).
    cost = build_cost_matrix(kspec, d_post)   # (J, J)

    # Initialise wage path by linear interpolation pre → post over the
    # first half, then flat at post-AI.
    wage_path = np.zeros((T_trans, J))
    half = max(T_trans // 2, 1)
    for t in range(T_trans):
        frac = min(t / half, 1.0)
        wage_path[t] = (1 - frac) * wages_pre + frac * wages_post

    # Workspace tensors
    V_path      = np.zeros((T_trans + 1, J))
    V_path[-1]  = V_post                                   # terminal
    policy_path = np.zeros((T_trans, J, J))
    mu_path     = np.zeros((T_trans + 1, J))
    mu_path[0]  = mu_pre                                   # initial

    prev_max_dw = None

    for outer in range(max_outer):
        t0 = time.time()
        log_w_path = np.log(np.maximum(wage_path, 1e-10))   # (T, J)

        # -------------------------------------------------------------------
        # 1. BACKWARD INDUCTION of V and policy  (single pass, vectorized)
        # -------------------------------------------------------------------
        V_next = V_post.copy()
        for t in range(T_trans - 1, -1, -1):
            # v_all[s, o] = (log w_{o,t} + a_o - cost[s,o] + β V_{t+1}(o)) / τ
            v_all = (log_w_path[t][None, :] + a_o[None, :] - cost
                     + beta * V_next[None, :]) / tau                 # (J, J)
            V_t = tau * logsumexp(v_all, axis=1)                     # (J,)
            v_max = v_all.max(axis=1, keepdims=True)
            exp_v = np.exp(v_all - v_max)
            policy_t = exp_v / exp_v.sum(axis=1, keepdims=True)      # (J, J)

            V_path[t]      = V_t
            policy_path[t] = policy_t
            V_next = V_t

        # -------------------------------------------------------------------
        # 2. FORWARD SWEEP of μ  (no entry; incumbents only)
        # -------------------------------------------------------------------
        mu_t = mu_pre.copy()
        wage_path_new = np.zeros_like(wage_path)
        for t in range(T_trans):
            mu_new = policy_path[t].T @ mu_t
            s = mu_new.sum()
            if s > 0:
                mu_new = mu_new / s
            mu_path[t + 1] = mu_new
            # Wage at time t consistent with L_t+1 = mu_new  (keeps the timing
            # convention used by simple_dynamic_model.solve_transition)
            L_eff = compute_L_eff(mu_new, p)
            wage_path_new[t] = compute_wages(B, L_eff, sigma)
            mu_t = mu_new

        # -------------------------------------------------------------------
        # 3. Convergence check + damped update
        # -------------------------------------------------------------------
        max_dw = float(np.max(np.abs(wage_path_new - wage_path)
                              / np.maximum(wage_path, 1e-10)))

        # Adaptive damping (shrink if getting worse, expand if improving)
        if prev_max_dw is not None:
            if max_dw > prev_max_dw * 1.02:
                damping = max(damping * 0.7, 0.02)
            elif max_dw < prev_max_dw * 0.8:
                damping = min(damping * 1.1, 0.40)
        prev_max_dw = max_dw

        if verbose:
            print(f"    [trans] outer={outer+1:2d}/{max_outer}  "
                  f"maxΔw={max_dw:.5f}  damp={damping:.3f}  "
                  f"[{time.time()-t0:.2f}s]")

        if max_dw < tol:
            if verbose:
                print(f"    [trans] converged after {outer+1} outer iters")
            break

        update = damping * (wage_path_new - wage_path)
        # Clip relative step size
        update = np.clip(update, -0.5 * wage_path, 0.5 * wage_path)
        wage_path = wage_path + update

    # Recompute diagnostics on final path
    var_log_w_path = np.array(
        [compute_var_log_occ_wage(wage_path[t], mu_path[t + 1])
         for t in range(T_trans)])
    gini_path = np.array(
        [compute_gini_occ_wages(wage_path[t], mu_path[t + 1])
         for t in range(T_trans)])
    mob_path = np.array(
        [1.0 - policy_path[t][np.arange(J), np.arange(J)]
              .dot(mu_path[t]) / max(mu_path[t].sum(), 1e-15)
         for t in range(T_trans)])
    # Aggregate output path
    def _Y(L):
        return (np.sum(B**(1.0 / sigma)
                       * np.maximum(L, 1e-10)**((sigma - 1.0) / sigma))
                ** (sigma / (sigma - 1.0)))
    Y_path = np.array([_Y(mu_path[t + 1]) for t in range(T_trans)])

    return {
        "wage_path":      wage_path,
        "mu_path":        mu_path,
        "policy_path":    policy_path,
        "V_path":         V_path,
        "var_log_w_path": var_log_w_path,
        "gini_path":      gini_path,
        "mob_path":       mob_path,
        "Y_path":         Y_path,
        "converged":      max_dw < tol,
        "final_maxdw":    max_dw,
    }


# ---------------------------------------------------------------------------
# Per-spec pre/post steady states + transition
# ---------------------------------------------------------------------------

def run_spec(tag, kspec, data, tier3, T_trans=80, max_outer=60, verbose=True):
    """Solve pre-SS, post-SS, and the transition path for one κ spec."""
    J       = data["J"]
    d_pre   = data["d_so"]
    d_post  = data["d_so_AI"]
    sigma   = data["sigma"]
    data_emp = data["employment"]

    p = SdmParams(data, tier3)

    print(f"\n[{tag}] --- pre-AI steady state ---")
    t0 = time.time()
    wages_pre, B_pre, V_pre, mu_pre, L_pre, pol_pre = solve_steady_state_vec(
        p, kspec, d_pre, data_emp,
        invert_amenities_flag=True,
        amenity_tol=1e-4, amenity_max_iter=80,
        max_iter=200, tol=1e-4, verbose=False,
    )
    a_o_pre = p.a_o.copy()
    print(f"[{tag}]   pre-SS solved in {time.time()-t0:.1f}s")

    print(f"[{tag}] --- post-AI steady state ---")
    t0 = time.time()
    p.a_o = a_o_pre.copy()
    wages_post, B_post, V_post, mu_post, L_post, pol_post = \
        solve_steady_state_vec(
            p, kspec, d_post, data_emp,
            invert_amenities_flag=False,
            B_ext=B_pre, wages_init=wages_pre,
            max_iter=500, tol=1e-5, damping=0.2, verbose=False,
        )
    print(f"[{tag}]   post-SS solved in {time.time()-t0:.1f}s")

    var_pre  = compute_var_log_occ_wage(wages_pre,  mu_pre)
    var_post = compute_var_log_occ_wage(wages_post, mu_post)
    gini_pre  = compute_gini_occ_wages(wages_pre,  mu_pre)
    gini_post = compute_gini_occ_wages(wages_post, mu_post)
    print(f"[{tag}]   Var(lnw): {var_pre:.4f} → {var_post:.4f}   "
          f"Gini: {gini_pre:.4f} → {gini_post:.4f}")

    print(f"[{tag}] --- transition path (T={T_trans}) ---")
    t0 = time.time()
    p.a_o = a_o_pre.copy()
    trans = solve_transition_vec(
        p, kspec, d_post, a_o_pre, B_pre,
        wages_pre, wages_post, mu_pre, V_post,
        T_trans=T_trans, max_outer=max_outer,
        tol=5e-4, damping=0.20, verbose=verbose,
    )
    print(f"[{tag}]   transition solved in {time.time()-t0:.1f}s  "
          f"(converged={trans['converged']}, "
          f"final max|Δw|={trans['final_maxdw']:.5f})")

    return {
        "tag":        tag,
        "a_o":        a_o_pre,
        "B":          B_pre,
        "pre":  {"wages": wages_pre,  "mu": mu_pre,  "V": V_pre,
                 "var_log_w": var_pre,  "gini": gini_pre},
        "post": {"wages": wages_post, "mu": mu_post, "V": V_post,
                 "var_log_w": var_post, "gini": gini_post},
        "trans": trans,
    }


def main(T_trans=80, max_outer=50):
    t_tot = time.time()
    print("=" * 72)
    print("  AI TRANSITION PATH — forward-looking, vectorized backward")
    print("  induction;  κ structural (same pre and post).")
    print("=" * 72)

    data = build_model_data(verbose=False)
    J = data["J"]
    print(f"J={J}  σ={data['sigma']}  β={data['beta']}  T_trans={T_trans}")

    # Load calibrated specs
    if not os.path.exists(KAPPA_JSON):
        raise FileNotFoundError(KAPPA_JSON)
    with open(KAPPA_JSON) as f:
        cal = json.load(f)
    kappa_scalar = cal["benchmark_scalar"]["kappa"]
    tau_scalar   = cal["benchmark_scalar"]["tau"]
    kappa_in_d   = np.array(cal["dest"]["kappa_in"])
    kappa_out_t  = np.array(cal["twoway"]["kappa_out"])
    kappa_in_t   = np.array(cal["twoway"]["kappa_in"])
    tier3 = {"kappa": kappa_scalar, "tau": tau_scalar}

    specs = [
        ("scalar", KappaSpec("scalar", J, kappa=kappa_scalar)),
        ("dest",   KappaSpec("dest",   J, kappa_in=kappa_in_d)),
        ("twoway", KappaSpec("twoway", J, kappa_out=kappa_out_t,
                              kappa_in=kappa_in_t)),
    ]

    results = {}
    for tag, kspec in specs:
        res = run_spec(tag, kspec, data, tier3,
                       T_trans=T_trans, max_outer=max_outer, verbose=True)
        results[tag] = res

    # ---- Summary table: horizon to half-adjustment ----
    print("\n" + "=" * 72)
    print("  TRANSITION SUMMARY")
    print("=" * 72)
    hdr = (f"{'spec':<8}  {'Var(lnw)_pre':>12} {'Var(lnw)_post':>14}  "
           f"{'T½(var)':>8}  {'T½(gini)':>9}  {'T½(mob)':>8}  {'ΔlogY peak':>10}")
    print(hdr)
    print("-" * len(hdr))

    def _half_life(path, pre_val, post_val):
        """Periods until path reaches halfway between pre and post."""
        target = pre_val + 0.5 * (post_val - pre_val)
        if post_val >= pre_val:
            hits = np.where(path >= target)[0]
        else:
            hits = np.where(path <= target)[0]
        return int(hits[0]) if len(hits) else -1

    summary_rows = []
    for tag in ["scalar", "dest", "twoway"]:
        r = results[tag]
        vp = np.array(r["trans"]["var_log_w_path"])
        gp = np.array(r["trans"]["gini_path"])
        mp = np.array(r["trans"]["mob_path"])
        Yp = np.array(r["trans"]["Y_path"])

        # Compute pre-SS Y from mu_pre for normalization
        mu0 = r["pre"]["mu"]
        B   = r["B"]; sigma = data["sigma"]
        Y0  = np.sum(B**(1/sigma)
                     * np.maximum(mu0, 1e-10)**((sigma-1)/sigma)
                     )**(sigma/(sigma-1))
        dlogY = np.log(Yp / Y0)

        T_half_var  = _half_life(vp, r["pre"]["var_log_w"],
                                  r["post"]["var_log_w"])
        T_half_gini = _half_life(gp, r["pre"]["gini"],
                                  r["post"]["gini"])
        # mobility pre is fm on pre SS — use trans[mob_path][0] as proxy
        mob_pre = float(mp[0])
        mob_post = float(mp[-1])
        T_half_mob = _half_life(mp, mob_pre, mob_post)

        print(f"{tag:<8}  {r['pre']['var_log_w']:>12.4f} "
              f"{r['post']['var_log_w']:>14.4f}  "
              f"{T_half_var:>8d}  {T_half_gini:>9d}  {T_half_mob:>8d}  "
              f"{dlogY.min() if dlogY.min() < 0 else dlogY.max():>+10.4f}")
        summary_rows.append({
            "spec": tag,
            "var_log_w_pre":  float(r["pre"]["var_log_w"]),
            "var_log_w_post": float(r["post"]["var_log_w"]),
            "gini_pre":       float(r["pre"]["gini"]),
            "gini_post":      float(r["post"]["gini"]),
            "T_half_var":     T_half_var,
            "T_half_gini":    T_half_gini,
            "T_half_mob":     T_half_mob,
            "dlogY_final":    float(dlogY[-1]),
            "dlogY_peak":     float(dlogY.max()),
            "dlogY_trough":   float(dlogY.min()),
            "mob_pre":        mob_pre,
            "mob_post":       mob_post,
            "mob_peak":       float(mp.max()),
        })
    print()

    # ---- Persist results ----
    out = {
        "T_trans": T_trans,
        "soc3_codes": data["soc3_codes"],
        "summary":  summary_rows,
        "by_spec":  {},
    }
    for tag, r in results.items():
        # Pre-SS and post-SS policies, computed once from the solved
        # value functions so analysis downstream has full flow matrices
        # without needing to re-solve.
        kspec = next(ks for (tname, ks) in specs if tname == tag)
        p_tmp = SdmParams(data, tier3)
        p_tmp.a_o = r["a_o"].copy()
        _, pol_pre  = solve_vf_vec(r["pre"]["wages"],  p_tmp, kspec,
                                    data["d_so"],
                                    return_policy=True, tol=1e-6)
        _, pol_post = solve_vf_vec(r["post"]["wages"], p_tmp, kspec,
                                    data["d_so_AI"],
                                    return_policy=True, tol=1e-6)
        out["by_spec"][tag] = {
            "wages_pre":   r["pre"]["wages"].tolist(),
            "wages_post":  r["post"]["wages"].tolist(),
            "mu_pre":      r["pre"]["mu"].tolist(),
            "mu_post":     r["post"]["mu"].tolist(),
            "policy_pre":  pol_pre.tolist(),
            "policy_post": pol_post.tolist(),
            "var_log_w_path": r["trans"]["var_log_w_path"].tolist(),
            "gini_path":       r["trans"]["gini_path"].tolist(),
            "mob_path":        r["trans"]["mob_path"].tolist(),
            "Y_path":          r["trans"]["Y_path"].tolist(),
            "mu_path":         r["trans"]["mu_path"].tolist(),
            "wage_path":       r["trans"]["wage_path"].tolist(),
            "converged":       bool(r["trans"]["converged"]),
            "final_maxdw":     float(r["trans"]["final_maxdw"]),
        }
    out_path = os.path.join(OUTPUT_DIR, "ai_transition_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] Wrote {out_path}")
    print(f"[done] Total elapsed: {time.time()-t_tot:.1f}s")


if __name__ == "__main__":
    import sys
    T = 60
    if "--long" in sys.argv:  T = 100
    if "--short" in sys.argv: T = 40
    main(T_trans=T, max_outer=50)
