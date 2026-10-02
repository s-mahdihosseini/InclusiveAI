"""
loglin_ge.py -- the LOG-LINEARIZED sufficient-statistic system of the note,
solved at a point's observable statistics. Nothing here knows the primitives
(alpha_i, A_i, Om_j, T_mi): only cost shares, budget shares, labor-income
shares, portfolio shares, and the elasticities enter -- that is the point.

The system, per unit GPT shock da = dln A_g (w_g = 1 numeraire, dln p_g = -da):

  (S)   dln p_i = (1-b_i) dln w_i + b_i dln p_g
  (D)   dln C_i = sD_i (dln I - dln p_i) + xi_i (1-sD_i) dln U + dln Gamma
  (LD)  dln L_i = dln C_i - sig_i b_i (dln w_i - dln p_g)     [Y_i = C_i]
  (LS)  dln L_i = 0                                (immobile, kappa = 0)
        dln L_i = k_i dln w_i - sum_j R_ij dln w_j (mobile: ripple matrix)

with GOOD-SPECIFIC demand elasticities sD_i, closed by the aggregate relations:

  income   dln I = sum_i s^L_i (dln w_i + dln L_i)      [+ s_g dln w_g = 0]
  utility  dln U = (dln I - sum_j om_j dln p_j) / xibar,  xibar = sum om_j xi_j
  Gamma    dln Gamma = -sum_k om_k (1-sD_k) (dln p_k - dln I + xi_k dln U)
           (the common cross-product index; identically zero when sD_k common)
  GPT      market clearing holds by Walras (checked, not imposed)

Every weight in the Gamma row is observable: budget shares times (1 - sD_k).

Real (base-consumption-bundle numeraire) changes subtract dln P_c = theta @ dp,
theta_j = current-price base-quantity shares (= budget shares at the base point).
"""
import numpy as np
from exact_ge import (occ, c, sig, sigD, xi, N, kap, Pi0, L0, w0, Lg,
                      structure_at, equilibrium)

# ------------------------------------------------- statistics at a state
def stats_from_state(s, w):
    """Observable sufficient statistics at a state produced by structure_at."""
    Ls, I = s["Ls"], s["I"]
    sL    = w * Ls / I                       # occupational labor-income shares
    mu    = (N[:, None] * s["Pi"]) / Ls[None, :]   # group m's share of occ i
    k     = (mu * kap[:, None]).sum(0)
    R     = np.einsum("m,mi,mj->ij", kap, mu, s["Pi"])
    theta = s["p"] * c / (s["p"] @ c)        # base-bundle deflator weights
    return dict(b=s["b"], sL=sL, om=s["om"], xibar=s["om"] @ xi,
                k=k, R=R, theta=theta, lam_g=s["lam_g"],
                xsh=s["X"] / s["X"].sum())

def stats_base(mobile=True):
    return stats_from_state(structure_at(w0, 1.0, mobile), w0)

# ------------------------------------------------- the linear system
def solve_loglin(st, mobile=True, da=1.0):
    """Solve the four-equation block + aggregates for dln w per unit shock,
    then scale by da. Returns dw, dL, dp, dI, dU, dGam, dPc, real versions."""
    b, sL, om, xb = st["b"], st["sL"], st["om"], st["xibar"]
    n = len(b)
    K = np.diag(st["k"]) - st["R"] if mobile else np.zeros((n, n))
    Aw, ap = np.diag(1.0 - b), -b                 # dp = Aw dw + ap  (da = 1)
    iota = sL + K.T @ sL if mobile else sL        # dI = iota @ dw
    uw = (iota - om @ Aw) / xb                    # dU = uw @ dw + ug
    ug = -(om @ ap) / xb
    nu = om * (1.0 - sigD)                        # Gamma-row weights
    gw = -(nu @ Aw) + nu.sum() * iota - (nu @ xi) * uw   # dGam = gw @ dw + gg
    gg = -(nu @ ap) - (nu @ xi) * ug
    Cw = (np.outer(sigD, iota) - np.diag(sigD) @ Aw
          + np.outer((1 - sigD) * xi, uw) + np.outer(np.ones(n), gw))
    cg = -sigD * ap + (1 - sigD) * xi * ug + gg   # dC = Cw dw + cg
    M  = Cw - np.diag(sig * b) - K                # (LD) = (LS)
    dw = np.linalg.solve(M, sig * b - cg) * da
    dL  = K @ dw
    dp  = Aw @ dw + ap * da
    dI  = iota @ dw
    dU  = uw @ dw + ug * da
    dGam = gw @ dw + gg * da
    dPc = st["theta"] @ dp
    return dict(dw=dw, dL=dL, dp=dp, dI=dI, dU=dU, dGam=dGam, dPc=dPc,
                dw_real=dw - dPc, dI_real=dI - dPc)

# ------------------------------------------------- internal consistency
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    for mob, tag in ((False, "immobile"), (True, "mobile  ")):
        st = stats_base(mob)
        r  = solve_loglin(st, mob)
        print(f"tangent at base, {tag}: dlnw/da (real) = {r['dw_real'].round(4)}"
              f"  dlnL/da = {r['dL'].round(4)}")
        # Hulten check (immobile): real income growth = Domar weight, exactly
        if not mob:
            print("  Hulten: dlnI_real/da = %.6f  vs  Domar lam_g = %.6f"
                  % (r["dI_real"], st["lam_g"]))
            assert abs(r["dI_real"] - st["lam_g"]) < 1e-12
        # Walras check: GPT demand growth = GPT supply growth (= da), using
        # dlnX_i = dlnL_i + sig_i (dlnw_i - dlnp_g)  from the CES input ratio
        dX = r["dL"] + sig * (r["dw"] + 1.0)
        gap = st["xsh"] @ dX - 1.0
        print("  GPT market (Walras): sum xsh dlnX - da = %.2e" % gap)
        assert abs(gap) < 1e-12
    print("loglin_ge: ALL INTERNAL CHECKS PASSED")
