"""
hetero_households.py -- ONE HOUSEHOLD PER WORKER GROUP: the distributional
demand feedback, exact and in summary statistics.

The economy is exact_ge.py's, with one change: instead of a pooled household,
each worker group m is a household earning its own labor income
    I_m = sum_i w_i L_mi          (+ the GPT sector's wage bill for college,
                                   whose workers staff the GPT sector),
and consuming out of it with the SAME implicitly additive preferences. Because
preferences are non-homothetic, households at different income levels buy
different baskets even with identical preferences; so when a GPT shock moves
group incomes differently (they work in different occupations), the
DISTRIBUTION of income feeds back into demand composition and hence into
wages. That feedback loop is the object of this file.

The log-linear system needs two new observable matrices and nothing else:

  psi_mi = w_i L_mi / I_m   (earnings incidence: who earns where)
  e_mi   = C_mi / C_i       (expenditure incidence: who buys what)

Per-group demand block (household-specific I_m, U_m, Gamma_m):

  dln C_mi = sD_i (dln I_m - dln p_i) + xi_i (1-sD_i) dln U_m + dln Gamma_m
  dln U_m  = (dln I_m - sum_j om_mj dln p_j) / xibar_m
  dln Gam_m= -sum_k om_mk (1-sD_k)(dln p_k - dln I_m + xi_k dln U_m)
  dln I_m  = sum_i psi_mi (dln w_i + dln L_mi)      [dln L_mi = kap_m(dw_i - dW_m)]
  dln C_i  = sum_m e_mi dln C_mi                    [aggregation]

Everything else (production, pricing, labor demand/supply) is unchanged.
Validation: (V1) derivative check vs exact GE including per-group incomes and
utilities; (V3) RK4 path integration to the exact x8 equilibrium; and the SIZE
of the distributional feedback: per-group vs pooled, exact and first-order.
"""
import numpy as np
from scipy.optimize import fsolve, brentq
from exact_ge import (occ, c, b0, sig, sigD, xi, N, kap, Pi0, L0, w0, X0, Lg,
                      alpha, A, rho, T, logit_shares)
import exact_ge
import loglin_ge

np.set_printoptions(precision=6, suppress=True)
LN8 = np.log(8.0)

# ------------------------------------------------- base incomes & tastes
Lmi0 = N[:, None] * Pi0
Im0  = (w0[None, :] * Lmi0).sum(1)
Im0[2] += Lg                              # college owns the GPT wage bill
ym0  = Im0 / N                            # per-capita incomes

def demand_m(Om, p, Im, Nm):
    """One group's demand: Nm identical households, per-capita income Im/Nm."""
    y  = Im / Nm
    lp = np.log(p) - np.log(y)
    G  = lambda u: np.log((Om * np.exp((1 - sigD) * (lp + xi * u))).sum())
    u  = brentq(G, -60.0, 60.0, xtol=1e-15)
    g  = Om * np.exp((1 - sigD) * (lp + xi * u))
    den = ((1 - sigD) * g).sum()
    om = (1 - sigD) * g / den
    return u, om, om * Im / p, -np.log(den)   # lnU, shares, consumption, lnGam

def calibrate_Omega():
    """Om such that AGGREGATE base consumption equals c at p = 1."""
    Om = exact_ge.Om.copy()
    for _ in range(800):
        Cagg = sum(demand_m(Om, np.ones(4), Im0[m], N[m])[2] for m in range(3))
        if np.max(np.abs(Cagg - c)) < 1e-14:
            break
        Om *= c / Cagg
        Om /= Om.sum()
    return Om

Om_h = calibrate_Omega()

# ------------------------------------------------- exact equilibrium
def structure_h(w, Ag, mobile=True):
    pg    = 1.0 / Ag
    Pi    = logit_shares(w) if mobile else Pi0
    Lmi   = N[:, None] * Pi
    Ls    = Lmi.sum(0)
    ratio = ((alpha * w) / ((1 - alpha) * pg)) ** sig
    y1    = A * ((1 - alpha) + alpha * ratio ** rho) ** (1.0 / rho)
    p     = (w + pg * ratio) / y1
    X, Y  = ratio * Ls, y1 * Ls
    Im    = (w[None, :] * Lmi).sum(1); Im = Im.copy(); Im[2] += Lg
    lU  = np.zeros(3); omh = np.zeros((3, 4)); Cm = np.zeros((3, 4)); lG = np.zeros(3)
    for m in range(3):
        lU[m], omh[m], Cm[m], lG[m] = demand_m(Om_h, p, Im[m], N[m])
    C = Cm.sum(0)
    return dict(pg=pg, p=p, Pi=Pi, Lmi=Lmi, Ls=Ls, X=X, Y=Y, Im=Im, lU=lU,
                omh=omh, Cm=Cm, lGam=lG, C=C, b=pg * X / (p * Y), Pc=p @ c,
                I=Im.sum(), lam_g=pg * Ag * Lg / Im.sum())

def equilibrium_h(Ag, mobile=True):
    def resid(lw):
        s = structure_h(np.exp(lw), Ag, mobile)
        return np.log(s["C"] / s["Y"])
    lw, info, ier, msg = fsolve(resid, np.log(w0), full_output=True, xtol=1e-13)
    assert ier == 1 or np.max(np.abs(resid(lw))) < 1e-10, msg
    s = structure_h(np.exp(lw), Ag, mobile)
    assert abs(s["X"].sum() / (Ag * Lg) - 1) < 1e-9        # Walras
    s["w"] = np.exp(lw)
    return s

# ------------------------------------------------- log-linear system
def stats_h(s, w):
    """Observable statistics: adds psi (who earns where) and e (who buys what)."""
    Lmi, Im, Ls = s["Lmi"], s["Im"], s["Ls"]
    psi  = (w[None, :] * Lmi) / Im[:, None]          # occupational income shares
    psig = np.array([0.0, 0.0, 1.0]) * (Lg / Im[2])  # college's GPT-income share
    e    = s["Cm"] / s["C"][None, :]                 # expenditure incidence
    mu   = Lmi / Ls[None, :]
    k    = (mu * kap[:, None]).sum(0)
    R    = np.einsum("m,mi,mj->ij", kap, mu, s["Pi"])
    return dict(b=s["b"], psi=psi, psig=psig, e=e, omh=s["omh"],
                xibar=s["omh"] @ xi, Pi=s["Pi"], k=k, R=R,
                theta=s["p"] * c / (s["p"] @ c), sI=Im / Im.sum(),
                lam_g=s["lam_g"], xsh=s["X"] / s["X"].sum())

def solve_loglin_h(st, mobile=True, da=1.0):
    b, psi, e, omh, xb = st["b"], st["psi"], st["e"], st["omh"], st["xibar"]
    n = len(b)
    K = np.diag(st["k"]) - st["R"] if mobile else np.zeros((n, n))
    Aw, ap = np.diag(1.0 - b), -b                    # dp = Aw dw + ap
    Cw, cg = np.zeros((n, n)), np.zeros(n)
    Iw = np.zeros((3, n))
    Uw, Ug, Gw, Gg = np.zeros((3, n)), np.zeros(3), np.zeros((3, n)), np.zeros(3)
    for m in range(3):
        occsh = psi[m].sum()                         # = 1 - psig[m]
        Iw[m] = psi[m] + (kap[m] * (psi[m] - occsh * st["Pi"][m]) if mobile else 0)
        Uw[m] = (Iw[m] - omh[m] @ Aw) / xb[m]
        Ug[m] = -(omh[m] @ ap) / xb[m]
        nu = omh[m] * (1 - sigD)
        Gw[m] = -(nu @ Aw) + nu.sum() * Iw[m] - (nu @ xi) * Uw[m]
        Gg[m] = -(nu @ ap) - (nu @ xi) * Ug[m]
        # dC_mi = sD_i(dI_m - dp_i) + xi_i(1-sD_i) dU_m + dGam_m, weighted e_mi
        Cw += (np.outer(e[m] * sigD, Iw[m]) - np.diag(e[m] * sigD) @ Aw
               + np.outer(e[m] * (1 - sigD) * xi, Uw[m])
               + np.outer(e[m], Gw[m]))
        cg += e[m] * (-sigD * ap + (1 - sigD) * xi * Ug[m] + Gg[m])
    M  = Cw - np.diag(sig * b) - K
    dw = np.linalg.solve(M, sig * b - cg) * da
    dL   = K @ dw
    dp   = Aw @ dw + ap * da
    dIm  = Iw @ dw
    dUm  = Uw @ dw + Ug * da
    dGm  = Gw @ dw + Gg * da
    dPc  = st["theta"] @ dp
    dI   = st["sI"] @ dIm                            # aggregate income
    return dict(dw=dw, dL=dL, dp=dp, dIm=dIm, dUm=dUm, dGm=dGm, dPc=dPc,
                dI=dI, dw_real=dw - dPc, dI_real=dI - dPc)

def stats_base_h(mobile=True):
    return stats_h(structure_h(w0, 1.0, mobile), w0)

# ===================================================================== main
if __name__ == "__main__":
    e0 = equilibrium_h(1.0)
    print("base point: w =", e0["w"].round(6), " (target", w0.round(6), ")")
    print("  per-capita incomes y_m:", (e0["Im"] / N).round(3),
          " budget shares by group:")
    for m, g in enumerate(["LowEd  ", "MidEd  ", "College"]):
        print(f"    {g}: om = {e0['omh'][m].round(3)}  (aggregate c = {c})")
    print("  expenditure incidence e_mi (rows: groups; cols: S,M,C,P):")
    print(stats_base_h()["e"].round(3))
    assert np.max(np.abs(np.log(e0["w"] / w0))) < 1e-9

    # ------------------------------------------------------------- (V1)
    print("=" * 72)
    print("(V1) derivative check at the base (central diff, h = 1e-5)")
    h = 1e-5
    for mob, tag in ((False, "immobile"), (True, "mobile")):
        ep, em = equilibrium_h(np.exp(h), mob), equilibrium_h(np.exp(-h), mob)
        fd_w = (np.log(ep["w"]) - np.log(em["w"])) / (2 * h)
        fd_L = (np.log(ep["Ls"]) - np.log(em["Ls"])) / (2 * h)
        fd_I = (np.log(ep["Im"]) - np.log(em["Im"])) / (2 * h)
        fd_U = (ep["lU"] - em["lU"]) / (2 * h)
        fd_G = (ep["lGam"] - em["lGam"]) / (2 * h)
        r = solve_loglin_h(stats_base_h(mob), mob)
        errs = dict(w=np.max(np.abs(fd_w - r["dw"])), L=np.max(np.abs(fd_L - r["dL"])),
                    Im=np.max(np.abs(fd_I - r["dIm"])), Um=np.max(np.abs(fd_U - r["dUm"])),
                    Gm=np.max(np.abs(fd_G - r["dGm"])))
        print(f"  {tag:>8}:  " + "  ".join(f"{k}: {v:.2e}" for k, v in errs.items()))
        assert max(errs.values()) < 1e-7
        if not mob:   # Hulten on AGGREGATE income, distribution-free
            gap = abs(r["dI_real"] - stats_base_h(mob)["lam_g"])
            print(f"            Hulten (aggregate real income = lam_g): gap {gap:.2e}")
            assert gap < 1e-12
        dX = r["dL"] + sig * (r["dw"] + 1.0)
        gap = stats_base_h(mob)["xsh"] @ dX - 1.0
        print(f"            Walras (GPT market): gap {gap:.2e}")
        assert abs(gap) < 1e-12

    # ------------------------------------------------------------- (V3)
    print("=" * 72)
    print("(V3) RK4 path integration of the per-group statistics, A_g: 1 -> 8")
    def tangent(lw, a, mob):
        s = structure_h(np.exp(lw), np.exp(a), mob)
        return solve_loglin_h(stats_h(s, np.exp(lw)), mob)["dw"]
    for mob, tag in ((False, "immobile"), (True, "mobile")):
        lw, a, hh = np.log(w0.copy()), 0.0, LN8 / 400
        for _ in range(400):
            k1 = tangent(lw, a, mob)
            k2 = tangent(lw + hh/2*k1, a + hh/2, mob)
            k3 = tangent(lw + hh/2*k2, a + hh/2, mob)
            k4 = tangent(lw + hh*k3, a + hh, mob)
            lw, a = lw + hh/6*(k1 + 2*k2 + 2*k3 + k4), a + hh
        err = np.max(np.abs(lw - np.log(equilibrium_h(8.0, mob)["w"])))
        print(f"  {tag:>8}: max|ln w_ODE - ln w_exact| = {err:.2e}")
        assert err < 1e-8

    # ---------------------------------------- the size of the feedback loop
    print("=" * 72)
    print("DISTRIBUTIONAL DEMAND FEEDBACK: per-group vs pooled household")
    # (a) first order: per-group tangent vs pooled tangent (same base point)
    for mob, tag in ((False, "immobile"), (True, "mobile")):
        rh = solve_loglin_h(stats_base_h(mob), mob)
        rp = loglin_ge.solve_loglin(loglin_ge.stats_base(mob), mob)
        print(f"  tangent dw/da, {tag}: per-group {rh['dw'].round(4)}")
        print(f"                  {'':{len(tag)}}  pooled    {rp['dw'].round(4)}"
              f"   max|diff| = {np.max(np.abs(rh['dw'] - rp['dw'])):.4f}")
    # (b) exact, at x8
    for mob, tag in ((False, "immobile"), (True, "mobile")):
        eh, ehb = equilibrium_h(8.0, mob), equilibrium_h(1.0, mob)
        epo, epb = exact_ge.equilibrium(8.0, mob), exact_ge.equilibrium(1.0, mob)
        dwh = np.log(eh["w"] / ehb["w"]) - np.log(eh["Pc"] / ehb["Pc"])
        dwp = np.log(epo["w"] / epb["w"]) - np.log(epo["Pc"] / epb["Pc"])
        print(f"  exact x8 real wages, {tag}: per-group {dwh.round(4)}")
        print(f"                        {'':{len(tag)}} pooled    {dwp.round(4)}"
              f"   max|diff| = {np.max(np.abs(dwh - dwp)):.4f}")
    # (c) group incomes and utilities at x8 (mobile): who collects the growth
    eh, ehb = equilibrium_h(8.0, True), equilibrium_h(1.0, True)
    dIm = np.log(eh["Im"] / ehb["Im"]) - np.log(eh["Pc"] / ehb["Pc"])
    print("  exact x8 (mobile): real group incomes dln I_m =", dIm.round(3),
          " [low, mid, college]")
    print("                     group utilities   dln U_m =",
          (eh["lU"] - ehb["lU"]).round(3))

    print("\nALL HETEROGENEOUS-HOUSEHOLD CHECKS PASSED")
