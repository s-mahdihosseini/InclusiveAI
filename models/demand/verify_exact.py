"""
Beyond first order: EXACT (global, non-linearized) results for the nhCES-GPT
economy, verified against exact nonlinear equilibria.

  (E1) EXACT SECTOR SEPARABILITY + GLOBAL CONDITIONAL MONOTONICITY.
       Given the aggregates (p_g, I, U), each equilibrium wage solves the
       scalar equation  L_i * Phi_i(w_i/p_g) = I^sigD Om_i U^{xi_i(1-sigD)} q_i^{-sigD},
       whose LHS is strictly increasing and RHS strictly decreasing in w_i:
       unique solution, and w_i is GLOBALLY monotone in p_g with the sign of
       sigma_i - sigma_D (the horse race holds at any shock size, conditionally).

  (E2) EXACT FLOW REPRESENTATION. The first-order sufficient-statistic system,
       evaluated at current equilibrium shares, is the exact tangent field of
       the equilibrium path: integrating dlnw/dlnA_g = Psi(w, A) by RK4 from
       A=1 to A=8 reproduces the exact nonlinear equilibrium. "First order" is
       exact in flow form; large-shock incidence = path integral of the stats.
       (Full model: mobility + per-group nhCES households.)

  (E3) EXACT ENGEL INCIDENCE (common substitution, sigma_i = sigma_D = sigma):
       closed-form wages  w_i = I [Om_i(1-th_i)/L_i]^{1/sig} U^{xi_i(1-sig)/sig},
       closed-form GPT price, GE reduced to ONE monotone scalar equation in U,
       and relative wages EXACTLY log-linear in U at any shock size:
       dln(w_i/w_j) = ((1-sig)/sig)(xi_i - xi_j) dlnU.

  (E4) EXACT DISPLACEMENT INCIDENCE (unit demand elasticity, sigma_D = 1,
       Cobb-Douglas demand): with nominal GDP as numeraire,
       w_i L_i = (1 - b_i) cbar_i EXACTLY, so dln w_i = dln(1 - b_i); the
       horse race sigma_i vs 1 holds globally; sectors with sigma_i = 2 admit
       a quadratic closed form for the wage given p_g.
"""
import numpy as np
from scipy.optimize import fsolve, brentq

np.set_printoptions(precision=8, suppress=True)

# =====================================================================
# Part I: the calibrated star-network economy (mobility + households),
# identical to verify_nhces.py  -- used for (E1) and (E2)
# =====================================================================
occ = ["S", "M", "C", "P"]; grp = ["LowEd", "MidEd", "College"]
c    = np.array([0.16, 0.28, 0.26, 0.30])
b0   = np.array([0.02, 0.15, 0.20, 0.15])
sig  = np.array([0.90, 2.20, 3.00, 0.45])
rho  = (sig - 1) / sig
sigD = 0.70
xi   = np.array([1.20, 0.65, 0.45, 1.55])
N    = np.array([0.58, 0.23, 0.19])   # 1980 hours shares, AR data (low/mid/college)
kap  = np.array([2.5, 2.0, 1.2])
Pi0  = np.array([[0.32, 0.42, 0.22, 0.04],
                 [0.16, 0.26, 0.44, 0.14],
                 [0.04, 0.06, 0.22, 0.68]])
L0 = N @ Pi0
w0 = (1 - b0) * c / L0
X0 = b0 * c
Lg = X0.sum()
r0 = X0 / L0
aux = r0 ** (1 - rho) / w0
alpha = aux / (1 + aux)
A = c / ((1 - alpha) * L0**rho + alpha * X0**rho) ** (1 / rho)
T = Pi0 / w0[None, :] ** kap[:, None]; T[Pi0 == 0] = 0.0
Lmi0 = N[:, None] * Pi0
I0 = (w0[None, :] * Lmi0).sum(1); I0[2] += Lg

def solveU(Om, p, I):
    u, lp = 0.0, np.log(p)
    for _ in range(200):
        z = (1 - sigD) * (lp + xi * u) + np.log(Om)
        zm = z.max()
        lE = zm / (1 - sigD) + np.log(np.exp(z - zm).sum()) / (1 - sigD)
        sh = np.exp(z - zm); sh /= sh.sum()
        h = lE - np.log(I)
        if abs(h) < 1e-14: break
        u -= h / (sh @ xi)
    return u, sh

def demand(Om, p, I, n=1.0):
    u, sh = solveU(Om, p, I / n)
    return I * sh / p, u, sh

nsize = N.copy()

def calibrate_Omega(incomes, sizes):
    Om = c.copy()
    for _ in range(600):
        Cagg = sum(demand(Om, np.ones(4), I_, n_)[0] for I_, n_ in zip(incomes, sizes))
        Om *= c / Cagg; Om /= Om.sum()
        if np.max(np.abs(Cagg - c)) < 1e-14: break
    return Om

Om_h = calibrate_Omega(I0, nsize)

def structure_at(w, Ag):
    """All primitives-implied objects at (w, A_g), w_g = 1 numeraire."""
    pg = 1.0 / Ag
    ratio = ((alpha * w) / ((1 - alpha) * pg)) ** sig
    y1 = A * ((1 - alpha) + alpha * ratio**rho) ** (1 / rho)
    p  = (w + pg * ratio) / y1
    b  = pg * ratio / (w + pg * ratio)                    # GPT cost shares
    u_ = T * w[None, :] ** kap[:, None]
    Pi = u_ / u_.sum(1, keepdims=True)
    Lmi = N[:, None] * Pi
    Ls  = Lmi.sum(0)
    Im  = (w[None, :] * Lmi).sum(1); Im[2] += Lg
    Cm = np.zeros((3, 4)); Um = np.zeros(3); Wsh = np.zeros((3, 4))
    for m in range(3):
        Cm[m], Um[m], Wsh[m] = demand(Om_h, p, Im[m], nsize[m])
    return dict(pg=pg, p=p, b=b, Pi=Pi, Lmi=Lmi, Ls=Ls, Im=Im, Cm=Cm,
                Um=Um, Wsh=Wsh, ratio=ratio, y1=y1)

def equilibrium(Ag):
    def resid(logw):
        s = structure_at(np.exp(logw), Ag)
        Y = s["y1"] * s["Ls"]
        return np.log(s["Cm"].sum(0) / Y)
    sol = fsolve(resid, np.log(w0), full_output=True, xtol=1e-13)
    assert sol[2] == 1, sol[3]
    return np.exp(sol[0])

# ------------------------------------------------------------ (E1)
def check_E1():
    Ag = 8.0
    w8 = equilibrium(Ag)
    s8 = structure_at(w8, Ag)
    I8, U8 = s8["Im"].sum(), None
    # pooled-equivalent aggregates for the conditional experiment: use the
    # HOUSEHOLD-m demand sum evaluated at frozen (Im, Um) while varying pg.
    Im8, Um8 = s8["Im"].copy(), s8["Um"].copy()
    def sector_wage(i, pg):
        """Solve L_i * Phi_i(w/pg) = sum_m C_mi(p_i(w,pg); Im8, Um8) frozen."""
        def res(logwi):
            wi = np.exp(logwi)
            ratio = ((alpha[i] * wi) / ((1 - alpha[i]) * pg)) ** sig[i]
            y1 = A[i] * ((1 - alpha[i]) + alpha[i] * ratio**rho[i]) ** (1 / rho[i])
            pi_ = (wi + pg * ratio) / y1
            C = sum(nsize[m] * (Im8[m]/nsize[m])**sigD * Om_h[i]
                    * np.exp(Um8[m])**(xi[i]*(1-sigD)) * pi_**(-sigD) for m in range(3))
            return np.log(y1 * s8["Ls"][i]) - np.log(C)
        sol = fsolve(res, np.log(w8[i]), full_output=True, xtol=1e-13)
        # scalar root; fsolve can flag slow progress at extreme p_g even when
        # the root is exact -- accept on the residual itself
        assert sol[2] == 1 or abs(res(sol[0])[0] if np.ndim(res(sol[0])) else res(sol[0])) < 1e-10, sol[3]
        return np.exp(sol[0][0])
    # global conditional monotonicity in pg, sign = sign(sigma_i - sigma_D)
    pgs = np.geomspace(s8["pg"] * 0.25, s8["pg"] * 4.0, 9)
    print("(E1) conditional wages w_i(p_g; I,U frozen), p_g grid (x%.2f..x%.0f):"
          % (0.25, 4))
    ok = True
    for i in range(4):
        ws = np.array([sector_wage(i, pg) for pg in pgs])
        mono_up = np.all(np.diff(ws) > 0)      # w rises as pg rises
        mono_dn = np.all(np.diff(ws) < 0)
        want_up = sig[i] > sigD                # displacement-dominant: cheaper GPT lowers w
        print(f"   {occ[i]} (sig={sig[i]:.2f} vs sigD={sigD}): "
              f"{'increasing' if mono_up else 'decreasing' if mono_dn else 'NON-MONOTONE'} in p_g")
        ok &= (mono_up if want_up else mono_dn)
    assert ok
    print("   => horse-race sign is GLOBAL conditional on aggregates; each sector")
    print("      equation has a unique solution (monotone LHS/RHS).")

check_E1()

# ------------------------------------------------------------ (E2)
def tangent(w, Ag):
    """Exact tangent dlnw/dlnA_g from the sufficient-statistic system at (w,A)."""
    s = structure_at(w, Ag)
    b, Pi, Lmi, Ls, Im, Cm, Wsh = s["b"], s["Pi"], s["Lmi"], s["Ls"], s["Im"], s["Cm"], s["Wsh"]
    pgh = -1.0                                             # w_g = 1 numeraire
    mu = Lmi / Ls[None, :]
    k  = (mu * kap[:, None]).sum(0)
    R  = np.einsum("m,mi,mj->ij", kap, mu, Pi)
    D  = sigD * (1 - b) + sig * b
    # dlnp_i = (1-b) w^ + b pgh ; dIm = sum psi (1+kap(...)) ; dUm = (dIm - om.dp)/xibar
    Aw = np.diag(1 - b); ag = b                            # dp = Aw w^ + ag pgh
    psi = (w[None, :] * Lmi) / Im[:, None]                 # college GPT share: dwg=0
    dIm_w = psi + kap[:, None] * (psi - psi.sum(1)[:, None] * Pi)
    xibar_m = Wsh @ xi
    dUm_w = (dIm_w - Wsh @ Aw) / xibar_m[:, None]
    dUm_g = (-(Wsh @ ag) * pgh) / xibar_m               # constant part, incl. pgh = -1
    e_mi = Cm / Cm.sum(0)[None, :]
    # f_i = b(sig-sigD) pgh + sum_m e_mi [sigD dIm + (1-sigD) xi_i dUm]
    Fw = np.zeros((4, 4)); fg = b * (sig - sigD) * pgh
    for m in range(3):
        Fw += np.outer(e_mi[m] * sigD, dIm_w[m]) \
            + np.outer(e_mi[m] * (1 - sigD) * xi, dUm_w[m])
        fg += e_mi[m] * (1 - sigD) * xi * dUm_g[m]
    M = np.diag(D + k) - R - Fw
    return np.linalg.solve(M, fg)

def check_E2(nsteps=200):
    a0, a1 = 0.0, np.log(8.0)
    h = (a1 - a0) / nsteps
    lw = np.log(w0.copy())
    for n in range(nsteps):                                 # RK4 on dlnw/da
        a = a0 + n * h
        k1 = tangent(np.exp(lw), np.exp(a))
        k2 = tangent(np.exp(lw + h/2*k1), np.exp(a + h/2))
        k3 = tangent(np.exp(lw + h/2*k2), np.exp(a + h/2))
        k4 = tangent(np.exp(lw + h*k3), np.exp(a + h))
        lw = lw + h/6*(k1 + 2*k2 + 2*k3 + k4)
    w_ode = np.exp(lw)
    w_ex  = equilibrium(8.0)
    err = np.max(np.abs(np.log(w_ode / w_ex)))
    print("\n(E2) exact-flow integration, A_g: 1 -> 8 (%d RK4 steps):" % nsteps)
    print("     ODE endpoint:", w_ode.round(6))
    print("     exact GE    :", w_ex.round(6), "  max|err| %.2e" % err)
    assert err < 1e-7
    print("     => the sufficient-statistic system is the EXACT tangent field:")
    print("        large-shock incidence = path integral of the statistics.")

check_E2()

# =====================================================================
# Part II: (E3) common substitution sigma_i = sigma_D  -- closed form
# =====================================================================
def check_E3():
    sg = 0.70                                              # sigma_i = sigma_D = 0.7
    th = np.array([0.02, 0.15, 0.20, 0.15])
    OmE = np.array([0.171, 0.251, 0.218, 0.360])
    LE  = L0.copy(); LgE = 0.14
    def closed_form(U, Ag):
        w  = (OmE * (1 - th) / LE) ** (1/sg) * U ** (xi * (1 - sg) / sg)   # I = 1
        pg = ((OmE * th * U ** (xi * (1 - sg))).sum() / (Ag * LgE)) ** (1/sg)
        p  = ((1 - th) * w**(1 - sg) + th * pg**(1 - sg)) ** (1 / (1 - sg))
        return w, pg, p
    def E_of(p, U):
        return ((OmE * (p * U**xi) ** (1 - sg)).sum()) ** (1 / (1 - sg))
    def solve_scalar(Ag):
        g = lambda lu: np.log(E_of(closed_form(np.exp(lu), Ag)[2], np.exp(lu)))  # E = I = 1
        return np.exp(brentq(g, -20, 20, xtol=1e-15))
    def solve_full(Ag):
        """independent full GE solve (I=1 numeraire) as a check"""
        def res(z):
            w, lpg, lU = np.exp(z[:4]), z[4], z[5]
            pg, U = np.exp(lpg), np.exp(lU)
            p = ((1 - th) * w**(1-sg) + th * pg**(1-sg)) ** (1/(1-sg))
            C = OmE * U**(xi*(1-sg)) * p**(-sg)            # I = 1
            LD = C * (1 - th) * (w / p) ** (-sg) / 1.0     # L = Y dq/dw = C (1-th) w^-sg q^sg / q^sg...
            LD = C * (1 - th) * w**(-sg) * p**sg
            XD = C * th * pg**(-sg) * p**sg
            return np.concatenate([np.log(LD / LE),
                                   [np.log(XD.sum() / (Ag * LgE)),
                                    np.log(E_of(p, U))]])
        z0 = np.concatenate([np.log(w0), [np.log(0.5), 0.0]])
        z, _, ier, msg = fsolve(res, z0, full_output=True, xtol=1e-13)
        # the I = 1 numeraire makes the last equation nearly dependent, so fsolve
        # can flag slow progress at the root; accept on the residual itself
        assert ier == 1 or np.max(np.abs(res(z))) < 1e-10, msg
        return np.exp(z[:4]), np.exp(z[4]), np.exp(z[5])
    print("\n(E3) common substitution sigma_i = sigma_D = %.1f: closed form" % sg)
    rows = []
    for Ag in (1.0, 8.0):
        U = solve_scalar(Ag)
        w_cf, pg_cf, _ = closed_form(U, Ag)
        w_fs, pg_fs, U_fs = solve_full(Ag)
        err = max(np.max(np.abs(np.log(w_cf / w_fs))), abs(np.log(pg_cf/pg_fs)),
                  abs(np.log(U / U_fs)))
        rows.append((Ag, U, w_cf))
        print(f"   A_g={Ag:.0f}: closed-form vs full GE, max|err| = {err:.2e}"
              f"   (U = {U:.5f})")
        assert err < 1e-10
    # exact log-linearity of relative wages in U:
    (A1, U1, wA), (A8, U8, wB) = rows
    lhs = np.log(wB / wA) - np.log(wB[0] / wA[0])          # relative to sector S
    rhs = (1 - sg) / sg * (xi - xi[0]) * np.log(U8 / U1)
    print("   dln(w_i/w_S) exact:", lhs.round(6))
    print("   ((1-s)/s)(xi_i-xi_S)dlnU:", rhs.round(6),
          " max|err| %.2e" % np.max(np.abs(lhs - rhs)))
    assert np.max(np.abs(lhs - rhs)) < 1e-10
    print("   => Engel incidence is EXACTLY log-linear in U at any shock size;")
    print("      GE reduces to one monotone scalar equation in U.")

check_E3()

# =====================================================================
# Part III: (E4) Cobb-Douglas demand (sigma_D = 1) -- exact displacement
# =====================================================================
def check_E4():
    cb  = np.array([0.16, 0.28, 0.26, 0.30])               # fixed expenditure shares
    sgE = np.array([0.90, 2.00, 3.00, 0.45])               # sigma_M = 2: quadratic case
    th  = np.array([0.02, 0.15, 0.20, 0.15])
    LE  = L0.copy(); LgE = 0.14
    def sector_w(i, pg):
        """(1-th)L w + th L pg^{1-s} w^s = (1-th) cb   (I = 1 numeraire)"""
        f = lambda lw: (1-th[i])*LE[i]*np.exp(lw) \
            + th[i]*LE[i]*pg**(1-sgE[i])*np.exp(lw)**sgE[i] - (1-th[i])*cb[i]
        return np.exp(brentq(f, -20, 10, xtol=1e-15))
    def solve(Ag):
        def gpt_gap(lpg):
            pg = np.exp(lpg)
            w = np.array([sector_w(i, pg) for i in range(4)])
            bshare = th*pg**(1-sgE) / ((1-th)*w**(1-sgE) + th*pg**(1-sgE))
            return np.log((bshare*cb).sum()) - np.log(pg*Ag*LgE)
        pg = np.exp(brentq(gpt_gap, -20, 5, xtol=1e-15))
        w = np.array([sector_w(i, pg) for i in range(4)])
        bshare = th*pg**(1-sgE) / ((1-th)*w**(1-sgE) + th*pg**(1-sgE))
        return w, pg, bshare
    print("\n(E4) Cobb-Douglas demand (sigma_D = 1): exact displacement incidence")
    w1, pg1, b1 = solve(1.0)
    paths = {Ag: solve(Ag) for Ag in (1.0, 2.0, 4.0, 8.0)}
    # (i) exact identity w_i L_i = (1-b_i) cb_i at every point
    for Ag, (w, pg, bs) in paths.items():
        assert np.max(np.abs(w * LE - (1 - bs) * cb)) < 1e-14
    print("   identity w_i L_i = (1-b_i) c_i: exact at every A_g (<1e-14)")
    # (ii) global horse race sigma_i vs 1 (wages relative to GDP=1)
    Ws = np.array([paths[Ag][0] for Ag in (1.0, 2.0, 4.0, 8.0)])
    for i in range(4):
        mono_dn = np.all(np.diff(np.log(Ws[:, i])) < 0)
        mono_up = np.all(np.diff(np.log(Ws[:, i])) > 0)
        want_dn = sgE[i] > 1
        print(f"   {occ[i]} (sig={sgE[i]:.2f}): w/GDP {'falls' if mono_dn else 'rises'}"
              f" monotonically; dln w (A_g x8) = {np.log(Ws[-1,i]/Ws[0,i]):+.3f}")
        assert (mono_dn if want_dn else mono_up)
    # (iii) quadratic closed form for the sigma = 2 sector
    i = 1
    for Ag in (1.0, 8.0):
        w, pg, _ = paths[Ag]
        a_, b_, c_ = th[i]*LE[i]/pg, (1-th[i])*LE[i], -(1-th[i])*cb[i]
        w_quad = (-b_ + np.sqrt(b_**2 - 4*a_*c_)) / (2*a_)
        assert abs(np.log(w_quad / w[i])) < 1e-12
    print("   sigma = 2 sector: quadratic closed form matches to 1e-12")
    print("   => with unit demand elasticity, displacement incidence is exact:")
    print("      dln w_i = dln(1-b_i), horse race sigma_i vs 1 holds globally.")

check_E4()

print("\nALL EXACT-RESULT CHECKS PASSED")
