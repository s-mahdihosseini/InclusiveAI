"""
exact_ge.py -- EXACT nonlinear general equilibrium of the calibrated
4-occupation + GPT economy of the note (Section: Numerical Validation).

Model (deliberately the same as the note's Sections 2-4, with ONE pooled
nonhomothetic-CES household so that Proposition 1's (I, U) are literally
the household's income and utility; the per-group-household version is
verified separately in verify_exact.py, check E2):

  production:   Y_i = A_i [ (1-alpha_i) L_i^rho_i + alpha_i X_i^rho_i ]^(1/rho_i),
                rho_i = (sigma_i - 1)/sigma_i        (reparametrized CES; the
                share parameter alpha_i is chosen to hit the base cost share b_i)
  GPT sector:   Y_g = A_g L_g,  L_g fixed, wage w_g = 1 (numeraire in the code;
                results are deflated by the base consumption bundle at the end)
  demand:       pooled household with IMPLICITLY (indirectly) ADDITIVE
                preferences (Hanoch 1975): the expenditure function E(p,U) is
                defined by  sum_j Om_j (p_j/E)^(1-sD_j) U^(xi_j (1-sD_j)) = 1,
                with a GOOD-SPECIFIC price elasticity sD_j. Shephard's lemma
                gives budget shares om_j = (1-sD_j) g_j / sum_k (1-sD_k) g_k,
                g_j = Om_j (p_j/E)^(1-sD_j) U^(xi_j(1-sD_j)). Log-differentiating
                reproduces the note's demand block EXACTLY, including the
                common cross-product index:  dln Gamma = -dln sum_k (1-sD_k) g_k.
                With sD_j common this collapses to nonhomothetic CES (CLM 2021)
                and Gamma is constant.
  workers:      logit/Roy:  pi_mi = T_mi w_i^kappa_m / sum_j T_mj w_j^kappa_m,
                L_i = sum_m N_m pi_mi;  mobile=False freezes pi at the base point
                (the kappa = 0 / immobile benchmark)

Base point: p_i = 1, I = 1, U = 1, A_g = 1. The GPT shock raises A_g.
"""
import numpy as np
from scipy.optimize import fsolve, brentq

# --------------------------------------------------------------- calibration
occ = ["S", "M", "C", "P"]
c    = np.array([0.16, 0.28, 0.26, 0.30])    # base final expenditure shares
b0   = np.array([0.02, 0.15, 0.20, 0.15])    # base GPT cost shares (exposure)
sig  = np.array([0.90, 2.20, 3.00, 0.45])    # labor-GPT substitution
sigD = np.array([0.95, 0.60, 0.45, 0.80])    # product demand elasticities
xi   = np.array([1.20, 0.65, 0.45, 1.55])    # income-elasticity parameters
N    = np.array([0.58, 0.23, 0.19])          # group sizes (1980 hours shares)
kap  = np.array([2.5, 2.0, 1.2])             # mobility elasticities
Pi0  = np.array([[0.32, 0.42, 0.22, 0.04],   # base occupation portfolios
                 [0.16, 0.26, 0.44, 0.14],
                 [0.04, 0.06, 0.22, 0.68]])

rho = (sig - 1.0) / sig
L0  = N @ Pi0                                # base occupation employment
w0  = (1.0 - b0) * c / L0                    # base wages (labor gets (1-b)c)
X0  = b0 * c                                 # base GPT purchases
Lg  = X0.sum()                               # GPT-sector labor force (Domar wt)
# taste shifters Om_j: at the base (p = I = U = 1) budget shares must equal c
# and the defining identity sum_j Om_j = 1 must hold; both are delivered by
Om  = (c / (1.0 - sigD)) / (c / (1.0 - sigD)).sum()

# CES share parameters alpha_i and TFPs A_i hitting (w0, b0, Y0 = c, p0 = 1):
_r0   = X0 / L0
_aux  = _r0 ** (1.0 / sig) / w0              # alpha/(1-alpha) = r0^(1/sig)/w0
alpha = _aux / (1.0 + _aux)
A     = c / ((1 - alpha) * L0 ** rho + alpha * X0 ** rho) ** (1.0 / rho)

# logit shifters rationalizing Pi0 at w0
T = Pi0 / w0[None, :] ** kap[:, None]
T[Pi0 == 0] = 0.0

# ------------------------------------------------------------------- demand
def solveU(p, I):
    """U (log), budget shares, and the Gamma index of the pooled household.
    lnU solves  sum_j Om_j exp[(1-sD_j)(ln p_j - ln I + xi_j lnU)] = 1,
    which is strictly increasing in lnU since every sD_j < 1. Budget shares
    follow from Shephard's lemma:  om_j = (1-sD_j) g_j / sum_k (1-sD_k) g_k."""
    lp = np.log(p) - np.log(I)
    G = lambda u: np.log((Om * np.exp((1 - sigD) * (lp + xi * u))).sum())
    u = brentq(G, -50.0, 50.0, xtol=1e-15)
    g = Om * np.exp((1 - sigD) * (lp + xi * u))
    denom = ((1 - sigD) * g).sum()
    return u, (1 - sigD) * g / denom, -np.log(denom)   # lnU, shares, ln Gamma

# -------------------------------------------------------------- equilibrium
def logit_shares(w):
    u = T * w[None, :] ** kap[:, None]
    return u / u.sum(1, keepdims=True)

def structure_at(w, Ag, mobile=True):
    """Everything implied by primitives at wages w and GPT productivity Ag
    (w_g = 1 numeraire). Used both by the exact solver (residuals) and by the
    log-linear system when its statistics are updated along the path."""
    pg    = 1.0 / Ag
    Pi    = logit_shares(w) if mobile else Pi0
    Ls    = N @ Pi
    ratio = ((alpha * w) / ((1 - alpha) * pg)) ** sig      # X_i / L_i
    y1    = A * ((1 - alpha) + alpha * ratio ** rho) ** (1.0 / rho)
    p     = (w + pg * ratio) / y1                          # unit cost = price
    X, Y  = ratio * Ls, y1 * Ls
    I     = w @ Ls + Lg                                    # w_g L_g = Lg
    lU, om, lGam = solveU(p, I)
    C     = om * I / p
    b     = pg * X / (p * Y)                               # GPT cost shares
    Pc    = p @ c                                          # base-bundle price
    return dict(pg=pg, p=p, Pi=Pi, Ls=Ls, X=X, Y=Y, I=I, lU=lU, om=om, C=C,
                lGam=lGam, b=b, Pc=Pc, lam_g=pg * Ag * Lg / I)

def equilibrium(Ag, mobile=True):
    """Solve the 4 goods-market-clearing conditions in log wages; GPT market
    then clears by Walras' law (asserted, not imposed)."""
    def resid(lw):
        s = structure_at(np.exp(lw), Ag, mobile)
        return np.log(s["C"] / s["Y"])
    lw, info, ier, msg = fsolve(resid, np.log(w0), full_output=True, xtol=1e-13)
    assert ier == 1 or np.max(np.abs(resid(lw))) < 1e-10, msg
    s = structure_at(np.exp(lw), Ag, mobile)
    assert abs(s["X"].sum() / (Ag * Lg) - 1) < 1e-9        # Walras check
    s["w"] = np.exp(lw)
    return s

# ------------------------------------------------------------------- report
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    e0 = equilibrium(1.0, mobile=True)
    print("base point (A_g = 1):")
    print("  w      :", e0["w"], " (target", w0.round(6), ")")
    print("  p      :", e0["p"], "  I = %.6f  lnU = %.2e" % (e0["I"], e0["lU"]))
    print("  b      :", e0["b"], " (target", b0, ")")
    print("  Domar  : %.4f" % e0["lam_g"])
    assert np.max(np.abs(np.log(e0["w"] / w0))) < 1e-10
    for mob, tag in ((False, "immobile"), (True, "mobile  ")):
        e1 = equilibrium(8.0, mobile=mob)
        dw = np.log(e1["w"] / w0) - np.log(e1["Pc"] / e0["Pc"])
        dL = np.log(e1["Ls"] / L0)
        dI = np.log(e1["I"] / e0["I"]) - np.log(e1["Pc"] / e0["Pc"])
        print(f"A_g x8, {tag}: real dlnw = {dw.round(4)}  dlnL = {dL.round(4)}"
              f"  real dlnI = {dI:+.4f}  Domar -> {e1['lam_g']:.4f}")
    print("exact_ge: ALL BASE CHECKS PASSED")
