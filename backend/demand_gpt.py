"""
InclusiveAI — Demand-structure (GPT incidence) model solver.

Faithful re-implementation of `models/demand/exact_ge.py` and
`models/demand/hetero_households.py` (GPTNetwork project: "The incidence of a
general-purpose technology"), parameterized so users can move the primitives:

  Economy: 4 occupation-goods — Manual services (S), Production (M),
  Clerical (C), Professional (P) — plus a GPT sector (Y_g = A_g L_g, w_g = 1).

  production:  Y_i = A_i [ (1-alpha_i) L_i^rho_i + alpha_i X_i^rho_i ]^(1/rho_i)
  demand:      implicitly additive (Hanoch) preferences: good-specific price
               elasticities sD_i (<1) and income-elasticity parameters xi_i.
  workers:     3 groups (low-ed, mid-ed, college) with logit/Roy mobility
               pi_mi ∝ T_mi w_i^kappa_m.

User-scalable knobs (all recalibrate the base point exactly):
  ag           final GPT productivity multiplier (path from 1 to ag)
  mobility     scales kappa (0 = immobile benchmark, 1 = calibrated)
  exposure     scales base GPT cost shares b0 (how much of each occupation's
               costs go to the GPT)
  eps_spread   demand-elasticity dispersion: sD = mean + spread*(sD_cal - mean)
               (0 = one common elasticity, CES benchmark)
  nonhom       income-elasticity dispersion: xi = 1 + nonhom*(xi_cal - 1)
               (0 = homothetic)
  sig_scale    labor-GPT substitution: ln sig = sig_scale * ln sig_cal
               (0 = all Cobb-Douglas-ish, 1 = calibrated)
  hetero       True: one household per worker group (distributional demand
               feedback); False: pooled household.

No scipy: 1-D roots by bisection, the 4-eq system by damped Newton with
finite-difference Jacobian, warm-started along the A_g path.
"""

from functools import lru_cache

import numpy as np

# ----------------------------------------------------------- calibrated base
OCC = ["Manual services", "Production", "Clerical", "Professional"]
GROUPS = ["Low education", "Mid education", "College"]

C_CAL = np.array([0.16, 0.28, 0.26, 0.30])    # base final expenditure shares
B_CAL = np.array([0.02, 0.15, 0.20, 0.15])    # base GPT cost shares (exposure)
SIG_CAL = np.array([0.90, 2.20, 3.00, 0.45])  # labor-GPT substitution
SD_CAL = np.array([0.95, 0.60, 0.45, 0.80])   # product demand elasticities
XI_CAL = np.array([1.20, 0.65, 0.45, 1.55])   # income-elasticity parameters
N_CAL = np.array([0.58, 0.23, 0.19])          # group sizes (1980 hours shares)
KAP_CAL = np.array([2.5, 2.0, 1.2])           # mobility elasticities
PI0_CAL = np.array([[0.32, 0.42, 0.22, 0.04],
                    [0.16, 0.26, 0.44, 0.14],
                    [0.04, 0.06, 0.22, 0.68]])


# ---------------------------------------------------------------- root tools
def bisect(f, lo, hi, tol=1e-14, max_iter=200):
    flo = f(lo)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if hi - lo < tol:
            return mid
        if (fm < 0) == (flo < 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def newton_system(resid, x0, tol=1e-11, max_iter=60, fd=1e-7):
    """Damped Newton with forward-difference Jacobian for small systems."""
    x = np.asarray(x0, float).copy()
    r = resid(x)
    for _ in range(max_iter):
        nr = np.max(np.abs(r))
        if nr < tol:
            return x, nr
        n = len(x)
        J = np.empty((n, n))
        for j in range(n):
            xp = x.copy()
            xp[j] += fd
            J[:, j] = (resid(xp) - r) / fd
        try:
            step = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(J, -r, rcond=None)[0]
        lam = 1.0
        for _ in range(30):
            x_new = x + lam * step
            r_new = resid(x_new)
            if np.max(np.abs(r_new)) < nr:
                x, r = x_new, r_new
                break
            lam *= 0.5
        else:
            x, r = x + 1e-3 * step, resid(x + 1e-3 * step)
    return x, np.max(np.abs(r))


# ---------------------------------------------------------------- the model
class GPTEconomy:
    def __init__(self, mobility=1.0, exposure=1.0, eps_spread=1.0,
                 nonhom=1.0, sig_scale=1.0, hetero=True):
        self.hetero = bool(hetero)
        self.c = C_CAL.copy()
        self.N = N_CAL.copy()
        self.Pi0 = PI0_CAL.copy()

        # --- user-transformed primitives -----------------------------------
        self.b0 = np.clip(exposure * B_CAL, 1e-4, 0.90)
        mean_sd = float(np.dot(self.c, SD_CAL))
        self.sD = np.clip(mean_sd + eps_spread * (SD_CAL - mean_sd), 0.02, 0.985)
        self.xi = 1.0 + nonhom * (XI_CAL - 1.0)
        sig = np.exp(sig_scale * np.log(SIG_CAL))
        # guard the CES reparametrization away from sigma = 1
        sig = np.where(np.abs(sig - 1.0) < 0.02,
                       np.where(sig >= 1.0, 1.02, 0.98), sig)
        self.sig = sig
        self.kap = np.clip(mobility, 0.0, None) * KAP_CAL

        # --- base-point calibration (identical formulas to exact_ge.py) ----
        self.rho = (self.sig - 1.0) / self.sig
        self.L0 = self.N @ self.Pi0
        self.w0 = (1.0 - self.b0) * self.c / self.L0
        self.X0 = self.b0 * self.c
        self.Lg = self.X0.sum()
        om_raw = self.c / (1.0 - self.sD)
        self.Om = om_raw / om_raw.sum()

        r0 = self.X0 / self.L0
        aux = r0 ** (1.0 / self.sig) / self.w0
        self.alpha = aux / (1.0 + aux)
        self.A = self.c / ((1 - self.alpha) * self.L0 ** self.rho
                           + self.alpha * self.X0 ** self.rho) ** (1.0 / self.rho)

        # logit shifters rationalizing Pi0 at w0
        self.T = self.Pi0 / self.w0[None, :] ** self.kap[:, None]
        self.T[self.Pi0 == 0] = 0.0

        # base group incomes (college owns the GPT wage bill)
        Lmi0 = self.N[:, None] * self.Pi0
        self.Im0 = (self.w0[None, :] * Lmi0).sum(1)
        self.Im0[2] += self.Lg

        self.Om_h = self._calibrate_Omega_h() if self.hetero else None

    # ------------------------------------------------------------- demand
    def _demand(self, Om, p, income_pc):
        """lnU, budget shares, lnGamma for one household with per-capita
        income income_pc at prices p (Hanoch implicitly additive)."""
        lp = np.log(p) - np.log(income_pc)

        def G(u):
            return np.log((Om * np.exp((1 - self.sD) * (lp + self.xi * u))).sum())

        u = bisect(G, -60.0, 60.0)
        g = Om * np.exp((1 - self.sD) * (lp + self.xi * u))
        den = ((1 - self.sD) * g).sum()
        om = (1 - self.sD) * g / den
        return u, om, -np.log(den)

    def _calibrate_Omega_h(self):
        """Om such that aggregate base consumption equals c at p = 1."""
        Om = self.Om.copy()
        ones = np.ones(4)
        for _ in range(800):
            Cagg = np.zeros(4)
            for m in range(3):
                _, om, _ = self._demand(Om, ones, self.Im0[m] / self.N[m])
                Cagg += om * self.Im0[m]          # p = 1
            if np.max(np.abs(Cagg - self.c)) < 1e-13:
                break
            Om *= self.c / Cagg
            Om /= Om.sum()
        return Om

    # -------------------------------------------------------- equilibrium
    def _structure(self, w, Ag):
        pg = 1.0 / Ag
        if np.all(self.kap == 0):
            Pi = self.Pi0
        else:
            u = self.T * w[None, :] ** self.kap[:, None]
            Pi = u / u.sum(1, keepdims=True)
        Lmi = self.N[:, None] * Pi
        Ls = Lmi.sum(0)
        ratio = ((self.alpha * w) / ((1 - self.alpha) * pg)) ** self.sig
        y1 = self.A * ((1 - self.alpha)
                       + self.alpha * ratio ** self.rho) ** (1.0 / self.rho)
        p = (w + pg * ratio) / y1
        X, Y = ratio * Ls, y1 * Ls
        Im = (w[None, :] * Lmi).sum(1).copy()
        Im[2] += self.Lg                          # w_g L_g = Lg (w_g = 1)
        I = Im.sum()

        if self.hetero:
            lU = np.zeros(3)
            omh = np.zeros((3, 4))
            Cm = np.zeros((3, 4))
            for m in range(3):
                lU[m], omh[m], _ = self._demand(self.Om_h, p, Im[m] / self.N[m])
                Cm[m] = omh[m] * Im[m] / p
            C = Cm.sum(0)
            lU_pooled = None
        else:
            lu, om, _ = self._demand(self.Om, p, I)
            C = om * I / p
            lU = None
            lU_pooled = lu
            Cm = None

        return dict(pg=pg, p=p, Pi=Pi, Ls=Ls, X=X, Y=Y, Im=Im, I=I, C=C,
                    Cm=Cm, lU=lU, lU_pooled=lU_pooled,
                    b=pg * X / (p * Y), Pc=p @ self.c,
                    lam_g=pg * Ag * self.Lg / I)

    def equilibrium(self, Ag, lw_init=None):
        lw0 = np.log(self.w0) if lw_init is None else lw_init

        def resid(lw):
            s = self._structure(np.exp(lw), Ag)
            return np.log(s["C"] / s["Y"])

        lw, err = newton_system(resid, lw0)
        s = self._structure(np.exp(lw), Ag)
        s["w"] = np.exp(lw)
        s["resid"] = err
        s["walras_gap"] = float(abs(s["X"].sum() / (Ag * self.Lg) - 1.0))
        return s


# -------------------------------------------------------------- API payload
@lru_cache(maxsize=128)
def solve(ag=8.0, mobility=1.0, exposure=1.0, eps_spread=1.0,
          nonhom=1.0, sig_scale=1.0, hetero=True, npoints=25):
    ag = max(float(ag), 1.0 + 1e-9)
    econ = GPTEconomy(mobility=mobility, exposure=exposure,
                      eps_spread=eps_spread, nonhom=nonhom,
                      sig_scale=sig_scale, hetero=hetero)

    base = econ.equilibrium(1.0)
    lnags = np.linspace(0.0, np.log(ag), int(npoints))

    path = {"ag": [], "dlnw_real": [], "dlnL": [], "dlnIm_real": [],
            "shares": [], "lam_g": []}
    lw = np.log(base["w"])
    worst_resid, worst_walras = base["resid"], base["walras_gap"]
    final = None
    for la in lnags:
        s = econ.equilibrium(np.exp(la), lw_init=lw)   # warm start
        lw = np.log(s["w"])
        dPc = np.log(s["Pc"] / base["Pc"])
        path["ag"].append(float(np.exp(la)))
        path["dlnw_real"].append((np.log(s["w"] / base["w"]) - dPc).tolist())
        path["dlnL"].append(np.log(s["Ls"] / base["Ls"]).tolist())
        path["dlnIm_real"].append((np.log(s["Im"] / base["Im"]) - dPc).tolist())
        path["shares"].append((s["p"] * s["C"] / s["I"]).tolist())
        path["lam_g"].append(float(s["lam_g"]))
        worst_resid = max(worst_resid, s["resid"])
        worst_walras = max(worst_walras, s["walras_gap"])
        final = s

    dPc = np.log(final["Pc"] / base["Pc"])
    out_final = {
        "dlnw_real": (np.log(final["w"] / base["w"]) - dPc).tolist(),
        "dlnL": np.log(final["Ls"] / base["Ls"]).tolist(),
        "dlnIm_real": (np.log(final["Im"] / base["Im"]) - dPc).tolist(),
        "dlnI_real": float(np.log(final["I"] / base["I"]) - dPc),
        "dlnp_real": (np.log(final["p"] / base["p"]) - dPc).tolist(),
        "lam_g": float(final["lam_g"]),
        "b": final["b"].tolist(),
        "shares": (final["p"] * final["C"] / final["I"]).tolist(),
        "shares_base": (base["p"] * base["C"] / base["I"]).tolist(),
    }
    if hetero:
        out_final["dlnUm"] = (final["lU"] - base["lU"]).tolist()

    return {
        "params": {"ag": ag, "mobility": mobility, "exposure": exposure,
                   "eps_spread": eps_spread, "nonhom": nonhom,
                   "sig_scale": sig_scale, "hetero": bool(hetero)},
        "labels": {"occupations": OCC, "groups": GROUPS},
        "primitives": {
            "expenditure_shares": econ.c.tolist(),
            "exposure_b0": econ.b0.tolist(),
            "sigma_labor_gpt": econ.sig.tolist(),
            "eps_demand": econ.sD.tolist(),
            "xi_income": econ.xi.tolist(),
            "kappa": econ.kap.tolist(),
        },
        "diagnostics": {"max_resid": float(worst_resid),
                        "max_walras_gap": float(worst_walras)},
        "path": path,
        "final": out_final,
    }


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    # (1) Base point reproduces calibration targets (both modes)
    for het in (False, True):
        econ = GPTEconomy(hetero=het)
        e0 = econ.equilibrium(1.0)
        assert np.max(np.abs(np.log(e0["w"] / econ.w0))) < 1e-9, "base wages"
        assert np.max(np.abs(e0["b"] - econ.b0)) < 1e-9, "base cost shares"
        assert e0["walras_gap"] < 1e-9, "Walras at base"
        print(f"(1) base point ok (hetero={het}); Domar = {e0['lam_g']:.4f}")

    # (2) Hulten: immobile, small shock — aggregate real income gain = lam_g
    econ = GPTEconomy(mobility=0.0, hetero=True)
    h = 1e-5
    ep, em = econ.equilibrium(np.exp(h)), econ.equilibrium(np.exp(-h))
    dI = (np.log(ep["I"] / em["I"]) - np.log(ep["Pc"] / em["Pc"])) / (2 * h)
    lam = econ.equilibrium(1.0)["lam_g"]
    print(f"(2) Hulten check: dI_real/da = {dI:.6f} vs lam_g = {lam:.6f} "
          f"(gap {abs(dI - lam):.2e})")
    assert abs(dI - lam) < 1e-5

    # (3) Note's benchmark: x8 shock, pooled vs per-group real wages
    for mob, tag in ((0.0, "immobile"), (1.0, "mobile  ")):
        eh = GPTEconomy(mobility=mob, hetero=True)
        ep = GPTEconomy(mobility=mob, hetero=False)
        rh, rp = {}, {}
        for name, econ2, store in (("h", eh, rh), ("p", ep, rp)):
            b_, f_ = econ2.equilibrium(1.0), None
            lw = np.log(b_["w"])
            for a in np.linspace(0, np.log(8), 15):
                f_ = econ2.equilibrium(np.exp(a), lw_init=lw)
                lw = np.log(f_["w"])
            store["dw"] = (np.log(f_["w"] / b_["w"])
                           - np.log(f_["Pc"] / b_["Pc"]))
        print(f"(3) x8 {tag}: per-group {np.round(rh['dw'], 4)}")
        print(f"           pooled    {np.round(rp['dw'], 4)}"
              f"   max|diff| = {np.max(np.abs(rh['dw'] - rp['dw'])):.4f}")

    # (4) Full API payload timing
    import time
    solve.cache_clear()
    t0 = time.time()
    out = solve(8.0, 1.0, 1.0, 1.0, 1.0, 1.0, True)
    print(f"(4) solve() {time.time() - t0:.2f}s; "
          f"max resid {out['diagnostics']['max_resid']:.1e}; "
          f"max Walras gap {out['diagnostics']['max_walras_gap']:.1e}")
    print("    final real dlnw:", np.round(out["final"]["dlnw_real"], 4))
    print("    final real dlnIm:", np.round(out["final"]["dlnIm_real"], 4))
    print("ALL DEMAND-MODEL CHECKS PASSED")
