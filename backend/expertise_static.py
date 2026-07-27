"""
InclusiveAI — Static expertise model solver.

Refactored from `models/expertise/static/Final_Supply.py`
("Generative AI and Occupational Entry Barriers", Hosseini & Lichtinger).

Model logic is unchanged:
  - Hierarchical feasibility: occupations ordered by entry barrier; ps_o = share
    of workers eligible; monotone-enforced.
  - Choice: Type-I EV taste shocks with scale tau over the feasible set
    -> logit shares within feasible prefix.
  - Labor demand: w_o = psi_o * A_o^{(sigma-1)/sigma} * L_o^{-1/sigma} (CES, elasticity sigma).
  - Exact GE via damped fixed point; PE formulas as in the paper.

New relative to the script:
  - Vectorized O(J) computation of implied labor shares (verified against the
    original O(J^2) loop).
  - User-scalable channel intensities:
      scarcity in [0, ~1.5]:   F_scaled = F0 + s*(F_AI - F0)  (barrier reduction intensity)
      productivity in [0, ~2]: pi_scaled = p * pi              (productivity gains intensity)
  - sigma as a free parameter (psi is re-calibrated for each sigma so the
    baseline is always matched exactly).
"""

import os
from functools import lru_cache

import numpy as np
import pandas as pd

EPS = 1e-12
MAX_ITERS = 400
TOL = 1e-12
DAMP = 0.50

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "..", "models", "expertise", "static", "Counterfactual.dta")
TITLES_PATH = os.path.join(_HERE, "..", "models", "expertise", "static", "occupation_titles.csv")

COL_WAGE = "mean_annual_wage_2024"
COL_EMP = "total_employment_2024"
COL_R = "avg_months_mapped_without_occ"
COL_RAI = "avg_months_mapped_with_occ"
COL_F0 = "F_at_R"
COL_FAI = "F_at_RAI"
PROD_COL = "prod_gain"


# ----------------------------------------------------------------------------
# Core model primitives (identical math to Final_Supply.py)
# ----------------------------------------------------------------------------
def enforce_monotone_ps(ps_sorted):
    ps = np.clip(np.asarray(ps_sorted, float), EPS, 1.0)
    F = np.clip(1.0 - ps, 0.0, 1.0)
    F_m = np.clip(np.maximum.accumulate(F), 0.0, 1.0)
    return np.clip(1.0 - F_m, EPS, 1.0)


def _mu_from_ps(ps):
    """Mass of workers whose highest feasible occupation is k."""
    F = 1.0 - np.clip(ps, EPS, 1.0)
    J = len(ps)
    mu = np.zeros(J)
    mu[: J - 1] = np.clip(F[1:] - F[:-1], 0.0, 1.0)
    mu[J - 1] = np.clip(1.0 - F[J - 1], 0.0, 1.0)
    s = mu.sum()
    return mu / s if s > 0 else np.ones(J) / J


def implied_Lshare(w_vec, ps_vec, tau=0.5):
    """
    Vectorized O(J) equivalent of the original build_S_and_mu + S @ mu:
      Lshare_i proportional to w_pow_i * sum_{k >= i} mu_k / denom_k,
      denom_k = cumsum(w_pow)[k].
    """
    w = np.asarray(w_vec, float)
    ps = np.asarray(ps_vec, float)
    mu = _mu_from_ps(ps)

    logw = np.log(np.clip(w, EPS, None))
    logw = logw - np.max(logw)
    w_pow = np.exp(logw / tau)

    denom = np.cumsum(w_pow)
    ratio = np.where(denom > 0, mu / np.clip(denom, EPS, None), 0.0)
    tail = np.cumsum(ratio[::-1])[::-1]  # sum_{k>=i} mu_k/denom_k
    Lshare = w_pow * tail
    return Lshare / Lshare.sum()


def solve_GE_fixed_point_exact(w_init, psi, A_mult_vec, ps_vec, L_total,
                               sigma=5.0, tau=0.5,
                               max_iters=MAX_ITERS, tol=TOL, damp=DAMP):
    w = np.clip(np.asarray(w_init, float), EPS, None)
    psi = np.asarray(psi, float)
    A_term = np.power(np.clip(np.asarray(A_mult_vec, float), EPS, None),
                      (sigma - 1.0) / sigma)
    ps = np.asarray(ps_vec, float)

    diff = np.inf
    for it in range(max_iters):
        L = np.clip(L_total * implied_Lshare(w, ps, tau=tau), EPS, None)
        w_new = psi * A_term * np.power(L, -1.0 / sigma)
        diff = np.max(np.abs(np.log(np.clip(w_new, EPS, None) / w)))
        w = (1.0 - damp) * w + damp * w_new
        if diff < tol:
            return w, it + 1, diff
    return w, max_iters, diff


# ----------------------------------------------------------------------------
# Stats helpers
# ----------------------------------------------------------------------------
def weighted_quantile(x, w, qs):
    x, w = np.asarray(x, float), np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[m], w[m]
    idx = np.argsort(x)
    x, w = x[idx], w[idx]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    return np.interp(qs, cw, x)


def weighted_var(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[m], w[m]
    w = w / w.sum()
    mu = np.sum(w * x)
    return float(np.sum(w * (x - mu) ** 2))


def ineq_stats_logw(wage, weights):
    logw = np.log(np.asarray(wage, float))
    w = np.asarray(weights, float)
    p10, p50, p90, p99 = weighted_quantile(logw, w, [0.10, 0.50, 0.90, 0.99])
    return {
        "var_logw": weighted_var(logw, w),
        "p10": float(p10), "p50": float(p50), "p90": float(p90), "p99": float(p99),
        "p50_p10": float(p50 - p10), "p90_p10": float(p90 - p10),
        "p90_p50": float(p90 - p50), "p99_p90": float(p99 - p90),
        "mean_logw": float(np.average(logw, weights=w)),
    }


def weighted_gini(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    idx = np.argsort(x)
    x, w = x[idx], w[idx]
    w = w / w.sum()
    cumw = np.cumsum(w)
    cumxw = np.cumsum(x * w)
    cumxw = cumxw / cumxw[-1]
    B = np.sum(w * (cumxw - 0.5 * (x * w) / np.sum(x * w)))
    # Standard discrete Gini
    return float(1.0 - 2.0 * np.sum(w * (cumxw - 0.5 * x * w / np.sum(x * w))))


def binscatter(x, y, w, nbins=20):
    """Employment-weighted quantile-binned means of y against x."""
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[m], y[m], w[m]
    edges = weighted_quantile(x, w, np.linspace(0, 1, nbins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    xs, ys = [], []
    for i in range(nbins):
        sel = (x > edges[i]) & (x <= edges[i + 1])
        if sel.sum() == 0:
            continue
        xs.append(float(np.average(x[sel], weights=w[sel])))
        ys.append(float(np.average(y[sel], weights=w[sel])))
    return xs, ys


def weighted_kde(logw, w, grid, bw_adjust=1.3):
    """Weighted Gaussian KDE with Scott's-rule bandwidth (matches scipy.stats.gaussian_kde)."""
    logw, w = np.asarray(logw, float), np.asarray(w, float)
    m = np.isfinite(logw) & (w > 0) & (logw >= grid[0]) & (logw <= grid[-1])
    x, wt = logw[m], w[m] / w[m].sum()
    neff = 1.0 / np.sum(wt ** 2)
    mu = np.sum(wt * x)
    var = np.sum(wt * (x - mu) ** 2)
    bw = np.sqrt(var) * neff ** (-1.0 / 5.0) * bw_adjust  # Scott's rule
    z = (grid[:, None] - x[None, :]) / bw
    dens = np.sum(wt[None, :] * np.exp(-0.5 * z ** 2), axis=1) / (bw * np.sqrt(2 * np.pi))
    return dens


def minimize_scalar_bounded(f, lo, hi, xatol=1e-4, max_iter=200):
    """Golden-section search (replaces scipy.optimize.minimize_scalar, bounded)."""
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lo), float(hi)
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if abs(b - a) < xatol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)
    return (a + b) / 2.0


def take(v, idx):
    return np.asarray(v)[idx]


def put_back(v_sorted, idx, n):
    out = np.empty(n, dtype=float)
    out[idx] = v_sorted
    return out


# ----------------------------------------------------------------------------
# Data (loaded once)
# ----------------------------------------------------------------------------
class _Data:
    def __init__(self):
        df = pd.read_stata(DATA_PATH)
        need = [c for c in ["onetsoccode", COL_WAGE, COL_EMP, COL_R, COL_RAI,
                            COL_F0, COL_FAI, PROD_COL, "pss"] if c in df.columns]
        df = df[need].dropna().copy()
        df = df[(df[COL_EMP] > 0) & (df[COL_WAGE] > 0)].copy()
        df[PROD_COL] = -np.log(1.0 - df[PROD_COL])

        try:
            titles = pd.read_csv(TITLES_PATH)
            df = df.merge(titles, on="onetsoccode", how="left")
            df["occupation_title"] = df["occupation_title"].fillna(df["onetsoccode"])
        except Exception:
            df["occupation_title"] = df["onetsoccode"]

        self.df = df.reset_index(drop=True)
        self.n = len(df)
        self.w0 = df[COL_WAGE].to_numpy(float)
        self.L_obs = df[COL_EMP].to_numpy(float)
        self.L0_total = float(self.L_obs.sum())
        self.F0 = df[COL_F0].to_numpy(float)
        self.FAI = df[COL_FAI].to_numpy(float)
        self.R0 = df[COL_R].to_numpy(float)
        self.RAI = df[COL_RAI].to_numpy(float)
        self.pi = df[PROD_COL].to_numpy(float)
        self.titles = df["occupation_title"].tolist()
        self.codes = df["onetsoccode"].tolist()

        # Baseline ordering + feasibility (does not depend on user params)
        self.idx0 = np.argsort(self.R0)
        self.ps0_raw = np.clip(1.0 - self.F0, EPS, 1.0)
        ps0_sorted = enforce_monotone_ps(take(self.ps0_raw, self.idx0))
        self.ps0_0 = ps0_sorted
        self.w0_0 = take(self.w0, self.idx0)
        Lobs_0 = take(self.L_obs, self.idx0)
        self.Lshare_obs_0 = Lobs_0 / Lobs_0.sum()

        # Calibrate tau to match baseline employment shares (independent of sigma)
        def tau_objective(log_tau):
            tau = float(np.exp(log_tau))
            Ls = implied_Lshare(self.w0_0, self.ps0_0, tau=tau)
            d = np.log(Ls + 1e-14) - np.log(self.Lshare_obs_0 + 1e-14)
            return float(np.sum(self.Lshare_obs_0 * d ** 2))

        self.tau_hat = float(np.exp(minimize_scalar_bounded(
            tau_objective, np.log(0.05), np.log(5.0), xatol=1e-4)))

        # Baseline model employment (tau-dependent only)
        self.Lshare_base_0 = implied_Lshare(self.w0_0, self.ps0_0, tau=self.tau_hat)
        self.L_base_0 = np.clip(self.L0_total * self.Lshare_base_0, EPS, None)
        self.L_base = put_back(self.L_base_0, self.idx0, self.n)


_DATA = None


def get_data():
    global _DATA
    if _DATA is None:
        _DATA = _Data()
    return _DATA


# ----------------------------------------------------------------------------
# Scenario solver
# ----------------------------------------------------------------------------
def _solve_scenario(sigma, scarcity, productivity, tau=None):
    """
    scarcity s: F_scaled = F0 + s*(F_AI - F0); barrier ordering interpolated too.
    productivity p: pi_scaled = p * pi.
    Returns dict of occupation-level arrays (original row order) + diagnostics.
    """
    D = get_data()
    n = D.n
    tau = D.tau_hat if tau is None else float(tau)

    # psi calibration depends on sigma (baseline A=1): psi = w0 * L_base^{1/sigma}
    psi0 = D.w0_0 * np.power(D.L_base_0, 1.0 / sigma)
    psi_unsorted = put_back(psi0, D.idx0, n)

    # Channel scaling
    F_s = np.clip(D.F0 + scarcity * (D.FAI - D.F0), 0.0, 1.0)
    R_s = D.R0 + scarcity * (D.RAI - D.R0)
    ps_with_raw = np.clip(1.0 - F_s, EPS, 1.0)
    pi_s = productivity * D.pi
    A_mult = np.exp(pi_s)

    # Scenario ordering by scaled barrier
    idxS = np.argsort(R_s)
    psS = enforce_monotone_ps(take(ps_with_raw, idxS))

    # ---- PE ----
    dlogw_PE = (-(1.0 / sigma) * np.log(np.clip(ps_with_raw, EPS, None)
                                        / np.clip(D.ps0_raw, EPS, None))
                + ((sigma - 1.0) / sigma) * pi_s)
    w_PE = D.w0 * np.exp(dlogw_PE)

    # ---- GE ----
    psi_S = take(psi_unsorted, idxS)
    A_S = take(A_mult, idxS)
    w_init = take(D.w0, idxS) * np.exp(((sigma - 1.0) / sigma) * take(pi_s, idxS))

    w_ge_S, iters, err = solve_GE_fixed_point_exact(
        w_init=w_init, psi=psi_S, A_mult_vec=A_S, ps_vec=psS,
        L_total=D.L0_total, sigma=sigma, tau=tau)

    w_GE = put_back(w_ge_S, idxS, n)
    dlogw_GE = np.log(w_GE / D.w0)

    # Employment reconstruction
    L_GE = put_back(D.L0_total * implied_Lshare(w_ge_S, psS, tau=tau), idxS, n)
    w_PE_S = take(w_PE, idxS)
    L_PE = put_back(D.L0_total * implied_Lshare(w_PE_S, psS, tau=tau), idxS, n)

    return {
        "w_PE": w_PE, "w_GE": w_GE,
        "dlogw_PE": dlogw_PE, "dlogw_GE": dlogw_GE,
        "L_PE": L_PE, "L_GE": L_GE,
        "iters": int(iters), "err": float(err), "tau": tau,
    }


@lru_cache(maxsize=256)
def solve(sigma=5.0, scarcity=1.0, productivity=1.0, nbins=20):
    """Full API payload for one (sigma, scarcity, productivity) scenario."""
    sigma = float(sigma)
    scarcity = float(scarcity)
    productivity = float(productivity)

    D = get_data()
    res = _solve_scenario(sigma, scarcity, productivity)

    # Channel decomposition (GE) at same sigma
    res_scar = _solve_scenario(sigma, scarcity, 0.0)
    res_prod = _solve_scenario(sigma, 0.0, productivity)

    logw0 = np.log(D.w0)

    # Inequality stats
    stats = {
        "baseline": ineq_stats_logw(D.w0, D.L_base),
        "pe": ineq_stats_logw(res["w_PE"], res["L_PE"]),
        "ge": ineq_stats_logw(res["w_GE"], res["L_GE"]),
        "ge_scarcity_only": ineq_stats_logw(res_scar["w_GE"], res_scar["L_GE"]),
        "ge_productivity_only": ineq_stats_logw(res_prod["w_GE"], res_prod["L_GE"]),
    }

    # Binned scatters (delta log wage vs log initial wage)
    bins = {}
    for key, dl in [("pe", res["dlogw_PE"]), ("ge", res["dlogw_GE"]),
                    ("ge_scarcity_only", res_scar["dlogw_GE"]),
                    ("ge_productivity_only", res_prod["dlogw_GE"])]:
        xs, ys = binscatter(logw0, dl, D.L_obs, nbins=nbins)
        bins[key] = {"x": xs, "y": ys}

    # KDEs of log wage distribution
    grid = np.linspace(10.0, 14.0, 200)
    kde = {
        "grid": grid.tolist(),
        "baseline": weighted_kde(np.log(D.w0), D.L_base, grid).tolist(),
        "pe": weighted_kde(np.log(res["w_PE"]), res["L_PE"], grid).tolist(),
        "ge": weighted_kde(np.log(res["w_GE"]), res["L_GE"], grid).tolist(),
    }

    # Occupation-level table (for gainers/losers)
    occ = []
    for i in range(D.n):
        occ.append({
            "code": D.codes[i],
            "title": D.titles[i],
            "w0": float(D.w0[i]),
            "emp0": float(D.L_obs[i]),
            "dlogw_pe": float(res["dlogw_PE"][i]),
            "dlogw_ge": float(res["dlogw_GE"][i]),
            "dlog_emp_ge": float(np.log(max(res["L_GE"][i], EPS) / max(D.L_base[i], EPS))),
        })

    return {
        "params": {"sigma": sigma, "scarcity": scarcity,
                   "productivity": productivity, "tau": D.tau_hat},
        "diagnostics": {"iters": res["iters"], "err": res["err"], "n_occupations": D.n},
        "stats": stats,
        "binscatter": bins,
        "kde": kde,
        "occupations": occ,
    }


# ----------------------------------------------------------------------------
# Verification: vectorized Lshare == original O(J^2) implementation
# ----------------------------------------------------------------------------
def _implied_Lshare_loop(w_vec, ps_vec, tau=0.5):
    w = np.asarray(w_vec, float)
    ps = np.asarray(ps_vec, float)
    mu = _mu_from_ps(ps)
    logw = np.log(np.clip(w, EPS, None))
    logw = logw - np.max(logw)
    w_pow = np.exp(logw / tau)
    denom_prefix = np.cumsum(w_pow)
    J = len(w)
    S = np.zeros((J, J))
    for k in range(J):
        if denom_prefix[k] > 0:
            S[: k + 1, k] = w_pow[: k + 1] / denom_prefix[k]
    Ls = S @ mu
    return Ls / Ls.sum()


if __name__ == "__main__":
    D = get_data()
    print(f"n = {D.n}, tau_hat = {D.tau_hat:.4f}")

    # Verify vectorization
    a = implied_Lshare(D.w0_0, D.ps0_0, tau=D.tau_hat)
    b = _implied_Lshare_loop(D.w0_0, D.ps0_0, tau=D.tau_hat)
    print("max |vectorized - loop| =", np.max(np.abs(a - b)))
    assert np.max(np.abs(a - b)) < 1e-12, "Vectorized Lshare mismatch!"

    import time
    t0 = time.time()
    out = solve(5.0, 1.0, 1.0)
    print(f"solve() took {time.time() - t0:.2f}s; GE iters={out['diagnostics']['iters']}")
    for k, v in out["stats"].items():
        print(f"{k:22s} var_logw={v['var_logw']:.4f} p90_p10={v['p90_p10']:.4f}")
