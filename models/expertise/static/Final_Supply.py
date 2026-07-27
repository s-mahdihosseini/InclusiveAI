################################################################################
# FULL WORKING CODE (EXACT GE via fixed point) — BASE MODEL (NO AUTOMATION)
# Structured like your NEW MODEL script, including the same end-of-file PSS exercise.
#
# Channels/scenarios computed (PE + exact GE):
# - Scarcity-only (ps changes; no productivity)
# - Productivity-only (pi changes; baseline ps)
# - Both channels (ps changes + pi changes)
################################################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotnine import ggplot, aes, geom_point, geom_line, geom_errorbar, geom_ribbon, labs, theme_bw
from binsreg import binsreg
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar

import statsmodels.api as sm
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 11,
    "font.size": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ----------------------------
# Settings
# ----------------------------
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Counterfactual.dta")

OUTDIR = "Results_BaseModel"
os.makedirs(OUTDIR, exist_ok=True)

sigma = 5.0   # Elasticity across occupations

# Column mapping
COL_WAGE = "mean_annual_wage_2024"
COL_EMP  = "total_employment_2024"

# Barrier proxies
COL_R    = "avg_months_mapped_without_occ"
COL_RAI  = "avg_months_mapped_with_occ"

# CDF objects
COL_F0   = "F_at_R"
COL_FAI  = "F_at_RAI"

# Productivity gain
PROD_COL = "prod_gain"
PROD_IS_PERCENT = False  # True if prod_gain is 10 for 10%

# Binsreg plotting settings
NBINS = 20
LINE = (3, 3)
CI   = (3, 3)
CB   = (3, 3)
POLY = 4

# Fixed point settings (GE exact)
MAX_ITERS = 400
TOL       = 1e-12
DAMP      = 0.50
EPS       = 1e-12


# ----------------------------
# Helper Functions
# ----------------------------
def binsreg_result_object(est):
    return est.data_plot[0]

def build_overlay_plot(result_pe, result_ge, xlabel, ylabel, title,
                       color_pe="blue", color_ge="red"):
    fig = ggplot() + labs(x=xlabel, y=ylabel, title=title) + theme_bw()

    # PE layer
    fig += geom_point(data=result_pe.dots, mapping=aes(x="x", y="fit"),
                      color=color_pe, size=2, shape="o")
    fig += geom_line(data=result_pe.line, mapping=aes(x="x", y="fit"),
                     color=color_pe, size=0.7)
    if hasattr(result_pe, "ci") and result_pe.ci is not None and not result_pe.ci.empty:
        fig += geom_errorbar(data=result_pe.ci, mapping=aes(x="x", ymin="ci_l", ymax="ci_r"),
                             color=color_pe, size=0.5, width=0.02)
    if hasattr(result_pe, "cb") and result_pe.cb is not None and not result_pe.cb.empty:
        fig += geom_ribbon(data=result_pe.cb, mapping=aes(x="x", ymin="cb_l", ymax="cb_r"),
                           fill=color_pe, alpha=0.12)

    # GE layer
    fig += geom_point(data=result_ge.dots, mapping=aes(x="x", y="fit"),
                      color=color_ge, size=2, shape="^")
    fig += geom_line(data=result_ge.line, mapping=aes(x="x", y="fit"),
                     color=color_ge, size=0.7)
    if hasattr(result_ge, "ci") and result_ge.ci is not None and not result_ge.ci.empty:
        fig += geom_errorbar(data=result_ge.ci, mapping=aes(x="x", ymin="ci_l", ymax="ci_r"),
                             color=color_ge, size=0.5, width=0.02)
    if hasattr(result_ge, "cb") and result_ge.cb is not None and not result_ge.cb.empty:
        fig += geom_ribbon(data=result_ge.cb, mapping=aes(x="x", ymin="cb_l", ymax="cb_r"),
                           fill=color_ge, alpha=0.12)
    return fig

def weighted_quantile(x, w, qs):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[m], w[m]
    idx = np.argsort(x)
    x, w = x[idx], w[idx]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    return np.interp(qs, cw, x)

def weighted_var(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[m], w[m]
    w = w / w.sum()
    mu = np.sum(w * x)
    return np.sum(w * (x - mu)**2)

# ============================
# CHANGE (1): add p99 and p99_p90
# ============================
def ineq_stats_logw(wage, weights):
    logw = np.log(np.asarray(wage, float))
    w = np.asarray(weights, float)
    mask = np.isfinite(logw) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return {k: np.nan for k in [
            "var_logw", "p10", "p50", "p90", "p99",
            "p50_p10", "p90_p10", "p90_p50", "p99_p90"
        ]}
    logw, w = logw[mask], w[mask]
    p10, p50, p90, p99 = weighted_quantile(logw, w, [0.10, 0.50, 0.90, 0.99])
    v = weighted_var(logw, w)
    return {
        "var_logw": float(v),
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "p99": float(p99),
        "p50_p10": float(p50 - p10),
        "p90_p10": float(p90 - p10),
        "p90_p50": float(p90 - p50),
        "p99_p90": float(p99 - p90),
    }

def enforce_monotone_ps(ps_sorted):
    """
    Hierarchical feasibility: barrier sorted low->high implies ps non-increasing.
    """
    ps = np.clip(np.asarray(ps_sorted, float), EPS, 1.0)
    F = 1.0 - ps
    F = np.clip(F, 0.0, 1.0)
    F_m = np.maximum.accumulate(F)   # enforce non-decreasing
    F_m = np.clip(F_m, 0.0, 1.0)
    ps_m = 1.0 - F_m
    return np.clip(ps_m, EPS, 1.0)


# ----------------------------
# Feasibility + choice objects
# ----------------------------
def build_S_and_mu(w, ps_vec, tau=0.5):
    """
    Assumes occupations ordered from lowest barrier to highest barrier.
    ps_vec is share eligible for each occupation (non-increasing in barrier).
    """
    w = np.asarray(w, float)
    ps = np.asarray(ps_vec, float)

    F = 1.0 - np.clip(ps, EPS, 1.0)
    J = len(w)

    mu = np.zeros(J)
    mu[:J-1] = np.clip(F[1:] - F[:-1], 0.0, 1.0)
    mu[J-1]  = np.clip(1.0 - F[J-1], 0.0, 1.0)
    mu = mu / mu.sum() if mu.sum() > 0 else np.ones(J) / J

    logw = np.log(np.clip(w, EPS, None))
    logw = logw - np.max(logw)
    w_pow = np.exp(logw / tau)

    denom_prefix = np.cumsum(w_pow)

    S = np.zeros((J, J))
    for k in range(J):
        denom = denom_prefix[k]
        if denom > 0:
            S[:k+1, k] = w_pow[:k+1] / denom
    return S, mu

def implied_Lshare_flexible(w_vec, ps_vec, tau=0.5):
    S, mu = build_S_and_mu(w_vec, ps_vec, tau=tau)
    Lshare = S @ mu
    return Lshare / Lshare.sum()


# ----------------------------
# Exact GE fixed point
# ----------------------------
def solve_GE_fixed_point_exact(
    w_init, psi, A_mult_vec, ps_vec, L_total,
    sigma=4.0, tau=0.5, max_iters=300, tol=1e-10, damp=0.5
):
    """
    Solves: w = psi * [A_mult]^{(σ-1)/σ} * L(w,ps)^{-1/σ}
    where L(w,ps) = L_total * implied_Lshare_flexible(w, ps, tau).
    Base model: A_mult = exp(pi) for productivity channels, otherwise 1.
    """
    w = np.clip(np.asarray(w_init, float), EPS, None)
    psi = np.asarray(psi, float)
    A_mult = np.asarray(A_mult_vec, float)
    ps = np.asarray(ps_vec, float)

    A_term = np.power(np.clip(A_mult, EPS, None), (sigma - 1.0) / sigma)

    diff = np.inf
    for it in range(max_iters):
        Lshare = implied_Lshare_flexible(w, ps, tau=tau)
        L = np.clip(L_total * Lshare, EPS, None)

        w_new = psi * A_term * np.power(L, -1.0 / sigma)

        diff = np.max(np.abs(np.log(np.clip(w_new, EPS, None) / w)))
        w = (1.0 - damp) * w + damp * w_new

        if diff < tol:
            return w, it + 1, diff

    return w, max_iters, diff


# ----------------------------
# Utility: reorder vectors
# ----------------------------
def take(v, idx):
    return np.asarray(v)[idx]

def put_back(v_sorted, idx, n):
    out = np.empty(n, dtype=float)
    out[idx] = v_sorted
    return out


# ----------------------------
# Weighted KDE helpers
# ----------------------------
def _weighted_kde(x, w, grid, bw_adjust=1.0):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    w = w / w.sum()
    kde = gaussian_kde(x, weights=w)
    kde.set_bandwidth(kde.factor * bw_adjust)
    return kde(grid)

def _trim_xyw(x, w, x_min=10.0, x_max=14.0):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0) & (x >= x_min) & (x <= x_max)
    x, w = x[m], w[m]
    if len(x) == 0:
        return x, w
    w = w / w.sum()
    return x, w

def plot_hist_three(w_base, w_pe, w_ge, L_base, L_pe, L_ge, fname, title):
    bins = 50
    plt.figure(figsize=(7,5))
    plt.hist(np.log(w_base), bins=bins, weights=L_base, density=True, alpha=0.3, label="Baseline")
    plt.hist(np.log(w_pe),   bins=bins, weights=L_pe,   density=True, alpha=0.3, label="PE")
    plt.hist(np.log(w_ge),   bins=bins, weights=L_ge,   density=True, alpha=0.3, label="GE (Exact)")
    plt.xlabel("log(annual wage)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname))
    plt.show()
    plt.close()

def plot_kde_three(w_base, w_pe, w_ge, L_base, L_pe, L_ge, fname, title,
                   bw_adjust=1.3, gridsize=400, x_min=10.0, x_max=14.0):
    x0_raw = np.log(np.asarray(w_base, float))
    x1_raw = np.log(np.asarray(w_pe, float))
    x2_raw = np.log(np.asarray(w_ge, float))

    x0, w0 = _trim_xyw(x0_raw, L_base, x_min=x_min, x_max=x_max)
    x1, w1 = _trim_xyw(x1_raw, L_pe,   x_min=x_min, x_max=x_max)
    x2, w2 = _trim_xyw(x2_raw, L_ge,   x_min=x_min, x_max=x_max)

    if len(x0) == 0 or len(x1) == 0 or len(x2) == 0:
        print(f"[WARN] KDE skipped for {fname}: trimming left empty sample "
              f"(lens: base={len(x0)}, pe={len(x1)}, ge={len(x2)}).")
        return

    grid = np.linspace(x_min, x_max, gridsize)

    d0 = _weighted_kde(x0, w0, grid, bw_adjust=bw_adjust)
    d1 = _weighted_kde(x1, w1, grid, bw_adjust=bw_adjust)
    d2 = _weighted_kde(x2, w2, grid, bw_adjust=bw_adjust)

    plt.figure(figsize=(7,5))
    plt.plot(grid, d0, linewidth=2.0, label="Current")
    plt.plot(grid, d1, linewidth=2.0, label="Post-AI (PE)")
    plt.plot(grid, d2, linewidth=2.0, label="Post-AI (GE)")
    plt.xlim(x_min, x_max)
    plt.xlabel("log(annual wage)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname))
    plt.show()
    plt.close()


# ----------------------------
# Data Cleaning & Prep
# ----------------------------
df = pd.read_stata(DATA_PATH)

need = [COL_WAGE, COL_EMP, COL_R, COL_RAI, COL_F0, COL_FAI, PROD_COL, "pss"]
df = df[need].dropna().copy()
df = df[(df[COL_EMP] > 0) & (df[COL_WAGE] > 0)].copy()
df[PROD_COL] = -np.log(1.0 - df[PROD_COL])

# Feasibility from CDF objects
df["ps_base_raw"] = np.clip(1.0 - df[COL_F0].to_numpy().astype(float), EPS, 1.0)
df["ps_with_raw"] = np.clip(1.0 - df[COL_FAI].to_numpy().astype(float), EPS, 1.0)

# Productivity multiplier: A_mult_prod = exp(pi)
pi = df[PROD_COL].to_numpy().astype(float)
if PROD_IS_PERCENT:
    pi = pi / 100.0
A_mult_prod = np.exp(pi)

# Basic objects
n = len(df)
w0 = df[COL_WAGE].to_numpy().astype(float)
L_obs = df[COL_EMP].to_numpy().astype(float)
L0_total = float(L_obs.sum())

# Scenario-specific orderings
idx0  = np.argsort(df[COL_R].to_numpy())     # baseline ordering
idxAI = np.argsort(df[COL_RAI].to_numpy())   # AI ordering

# Sorted baseline objects (and monotone ps)
w0_0   = take(w0, idx0)
Lobs_0 = take(L_obs, idx0)
Lshare_obs_0 = Lobs_0 / Lobs_0.sum()

ps0_0_raw = take(df["ps_base_raw"].to_numpy(), idx0)
ps0_0 = enforce_monotone_ps(ps0_0_raw)

# Sorted AI feasibility (and monotone ps)
psAI_raw = take(df["ps_with_raw"].to_numpy(), idxAI)
psAI = enforce_monotone_ps(psAI_raw)


# ============================================================
# 1) CALIBRATE tau TO MATCH BASELINE EMPLOYMENT SHARES
# ============================================================
def tau_objective(log_tau):
    tau = float(np.exp(log_tau))
    Lshare_model = implied_Lshare_flexible(w0_0, ps0_0, tau=tau)
    eps = 1e-14
    diff = np.log(Lshare_model + eps) - np.log(Lshare_obs_0 + eps)
    return float(np.sum(Lshare_obs_0 * diff**2))

res = minimize_scalar(
    tau_objective,
    bounds=(np.log(0.05), np.log(5.0)),
    method="bounded",
    options={"xatol": 1e-4}
)
tau_hat = float(np.exp(res.x))

print("\n=== tau calibration ===")
print(f"tau_hat = {tau_hat:.4f}")
print(f"objective = {res.fun:.6e}   success={res.success}")

Lshare_fit = implied_Lshare_flexible(w0_0, ps0_0, tau=tau_hat)
rmse = np.sqrt(np.mean((Lshare_fit - Lshare_obs_0)**2))
corr = np.corrcoef(Lshare_fit, Lshare_obs_0)[0, 1]
print(f"RMSE(Lshare) = {rmse:.4e},  Corr = {corr:.4f}")

tau = tau_hat


# ------------------------------------------------------------
# 2) BASELINE CALIBRATION OF psi (baseline has A_mult=1)
# ------------------------------------------------------------
Lshare_base_model_0 = implied_Lshare_flexible(w0_0, ps0_0, tau=tau)
L_base_model_0 = np.clip(L0_total * Lshare_base_model_0, EPS, None)

# Baseline: A_mult = 1 so A_term=1 => psi = w * L^{1/sigma}
psi0 = w0_0 * np.power(L_base_model_0, 1.0 / sigma)

df["L_base_model"] = put_back(L_base_model_0, idx0, n)
psi_unsorted = put_back(psi0, idx0, n)


# ============================================================
# PARTIAL EQUILIBRIUM (PE) wage changes (BASE MODEL)
# ============================================================
ps_base = df["ps_base_raw"].to_numpy()
ps_with = df["ps_with_raw"].to_numpy()

# Scarcity PE
df["dlogw_PE_scar"] = -(1.0 / sigma) * np.log(np.clip(ps_with, EPS, None) / np.clip(ps_base, EPS, None))
df["w_PE_scar"] = w0 * np.exp(df["dlogw_PE_scar"])

# Productivity PE
df["dlogw_PE_prod"] = ((sigma - 1.0) / sigma) * pi
df["w_PE_prod"] = w0 * np.exp(df["dlogw_PE_prod"])

# Both PE
df["dlogw_PE_all"] = df["dlogw_PE_scar"] + df["dlogw_PE_prod"]
df["w_PE_all"] = w0 * np.exp(df["dlogw_PE_all"])


# ============================================================
# EXACT GE WAGES (fixed point) — scenario-consistent ordering
# ============================================================

# --- 1) Scarcity-only GE: order by AI barrier; A_mult=1; ps=psAI
w_init_AI = take(w0, idxAI)
psi_AI    = take(psi_unsorted, idxAI)

w_ge_scar_AI, its1, err1 = solve_GE_fixed_point_exact(
    w_init=w_init_AI,
    psi=psi_AI,
    A_mult_vec=np.ones(n),
    ps_vec=psAI,
    L_total=L0_total,
    sigma=sigma, tau=tau, max_iters=MAX_ITERS, tol=TOL, damp=DAMP
)
df["w_GE_scar"] = put_back(w_ge_scar_AI, idxAI, n)
df["dlogw_GE_scar"] = np.log(df["w_GE_scar"] / w0)

# --- 2) Productivity-only GE: order by baseline barrier; A_mult=exp(pi); ps=ps0_0
A0_prod = take(A_mult_prod, idx0)
pi0 = take(pi, idx0)
w_init_prod_0 = w0_0 * np.exp(((sigma - 1.0) / sigma) * pi0)

w_ge_prod_0, its2, err2 = solve_GE_fixed_point_exact(
    w_init=w_init_prod_0,
    psi=psi0,
    A_mult_vec=A0_prod,
    ps_vec=ps0_0,
    L_total=L0_total,
    sigma=sigma, tau=tau, max_iters=MAX_ITERS, tol=TOL, damp=DAMP
)
df["w_GE_prod"] = put_back(w_ge_prod_0, idx0, n)
df["dlogw_GE_prod"] = np.log(df["w_GE_prod"] / w0)

# --- 3) Both channels GE: order by AI barrier; A_mult=exp(pi); ps=psAI
AAI_prod = take(A_mult_prod, idxAI)
piAI = take(pi, idxAI)
w_init_all_AI = take(w0, idxAI) * np.exp(((sigma - 1.0) / sigma) * piAI)

w_ge_all_AI, its3, err3 = solve_GE_fixed_point_exact(
    w_init=w_init_all_AI,
    psi=psi_AI,
    A_mult_vec=AAI_prod,
    ps_vec=psAI,
    L_total=L0_total,
    sigma=sigma, tau=tau, max_iters=MAX_ITERS, tol=TOL, damp=DAMP
)
df["w_GE_all"] = put_back(w_ge_all_AI, idxAI, n)
df["dlogw_GE_all"] = np.log(df["w_GE_all"] / w0)

print("\nGE fixed point diagnostics:")
print(f"  tau (calibrated) = {tau:.4f}")
print(f"  Scarcity-only:  iters={its1}, final max log diff={err1:.2e}")
print(f"  Prod-only:      iters={its2}, final max log diff={err2:.2e}")
print(f"  Both channels:  iters={its3}, final max log diff={err3:.2e}")


# ============================================================
# EMPLOYMENT RECONSTRUCTION (scenario-consistent ordering)
# ============================================================

# Scarcity-only (AI ordering)
Lshare_scar_AI = implied_Lshare_flexible(w_ge_scar_AI, psAI, tau=tau)
df["L_GE_scar"] = put_back(L0_total * Lshare_scar_AI, idxAI, n)

w_PE_scar_AI = take(df["w_PE_scar"].to_numpy(), idxAI)
Lshare_PE_scar_AI = implied_Lshare_flexible(w_PE_scar_AI, psAI, tau=tau)
df["L_PE_scar"] = put_back(L0_total * Lshare_PE_scar_AI, idxAI, n)

# Prod-only (baseline ordering)
Lshare_prod_0 = implied_Lshare_flexible(w_ge_prod_0, ps0_0, tau=tau)
df["L_GE_prod"] = put_back(L0_total * Lshare_prod_0, idx0, n)

w_PE_prod_0 = take(df["w_PE_prod"].to_numpy(), idx0)
Lshare_PE_prod_0 = implied_Lshare_flexible(w_PE_prod_0, ps0_0, tau=tau)
df["L_PE_prod"] = put_back(L0_total * Lshare_PE_prod_0, idx0, n)

# Both channels (AI ordering)
Lshare_all_AI = implied_Lshare_flexible(w_ge_all_AI, psAI, tau=tau)
df["L_GE_all"] = put_back(L0_total * Lshare_all_AI, idxAI, n)

w_PE_all_AI = take(df["w_PE_all"].to_numpy(), idxAI)
Lshare_PE_all_AI = implied_Lshare_flexible(w_PE_all_AI, psAI, tau=tau)
df["L_PE_all"] = put_back(L0_total * Lshare_PE_all_AI, idxAI, n)


# ============================================================
# PLOTTING & OUTPUTS
# ============================================================

df["logw0"] = np.log(df[COL_WAGE])

# 1) Scarcity-only binsreg
est_pe = binsreg(y="dlogw_PE_scar", x="logw0", w=df[COL_EMP],
                 data=df, line=LINE, ci=CI, cb=CB, polyreg=POLY)
est_ge = binsreg(y="dlogw_GE_scar", x="logw0", w=df[COL_EMP],
                 data=df, line=LINE, ci=CI, cb=CB, polyreg=POLY)
fig = build_overlay_plot(binsreg_result_object(est_pe), binsreg_result_object(est_ge),
                         "log(initial wage)", "Δ log wage",
                         "Scarcity-only: Δlog w vs log initial wage (Exact GE, tau calibrated)")
fig.save(os.path.join(OUTDIR, "binsreg_scarcity_only_logw0_exactGE_taucalib_basemodel.pdf"), width=7, height=4)
print(fig)

# 2) Productivity-only binsreg
est_pe = binsreg(y="dlogw_PE_prod", x="logw0", w=df[COL_EMP],
                 data=df, line=LINE, ci=CI, cb=CB, polyreg=POLY)
est_ge = binsreg(y="dlogw_GE_prod", x="logw0", w=df[COL_EMP],
                 data=df, line=LINE, ci=CI, cb=CB, polyreg=POLY)
fig = build_overlay_plot(binsreg_result_object(est_pe), binsreg_result_object(est_ge),
                         "log(initial wage)", "Δ log wage",
                         "Productivity-only: Δlog w vs log initial wage (Exact GE, tau calibrated)")
fig.save(os.path.join(OUTDIR, "binsreg_productivity_only_logw0_exactGE_taucalib_basemodel.pdf"), width=7, height=4)
print(fig)

# 3) Both channels binsreg
est_pe = binsreg(y="dlogw_PE_all", x="logw0", w=df[COL_EMP],
                 data=df, line=LINE, ci=CI, cb=CB, polyreg=POLY)
est_ge = binsreg(y="dlogw_GE_all", x="logw0", w=df[COL_EMP],
                 data=df, line=LINE, ci=CI, cb=CB, polyreg=POLY)
fig = build_overlay_plot(binsreg_result_object(est_pe), binsreg_result_object(est_ge),
                         "log(initial wage)", "Δ log wage",
                         "Both channels: Δlog w vs log initial wage (Exact GE, tau calibrated)")
fig.save(os.path.join(OUTDIR, "binsreg_bothchannels_logw0_exactGE_taucalib_basemodel.pdf"), width=7, height=4)
print(fig)

# --- Histograms (employment-weighted; scenario-consistent L) ---
plot_hist_three(w0, df["w_PE_scar"], df["w_GE_scar"],
                df["L_base_model"], df["L_PE_scar"], df["L_GE_scar"],
                "hist_scarcity_exactGE_taucalib_basemodel.pdf",
                "Wage Dist: Scarcity-only (Exact GE, tau calibrated)")

plot_hist_three(w0, df["w_PE_prod"], df["w_GE_prod"],
                df["L_base_model"], df["L_PE_prod"], df["L_GE_prod"],
                "hist_prod_exactGE_taucalib_basemodel.pdf",
                "Wage Dist: Productivity-only (Exact GE, tau calibrated)")

plot_hist_three(w0, df["w_PE_all"], df["w_GE_all"],
                df["L_base_model"], df["L_PE_all"], df["L_GE_all"],
                "hist_bothchannels_exactGE_taucalib_basemodel.pdf",
                "Wage Dist: Both channels (Exact GE, tau calibrated)")

# --- KDE plots (employment-weighted; trimmed) ---
plot_kde_three(w0, df["w_PE_scar"], df["w_GE_scar"],
               df["L_base_model"], df["L_PE_scar"], df["L_GE_scar"],
               "kde_scarcity_exactGE_taucalib_basemodel.pdf", "",
               bw_adjust=1.3)

plot_kde_three(w0, df["w_PE_prod"], df["w_GE_prod"],
               df["L_base_model"], df["L_PE_prod"], df["L_GE_prod"],
               "kde_prod_exactGE_taucalib_basemodel.pdf", "",
               bw_adjust=1.3)

plot_kde_three(w0, df["w_PE_all"], df["w_GE_all"],
               df["L_base_model"], df["L_PE_all"], df["L_GE_all"],
               "kde_bothchannels_exactGE_taucalib_basemodel.pdf", "",
               bw_adjust=1.3)

# ============================================================
# INEQUALITY TABLES
# ============================================================
rows = []
rows.append({"scenario":"Baseline",           **ineq_stats_logw(df[COL_WAGE],     df["L_base_model"])})
rows.append({"scenario":"Scarcity PE",        **ineq_stats_logw(df["w_PE_scar"],  df["L_PE_scar"])})
rows.append({"scenario":"Scarcity GE",        **ineq_stats_logw(df["w_GE_scar"],  df["L_GE_scar"])})
rows.append({"scenario":"Productivity PE",    **ineq_stats_logw(df["w_PE_prod"],  df["L_PE_prod"])})
rows.append({"scenario":"Productivity GE",    **ineq_stats_logw(df["w_GE_prod"],  df["L_GE_prod"])})
rows.append({"scenario":"Both channels PE",   **ineq_stats_logw(df["w_PE_all"],   df["L_PE_all"])})
rows.append({"scenario":"Both channels GE",   **ineq_stats_logw(df["w_GE_all"],   df["L_GE_all"])})

ineq_table = pd.DataFrame(rows)
print("\nInequality Statistics:")
print(ineq_table)
ineq_table.to_csv(os.path.join(OUTDIR, "inequality_table_exactGE_taucalib_basemodel.csv"), index=False)

# Save main data
df.to_csv(os.path.join(OUTDIR, "counterfactual_results_full_exactGE_taucalib_basemodel.csv"), index=False)

print("\nAll done. Results saved to", OUTDIR)


# ============================================================
# OPTIONAL: Same PSS regressions / binscatters using Scarcity-only GE outcomes
# (copied structure from your NEW MODEL end exercise)
# ============================================================
df["PSS"] = df["pss"]
df["PSS2"] = df["PSS"]**2

df["dlog_w_GE_scar"] = np.log(df["w_GE_scar"]) - np.log(df[COL_WAGE])
df["dlog_L_GE_scar"] = np.log(df["L_GE_scar"]) - np.log(df["L_base_model"])

# Quadratic WLS regressions (exclude small PSS)
mask = (
    np.isfinite(df["PSS"]) &
    (df["PSS"] > 0.1) &
    np.isfinite(df["dlog_w_GE_scar"]) &
    np.isfinite(df["dlog_L_GE_scar"]) &
    np.isfinite(df[COL_EMP]) &
    (df[COL_EMP] > 0)
)
wts = df.loc[mask, COL_EMP]

X_w = sm.add_constant(df.loc[mask, ["PSS", "PSS2"]])
y_w = df.loc[mask, "dlog_w_GE_scar"]
res_w = sm.WLS(y_w, X_w, weights=wts).fit()

print("\nSCARCITY-ONLY GE wage response vs PSS (quadratic, PSS > 0.1):")
print(f"  coef(PSS)  = {res_w.params['PSS']:.4f}")
print(f"  coef(PSS²) = {res_w.params['PSS2']:.4f}")
print(f"  R^2        = {res_w.rsquared:.4f}")

X_l = sm.add_constant(df.loc[mask, ["PSS", "PSS2"]])
y_l = df.loc[mask, "dlog_L_GE_scar"]
res_l = sm.WLS(y_l, X_l, weights=wts).fit()

print("\nSCARCITY-ONLY GE employment response vs PSS (quadratic, PSS > 0.1):")
print(f"  coef(PSS)  = {res_l.params['PSS']:.4f}")
print(f"  coef(PSS²) = {res_l.params['PSS2']:.4f}")
print(f"  R^2        = {res_l.rsquared:.4f}")

# Linear binscatter panels (PSS >= 0.05)
mask2 = (
    np.isfinite(df["PSS"]) &
    (df["PSS"] >= 0.05) &
    np.isfinite(df["dlog_w_GE_scar"]) &
    np.isfinite(df["dlog_L_GE_scar"]) &
    np.isfinite(df[COL_EMP]) &
    (df[COL_EMP] > 0)
)
df2 = df.loc[mask2].copy()

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.right": False,
    "axes.spines.top": False,
})

x = df2["PSS"].to_numpy()
wts2 = df2[COL_EMP].to_numpy()

# Wage panel
Xw = sm.add_constant(x)
yw = df2["dlog_w_GE_scar"].to_numpy()
rw = sm.WLS(yw, Xw, weights=wts2).fit()

est_w = binsreg(
    y="dlog_w_GE_scar",
    x="PSS",
    w=df2[COL_EMP],
    data=df2,
    line=(3,3), ci=(3,3), cb=(3,3),
    polyreg=1,
    nbins=20
)
bw = est_w.data_plot[0]

plt.figure(figsize=(6.3,4.0))
plt.scatter(bw.dots["x"], bw.dots["fit"], s=40, edgecolor="k", linewidth=0.3)
plt.plot(bw.line["x"], bw.line["fit"], linewidth=1.5)
plt.fill_between(bw.cb["x"], bw.cb["cb_l"], bw.cb["cb_r"], alpha=0.12)
plt.axhline(0, color="0.45", linewidth=0.8, linestyle="--")

a_w, b_w = rw.params
eq_w = f"$\\Delta\\log w = {a_w:.3g} {b_w:+.3g}\\cdot PSS$\\n$R^2 = {rw.rsquared:.3f}$"
plt.annotate(eq_w, xy=(0.97, 0.05), xycoords="axes fraction",
             ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"),
             fontsize=9)

plt.xlabel("Potential Supply Shift (PSS)")
plt.ylabel("Δ log wage (GE, scarcity only)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "binscatter_scarcity_GE_wage_linear_basemodel.png"))
plt.show()
plt.close()

# Employment panel
Xl = sm.add_constant(x)
yl = df2["dlog_L_GE_scar"].to_numpy()
rl = sm.WLS(yl, Xl, weights=wts2).fit()

est_l = binsreg(
    y="dlog_L_GE_scar",
    x="PSS",
    w=df2[COL_EMP],
    data=df2,
    line=(3,3), ci=(3,3), cb=(3,3),
    polyreg=1,
    nbins=20
)
bl = est_l.data_plot[0]

plt.figure(figsize=(6.3,4.0))
plt.scatter(bl.dots["x"], bl.dots["fit"], s=40, edgecolor="k", linewidth=0.3)
plt.plot(bl.line["x"], bl.line["fit"], linewidth=1.5)
plt.fill_between(bl.cb["x"], bl.cb["cb_l"], bl.cb["cb_r"], alpha=0.12)
plt.axhline(0, color="0.45", linewidth=0.8, linestyle="--")

a_l, b_l = rl.params
eq_l = f"$\\Delta\\log L = {a_l:.3g} {b_l:+.3g}\\cdot PSS$\\n$R^2 = {rl.rsquared:.3f}$"
plt.annotate(eq_l, xy=(0.97, 0.05), xycoords="axes fraction",
             ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"),
             fontsize=9)

plt.xlabel("Potential Supply Shift (PSS)")
plt.ylabel("Δ log employment (GE, scarcity only)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "binscatter_scarcity_GE_employment_linear_basemodel.png"))
plt.show()
plt.close()

print("\nScarcity-only GE linear regressions (PSS ≥ 0.05):")
print(f"Wage:       coef(PSS) = {b_w:.4f}, R^2 = {rw.rsquared:.4f}")
print(f"Employment: coef(PSS) = {b_l:.4f}, R^2 = {rl.rsquared:.4f}")









# ============================================================
# QQ-style percentile table + scatter (LOG wages)
# x-axis: log(wage) percentile pre-AI
# y-axis: log(wage) percentile post-AI (same percentile)
# Includes 20 percentiles: 5,10,...,95,99 (includes 90/95/99)
# Runs for: Scarcity channel, Productivity channel, Total effect
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------- choose GE vs PE ----------
USE_GE = True  # False -> PE

if USE_GE:
    scen_map = {
        "Scarcity":     ("w_GE_scar", "L_GE_scar"),
        "Productivity": ("w_GE_prod", "L_GE_prod"),
        "Total":        ("w_GE_all",  "L_GE_all"),
    }
    tag = "GE"
else:
    scen_map = {
        "Scarcity":     ("w_PE_scar", "L_PE_scar"),
        "Productivity": ("w_PE_prod", "L_PE_prod"),
        "Total":        ("w_PE_all",  "L_PE_all"),
    }
    tag = "PE"

# Baseline wage + baseline weights
COL_WAGE = "mean_annual_wage_2024"
COL_EMP  = "total_employment_2024"

w_pre = df[COL_WAGE].to_numpy(dtype=float)
w_pre_wts = (df["L_base_model"] if "L_base_model" in df.columns else df[COL_EMP]).to_numpy(dtype=float)

# 20 percentiles: 5..95 by 5 + 99
pct_list = list(range(5, 100, 5)) + [97,98,99]
q_list = np.array(pct_list, dtype=float) / 100.0

def weighted_quantile_1d(x, w, qs):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    qs = np.asarray(qs, float)

    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[m], w[m]
    if x.size == 0:
        return np.full_like(qs, np.nan, dtype=float)

    idx = np.argsort(x)
    x = x[idx]
    w = w[idx]

    cw = np.cumsum(w)
    cw = cw / cw[-1]
    return np.interp(qs, cw, x)

# ---------- baseline LOG-quantiles ----------
logw_pre = np.log(np.clip(w_pre, 1e-12, None))
pre_q = weighted_quantile_1d(logw_pre, w_pre_wts, q_list)

# ---------- build table (log wages) ----------
rows = []
for name, (w_col, L_col) in scen_map.items():
    w_post = df[w_col].to_numpy(dtype=float)
    w_post_wts = df[L_col].to_numpy(dtype=float)

    logw_post = np.log(np.clip(w_post, 1e-12, None))
    post_q = weighted_quantile_1d(logw_post, w_post_wts, q_list)

    for p, xq, yq in zip(pct_list, pre_q, post_q):
        rows.append({
            "scenario": name,
            "percentile": p,
            "logw_pre": xq,
            "logw_post": yq,
            "dlogw_q": (yq - xq) if (np.isfinite(xq) and np.isfinite(yq)) else np.nan,
            "pct_change_q": 100.0 * (np.exp(yq - xq) - 1.0) if (np.isfinite(xq) and np.isfinite(yq)) else np.nan,
        })

q_table_long = pd.DataFrame(rows)

wide = (
    q_table_long
    .pivot(index="percentile", columns="scenario", values="logw_post")
    .reset_index()
)
wide.insert(1, "logw_pre", pre_q)
wide = wide.sort_values("percentile")

print("\nLOG wage percentile table (employment-weighted):")
print(wide.to_string(index=False))

wide.to_csv(os.path.join(OUTDIR, f"logw_percentiles_pre_vs_post_{tag}.csv"), index=False)
q_table_long.to_csv(os.path.join(OUTDIR, f"logw_percentiles_pre_vs_post_long_{tag}.csv"), index=False)

# ---------- plotting ----------
def qq_scatter_log(pre_q, post_q, pct_list, title, outpath):
    pre_q = np.asarray(pre_q, float)
    post_q = np.asarray(post_q, float)

    m = np.isfinite(pre_q) & np.isfinite(post_q)
    x = pre_q[m]
    y = post_q[m]
    pcts = np.asarray(pct_list)[m]

    if x.size == 0:
        print(f"[WARN] No finite points for plot: {outpath}")
        return

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 0.5
    lo -= pad
    hi += pad

    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(x, y, s=45, edgecolor="k", linewidth=0.3)
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="0.45")  # 45-degree line in log space

    for xi, yi, pp in zip(x, y, pcts):
        plt.annotate(str(int(pp)), (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=8)

    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("log(wage) percentile pre-AI")
    plt.ylabel("log(wage) percentile post-AI")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.show()
    plt.close()

for name, (w_col, L_col) in scen_map.items():
    w_post = df[w_col].to_numpy(dtype=float)
    w_post_wts = df[L_col].to_numpy(dtype=float)

    logw_post = np.log(np.clip(w_post, 1e-12, None))
    post_q = weighted_quantile_1d(logw_post, w_post_wts, q_list)

    outpath = os.path.join(OUTDIR, f"qq_log_pre_vs_post_{name.lower()}_{tag}.png")
    qq_scatter_log(
        pre_q=pre_q,
        post_q=post_q,
        pct_list=pct_list,
        title=f"Pre-AI vs Post-AI log-wage percentiles ({name}, {tag})",
        outpath=outpath
    )

print(f"\nSaved log-percentile tables + log-QQ scatter plots to: {OUTDIR}")







import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------- settings ----------
USE_GE = True  # False -> PE
WEIGHT_POST_WITH_BASELINE = True  # True: weights for y use baseline weights; False: use post weights

if USE_GE:
    scen_map = {
        "Scarcity":     ("w_GE_scar", "L_GE_scar"),
        "Productivity": ("w_GE_prod", "L_GE_prod"),
        "Total":        ("w_GE_all",  "L_GE_all"),
    }
    tag = "GE"
else:
    scen_map = {
        "Scarcity":     ("w_PE_scar", "L_PE_scar"),
        "Productivity": ("w_PE_prod", "L_PE_prod"),
        "Total":        ("w_PE_all",  "L_PE_all"),
    }
    tag = "PE"

COL_WAGE = "mean_annual_wage_2024"
COL_EMP  = "total_employment_2024"

w_pre = df[COL_WAGE].to_numpy(float)
w_pre_wts = (df["L_base_model"] if "L_base_model" in df.columns else df[COL_EMP]).to_numpy(float)

# 20 cutpoints: 0,5,10,...,95,99,100  (note last bin is 99-100)
pct_list = list(range(5, 100, 5)) + [99]
cuts = np.array([0] + pct_list + [100], float) / 100.0

def weighted_quantile_1d(x, w, qs):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    qs = np.asarray(qs, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[m], w[m]
    if x.size == 0:
        return np.full_like(qs, np.nan, dtype=float)
    idx = np.argsort(x)
    x, w = x[idx], w[idx]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    return np.interp(qs, cw, x)

def weighted_mean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan
    x, w = x[m], w[m]
    return float(np.sum(w * x) / np.sum(w))

# ---------- define bins by PRE wages (using baseline employment weights) ----------
# Compute wage thresholds in levels (or log; monotone either way). Use levels for clarity.
thr = weighted_quantile_1d(w_pre, w_pre_wts, cuts)

# Assign each occupation to a bin based on pre wage
# Bin i is [thr[i], thr[i+1]) except last includes endpoint
bin_id = np.full(len(df), -1, dtype=int)
for i in range(len(thr) - 1):
    lo, hi = thr[i], thr[i+1]
    if i < len(thr) - 2:
        m = (w_pre >= lo) & (w_pre < hi)
    else:
        m = (w_pre >= lo) & (w_pre <= hi)
    bin_id[m] = i

# Bin labels corresponding to upper percentile cut (5,10,...,95,99,100)
bin_label = [int(c*100) for c in cuts[1:]]  # length = number of bins

# Pre x-values for each bin: weighted mean log(pre wage)
logw_pre = np.log(np.clip(w_pre, 1e-12, None))
x_bybin = []
for i in range(len(bin_label)):
    m = bin_id == i
    x_bybin.append(weighted_mean(logw_pre[m], w_pre_wts[m]))
x_bybin = np.array(x_bybin, float)

# ---------- for each scenario: y-values are log(post wage) for SAME occupations in each pre bin ----------
out_rows = []
for name, (w_col, L_col) in scen_map.items():
    w_post = df[w_col].to_numpy(float)
    logw_post = np.log(np.clip(w_post, 1e-12, None))

    if WEIGHT_POST_WITH_BASELINE:
        wts_y = w_pre_wts
        wts_tag = "baselineWts"
    else:
        wts_y = df[L_col].to_numpy(float)
        wts_tag = "postWts"

    y_bybin = []
    for i in range(len(bin_label)):
        m = bin_id == i
        y_bybin.append(weighted_mean(logw_post[m], wts_y[m]))
    y_bybin = np.array(y_bybin, float)

    # save table rows
    for lab, xq, yq in zip(bin_label, x_bybin, y_bybin):
        out_rows.append({
            "scenario": name,
            "bin_upper_percentile": lab,  # e.g. 5 means the 0-5 bin, 10 means 5-10 bin, ... 99 means 95-99, 100 means 99-100
            "logw_pre_binmean": xq,
            "logw_post_binmean": yq,
            "dlogw_bin": (yq - xq) if (np.isfinite(xq) and np.isfinite(yq)) else np.nan,
            "pct_change_bin": 100.0 * (np.exp(yq - xq) - 1.0) if (np.isfinite(xq) and np.isfinite(yq)) else np.nan,
        })

    # ---------- plot scatter with 45-degree line ----------
    m = np.isfinite(x_bybin) & np.isfinite(y_bybin)
    x = x_bybin[m]
    y = y_bybin[m]
    labs = np.array(bin_label)[m]

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 0.5
    lo -= pad
    hi += pad

    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(x, y, s=55, edgecolor="k", linewidth=0.3)
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="0.45")

    for xi, yi, pp in zip(x, y, labs):
        # annotate only the key ones if you prefer; currently annotates all bins
        plt.annotate(str(int(pp)), (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=8)

    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("log(wage) pre-AI (bin mean by pre-AI percentile bins)")
    plt.ylabel("log(wage) post-AI (same occupations; bin mean)")
    plt.title(f"Pre-binned occupations: log-wage pre vs post ({name}, {tag}, {wts_tag})")
    plt.tight_layout()

    fname = f"scatter_prebin_log_pre_vs_post_{name.lower()}_{tag}_{wts_tag}.png"
    plt.savefig(os.path.join(OUTDIR, fname), dpi=200)
    plt.show()
    plt.close()

# ---------- save table ----------
prebin_table = pd.DataFrame(out_rows)
prebin_table.to_csv(os.path.join(OUTDIR, f"prebin_logw_pre_vs_post_bins_{tag}.csv"), index=False)

print("\nSaved pre-binned (fixed percentile groups) table + plots to:", OUTDIR)





# ============================================================
# Wage changes by pre-AI wage percentiles
#   - Fixed pre-AI percentile bins
#   - Same occupations tracked post-AI
#   - Smoothed curves via LOWESS + interpolation
#   - Publication-quality (LaTeX-style) plots
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import warnings
from scipy.interpolate import interp1d
import statsmodels.api as sm


def make_prebin_changes(df, w_post_col, out_prefix, OUTDIR,
                        COL_WAGE="mean_annual_wage_2024",
                        COL_EMP="total_employment_2024",
                        percentiles=(0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,
                                     0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99)):
    """
    Fixed pre-AI percentile bins:
      1) Compute pre-AI wage cutoffs using baseline employment weights
      2) Bin occupations by pre-AI wage
      3) For each bin: employment-weighted mean log(w_pre) and log(w_post)
      4) Plot Δlog wage and % change vs pre-AI percentile (smoothed)
    """

    # ----------------------------
    # Data prep
    # ----------------------------
    d = df[[COL_WAGE, COL_EMP, w_post_col]].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    d = d[(d[COL_WAGE] > 0) & (d[COL_EMP] > 0) & (d[w_post_col] > 0)].copy()

    w_pre  = d[COL_WAGE].to_numpy(float)
    w_post = d[w_post_col].to_numpy(float)
    emp    = d[COL_EMP].to_numpy(float)

    # Weighted pre-AI wage cutoffs
    qs = np.array(percentiles, float)
    cuts = weighted_quantile(w_pre, emp, qs)

    edges  = np.r_[-np.inf, cuts]
    labels = (qs * 100).astype(int)

    d["bin"] = pd.cut(w_pre, bins=edges, labels=labels,
                      right=True, include_lowest=True)

    # ----------------------------
    # Bin means (employment-weighted)
    # ----------------------------
    x_bybin, y_bybin, p = [], [], []

    for lab in labels:
        g = d[d["bin"] == lab]
        if len(g) == 0:
            x_bybin.append(np.nan)
            y_bybin.append(np.nan)
            p.append(lab)
            continue

        ww = g[COL_EMP].to_numpy(float)
        ww = ww / ww.sum()

        x = np.log(g[COL_WAGE].to_numpy(float))
        y = np.log(g[w_post_col].to_numpy(float))

        x_bybin.append(np.sum(ww * x))
        y_bybin.append(np.sum(ww * y))
        p.append(lab)

    x_bybin = np.array(x_bybin)
    y_bybin = np.array(y_bybin)
    p       = np.array(p, float)

    dlog_bybin   = y_bybin - x_bybin
    pctchg_bybin = 100.0 * (np.exp(dlog_bybin) - 1.0)

    m = np.isfinite(p) & np.isfinite(dlog_bybin)
    p, dlog_bybin, pctchg_bybin = p[m], dlog_bybin[m], pctchg_bybin[m]

    # ----------------------------
    # Plot styling (LaTeX look)
    # ----------------------------
    _orig = dict(mpl.rcParams)
    try:
        mpl.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.labelsize": 12,
            "font.size": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        })
    except Exception:
        pass

    def smooth_curve(x, y, xgrid, lowess_frac=0.35, interp_kind="cubic"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lo = sm.nonparametric.lowess(y, x, frac=lowess_frac,
                                         it=1, return_sorted=True)
        xs, ys = lo[:,0], lo[:,1]
        if interp_kind == "cubic" and len(xs) < 4:
            interp_kind = "linear"
        f = interp1d(xs, ys, kind=interp_kind,
                     bounds_error=False, fill_value="extrapolate")
        return f(xgrid)

    xgrid = np.linspace(p.min(), p.max(), 250)
    xticks = [5, 25, 50, 75, 90, 95, 99]
    xticks = [t for t in xticks if p.min() <= t <= p.max()]

    # ----------------------------
    # Δ log wage plot
    # ----------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=200)
    ax.plot(xgrid,
            smooth_curve(p, dlog_bybin, xgrid),
            linewidth=2.2)
    ax.scatter(p, dlog_bybin, s=35, zorder=3)
    ax.axhline(0, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"Pre-AI wage percentile")
    ax.set_ylabel(r"$\Delta \log w$ (post $-$ pre)")
    ax.set_xticks(xticks)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"dlog_by_prebin_{out_prefix}.pdf"))
    fig.savefig(os.path.join(OUTDIR, f"dlog_by_prebin_{out_prefix}.png"))
    plt.close(fig)

    # ----------------------------
    # % wage change plot
    # ----------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=200)
    ax.plot(xgrid,
            smooth_curve(p, pctchg_bybin, xgrid),
            linewidth=2.2)
    ax.scatter(p, pctchg_bybin, s=35, zorder=3)
    ax.axhline(0, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"Pre-AI wage percentile")
    ax.set_ylabel(r"\% change in wage")
    ax.set_xticks(xticks)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"pctchg_by_prebin_{out_prefix}.pdf"))
    fig.savefig(os.path.join(OUTDIR, f"pctchg_by_prebin_{out_prefix}.png"))

    mpl.rcParams.update(_orig)

    return pd.DataFrame({
        "pctl_bin_upper": p,
        "mean_log_w_pre": x_bybin[m],
        "mean_log_w_post": y_bybin[m],
        "dlog_w": dlog_bybin,
        "pct_change": pctchg_bybin
    })


# ============================================================
# CALLS (GE outcomes)
# ============================================================

# Scarcity-only GE
tab_scar = make_prebin_changes(
    df=df,
    w_post_col="w_GE_scar",
    out_prefix="scarcity_GE",
    OUTDIR=OUTDIR,
    COL_WAGE=COL_WAGE,
    COL_EMP=COL_EMP
)

# Productivity-only GE
tab_prod = make_prebin_changes(
    df=df,
    w_post_col="w_GE_prod",
    out_prefix="productivity_GE",
    OUTDIR=OUTDIR,
    COL_WAGE=COL_WAGE,
    COL_EMP=COL_EMP
)

# Total effect (scarcity + productivity) GE
tab_tot = make_prebin_changes(
    df=df,
    w_post_col="w_GE_all",
    out_prefix="tot_GE",
    OUTDIR=OUTDIR,
    COL_WAGE=COL_WAGE,
    COL_EMP=COL_EMP
)

# Optional: save tables
tab_scar.to_csv(os.path.join(OUTDIR, "prebin_changes_scarcity_GE.csv"), index=False)
tab_prod.to_csv(os.path.join(OUTDIR, "prebin_changes_productivity_GE.csv"), index=False)
tab_tot.to_csv(os.path.join(OUTDIR, "prebin_changes_total_GE.csv"), index=False)

print("Saved smoothed pre-AI percentile wage-change plots and tables.")


# Optional: save the bin tables
tab_scar.to_csv(os.path.join(OUTDIR, "prebin_changes_scarcity_GE.csv"), index=False)
tab_prod.to_csv(os.path.join(OUTDIR, "prebin_changes_productivity_GE.csv"), index=False)