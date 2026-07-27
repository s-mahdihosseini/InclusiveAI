"""
Option B robustness check: redefine the inflow-share moment as
    inflow_share[o] = Σ_s  μ_s · δ_s · π(o|s,switch) / Σ_{s,o} μ_s · δ_s · π(o|s,switch)
i.e. weight by actual switcher mass (employment × outflow rate), not just
employment.

This script:
  1. Recomputes the data moment under Option B
  2. Monkey-patches compute_flow_moments to use Option B model moment
  3. Recalibrates twoway κ
  4. Runs AI steady-state counterfactual
  5. Runs transition path (T=100)
  6. Produces comparison figures (inequality dynamics + top 30)
  7. Saves everything to sdm_output/option_b/

Nothing in the existing codebase is overwritten.
"""

import os, sys, json, time, warnings
import numpy as np

warnings.filterwarnings("ignore")

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
OUTDIR = os.path.join(THIS_DIR, "sdm_output", "option_b")
os.makedirs(OUTDIR, exist_ok=True)

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
    compute_flow_moments as compute_flow_moments_A,  # Option A original
    calibrate_twoway_stable as calibrate_twoway_stable_A,
    _solve_and_moments as _solve_and_moments_A,
)
from flow_data import load_flow_moments

# =====================================================================
# 0. Load data
# =====================================================================
print("=" * 70)
print("OPTION B ROBUSTNESS CHECK")
print("=" * 70)

data = build_model_data(verbose=True)
J = data["J"]
soc3_codes = data["soc3_codes"]
soc3_names = data["soc3_names"]
data_emp   = data["employment"]
d_pre      = data["d_so"]
d_post     = data["d_so_AI"]

# Load flow data (Option A moments — we will recompute the inflow share)
flow_A = load_flow_moments(soc3_codes, data_emp, verbose=True)

# =====================================================================
# 1. Recompute data inflow share under Option B
# =====================================================================
# Option B: weight by switch_mass[s] (= emp[s] * outflow_rate[s])
# which is exactly the column-sum of the count matrix N.
# flow_A already has switch_mass_data and pi_switch_data.

switch_mass = flow_A["switch_mass_data"]  # (J,) raw cross-SOC3 switchers
pi_switch_data = flow_A["pi_switch_data"]  # (J, J) conditional on switching

# Option B inflow = Σ_s switch_mass[s] * π(o|s,switch) / total_switches
# which is just column-sum of N / total
w_switcher = switch_mass.copy()
w_switcher = w_switcher / max(w_switcher.sum(), 1e-15)
inflow_share_B = (w_switcher[:, None] * pi_switch_data).sum(axis=0)
inflow_share_B = inflow_share_B / inflow_share_B.sum()

# Compare with Option A
inflow_share_A = flow_A["inflow_share_switch_data"]
corr_AB = np.corrcoef(inflow_share_A, inflow_share_B)[0, 1]
max_diff = np.max(np.abs(inflow_share_A - inflow_share_B))
print(f"\n--- Inflow share moment comparison ---")
print(f"  Correlation(A, B)    = {corr_AB:.6f}")
print(f"  Max absolute diff    = {max_diff:.6f}")
print(f"  Mean absolute diff   = {np.mean(np.abs(inflow_share_A - inflow_share_B)):.6f}")
print(f"  Option A range: [{inflow_share_A.min():.5f}, {inflow_share_A.max():.5f}]")
print(f"  Option B range: [{inflow_share_B.min():.5f}, {inflow_share_B.max():.5f}]")

# Build Option B flow data dict (same outflow rates, different inflow share)
flow_B = dict(flow_A)
flow_B["inflow_share_switch_data"] = inflow_share_B

# =====================================================================
# 2. Monkey-patch compute_flow_moments for Option B
# =====================================================================
def compute_flow_moments_B(policy, mu, data_emp):
    """
    Option B: inflow share weighted by model-implied switcher mass
    (μ_s × switch_prob_s) instead of just employment.
    """
    J = policy.shape[0]
    s_idx = np.arange(J)
    stay_prob = policy[s_idx, s_idx]
    switch_prob = 1.0 - stay_prob

    # π^switch(o | s) = π(o | s) / (1 - π(s|s))
    pi_switch = policy.copy()
    np.fill_diagonal(pi_switch, 0.0)
    row_sum = pi_switch.sum(axis=1)
    pi_switch = pi_switch / np.maximum(row_sum[:, None], 1e-15)

    # Outflow rate per origin
    outflow_rate_model = switch_prob.copy()

    # Option B: weight by μ_s * switch_prob_s (model-implied switcher mass)
    switcher_mass = mu * switch_prob
    w_switcher = switcher_mass / max(switcher_mass.sum(), 1e-15)
    inflow_share_model = (w_switcher[:, None] * pi_switch).sum(axis=0)
    inflow_share_model = inflow_share_model / max(inflow_share_model.sum(), 1e-15)

    # Aggregate mobility
    mobility_rate = (mu * switch_prob).sum() / max(mu.sum(), 1e-15)

    return {
        "pi_switch_model":          pi_switch,
        "inflow_share_switch_model": inflow_share_model,
        "outflow_rate_model":        outflow_rate_model,
        "mobility_rate":             mobility_rate,
    }

# Monkey-patch the module so calibrate_twoway_stable uses Option B
import sdm_kappa_vec
_original_cfm = sdm_kappa_vec.compute_flow_moments
sdm_kappa_vec.compute_flow_moments = compute_flow_moments_B

# =====================================================================
# 3. Calibrate twoway κ under Option B
# =====================================================================
print("\n" + "=" * 70)
print("CALIBRATING TWO-WAY κ (Option B moments)")
print("=" * 70)

# Load scalar κ for initial values
KAPPA_JSON = os.path.join(THIS_DIR, "sdm_output", "kappa_heterogeneity_results.json")
with open(KAPPA_JSON) as f:
    kh = json.load(f)
scalar_kappa = kh["benchmark_scalar"]["kappa"]

tier3 = {"kappa": scalar_kappa, "tau": kh["benchmark_scalar"]["tau"]}
p = SdmParams(data, tier3)

kspec_B, hist_B, (wages_cal, B_cal, V_cal, mu_cal, L_cal, pol_cal) = \
    calibrate_twoway_stable_A(
        p, d_pre, data_emp, flow_B,
        kappa_out_init=scalar_kappa / 2,
        kappa_in_init=scalar_kappa / 2,
        step_out=0.20, step_in=0.20,
        max_iter=120, tol=5e-3,
        reinvert_amenity_every=4,
        amenity_tol=2e-3, amenity_max_iter=30,
        verbose=True,
    )

print(f"\nCalibrated κ_out range: [{kspec_B.kappa_out.min():.3f}, {kspec_B.kappa_out.max():.3f}]")
print(f"Calibrated κ_in  range: [{kspec_B.kappa_in.min():.3f}, {kspec_B.kappa_in.max():.3f}]")

# Also load Option A κ for comparison
kappa_out_A = np.array(kh["twoway"]["kappa_out"])
kappa_in_A  = np.array(kh["twoway"]["kappa_in"])
corr_ko = np.corrcoef(kappa_out_A, kspec_B.kappa_out)[0, 1]
corr_ki = np.corrcoef(kappa_in_A,  kspec_B.kappa_in)[0, 1]
print(f"\nκ_out correlation (A vs B): {corr_ko:.4f}")
print(f"κ_in  correlation (A vs B): {corr_ki:.4f}")

# Save calibration
cal_out = {
    "kappa_out": kspec_B.kappa_out.tolist(),
    "kappa_in":  kspec_B.kappa_in.tolist(),
    "history":   hist_B,
    "kappa_out_A": kappa_out_A.tolist(),
    "kappa_in_A":  kappa_in_A.tolist(),
    "inflow_share_data_A": inflow_share_A.tolist(),
    "inflow_share_data_B": inflow_share_B.tolist(),
    "corr_kappa_out_AB": corr_ko,
    "corr_kappa_in_AB":  corr_ki,
}
with open(os.path.join(OUTDIR, "calibration_optB.json"), "w") as f:
    json.dump(cal_out, f, indent=2)
print(f"Saved calibration to {OUTDIR}/calibration_optB.json")

# =====================================================================
# 4. AI steady-state counterfactual
# =====================================================================
print("\n" + "=" * 70)
print("AI STEADY-STATE COUNTERFACTUAL (Option B)")
print("=" * 70)

# Re-solve pre-AI to get clean a_o, B
p_cf = SdmParams(data, tier3)
wages_pre, B_pre, V_pre, mu_pre, L_pre, pol_pre = solve_steady_state_vec(
    p_cf, kspec_B, d_pre, data_emp,
    invert_amenities_flag=True,
    amenity_tol=1e-4, amenity_max_iter=80,
    max_iter=200, tol=1e-4, verbose=False,
)
a_o = p_cf.a_o.copy()

# Post-AI: freeze a_o, B
p_cf.a_o = a_o.copy()
wages_post, B_post, V_post, mu_post, L_post, pol_post = solve_steady_state_vec(
    p_cf, kspec_B, d_post, data_emp,
    invert_amenities_flag=False,
    B_ext=B_pre,
    wages_init=wages_pre,
    max_iter=500, tol=1e-5, damping=0.2, verbose=False,
)

var_pre  = compute_var_log_occ_wage(wages_pre, mu_pre)
var_post = compute_var_log_occ_wage(wages_post, mu_post)
gini_pre  = compute_gini_occ_wages(wages_pre, mu_pre)
gini_post = compute_gini_occ_wages(wages_post, mu_post)

# Employment-weighted percentile gaps
def _percentile_gaps(wages, mu):
    w = mu / max(mu.sum(), 1e-15)
    lw = np.log(np.maximum(wages, 1e-10))
    idx = np.argsort(lw)
    cum = np.cumsum(w[idx])
    def _pct(q):
        i = np.searchsorted(cum, q)
        i = min(i, len(lw) - 1)
        return lw[idx[i]]
    return _pct(0.9) - _pct(0.5), _pct(0.9) - _pct(0.1)

p90p50_pre, p90p10_pre   = _percentile_gaps(wages_pre, mu_pre)
p90p50_post, p90p10_post = _percentile_gaps(wages_post, mu_post)

sigma = data["sigma"]
B = B_pre
def _Y(L):
    return (np.sum(B**(1.0/sigma) * np.maximum(L, 1e-10)**((sigma-1)/sigma))
            **(sigma/(sigma-1)))
Y_pre = _Y(mu_pre)
Y_post = _Y(mu_post)
dlogY = np.log(Y_post / Y_pre)

print(f"Pre-AI:  Var(log w) = {var_pre:.4f},  Gini = {gini_pre:.4f}")
print(f"Post-AI: Var(log w) = {var_post:.4f}, Gini = {gini_post:.4f}")
print(f"ΔVar(log w) = {var_post - var_pre:+.4f}  ({100*(var_post-var_pre)/var_pre:+.1f}%)")
print(f"p90-p50: {p90p50_pre:.4f} → {p90p50_post:.4f}")
print(f"p90-p10: {p90p10_pre:.4f} → {p90p10_post:.4f}")
print(f"Δlog Y = {dlogY:+.4f}")

ss_out = {
    "var_log_w_pre": var_pre, "var_log_w_post": var_post,
    "gini_pre": gini_pre, "gini_post": gini_post,
    "p90_p50_pre": p90p50_pre, "p90_p50_post": p90p50_post,
    "p90_p10_pre": p90p10_pre, "p90_p10_post": p90p10_post,
    "dlogY": dlogY,
    "wages_pre": wages_pre.tolist(), "wages_post": wages_post.tolist(),
    "mu_pre": mu_pre.tolist(), "mu_post": mu_post.tolist(),
}
with open(os.path.join(OUTDIR, "counterfactual_ss_optB.json"), "w") as f:
    json.dump(ss_out, f, indent=2)
print(f"Saved SS results to {OUTDIR}/counterfactual_ss_optB.json")

# =====================================================================
# 5. Transition path
# =====================================================================
print("\n" + "=" * 70)
print("TRANSITION PATH (Option B, T=100)")
print("=" * 70)

T_trans = 100

from ai_transition_path import solve_transition_vec

p_tr = SdmParams(data, tier3)
p_tr.a_o = a_o.copy()

trans = solve_transition_vec(
    p_tr, kspec_B, d_post, a_o, B_pre,
    wages_pre, wages_post, mu_pre, V_post,
    T_trans=T_trans, max_outer=60, tol=5e-4,
    damping=0.20, verbose=True,
)

wage_path   = trans["wage_path"]
mu_path     = trans["mu_path"]
var_path    = trans["var_log_w_path"]
gini_path   = trans["gini_path"]

# Half-lives
def _half_life(path, v0, v1):
    target = v0 + 0.5 * (v1 - v0)
    if v1 < v0:
        hit = np.where(path <= target)[0]
    else:
        hit = np.where(path >= target)[0]
    return int(hit[0]) + 1 if len(hit) else -1

t_half_var  = _half_life(var_path, var_pre, var_post)
t_half_gini = _half_life(gini_path, gini_pre, gini_post)

print(f"\nVar(log w) half-life: {t_half_var} years")
print(f"Gini half-life:       {t_half_gini} years")

# Save transition
trans_out = {
    "T_trans": T_trans,
    "var_log_w_path": var_path.tolist(),
    "gini_path": gini_path.tolist(),
    "var_log_w_pre": var_pre,
    "var_log_w_post": var_post,
    "wages_pre": wages_pre.tolist(),
    "wages_post": wages_post.tolist(),
    "mu_pre": mu_pre.tolist(),
    "mu_post": mu_post.tolist(),
    "t_half_var": t_half_var,
    "t_half_gini": t_half_gini,
    "p90_p50_pre": p90p50_pre, "p90_p50_post": p90p50_post,
    "p90_p10_pre": p90p10_pre, "p90_p10_post": p90p10_post,
}
with open(os.path.join(OUTDIR, "transition_optB.json"), "w") as f:
    json.dump(trans_out, f, indent=2)
print(f"Saved transition to {OUTDIR}/transition_optB.json")

# =====================================================================
# 6. Figures — comparison with Option A
# =====================================================================
print("\n" + "=" * 70)
print("GENERATING COMPARISON FIGURES")
print("=" * 70)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "text.usetex":        False,
    "mathtext.fontset":   "cm",
    "mathtext.rm":        "serif",
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Bitstream Vera Serif",
                           "Palatino", "Times"],
    "axes.labelsize":     13,
    "axes.titlesize":     14,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.9,
    "legend.fontsize":    10,
    "pdf.fonttype":       42,
})

# --- Load Option A transition for comparison ---
RESULTS_A = os.path.join(THIS_DIR, "sdm_output", "ai_transition_results.json")
with open(RESULTS_A) as f:
    rA = json.load(f)
tw_A = rA["by_spec"]["twoway"]
var_path_A = np.array(tw_A["var_log_w_path"])
summary_A = next(s for s in rA["summary"] if s["spec"] == "twoway")

# ---------- Figure 1: Inequality dynamics comparison ----------
fig, ax = plt.subplots(figsize=(9, 5.4))
t_axis = np.arange(1, T_trans + 1)

ax.plot(t_axis, var_path_A, color="#1f3b73", lw=2.4, label="Option A (employment-weighted)")
ax.plot(t_axis, var_path, color="#b03030", lw=2.4, ls="--",
        label="Option B (switcher-mass-weighted)")

ax.axhline(summary_A["var_log_w_pre"], color="#555", ls=":", lw=0.8, alpha=0.5)
ax.axhline(summary_A["var_log_w_post"], color="#1f3b73", ls=":", lw=0.8, alpha=0.5)
ax.axhline(var_post, color="#b03030", ls=":", lw=0.8, alpha=0.5)

# Annotation
pctA = 100 * (summary_A["var_log_w_post"] - summary_A["var_log_w_pre"]) / summary_A["var_log_w_pre"]
pctB = 100 * (var_post - var_pre) / var_pre
ax.text(0.985, 0.15,
        f"Option A: $\\Delta$Var = {summary_A['var_log_w_post']-summary_A['var_log_w_pre']:+.4f} ({pctA:+.1f}%)\n"
        f"Option B: $\\Delta$Var = {var_post-var_pre:+.4f} ({pctB:+.1f}%)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="#f6f6f0", ec="#bbb"))

ax.set_xlabel("Years after AI shock")
ax.set_ylabel(r"Variance of log occupational wages")
ax.set_title("Inequality Transition: Option A vs Option B Moment Definition",
             fontweight="bold")
ax.legend(loc="upper right", framealpha=0.95)
ax.set_xlim(0, 30)
# y range from visible data
ymin = min(var_path[:30].min(), var_path_A[:30].min())
ymax = max(var_path[:30].max(), var_path_A[:30].max(), var_pre)
pad = 0.03 * (ymax - ymin)
ax.set_ylim(ymin - pad, ymax + pad)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "inequality_comparison_AB.pdf"), dpi=200)
fig.savefig(os.path.join(OUTDIR, "inequality_comparison_AB.png"), dpi=180)
plt.close(fig)
print("Saved inequality comparison figure")

# ---------- Figure 2: Top 30 inflows/outflows (Option B) ----------
emp_total = float(data_emp.sum())
mu_pre_B  = np.array(mu_pre)
mu_post_B = np.array(mu_post)
w_pre_B   = np.array(wages_pre)
w_post_B  = np.array(wages_post)
L_pre_B   = mu_pre_B * emp_total
L_post_B  = mu_post_B * emp_total
dL        = L_post_B - L_pre_B
dlogw     = np.log(w_post_B) - np.log(w_pre_B)

TOP_N = 30
idx_in  = np.argsort(-dL)[:TOP_N]
idx_out = np.argsort( dL)[:TOP_N]

names = data["soc3_names"]

def _nice_name(soc3, name, maxlen=34):
    n = name.replace(" Occupations", "").replace(" Workers", "")
    if len(n) > maxlen:
        n = n[:maxlen - 1].rstrip(",;: ") + "..."
    return f"{soc3}  {n}"

fig, axes = plt.subplots(1, 2, figsize=(17, 11), gridspec_kw={"wspace": 0.38})

# Panel A — gainers
ax = axes[0]
y = np.arange(TOP_N)[::-1]
vals_k = dL[idx_in] / 1e3
ax.barh(y, vals_k, color="#2d8a4e", edgecolor="black", linewidth=0.35, height=0.75)
labels = [_nice_name(soc3_codes[i], names[i]) for i in idx_in]
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlabel(r"$\Delta L$  (thousand workers)")
ax.set_title("Top 30 — Net Inflows (Option B)", pad=14, fontweight="bold")
ax.grid(True, axis="x", alpha=0.32, linestyle=":")
ax.axvline(0, color="black", lw=0.6)
ax.set_xlim(0, max(vals_k) * 1.22)
for i, v in enumerate(vals_k):
    pct = 100 * dlogw[idx_in[i]]
    ax.text(v, y[i], fr"  ${pct:+.1f}\%$", va="center", fontsize=11, color="#222")

# Panel B — losers
ax = axes[1]
vals_k = dL[idx_out] / 1e3
ax.barh(y, vals_k, color="#b03030", edgecolor="black", linewidth=0.35, height=0.75)
labels = [_nice_name(soc3_codes[i], names[i]) for i in idx_out]
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlabel(r"$\Delta L$  (thousand workers)")
ax.set_title("Top 30 — Net Outflows (Option B)", pad=14, fontweight="bold")
ax.grid(True, axis="x", alpha=0.32, linestyle=":")
ax.axvline(0, color="black", lw=0.6)
ax.set_xlim(min(vals_k) * 1.22, 0)
for i, v in enumerate(vals_k):
    pct = 100 * dlogw[idx_out[i]]
    ax.text(v, y[i], fr"${pct:+.1f}\%$  ", va="center", ha="right", fontsize=11, color="#222")

plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "top30_inflows_outflows_optB.pdf"), dpi=220, bbox_inches="tight")
fig.savefig(os.path.join(OUTDIR, "top30_inflows_outflows_optB.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved top-30 figure (Option B)")

# ---------- Figure 3: κ comparison scatter ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

ax = axes[0]
ax.scatter(kappa_out_A, kspec_B.kappa_out, s=20, alpha=0.7, c="#1f3b73")
mn, mx = min(kappa_out_A.min(), kspec_B.kappa_out.min()), max(kappa_out_A.max(), kspec_B.kappa_out.max())
ax.plot([mn, mx], [mn, mx], "k--", lw=0.8, alpha=0.5)
ax.set_xlabel(r"$\kappa_{\mathrm{out}}$ Option A")
ax.set_ylabel(r"$\kappa_{\mathrm{out}}$ Option B")
ax.set_title(f"Origin barriers  (corr = {corr_ko:.3f})", fontweight="bold")
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(kappa_in_A, kspec_B.kappa_in, s=20, alpha=0.7, c="#b03030")
mn, mx = min(kappa_in_A.min(), kspec_B.kappa_in.min()), max(kappa_in_A.max(), kspec_B.kappa_in.max())
ax.plot([mn, mx], [mn, mx], "k--", lw=0.8, alpha=0.5)
ax.set_xlabel(r"$\kappa_{\mathrm{in}}$ Option A")
ax.set_ylabel(r"$\kappa_{\mathrm{in}}$ Option B")
ax.set_title(f"Destination barriers  (corr = {corr_ki:.3f})", fontweight="bold")
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "kappa_scatter_AB.pdf"), dpi=200)
fig.savefig(os.path.join(OUTDIR, "kappa_scatter_AB.png"), dpi=180)
plt.close(fig)
print("Saved κ scatter comparison")

# ---------- Figure 4: Top-30 side-by-side with Option A ----------
# Load Option A employment data
mu_pre_A  = np.array(tw_A["mu_pre"])
mu_post_A = np.array(tw_A["mu_post"])
dL_A      = (mu_post_A - mu_pre_A) * emp_total
idx_in_A  = np.argsort(-dL_A)[:TOP_N]
idx_out_A = np.argsort( dL_A)[:TOP_N]

# Check overlap in top-30 lists
set_in_A  = set(idx_in_A)
set_in_B  = set(idx_in)
set_out_A = set(idx_out_A)
set_out_B = set(idx_out)
overlap_in  = len(set_in_A & set_in_B)
overlap_out = len(set_out_A & set_out_B)

print(f"\n--- Top-30 Overlap ---")
print(f"  Inflow  list overlap: {overlap_in}/30")
print(f"  Outflow list overlap: {overlap_out}/30")

# =====================================================================
# 7. Summary comparison
# =====================================================================
print("\n" + "=" * 70)
print("SUMMARY COMPARISON: OPTION A vs OPTION B")
print("=" * 70)
print(f"{'':30s}  {'Option A':>12s}  {'Option B':>12s}  {'Diff':>10s}")
print("-" * 70)

rows = [
    ("Var(log w) pre",     summary_A["var_log_w_pre"], var_pre),
    ("Var(log w) post",    summary_A["var_log_w_post"], var_post),
    ("ΔVar(log w)",        summary_A["var_log_w_post"]-summary_A["var_log_w_pre"],
                           var_post - var_pre),
    ("ΔVar(log w) %",     100*(summary_A["var_log_w_post"]-summary_A["var_log_w_pre"])/summary_A["var_log_w_pre"],
                           100*(var_post-var_pre)/var_pre),
    ("p90-p50 pre",        tw_A.get("p90_p50_pre", float('nan')), p90p50_pre),
    ("p90-p50 post",       tw_A.get("p90_p50_post", float('nan')), p90p50_post),
    ("p90-p10 pre",        tw_A.get("p90_p10_pre", float('nan')), p90p10_pre),
    ("p90-p10 post",       tw_A.get("p90_p10_post", float('nan')), p90p10_post),
    ("Half-life Var (yrs)", float(summary_A.get("T_half_var", -1)), float(t_half_var)),
    ("Δlog Y",             float(summary_A.get("dlogY_final", float('nan'))), dlogY),
    ("Top-30 inflow overlap",  30, overlap_in),
    ("Top-30 outflow overlap", 30, overlap_out),
    ("κ_out corr(A,B)",   1.0, corr_ko),
    ("κ_in  corr(A,B)",   1.0, corr_ki),
]

for label, vA, vB in rows:
    if isinstance(vA, int) and isinstance(vB, int):
        print(f"  {label:30s}  {vA:12d}  {vB:12d}  {vB-vA:+10d}")
    else:
        diff = vB - vA if not (np.isnan(vA) or np.isnan(vB)) else float('nan')
        print(f"  {label:30s}  {vA:12.4f}  {vB:12.4f}  {diff:+10.4f}")

# Save summary
summary_out = {
    "option_A": {
        "var_log_w_pre": summary_A["var_log_w_pre"],
        "var_log_w_post": summary_A["var_log_w_post"],
        "t_half_var": summary_A.get("T_half_var"),
    },
    "option_B": {
        "var_log_w_pre": var_pre,
        "var_log_w_post": var_post,
        "p90_p50_pre": p90p50_pre, "p90_p50_post": p90p50_post,
        "p90_p10_pre": p90p10_pre, "p90_p10_post": p90p10_post,
        "t_half_var": t_half_var,
        "dlogY": dlogY,
    },
    "comparison": {
        "corr_kappa_out": corr_ko,
        "corr_kappa_in":  corr_ki,
        "top30_inflow_overlap": overlap_in,
        "top30_outflow_overlap": overlap_out,
    },
}
with open(os.path.join(OUTDIR, "summary_comparison_AB.json"), "w") as f:
    json.dump(summary_out, f, indent=2)

print(f"\nAll Option B outputs saved to: {OUTDIR}/")
print("Done.")
