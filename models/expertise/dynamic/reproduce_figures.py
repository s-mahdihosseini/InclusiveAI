#!/usr/bin/env python3
"""
reproduce_figures.py
====================
Self-contained script that regenerates the two publication figures for
Section 5 ("Dynamic Extension") of Hosseini & Lichtinger (2026).

  Figure 1 — Inequality IRF:
      Transition path of Var(log w) over 30 years after the AI shock,
      with 50% / 80% adjustment markers and pre/post steady-state lines.

  Figure 2 — Top-30 inflows & outflows:
      Two-panel bar chart of the 30 occupations gaining the most workers
      (left, green) and 30 losing the most (right, red), annotated with
      the corresponding wage change (Δlog w in %).

Inputs  (read-only, nothing is overwritten)
------
  sdm_output/ai_transition_results.json
      Produced by ai_transition_path.py.  Contains the T=100 transition
      path (wages, employment shares, Var(log w), Gini at each period)
      and pre/post steady-state values, for all three κ specs.

  build_model_data()  (via simple_dynamic_model.py + paths_override.py)
      Only used for occupation names and total employment (to convert
      employment shares to worker counts).

Outputs
-------
  sdm_output/sdm_inequality_evolution.pdf   +  .png
  sdm_output/top30_inflows_outflows.pdf     +  .png

Usage
-----
    cd "Dynamic model/2. Dynamic Model"
    python reproduce_figures.py
"""

import os
import sys
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── Paths ────────────────────────────────────────────────────────────
THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(THIS_DIR, "sdm_output", "ai_transition_results.json")
OUT_DIR      = os.path.join(THIS_DIR, "sdm_output")

sys.path.insert(0, THIS_DIR)
import paths_override  # noqa: F401  — redirects data paths
from simple_dynamic_model import build_model_data

# ── Global matplotlib style (LaTeX-like serif + CM math) ─────────────
rcParams.update({
    "text.usetex":        False,
    "mathtext.fontset":   "cm",
    "mathtext.rm":        "serif",
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Bitstream Vera Serif",
                           "Palatino", "Times"],
    "axes.labelsize":     14,
    "axes.titlesize":     15,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.9,
    "legend.fontsize":    11,
    "pdf.fonttype":       42,       # embed TrueType in PDF
})


# =====================================================================
# Load data
# =====================================================================
print("Loading model data …")
data = build_model_data(verbose=False)
soc3_codes = data["soc3_codes"]
soc3_names = data["soc3_names"]
emp_total  = float(data["employment"].sum())

print(f"Loading transition results from {RESULTS_JSON} …")
with open(RESULTS_JSON) as f:
    results = json.load(f)

tw      = results["by_spec"]["twoway"]
summary = next(s for s in results["summary"] if s["spec"] == "twoway")


# =====================================================================
# Figure 1 — Inequality IRF (Var(log w) transition path)
# =====================================================================
def plot_inequality_irf():
    var_path = np.array(tw["var_log_w_path"])
    T        = len(var_path)
    var_pre  = summary["var_log_w_pre"]
    var_post = summary["var_log_w_post"]

    # Half-life (50% adjustment)
    target50 = var_pre + 0.50 * (var_post - var_pre)
    hit50    = np.where(var_path <= target50)[0]
    t_half   = int(hit50[0]) + 1 if len(hit50) else None

    # 80% adjustment
    target80 = var_pre + 0.80 * (var_post - var_pre)
    hit80    = np.where(var_path <= target80)[0]
    t_80     = int(hit80[0]) + 1 if len(hit80) else None

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(9, 5.4))
    t_axis = np.arange(1, T + 1)

    ax.plot(t_axis, var_path, color="#1f3b73", lw=2.4,
            label=r"Var$(\log w_o)$ along transition")
    ax.axhline(var_pre,  color="#555",    ls="--", lw=1.1,
               label=f"Pre-AI steady state = {var_pre:.3f}")
    ax.axhline(var_post, color="#a62a2a", ls="--", lw=1.1,
               label=f"Post-AI steady state = {var_post:.3f}")

    # 50% marker
    if t_half is not None:
        ax.axvline(t_half, color="#2a7f3f", ls=":", lw=1.2)
        ax.annotate(f"50% adjustment\nby year {t_half}",
                    xy=(t_half, target50),
                    xytext=(t_half + 6, target50 + 0.004),
                    fontsize=10, color="#2a7f3f",
                    arrowprops=dict(arrowstyle="->", color="#2a7f3f", lw=1))

    # 80% marker
    if t_80 is not None:
        ax.axvline(t_80, color="#8a6d00", ls=":", lw=1.0)
        ax.annotate(f"80% by year {t_80}",
                    xy=(t_80, target80),
                    xytext=(t_80 + 6, target80 + 0.002),
                    fontsize=9.5, color="#8a6d00",
                    arrowprops=dict(arrowstyle="->", color="#8a6d00", lw=0.9))

    # Long-run change box
    pct = 100 * (var_post - var_pre) / var_pre
    ax.text(0.985, 0.15,
            f"Long-run change:   "
            fr"$\Delta\mathrm{{Var}}(\log w)$ = ${var_post - var_pre:+.3f}$   "
            f"({pct:+.1f}%)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f6f6f0", ec="#bbb"))

    ax.set_xlabel("Years after AI shock")
    ax.set_ylabel(r"Variance of log occupational wages")
    ax.set_title("Transition Path of Between-Occupation Wage Inequality",
                 fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # x-axis: 30 years (all action happens by ~20)
    ax.set_xlim(0, 30)
    var_30 = var_path[:30]
    pad = 0.02 * (var_pre - var_30[-1])
    ax.set_ylim(var_30[-1] - pad, var_pre + pad)

    plt.tight_layout()
    pdf = os.path.join(OUT_DIR, "sdm_inequality_evolution.pdf")
    png = os.path.join(OUT_DIR, "sdm_inequality_evolution.png")
    fig.savefig(pdf, dpi=200)
    fig.savefig(png, dpi=180)
    plt.close(fig)

    print(f"  → {pdf}")
    print(f"  → {png}")
    print(f"    half-life = {t_half} yrs,  80% by year {t_80}")
    print(f"    Var(log w): {var_pre:.4f} → {var_post:.4f}  ({pct:+.1f}%)")


# =====================================================================
# Figure 2 — Top-30 inflows / outflows bar chart
# =====================================================================
def plot_top30():
    mu_pre  = np.array(tw["mu_pre"])
    mu_post = np.array(tw["mu_post"])
    w_pre   = np.array(tw["wages_pre"])
    w_post  = np.array(tw["wages_post"])

    L_pre  = mu_pre  * emp_total
    L_post = mu_post * emp_total
    dL     = L_post - L_pre
    dlogw  = np.log(w_post) - np.log(w_pre)

    TOP_N   = 30
    idx_in  = np.argsort(-dL)[:TOP_N]
    idx_out = np.argsort( dL)[:TOP_N]

    def _nice_name(soc3, name, maxlen=34):
        n = name.replace(" Occupations", "").replace(" Workers", "")
        if len(n) > maxlen:
            n = n[:maxlen - 1].rstrip(",;: ") + "…"
        return f"{soc3}  {n}"

    fig, axes = plt.subplots(1, 2, figsize=(17, 11),
                              gridspec_kw={"wspace": 0.38})

    # ── Panel A: gainers ──
    ax = axes[0]
    y  = np.arange(TOP_N)[::-1]
    vals_k = dL[idx_in] / 1e3
    ax.barh(y, vals_k, color="#2d8a4e",
            edgecolor="black", linewidth=0.35, height=0.75)
    labels = [_nice_name(soc3_codes[i], soc3_names[i]) for i in idx_in]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel(r"$\Delta L$  (thousand workers)")
    ax.set_title("Top 30 Occupations — Net Inflows",
                 pad=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.32, linestyle=":")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlim(0, max(vals_k) * 1.22)
    for i, v in enumerate(vals_k):
        pct = 100 * dlogw[idx_in[i]]
        ax.text(v, y[i], fr"  ${pct:+.1f}\%$",
                va="center", fontsize=11, color="#222")

    # ── Panel B: losers ──
    ax = axes[1]
    vals_k = dL[idx_out] / 1e3
    ax.barh(y, vals_k, color="#b03030",
            edgecolor="black", linewidth=0.35, height=0.75)
    labels = [_nice_name(soc3_codes[i], soc3_names[i]) for i in idx_out]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel(r"$\Delta L$  (thousand workers)")
    ax.set_title("Top 30 Occupations — Net Outflows",
                 pad=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.32, linestyle=":")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlim(min(vals_k) * 1.22, 0)
    for i, v in enumerate(vals_k):
        pct = 100 * dlogw[idx_out[i]]
        ax.text(v, y[i], fr"${pct:+.1f}\%$  ",
                va="center", ha="right", fontsize=11, color="#222")

    plt.tight_layout()
    pdf = os.path.join(OUT_DIR, "top30_inflows_outflows.pdf")
    png = os.path.join(OUT_DIR, "top30_inflows_outflows.png")
    fig.savefig(pdf, dpi=220, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  → {pdf}")
    print(f"  → {png}")
    print(f"    Top gainer:  {soc3_codes[idx_in[0]]}  "
          f"({soc3_names[idx_in[0]]}):  ΔL = {dL[idx_in[0]]:+,.0f}")
    print(f"    Top loser:   {soc3_codes[idx_out[0]]}  "
          f"({soc3_names[idx_out[0]]}):  ΔL = {dL[idx_out[0]]:+,.0f}")


# =====================================================================
# Run
# =====================================================================
if __name__ == "__main__":
    print("\n─── Figure 1: Inequality IRF ───")
    plot_inequality_irf()

    print("\n─── Figure 2: Top-30 Inflows / Outflows ───")
    plot_top30()

    print("\nDone.  Upload the PDFs to Overleaf:")
    print("    sdm_output/sdm_inequality_evolution.pdf")
    print("    sdm_output/top30_inflows_outflows.pdf")
