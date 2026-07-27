"""
Generate a clean publication-ready figure of the Var(log w) transition path
for the paper (sdm_inequality_evolution.pdf).

Uses the T=100 two-way κ transition results from
sdm_output/ai_transition_results.json.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ----- LaTeX-style fonts: Computer Modern math + serif family -----
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

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(THIS_DIR, "sdm_output", "ai_transition_results.json")
OUT_PDF = os.path.join(THIS_DIR, "sdm_output", "sdm_inequality_evolution.pdf")

with open(RESULTS_JSON) as f:
    r = json.load(f)

tw = r["by_spec"]["twoway"]
var_path = np.array(tw["var_log_w_path"])
T = len(var_path)

# Pre-AI baseline = value at t = 0 of the transition path, which was
# initialized at the pre-AI steady state.  Summary also reports it.
summary = next(s for s in r["summary"] if s["spec"] == "twoway")
var_pre  = summary["var_log_w_pre"]
var_post = summary["var_log_w_post"]

# Compute half-life
target = var_pre + 0.5 * (var_post - var_pre)
# var is decreasing, find first index where path <= target
hit = np.where(var_path <= target)[0]
t_half = int(hit[0]) + 1 if len(hit) else None

# 80% adjustment
target80 = var_pre + 0.80 * (var_post - var_pre)
hit80 = np.where(var_path <= target80)[0]
t_80 = int(hit80[0]) + 1 if len(hit80) else None

# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.4))

t_axis = np.arange(1, T + 1)
ax.plot(t_axis, var_path, color="#1f3b73", lw=2.4, label=r"Var$(\log w_o)$ along transition")

ax.axhline(var_pre,  color="#555", ls="--", lw=1.1,
           label=f"Pre-AI steady state = {var_pre:.3f}")
ax.axhline(var_post, color="#a62a2a", ls="--", lw=1.1,
           label=f"Post-AI steady state = {var_post:.3f}")

# Half-life shaded marker
if t_half is not None:
    ax.axvline(t_half, color="#2a7f3f", ls=":", lw=1.2)
    ax.annotate(f"50% adjustment\nby year {t_half}",
                xy=(t_half, target),
                xytext=(t_half + 6, target + 0.004),
                fontsize=10, color="#2a7f3f",
                arrowprops=dict(arrowstyle="->", color="#2a7f3f", lw=1))

if t_80 is not None:
    ax.axvline(t_80, color="#8a6d00", ls=":", lw=1.0)
    ax.annotate(f"80% by year {t_80}",
                xy=(t_80, target80),
                xytext=(t_80 + 6, target80 + 0.002),
                fontsize=9.5, color="#8a6d00",
                arrowprops=dict(arrowstyle="->", color="#8a6d00", lw=0.9))

# Total change annotation
pct = 100 * (var_post - var_pre) / var_pre
ax.text(0.985, 0.15,
        f"Long-run change:   "
        fr"$\Delta\mathrm{{Var}}(\log w)$ = ${var_post-var_pre:+.3f}$   "
        f"({pct:+.1f}%)",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="#f6f6f0", ec="#bbb"))

ax.set_xlabel("Years after AI shock", fontsize=11)
ax.set_ylabel(r"Variance of log occupational wages", fontsize=11)
ax.set_title("Transition Path of Between-Occupation Wage Inequality",
             fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
ax.set_xlim(0, 30)
# Tight y-range — compute from visible data only
var_path_30 = var_path[:30]
pad = 0.02 * (var_pre - var_path_30[-1])
ax.set_ylim(var_path_30[-1] - pad, var_pre + pad)

plt.tight_layout()
plt.savefig(OUT_PDF, dpi=200)
# Also save a PNG copy for preview
png_out = OUT_PDF.replace(".pdf", ".png")
plt.savefig(png_out, dpi=180)
plt.close(fig)

print(f"Wrote {OUT_PDF}")
print(f"Wrote {png_out}")
print(f"T = {T} periods, half-life = {t_half} yrs, 80% adjustment = {t_80} yrs")
print(f"Var(log w): {var_pre:.4f} -> {var_post:.4f}  ({pct:+.1f}%)")
