"""
Generate a publication-quality top-30 inflow/outflow figure:
 - LaTeX font (serif, mathtext), larger axis/tick labels
 - Clean bar design, thinner frame, gridlines only on x-axis
 - Two panels: gainers (green) on the left, losers (red) on the right
 - Each bar labelled with SOC3 + truncated occupation name
 - Wage-change annotation outside each bar
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ----- aesthetic settings: mathtext + DejaVu Serif for portable look -----
rcParams.update({
    "text.usetex":        False,
    "mathtext.fontset":   "cm",          # Computer Modern math
    "mathtext.rm":        "serif",
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Bitstream Vera Serif",
                           "Palatino", "Times"],
    "axes.labelsize":     17,
    "axes.titlesize":     19,
    "xtick.labelsize":    14,
    "ytick.labelsize":    13,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.9,
    "legend.fontsize":    13,
    "pdf.fonttype":       42,            # embed TrueType for PDF
})

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS  = os.path.join(THIS_DIR, "sdm_output", "ai_transition_results.json")
OUT_PDF  = os.path.join(THIS_DIR, "sdm_output", "top30_inflows_outflows.pdf")

# ----- data -----
with open(RESULTS) as f:
    r = json.load(f)

tw = r["by_spec"]["twoway"]
codes = r["soc3_codes"]
# The JSON stores mu_pre/post and employment shares; use wages and mu to
# compute ΔL in absolute workers.
import paths_override  # noqa: F401
from simple_dynamic_model import build_model_data   # noqa
data = build_model_data(verbose=False)
names = data["soc3_names"]
emp_total = float(data["employment"].sum())

mu_pre   = np.array(tw["mu_pre"])
mu_post  = np.array(tw["mu_post"])
w_pre    = np.array(tw["wages_pre"])
w_post   = np.array(tw["wages_post"])
L_pre    = mu_pre  * emp_total
L_post   = mu_post * emp_total
dL       = L_post - L_pre
dL_pct   = dL / np.maximum(L_pre, 1e-6)
dlogw    = np.log(w_post) - np.log(w_pre)

TOP_N = 30
idx_in  = np.argsort(-dL)[:TOP_N]
idx_out = np.argsort( dL)[:TOP_N]


def _nice_name(soc3, name, maxlen=34):
    # Shorten long names and drop trailing " Occupations"
    n = name.replace(" Occupations", "").replace(" Workers", "")
    if len(n) > maxlen:
        n = n[:maxlen - 1].rstrip(",;: ") + "..."
    return f"{soc3}  {n}"


fig, axes = plt.subplots(1, 2, figsize=(17, 11),
                          gridspec_kw={"wspace": 0.38})

# --- Panel A — gainers ----------------------------------------------------
ax = axes[0]
y  = np.arange(TOP_N)[::-1]
vals_k = dL[idx_in] / 1e3           # thousand workers
bars = ax.barh(y, vals_k, color="#2d8a4e",
               edgecolor="black", linewidth=0.35, height=0.75)
labels = [_nice_name(codes[i], names[i]) for i in idx_in]
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlabel(r"$\Delta L$  (thousand workers)")
ax.set_title("Top 30 Occupations — Net Inflows", pad=14, fontweight="bold")
ax.grid(True, axis="x", alpha=0.32, linestyle=":")
ax.axvline(0, color="black", lw=0.6)
# Extend x-axis a bit so annotations fit
ax.set_xlim(0, max(vals_k) * 1.22)
for i, v in enumerate(vals_k):
    pct = 100 * dlogw[idx_in[i]]
    ax.text(v, y[i], fr"  ${pct:+.1f}\%$",
            va="center", fontsize=11, color="#222")

# --- Panel B — losers -----------------------------------------------------
ax = axes[1]
vals_k = dL[idx_out] / 1e3
bars = ax.barh(y, vals_k, color="#b03030",
               edgecolor="black", linewidth=0.35, height=0.75)
labels = [_nice_name(codes[i], names[i]) for i in idx_out]
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlabel(r"$\Delta L$  (thousand workers)")
ax.set_title("Top 30 Occupations — Net Outflows", pad=14, fontweight="bold")
ax.grid(True, axis="x", alpha=0.32, linestyle=":")
ax.axvline(0, color="black", lw=0.6)
ax.set_xlim(min(vals_k) * 1.22, 0)
for i, v in enumerate(vals_k):
    pct = 100 * dlogw[idx_out[i]]
    ax.text(v, y[i], fr"${pct:+.1f}\%$  ",
            va="center", ha="right", fontsize=11, color="#222")

plt.tight_layout()
plt.savefig(OUT_PDF, dpi=220, bbox_inches="tight")
png = OUT_PDF.replace(".pdf", ".png")
plt.savefig(png, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {OUT_PDF}")
print(f"Wrote {png}")
