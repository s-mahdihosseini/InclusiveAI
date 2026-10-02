"""
Analyze the AI transition-path results and produce the paper deliverables:

    (1) Top 50 occupations by employment INFLOW (largest positive ΔL)
    (2) Top 50 occupations by employment OUTFLOW (largest negative ΔL)
    (3) Top occupation-to-occupation flow pairs
        (3a) biggest switcher flows in the post-AI steady state
        (3b) biggest CHANGES in switcher flows (post − pre)
    (4) Heatmaps of the J×J occ-to-occ flow matrix pre and post AI,
        with illustrative labels for the most salient cells.

Focuses on the TWO-WAY κ specification (the one that matches observed
flow data).  The dest-only and scalar specs are processed in parallel
for comparison; their outputs go into sub-directories.

Outputs land in:  sdm_output/transition_analysis/<spec>/
    top50_inflows.csv
    top50_outflows.csv
    top_pair_flows_post.csv
    top_pair_flow_changes.csv
    top50_inflows_outflows.png
    heatmap_flows_pre.png
    heatmap_flows_post.png
    heatmap_flows_change.png
    summary.md
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import paths_override  # noqa: F401
from simple_dynamic_model import build_model_data


THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(THIS_DIR, "sdm_output")
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "transition_analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

TRANS_JSON  = os.path.join(OUTPUT_DIR, "ai_transition_results.json")


# ---------------------------------------------------------------------------
# Flow-matrix utilities
# ---------------------------------------------------------------------------

def switcher_flow_matrix(mu, policy, total_emp):
    """
    Returns an (J, J) matrix F where F[s, o] = number of workers moving
    from s to o per year (off-diagonal only; diagonal set to 0).
    """
    F = mu[:, None] * policy * total_emp
    np.fill_diagonal(F, 0.0)
    return F


def topk_indices_by(x, k, largest=True):
    """Top k indices of x (argmax if largest, argmin if not)."""
    if largest:
        return np.argsort(-x)[:k]
    return np.argsort(x)[:k]


# ---------------------------------------------------------------------------
# Major-group helpers for nicer axis labels
# ---------------------------------------------------------------------------

MAJOR_GROUP_NAMES = {
    "11": "Management",
    "13": "Business/Financial",
    "15": "Computer/Math",
    "17": "Engineering",
    "19": "Sciences",
    "21": "Community/Social",
    "23": "Legal",
    "25": "Education",
    "27": "Arts/Media",
    "29": "Healthcare Prac.",
    "31": "Healthcare Supp.",
    "33": "Protective",
    "35": "Food Prep",
    "37": "Building Maint.",
    "39": "Personal Care",
    "41": "Sales",
    "43": "Office/Admin",
    "45": "Farming",
    "47": "Construction",
    "49": "Repair",
    "51": "Production",
    "53": "Transportation",
}


def group_label_blocks(codes):
    """Return list of (major_code, start_idx, end_idx, label) for a code list
    that is sorted by SOC3 (major-minor)."""
    majors = [c.split("-")[0] for c in codes]
    blocks = []
    cur = majors[0]; start = 0
    for i, m in enumerate(majors[1:], start=1):
        if m != cur:
            blocks.append((cur, start, i - 1,
                           MAJOR_GROUP_NAMES.get(cur, cur)))
            cur = m; start = i
    blocks.append((cur, start, len(majors) - 1,
                   MAJOR_GROUP_NAMES.get(cur, cur)))
    return blocks


# ---------------------------------------------------------------------------
# Analysis for one spec
# ---------------------------------------------------------------------------

def analyze_spec(tag, spec_data, codes, names, employment, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    J = len(codes)
    total_emp = float(employment.sum())
    wages_pre  = np.array(spec_data["wages_pre"])
    wages_post = np.array(spec_data["wages_post"])
    mu_pre    = np.array(spec_data["mu_pre"])
    mu_post   = np.array(spec_data["mu_post"])
    pol_pre   = np.array(spec_data["policy_pre"])
    pol_post  = np.array(spec_data["policy_post"])

    # Absolute employment change (workers)
    # mu_* are shares; mapping back to workers via total_emp
    L_pre_workers  = mu_pre  * total_emp
    L_post_workers = mu_post * total_emp
    dL_workers = L_post_workers - L_pre_workers
    dL_pct     = dL_workers / np.maximum(L_pre_workers, 1e-6)

    # --------------------------------------------------------------
    # (1) top 50 inflows (gainers)
    # --------------------------------------------------------------
    idx_in = np.argsort(-dL_workers)[:50]
    df_in = pd.DataFrame({
        "rank":         np.arange(1, len(idx_in) + 1),
        "soc3":         [codes[i] for i in idx_in],
        "occupation":   [names[i] for i in idx_in],
        "wage_pre":     wages_pre[idx_in].round(0),
        "wage_post":    wages_post[idx_in].round(0),
        "dlog_wage":    (np.log(wages_post[idx_in])
                         - np.log(wages_pre[idx_in])).round(4),
        "L_pre":        L_pre_workers[idx_in].round(0),
        "L_post":       L_post_workers[idx_in].round(0),
        "dL_workers":   dL_workers[idx_in].round(0),
        "dL_pct":       (100 * dL_pct[idx_in]).round(2),
    })
    df_in.to_csv(os.path.join(out_dir, "top50_inflows.csv"), index=False)

    # --------------------------------------------------------------
    # (2) top 50 outflows (losers)
    # --------------------------------------------------------------
    idx_out = np.argsort(dL_workers)[:50]
    df_out = pd.DataFrame({
        "rank":         np.arange(1, len(idx_out) + 1),
        "soc3":         [codes[i] for i in idx_out],
        "occupation":   [names[i] for i in idx_out],
        "wage_pre":     wages_pre[idx_out].round(0),
        "wage_post":    wages_post[idx_out].round(0),
        "dlog_wage":    (np.log(wages_post[idx_out])
                         - np.log(wages_pre[idx_out])).round(4),
        "L_pre":        L_pre_workers[idx_out].round(0),
        "L_post":       L_post_workers[idx_out].round(0),
        "dL_workers":   dL_workers[idx_out].round(0),
        "dL_pct":       (100 * dL_pct[idx_out]).round(2),
    })
    df_out.to_csv(os.path.join(out_dir, "top50_outflows.csv"), index=False)

    # --------------------------------------------------------------
    # (3) Pair-flows pre / post / change
    # --------------------------------------------------------------
    F_pre  = switcher_flow_matrix(mu_pre,  pol_pre,  total_emp)
    F_post = switcher_flow_matrix(mu_post, pol_post, total_emp)
    F_diff = F_post - F_pre

    def _pair_df(F, k=50, largest=True, ref_F=None):
        flat = F.flatten()
        if largest:
            idx = np.argsort(-np.abs(flat) if ref_F is None else -flat)[:k]
        else:
            idx = np.argsort(flat)[:k]
        rows = []
        for rnk, ii in enumerate(idx):
            s, o = np.unravel_index(ii, F.shape)
            rows.append({
                "rank":      rnk + 1,
                "origin_soc3":   codes[s],
                "origin_name":   names[s],
                "dest_soc3":     codes[o],
                "dest_name":     names[o],
                "flow":          float(F[s, o]),
                "flow_pre":      float(F_pre[s, o])  if ref_F is None else float(F_pre[s, o]),
                "flow_post":     float(F_post[s, o]) if ref_F is None else float(F_post[s, o]),
                "delta":         float(F_post[s, o] - F_pre[s, o]),
            })
        return pd.DataFrame(rows)

    # 3a — largest post-AI switcher flows
    df_pair_post = _pair_df(F_post, k=50, largest=True)
    df_pair_post.to_csv(os.path.join(out_dir, "top_pair_flows_post.csv"),
                        index=False)

    # 3b — largest CHANGES (absolute magnitude)
    flat_diff = F_diff.flatten()
    idx_abs = np.argsort(-np.abs(flat_diff))[:50]
    rows_chg = []
    for rnk, ii in enumerate(idx_abs):
        s, o = np.unravel_index(ii, F_diff.shape)
        rows_chg.append({
            "rank":        rnk + 1,
            "origin_soc3": codes[s],
            "origin_name": names[s],
            "dest_soc3":   codes[o],
            "dest_name":   names[o],
            "flow_pre":    float(F_pre[s, o]),
            "flow_post":   float(F_post[s, o]),
            "delta":       float(F_diff[s, o]),
            "pct_change":  (100 * F_diff[s, o] / max(F_pre[s, o], 1e-6)),
        })
    df_pair_chg = pd.DataFrame(rows_chg)
    df_pair_chg.to_csv(os.path.join(out_dir, "top_pair_flow_changes.csv"),
                       index=False)

    # --------------------------------------------------------------
    # (4) PLOTS
    # --------------------------------------------------------------
    _plot_top_inflows_outflows(df_in, df_out, out_dir, tag)
    _plot_top_pair_flows(df_pair_post, df_pair_chg, out_dir, tag)
    _plot_heatmap(F_pre,  codes, "Pre-AI occ-to-occ flows (workers/yr)",
                  os.path.join(out_dir, "heatmap_flows_pre.png"),
                  log_scale=True, max_label_pairs=15, F_annot=F_pre,
                  names=names)
    _plot_heatmap(F_post, codes, "Post-AI occ-to-occ flows (workers/yr)",
                  os.path.join(out_dir, "heatmap_flows_post.png"),
                  log_scale=True, max_label_pairs=15, F_annot=F_post,
                  names=names)
    _plot_heatmap(F_diff, codes,
                  "Δ Flow (post − pre AI)   (workers/yr, diverging)",
                  os.path.join(out_dir, "heatmap_flows_change.png"),
                  log_scale=False, diverging=True,
                  max_label_pairs=15, F_annot=F_diff, names=names)

    # Summary markdown
    _write_summary(tag, df_in, df_out, df_pair_post, df_pair_chg,
                   F_pre, F_post, out_dir)

    return dict(
        top_inflows_csv  = os.path.join(out_dir, "top50_inflows.csv"),
        top_outflows_csv = os.path.join(out_dir, "top50_outflows.csv"),
        pair_post_csv    = os.path.join(out_dir, "top_pair_flows_post.csv"),
        pair_chg_csv     = os.path.join(out_dir, "top_pair_flow_changes.csv"),
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_top_inflows_outflows(df_in, df_out, out_dir, tag):
    fig, axes = plt.subplots(1, 2, figsize=(18, 14))

    # Top 50 inflows
    ax = axes[0]
    y = np.arange(len(df_in))[::-1]
    ax.barh(y, df_in["dL_workers"] / 1e3,
            color="#2a7f3f", edgecolor="black", linewidth=0.3)
    lbls = [f"{r.soc3}  {str(r.occupation)[:32]}"
            for _, r in df_in.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(lbls, fontsize=8)
    ax.set_xlabel("ΔL (thousand workers)")
    ax.set_title(f"[{tag}]  Top 50 occupations — biggest inflows",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.axvline(0, color="black", lw=0.8)
    # Annotate wage change
    for i, (_, r) in enumerate(df_in.iterrows()):
        dx = r["dL_workers"] / 1e3
        ax.text(dx, y[i], f"  {100*r['dlog_wage']:+.1f}% Δw",
                va="center", fontsize=6.5, color="#333")

    # Top 50 outflows
    ax = axes[1]
    y = np.arange(len(df_out))[::-1]
    ax.barh(y, df_out["dL_workers"] / 1e3,
            color="#a62a2a", edgecolor="black", linewidth=0.3)
    lbls = [f"{r.soc3}  {str(r.occupation)[:32]}"
            for _, r in df_out.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(lbls, fontsize=8)
    ax.set_xlabel("ΔL (thousand workers)")
    ax.set_title(f"[{tag}]  Top 50 occupations — biggest outflows",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.axvline(0, color="black", lw=0.8)
    for i, (_, r) in enumerate(df_out.iterrows()):
        dx = r["dL_workers"] / 1e3
        ax.text(dx, y[i], f"  {100*r['dlog_wage']:+.1f}% Δw",
                va="center", fontsize=6.5, color="#333", ha="right")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "top50_inflows_outflows.png"), dpi=150)
    plt.close(fig)


def _plot_top_pair_flows(df_pp, df_pc, out_dir, tag, top_n=25):
    """Two-panel: top-N post-AI pair flows (left) and top-N flow CHANGES
    (right).  Horizontal bars with origin → dest labels."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 12))

    # Panel A — top-N post-AI pair flows
    ax = axes[0]
    df_A = df_pp.head(top_n).copy()
    y = np.arange(len(df_A))[::-1]
    ax.barh(y, df_A["flow_post"] / 1e3, color="#1f77b4",
            edgecolor="black", linewidth=0.3, label="post-AI")
    ax.barh(y, df_A["flow_pre"] / 1e3, color="#1f77b4", alpha=0.35,
            edgecolor="black", linewidth=0.2, label="pre-AI")
    labels = [f"{r['origin_soc3']} {str(r['origin_name'])[:22]}  →  "
              f"{r['dest_soc3']} {str(r['dest_name'])[:22]}"
              for _, r in df_A.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Workers per year (thousand)")
    ax.set_title(f"[{tag}]  Top {top_n} post-AI switcher pair flows",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)

    # Panel B — top-N flow CHANGES
    ax = axes[1]
    df_B = df_pc.head(top_n).copy()
    y = np.arange(len(df_B))[::-1]
    colors = ["#2a7f3f" if d > 0 else "#a62a2a" for d in df_B["delta"]]
    ax.barh(y, df_B["delta"] / 1e3, color=colors,
            edgecolor="black", linewidth=0.3)
    labels = [f"{r['origin_soc3']} {str(r['origin_name'])[:22]}  →  "
              f"{r['dest_soc3']} {str(r['dest_name'])[:22]}"
              for _, r in df_B.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Δ flow (post − pre), thousand workers/yr")
    ax.set_title(f"[{tag}]  Top {top_n} pair flow CHANGES",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.axvline(0, color="black", lw=0.8)
    # Annotate percentage change
    for i, (_, r) in enumerate(df_B.iterrows()):
        dx = r["delta"] / 1e3
        pct = r["pct_change"]
        ax.text(dx, y[i], f"  {pct:+.0f}%",
                va="center", fontsize=7, color="#333",
                ha="left" if dx >= 0 else "right")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "top_pair_flows.png"), dpi=150)
    plt.close(fig)


def _plot_heatmap(F, codes, title, out_path,
                  log_scale=True, diverging=False,
                  max_label_pairs=15, F_annot=None, names=None):
    J = len(codes)
    fig, ax = plt.subplots(figsize=(15, 13))

    if diverging:
        # Symmetric power-norm so mid-range changes show.
        vmax = np.max(np.abs(F))
        # use PowerNorm with gamma<1 for both signs separately via asinh-style
        scale = vmax if vmax > 0 else 1.0
        F_norm = np.sign(F) * (np.abs(F) / scale) ** 0.5
        im = ax.imshow(F_norm, cmap="RdBu_r", vmin=-1, vmax=1,
                       aspect="equal")
        # Build a custom colorbar with actual-value ticks
        tick_vals = np.array([-vmax, -vmax/2, -vmax/10, 0,
                              vmax/10, vmax/2, vmax])
        tick_norm = np.sign(tick_vals) * (np.abs(tick_vals) / scale) ** 0.5
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                             ticks=tick_norm)
        cbar.set_ticklabels([f"{int(v):+,}" for v in tick_vals])
        cbar.set_label("Δ workers per year "
                       "(colour scale: sqrt-asymmetric)")
    else:
        if log_scale:
            F_plot = np.log10(F + 1.0)   # +1 avoids log(0)
            im = ax.imshow(F_plot, cmap="viridis", aspect="equal")
            cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cbar.set_label("log10(1 + flow)   [flow in workers/yr]")
        else:
            im = ax.imshow(F, cmap="viridis", aspect="equal")
            cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cbar.set_label("workers per year")

    # Major-group bracket ticks and lines
    blocks = group_label_blocks(codes)
    for maj, s, e, lbl in blocks:
        ax.axhline(s - 0.5, color="white", lw=0.4, alpha=0.6)
        ax.axvline(s - 0.5, color="white", lw=0.4, alpha=0.6)
    # Major-group text labels along top and left
    for maj, s, e, lbl in blocks:
        mid = 0.5 * (s + e)
        ax.text(-2.5, mid, f"{maj}  {lbl}", ha="right", va="center",
                fontsize=7.5, color="#222")
        ax.text(mid, -2.5, f"{maj}\n{lbl.split('/')[0]}", ha="center",
                va="bottom", fontsize=6.5, rotation=90, color="#222")

    ax.set_xlabel("Destination occupation (SOC3)")
    ax.set_ylabel("Origin occupation (SOC3)")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=40)
    ax.set_xticks([])
    ax.set_yticks([])

    # Mark top-N cells with small ring markers (no text — too cluttered)
    if F_annot is not None and names is not None and max_label_pairs > 0:
        F_abs = np.abs(F_annot)
        np.fill_diagonal(F_abs, 0.0)
        flat = F_abs.flatten()
        idx = np.argsort(-flat)[:max_label_pairs]
        for ii in idx:
            s, o = np.unravel_index(ii, F_annot.shape)
            ax.plot(o, s, marker="o", markersize=8,
                    markerfacecolor="none",
                    markeredgecolor="yellow", markeredgewidth=1.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=175)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Written summary
# ---------------------------------------------------------------------------

def _write_summary(tag, df_in, df_out, df_pp, df_pc, F_pre, F_post, out_dir):
    lines = []
    lines.append(f"# Transition analysis — κ spec: **{tag}**\n")

    tot_workers = F_pre.sum()
    tot_workers_post = F_post.sum()
    lines.append(f"Aggregate switcher flow (workers/yr): "
                 f"pre = {tot_workers:,.0f},  post = {tot_workers_post:,.0f},  "
                 f"Δ = {tot_workers_post - tot_workers:+,.0f} "
                 f"({100*(tot_workers_post - tot_workers)/max(tot_workers,1):+.1f}%).\n")

    lines.append("## Top 10 occupations — biggest INFLOWS (net workers added)\n")
    lines.append(df_in.head(10).to_markdown(index=False))
    lines.append("")
    lines.append("## Top 10 occupations — biggest OUTFLOWS (net workers lost)\n")
    lines.append(df_out.head(10).to_markdown(index=False))
    lines.append("")
    lines.append("## Top 15 pair flows — post AI (steady state)\n")
    lines.append(df_pp.head(15)[["rank", "origin_soc3", "origin_name",
                                   "dest_soc3", "dest_name",
                                   "flow_pre", "flow_post", "delta"]]
                 .to_markdown(index=False))
    lines.append("")
    lines.append("## Top 15 pair changes — |Δ flow| (post − pre)\n")
    lines.append(df_pc.head(15)[["rank", "origin_soc3", "origin_name",
                                   "dest_soc3", "dest_name",
                                   "flow_pre", "flow_post", "delta",
                                   "pct_change"]]
                 .to_markdown(index=False))
    lines.append("")

    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(TRANS_JSON):
        raise FileNotFoundError(TRANS_JSON)
    with open(TRANS_JSON) as f:
        trans = json.load(f)

    data = build_model_data(verbose=False)
    codes = data["soc3_codes"]
    names = data["soc3_names"]
    employment = data["employment"]
    J = data["J"]

    for spec_tag in ["twoway", "dest", "scalar"]:
        if spec_tag not in trans["by_spec"]:
            continue
        print(f"\n=== analyzing spec: {spec_tag} ===")
        out_dir = os.path.join(ANALYSIS_DIR, spec_tag)
        paths = analyze_spec(spec_tag, trans["by_spec"][spec_tag],
                             codes, names, employment, out_dir)
        print(f"  wrote outputs to {out_dir}")
        for k, v in paths.items():
            print(f"    {k}: {v}")

    print("\n[done] all spec analyses complete")


if __name__ == "__main__":
    main()
