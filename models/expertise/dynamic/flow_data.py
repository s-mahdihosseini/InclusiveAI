"""
Loader for the occupation-to-occupation transition dataset.

Input  : occupation_transitions_public_data_set.dta
         (6-digit SOC pairs with transition_share conditional on switching,
         and total_obs counts of switchers.)

Output : three numpy arrays matched to the model's SOC3 universe (J=94):

    pi_switch_data[s, o]       shape (J, J), rows sum to 1, diagonal 0
        Destination distribution conditional on switching FROM s.

    inflow_share_switch_data[o]   shape (J,), sums to 1
        Share of aggregate switchers landing in destination o,
        weighted by DATA employment at the origin.

    outflow_rate_data[s]          shape (J,)
        Occupation-specific outflow (1-yr switching) rate, normalized so
        the employment-weighted mean equals CPS_AGG (default 0.12).

Aggregation from 6-digit SOC to SOC3 follows the same convention used
by `_load_retraining_matrix_soc3` in simple_dynamic_model.py:
    soc3 = first 4 characters of the 6-digit code (e.g. '29-1141' -> '29-1').

The flow file reports `transition_share` conditional on the worker having
switched occupations, so the raw matrix has no diagonal.  When aggregating
6-digit -> SOC3 pairs, some pairs that are off-diagonal at 6-digit become
WITHIN-minor-group (diagonal at SOC3).  Those flows are *dropped* from
the numerator, because at SOC3 the diagonal is the 'stay' event which
by definition is outside the conditional-switch matrix.  Rows are then
re-normalized to sum to 1.
"""

import os
import numpy as np
import pandas as pd

DEFAULT_FLOW_DTA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "occupation_transitions_public_data_set.dta",
)
DEFAULT_FLOW_DTA = os.path.normpath(DEFAULT_FLOW_DTA)

CPS_AGG_MOBILITY = 0.12   # 1-yr occupational switching rate target


def _soc6_to_soc3(soc6):
    """First 4 chars of 6-digit SOC code (matching model convention)."""
    return str(soc6)[:4]


def load_flow_moments(soc3_codes, data_emp, flow_path=DEFAULT_FLOW_DTA,
                      agg_mobility=CPS_AGG_MOBILITY, verbose=True):
    """
    Aggregate the 6-digit flow matrix to the model's SOC3 universe.

    Parameters
    ----------
    soc3_codes : list[str] of length J=94
        Model SOC3 codes from build_model_data()["soc3_codes"].
    data_emp : ndarray shape (J,)
        Data employment per SOC3 from build_model_data()["employment"].
    flow_path : str
        Path to the .dta flow file.
    agg_mobility : float
        Aggregate 1-yr switching rate used to rescale outflow_rate_data[s].
    verbose : bool

    Returns
    -------
    dict with keys
        pi_switch_data           : (J, J) conditional-on-switch probs
        inflow_share_switch_data : (J,)  aggregate switcher destination share
        outflow_rate_data        : (J,)  unconditional origin outflow rate
        switch_mass_data         : (J,)  Σ_o total_obs[s,o] at SOC3 (pre-norm)
        J
    """
    if not os.path.exists(flow_path):
        raise FileNotFoundError(f"Flow data not found: {flow_path}")

    J = len(soc3_codes)
    code2idx = {c: i for i, c in enumerate(soc3_codes)}

    if verbose:
        print(f"[flow_data] Reading {flow_path} ...")
    df = pd.read_stata(flow_path)

    required = {"soc1", "soc2", "total_obs", "transition_share"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"flow file missing columns: {missing}")

    df["orig_soc3"] = df["soc1"].map(_soc6_to_soc3)
    df["targ_soc3"] = df["soc2"].map(_soc6_to_soc3)

    # Raw switch count = transition_share * total_obs.  Here total_obs is the
    # count of switchers from soc1 (regardless of destination), and
    # transition_share is the destination distribution given origin.  The
    # product is the observed number of s->o transitions at 6-digit.
    df["switch_count"] = df["transition_share"].astype(float) * df["total_obs"].astype(float)

    # Aggregate to SOC3
    agg = (df.groupby(["orig_soc3", "targ_soc3"])["switch_count"]
             .sum()
             .reset_index())

    # Build J x J switch-count matrix (SOC3 level)
    N = np.zeros((J, J), dtype=float)
    dropped_rows = 0
    for _, row in agg.iterrows():
        i = code2idx.get(row["orig_soc3"])
        j = code2idx.get(row["targ_soc3"])
        if i is None or j is None:
            dropped_rows += 1
            continue
        N[i, j] += float(row["switch_count"])

    if verbose and dropped_rows:
        print(f"  dropped {dropped_rows} 6-digit cells "
              f"(SOC3 not in model universe)")

    # Drop within-SOC3 cells: at SOC3 those count as 'stay' events.
    diag_mass = np.diag(N).sum()
    total_mass = N.sum()
    if verbose:
        print(f"  within-SOC3 mass = {diag_mass/total_mass:.2%} of total switch mass "
              f"(will be dropped from conditional-switch matrix)")
    np.fill_diagonal(N, 0.0)

    # Row sums = total switch count at SOC3 (cross-SOC3 switchers only)
    switch_mass = N.sum(axis=1)

    # π(o | s, switch) — destination distribution conditional on switching
    pi_switch = np.zeros_like(N)
    for i in range(J):
        if switch_mass[i] > 0:
            pi_switch[i] = N[i] / switch_mass[i]
        # else: leave as zeros (origin has no observed switchers in data;
        # will be handled gracefully by the calibration loop — those origins
        # contribute zero weight to destination-inflow moments.)

    # Inflow share among switchers, weighted by data employment at origin
    w_origin = data_emp.astype(float).copy()
    w_origin = w_origin / w_origin.sum()
    inflow_share = (w_origin[:, None] * pi_switch).sum(axis=0)
    # Normalize so it sums to exactly 1 (it already does up to numerical)
    inflow_share = inflow_share / inflow_share.sum()

    # Outflow rate per origin, normalized to target aggregate mobility.
    # Raw proxy: switch_mass[s] / data_emp[s] (CPS/ASEC-style rate but
    # on different sampling; we only use relative cross-s variation and
    # renormalize the mean).
    emp = data_emp.astype(float).copy()
    emp = np.maximum(emp, 1e-8)
    raw_rate = switch_mass / emp
    # Origins with zero observed switchers are not identified by this data.
    # Build a valid_mask = False for them; do NOT target their outflow.
    valid_mask = switch_mass > 0

    # Rescale so the employment-weighted mean outflow (over VALID origins)
    # equals the CPS aggregate.
    emp_valid = emp.copy()
    emp_valid[~valid_mask] = 0.0
    w_origin_valid = emp_valid / max(emp_valid.sum(), 1e-15)
    raw_mean = (w_origin_valid * raw_rate).sum()
    if raw_mean > 0:
        outflow_rate = raw_rate * (agg_mobility / raw_mean)
    else:
        outflow_rate = np.full(J, agg_mobility)
    outflow_rate = np.minimum(outflow_rate, 0.99).astype(float)
    # Non-valid origins: NaN so any accidental use downstream fails loudly.
    outflow_rate[~valid_mask] = np.nan

    if verbose:
        n_drop = int((~valid_mask).sum())
        print(f"  outflow rate (valid origins only, n={int(valid_mask.sum())}): "
              f"min={np.nanmin(outflow_rate):.3f} "
              f"p50={np.nanmedian(outflow_rate):.3f} "
              f"max={np.nanmax(outflow_rate):.3f}")
        print(f"  inflow  share: min={inflow_share.min():.4f} "
              f"p50={np.median(inflow_share):.4f} max={inflow_share.max():.4f}")
        if n_drop > 0:
            print(f"  dropped {n_drop} origin(s) with zero switch_mass "
                  f"(not identified by flow data)")

    return {
        "pi_switch_data":           pi_switch,
        "inflow_share_switch_data": inflow_share,
        "outflow_rate_data":        outflow_rate,
        "switch_mass_data":         switch_mass,
        "valid_origin_mask":        valid_mask,
        "J":                        J,
    }


if __name__ == "__main__":
    import paths_override  # noqa: F401
    from simple_dynamic_model import build_model_data
    data = build_model_data(verbose=False)
    fm = load_flow_moments(data["soc3_codes"], data["employment"])
    print("OK — loaded flow moments.")
    print(" pi_switch_data rows sum to 1?  ",
          np.allclose(fm["pi_switch_data"].sum(axis=1)[
              fm["switch_mass_data"] > 0], 1.0))
