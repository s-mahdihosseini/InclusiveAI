"""
Kappa-heterogeneity extension of the Simple Dynamic Model.

Supports three parameterizations of the switching-cost coefficient:

    'scalar'  : cost[s, o] = kappa        * d[s, o]                (1 param)
    'dest'    : cost[s, o] = kappa_in[o]  * d[s, o]                (J params)
    'twoway'  : cost[s, o] = (kappa_out[s] + kappa_in[o]) * d[s,o] (2J params)

Structural interpretation:
    kappa_out[s] = origin 'stickiness'   (how costly to leave s)
    kappa_in [o] = destination 'barrier' (how costly to enter o)

This module re-implements the minimum set of functions from
simple_dynamic_model that depend on kappa, so that scalar / origin /
destination / two-way kappa all reduce to supplying a J x J cost
multiplier matrix.  Everything else (production, CES demand, stationary
distribution, amenity inversion logic) is reused.

The model is by assumption *structural in kappa*: the AI shock is
modelled only via d_so -> d_so_AI, so kappa vectors do NOT change
between pre- and post-AI steady states.
"""

import numpy as np
from simple_dynamic_model import (
    logsumexp,
    compute_stationary_distribution,
    compute_L_eff,
    compute_wages,
    invert_B,
    compute_var_log_occ_wage,
)


# ---------------------------------------------------------------------------
# 1. Kappa container
# ---------------------------------------------------------------------------

class KappaSpec:
    """
    Holds kappa in its native parameterization plus a materialized J x J
    multiplier matrix K such that cost[s, o] = K[s, o] * d[s, o].
    """

    def __init__(self, kind, J, kappa=None, kappa_out=None, kappa_in=None):
        self.kind = kind
        self.J = J
        if kind == "scalar":
            if kappa is None:
                raise ValueError("scalar kappa needs `kappa`")
            self.kappa = float(kappa)
            self.kappa_out = None
            self.kappa_in  = None
        elif kind == "dest":
            if kappa_in is None:
                raise ValueError("dest needs `kappa_in` (J,)")
            self.kappa = None
            self.kappa_out = None
            self.kappa_in  = np.asarray(kappa_in, float).copy()
            assert self.kappa_in.shape == (J,)
        elif kind == "twoway":
            if kappa_out is None or kappa_in is None:
                raise ValueError("twoway needs `kappa_out` AND `kappa_in`")
            self.kappa = None
            self.kappa_out = np.asarray(kappa_out, float).copy()
            self.kappa_in  = np.asarray(kappa_in, float).copy()
            assert self.kappa_out.shape == (J,)
            assert self.kappa_in.shape == (J,)
        else:
            raise ValueError(f"unknown kind: {kind}")

    def multiplier(self):
        """Return the (J, J) multiplier matrix K."""
        J = self.J
        if self.kind == "scalar":
            return np.full((J, J), self.kappa)
        if self.kind == "dest":
            return np.broadcast_to(self.kappa_in[None, :], (J, J)).copy()
        if self.kind == "twoway":
            return self.kappa_out[:, None] + self.kappa_in[None, :]
        raise RuntimeError

    def copy(self):
        if self.kind == "scalar":
            return KappaSpec("scalar", self.J, kappa=self.kappa)
        if self.kind == "dest":
            return KappaSpec("dest", self.J, kappa_in=self.kappa_in.copy())
        return KappaSpec("twoway", self.J,
                         kappa_out=self.kappa_out.copy(),
                         kappa_in=self.kappa_in.copy())


# ---------------------------------------------------------------------------
# 2. Model primitives (re-implement kappa-using functions)
# ---------------------------------------------------------------------------

def build_cost_matrix(kspec: KappaSpec, d):
    """cost[s, o] = multiplier[s, o] * d[s, o] with zero diagonal."""
    K = kspec.multiplier()
    d_c = d.copy()
    np.fill_diagonal(d_c, 0.0)
    cost = K * np.maximum(d_c, 0.0)
    np.fill_diagonal(cost, 0.0)
    return cost


def solve_vf_vec(wages, p, kspec, d, tol=1e-6, max_iter=500,
                 V_init=None, return_policy=False):
    """Bellman value iteration with vector kappa."""
    J     = p.J
    tau   = p.tau
    beta  = p.beta_eff
    log_w = np.log(np.maximum(wages, 1e-10))

    cost = build_cost_matrix(kspec, d)
    a_o  = p.a_o

    V = V_init.copy() if V_init is not None else log_w.copy()
    for _ in range(max_iter):
        v_all = (log_w[None, :] + a_o[None, :] - cost + beta * V[None, :]) / tau
        V_new = tau * logsumexp(v_all, axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new

    if return_policy:
        v_all = (log_w[None, :] + a_o[None, :] - cost + beta * V[None, :]) / tau
        v_max = v_all.max(axis=1, keepdims=True)
        exp_v = np.exp(v_all - v_max)
        policy = exp_v / exp_v.sum(axis=1, keepdims=True)
        return V, policy
    return V


def invert_amenities_vec(wages, p, kspec, d, data_emp,
                         max_iter=80, tol=1e-4, verbose=False):
    """BLP-style amenity inversion, re-implemented with vector kappa."""
    J = p.J
    emp_data = data_emp.astype(float).copy()
    emp_data = emp_data / emp_data.sum()
    emp_data = np.maximum(emp_data, 1e-8)
    log_emp_data = np.log(emp_data)

    p.a_o = np.zeros(J)
    dynamic_step = (1.0 - p.beta)
    damping = 0.7
    prev_dev = None

    for it in range(max_iter):
        V, policy = solve_vf_vec(wages, p, kspec, d,
                                  return_policy=True, tol=1e-6)
        mu = compute_stationary_distribution(V, p, d, policy=policy, tol=1e-9)
        emp_model = np.maximum(mu / max(mu.sum(), 1e-15), 1e-8)
        max_dev = np.max(np.abs(emp_model - emp_data))
        if max_dev < tol:
            if verbose:
                print(f"    amenity converged iter={it} max_dev={max_dev:.2e}")
            break
        if prev_dev is not None and max_dev > prev_dev * 1.1:
            damping *= 0.8
        prev_dev = max_dev
        p.a_o = p.a_o + dynamic_step * damping * (
            log_emp_data - np.log(emp_model))
        p.a_o = p.a_o - p.a_o.mean()
    return p


def solve_steady_state_vec(p, kspec, d, data_emp,
                           max_iter=200, tol=1e-3, damping=0.15,
                           invert_amenities_flag=True,
                           amenity_tol=1e-4, amenity_max_iter=80,
                           B_ext=None, wages_init=None, verbose=False):
    """Pre-AI steady state with vector kappa and amenity / B inversion."""
    J     = p.J
    sigma = p.sigma
    wages = (wages_init.copy() if wages_init is not None
             else p.data["mean_wage"].copy())

    if invert_amenities_flag:
        invert_amenities_vec(wages, p, kspec, d, data_emp,
                             max_iter=amenity_max_iter, tol=amenity_tol,
                             verbose=verbose)

    if B_ext is not None:
        B = B_ext.copy()
    else:
        V, policy = solve_vf_vec(wages, p, kspec, d, return_policy=True)
        mu = compute_stationary_distribution(V, p, d, policy=policy)
        L_eff = compute_L_eff(mu, p)
        B = invert_B(wages, L_eff, sigma)

    prev_dw = None
    for it in range(max_iter):
        V, policy = solve_vf_vec(wages, p, kspec, d, return_policy=True)
        mu = compute_stationary_distribution(V, p, d, policy=policy)
        L_eff = compute_L_eff(mu, p)
        w_new = compute_wages(B, L_eff, sigma)
        dw = np.max(np.abs(w_new - wages) / np.maximum(wages, 1e-10))
        if dw < tol:
            break
        if prev_dw is not None and dw > prev_dw * 1.05:
            damping = max(damping * 0.85, 0.02)
        prev_dw = dw
        wages = wages + damping * (w_new - wages)

    return wages, B, V, mu, L_eff, policy


# ---------------------------------------------------------------------------
# 3. Flow moments from model
# ---------------------------------------------------------------------------

def compute_flow_moments(policy, mu, data_emp):
    """
    Return model counterparts of the flow-data moments.

    policy[s, o]  = unconditional π(o | s), diagonal = stay prob.
    """
    J = policy.shape[0]
    s_idx = np.arange(J)
    stay_prob = policy[s_idx, s_idx]
    switch_prob = 1.0 - stay_prob

    # π^switch(o | s) = π(o | s) / (1 - π(s|s))  for o != s
    pi_switch = policy.copy()
    np.fill_diagonal(pi_switch, 0.0)
    row_sum = pi_switch.sum(axis=1)
    pi_switch = pi_switch / np.maximum(row_sum[:, None], 1e-15)

    # Outflow rate per origin (unconditional 1-yr switching rate)
    outflow_rate_model = switch_prob.copy()

    # Inflow share among switchers, weighted by DATA employment
    w_origin = data_emp.astype(float) / data_emp.sum()
    inflow_share_model = (w_origin[:, None] * pi_switch).sum(axis=0)
    inflow_share_model = inflow_share_model / inflow_share_model.sum()

    # Aggregate 1-yr mobility rate using model mu
    mobility_rate = (mu * switch_prob).sum() / max(mu.sum(), 1e-15)

    return {
        "pi_switch_model":          pi_switch,
        "inflow_share_switch_model": inflow_share_model,
        "outflow_rate_model":        outflow_rate_model,
        "mobility_rate":             mobility_rate,
    }


# ---------------------------------------------------------------------------
# 4. Fixed-point calibrators
# ---------------------------------------------------------------------------

def calibrate_dest(p, d, data_emp, data_flow,
                   kappa_init=5.0,
                   step=0.6, max_iter=40, tol=1e-3,
                   reinvert_amenity_every=5,
                   amenity_tol=1e-3, amenity_max_iter=40,
                   verbose=True):
    """
    Calibrate destination-specific kappa_in[o] to match destination
    inflow shares among switchers.

    Update rule (log-space, keeps kappa_in positive):
        kappa_in[o] <- kappa_in[o] * exp( step * (log f_model - log f_data) )
    Intuition: if model inflow to o is too HIGH relative to data, the
    entry barrier kappa_in[o] is too LOW, so raise it.
    """
    J = p.J
    f_data = data_flow["inflow_share_switch_data"]
    f_data = np.maximum(f_data, 1e-8)

    kspec = KappaSpec("dest", J, kappa_in=np.full(J, float(kappa_init)))

    history = []
    B_cache = None

    for it in range(max_iter):
        # Re-invert amenities and B periodically; otherwise reuse B and
        # just re-solve wages.  First iter always re-inverts.
        reinvert = (it % reinvert_amenity_every == 0) or (it == 0)
        wages, B, V, mu, L_eff, policy = solve_steady_state_vec(
            p, kspec, d, data_emp,
            invert_amenities_flag=reinvert,
            amenity_tol=amenity_tol, amenity_max_iter=amenity_max_iter,
            B_ext=(None if reinvert else B_cache),
            max_iter=60, tol=1e-3, verbose=False,
        )
        B_cache = B

        fm = compute_flow_moments(policy, mu, data_emp)
        f_mod = np.maximum(fm["inflow_share_switch_model"], 1e-8)

        log_dev = np.log(f_mod) - np.log(f_data)
        max_log_dev = np.max(np.abs(log_dev))
        mob = fm["mobility_rate"]
        var_lw = compute_var_log_occ_wage(wages, mu)

        history.append({
            "iter": it,
            "max_log_dev_inflow": float(max_log_dev),
            "mobility_rate":      float(mob),
            "var_log_occ_wage":   float(var_lw),
            "kappa_in_min":       float(kspec.kappa_in.min()),
            "kappa_in_max":       float(kspec.kappa_in.max()),
        })
        if verbose:
            print(f"  [dest] iter={it:3d}  maxΔlog(f)={max_log_dev:.4f}  "
                  f"mob={mob:.4f}  var(log w)={var_lw:.4f}  "
                  f"κ_in∈[{kspec.kappa_in.min():.2f},{kspec.kappa_in.max():.2f}]")

        if max_log_dev < tol:
            if verbose:
                print(f"  [dest] converged in {it} iters")
            break

        # Multiplicative update, clipped for stability
        update = np.exp(np.clip(step * log_dev, -0.5, 0.5))
        kspec.kappa_in = np.clip(kspec.kappa_in * update, 1e-3, 1e3)

    return kspec, history, (wages, B, V, mu, L_eff, policy)


def calibrate_twoway(p, d, data_emp, data_flow,
                     kappa_out_init=2.0, kappa_in_init=3.0,
                     step_out=0.4, step_in=0.4,
                     max_iter=50, tol=1e-3,
                     reinvert_amenity_every=5,
                     amenity_tol=1e-3, amenity_max_iter=40,
                     verbose=True):
    """
    Calibrate two-way κ_out[s] + κ_in[o] to jointly match:
        - origin outflow rate m_s             (J moments, -> κ_out)
        - destination inflow share f_o        (J moments, -> κ_in)

    Update rules:
        κ_out[s] <- κ_out[s] * exp( step_out * (log m_model[s] - log m_data[s]) * (-1) )
                 i.e. if model outflow > data, raise κ_out to reduce outflow.
        κ_in [o] <- κ_in [o] * exp( step_in  * (log f_model[o] - log f_data[o]) )
                 i.e. if model inflow  > data, raise κ_in  to reduce inflow.
    """
    J = p.J
    # Valid-origin mask: origins with zero switch mass in the data are
    # NOT identified by this data and their outflow moment is dropped.
    valid_mask = data_flow.get("valid_origin_mask",
                               np.ones(J, dtype=bool))
    m_data_raw = data_flow["outflow_rate_data"]
    # Fill NaN for non-valid origins with a placeholder; never used.
    m_data = np.where(valid_mask, np.nan_to_num(m_data_raw, nan=0.0), 0.0)
    m_data = np.maximum(m_data, 1e-5)
    f_data = np.maximum(data_flow["inflow_share_switch_data"], 1e-8)

    kspec = KappaSpec(
        "twoway", J,
        kappa_out=np.full(J, float(kappa_out_init)),
        kappa_in =np.full(J, float(kappa_in_init)),
    )

    history = []
    B_cache = None

    for it in range(max_iter):
        reinvert = (it % reinvert_amenity_every == 0) or (it == 0)
        wages, B, V, mu, L_eff, policy = solve_steady_state_vec(
            p, kspec, d, data_emp,
            invert_amenities_flag=reinvert,
            amenity_tol=amenity_tol, amenity_max_iter=amenity_max_iter,
            B_ext=(None if reinvert else B_cache),
            max_iter=60, tol=1e-3, verbose=False,
        )
        B_cache = B

        fm = compute_flow_moments(policy, mu, data_emp)
        m_mod = np.maximum(fm["outflow_rate_model"],        1e-5)
        f_mod = np.maximum(fm["inflow_share_switch_model"], 1e-8)

        log_dev_m = np.log(m_mod) - np.log(m_data)
        log_dev_f = np.log(f_mod) - np.log(f_data)
        # Zero-out deviations on non-valid origins so they affect neither
        # the convergence check nor the kappa_out update.
        log_dev_m = np.where(valid_mask, log_dev_m, 0.0)
        max_dev_m = np.max(np.abs(log_dev_m))
        max_dev_f = np.max(np.abs(log_dev_f))
        mob = fm["mobility_rate"]
        var_lw = compute_var_log_occ_wage(wages, mu)

        history.append({
            "iter":                it,
            "max_log_dev_outflow": float(max_dev_m),
            "max_log_dev_inflow":  float(max_dev_f),
            "mobility_rate":       float(mob),
            "var_log_occ_wage":    float(var_lw),
            "kappa_out_min":       float(kspec.kappa_out.min()),
            "kappa_out_max":       float(kspec.kappa_out.max()),
            "kappa_in_min":        float(kspec.kappa_in.min()),
            "kappa_in_max":        float(kspec.kappa_in.max()),
        })
        if verbose:
            print(f"  [twoway] iter={it:3d}  maxΔlog(m)={max_dev_m:.4f}  "
                  f"maxΔlog(f)={max_dev_f:.4f}  mob={mob:.4f}  "
                  f"var(log w)={var_lw:.4f}  "
                  f"κ_out∈[{kspec.kappa_out.min():.2f},{kspec.kappa_out.max():.2f}] "
                  f"κ_in∈[{kspec.kappa_in.min():.2f},{kspec.kappa_in.max():.2f}]")

        if max_dev_m < tol and max_dev_f < tol:
            if verbose:
                print(f"  [twoway] converged in {it} iters")
            break

        up_out = np.exp(np.clip(step_out * log_dev_m, -0.5, 0.5))
        up_in  = np.exp(np.clip(step_in  * log_dev_f, -0.5, 0.5))
        kspec.kappa_out = np.clip(kspec.kappa_out * up_out, 1e-3, 1e3)
        kspec.kappa_in  = np.clip(kspec.kappa_in  * up_in,  1e-3, 1e3)

    return kspec, history, (wages, B, V, mu, L_eff, policy)


# ---------------------------------------------------------------------------
# 5. Stabilized two-way calibrator (kills period-2 limit cycles)
# ---------------------------------------------------------------------------

def _solve_and_moments(p, kspec, d, data_emp, B_cache, do_reinvert,
                       amenity_tol, amenity_max_iter):
    """Helper: solve steady state (optionally re-invert amenities/B) and
    return model flow moments.  B_cache is reused when do_reinvert=False."""
    wages, B, V, mu, L_eff, policy = solve_steady_state_vec(
        p, kspec, d, data_emp,
        invert_amenities_flag=do_reinvert,
        amenity_tol=amenity_tol, amenity_max_iter=amenity_max_iter,
        B_ext=(None if do_reinvert else B_cache),
        max_iter=60, tol=1e-3, verbose=False,
    )
    fm = compute_flow_moments(policy, mu, data_emp)
    return wages, B, V, mu, L_eff, policy, fm


def calibrate_twoway_stable(p, d, data_emp, data_flow,
                            kappa_out_init=2.0, kappa_in_init=3.0,
                            step_out=0.20, step_in=0.20,
                            max_iter=60, tol=5e-3,
                            reinvert_amenity_every=4,
                            amenity_tol=2e-3, amenity_max_iter=30,
                            ema_window=4,
                            adapt_step=True,
                            gauss_seidel=True,
                            verbose=True):
    """
    Stabilized calibration for the two-way κ_out[s] + κ_in[o] specification.

    Stabilization mechanisms (stacked — each addresses a distinct source of
    Jacobi-fixed-point oscillation observed in the vanilla calibrator):

        (i)  Smaller base step (0.20 vs. 0.40 in the plain version).
        (ii) Gauss-Seidel alternation: update κ_in, re-equilibrate, update
             κ_out, re-equilibrate — within each outer iter.  Decouples the
             simultaneous feedback that causes period-2 cycles.
        (iii) Log-space EMA over the most recent `ema_window` iterates of
             (log κ_out, log κ_in).  Krasnoselskii–Mann averaging: kills
             period-2 (and higher-period-small) limit cycles of nonexpansive
             maps.  The "published" estimate at any iter is the EMA state.
        (iv) Adaptive step shrinking: if the max log-deviation has not
             improved over the last two EMA-smoothed iterates, multiply the
             step by 0.7.  Prevents permanent oscillation near the fixed
             point.

    Unidentified origins (valid_origin_mask[s] = False in data_flow) are
    held at the init value for κ_out[s] and excluded from the convergence
    metric on outflow.
    """
    import collections
    J = p.J

    # ----- data moments + valid mask -----
    valid_mask = data_flow.get("valid_origin_mask", np.ones(J, dtype=bool))
    m_data_raw = data_flow["outflow_rate_data"]
    m_data = np.where(valid_mask, np.nan_to_num(m_data_raw, nan=0.0), 0.0)
    m_data = np.maximum(m_data, 1e-5)
    f_data = np.maximum(data_flow["inflow_share_switch_data"], 1e-8)
    log_m_data = np.log(m_data)
    log_f_data = np.log(f_data)

    # ----- state -----
    kspec = KappaSpec(
        "twoway", J,
        kappa_out=np.full(J, float(kappa_out_init)),
        kappa_in =np.full(J, float(kappa_in_init)),
    )
    log_ko_hist = collections.deque(maxlen=ema_window)
    log_ki_hist = collections.deque(maxlen=ema_window)

    step_o = float(step_out)
    step_i = float(step_in)
    prev_total_dev = np.inf
    worse_streak   = 0

    history = []
    B_cache = None

    for it in range(max_iter):
        reinvert_outer = (it % reinvert_amenity_every == 0) or (it == 0)

        # -----------------------------------------------------------------
        # Gauss-Seidel half-step 1 : update κ_in given current κ_out
        # -----------------------------------------------------------------
        wages, B, V, mu, L_eff, policy, fm = _solve_and_moments(
            p, kspec, d, data_emp, B_cache, reinvert_outer,
            amenity_tol, amenity_max_iter)
        B_cache = B

        f_mod = np.maximum(fm["inflow_share_switch_model"], 1e-8)
        log_dev_f = np.log(f_mod) - log_f_data
        # multiplicative update in log space
        log_ki_new = np.log(kspec.kappa_in) + np.clip(step_i * log_dev_f,
                                                     -0.4, 0.4)
        kspec.kappa_in = np.clip(np.exp(log_ki_new), 1e-3, 1e3)

        if gauss_seidel:
            # Re-solve intermediate steady state so κ_out update sees the
            # new κ_in's equilibrium impact
            wages, B, V, mu, L_eff, policy, fm = _solve_and_moments(
                p, kspec, d, data_emp, B_cache, False,
                amenity_tol, amenity_max_iter)

        # -----------------------------------------------------------------
        # Gauss-Seidel half-step 2 : update κ_out given current κ_in
        # -----------------------------------------------------------------
        m_mod = np.maximum(fm["outflow_rate_model"], 1e-5)
        log_dev_m = np.log(m_mod) - log_m_data
        log_dev_m = np.where(valid_mask, log_dev_m, 0.0)
        log_ko_new = np.log(kspec.kappa_out) + np.clip(step_o * log_dev_m,
                                                       -0.4, 0.4)
        kspec.kappa_out = np.clip(np.exp(log_ko_new), 1e-3, 1e3)

        # -----------------------------------------------------------------
        # Re-equilibrate after full Gauss-Seidel pass and compute final
        # moments that will be reported this iter.
        # -----------------------------------------------------------------
        wages, B, V, mu, L_eff, policy, fm = _solve_and_moments(
            p, kspec, d, data_emp, B_cache, False,
            amenity_tol, amenity_max_iter)

        m_mod = np.maximum(fm["outflow_rate_model"],        1e-5)
        f_mod = np.maximum(fm["inflow_share_switch_model"], 1e-8)
        log_dev_m = np.log(m_mod) - log_m_data
        log_dev_f = np.log(f_mod) - log_f_data
        log_dev_m = np.where(valid_mask, log_dev_m, 0.0)
        max_dev_m = float(np.max(np.abs(log_dev_m)))
        max_dev_f = float(np.max(np.abs(log_dev_f)))
        total_dev = max_dev_m + max_dev_f

        # -----------------------------------------------------------------
        # Adaptive EMA smoothing of log-κ  (Krasnoselskii–Mann averaging).
        # Detect oscillation by checking whether the current update-direction
        # flipped sign (coord-wise) relative to the previous update.  If the
        # flip fraction exceeds 30%, apply full EMA averaging.  Otherwise
        # the sequence is progressing monotonically and we skip smoothing.
        # -----------------------------------------------------------------
        log_ko_hist.append(np.log(kspec.kappa_out))
        log_ki_hist.append(np.log(kspec.kappa_in))
        flip_detected = False
        if len(log_ko_hist) >= 3:
            d1o = log_ko_hist[-1] - log_ko_hist[-2]
            d2o = log_ko_hist[-2] - log_ko_hist[-3]
            d1i = log_ki_hist[-1] - log_ki_hist[-2]
            d2i = log_ki_hist[-2] - log_ki_hist[-3]
            flip_o = float(np.mean(d1o * d2o < 0))
            flip_i = float(np.mean(d1i * d2i < 0))
            flip_detected = (flip_o > 0.3) or (flip_i > 0.3)
            if flip_detected:
                log_ko_avg = np.mean(np.stack(list(log_ko_hist), axis=0),
                                     axis=0)
                log_ki_avg = np.mean(np.stack(list(log_ki_hist), axis=0),
                                     axis=0)
                kspec.kappa_out = np.clip(np.exp(log_ko_avg), 1e-3, 1e3)
                kspec.kappa_in  = np.clip(np.exp(log_ki_avg), 1e-3, 1e3)

        # -----------------------------------------------------------------
        # Adaptive step shrink on lack-of-progress (after warmup iters)
        # -----------------------------------------------------------------
        if adapt_step and it >= ema_window:
            if total_dev > prev_total_dev * 0.995:
                worse_streak += 1
                if worse_streak >= 2:
                    step_o *= 0.7
                    step_i *= 0.7
                    worse_streak = 0
                    if verbose:
                        print(f"    [adapt] shrink steps -> "
                              f"step_out={step_o:.3f} step_in={step_i:.3f}")
            else:
                worse_streak = 0
        prev_total_dev = total_dev

        var_lw = compute_var_log_occ_wage(wages, mu)
        history.append({
            "iter":                it,
            "max_log_dev_outflow": max_dev_m,
            "max_log_dev_inflow":  max_dev_f,
            "mobility_rate":       float(fm["mobility_rate"]),
            "var_log_occ_wage":    float(var_lw),
            "step_out":            float(step_o),
            "step_in":             float(step_i),
            "flip_detected":       bool(flip_detected),
            "kappa_out_min":       float(kspec.kappa_out.min()),
            "kappa_out_max":       float(kspec.kappa_out.max()),
            "kappa_in_min":        float(kspec.kappa_in.min()),
            "kappa_in_max":        float(kspec.kappa_in.max()),
        })

        if verbose:
            flag = " EMA" if flip_detected else ""
            print(f"  [twoway-stable] it={it:3d}  "
                  f"|Δlog|m={max_dev_m:.4f}  |Δlog|f={max_dev_f:.4f}  "
                  f"mob={fm['mobility_rate']:.4f}  var(lnw)={var_lw:.4f}  "
                  f"step=({step_o:.3f},{step_i:.3f}){flag}")

        if max_dev_m < tol and max_dev_f < tol:
            if verbose:
                print(f"  [twoway-stable] converged in {it} iters")
            break

    return kspec, history, (wages, B, V, mu, L_eff, policy)
