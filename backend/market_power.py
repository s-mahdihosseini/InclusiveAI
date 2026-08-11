"""
InclusiveAI — AI supplier market power & compute investment model.

Closed-form implementation of Hosseini & Lichtinger (2026), "AI Productivity,
Upward-Sloping Factor Supply, and the Dynamics of Compute Investment".

Static block (Proposition 1), evaluated at an equilibrium with AI revenue
share s_X (s_1 = 1 - s_X), theta = s_1/sigma:

    D  = 1 + eta * (1 - beta + beta*theta)
    H  = (1 + eta) / D
    E[X,A] = H;  E[pX,A] = -theta*H;  E[L2,A] = eta*(1-theta)/D
    E[w2,A] = (1-theta)/D;  E[Y,A] = s_X*H;  E[w1,A] = (s_X/sigma)*H
    dlog s_X/dlog A = rho*s_1*H,  rho = (sigma-1)/sigma

Dynamic block (Propositions: transition after a permanent shock epsilon):

    B = (1-theta)*H,  a = 1 - gamma*B  (stability: a > 0)
    K_inf = B/a
    r*K = 1/b - (1-delta)
    T(phi) = 1 + 1/b + r*K * a / phi
    lambda = [T - sqrt(T^2 - 4/b)] / 2          (stable root in (0,1))

    K_t  = K_inf * (1 - lambda^t)               (log dev, per unit epsilon)
    i_t  = K_inf * (1 - lambda) * lambda^t      (level dev of I/K)
    q_t  = phi * i_t                            (level dev; q* = 1)
    rK_t = B * lambda^t
    all static variables scale with (1 + gamma*K_t).

Market-power extension (Appendix D): with the aggregate GE demand curve the
monopolist's marginal-revenue wedge is m = 1 - theta; AI-revenue split becomes
labor beta*m, compute gamma*m, profit 1-(beta+gamma)*m (competitive:
beta, gamma, 1-beta-gamma).

Fixed calibration constants (quarterly): b = 0.99, delta = 0.025, horizon 40.
"""

from functools import lru_cache

import numpy as np

B_DISC = 0.99   # discount factor (quarterly)
DELTA = 0.025   # compute depreciation (quarterly)
HORIZON = 40    # quarters plotted


def statics(sigma, sx0, eta, beta):
    s1 = 1.0 - sx0
    theta = s1 / sigma
    rho = (sigma - 1.0) / sigma
    D = 1.0 + eta * (1.0 - beta + beta * theta)
    H = (1.0 + eta) / D
    return {
        "s1": s1, "theta": theta, "rho": rho, "D": D, "H": H,
        "E_X": H,
        "E_pX": -theta * H,
        "E_L2": eta * (1.0 - theta) / D,
        "E_w2": (1.0 - theta) / D,
        "E_Y": sx0 * H,
        "E_w1": (sx0 / sigma) * H,
        "E_logsX": rho * s1 * H,          # dlog s_X / dlog A
        "E_Pi": (1.0 - theta) * H,        # dlog Pi_Z / dlog A
    }


def dynamics(st, gamma, phi):
    theta, H = st["theta"], st["H"]
    B = (1.0 - theta) * H
    a = 1.0 - gamma * B
    if a <= 1e-9:
        return None, B, a                  # unstable configuration
    K_inf = B / a
    r_star = 1.0 / B_DISC - (1.0 - DELTA)
    T = 1.0 + 1.0 / B_DISC + r_star * a / phi
    disc = T * T - 4.0 / B_DISC
    lam = 0.5 * (T - np.sqrt(disc))
    return {"B": B, "a": a, "K_inf": K_inf, "r_star": r_star,
            "T": T, "lam": float(lam),
            "half_life": float(np.log(0.5) / np.log(lam))}, B, a


@lru_cache(maxsize=256)
def solve(sigma=2.0, sx0=0.5, eta=1.0, beta=0.4, gamma=0.4,
          phi=8.0, shock=1.0):
    """Full API payload. `shock` is the permanent productivity increase in %."""
    sigma = max(float(sigma), 0.05)
    sx0 = min(max(float(sx0), 0.01), 0.95)
    eta = max(float(eta), 0.0)
    beta = min(max(float(beta), 0.01), 0.94)
    gamma = min(max(float(gamma), 0.01), 0.95 - beta)  # enforce beta+gamma<=0.95
    phi = max(float(phi), 0.05)
    eps = float(shock)                                  # percent

    st = statics(sigma, sx0, eta, beta)
    dyn, B, a = dynamics(st, gamma, phi)
    if dyn is None:
        return {"error": (
            f"Unstable configuration: gamma*B = {gamma * B:.3f} >= 1. "
            "The scale feedback from compute accumulation explodes — lower "
            "gamma, lower eta, raise sigma, or lower the AI share.")}

    lam, K_inf = dyn["lam"], dyn["K_inf"]
    t = np.arange(HORIZON + 1)
    lam_t = lam ** t

    K = K_inf * (1.0 - lam_t)                # log dev per unit shock
    scale = 1.0 + gamma * K                  # (1 + gamma*K_t)
    i_pp = K_inf * (1.0 - lam) * lam_t       # level dev of I/K
    q = phi * i_pp
    rK = B * lam_t

    def pct(x):
        return (np.asarray(x) * eps).tolist()   # scale by shock (%)

    path = {
        "t": t.tolist(),
        "K": pct(K),
        "i_pp": pct(i_pp),
        "q": pct(q),
        "rK": pct(rK),
        "X": pct(st["E_X"] * scale),
        "pX": pct(st["E_pX"] * scale),
        "w1": pct(st["E_w1"] * scale),
        "w2": pct(st["E_w2"] * scale),
        "L2": pct(st["E_L2"] * scale),
        "Y": pct(st["E_Y"] * scale),
        # Rents to the hard-to-replicate fixed factor: Pi = (1-b-g) pX X
        "Pi": pct((1.0 - st["theta"]) * st["H"] * scale),
    }

    # Income-share paths (level changes, percentage points of GDP):
    # dlog s_X,t = rho*s1*H*(1+gamma*K_t)*eps -> ds_X = s_X * dlog s_X
    dlog_sx = st["E_logsX"] * scale * (eps / 100.0)
    ds_x = sx0 * dlog_sx * 100.0                       # pp of GDP
    path["shares"] = {
        "dsX": ds_x.tolist(),
        "ds1": (-ds_x).tolist(),
        "ds2": (beta * ds_x).tolist(),
        "dsK": (gamma * ds_x).tolist(),
        "dsPi": ((1.0 - beta - gamma) * ds_x).tolist(),
    }

    # Market-power extension: split of AI revenue
    m = 1.0 - st["theta"]                               # aggregate-demand markup wedge
    split = {
        "markup_wedge": m,
        "demand_elasticity": sigma / st["s1"],
        "competitive": {"labor": beta, "compute": gamma,
                        "profit": 1.0 - beta - gamma},
        "monopoly": {"labor": beta * m, "compute": gamma * m,
                     "profit": 1.0 - (beta + gamma) * m},
    }

    impact = {k: float(np.asarray(v)[0]) for k, v in path.items()
              if k not in ("t", "shares")}
    # True long run (t -> infinity): lambda^t -> 0, K -> K_inf, q,i,rK -> steady state
    scale_inf = 1.0 + gamma * K_inf
    longrun = {
        "K": K_inf * eps, "i_pp": 0.0, "q": 0.0, "rK": 0.0,
        "X": st["E_X"] * scale_inf * eps,
        "pX": st["E_pX"] * scale_inf * eps,
        "w1": st["E_w1"] * scale_inf * eps,
        "w2": st["E_w2"] * scale_inf * eps,
        "L2": st["E_L2"] * scale_inf * eps,
        "Y": st["E_Y"] * scale_inf * eps,
        "Pi": (1.0 - st["theta"]) * st["H"] * scale_inf * eps,
    }

    return {
        "params": {"sigma": sigma, "sx0": sx0, "eta": eta, "beta": beta,
                   "gamma": gamma, "phi": phi, "shock": eps,
                   "b": B_DISC, "delta": DELTA},
        "derived": {"theta": st["theta"], "D": st["D"], "H": st["H"],
                    "B": dyn["B"], "a": dyn["a"], "lambda": lam,
                    "half_life": dyn["half_life"], "K_inf": K_inf,
                    "r_star": dyn["r_star"],
                    "scale_vs_price": "scale" if st["theta"] < 1 else "price"},
        "impact": impact,
        "longrun": longrun,
        "path": path,
        "markup": split,
    }


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    # Reproduce the note's illustrative calibration exactly:
    # sigma=2, s1=0.5, eta=1, beta=0.4, gamma=0.4, b=0.99, delta=0.025
    # -> lambda = 0.904, 0.953, 0.977 for phi = 2, 8, 30; K_inf = 1.3636; B = 0.8824
    st = statics(sigma=2.0, sx0=0.5, eta=1.0, beta=0.4)
    assert abs(st["theta"] - 0.25) < 1e-12
    assert abs(st["H"] - 2.0 / 1.7) < 1e-12
    roots = {}
    for phi, target in [(2.0, 0.903624), (8.0, 0.952796), (30.0, 0.977383)]:
        dyn, B, a = dynamics(st, gamma=0.4, phi=phi)
        roots[phi] = dyn["lam"]
        assert abs(dyn["lam"] - target) < 1e-5, (phi, dyn["lam"], target)
    dyn, B, a = dynamics(st, gamma=0.4, phi=8.0)
    assert abs(B - 0.882353) < 1e-5
    assert abs(dyn["K_inf"] - 1.363636) < 1e-5
    print("paper calibration reproduced:",
          {k: round(v, 6) for k, v in roots.items()},
          f"B={B:.6f} K_inf={dyn['K_inf']:.6f}")

    # Accounting identity: shares sum to zero change
    out = solve()
    sh = out["path"]["shares"]
    tot = (np.array(sh["ds1"]) + np.array(sh["ds2"])
           + np.array(sh["dsK"]) + np.array(sh["dsPi"]))
    assert np.max(np.abs(tot)) < 1e-12, "share accounting violated"

    # q = 1 + phi(i - delta) linearized: q_t = phi * i_t
    q0, i0, phi0 = out["impact"]["q"], out["impact"]["i_pp"], out["params"]["phi"]
    assert abs(q0 - phi0 * i0) < 1e-12

    # Monopoly split adds to 1
    mp = out["markup"]["monopoly"]
    assert abs(mp["labor"] + mp["compute"] + mp["profit"] - 1.0) < 1e-12

    # Edge cases solve without error
    for args in [(0.6, 0.1, 0.0, 0.1, 0.1, 0.5, 10.0),
                 (4.0, 0.8, 5.0, 0.5, 0.45, 50.0, 20.0),
                 (1.0, 0.5, 1.0, 0.4, 0.4, 8.0, 1.0)]:
        r = solve(*args)
        assert "error" in r or np.isfinite(r["derived"]["lambda"])
    print("ALL MARKET-POWER MODEL CHECKS PASSED")
