"""Steady-state solver for the AI extension of Moll--Rachel--Restrepo.

The financial side is the extended model in *Uneven Growth*: a fraction ``chi``
can hold risky business capital, all other households hold only a zero-net-supply
safe bond, wealth is reset at Poisson rate ``p``, and the risky portfolio share is
the closed-form Merton rule.  The production side keeps their task structure but
allows occupation-specific wage markdowns.  Labor-markdown rents and other business
profits accrue to the same consolidated risky equity claim.

There is deliberately no Aiyagari asset grid and no transition routine here.  The
stationary wealth distributions are the exact Pareto and double-Pareto solutions of
the Kolmogorov forward equations in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class Parameters:
    rho: float
    sigma: float
    gamma: float
    nu: float
    chi: float
    dissipation: float
    growth: float
    delta: float
    leverage_limit: float
    productivity: float
    masses: np.ndarray
    eta: np.ndarray
    alpha: np.ndarray
    psi: np.ndarray
    wage_markdown: np.ndarray
    other_profit_share: float
    quadrature_size: int = 20_000


@dataclass(frozen=True)
class Equilibrium:
    output: float
    capital: float
    normalized_capital: float
    safe_bonds_households: float
    safe_bonds_investors: float
    r_safe: float
    r_risky: float
    r_portfolio: float
    r_wealth: float
    risky_share: float
    investor_effective_wealth: float
    household_effective_wealth: float
    wages: np.ndarray
    marginal_products_labor: np.ndarray
    labor_bill: float
    markdown_profit: float
    other_profit: float
    total_profit: float
    physical_capital_share: float
    ordinary_capital_share: float
    other_profit_share: float
    risky_payout_share: float
    inverse_tail_households: float
    inverse_tail_investors: float
    inverse_left_tail_investors: float
    statistics: Dict[str, float]
    residuals: Dict[str, float]


def validate(p: Parameters) -> None:
    n = len(p.masses)
    if n < 1 or any(len(x) != n for x in (p.eta, p.alpha, p.psi, p.wage_markdown)):
        raise ValueError("Occupation arrays must have the same positive length")
    if np.any(p.masses <= 0) or not np.isclose(p.masses.sum(), 1.0):
        raise ValueError("Occupation masses must be positive and sum to one")
    if np.any(p.eta <= 0) or not np.isclose(p.eta.sum(), 1.0):
        raise ValueError("Task weights must be positive and sum to one")
    if np.any(p.alpha <= 0) or np.any(p.alpha >= 1):
        raise ValueError("Automation shares must lie strictly between zero and one")
    if np.any(p.psi <= 0):
        raise ValueError("Labor productivity must be positive")
    if np.any(p.wage_markdown < 0) or np.any(p.wage_markdown >= 1):
        raise ValueError("Wage markdowns must lie in [0,1)")
    if p.other_profit_share < 0:
        raise ValueError("The other-business-profit share cannot be negative")
    if not (0 < p.chi < 1 and p.nu > 0 and p.gamma > 0):
        raise ValueError("Risky participation and return-risk parameters are invalid")
    if min(p.sigma, p.dissipation, p.leverage_limit, p.productivity) <= 0:
        raise ValueError("Preference, dissipation, leverage, and productivity values must be positive")
    if p.rho + (p.sigma - 1) * p.growth <= 0:
        raise ValueError("The finite-human-wealth condition is violated")


def production_shares(p: Parameters) -> Dict[str, np.ndarray | float]:
    labor_task = p.eta * (1.0 - p.alpha)
    physical = float(p.eta @ p.alpha)
    labor = float(np.sum((1.0 - p.wage_markdown) * labor_task))
    markdown_profit = float(np.sum(p.wage_markdown * labor_task))
    other_profit = float(p.other_profit_share)
    ordinary_capital = physical - other_profit
    if ordinary_capital < -1e-12:
        raise ValueError("Other business profits cannot exceed the capital-product share")
    total_profit = markdown_profit + other_profit
    risky = physical + markdown_profit
    if not np.isclose(labor + ordinary_capital + total_profit, 1.0, atol=1e-12):
        raise AssertionError("Task shares do not exhaust output")
    return {
        "labor_task": labor_task,
        "physical_capital": physical,
        "ordinary_capital": ordinary_capital,
        "labor": labor,
        "markdown_profit": markdown_profit,
        "other_profit": other_profit,
        "total_profit": total_profit,
        "risky_payout": risky,
    }


def _state_from_unknowns(u: np.ndarray, p: Parameters) -> Dict[str, float]:
    shares = production_shares(p)
    risky_payout = float(shares["risky_payout"])
    k = float(np.exp(np.clip(u[0], -30, 20)))
    b_h = float(np.exp(np.clip(u[1], -30, 20)))
    r_safe = float(p.growth + np.exp(np.clip(u[2], -30, 5)))
    r_risky = risky_payout / (1.0 - risky_payout) * (r_safe - p.growth) / k - p.delta
    risky_share = float(
        np.clip((r_risky - r_safe) / (p.gamma * p.nu**2), 0.0, p.leverage_limit)
    )
    x_i = k - b_h + p.chi
    x_h = b_h + 1.0 - p.chi
    r_portfolio = risky_share * r_risky + (1.0 - risky_share) * r_safe
    r_wealth = r_portfolio + 0.5 * (p.sigma - 1.0) * p.gamma * p.nu**2 * risky_share**2
    return {
        "k": k,
        "b_h": b_h,
        "r_safe": r_safe,
        "r_risky": r_risky,
        "risky_share": risky_share,
        "x_i": x_i,
        "x_h": x_h,
        "r_portfolio": r_portfolio,
        "r_wealth": r_wealth,
    }


def _equilibrium_residuals(u: np.ndarray, p: Parameters) -> tuple[np.ndarray, Dict[str, float]]:
    s = _state_from_unknowns(u, p)
    if s["x_i"] <= 0 or s["r_risky"] <= p.growth or s["r_safe"] <= p.growth:
        return np.full(3, 1e3), s
    investor_saving = ((s["r_wealth"] - p.rho) / p.sigma - p.growth) * s["x_i"]
    household_saving = ((s["r_safe"] - p.rho) / p.sigma - p.growth) * s["x_h"]
    residual = np.array(
        [
            investor_saving - p.dissipation * (s["k"] - s["b_h"]),
            s["risky_share"] * s["x_i"] - s["k"],
            household_saving - p.dissipation * s["b_h"],
        ]
    )
    return residual, s


def _solve_three_equations(p: Parameters) -> tuple[np.ndarray, Dict[str, float]]:
    # The original MATLAB routine solves these same three equations.  Logs enforce
    # positive capital, household bond holdings, and r_B-g.
    starts = (
        np.array([0.13, 0.05, 0.03]),
        np.array([0.28, 0.12, 0.04]),
        np.array([0.35, 0.08, 0.04]),
        np.array([0.45, 0.05, 0.05]),
        np.array([max(0.15, 0.8 * p.chi), max(0.015, 0.12 * (1.0 - p.chi)), 0.04]),
    )
    last_residual = np.full(3, np.nan)
    for start in starts:
        u = np.log(start)
        for _ in range(200):
            residual, state = _equilibrium_residuals(u, p)
            last_residual = residual
            norm = float(np.max(np.abs(residual)))
            if norm < 2e-12:
                return u, state
            h = 2e-5
            jac = np.empty((3, 3))
            for j in range(3):
                shifted = u.copy()
                shifted[j] += h
                shifted_residual, _ = _equilibrium_residuals(shifted, p)
                jac[:, j] = (shifted_residual - residual) / h
            try:
                direction = np.linalg.solve(jac, -residual)
            except np.linalg.LinAlgError:
                direction = -np.linalg.pinv(jac) @ residual
            direction = np.clip(direction, -2.0, 2.0)
            accepted = False
            for damping in (1.0, 0.5, 0.25, 0.1, 0.04, 0.01, 0.002):
                candidate = u + damping * direction
                candidate_residual, candidate_state = _equilibrium_residuals(candidate, p)
                if (
                    candidate_state["x_i"] > 0
                    and candidate_state["risky_share"] > 0
                    and np.max(np.abs(candidate_residual)) < norm
                ):
                    u = candidate
                    accepted = True
                    break
            if not accepted:
                break
    raise RuntimeError(
        "Moll--Rachel--Restrepo equilibrium did not converge from multiple starts; "
        f"last residual={last_residual}"
    )


def _production_levels(p: Parameters, state: Dict[str, float]) -> Dict[str, np.ndarray | float]:
    shares = production_shares(p)
    alpha = float(shares["physical_capital"])
    risky_payout = float(shares["risky_payout"])
    labor_task = np.asarray(shares["labor_task"])
    capital_output = risky_payout / (state["r_risky"] + p.delta)
    unit = (capital_output / alpha) ** alpha * float(
        np.prod((p.psi * p.masses / labor_task) ** labor_task)
    )
    output = float((p.productivity * unit) ** (1.0 / (1.0 - alpha)))
    capital = capital_output * output
    marginal_products = labor_task * output / p.masses
    wages = (1.0 - p.wage_markdown) * marginal_products
    labor_bill = float(p.masses @ wages)
    markdown_profit = float(shares["markdown_profit"]) * output
    other_profit = float(shares["other_profit"]) * output
    total_profit = float(shares["total_profit"]) * output
    ordinary_capital_income = float(shares["ordinary_capital"]) * output
    human_wealth = labor_bill / (state["r_safe"] - p.growth)
    normalized_capital = capital / human_wealth
    return {
        "output": output,
        "capital": capital,
        "capital_output": capital_output,
        "marginal_products": marginal_products,
        "wages": wages,
        "labor_bill": labor_bill,
        "markdown_profit": markdown_profit,
        "other_profit": other_profit,
        "total_profit": total_profit,
        "ordinary_capital_income": ordinary_capital_income,
        "human_wealth": human_wealth,
        "normalized_capital": normalized_capital,
    }


def _tail_parameters(p: Parameters, state: Dict[str, float]) -> tuple[float, float, float]:
    inv_h = (state["r_safe"] - p.rho - p.sigma * p.growth) / (p.dissipation * p.sigma)
    drift = (
        state["r_wealth"]
        - p.rho
        - p.sigma * p.growth
        - 0.5 * p.sigma * state["risky_share"] ** 2 * p.nu**2
    )
    variance = state["risky_share"] ** 2 * p.nu**2
    root = np.sqrt(drift**2 + 2.0 * p.sigma**2 * variance * p.dissipation)
    inv_p = (drift + root) / (2.0 * p.dissipation * p.sigma)
    inv_n = (drift - root) / (2.0 * p.dissipation * p.sigma)
    return float(inv_h), float(inv_p), float(inv_n)


def _quadrature_edges(n: int, extra: float) -> np.ndarray:
    """Quantile cells with logarithmic resolution in the Pareto tail."""
    body_n = max(200, n // 2)
    tail_n = max(200, n - body_n)
    body = np.linspace(0.0, 0.99, body_n + 1)
    survival = np.geomspace(0.01, 1e-12, tail_n + 1)
    tail = 1.0 - survival
    return np.unique(np.r_[body, tail, extra, 1.0])


def _mean_household_wealth(edges: np.ndarray, inv_tail: float) -> np.ndarray:
    left, right = edges[:-1], edges[1:]
    integral = ((1.0 - left) ** (1.0 - inv_tail) - (1.0 - right) ** (1.0 - inv_tail)) / (
        1.0 - inv_tail
    )
    return integral / (right - left)


def _mean_investor_wealth(edges: np.ndarray, inv_p: float, inv_n: float) -> np.ndarray:
    denominator = inv_p - inv_n
    mass_low = -inv_n / denominator
    mass_high = inv_p / denominator
    left, right = edges[:-1], edges[1:]
    result = np.empty_like(left)
    low = right <= mass_low + 1e-15
    high = left >= mass_low - 1e-15
    power = -inv_n
    low_integral = mass_low ** (-power) * (
        right[low] ** (power + 1.0) - left[low] ** (power + 1.0)
    ) / (power + 1.0)
    result[low] = low_integral / (right[low] - left[low])
    high_integral = mass_high**inv_p * (
        (1.0 - left[high]) ** (1.0 - inv_p)
        - (1.0 - right[high]) ** (1.0 - inv_p)
    ) / (1.0 - inv_p)
    result[high] = high_integral / (right[high] - left[high])
    if np.any(~(low | high)):
        raise AssertionError("Investor quadrature must split at the double-Pareto kink")
    return result


def stationary_atoms(p: Parameters, e: Equilibrium, quadrature_size: int | None = None) -> Dict[str, np.ndarray]:
    """Deterministic atoms for exact stationary distributions.

    Midpoint inverse-CDF quadrature introduces no simulation noise.  It is used only
    for mixed-occupation Ginis and percentile decompositions; aggregate equilibrium
    conditions and Pareto tail indices are analytical.
    """
    n_requested = int(p.quadrature_size if quadrature_size is None else quadrature_size)
    mass_low = -e.inverse_left_tail_investors / (
        e.inverse_tail_investors - e.inverse_left_tail_investors
    )
    edges = _quadrature_edges(n_requested, mass_low)
    cell_weight = np.diff(edges)
    x_h = _mean_household_wealth(edges, e.inverse_tail_households)
    x_i = _mean_investor_wealth(
        edges, e.inverse_tail_investors, e.inverse_left_tail_investors
    )
    n = len(cell_weight)
    arrays: Dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "weights", "occupation", "investor", "normalized_effective_wealth",
            "effective_wealth", "net_worth", "equity", "bonds", "labor",
            "markdown_profit", "other_profit", "profit", "equity_income",
            "bond_income", "capital_income", "total",
        )
    }
    markdown_profit_yield = e.markdown_profit / e.capital
    other_profit_yield = e.other_profit / e.capital
    profit_yield = e.total_profit / e.capital
    for j, mass in enumerate(p.masses):
        human = e.wages[j] / (e.r_safe - p.growth)
        for is_investor, class_mass, x_norm in (
            (False, mass * (1.0 - p.chi), x_h),
            (True, mass * p.chi, x_i),
        ):
            effective = human * x_norm
            net_worth = effective - human
            equity = e.risky_share * effective if is_investor else np.zeros(n)
            bonds = net_worth - equity
            labor = np.full(n, e.wages[j])
            equity_income = e.r_risky * equity
            bond_income = e.r_safe * bonds
            markdown_profit = markdown_profit_yield * equity
            other_profit = other_profit_yield * equity
            profit = profit_yield * equity
            capital_income = equity_income + bond_income
            total = labor + capital_income
            values = {
                "weights": class_mass * cell_weight,
                "occupation": np.full(n, j, dtype=float),
                "investor": np.full(n, float(is_investor)),
                "normalized_effective_wealth": x_norm,
                "effective_wealth": effective,
                "net_worth": net_worth,
                "equity": equity,
                "bonds": bonds,
                "labor": labor,
                "markdown_profit": markdown_profit,
                "other_profit": other_profit,
                "profit": profit,
                "equity_income": equity_income,
                "bond_income": bond_income,
                "capital_income": capital_income,
                "total": total,
            }
            for name, value in values.items():
                arrays[name].append(np.asarray(value))
    return {name: np.concatenate(parts) for name, parts in arrays.items()}


def weighted_gini(values: np.ndarray, weights: np.ndarray) -> float:
    """Generalized weighted Gini; defined when aggregate income is positive.

    Capital income can be negative for leveraged investors, as in the paper.  In
    that case the generalized Gini may exceed one and should be read as a
    concentration index rather than as a number bounded by zero and one.
    """
    values, weights = np.asarray(values), np.asarray(weights)
    keep = weights > 0
    values, weights = values[keep], weights[keep]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    total_weight = float(weights.sum())
    total_value = float(np.sum(values * weights))
    if total_value <= 0:
        return float("nan")
    cum_before = np.cumsum(weights) - weights
    return float(
        2.0 * np.sum(values * weights * (cum_before + 0.5 * weights))
        / (total_weight * total_value)
        - 1.0
    )


def top_share(ranking: np.ndarray, component: np.ndarray, weights: np.ndarray, fraction: float) -> float:
    order = np.argsort(ranking)[::-1]
    ranking = np.asarray(ranking)[order]
    component = np.asarray(component)[order]
    weights = np.asarray(weights)[order]
    remaining = fraction * float(weights.sum())
    numerator = 0.0
    for value, mass in zip(component, weights):
        if remaining <= 0:
            break
        used = min(float(mass), remaining)
        numerator += float(value) * used
        remaining -= used
    denominator = float(np.sum(component * weights))
    return numerator / denominator if denominator > 0 else float("nan")


def quantile_bin_means(
    ranking: np.ndarray,
    components: Dict[str, np.ndarray],
    weights: np.ndarray,
    edges: np.ndarray,
) -> Dict[str, np.ndarray]:
    order = np.argsort(ranking)
    weights = np.asarray(weights)[order]
    ordered = {name: np.asarray(values)[order] for name, values in components.items()}
    total_mass = float(weights.sum())
    result = {name: np.zeros(len(edges) - 1) for name in components}
    bin_mass = np.diff(edges) * total_mass
    cumulative = 0.0
    b = 0
    for i, mass in enumerate(weights):
        left, right = cumulative / total_mass, (cumulative + mass) / total_mass
        while b < len(bin_mass) and left < edges[-1]:
            overlap = max(0.0, min(right, edges[b + 1]) - max(left, edges[b]))
            if overlap > 0:
                atom_mass = overlap * total_mass
                for name in result:
                    result[name][b] += ordered[name][i] * atom_mass
            if right <= edges[b + 1] + 1e-15:
                break
            b += 1
        cumulative += mass
    for name in result:
        result[name] /= bin_mass
    return result


def solve_equilibrium(p: Parameters, compute_distribution: bool = True) -> Equilibrium:
    validate(p)
    u, state = _solve_three_equations(p)
    equation_residual, state = _equilibrium_residuals(u, p)
    levels = _production_levels(p, state)
    shares = production_shares(p)
    inv_h, inv_p, inv_n = _tail_parameters(p, state)

    # Consumption rates from the closed-form household policies.  Dissipation
    # consumes the economy's financial wealth; including pK closes resources.
    c_i_rate = (
        p.rho
        + (p.sigma - 1.0) * state["r_portfolio"]
        - 0.5 * (p.sigma - 1.0) * p.gamma * p.nu**2 * state["risky_share"] ** 2
    ) / p.sigma
    c_h_rate = (p.rho + (p.sigma - 1.0) * state["r_safe"]) / p.sigma
    human = float(levels["human_wealth"])
    regular_consumption = human * (c_i_rate * state["x_i"] + c_h_rate * state["x_h"])
    dissipation_consumption = p.dissipation * float(levels["capital"])
    resource_residual = (
        float(levels["output"])
        - (p.growth + p.delta) * float(levels["capital"])
        - regular_consumption
        - dissipation_consumption
    )
    factor_residual = float(levels["output"]) - (
        float(levels["labor_bill"])
        + float(levels["markdown_profit"])
        + float(levels["other_profit"])
        + float(levels["ordinary_capital_income"])
    )
    normalized_capital_residual = float(levels["normalized_capital"]) - state["k"]

    empty_stats: Dict[str, float] = {
        "labor_income_gini": float("nan"),
        "profit_income_gini": float("nan"),
        "capital_income_gini": float("nan"),
        "total_income_gini": float("nan"),
        "net_worth_gini": float("nan"),
        "top1_net_worth_share": float("nan"),
        "top10_net_worth_share": float("nan"),
        "top1_equity_share": float("nan"),
        "top10_total_income_share": float("nan"),
    }
    provisional = Equilibrium(
        output=float(levels["output"]),
        capital=float(levels["capital"]),
        normalized_capital=state["k"],
        safe_bonds_households=state["b_h"],
        safe_bonds_investors=-state["b_h"],
        r_safe=state["r_safe"],
        r_risky=state["r_risky"],
        r_portfolio=state["r_portfolio"],
        r_wealth=state["r_wealth"],
        risky_share=state["risky_share"],
        investor_effective_wealth=state["x_i"],
        household_effective_wealth=state["x_h"],
        wages=np.asarray(levels["wages"]),
        marginal_products_labor=np.asarray(levels["marginal_products"]),
        labor_bill=float(levels["labor_bill"]),
        markdown_profit=float(levels["markdown_profit"]),
        other_profit=float(levels["other_profit"]),
        total_profit=float(levels["total_profit"]),
        physical_capital_share=float(shares["physical_capital"]),
        ordinary_capital_share=float(shares["ordinary_capital"]),
        other_profit_share=float(shares["other_profit"]),
        risky_payout_share=float(shares["risky_payout"]),
        inverse_tail_households=inv_h,
        inverse_tail_investors=inv_p,
        inverse_left_tail_investors=inv_n,
        statistics=empty_stats,
        residuals={
            "investor_saving": float(equation_residual[0]),
            "risky_capital_market": float(equation_residual[1]),
            "bond_saving": float(equation_residual[2]),
            "bond_market": state["b_h"] - state["b_h"],
            "normalized_capital": normalized_capital_residual,
            "factor_exhaustion": factor_residual,
            "resources": resource_residual,
        },
    )
    if not compute_distribution:
        return provisional

    atoms = stationary_atoms(p, provisional)
    weights = atoms["weights"]
    stats = {
        "labor_income_gini": weighted_gini(atoms["labor"], weights),
        "profit_income_gini": weighted_gini(atoms["profit"], weights),
        "capital_income_gini": weighted_gini(atoms["capital_income"], weights),
        "total_income_gini": weighted_gini(atoms["total"], weights),
        "net_worth_gini": weighted_gini(atoms["net_worth"], weights),
        "top1_net_worth_share": top_share(atoms["net_worth"], atoms["net_worth"], weights, 0.01),
        "top10_net_worth_share": top_share(atoms["net_worth"], atoms["net_worth"], weights, 0.10),
        "top1_equity_share": top_share(atoms["equity"], atoms["equity"], weights, 0.01),
        "top10_total_income_share": top_share(atoms["total"], atoms["total"], weights, 0.10),
    }
    return replace(provisional, statistics=stats)


def normalize_productivity(p: Parameters, target_output: float = 1.0) -> Parameters:
    e = solve_equilibrium(p, compute_distribution=False)
    alpha = e.physical_capital_share
    adjusted = p.productivity * (target_output / e.output) ** (1.0 - alpha)
    return replace(p, productivity=float(adjusted))
