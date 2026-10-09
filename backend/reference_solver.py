"""Steady-state solver for the AI extension of Moll--Rachel--Restrepo.

The financial side is the extended model in *Uneven Growth*: a fraction ``chi``
can hold risky business capital, all other households hold only a zero-net-supply
safe bond, wealth is reset at Poisson rate ``p``, and the risky portfolio share is
the closed-form Merton rule.  The production side keeps their task structure,
allows occupation-specific wage markdowns, and activates their constant product
markup.  One risky capital stock is allocated between fully automated tasks and
fixed-proportions compute used alongside workers.  Labor-markdown rents and product-
markup profits accrue to the same consolidated risky equity claim.

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
    augmentation_compute_requirement: np.ndarray
    augmentation_gain: np.ndarray
    psi: np.ndarray
    wage_markdown: np.ndarray
    product_markup: float
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
    capital_rental: float
    risky_share: float
    investor_effective_wealth: float
    household_effective_wealth: float
    wages: np.ndarray
    marginal_products_labor: np.ndarray
    labor_bill: float
    markdown_profit: float
    product_profit: float
    total_profit: float
    automation_capital: float
    augmentation_capital: float
    automation_capital_share: float
    augmentation_capital_share: float
    physical_capital_share: float
    ordinary_capital_share: float
    product_profit_share: float
    risky_payout_share: float
    inverse_tail_households: float
    inverse_tail_investors: float
    inverse_left_tail_investors: float
    augmentation_adoption_slack: np.ndarray
    statistics: Dict[str, float]
    residuals: Dict[str, float]


def validate(p: Parameters) -> None:
    n = len(p.masses)
    if n < 1 or any(
        len(x) != n
        for x in (
            p.eta,
            p.alpha,
            p.augmentation_compute_requirement,
            p.augmentation_gain,
            p.psi,
            p.wage_markdown,
        )
    ):
        raise ValueError("Occupation arrays must have the same positive length")
    if np.any(p.masses <= 0) or not np.isclose(p.masses.sum(), 1.0):
        raise ValueError("Occupation masses must be positive and sum to one")
    if np.any(p.eta <= 0) or not np.isclose(p.eta.sum(), 1.0):
        raise ValueError("Task weights must be positive and sum to one")
    if np.any(p.alpha <= 0) or np.any(p.alpha >= 1):
        raise ValueError("Automation shares must lie strictly between zero and one")
    if np.any(p.augmentation_compute_requirement < 0):
        raise ValueError("Augmentation compute requirements must be nonnegative")
    if np.any(p.augmentation_gain < 1):
        raise ValueError("Augmentation productivity gains must be at least one")
    if np.any(p.psi <= 0):
        raise ValueError("Labor productivity must be positive")
    if np.any(p.wage_markdown < 0) or np.any(p.wage_markdown >= 1):
        raise ValueError("Wage markdowns must lie in [0,1)")
    if p.product_markup < 1:
        raise ValueError("The product markup must be at least one")
    if not (0 < p.chi < 1 and p.nu > 0 and p.gamma > 0):
        raise ValueError("Risky participation and return-risk parameters are invalid")
    if min(p.sigma, p.dissipation, p.leverage_limit, p.productivity) <= 0:
        raise ValueError("Preference, dissipation, leverage, and productivity values must be positive")
    if p.rho + (p.sigma - 1) * p.growth <= 0:
        raise ValueError("The finite-human-wealth condition is violated")


def production_shares(p: Parameters, e: Equilibrium) -> Dict[str, np.ndarray | float]:
    """Revenue shares implied by the solved fixed-proportions technology.

    MRR automation contributes the constant task share ``sum eta_j alpha_j``.
    Augmentation capital is instead a quantity requirement.  Its payment share is
    endogenous because the rental price is determined in general equilibrium.
    """
    remaining_task = p.eta * (1.0 - p.alpha)
    labor_task = (
        p.product_markup
        * p.masses
        * e.marginal_products_labor
        / e.output
    )
    return {
        "remaining_task": remaining_task,
        "labor_task": labor_task,
        "automation_capital": e.automation_capital_share,
        "augmentation_capital": e.augmentation_capital_share,
        "physical_capital": e.physical_capital_share,
        "ordinary_capital": e.ordinary_capital_share,
        "labor": e.labor_bill / e.output,
        "markdown_profit": e.markdown_profit / e.output,
        "product_profit": e.product_profit_share,
        "total_profit": e.total_profit / e.output,
        "risky_payout": e.risky_payout_share,
    }


def _production_at_capital(p: Parameters, capital: float) -> Dict[str, np.ndarray | float]:
    """Production and payouts for a candidate aggregate capital stock.

    On MRR automated tasks, one unit of task output requires one unit of capital.
    On a nonautomated task, effective labor and augmentation compute are required
    in fixed proportions.  If ``m_j`` is compute per unit of augmented task output,
    total augmentation capital is ``m_j psi_j^A L_j``.  The remainder of aggregate
    capital is allocated across automated tasks exactly as in MRR Appendix A.1.
    """
    automation_share = float(p.eta @ p.alpha)
    labor_exponents = p.eta * (1.0 - p.alpha)
    effective_psi = p.psi * p.augmentation_gain
    augmentation_capital_by_group = (
        p.augmentation_compute_requirement * effective_psi * p.masses
    )
    augmentation_capital = float(np.sum(augmentation_capital_by_group))
    automation_capital = capital - augmentation_capital
    if automation_capital <= 0:
        raise ValueError("Aggregate capital does not cover augmentation compute")

    unit = (automation_capital / automation_share) ** automation_share * float(
        np.prod((effective_psi * p.masses / labor_exponents) ** labor_exponents)
    )
    output = float(p.productivity * unit)
    inverse_markup = 1.0 / p.product_markup
    capital_rental = inverse_markup * automation_share * output / automation_capital

    # The first term is the MRR marginal revenue product of labor.  The second is
    # the extra compute bill required when one more worker operates at the
    # augmented productivity level.
    shadow_wages = (
        inverse_markup * labor_exponents * output / p.masses
        - capital_rental * p.augmentation_compute_requirement * effective_psi
    )
    if np.any(shadow_wages <= 0):
        raise ValueError("Augmentation compute makes a labor shadow wage nonpositive")
    wages = (1.0 - p.wage_markdown) * shadow_wages
    labor_bill = float(p.masses @ wages)
    markdown_profit = float(p.masses @ (p.wage_markdown * shadow_wages))
    product_profit = float((1.0 - inverse_markup) * output)
    total_profit = markdown_profit + product_profit
    ordinary_capital_income = capital_rental * capital

    augmentation_capital_share = float(
        p.product_markup * capital_rental * augmentation_capital / output
    )
    physical_capital_share = automation_share + augmentation_capital_share
    ordinary_capital_share = inverse_markup * physical_capital_share
    risky_payout = ordinary_capital_income + total_profit
    risky_payout_share = risky_payout / output
    r_risky = risky_payout / capital - p.delta

    baseline_psi = p.psi
    productivity_saving = (
        shadow_wages / baseline_psi * (1.0 - 1.0 / p.augmentation_gain)
    )
    compute_cost = capital_rental * p.augmentation_compute_requirement
    adoption_slack = productivity_saving - compute_cost
    inactive = np.isclose(p.augmentation_gain, 1.0) & np.isclose(
        p.augmentation_compute_requirement, 0.0
    )
    adoption_slack = np.where(inactive, np.nan, adoption_slack)

    factor_residual = output - (
        labor_bill + markdown_profit + product_profit + ordinary_capital_income
    )
    if abs(factor_residual) > 5e-10 * max(1.0, output):
        raise AssertionError("Production payments do not exhaust output")
    return {
        "output": output,
        "capital": capital,
        "automation_capital": automation_capital,
        "augmentation_capital": augmentation_capital,
        "capital_rental": capital_rental,
        "r_risky": r_risky,
        "shadow_wages": shadow_wages,
        "wages": wages,
        "labor_bill": labor_bill,
        "markdown_profit": markdown_profit,
        "product_profit": product_profit,
        "total_profit": total_profit,
        "ordinary_capital_income": ordinary_capital_income,
        "automation_capital_share": automation_share,
        "augmentation_capital_share": augmentation_capital_share,
        "physical_capital_share": physical_capital_share,
        "ordinary_capital_share": ordinary_capital_share,
        "product_profit_share": 1.0 - inverse_markup,
        "risky_payout_share": risky_payout_share,
        "adoption_slack": adoption_slack,
        "factor_residual": factor_residual,
    }


def _state_from_unknowns(u: np.ndarray, p: Parameters) -> Dict[str, float]:
    capital = float(np.exp(np.clip(u[0], -30, 20)))
    bonds_h = float(np.exp(np.clip(u[1], -30, 20)))
    r_safe = float(p.growth + np.exp(np.clip(u[2], -30, 5)))
    try:
        levels = _production_at_capital(p, capital)
    except (ValueError, FloatingPointError):
        return {
            "capital": capital, "bonds_h": bonds_h, "r_safe": r_safe,
            "invalid": 1.0,
        }
    r_risky = float(levels["r_risky"])
    risky_share = float(
        np.clip((r_risky - r_safe) / (p.gamma * p.nu**2), 0.0, p.leverage_limit)
    )
    human_wealth = float(levels["labor_bill"]) / (r_safe - p.growth)
    x_i_level = capital - bonds_h + p.chi * human_wealth
    x_h_level = bonds_h + (1.0 - p.chi) * human_wealth
    k = capital / human_wealth
    b_h = bonds_h / human_wealth
    x_i = x_i_level / human_wealth
    x_h = x_h_level / human_wealth
    r_portfolio = risky_share * r_risky + (1.0 - risky_share) * r_safe
    r_wealth = r_portfolio + 0.5 * (p.sigma - 1.0) * p.gamma * p.nu**2 * risky_share**2
    return {
        "invalid": 0.0,
        "capital": capital,
        "bonds_h": bonds_h,
        "human_wealth": human_wealth,
        "x_i_level": x_i_level,
        "x_h_level": x_h_level,
        "k": k,
        "b_h": b_h,
        "r_safe": r_safe,
        "r_risky": r_risky,
        "risky_share": risky_share,
        "x_i": x_i,
        "x_h": x_h,
        "r_portfolio": r_portfolio,
        "r_wealth": r_wealth,
        "levels": levels,
    }


def _equilibrium_residuals(u: np.ndarray, p: Parameters) -> tuple[np.ndarray, Dict[str, float]]:
    s = _state_from_unknowns(u, p)
    if s.get("invalid", 0.0) or s["x_i_level"] <= 0 or s["r_risky"] <= p.growth:
        return np.full(3, 1e3), s
    investor_saving = (
        ((s["r_wealth"] - p.rho) / p.sigma - p.growth) * s["x_i_level"]
    )
    household_saving = (
        ((s["r_safe"] - p.rho) / p.sigma - p.growth) * s["x_h_level"]
    )
    residual = np.array(
        [
            investor_saving - p.dissipation * (s["capital"] - s["bonds_h"]),
            s["risky_share"] * s["x_i_level"] - s["capital"],
            household_saving - p.dissipation * s["bonds_h"],
        ]
    )
    return residual, s


def _solve_three_equations(p: Parameters) -> tuple[np.ndarray, Dict[str, float]]:
    # The original MATLAB routine solves these same three equations.  Logs enforce
    # positive capital, household bond holdings, and r_B-g.
    starts = (
        np.array([3.0, 2.0, 0.034]),
        np.array([5.0, 3.0, 0.040]),
        np.array([2.0, 1.0, 0.030]),
        np.array([8.0, 5.0, 0.050]),
        np.array([1.0, 0.5, 0.025]),
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
                    not candidate_state.get("invalid", 0.0)
                    and candidate_state["x_i_level"] > 0
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
            "markdown_profit", "product_profit", "profit", "equity_income",
            "bond_income", "capital_income", "total",
        )
    }
    markdown_profit_yield = e.markdown_profit / e.capital
    product_profit_yield = e.product_profit / e.capital
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
            product_profit = product_profit_yield * equity
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
                "product_profit": product_profit,
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
    levels = state["levels"]
    inv_h, inv_p, inv_n = _tail_parameters(p, state)

    # Consumption rates from the closed-form household policies.  Dissipation
    # consumes the economy's financial wealth; including pK closes resources.
    c_i_rate = (
        p.rho
        + (p.sigma - 1.0) * state["r_portfolio"]
        - 0.5 * (p.sigma - 1.0) * p.gamma * p.nu**2 * state["risky_share"] ** 2
    ) / p.sigma
    c_h_rate = (p.rho + (p.sigma - 1.0) * state["r_safe"]) / p.sigma
    human = state["human_wealth"]
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
        + float(levels["product_profit"])
        + float(levels["ordinary_capital_income"])
    )
    normalized_capital_residual = float(levels["capital"]) / human - state["k"]

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
        capital_rental=float(levels["capital_rental"]),
        risky_share=state["risky_share"],
        investor_effective_wealth=state["x_i"],
        household_effective_wealth=state["x_h"],
        wages=np.asarray(levels["wages"]),
        marginal_products_labor=np.asarray(levels["shadow_wages"]),
        labor_bill=float(levels["labor_bill"]),
        markdown_profit=float(levels["markdown_profit"]),
        product_profit=float(levels["product_profit"]),
        total_profit=float(levels["total_profit"]),
        automation_capital=float(levels["automation_capital"]),
        augmentation_capital=float(levels["augmentation_capital"]),
        automation_capital_share=float(levels["automation_capital_share"]),
        augmentation_capital_share=float(levels["augmentation_capital_share"]),
        physical_capital_share=float(levels["physical_capital_share"]),
        ordinary_capital_share=float(levels["ordinary_capital_share"]),
        product_profit_share=float(levels["product_profit_share"]),
        risky_payout_share=float(levels["risky_payout_share"]),
        inverse_tail_households=inv_h,
        inverse_tail_investors=inv_p,
        inverse_left_tail_investors=inv_n,
        augmentation_adoption_slack=np.asarray(levels["adoption_slack"]),
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
    alpha = e.automation_capital_share
    adjusted = p.productivity * (target_output / e.output) ** (1.0 - alpha)
    return replace(p, productivity=float(adjusted))
