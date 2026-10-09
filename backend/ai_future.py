"""Website controls and appendix scenarios for the reviewed AI model.

The backend vendors ai_mrr_reviewed/solver.py and technology.py. Augmentation
combines workers with compute in fixed proportions. Firms compare capital
costs with the shadow value of labor when choosing tasks and techniques. With
fixed occupation employment, wage markdowns redistribute labor income to owners
and affect capital accumulation through the financial equilibrium. The reference
solver reproduces the baseline calibration. Existing fields remain available.

Automation/augmentation questions tilt task opportunities d_j/e_j at a fixed
production-weighted mean. Advanced controls set lambda_A, lambda_P, c_P, xi,
the task tier, and optional exact participation chi. Employer-power answers
apply additional multipliers to the wage markdown after the xi sharing rule.
Named presets reproduce the five appendix scenarios.
"""

from __future__ import annotations

import csv
import os
from dataclasses import replace
from functools import lru_cache
from typing import Dict

import numpy as np

from mrr_solver import (
    Equilibrium,
    Parameters,
    _production_at_capital,
    production_shares,
    solve_equilibrium,
    stationary_atoms,
    weighted_gini,
)

from reference_solver import normalize_productivity, solve_equilibrium as solve_baseline_equilibrium

HERE = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_FILE = os.path.join(HERE, "data", "ai_occupation_calibration.csv")

# ---------------------------------------------------------------------------
# Calibration (identical to ai_mrr_reviewed/run_scenarios.py)
# ---------------------------------------------------------------------------
MRR = {
    "rho": 0.01, "sigma": 2.0, "gamma": 2.0, "nu": 0.077, "chi": 0.066,
    "dissipation": 0.045, "growth": 0.015, "delta": 0.05, "leverage_limit": 2.0,
}
LABOR_SHARE_TARGET = 0.66
MARKDOWN_PROFIT_SHARE_TARGET = 0.078
PRODUCT_MARKUP_PROFIT_SHARE_TARGET = 0.030
PRODUCT_MARKUP = 1.0 / (1.0 - PRODUCT_MARKUP_PROFIT_SHARE_TARGET)
BASELINE_ORDINARY_CAPITAL_SHARE = (
    1.0 - LABOR_SHARE_TARGET - MARKDOWN_PROFIT_SHARE_TARGET - PRODUCT_MARKUP_PROFIT_SHARE_TARGET
)
# With no pre-AI augmentation capital, alpha/mu is ordinary capital income.
BASELINE_AUTOMATION_SHARE = PRODUCT_MARKUP * BASELINE_ORDINARY_CAPITAL_SHARE
AUTOMATED_TASK_COST_GAP = 1.30
# Compute requirement per augmented task output: m_j = c_P * s_j.
DEFAULT_AUGMENTATION_COST = 0.25
SEEGMILLER_INVERSE_MARKDOWN = np.r_[np.full(4, 1.14), np.full(4, 1.16), np.full(2, 1.23)]

# Exposure tier used for the occupation profiles (T3--T4 automation).
TIER = "modest"
TIERS = ("limited", "modest", "broad")
DEFAULT_WAGE_SHARING = 1.0
# Keep gains finite and task frontiers interior at full shock intensity.
PROFILE_MAX = 1.0 - 1e-8
DEFAULT_LAMBDA_A = 0.5
DEFAULT_LAMBDA_P = 0.5

# Quadrature for distributional statistics. 6,000 cells per type reproduces the
# 20,000-cell statistics of the paper code to about 1e-4 at a quarter of the cost.
QUADRATURE = 6000

# ---------------------------------------------------------------------------
# Belief scales: answer index -2..2 -> parameter value
# ---------------------------------------------------------------------------
TILT = {-2: -0.9, -1: -0.45, 0: 0.0, 1: 0.45, 2: 0.9}
CHI = {-2: 0.035, -1: 0.05, 0: 0.066, 1: 0.15, 2: 0.30}
MARKDOWN_MULTIPLIER = {-2: 0.25, -1: 0.6, 0: 1.0, 1: 1.5, 2: 2.0}

QUESTIONS = [
    {
        "id": "auto",
        "title": "Which jobs will AI automate more?",
        "text": "Which jobs have more tasks that AI could carry out? Firms choose whether to automate at equilibrium costs.",
        "has_data_option": True,
        "options": [
            "Mostly low-wage jobs", "Somewhat more low-wage jobs", "Evenly across jobs",
            "Somewhat more high-wage jobs", "Mostly high-wage jobs",
        ],
        "maps_to": "Automation opportunity d_j; its production-weighted mean is held at the task-data value",
    },
    {
        "id": "aug",
        "title": "Which workers will AI make more productive?",
        "text": "Think of tasks that stay with people but get done faster or better with AI.",
        "has_data_option": True,
        "options": [
            "Mostly low-wage workers", "Somewhat more low-wage workers", "Evenly across workers",
            "Somewhat more high-wage workers", "Mostly high-wage workers",
        ],
        "maps_to": "Time-saving opportunity e_j; its production-weighted mean is held at the task-data value",
    },
    {
        "id": "own",
        "title": "Who will own the companies and capital that run AI?",
        "text": "The baseline gives 6.6 percent of households access to risky business equity. Wider access changes saving and ownership concentration.",
        "has_data_option": False,
        "options": [
            "Much narrower access", "Narrower access", "Baseline access",
            "Broader ownership", "Much broader ownership",
        ],
        "values": [CHI[k] for k in range(-2, 3)],
        "maps_to": "chi: share of households with access to risky business/AI equity",
    },
    {
        "id": "mp",
        "title": "How much power will employers have over wages?",
        "text": "Employers with market power pay workers less than the value they produce. "
                "Answer separately for low-wage and high-wage workers. Wage markdowns shift income from workers to owners and can change capital accumulation.",
        "has_data_option": False,
        "options": ["Much weaker", "Weaker", "Baseline multiplier", "Stronger", "Much stronger"],
        "values": [MARKDOWN_MULTIPLIER[k] for k in range(-2, 3)],
        "maps_to": "Additional multipliers on wage markdowns after the augmentation wage-sharing rule",
    },
]


def _read_calibration() -> Dict[str, np.ndarray]:
    with open(CALIBRATION_FILE) as stream:
        rows = list(csv.DictReader(stream))
    get = lambda name: np.array([float(row[name]) for row in rows])
    return {name: get(name) for name in (
        "mass", "mean_wage", "payroll_share", "ppg",
        "auto_limited", "auto_modest", "auto_broad",
        "aug_limited", "aug_modest", "aug_broad",
    )}


def _baseline_markdowns(payroll_share: np.ndarray) -> np.ndarray:
    raw = 1.0 - 1.0 / SEEGMILLER_INVERSE_MARKDOWN
    raw_logit = np.log(raw / (1.0 - raw))
    target = 1.0 + MARKDOWN_PROFIT_SHARE_TARGET / LABOR_SHARE_TARGET
    low, high = -6.0, 2.0
    for _ in range(120):
        shift = 0.5 * (low + high)
        tau = 1.0 / (1.0 + np.exp(-(raw_logit + shift)))
        if float(np.sum(payroll_share / (1.0 - tau))) < target:
            low = shift
        else:
            high = shift
    return 1.0 / (1.0 + np.exp(-(raw_logit + 0.5 * (low + high))))


@lru_cache(maxsize=1)
def _base() -> tuple[Parameters, Dict[str, np.ndarray]]:
    data = _read_calibration()
    masses = data["mass"] / data["mass"].sum()
    markdown = _baseline_markdowns(data["payroll_share"])
    revenue = data["payroll_share"] / (1.0 - markdown)
    p = Parameters(
        **MRR,
        productivity=1.0,
        masses=masses,
        eta=revenue / revenue.sum(),
        alpha=np.full(len(masses), BASELINE_AUTOMATION_SHARE),
        augmentation_compute_requirement=np.zeros(len(masses)),
        augmentation_gain=np.ones(len(masses)),
        psi=np.ones(len(masses)),
        wage_markdown=markdown,
        product_markup=PRODUCT_MARKUP,
        quadrature_size=QUADRATURE,
    )
    p = normalize_productivity(p, 1.0)
    e = solve_baseline_equilibrium(p, compute_distribution=False)
    psi = e.wages / (AUTOMATED_TASK_COST_GAP * e.capital_rental)
    p = normalize_productivity(replace(p, psi=psi), 1.0)
    return p, data


# ---------------------------------------------------------------------------
# Beliefs -> parameters
# ---------------------------------------------------------------------------
def tilted_profile(observed: np.ndarray, eta: np.ndarray, answer) -> np.ndarray:
    """Bounded opportunity profile with the production-weighted mean preserved.

    A positive answer tilts toward high-wage occupations. For profiles that reach
    the upper bound, rescale the uncapped groups to preserve the same mean.
    """
    if answer == "data":
        return observed.copy()
    mean = float(eta @ observed)
    n = len(observed)
    rank = np.linspace(-1.0, 1.0, n)
    shape = 1.0 + TILT[int(answer)] * rank
    linear = shape * mean / float(eta @ shape)
    if np.max(linear) <= PROFILE_MAX:
        return linear
    low, high = 0.0, PROFILE_MAX / float(np.min(shape))
    for _ in range(80):
        scale = .5 * (low + high)
        if float(eta @ np.minimum(scale * shape, PROFILE_MAX)) < mean:
            low = scale
        else:
            high = scale
    return np.minimum(.5 * (low + high) * shape, PROFILE_MAX)


def markdown_schedule(baseline: np.ndarray, mp_low: int, mp_high: int) -> np.ndarray:
    m_low, m_high = MARKDOWN_MULTIPLIER[int(mp_low)], MARKDOWN_MULTIPLIER[int(mp_high)]
    multiplier = np.r_[np.full(4, m_low), np.full(4, np.sqrt(m_low * m_high)), np.full(2, m_high)]
    return np.clip(baseline * multiplier, 0.0, 0.95)


def scenario_parameters(auto, aug, own, mp_low, mp_high, lam_a, lam_p,
                        c_p=DEFAULT_AUGMENTATION_COST, tier=TIER,
                        xi=DEFAULT_WAGE_SHARING, chi=None):
    base, data = _base()
    if tier not in TIERS:
        raise ValueError("task tier must be limited, modest, or broad")
    if not all(np.isfinite(v) for v in (lam_a, lam_p, c_p, xi)):
        raise ValueError("scenario parameters must be finite")
    if not (0 <= lam_a <= 1 and 0 <= lam_p <= 1 and 0 <= c_p <= .9 and 0 <= xi <= 1):
        raise ValueError("scenario parameter is outside its supported range")
    participation = CHI[int(own)] if chi is None else float(chi)
    if not np.isfinite(participation) or not (.035 <= participation <= .6):
        raise ValueError("participation must be between .035 and .6")
    a_profile = tilted_profile(data[f"auto_{tier}"], base.eta, auto)
    q_profile = tilted_profile(data[f"aug_{tier}"], base.eta, aug)
    alpha = BASELINE_AUTOMATION_SHARE + (1.0 - BASELINE_AUTOMATION_SHARE) * lam_a * a_profile
    time_saving = lam_p * q_profile
    gain = 1.0 / (1.0 - time_saving)
    sharing_markdown = 1.0 - (1.0 - base.wage_markdown) * gain ** (xi - 1.0)
    p = replace(
        base,
        chi=participation,
        alpha=alpha,
        augmentation_compute_requirement=c_p * time_saving,
        augmentation_gain=gain,
        wage_markdown=markdown_schedule(sharing_markdown, mp_low, mp_high),
    )
    return p, a_profile, q_profile, gain


# ---------------------------------------------------------------------------
# Fast distributional statistics on the analytical stationary distributions
# ---------------------------------------------------------------------------
def _cumulative(ranking, component, weights):
    order = np.argsort(ranking, kind="mergesort")
    w = weights[order]
    cm = np.r_[0.0, np.cumsum(w)]
    cv = np.r_[0.0, np.cumsum(component[order] * w)]
    return cm, cv


def _bin_means(ranking, component, weights, edges):
    cm, cv = _cumulative(ranking, component, weights)
    total = cm[-1]
    values = np.interp(np.asarray(edges) * total, cm, cv)
    return np.diff(values) / (np.diff(edges) * total)


def _top_share(ranking, component, weights, fraction):
    cm, cv = _cumulative(ranking, component, weights)
    total_mass, total_value = cm[-1], cv[-1]
    below = np.interp((1.0 - fraction) * total_mass, cm, cv)
    return float((total_value - below) / total_value)


PERCENTILE_EDGES = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .99, 1.0]
PERCENTILE_LABELS = ["P0–10", "P10–20", "P20–30", "P30–40", "P40–50", "P50–60",
                     "P60–70", "P70–80", "P80–90", "P90–99", "Top 1%"]


def _describe(p: Parameters, e: Equilibrium) -> Dict[str, object]:
    atoms = stationary_atoms(p, e)
    w = atoms["weights"]
    total, labor, cap, nw = atoms["total"], atoms["labor"], atoms["capital_income"], atoms["net_worth"]
    sh = production_shares(p, e)
    top10_cap = _top_share(total, cap, w, 0.10) * float(np.sum(cap * w))
    top10_tot = _top_share(total, total, w, 0.10) * float(np.sum(total * w))
    return {
        "output": e.output,
        "capital": e.capital,
        "capital_rental": e.capital_rental,
        "r_risky": e.r_risky,
        "r_safe": e.r_safe,
        "risky_share": e.risky_share,
        "tail_inverse": e.inverse_tail_investors,
        "chi": p.chi,
        "shares": {
            "labor": sh["labor"],
            "ordinary_capital": sh["ordinary_capital"],
            "markdown_profit": sh["markdown_profit"],
            "product_profit": sh["product_profit"],
            # Capital-service payments by use, as shares of gross output.
            "automation_capital": sh["automation_capital"] / p.product_markup,
            "augmentation_capital": sh["augmentation_capital"] / p.product_markup,
        },
        "gini": {
            "labor": weighted_gini(labor, w),
            "capital": weighted_gini(cap, w),
            "total": weighted_gini(total, w),
            "wealth": weighted_gini(nw, w),
            "equity": weighted_gini(atoms["equity"], w),
        },
        "top10_income": _top_share(total, total, w, 0.10),
        "top1_income": _top_share(total, total, w, 0.01),
        "top1_equity": _top_share(atoms["equity"], atoms["equity"], w, 0.01),
        "top10_equity": _top_share(atoms["equity"], atoms["equity"], w, 0.10),
        "top1_wealth": _top_share(nw, nw, w, 0.01),
        "top10_wealth": _top_share(nw, nw, w, 0.10),
        "top10_capital_income_share_of_income": top10_cap / top10_tot,
        "capital_income_share_of_income": float(np.sum(cap * w) / np.sum(total * w)),
        "top10_labor_concentration": _top_share(total, labor, w, .10),
        "top10_capital_concentration": _top_share(total, cap, w, .10),
        "wages": e.wages.tolist(),
        "percentiles": {
            "labor": _bin_means(total, labor, w, PERCENTILE_EDGES).tolist(),
            "capital": _bin_means(total, cap, w, PERCENTILE_EDGES).tolist(),
            "total": _bin_means(total, total, w, PERCENTILE_EDGES).tolist(),
        },
    }


@lru_cache(maxsize=1)
def _baseline_result() -> Dict[str, object]:
    base, _ = _base()
    return _describe(base, solve_equilibrium(base, compute_distribution=False))


@lru_cache(maxsize=512)
def solve(auto="data", aug="data", own=0, mp_low=0, mp_high=0,
          lam_a=DEFAULT_LAMBDA_A, lam_p=DEFAULT_LAMBDA_P,
          c_p=DEFAULT_AUGMENTATION_COST, tier=TIER,
          xi=DEFAULT_WAGE_SHARING, chi=None) -> Dict[str, object]:
    base, data = _base()
    p, a_profile, q_profile, psi_gain = scenario_parameters(
        auto, aug, own, mp_low, mp_high, lam_a, lam_p, c_p, tier, xi, chi)
    e = solve_equilibrium(p, compute_distribution=False)
    worst = max(abs(v) for v in e.residuals.values())
    before, after = _baseline_result(), _describe(p, e)
    adoption = _production_at_capital(p, e.capital)["technique_details"]
    return {
        "inputs": {"auto": auto, "aug": aug, "own": own, "mp_low": mp_low,
                   "mp_high": mp_high, "lam_a": lam_a, "lam_p": lam_p, "c_p": c_p,
                   "tier": tier, "xi": xi, "chi": chi},
        "baseline": before,
        "scenario": after,
        "deciles": {
            "mean_wage_data": data["mean_wage"].tolist(),
            "automation_profile": a_profile.tolist(),
            "automation_data": data[f"auto_{tier}"].tolist(),
            "augmentation_profile": q_profile.tolist(),
            "augmentation_data": data[f"aug_{tier}"].tolist(),
            "alpha_baseline": base.alpha.tolist(),
            "alpha": p.alpha.tolist(),
            "productivity_gain": (psi_gain - 1.0).tolist(),
            "augmentation_capital": p.augmentation_compute_requirement.tolist(),
            "augmentation_gain": p.augmentation_gain.tolist(),
            "time_saving": (1.0 - 1.0 / p.augmentation_gain).tolist(),
            "automation_adoption": adoption["automation_adoption"].tolist(),
            "employer_labor_cost": e.wages.tolist(),
            "labor_resource_value": e.marginal_products_labor.tolist(),
            "private_labor_cost_ratio": ((1.0 - p.wage_markdown) * adoption["v"]).tolist(),
            "shadow_labor_cost_ratio": adoption["v"].tolist(),
            "realized_automation": (p.alpha * adoption["automation_adoption"]).tolist(),
            "augmentation_adoption": adoption["augmentation_adoption"].tolist(),
            "markdown_baseline": base.wage_markdown.tolist(),
            "markdown": p.wage_markdown.tolist(),
            "wage_change": (np.asarray(after["wages"]) / np.asarray(before["wages"]) - 1.0).tolist(),
        },
        "percentile_labels": PERCENTILE_LABELS,
        "max_residual": worst,
    }


PRESET_DEFAULTS = {
    "auto": "data", "aug": "data", "own": 0, "mp_low": 0, "mp_high": 0,
    "lam_a": DEFAULT_LAMBDA_A, "lam_p": DEFAULT_LAMBDA_P,
    "c_p": DEFAULT_AUGMENTATION_COST, "tier": TIER,
    "xi": DEFAULT_WAGE_SHARING, "chi": None,
}
PRESETS = [
    {"id": "pre_ai", "label": "Pre-AI baseline",
     "inputs": {**PRESET_DEFAULTS, "lam_a": 0., "lam_p": 0.}},
    {"id": "concentrated", "label": "Concentrated AI",
     "inputs": {**PRESET_DEFAULTS, "lam_a": .65, "lam_p": .35, "xi": .25}},
    {"id": "wage_sharing", "label": "Higher wage sharing",
     "inputs": {**PRESET_DEFAULTS, "lam_a": .65, "lam_p": .35}},
    {"id": "ownership", "label": "Broader ownership",
     "inputs": {**PRESET_DEFAULTS, "lam_a": .65, "lam_p": .35, "xi": .25, "chi": .25}},
    {"id": "combined", "label": "Combined scenario",
     "inputs": {**PRESET_DEFAULTS, "tier": "limited", "lam_a": .5, "lam_p": 1., "chi": .25}},
]


def meta() -> Dict[str, object]:
    return {
        "questions": QUESTIONS,
        "defaults": PRESET_DEFAULTS,
        "tier": TIER,
        "tiers": [{"id": "limited", "label": "Limited: T4"},
                  {"id": "modest", "label": "Modest: T3–T4"},
                  {"id": "broad", "label": "Broad: T2–T4"}],
        "presets": PRESETS,
        "model_version": "ai_mrr_reviewed-shadow-adoption-2026-10-09",
        "parameter_mapping": {
            "auto": "d_j: automation opportunities; sum(eta_j*d_j) held fixed",
            "aug": "e_j: time-saving opportunities; sum(eta_j*e_j) held fixed",
            "lam_a": "alpha_j = alpha_0 + (1-alpha_0)*lambda_A*d_j",
            "lam_p": "s_j = lambda_P*e_j; G_j = 1/(1-s_j)",
            "c_p": "m_j = c_P*s_j",
            "xi": "tau_sharing_j = 1-(1-tau_0_j)*G_j**(xi-1)",
            "mp_low,mp_high": "tau_j = min(.95, multiplier_j*tau_sharing_j)",
            "own,chi": "chi overrides the ownership answer when supplied",
            "adoption": "shadow-cost rule: R < omega_j/psi_j0; R*m_j < omega_j/psi_j0*(1-1/G_j)",
        },
        "task_sources": ["O*NET", "Eloundou et al. (2024)", "Hosseini and Lichtinger (2026)"],
    }
