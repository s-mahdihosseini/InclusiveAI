"""'Your AI future': beliefs -> parameters -> steady state of the AI--MRR model.

The economics is the maintained model in
``model/ai_mrr_extended`` (AI extension of Moll, Rachel and Restrepo 2022,
"Uneven Growth"). ``mrr_solver.py`` is an unmodified copy of that project's
``solver.py``; the calibration below reproduces ``run_scenarios.base_parameters``.

A visitor answers four questions. Each answer moves one block of parameters:

1. Who does AI automate?      -> the cross-decile profile of automation a_j
2. Whom does AI augment?      -> the cross-decile profile of augmentation q_j
3. Who owns AI capital?       -> chi, the share of households with risky equity
4. Employer market power      -> wage markdowns tau_j for low- and high-wage work

The aggregate size of the shock (lambda_A, lambda_P) is held at a benchmark and
can be changed under "advanced". Questions 1--2 change only *who* is exposed:
the value-added-weighted mean of the profile equals its value in the task data.
Every result compares two balanced-growth steady states: the calibrated pre-AI
economy and the economy implied by the visitor's answers.
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
    normalize_productivity,
    production_shares,
    solve_equilibrium,
    stationary_atoms,
    weighted_gini,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_FILE = os.path.join(HERE, "data", "ai_occupation_calibration.csv")

# ---------------------------------------------------------------------------
# Calibration (identical to ai_mrr_extended/run_scenarios.py)
# ---------------------------------------------------------------------------
MRR = {
    "rho": 0.01, "sigma": 2.0, "gamma": 2.0, "nu": 0.077, "chi": 0.066,
    "dissipation": 0.045, "growth": 0.015, "delta": 0.05, "leverage_limit": 2.0,
}
LABOR_SHARE_TARGET = 0.66
MARKDOWN_PROFIT_SHARE_TARGET = 0.078
OTHER_BUSINESS_PROFIT_SHARE_TARGET = 0.030
BASELINE_PHYSICAL_CAPITAL_SHARE = 1.0 - LABOR_SHARE_TARGET - MARKDOWN_PROFIT_SHARE_TARGET
AUTOMATED_TASK_COST_GAP = 1.30
SEEGMILLER_INVERSE_MARKDOWN = np.r_[np.full(4, 1.14), np.full(4, 1.16), np.full(2, 1.23)]

# Exposure tier used for the occupation profiles (T3--T4 automation).
TIER = "modest"
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
        "text": "Think of tasks that AI and machines take over completely.",
        "has_data_option": True,
        "options": [
            "Mostly low-wage jobs", "Somewhat more low-wage jobs", "Evenly across jobs",
            "Somewhat more high-wage jobs", "Mostly high-wage jobs",
        ],
        "maps_to": "Cross-occupation profile of automation a_j (mean held at the task-data value)",
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
        "maps_to": "Cross-occupation profile of augmentation q_j (mean held at the task-data value)",
    },
    {
        "id": "own",
        "title": "Who will own the companies and capital that run AI?",
        "text": "Today roughly 7 percent of households hold most risky business equity.",
        "has_data_option": False,
        "options": [
            "Even fewer people than today", "Somewhat fewer than today", "About as few as today",
            "Broader ownership", "Much broader ownership",
        ],
        "values": [CHI[k] for k in range(-2, 3)],
        "maps_to": "chi: share of households with access to risky business/AI equity",
    },
    {
        "id": "mp",
        "title": "How much power will employers have over wages?",
        "text": "Employers with market power pay workers less than the value they produce. "
                "Answer separately for low-wage and high-wage workers.",
        "has_data_option": False,
        "options": ["Much weaker", "Weaker", "Same as today", "Stronger", "Much stronger"],
        "values": [MARKDOWN_MULTIPLIER[k] for k in range(-2, 3)],
        "maps_to": "Wage markdowns tau_j, scaled relative to the Seegmiller-based pre-AI schedule",
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
        alpha=np.full(len(masses), BASELINE_PHYSICAL_CAPITAL_SHARE),
        psi=np.ones(len(masses)),
        wage_markdown=markdown,
        other_profit_share=OTHER_BUSINESS_PROFIT_SHARE_TARGET,
        quadrature_size=QUADRATURE,
    )
    p = normalize_productivity(p, 1.0)
    e = solve_equilibrium(p, compute_distribution=False)
    psi = e.wages / (AUTOMATED_TASK_COST_GAP * (e.r_risky + p.delta))
    p = normalize_productivity(replace(p, psi=psi), 1.0)
    return p, data


# ---------------------------------------------------------------------------
# Beliefs -> parameters
# ---------------------------------------------------------------------------
def tilted_profile(observed: np.ndarray, eta: np.ndarray, answer) -> np.ndarray:
    """Exposure profile across wage deciles with the eta-weighted mean held fixed.

    ``answer == 'data'`` returns the measured profile. Otherwise the profile is
    linear in wage rank, mean * (1 + s z_j), with z_j running from -1 (lowest
    decile) to +1 (highest) and s set by the answer.
    """
    if answer == "data":
        return observed.copy()
    mean = float(eta @ observed)
    n = len(observed)
    z = (np.arange(n) - (n - 1) / 2) / ((n - 1) / 2)
    profile = 1.0 + TILT[int(answer)] * z
    return profile * mean / float(eta @ profile)


def markdown_schedule(baseline: np.ndarray, mp_low: int, mp_high: int) -> np.ndarray:
    m_low, m_high = MARKDOWN_MULTIPLIER[int(mp_low)], MARKDOWN_MULTIPLIER[int(mp_high)]
    multiplier = np.r_[np.full(4, m_low), np.full(4, np.sqrt(m_low * m_high)), np.full(2, m_high)]
    return np.clip(baseline * multiplier, 0.0, 0.95)


def scenario_parameters(auto, aug, own, mp_low, mp_high, lam_a, lam_p):
    base, data = _base()
    a_profile = tilted_profile(data[f"auto_{TIER}"], base.eta, auto)
    q_profile = tilted_profile(data[f"aug_{TIER}"], base.eta, aug)
    alpha = BASELINE_PHYSICAL_CAPITAL_SHARE + (1.0 - BASELINE_PHYSICAL_CAPITAL_SHARE) * lam_a * a_profile
    psi_gain = 1.0 / (1.0 - lam_p * q_profile)
    p = replace(
        base,
        chi=CHI[int(own)],
        alpha=np.clip(alpha, 1e-6, 1 - 1e-6),
        psi=base.psi * psi_gain,
        wage_markdown=markdown_schedule(base.wage_markdown, mp_low, mp_high),
    )
    return p, a_profile, q_profile, psi_gain


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
    sh = production_shares(p)
    top10_cap = _top_share(total, cap, w, 0.10) * float(np.sum(cap * w))
    top10_tot = _top_share(total, total, w, 0.10) * float(np.sum(total * w))
    return {
        "output": e.output,
        "r_risky": e.r_risky,
        "r_safe": e.r_safe,
        "risky_share": e.risky_share,
        "tail_inverse": e.inverse_tail_investors,
        "chi": p.chi,
        "shares": {
            "labor": sh["labor"],
            "ordinary_capital": sh["ordinary_capital"],
            "markdown_profit": sh["markdown_profit"],
            "other_profit": sh["other_profit"],
        },
        "gini": {
            "labor": weighted_gini(labor, w),
            "capital": weighted_gini(cap, w),
            "total": weighted_gini(total, w),
            "wealth": weighted_gini(nw, w),
        },
        "top10_income": _top_share(total, total, w, 0.10),
        "top1_income": _top_share(total, total, w, 0.01),
        "top1_wealth": _top_share(nw, nw, w, 0.01),
        "top10_wealth": _top_share(nw, nw, w, 0.10),
        "top10_capital_income_share_of_income": top10_cap / top10_tot,
        "capital_income_share_of_income": float(np.sum(cap * w) / np.sum(total * w)),
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
          lam_a=DEFAULT_LAMBDA_A, lam_p=DEFAULT_LAMBDA_P) -> Dict[str, object]:
    base, data = _base()
    p, a_profile, q_profile, psi_gain = scenario_parameters(
        auto, aug, own, mp_low, mp_high, lam_a, lam_p)
    e = solve_equilibrium(p, compute_distribution=False)
    worst = max(abs(v) for v in e.residuals.values())
    before, after = _baseline_result(), _describe(p, e)
    return {
        "inputs": {"auto": auto, "aug": aug, "own": own, "mp_low": mp_low,
                   "mp_high": mp_high, "lam_a": lam_a, "lam_p": lam_p},
        "baseline": before,
        "scenario": after,
        "deciles": {
            "mean_wage_data": data["mean_wage"].tolist(),
            "automation_profile": a_profile.tolist(),
            "automation_data": data[f"auto_{TIER}"].tolist(),
            "augmentation_profile": q_profile.tolist(),
            "augmentation_data": data[f"aug_{TIER}"].tolist(),
            "alpha_baseline": base.alpha.tolist(),
            "alpha": p.alpha.tolist(),
            "productivity_gain": (psi_gain - 1.0).tolist(),
            "markdown_baseline": base.wage_markdown.tolist(),
            "markdown": p.wage_markdown.tolist(),
            "wage_change": (np.asarray(after["wages"]) / np.asarray(before["wages"]) - 1.0).tolist(),
        },
        "percentile_labels": PERCENTILE_LABELS,
        "max_residual": worst,
    }


def meta() -> Dict[str, object]:
    return {
        "questions": QUESTIONS,
        "defaults": {"auto": "data", "aug": "data", "own": 0, "mp_low": 0, "mp_high": 0,
                     "lam_a": DEFAULT_LAMBDA_A, "lam_p": DEFAULT_LAMBDA_P},
        "tier": TIER,
    }
