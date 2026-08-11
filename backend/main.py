"""
InclusiveAI — API server.

Run from the backend/ directory:
    uvicorn main:app --reload --port 8000

Serves:
  - GET /api/expertise/solve?sigma=&scarcity=&productivity=   -> model results (JSON)
  - GET /api/expertise/meta                                    -> parameter metadata
  -     /                                                      -> frontend (static)
"""

import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import demand_gpt
import expertise_static
import market_power

app = FastAPI(title="InclusiveAI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PARAM_META = {
    "models": [
        {
            "id": "expertise",
            "name": "AI & Occupational Entry Barriers",
            "status": "live",
            "paper": "Generative AI and Occupational Entry Barriers (Hosseini & Lichtinger, 2026)",
            "description": (
                "Static GE model of occupational choice with hierarchical entry "
                "barriers. AI affects wages via two channels: (1) scarcity — AI "
                "lowers retraining costs, eroding expertise rents at the top; "
                "(2) productivity — AI raises task productivity differentially "
                "across occupations."
            ),
            "parameters": [
                {"id": "sigma", "name": "σ — Elasticity of substitution across occupations",
                 "min": 1.5, "max": 15.0, "step": 0.5, "default": 5.0,
                 "help": "Higher σ: occupations are closer substitutes in production, so wage responses to labor reallocation are smaller."},
                {"id": "scarcity", "name": "Scarcity channel intensity",
                 "min": 0.0, "max": 1.5, "step": 0.05, "default": 1.0,
                 "help": "0 = AI does not reduce entry barriers; 1 = LLM-estimated barrier reduction (paper baseline); >1 = amplified erosion of expertise rents."},
                {"id": "productivity", "name": "Productivity channel intensity",
                 "min": 0.0, "max": 2.0, "step": 0.05, "default": 1.0,
                 "help": "0 = no productivity gains; 1 = LLM-estimated occupation-level productivity gains (paper baseline); >1 = amplified gains."},
            ],
        },
        {
            "id": "demand",
            "name": "Demand Structure & GPT Incidence",
            "status": "live",
            "paper": "The Incidence of a General-Purpose Technology (GPTNetwork project)",
            "description": (
                "Exact GE model of 4 occupation-goods plus a GPT sector. A cheaper "
                "GPT helps an occupation when the demand elasticity for its output "
                "exceeds its labor-GPT substitution elasticity; nonhomothetic "
                "preferences shift spending toward income-elastic services as the "
                "economy grows; worker mobility and household heterogeneity close "
                "the loop."
            ),
            "parameters": [
                {"id": "ag", "name": "GPT productivity multiplier (A_g)",
                 "min": 1.5, "max": 16.0, "step": 0.5, "default": 8.0,
                 "help": "Final size of the GPT shock. The paper's benchmark is x8 (roughly the computer era)."},
                {"id": "mobility", "name": "Worker mobility (κ scale)",
                 "min": 0.0, "max": 2.0, "step": 0.1, "default": 1.0,
                 "help": "0 = workers stay put (wages absorb everything); 1 = calibrated Roy/logit mobility; >1 = more fluid labor markets."},
                {"id": "exposure", "name": "GPT exposure scale (b₀)",
                 "min": 0.25, "max": 2.0, "step": 0.05, "default": 1.0,
                 "help": "Scales the share of each occupation's production costs that go to the GPT (clerical 20%, production/professional 15%, manual services 2% at baseline)."},
                {"id": "eps_spread", "name": "Demand-elasticity dispersion (ε)",
                 "min": 0.0, "max": 1.5, "step": 0.05, "default": 1.0,
                 "help": "0 = every good has the same price elasticity (CES benchmark: no immiserizing growth); 1 = calibrated (services elastic, clerical/production inelastic)."},
                {"id": "nonhom", "name": "Nonhomotheticity (ξ)",
                 "min": 0.0, "max": 1.5, "step": 0.05, "default": 1.0,
                 "help": "0 = homothetic (spending shares don't move with income); 1 = calibrated Engel forces (growth shifts spending toward professional & manual services)."},
                {"id": "sig_scale", "name": "Labor–GPT substitutability (σ scale)",
                 "min": 0.0, "max": 1.5, "step": 0.05, "default": 1.0,
                 "help": "Scales log σ. 0 = all occupations near Cobb-Douglas; 1 = calibrated (clerical highly substitutable σ=3, professional complementary σ=0.45)."},
                {"id": "hetero", "name": "Household heterogeneity",
                 "type": "toggle", "default": 1,
                 "help": "On: one household per education group — income distribution feeds back into demand composition. Off: pooled representative household."},
            ],
        },
        {
            "id": "market_power",
            "name": "AI Supplier Market Power & Compute",
            "status": "live",
            "paper": "AI Productivity, Upward-Sloping Factor Supply, and the Dynamics of Compute Investment (Hosseini & Lichtinger, 2026)",
            "description": (
                "Closed-form GE model of the AI supply side: a productivity shock "
                "raises desired AI scale, which runs into two bottlenecks — "
                "specialized labor on an upward-sloping supply curve, and compute "
                "capital that takes time to build. Who gains, how fast prices "
                "fall, and how income shares move all follow in closed form."
            ),
            "parameters": [
                {"id": "shock", "name": "AI productivity shock (%, permanent)",
                 "min": 0.5, "max": 20.0, "step": 0.5, "default": 1.0,
                 "help": "Size of the permanent increase in AI productivity A. All responses scale linearly (first-order solution)."},
                {"id": "sigma", "name": "σ — substitution: AI services vs. labor",
                 "min": 0.5, "max": 4.0, "step": 0.05, "default": 2.0,
                 "help": "Elasticity of substitution in final production. σ > 1: AI and workers are substitutes — AI's income share rises with productivity. σ < 1: complements — the share falls."},
                {"id": "sx0", "name": "Initial AI revenue share (s_X)",
                 "min": 0.05, "max": 0.8, "step": 0.01, "default": 0.5,
                 "help": "AI services' share of final-good revenue at the initial equilibrium. Sets θ = (1−s_X)/σ, the key sufficient statistic: scale effect wins when θ < 1."},
                {"id": "eta", "name": "η — specialized-labor supply elasticity",
                 "min": 0.0, "max": 5.0, "step": 0.1, "default": 1.0,
                 "help": "Slope of the AI-talent supply curve. 0 = fixed supply (all adjustment through wages); large = flat supply (adjustment through employment)."},
                {"id": "beta", "name": "β — AI-labor share of AI revenue",
                 "min": 0.05, "max": 0.9, "step": 0.05, "default": 0.4,
                 "help": "Cobb-Douglas weight on specialized labor in AI production."},
                {"id": "gamma", "name": "γ — compute share of AI revenue",
                 "min": 0.05, "max": 0.9, "step": 0.05, "default": 0.4,
                 "help": "Cobb-Douglas weight on compute capital. The remainder 1−β−γ is the rent to the fixed factor (data, organization). β+γ is capped at 0.95."},
                {"id": "phi", "name": "φ — compute adjustment cost",
                 "min": 0.5, "max": 50.0, "step": 0.5, "default": 8.0,
                 "help": "How costly it is to build compute quickly (data centers, power, chips). Changes the speed of the transition — not the impact response or the long run."},
            ],
        },
    ]
}


@app.get("/api/expertise/meta")
def meta():
    return PARAM_META


@app.get("/api/expertise/solve")
def solve(
    sigma: float = Query(5.0, ge=1.05, le=25.0),
    scarcity: float = Query(1.0, ge=0.0, le=2.0),
    productivity: float = Query(1.0, ge=0.0, le=3.0),
):
    try:
        # Round to grid so the lru_cache is effective
        out = expertise_static.solve(
            round(float(sigma), 2), round(float(scarcity), 2),
            round(float(productivity), 2))
        return out
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/solve")
def solve_market(
    sigma: float = Query(2.0, ge=0.1, le=8.0),
    sx0: float = Query(0.5, ge=0.01, le=0.9),
    eta: float = Query(1.0, ge=0.0, le=10.0),
    beta: float = Query(0.4, ge=0.01, le=0.9),
    gamma: float = Query(0.4, ge=0.01, le=0.9),
    phi: float = Query(8.0, ge=0.1, le=100.0),
    shock: float = Query(1.0, ge=0.1, le=50.0),
):
    try:
        out = market_power.solve(
            round(float(sigma), 3), round(float(sx0), 3), round(float(eta), 3),
            round(float(beta), 3), round(float(gamma), 3),
            round(float(phi), 2), round(float(shock), 2))
        if "error" in out:
            raise HTTPException(status_code=422, detail=out["error"])
        return out
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e))


import json as _json

_SCEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scenarios")
_scen_cache = {"mtime": None, "data": None}


@app.get("/api/scenarios")
def scenarios():
    """scenarios.json enriched with each source's central_scenario_summary,
    author/title, and one representative quote from extractions/."""
    path = os.path.join(_SCEN_DIR, "scenarios.json")
    mtime = os.path.getmtime(path)
    if _scen_cache["mtime"] == mtime:
        return _scen_cache["data"]

    with open(path, encoding="utf-8") as f:
        data = _json.load(f)

    for sc in data["scenarios"]:
        enriched = []
        for sid in sc["sources"]:
            entry = {"id": sid}
            epath = os.path.join(_SCEN_DIR, "extractions", f"{sid}.json")
            try:
                with open(epath, encoding="utf-8") as f:
                    ex = _json.load(f)
                entry["author"] = ex.get("author", sid)
                entry["title"] = ex.get("title", "")
                entry["year"] = ex.get("year", "")
                entry["url"] = ex.get("url", "")
                entry["summary"] = ex.get("central_scenario_summary", "")
                for dim in ex.get("dimensions", {}).values():
                    if isinstance(dim, dict) and dim.get("quotes"):
                        entry["quote"] = dim["quotes"][0].get("text", "")
                        break
            except Exception:
                pass
            enriched.append(entry)
        sc["sources"] = enriched

    _scen_cache.update(mtime=mtime, data=data)
    return data


@app.get("/api/demand/solve")
def solve_demand(
    ag: float = Query(8.0, ge=1.01, le=32.0),
    mobility: float = Query(1.0, ge=0.0, le=4.0),
    exposure: float = Query(1.0, ge=0.05, le=3.0),
    eps_spread: float = Query(1.0, ge=0.0, le=2.0),
    nonhom: float = Query(1.0, ge=0.0, le=2.0),
    sig_scale: float = Query(1.0, ge=0.0, le=2.0),
    hetero: int = Query(1, ge=0, le=1),
):
    try:
        return demand_gpt.solve(
            round(float(ag), 2), round(float(mobility), 2),
            round(float(exposure), 2), round(float(eps_spread), 2),
            round(float(nonhom), 2), round(float(sig_scale), 2),
            bool(hetero))
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e))


# Serve the frontend at / (must be mounted last)
_FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.isdir(_FRONTEND):
    app.mount("/", StaticFiles(directory=_FRONTEND, html=True), name="frontend")
