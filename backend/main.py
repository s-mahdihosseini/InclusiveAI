"""
InclusiveAI — API server.

Run from the backend/ directory:
    uvicorn main:app --reload --port 8000

Serves:
  - GET /api/future/meta                                        -> questions and defaults
  - GET /api/future/solve?auto=&aug=&own=&mp_low=&mp_high=      -> steady-state comparison
                         &lam_a=&lam_p=&c_p=
  -     /                                                       -> frontend (static)

The earlier models (expertise, demand structure, compute bottlenecks) and their
endpoints are archived in ../archive/v1.
"""

import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import ai_future

app = FastAPI(title="InclusiveAI API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _answer(value: str, allow_data: bool):
    if allow_data and value == "data":
        return "data"
    try:
        v = int(value)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"invalid answer {value!r}")
    if v not in (-2, -1, 0, 1, 2):
        raise HTTPException(status_code=422, detail="answers must be between -2 and 2")
    return v


@app.get("/api/future/meta")
def future_meta():
    return ai_future.meta()


@app.get("/api/future/solve")
def future_solve(
    auto: str = Query("data"),
    aug: str = Query("data"),
    own: str = Query("0"),
    mp_low: str = Query("0"),
    mp_high: str = Query("0"),
    lam_a: float = Query(ai_future.DEFAULT_LAMBDA_A, ge=0.0, le=1.0),
    lam_p: float = Query(ai_future.DEFAULT_LAMBDA_P, ge=0.0, le=1.0),
    c_p: float = Query(ai_future.DEFAULT_AUGMENTATION_COST, ge=0.0, le=0.9),
):
    try:
        return ai_future.solve(
            _answer(auto, True), _answer(aug, True), _answer(own, False),
            _answer(mp_low, False), _answer(mp_high, False),
            round(float(lam_a), 2), round(float(lam_p), 2), round(float(c_p), 2))
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/healthz")
def healthz():
    return {"ok": True}


# Serve the frontend at / (must be mounted last)
_FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.isdir(_FRONTEND):
    app.mount("/", StaticFiles(directory=_FRONTEND, html=True), name="frontend")
