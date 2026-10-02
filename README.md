
---
title: InclusiveAI Simulator
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
# InclusiveAI

Interactive general-equilibrium models of how AI shapes the distribution of prosperity.
Companion site to *AI and the Distribution of Prosperity*.

## Structure

```
InclusiveAI/
├── backend/
│   ├── main.py               # FastAPI app: API endpoints + serves the frontend
│   ├── expertise_static.py   # Static expertise model solver (refactored from Final_Supply.py)
│   ├── ai_future.py          # "Your AI Future" page: four belief questions -> AI-MRR steady state
│   ├── mrr_solver.py         # Copy of model/ai_mrr_extended/solver.py (AI extension of Moll-Rachel-Restrepo)
│   ├── data/ai_occupation_calibration.csv  # Occupation deciles from simple_aiyagari_ge/build_calibration.py
│   └── requirements.txt
├── frontend/
│   ├── index.html            # Single-page app (Chart.js, no build step)
│   └── future.html           # Separate "Your AI Future" page (AI-MRR model)
└── models/
    └── expertise/
        ├── static/           # Reference code + data (Final_Supply.py, Counterfactual.dta,
        │                     #   create_paper_results.py, occupation_titles.csv)
        └── dynamic/          # Dynamic model scripts + codebase guide (for a future sub-tab)
```

## Run locally

```bash
cd backend
pip install -r requirements.txt
python3 -m uvicorn main:app --port 8000
# open http://127.0.0.1:8000
```

## Tabs

1. **Expertise & Entry Barriers** (live) — Hosseini & Lichtinger (2026). Static GE model,
   885 O*NET occupations. Two channels, each user-scalable:
   - *Scarcity*: AI lowers retraining barriers → feasibility ps interpolated between
     baseline F_at_R and AI F_at_RAI by the intensity slider.
   - *Productivity*: occupation-level LLM-estimated productivity gains π, scaled by slider.
   - σ (CES elasticity across occupations) is free; ψ is re-calibrated per σ so the
     baseline is always matched. τ (taste-shock scale) is calibrated once to baseline
     employment shares (τ̂ ≈ 0.717).
2. **Demand Structure & GPT Incidence** (live) — GPTNetwork project. Exact GE of
   4 occupation-goods (manual services, production, clerical, professional) + a GPT
   sector, with implicitly additive (Hanoch) nonhomothetic demand, Roy/logit worker
   mobility across 3 education groups, and optional per-group households
   (distributional demand feedback). User-scalable: A_g shock size, mobility κ,
   GPT exposure b₀, demand-elasticity dispersion ε, nonhomotheticity ξ,
   labor–GPT substitutability σ, pooled vs heterogeneous households.
   Solver: scipy-free damped Newton warm-started along the A_g path
   (`backend/demand_gpt.py`; verified against the note: Hulten check ~1e-13, and
   the hetero-vs-pooled feedback reproduces the paper's 0.05 / 0.01 log points).
3. **AI Market Power** (planned) — model from the AI and Market Power project.

## Expert scenarios

Both live tabs have an "Expert scenario" dropdown: 5 presets distilled from 28
expert sources (economists, lab leaders, forecasters, skeptics) via a fixed
extraction rubric. Selecting one sets all sliders and shows the narrative +
sources with quotes. See `scenarios/README.md` for the pipeline
(corpus → rubric extraction → synthesis → scenarios.json) and how to scale it.
API: `GET /api/scenarios`.

## Implementation notes

- The solver is a vectorized O(J) rewrite of the original O(J²) labor-share computation,
  verified to match the original to machine precision (`python3 backend/expertise_static.py`
  runs the check).
- PE formulas and the exact-GE damped fixed point are unchanged from `Final_Supply.py`.
- No scipy dependency: τ calibration uses golden-section search; KDE is a NumPy
  Gaussian KDE with Scott's rule (matches scipy defaults).
- Responses are cached (`lru_cache`) on a rounded parameter grid; a solve takes ~10 ms,
  so live sliders are feasible even without the cache.

## API

- `GET /api/expertise/meta` — model + parameter metadata (drives the frontend controls).
- `GET /api/expertise/solve?sigma=5&scarcity=1&productivity=1` — returns inequality stats
  (baseline / PE / GE / channel decomposition), binned Δlog-wage scatters, wage-density
  KDEs, and the full occupation-level table.

## Deployment

Any host that runs Python works (Render, Railway, Fly.io, a university server):
`uvicorn main:app --host 0.0.0.0 --port $PORT`. The frontend is static and served by
FastAPI itself, so a single service is enough.

## "Your AI Future" page (`/future.html`)

A separate page built on the AI extension of Moll, Rachel and Restrepo (2022)
maintained in `model/ai_mrr_extended` (note: `output/pdf/model.pdf`). Visitors answer
four questions; each maps to one block of parameters:

| Question | Parameter |
|---|---|
| Which jobs will AI automate more? | tilt of the automation profile a_j across wage deciles (mean held at the task-data value) |
| Which workers will AI make more productive? | tilt of the augmentation profile q_j (mean held fixed) |
| Who will own AI capital? | chi, share of households with risky equity: 3.5, 5, 6.6 (MRR), 15, 30% |
| Employer power over wages (low / high wage) | multiplier on pre-AI markdowns: 0.25, 0.6, 1, 1.5, 2 (deciles 1-4 and 9-10; 5-8 geometric mean) |

The aggregate shock size is lambda_A = lambda_P = 0.5 on the T3-T4 tier by default and can be
changed under "Advanced". The page compares the pre-AI and post-AI balanced-growth steady
states (about 0.1 s per solve). `mrr_solver.py` is an unmodified copy of the model's
`solver.py`; if the model changes, copy it again together with the calibration CSV.

API: `GET /api/future/meta`, `GET /api/future/solve?auto=data|-2..2&aug=...&own=-2..2&mp_low=-2..2&mp_high=-2..2&lam_a=0.5&lam_p=0.5`.
