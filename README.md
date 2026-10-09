
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

An interactive general-equilibrium model of how AI shapes the distribution of
prosperity. Visitors answer four questions about AI; the site solves the model
and compares the pre-AI and post-AI long-run steady states.
Companion site to *AI and the Distribution of Prosperity*.

The model builds on Moll, Rachel and Restrepo (2022) and uses the reviewed
`model/ai_mrr_reviewed` implementation: automation and fixed-proportions worker
augmentation with compute. Firms choose techniques by comparing capital costs
with labor’s shadow value. Wage markdowns redistribute income between workers
and owners; the financial equilibrium determines their effect on capital
accumulation. The revised appendix and website use the same
calibration and five scenario presets. The first version of the site (expertise,
demand-structure and compute-bottleneck models) is archived in `archive/v1`.

## Structure

```
InclusiveAI/
├── backend/
│   ├── main.py               # FastAPI app: /api/future/* + serves the frontend
│   ├── ai_future.py          # Four belief questions -> parameters -> steady state
│   ├── mrr_solver.py         # Copy of model/ai_mrr_reviewed/solver.py
│   ├── technology.py         # Task allocation at shadow labor and capital costs
│   ├── reference_solver.py   # Reproduces the baseline normalization
│   ├── test_integration.py   # Appendix parity and economic checks
│   ├── data/ai_occupation_calibration.csv  # Occupation deciles (simple_aiyagari_ge/build_calibration.py)
│   └── requirements.txt
├── frontend/
│   ├── index.html            # The page (Chart.js, no build step)
│   └── future.html           # Redirect to / for old links
└── archive/v1/               # Retired v1 site and models (not served)
```

## Run locally

```bash
bash run.sh
# open http://127.0.0.1:8000
```

## Deploy

Render (Docker) builds from this repo; every push to GitHub `main` redeploys.

## Model controls and appendix connection

The four questions change the incidence of automation and augmentation,
participation in risky equity, and wage markdowns. Advanced controls set the task
tier, automation feasibility, augmentation time savings, compute requirement,
wage sharing, and optional exact equity participation. The preset selector
reproduces all five revised appendix cases.

[MODEL_PARAMETERS.md](MODEL_PARAMETERS.md) maps every control to its equation,
explains the units and adoption decisions, and gives settings for reproducing
the appendix's comparative statics. The existing employer-power controls apply
additional markdown multipliers; the new `xi` control implements the appendix's
wage-sharing rule.

API: `GET /api/future/meta` supplies questions, defaults, presets, and parameter
mappings. `GET /api/future/solve` accepts the existing `auto`, `aug`, `own`,
`mp_low`, `mp_high`, `lam_a`, `lam_p`, and `c_p` fields, plus optional `tier`,
`xi`, and `chi`. Existing request URLs continue to work.

## Verify the integration

```bash
.venv/bin/python -m unittest discover -s backend -p 'test_*.py' -v
```

Tests verify all five appendix equilibria, bounded fixed-mean task profiles,
shadow-cost adoption conditions, fixed-capital markdown invariance, factor
payments, household income accounting, and control
extremes. The reviewed solver and its two supporting modules are vendored into
this repository so Docker deployments remain self-contained.
