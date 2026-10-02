
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

The model is the AI extension of Moll, Rachel and Restrepo (2022) maintained in
`model/ai_mrr_extended` (note: `output/pdf/model.pdf`), with a constant product
markup and capital-using augmentation. The first version of the site (expertise,
demand-structure and compute-bottleneck models) is archived in `archive/v1`.

## Structure

```
InclusiveAI/
├── backend/
│   ├── main.py               # FastAPI app: /api/future/* + serves the frontend
│   ├── ai_future.py          # Four belief questions -> parameters -> steady state
│   ├── mrr_solver.py         # Copy of model/ai_mrr_extended/solver.py
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

## The model page

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
changed under "Advanced", together with c_P (AI capital needed per unit of augmentation relative to
full automation; default 0.25). The model version has a constant product markup (3% of output) and
capital-using augmentation, matching ai_mrr_extended as of 2 Oct 2026. The page compares the pre-AI and post-AI balanced-growth steady
states (about 0.1 s per solve). `mrr_solver.py` is an unmodified copy of the model's
`solver.py`; if the model changes, copy it again together with the calibration CSV.

API: `GET /api/future/meta`, `GET /api/future/solve?auto=data|-2..2&aug=...&own=-2..2&mp_low=-2..2&mp_high=-2..2&lam_a=0.5&lam_p=0.5`.
