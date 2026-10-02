# Archived: InclusiveAI v1 (through Oct 2026)

The first version of the site, retired when the site switched to the AI-MRR model
(`backend/ai_future.py`). Nothing here is served or built by the Dockerfile.

Contents:

- `frontend/index.html` — the multi-tab site (Home, Expertise & Entry Barriers,
  Demand Structure, Compute Bottlenecks, Underlying Models, About).
- `backend/expertise_static.py`, `demand_gpt.py`, `market_power.py` — the three model solvers.
- `backend/main.py`, `backend/requirements.txt`, `Dockerfile`, `DEPLOY.md` — the v1 server,
  dependencies (needs pandas), container, and deploy notes as they were.
- `models/` — reference code and data for the expertise and demand models.
- `scenarios/` — the expert-scenario corpus, extractions, and rubric used by the v1 tabs.

To run v1 again, restore these paths to the repository root (`frontend/`, `backend/`,
`models/`, `scenarios/`) and use the archived `Dockerfile` and `requirements.txt`.
