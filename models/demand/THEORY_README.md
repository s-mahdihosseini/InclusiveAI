# Productivity Growth, Factor Shares, and Wages

A short theory exploration of one recurring question in three progressively richer general-equilibrium setups:

> **When a factor (or the sector that uses it) becomes more productive, do its own wages / income share go up or down?**

The answer always has the same shape — a productivity gain raises quantity but pushes price down, and who wins is decided by a single elasticity. What changes across the three models is *which* elasticity.

---

## The three models

### 1. One good, two labor types combined by CES (skill-biased technical change)

A single final good from a CES aggregate of two efficiency-labor bundles,

```
Y = [ (A1 L1)^((σ-1)/σ) + (A2 L2)^((σ-1)/σ) ]^(σ/(σ-1))
```

with `L1, L2` supplied inelastically. Competitive wages `w_i = ∂Y/∂L_i`. The revenue share collapses to each factor's weight in the aggregator:

```
s1 = (A1 L1)^((σ-1)/σ) / [ (A1 L1)^((σ-1)/σ) + (A2 L2)^((σ-1)/σ) ]
```

**Comparative static (as A1 rises relative to A2):**

```
d ln(s1/s2) / d ln(A1/A2) = (σ - 1) / σ
```

- σ > 1 (substitutes): skill-biased tech change **raises** the skilled share.
- σ < 1 (complements): it **lowers** it.
- σ = 1 (**Cobb–Douglas**): shares are constant (`s1 = α`), independent of productivity — the knife-edge.

### 2. Two goods, specific labor, different demand elasticities

Each labor type produces its own good, `Y_i = A_i L_i`, sold to the household with price elasticity `ε_i`. Since labor is the only input, **sector wage bill = sector revenue** (`w_i L_i = P_i Y_i`). Productivity raises output one-for-one but slides price down the demand curve, giving:

```
d ln w_i / d ln A_i = 1 - 1/ε_i = (ε_i - 1) / ε_i
```

- Elastic good (ε > 1): productivity **raises** the wage.
- Inelastic good (ε < 1): productivity **lowers** it — *immiserizing growth*.
- Unit elastic (ε = 1): wage unchanged (constant revenue).

This required leaving CES on the demand side, since CES forces both goods to share one elasticity.

### 3. Explicit Stone–Geary microfoundation

To get genuinely different elasticities across goods (and stop asserting them), the household is given Stone–Geary preferences:

```
U = Σ β_i ln(C_i − γ_i),   Σ β_i = 1
```

with `γ_i` the subsistence/committed quantity of good `i`. This yields the Linear Expenditure System and, derived rather than assumed:

```
own-price elasticity:  ε_ii = 1 − (1 − β_i) γ_i / C_i
income elasticity:      η_i  = β_i I / (P_i C_i)
```

so `γ_i > 0` (necessity) ⟹ price- and income-**inelastic**; `γ_i < 0` (luxury) ⟹ **elastic**; `γ_i = 0` ⟹ Cobb–Douglas. Closing GE (`w_i = P_i A_i`, market clearing) gives a closed-form relative price and the explicit wage result:

```
d ln w1 / d ln A1 = −γ1 / (A1 L1 − γ1) = (ε1 − 1)/ε1,   ε1 = 1 − γ1/(A1 L1)
```

The sign is controlled entirely by the primitive `γ1`. Example: a necessity sector (β1 = 0.4, γ1 = 0.6) made 30% more productive sees its price fall 33% and its wage fall **13.3%**.

---

## The unifying idea

All three results are the same object — `(x − 1)/x` — with `x` being the elasticity that governs how much price must fall to absorb extra output:

| Model | Elasticity `x` | Wage/share rises iff |
|-------|----------------|----------------------|
| 1. CES factor bundles | substitution elasticity σ | σ > 1 |
| 2. Two sectors | demand elasticity ε | ε > 1 |
| 3. Stone–Geary | GE demand elasticity ε = 1 − γ/(AL) | γ < 0 (luxury) |

The economics: a factor benefits from its own productivity growth only if the world wants proportionally more of what it makes. This is the mechanism behind Baumol-style structural change (agriculture: huge productivity growth + inelastic demand ⟹ falling farm prices, incomes, and employment) and skill-biased technical change (same formula with σ).

---

## Folder layout

```
paper/       gpt_wages_paper.tex / .pdf, references.bib   (the main paper)
notes/       km_revisited, nhces_mobility_note, wages_note (earlier companions)
code/        verify_*.py, ar_data_bridge.py                (numerical verification)
data/        19815_Data_and_Programs/                      (AR 2022 replication kit)
literature/  input PDFs (AA 2011, AR 2022, CLM, Barany-Siegel, ...)
```

## Files

- **`wages_note.pdf`** — a 2-page LaTeX note covering models 2 and 3, with the `(ε−1)/ε` figure drawn natively in pgfplots and the full Stone–Geary derivation.
- **`wages_note.tex`** — LaTeX source (edit and recompile to restyle).
- **`verify_nhces.py`** — companion for the non-homothetic CES (Comin–Lashkari–Mestieri) version, the serious quantitative demand system. Verifies: the exact global demand log-differentials from the closed-form expenditure function E(p,U); the household identity Û_m = (Î_m − ω_m·p̂)/ε̄_m; the mobility wage system under nhCES; the **complete closed-form first-order GE** (Hulten pins Î, Û = Î/ε̄, explicit p̂_g and wages — Prop. "Complete closed form under nhCES"); the failure of Gorman aggregation under common nhCES preferences (endogenous but modest distributional channel); and the calibrated A_g×8 illustration (polarization survives a common price elasticity σ_D = 0.7; Engel heterogeneity persists instead of fading).
- **`nhces_mobility_note.pdf`** — the pedagogical note (10 pp.): "Automation, Engel Curves, and Occupational Mobility: One Model, Built Slowly." nhCES utility set up from first principles with every derivation shown step by step (implicit utility → expenditure function → Marshallian demand → elasticities → Engel curves → non-aggregation → welfare), occupational choice derived from Gumbel draws (with the integral in a footnote), and one **integrated** three-sector economy (S/R/A + capital-produced automation, two overlapping worker groups who are also the nhCES households). Delivers simultaneously: falling labor share (closed-form harmonic-mean condition H = Σχ·h_i > 1, with h_i the cost-share-weighted harmonic mean of σ_D and σ_i), wage/employment polarization, reallocation, the Engel loop, and earnings ≠ welfare. General N-sector model stated at the end. No ad hoc linearized shifters anywhere.
- **`nhces_mobility_note.tex`** — LaTeX source.
- **`verify_three_sector.py`** — numerical companion: exact nonlinear GE of the integrated model; verifies the derived demand differentials, the integrated wage system, the labor-share closed form (5e-11), and the A_g×8 illustration.
- **`gpt_wages_paper.pdf`** — the full paper (52 pp., house style): "The Incidence of General-Purpose Technologies: Sufficient Statistics with Non-Homothetic Demand and Worker Mobility." Framework-first framing: observable sufficient statistics (exposure b_i, substitution σ_i, demand pair (σ_D, ξ_i), mobility matrix) govern the incidence of a GPT shock; nhCES demand throughout (no LES); closed-form baseline GE; ripple propagation; two-exposure network results with the derived-demand overhead microfoundation; endogenous Engel channel. Section 7 shows a single calibrated shock matching five stylized facts of the computer era (F1 real wage declines amid growth, F2 intensifying wage polarization, F3 employment polarization, F4 occupational reallocation with occupations predicting wages, F5 displacement). Section 8 prices the birth of new occupations in the same calibrated economy. Appendices: proofs, demand-system facts, exact (beyond-first-order) results, calibration-to-estimation review, calibration + verification table.
- **`verify_exact.py`** — beyond-first-order companion: (E1) exact block separability + **global** conditional horse-race monotonicity (16-fold p_g grid); (E2) the sufficient-statistic system is the **exact tangent field** — RK4 integration over A_g ∈ [1,8] reproduces the exact nonlinear GE to 1.1e-13; (E3) common-substitution economy (σ_i ≡ σ_D): closed-form wages/GPT price, GE = one scalar equation, relative wages exactly log-linear in U (1e-16); (E4) Cobb–Douglas demand: exact identity w_iL_i = (1−b_i)c_i, global σ_i-vs-1 horse race, quadratic closed form for σ=2 (1e-12). Written up as the paper's "Beyond the First Order" subsection + appendix.
- **`verify_new_occupations.py`** — new-occupations companion: latent GPT-complements (σ_i < min(1, σ_D), high θ_i) as embryonic occupations. Verifies: (X1) the conditional birth curve L ∝ q^(σ_i−σ_D) in closed form (2e-16) with latency iff σ_i < σ_D; (X2) the exact logistic law db/dlnp_g = (1−σ)b(1−b) for the GPT cost share; (X3) the monotone GE birth path in a 5-occupation economy — employment ×24 and real wage ×64 as A_g goes from 1/64 to 8, with the 5-sector wage system verifying at 9e-11; (X4) absorption effects on incumbents. Written up as the paper's Section "New Occupations: Birth by Complementarity" + quant subsection + appendix proof. Punchline: the birth condition IS the horse-race condition — dying, thriving, and newborn occupations are one statistic read at three points of its range.
- **`verify_network_nhces.py`** — the COMBINED full system: production network (s>0, CES_zeta outer nest, CES_rho bundle) + nhCES per-group households (per-capita utility) + logit mobility, all active at the paper's calibration. (NN1) the stacked block-linear system of the paper's Appendix "Assembly of the Full System" reproduces FD comparative statics of the exact nonlinear equilibrium at 3e-10, for rho in {1.0, 0.5}; (NN2) the displacement identity in the combined economy (1e-11); (NN3) A_g x8 illustration — network incidence tracks the star headline experiment, slightly amplified (clerical employment -0.50 vs -0.48).
- **`ar_data_bridge.py`** — bridge from the Acemoglu–Restrepo (2022) replication kit (`19815_Data_and_Programs/`) to the paper: (D1) education-group calibration targets — 1980 hours shares (0.58, 0.23, 0.19) and relative wages (1.00, 1.11, 1.62); (D2) the incidence profile in the data — real wage change 1980–2016 by ventile of the 1980 group wage distribution, with 53% of 1980 employment in groups whose real wages fell; (D3) sufficient-statistic correlations across 500 groups — task displacement vs Δlnw −0.82, Webb software exposure vs task displacement +0.73, exposure vs Δlnw −0.78. The paper's worker groups are recalibrated to these targets (with per-capita household utility E(p,U_m)=I_m/n_m), and Section "The Incidence in the Data, 1980–2016" presents the facts figure and correlations.
- Paper appendix "From Calibration to Estimation: The Elasticities and Their Data" — parameter-by-parameter literature review (CLM 2021 for σ_D and ξ; KORV 2000, Eden–Gaggl 2018, Hubmer 2023, Oberfield–Raval 2021 for σ_i; HHJK 2019, Burstein–Morales–Vogel 2019, Traiberman 2019 for κ; Atalay 2017 for ρ, ζ), measured exposure anchors from the AR kit's capital-compensation files (equipment+IP income / value added: median 0.15, p90 0.30 across 49 industries), the estimating equations (CLM share regression; the CES cost-share equation; the logit transition equation), a σ_D sensitivity table (results sharpen at CLM's estimates — baseline conservative), and the full data checklist.
- **`gpt_wages_paper.tex`** — LaTeX source (compile: pdflatex → bibtex → pdflatex ×2).
- **`references.bib`** — bibliography (natbib/apalike).
- **`km_revisited.pdf`** — "Katz and Murphy Revisited": the full model. N occupation sectors + one GPT sector (star input–output network), LES demand, sector-specific labor–GPT substitution. Main result: `d ln w_i = [(ε_i−1) dlnA_i + b_i(σ_i−ε_i) dlnp_g + η_i dlnI] / [φ_i + ε_i(1−b_i) + σ_i b_i]` — nests Katz–Murphy `(σ−1)/σ` and this note's `(ε−1)/ε`, and resolves Acemoglu–Autor (2011) puzzles 1–5 (real wage declines, wage & employment polarization, occupational reallocation, machine displacement) without tasks. Includes a calibrated 4-occupation polarization example.
- **`km_revisited.tex`** — LaTeX source.
- **`verify_gpt_network.py`** — numerical companion: solves the exact nonlinear GE and verifies every formula in `km_revisited` by finite differences (Prop 1 with φ=0 and φ>0, own-productivity terms, the 2×2 aggregate closure, Hulten's theorem, Katz–Murphy nesting). All match to ~1e-9. Requires numpy + scipy.
- **`verify_mobility.py`** — companion for the worker-mobility/ripple extension (Section 5 of `km_revisited`): M worker groups with logit occupational choice over overlapping portfolios. Verifies the ripple wage system `[Λ−R] ŵ = f`, group wage indices `Ŵ_m = Σ π_mi ŵ_i`, reallocation `κ_m(ŵ_i−Ŵ_m)`, the propagation matrix Neumann series (minimal Acemoglu–Restrepo 2022 ripple), growth accounting, and a 3-occupation chain showing displacement attenuating with network distance.
- **`verify_full_ge.py`** — companion for the production-network + heterogeneous-households model. The intermediate bundle is CES with elasticity ρ (ρ=1 Cobb–Douglas, used by `km_revisited`; ρ=0.5 the paper's **overhead bundle**, under which customer j's derived-demand elasticity for input i is ρ(1−ω̃_ji)+ζ_j·ω̃_ji ≈ 0.55 — inelastic clerical demand microfounded from downstream technology). Verifies (for ρ ∈ {1, 0.5}): the stacked block-linear system; Leontief price exposure (prices only — the displacement identity dlnL = dlnY − ζs(dlnq−dlnP^M) − σθ(dlnw−dlnp_g) keeps the direct margins); Gorman aggregation under common β; growth accounting; exact LES welfare.
- **`README.md`** — this file.

### One-command replication

```
cd code && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && ./run_all.sh
```

### Rebuilding the PDF

```bash
pdflatex wages_note.tex && pdflatex wages_note.tex
```

(Two passes are needed for the `fillbetween` shading. Requires TeX Live with `pgfplots`.)

---

## Verification

Every formula was checked numerically (finite-difference comparative statics against closed forms): the CES revenue share and its `(σ−1)/σ` elasticity; the `(ε−1)/ε` wage response; and the full Stone–Geary GE (both markets clear, `ε_ii = 1 − (1−β)γ/C`, and `d ln w1/d ln A1 = −γ1/(A1L1−γ1)`). All matched to ~1e-5 or better.
