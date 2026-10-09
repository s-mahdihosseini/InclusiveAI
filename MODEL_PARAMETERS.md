# Website controls and the appendix model

The website runs the reviewed task model in `backend/mrr_solver.py` and
`backend/technology.py`. The five named presets reproduce the revised appendix's
balanced-growth equilibria. The four-question layout also supports additional
comparisons of the distribution of task opportunities and wage markdowns.

## Controls, equations, and experiments

The occupation index is `j`, the fixed production weight is `eta_j`, and the
baseline automation frontier is `alpha_0 = 0.2391752577`. The task-data profiles
are automation opportunities `d_j` and conditional time-saving opportunities
`e_j`.

| Website control | API field | Appendix object and equation | Comparison |
|---|---|---|---|
| Which jobs could be automated? | `auto` | Profile `d_j`, holding `sum(eta_j*d_j)` fixed | Incidence of feasible automation across occupation-wage groups |
| Which workers could save time? | `aug` | Profile `e_j`, holding `sum(eta_j*e_j)` fixed | Incidence of augmentation opportunities |
| Ownership access | `own`, optional `chi` | Fraction `chi` of each occupation with risky-equity access | Capital supply, returns, and ownership concentration |
| Employer power, low/high wages | `mp_low`, `mp_high` | Additional multiplier `M_j` on the wage markdown | Occupation-specific wage-setting experiment |
| Automation opportunity tier | `tier` | Limited: T4; modest: T3–T4; broad: T2–T4 | Which tasks enter `d_j`; `e_j` averages over the remaining set |
| Automation feasibility | `lam_a` | `alpha_j = alpha_0 + (1-alpha_0)*lambda_A*d_j` | Expansion of the technological frontier |
| Augmentation time savings | `lam_p` | `s_j = lambda_P*e_j`; `G_j = 1/(1-s_j)` | Labor time saved on an augmented task |
| AI capital requirement | `c_p` | `m_j = c_P*s_j` | Capital required per unit of augmented task output |
| Wage sharing | `xi` | `tau_sharing_j = 1-(1-tau_0_j)*G_j^(xi-1)` | Appendix wage-sharing comparative static |

The final wage markdown is `tau_j = min(.95, M_j*tau_sharing_j)` and the wage is
`w_j = (1-tau_j)*omega_j`, where `omega_j` is labor's marginal resource value
in production. The employer's labor payment is `w_j`. At fixed capital and
occupation employment, markdowns shift income between workers and owners.
Their effect on saving and capital accumulation can change technology use in
general equilibrium.
The employer-power answer sets `M_j` to 0.25, 0.6, 1, 1.5, or 2. Low-wage
answers apply to deciles 1–4, high-wage answers to deciles 9–10, and deciles 5–8
use their geometric mean. All appendix presets use `M_j=1`. The `xi` control
then implements the appendix's wage-sharing equation exactly.

The ownership answer selects 3.5, 5, 6.6, 15, or 30 percent. An exact `chi`
value overrides that answer; clicking the ownership dial clears the override.
The appendix's broader-ownership cases use `chi=.25`. Participation is constant
within an equilibrium and independent of occupation.

The `data` choice uses the observed task profile. Other answers construct a
profile proportional to `1+t*z_j`, where wage rank `z_j` runs from -1 to 1
and `t` takes values -0.9, -0.45, 0, 0.45, or 0.9. Rescaling preserves its
production-weighted mean. If the profile reaches its upper bound, the remaining
groups are rescaled to preserve that mean. Profiles remain below one, ensuring
finite augmentation gains at full intensity. The appendix's incidence table uses
sorted task profiles and a uniform profile; the website adds intermediate linear
tilts with the same fixed-mean interpretation.

## Feasibility and adoption

Firms compare capital costs at the solved rental `R` with the shadow value
of labor `omega_j`:

- Automate feasible tasks when `R < omega_j/psi_j0`.
- Augment remaining tasks when `R*m_j < (omega_j/psi_j0)*(1-1/G_j)`.

The two labor-technique costs are `omega_j/psi_j0` and
`omega_j/(psi_j0*G_j)+R*m_j`. Employment is fixed within each occupation,
so a worker’s time freed by automation or augmentation can be used on another
task. The value of that time is `omega_j`.

At fixed capital, wage markdowns change wage payments and owner rents while
leaving output, task allocation, and resource values unchanged. The employer-power
and wage-sharing controls affect adoption through saving, capital accumulation,
and the resulting equilibrium resource values. Task allocation and these values
are solved jointly with the asset-market equilibrium.

The chosen fractions are `theta_j` for capital on feasible automation tasks and
`z_j` for augmentation on remaining tasks. At cost ties, firms can use a mixture.
The decile table reports feasible automation `alpha_j`, actual automation
`alpha_j*theta_j`, productivity `G_j`, compute requirement `m_j`, augmentation
use `z_j`, and wage markdown `tau_j`. The API also reports `employer_labor_cost`,
`labor_resource_value`, `private_labor_cost_ratio = w_j/(R*psi_j0)`, and
`shadow_labor_cost_ratio = omega_j/(R*psi_j0)` for each occupation. The shadow
ratio governs adoption; the private ratio records wage payments. Baseline labor productivity `psi_j0` remains
available as a technique after AI arrives.

`c_P` and `m_j` are quantities in the model's capital units. The share of output
paid to augmentation capital is `R*K_G/Y`, determined in equilibrium.

## Exact appendix presets

All presets use observed task profiles, `c_P=.25`, and neutral employer-power
multipliers. The page's initial custom settings retain the previous site's
`lambda_A=lambda_P=.5`, modest tier, and baseline ownership, with `xi=1`.

| Preset | Tier | `lambda_A` | `lambda_P` | `xi` | `chi` |
|---|---|---:|---:|---:|---:|
| Pre-AI baseline | Modest (zero AI intensity) | 0 | 0 | 1 | .066 |
| Concentrated AI | Modest | .65 | .35 | .25 | .066 |
| Higher wage sharing | Modest | .65 | .35 | 1 | .066 |
| Broader ownership | Modest | .65 | .35 | .25 | .25 |
| Combined scenario | Limited | .50 | 1 | 1 | .25 |

Select a preset and change one control to study its consequences. Changing an
answer switches the selector to Custom settings. The pre-AI comparison remains
the same calibrated economy throughout.

## Reproducing the main comparative statics

For the automation curve, set the modest tier, observed profiles, `lambda_P=0`,
`xi=1`, baseline ownership, and neutral employer-power multipliers; vary
`lambda_A` from 0 to 1. For the other three curves, start with Concentrated AI
and vary just `c_P`, `xi`, or exact `chi`. The paper's ownership points are
`.04, .066, .10, .15, .25, .40, .60`.

The API accepts six decimal places, allowing the exact grid points used in the
appendix even when a slider uses coarser steps. For example, the following request
reproduces the Concentrated AI specification with `c_P=.09375`:

```text
/api/future/solve?auto=data&aug=data&own=0&mp_low=0&mp_high=0&lam_a=.65&lam_p=.35&c_p=.09375&tier=modest&xi=.25&chi=.066
```

The wage-sharing-incidence comparison in the appendix varies `xi_j` across
occupations. The current website supplies a common `xi` and separate low/high
markdown multipliers, providing an additional way to study wage-setting incidence.

## Income and ownership measures

Household labor income is `w_j`; net capital income is `r_K*a+r_B*b`; total
income is their sum. Aggregate total income is `Y-delta*K`. Capital rentals
and both business-rent flows are included in the consolidated equity return.
The factor-share ribbon reports gross output shares, while household income
shares use net income.

The top one percent equity share ranks households by equity and matches the
appendix's ownership measure. Net worth is equity plus bonds. Effective wealth
adds the present value of wages; `kappa=a/x` is the equity share of effective
wealth. The income chart compares percentile groups ranked within each
equilibrium. Group membership can change across the equilibria.

## Calibration, sources, and model updates

O*NET supplies tasks. Eloundou et al. (2024) supply exposure measures; Hosseini
and Lichtinger (2026) supply updated automation ratings. The calibration CSV is
identical to the paper's maintained file. Baseline financial parameters follow
Moll, Rachel, and Restrepo (2022). Wage-markdown calibration uses the same
Seegmiller-based schedule and aggregate rent target as the appendix.

Baseline labor productivity is calibrated using
`psi_j0 = w_j0/(1.30*R_0)`. This calibrates the baseline private labor cost
relative to automation capital, using the wage after the markdown.
`reference_solver.py` reproduces the initial normalization. All website
counterfactuals use the task-allocation routine at shadow labor and capital
costs. The paid-wage normalization of `psi_j0` is retained to preserve the
existing calibration; it implies a baseline shadow ratio `1.30/(1-tau_j0)`. The wage-income, markup, and markdown-rent calibration targets
and financial parameters are retained.

To synchronize a subsequent model revision, copy the reviewed `solver.py` to
`backend/mrr_solver.py`, together with `technology.py` and `reference_solver.py`,
then review `ai_future.py`'s calibration and parameter map. Run the integration
tests before publishing. The website is standalone and needs no path to the
paper's repository at runtime.

```bash
.venv/bin/python -m unittest discover -s backend -p 'test_*.py' -v
```

The tests compare all five appendix equilibria, verify baseline calibration,
check shadow-cost adoption, markdown invariance at fixed capital, income
accounting, and exercise control
extremes. Website statistics use 6,000 quadrature cells; the paper uses 20,000.
The equilibrium quantities and returns agree within `1e-8`, and tested Ginis
agree within `1e-5`.
