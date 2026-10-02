# Scenario synthesis report

28 extractions.

## Score distributions

- **capability_pace**: 1:6  2:6  3:8  4:8
- **expertise_erosion**: null:8  0:6  1:2  2:5  3:7
- **productivity_breadth**: 0:4  1:7  2:17
- **substitution**: null:1  0:3  1:12  2:12
- **demand_absorption**: null:9  0:4  1:7  2:8
- **mobility_adjustment**: null:13  0:5  1:10
- **rent_concentration**: null:6  0:5  1:8  2:9

## Source x dimension matrix

| source | capability_p | expertise_er | productivity | substitution | demand_absor | mobility_adj | rent_concent |
|---|---|---|---|---|---|---|---|
| acemoglu-dont-believe-hype | 1 | · | 0 | 1 | · | · | 1 |
| acemoglu-johnson-power | 1 | 0 | 0 | 2 | 0 | 0 | 2 |
| acemoglu-simple-macro | 1 | 1 | 0 | 1 | 1 | · | 1 |
| aghion-jones-jones | 1 | 0 | 1 | 1 | 0 | · | 1 |
| ai-2027 | 4 | 3 | 2 | 2 | 1 | 0 | 2 |
| altman-gentle-singularity | 4 | 2 | 2 | 1 | 2 | 1 | 0 |
| altman-intelligence-age | 4 | 2 | 2 | 1 | 2 | 1 | 0 |
| altman-three-observations | 3 | 2 | 2 | 1 | 2 | 1 | 1 |
| amodei-jobs-warning | 3 | 0 | 1 | 2 | 0 | 0 | · |
| amodei-loving-grace | 4 | 3 | 2 | 1 | 2 | 1 | 0 |
| anthropic-economic-index | 2 | · | 1 | 1 | · | · | · |
| aschenbrenner-situational | 4 | 3 | 2 | 2 | · | · | 2 |
| autor-middle-class | 2 | 2 | 1 | 0 | 2 | 1 | 0 |
| brookings-korinek-work | 3 | 3 | 2 | 2 | · | · | 2 |
| brynjolfsson-turing-trap | 2 | 0 | 2 | 1 | 2 | 1 | 2 |
| dwarkesh-timelines | 3 | · | 2 | 2 | · | · | · |
| eloundou-gpts-gpts | 2 | · | 2 | 1 | 1 | · | · |
| epoch-explosive-growth | 3 | · | 2 | 2 | 2 | · | · |
| forethought-intelligence-explosion | 4 | 3 | 2 | 2 | 2 | 0 | 2 |
| grace-ai-survey | 3 | · | 2 | · | · | · | 1 |
| imf-genai-work | 2 | 0 | 1 | 1 | 1 | 1 | 1 |
| intelligence-curse | 3 | 3 | 2 | 2 | 0 | 0 | 2 |
| karpathy-dwarkesh | 1 | 1 | 0 | 1 | 1 | 1 | · |
| korinek-agi-scenarios | 3 | 0 | 2 | 2 | · | · | 2 |
| mollick-latent-expertise | 2 | 2 | 1 | 0 | · | · | 0 |
| narayanan-kapoor-normal | 1 | · | 1 | 0 | 1 | 1 | 1 |
| sutskever-interviews | 4 | 3 | 2 | 2 | 1 | 1 | 1 |
| trammell-korinek-growth | 4 | · | 2 | 2 | · | · | 2 |

## Greedy grouping (by profile similarity)

### Group around capability_pace=1 (seed: acemoglu-dont-believe-hype)
- acemoglu-dont-believe-hype (Daron Acemoglu)
- acemoglu-johnson-power (Daron Acemoglu & Simon Johnson)
- acemoglu-simple-macro (Daron Acemoglu)
- aghion-jones-jones (Philippe Aghion, Benjamin F. Jones & Charles I. Jones)
- karpathy-dwarkesh (Andrej Karpathy)
- narayanan-kapoor-normal (Arvind Narayanan & Sayash Kapoor)

Implied preset (median scores through rubric maps_to):
  - demand.ag = 3.0   [capability_pace median 1]
  - expertise.scarcity = 0.3   [expertise_erosion median 1]
  - expertise.productivity = 0.5   [productivity_breadth median 0]
  - demand.exposure = 0.5   [productivity_breadth median 0]
  - demand.sig_scale = 1.0   [substitution median 1]
  - demand.eps_spread = 1.0   [demand_absorption median 1]
  - demand.nonhom = 1.0   [demand_absorption median 1]
  - demand.mobility = 1.0   [mobility_adjustment median 1]

### Group around capability_pace=2 (seed: anthropic-economic-index)
- amodei-jobs-warning (Dario Amodei)
- anthropic-economic-index (Anthropic)
- autor-middle-class (David Autor)
- brynjolfsson-turing-trap (Erik Brynjolfsson)
- eloundou-gpts-gpts (Tyna Eloundou, Sam Manning, Pamela Mishkin & Daniel Rock)
- imf-genai-work (Mauro Cazzaniga, Florence Jaumotte, Longji Li, Giovanni Melina, Augustus J. Panton, Carlo Pizzinelli, Emma J. Rockall, Marina Mendes Tavares (IMF))
- mollick-latent-expertise (Ethan Mollick)

Implied preset (median scores through rubric maps_to):
  - demand.ag = 8.0   [capability_pace median 2]
  - expertise.scarcity = 0.0   [expertise_erosion median 0]
  - expertise.productivity = 1.0   [productivity_breadth median 1]
  - demand.exposure = 1.0   [productivity_breadth median 1]
  - demand.sig_scale = 1.0   [substitution median 1]
  - demand.eps_spread = 1.0   [demand_absorption median 1]
  - demand.nonhom = 1.0   [demand_absorption median 1]
  - demand.mobility = 1.0   [mobility_adjustment median 1]

### Group around capability_pace=3 (seed: altman-three-observations)
- altman-gentle-singularity (Sam Altman)
- altman-intelligence-age (Sam Altman)
- altman-three-observations (Sam Altman)
- amodei-loving-grace (Dario Amodei)
- epoch-explosive-growth (Ege Erdil & Tamay Besiroglu (Epoch AI))
- grace-ai-survey (Katja Grace, Harlan Stewart, Julia Fabienne Sandkühler, Stephen Thomas, Ben Weinstein-Raun, Jan Brauner, Richard C. Korzekwa (AI Impacts))

Implied preset (median scores through rubric maps_to):
  - demand.ag = 16.0   [capability_pace median 4]
  - expertise.scarcity = 1.0   [expertise_erosion median 2]
  - expertise.productivity = 1.8   [productivity_breadth median 2]
  - demand.exposure = 1.8   [productivity_breadth median 2]
  - demand.sig_scale = 1.0   [substitution median 1]
  - demand.eps_spread = 0.5   [demand_absorption median 2]
  - demand.nonhom = 1.2   [demand_absorption median 2]
  - demand.mobility = 1.0   [mobility_adjustment median 1]

### Group around capability_pace=4 (seed: ai-2027)
- ai-2027 (Daniel Kokotajlo, Scott Alexander, Thomas Larsen, Eli Lifland, Romeo Dean (AI Futures Project))
- aschenbrenner-situational (Leopold Aschenbrenner)
- brookings-korinek-work (Anton Korinek)
- dwarkesh-timelines (Dwarkesh Patel)
- forethought-intelligence-explosion (William MacAskill & Fin Moorhouse (Forethought))
- intelligence-curse (Luke Drago & Rudolf Laine)
- korinek-agi-scenarios (Anton Korinek & Donghyun Suh)
- sutskever-interviews (Ilya Sutskever)
- trammell-korinek-growth (Philip Trammell & Anton Korinek)

Implied preset (median scores through rubric maps_to):
  - demand.ag = 16.0   [capability_pace median 4]
  - expertise.scarcity = 1.5   [expertise_erosion median 3]
  - expertise.productivity = 1.8   [productivity_breadth median 2]
  - demand.exposure = 1.8   [productivity_breadth median 2]
  - demand.sig_scale = 1.4   [substitution median 2]
  - demand.eps_spread = 1.0   [demand_absorption median 1]
  - demand.nonhom = 1.0   [demand_absorption median 1]
  - demand.mobility = 0.3   [mobility_adjustment median 0]
