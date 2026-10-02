# InclusiveAI — Expert Scenarios Pipeline

Maps what technologists, economists, and forecasters say about future AI onto
concrete parameter presets for the InclusiveAI models.

## Layout

```
scenarios/
├── rubric.json          # Extraction schema: 7 scored dimensions + timeline +
│                        #   inequality prediction, each with anchored scales and
│                        #   maps_to tables linking scores to model parameters
├── sources.csv          # Master source list (~50 entries; priority 1 = core)
├── corpus/              # Cleaned full text, one .txt per source  (28 collected)
├── extractions/         # Rubric extraction JSON per source        (28 done)
├── synthesis_report.md  # Auto-generated: score matrix, clusters, implied presets
├── scenarios.json       # THE DELIVERABLE: 5 named scenarios with parameter
│                        #   presets + source attribution (human-reviewed)
└── scripts/
    ├── download_corpus.py  # local: fetch sources.csv → corpus/
    ├── extract.py          # local: Anthropic API rubric extraction → extractions/
    └── synthesize.py       # aggregate extractions → synthesis_report.md
```

## The five scenarios (wave 1)

| Scenario | Anchored in | Expertise preset | Demand preset |
|---|---|---|---|
| Normal Technology | Narayanan-Kapoor, Acemoglu, Aghion-Jones-Jones, Karpathy | scarcity 0.3, prod 0.5 | A_g 3, exposure 0.5 |
| Expertise Democratized | Autor, Mollick, Brynjolfsson, Anthropic Economic Index | scarcity 1.3, prod 1.0 | A_g 6, σ-scale 0.6, mobility 1.5 |
| White-Collar Displacement | Amodei (jobs warning), IMF, Eloundou et al., Acemoglu-Johnson | scarcity 0.8, prod 1.2 | A_g 8, σ-scale 1.4, mobility 0.4 |
| Transformative AGI, Broadly Shared | Amodei (Loving Grace), Altman ×3, Sutskever, Epoch, Grace survey | scarcity 1.5, prod 1.8 | A_g 16, ε compressed |
| Concentrated AGI / Intelligence Curse | Drago-Laine, AI-2027, Aschenbrenner, Forethought, Korinek | scarcity 1.5, prod 2.0 | A_g 16, mobility 0.3, exposure 2.0 |

Attribution rule: scenarios are "consistent with views expressed by" the listed
sources — never "author X's parameters." The mapping table (rubric.json) is the
reviewable scientific object.

## Scaling up (local, no chat tokens)

```bash
cd scenarios
pip install requests beautifulsoup4 pypdf anthropic
python3 scripts/download_corpus.py --priority 3      # fetch everything in sources.csv
export ANTHROPIC_API_KEY=...
python3 scripts/extract.py                            # rubric extraction per document
python3 scripts/synthesize.py                         # regenerate the score matrix
# then hand-edit scenarios.json in light of synthesis_report.md
```

Add sources by appending rows to sources.csv (papers, podcast transcripts,
interviews — anything with a fetchable URL). For paywalled/PDF-only items, drop
the text manually into corpus/<id>.txt with the standard header.

## Caveats for the paper

- Most experts don't speak in elasticities; scores are judgment calls anchored by
  verbatim quotes stored in extractions/. Spot-check before citing.
- Dimension `rent_concentration` is recorded but not yet mapped to a model — it
  awaits the AI Market Power tab.
- Wave 1 = 28 sources, single-extractor. For publication: second extractor pass
  (different model or human) + inter-rater agreement on the score matrix.
