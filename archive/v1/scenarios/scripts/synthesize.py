#!/usr/bin/env python3
"""
Aggregate extractions/*.json into a synthesis report to inform scenarios.json.

Usage (from the scenarios/ directory):
    python3 scripts/synthesize.py

Produces synthesis_report.md: per-dimension score distributions, a source x
dimension matrix, simple k-like grouping of sources by score profile, and the
implied parameter presets per group (using rubric.json's maps_to tables).
The final scenarios.json is a HUMAN-REVIEWED artifact: edit it by hand (or with
Claude) using this report; do not overwrite it blindly.
"""

import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = os.path.dirname(HERE)
EXTRACT = os.path.join(SCEN, "extractions")

DIMS = ["capability_pace", "expertise_erosion", "productivity_breadth",
        "substitution", "demand_absorption", "mobility_adjustment",
        "rent_concentration"]


def load():
    out = []
    for f in sorted(os.listdir(EXTRACT)):
        if f.endswith(".json"):
            with open(os.path.join(EXTRACT, f), encoding="utf-8") as g:
                out.append(json.load(g))
    return out


def profile(e):
    return tuple(e["dimensions"].get(d, {}).get("score") for d in DIMS)


def dist(v, w):
    """Distance between two score profiles, ignoring nulls."""
    pairs = [(a, b) for a, b in zip(v, w) if a is not None and b is not None]
    if not pairs:
        return 99.0
    return sum(abs(a - b) for a, b in pairs) / len(pairs)


def main():
    ex = load()
    if not ex:
        print("no extractions found — run extract.py first")
        return
    with open(os.path.join(SCEN, "rubric.json"), encoding="utf-8") as f:
        rubric = json.load(f)

    lines = ["# Scenario synthesis report", "",
             f"{len(ex)} extractions.", ""]

    # 1. score distributions
    lines += ["## Score distributions", ""]
    for d in DIMS:
        counts = defaultdict(int)
        for e in ex:
            s = e["dimensions"].get(d, {}).get("score")
            counts["null" if s is None else s] += 1
        row = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=str))
        lines.append(f"- **{d}**: {row}")
    lines.append("")

    # 2. matrix
    lines += ["## Source x dimension matrix", "",
              "| source | " + " | ".join(d[:12] for d in DIMS) + " |",
              "|" + "---|" * (len(DIMS) + 1)]
    for e in ex:
        p = profile(e)
        lines.append(f"| {e['source_id']} | "
                     + " | ".join("·" if s is None else str(s) for s in p) + " |")
    lines.append("")

    # 3. greedy grouping around seeds (capability_pace anchors the debate)
    lines += ["## Greedy grouping (by profile similarity)", ""]
    seeds = {}
    for e in ex:
        cp = e["dimensions"].get("capability_pace", {}).get("score")
        if cp is not None and cp not in seeds:
            seeds[cp] = e
    groups = defaultdict(list)
    for e in ex:
        best = min(seeds, key=lambda k: dist(profile(e), profile(seeds[k])))
        groups[best].append(e)
    for k in sorted(groups):
        lines.append(f"### Group around capability_pace={k} "
                     f"(seed: {seeds[k]['source_id']})")
        for e in groups[k]:
            lines.append(f"- {e['source_id']} ({e.get('author','?')})")
        # implied parameters: median score per dim -> maps_to
        lines.append("")
        lines.append("Implied preset (median scores through rubric maps_to):")
        for d in DIMS:
            scores = [e["dimensions"].get(d, {}).get("score") for e in groups[k]]
            scores = sorted(s for s in scores if s is not None)
            if not scores:
                continue
            med = scores[len(scores) // 2]
            maps = rubric["dimensions"][d].get("maps_to", {})
            for target, table in maps.items():
                if isinstance(table, dict) and str(med) in table:
                    lines.append(f"  - {target} = {table[str(med)]}   "
                                 f"[{d} median {med}]")
        lines.append("")

    out = os.path.join(SCEN, "synthesis_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
