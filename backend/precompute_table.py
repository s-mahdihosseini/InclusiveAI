"""Precompute the equilibrium table used by ai_future.warm_start.

Run from the backend directory:
    python precompute_table.py            # builds or resumes data/equilibrium_table.json
    python precompute_table.py --check 50 # re-solves 50 random entries from scratch

The table stores, for every combination of the four question answers under the
default advanced settings and under each preset's advanced settings, the three
log unknowns (log K, log B_S, log(r_B-g)) of the balanced-growth equilibrium.
The website uses them only as Newton starting points; every served equilibrium
is still verified to the solver's 2e-12 tolerance.  Rebuild the table after any
change to mrr_solver.py, technology.py, reference_solver.py, the calibration
file, or the question-to-parameter mapping in ai_future.py.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

import ai_future
from mrr_solver import equilibrium_unknowns, solve_equilibrium

HERE = os.path.dirname(os.path.abspath(__file__))
ANSWERS = (-2, -1, 0, 1, 2)
ANSWERS_WITH_DATA = ("data",) + ANSWERS
SOURCE_FILES = ("mrr_solver.py", "technology.py", "reference_solver.py", "ai_future.py",
                os.path.join("data", "ai_occupation_calibration.csv"))


def source_hash() -> str:
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        with open(os.path.join(HERE, name), "rb") as stream:
            digest.update(stream.read())
    return digest.hexdigest()


def advanced_settings() -> dict:
    """Default custom settings plus each preset's advanced settings (deduplicated)."""
    keys = ("lam_a", "lam_p", "c_p", "tier", "xi", "chi")
    settings = {"custom_default": {k: ai_future.PRESET_DEFAULTS[k] for k in keys}}
    for preset in ai_future.PRESETS:
        candidate = {k: preset["inputs"][k] for k in keys}
        if candidate not in settings.values():
            settings[preset["id"]] = candidate
    return settings


def work_units() -> list:
    """(label, settings, auto, aug) units; each solves the own x mp_low x mp_high block."""
    units = []
    for label, s in advanced_settings().items():
        if float(s["lam_a"]) == 0.0 and float(s["lam_p"]) == 0.0:
            tilts = [("data", "data")]          # tilts are irrelevant without AI intensity
        else:
            tilts = list(itertools.product(ANSWERS_WITH_DATA, ANSWERS_WITH_DATA))
        for auto, aug in tilts:
            units.append((label, s, auto, aug))
    return units


def solve_unit(unit) -> tuple:
    label, s, auto, aug = unit
    owns = (0,) if s["chi"] is not None else ANSWERS   # an exact chi overrides the dial
    rows, failures, guess = [], [], None
    started = time.time()
    for own in owns:
        for mp_low in ANSWERS:
            for mp_high in ANSWERS:
                args = (auto, aug, own, mp_low, mp_high, s["lam_a"], s["lam_p"], s["c_p"],
                        s["tier"], s["xi"], s["chi"])
                key = ai_future.table_key(*args)
                try:
                    p, *_ = ai_future.scenario_parameters(*args)
                    e = solve_equilibrium(p, compute_distribution=False, initial_guess=guess)
                    guess = equilibrium_unknowns(p, e)
                    rows.append((key, guess.tolist(), max(abs(v) for v in e.residuals.values())))
                except Exception as error:  # record and continue
                    failures.append((key, repr(error)))
    return label, rows, failures, time.time() - started


def build(workers: int) -> None:
    path = ai_future.TABLE_FILE
    table = {"model_version": ai_future.meta()["model_version"], "source_sha256": source_hash(),
             "created": time.strftime("%Y-%m-%d %H:%M:%S"), "entries": {}, "failures": []}
    if os.path.exists(path):
        with open(path) as stream:
            previous = json.load(stream)
        if previous.get("source_sha256") == table["source_sha256"]:
            table["entries"] = previous.get("entries", {})
            print(f"resuming: {len(table['entries'])} entries already stored", flush=True)
    units = [u for u in work_units()
             if any(ai_future.table_key(u[2], u[3], own, ml, mh, u[1]["lam_a"], u[1]["lam_p"],
                                        u[1]["c_p"], u[1]["tier"], u[1]["xi"], u[1]["chi"])
                    not in table["entries"]
                    for own in ((0,) if u[1]["chi"] is not None else ANSWERS)
                    for ml in ANSWERS for mh in ANSWERS)]
    print(f"{len(units)} work units, {workers} workers", flush=True)
    worst = 0.0
    with Pool(workers) as pool:
        for i, (label, rows, failures, seconds) in enumerate(pool.imap_unordered(solve_unit, units), 1):
            for key, unknowns, residual in rows:
                table["entries"][key] = unknowns
                worst = max(worst, residual)
            table["failures"].extend(failures)
            print(f"[{i}/{len(units)}] {label}: {len(rows)} solved, {len(failures)} failed, "
                  f"{seconds:.1f}s; total {len(table['entries'])}", flush=True)
            if True:  # save after every unit; the build may be interrupted and resumed
                table["max_residual"] = worst
                table["count"] = len(table["entries"])
                tmp = path + ".tmp"
                with open(tmp, "w") as stream:
                    json.dump(table, stream, separators=(",", ":"))
                os.replace(tmp, path)
    print(f"done: {len(table['entries'])} entries, {len(table['failures'])} failures, "
          f"max residual {worst:.2e}", flush=True)


def check(sample: int) -> None:
    """Re-solve random entries from the fixed starts and compare the unknowns."""
    table = ai_future._table()
    keys = random.Random(0).sample(sorted(table["entries"]), sample)
    worst = 0.0
    for key in keys:
        auto, aug, own, mp_low, mp_high, lam_a, lam_p, c_p, tier, xi, chi = key.split("|")
        auto = auto if auto == "data" else int(auto)
        aug = aug if aug == "data" else int(aug)
        chi_value = None if chi == "none" else float(chi)
        p, *_ = ai_future.scenario_parameters(auto, aug, int(own), int(mp_low), int(mp_high),
                                              float(lam_a), float(lam_p), float(c_p), tier,
                                              float(xi), chi_value)
        e = solve_equilibrium(p, compute_distribution=False)   # fixed starts, no warm start
        difference = float(np.max(np.abs(equilibrium_unknowns(p, e) - np.asarray(table["entries"][key]))))
        worst = max(worst, difference)
    print(f"{sample} entries re-solved from scratch; max |difference in log unknowns| = {worst:.2e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        check(int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    else:
        build(int(sys.argv[1]) if len(sys.argv) > 1 else max(1, (os.cpu_count() or 2)))
