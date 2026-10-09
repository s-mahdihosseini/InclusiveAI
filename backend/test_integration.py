"""Regression and economic checks for the website's reviewed model adapter.

Run from the repository root with:
    python -m unittest discover -s backend -p 'test_*.py' -v

The numerical targets are from ai_mrr_reviewed/output/scenario_summary.csv.
They are recorded here so the website tests run without the paper's repository.
Website distribution statistics use 6,000 quadrature cells, versus 20,000 in
those reference results; aggregate equilibria do not depend on quadrature.
"""
from __future__ import annotations

import json
import unittest
from dataclasses import replace

import numpy as np

import ai_future
from mrr_solver import _production_at_capital, solve_equilibrium, stationary_atoms


# Y, K, R, r_B, r_K, labor share, ordinary-capital share, markdown-rent share,
# total-income Gini, labor-income Gini, inverse investor tail exponent.
REFERENCE = {
    'Pre-AI baseline': (
        1.0000000000000009, 2.9742435155021236, 0.07800302792652572,
        0.04883532633469889, 0.06431478230611525, 0.6600000000000001,
        0.23199999999999998, 0.0779999999999998, 0.33794462732364217,
        0.2717958342833682, 0.5332235361973484,
    ),
    'Concentrated AI': (
        1.2484369345423898, 4.930193089027259, 0.09386190861704402,
        0.05436836929806883, 0.07313786466941821, 0.5137171669682555,
        0.37066937094125674, 0.08561346209048742, 0.3949275839707529,
        0.24601440797002105, 0.7192356389191119,
    ),
    'High wage pass-through': (
        1.2200613526682444, 4.642221080919587, 0.09722997192622791,
        0.053336238295730594, 0.07172297206166027, 0.5368553018247467,
        0.36995108843177016, 0.06319360974348291, 0.3856393448044859,
        0.2524769344949489, 0.6933867575663587,
    ),
    'Broad AI ownership': (
        1.2699612267658094, 5.155847076370275, 0.09122012086269449,
        0.059235135185746995, 0.06970996026000065, 0.5139959901053391,
        0.3703396478125679, 0.08566436208209294, 0.34381857921548975,
        0.24611550162030538, 0.43477063920461895,
    ),
    'Inclusive AI': (
        1.2560185821165084, 4.177153228628252, 0.07913620389400876,
        0.053203708882098466, 0.06060435160448141, 0.6321620308941116,
        0.26318404385404165, 0.07465392525184657, 0.3114220865077937,
        0.26802478455375467, 0.2873317530088597,
    ),
}



def appendix_inputs(name: str) -> dict:
    inputs = dict(
        auto="data", aug="data", own=0, mp_low=0, mp_high=0,
        lam_a=0.65, lam_p=0.35, c_p=0.25,
        tier="modest", xi=0.25, chi=0.066,
    )
    if name == "Pre-AI baseline":
        inputs.update(lam_a=0.0, lam_p=0.0, xi=1.0)
    elif name == "High wage pass-through":
        inputs.update(xi=1.0)
    elif name == "Broad AI ownership":
        inputs.update(chi=0.25)
    elif name == "Inclusive AI":
        inputs.update(tier="limited", lam_a=0.5, lam_p=1.0, xi=1.0, chi=0.25)
    return inputs


class WebsiteModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.data = ai_future._base()
        cls.parameters = {}
        cls.equilibria = {}
        cls.responses = {}
        for name in REFERENCE:
            inputs = appendix_inputs(name)
            p, *_ = ai_future.scenario_parameters(**inputs)
            cls.parameters[name] = p
            cls.equilibria[name] = solve_equilibrium(p, compute_distribution=False)
            cls.responses[name] = ai_future.solve(**inputs)

    def test_all_five_appendix_equilibria(self):
        for name, target in REFERENCE.items():
            with self.subTest(scenario=name):
                e = self.equilibria[name]
                actual = (
                    e.output, e.capital, e.capital_rental, e.r_safe, e.r_risky,
                    e.labor_bill / e.output, e.ordinary_capital_share,
                    e.markdown_profit / e.output,
                )
                np.testing.assert_allclose(actual, target[:8], rtol=0, atol=1e-8)
                self.assertAlmostEqual(e.product_profit_share, 0.03, delta=1e-10)
                self.assertAlmostEqual(e.inverse_tail_investors, target[10], delta=1e-8)
                self.assertLess(max(abs(v) for v in e.residuals.values()), 1e-8)

    def test_website_results_match_appendix_statistics(self):
        for name, target in REFERENCE.items():
            with self.subTest(scenario=name):
                result = self.responses[name]["scenario"]
                np.testing.assert_allclose(
                    [result["output"], result["r_safe"], result["r_risky"],
                     result["shares"]["labor"], result["shares"]["ordinary_capital"],
                     result["shares"]["markdown_profit"]],
                    [target[0], target[3], target[4], target[5], target[6], target[7]],
                    rtol=0, atol=1e-8,
                )
                self.assertAlmostEqual(result["gini"]["total"], target[8], delta=1e-5)
                self.assertAlmostEqual(result["gini"]["labor"], target[9], delta=1e-5)

    def test_baseline_calibration_uses_paid_labor_cost(self):
        e = self.equilibria["Pre-AI baseline"]
        self.assertAlmostEqual(e.output, 1.0, delta=1e-10)
        np.testing.assert_allclose(
            e.wages / (self.base.psi * e.capital_rental),
            np.full(len(self.base.masses), 1.3), atol=1e-10,
        )
        np.testing.assert_allclose(
            e.wages / float(self.base.masses @ e.wages),
            self.data["mean_wage"] / float(self.base.masses @ self.data["mean_wage"]),
            atol=1e-10,
        )
        response = self.responses["Pre-AI baseline"]
        np.testing.assert_allclose(response["baseline"]["wages"], response["scenario"]["wages"], atol=1e-10)
        np.testing.assert_allclose(response["deciles"]["wage_change"], 0.0, atol=1e-10)

    def test_exposure_tilts_are_bounded_and_preserve_means(self):
        for tier in ("limited", "modest", "broad"):
            for kind in ("auto", "aug"):
                observed = self.data[f"{kind}_{tier}"]
                for answer in ("data", -2, -1, 0, 1, 2):
                    with self.subTest(tier=tier, kind=kind, answer=answer):
                        profile = ai_future.tilted_profile(observed, self.base.eta, answer)
                        self.assertTrue(np.all(np.isfinite(profile)))
                        self.assertTrue(np.all((profile >= 0) & (profile <= 1)))
                        self.assertAlmostEqual(float(self.base.eta @ profile),
                                               float(self.base.eta @ observed), delta=2e-12)
                        if answer == "data":
                            np.testing.assert_array_equal(profile, observed)

    def test_augmentation_gain_and_compute_are_distinct_quantities(self):
        for lam_p, c_p in ((0.0, 0.9), (1.0, 0.0), (1.0, 0.9), (0.37, 0.25)):
            with self.subTest(lam_p=lam_p, c_p=c_p):
                p, _, q, gain = ai_future.scenario_parameters(
                    "data", "data", 0, 0, 0, 0.5, lam_p, c_p)
                savings = lam_p * q
                np.testing.assert_allclose(p.augmentation_gain, 1 / (1 - savings), atol=1e-14)
                np.testing.assert_allclose(gain, p.augmentation_gain, atol=1e-14)
                np.testing.assert_allclose(p.augmentation_compute_requirement, c_p * savings, atol=1e-14)
                np.testing.assert_array_equal(p.psi, self.base.psi)
                if lam_p == 0 or c_p == 0:
                    e = solve_equilibrium(p, compute_distribution=False)
                    self.assertAlmostEqual(e.augmentation_capital, 0.0, delta=1e-12)

    def test_wage_sharing_and_legacy_markdowns_have_explicit_mappings(self):
        for xi in (0.0, 0.25, 1.0):
            with self.subTest(xi=xi):
                p, _, _, gain = ai_future.scenario_parameters(
                    "data", "data", 0, 0, 0, 0.65, 0.35, xi=xi)
                np.testing.assert_allclose(
                    1 - p.wage_markdown,
                    (1 - self.base.wage_markdown) * gain ** (xi - 1),
                    atol=1e-13,
                )
        p, *_ = ai_future.scenario_parameters("data", "data", 0, -1, 2, 0.5, 0.0)
        np.testing.assert_allclose(p.wage_markdown,
                                   ai_future.markdown_schedule(self.base.wage_markdown, -1, 2), atol=1e-14)

    def test_shadow_cost_adoption_conditions_and_labor_clearing(self):
        for name, p in self.parameters.items():
            with self.subTest(scenario=name):
                e = self.equilibria[name]
                d = _production_at_capital(p, e.capital)["technique_details"]
                v = d["v"]
                aa, gg = d["automation_adoption"], d["augmentation_adoption"]
                for fractions in (aa, gg):
                    self.assertTrue(np.all((fractions >= 0) & (fractions <= 1)))
                auto_labor, auto_capital = v, np.ones_like(v)
                other_labor = v
                augmented = v / p.augmentation_gain + p.augmentation_compute_requirement
                for regret in (
                    aa * (auto_capital - np.minimum(auto_labor, auto_capital)),
                    (1 - aa) * (auto_labor - np.minimum(auto_labor, auto_capital)),
                    gg * (augmented - np.minimum(other_labor, augmented)),
                    (1 - gg) * (other_labor - np.minimum(other_labor, augmented)),
                ):
                    self.assertLess(float(np.max(np.abs(regret))), 1e-9)
                np.testing.assert_allclose(d["labor_implied"], p.masses, atol=1e-10)
                self.assertAlmostEqual(d["capital"], e.capital, delta=1e-9)
                self.assertLessEqual(e.automation_capital_share, float(p.eta @ p.alpha) + 1e-10)

    def test_markdowns_redistribute_at_fixed_capital_and_leave_adoption_unchanged(self):
        # Each case has an interior technique mixture at fixed K and labor.
        # A markdown changes wage payments and owner rents in equal amounts.
        p = replace(
            self.base, productivity=1.0, product_markup=1.0,
            masses=np.ones(1), eta=np.ones(1), psi=np.ones(1),
            alpha=np.array([0.5]), augmentation_gain=np.array([2.0]),
            augmentation_compute_requirement=np.array([0.2]),
            wage_markdown=np.zeros(1),
        )
        cases = (
            (p, "automation_adoption", 2 / 7),
            (replace(p, alpha=np.array([0.2]), augmentation_gain=np.array([1.2])),
             "augmentation_adoption", 0.375),
        )
        for parameters, mixed_key, mixed_fraction in cases:
            with self.subTest(technique=mixed_key):
                low = _production_at_capital(parameters, 0.4)
                high = _production_at_capital(
                    replace(parameters, wage_markdown=np.array([0.5])), 0.4)
                self.assertAlmostEqual(low["technique_details"][mixed_key][0],
                                       mixed_fraction, delta=1e-10)
                for key in ("output", "capital_rental", "shadow_wages",
                            "automation_capital", "augmentation_capital"):
                    np.testing.assert_allclose(high[key], low[key], atol=1e-12)
                for key in ("automation_adoption", "augmentation_adoption", "v",
                            "labor_implied", "capital"):
                    np.testing.assert_allclose(high["technique_details"][key],
                                               low["technique_details"][key], atol=1e-12)
                np.testing.assert_allclose(high["wages"], 0.5 * low["wages"], atol=1e-12)
                self.assertAlmostEqual(high["markdown_profit"] - low["markdown_profit"],
                                       low["labor_bill"] - high["labor_bill"], delta=1e-12)

    def test_payments_and_stationary_net_income_conserve_resources(self):
        for name, p in self.parameters.items():
            with self.subTest(scenario=name):
                e = self.equilibria[name]
                self.assertAlmostEqual(e.output,
                    e.labor_bill + e.capital_rental * e.capital + e.markdown_profit + e.product_profit,
                    delta=1e-9)
                atoms = stationary_atoms(p, e)
                w = atoms["weights"]
                self.assertAlmostEqual(float(w.sum()), 1.0, delta=1e-12)
                self.assertAlmostEqual(float(w @ atoms["bonds"]), 0.0, delta=2e-8)
                self.assertAlmostEqual(float(w @ atoms["equity"]), e.capital, delta=2e-8)
                self.assertAlmostEqual(float(w @ atoms["labor"]), e.labor_bill, delta=2e-8)
                self.assertAlmostEqual(float(w @ atoms["total"]), e.output - p.delta * e.capital, delta=2e-8)
                after = self.responses[name]["scenario"]
                shares = after["shares"]
                self.assertAlmostEqual(sum(shares[k] for k in
                    ("labor", "ordinary_capital", "markdown_profit", "product_profit")), 1.0, delta=1e-10)
                self.assertAlmostEqual(shares["automation_capital"] + shares["augmentation_capital"],
                                       shares["ordinary_capital"], delta=1e-10)
                self.assertAlmostEqual(after["capital_income_share_of_income"],
                                       e.r_risky * e.capital / (e.output - p.delta * e.capital), delta=2e-8)
                bins = after["percentiles"]
                np.testing.assert_allclose(np.asarray(bins["labor"]) + np.asarray(bins["capital"]),
                                           bins["total"], atol=1e-10)

    def test_existing_positional_api_and_control_corners_return_finite_json(self):
        cases = (
            ("data", "data", 0, 0, 0, 0.5, 0.5, 0.25),
            (-2, -2, -2, 2, 2, 1.0, 1.0, 0.9),
            (2, 2, 2, -2, -2, 1.0, 1.0, 0.9),
            (-2, "data", -2, -2, 2, 1.0, 0.0, 0.0),
            ("data", -2, 2, 0, 0, 0.0, 1.0, 0.0),
        )
        # Full broad-tier automation can place individual frontiers near one.
        # Exercise those solves as well as checking profile bounds separately.
        calls = [(args, {}) for args in cases] + [
            ((-2, -2, 0, 0, 0, 1.0, 1.0, 0.0), dict(tier="broad", chi=0.035, xi=0.0)),
            ((2, 2, 0, 0, 0, 1.0, 1.0, 0.9), dict(tier="broad", chi=0.6, xi=1.0)),
            ((-2, 2, 0, 0, 0, 1.0, 1.0, 0.9), dict(tier="broad", chi=0.6, xi=1.0)),
            ((2, -2, 0, 0, 0, 1.0, 1.0, 0.0), dict(tier="broad", chi=0.035, xi=0.0)),
        ]
        required = {"inputs", "baseline", "scenario", "deciles", "percentile_labels", "max_residual"}
        for args, overrides in calls:
            with self.subTest(inputs=args, overrides=overrides):
                result = ai_future.solve(*args, **overrides)
                self.assertTrue(required <= result.keys())
                json.dumps(result, allow_nan=False)
                self.assertLess(result["max_residual"], 1e-8)
                self.assertEqual(len(result["scenario"]["wages"]), len(self.base.masses))
                self.assertEqual(len(result["scenario"]["percentiles"]["total"]), len(result["percentile_labels"]))
        for result in self.responses.values():
            json.dumps(result, allow_nan=False)

    def test_high_ownership_curve_point_returns_finite_json(self):
        # The appendix ownership curve reaches chi=.6. This point needs the
        # robust financial solve is available if the Newton starts do not converge.
        inputs = appendix_inputs("Concentrated AI")
        inputs["chi"] = 0.6
        result = ai_future.solve(**inputs)
        scenario = result["scenario"]
        np.testing.assert_allclose(
            [scenario["capital"], scenario["r_safe"], scenario["r_risky"]],
            [5.275699496227166, 0.06264573610703184, 0.06798755597402638],
            atol=1e-8, rtol=0,
        )
        self.assertLess(result["max_residual"], 1e-8)
        self.assertEqual(scenario["chi"], 0.6)
        np.testing.assert_allclose(
            result["deciles"]["private_labor_cost_ratio"],
            np.asarray(result["deciles"]["employer_labor_cost"])
            / (scenario["capital_rental"] * self.base.psi),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result["deciles"]["shadow_labor_cost_ratio"],
            np.asarray(result["deciles"]["labor_resource_value"])
            / (scenario["capital_rental"] * self.base.psi),
            atol=1e-12,
        )
        json.dumps(result, allow_nan=False)

    def test_metadata_presets_reproduce_all_appendix_cases(self):
        presets = ai_future.meta()["presets"]
        self.assertEqual(len(presets), len(REFERENCE))
        matched = set()
        for preset in presets:
            self.assertTrue({"id", "label", "inputs"} <= preset.keys())
            inputs = preset["inputs"]
            self.assertEqual(inputs.get("auto", "data"), "data")
            self.assertEqual(inputs.get("aug", "data"), "data")
            self.assertEqual(inputs.get("mp_low", 0), 0)
            self.assertEqual(inputs.get("mp_high", 0), 0)
            result = ai_future.solve(**inputs)
            candidates = [name for name, target in REFERENCE.items()
                          if abs(result["scenario"]["output"] - target[0]) < 1e-8]
            self.assertEqual(len(candidates), 1, preset["id"])
            matched.add(candidates[0])
        self.assertEqual(matched, set(REFERENCE))


if __name__ == "__main__":
    unittest.main()
