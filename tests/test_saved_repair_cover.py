"""Validate the geometric cover of declared Newton repairs on public replay."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from engine.monodromy import monodromy_zeta
from engine.resolution_complex import dual_complex
from engine.resolve import resolve
from engine.strata import InvalidChart
from engine.topological import topological_zeta


class SavedRepairCoverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = resolve({"polynomial": "x^4-x^2+y^2*z^2", "variables": ["x", "y", "z"],
                              "mode": "intrinsic", "search": "quick"})

    def test_public_replay_requires_every_declared_repair_over_a_deleted_centre(self):
        self.assertEqual(self.record["status"], "resolved")
        for damage in ("uncovered centre", "no repair children", "spurious plane marker", "unmarked orphan",
                       "duplicate pivot", "missing parent cone", "wrong projection", "changed composite map",
                       "empty ownership"):
            with self.subTest(damage=damage):
                record = deepcopy(self.record)
                charts = record["atlas"]
                child = next(c for c in charts if c.get("repair_projection"))
                if damage == "uncovered centre":
                    charts[:] = [c for c in charts if not any(
                        row["id"].startswith("P1.") for row in c["components"])]
                elif damage in ("no repair children", "spurious plane marker"):
                    charts[:] = [c for c in charts if not c.get("repair_projection")]
                    if damage == "spurious plane marker":
                        charts[0]["point_blowup"] = {"parent": -1, "center": []}
                elif damage == "unmarked orphan":
                    parent = next(c for c in charts if c.get("torus") and c["removed"])
                    parent.pop("torus")
                    charts[:] = [parent]
                elif damage == "duplicate pivot":
                    charts.append(deepcopy(child))
                elif damage == "missing parent cone":
                    # Leave the numerical parent index of every child unchanged.
                    outer = next(c for c in reversed(charts) if c.get("torus") and not c["removed"])
                    charts.remove(outer)
                elif damage == "wrong projection":
                    child["repair_projection"]["map"][0] += "+1"
                elif damage == "changed composite map":
                    child["map"]["x"] += "+1"
                else:
                    child["region"].append("1")
                for mode in ("embedded", "intrinsic"):
                    with self.subTest(mode=mode):
                        graph = dual_complex({**record, "mode": mode})
                        self.assertEqual(graph["status"], "partial")
                        self.assertIn("complete cover", graph["reason"])
                        self.assertEqual(graph["cells"], [])

    def test_geometry_only_solve_requires_the_whole_repair_cover(self):
        for mode in ("embedded", "intrinsic"):
            with self.subTest(mode=mode):
                request = dict(polynomial=self.record["polynomial"], variables=self.record["variables"], mode=mode)
                damaged = deepcopy(self.record)
                damaged["mode"] = mode
                damaged["atlas"] = [c for c in damaged["atlas"] if not c.get("repair_projection")]
                with patch("engine.resolve._resolve_polynomial", return_value=damaged):
                    quick = resolve(dict(request, search="quick"))
                self.assertEqual(quick["status"], "inconclusive")
                self.assertIn("complete cover", quick["diagnostic"]["description"])

                replacement = resolve(dict(request, search="quick"))
                self.assertEqual(replacement["status"], "resolved")
                with patch("engine.resolve._resolve_polynomial", return_value=damaged), \
                     patch("engine.general_resolution.search", return_value=replacement) as fallback:
                    result = resolve(dict(request, search="exhaustive"))
                fallback.assert_called_once()
                self.assertEqual(result["status"], "resolved")
                self.assertEqual(result["dual_complex"]["status"], "complete")

    def test_zeta_replay_requires_a_complete_repair_cover(self):
        record = resolve({"polynomial": "z^3+y^4+(1/432)*x^6+x*y^2*z+x^3*y*z+x*y*z^2",
                          "variables": list("xyz"), "mode": "embedded", "monodromy": True})
        for key, compute in (("monodromy_zeta", monodromy_zeta), ("topological_zeta", topological_zeta)):
            with self.subTest(invariant=key):
                expected = record[key]
                saved = {k: v for k, v in record.items() if k not in ("monodromy_zeta", "topological_zeta", "dual_complex")}
                actual = compute(saved)
                self.assertEqual(actual["status"], "computed")
                self.assertEqual(actual["expression"], expected["expression"])
                charts = saved["atlas"]
                first = next(i for i, c in enumerate(charts) if c.get("repair_projection"))
                changed = deepcopy(charts)
                changed[first]["repair_projection"]["map"][0] += "+1"
                for reduced in (charts[:first] + charts[first+1:],
                                [c for c in charts if not c.get("repair_projection")], changed):
                    with self.assertRaisesRegex(InvalidChart, "complete cover"):
                        compute({**saved, "atlas": reduced})
