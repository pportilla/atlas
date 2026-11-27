from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from engine.monodromy import monodromy_zeta
from engine.resolution_complex import dual_complex
from engine.resolve import resolve
from engine.strata import InvalidChart


class CanonicalReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request = dict(polynomial="(x+y)^2+z^3+x^3", variables=list("xyz"), search="quick")
        cls.record = json.loads(json.dumps(resolve(cls.request)))

    def test_missing_and_repeated_pivots_fail_in_both_modes(self):
        self.assertEqual(self.record["engine"], "canonical-blowups")
        for mode in ("embedded", "intrinsic"):
            record = json.loads(json.dumps(resolve(dict(self.request, mode=mode))))
            self.assertEqual(dual_complex(record)["status"], "complete")
            for change in ("omit", "duplicate", "strict transform", "centre", "step limit"):
                with self.subTest(mode=mode, change=change):
                    saved = deepcopy(record)
                    if change == "omit":
                        for k in ("atlas", "charts"):
                            saved[k] = saved[k][-1:]
                    elif change == "duplicate":
                        for k in ("atlas", "charts"):
                            saved[k][0] = saved[k][-1]
                    elif change == "strict transform":
                        saved["charts"][0]["strict_transform"] = ["1"]
                    elif change == "centre":
                        saved["steps"][0]["center_charts"][0]["ideal"] = ["1"]
                    else:
                        saved["steps"].pop()
                    for replay in (dual_complex, monodromy_zeta):
                        with self.assertRaises(InvalidChart):
                            replay(saved)

    def test_intrinsic_sheared_cusp_stops_at_one_and_replays_the_old_three_blowups(self):
        from engine.resolution_zetas import replay_zetas
        request = dict(self.request, polynomial="(x+y+z)^2-(x-y)^3", zetas=["monodromy", "topological"])
        intrinsic = json.loads(json.dumps(resolve(dict(request, mode="intrinsic"))))
        embedded = json.loads(json.dumps(resolve(dict(request, mode="embedded"))))
        # The singular line A=B=0 is the locus of maximal order, which the
        # maximal order rule blows up first along a non-isolated locus.
        self.assertEqual([s["rule"] for s in intrinsic["steps"]], ["maximal order"])
        self.assertEqual([s["rule"] for s in embedded["steps"]], ["maximal order", "surface", "surface"])
        self.assertEqual(intrinsic["steps"][0], embedded["steps"][0])
        self.assertEqual(len(intrinsic["charts"]), 2)
        self.assertIn("the strict transform is smooth", intrinsic["certificate"]["description"])
        self.assertNotIn("strata", intrinsic["certificate"])
        self.assertIn("embedded_companion", intrinsic)
        dual = dual_complex(intrinsic)
        self.assertEqual(dual, intrinsic["dual_complex"])
        self.assertEqual((dual["status"], dual["vertices"], dual["cells"]),
                         ("complete", [dict(dimension=1, id="B1")], []))
        with self.assertRaises(InvalidChart):
            dual_complex(dict(intrinsic, mode="embedded"))

        # Before mode-specific stopping the origin solver serialized this entire
        # three-blowup cover for intrinsic requests too.
        legacy = dict(embedded, mode="intrinsic")
        self.assertEqual(dual_complex(legacy)["vertices"], [dict(dimension=1, id="B3")])
        for saved in intrinsic, legacy:
            replay = replay_zetas(dict(record=saved, zetas=request["zetas"]))
            for kind in request["zetas"]:
                key = kind + "_zeta"
                self.assertEqual(replay[key]["expression"], embedded[key]["expression"])
                self.assertEqual(replay[key]["status"], "computed")

    def test_original_weighted_divisor_and_older_records(self):
        weighted = resolve(dict(self.request, polynomial=f'({self.request["polynomial"]})^3'))
        self.assertEqual(dual_complex(weighted)["status"], "complete")
        self.assertEqual(monodromy_zeta(weighted)["expression"], "(1-T^9)/((1-T^6)*(1-T^18))")
        older = deepcopy(self.record)
        older.pop("atlas")
        self.assertEqual(dual_complex(older), dual_complex(self.record))
        older.pop("steps")
        partial = dual_complex(older)
        self.assertEqual(partial["status"], "partial")
        self.assertFalse(partial["cells"])

    def test_fresh_solve_checks_an_unverified_proposal_without_repeating_its_own_blowups(self):
        with patch("engine.canonical_replay._run", side_effect=AssertionError("duplicated canonical solve")):
            good = resolve(self.request)
        self.assertEqual(good["engine"], "canonical-blowups")
        for mode in ("embedded", "intrinsic"):
            with self.subTest(mode=mode):
                bad = dict(good, mode=mode, charts=good["charts"][-1:], atlas=good["atlas"][-1:])
                # A bounded attempt reports the rejection. A quick request goes on after it,
                # with the projective search: the smooth-centre rules made this proposal.
                with patch("engine.resolve._resolve_polynomial", return_value=bad):
                    result = resolve(dict(self.request, mode=mode, bounded=True))
                self.assertEqual(result["status"], "inconclusive")
                self.assertIn("complete reconstructed cover", result["diagnostic"]["description"])
