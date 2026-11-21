"""A geometry proposal must supply the evidence requested by an exhaustive solve."""
import unittest
from unittest.mock import Mock, patch

from engine.resolve import resolve
from engine.resolution_zetas import replay_zetas
from engine.strata import InvalidZetaCertificate


class ResolutionCompletionTests(unittest.TestCase):
    def test_replay_of_a_computed_exponent_preserves_host_decimal_policy(self):
        import sys
        from engine.polynomial import InputError, integer_text, parse
        from engine.monodromy import monodromy_zeta

        limit = sys.get_int_max_str_digits()
        if not limit:
            self.skipTest("This interpreter permits unrestricted decimal literals.")
        power = "1" + "0" * (limit // 2 + 1)
        request = dict(polynomial=f"(x^{power})^{power}", variables=["x"], monodromy=True)
        record = resolve(request)
        expanded = parse(request["polynomial"], ["x"], bounded=False)
        exponent, = next(iter(expanded))
        self.assertGreater(len(integer_text(exponent)), limit)
        self.assertEqual(monodromy_zeta(record, polynomial=expanded)["expression"],
                         f"1/(1-T^{integer_text(exponent)})")
        self.assertEqual(replay_zetas(dict(record=record, zetas=["topological"]))["topological_zeta"],
                         record["topological_zeta"])
        with self.assertRaisesRegex(InputError, "active limit"):
            parse(record["polynomial"], ["x"], bounded=False)
        self.assertEqual(sys.get_int_max_str_digits(), limit)

    def test_coordinate_normal_crossings_in_nine_variables(self):
        variables = [f"x{i}" for i in range(9)]
        record = resolve(dict(polynomial="*".join(variables), variables=variables, monodromy=True))
        self.assertEqual(record["engine"], "normal-crossings-at-origin")
        self.assertEqual(record["topological_zeta"]["poles"][0]["order"], 9)
        self.assertEqual(record["monodromy_zeta"]["expression"], "1")
        self.assertEqual(record["monodromy_check"]["status"], "verified")

    def test_fixed_geometric_effort_gets_increasing_time(self):
        from itertools import islice
        from engine.resolution_search import _retry_efforts
        rounds = list(islice(_retry_efforts(), 64))
        for effort in range(1, 6):
            times = [seconds for seconds, depth in rounds if depth == effort]
            self.assertGreaterEqual(len(times), 2)
            self.assertGreater(times[-1], times[0])

    def test_a_failed_verifier_is_distinct_from_a_timed_out_candidate(self):
        from engine.general_resolution import _run_attempt
        worker = Mock(answered=0)
        worker.ask.side_effect = [None, '["worker_error","NameError: missing verifier",null]']
        with patch("engine.general_resolution._workers") as workers:
            workers.take.return_value = worker
            self.assertEqual(_run_attempt({}, 1), ("undecided", None))
            with self.assertRaisesRegex(RuntimeError, "missing verifier"):
                _run_attempt({}, 1)
            workers.take.side_effect = OSError("no executable")
            with self.assertRaisesRegex(RuntimeError, "no executable"):
                _run_attempt({}, 1)

    def test_failed_zeta_check_resumes_with_an_integrated_certificate(self):
        from engine.general_resolution import search
        from engine.resolution_zetas import zetas_from_strata

        request = dict(variables=list("xyz"), polynomial="(1+z)*(x^2-y^2*z)",
                       search="exhaustive", zetas=["monodromy", "topological"])
        attempts = []

        def read_strata(record, kinds, strata, **options):
            attempts.append(record["engine"])
            if len(attempts) == 1:
                raise InvalidZetaCertificate("The proposed strata failed their integral identity.")
            return zetas_from_strata(record, kinds, strata, **options)

        with patch("engine.resolve.zetas_from_strata", side_effect=read_strata), \
             patch("engine.general_resolution.search", wraps=search) as complete:
            record = resolve(request)
        self.assertGreaterEqual(len(attempts), 2)
        self.assertTrue(complete.call_args.kwargs["integrals"])
        self.assertEqual(record["engine"], "verified-ideal-blowup")
        self.assertEqual(record["monodromy_check"]["status"], "verified")
        self.assertEqual(record["monodromy_zeta"]["expression"], "1/(1-T^2)")
        replay = replay_zetas(dict(record=record, zetas=["monodromy", "topological"]))
        self.assertEqual(replay["topological_zeta"]["expression"], record["topological_zeta"]["expression"])

    def test_intrinsic_geometry_and_its_companion_survive_separate_replay(self):
        request = dict(variables=["x", "y"], polynomial="y^2-x^3", mode="intrinsic")
        geometry = resolve(request)
        for record in (geometry, resolve(dict(request, monodromy=True))):
            results = replay_zetas(dict(record=record, zetas=["topological", "monodromy"]))
            self.assertEqual(results["monodromy_check"]["status"], "verified")
            self.assertEqual(results["topological_zeta"]["expression"], "(4*s + 5)/((s + 1)*(6*s + 5))")
            self.assertEqual(results["topological_zeta"]["source"], "companion embedded resolution")
