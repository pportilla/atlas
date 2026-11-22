"""Exact pole checks through the shared Atlas session."""
import io
import json
import unittest
from contextlib import redirect_stdout
from decimal import Decimal
from fractions import Fraction
from textwrap import dedent
from unittest.mock import Mock, patch

from engine.app import calculate_resolution
from engine.resolve import inconclusive
from engine.session import new_session, print_json
from engine.session_expressions import SessionError
from engine.session_values import Resolution
from engine.strata import resolution_strata
from tests.cli_support import run_prompt, run_script


class SessionMonodromyTests(unittest.TestCase):
    def test_json_at_both_entry_points_includes_cancelled_candidate_data(self):
        source = dedent("""
            x, y = coordinates("x", "y")
            f = y^2 - x^5
            C = monodromy_check(f)
            print_json(poles=C.candidate_poles, label="line\\nbreak", equation=f, data=(2/4, True, 9, None))
        """).lstrip()
        script = run_script(source)
        self.assertEqual((script.returncode, script.stderr), (0, ""))
        status, out, err = run_prompt(source + "quit\n")
        self.assertEqual((status, err), (0, ""))
        self.assertEqual(script.stdout.splitlines(), out.replace("atlas> ", "").strip().splitlines())
        line, = script.stdout.splitlines()
        record = json.loads(line)
        self.assertEqual(line, json.dumps(record, sort_keys=True))
        self.assertEqual(record, {
            "equation": "-x^5+y^2", "label": "line\nbreak", "data": ["1/2", True, 9, None],
            "poles": [{"value": "-1", "components": ["E1", "strict"], "pole_order": 1},
                      {"value": "-4/5", "components": ["E3"], "pole_order": 0},
                      {"value": "-3/4", "components": ["E2"], "pole_order": 0},
                      {"value": "-7/10", "components": ["E4"], "pole_order": 1}]})

    def test_json_preserves_large_integers_and_existing_floats(self):
        value = 10**4501 + 3
        output = io.StringIO()
        with redirect_stdout(output):
            print_json(data={"ratio": Fraction(value, 7), "integer": value, "time": 0.25})
        self.assertEqual(json.loads(output.getvalue(), parse_int=Decimal), {"data": {
            "ratio": f"{Decimal(value)}/7", "integer": Decimal(value), "time": 0.25}})
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaisesRegex(ValueError, "finite"):
            print_json(data=[float("nan")])
        self.assertEqual(output.getvalue(), "")

    def test_signed_orders_h0_and_nearby_witnesses_at_both_entry_points(self):
        source = dedent("""
            x, y, z, w = coordinates("x", "y", "z", "w", coeffs=QQ)
            R = resolve(x^3+y^3+z^3+w^3, search="quick", monodromy=True)
            C = monodromy_check(R)
            p = C.poles[0]
            print(C.status, C.resolution == R, p.value, p.eigenvalue_order, p.cyclotomic_exponent)
            C = monodromy_check(x*y)
            print(C.status, C.poles[0].order, C.poles[0].witness, C.poles[0].cyclotomic_exponent)
            C = monodromy_check((x+y^2)^6*(z+w^2)^4, check_isolation=True)
            p = C.poles[0]
            print(C.status, C.milnor_number, C.monodromy_zeta.expression)
            print(p.eigenvalue_order, p.cyclotomic_exponent, p.witness, p.witness_multiplicity)
        """).lstrip()
        expected = ["verified True -4/3 3 5", "verified 2 origin H^0 0", "verified None 1",
                    "4 0 nearby smooth component 4"]
        script = run_script(source)
        self.assertEqual((script.returncode, script.stderr), (0, ""))
        self.assertEqual(script.stdout.splitlines(), expected)
        status, out, err = run_prompt(source + "quit\n")
        self.assertEqual((status, err), (0, ""))
        self.assertEqual(out.replace("atlas> ", "").strip().splitlines(), expected)

    def test_existing_resolution_replays_strata_without_resolving_again(self):
        calculate = Mock(wraps=calculate_resolution)
        session = new_session(calculate)
        list(session.run('x,y = coordinates("x","y")\nR = resolve(y^2-x^3)'))
        with patch("engine.strata.resolution_strata", wraps=resolution_strata) as replay:
            list(session.run("C = monodromy_check(R)"))
            session.variables["C"]["topological_zeta"]["poles"].clear()
            session.variables["C"]["monodromy_zeta"]["factors"].clear()
            list(session.run("C = monodromy_check(R)"))
            self.assertEqual(replay.call_count, 1)
        self.assertEqual(calculate.call_count, 1)
        result = session.variables["C"]
        self.assertEqual(result["status"], "verified")
        pole = next(p for p in result["poles"] if p["value"] == "-5/6")
        self.assertEqual((pole["witness"], pole["cyclotomic_exponent"]), ("origin zeta", 1))
        self.assertIs(result["resolution"], session.variables["R"])

    def test_imported_resolution_replays_saved_atlas_even_with_stored_zeta_values(self):
        status, record = calculate_resolution("x*y", ["x", "y"], mode="embedded",
                                              search="quick", monodromy=True)
        self.assertEqual(status, 200)
        record["topological_zeta"]["poles"].clear()
        record["monodromy_zeta"]["factors"].append({"multiplicity": 13, "exponent": 1})
        record["monodromy_check"]["status"] = "unresolved"
        session = new_session(None)
        session.variables["R"] = Resolution(record)
        list(session.run("C = monodromy_check(R)\nstatus = R.monodromy_check.status\nfactors = R.monodromy_zeta.factors"))
        self.assertEqual(session.variables["status"], "verified")
        self.assertEqual(session.variables["factors"], [])
        pole, = session.variables["C"]["poles"]
        self.assertEqual((pole["value"], pole["order"], pole["witness"]), ("-1", 2, "origin H^0"))

    def test_five_variable_binomial_retains_two_poles_with_the_same_eigenvalue_order(self):
        session = new_session(calculate_resolution)
        list(session.run('x,y,z,t,u=coordinates("x","y","z","t","u")\n'
                         'C=monodromy_check(x^3*y^6-z^2*t^2*u^4)'))
        result = session.variables["C"]
        self.assertEqual(result["status"], "verified")
        poles = [p for p in result["poles"] if p["eigenvalue_order"] == 12]
        self.assertEqual({p["value"] for p in poles}, {"-7/12", "-5/12"})
        for p in poles:
            self.assertEqual((p["cyclotomic_exponent"], p["witness"]), (0, "nearby transverse binomial"))
            self.assertEqual(p["transverse"]["coordinates"], ("x", "u"))
            self.assertEqual(p["transverse"]["powers"], (3, 4))
            self.assertEqual(p["transverse"]["cyclotomic_exponent"], 1)
            strata = result["monodromy_zeta"]["nearby_strata"]
            matches = {tuple(strata[m["stratum_index"]]["coordinates"]): m["cyclotomic_exponent"]
                       for m in p["nearby_matches"]}
            self.assertEqual(matches, {("x", "u"): 1, ("y", "u"): 2})

    def test_wide_nearby_zeta_at_prompt_and_in_scripts(self):
        source = dedent("""
            x,y,z = coordinates("x","y","z", coeffs=QQ)
            f = (123456789012345678901234567890123456789/7)*x^9007199254740993*y^4
            C = monodromy_check(f, search="exhaustive")
            p = C.poles[1]
            m = p.nearby_matches[0]
            a = C.monodromy_zeta.nearby_strata[m.stratum_index]
            print(C.status, p.value, p.eigenvalue_order, m.cyclotomic_exponent, a.expression)
        """).lstrip()
        expected = "verified -1/9007199254740993 9007199254740993 -1 1/(1-T^9007199254740993)"
        script = run_script(source)
        self.assertEqual((script.returncode, script.stderr, script.stdout.strip()), (0, "", expected))
        status, out, err = run_prompt(source + "quit\n")
        self.assertEqual((status, err, out.replace("atlas> ", "").strip()), (0, "", expected))

    def test_transverse_composition_witnesses_at_prompt_and_in_scripts(self):
        # The order-6 eigenvalue lives on the lines x = y^2+z^2 = 0, where f = u1^2+u2^3.
        source = dedent("""
            x, y, z = coordinates("x", "y", "z")
            C = monodromy_check((x^2+y^2+z^2)^2+x^3)
            p = C.poles[1]
            t = p.transverse
            print(C.status, p.value, p.eigenvalue_order, p.witness)
            print(t.coordinates, t.transverse_coordinates, t.transverse_germ, t.cyclotomic_exponent)
            print(C.monodromy_zeta.nearby_strata[p.nearby_matches[0].stratum_index].expression)
        """).lstrip()
        expected = ["verified -5/6 6 nearby transverse composition",
                    "('x^2+y^2+z^2', 'x') ('u1', 'u2') u2^3+u1^2 1", "(1-T^6)/((1-T^2)*(1-T^3))"]
        script = run_script(source)
        self.assertEqual((script.returncode, script.stderr, script.stdout.splitlines()), (0, "", expected))
        status, out, err = run_prompt(source + "quit\n")
        self.assertEqual((status, err, out.replace("atlas> ", "").strip().splitlines()), (0, "", expected))

    def test_quick_limits_keep_the_reason_and_worker_failures_raise(self):
        request = dict(polynomial="x^200", variables=["x"], mode="embedded", monodromy=True)
        record = inconclusive(request, {"description": "quick worker timed out"}, 2)
        calculate = Mock(side_effect=[(200, record), (422, {"error": "budget exhausted", "code": "computation_limit"}),
                                      (503, {"error": "Worker stopped"}),
                                      (400, {"error": "The germ must pass through the origin."})])
        session = new_session(calculate)
        expr = 'C = monodromy_check("x^200", variables=["x"], search="quick")'
        list(session.run(expr))
        result = session.variables["C"]
        self.assertEqual((result["status"], result["poles"], result["reason"]),
                         ("inconclusive", [], "quick worker timed out"))
        self.assertEqual(result["topological_zeta"]["status"], "inconclusive")
        self.assertEqual(result["monodromy_zeta"]["status"], "inconclusive")
        self.assertIsNone(result["candidate_poles"])
        list(session.run(expr))
        self.assertEqual((session.variables["C"]["reason"], session.variables["C"]["resolution"]),
                         ("budget exhausted", None))
        with self.assertRaisesRegex(SessionError, "Worker stopped"):
            list(session.run(expr))
        with self.assertRaisesRegex(SessionError, "origin"):
            list(session.run('monodromy_check("1+x", variables=["x"])'))

    def test_default_search_and_saved_quick_record_finish_at_prompt_and_in_scripts(self):
        # A degree beyond the quick parser's limit, with a known one-dimensional Milnor fibre.
        source = ('x=coordinates("x")\nf=x^200\nQ=monodromy_check(f,search="quick")\n'
                  'R=Q.resolution\nC=monodromy_check(R)\nD=monodromy_check(f)\n'
                  'print(Q.status,C.status,D.status,C.poles[0].value,R.monodromy_zeta.expression)\n')
        expected = "inconclusive verified verified -1/200 1/(1-T^200)"
        completed = run_script(source)
        self.assertEqual((completed.returncode, completed.stderr, completed.stdout.strip()), (0, "", expected))
        status, out, err = run_prompt(source + "quit\n")
        self.assertEqual((status, err, out.replace("atlas> ", "").strip()), (0, "", expected))

    def test_candidate_requires_nontrivial_missing_order_and_finite_local_number(self):
        # Deliberately inconsistent test double. An actual counterexample is
        # unavailable as a fixture. Keep the true cusp zeta (and its Euler
        # characteristic), but insert a spurious order-five topological pole.
        status, record = calculate_resolution("y^2-x^3", ["x", "y"], mode="embedded",
                                              search="quick", monodromy=True)
        self.assertEqual(status, 200)
        record["topological_zeta"]["poles"] = [{"value": "-2/5", "order": 1, "components": []}]
        session = new_session(Mock(return_value=(200, record)))
        list(session.run('x,y=coordinates("x","y")\nR=resolve(y^2-x^3, monodromy=True)\nC=monodromy_check(R,search="quick")'))
        result = session.variables["C"]
        self.assertEqual((result["status"], result["milnor_number"], result["poles"][0]["witness"]),
                         ("unresolved", None, None))
        list(session.run('C=monodromy_check(R, search="quick", check_isolation=True)'))
        result = session.variables["C"]
        self.assertEqual((result["status"], result["milnor_number"].value), ("isolated_candidate", 2))
        list(session.run('C=monodromy_check(R, check_isolation=True)'))
        result = session.variables["C"]
        self.assertEqual(result["status"], "refuted")
        self.assertEqual(result["local_bernstein_sato"]["by_pole"]["-2/5"],
                         {"root": None, "tested_roots": ["-2/5", "-7/5"]})
        record["monodromy_zeta"]["milnor_fiber_euler_characteristic"] += 1
        list(session.run('R=resolve(y^2-x^3, monodromy=True)\nC=monodromy_check(R, search="quick", check_isolation=True)'))
        self.assertEqual(session.variables["C"]["status"], "unresolved")
        record["polynomial"] = "x^2*y^2"
        list(session.run('R=resolve(x^2*y^2, monodromy=True)\nC=monodromy_check(R, search="quick", check_isolation=True)'))
        result = session.variables["C"]
        self.assertFalse(result["milnor_number"].finite)
        self.assertEqual(result["status"], "unresolved")

    def test_exact_nearby_eigenvalues_at_both_entry_points_and_replay(self):
        source = dedent("""
            x, y, z = coordinates("x", "y", "z", coeffs=QQ)
            f = z^3 + x^4*y + x^5
            R = resolve(f, search="quick", monodromy=True)
            C = monodromy_check(R)
            print(C.status, R.monodromy_check.status, monodromy_check(f).status)
            print(C.local_bernstein_sato.by_pole["-7/12"].root)
        """).lstrip()
        expected = ["verified verified verified", "-7/12"]
        done = run_script(source)
        self.assertEqual((done.returncode, done.stderr, done.stdout.splitlines()), (0, "", expected))
        code, out, err = run_prompt(source + "quit\n")
        self.assertEqual((code, err, out.replace("atlas> ", "").strip().splitlines()), (0, "", expected))
        from engine.resolve_cli import needs_exhaustive_search
        from engine.resolution_zetas import replay_zetas
        _, record = calculate_resolution("z^3+x^4*y+x^5", list("xyz"), mode="embedded", search="quick", monodromy=True)
        self.assertTrue(needs_exhaustive_search({"monodromy": True}, 200, record))
        record["monodromy_check"] = {"status": "refuted", "local_bernstein_sato": {"by_pole": {}}}
        replay = replay_zetas({"record": record, "zetas": ["monodromy", "topological"]})
        self.assertEqual(replay["monodromy_check"]["local_bernstein_sato"]["by_pole"]["-7/12"]["root"], "-7/12")

    def test_local_roots_exclude_distant_points_and_allow_integer_shifts(self):
        from engine.monodromy import _request
        from engine.point_geometry import encode_terms
        from engine.polynomial import parse
        def roots(f, names, values):
            terms = encode_terms(parse(f, list(names), bounded=False))
            return _request("local_eigenvalues", terms=terms, width=len(names), values=values)["by_pole"]
        cases = roots("(x+y)^2*y*(x-1)^3", "xy", ["-1/3", "-2/3", "-1/2"])
        self.assertEqual([v["root"] for v in (cases["-1/3"], cases["-2/3"], cases["-1/2"])], [None, None, "-1/2"])
        self.assertEqual(roots("x^2+y^2+z^2", "xyz", ["-1/2"])["-1/2"],
                         {"tested_roots": ["-1/2", "-3/2"], "root": "-3/2"})
        power = 2**65 + 1
        slope = f"-1/{power}"
        self.assertEqual(roots(f"{2**127}*x^{power}*(1+x)", "x", [slope])[slope]["root"], slope)
