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
from tests.test_cli_session import run_prompt, run_script


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
        session = new_session(None)
        session.variables["R"] = Resolution(record)
        list(session.run("C = monodromy_check(R)"))
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

    def test_unavailable_records_keep_the_reason_and_invalid_input_still_fails(self):
        request = dict(polynomial="x^200", variables=["x"], mode="embedded", monodromy=True)
        record = inconclusive(request, {"description": "quick worker timed out"}, 2)
        calculate = Mock(side_effect=[(200, record), (422, {"error": "budget exhausted"}),
                                      (400, {"error": "The germ must pass through the origin."})])
        session = new_session(calculate)
        expr = 'C = monodromy_check("x^200", variables=["x"])'
        list(session.run(expr))
        result = session.variables["C"]
        self.assertEqual((result["status"], result["poles"], result["reason"]),
                         ("unavailable", [], "quick worker timed out"))
        self.assertEqual(result["topological_zeta"]["status"], "unavailable")
        self.assertEqual(result["monodromy_zeta"]["status"], "unavailable")
        self.assertIsNone(result["candidate_poles"])
        list(session.run(expr))
        self.assertEqual((session.variables["C"]["reason"], session.variables["C"]["resolution"]),
                         ("budget exhausted", None))
        with self.assertRaisesRegex(SessionError, "origin"):
            list(session.run('monodromy_check("1+x", variables=["x"])'))

    def test_candidate_requires_nontrivial_missing_order_and_finite_local_number(self):
        # Deliberately inconsistent test double. An actual counterexample is
        # unavailable as a fixture. Keep the true cusp zeta (and its Euler
        # characteristic), but insert a spurious order-five topological pole.
        status, record = calculate_resolution("y^2-x^3", ["x", "y"], mode="embedded",
                                              search="quick", monodromy=True)
        self.assertEqual(status, 200)
        record["topological_zeta"]["poles"] = [{"value": "-2/5", "order": 1, "components": []}]
        session = new_session(Mock(return_value=(200, record)))
        list(session.run('x,y=coordinates("x","y")\nR=resolve(y^2-x^3, monodromy=True)\nC=monodromy_check(R)'))
        result = session.variables["C"]
        self.assertEqual((result["status"], result["milnor_number"], result["poles"][0]["witness"]),
                         ("unresolved", None, None))
        list(session.run('C=monodromy_check(R, check_isolation=True)'))
        result = session.variables["C"]
        self.assertEqual((result["status"], result["milnor_number"].value), ("isolated_candidate", 2))
        record["monodromy_zeta"]["milnor_fiber_euler_characteristic"] += 1
        list(session.run('R=resolve(y^2-x^3, monodromy=True)\nC=monodromy_check(R, check_isolation=True)'))
        self.assertEqual(session.variables["C"]["status"], "unresolved")
        record["polynomial"] = "x^2*y^2"
        list(session.run('R=resolve(x^2*y^2, monodromy=True)\nC=monodromy_check(R, check_isolation=True)'))
        result = session.variables["C"]
        self.assertFalse(result["milnor_number"].finite)
        self.assertEqual(result["status"], "unresolved")
