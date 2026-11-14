"""Point metadata, typed results and portable replay across the public interfaces."""
import json
import shlex
import unittest

from engine.session import new_session
from engine.session_expressions import SessionError
from tests.large_coupled_ideal import COUPLED_SQUARES, EXPONENT, POINT, VARIABLES
from tests.cli_support import run_script, shell
from tests.http_support import endpoint, post


class LocalDimensionInterfaceTests(unittest.TestCase):
    def test_wide_coupled_ideal_cli_and_script(self):
        arguments = ["--vars", ",".join(VARIABLES), "--order", "grevlex",
                     "--point", ",".join(POINT), "--compute", "cpu", "--json"]
        for generator in COUPLED_SQUARES:
            arguments.extend(["--gen", generator])
        for operation in ("local-dimension", "local-smoothness"):
            with self.subTest(operation=operation):
                cli = shell(["./atlas", "algebra", operation, *arguments])
                self.assertEqual(cli.returncode, 0, cli.stdout + cli.stderr)
                record = json.loads(cli.stdout)
                result = record["result"]
                self.assertEqual((result["scope"], result["point"], result["local_dimension"]), ("point", POINT, 0))
                if operation == "local-dimension":
                    self.assertEqual((result["type"], result["value"]), ("integer", 0))
                else:
                    self.assertIs(result["value"], False)
                    self.assertEqual((result["embedding_dimension"], result["jacobian_rank"]), (3, 0))
                self.assertIn(str(EXPONENT), record["replay"]["script"])
                replay = run_script(record["replay"]["script"])
                self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
                self.assertEqual(json.loads(replay.stdout)["result"], result)

    def test_junction_and_smooth_branch_above_translation_cap(self):
        # At (1,1,0) these are the three axes in the local coordinates
        # (x^1001-1, y^1001-1, z). At z=2 only the z-axis remains.
        u, v = "(x^1001-1)", "(y^1001-1)"
        with endpoint() as url:
            for z, smooth, embedding in (("0", False, 3), ("2", True, 1)):
                with self.subTest(z=z):
                    point = ["1", "1", z]
                    status, record = post(url, "local-smoothness",
                                          {"ideal": [f"{u}*{v}", f"{u}*z", f"{v}*z"], "point": point},
                                          ring={"variables": VARIABLES}, compute="auto")
                    self.assertEqual(status, 200, record)
                    result = record["result"]
                    self.assertEqual((result["local_dimension"], result["embedding_dimension"], result["point"]),
                                     (1, embedding, point))
                    self.assertIs(result["value"], smooth)

    def test_different_components_http_cli_and_replay(self):
        # The plane x=0 and the line y=z=0 meet only at the origin.
        ring = {"field": "QQ", "variables": ["x", "y", "z"], "order": "lex"}
        with endpoint() as url:
            for point, dimension, embedding in (
                (["0", "1/2", "0"], 2, 2), (["2/3", "0", "0"], 1, 1),
                (["0", "0", "0"], 2, 3), (None, 2, 3),
            ):
                operands = {"ideal": ["x*y", "x*z"]}
                if point is not None:
                    operands["point"] = point
                for operation in ("local-dimension", "local-smoothness"):
                    with self.subTest(operation=operation, point=point):
                        status, record = post(url, operation, operands, ring=ring)
                        self.assertEqual(status, 200, record)
                        result = record["result"]
                        self.assertEqual(result["local_dimension"], dimension)
                        self.assertEqual(result["scope"], "origin" if point is None else "point")
                        self.assertEqual(result.get("point"), point)
                        if operation == "local-smoothness":
                            self.assertIs(result["value"], dimension == embedding)
                            self.assertEqual(result["embedding_dimension"], embedding)
                            self.assertEqual(result["jacobian_rank"], 3 - embedding)
                        else:
                            self.assertEqual(result["value"], dimension)
                        command = shlex.split(record["replay"]["command"])
                        cli = shell(command)
                        self.assertEqual(cli.returncode, 0, cli.stderr)
                        self.assertEqual(json.loads(cli.stdout)["result"], result)
                        self.assertEqual(record["execution"]["backend"], "cpu")
                        # Exercise one CLI-generated replay for each operation.
                        if point == ["2/3", "0", "0"]:
                            replay = run_script(json.loads(cli.stdout)["replay"]["script"])
                            self.assertEqual(replay.returncode, 0, replay.stderr)
                            self.assertEqual(json.loads(replay.stdout)["result"], result)
                            command.remove("--json")
                            text = shell(command)
                            self.assertIn("Local dimension: 1", text.stdout)
                            if operation == "local-smoothness":
                                self.assertIn("Smooth at point: yes", text.stdout)

    def test_saved_objects_and_fields(self):
        session = new_session(None)
        list(session.run("""
x,y = coordinates("x","y")
I = ideal((y-2)^2-(x-1/2)^3)
B = local_standard_basis(I,point=(1/2,2))
C = tangent_cone(B)
d = local_dimension(B)
s = local_smoothness(B)
v = local_smoothness(C)
a = local_smoothness(I, point=(3/2,3))
"""))
        d, s, v, a = (session.variables[k] for k in "dsva")
        self.assertEqual((d.local_dimension, d.value, d.point), (1, 1, ["1/2", "2"]))
        self.assertEqual((s.value, s.embedding_dimension, s.jacobian_rank, s.point), (False, 2, 0, d.point))
        self.assertEqual((v.value, v.local_dimension, v.point), (False, 1, ["0", "0"]))
        self.assertEqual((a.value, a.embedding_dimension, a.local_dimension), (True, 1, 1))
        for op in ("local_dimension", "local_smoothness"):
            for arg in ("C, point=(1,2)", "B, point=(0,0)", "I, point=(0,0)"):
                with self.subTest(operation=op, args=arg), self.assertRaisesRegex(SessionError, "point"):
                    list(session.run(f"{op}({arg})"))
