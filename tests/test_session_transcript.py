import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.cli_support import ROOT, run_prompt


class SessionTranscriptTests(unittest.TestCase):
    def test_repeated_assignment_resolves_live_names_and_keeps_only_successes(self):
        from engine.session import new_session
        from engine.session_expressions import SessionError, assignment_syntax
        from engine.session_transcript import SessionTranscript

        session, log = new_session(None), SessionTranscript()
        source = "out = op(a, b)"
        self.assertIsNotNone(assignment_syntax(source))
        session.variables.update(op=lambda a, b: a, a=1, b=2)
        list(session.run(source, transcript=log))
        session.variables.update(op=lambda a, b: b, b=9)
        list(session.run(source, transcript=log))
        self.assertEqual(session.variables["out"], 9)

        session.variables.update(op=session.builtins["coordinates"], a="x", b="y")
        with self.assertRaisesRegex(SessionError, "would replace"):
            list(session.run(source, transcript=log))

        def fail_on_two(a, b):
            if a == 2:
                raise ValueError("second iteration")
            return a

        session.variables["op"] = fail_on_two
        with self.assertRaisesRegex(SessionError, "second iteration") as context:
            list(session.run("for a in (1, 2):\n    out = op(a, b)", transcript=log))
        self.assertEqual((context.exception.line, context.exception.function), (2, "op"))
        self.assertEqual((session.variables["a"], session.variables["out"]), ("x", 9))
        self.assertEqual(log.statements, [source, source])

    def test_only_complete_public_calls_bypass_the_full_parser(self):
        from engine.session_expressions import Evaluator, SessionError, assignment_syntax

        session = Evaluator({"f": lambda a, b: a + b})
        session.variables["a"] = 3
        for text in ("class = f(a, 2)", "ans = f(a, 02)", "ans = f(a, 2", "  ans = f(a, 2)", "ans = a ^ 02"):
            with self.subTest(text=text):
                self.assertIsNone(assignment_syntax(text))
                with self.assertRaises(SessionError):
                    list(session.run(text))
        text = "ans = f(a, 2)  # ^ in a comment"
        self.assertIsNone(assignment_syntax(text))
        list(session.run(text))
        self.assertEqual(session.variables["ans"], 5)
        self.assertIsNotNone(assignment_syntax("ans = a^2"))
        self.assertIsNone(assignment_syntax("ans = a^2^3"))
        list(session.run("ans = a^2^3"))
        self.assertEqual(session.variables["ans"], 3**8)

    @unittest.skipUnless(hasattr(sys, "set_int_max_str_digits"), "Requires a configurable decimal literal limit")
    def test_repeated_source_obeys_the_current_decimal_limit(self):
        from engine.session import new_session
        from engine.session_expressions import SessionError

        session = new_session(None)
        source = "n = " + "1" * 650
        previous = sys.get_int_max_str_digits()
        try:
            sys.set_int_max_str_digits(0)
            list(session.run(source))
            sys.set_int_max_str_digits(640)
            with self.assertRaisesRegex(SessionError, "640 digits"):
                list(session.run(source))
        finally:
            sys.set_int_max_str_digits(previous)

    def replay(self, file):
        result = subprocess.run([str(ROOT / "atlas"), "run", str(file)], capture_output=True,
                                text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.splitlines()

    def test_resolution_and_failed_statements(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "my session.atlas"
            prompt = "\n".join([
                'x, y = coordinates("x","y")',
                'R = resolve(y^2-x^3); p=x^2; print("β;", p); bad=1/0; skipped=3',
                'p = p^2',
                'if 1:',
                '    p = 0',
                '    bad = 1/0',
                '',
                'for v in [1,2]:',
                '    print(v,p)',
                '',
                'print(R.branch if True else 0)',
                'print(R.status, R.branches[0].meets[0].id)',
                'help save',
                ':save ' + shlex.quote(str(output)),
                'quit',
            ])
            status, stdout, stderr = run_prompt(prompt)
            self.assertEqual(status, 0)
            self.assertIn('Saved session', stdout)
            self.assertIn('Division by zero', stderr)
            script = output.read_text()
            self.assertIn('R = resolve(y^2-x^3)', script)
            self.assertIn('print("β;", p)', script)
            self.assertIn('for v in [1,2]:\n    print(v,p)', script)
            for omitted in ('help save', ':save', 'bad=', 'skipped', 'p = 0', 'if True'):
                self.assertNotIn(omitted, script)
            self.assertEqual(self.replay(output), ['β; x^2', '1 x^4', '2 x^4', 'resolved E3'])

    def test_repeated_save_and_write_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'count.atlas'
            command = ':save ' + shlex.quote(str(output))
            prompt = '\n'.join([command, 'n = 3', 'print(n)', command,
                                ':save ' + shlex.quote(directory), 'n = 4',
                                'print(n)', command, 'quit'])
            status, _, stderr = run_prompt(prompt)
            self.assertEqual(status, 0)
            self.assertIn('Cannot save session', stderr)
            self.assertEqual(self.replay(output), ['3', '4'])
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_two_normal_forms_reuse_one_named_basis(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'basis.atlas'
            prompt = '\n'.join([
                'x,y=coordinates("x","y")',
                'I=ideal(x^2-1,y-x,order="lex")',
                'G=groebner(I,compute="cpu")',
                'a=normal_form(x^2+y/2,G,compute="cpu")',
                'b=normal_form(y^2-1,G,compute="cpu")',
                'print(a==1+y/2,b==0)',
                ':save ' + shlex.quote(str(output)),
                'quit',
            ])
            status, stdout, stderr = run_prompt(prompt)
            self.assertEqual((status, stderr), (0, ""))
            self.assertIn('True True', stdout)
            self.assertEqual(self.replay(output), ['True True'])
