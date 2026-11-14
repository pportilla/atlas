import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.cli_support import ROOT, run_prompt


class SessionTranscriptTests(unittest.TestCase):
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
