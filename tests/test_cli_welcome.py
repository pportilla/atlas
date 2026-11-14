import io
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.app import main
from engine.welcome import LOGO, SMALL_LOGO, WORDMARK, atlas_version, render_welcome


ROOT = Path(__file__).resolve().parents[1]


class Terminal(io.StringIO):
    def isatty(self):
        return True


class CliWelcomeTests(unittest.TestCase):
    def assert_logo(self, output, logo):
        rows = output.splitlines()
        start = next(i for i, line in enumerate(rows) if line.endswith(logo[0]))
        self.assertEqual([row[-len(mark):] for row, mark in zip(rows[start:], logo)], list(logo))

    def test_bare_launch_with_redirected_streams(self):
        env = dict(os.environ, TERM='xterm', COLUMNS='80')
        env.pop('NO_COLOR', None)
        result = subprocess.run(
            [sys.executable, str(ROOT / 'atlas')], input='unknown-command\n',
            capture_output=True, text=True, env=env, timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, '')
        rows = result.stdout.splitlines()
        wordmark_end = next(i for i, row in enumerate(rows) if WORDMARK[-1] in row)
        version = next(i for i, row in enumerate(rows) if row.lstrip().startswith('version ' + atlas_version()))
        contact = next(i for i, row in enumerate(rows) if row.lstrip().startswith('pportilla <p.portilla@upm.es>'))
        self.assertLess(wordmark_end, version)
        self.assertLess(version, contact)
        self.assertNotIn('Developer', result.stdout)
        self.assertNotIn('Resolved curve', result.stdout)
        self.assertIn('Welcome to Atlas', result.stdout)
        self.assertNotIn('Try ', result.stdout)
        self.assertNotIn('./atlas resolve', result.stdout)
        self.assertNotIn('atlas>', result.stdout)
        self.assertNotIn('\x1b', result.stdout)

    def test_version_falls_back_for_a_source_download(self):
        with patch('engine.welcome.Path.exists', return_value=False), \
                patch('engine.welcome.subprocess.check_output') as git:
            self.assertEqual(atlas_version(), 'development')
        git.assert_not_called()

    def test_welcome_art_fits_terminal_width(self):
        for version in ('abc1234', 'release-' + 'b' * 160):
            with patch('engine.welcome.atlas_version', return_value=version):
                for width in range(1, 121):
                    with self.subTest(width=width, version=version):
                        banner = render_welcome(width, color=False, interactive=True)
                        self.assertTrue(banner.isascii())
                        self.assertLessEqual(max(map(len, banner.splitlines())), width)

    def test_compact_logo_survives_side_by_side_and_stacked_layouts(self):
        for width in (18, 50):
            with self.subTest(width=width):
                output = render_welcome(width, color=False, interactive=False)
                self.assert_logo(output, SMALL_LOGO)

    def test_paper_map_has_three_divisors_and_the_strict_transform(self):
        for logo in (LOGO, SMALL_LOGO):
            with self.subTest(logo=logo):
                drawing = '\n'.join(logo)
                self.assertEqual(drawing.count('o'), 3)
                self.assertEqual(drawing.count('^'), 1)
                self.assertNotRegex(drawing, r'[0-9A-Z<>v]')
                row = next(i for i, line in enumerate(logo) if 'o' in line)
                chain = re.search(r'o-+o-+o', logo[row])
                self.assertIsNotNone(chain)
                column = logo[row].index('o', chain.start() + 1)
                tip = next(i for i, line in enumerate(logo) if '^' in line)
                self.assertLess(tip, row)
                self.assertEqual(logo[tip].index('^'), column)
                self.assertTrue(all(line[column] == '|' for line in logo[tip + 1:row]))

    def launch_session(self, commands, term='xterm', no_color='1'):
        stdin, stdout, stderr = Terminal(commands), Terminal(), io.StringIO()
        with (
            patch.object(sys, 'argv', ['atlas']),
            patch.object(sys, 'stdin', stdin),
            patch.object(sys, 'stdout', stdout),
            patch.object(sys, 'stderr', stderr),
            patch.dict(os.environ, {'TERM': term, 'COLUMNS': '80'}),
        ):
            if no_color is None:
                os.environ.pop('NO_COLOR', None)
            else:
                os.environ['NO_COLOR'] = no_color
            self.assertEqual(main(), 0)
        return stdout.getvalue(), stderr.getvalue(), stdin.read()

    def test_prompt_recovers_from_errors_and_quits(self):
        out, err, unread = self.launch_session(
            "unknown-command\nresolve 'unfinished\nhelp resolve\nquit\nafter quit\n"
        )
        self.assertIn('unknown-command', err)
        self.assertIn('Cannot read command', err)
        self.assertIn('--vars', out)
        self.assertIn('atlas>', out)
        self.assertNotIn('\x1b', out)
        self.assertEqual(unread, 'after quit\n')

    def test_eof_and_terminal_color(self):
        for term, has_color in [('xterm', True), ('dumb', False)]:
            with self.subTest(term=term):
                out, err, _ = self.launch_session('', term=term, no_color=None)
                self.assertIn('Welcome to Atlas', out)
                self.assertIn('atlas>', out)
                self.assertEqual('\x1b[' in out, has_color)
                self.assertEqual(err, '')

    def test_both_launch_modes_work_without_the_solver(self):
        with (
            patch('engine.app.calculate_resolution', side_effect=AssertionError('unexpected resolution')) as solve,
            patch('engine.native_library._kernel', side_effect=OSError('missing native kernel')) as kernel,
        ):
            out, err, _ = self.launch_session('help resolve\nquit\n')
            self.assertIn('--vars', out)
            self.assertIn('atlas>', out)
            self.assertEqual(err, '')
            self.assert_logo(out, LOGO)
            redirected = io.StringIO()
            with (
                patch.object(sys, 'argv', ['atlas']),
                patch.object(sys, 'stdin', io.StringIO()),
                patch.object(sys, 'stdout', redirected),
                patch.dict(os.environ, {'COLUMNS': '80'}),
            ):
                self.assertEqual(main(), 0)
            self.assert_logo(redirected.getvalue(), LOGO)
            solve.assert_not_called()
            kernel.assert_not_called()


if __name__ == '__main__':
    unittest.main()
