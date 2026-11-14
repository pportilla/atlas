import io
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.app import main
from engine.welcome import SMALL_WORDMARK, WORDMARK, atlas_version, print_welcome, render_welcome


ROOT = Path(__file__).resolve().parents[1]


class Terminal(io.StringIO):
    def isatty(self):
        return True


class CliWelcomeTests(unittest.TestCase):
    def assert_logo(self, output, logo):
        rows = output.splitlines()
        start = next(i for i, line in enumerate(rows) if line.endswith(logo[0]))
        indent = len(rows[start]) - len(logo[0])
        self.assertEqual(rows[start:start + len(logo)], [' ' * indent + row for row in logo])

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
        self.assert_logo(result.stdout, WORDMARK)
        version = next(i for i, row in enumerate(rows) if row.lstrip().startswith('version ' + atlas_version()))
        contact = next(i for i, row in enumerate(rows) if row.lstrip().startswith('pportilla <p.portilla@upm.es>'))
        self.assertEqual([row.lstrip() for row in rows[:version] if row.strip()],
                         [row.lstrip() for row in WORDMARK])
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
                        colored = render_welcome(width, color=True, interactive=True)
                        self.assertEqual(re.sub(r'\x1b\[[0-9;]*m', '', colored), banner)

    def test_compact_wordmark_and_text_fallback(self):
        width = max(map(len, SMALL_WORDMARK)) + 2
        self.assert_logo(render_welcome(width, color=False, interactive=True), SMALL_WORDMARK)
        narrow = render_welcome(5, color=False, interactive=True)
        self.assertEqual(narrow.lstrip().splitlines()[0], 'ATLAS')

    def launch_session(self, commands, term='xterm', no_color='1', stdout=None):
        stdin, stderr = Terminal(commands), io.StringIO()
        if stdout is None:
            stdout = Terminal()
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

    def test_animation_finishes_at_the_original_wordmark_and_prompt(self):
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted):
                stdout = Terminal()
                frame_ends = []

                def pause(_seconds):
                    frame_ends.append(stdout.tell())
                    if interrupted:
                        raise KeyboardInterrupt

                with (
                    patch.object(stdout, 'fileno', return_value=1),
                    patch('os.get_terminal_size', return_value=os.terminal_size((80, 24))),
                    patch('engine.welcome.time') as clock,
                ):
                    clock.sleep.side_effect = pause
                    out, err, _ = self.launch_session('quit\n', stdout=stdout)

                self.assertEqual(err, '')
                if interrupted:
                    self.assertEqual(len(frame_ends), 1)
                else:
                    frames = [
                        out[start:end].rsplit('\x1b[J', 1)[-1].strip('\r\n').splitlines()
                        for start, end in zip([0] + frame_ends[:-1], frame_ends)
                    ]
                    self.assert_unfolding_sequence(frames)
                final = out[frame_ends[-1]:]
                cleanup = final.index('\x1b[J')
                self.assertIn('\x1b8', final[:cleanup])
                self.assertIn('\x1b[?25h', final[cleanup:])
                self.assertIn('\x1b7', out[:frame_ends[0]])
                self.assert_logo(final[cleanup:], WORDMARK)
                self.assertLess(final.index('Welcome to Atlas'), final.index('atlas>'))
                self.assertEqual(out.count('Welcome to Atlas'), 1)

    def assert_unfolding_sequence(self, frames):
        reveals = []
        blank_widths = []
        wordmark_widths = []
        for rows in frames:
            self.assertNotRegex(''.join(rows), '[A-Za-z]')
            if ':' in ''.join(rows):
                self.assertFalse(wordmark_widths, 'Paper reappeared after the wordmark')
                panels = [row.strip()[1:-1].split(':') for row in rows
                          if row.strip().startswith('|') and row.strip().endswith('|')
                          and row.count(':') == 4]
                if any(cell.strip() for row in panels for cell in row):
                    reveals.append(panels)
                else:
                    self.assertFalse(reveals, 'Lettering disappeared during the reveal')
                    blank_widths.append(max(len(row.strip()) for row in rows))
            else:
                self.assertTrue(reveals, 'Paper disappeared before lettering was drawn')
                ink = [row.strip().replace(' ', '') for row in rows if row.strip()]
                self.assertEqual(ink, [row.replace(' ', '') for row in WORDMARK])
                wordmark_widths.append(max(len(row.strip()) for row in rows))

        self.assertGreater(len(blank_widths), 1)
        self.assertLess(blank_widths[0], blank_widths[-1])
        self.assertEqual(sorted(blank_widths), blank_widths)
        complete = reveals[-1]
        self.assertEqual([''.join(row).replace(' ', '') for row in complete],
                         [row.replace(' ', '') for row in WORDMARK])
        self.assertTrue(all(len(row) == 5 for row in complete))
        self.assertEqual(len(reveals), 4 * (len('ATLAS') - 1) + len(WORDMARK))
        for step, panels in enumerate(reveals):
            with self.subTest(reveal_step=step):
                # A new letter starts every fourth step while the previous one finishes.
                for y, row in enumerate(complete):
                    expected = [cell if step >= 4 * letter + y else ' ' * len(cell)
                                for letter, cell in enumerate(row)]
                    self.assertEqual(panels[y], expected)
        self.assertGreater(len(wordmark_widths), 1)
        self.assertGreater(wordmark_widths[0], wordmark_widths[-1])
        self.assertEqual(sorted(wordmark_widths, reverse=True), wordmark_widths)

    def test_static_welcome_skips_animation(self):
        for stream, interactive, term in ((io.StringIO(), False, 'xterm'),
                                          (Terminal(), False, 'xterm'), (Terminal(), True, 'dumb')):
            with (
                self.subTest(interactive=interactive, term=term),
                patch.object(sys, 'stdout', stream),
                patch.object(stream, 'fileno', return_value=1),
                patch('os.get_terminal_size', return_value=os.terminal_size((80, 24))),
                patch.dict(os.environ, {'TERM': term, 'COLUMNS': '80', 'NO_COLOR': '1'}),
                patch('engine.welcome.time') as clock,
            ):
                print_welcome(interactive=interactive)
                clock.sleep.assert_not_called()
                self.assertNotIn('\x1b', stream.getvalue())
                self.assert_logo(stream.getvalue(), WORDMARK)

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
            self.assert_logo(out, WORDMARK)
            redirected = io.StringIO()
            with (
                patch.object(sys, 'argv', ['atlas']),
                patch.object(sys, 'stdin', io.StringIO()),
                patch.object(sys, 'stdout', redirected),
                patch.dict(os.environ, {'COLUMNS': '80'}),
            ):
                self.assertEqual(main(), 0)
            self.assert_logo(redirected.getvalue(), WORDMARK)
            solve.assert_not_called()
            kernel.assert_not_called()


if __name__ == '__main__':
    unittest.main()
