import io
import os
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path
from random import Random
from unittest.mock import patch

from engine.app import main
from engine.identity_atlas import rational_factors
from engine.plane_backend import resolve_plane
from engine.polynomial import parse
from engine.welcome import LOGO, SMALL_LOGO, atlas_version, render_welcome
from engine.welcome_curve import resolve_welcome_curve
from engine.welcome_graph import graph_rows


ROOT = Path(__file__).resolve().parents[1]


class Terminal(io.StringIO):
    def isatty(self):
        return True


def divisors_on_arrow_path(rows):
    """Count exceptional vertices on the connected path between the two arrowheads."""
    cells = {(x, y): glyph for y, row in enumerate(rows)
             for x, glyph in enumerate(row) if glyph != ' '}
    start, end = (p for p, glyph in cells.items() if glyph in '<>^v')
    pending = [(start, 0)]
    visited = {start}
    # Both glyphs must have a stroke in the direction of travel.
    links = [(1, 0, 'o+-<', 'o+->'), (-1, 0, 'o+->', 'o+-<'),
             (0, 1, 'o+|^', 'o+|v'), (0, -1, 'o+|v', 'o+|^')]
    while pending:
        (x, y), count = pending.pop()
        if (x, y) == end:
            return count
        for dx, dy, outgoing, incoming in links:
            neighbor = (x + dx, y + dy)
            if (cells[x, y] in outgoing and neighbor not in visited
                    and cells.get(neighbor, ' ') in incoming):
                visited.add(neighbor)
                pending.append((neighbor, count + (cells[neighbor] == 'o')))
    raise AssertionError("The two branch arrows are disconnected.")


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
        first, second, *_ = result.stdout.strip().splitlines()
        self.assertEqual(first, 'Atlas version ' + atlas_version())
        self.assertIn('Developer: pportilla <p.portilla@upm.es>', second)
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
        example = resolve_welcome_curve(Random(0))
        for width in range(1, 121):
            with self.subTest(width=width):
                banner = render_welcome(width, color=False, interactive=True, example=example)
                self.assertTrue(banner.isascii())
                self.assertLessEqual(max(map(len, banner.splitlines())), width)

    def test_compact_logo_survives_side_by_side_and_stacked_layouts(self):
        for width in (18, 50):
            with self.subTest(width=width):
                output = render_welcome(width, color=False, interactive=False, example=("x*y", {}))
                self.assert_logo(output, SMALL_LOGO)

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

    def test_each_launch_computes_once_and_uses_its_own_example(self):
        examples = [resolve_welcome_curve(Random(seed)) for seed in (0, 1)]
        with patch('engine.welcome.resolve_welcome_curve', side_effect=examples) as compute:
            for polynomial, _ in examples:
                out, err, _ = self.launch_session('quit\n')
                self.assertIn(polynomial, out)
                self.assertNotIn('Try ', out)
                self.assert_logo(out, LOGO)
                self.assertEqual(err, '')
        self.assertEqual(compute.call_count, len(examples))

    def test_unavailable_solver_leaves_prompt_usable(self):
        with patch('engine.welcome.resolve_welcome_curve', side_effect=RuntimeError('missing worker')):
            out, _, _ = self.launch_session('help resolve\nquit\n')
        self.assertIn(LOGO[0], out)
        self.assertIn('missing worker', out)
        self.assertIn('--vars', out)
        self.assertIn('atlas>', out)

    def test_sampled_germs_have_two_branches_and_at_most_two_pairs(self):
        variables = ['x', 'y']
        seen = set()
        for seed in range(8):
            polynomial, result = resolve_welcome_curve(Random(seed))
            with self.subTest(polynomial=polynomial):
                poly = parse(polynomial, variables)
                self.assertEqual(poly, parse(result['polynomial'], variables))
                self.assertEqual(result['mode'], 'embedded')
                self.assertEqual(result['dual_complex']['status'], 'complete')
                self.assertEqual(len(result['branches']), 2)
                factors = rational_factors(poly, variables)
                self.assertEqual([multiplicity for _, multiplicity in factors], [1, 1])
                pairs = []
                for factor, _ in factors:
                    if min(sum(exponents) for exponents in factor) == 1:
                        pairs.append(0)
                        continue
                    branch = resolve_plane(factor, variables, 'embedded')
                    self.assertEqual(len(branch['branches']), 1)
                    # A plane branch has one rupture vertex per Puiseux pair.
                    # Its valency counts the strict transform as an incident edge.
                    degree = Counter(v for c in branch['dual_complex']['cells'] for v in c['vertices'])
                    degree.update(branch['branches'][0]['ambient_components'])
                    pairs.append(sum(d >= 3 for d in degree.values()))
                self.assertLessEqual(sum(pairs), 2)
                seen.add(tuple(sorted(pairs)))
        self.assertIn((0, 2), seen)
        self.assertGreater(len(seen), 1)

    def test_both_branch_arrows_survive_at_every_graph_size(self):
        variables = ['x', 'y']
        cases = [('y*(y-x^3)', ['E3', 'E3'], 1),
                 ('(y^2-x^3)*(x^2-y^3)', ['E3', 'E5'], 3)]
        for polynomial, attachments, nodes_between_arrows in cases:
            result = resolve_plane(parse(polynomial, variables), variables, 'embedded')
            self.assertCountEqual([b['ambient_components'][0] for b in result['branches']], attachments)
            for width in (5, 13, 29):
                with self.subTest(polynomial=polynomial, width=width):
                    rows = graph_rows(result, width)
                    self.assertLessEqual(max(map(len, rows)), width)
                    drawing = ''.join(rows)
                    self.assertEqual(drawing.count('o'), len(result['dual_complex']['vertices']))
                    self.assertEqual(sum(c in '<>^v' for c in drawing), 2)
                    self.assertEqual(divisors_on_arrow_path(rows), nodes_between_arrows)

    def test_distinct_attachments_share_a_horizontal_path_on_wide_paper(self):
        variables = ['x', 'y']
        curve = parse('(y^2-x^3)*(x^2-y^3)', variables)
        rows = graph_rows(resolve_plane(curve, variables, 'embedded'), 29)
        # The unique path between the two attachment points is E3--E1--E5.
        self.assertRegex('\n'.join(rows), r'<-{1,2}o-+o-+o-{1,2}>')

    def test_shared_attachment_has_short_up_and_right_arrows_on_wide_paper(self):
        variables = ['x', 'y']
        curve = parse('y*(y-x^3)', variables)
        rows = graph_rows(resolve_plane(curve, variables, 'embedded'), 29)
        tip_rows = [i for i, row in enumerate(rows) if '^' in row]
        self.assertEqual(len(tip_rows), 1)
        top = tip_rows[0]
        column = rows[top].index('^')
        vertical = ''.join(row.ljust(column + 1)[column] for row in rows[top:])
        self.assertRegex(vertical, r'^\^\|{1,2}o')
        attachment_row = rows[top + vertical.index('o')]
        self.assertRegex(attachment_row[column:], r'^o-{1,2}>')


if __name__ == '__main__':
    unittest.main()
