"""Left-right coordinate invariance of the extended tangent quotient."""
import re
import shutil
import unittest

from tests.test_local_algebra_singular import calculate, singular


@unittest.skipUnless(shutil.which("Singular"), "Singular is not installed")
class AeCoordinateChanges(unittest.TestCase):
    def compare(self, variables, components):
        maps = [components]
        if variables == ("x", "y"):
            # A triangular source automorphism fixing the origin.
            maps.append(tuple(re.sub(r"\bx\b", "(x+2*y^2)", f) for f in components))
        # Replace the last target coordinate by itself plus the square of the
        # first, then cycle it to the front. The inverse is again polynomial.
        maps.append((f"({components[-1]})+({components[0]})^2", *components[:-1]))
        # coDimMap uses standard bases internally. Keep Singular's default
        # reduction flags for these local modules.
        commands = ['option(noredSB);', 'LIB "classifyMapGerms.lib";', "ideal f;", "list d;"]
        for polynomials in maps:
            commands += [f"f={','.join(polynomials)};", "d=coDimMap(f);", 'print("ANSWER "+string(d[2]));']
        expected = singular("(c,ds)", commands, variables)
        self.assertEqual(len(expected), len(maps))
        self.assertEqual(expected, [expected[0]] * len(maps))
        for index, polynomials in enumerate(maps):
            with self.subTest(map=polynomials):
                result = calculate("ae-codimension", variables=variables, components=list(polynomials),
                                   order="lex" if index == 1 else "grevlex")
                self.assertIs(result["finite"], True, result)
                self.assertEqual(result["value"], expected[index])

    def test_surface_maps(self):
        for components in (
            ("x", "y^2"), ("x", "y^3+x*y"), ("x", "y^3+x^2*y"),
            ("x", "y^3-x^2*y"), ("x", "y^4+x*y"), ("x", "y^5+x*y"),
            ("x", "y^3+x^3*y"), ("x", "y^2", "x*y"), ("x", "y^2", "x^2*y+y^3"),
        ):
            self.compare(("x", "y"), components)

    def test_curve_parametrizations(self):
        for components in (
            ("t^2", "t^3"), ("t^3", "t^4"), ("t^2", "t^7"),
            ("t^3", "t^5", "t^7"), ("t^4", "t^6", "t^9"),
        ):
            self.compare(("t",), components)
