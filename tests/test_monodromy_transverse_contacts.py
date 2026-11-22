import unittest

from engine.monodromy import _request
from engine.point_geometry import encode_terms
from engine.polynomial import parse


def contacts(equation, variables="xyz"):
    terms = encode_terms(parse(equation, list(variables), bounded=False))
    rows = _request("transverse_contacts", width=len(variables), terms=terms)
    return {tuple(row["axes"]): (row["contact_order"], row["factors"]) for row in rows}


class TransverseContactsTests(unittest.TestCase):
    def test_noncoordinate_sheets_with_third_order_contact(self):
        self.assertEqual(contacts("(x*z-y^2)*(x*z+y*z-y^2)"),
                         {(1, 2): ("3", [["2", "-1"], ["6", "1"]])})

    def test_symbolic_parameters_exclude_a_specialization_with_higher_contact(self):
        factor = 2**119 + 1
        equation = f"z*y*(y+{factor}*(t-1)*x^3+x^5)"
        self.assertEqual(contacts(equation, "xyzt")[(0, 1)],
                         ("3", [["2", "-1"], ["6", "1"]]))

    def test_repeated_singular_or_three_branches_do_not_give_a_two_sheet_certificate(self):
        for equation in ("x^2*(x+y^4*z)", "x*(x^2+y^3*z)", "x*(x+y^2*z)*(x-y^2*z)"):
            with self.subTest(equation=equation):
                self.assertNotIn((0, 1), contacts(equation))

    def test_wide_exponent_defers_the_shortcut(self):
        self.assertEqual(contacts(f"x*(x+y^{2**80}*z)"), {})
