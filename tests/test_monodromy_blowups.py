import unittest

from tests.blowup_monodromy_oracle import blowup_contributions
from engine.resolve import resolve


def atlas(polynomial, variables, mode="embedded"):
    return resolve({"variables":list(variables), "polynomial":polynomial,
                    "mode":mode, "search":"quick"})


def pairs(records):
    return [(item["multiplicity"],item["euler_characteristic"]) for item in records]


class BlowupMonodromy(unittest.TestCase):
    def test_cusp_and_its_weighted_divisor(self):
        result = atlas("y^2-x^3","xy")
        terms = blowup_contributions(result)
        self.assertEqual(pairs(terms),[(2,1),(3,1),(6,-1)])
        self.assertEqual(sum(n*chi for n,chi in pairs(terms)),-1)  # 1−μ
        weighted = atlas("(1+x)^2*(y^2-x^3)^2","xy")
        self.assertEqual(pairs(blowup_contributions(weighted)),[(4,1),(6,1),(12,-1)])
        self.assertIsNone(blowup_contributions(atlas("y^2-x^3","xy","intrinsic")))

    def test_geometric_branches_on_overlapping_charts(self):
        result = atlas("x*y*(x^2+y^2)","xy")
        self.assertTrue(result["algebraic_numbers"])
        record, = blowup_contributions(result)
        self.assertEqual((record["multiplicity"],record["euler_characteristic"]),(4,-2))
        self.assertEqual(record["strict_transform_intersection_points"],4)
        self.assertEqual(record["exceptional_intersection_points"],0)
        self.assertEqual(pairs(blowup_contributions(atlas("x*y","xy"))),[(2,0)])

    def test_ordinary_complement_and_nonreduced_multiplicity(self):
        # A smooth plane quartic has genus 3. Its projective complement
        # has χ=3−(−4)=7. A smooth quartic K3 has χ=24 in P³.
        g = "x^4+y^4+z^4"
        for mode in ("embedded","intrinsic"):
            self.assertEqual(pairs(blowup_contributions(atlas(g,"xyz",mode))),[(4,7)])
        self.assertEqual(pairs(blowup_contributions(atlas(f"(1+x)*({g})^3","xyz"))),[(12,7)])
        record, = blowup_contributions(atlas("x^4+y^4+z^4+t^4","xyzt"))
        self.assertEqual(record["euler_characteristic"], -20)
        self.assertEqual(record["multiplicity"]*record["euler_characteristic"],1-3**4)


if __name__=="__main__":
    unittest.main()
