import unittest

from engine.fan_centres import singular_ray_images
from engine.monomial_atlas import fan_outputs
from engine.polynomial import parse
from engine.resolution_complex import dual_complex
from engine.strata import InvalidChart


class FanCentreReplayTests(unittest.TestCase):
    def test_line_blowup_checks_its_image_in_the_original_coordinates(self):
        names = list("xyz")
        cones = [[3, 1, 2], [0, 3, 2]]
        charts = [dict(coordinates=list("uvw"), ray_indices=cone, map=dict(zip(names, images)))
                  for cone, images in zip(cones, (("u+w", "u*v", "w"), ("u*v+w", "v", "w")))]
        source = "(x-z)^2-y^2*z"
        record = dict(status="resolved", engine="newton-polyhedron", mode="embedded", polynomial=source,
                      variables=names, charts=charts,
                      fan=dict(rays=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], cones=cones),
                      certificate=dict(linear_map_to_input=[[1, 0, 1], [0, 1, 0], [0, 0, 1]]))
        owned, _ = fan_outputs(record, parse(source, names), integrals=False)
        for archive in (record, dict(record, atlas=owned)):
            for mode in ("embedded", "intrinsic"):
                with self.subTest(owned="atlas" in archive, mode=mode):
                    self.assertEqual(dual_complex(dict(archive, mode=mode))["status"], "complete")
                    for polynomial in ("z", "x^2-y^2*z"):
                        with self.assertRaisesRegex(InvalidChart, "smooth locus"):
                            dual_complex(dict(archive, mode=mode, polynomial=polynomial))

    def test_marked_frame_reduces_coefficients_and_checks_the_reduced_support(self):
        matrix = ((1, 0, "a"), (0, 1, 0), (0, 0, 1))
        mark = dict(generator="a", minimal_polynomial="a^2-2")
        def verify(equation):
            return singular_ray_images(equation, list("xyz"), [(1, 1, 0)], matrix, mark)
        self.assertEqual(verify("(x^2-2*z^2)^2-y^2*z"), [["x", "y"]])
        self.assertIsNone(verify("(x^2-3*z^2)^2-y^2*z"))
        self.assertIsNone(verify("y^18446744073709551617"))

    def test_ordinary_point_blowup_of_a_smooth_source(self):
        names, coords = list("xyz"), list("uvw")
        charts = []
        for i, pivot in enumerate(coords):
            mapping = [v if j == i else f"{pivot}*{v}" for j, v in enumerate(coords)]
            charts.append(dict(coordinates=coords, pivot=names[i], map=dict(zip(names, mapping)),
                               exceptional=[{"id": "E1", "equation": pivot}]))
        for mode in ("embedded", "intrinsic"):
            with self.assertRaisesRegex(InvalidChart, "smooth locus"):
                dual_complex(dict(status="resolved", variables=names, polynomial="z", mode=mode, charts=charts))
