"""Geometric boundary curves on a smooth surface in a regular threefold fan.

For an internal ray, its divisor is a toric surface over the origin. In a
maximal cone its intersection with the strict surface is a plane curve. A
coordinate factor is the orbit closure of the corresponding two-dimensional
cone. Every other geometrically irreducible factor has a dense open in the
ray's torus. Two such curves agree precisely when their Laurent equations in
the common character lattice differ by a unit. These two descriptions glue
the actual curves, including distinct branches in the same ambient divisor.

The Jacobian criterion on the strict surface verifies smooth curves,
transverse pairs and absence of triples on each owned region. Their finite
pair intersections are reduced, so their Euler characteristics count edges.
"""
from .kernel_json import kernel_json
from .polynomial import encode_sparse_row, integer_text


def surface_complex(charts, *, local_coordinates=None):
    rays, pieces = {}, []
    for cone, boundary, terms, owned in charts:
        pieces.append({"rays": [rays.setdefault(r, len(rays)) for r in cone],
                       "boundary": list(boundary.items()), "strict": encode_sparse_row(dict(terms)),
                       "owned": owned})
    if local_coordinates is not None:
        for piece, coords in zip(pieces, local_coordinates, strict=True):
            piece["coordinates"] = coords
    return kernel_json("atlas_torus_sections", "surface", charts=pieces,
                       rays=[list(map(integer_text, ray)) for ray in rays])
