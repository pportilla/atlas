"""Smooth toric blowups over the singular locus of a binomial."""
from collections import Counter
from itertools import combinations


def _terminal(cone, signed_orders, dimension):
    """Test smoothness and transversality to every exceptional stratum.

    After dividing the common monomial, opposite signs have disjoint supports.
    A unit term forces every coordinate in the other term to be invertible
    along the hypersurface. A degree-one original term gives an affine graph
    even after imposing any subset of the exceptional coordinates.
    """
    positive = [ray for ray in cone if signed_orders[ray] > 0]
    negative = [ray for ray in cone if signed_orders[ray] < 0]
    if not positive or not negative:
        return {"criterion": "unit term"}
    for side in (positive, negative):
        if len(side) == 1:
            ray = side[0]
            if ray < dimension and abs(signed_orders[ray]) == 1:
                return {"criterion": "linear nonexceptional coordinate",
                        "linear_ray_index": ray, "coordinate_index": cone.index(ray)}
    # When both terms have degree >= 2, the chart origin is singular. For
    # linear exceptional terms, set those coordinates to zero: the remaining
    # monomial is singular at the origin, or the entire stratum is contained.
    return None


def _choose_center(cones, bad_cones, rays, signed_orders, exponents):
    dimension = len(rays[0])
    for size in range(2, dimension + 1):
        candidates = {}
        for cone in bad_cones:
            for face in combinations(sorted(cone), size):
                if face in candidates:
                    continue
                orders = [signed_orders[i] for i in face]
                if not min(orders) < 0 < max(orders):
                    continue
                image = tuple(j for j in range(dimension) if any(rays[i][j] for i in face))
                # A coordinate subspace is contained in the original singular
                # locus if both terms have order at least two in its ideal.
                image_orders = tuple(sum(term[j] for j in image) for term in exponents)
                if min(image_orders) < 2:
                    continue
                positive = sum(w for w in orders if w > 0)
                negative = -sum(w for w in orders if w < 0)
                severity = (max(positive, negative), min(positive, negative))
                candidates[face] = (severity, image, image_orders)
        if candidates:
            incidence = Counter(face for cone in cones for face in combinations(sorted(cone), size)
                                if face in candidates)
            def priority(face):
                larger, smaller = candidates[face][0]
                return -larger, -smaller, incidence[face], face

            face = min(candidates, key=priority)
            _, image, image_orders = candidates[face]
            return face, image, image_orders, incidence[face]
    return None


def build_binomial_fan(exponents, *, variables=None, max_steps=256, max_cones=1024):
    """Return the completed fan and blowup sequence, or defer within the budget.

    Input consists of two distinct ordinary monomials with disjoint support.
    Coefficients can be any two nonzero constants in
    characteristic zero. Each chosen face belongs to a unimodular cone, and
    subdividing its entire star at the sum of its rays is the blowup of its
    smooth orbit closure. The image of this closure is the recorded original
    coordinate subspace. Indices of rays and coordinates start at zero.
    """
    exponents = tuple(tuple(term) for term in exponents)
    if len(exponents) != 2:
        return None
    left, right = exponents
    dimension = len(left)
    if (dimension < 2 or len(right) != dimension or max_cones < 1 or max_steps < 0
            or any(type(value) is not int or value < 0 for term in exponents for value in term)
            or any(a and b for a, b in zip(left, right))):
        return None
    names = list(range(dimension)) if variables is None else list(variables)
    if len(names) != dimension:
        raise ValueError("Supply one variable name per exponent coordinate.")
    signed_orders = [a - b for a, b in zip(left, right)]
    if not any(signed_orders):
        return None

    from . import binomial_kernel
    from .native_library import CHECK
    rows = binomial_kernel.run(f"fan {dimension} {max_steps} {max_cones}", left, right)
    if rows is not None:
        output = binomial_kernel.fan(rows, dimension, names)
        if CHECK and output != _reference(exponents, names, max_steps, max_cones):
            raise AssertionError("The binomial fan differs from its reference subdivision.")
        return output
    return _reference(exponents, names, max_steps, max_cones)


def _reference(exponents, names, max_steps, max_cones):
    left, right = exponents
    dimension = len(left)
    signed_orders = [a - b for a, b in zip(left, right)]
    rays = [tuple(int(i == j) for j in range(dimension)) for i in range(dimension)]
    cones = [tuple(range(dimension))]
    steps = []
    while True:
        certificates = [_terminal(c, signed_orders, dimension) for c in cones]
        bad = [cone for cone, cert in zip(cones, certificates) if cert is None]
        if not bad:
            for cone, cert in zip(cones, certificates):
                cert.update(ray_indices=list(cone), signed_orders=[signed_orders[i] for i in cone])
            return {"rays": [list(ray) for ray in rays], "cones": [list(c) for c in cones],
                    "terminal_certificates": certificates, "steps": steps}
        if len(steps) >= max_steps:
            return None
        choice = _choose_center(cones, bad, rays, signed_orders, exponents)
        if choice is None:
            return None
        face, image, image_orders, incidence = choice
        if len(cones) + incidence * (len(face) - 1) > max_cones:
            return None
        new = len(rays)
        rays.append(tuple(map(sum, zip(*(rays[i] for i in face)))))
        signed_orders.append(sum(signed_orders[i] for i in face))
        steps.append({"type": "blowup", "id": f"blowup-{len(steps) + 1}",
                      "center_ray_indices": list(face), "new_ray_index": new,
                      "image_zero_coordinates": [names[j] for j in image],
                      "image_monomial_orders": list(image_orders)})
        face_set = frozenset(face)
        refined = []
        for cone in cones:
            if face_set.issubset(cone):
                for removed in face:
                    refined.append(tuple(new if j == removed else j for j in cone))
            else:
                refined.append(cone)
        cones = refined
