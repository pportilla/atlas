"""Smooth toric blowups over the singular locus of a binomial."""
def build_binomial_fan(exponents, *, variables=None, max_steps=256, max_cones=1024, intrinsic=False):
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
    if left == right:
        return None
    from . import binomial_kernel
    operation = "intrinsic_fan" if intrinsic else "fan"
    rows = binomial_kernel.run(f"{operation} {dimension} {max_steps} {max_cones}", left, right)
    return binomial_kernel.fan(rows, dimension, names)
