"""Sparse point-local examples whose expansion exceeds one million terms."""

EXPONENT = 2**64 + 7
VARIABLES = ["x", "y", "z"]
POINT = ["1", "0", "0"]

# At p, u=x^N-1 has nonzero derivative. The independent linear forms
# u+y, u+z, y+z give a local coordinate system. Their squares cut out an
# isolated nonreduced point with zero Jacobian rank.
def coupled_squares(exponent):
    return [f"(x^{exponent}-1+y)^2", f"(x^{exponent}-1+z)^2", "(y+z)^2"]


COUPLED_SQUARES = coupled_squares(EXPONENT)

# The extra x prevents a common power substitution in the moving axis.
# The unsquared linear equation gives a rank-one Jacobian at the same point.
COUPLED_RANK_ONE = [f"(x^{EXPONENT}+x-2+y)^2", f"(x^{EXPONENT}+x-2+z)^2", "y+z"]
