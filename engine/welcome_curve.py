"""Small, reduced plane germs for the opening resolution diagram."""
import random

from .resolve import resolve


# A binomial with coprime exponents has one geometric branch and one Puiseux
# pair. Keeping exponents small also keeps the drawing and the blowups short.
_CUSP_POWERS = ((2, 3), (2, 5), (3, 4), (3, 5))


def _binomial(x, y, powers, sign):
    p, q = powers
    return f"{y}^{p}{sign}{x}^{q}"


def random_curve(rng=None):
    """Sample two distinct branches with at most two Puiseux pairs in total."""
    if rng is None:
        rng = random
    x, y = rng.choice((("x", "y"), ("y", "x")))
    family = rng.choice(("tangent", "cusp_line", "two_cusps", "two_pairs_line"))

    if family == "tangent":
        # Two smooth graphs with contact order k.
        k = rng.randint(2, 5)
        return f"({y}-{x}^{k})*({y}+{x}^{k})"

    if family == "two_pairs_line":
        # For y=t^6+a*t^7, x=t^4, elimination gives
        # (y^2-x^3)^2 - 4*a^2*x^5*y - a^4*x^7.
        # The characteristic exponents 6 and 7 drop gcd(4,...) to 2 and 1;
        # conjugating t by fourth roots of unity accounts for all four roots.
        a = rng.choice((1, 2))
        coefficient = "" if a == 1 else f"{a**4}*"
        branch = f"({y}^2-{x}^3)^2-{4*a*a}*{x}^5*{y}-{coefficient}{x}^7"
        return f"{rng.choice((x, y))}*({branch})"

    sign = rng.choice(("-", "+"))
    first = _binomial(x, y, rng.choice(_CUSP_POWERS), sign)
    if family == "cusp_line":
        return f"{rng.choice((x, y))}*({first})"

    # Swapping axes makes the tangents distinct. With common axes, the
    # opposite coefficient keeps the branches distinct even for equal powers.
    u, v = rng.choice(((x, y), (y, x)))
    other_sign = "+" if sign == "-" else "-"
    second = _binomial(u, v, rng.choice(_CUSP_POWERS), other_sign)
    return f"({first})*({second})"


def resolve_welcome_curve(rng=None):
    """Return the factored input and its freshly computed embedded resolution."""
    polynomial = random_curve(rng)
    record = resolve({"polynomial": polynomial, "variables": ["x", "y"],
                      "mode": "embedded", "search": "quick", "compute": "cpu"})
    if (record["status"] != "resolved" or record["dual_complex"]["status"] != "complete"
            or len(record["branches"]) != 2):
        raise RuntimeError("Atlas did not certify a two-branch embedded resolution.")
    return polynomial, record
