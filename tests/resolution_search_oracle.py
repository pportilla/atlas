"""Independent symbolic checks of the resolution candidate family and exhaustive code syntax."""
from fractions import Fraction

from engine import native_polynomial as np
from engine.chart_expressions import chart_expressions
from engine.reference import QQ, sp
from engine.resolution_search import ResolutionCandidate, _projective


def support_profile(variables, f):
    exps = [m for m, _ in sp.Poly(f, *variables, domain=QQ).terms()]
    return (max(map(sum, exps)), len(exps),
            tuple(i for i in range(len(variables)) if any(m[i] for m in exps)))


def singular_subspaces(variables, f, subset_batch):
    exps = [m for m, _ in sp.Poly(f, *variables, domain=QQ).terms()]
    return tuple(j for j, indices in enumerate(subset_batch)
                 if all(sum(e[i] for i in indices) >= 2 for e in exps))


def repeated_directions(variables, expr, i, j):
    s = sp.Dummy("slope")
    terms = sp.Poly(expr, *variables, domain=QQ).terms()
    order = min(e[i] + e[j] for e, _ in terms)
    if order < 2:
        return False, ()
    forms = {}
    for e, c in terms:
        if e[i] + e[j] == order:
            key = e[:i] + e[i+1:j] + e[j+1:]
            forms[key] = forms.get(key, 0) + c*s**e[j]
    polynomials = [sp.Poly(form, s, domain=QQ) for form in forms.values()]
    common = polynomials[0]
    for form in polynomials[1:]:
        common = common.gcd(form)
    slopes = sorted(Fraction(root) for root, count in common.ground_roots().items() if count >= 2)
    return all(form.degree() <= order-2 for form in polynomials), tuple(slopes)


def squarefree(expr, variables):
    return sp.Poly(expr, *variables, domain=QQ).sqf_part().monic().as_expr()


class _InvalidCode(Exception):
    pass


class _Reader:
    """Read a self-delimiting stream of naturals (Elias gamma on value+1)."""
    def __init__(self, code):
        self.bits = bin(code + 1)[3:]  # discard the leading 1
        self.at = 0

    def nat(self):
        start = self.at
        # The leading zeroes and the value bits after the first one are
        # equally long. Once this prefix is all zeroes, no suffix can fit
        # a complete natural number in the remaining bit length.
        last = start + (len(self.bits) - start + 1)//2
        one = self.bits.find("1", start, last)
        if one < 0:
            self.at = last
            raise _InvalidCode
        self.at = one + (one-start) + 1
        return int(self.bits[one:self.at], 2) - 1

    def signed(self):
        n = self.nat()
        return (n+1)//2 if n%2 else -(n//2)

    def size(self, minimum=0):
        n = minimum + self.nat()
        # Every item consumes at least one bit; a short malformed code cannot
        # claim an enormous vector that blocks subsequent valid candidates.
        if n > len(self.bits)-self.at:
            raise _InvalidCode
        return n


def _read_field(reader, api):
    field_degree = reader.nat() + 1
    if field_degree == 1:
        return QQ, None, 1
    if field_degree > len(reader.bits)-reader.at:
        raise _InvalidCode
    coeff = tuple(reader.signed() for _ in range(field_degree))
    root_index = reader.nat()
    if root_index >= field_degree:
        raise _InvalidCode
    t = sp.Symbol("T")
    coefficients = (1, *reversed(coeff))
    if not sp.Poly.from_list(coefficients, t, domain=sp.QQ).is_irreducible:
        raise _InvalidCode
    if api is np:
        from engine.candidate_field import indexed_field
        domain, primitive = indexed_field(coefficients, root_index)
    else:
        from engine.field_coefficient import indexed_field
        domain, primitive = indexed_field(coefficients, t, root_index)
    return domain, primitive, field_degree


def _read_poly(reader, variables, field_degree, primitive, projective_start=None):
    api = chart_expressions(variables)
    terms = []
    projective_support = []
    for _ in range(reader.size(1)):
        coeff = reader.signed()
        if coeff == 0:
            raise _InvalidCode
        if field_degree > 1:
            k = reader.nat()
            if k >= field_degree:
                raise _InvalidCode
            coeff *= primitive**k
        exps = tuple(reader.nat() for _ in variables)
        if projective_start is not None:
            projective_support.append(exps[projective_start:])
        terms.append(coeff * api.prod(v**e for v,e in zip(variables,exps)))
    if projective_support:
        if len(set(map(sum, projective_support))) != 1:
            raise _InvalidCode
    poly = api.Add(*terms)
    if poly == 0:
        raise _InvalidCode
    return poly


def decode_range(variables, code):
    """Decode one code and advance past every suffix its prefix ruled out."""
    variables = tuple(variables)
    r = _Reader(code)
    try:
        domain, primitive, degree = _read_field(r, chart_expressions(variables))
        center = tuple(_read_poly(r, variables, 1, None) for _ in range(r.size(1)))
        u = _projective(variables, len(center))
        homogeneous = (*variables,*u)
        components = []
        for _ in range(r.size()):
            ideal = tuple(_read_poly(r, homogeneous, degree, primitive, len(variables))
                          for _ in range(r.size(1)))
            components.append(ideal)
        if r.at == len(r.bits):
            candidate = ResolutionCandidate(center, u, tuple(components), domain, primitive, f"integer code {code:#x}")
            return candidate, code + 1
    except _InvalidCode:
        pass
    # The parse failed, or finished before the end. For this fixed length
    # all completions of the prefix that was read have the same failure.
    # Work with code+1 because its leading sentinel determines the length.
    unread = len(r.bits) - r.at
    next_code = (((code + 1) >> unread) + 1) << unread
    return None, next_code - 1


def first_code(variables):
    # The two shortest payloads have a rational field, a single centre
    # generator +1 or -1, all exponents zero, and empty boundary. Gamma
    # code 1 is "010", the smaller of the two nonzero signed coefficients.
    # In code+1 the leading sentinel and the three one-bit counts come first.
    return (0b1111011 << (len(variables) + 1)) - 2




def coefficient_rows(expr, variables, domain):
    terms = sp.Poly(expr, *variables, domain=domain).rep.to_dict()
    return tuple(sorted((e, (v,) if domain == QQ else tuple(v.to_list())) for e, v in terms.items()))


def check_removed_unit(unit, polynomial, active):
    if unit != sp.cancel(polynomial/active):
        raise AssertionError("The removed unit differs from the exact quotient.")
