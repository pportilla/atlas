"""Parameter and return descriptions for the callable Atlas session interface.

Defaults and argument order come from the live signature. Keep accepted values,
ownership requirements and return fields here alongside the runnable help cards.
"""
from typing import NamedTuple

from engine.session_function_help import session_signature


class FunctionReference(NamedTuple):
    returns: str
    inputs: dict[str, str]


def entry(returns, **inputs):
    return FunctionReference(returns, inputs)


COMPUTE = ('Device policy. Use `"auto"` to let Atlas choose, `"cpu"` for the CPU, '
           'or `"gpu"` to request acceleration on eligible stages.')
ORDER = ('Monomial order, either `"lex"` (lexicographic) or `"grevlex"` (graded reverse '
         'lexicographic). Omission inherits an order carried by the input or uses `"grevlex"`.')
VARIABLES = ('All generators of one live ring in declaration order, such as `R.coordinates`. '
             'This can specify a parent even when every input is a scalar.')
POINT_COORDINATES = ('List or tuple with one rational entry per coordinate in declaration order. '
                     'Both `(1/2, 0)` and `("1/2", 0)` specify the same point.')
POINT = POINT_COORDINATES + ' The default is the origin.'
GLOBAL_IDEAL = 'Ideal in a polynomial ring over `QQ`, or its saved Gröbner basis.'
QUOTIENT_IDEAL = 'Ideal in a polynomial or quotient ring over `QQ`, or its saved Gröbner basis.'
LOCAL_IDEAL = GLOBAL_IDEAL + ' A saved local standard basis also supplies the point.'
CONE_IDEAL = LOCAL_IDEAL + ' A saved tangent cone selects its vertex.'
LOCAL_POINT = POINT_COORDINATES + ' Defaults to the origin for an ideal and to the recorded point for a saved basis.'
CONE_POINT = LOCAL_POINT + ' For a saved tangent cone, use the origin of its displacement coordinates.'
POLYNOMIAL = 'Polynomial over `QQ` or an exact rational scalar.'
IN_IDEAL_RING = 'Polynomial in the ideal’s ring or an exact rational scalar.'
QUOTIENT_OPERAND = IN_IDEAL_RING + ' For a quotient ideal, an ambient polynomial is projected into the quotient.'
EXPONENT = 'Nonnegative integer, also accepted as an unsigned decimal string. Zero gives the unit ideal.'
SERIES = ('A graded series with signed `.numerator` terms (degree and coefficient), '
          'denominator `(1-t)^denominator_power`, Krull `.dimension` and `.multiplicity`.')
BASIS = ('A quotient-basis result. When `.finite` is true, `.basis` lists polynomial representatives '
         'of a vector-space basis over `QQ` and `.dimension` is its length. Both are `None` for infinite dimension.')

DECLARATION = dict(
    names='One or more distinct coordinate names, passed as individual strings in generator order. '
          'Use letters, digits and underscores, beginning with a letter, up to 24 characters.',
    coeffs='Coefficient field. Atlas accepts the rational field `QQ`.',
    fresh='With `False`, repeated ordinary declarations in the same session reuse the parent. '
          'With `True`, create an independent ring and allow its coordinates to rebind saved names.',
)
POLYNOMIAL_OPTIONS = dict(order=ORDER, variables=VARIABLES, compute=COMPUTE)
IDEAL_INPUT = dict(ideal=GLOBAL_IDEAL, compute=COMPUTE)
IDEAL_PAIR = dict(left=QUOTIENT_IDEAL, right='Ideal with the same live parent as `left`.', compute=COMPUTE)
ORDERED_IDEALS = dict(ideal=GLOBAL_IDEAL, other_ideal='Ideal with the same live parent and monomial order.',
                      compute=COMPUTE)
LOCAL_INPUT = dict(ideal=LOCAL_IDEAL, point=LOCAL_POINT, compute=COMPUTE)
CONE_INPUT = dict(ideal=CONE_IDEAL, point=CONE_POINT, compute=COMPUTE)
SCHEME_POINT_INPUT = dict(CONE_INPUT, point=CONE_POINT + ' All ideal generators must vanish at the point.')
HYPERSURFACE = dict(polynomial='Polynomial over `QQ` defining the hypersurface germ.',
                    point=POINT + ' The polynomial must vanish there.', **POLYNOMIAL_OPTIONS)
RESOLUTION_INPUT = dict(
    variables='For a formula string, use an ordered list of coordinate names or the ring’s live generators. '
              'Choose either `variables` or `ring`.',
    ring='For a formula string, supply its live affine polynomial ring over `QQ`. '
         'Omitting both `ring` and `variables` uses the active ring. A typed input carries its own ring.',
    search='Use `"quick"` for a bounded attempt or `"exhaustive"` to continue the search.',
    compute=COMPUTE + ' With `None`, resolution uses the configured backend (normally CPU).',
)

FUNCTIONS = {
    'coordinates': entry('One polynomial generator for a single name, or an ordered tuple of generators '
                         'for multiple names. Each generator has the declared ring as its `.parent`.',
                         **DECLARATION),
    'ring': entry('The live polynomial ring. A direct assignment such as `R = ring("x", "y")` also '
                  'binds its generators to `x` and `y`. Read them in order from `R.coordinates`.', **DECLARATION),
    'use_ring': entry('The selected ring, or `None` after clearing the selection.',
                     ring='Live polynomial, chart or quotient ring to select, or `None` to clear it. '
                           'The ground field `QQ` has no coordinates to select.'),
    'active_ring': entry('The selected ring, or `None` when the session has no active ring.'),
    'ring_name': entry('A saved name as a string, including an automatic name such as `"R1"`. '
                       'Returns `"QQ"` for the ground field and `None` for an unnamed ring.',
                       ring='Live ring to look up, such as `f.parent`. A ring with several names gives one alias.'),
    'delete_ring': entry('A report with the removed bindings in `.names` and the Boolean `.was_active` '
                         'indicating whether the active selection was removed.',
                         ring='Live ring to delete with its dependent bindings. The built-in ground field is permanent.'),
    'polynomial': entry('A polynomial in the selected ring, or a reduced residue class if that ring is a quotient.',
                        source='Nonempty expression string in the ring’s names, such as `"x^2 + y/2"`.',
                        ring='Live parent in which to read the formula. If both parent options are omitted, '
                             'the active ring supplies the coordinates.',
                        variables='All generators of the parent in order. Use this in place of `ring`.'),
    'ideal': entry('An ideal with its live `.parent`, `.generators` and monomial `.order`.',
                   generators='Polynomials, exact rational scalars or residue classes, passed separately or '
                              'as one list or tuple. No generators gives the zero ideal.',
                   order=ORDER + ' A quotient ideal uses its parent’s order.',
                   variables='Ordered generators of the target ring. Use this in place of `ring`.',
                   ring='Live parent ring. Ring-valued generators normally supply it. For an empty or '
                        'scalar-only ideal, name the ring or select an active one. Explicit `ring=K` '
                        'projects ambient generators into the quotient `K`.'),
    'quotient_ring': entry('The ring of residue classes. Its `.coordinates` belong to the quotient, '
                           '`.ambient` gives the polynomial ring, and `.projection` maps the '
                           'defining ideal’s parent to the quotient.',
                           ideal=QUOTIENT_IDEAL, compute=COMPUTE,
                           fresh='Set to `True` for an independent quotient. By default the same '
                                 'ideal object gives the same parent, including across compute policies.'),
    'hom': entry('A callable ring homomorphism with `.domain`, `.codomain` and ordered `.images`. '
                 'Apply it to an element of the domain to substitute all generator images simultaneously.',
                 domain='Live source ring over `QQ`. Its defining relations must vanish on the supplied images.',
                 codomain='Live target ring in which to evaluate. A chart over an algebraic coefficient field '
                          'can also serve as the target.',
                 images='List or tuple of one target element per source generator, in declaration order. '
                        'For a quotient target, use its residue classes or rational scalars.'),
    'gcd': entry('The monic greatest common divisor as a polynomial in the common parent, or zero '
                 'when both inputs vanish.', left=POLYNOMIAL, right='Second polynomial in the same ring or a scalar.',
                 **POLYNOMIAL_OPTIONS),
    'factor': entry('A factorization with rational `.unit` and ordered `.factors`. Each factor has a monic '
                    '`.polynomial` and positive integer `.multiplicity`.',
                    polynomial='Nonzero polynomial over `QQ` or nonzero rational scalar.', **POLYNOMIAL_OPTIONS),
    'resultant': entry('The signed resultant as a polynomial in the surviving coordinates. Its `.inclusion` '
                       'maps the result back to the original ring.', left=POLYNOMIAL,
                       right='Second polynomial in the same ring or a scalar.',
                       variable='One generator of the common ring, or its name as a string, to eliminate.',
                       **POLYNOMIAL_OPTIONS),
    'groebner': entry('A basis in the input ring. Read its `.generators`, or pass the saved basis directly '
                      'to `normal_form` and `membership`.', ideal=QUOTIENT_IDEAL, compute=COMPUTE),
    'normal_form': entry('The reduced remainder in the ideal’s parent ring, a residue class for a quotient ideal.',
                         polynomial=QUOTIENT_OPERAND, ideal=QUOTIENT_IDEAL, compute=COMPUTE),
    'membership': entry('A result with Boolean `.value`. It is true exactly when the remainder is zero.',
                        polynomial=QUOTIENT_OPERAND, ideal=QUOTIENT_IDEAL, compute=COMPUTE),
    'eliminate': entry('An ideal in the polynomial ring on the surviving coordinates. Its `.inclusion` '
                       'maps those coordinates back to the source.',
                       variables='One or more distinct generators to eliminate, or their names. '
                                 'Pass them separately or as one list or tuple.',
                       **IDEAL_INPUT),
    'intersection': entry('The ideal of polynomials belonging to both input ideals, in their common parent.',
                          **ORDERED_IDEALS),
    'quotient': entry('The colon ideal of polynomials whose product with every generator of `other_ideal` '
                      'belongs to `ideal`.', **ORDERED_IDEALS),
    'saturation': entry('The ideal of elements carried into the input ideal by multiplication by some '
                        'nonnegative power of the supplied polynomial.', polynomial=IN_IDEAL_RING, **IDEAL_INPUT),
    'ideal_sum': entry('An ideal generated by the combined generators in their common ring, with the left '
                       'ideal’s order. Also available as `I + J`.', **IDEAL_PAIR),
    'ideal_product': entry('An ideal generated by all products of a left generator and a right generator, '
                           'with the left ideal’s order. Also available as `I * J`.', **IDEAL_PAIR),
    'ideal_power': entry('The ordinary power in the input ring, also available as `I^n`.',
                         ideal=QUOTIENT_IDEAL, n=EXPONENT, compute=COMPUTE),
    'symbolic_power': entry('The intersection of the contractions of the ordinary power localized at '
                            'the minimal primes, as an ideal in the original ring.',
                            ideal='A radical ideal (including a prime ideal) in a polynomial ring over `QQ`.',
                            n=EXPONENT, compute=COMPUTE),
    'radical': entry('The ideal of all polynomials having a positive power in the input ideal, over `QQ`.',
                     **IDEAL_INPUT),
    'primary_decomposition': entry('A decomposition with one entry in `.components` for each primary '
                                   'component. Each entry pairs the ideal `.primary` with its associated `.prime`.',
                                   **IDEAL_INPUT),
    'localize': entry('A localization with its verified `.prime`, contracted ideal `.contraction`, '
                      'and optional fraction `.element`. For that fraction, read `.member` and `.unit`.',
                      ideal='Ideal to extend to the local ring. In the one-argument form, this is '
                            'the prime at which to localize and the extended ideal is zero.',
                      prime='Proper prime ideal in the same polynomial ring and order as the ideal to extend.',
                      numerator='Polynomial or scalar numerator of the fraction to test. Defaults to one '
                                'when only a denominator is supplied.',
                      denominator='Polynomial or scalar denominator outside the prime. Defaults to one '
                                  'when only a numerator is supplied.', compute=COMPUTE),
    'normalization': entry('The normalization of the reduced quotient. Each entry of `.components` '
                           'includes its prime, target relations, coordinate images, fractions and map. '
                           '`.input_reduced` records whether the original quotient was reduced.', **IDEAL_INPUT),
    'dimension': entry('A result whose integer `.value` is the dimension of the quotient (−1 for the zero ring).',
                       **IDEAL_INPUT),
    'quotient_basis': entry(BASIS, **IDEAL_INPUT),
    'multiplicity': entry('A global length with `.finite` and `.value`. The value is the vector-space '
                          'dimension over `QQ` when finite, and `None` for infinite colength.', **IDEAL_INPUT),
    'hilbert_series': entry(SERIES, ideal='An ideal homogeneous in the standard grading of a '
                            'polynomial ring over `QQ`.', compute=COMPUTE),
    'local_standard_basis': entry('A local basis with polynomial `.generators` and the selected `.point`. '
                                  'Pass the result to `local_normal_form` to reuse it.',
                                  ideal=GLOBAL_IDEAL, point=POINT, compute=COMPUTE),
    'local_normal_form': entry('A local reduction with `.remainder` and `.unit`. The unit evaluates to '
                               'one at the point, and multiplying the input by the unit gives the remainder '
                               'modulo the localized ideal. A zero remainder certifies local membership.',
                               polynomial=IN_IDEAL_RING, **LOCAL_INPUT),
    'local_quotient_basis': entry(BASIS, **LOCAL_INPUT),
    'hilbert_samuel_function': entry('A result whose `.values` are the lengths modulo the requested powers '
                                     'of the maximal ideal, with degree n corresponding to power n+1. '
                                     '`.graded_values` gives the dimensions of the graded pieces in the same order.',
                                     degrees='List or tuple of nonnegative integer degrees in the desired output order. '
                                             'Unsigned decimal strings, repeated entries and an empty list are allowed. '
                                             'Omission selects 0 through 5.',
                                     **LOCAL_INPUT),
    'hilbert_samuel_multiplicity': entry('The local Hilbert–Samuel multiplicity in `.value`, with the local '
                                         'Krull dimension in `.dimension`.', **LOCAL_INPUT),
    'tangent_cone': entry('A cone with homogeneous `.generators` in fresh displacement `.coordinates`, '
                          'together with its `.dimension` and `.multiplicity`.', **LOCAL_INPUT),
    'tangent_cone_hilbert_series': entry(SERIES, **CONE_INPUT),
    'zariski_tangent_space': entry('A tangent space with independent linear `.equations` in displacement '
                                   'coordinates, rational direction vectors in `.basis` and their `.dimension`.',
                                   **SCHEME_POINT_INPUT),
    'embedding_dimension': entry('The tangent-space dimension in `.value`, with `.ambient_dimension` '
                                 'and `.jacobian_rank`.', **SCHEME_POINT_INPUT),
    'local_dimension': entry('The local Krull dimension at a point of the scheme in `.value`.', **SCHEME_POINT_INPUT),
    'local_smoothness': entry('A smoothness result. Its Boolean `.value` compares `.local_dimension` '
                              'and `.embedding_dimension` at the point of the scheme.', **SCHEME_POINT_INPUT),
    'local_jacobian_algebra': entry('The local algebra modulo the partial derivatives. Read the derivatives '
                                    'in `.generators` and check `.finite` before using `.basis` or `.dimension`.',
                                    **HYPERSURFACE),
    'milnor_number': entry('A local length with `.finite` and `.value`, computed modulo the partial '
                           'derivatives. An infinite length has value `None`.', **HYPERSURFACE),
    'tjurina_number': entry('A local length with `.finite` and `.value`, computed modulo the equation '
                            'and its partial derivatives. An infinite length has value `None`.', **HYPERSURFACE),
    'ae_codimension': entry('A codimension result. When `.finite` is true, `.value` is exact. False certifies '
                            'infinity. When it is `None`, inspect `.lower_bound` and `.reason`.',
                            components='One or more map components in target order, all vanishing at the origin. '
                                       'Use one source ring over `QQ` with at least one coordinate. Pass the '
                                       'components separately or as one list or tuple.',
                            **POLYNOMIAL_OPTIONS),
    'resolve': entry('A resolution with `.status`, `.charts`, `.divisors`, `.branches` and `.dual_complex`. '
                     'Read the status to see whether the search certified the resolution.',
                     polynomial='Hypersurface equation vanishing at the origin, as a polynomial over `QQ` '
                                'or as a formula string.',
                     mode='Use `"embedded"` for normal crossings of the total transform, including multiplicities, '
                          'or `"intrinsic"` to resolve the strict transform of a reduced germ.',
                     method='Use `"auto"` for the standard resolution search. Choose `"weighted"` for embedded '
                            'weighted blowups in 1–16 variables with `monodromy=False`. Inspect the returned '
                            '`.steps`, `.divisors` and `.certificate`.',
                     monodromy='Set to `True` to request both the monodromy and local topological zeta functions.',
                     **RESOLUTION_INPUT),
    'recompute_dual_complex': entry('A dual complex with `.vertices` linked to divisors on the supplied '
                                    'resolution, `.cells` describing incidences, and a completeness `.status`.',
                                    resolution='A saved resolution returned by `resolve`. Its charts supply the incidence data.'),
    'dual_complex': entry('A dictionary with rebuilt vertices, cells and status. For linked divisor objects '
                          'use `recompute_dual_complex`.', resolution='A saved resolution returned by `resolve`.'),
    'monodromy_check': entry('A record with `.status`, `.reason` and the witnesses for actual `.poles`. '
                             'A verified status means every pole has an eigenvalue witness.',
                             polynomial='Equation of a hypersurface germ at the origin, a formula string, or '
                                        'a saved resolution. A saved resolution supplies its charts directly.',
                             check_isolation='With `True`, try the local Milnor number when an eigenvalue '
                                             'witness is unresolved, to check for an isolated candidate.',
                             **RESOLUTION_INPUT),
    'print': entry('Writes text and returns `None`.',
                   args='Zero or more values in display order.', sep='Text placed between values.',
                   end='Text written after the final value, normally a newline.',
                   file='Use `None` for the session’s standard output.',
                   flush='Set to `True` to flush the output stream immediately.'),
    'len': entry('An integer count.', obj='A sequence, mapping or other value with a length, '
                 'such as `I.generators`. Pass it positionally.'),
    'print_json': entry('Writes one JSON object followed by a newline and returns `None`.',
                        fields='Keyword arguments giving field names and serializable values, '
                               'such as `print_json(length=2)`. Exact rationals export as strings.'),
    'algebra_request': entry('The structured value for the requested operation, reconstructed in a fresh '
                             'coordinate space for this request.', source='JSON string specifying an algebra '
                             'operation, its ordered ring and its operands.'),
}


def reference_for(name, function):
    """Require descriptions for exactly the arguments the session accepts, in signature order."""
    data = FUNCTIONS[name]
    parameters = session_signature(function).parameters
    if difference := parameters.keys() ^ data.inputs.keys():
        raise ValueError(f"Update the descriptions of {name} parameters: {sorted(difference)}")
    return [(key, data.inputs[key]) for key in parameters], data.returns
