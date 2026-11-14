"""Local coordinate frames for computing the invariant of a blowup centre."""

from .affine_kernel import (frame_derivative, frame_field, frame_function, frame_linear,
                            frame_locus, frame_restrict)
from .chart_expressions import chart_expressions
from .groebner import groebner


def _membership(variables, equations, f, known_nonempty):
    from .field_wide import _expanded_row
    from .native_library import ideal_request
    rows = [_expanded_row(variables, g) for g in (*equations, f)]
    result, = ideal_request(f"frame_membership {len(variables)} {len(equations)} {int(known_nonempty)}", rows)
    return result


class Frame:
    """A submanifold N of affine space, on an open set, with coordinates and derivations.

    N is given by the equations `rows` in the `variables`, on the open set
    where the `units` do not vanish. Variables determined rationally by an
    equation have been eliminated: `solved` keeps those equations, for the way back.
    """

    def __init__(self, variables, coords, rows=(), units=(), solved=()):
        self.variables = tuple(variables)
        self.coords = tuple(coords)         # variables that are local coordinates on N
        self.rows = tuple(rows)             # equations with invertible Jacobian in the other variables
        self.units = tuple(u for u in units if not getattr(u, "is_number", True) or u == 0)
        self.solved = tuple(solved)
        self._fields = {}
        self._basis = None
        self._reducer = None
        self._jets = {}
        self._inverted = {}
        self._cuts = {}
        self._cut_coordinates = {}
        self._empty = {}
        self._vanishing = {}
        self._regular = {}
        self._known_nonempty = False
        self._scalar_units = set()
        self._unit = None

    def field(self, k):
        """The vector field D_k as {variable: coefficient}."""
        if k not in self._fields:
            others = [v for v in self.variables if v not in self.coords]
            if not others:
                self._fields[k] = {k: chart_expressions(self.variables).sympify(1)}
            else:
                self._fields[k] = frame_field(self.variables, self.rows, (*others, k))
        return self._fields[k]

    def basis(self):
        if self._basis is None:
            contracted = frame_locus(self.variables, self.rows, self.units, closure=True)
            self._basis = groebner(contracted, self.variables, known=len(contracted))
        return self._basis

    def over(self, x):
        """A variable t and a basis of (rows) + (t*x - 1) for an order that
        compares t first. The units are left out of it on purpose."""
        if x not in self._inverted:
            t = chart_expressions(self.variables).Dummy("over")
            self._inverted[x] = t, groebner((*self.rows, t*x - 1), (t, *self.variables), order="first")
        return self._inverted[x]

    def unit(self):
        """The product of the units, as a function on V(rows)."""
        if self._unit is None:
            value = chart_expressions(self.variables).sympify(1)
            for u in self.units:
                value = self.reduced(value*self.reduced(u))
            self._unit = value
        return self._unit

    def cut(self, x):
        """Bases that decide whether x divides a function on N: one of the
        ideal I0 = (rows) + (x), and one of I0 + (s*u - 1) for the product u
        of the units taken modulo I0 (None when u is constant there, False
        when u vanishes there: N does not meet x = 0)."""
        if x not in self._cuts:
            first = groebner((*self.rows, x), self.variables)
            u = chart_expressions(self.variables).sympify(1)
            for factor in self.units:
                factor = first.reduce(frame_function(self.variables, factor))[1]
                u = first.reduce(frame_function(self.variables, u*factor))[1]
            if u == 0:
                second = False
            elif u.is_number:
                second = None
            else:
                s = chart_expressions(self.variables).Dummy("unit")
                self._cut_coordinates[x] = (s, *self.variables)
                second = groebner((*first.exprs, s*u - 1), self._cut_coordinates[x])
            self._cuts[x] = first, second
        return self._cuts[x]

    def reduced(self, f):
        """A smaller representative of the function f on N."""
        if not self.rows:
            return f
        return self._reduced_terms(frame_function(self.variables, f))

    def _reduced_terms(self, value):
        if self._reducer is None:
            self._reducer = groebner(self.rows, self.variables)
        return self._reducer.reduce(value)[1]

    def derivative(self, f, k):
        field = self.field(k)
        value = frame_derivative(self.variables, f, field)
        if len(field) == 1 and field.get(k) == 1:
            return value
        return self._reduced_terms(value)

    def graph_regular(self, f):
        """Whether native graph elimination certifies a nonzerodivisor modulo the frame equations."""
        if self.known_scalar_unit(f):
            return True
        if f not in self._regular:
            from .field_wide import _expanded_row
            from .native_library import ideal_request
            rows = [_expanded_row(self.variables, p) for p in (*self.rows, f)]
            result, = ideal_request(f"graph_regular {len(self.variables)} {len(self.rows)}", rows)
            self._regular[f] = result == "yes"
        return self._regular[f]

    def known_scalar_unit(self, f):
        """Whether the native membership check already proved f is a nonzero rational on this frame."""
        return f in self._scalar_units

    def vanishes(self, f):
        """Whether f is zero on N."""
        if f == 0:
            return True
        if not self.variables:
            return self.empty()
        if not self.rows and 0 not in self.units:
            return False
        if f in self._vanishing:
            return self._vanishing[f]
        result = _membership(self.variables, self.rows, f, self._known_nonempty)
        if result != "unknown":
            # Both "no" and "scalar" identify a nonzero rational. The latter
            # still needs a decision about the open frame itself.
            answer = result == "yes" or (result == "scalar" and self.empty())
            if not answer:
                self._scalar_units.add(f)
            self._vanishing[f] = answer
            return answer
        residue = self.reduced(f)
        if not residue:
            answer = True
        elif self._basis is None and self.coords and self.units and self._witness(residue):
            answer = False
        else:
            answer = self.basis().contains(residue)
        self._vanishing[f] = answer
        return answer

    def _witness(self, f):
        """One point on the open frame with f nonzero disproves membership.
        Specialization only supplies witnesses. If these fibres miss the
        open set, use the exact saturated ideal of the whole chart."""
        for start in (2, -3, 5):
            fixed = tuple(v - (start+i) for i, v in enumerate(self.coords))
            if not self._empty_on(fixed, (f,)):
                return True
        return False

    def jet(self, f, order):
        """f and its nonvanishing derivatives of orders strictly smaller than order."""
        key = (f, order)
        if key not in self._jets:
            from . import native_library
            if order <= 1 or not self.coords or 0 in self.units:
                return (f,)
            if native_library._MODE == "sympy":
                from tests.centre_weight_oracle import jet
                return jet(self, f, order)
            from .affine_kernel import _request, _ring, _sparse_expression
            if self.rows:
                from .field_wide import _expanded_row
                lines = self._implicit_input(f)
                lines.extend((str(order), *self._basis_lines(self.variables, self.basis()), str(len(self.coords))))
                for k in self.coords:
                    field = self.field(k)
                    lines.extend((str(int(len(field) == 1 and field.get(k) == 1)), str(len(field))))
                    for axis,c in field.items():
                        lines.extend((str(self.variables.index(axis)), _expanded_row(self.variables,c)))
                rows = _request(["implicit_jet", *lines])
            else:
                ring = _ring(self.variables)
                rows = _request(["affine_jet", ring.header, str(order),
                                 " ".join(str(self.variables.index(c)) for c in self.coords), ring.text(f)])
            result = (f, *(_sparse_expression(self.variables, row) for row in rows[1:]))
            if native_library.CHECK:
                from tests.centre_weight_oracle import jet
                if result != jet(self, f, order):
                    raise AssertionError("The native frame jet differs from iterated differentiation.")
            self._jets[key] = result
        return self._jets[key]

    @staticmethod
    def _basis_lines(variables, basis):
        from .field_wide import _expanded_row
        return (str(len(basis.exprs)), *(_expanded_row(variables, f) for f in basis.exprs))

    def _implicit_input(self, f):
        from .field_wide import _expanded_row
        if self._reducer is None:
            self._reducer = groebner(self.rows, self.variables)
        return [str(len(self.variables)), *self._basis_lines(self.variables, self._reducer),
                _expanded_row(self.variables, f)]

    def empty(self, *gens):
        return self._empty_on(gens)

    def inside(self, unit, gens):
        """Whether the zero set of the ideal lies in that of `unit`, on N."""
        return self._empty_on(tuple(gens), (unit,))

    def _empty_on(self, gens, extra_units=()):
        key = (gens, extra_units)
        if key not in self._empty:
            self._empty[key] = frame_locus(self.variables, (*self.rows, *gens), (*self.units, *extra_units))
            if not self._empty[key]:
                self._known_nonempty = True
        return self._empty[key]

    def restricted(self, z, k, unit, *, witness_loci=()):
        """The submanifold z = 0 of N, where D_k z is invertible, on the open set unit != 0."""
        child = Frame(self.variables, tuple(c for c in self.coords if c != k), (*self.rows, z),
                      (*self.units, unit), self.solved)
        # A hit in the parent witnesses this child only if it lies on z=0
        # and all the child's units were inverted in the nonemptiness test.
        for gens in witness_loci:
            if (self._empty.get((gens, (unit,))) is False
                    and _membership(self.variables, (*self.rows, *gens), z, True) == "yes"):
                child._known_nonempty = True
                break
        return child

    def eliminated(self, z, k, coefficients=None):
        """The submanifold z = 0 when z = a*k + b and a is invertible: k = -b/a
        there. Returns the frame without k and the map taking a function to a
        polynomial that is a unit times its restriction (the numerator)."""
        a, b = frame_linear(self.variables, z, k) if coefficients is None else coefficients

        def move(f):
            return frame_restrict(self.variables, f, k, a, b)

        rows = tuple(g for g in map(move, self.rows) if g != 0)
        return Frame(tuple(v for v in self.variables if v != k), tuple(c for c in self.coords if c != k), rows,
                     (*map(move, self.units), a), (*self.solved, z)), move

    def changed(self, swap):
        """The same submanifold after a substitution of the coordinates."""
        def move(gens, variables):
            return tuple(frame_function(variables, chart_expressions(variables).sympify(g).xreplace(swap)) for g in gens)
        # Saved graph equations still use the eliminated coordinates.
        extra = set().union(*(g.free_symbols for g in self.solved)) - set(self.variables)
        original = (*self.variables, *sorted(extra, key=chart_expressions(self.variables).default_sort_key))
        return Frame(self.variables, self.coords, move(self.rows, self.variables),
                     move(self.units, self.variables), move(self.solved, original))
