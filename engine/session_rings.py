"""Ring names, the active ring and ring deletion for one Atlas session.

Every ring that owns a saved value has a name. Atlas calls a ring R1, R2, …
until you assign it to a name of your own. Choosing names, the active ring and
deletions only rebind session names. They never construct or change a ring.
"""

from fractions import Fraction
from types import MethodType
from typing import NamedTuple

from .session_algebra import (Ideal, _AbsoluteFactorization, _EliminatedIdeal, _Normalization, _NormalizedComponent, _Resultant,
                              _RingValue, declare_ideal, polynomial as read_polynomial)
from .session_ideal_native import ideal_constructor
from .session_maps import RingHomomorphism
from .session_polynomial import QQ, Polynomial, _Coordinates, _CoordinateSpace
from .session_quotient import QuotientElement, QuotientRing
from .session_values import Branch, Chart, Divisor, DualComplex, Resolution


# Named calls with two positional arguments whose results always lie in a ring
# of an operand, in QQ, or in the active ring. Only these run natively as a
# group in scripts, so every other call names its new rings statement by statement,
# exactly as at the prompt. Session variables cannot shadow these built-in names.
RING_PRESERVING_CALLS = frozenset({
    "gcd", "ideal", "ideal_power", "ideal_product", "ideal_sum", "intersection", "local_normal_form",
    "localization", "localize", "membership", "normal_form", "print", "quotient", "saturate",
    "saturation", "symbolic_power",
})


# Saved values of these types never refer to a ring.
_PLAIN = frozenset({int, bool, str, Fraction, float, type(None)})


def owner(value):
    """Return the parent under which the workspace lists a value, or None.

    A chart stands in for its own ring until that ring has been constructed,
    because constructing it can involve a number-field calculation.
    """
    if type(value) in _PLAIN:
        return None
    if isinstance(value, (_CoordinateSpace, QuotientRing)):
        return value
    if isinstance(value, (Polynomial, QuotientElement, _Coordinates)):
        return value.parent
    if isinstance(value, _RingValue):
        return value._space
    if isinstance(value, Resolution):
        return value._source_space
    if isinstance(value, Chart):
        if "_space" in vars(value):
            return value._space
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], (Polynomial, QuotientElement)):
        space = value[0].parent
        if all(isinstance(item, (Polynomial, QuotientElement)) and item.parent is space for item in value):
            return space
    return None


def is_ring(value):
    """Polynomial, chart and quotient rings can be named, selected and deleted. QQ is built in."""
    return isinstance(value, (_CoordinateSpace, QuotientRing)) and value is not QQ


def _collection(value):
    return isinstance(value, (dict, list, tuple)) and not isinstance(value, _Coordinates)


def _collection_rings(value, cache):
    if id(value) in cache:
        return cache[id(value)]
    if isinstance(value, (tuple, list)) and value:
        first = value[0]
        if isinstance(first, (Polynomial, QuotientElement)) and not isinstance(first, _Resultant):
            # A homogeneous polynomial list has one parent. A resultant can
            # also carry an inclusion, so handle mixed types individually.
            kind, parent = type(first), first.parent
            if all(type(item) is kind and item.parent is parent for item in value):
                return referenced_rings(parent)
    ordered, seen = {}, set()
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) in _PLAIN:
            continue
        if _collection(item):
            key = id(item)
            if key in seen:
                continue
            seen.add(key)
            if key in cache:
                ordered.update(dict.fromkeys(cache[key]))
            else:
                contents = item.values() if isinstance(item, dict) else item
                pending.extend(reversed(contents))
        else:
            ordered.update(dict.fromkeys(referenced_rings(item, cache)))
    rings = tuple(ordered)
    cache[id(value)] = rings
    return rings


def referenced_rings(value, containers=None):
    """Read ring dependencies from live metadata in declaration order, foundations first.

    A deletion shares the collection cache across bindings, since several saved
    names can point to the same large list or record.
    """
    if type(value) in _PLAIN:
        return ()
    if isinstance(value, RingHomomorphism):
        items = (value.domain, value.codomain)
    elif isinstance(value, MethodType):
        return referenced_rings(value.__self__, containers)
    elif isinstance(value, (_AbsoluteFactorization, _EliminatedIdeal, _Resultant)):
        # The inclusion retains the original ring even when the result is a scalar.
        items = (value.inclusion,)
    elif isinstance(value, _Normalization):
        items = (value.ambient, *value.components)
    elif isinstance(value, _NormalizedComponent):
        items = (value.prime.parent, value.ambient)
    elif isinstance(value, (Chart, Divisor, Branch, DualComplex)):
        # Inspect the chart ring only if it has already been constructed.
        items = (value._resolution._source_space, vars(value).get("_space"))
    elif _collection(value):
        return _collection_rings(value, {} if containers is None else containers)
    else:
        parent = owner(value)
        if not is_ring(parent):
            return ()
        if not isinstance(parent, QuotientRing):
            return (parent,)
        # Every further quotient retains its intermediate parents, down to the ambient ring.
        chain = [parent]
        while isinstance(parent, QuotientRing):
            defining = parent.defining_ideal
            if isinstance(defining, _EliminatedIdeal):
                # K.defining_ideal.inclusion still exposes both endpoint rings.
                return (*referenced_rings(defining, containers), *reversed(chain))
            parent = defining.parent
            if is_ring(parent):
                chain.append(parent)
        return tuple(reversed(chain))
    return tuple(dict.fromkeys(ring for item in items for ring in referenced_rings(item, containers)))


class RingDeletion(NamedTuple):
    """The names removed with a ring, and whether the active ring was among the deleted rings."""

    ring: str
    names: tuple
    was_active: bool

    def session_attribute(self, name):
        if name in self._fields:
            return getattr(self, name)
        raise AttributeError(f"A ring deletion has no Atlas field {name!r}.")

    def __repr__(self):
        text = f"Deleted {self.ring}" + (f": {', '.join(self.names)}" if self.names else "")
        return text + ("\nNo ring is active." if self.was_active else "")

    __str__ = __repr__


class SessionRings:
    """Keep a name for every ring in use, and the ring that new polynomials use by default."""

    def __init__(self, session):
        self._session = session
        self.active = None
        self._ring_of = {}       # Every saved name currently bound to a ring.
        self._name_of = {}       # One of those names for each ring.
        self._automatic = set()  # The names Atlas chose.
        self._checkpoint = False  # False when no statement is being watched.
        # A live view of the saved names that refer to rings. Restoring a
        # state updates the registry in place, so this view stays current.
        self.names = self._ring_of.keys()

    def is_automatic(self, name):
        """Whether name is a name that Atlas chose for a ring and still refers to that ring."""
        return name in self._automatic and self._ring_of.get(name) is self._session.variables.get(name)

    # The registry records names bound by the evaluator. Compare it with the
    # live bindings, since a failed prompt command restores earlier values.
    def name_of(self, ring):
        """Return a saved name of the ring, or None."""
        variables = self._session.variables
        name = self._name_of.get(ring)
        if name is not None and variables.get(name) is ring:
            return name
        for name, value in self._ring_of.items():
            if value is ring and variables.get(name) is ring:
                self._name_of[ring] = name
                return name
        return None

    def may_group(self, targets):
        """Native groups of calls may run unless one of them rebinds a ring name."""
        return targets.isdisjoint(self._ring_of)

    def note(self, name, value, *, naming=False, known_collection=False):
        """Record a new binding and name each ring it introduces.

        naming marks a direct assignment. Assigning a ring to a name of your
        own replaces the name that Atlas chose for it.
        """
        if name in self._ring_of:
            self._unbind(name, value)
        if type(value) in _PLAIN or (known_collection and _collection(value)):
            return
        rings = referenced_rings(value)
        if not rings:
            return
        if is_ring(value):
            if naming:
                self._withdraw_automatic(value, name)
            self._ring_of[name] = value
            if self.name_of(value) is None:
                self._name_of[value] = name
        for ring in rings:
            if self.name_of(ring) is None:
                self._name_automatically(ring)

    def _unbind(self, name, value):
        ring = self._ring_of.pop(name)
        self._automatic.discard(name)
        if self._name_of.get(ring) == name:
            del self._name_of[ring]
        if ring is not value and self.name_of(ring) is None and self._in_use(ring):
            self._name_automatically(ring)

    def _in_use(self, ring):
        if self.active is not None and ring in referenced_rings(self.active):
            return True
        containers = {}
        return any(ring in referenced_rings(value, containers) for value in self._session.variables.values())

    def _withdraw_automatic(self, ring, keep):
        variables = self._session.variables
        for alias in [alias for alias in self._automatic if alias != keep and self._ring_of.get(alias) is ring]:
            if variables.get(alias) is ring:
                del variables[alias]
            del self._ring_of[alias]
            self._automatic.discard(alias)
            if self._name_of.get(ring) == alias:
                del self._name_of[ring]

    def _name_automatically(self, ring):
        session = self._session
        index = 1
        while (name := f"R{index}") in session.variables or name in session.builtins or name in session.reserved:
            index += 1
        session.variables[name] = ring
        self._ring_of[name] = ring
        self._name_of[ring] = name
        self._automatic.add(name)
        return name

    # A prompt command that fails leaves no trace. Statements that change the
    # active ring or delete names save the session before their first change.
    def state(self):
        return self.active, dict(self._ring_of), dict(self._name_of), set(self._automatic)

    def restore(self, state):
        self.active, ring_of, name_of, automatic = state
        for registry, saved in ((self._ring_of, ring_of), (self._name_of, name_of), (self._automatic, automatic)):
            registry.clear()
            registry.update(saved)

    def watch(self):
        self._checkpoint = None

    def unwatch(self):
        """Stop watching and return the state saved before the statement's first change, if any."""
        saved, self._checkpoint = self._checkpoint, False
        return saved or None

    def _save(self):
        if self._checkpoint is None:
            self._checkpoint = (self._session.variables.copy(), self.state())

    def default(self):
        """Return the active ring for an operation that was given no ring."""
        if self.active is None:
            raise ValueError("No ring is active. Pass ring=f.parent, or choose a default with use_ring(f.parent).")
        return self.active

    def use(self, ring):
        if ring is QQ:
            raise ValueError("QQ has no coordinates. Choose a polynomial or quotient ring.")
        if ring is not None and not is_ring(ring):
            raise TypeError("Pass a ring, for example use_ring(R1) or use_ring(f.parent), or None.")
        self._save()
        self.active = ring
        if ring is not None:
            for each in referenced_rings(ring):
                if self.name_of(each) is None:
                    self._name_automatically(each)
        return ring

    def delete(self, ring):
        if ring is QQ:
            raise ValueError("QQ is built in and cannot be deleted.")
        if not is_ring(ring):
            raise TypeError("Pass a ring, for example delete_ring(R1) or delete_ring(f.parent).")
        variables = self._session.variables
        containers = {}
        names = tuple(name for name, value in variables.items() if ring in referenced_rings(value, containers))
        was_active = ring in referenced_rings(self.active, containers)
        self._save()
        for name in names:
            del variables[name]
            self._automatic.discard(name)
            if self._ring_of.pop(name, None) is not None:
                for named in [named for named, alias in self._name_of.items() if alias == name]:
                    del self._name_of[named]
        if was_active:
            self.active = None
        return RingDeletion(repr(ring), names, was_active)


def _carries_ring(generators):
    if len(generators) == 1 and isinstance(generators[0], (tuple, list)):
        generators = generators[0]
    return any(isinstance(f, (Polynomial, QuotientElement)) for f in generators)


def ring_bindings(rings):
    """Return the session functions that name, select and delete rings, and the readers that use the active ring."""

    def use_ring(ring):
        """Make a ring the default for new polynomials and ideals, and return that ring.

        >>> x, y = coordinates("x", "y")
        >>> use_ring(x.parent)  # QQ[x, y]
        >>> f = polynomial("x^2 + y")
        >>> f.parent  # QQ[x, y]

        Docs: docs/tutorial/rings.md#name-select-and-delete-rings
        """
        return rings.use(ring)

    def active_ring():
        """Return the ring that new polynomials and ideals use by default, or None.

        >>> x, y = coordinates("x", "y")
        >>> use_ring(x.parent)  # QQ[x, y]
        >>> active_ring() == x.parent  # True

        Docs: docs/tutorial/rings.md#name-select-and-delete-rings
        """
        return rings.active

    def ring_name(ring):
        """Return a saved name of the given ring in this session, or None.

        >>> x = coordinates("x")
        >>> ring_name(x.parent)  # R1
        >>> A = x.parent
        >>> ring_name(A)  # A

        Docs: docs/tutorial/rings.md#name-select-and-delete-rings
        """
        if ring is QQ:
            return "QQ"
        if not is_ring(ring):
            raise TypeError("Pass a live ring, for example ring_name(f.parent).")
        return rings.name_of(ring)

    def delete_ring(ring):
        """Delete a ring with every saved value in it, and return the names removed.

        >>> x, y = coordinates("x", "y")
        >>> f = y^2 - x^3
        >>> delete_ring(x.parent)  # Deleted QQ[x, y]: x, y, R1, f

        Docs: docs/tutorial/rings.md#name-select-and-delete-rings
        """
        return rings.delete(ring)

    def polynomial(source, *, ring=None, variables=None):
        """Return the polynomial written in source, read in ring= or else in the active ring.

        >>> x, y = coordinates("x", "y")
        >>> g = polynomial("x^2+y", ring=x.parent)
        >>> use_ring(x.parent)  # QQ[x, y]
        >>> h = polynomial("x*y")
        >>> g + h  # x^2+x*y+y

        Docs: docs/tutorial/rings.md#choose-the-ring-of-a-polynomial
        """
        if ring is None and variables is None:
            ring = rings.default()
        return read_polynomial(source, ring=ring, variables=variables)

    def ideal(*generators, order=None, variables=None, ring=None):
        # Polynomial generators select their own parent. The active ring holds
        # an empty ideal or one generated by rational numbers.
        if ring is None and variables is None and not _carries_ring(generators):
            if rings.active is None:
                raise ValueError("For an empty ideal or one generated by numbers, pass ring=f.parent, ring=QQ "
                                 "or variables=(x, y), or choose a default ring with use_ring(f.parent).")
            ring = rings.active
        return declare_ideal(*generators, order=order, variables=variables, ring=ring)

    ideal.__doc__ = declare_ideal.__doc__
    return {"use_ring": use_ring, "active_ring": active_ring, "delete_ring": delete_ring, "ring_name": ring_name,
            "polynomial": polynomial, "ideal": ideal_constructor(ideal, Polynomial, Ideal)}
