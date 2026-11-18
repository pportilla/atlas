"""Explicit homomorphisms of Atlas rings with chosen generator images."""
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property

from .session_algebra import _polynomial
from .session_polynomial import Polynomial, QQ, _CoordinateSpace
from .session_pullback import _ChartSpace
from .session_quotient import QuotientElement, QuotientRing


def _element(parent, value):
    if isinstance(parent, QuotientRing):
        if isinstance(value, QuotientElement) and value.parent is parent:
            return value
        scalar = type(value) is int or isinstance(value, Fraction)
        if scalar or (isinstance(value, Polynomial) and value.parent is QQ):
            return parent(value)
        raise ValueError("Use a residue class of the specified quotient ring. "
                         "Apply its projection to an ambient polynomial first.")
    return _polynomial(value, parent)


def _representative(element):
    return element.representative if isinstance(element, QuotientElement) else element


@dataclass(frozen=True, eq=False)
class RingHomomorphism:
    domain: _CoordinateSpace | QuotientRing
    codomain: _CoordinateSpace | QuotientRing
    images: tuple
    _fields = {"domain", "codomain", "images"}

    def session_attribute(self, name):
        if name in self._fields:
            return getattr(self, name)
        raise AttributeError(f"A ring homomorphism has no Atlas field {name!r}.")

    def __repr__(self):
        return f"hom({self.domain} → {self.codomain}, images={self.images})"

    @cached_property
    def _substitutions(self):
        source = self.domain.ambient if isinstance(self.domain, QuotientRing) else self.domain
        return {generator: _representative(image)._value
                for generator, image in zip(source.generators, self.images)}

    def _apply_representative(self, polynomial):
        target = self.codomain
        value = polynomial._value.xreplace(self._substitutions)
        if isinstance(target, QuotientRing):
            return target(Polynomial(target.ambient, value))
        return Polynomial(target, value)

    def __call__(self, value):
        element = _element(self.domain, value)
        return self._apply_representative(_representative(element))


def hom(domain, codomain, *, images):
    """Construct a QQ-algebra homomorphism from the ordered images of the domain generators.

    Pass live rings and give each image in the codomain. For a quotient or a rational
    graph chart, its defining relations must vanish under the map. Call the map on
    a domain element to substitute all images simultaneously. Charts over algebraic
    coefficient fields can serve as targets. Maps from them need a choice of
    coefficient-field embedding, which hom currently supports only for QQ.

    >>> x, y = coordinates("x", "y")
    >>> u, v, w = coordinates("u", "v", "w")
    >>> phi = hom(x.parent, u.parent, images=(u, w))
    >>> phi(x*y + 1) == u*w + 1  # True
    """
    rings = (_CoordinateSpace, QuotientRing)
    if not isinstance(domain, rings) or not isinstance(codomain, rings):
        raise TypeError("Pass live parent rings as the domain and codomain of hom.")
    if domain.coeffs is not QQ:
        raise ValueError("Maps from a chart over an algebraic coefficient field require "
                         "a choice of coefficient-field embedding. Currently hom accepts domains over QQ.")
    relations = ()
    if isinstance(domain, QuotientRing):
        relations = domain.relations
    elif domain.has_relations:
        if not isinstance(domain, _ChartSpace):
            raise ValueError("The domain must provide its defining relations.")
        relations = domain.presentation_relations
    if not isinstance(images, (tuple, list)) or len(images) != len(domain.variables):
        raise ValueError("Supply one image per domain generator in declaration order, as a tuple or list.")
    mapping = RingHomomorphism(domain, codomain, tuple(_element(codomain, image) for image in images))
    for relation in relations:
        if _representative(mapping._apply_representative(relation)):
            raise ValueError(f"The images do not annihilate the domain relation {relation}.")
    return mapping
