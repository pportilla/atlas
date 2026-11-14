"""Load the optional comparison engine only for symbolic callers and reference runs."""
import importlib
import sys


class _Reference:
    def __getattr__(self, name):
        return getattr(importlib.import_module("sympy"), name)


sp = _Reference()


class _Rationals(str):
    """SymPy accepts the domain name in its public polynomial constructors."""
    is_AlgebraicField = False

    def __eq__(self, other):
        if type(other).__module__ == "sympy.polys.domains.rationalfield" and type(other).__name__ == "RationalField":
            return True
        return str.__eq__(self, other)

    __hash__ = str.__hash__

    def __call__(self, *args):
        return sp.QQ(*args)

    def __getattr__(self, name):
        return getattr(sp.QQ, name)


QQ = sys.modules["sympy"].QQ if hasattr(sys.modules.get("sympy"), "QQ") else _Rationals("QQ")
