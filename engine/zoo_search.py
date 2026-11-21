"""Transport the Resolution Zoo's compact index and queries to its native matcher."""
import ctypes
from functools import lru_cache
import json
from pathlib import Path
import sys

from .session_polynomial import Polynomial, QQ

ROOT = Path(__file__).resolve().parents[1]
_INDEX = ROOT / "web" / "zoo-data" / "index.json"


@lru_cache(maxsize=1)
def _library():
    suffix = "dylib" if sys.platform == "darwin" else "so"
    try:
        lib = ctypes.CDLL(str(ROOT / "native" / "build" / f"libzoo_search.{suffix}"))
    except OSError as error:
        raise RuntimeError("Build the Resolution Zoo search kernel with make ordinary.") from error
    lib.atlas_zoo_create.argtypes = [ctypes.c_char_p]
    lib.atlas_zoo_create.restype = ctypes.c_void_p
    lib.atlas_zoo_destroy.argtypes = [ctypes.c_void_p]
    lib.atlas_zoo_destroy.restype = None
    lib.atlas_zoo_search.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.atlas_zoo_search.restype = ctypes.c_char_p
    lib.atlas_zoo_error.argtypes = []
    lib.atlas_zoo_error.restype = ctypes.c_char_p
    return lib


class _Catalogue:
    def __init__(self, entries):
        lines = []
        for entry in entries:
            if not entry["id"] or any(c in entry["id"] for c in "\t\r\n\0"):
                raise ValueError("Invalid Resolution Zoo record ID.")
            terms = [" ".join([t["coefficient"], *t["exponents"]]) for t in entry["terms"]]
            lines.append("\t".join([entry["id"], *terms]))
        wire = "\n".join(lines)
        if "\0" in wire:
            raise ValueError("Unexpected null byte in Resolution Zoo index.")
        self.lib = _library()
        self.handle = self.lib.atlas_zoo_create(wire.encode())
        if not self.handle:
            raise ValueError(self.lib.atlas_zoo_error().decode())

    def search(self, query):
        if isinstance(query, (list, tuple)):
            if any(not isinstance(term, str) or not term.strip() for term in query):
                raise ValueError("Pass a list of nonempty monomial strings.")
            query = ",".join(query)
        if not isinstance(query, str) or "\0" in query:
            raise ValueError("Pass a monomial query string or a list of monomial strings.")
        result = self.lib.atlas_zoo_search(self.handle, query.encode())
        if result is None:
            raise ValueError(self.lib.atlas_zoo_error().decode())
        return {"ids": result.decode().splitlines()}

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.atlas_zoo_destroy(self.handle)


@lru_cache(maxsize=1)
def _installed_index(path, mtime_ns, size):
    # File identity is part of the cache key so a regenerated index is visible.
    return _Catalogue(json.loads(path.read_bytes())["entries"])


def zoo_search(query):
    """Return the IDs of Resolution Zoo records containing all requested monomials.

    >>> hits = zoo_search("x^2, y^2")
    >>> print(len(hits.ids) > 0)  # True

    Docs: docs/reference/resolution.md#search-the-resolution-zoo
    """
    terms = query if isinstance(query, (list, tuple)) else [query]
    if any(isinstance(term, Polynomial) for term in terms):
        if not all(isinstance(term, Polynomial) for term in terms):
            raise ValueError("Pass rational polynomials together or supply the query as text.")
        parent = terms[0].parent
        if parent.has_relations or parent.coeffs is not QQ:
            raise ValueError("Zoo polynomial queries require polynomial coordinates over QQ.")
        if any(term.parent is not parent for term in terms):
            raise ValueError("Query polynomials belong to different coordinate spaces.")
        query = [str(term) for term in terms]
    info = _INDEX.stat()
    return _installed_index(_INDEX, info.st_mtime_ns, info.st_size).search(query)
