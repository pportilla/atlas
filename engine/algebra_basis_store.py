"""Server-owned references to exact Gröbner bases returned by isolated workers."""

from collections import OrderedDict
import json
import secrets
from threading import Lock

from .algebra import calculate_algebra, error_record


class AlgebraBasisStore:
    def __init__(self, *, max_entries=128, max_bytes=16 << 20):
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._bases = OrderedDict()
        self._bytes = 0
        self._lock = Lock()

    def _remember(self, basis):
        # Serialized entries are immutable and their aggregate storage is bounded.
        data = json.dumps([basis["ring"], basis["generators"]], separators=(",", ":")).encode()
        if len(data) > self.max_bytes or self.max_entries < 1:
            return None
        ref = secrets.token_urlsafe(24)
        with self._lock:
            while len(self._bases) >= self.max_entries or self._bytes + len(data) > self.max_bytes:
                _, removed = self._bases.popitem(last=False)
                self._bytes -= len(removed)
            self._bases[ref] = data
            self._bytes += len(data)
        return ref

    def _lookup(self, ref):
        with self._lock:
            data = self._bases.get(ref)
            if data is None:
                return None
            self._bases.move_to_end(ref)
        return json.loads(data)

    def calculate(self, request, timeout, *, owner=None):
        operands = request.get("operands") if isinstance(request, dict) else None
        if isinstance(operands, dict) and "basis_ref" in operands:
            if request.get("operation") not in ("normal-form", "membership", "multiplicity") or "ideal" in operands:
                return 400, error_record("invalid_request",
                    "Use basis_ref in place of ideal for normal-form, membership or multiplicity.")
            ref = operands["basis_ref"]
            if not isinstance(ref, str) or not ref:
                return 400, error_record("invalid_request", "basis_ref must be a nonempty string.")
            basis = self._lookup(ref)
            if basis is None:
                return 410, error_record("stale_basis_ref",
                    "This server has no saved basis with that reference. Compute the basis again.")
            basis_ring, generators = basis
            ring = request.get("ring")
            if not isinstance(ring, dict):
                return 400, error_record("invalid_request", "ring must be an object.")
            if {"field": "QQ", "order": "grevlex", **ring} != basis_ring:
                return 400, error_record("basis_ring_mismatch",
                    "The saved basis requires its original field, ordered variables and monomial order.")
            expanded = dict(operands, ideal=generators)
            del expanded["basis_ref"]
            # Only a stored server result can set this worker argument. The
            # expanded request also yields portable command and script replays.
            return calculate_algebra(dict(request, operands=expanded), known_basis=True, timeout=timeout, owner=owner)

        status, record = calculate_algebra(request, timeout=timeout, owner=owner)
        if status == 200 and record["operation"] == "groebner":
            ref = self._remember(record["result"])
            if ref is not None:
                record["basis_ref"] = ref
        return status, record
