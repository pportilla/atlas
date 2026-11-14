"""Serialize exact algebra results without changing the host integer conversion policy."""
import json
from fractions import Fraction

from .polynomial import integer_text, rational_text


def sorted_record(value):
    """Encode an exact witness, including its rational entries, in canonical key order."""
    if isinstance(value, dict):
        return {key: sorted_record(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [sorted_record(item) for item in value]
    if isinstance(value, Fraction):
        return rational_text(value)
    return value


def _result_json(value):
    """Encode a worker record with exact JSON integers under the host's digit limit."""
    if type(value) is int:
        return integer_text(value)
    if isinstance(value, dict):
        fields = (f"{json.dumps(key, ensure_ascii=False)}: {_result_json(item)}" for key, item in value.items())
        return "{" + ", ".join(fields) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_result_json(item) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False)


def result_json(record, *, indent=None):
    try:
        return json.dumps(record, indent=indent, ensure_ascii=False)
    except ValueError:
        # Computed multiplicities and quotient dimensions can exceed the
        # interpreter's integer-to-decimal conversion limit.
        return _result_json(record)
