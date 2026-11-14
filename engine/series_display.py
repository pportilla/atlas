"""Exact display of a sparse Hilbert numerator over an ambient polynomial ring."""
from .polynomial import integer_text


def format_hilbert_series(numerator, denominator_power):
    if not numerator:
        return "0"
    parts = []
    for term in numerator:
        degree, coefficient = term["degree"], term["coefficient"]
        if coefficient < 0:
            parts.append(" - " if parts else "-")
        elif parts:
            parts.append(" + ")
        magnitude = abs(coefficient)
        if degree == 0:
            parts.append(integer_text(magnitude))
        else:
            if magnitude != 1:
                parts.append(integer_text(magnitude) + "*")
            parts.append("t" if degree == 1 else "t^" + integer_text(degree))
    top = "".join(parts)
    if not denominator_power:
        return top
    if len(numerator) > 1:
        top = "(" + top + ")"
    power = "" if denominator_power == 1 else "^" + integer_text(denominator_power)
    return top + "/(1 - t)" + power
