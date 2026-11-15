"""Render a short Atlas usage card for the terminal."""

from textwrap import indent


def usage_card(summary, examples, reference, *, signature=None):
    heading = f"{signature}\n{summary}" if signature else summary
    url = f"https://github.com/pportilla/atlas/blob/main/{reference}"
    return f"{heading}\n\nExample\n{indent(examples, '  ')}\n\nDocs  {url}"
