"""The terminal wordmark and lettered map as HTML for the web workspace and the docs.

Both renderings come from the rows in engine/welcome.py. Each coloured run
becomes a span whose class names its terminal role, styled by web/identity.css.
"""
from html import escape
from textwrap import dedent

from .welcome import WORDMARK, lettered_map

_SLOTS = {
    "<!--atlas:wordmark-->": lambda: wordmark_html(),
    "<!--atlas:map-->": lambda: lettered_map_html(),
}


def _span(text, role, color=True):
    if not (color and role and text):
        return escape(text)
    return f'<span class="t-{role}">{escape(text)}</span>'


def wordmark_html():
    return "\n".join(_span(row, "brand") for row in WORDMARK)


def lettered_map_html():
    # Each row carries a two-space terminal margin outside its spans.
    return dedent("\n".join(row.rstrip() for row in lettered_map(painter=_span)))


def fill_brand(page):
    """Replace the brand placeholders in an HTML page."""
    for slot, render in _SLOTS.items():
        if slot in page:
            page = page.replace(slot, render())
    return page
