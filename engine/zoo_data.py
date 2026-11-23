"""Locate the separate Resolution Zoo checkout used by Atlas and its local server."""

import os
from pathlib import Path


def catalogue_directory():
    configured = os.environ.get("ATLAS_ZOO_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "resolution-zoo"


def catalogue_file(relative):
    directory = catalogue_directory().resolve()
    path = directory / relative
    # A catalogue filename must not redirect the server to .git/config or another local file.
    if path.resolve() != path or not path.is_relative_to(directory):
        raise FileNotFoundError("Unknown catalogue record.")
    if not (directory / "index.json").is_file():
        raise RuntimeError("Clone https://github.com/pportilla/resolution-zoo next to Atlas "
                           "or set ATLAS_ZOO_DIR to the catalogue checkout.")
    return path
