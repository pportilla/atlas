"""Small local catalogues for the Zoo interfaces; tests never need network access."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch


class TestZooMixin:
    """Use a temporary catalogue for this test class, including child Atlas processes."""

    zoo_records = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        if cls.zoo_records:
            make_browser_catalogue(directory)
        else:
            entries = [
                {"id": "node", "terms": [{"coefficient": "1", "exponents": ["2", "0"]},
                                           {"coefficient": "1", "exponents": ["0", "2"]}]},
                {"id": "whitney", "terms": [{"coefficient": "1", "exponents": ["2", "0", "0"]},
                                             {"coefficient": "-1", "exponents": ["0", "2", "1"]}]},
            ]
            (directory / "index.json").write_text(json.dumps({"schema_version": 1, "entries": entries}))
        setting = patch.dict(os.environ, ATLAS_ZOO_DIR=str(directory))
        setting.start()
        cls.addClassCleanup(setting.stop)


def make_browser_catalogue(directory):
    from tools.build_resolution_zoo import generate

    groups = [
        (["x", "y"], [("plane-cusp", "y^2-x^3")]),
        (["x", "y", "z"], [("a1", "x^2+y^2+z^2"), ("a2", "x*y-z^3"), ("a3", "x*y-z^4"),
                           ("whitney-umbrella", "x^2-y^2*z"),
                           ("mixed-quartic", "x^4+y^4+z^4-1/2*x*y^3+x*y*z^2")]),
        (["x", "y", "z", "t"], [("threefold-node", "x*y-z*t")]),
        (["x", "y", "z", "t", "u"], [("fourfold-node", "x^2+y^2+z^2+t^2+u^2")]),
        (["x", "y", "z", "t", "u", "v"], [("fivefold-node", "x^2+y^2+z^2+t^2+u^2+v^2")]),
    ]
    seeds = directory / "seeds.json"
    seeds.write_text(json.dumps([
        {"dimension": len(variables) - 1, "variables": variables,
         "entries": [{"id": ident, "name": ident.replace("-", " ").capitalize(),
                      "equation": equation, "labels": []} for ident, equation in cases]}
        for variables, cases in groups
    ]))
    with contextlib.redirect_stdout(io.StringIO()):
        generate(seeds, directory, jobs=2)
    # Enough equivalent entries to exercise a second page without extra solver runs.
    record = json.loads((directory / "records" / "a1.json").read_text())
    index_path = directory / "index.json"
    index = json.loads(index_path.read_text())
    entry = next(row for row in index["entries"] if row["id"] == "a1")
    for number in range(40):
        ident = f"fixture-node-{number}"
        index["entries"].append({**entry, "id": ident})
        (directory / "records" / (ident + ".json")).write_text(json.dumps({**record, "id": ident}))
    index_path.write_text(json.dumps(index))
