"""Replay completed native calls through both settings and compare exact replies.

Use a recording directory, JSONL file, or JSONL.gz archive. Pending JSON inputs
left by a cancelled solver lane have no captured reply and are available in the
recording directory for profiling.
"""

import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def json_symbols():
    # The JSON transport has one specific C ABI. Other exported functions
    # have different signatures and must never be invoked from corpus data.
    signature = re.compile(r'extern\s+"C"\s+const\s+char\s*\*\s*(atlas_\w+)'
                           r'\(\s*const\s+char\s*\*\s*\w*\s*\)')
    return {symbol for source in (ROOT / "native").glob("*.cpp")
            for symbol in signature.findall(source.read_text())}


def replay(corpus):
    from engine.kernel_json import _json_reply
    from engine.native_library import _affine_reply, _library
    symbols = json_symbols()
    counts = Counter()
    paths = (sorted([*corpus.rglob("requests-*.jsonl"), *corpus.rglob("*.jsonl.gz")])
             if corpus.is_dir() else [corpus])
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            for i, line in enumerate(stream, 1):
                row = json.loads(line)
                transport = row.get("transport")
                if transport == "json":
                    symbol = row["symbol"]
                    if symbol not in symbols:
                        raise ValueError(f"{path}:{i}: unknown JSON kernel {symbol!r}")
                    actual = _json_reply(symbol, row["request"]).decode("utf-8")
                    error = None
                    operation = json.loads(row["request"])["operation"]
                elif transport == "affine":
                    lines = row["request"].splitlines()
                    operation = lines[0]
                    try:
                        actual = _affine_reply(_library(), row["request"].encode("utf-8"), row.get("seconds")).decode("ascii")
                        error = None
                    except ArithmeticError as exc:
                        actual, error = None, str(exc)
                else:
                    raise ValueError(f"{path}:{i}: unknown transport {transport!r}")
                if (actual, error) != (row["response"], row.get("error")):
                    raise ValueError(f"{path}:{i}: {transport}/{operation} reply differs "
                                     f"with ATLAS_LINEAR_ALGEBRA={os.environ['ATLAS_LINEAR_ALGEBRA']}\n"
                                     f"expected {str(row.get('error') or row['response'])[:240]}\n"
                                     f"received {str(error or actual)[:240]}")
                counts[f"{transport}/{operation}"] += 1
    if not counts:
        raise ValueError(f"No requests in {corpus}")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="Recording directory or a JSONL or JSONL.gz file")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.worker:
        sys.path.insert(0, str(ROOT))
        print(json.dumps(replay(options.corpus), sort_keys=True))
        return
    env = os.environ.copy()
    env.pop("ATLAS_RECORD_REQUESTS", None)
    for switch in ("0", "1"):
        env["ATLAS_LINEAR_ALGEBRA"] = switch
        counts = json.loads(subprocess.check_output(
            [sys.executable, __file__, options.corpus, "--worker"], env=env, text=True))
        print(f"ATLAS_LINEAR_ALGEBRA={switch}: {sum(counts.values())} exact replies matched")


if __name__ == "__main__":
    main()
