"""Private Linux supervisor entry point for the regular resolve CLI."""
import os
import sys

# Match the public atlas script's import path when invoked from another cwd.
sys.path[0] = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
from engine.resolve_cli import main

if __name__ == "__main__":
    control = sys.argv.pop(1)
    unlimited = ("int_max_str_digits" not in sys._xoptions and
                 (sys.flags.ignore_environment or not os.environ.get("PYTHONINTMAXSTRDIGITS")))
    raise SystemExit(main(unlimited_decimals=unlimited, supervisor=control))
