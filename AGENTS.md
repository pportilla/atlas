# Atlas mathematical operations

Implement every new mathematical computation in C++ under `native/`. This
rule covers arithmetic, invariants, geometric predicates, algorithmic
shortcuts, certificate verification, and fallbacks. Every production entry
point (prompt, script, CLI, HTTP, worker, and saved-record replay) must call
the same native computation. A missing native operation must be implemented
in C++ before exposing it.

Use Python for orchestration, input validation, transport, presentation, and
tests. Independent Python reference calculations belong in tests. When
migrating an operation, remove its superseded Python implementation and
exercise the public entry points, including wide exponents and coefficients.
Run `make check-native-boundary` when changing production Python. Code review
must also check that orchestration delegates mathematical decisions to C++.

Expose new user-facing mathematical computations as functions that return
structured values. Register them in the shared session in `engine/session.py`
so the same function, fields, and arguments work at the prompt and in scripts.
Keep printing and command-line formatting at the interface boundary.

Carry coordinate spaces with polynomials and resolution ownership with linked
geometric objects. For operations on rational polynomials, default the input
coefficient field to `QQ` and accept `coeffs=QQ` when declaring coordinates.
The solver determines the geometric field of the result.

For each new mathematical operation that Singular supports, add repeatable
timing tests on the same inputs in both systems. Check mathematical agreement,
run multiple trials, and record both timings and the comparison conditions.
Use slower cases to guide optimization and rerun the original cases after
each change, aiming to beat Singular consistently across the comparison
suite. Report any remaining slower cases.
