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

In the shared session, use live parent objects to check ownership before
serializing a polynomial or coordinate. Identically spelled generators can
belong to independent rings. Reject a direct coordinate declaration that
would replace a saved binding before changing any name. An explicit
`fresh=True` requests an independent ring and permits rebinding. Reuse the
parent for repeated ordinary declarations in one session, and use explicit
maps with a domain, codomain and ordered generator images for transport.
Test changes to these rules with independent same-named
rings at the prompt and in scripts, including what survives a failed command.

A ring owns its ordered coordinates, including an empty coordinate tuple.
For a quotient `K`, `K.coordinates` belongs to `K` and
`K.ambient.coordinates` belongs to its polynomial ring. A saved name is
an alias for its value. Renaming a value or selecting a compute backend must
preserve its live parent. Group workspace bindings by that parent and show
maps with both endpoints. Help, completion and workspace display must be
able to describe these relationships without running a computation.

For each new mathematical operation that Singular supports, add repeatable
timing tests on the same inputs in both systems. Check mathematical agreement,
run multiple trials, and record both timings and the comparison conditions.
Use slower cases to guide optimization and rerun the original cases after
each change, aiming to beat Singular consistently across the comparison
suite. Report any remaining slower cases.

## Prompt help and public docstrings

Give each new public session function a docstring with a one-sentence first
line describing what it returns. Follow it with a short, runnable Atlas
example using `>>>` lines, including any setup it needs in a fresh session.
End with `Docs: path/to/guide.md#section` pointing to the exact section of
the user guide. Put extended explanations, conventions and exceptions in
that section. Command help uses the real signature and these three parts.

For a new public value type, add a saved-value help card. Identify the
actual value and its parent or owner from existing session metadata. Give
short expressions using the queried name and its public fields. Showing
help must leave bindings and cached mathematical results untouched.
Exercise the card at the prompt, run its examples, and check its documentation
anchor. Aliases must describe the same object or operation.
