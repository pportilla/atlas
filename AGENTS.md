# Atlas mathematical operations

Expose new user-facing mathematical computations as functions that return
structured values. Register them in the shared session in `engine/session.py`
so the same function, fields, and arguments work at the prompt and in scripts.
Keep printing and command-line formatting at the interface boundary.

Carry coordinate spaces with polynomials and resolution ownership with linked
geometric objects. For operations on rational polynomials, default the input
coefficient field to `QQ` and accept `coeffs=QQ` when declaring coordinates.
The solver determines the geometric field of the result.
