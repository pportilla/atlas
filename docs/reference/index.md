# Reference manual

This manual documents every function of the Atlas session and the values they return. The same functions, arguments and fields work at the prompt and in scripts. To look up a name, use the [function index](genindex.md). At the prompt, `help NAME` prints the short form of an entry, and `help Z` describes a value you saved as `Z`.

Each subject page opens with a short introduction and a complete example, followed by one entry per function. An entry gives the signature, what each input and option means, what the function returns and which fields to read from the result, and an example that the test suite runs. It links to the guide sections that use the function, to its C++ implementation and to its tests.

In a signature, the arguments after `*` or after a variable-length argument such as `*generators` are passed by name, as in `ideal(x, order="lex")` or `resolve(f, mode="intrinsic")`.

## Session functions

### Rings and polynomials

- [Coordinates, polynomials and ring maps](rings.md). Declare, name, select and delete rings, read polynomials, factor them, and map one ring to another.

### Ideals and quotients

- [Gröbner bases and ideal operations](ideals.md). Bases, normal forms, membership, elimination and ideal arithmetic.
- [Components, localization and normalization](components.md). Radicals, primary decomposition, localization at a prime and normalization.
- [Global dimension, length and Hilbert series](dimension.md). Krull dimension, quotient bases, lengths and graded Hilbert series.

### Local algebra and geometry

- [Local algebra at a point](local-algebra.md). Standard bases, normal forms, quotient bases and Hilbert–Samuel functions at a rational point.
- [Tangent geometry and smoothness](tangent-geometry.md). Tangent cones, Zariski tangent spaces, local dimension and smoothness.
- [Hypersurface and map germ invariants](germ-invariants.md). Jacobian algebras, Milnor and Tjurina numbers, and the codimension of a map germ.

### Resolution of singularities

- [Resolution, zeta functions and monodromy](resolution.md). Resolve a germ, read its topological and monodromy zeta functions, check the monodromy conjecture and search the Resolution Zoo.
- [Resolution objects and zeta functions](resolution-values.md). Every field of a resolution, its charts, divisors, branches and dual complex, its two zeta functions and a pole check.

### Scripts

- [Script output and replay](output.md). Print values, write them as JSON and replay a saved algebra request.

## Saved values

- [Methods on saved values](saved-values.md). Residue classes in a quotient ring, fractions in a local ring and ring homomorphisms.

## Interfaces

- [Command line](command-line.md). Run scripts, resolve a germ, compute from a saved record and run algebra operations from the shell.
- [HTTP API and compute options](http-and-compute.md). Run the server, request a resolution, follow a background search, replay zeta functions and choose a device.
- [Algebra requests and replay](algebra-requests.md). Describe a ring and its operands in JSON, with the operations and the fields of their results.

## Algorithms and certificates

- [The exact kernel](../../native/KERNEL.md). How Atlas computes and verifies Gröbner bases over the rationals, with pointers to the notes below.
  - [Polynomial, ideal and quotient algorithms](../../native/ideal-algorithms.md)
  - [Local algebra algorithms](../../native/local-algorithms.md)
  - [Certificates for the extended tangent quotient](../../native/map-germ-algorithms.md)
  - [Resolution and chart algorithms](../../native/resolution-algorithms.md)

## Indices

- [Function index](genindex.md). Every session function and alias in alphabetical order.
