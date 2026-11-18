# Atlas

Atlas is a system for computing with polynomial rings over the rationals and for resolving hypersurface singularities. The examples in this manual use the Atlas prompt. You can also save a calculation as a script and run it again.

## Start here

In $\mathbb Q[x,y]$, suppose we impose the relation $x^2=y$. What does $x^3$ become? Name the ring, put the relation in an ideal, and ask for the normal form.

```text
atlas> R = ring("x", "y")
atlas> I = ideal(x^2 - y)
atlas> normal_form(x^3, I)
x*y
```

The answer says that $x^3$ and $xy$ represent the same class modulo $I$. A second way to check this is to ask whether their difference belongs to the ideal.

```text
atlas> membership(x^3 - x*y, I).value
True
```

[Continue with rings and ideals](examples/algebra-api.md) or [open your first Atlas session](docs/sessions.md).

## Find a calculation

| If you want to… | Read… |
| --- | --- |
| Factor a polynomial, eliminate a variable, or test ideal membership | [Polynomials and ideals](examples/polynomial-ideals.md) |
| Construct a quotient and compare two ideals whose images coincide | [Ideals before and after quotienting](examples/quotient-algebras.md#compare-ideals-before-and-after-quotienting) |
| Compute the Milnor number of a plane curve at a specified point | [Local algebra](examples/local-algebra.md) |
| Compare the tangent space and tangent cone at a singular point | [Local geometry](examples/local-geometry.md) |
| Resolve a cusp and follow its equation through a blowup | [Resolving a plane cusp](docs/resolving-a-cusp.md) |
| Recover exceptional divisors, incidences and zeta functions from a resolution | [Reading a resolution record](docs/resolution-records.md) |

## Browse the documentation and code

At the prompt, type `help normal_form` to see the arguments and an example, or `help I` to inspect the ideal you just made. In the built manual, **Functions** lists the operations by subject and **Index** looks them up by name. Methods such as `K.element(f)` have their own [reference for saved values](docs/value-methods.md).

For the mathematics behind a result, the [algorithm guide](native/KERNEL.md) describes the computations and points to the source and the literature. For resolution, start with [how Atlas chooses and checks a resolution](docs/resolution-pipeline.md). The [benchmarks](benchmarks/README.md) give inputs and timings. The [development notes](docs/development-roadmap.md) and [formal verification plan](docs/lean-verification-roadmap.md) describe work in progress.

## Installation

The [installation guide](docs/installation.md) covers building Atlas on Linux, starting the prompt and building a local copy of this manual. You can read and build the manual independently of the mathematical software.

## Command-line and web interfaces

To run an Atlas file, use `./atlas run FILE`. For the browser workspace or a program that submits calculations over HTTP, see [HTTP requests and compute options](docs/http-and-compute.md).
