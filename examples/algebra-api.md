# Algebra examples

Each guide starts from a concrete ideal, polynomial or map over the rationals. The examples give the expected values, and the accompanying links open the Atlas tests and C++ calculations behind them. Use a fresh Atlas session for a standalone block, or save it in an `.atlas` file and run it with `./atlas run`.

## Run a first example

<a id="shell-prompt-and-scripts"></a>

```sh
./atlas run examples/groebner.atlas
```

The script computes a Gröbner basis of $(x^2-1,y-x)$ in lex order, reduces $x^2+y/2$ to $1+y/2$, and tests two polynomials for membership. The last two outputs are `True` for $y^2-1$ and `False` for $y$. [Read the script](groebner.atlas) · [Follow the calculation](polynomial-ideals.md#reduce-a-polynomial-modulo-an-ideal) · [Prompt and scripts](../docs/sessions.md)

The matching regression case checks that repeated reductions use the saved basis and that the result can be used in another ideal. To run just this case from a built checkout, use the command below. [Read the test](../tests/test_session_algebra.py#L62)

```sh
python3 -m unittest tests.test_session_algebra.AlgebraSessionTests.test_reuse_basis_and_polynomial_in_original_coordinates
```

## Choose a worked example

| Guide | Examples |
| --- | --- |
| [Polynomials and ideals](polynomial-ideals.md) | <a id="polynomial-factorization"></a>[Reconstruct a factorization](polynomial-ideals.md#factor-and-reconstruct-a-polynomial), <a id="polynomial-gcd"></a>[find a GCD](polynomial-ideals.md#find-a-common-factor), or <a id="polynomial-resultants"></a>[eliminate a coordinate](polynomial-ideals.md#eliminate-a-coordinate). |
| [Ideal structure](polynomial-ideals.md#intersect-divide-and-saturate-ideals) | <a id="radicals-of-polynomial-ideals"></a>[Recover the reduced support](polynomial-ideals.md#recover-the-reduced-support), <a id="primary-decomposition-over-qq"></a>[reconstruct a primary decomposition](polynomial-ideals.md#reconstruct-an-ideal-from-its-primary-components), or <a id="ordinary-and-symbolic-powers-over-qq"></a>[compare ordinary and symbolic powers](polynomial-ideals.md#compare-ordinary-and-symbolic-powers). |
| [Quotient rings and affine algebras](quotient-algebras.md) | <a id="arithmetic-in-declared-quotient-rings-over-qq"></a>[Compute with classes and ring maps](quotient-algebras.md#compute-with-residue-classes), <a id="localization-at-a-prime-over-qq"></a>[localize at a prime](quotient-algebras.md#localize-at-a-prime), <a id="normalization-of-affine-algebras-over-qq"></a>[normalize a cusp](quotient-algebras.md#normalize-the-cusp), or <a id="hilbert-series-of-homogeneous-quotients"></a>[read a graded Hilbert series](quotient-algebras.md#read-the-graded-hilbert-series). |
| [Local algebras and map germs](local-algebra.md) | <a id="quotient-bases-and-local-hypersurface-invariants"></a><a id="multiplicity-of-finite-colength-quotients"></a>[Count a local length](local-algebra.md#count-global-and-local-lengths), <a id="extended-right-left-codimension-of-a-map-germ"></a>[compare a cusp and its parametrization](local-algebra.md#compare-the-cusp-and-its-parametrization), <a id="one-curve-at-three-rational-points"></a>[evaluate one curve at three points](local-algebra.md#evaluate-one-curve-at-three-points), or <a id="local-standard-bases-and-normal-forms-at-a-rational-point"></a>[reduce at a rational point](local-algebra.md#reduce-at-a-rational-point). |
| [Geometry at a rational point](local-geometry.md) | <a id="local-dimension-and-smoothness"></a>[Check smoothness](local-geometry.md#compare-local-dimension-and-smoothness), <a id="zariski-tangent-spaces-and-embedding-dimensions"></a>[find a tangent space](local-geometry.md#compute-the-tangent-space), <a id="tangent-cones-and-their-hilbert-series"></a>[compute a tangent cone](local-geometry.md#compute-the-tangent-cone), or <a id="hilbert-samuel-functions"></a><a id="hilbertsamuel-functions-and-multiplicities"></a>[sample the Hilbert–Samuel function](local-geometry.md#sample-the-hilbert-samuel-function). |

## Use the shell or HTTP API

<a id="example-request"></a><a id="example-success"></a><a id="example-error"></a><a id="operands-and-results"></a><a id="reuse-a-basis-over-http"></a>

Run `./atlas algebra --help` to list shell operations, or `./atlas algebra normal-form --help` for a single command. The HTTP guide gives a complete request, its expected result, the operation and operand tables, and how to reuse a basis or replay a record. [Algebra requests and replay](algebra-http.md) · [Server and compute options](../docs/http-and-compute.md)
