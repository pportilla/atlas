# Algebra examples

## Exact rational linear algebra

`matrix_analysis` treats the rows as a matrix with entries in `QQ`. The pivot indices
are zero-based, `.image` lists the corresponding original columns, and `.kernel` lists
a basis of column vectors annihilated by the matrix. The reduced row echelon matrix
has the same shape as the input. A rectangular matrix has determinant `None`. The
empty square matrix has determinant one. Use `columns=n` to specify a matrix with
zero rows and `n` columns.

```atlas
M = matrix_analysis([[1, 2], [2, 4]], rhs=[3, 6])
print(M.rank, M.determinant, M.pivot_columns)  # 1 0 (0,)
print(M.solution.status)                       # underdetermined
print(M.solution.particular[0])                # 3
print(M.kernel[0][0])                          # -2
```

With `rhs`, the solution status is `unique`, `underdetermined`, or `inconsistent`.
When consistent, all solutions are the particular vector plus linear combinations
of the kernel vectors. For an inconsistent system the particular vector is `None`.
Exact coefficients may be integers, rational strings, or session rational values.
The scalar field is independent of the active polynomial ring.

`compute="gpu"` dispatches eligible row selection over a prime field to an available
accelerator. Atlas certifies that the selected rows span the matrix over `QQ`, including
the right-hand side. This recovers rank lost at an unlucky prime. For square matrices
through order six, Atlas removes the rational row contents and checks an integer
bound before evaluating the determinant directly on the GPU. Independent prime images
certify this result over the integers. Larger determinants and exact vector reconstruction
use the CPU. The `.execution`
record lists the stages and device. The command line accepts repeated `--row` arguments.
[CPU, Triton and Singular timings](../benchmarks/matrix-triton.md)

```sh
./atlas algebra matrix-analysis --row 1,2 --row 2,4 --rhs 3,6 --compute gpu
```

## Run a first example

<a id="shell-prompt-and-scripts"></a>

What can we deduce from the equations $x^2=1$ and $y=x$? For instance, $y^2=1$ follows immediately. Atlas can check this by reducing $y^2-1$ modulo the ideal of the two equations.

```text
atlas> R = ring("x", "y")
atlas> I = ideal(x^2-1, y-x, order="lex")
atlas> B = groebner(I)
atlas> B.generators
(x-y, y^2-1)
atlas> membership(y^2-1, B).value
True
```

The call `ring("x", "y")` gives us the ring $\mathbb Q[x,y]$ and names its two generators. The Gröbner basis describes the same ideal as `I`. We can now use `B` to simplify other polynomials or test more equations. The complete calculation is also available as a script. [Reduction modulo an ideal](polynomial-ideals.md#reduce-a-polynomial-modulo-an-ideal) · [Example script](groebner.atlas)

```sh
./atlas run examples/groebner.atlas
```

## Choose a worked example

Each section begins with the declarations it needs. To try a different section, start a fresh session. You can paste its Atlas commands at the prompt or save them together in a script.

<a id="polynomial-factorization"></a><a id="polynomial-gcd"></a><a id="polynomial-resultants"></a><a id="radicals-of-polynomial-ideals"></a><a id="primary-decomposition-over-qq"></a><a id="ordinary-and-symbolic-powers-over-qq"></a>

Eliminate the parameter from $(t^2,t^3)$, find a common factor, or recover the primary components of an ideal. [Polynomials and ideals](polynomial-ideals.md)

<a id="arithmetic-in-declared-quotient-rings-over-qq"></a><a id="localization-at-a-prime-over-qq"></a><a id="normalization-of-affine-algebras-over-qq"></a><a id="hilbert-series-of-homogeneous-quotients"></a>

Impose $y^2=0$ and see why two different ideals acquire the same image. Then try inverting a residue class, localizing at a prime or normalizing a cusp. [Quotient rings, localization and normalization](quotient-algebras.md)

<a id="quotient-bases-and-local-hypersurface-invariants"></a><a id="multiplicity-of-finite-colength-quotients"></a><a id="extended-right-left-codimension-of-a-map-germ"></a><a id="one-curve-at-three-rational-points"></a><a id="local-standard-bases-and-normal-forms-at-a-rational-point"></a>

Count the contribution of a single point to a quotient algebra, or compute the Milnor number at several points of the same curve. [Local algebras and map germs](local-algebra.md)

<a id="local-dimension-and-smoothness"></a><a id="zariski-tangent-spaces-and-embedding-dimensions"></a><a id="tangent-cones-and-their-hilbert-series"></a><a id="hilbertsamuel-functions-and-multiplicities"></a>

Identify the singular point where a plane meets a line, then compare the tangent space and tangent cone of a cusp. [Geometry at a rational point](local-geometry.md)

## Use the shell or HTTP API

<a id="example-request"></a><a id="example-success"></a><a id="example-error"></a><a id="operands-and-results"></a><a id="reuse-a-basis-over-http"></a>

From the shell, use `./atlas algebra --help` to list the available calculations, or `./atlas algebra normal-form --help` for the arguments of a single command. If you are calling Atlas from another program, the HTTP guide carries the first example above through a request, a response and a second reduction using the saved basis. [Algebra requests and replay](algebra-http.md) · [Server and compute options](../docs/http-and-compute.md)
