# Resolution Atlas

Atlas resolves hypersurface germs at the origin from a polynomial with rational
coefficients. It returns affine charts, their maps to the original coordinates,
strict transforms, and a labelled dual complex that distinguishes geometric
components over the algebraic closure. It can also compute the local monodromy
and topological zeta functions.

## Build and run

Install Python 3.10 or later, a C++17 compiler, Make, pkg-config, and the
development libraries for GMP (including gmpxx), FLINT (tested with 3.6.0),
JsonCpp, and libnormaliz.
Build from the repository root and resolve the plane cusp in a terminal.

```sh
make -j4
./atlas resolve 'y^2-x^3' --vars x,y --mode embedded --monodromy
```

The cusp takes three point blowups. Its exceptional curves have multiplicities
2, 3, and 6. The last curve meets each of the first two. Embedded mode resolves
the total divisor, including multiplicities. Use `--mode intrinsic` for a
reduced germ to compute the exceptional divisor on its smooth strict transform.

Save the full chart record and both zetas as JSON, then reconstruct the complex
from the saved charts.

```sh
./atlas resolve 'y^2-x^3' --vars x,y --monodromy --json > cusp.json
./atlas dual-complex cusp.json
```

## Work with a result

Start `./atlas` for the interactive prompt. Type `help functions` for the
operation list or `help factor` for a worked example. To see how to inspect
one divisor across charts, run `./atlas run examples/cusp.atlas`.

| Task | Guide |
| --- | --- |
| Inspect a chart or save a session | [Prompt and scripts](docs/sessions.md) |
| Read a resolution, its cell attachments, or its zetas | [Resolution records](docs/resolution-records.md) |
| Understand how Atlas chooses and verifies a blowup | [Resolution pipeline](docs/resolution-pipeline.md) |
| Compute with polynomials, ideals, and quotient rings | [Algebra examples](examples/algebra-api.md) |
| Run a local server or select a compute device | [HTTP and compute options](docs/http-and-compute.md) |
| Compare algorithms and timings | [Benchmarks](benchmarks/README.md) · [Atlas and Singular](README-singular-comparison.md) |

## Development

Exact arithmetic and geometric verification live in the C++ kernel. Python
coordinates the search, moves records between workers, and serves the CLI and
HTTP interfaces. [Kernel algorithms and proofs](native/KERNEL.md) ·
[Development conventions](AGENTS.md)

For the test suite, install the optional reference dependencies with
`python3 -m pip install -r requirements-test.txt`, then run `make test`.
Run `make check-native-boundary` to check the separation between the kernel
and the Python interfaces.
