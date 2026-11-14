# Atlas

Atlas is a computer algebra system for exact computation in commutative algebra
and algebraic geometry, with a focus on resolution of singularities. It works
with multivariate polynomials, ideals, and quotient rings over the rationals.
The C++ kernel performs the mathematics. Python provides the interactive
console, scripting environment, and HTTP interface.

## Capabilities

- **Polynomial and ideal arithmetic.** Factor polynomials, compute GCDs and
  resultants, and use Gröbner bases for normal forms, ideal membership, and
  elimination. Atlas also computes radicals, primary decompositions, and
  ordinary and symbolic powers of ideals.
- **Local algebra and geometry.** Compute in quotient rings and localizations,
  normalize affine algebras, and study singular points through local dimension,
  tangent cones, multiplicities, and Milnor and Tjurina numbers. For maps
  between smooth germs, compute the extended right-left codimension.
- **Resolution of singularities.** Compute embedded and intrinsic resolutions
  of hypersurface germs at the origin. Inspect the chart maps and strict
  transforms, recover the labelled dual complex of the exceptional divisor,
  and compute the local monodromy and topological zeta functions.

## Installation

Build Atlas from source on Linux with Python 3.10 or later, a C++17 compiler,
GNU Make, pkg-config, and the development libraries for GMP (including gmpxx),
FLINT, JsonCpp, and Normaliz. The CPU runtime uses the Python standard library.

On Debian or Ubuntu, install the build dependencies and check the FLINT version.

```sh
sudo apt update
sudo apt install git build-essential pkg-config python3 libgmp-dev libmpfr-dev \
    libflint-dev libjsoncpp-dev libnormaliz-dev
pkg-config --modversion flint
```

Atlas requires FLINT 3.6 or later. If your distribution ships an older release,
install FLINT 3.6.0, the version tested here, and check the version again.
[FLINT releases](https://flintlib.org/downloads.html) · [FLINT installation guide](https://flintlib.org/doc/guide/getting-started/building.html)

```sh
git clone https://github.com/pportilla/atlas.git
cd atlas
make -j4
./atlas --help
```

## Using Atlas

Run `./atlas` from the checkout to open the console. `./atlas run FILE`
executes an Atlas script, and `./atlas algebra --help` lists the algebra
commands. For the Resolution web interface and HTTP API, start `./atlas serve`
and open `http://127.0.0.1:8765`. Submit a polynomial, then expand its row to
follow the search. You can submit another polynomial while the first runs.
Finished rows offer the recorded complex, charts, and follow-up zeta calculations.

## Documentation

- [Algebra operations](examples/algebra-api.md) and [console and scripting guide](docs/sessions.md)
- [Resolution algorithms](docs/resolution-pipeline.md) and [resolution records](docs/resolution-records.md)
- [HTTP API and optional GPU support](docs/http-and-compute.md)
- [Kernel algorithms and proofs](native/KERNEL.md) and [benchmarks](benchmarks/README.md)
- [Lean verification roadmap](docs/lean-verification-roadmap.md)
