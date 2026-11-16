# Atlas

Atlas computes with rational polynomials, ideals, quotient rings and hypersurface germs. You can use the same mathematical functions at the prompt and in an Atlas script. The C++ kernel computes the results.

## Start here

After [building Atlas](#installation), run `./atlas` from the checkout. For example, the Milnor number of the plane cusp $y^2-x^3$ is the dimension of its local Jacobian algebra. The partial derivatives generate $(x^2,y)$, so the classes of $1$ and $x$ give a basis.

```text
atlas> x, y = coordinates("x", "y")
atlas> f = y^2 - x^3
atlas> milnor_number(f).value
2
```

Use `help milnor_number` for its arguments or `help functions` to see the available operations. [Prompt and scripts](docs/sessions.md) · [Local algebra test cases](tests/test_local_algebra_session.py)

## Find a calculation

| What you want to do | Where to start |
| --- | --- |
| Compute with polynomials, ideals or rings, or study a germ at a point | [Algebra guides and worked examples](examples/algebra-api.md) |
| Resolve a curve and inspect its exceptional divisors and chart equations | [Resolve a plane cusp](docs/resolving-a-cusp.md) |
| Read a saved resolution or compute its dual complex and zeta functions | [Reading a resolution record](docs/resolution-records.md) |
| Submit a calculation through the web interface or an HTTP client | [HTTP requests and compute options](docs/http-and-compute.md) |
| Follow a resolution from search to verification | [Resolution algorithms](docs/resolution-pipeline.md) |

## Browse the documentation and code

From the checkout, build the site with the Python environment you use for Atlas.

```sh
python3 -m pip install markdown-it-py
make docs
```

Open `build/docs/index.html` in a browser. In **Functions**, choose a topic and open **Implementation and tests** under a function to jump to its code or a test case in this checkout. Use **Related documentation** on either file to return to the guide. The alphabetical index includes alternate function names.

- Follow a mathematical argument into the C++ implementation. [Kernel algorithms and proofs](native/KERNEL.md)
- Compare timings for the same inputs and see the conditions of each run. [Benchmark reports](benchmarks/README.md)
- Read about proposed operations and formal verification. [Development roadmap](docs/development-roadmap.md) · [Lean verification roadmap](docs/lean-verification-roadmap.md)

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

## Command-line and web interfaces

Use `./atlas run FILE` to execute an Atlas script and `./atlas algebra --help` to list algebra commands. For the resolution workspace, start `./atlas serve` and open `http://127.0.0.1:8765`. Expand a submitted polynomial to follow its search. A finished row gives access to the charts, complex and follow-up zeta calculations.
