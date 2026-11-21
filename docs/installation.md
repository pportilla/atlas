# Installation guide

Choose [Build Atlas](#build-atlas) to run the examples on Linux, or [Build the manual](#build-the-manual) for a browsable copy of the documentation.

## Build Atlas

You will need Python 3.10 or later, a C++17 compiler, GNU Make, pkg-config, and the development libraries for GMP (including gmpxx), FLINT, JsonCpp and Normaliz. The CPU runtime uses the Python standard library. On Debian or Ubuntu, install the prerequisites and check the version of FLINT.

```sh
sudo apt update
sudo apt install git build-essential pkg-config python3 libgmp-dev libmpfr-dev \
    libflint-dev libjsoncpp-dev libnormaliz-dev
pkg-config --modversion flint
```

Atlas requires FLINT 3.6 or later. If your distribution supplies an older version, install FLINT 3.6.0, the version tested with Atlas, and check that pkg-config finds it. [FLINT releases](https://flintlib.org/downloads.html) · [FLINT build instructions](https://flintlib.org/doc/guide/getting-started/building.html)

Clone the repository and compile Atlas.

```sh
git clone https://github.com/pportilla/atlas.git
cd atlas
make -j4
```

In this directory, `./atlas` opens the mathematical prompt and `./atlas --help` lists the command-line interfaces. Take the [tour](tour.md) or [run an Atlas script](tutorial/prompt-and-scripts.md#run-or-save-a-calculation).

## Build the manual

The manual needs Python 3.10 or later and GNU Make. From the top of a source checkout or an unpacked archive, install the documentation dependencies in a virtual environment. On Debian and Ubuntu, the `python3-venv` package supplies the environment creator.

```sh
python3 -m venv build/docs-env
. build/docs-env/bin/activate
python -m pip install -r requirements-docs.txt
make docs
```

Open `build/docs/index.html` in a browser. The pages and search index are static. For typeset formulas and diagrams, the browser loads KaTeX and Mermaid from jsDelivr. To publish a copy, serve the contents of `build/docs/` with a static web server. To choose another output directory, use `make docs DOCS_OUT=path/to/site`.

## Check a build

Once Atlas and the documentation dependencies are installed, run `make check-docs` to check internal links, the function reference and the results printed in the worked examples. To run the full test suite, use the same virtual environment.

```sh
python -m pip install -r requirements-test.txt
make test
```

The comparison tests use Singular when it is installed. The [benchmark reports](../benchmarks/README.md) describe how to reproduce individual comparisons.
