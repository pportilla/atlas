# Atlas

Atlas computes exactly with polynomial rings over the rationals, studies a variety at a rational point, and resolves hypersurface singularities. You can use it at an interactive prompt, in scripts, from the command line and over HTTP. Its documentation is organized as a set of documents, each written for a different purpose.

Run `./atlas serve` and open the **Resolution Zoo** in the resolution workspace to browse computed examples. Search by monomials, compare intrinsic and embedded resolutions, and inspect the strata contributing to each zeta function. [Catalogue format and regeneration](docs/thematic/resolution-records.md#resolution-zoo-catalogue)

## Tutorials

- **[A tour of Atlas](docs/tour.md).** A one-page introduction: reduce a polynomial modulo an ideal, compute an invariant of a cusp and resolve it.
- **[Tutorial](docs/tutorial/index.md).** The best way to become familiar with Atlas in a few hours. Each chapter works through mathematical questions at the prompt, from rings and ideals to local algebra and a first resolution.
- **[Constructions](docs/constructions.md).** Short answers to questions of the form “How do I compute … in Atlas?”, each a complete script with its output.
- **[Thematic tutorials](docs/thematic/index.md).** In-depth treatments of one subject, such as the zeta functions of a germ and the monodromy conjecture, saved resolution records, and how Atlas chooses and certifies its blowups.

## Reference manual

- **[Reference manual](docs/reference/index.md).** Every session function by subject, with its inputs, return value and a tested example, and links to its implementation, its tests and the guides that use it. It also lists every field of a resolution and its zeta functions, the methods of other saved values, the command line, the HTTP interface and the algorithms behind the results.
- **[Function index](docs/reference/genindex.md).** Every session function and alias in alphabetical order.

## Installation and development

- **[Installation guide](docs/installation.md).** Build Atlas on Linux, start the prompt and build a local copy of this manual.
- **[Developer guide](docs/developer/index.md).** Where the code lives, the rules for adding a calculation and documenting a function, and the benchmarks and roadmaps.
