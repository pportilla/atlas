# Developer guide

This guide is for people who change Atlas: its C++ kernel, the Python session or this manual. It describes where each part lives, the rules every new calculation follows and how to check a change. The chapters that follow record how the current code was measured and where it is going.

## Where things live

| Directory | Contents |
| --- | --- |
| `native/` | The C++ kernel, which performs every mathematical computation, with the algorithm notes beside the code |
| `engine/` | The Python session, prompt, command-line interface, HTTP server and workers |
| `docs/` | This manual, with its builder in `docs/site/` |
| `examples/` | Atlas scripts and saved records that the manual refers to |
| `tests/` | Unit and interface tests, including the worked examples of this manual |
| `benchmarks/` | Timing scripts, inputs and reports |
| `web/` | The browser workspace |

## Add a mathematical operation

Implement the computation in C++ under `native/`, including its arithmetic, geometric predicates, shortcuts, certificate checks and fallbacks. Every entry point calls the same native operation: the prompt, scripts, the command line, HTTP, workers and the replay of a saved record. Python validates input, transports values and presents results. Independent reference calculations in Python belong in the tests. Run `make check-native-boundary` whenever you change production Python.

Expose a calculation that users call as a session function returning a structured value, and register it in `engine/session.py`. The same call, fields and arguments then work at the prompt and in scripts. Polynomials carry their rings, and the session checks ownership through live parent objects. The [chapter on rings](../tutorial/rings.md) shows the conventions users see.

When an established system can perform the same operation, add repeatable timing comparisons on the same inputs. Check that the answers agree, run several trials and record the machine and conditions in a [benchmark report](../../benchmarks/README.md).

## Document a function

A public session function needs a docstring with three parts. The first line says what the function returns. A short example with `>>>` lines follows, including any setup it needs in a fresh session. The last line names the guide section that explains the function, such as `Docs: docs/tutorial/rings.md#name-select-and-delete-rings`. The prompt's `help NAME` card shows these parts with the real signature.

The [reference manual](../reference/index.md) has one page for each subject. The builder completes these pages with the cards of their functions, much as Sphinx's autodoc completes Sage's reference manual from docstrings. To add a function to the manual, list it under its subject page in `docs/site/catalog.py` with links to its implementation and a test. Describe its inputs and return value in `docs/site/function_reference.py`. The build fails if either file omits a session function.

Each kind of material has its own place in the manual.

| Material | Place |
| --- | --- |
| A worked calculation that a newcomer reads in order | A chapter of the [tutorial](../tutorial/index.md) |
| One subject treated in depth | A [thematic tutorial](../thematic/index.md) |
| A short answer to “How do I…?” | [Constructions](../constructions.md) |
| Inputs, return values and conventions | The [reference manual](../reference/index.md) |
| Every field of a value that a function returns | The subject's page of the reference manual, such as [Resolution objects and zeta functions](../reference/resolution-values.md), with an example that starts from an empty session |
| How a result is computed and certified | The algorithm notes in `native/` |
| Timings and their conditions | A [benchmark report](../../benchmarks/README.md) |

Every page belongs to one document in the navigation that `docs/site/build.py` defines, and the landing page of that document links to it.

## Build and check

`make docs` builds the manual in `build/docs/`. `make check-docs` checks its links and anchors, the function reference and the output of every worked example. `make test` runs the full test suite. [Installation guide](../installation.md#check-a-build)

## Chapters

- [Benchmarks](../../benchmarks/README.md). Timings for algebra, resolution and devices, with performance logs for individual changes.
- [Development roadmap](development-roadmap.md). Proposed calculations and the examples that should validate them.
- [Lean verification roadmap](lean-verification-roadmap.md). A plan to prove statements about computations that Atlas performs.
- [Visual identity](identity.md). The faces, colours and terminal roles shared by the prompt, the web workspace and this manual.
