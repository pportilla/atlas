# Command line

Run `./atlas` from the repository directory. Without arguments it opens the Atlas prompt. With a command it runs that command and exits. `./atlas --help` lists the commands and `./atlas COMMAND --help` describes one of them.

| Command | Purpose |
| --- | --- |
| `./atlas` | Open the interactive prompt. |
| `./atlas run FILE` | Run an Atlas script. |
| `./atlas resolve POLYNOMIAL --vars x,y,…` | Resolve a hypersurface germ at the origin, optionally with its zeta functions. |
| `./atlas monodromy RECORD` | Compute the monodromy zeta function from a saved resolution. |
| `./atlas recompute-dual-complex RECORD` | Rebuild the exceptional dual complex from a saved resolution. Also available as `dual-complex`. |
| `./atlas algebra OPERATION …` | Compute with polynomials and ideals over $\mathbb Q$. |
| `./atlas serve` | Start the web workspace and the HTTP API. |
| `./atlas devices` | List the GPU devices that Atlas can use. |

## Open the prompt

`./atlas` opens a session in which you enter Atlas statements. `help NAME` describes a function, and `help Z` describes the value saved as `Z`. `:workspace` lists the saved values grouped by ring, `:save "file.atlas"` writes the successful statements to a script, and `quit` ends the session. Tab completes names and fields. [The prompt and scripts](../tutorial/prompt-and-scripts.md)

The prompt also accepts the `resolve` and `algebra` commands of this page without `./atlas`, for example `resolve "y^2-x^3" --vars x,y`.

## Run a script

`./atlas run FILE` runs the statements of a UTF-8 file in a fresh session, and `./atlas run -` reads them from standard input. A script prints only what it passes to `print` or `print_json`. If a statement fails, Atlas reports the file and line on standard error and exits with status 2. [Run or save a calculation](../tutorial/prompt-and-scripts.md#run-or-save-a-calculation)

## Resolve a germ

`./atlas resolve` takes the equation as a quoted string and its coordinates in order. For the cusp, run the following command.

```sh
./atlas resolve 'y^2-x^3' --vars x,y
```

The output summarizes the resolution and lists each chart with its map, strict transform and exceptional divisors. The options correspond to those of the session function `resolve`.

| Option | Effect |
| --- | --- |
| `--vars x,y,…` | Required. The coordinates in order, separated by commas. |
| `--mode embedded`, `--mode intrinsic` | Resolve the hypersurface in its ambient space (default), or the hypersurface itself. [Resolution modes](../thematic/resolution-records.md#resolution-modes) |
| `--method auto`, `--method weighted` | Let Atlas choose the construction (default), or use weighted blowups in embedded mode. [Weighted resolution](../thematic/weighted-resolution.md) |
| `--method jung` | Resolve a surface in three variables by Jung's method, after checking that it applies; the mode defaults to intrinsic. [Jung's method](../thematic/jung-method.md#resolve-with-method-jung) |
| `--exhaustive`, `--quick` | Search until the resolution is certified (default), or give the same search a time budget of 20 seconds. [Quick search](../thematic/resolution-pipeline.md#quick-search) |
| `--seconds N` | The time budget of a quick search in seconds. It implies `--quick` and does not go with `--exhaustive`. |
| `--monodromy` | Also compute the topological and monodromy zeta functions. |
| `--zetas-only` | Compute both zeta functions and the pole check. Atlas may then use a certificate without charts, as for germs made of polynomial graphs. [Zeta functions](../thematic/zeta-functions.md#germs-made-of-polynomial-graphs) |
| `--json` | Print the complete record as JSON instead of the summary. |
| `--compute cpu`, `--compute auto`, `--compute gpu`, `--gpu` | Choose the device. [Select a device](http-and-compute.md#select-a-device) |

Save the record of a resolution with its zeta functions for later use.

```sh
./atlas resolve 'y^2-x^3' --vars x,y --monodromy --json > cusp.json
```

The exit status is 0 when the resolution is certified and every requested zeta function was computed, and 2 for an invalid input or an error. It is 3 when a quick search is not finished: its time budget ended. The command then prints one line, such as `Not resolved within 20 seconds. Stopped while blowing up centre B7 (order method), after 6 blowups. Run without --quick to continue the search.` With `--json` it prints the record of the expired budget, with `status` `"inconclusive"` and the field `budget`. Nothing else ends a quick search, except an input beyond the limits that follow, which also exits with 3 before any search. It reads at most eight variables, 8,192 characters, degree 128 and 4,096 expanded terms, with integers of up to 256 decimal digits and coefficients of up to 4,096 bits during expansion. An exhaustive search has no such bounds.

## Read a saved record

These commands compute from a record saved with `--json` without resolving again. Give `-` to read the record from standard input.

```sh
./atlas monodromy cusp.json
./atlas monodromy --json cusp.json
./atlas recompute-dual-complex cusp.json
```

`./atlas monodromy` replays the certificate of the record and prints the monodromy zeta function and the Euler characteristic of the Milnor fibre. With `--json` it prints the complete zeta function record, including A’Campo's contribution of each divisor. `./atlas recompute-dual-complex` prints the dual complex rebuilt from the charts as JSON, and also reads compressed `.json.gz` records. Both exit with status 0 on success and 3 when the result is incomplete. [Resolution records](../thematic/resolution-records.md#save-and-replay-the-cusp)

## Compute with polynomials and ideals

`./atlas algebra OPERATION` runs one algebra operation. The operations are those of the session functions, written with hyphens, such as `normal-form`, `groebner`, `primary-decomposition` or `milnor-number`. `./atlas algebra --help` lists them, and `./atlas algebra normal-form --help` gives the options of one operation.

```sh
./atlas algebra normal-form --vars x,y --gen 'x^2-y' --poly 'x^3'
```

The ring is given by `--vars` in order, with `--order lex` or `--order grevlex` and, for a quotient, one `--relation` per defining relation. Ideal generators are repeated `--gen` options and the polynomial operand is `--poly`. With `--json`, the result includes `replay.command` and `replay.script`, which repeat the calculation from the shell and from an Atlas script. [Algebra requests and replay](algebra-requests.md)

## Start the web workspace

`./atlas serve` starts the web workspace and the HTTP API at `http://127.0.0.1:8765`. Choose another port with `--port`. The workspace resolves germs in the browser. [HTTP API and compute options](http-and-compute.md)

## List devices

`./atlas devices` lists the Triton and OpenCL GPU devices that passed an exact integer probe on this machine. [Select a device](http-and-compute.md#select-a-device)
