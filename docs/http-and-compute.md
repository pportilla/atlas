# HTTP API and compute options

Start the local server with `./atlas serve`. Open `http://127.0.0.1:8765` for the resolution workspace or send JSON to the same address. Use `--port PORT` to select another port.

## Request a resolution

This request resolves the cusp in embedded mode and asks for both zetas.

```sh
curl -sS http://127.0.0.1:8765/api/resolve -H 'Content-Type: application/json' \
  -d '{"polynomial":"y^2-x^3","variables":["x","y"],"mode":"embedded","search":"quick","zetas":["monodromy","topological"]}'
```

Read the returned `status` first. For this cusp, the resolution is `resolved`. Its `monodromy_zeta.expression` is `(1-T^6)/((1-T^2)*(1-T^3))`. The `monodromy_check.poles` entries pair −1 with eigenvalue order 1 and −5/6 with order 6. [Cusp response checks](../tests/test_http_search.py#L101) · [Zeta and chart fields](resolution-records.md)

`POST /api/resolve` with `"search":"quick"` returns the bounded result directly. Omit `search` to try quick resolution and continue in the background if the attempt is inconclusive, hits a computation limit or misses a requested zeta. A direct result is HTTP 200. A background job is HTTP 202 and includes an `id`. Setting `"search":"exhaustive"` starts a background job immediately.

## Follow a background search

`POST /api/searches` accepts the same resolution input. It always returns a job on successful submission. The default search policy there is `"exhaustive"`. Set `"search":"auto"` to try quick first or `"search":"quick"` to keep the attempt bounded. Two jobs can run at once. Additional jobs start in submission order. The server retains finished jobs for 30 minutes.

| Request | Response |
| --- | --- |
| `GET /api/searches/{id}` | The job's `state`, any `progress` snapshot and its `result` when completed. Progress can include a solver stage and blowup count. |
| `GET /api/searches` | A `jobs` array, newest first. |
| `GET /api/searches/current` | A `job` field with the most recently submitted job or `null`. |
| `DELETE /api/searches/{id}` | The updated state of the job after a cancellation request. |

A job's `state` describes the worker. After `state` becomes `completed`, inspect `result.status` for the mathematical outcome. [Queue, reattachment and cancellation examples](../tests/test_http_search.py) · [Job implementation](../engine/search_jobs.py)

## Replay zeta functions

Select the names `"monodromy"` and `"topological"` in the `zetas` array. On a new resolution, `"zetas":[]` requests geometry alone. For a completed job, send one or both names to `POST /api/searches/{id}/zetas`. To replay an exported resolution after the job expires, send a JSON object with the saved `record` and the requested `zetas` to `POST /api/zetas`.

Replay uses the saved charts, including an available embedded companion for an intrinsic resolution. When the charts cannot certify the required strata, the zeta's `status` is `unavailable` and its `reason` explains why. If the other zeta was already computed, Atlas recomputes the pair on the same strata. When both are computed, `monodromy_check` lists the poles and witnesses. [Replay logic](../engine/resolution_zetas.py) · [Native monodromy](../native/monodromy.cpp) · [Native topological zeta](../native/zeta.cpp)

## Submit an algebra request

`POST /api/algebra` accepts an operation, its ordered ring and the operands. The ordinary algebra and quick resolution endpoints accept up to 32,768 bytes and use a 55-second worker deadline. For larger requests, use `POST /api/algebra/exhaustive` for algebra or `POST /api/searches` for resolution. The exhaustive algebra endpoint keeps the request open until the worker returns. `GET /api/health` returns `{"status":"ok"}`. [Normal forms and reusing a basis over HTTP](../examples/algebra-http.md) · [Wide request tests](../tests/test_http_exhaustive.py) · [HTTP handlers](../engine/app.py)

## Select a device

Resolution defaults to CPU execution. Rational ideal arithmetic defaults to
automatic selection. List the usable backends with `./atlas devices`. To select
a policy, pass `--compute cpu`, `--compute auto`, or `--compute gpu` at the CLI,
or use the `compute` field in an HTTP request.

```sh
./atlas devices
./atlas resolve 'x*y-z^5' --vars x,y,z --mode intrinsic --compute cpu --json
```

To enable OpenCL, install its headers, loader, and device driver and run
`make gpu`. Triton uses a compatible GPU-enabled PyTorch and Triton installation.
To select a specific backend and device, set `RESOLUTION_GPU_BACKEND` to
`triton` or `opencl` and set `RESOLUTION_GPU_INDEX` to the device ordinal.
See `execution.stages` for the backend used. Resolution records include
fallback details on the chart or rank stage. Algebra results collect the
reasons in `execution.fallback_reasons`.
[Device calibration and full solves](../benchmarks/groebner-timings.md#profiling-katsura-6--7-and--8) · [Tests for CPU fallback and device agreement](../tests/test_gpu_resolution.py)
