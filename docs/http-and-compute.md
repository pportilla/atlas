# HTTP and compute options

Run `./atlas serve` to start the local server on port 8765, or supply
`--port PORT`. A terminal client can use the same server on a headless
machine.

```sh
curl -s http://127.0.0.1:8765/api/resolve -H 'Content-Type: application/json' \
  -d '{"polynomial":"y^2-x^3","variables":["x","y"],"mode":"embedded","monodromy":true}'
```

| Request | Result |
| --- | --- |
| `POST /api/resolve` | Returns a finished result for a fast computation. If the bounded attempt is inconclusive, reaches a computation limit, or lacks a requested zeta, the server starts a background search and returns HTTP 202 with its ID. |
| `POST /api/searches` | Starts an exhaustive resolution in the background and returns a job with an `id`. |
| `GET /api/searches/{id}` | Returns the job status and its result when available. |
| `DELETE /api/searches/{id}` | Cancels the specified job. |
| `POST /api/algebra` | Runs an algebra operation and returns its structured result. |
| `POST /api/algebra/exhaustive` | Returns the algebra result on the same connection, with the ordinary input-size and worker time limits lifted. |

Set `"search":"quick"` on a resolution request to return the bounded result
directly. The ordinary endpoints accept bodies up to 32,768 bytes and run
workers with a 55-second deadline. Use `/api/searches` or
`/api/algebra/exhaustive` for larger bodies. [Algebra request examples](../examples/algebra-api.md) · [Server implementation](../engine/app.py)

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
[Device calibration and full solves](../benchmarks/README.md#profiling-katsura-6--7-and--8)
