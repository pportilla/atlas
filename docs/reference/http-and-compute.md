# HTTP API and compute options

The Atlas server provides the resolution workspace in a browser and accepts calculations as JSON. Start it with `./atlas serve` on the machine that runs Atlas. Its default address is `http://127.0.0.1:8765`. Choose another port with `--port`.

When Atlas runs on a remote machine, forward the port from the machine with your browser. In a terminal on that machine, use `ssh -N -L 8765:127.0.0.1:8765 user@host` with your usual SSH destination in place of `user@host`. Keep the SSH session open, then open the local address in your browser. The same tunnel carries the requests on this page.

## Request a resolution

To resolve the plane cusp $y^2=x^3$, send its equation and ordered coordinates to `POST /api/resolve`. The following request asks for the embedded resolution and both zeta functions.

```sh
curl -sS http://127.0.0.1:8765/api/resolve \
  -H 'Content-Type: application/json' \
  -d '{
    "polynomial": "y^2-x^3",
    "variables": ["x", "y"],
    "mode": "embedded",
    "search": "quick",
    "zetas": ["monodromy", "topological"]
  }'
```

For the cusp the response is HTTP 200 and `status` is `"resolved"`. In the record, `monodromy_zeta.expression` gives `(1-T^6)/((1-T^2)*(1-T^3))`. The topological zeta function has poles $-1$ and $-5/6$. Their entries in `monodromy_check.poles` report monodromy eigenvalues of order one and six respectively. [Reading the exceptional divisors and zeta functions](../thematic/resolution-records.md)

For zeta functions alone, send `"zetas_only": true` in embedded mode. Atlas can then return `"status": "zetas_computed"` with a replayable plane contact certificate. The default selection includes both functions. An explicit `zetas` list selects individual functions. [Three-graph example](../thematic/resolution-records.md#verify-three-polynomial-graphs) · [Four-graph example](../thematic/resolution-records.md#verify-four-polynomial-graphs) · [Three quadratic translates](../thematic/resolution-records.md#verify-three-quadratic-translates)

Choose `search` according to how your client will receive the result.

| Request to `/api/resolve` | How to read the response |
| --- | --- |
| `"search": "quick"` | Runs a bounded search and returns its outcome directly. Read `status` to see whether the calculation resolved the input. |
| Omit `search` | Tries quick resolution first. If the attempt reaches a limit or a requested result still needs a certificate, Atlas returns HTTP 202 with the ID of an exhaustive background job. |
| `"search": "exhaustive"` | Starts a background job immediately. Use its returned `id` to follow progress. |

The default `method` is `"auto"`. For embedded resolution by weighted blowups, choose `"method": "weighted"`. Use `"zetas": []` for geometry alone or select the zetas to compute using an embedded companion. [Weighted resolution options and output](../thematic/weighted-resolution.md#resolve-with-method-weighted)

## Follow a background search

To submit a job and return immediately, use `POST /api/searches`. On successful submission the response is HTTP 202. Here is a shell example that saves the job ID and checks its state. It uses Python 3 to read the ID from the JSON.

```sh
job_id=$(curl -fsS http://127.0.0.1:8765/api/searches \
  -H 'Content-Type: application/json' \
  -d '{"polynomial":"y^2-x^3","variables":["x","y"],"mode":"embedded","zetas":[]}' \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["id"])')
curl -sS "http://127.0.0.1:8765/api/searches/$job_id"
```

Repeat the second command to check progress. A `completed` job has a `result`. Read `result.status` to learn whether the input was resolved. While a job is running, `progress` can include the current stage and blowup count.

| Request | Response |
| --- | --- |
| `GET /api/searches/{id}` | The selected job, including `state`, available `progress` and the completed `result`. |
| `GET /api/searches` | A `jobs` array, newest submission first. |
| `GET /api/searches/current` | The most recently submitted `job`, or `null` if there is none. |
| `DELETE /api/searches/{id}` | The state of the job after asking Atlas to cancel it. |

Submission to `/api/searches` defaults to `"search": "exhaustive"`. Use `"auto"` to try quick resolution first, or `"quick"` for a bounded background search. The server runs up to two jobs at once and queues the rest in submission order. It retains finished jobs for 30 minutes. Save `result` if you need the record later.

## Replay zeta functions

You can obtain a resolution first and decide which zeta functions to compute later. Send `"zetas": []` for geometry alone. For a completed job, post the names you want to `/api/searches/{id}/zetas`. For example, with `job_id` from the preceding section, this request adds the topological zeta function.

```sh
curl -sS "http://127.0.0.1:8765/api/searches/$job_id/zetas" \
  -H 'Content-Type: application/json' \
  -d '{"zetas":["topological"]}'
```

The updated job contains `result.topological_zeta`. For the cusp its expression is $(4s+5)/((s+1)(6s+5))$. To request both functions together, give `["monodromy", "topological"]`. If one was already computed, Atlas recomputes the pair using the same strata.

Once the job has expired, use `POST /api/zetas`. Its JSON body takes the saved resolution object as `record` and the names as `zetas`. This endpoint returns the zeta outputs directly. Replay checks a saved embedded or contact certificate against the polynomial. For an intrinsic record without embedded data or an inconclusive quick record, Atlas can start a fresh embedded calculation from the recorded equation and ordered coordinates. [Local monodromy zeta function](../thematic/resolution-records.md#local-monodromy-zeta-function) · [Topological zeta function](../thematic/resolution-records.md#topological-zeta-function)

## Submit an algebra request

Use `POST /api/algebra` for polynomial and ideal arithmetic. A request specifies the operation, an ordered ring and its operands. The result is returned directly. A Gröbner basis response can also include a reference for later normal forms or membership tests. [Worked algebra requests and complete operation tables](algebra-requests.md)

The standard algebra and resolution endpoints accept up to 32,768 bytes per request. A standard algebra request and a quick resolution have a 55-second worker deadline. For a larger algebra calculation, post to `/api/algebra/exhaustive`, which keeps the connection open until the worker finishes. For a larger resolution request, use `/api/searches` to obtain a job ID. To check that your client can reach the server, `GET /api/health` returns `{"status":"ok"}`.

## Select a device

Resolution defaults to CPU execution. Rational ideal arithmetic defaults to automatic device selection. To see the available devices on the machine doing the calculation, run `./atlas devices`.

| Policy | Effect |
| --- | --- |
| `cpu` | Runs the calculation on the CPU. |
| `auto` | Allows Atlas to select a supported device for eligible stages. |
| `gpu` | Requests GPU acceleration for eligible stages. The record describes any CPU fallback. |

Pass the policy as `--compute` from the shell or as `compute` in a JSON request. For example, this resolves the surface $xy=z^5$ on the CPU.

```sh
./atlas resolve 'x*y-z^5' --vars x,y,z --mode intrinsic --compute cpu --json
```

Look in `execution.stages` to see which backend actually ran. Resolution records include the reason for fallback on the affected chart or rank stage. Algebra records collect the reasons in `execution.fallback_reasons`.

For OpenCL, install its headers, loader and device driver, then build with `make gpu`. Triton uses a compatible GPU-enabled PyTorch and Triton installation. Set `RESOLUTION_GPU_BACKEND` to `triton` or `opencl` and `RESOLUTION_GPU_INDEX` to the device ordinal when you need to choose a particular device. [Device calibration and Gröbner basis timings](../../benchmarks/groebner-timings.md#profiling-katsura-6--7-and--8)

On the tested Radeon 680M with ROCm 7.1, initializing PyTorch before the HIP driver caused host-to-device transfers to fail. Atlas initializes the driver during device discovery. In a Python application embedding Atlas on this system, call `engine.gpu_devices.available_devices()` before importing PyTorch. Supply `TRITON_LIBHIP_PATH` when the HIP runtime lies outside the system library path.

## Exact arithmetic on Triton

Use `compute="gpu"` in an Atlas session or `--compute gpu` on the algebra CLI. The HTTP API accepts the same `compute` value. Run `./atlas devices` to see the devices that passed an integer kernel probe. Record the physical device with the result by inspecting `.execution` at the prompt or `execution.stages` in the JSON output.

| Operation family | Atlas entry points | Work on the GPU |
|---|---|---|
| Scalar and polynomial arithmetic | `polynomial_arithmetic` | Coefficient addition, subtraction, products, integer limbs, powers, derivatives and simultaneous substitutions. Marked number fields retain their exact defining equation. |
| Finite fields | `modular_arithmetic` | Arithmetic in prime fields and finite extensions given by `field_polynomial`. The native host checks primality and irreducibility for extensions. |
| Division and normal forms | `polynomial_division`, `normal_form`, `cancel_fraction` | Independent reductions for binomials, divisor witnesses for monomial ideals and eligible GCD stages in rational fraction cancellation. |
| GCD and Bézout coefficients | `gcd`, `extended_gcd` | Modular root screening on eligible coordinate sections and modular basis stages. Exact division and Bézout identities are obtained in the rational polynomial ring. |
| Factorization | `factor`, `squarefree_factor`, `absolute_factor` | Modular root screening. The host verifies multiplicities, factors the residual polynomial and constructs any exact algebraic coefficient field needed for the result. |
| Linear algebra | `matrix_analysis` | Independent row witnesses and eligible bounded determinants. The host certifies results over the rationals and obtains the exact kernel, image and solution. |
| Modular methods | `modular_arithmetic`, `hensel_factors` | Reduction, Chinese remaindering, bounded rational reconstruction and lifting simple polynomial roots. Two-factor Hensel lifting uses device convolution and the host verifies the lifted product modulo the squared prime. The modular Gröbner solver can also combine prime images on the GPU. |

GPU kernels run on eligible shapes and coefficient ranges. The exact native CPU solver handles larger moduli, residual factors and other cases outside an individual kernel’s range. Automatic mode generally uses the CPU for small inputs. The timings include data movement and show where explicit GPU execution paid off on the Radeon 680M. [Polynomial arithmetic](../../benchmarks/gpu-polynomial.md) · [Arithmetic over a number field](../../benchmarks/triton-marked-arithmetic.md) · [Finite fields](../../benchmarks/triton-finite-fields.md) · [GCD, division and factorization](../../benchmarks/gpu-factor-division.md) · [Matrices and modular lifting](../../benchmarks/matrix-triton.md) · [Modular arithmetic](../../benchmarks/triton-modular.md) · [Hensel factor lifting](../../benchmarks/hensel-factors.md)
