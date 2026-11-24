# Benchmarks

For a computation you launch from the shell, look for **fresh CLI** times. They include starting the program and writing its answer. For repeated queries in one session, look for **warm handler** times and check whether the report reuses an ideal or a prepared basis. A **kernel** time isolates the native calculation specified in the report.

Each report identifies the inputs, the check for agreement and the machine and trial conditions. Run its reproduction commands from the repository root. When comparing solvers, check what each program returns. For example, in the inclined pinch-point comparison Atlas printed the complete resolution in JSON, while Singular printed the final chart count. [Reduced singular locus timings](reduced-centre-resolution.md)

## Algebra

<a id="gröbner-kernel"></a><a id="the-verifiers-bases"></a><a id="ideals-whose-coefficients-swell"></a><a id="phase-1-rational-ideal-arithmetic"></a><a id="membership-with-a-prepared-basis"></a><a id="replays-on-the-gpu"></a><a id="full-katsura-7-solve-on-cpu-and-gpu"></a><a id="profiling-katsura-6--7-and--8"></a><a id="one-verification-end-to-end"></a>

- [Gröbner basis and normal form timings](groebner-timings.md) — Basis verification, coefficient growth, prepared membership and modular replay on CPU and GPU.
- [The F4 trace of the modular Gröbner route](groebner-f4.md) — Katsura-11 modulo one prime and over Q, and standard systems, against msolve, giac and Groebner.jl.
- [Multivariate polynomial GCD benchmark](polynomial_gcd.py) compares the public Atlas call, generic FLINT and Singular on nine Liao–Fateman examples and a 401-bit coefficient case. The reports include the original single-call comparison, repeated calls on prepared inputs, and first use on fresh operands. [Pinned CPU](polynomial-gcd-prepared-ryzen-6800h.json) · [Default scheduling](polynomial-gcd-prepared-unpinned-ryzen-6800h.json) · [Before, pinned](polynomial-gcd-ryzen-6800h.json) · [Before, default scheduling](polynomial-gcd-unpinned-ryzen-6800h.json)

<a id="primary-decomposition-over-qq"></a><a id="normalization-of-affine-algebras-over-qq"></a><a id="localization-at-a-prime-over-qq"></a><a id="ordinary-and-symbolic-powers-over-qq"></a><a id="arithmetic-in-declared-quotient-rings-over-qq"></a>

- [Ideal and quotient arithmetic timings](ideal-arithmetic.md) — Primary decomposition, normalization, prime localization, ideal powers and arithmetic in quotient rings.
- [Ideal sums and products in scripts](ideal-operators.md) measures how long it takes to declare two ideals and compute with them in the same session.

<a id="local-dimension-and-smoothness-against-singular"></a>

- [Local dimension and smoothness timings](local-algebra.md) — Local Krull dimension and smoothness on schemes of different dimensions, at rational points.

<a id="extended-right-left-codimension-of-map-germs"></a><a id="pairwise-coprime-powers-of-three-lines"></a><a id="atlas-kernel-timing-for-graded-surface-maps"></a><a id="published-map-germs"></a>

- [Map-germ codimension timings](map-germ-timings.md) — Finite and infinite map-germ examples, graded families and published inputs.
- [Performance log for map-germ codimension](../docs/developer/perf-log-ae-codimension.md) follows individual changes to the Hilbert, graded rank and jet calculations on a fixed family.

- [Applying homomorphisms to polynomials](ideal-arithmetic.md#ring-homomorphisms) measures three maps on prepared inputs.
- [Coupled sparse local systems](large-coupled-local.md) and [quotient bases and hypersurface invariants at a point](point-quotient-invariants.md) measure shifted equations whose expansions can be very large.
- [Shifted standard bases and weak normal forms](shifted-local-standard.md) describes how the local ideals and remainders are certified.

## Geometric resolution

- [Local monodromy eigenvalues and full Bernstein–Sato polynomials](local-monodromy-roots.md) compares targeted local decisions in Atlas and Singular with full polynomial computations on the same weighted homogeneous inputs.

<a id="same-host-resolution-and-zeta-functions"></a><a id="measurements-on-the-headless-host"></a><a id="topological-zeta-reduction-from-certified-strata"></a><a id="verifying-the-3915-chart-fan"></a><a id="pullbacks-and-jacobian-orders-on-the-regular-fan"></a><a id="projection-formatting-and-verification-on-the-regular-fan"></a><a id="newton-coordinate-choices"></a><a id="native-chart-equations-and-intrinsic-orbit-restrictions"></a><a id="formatting-the-independently-verified-fan-equations"></a><a id="general-blowups-and-intrinsic-chains"></a><a id="affine-euler-characteristics-on-the-double-quadric-resolution"></a><a id="shared-torus-verification"></a><a id="published-inputs-and-the-four-variable-germ"></a><a id="skipping-saturation-during-resolution"></a><a id="residual-order-at-smooth-rational-points"></a>

- [Resolution and zeta timings](resolution-timings.md) — Complete resolutions, verified charts, exceptional strata and zeta functions.

- For 21 previously rejected germs, the driver compares fresh Atlas embedded and intrinsic resolutions with Singular's checked embedded resolution. It also compares Milnor and Tjurina numbers and the Milnor fibre Euler characteristic for each input. On the Ryzen 7 6800H, Atlas completed all three trials in both modes for every germ. Singular finished all three trials for ten germs, and Atlas had the lower embedded median in all ten. Nine Singular cases exceeded 15 seconds in every trial and two returned an undefined `invSat` error. [Comparison driver](zoo_resolution.py) · [Conditions and individual trials](zoo-resolution-ryzen-6800h.json)

- Run `python3 benchmarks/zoo_search.py --copies 20 --trials 5` to compare the sorted catalogue and three monomial queries with Singular. All 4,920 row IDs agreed, including the order of equal polynomials. Median ordering times were 84.0 ms for Atlas, including index construction, and 3,941 ms for Singular on already parsed polynomials. Median search times ranged from 0.17 to 0.42 ms in Atlas and from 437 to 602 ms in Singular. Repeating the original 5,000-row workload with `--limit-surfaces 50 --copies 100 --trials 5` gave ordering medians of 13.8 and 2,698 ms and search medians of 0.04–0.08 and 198–259 ms, respectively. Each comparison uses one warmup and five measured trials. [Search driver](zoo_search.py) · [Conditions and individual trials for 4,920 rows](zoo-search-ryzen-6800h.json)

- A separate benchmark compares the intrinsic dual complex of the double-quadric surface with and without the canonical divisor labels. [Intrinsic gluing driver](zoo_intrinsic_gluing.py) · [Gluing trials](zoo-intrinsic-gluing-ryzen-6800h.json)

- [Arithmetic on chart coordinates](resolution-timings.md#arithmetic-on-chart-coordinates) times multiplication on a graph chart and division by an algebraic coefficient.
- [Canonical blowup stages and centre choices in one native call, before and after](canonical-native-steps.md)
- [Centre selection and exceptional intersections](canonical-chart-batches.md) and [replay of saved charts](native-resolution-boundary.md) compare the geometry returned by whole CLI runs.
- [Performance log for canonical blowups](../docs/developer/perf-log-resolve.md) traces the cost of successive stages for one surface and records a correction to the ninth-blowup measurements.
- [Performance log for Jung's method](../docs/developer/perf-log-jung.md) times `jung_resolution` next to `atlas resolve` on the same surfaces. [Driver](jung_timings.py)
- The inclined pinch point illustrates why the choice of centre matters. The early Atlas run took nine blowups and produced 36 charts. Blowing up the singular line directly needed one blowup and two charts. The intervening studies include inclined and nonlinearly sheared centres. [Early search](exhaustive-search-resolution.md) · [Inclined centres](inclined-centre-resolution.md) · [Sheared centres](sheared-centre-resolution.md) · [Reduced singular locus](reduced-centre-resolution.md)

## Execution and devices

<a id="cpu-versus-opencl-on-intel-iris-xe"></a><a id="independent-chart-transformations"></a><a id="parallel-modular-rank"></a><a id="reproduce-on-another-device"></a>

- [Device timings for chart transformations and rank](device-timings.md) — Independent chart substitutions and modular rank on Intel Iris Xe.

- [Default CPU paths](symbolic-paths.md) tracks the change from symbolic computations to native workers and explains how the path was audited.
