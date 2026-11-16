# Local dimension and smoothness timings

The CPU comparison measures the local Krull dimension and smoothness over QQ on 12 ideal-and-point inputs, including a point outside the scheme, with a separate timing for each operation. Both engines compute the local dimension. For smoothness they also compute the embedding dimension, Jacobian rank, and regularity verdict. A point outside the scheme must be rejected. The corpus includes a translated cusp, a nonreduced point, a plane meeting a line, and two six-coordinate rank-one matrix varieties. Three minors define the matrix variety of dimension four. The harder presentation applies a nonlinear coordinate change and translates the vertex to a rational point. [Inputs and timing harness](local_dimension_cpu.py) · [Independent Singular checks](../tests/test_local_smoothness_singular.py)

On the Ryzen 7 6800H, both engines used CPU only and were pinned to logical CPU 5. Every warm sample averages 96 recomputations on pre-parsed inputs after one discarded warm-up. The reported times are medians of seven such samples with rotating engine order. Atlas’s public handler includes the execution report returned to the application. The separate C++ measurement starts with a typed ideal and point. Singular 4.4.1 runs one thread, and its comparison time is the lower median across two procedures. One uses a generic local standard basis. The other also uses incidence, height, and Jacobian-rank shortcuts. Cold times are medians of five fresh CLI processes for each engine and procedure, including parsing and output. [Raw samples, generator lists, and build hashes](local-dimension-cpu-ryzen-6800h.json)

| Local smoothness | Atlas warm C++ | Atlas warm public handler | Singular warm | Atlas cold CLI | Singular cold CLI |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shifted cusp | 2.56 µs | 13.35 µs | 48.57 µs | 4.66 ms | 7.28 ms |
| Plane meeting a line | 0.43 µs | 9.54 µs | 28.98 µs | 4.74 ms | 6.96 ms |
| Nonlinearly transformed rank-one matrix vertex | 38.54 µs | 68.61 µs | 309.79 µs | 5.28 ms | 8.07 ms |

Atlas had the lower median in all 24 case/operation pairs at each of the three timing boundaries. The median of the 24 per-case Singular/Atlas ratios was 25.68× for the warm C++ kernel, 3.50× for the warm public handler, and 1.52× for the cold CLI. On the transformed matrix vertex, the local-dimension handler took 67.09 µs against Singular’s 255.58 µs. These timings apply to the listed corpus, build, and machine.

Run these commands from the repository root using the Atlas build environment.

```sh
make -C native -f groebner.mk build/local_algebra build/bench_local_dimension
python3 benchmarks/local_dimension_cpu.py --batch 96 --repeats 7 --cpu 5 --cold-repeats 5 --output /tmp/local-dimension-cpu.json
```

The large coupled ideal comparison measures the same two operations on sparse systems that exceed the former translation limit. It includes separate Singular runs on the original equations and in verified local coordinates. [Results and reproduction instructions](large-coupled-local.md)

The shifted local standard-basis and normal-form benchmark checks returned polynomials, leading monomials and unit certificates, and times fresh CPU invocations on the original equations. [Results and reproduction instructions](shifted-local-standard.md)

The rational-point quotient benchmark compares the local quotient basis, Jacobian algebra, Milnor number and Tjurina number with Singular. It checks the rank of each finite basis in the source coordinates and includes a colength exceeding 64 bits. [Results and reproduction instructions](point-quotient-invariants.md)
