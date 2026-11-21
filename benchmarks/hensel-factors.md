# Coprime polynomial Hensel lifting on the Radeon 680M

Atlas and Singular recovered the same ordered monic factors modulo the squared prime on all three inputs. The timings below are medians in milliseconds from seven warm trials. The Atlas CPU was faster than the Triton route on all three, and Singular was faster than both. [Individual timings and software versions](hensel-factors-680m.json)

| Factor degrees | Prime | Atlas CPU | Atlas Triton | Singular |
|---|---:|---:|---:|---:|
| 1, 1 | 7 | 0.294 | 0.617 | 0.103 |
| 32, 33 | 257 | 1.113 | 1.685 | 0.195 |
| 128, 129 | 257 | 3.441 | 4.636 | 0.800 |

The quadratic case lifts the factors of $x^2-2$ modulo seven. The two larger cases use dense seeded inputs with known lifts. Atlas timings cover the public dispatcher, input parsing, memory transfers, a native coefficient-by-coefficient product check and the result record. Each timed GPU run required a physical Triton convolution. The finite-field inverse, corrections and product certification run on the CPU.

The first implementation used sparse Euclid on every input and serialized its intermediate product in both modes. Dense inputs now use FLINT's univariate modular inverse, and the CPU route reuses the projected polynomials in one native call. On the largest original input the CPU median went from 5.138 to 3.441 ms and the Triton median went from 5.605 to 4.636 ms. Sparse inputs with arbitrary-size exponents retain sparse Euclid.

Run `python benchmarks/hensel_factors.py --trials 7` with a physical Triton device at index zero and Singular on the path. These measurements used a Ryzen 7 6800H with integrated Radeon 680M, HIP, Triton 3.5.1 and Singular 4.4.1. Atlas alternates CPU and GPU order after a warmup. Single-threaded Singular times input construction, the correction over the prime field using `extgcd` and conversion to the squared-prime residue ring. It checks both returned factors and their product there outside the timer. Process startup is excluded in both systems. [Benchmark driver](hensel_factors.py)
