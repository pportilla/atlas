# Ideal arithmetic at the prompt and in resolution charts

## Public operators

On two pairs of ideals in `QQ[x,y]`, the sum, product and ordinary square agreed with Singular by mutual ideal containment in every trial. Singular was faster on all six cases. Five trials of 16 operations measured prepared operands on one CPU core. The separate Atlas native timings include sparse input decoding and result serialization. [Inputs and timing driver](ideal_operators.py) · [Individual trials and medians](ideal-operators-ryzen-6800h.json)

```sh
python3 benchmarks/ideal_operators.py --cpu 5 --trials 5 --batch 16
```

## Products of strict components on a resolution chart

Each measurement times construction of the strict-transform ideal, including the final choice of generators. Charts have three private native coordinates over `QQ` and an empty ambient ideal. Each batch uses fresh coordinates on 20 charts with integer parameter `i` ranging from 1 to 20. Chart construction happens before the timer starts. After discarding one warm-up batch, the driver reports all seven trials and their median in milliseconds per chart. The measurements below used the Ryzen 7 6800H with ordinary process affinity. [Chart inputs and timing driver](strict_component_products.py)

| Strict component ideals | Individual polynomial products (ms/chart) | Batched native product (ms/chart) |
| --- | ---: | ---: |
| `(x³+y²+iz)` | 0.080 | 0.077 |
| `(x³+y²+iz), (y−z), (x+iy), (z+ix)` | 0.443 | 0.169 |
| `(x+iz,y), (y+ix,z), (z+iy,x)` | 1.838 | 0.832 |

Build the native kernel and run the following command from the repository root. To repeat the baseline on the earlier checkout (`702548e`), use the same driver with that checkout on `PYTHONPATH` and its native kernel built.

```sh
python3 -m benchmarks.strict_component_products
```
