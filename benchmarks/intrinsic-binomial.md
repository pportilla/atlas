# Intrinsic resolution of cusp cylinders

For $f=x^d-y^{d+1}$ in $\mathbb A^3_{\mathbb Q}$, the singular locus is
the line $x=y=0$. Blowing up this line resolves the surface. On the
$y$-chart the strict equation is $u^d-y$ and the map is
$(x,y,z)=(yu,y,z)$. Its reduced exceptional locus has equations $y=u=0$.
On the $x$-chart the strict equation is $1-xv^{d+1}$, which has no points
on $x=0$. For $d\ge2$ the surface starts singular, so a single blowup is
minimal for this family. Embedded resolution continues through the contact
with the exceptional divisor.

The Atlas executable runs exhaustive intrinsic resolution on CPU and emits
its full JSON record. Singular computes the reduced Jacobian locus, blows
it up with `blowUpBO` from `resolve.lib`, and retains both affine opens.
The driver checks the centre, chart count, exceptional order, and reduced
exceptional line in the Atlas record. In Singular it checks the projection,
total-transform identity, smoothness, and reduced exceptional ideals on the
two charts. Each fresh-process timing includes startup. Both systems omit
zeta functions. The warm Atlas comparison uses the same intrinsic verifier
with the previous embedded fan stopping rule or the intrinsic graph rule.

Median elapsed milliseconds from five trials after one warmup, alternating
the order on a shared AMD Ryzen 7 6800H, Python 3.14.4 and Singular 4.4.1.

| $d$ | Atlas fresh | Singular fresh | Atlas warm, previous fan | Atlas warm, intrinsic fan |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 74.36 | 39.66 | 1.33 | 1.00 |
| 5 | 69.25 | 37.81 | 1.77 | 1.02 |
| 17 | 70.24 | 38.35 | 3.39 | 1.09 |

Singular finished faster in all three fresh-process comparisons. Atlas's
intrinsic fan used one blowup in each case, compared with 3, 6 and 18
respectively when using its embedded fan.

```sh
python3 benchmarks/intrinsic_binomial.py --trials 5
python3 benchmarks/intrinsic_binomial.py --compare-fans --trials 5
```

## The canonical route in sheared coordinates

For $(x+y+z)^2-(x-y)^3$, take $A=x+y+z$ and $B=x-y$. The coordinates
$(A,B,z)$ are independent. The first canonical centre is $A=B=0$. In the
$B$-chart, $A=Bv$ gives the smooth strict equation $v^2-B=0$. Its reduced
exceptional section is the line $B=v=0$. In the $A$-chart the strict
equation $1-Au^3=0$ misses $A=0$. Embedded resolution uses two further
blowups to make the total transform transverse. The intrinsic calculation
can stop after one.

The warm Atlas comparison includes centre selection, blowups, chart
serialization and final intrinsic verification. Both runs use the same
canonical rule. Enabling its smooth-strict-transform stopping test took
33.36 ms median. Continuing to embedded normal crossings took 41.78 ms
median. The fresh-process comparison used the public quick CPU resolution,
including chart output. Atlas took 235.20 ms median and Singular took
57.26 ms median on the same shared machine, with one warmup and five trials.
Singular checked the radical centre against $(A,B)$, built the two blowup
charts, and verified the exceptional order, smoothness, and reduced
exceptional line on its chart cover.

```sh
python3 benchmarks/intrinsic_binomial.py --sheared --trials 5
python3 benchmarks/intrinsic_binomial.py --compare-canonical --trials 5
```
