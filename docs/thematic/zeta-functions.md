# Zeta functions and the monodromy conjecture

Let $f\in\mathbb Q[x_1,\dots,x_n]$ vanish at the origin. Two invariants of the germ of $f=0$ at the origin can be read from an embedded resolution.

- The **local topological zeta function** $Z_{\mathrm{top},0}(s)$ of Denef and Loeser, a rational function of $s$ built from the numerical data of the resolution.
- The **local monodromy zeta function** $\zeta_{f,0}(T)$, which records the eigenvalues of the monodromy on the cohomology of the Milnor fibre at the origin. A’Campo's formula computes it from the same resolution.

The topological monodromy conjecture of Denef and Loeser predicts that each pole $s_0$ of $Z_{\mathrm{top},0}$ gives an eigenvalue $e^{2\pi i s_0}$ of the monodromy at some point of $f=0$ near the origin. Atlas computes both zeta functions exactly and looks for such an eigenvalue for every pole.

This tutorial computes the two functions for the cusp, shows where every term comes from, and then checks the conjecture on germs whose eigenvalues appear only away from the origin. Each section is a complete script that starts from an empty session. The [cusp tutorial](../tutorial/resolving-a-cusp.md) introduces resolutions, and [Resolution objects and zeta functions](../reference/resolution-values.md) lists every field used here.

## Compute both zeta functions

Declare the coordinates, write the equation and resolve it. The zeta functions are fields of the resolution.

```atlas
R = ring("x", "y")
f = y^2 - x^3
Z = resolve(f)
print(Z.topological_zeta.expression)    # (4*s + 5)/((s + 1)*(6*s + 5))
print(Z.monodromy_zeta.expression)      # (1-T^6)/((1-T^2)*(1-T^3))
print(Z.topological_zeta.poles[1].value)    # -5/6
```

`resolve(f)` computes the geometry. The first time you read `Z.topological_zeta` or `Z.monodromy_zeta`, Atlas computes both functions from the saved charts and keeps them. To compute them during the call, write `resolve(f, monodromy=True)`. When only the zeta functions matter, `monodromy_check(f)` computes them together with the pole check described below.

Both functions are always computed on an embedded resolution. If `Z` is an intrinsic or weighted resolution, Atlas resolves the same equation in embedded mode when you first read a zeta function.

### From the shell

The command line computes the same functions and saves them with the resolution.

```sh
./atlas resolve 'y^2-x^3' --vars x,y --monodromy --json > cusp.json
./atlas monodromy cusp.json
```

The second command prints the monodromy zeta function of the saved record and the Euler characteristic of the Milnor fibre. Over HTTP, add `"zetas": ["monodromy", "topological"]` to a resolution request. [Command line](../reference/command-line.md#resolve-a-germ) · [HTTP requests](../reference/http-and-compute.md#request-a-resolution)

## The topological zeta function

Let $\pi$ be an embedded resolution and $E_i$ the components of the total transform $\pi^{-1}(f=0)$, the strict transform included. Each component has two numbers. The multiplicity $N_i$ is the order of $f\circ\pi$ along $E_i$, and $\nu_i-1$ is the order of the pulled-back volume form $\pi^*(dx_1\wedge\dots\wedge dx_n)$. For a set $I$ of components, $E_I^\circ$ denotes the points lying on exactly the components in $I$. [J. Denef and F. Loeser, *Caractéristiques d’Euler-Poincaré, fonctions zêta locales et modifications analytiques* (1992)](https://webusers.imj-prg.fr/~francois.loeser/jams92.pdf)

$$Z_{\mathrm{top},0}(s)=\sum_I \chi\big(E_I^\circ\cap\pi^{-1}(0)\big)\prod_{i\in I}\frac{1}{N_i s+\nu_i}.$$

Atlas returns the numerical data and the Euler characteristics of the strata.

```atlas
R = ring("x", "y")
T = resolve(y^2 - x^3).topological_zeta
print(T.numerical_data[0])    # {'component': 'E1', 'multiplicity': 2, 'nu': 2}
print(T.numerical_data[1])    # {'component': 'E2', 'multiplicity': 3, 'nu': 3}
print(T.numerical_data[2])    # {'component': 'E3', 'multiplicity': 6, 'nu': 5}
print(T.numerical_data[3])    # {'component': 'strict', 'multiplicity': 1, 'nu': 1}
print(len(T.strata))          # 6
print(T.strata[2])            # {'components': ['E3'], 'euler_characteristic': -1}
print(T.strata[5])            # {'components': ['E3', 'strict'], 'euler_characteristic': 1}
```

The curves `E1` and `E2` are projective lines, each meeting the rest of the total transform in one point, so each open part has Euler characteristic $1$. `E3` meets `E1`, `E2` and the strict transform, so its open part is a projective line minus three points, with Euler characteristic $-1$. The three intersection points each contribute $1$. The strict transform is not compact, and only its intersection with $\pi^{-1}(0)$ counts. Summing the six terms gives

$$\frac{1}{2s+2}+\frac{1}{3s+3}-\frac{1}{6s+5}+\frac{1}{(2s+2)(6s+5)}+\frac{1}{(3s+3)(6s+5)}+\frac{1}{(s+1)(6s+5)}=\frac{4s+5}{(s+1)(6s+5)}.$$

Every pole lies among the candidates $-\nu_i/N_i$. Here they are $-1$ (from `E1`, `E2` and the strict transform) and $-5/6$ (from `E3`). In general a candidate can cancel. `T.poles` keeps only the poles that survive, with their orders.

## The monodromy zeta function

Let $F_0$ be the Milnor fibre of $f$ at the origin and $h$ its monodromy. Atlas uses the convention

$$\zeta_{f,0}(T)=\prod_q\det\big(1-T\,h^*\mid H^q(F_0;\mathbb Q)\big)^{(-1)^{q+1}},$$

for the **unreduced** cohomology, so the trivial eigenvalue of $H^0$ is included. Write $E_i^\circ$ for $E_i$ with its intersections with all other components removed, including the strict transform. A’Campo's formula only needs the exceptional components over the origin. [N. A’Campo, *La fonction zêta d’une monodromie*, Theorem 3 (1975)](https://doi.org/10.1007/BF02565748)

$$\zeta_{f,0}(T)=\prod_i\big(1-T^{N_i}\big)^{-\chi(E_i^\circ\cap\pi^{-1}(0))}.$$

```atlas
R = ring("x", "y")
f = y^2 - x^3
M = resolve(f).monodromy_zeta
print(M.contributions[0])    # {'component': 'E1', 'multiplicity': 2, 'euler_characteristic': 1}
print(M.contributions[1])    # {'component': 'E2', 'multiplicity': 3, 'euler_characteristic': 1}
print(M.contributions[2])    # {'component': 'E3', 'multiplicity': 6, 'euler_characteristic': -1}
print(M.expression)          # (1-T^6)/((1-T^2)*(1-T^3))
print(M.milnor_fiber_euler_characteristic)    # -1
print(milnor_number(f).value)                 # 2
```

The contributions give $(1-T^2)^{-1}(1-T^3)^{-1}(1-T^6)^{1}$. The Euler characteristic of the Milnor fibre is minus the degree of $\zeta_{f,0}$, namely $\chi(F_0)=\sum_i N_i\,\chi(E_i^\circ\cap\pi^{-1}(0))=2+3-6=-1$. For an isolated plane curve singularity $\chi(F_0)=1-\mu$, which agrees with the Milnor number $\mu=2$.

## Check the monodromy conjecture

The conjecture asks, for each pole $s_0$ of $Z_{\mathrm{top},0}$, for a point $x$ of $f=0$ arbitrarily close to the origin at which $e^{2\pi i s_0}$ is an eigenvalue of the monodromy on $H^*(F_x;\mathbb C)$. If $s_0=-a/q$ in lowest terms, $e^{2\pi i s_0}$ is a primitive $q$-th root of unity. It is an eigenvalue exactly when the cyclotomic polynomial $\Phi_q$ divides the numerator or the denominator of the local zeta function at $x$.

`monodromy_check(f)` resolves $f$, computes both zeta functions and records a witness for every pole.

```atlas
R = ring("x", "y")
f = y^2 - x^3
P = monodromy_check(f)
print(P.status)                                     # verified
print(P.topological_zeta.expression)                # (4*s + 5)/((s + 1)*(6*s + 5))
p = P.poles[1]
print(p.value, p.eigenvalue_order)                  # -5/6 6
print(p.cyclotomic_exponent, p.witness)             # 1 origin zeta
q = P.poles[0]
print(q.value, q.eigenvalue_order, q.witness)       # -1 1 origin zeta
```

For the pole $-5/6$, the polynomial $\Phi_6=T^2-T+1$ divides $1-T^6$ but neither $1-T^2$ nor $1-T^3$. Its exponent in the origin zeta is $+1$, so the origin is a witness. The pole $-1$ predicts the eigenvalue $1$, and $\Phi_1=1-T$ appears with exponent $-1$.

The `status` summarizes the check. `verified` means every pole has a witness. `refuted` means that an exhaustive check proved that some pole's eigenvalue is absent from every Milnor fibre near the origin, which would contradict the conjecture. A quick check can end with `unresolved` or `inconclusive`, described in [Quick checks](#quick-checks). For a resolution you already have, `Z.monodromy_check` gives the same answer as `monodromy_check(Z)`. [Fields of a pole check](../reference/resolution-values.md#pole-check)

## Monodromy near the origin

The eigenvalue predicted by a pole need not occur at the origin. Consider the four planes $xyz(x+y)=0$. The origin zeta function is $1$, yet the topological zeta function has a pole at $-2/3$.

```atlas
R = ring("x", "y", "z")
P = monodromy_check(x*y*z*(x + y))
print(P.status)                              # verified
print(P.monodromy_zeta.expression)           # 1
print(P.topological_zeta.expression)         # -(s - 2)/((s + 1)^2*(3*s + 2))
p = P.poles[1]
print(p.value, p.cyclotomic_exponent)        # -2/3 0
print(p.witness)                             # nearby ordinary multiple point
print(p.transverse.point, p.transverse.transverse_germ)   # ('0', '0', '1') u1^2*u2+u1*u2^2
N = P.monodromy_zeta.nearby_strata
print(N[p.nearby_matches[0].stratum_index].expression)    # (1-T^3)
```

The three planes $x=0$, $y=0$ and $x+y=0$ contain the $z$-axis. At the point $(0,0,1)$ the germ is a product of a smooth factor and the transverse triple point $u_1u_2(u_1+u_2)$, whose zeta function $1-T^3$ contains $\Phi_3$. Such points lie arbitrarily close to the origin, so they witness the eigenvalue $e^{-4\pi i/3}$ of order three. The double pole $-1$ is witnessed by $H^0$ of the Milnor fibre at the origin, which always carries the eigenvalue $1$.

`P.monodromy_zeta.nearby_strata` lists every zeta function Atlas certified near the origin. Each entry has a `model` and the data of that model. The models are smooth points of a component of $f=0$, transverse binomials and Morse singularities on coordinate loci, transverse ordinary multiple points along the intersection lines of a plane arrangement, two smooth branches with generic contact along a coordinate subspace of codimension two, constant transverse types along unused coordinates, and transverse compositions. In `p.nearby_matches`, each `stratum_index` points into this list and `cyclotomic_exponent` is the exponent of $\Phi_q$ in that stratum's zeta function, positive for a zero and negative for a pole. The list may cover only part of the singular locus, so an empty list does not mean that the singularity is isolated.

The web workspace started by `./atlas serve` shows the same information. With both zeta functions selected, its witness table lists each actual pole and the strata whose local zeta function contains the predicted eigenvalue. For example, its **Singular line** example $x^2+y^2+tz^2+z^3$ has zeta function $1$ at the origin. At points of the line $x=y=z=0$ with $t\ne0$ the transverse Morse singularity has zeta function $1/(1-T^2)$, which supplies the eigenvalue $-1$ for the pole $-3/2$.

### Transverse compositions

A transverse composition writes $f=G(h_1,\dots,h_k)$, where the $h_i$ vanish on a component of the critical locus of codimension $k$. Near a point of $h_1=\dots=h_k=0$ where $dh_1,\dots,dh_k$ are independent, the $h_i$ are part of a coordinate system. There $f$ is $G$ times a smooth factor and has the zeta function of $G$. Atlas checks $f=G(h)$ as an identity of polynomials, and checks that the points where $dh$ drops rank form a smaller set, so that good points exist arbitrarily close to the origin.

This search decomposes the critical locus of $f$, which can take long: for the six hyperplanes $xyztu(x+y+z+t+u)=0$ it is still working after two minutes. The zeta functions do not depend on it. A quick search therefore gives it 15% of its time budget (3 of 20 seconds), keeps the compositions certified by then, and records in `monodromy_zeta.nearby_search` that the search stopped. An exhaustive search lets it finish. [Search](../../native/monodromy.cpp) · [Time limit](../../engine/resolve.py)

For $f=(x^2+y^2+z^2)^2+x^5$ the singular locus is the pair of lines $x=y^2+z^2=0$. With $u_1=x^2+y^2+z^2$ and $u_2=x$, Atlas finds $G=u_1^2+u_2^5$, whose zeta function $(1-T^{10})/((1-T^2)(1-T^5))$ witnesses the pole $-7/10$. That eigenvalue of order ten is absent from the zeta function at the origin.

```atlas
R = ring("x", "y", "z")
P = monodromy_check((x^2+y^2+z^2)^2+x^5)
print(P.status)                                # verified
t = P.poles[2].transverse
print(t.coordinates)                           # ('x^2+y^2+z^2', 'x')
print(t.transverse_germ)                       # u2^5+u1^2
```

Atlas computes the zeta function of a Brieskorn–Pham germ $\sum c_iu_i^{a_i}$ from its join formula and resolves any other $G$. The search tries the generators of the component's prime ideal and the irreducible factors of $f$ and its partial derivatives that vanish on the component. It covers at most eight variables and inputs of degree at most 32 with at most 128 terms.

## Decide the remaining poles exactly

When no zeta function at or near the origin contains the predicted eigenvalue, an exhaustive check runs an exact test. For a pole $s_0$, Atlas tests the roots $\alpha$ of the local Bernstein–Sato polynomial $b_{f,0}$ with $s_0-\alpha\in\mathbb Z$. The numbers $e^{2\pi i\alpha}$ are exactly the monodromy eigenvalues on the Milnor fibres at points sufficiently near the origin, the origin included. The test therefore decides every remaining pole, for non-isolated and non-reduced germs as well. [Saito, Remark 2.12](https://www.numdam.org/item/10.24033/bsmf.2227.pdf#page=15) · [Native root test and proof](../../native/resolution-algorithms.md#deciding-nearby-monodromy-eigenvalues-for-a-pole)

```atlas
R = ring("x", "y", "z")
P = monodromy_check(z^3 + x^4*y + x^5)
print(P.status)                                # verified
print(P.monodromy_zeta.expression)             # 1/(1-T^3)
p = P.poles[1]
print(p.value, p.cyclotomic_exponent)          # -7/12 0
print(p.witness, p.bernstein_root)             # local Bernstein–Sato root -7/12
print(P.local_bernstein_sato.by_pole["-7/12"].tested_roots)   # ['-7/12']
```

The pole $-7/12$ predicts an eigenvalue of order twelve, which the origin zeta $1/(1-T^3)$ lacks. The root $-7/12$ of $b_{f,0}$ supplies it. A check is `refuted` exactly when this test finds no root for some pole. The stronger form of the conjecture would ask for $b_{f,0}(s_0)=0$ for the pole itself.

## Germs made of polynomial graphs

Some germs have zeta functions that Atlas computes without charts in three dimensions. Suppose that $f$ factors over $\mathbb Q$ into graphs $z=c_i(x,y)$ through the origin. The pairwise differences of the roots define a plane curve. When that curve is reduced at the origin, Atlas resolves it and integrates over the fibre coordinate $z$. The record then has status `zetas_computed`, and its plane resolution is saved in `contact_certificate.base_resolution`. [Derivation of both formulas](../../native/resolution-algorithms.md#integrating-three-polynomial-graphs-over-a-resolved-contact-curve)

```atlas
R = ring("x", "y", "z")
f = z*(z-y^2+x^3)*(z-y^2+x^3-x^4-y^3)
P = monodromy_check(f)
print(P.status, P.resolution.status)           # verified zetas_computed
print(P.monodromy_zeta.expression)             # (1-T^6)*(1-T^8)*(1-T^9)*(1-T^11)/((1-T^3)*(1-T^18)*(1-T^22)*(1-T^32))
print(len(P.poles))                            # 4
```

The four poles are $-1$, $-11/18$, $-13/22$ and $-19/32$, all witnessed at the origin. `P.candidate_poles` also shows the candidates that cancel. For example $-5/8$ has eigenvalue order $8$, cyclotomic exponent $0$ and pole order $0$.

### Four graphs

Adding the graph $z=y$ changes both zeta functions. At a point of a strict contact branch two roots coincide and count as a double root.

```atlas
R = ring("x", "y", "z")
f = z*(z-y^2+x^3)*(z-y^2+x^3-x^4-y^3)*(z-y)
P = monodromy_check(f)
print(P.status)                                # verified
print(len(P.poles), P.poles[0].value, P.poles[0].order)   # 4 -1 2
```

### Three quadratic translates

Atlas also treats three factors $(z-c_i(x,y))^2-D(x,y)$ over $\mathbb Q$ with centres $c_i$ and a common radicand $D$, all vanishing at the origin. It recovers the centres and the radicand from the factorization. The contact calculation applies when $D$, the centre differences $d_{ij}=c_i-c_j$ and the polynomials $d_{ij}^2-4D$ have a reduced product at the origin. [Regression examples](../../tests/test_quadratic_translate_monodromy.py)

```atlas
R = ring("x", "y", "z")
D = y^3 - x^5
A = y^2 - x^3
B = x^4 + y^3
f = (z^2 - D)*((z - A)^2 - D)*((z - A - B)^2 - D)
P = monodromy_check(f)
print(P.status)                                        # verified
print(P.monodromy_zeta.milnor_fiber_euler_characteristic)   # 415
```

The six roots $c_i\pm\sqrt D$ have contact orders and ramification that determine the integrals on the plane resolution. All eight poles, a double pole at $-1$ and seven simple poles, have witnesses at the origin.

## Quick checks

By default `monodromy_check` and `resolve` search exhaustively, until the result is certified or you interrupt them. With `search="quick"` the same search has a time budget of 20 seconds, or the `seconds=N` you give, and the resolution is inconclusive only when the budget ends. [Quick search](resolution-pipeline.md#quick-search)

```atlas
t = coordinates("t")
f = t^200
Q = monodromy_check(f, search="quick")
print(Q.status)                       # inconclusive
print(Q.reason)                       # Quick parsing accepts exponents up to 128.
P = monodromy_check(Q.resolution)
print(P.status, P.poles[0].value)     # verified -1/200
```

A quick check reports `inconclusive` when it stopped before both zeta functions were computed, and `unresolved` when they were computed but a pole still lacks a witness. Passing the resolution of a quick check, `Q.resolution`, to `monodromy_check` continues the check exhaustively from the saved record. With `check_isolation=True`, a check that ends `unresolved` or `refuted` also computes the local Milnor number in `P.milnor_number`. For an isolated singularity, a quick check that misses a predicted eigenvalue then reports `isolated_candidate`, a case to check independently.
