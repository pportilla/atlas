# Development roadmap

Atlas can compute with ideals in rational polynomial rings and their quotients, study a germ at a rational point, and resolve a hypersurface germ at the origin. What would it take to compute a free resolution of a module, or to resolve a germ defined by several equations? The examples below are proposed extensions. Each gives a concrete calculation that a future implementation should handle. For computations available today, see the [tutorial](../tutorial/index.md), the [reference manual](../reference/index.md) or [reading a resolution record](../thematic/resolution-records.md).

## Rings and modules

The public global orders are lex and grevlex. Weighted and block orders, user-declared finite fields and algebraic extensions would need declarations, coefficient arithmetic and ring-aware maps that work across interfaces.

A first example for polynomial matrices and modules is the residue field of $R=\mathbb Q[x,y]$. Present it as $R/(x,y)$, recover the relation $(-y,x)$ on the two generators and check the free ranks $1,2,1$ of its minimal graded free resolution. This would give a starting point for syzygies, Betti tables and the later computation of Hom, Ext and Tor.

## Geometric resolution

Proposed inputs include several equations defining one germ, a user-selected blowup centre and an affine variety with several singular points. For a first multi-equation example, resolving $(z,y^2-x^3)$ should recover the intrinsic cusp after projection onto the first two coordinates. For a global example, check the map and smoothness above every singular point. [Current chart and complex conventions](../thematic/resolution-records.md#chart-and-complex-conventions)

Further questions include branch parametrizations and deformations, and monodromy invariants that distinguish Jordan block sizes. The existing resolution records provide geometric branch attachments and zeta functions. [Monodromy and topological zetas](../thematic/resolution-records.md#local-monodromy-zeta-function)

## Other proposed calculations

| Area | Example to validate |
| --- | --- |
| Real algebraic systems | Isolate the real roots of a zero-dimensional rational ideal and return intervals with exact defining polynomials. |
| Algebraic geometry and invariant theory | Compute a Riemann–Roch space of a specified divisor or generators of an invariant ring for a specified group action. |
| Tropical geometry | Return the cones and weights of the tropical fan associated to a supplied ideal. |
| Noncommutative algebra | Declare the Weyl algebra and recover the relation $\partial x - x\partial = 1$. |

The [Lean verification roadmap](lean-verification-roadmap.md) describes a separate proposal to prove a statement about a computation Atlas already performs, starting with rational ideal membership.
