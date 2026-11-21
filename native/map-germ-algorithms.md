# Certificates for the extended tangent quotient

For a parametrized cusp $f(t)=(t^2,t^3)$, which infinitesimal changes of the map survive changes of coordinates in source and target? The function `ae_codimension` counts the independent directions that survive. It returns a certified finite dimension, a proof of infinite dimension, or a lower bound when the finite-jet calculation reaches its work limit. [Compare the cusp equation and its parametrization](../docs/tutorial/local-algebra.md#compare-the-cusp-and-its-parametrization)

For a single map germ $f:(\mathbb Q^n,0)\to(\mathbb Q^p,0)$ let
$R=\mathbb Q[[x]]$, $B=\mathbb Q[[y]]$ and regard $R$ as a $B$-module by
substitution $y\mapsto f$. The extended tangent quotient is
$N=R^p/(df\,R^n+f^*(B^p))$. Its vector-space dimension is the
$\mathcal A_e$-codimension. Constant vector fields on both spaces take part.

For the cusp, substitution from the target produces the constants and all powers of $t$ of degree at least two. Thus only $(t,0)$ and $(0,t)$ survive the target vector fields. The derivative $(2t,3t^2)$ kills the first direction modulo those fields. Its multiples by $t$ have order at least two in both entries, so the second direction survives. The quotient has dimension one. [David Mond, *Some Open Problems in the Theory of Singularities of Mappings* (2015), §1.1](https://www.journalofsingularities.com/volume12/mond.pdf)

For more complicated germs, the sections below describe how Atlas certifies that the tangent quotient has the reported dimension. [Native dispatcher and filtered jets](ae_codimension.cpp#L395) · [Public interface cases](../tests/test_ae_codimension_interface.py) · [Timing reports](../benchmarks/map-germ-timings.md)

## Filtered jets

Set $M=R^p/df\,R^n$ and $I=(f)R$. At jet degree $D$,
sparse rational echelon bases compute the source Jacobian module and the
target-vector span separately. Independent target rows generate successive
products by $f_j$ modulo the Jacobian module. The finite-jet quotient dimension
is always a lower bound. Full contact layers from degree $k\le D$ imply
$\mathfrak m^k M\subset IM$ by Nakayama over $R$ applied modulo $IM$.
Thus $M$ is finite over $B$ by completeness. For $L_d$ the image of
$\mathfrak m^d M$ in $N$, full tangent layers from degree $d$ through $D$
give $L_d=L_{D+1}$. If $d+k\le D+1$, then
$L_d=L_{d+k}\subset (y)L_d$ and Nakayama over $B$ gives $L_d=0$.
This proves the reported dimension, even when earlier jet dimensions have
temporarily stopped growing. A positive-dimensional support for
$M/IM$, detected locally using $f$ and the maximal minors of $df$, certifies
infinity because target fields contribute at most $p$ constants modulo $I$.
[Filtered computation](ae_codimension.cpp)

## Weighted tangent ranks

Suppose the source variables have positive weights
$w_i$ and each component $f_j$ is homogeneous of degree $d_j>0$. In weight
$q$ the ambient vector fields have component degrees $q+d_j$. The source
tangent rows are $x^a\partial_i f$ with $|a|_w=q+w_i$. The target rows are
$f^b e_j$ with $|b|_d=q+d_j$. These span the full extended tangent space in
that weight. The matrix corank over $\mathbb Q$ is therefore $\dim N_q$.
Clearing denominators by an invertible target scaling makes the rows integral.
Maximal rank modulo one prime proves maximal rational rank. Every other rank
is computed exactly over the integers.

Euler's identity $\sum_i w_i x_i\partial_i f=(d_j f_j)_j$ removes redundant
rows. Fix a target position $s$. For each target monomial $f^b$ of weight
$q$, the row $f^b f_s e_s$ is a linear combination of source Jacobian rows
and rows in the other target positions, since $d_s$ is nonzero.

When one target component is a nonzero multiple of a monomial $g=x^a$, let
$h$ be its weight. Multiplication by $g$ embeds the ambient space at weight
$q-h$ into the space at weight $q$, and carries tangent relations to tangent
relations. Each column retains its component and its monomial with all
powers of $g$ divided out. At fixed weight this labels columns uniquely.
The new columns are exactly the monomials not divisible by $g$. The only
new relations needed are derivatives with multipliers not divisible by $g$
and target rows omitting the selected target variable. Euler's identity
still removes rows in a different target position.

For each residue modulo $h$, the kernel keeps a basis for the annihilator
of the preceding tangent space. Inherited vector fields are evaluated on
this basis. The new monomials contribute independent coordinates. Computing
the right nullspace of the new relations on these two sets of coordinates
therefore gives the exact dual of the current normal space. A full-column-rank
modular image certifies a zero piece directly. In every other case the kernel
computes an integral nullspace spanning the rational kernel and carries it
forward in the original monomial coordinates. A homogeneous component
$c x_i+H$ with $H$ independent of $x_i$ can supply a monomial after the
bounded source substitution $x_i\mapsto x_i-H/c$. This automorphism preserves
the grading and the contact certificate. Before exact elimination, the kernel
uses Hadamard's inequality to bound minor sizes from the heights of individual
matrix columns. When it carries the nullspace basis into the next weight, it
charges work only for the nonzero change-of-basis entries it multiplies.
[Native quotient calculation](ae_weighted_quotient.cpp)

Let $k$ be the contact order proved by the filtered jets. Since
$\mathfrak m^kM\subset(f)M$, the monomial vectors of ordinary degree less
than $k$ span $M/(f)M$. Their weights are at most
$G=(k-1)\max_i w_i-\min_j d_j$. Thus for $q>G$,
$N_q=\sum_j f_j N_{q-d_j}$. A run of $\max_j d_j$ consecutive zero pieces
ending at or above $G$ proves by induction that every later piece vanishes.
For this method, `contact_order` records $k$ and `last_weight` records the
weight at the end of that run of zeros.
The sum of the computed coranks is then the dimension of the completed normal
space, since positive weights give finite pieces and tend to infinity with
ordinary degree. The kernel caches the Jacobian and the pullbacks across
weights. If the bounded rank calculation runs out of work, the graded graph
can still prove a finite or infinite answer. [Weighted rank computation](ae_weighted_ranks.cpp)

## Graded graph elimination

When the source admits positive weights making
each component homogeneous, introduce graph equations $y_j-f_j$ and square-zero
markers $z_j$ with every $z_i z_j=0$. Impose the Jacobian rows
$\sum_j (\partial_i f_j)z_j$. In the resulting ring the marker-degree-one part
is the polynomial Jacobian cokernel. Under lex order $x>y>z$, reduction of
a target vector introduces no source variable. Standard monomials linear
in the markers and divisible by some $x_i$ therefore form a basis of the
normal quotient. Partitioning by the first such $x_i$ gives monomial
colengths, including a proof of infinity when a section has infinite length.
Positive weights identify the completion with the product of the finite
weight pieces, so the finite count or infinite conclusion applies at the origin.
[Graded graph computation](ae_graded.cpp)

## Scalar germs and curves

For an isolated scalar germ, the Jacobian algebra
has length $\mu$. The image of $B$ is spanned by $1,[f],\dots,[f]^{e-1}$,
where $e$ is the nilpotence index of $[f]$. Its minimal polynomial is $T^e$,
so these classes are independent and the answer is $\mu-e$. For a primitive
monomial curve with numerical semigroup $S$, there are initially $p\,g(S)$
missing powers across the $p$ target positions. The disjoint source vectors
$t^j df$ eliminate $1+g(S)-\mathrm{type}(S)$ directions. The kernel counts
the gaps and pseudo-Frobenius numbers using the Apéry set. Regular maps, ramified
covers and split unfoldings admit further direct tangent-space calculations.
[Native formulas and their hypotheses](ae_special.cpp)

## Homogeneous surface maps with a finite coordinate projection

Suppose
$f=(x^a,y^b,h(x,y))$ and $h$ is a homogeneous binary form of degree $c$.
The polynomial source $R=\mathbb Q[x,y]$ is free over $S=\mathbb Q[x^a,y^b]$
with basis $x^iy^j$ for $i<a$, $j<b$. At weight $w$ the three positions of
an ambient vector field have polynomial degrees $w+a,w+b,w+c$. In the
first position, divide by $a x^{a-1}$ and carry the quotient times
$-\partial_xh$ into the third position. Use $b y^{b-1}$ and $-\partial_yh$
in the second position. These independent pivots remove precisely the source
vector fields. This leaves a finite-dimensional vector space in each weight.

Target rows are pullbacks of monomials $X^iY^jZ^k$ in each position.
Multiplication by $h$ is an endomorphism of the free $S$-module $R$ of rank
$ab$. Its homogeneous characteristic equation bounds the necessary powers by
$k<ab$. In the third position it suffices to use $k=0$, since
$df\cdot(x,y)^{\mathsf T}=(a x^a,b y^b,c h)^{\mathsf T}$ by Euler's identity.
The remaining third-position rows are distinct coordinate vectors, so remove
their rows and columns before computing rank. Whenever the Jacobian pivot
divides a whole pullback, its contribution to the third position is a
monomial shift of $-h^k\partial_x h$ or $-h^k\partial_y h$. The modular
calculation reuses these products across rows and weights. It reduces entries
as pivots are encountered and skips the zero suffix of each pivot row.
A full-rank minor modulo the prime certifies full rank over $\mathbb Q$,
so modular ranks alone prove the zero pieces of the tail below. Atlas
looks for that tail first, modulo 65521. The search ends at weight
$abc+\max(a,b)$ or when a piece exceeds the matrix bound. If no tail has
been proved by then, the pieces left unproved are ranked again modulo 65519,
since one prime can lower the ranks of every piece of a map. Once a tail is
proved, Atlas constructs the integer matrix of each piece still unproved,
in order of weight, and computes its exact rank with the surviving columns
in their original order, stopping at the first tail proved over
$\mathbb Q$. An unlucky prime can only postpone the modular proof. Maps of
infinite codimension therefore never reach the exact ranks in this
calculation, and graph elimination decides their codimension.

The remaining quotient $N$ is generated over $S$ by the residue monomials in
each position. Their maximum weight is $W=a+b-2-\min(a,b,c)$. For $w>W$,
$N_w=x^aN_{w-a}+y^bN_{w-b}$. Consequently $\max(a,b)$ consecutive zero
pieces ending at or above $W$ prove that every higher piece vanishes. The
reported dimension is the sum through this certified tail, and
`last_weight` records the last weight computed. The positive grading gives
the same finite dimension for the completed germ at the origin. Three
powers of distinct rational linear forms can also be put into the coordinate
projection by a linear source change and independent target rescalings.
[Native rank computation](ae_graded_surface.cpp) · [Independent rational reference](../tests/test_ae_coprime_reference.py)

## Multiplicity-two maps into a hypersurface

For $f=(u,v^2,h(u,v))$ set
$s=v^2$, $A=\mathbb Q[[u,s]]$ and $R=A\oplus vA$. The even part of $h$ can
be removed by a target coordinate change. Writing its odd part as $vP(u,s)$,
the target subalgebra is $A\oplus vP A$. Each target position therefore
contributes $v(A/P)$ before quotienting by source fields. Odd multiples of
$\partial_{u_i}f$ and even multiples of $\partial_vf$ eliminate the first
$n$ positions. The complementary multiples give the ideal
$(P,\partial_{u_1}P,\dots,\partial_{u_{n-1}}P,s\partial_sP)$ in the last
position. Atlas computes its local colength. For $h=v^3+u^k v$ this gives
$k-1$, including exponents beyond 64 bits. [Native reduction](ae_special.cpp)
