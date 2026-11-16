# Prompt and scripts

Start `./atlas` from the checkout. Enter an assignment to keep a value, then enter an expression to inspect it. Here the relation $t^2=1$ makes $t^3-t$ vanish in the quotient.

```text
atlas> t = coordinates("t")
atlas> I = ideal(t^2 - 1)
atlas> membership(t^3 - t, I).value
True
atlas> membership(t, I).value
False
```

Use `^` for powers, `*` for multiplication and fractions such as `1/2` for exact coefficients. Functions return values you can assign, pass to another function or inspect through their fields. Type `help membership` for the signature and example, `help I` for the fields on this ideal, or `help functions` to list operations. Tab completes names, fields and methods. Type `:workspace` to inspect the rings and variables in use. To name a ring while binding its generators, enter `R = ring("x", "y")`.

## Run or save a calculation

Scripts share the prompt’s functions. To replay the membership calculation above and see its result, enter `print(membership(t^3 - t, I).value)` at the prompt. Then enter `:save "my session.atlas"`. From the shell, run the saved file.

```sh
./atlas run "my session.atlas"
```

The script prints `True`. Saving captures successful statements in order, including explicit prints. Saving again replaces the file. Use `./atlas run -` to read a script from standard input. At the prompt, a blank line finishes an indented block. A failed or interrupted block restores the previous variable bindings and active ring. [Session save and replay cases](../tests/test_session_transcript.py)

Use `len(I.generators)` to count the generators. To write the count as a JSON
record, use `print_json(count=len(I.generators))`.

Use `quit` or `exit` to leave the prompt.

## Inspect a saved value

Type `help` followed by the name you saved. The first line identifies the value.
For a polynomial, it names the parent ring. The examples use your binding, so
you can enter them directly at the prompt. The documentation link opens the
section for that kind of value.

```text
atlas> R = ring("x", "y")
atlas> f = y^2 - x^3
atlas> help f
f is a polynomial in R (QQ[x, y]).

Example
  f.parent
  f^2
  polynomial("x+1", ring=f.parent)

Docs  https://github.com/pportilla/atlas/blob/main/docs/sessions.md#choose-the-ring-of-a-polynomial
```

`f -h` and `f help` open the same card. Use `help R` or `help f.parent` to
inspect the ring, `help polynomial` for the constructor, and `:workspace` to
see all your saved names.

## Choose the ring of a polynomial

`R = ring("x", "y")` names the ring `QQ[x, y]` and binds `x` and `y` to its
generators at the prompt or in a script. `R.coordinates` gives those same
generators in order. `polynomial("x^2", ring=R)` reads a formula in the ring.
Repeating the declaration or calling `coordinates("x", "y")` in the same
session reuses the ring and its generators.

```text
atlas> R = ring("x", "y")
atlas> f = y^2 + x^5
atlas> f.parent == R
True
atlas> g = f + polynomial("x^2", ring=R)
atlas> g.parent == f.parent
True
```

The `Rings` row now shows `R = QQ[x, y] → x, y, f, g`. Here the four names
are polynomials with that parent. A ring that you have not named, such as one
declared only with `coordinates`, appears under a name such as `R1`.
[Name, select and delete rings](#name-select-and-delete-rings) explains these
names and the active ring. If you save an ideal, Atlas lists it under
its ambient ring. A resolution appears under the ring of its input equation.
Type `:workspace` to see each ring's coordinates, polynomials and ideals together.
Maps appear separately with their domain and codomain.
Type `help f` or `help f.parent` to inspect a saved polynomial or its ring.
The same help command accepts a saved ideal, quotient ring, or ring map.

### Name a quotient and its coordinates

Every ring owns its coordinates. Assigning a name to a ring or to one of its
coordinates gives you another way to refer to the same object.

```text
atlas> x, y = coordinates("x", "y")
atlas> I = ideal(y^2+x^5)
atlas> K = I.quotient_ring
atlas> K.variables
('x', 'y')
atlas> K.coordinates
([x], [y])
atlas> u, v = K.coordinates
atlas> u.parent == K
True
atlas> v^2 + u^5
0
```

The workspace groups `x, y` and `I` in the ambient polynomial ring `R`. Under
`K = R/(x^5+y^2)`, the coordinate entry `u = [x]` means the class of
`x` in this quotient. `K.ambient.coordinates` gives the original polynomials.
If you also write `A = quotient_ring(I)`, the heading shows both names for one
ring. A different compute backend still yields that same parent.
Use `quotient_ring(I, fresh=True)` to request an independent copy.

`polynomial` reads a nonempty string. Select this quotient with `ring=K` or `variables=K.coordinates`. If you have made it active with `use_ring(K)`, you may omit both arguments. The string
uses the names in `K.variables`, here `x` and `y`. For an existing ambient
polynomial `f`, use `K(f)` to take its residue class directly.

```text
atlas> g = polynomial("y^2", ring=K)
atlas> h = polynomial("x^5", ring=K)
atlas> K(y^2) == g
True
atlas> g == h
False
atlas> g == -h
True
```

The defining relation is `y^2 + x^5 = 0` in `K`, so the two classes are
negatives of each other. In a script, `print(g == h, g == -h)` prints
`False True`. With `ring=K.ambient`, the same strings give polynomials in
`QQ[x, y]`. A saved coordinate tuple carries its own `.parent`, including
when the ring has no generators.

`K.defining_ideal` is the ideal you supplied. `K.relations` gives its reduced
basis. The serializable presentation in `K.ring` includes the relations.

### Create an ideal in a quotient ring

With the cusp ring `K` above, pass `ring=K` to use the images of the ambient
polynomials as generators. Atlas lists the resulting ideal under `K` in the workspace.

```text
atlas> J = ideal((y, x^5), ring=K)
atlas> J.parent == K
True
atlas> J.generators
([y], [-y^2])
atlas> membership(K(x^5), J).value
True
atlas> membership(K(x), J).value
False
```

The brackets denote residue classes. Since `x^5 = -y^2` in `K`, the second
generator is a multiple of the first, so `J` is generated by the class of `y`
alone. If `u, v = K.coordinates`, then `ideal(v, u^5)` also constructs this
ideal in `K`. When mixing a residue class with an ambient polynomial, name
the target explicitly, as in `ideal(v, x^5, ring=K)`. Use `ideal(ring=K)`
for the zero ideal.

Sums, products, powers, Gröbner bases, membership and normal forms all use the
relations of the parent. For example, `normal_form(K(x^5+x), J) == K(x)`
returns `True`. Sums and products require both ideals to have the same live
parent. These calls also work in scripts. Use `print(...)` to display a result.
`intersection` and `primary_decomposition` currently take ideals in
polynomial rings only.

To form the further quotient `K/J`, write `L = J.quotient_ring`.
`L.defining_ideal` is `J`, and `L.projection` maps `K` to `L`. For example,
`L.projection(v) == 0` returns `True`. Writing `ideal(v, ring=L)` uses this
projection, so it gives the zero ideal of `L`. Polynomial representatives
still live in `L.ambient`, the original ring `QQ[x, y]`.

### Reuse and rebind coordinate names

Atlas uses the coefficient field and the generator names in the order you
give them. Omitting `coeffs=QQ` selects the rationals. If a generator would
replace a saved value, Atlas rejects the whole declaration. For example, with
`x` already in `R = QQ[x, y]`, `S = ring("x", "z")` would give `x` a
different parent. Atlas reports the conflict before assigning `S` or `z`.
Use `coordinates` to choose available names for the generators of the second
ring.

```text
atlas> x2, z2 = coordinates("x", "z")
atlas> S = x2.parent
atlas> S
QQ[x, z]
atlas> x.parent == f.parent
True
```

Here `x2` and `z2` name the generators spelled `x` and `z` in `S`. Direct
`coordinates` assignments also check the names you choose, including when
you select a generator by index. For example,
`x3, y3, z3 = coordinates("x", "y", "z")` selects `QQ[x, y, z]` while `x`,
`y`, and `f` still belong to `QQ[x, y]`. Asking for
`coordinates("y", "x")` also selects a different parent because the order changed.

To work with two independent copies of `QQ[x, y]`, request a fresh one
explicitly. You can still read a formula in either ring by passing its parent.

```text
atlas> u, v = coordinates("x", "y", fresh=True)
atlas> u.parent == f.parent
False
atlas> g = polynomial("x^2+y", ring=f.parent)
atlas> g.parent == f.parent
True
atlas> h = u^2 + v
atlas> h.parent == u.parent
True
```

Another ordinary call to `coordinates("x", "y")` returns the generators of
`f.parent`. To deliberately rebind `x` and `y` to an independent copy, use
`T = ring("x", "y", fresh=True)`. Then `x.parent == T` and `f.parent == R`.
Each copy of `QQ[x, y]` has its own name in the workspace. Ordinary assignment and unpacking a
saved tuple also rebind names. To restore the original generators,
assign `x, y = f.parent.coordinates`.

### Name, select and delete rings

Every ring that holds a saved value has a name. `R = ring("x", "y")` names a
ring as you create it. When a declaration or a calculation introduces a ring
that you have not named, Atlas binds the first free name among `R1`, `R2`,
and so on. Assigning that ring to a name of your own renames it. In a new
session:

```text
atlas> x, y = coordinates("x", "y")
atlas> u, v = coordinates("x", "y", fresh=True)
atlas> R1 == x.parent, R2 == u.parent
(True, True)
atlas> P = R2
atlas> P == u.parent
True
```

The workspace now lists `R1 = QQ[x, y] → x, y` and `P = QQ[x, y] → u, v`.
`R2` is free again, and the next unnamed ring receives it. Names you choose
yourself stay until you rebind them. If you rebind a ring's last name while
its values remain, Atlas gives the ring a free `R` name again.

Use `use_ring` to choose the active ring. A string passed to `polynomial`
without `ring=` or `variables=` is read in the active ring. `ideal()` and ideals
generated only by numbers, such as `ideal(1)`, also belong to it.

```text
atlas> use_ring(R1)
QQ[x, y]
atlas> f = polynomial("y^2 - x^3")
atlas> f.parent == R1
True
atlas> I = ideal()
atlas> I.parent == R1
True
atlas> ideal(u).parent == P
True
```

Polynomials and coordinates keep their own parent, so `ideal(u)` lies in `P`.
`S = ring("s", "t")` binds the generators, and `use_ring(S)` selects the
ring. A call nested inside an expression returns only the ring. For example,
`S = use_ring(ring("s", "t"))` selects the ring in one line, and
`s, t = S.coordinates` then binds its generators.
`resolve("y^2-x^3")` and `monodromy_check` also read a string in the active
ring when it is a polynomial ring over QQ. The workspace marks the active ring
with `(active)`. `active_ring()` returns it, and `use_ring(None)` clears the
choice. A quotient ring can be active as well. Then `polynomial("y^2")` gives
a residue class in that quotient.

`delete_ring` removes a ring and every saved name that depends on it. That
includes the names of the ring and of its quotients, the values in these rings,
maps to or from them, and resolutions of their polynomials with their charts.
Other rings keep their values.

```text
atlas> J = ideal(f)
atlas> K = J.quotient_ring
atlas> use_ring(K)
QQ[x, y]/(x^3-y^2) (grevlex)
atlas> delete_ring(R1)
Deleted QQ[x, y]: x, y, R1, f, I, J, K
No ring is active.
```

The result lists the removed names in `.names`, and `.was_active` records
whether the active ring was deleted. In that case no ring is active
afterwards. A value that only refers to a deleted ring through a field stays.
For example, the result of `eliminate` keeps its `.inclusion` map.

At the prompt, a failed command that has already selected or deleted a ring
leaves the names and the active ring as they were. These functions work the
same way in scripts, and `:save` records them with your other statements.

## Add, multiply, and raise ideals to powers

At the prompt, use `I + J` for an ideal sum, `I * J` for a product, and
`I^n` for an ordinary power. Use the corresponding function names in scripts:

```atlas
x, y = coordinates("x", "y")
I = ideal(x^2, y)
J = ideal(x, y^2)
S = ideal_sum(I, J)
P = ideal_product(I, J)
print(groebner(S).generators)  # (x, y)
print(groebner(P).generators)  # (x^3, y^3, x*y)
print(ideal_power(I, 2).generators)
```

Run [this example script](../examples/ideal-arithmetic.atlas) with `./atlas run examples/ideal-arithmetic.atlas`.
The sum uses generators from both ideals. The product uses pairwise products.
Both operands must be ideals with the same live parent ring. To add the
principal ideal generated by a polynomial `f`, write `I + ideal(f)`.
The result uses the left ideal's monomial order. Both spellings work
at the prompt and in scripts.

## Move a polynomial between rings

To add `z` to a polynomial in `QQ[x, y]`, declare the larger ring under
available names and specify where each generator of the original ring goes.

```text
atlas> x, y = coordinates("x", "y")
atlas> f = y^2 + x^3
atlas> X, Y, z = coordinates("x", "y", "z")
atlas> include = hom(f.parent, z.parent, images=(X, Y))
atlas> g = include(f) + z
atlas> g.parent
QQ[x, y, z]
atlas> g == Y^2 + X^3 + z
True
```

`include.domain`, `include.codomain`, and `include.images` describe the map.
The images follow the generator order of the domain and are substituted
simultaneously. Between `QQ[x, y]` and `QQ[x, z]`, choosing images
`(x2, z2)` sends `y` to the generator spelled `z`.

Use `I.parent` for the live ring of an ideal. For example,
`I = ideal(f)`, then `polynomial("x+y", ring=I.parent)` uses the same
generators as `f`. The field `I.ring` contains the serializable description
and monomial order. Use `ideal(ring=f.parent)` for the zero ideal and
`polynomial("1/2", ring=QQ)` for a scalar. Both typed and quoted inputs to
`resolve` can use an existing ring: `resolve(f)` and
`resolve("y^2+x^3", ring=f.parent)` have the same original parent.

For `A = quotient_ring(I)`, `A.ambient` is the original polynomial ring,
`A.coordinates` and `A.gens` give its generators' residue classes, and `A.projection` sends
elements of `I.parent` to their classes. Compare inside the quotient with `A(f) == A(0)`. For
`J = eliminate(I, x)`, `J.inclusion` sends a polynomial in the surviving
coordinates back into the original ring.

The [ring map tests](../tests/test_session_maps.py) check simultaneous substitution, quotient relations and the surviving coordinates after elimination.

<a id="inspect-a-cusp"></a>
<a id="look-inside-a-chart"></a>

For a map arising from a blowup, [follow a cusp into its chart](resolving-a-cusp.md#look-inside-a-chart).
