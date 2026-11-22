# Resolution, exceptional divisors and monodromy

These functions resolve a hypersurface germ at the origin, rebuild the dual complex of the exceptional divisor from the saved charts, and check the poles of the local topological zeta function for monodromy eigenvalues. The fields of a saved resolution, its charts and its divisors are described with the [methods on saved values](saved-values.md#resolution-charts-divisors-and-polynomial-multiplicity). [Resolving a plane cusp](../tutorial/resolving-a-cusp.md) · [Reading a resolution record](../thematic/resolution-records.md) · [Resolution and chart algorithms](../../native/resolution-algorithms.md)

## Search the Resolution Zoo

At the prompt or in a script, `zoo_search("1/2*x*y^3, -x^2")` returns a dictionary whose `ids` field lists matching record IDs in catalogue order. The search box in the Resolution Zoo accepts the same text. Atlas looks for every requested term in the expanded equation. All terms share one renaming of variables and one overall choice of sign. Distinct query variables must denote distinct coordinates. Thus the example matches an equation containing `-1/2*v*u^3` and `v^2`, even when the equation contains additional terms. Coefficients match their exact rational values.

Separate terms with commas or write a signed sum such as `x*y^2 - y^3`. The session also accepts a list such as `["x*y^2", "-y^3"]`. Use `*` or whitespace between variables and use `^` or `**` for nonnegative integer exponents. A name like `xy` is one coordinate. A blank query or an empty list returns all records.

With `x, y = coordinates("x", "y", coeffs=QQ)`, the polynomial query `zoo_search([x*y^2, -y^3])` gives the same result. Supply polynomial inputs from a single live parent. Text queries use their own variable names as placeholders and require no coordinate declaration.

The catalogue writes equations in coordinates `x, y, z, w, u, v`, followed by `x7, x8, …`. Within each equation, monomials appear in descending lexicographic order with `x` first. The list compares these term sequences in ascending order, comparing exponent vectors first and exact coefficients second. A shorter matching prefix comes first. Identical polynomials are ordered by record ID. Filtering and pagination preserve this order.

Select an equation to see its recorded name, labels and resolution engine. **Copy LaTeX** and **Copy Python** copy the polynomial without `= 0`. For example, the Whitney umbrella gives `x^{2} - y^{2}z` in LaTeX and `x**2 - y**2*z` in Python. Declare the displayed coordinates before evaluating the Python expression. Where the resolution data uses different coordinate names, the record shows their correspondence.
