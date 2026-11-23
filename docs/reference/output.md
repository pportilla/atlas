# Script output and replay

These functions display values at the prompt or in a script, write them as JSON, and replay an algebra request saved from the command line or the HTTP API. At the prompt, an expression on a line by itself is displayed without `print`. A script shows only what it prints. [Run or save a calculation](../tutorial/prompt-and-scripts.md#run-or-save-a-calculation) · [Algebra requests and replay](algebra-requests.md)

## Print and export results

```atlas
R = ring("x", "y")
f = y^2 - x^3
print("Milnor number", milnor_number(f).value)            # Milnor number 2
print(len(groebner(ideal(x^2, y^3)).generators))          # 2
```

In a script, `print_json(milnor=milnor_number(f).value)` writes the line `{"milnor": 2}`. Each call writes one JSON object, with polynomials as strings and exact rationals as decimal strings.
