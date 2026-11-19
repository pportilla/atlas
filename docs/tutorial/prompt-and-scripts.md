# The prompt and scripts

Does $t^2-1$ divide $t^3-t$? Open the prompt with `./atlas` in the project directory. Declare an indeterminate, form the principal ideal, and ask whether the second polynomial belongs to it.

```text
atlas> t = coordinates("t")
atlas> I = ideal(t^2 - 1)
atlas> membership(t^3 - t, I).value
True
atlas> membership(t, I).value
False
```

An assignment keeps its result under a name. An expression on a line by itself displays its value. Use `^` for powers, `*` for multiplication and `1/2` for an exact rational coefficient. Tab completes names and fields. A blank line finishes an indented block. If a command fails or you interrupt a block, its assignments and active-ring selection are rolled back. Type `quit` to leave the session.

## Run or save a calculation

Suppose you want to send the divisibility calculation to a colleague. Put the following lines in a file called `divides.atlas`.

```atlas
t = coordinates("t")
I = ideal(t^2 - 1)
print(membership(t^3 - t, I).value)
```

From the shell, `./atlas run divides.atlas` prints `True`. The prompt and scripts accept the same expressions. Use `print` to choose what a script displays, or `print_json(count=len(I.generators))` to write a named result as JSON.

You can also save a calculation you've already made. At the prompt, enter the `print` statement above, then `:save "my session.atlas"`. This writes the successful statements in order, including your explicit prints. Running `./atlas run "my session.atlas"` reproduces the calculation. Saving again to the same path replaces the file. Use `./atlas run -` to read from standard input.

## Inspect a saved value

There are two kinds of help you might want while working. Type `help membership` to look up the function's arguments and an example. Type `help I` to ask about the particular ideal you saved. Try the latter with a polynomial.

```text
atlas> R = ring("x", "y")
atlas> f = y^2 - x^3
atlas> f.parent
QQ[x, y]
```

Now type `help f`. The first line names `R` as the parent of `f`, and the examples use the name `f` so you can try them directly. You can also ask for `help f.parent` or use `:workspace` to see the names grouped under each ring. For the arguments to a method such as `K.element(f)` or `C.pullback(f)`, see [Methods on saved values](../reference/saved-values.md).
