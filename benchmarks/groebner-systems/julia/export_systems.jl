# Write example systems of Groebner.jl as text: the number of variables, then
# one polynomial per line with terms "coefficient e1 ... en" separated by ';'.
# Coefficients are integers: each polynomial is multiplied by the least common
# denominator of its coefficients.
using Groebner, AbstractAlgebra

function dump(path, system)
    n = nvars(parent(system[1]))
    open(path, "w") do io
        println(io, n)
        for f in system
            cs = collect(coefficients(f))
            scale = reduce(lcm, map(denominator, cs); init = BigInt(1))
            terms = String[]
            for (c, e) in zip(cs, exponent_vectors(f))
                push!(terms, string(numerator(c * scale), " ", join(e, " ")))
            end
            println(io, join(terms, ";"))
        end
    end
end

E = Groebner.Examples
out = ARGS[1]
for n in 5:9
    dump(joinpath(out, "cyclic-$n.txt"), E.cyclicn(n))
end
for n in 5:9
    dump(joinpath(out, "noon-$n.txt"), E.noonn(n))
end
for n in 4:8
    dump(joinpath(out, "reimer-$n.txt"), E.reimern(n))
end
for n in 7:13
    dump(joinpath(out, "eco-$n.txt"), E.econ(n))
end
dump(joinpath(out, "henrion-5.txt"), E.henrion5())
dump(joinpath(out, "henrion-6.txt"), E.henrion6())
dump(joinpath(out, "henrion-7.txt"), E.henrion7())
for n in 4:6
    dump(joinpath(out, "katsura-gjl-$n.txt"), E.katsuran(n))
end
