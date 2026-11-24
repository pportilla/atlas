# Time Groebner.jl on a system written by the benchmark driver.
#
#   julia -t N --project=. run_groebner.jl SYSTEM CHARACTERISTIC MODE REPEATS BASIS [CHECK_PRIME [BUDGET]]
#
# SYSTEM: the number of variables, then one polynomial per line with terms
# "coefficient e1 ... en" separated by ';'. The first variable is the largest
# in the graded reverse lexicographic order.
# MODE modulo a prime: deterministic, randomized, las_vegas (linear algebra),
# or learn_apply (learn at the prime, apply at the previous prime).
# MODE over Q (CHARACTERISTIC 0): default (certify=false), or certify:
# certify=true on the homogenized ideal with deterministic linear algebra
# (with randomized linear algebra certify=true fails in Groebner.jl 0.10.11).
# The basis of the last run is written to BASIS in the format of SYSTEM;
# over Q its coefficients are written modulo CHECK_PRIME.
# Prints one JSON line with the seconds of every run; runs are not repeated
# once they have used BUDGET seconds. A small system of the same kind is
# solved first, and a first run of the system shorter than five seconds is
# not counted, so that compilation is not timed.
using Groebner, AbstractAlgebra

function read_system(path, K)
    lines = readlines(path)
    n = parse(Int, lines[1])
    R, _ = polynomial_ring(K, ["x$i" for i in 1:n]; internal_ordering = :degrevlex)
    system = elem_type(R)[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        ctx = MPolyBuildCtx(R)
        for term in split(line, ";")
            parts = split(term)
            push_term!(ctx, K(parse(BigInt, parts[1])), [parse(Int, e) for e in parts[2:end]])
        end
        push!(system, finish(ctx))
    end
    return system
end

previous_prime(p) = (q = p - 1; while !AbstractAlgebra.is_probable_prime(BigInt(q)) q -= 1 end; q)

function write_basis(path, basis, modulus)
    open(path, "w") do io
        for f in basis
            terms = String[]
            for (c, e) in zip(coefficients(f), exponent_vectors(f))
                value = if modulus == 0
                    string(lift(c))
                else
                    string(mod(numerator(c) * invmod(denominator(c), modulus), modulus))
                end
                push!(terms, string(value, " ", join(e, " ")))
            end
            println(io, join(terms, ";"))
        end
    end
end

function main()
    path, characteristic, mode, repeats, out = ARGS[1], parse(Int, ARGS[2]), ARGS[3], parse(Int, ARGS[4]), ARGS[5]
    check = length(ARGS) >= 6 ? parse(BigInt, ARGS[6]) : BigInt(0)
    budget = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : Inf
    started = time()
    K = characteristic == 0 ? QQ : GF(characteristic)
    system = read_system(path, K)
    small = Groebner.Examples.katsuran(4; k = K)
    seconds = Dict{String, Vector{Float64}}()
    basis = nothing
    if mode == "learn_apply"
        other = previous_prime(characteristic)
        system2 = read_system(path, GF(other))
        small2 = Groebner.Examples.katsuran(4; k = GF(other))
        trace, _ = groebner_learn(small)
        groebner_apply!(trace, small2)
        seconds["learn"] = Float64[]
        seconds["apply"] = Float64[]
        warm = false
        while length(seconds["learn"]) < repeats
            t = time_ns()
            trace, basis = groebner_learn(system)
            learn = (time_ns() - t) / 1e9
            t = time_ns()
            ok, _ = groebner_apply!(trace, system2)
            apply = (time_ns() - t) / 1e9
            ok || error("apply failed")
            if !warm && learn + apply < 5
                warm = true
                continue
            end
            warm = true
            push!(seconds["learn"], learn)
            push!(seconds["apply"], apply)
            time() - started > budget && break
        end
    else
        options = if characteristic == 0
            mode == "certify" ? (certify = true, linalg = :deterministic, homogenize = :yes) : (certify = false,)
        else
            (linalg = Symbol(mode),)
        end
        groebner(small; options...)
        seconds[mode] = Float64[]
        warm = false
        while length(seconds[mode]) < repeats
            t = time_ns()
            basis = groebner(system; options...)
            elapsed = (time_ns() - t) / 1e9
            if !warm && elapsed < 5
                warm = true
                continue
            end
            warm = true
            push!(seconds[mode], elapsed)
            time() - started > budget && break
        end
    end
    out != "-" && write_basis(out, basis, characteristic == 0 ? check : BigInt(0))
    fields = join(["\"$k\": [" * join(v, ", ") * "]" for (k, v) in seconds], ", ")
    println("{\"elements\": $(length(basis)), \"threads\": $(Threads.nthreads()), \"seconds\": {$fields}}")
end

main()
