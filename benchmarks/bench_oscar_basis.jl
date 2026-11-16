# Persistent OSCAR worker for the native reduced-basis comparison.
# Each method line is followed by one bench_groebner_basis.cpp request.
import Oscar
using Oscar: QQ, QQFieldElem, QQMPolyRingElem, base_ring, coeff, degrevlex,
             divexact, exponent_vector, fglm, groebner_basis, groebner_basis_f4, ideal,
             lex, polynomial_ring

const SERIAL_F4 = Symbol("f4-serial")

function read_polynomial(io, ring, n)
    line = readline(io)
    isempty(line) && return zero(ring)
    coefficients = QQFieldElem[]
    exponents = Vector{Int}[]
    for term in split(line, ';')
        fields = split(term)
        length(fields) == n + 1 || error("Wrong number of exponents")
        rational = split(fields[1], '/')
        numerator = parse(BigInt, rational[1])
        denominator = length(rational) == 1 ? 1 : parse(BigInt, only(rational[2:end]))
        push!(coefficients, QQ(numerator // denominator))
        push!(exponents, parse.(Int, fields[2:end]))
    end
    return ring(coefficients, exponents)
end

struct OscarBasisFailure <: Exception
    cause::Exception
end

function solve(method, input, order)
    # Distinguish errors from the actual solver call from benchmark or
    # protocol errors. Only the former can mark an individual case failed.
    basis = try
        if method == :default
            groebner_basis(input; ordering=order, complete_reduction=true)
        elseif method in (:f4, SERIAL_F4)
            threads = method == SERIAL_F4 ? 1 : Threads.nthreads()
            groebner_basis_f4(input; nr_thrds=threads, complete_reduction=true)
        else
            fglm(input; start_ordering=degrevlex(base_ring(input)), destination_ordering=order)
        end
    catch ex
        ex isa Union{InterruptException,OutOfMemoryError,StackOverflowError} && rethrow()
        throw(OscarBasisFailure(ex))
    end

    # Indexing IdealGens materializes the coefficients in the OSCAR ring,
    # including Singular -> Nemo conversion for the default and FGLM.
    result = QQMPolyRingElem[]
    for polynomial in basis
        iszero(polynomial) && continue  # Some APIs present the zero basis as [0].
        # The storage order of this ring is the requested term order.
        leading = coeff(polynomial, 1)
        push!(result, isone(leading) ? polynomial : divexact(polynomial, leading))
    end
    return result
end

function measure(method, ring, generators, order, batch)
    # Each ideal has its own basis cache and its own solver-side representation.
    # Only the original, already parsed QQ polynomials are shared across solves.
    solve(method, ideal(ring, generators), order)
    inputs = [ideal(ring, generators) for _ in 1:batch]
    results = Vector{Vector{QQMPolyRingElem}}(undef, batch)
    start = time_ns()
    for i in eachindex(inputs)
        results[i] = solve(method, inputs[i], order)
    end
    ms = (time_ns() - start) / (1e6 * batch)
    basis = last(results)
    for other in @view results[1:end-1]
        if other != basis && sort(sparse_rows(other)) != sort(sparse_rows(basis))
            error("Inconsistent OSCAR bases in one batch")
        end
    end
    return ms, basis
end

function sparse_rows(basis)
    return map(basis) do polynomial
        io = IOBuffer()
        for i in 1:length(polynomial)
            c = coeff(polynomial, i)
            i > 1 && print(io, ';')
            print(io, numerator(c))
            isone(denominator(c)) || print(io, '/', denominator(c))
            for e in exponent_vector(polynomial, i)
                print(io, ' ', e)
            end
        end
        String(take!(io))
    end
end

function run_case(input, output, method; startup=false)
    method in (:default, :f4, SERIAL_F4, :fglm) || error("Unknown OSCAR method: $method")
    rows, batch = parse.(Int, split(readline(input)))
    n, order_id, known = parse.(Int, split(readline(input)))
    rows >= 0 && batch >= 1 && known == 0 && order_id in (0, 1) ||
        error("Invalid benchmark request")
    ring, _ = polynomial_ring(QQ, [Symbol("x", i) for i in 1:n];
                              internal_ordering=order_id == 0 ? :degrevlex : :lex, cached=false)
    generators = QQMPolyRingElem[read_polynomial(input, ring, n) for _ in 1:rows]
    order = order_id == 0 ? degrevlex(ring) : lex(ring)
    if method in (:f4, SERIAL_F4) && order_id != 0
        println(output, "SKIP msolve F4 supports degrevlex only")
    elseif method == :fglm && order_id != 1
        println(output, "SKIP the FGLM comparison converts degrevlex to lex")
    else
        try
            ms, basis = measure(method, ring, generators, order, batch)
            rows = sparse_rows(basis)
            println(output, "TIME ", ms)
            println(output, "BASIS ", length(rows))
            foreach(row -> println(output, row), rows)
        catch ex
            ex isa OscarBasisFailure || rethrow()
            startup && throw(ex.cause)
            summary = first(split(sprint(showerror, ex.cause; context=:limit => true), '\n'))
            label = order_id == 0 ? "grevlex" : "lex"
            println(output, "SKIP BACKEND_ERROR: $method $label n=$n generators=$(length(generators)): ",
                    nameof(typeof(ex.cause)), ": ", summary)
        end
    end
    flush(output)
end

function main()
    # Compile the full reader/solver/writer path before the caller starts a
    # case timeout. All methods use the same two-variable zero-dimensional ideal.
    for (method, order) in ((:default, 0), (:default, 1), (:f4, 0), (:fglm, 1))
        small = "2 2\n2 $order 0\n2 2 0;-1 0 1\n1 0 2;-1 0 0\n"
        run_case(IOBuffer(small), devnull, method; startup=true)
    end
    println("READY ", VERSION, ' ', Base.pkgversion(Oscar))
    flush(stdout)
    for line in eachline(stdin)
        run_case(stdin, stdout, Symbol(line))
    end
end

main()
