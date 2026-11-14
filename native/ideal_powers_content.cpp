// Symbolic powers of a radical ideal with a common hypersurface factor.
#include "ideal_powers.hpp"
#include "local_io.hpp"
#include "primary_decomposition.hpp"
#include "sparse_flint.hpp"

namespace atlas::ideal_powers {

std::optional<Basis> content_symbolic(const Basis& rows, size_t width, bool lex, const mpz_class& exponent) {
    auto h = local::polynomial_gcd(rows,width);
    if (!h || h->empty() || h->rbegin()->first == local::Powers(width)) return std::nullopt;
    auto residual = local::divide_generators(rows,*h,width);

    // Write I=hK. If I is radical, K=I:h is radical and h avoids its
    // minimal primes: colon by h removes exactly the primes containing h.
    // Conversely, when h is squarefree, K is radical, and h is regular
    // modulo K, the equality I=(h) ∩ K proves that I is radical.
    const auto kernel = primary::reduced(residual,width,true);
    if (primary::saturate_lex(kernel,*h,width) != kernel)
        throw InvalidInput("Symbolic powers require a prime or radical ideal over QQ.");

    // No irreducible factor of h divides all generators of K, since h
    // was their gcd before division. Thus the minimal primes of I are
    // those of (h) and of K. The latter give primary components of K^(n)
    // on which h stays regular, so (h^n) ∩ K^(n) = h^n K^(n).
    // Both recursive calls validate radicality even at n=0 or 1.
    const auto hypersurface = power({*h},width,lex,exponent,true);
    auto generators = power(residual,width,lex,exponent,true);
    if (exponent == 0) return generators;
    local::SparseLexRing ring(width);
    PolyValue factor(ring.ctx), row(ring.ctx), product(ring.ctx);
    ring.write(factor.p,hypersurface.front());
    for (auto& g : generators) {
        ring.write(row.p,g);
        fmpq_mpoly_mul(product.p,factor.p,row.p,ring.ctx);
        g = ring.read(product.p);
    }
    return primary::reduced(generators,width,lex);
}

} // namespace atlas::ideal_powers
