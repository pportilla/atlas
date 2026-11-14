// Exact Macaulay elimination for small plane complete intersections.
#include "ideal_powers.hpp"
#include "groebner.hpp"
#include <flint/fmpz_poly.h>
#include <algorithm>

namespace atlas::ideal_powers {
std::optional<std::vector<Poly>> echelon_products(const Ring&,const std::vector<Poly>&,size_t,size_t);
// For two degree-d forms with relatively prime highest homogeneous parts,
// the points at infinity are empty. R/I has length d², and
// length(R/I^n) = d²*n(n+1)/2 by the free conormal filtration. Every row
// below belongs to I^n; a zero-dimensional candidate initial ideal of this
// colength therefore certifies completeness without further pair reductions.
std::optional<std::vector<Poly>> plane_products(const Ring& ring, const std::vector<Poly>& inputs,
                                                  const std::vector<Poly>& products, size_t n) {
    if (ring.lex || ring.n != 2 || inputs.size()!=2 || n < 2) return std::nullopt;
    const size_t d = inputs[0].m.front();
    if (inputs[1].m.front()!=d || !d || (n+1)*d>24) return std::nullopt;
    // A nonzero binary homogeneous resultant is equivalent to coprimality.
    // Euclidean gcd over QQ of f(1,t),g(1,t) plus the possible root [0:1].
    fmpz_poly_t a,b,g;
    fmpz_poly_init(a); fmpz_poly_init(b); fmpz_poly_init(g);
    for (size_t i=0;i<2;++i) for (size_t j=0;j<inputs[i].size();++j) {
        auto m = inputs[i].m.data()+ring.stride*j;
        if (m[0] != d) break;
        fmpz_poly_set_coeff_fmpz(i ? b : a,Ring::field(m,ring.position(1)),&inputs[i].c[j].v);
    }
    bool coprime = std::max(fmpz_poly_degree(a),fmpz_poly_degree(b)) == static_cast<slong>(d);
    if (coprime) { fmpz_poly_gcd(g,a,b); coprime = fmpz_poly_degree(g)==0; }
    fmpz_poly_clear(a); fmpz_poly_clear(b); fmpz_poly_clear(g);
    if (!coprime) return std::nullopt;
    return echelon_products(ring,products,d-1,d*d*n*(n+1)/2);
}
} // namespace atlas::ideal_powers
