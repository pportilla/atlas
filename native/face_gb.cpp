// Decide whether a hypersurface in an algebraic torus has a singularity
// over Qbar. FLINT's primitive integer Buchberger routine computes over Q.
#include <flint/fmpz.h>
#include <flint/fmpz_mpoly.h>
#include <flint/fmpz_poly.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

int main() {
    int n, count;
    if (!(std::cin >> n >> count) || n<1 || n>8 || count<1 || count>4096) return 2;
    fmpz_mpoly_ctx_t ctx;
    fmpz_mpoly_ctx_init(ctx,n+1,ORD_DEGREVLEX);
    fmpz_mpoly_t f, temp;
    fmpz_mpoly_init(f,ctx);
    fmpz_mpoly_init(temp,ctx);
    std::vector<ulong> powers(n+1);
    fmpz_t c;
    fmpz_init(c);
    for (int k=0;k<count;++k) {
        std::string coefficient;
        if (!(std::cin >> coefficient) || coefficient.size()>8192 ||
            fmpz_set_str(c,coefficient.c_str(),10) || fmpz_is_zero(c)) return 2;
        for (int i=0;i<n;++i)
            if (!(std::cin >> powers[i]) || powers[i]>4096) return 2;
        fmpz_mpoly_push_term_fmpz_ui(f,c,powers.data(),ctx);
    }
    fmpz_mpoly_sort_terms(f,ctx);
    if (n==1) {
        fmpz_poly_t p, derivative, common;
        fmpz_poly_init(p); fmpz_poly_init(derivative); fmpz_poly_init(common);
        if (!fmpz_mpoly_get_fmpz_poly(p,f,0,ctx)) return 2;
        fmpz_poly_derivative(derivative,p);
        fmpz_poly_gcd(common,p,derivative);
        std::cout << (fmpz_poly_degree(common)<=0 ? "nondegenerate" : "degenerate")
                  << ' ' << fmpz_poly_degree(common) << '\n';
        fmpz_poly_clear(common); fmpz_poly_clear(derivative); fmpz_poly_clear(p);
    } else {
        fmpz_mpoly_vec_t generators, standard_basis;
        fmpz_mpoly_vec_init(generators,0,ctx);
        fmpz_mpoly_vec_init(standard_basis,0,ctx);
        fmpz_mpoly_vec_append(generators,f,ctx);
        for (int i=0; i<n; ++i) {
            fmpz_mpoly_derivative(temp,f,i,ctx);
            if (!fmpz_mpoly_is_zero(temp,ctx)) fmpz_mpoly_vec_append(generators,temp,ctx);
        }
        // The final parameter inverts every torus coordinate simultaneously.
        fmpz_mpoly_one(temp,ctx);
        std::fill(powers.begin(),powers.end(),1);
        fmpz_set_si(c,-1);
        fmpz_mpoly_set_coeff_fmpz_ui(temp,c,powers.data(),ctx);
        fmpz_mpoly_vec_append(generators,temp,ctx);
        int complete=fmpz_mpoly_buchberger_naive_with_limits(standard_basis,generators,256,4096,16384,ctx);
        bool unit=false;
        for (slong i=0; i<standard_basis->length; ++i) {
            const auto* g=fmpz_mpoly_vec_entry(standard_basis,i);
            if (fmpz_mpoly_is_fmpz(g,ctx) && !fmpz_mpoly_is_zero(g,ctx)) unit=true;
        }
        // Finding a nonzero scalar is conclusive even if a resource limit
        // stopped the remaining pairs. A proper ideal needs a completed basis.
        std::cout << (unit ? "nondegenerate" : complete ? "degenerate" : "undetermined")
                  << ' ' << standard_basis->length << '\n';
        fmpz_mpoly_vec_clear(standard_basis,ctx);
        fmpz_mpoly_vec_clear(generators,ctx);
    }
    fmpz_mpoly_clear(temp,ctx);
    fmpz_mpoly_clear(f,ctx);
    fmpz_clear(c);
    fmpz_mpoly_ctx_clear(ctx);
}
