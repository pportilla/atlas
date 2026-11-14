// Membership tests on reduced lex bases. Batch reductions of wide sparse
// monomials without expanding the corresponding (potentially enormous) powers.
#include "localization.hpp"
#include "primary_decomposition.hpp"
#include "local_point_basis.hpp"
#include <flint/fmpz.h>
#include <flint/fmpz_poly.h>
#include <algorithm>

namespace atlas::localization {
namespace {
using local::Powers;

bool divisible(const Powers& e, const Powers& divisor) {
    return std::equal(divisor.begin(),divisor.end(),e.begin(),std::less_equal<>());
}

// The coefficient of a reduced monomial is stored as a polynomial in the
// (nonzero) rational multipliers from the binomial divisors, evaluated only
// after all terms reducing to the same monomial have been combined.
bool nonzero_coefficient(const Polynomial& coefficient, const std::vector<mpq_class>& multipliers) {
    if (coefficient.empty()) return false;
    if (coefficient.size() == 1) return true;
    return local::nonzero_at_point(coefficient,multipliers);
}

bool binomial_member(const Polynomial& f, const Basis& basis) {
    std::vector<mpq_class> multipliers(basis.size(),1);
    for (size_t i = 0; i < basis.size(); ++i) if (basis[i].size() > 1)
        multipliers[i] = -basis[i].begin()->second / basis[i].rbegin()->second;
    std::map<Powers,Polynomial> remainders;
    for (const auto& [e,c] : f) {
        auto monomial = e;
        Powers scalar_powers(basis.size());
        mpq_class scalar = c;
        for (;;) {
            auto row = std::find_if(basis.begin(),basis.end(),[&](const auto& g) {
                return divisible(monomial,g.rbegin()->first);
            });
            if (row == basis.end()) {
                auto& group = remainders[monomial];
                auto coefficient = group.try_emplace(std::move(scalar_powers),0).first;
                coefficient->second += scalar;
                if (coefficient->second == 0) group.erase(coefficient);
                break;
            }
            if (row->size() == 1) break;
            const auto& head = row->rbegin()->first;
            const auto& tail = row->begin()->first;
            // The rewrite removes head and adds tail. Its lex decrease means
            // some coordinate decreases, bounding the number of repetitions.
            std::optional<mpz_class> copies;
            for (size_t j = 0; j < head.size(); ++j) if (head[j] > tail[j]) {
                mpz_class bound = 1 + (monomial[j]-head[j])/(head[j]-tail[j]);
                if (!copies || bound < *copies) copies = std::move(bound);
            }
            for (size_t j = 0; j < head.size(); ++j) monomial[j] += *copies * (tail[j]-head[j]);
            const auto i = row-basis.begin();
            if (multipliers[i] == -1) {
                if (mpz_odd_p(copies->get_mpz_t())) scalar = -scalar;
            } else if (multipliers[i] != 1) scalar_powers[i] += *copies;
        }
    }
    return std::none_of(remainders.begin(),remainders.end(),[&](const auto& term) {
        return nonzero_coefficient(term.second,multipliers);
    });
}

// If Phi_k(x_i) belongs to the ideal then x_i^k=1 in its quotient. Dense
// FLINT cyclotomic recognition is only attempted on small univariate basis
// rows. The input powers reduced modulo the certified period remain unbounded.
Polynomial reduce_cyclic_powers(const Polynomial& f, const Basis& basis) {
    const size_t n = f.begin()->first.size();
    std::vector<unsigned long> periods(n);
    for (const auto& g : basis) {
        const auto& head = g.rbegin()->first;
        const auto first = std::find_if(head.begin(),head.end(),[](const auto& e) { return e != 0; });
        if (first == head.end() || *first > 4096) continue;
        const size_t i = first-head.begin();
        if (!std::all_of(g.begin(),g.end(),[&](const auto& term) {
                const auto& e = term.first;
                return term.second.get_den() == 1 && std::count(e.begin(),e.end(),0) + (e[i] != 0) == static_cast<long>(n);
            })) continue;
        fmpz_poly_t relation;
        fmpz_poly_init(relation);
        fmpz_t coefficient;
        fmpz_init(coefficient);
        for (const auto& [e,c] : g) {
            fmpz_set_mpz(coefficient,mpq_numref(c.get_mpq_t()));
            fmpz_poly_set_coeff_fmpz(relation,e[i].get_ui(),coefficient);
        }
        fmpz_clear(coefficient);
        periods[i] = fmpz_poly_is_cyclotomic(relation);
        fmpz_poly_clear(relation);
    }
    if (std::all_of(periods.begin(),periods.end(),[](auto p) { return p == 0; })) return f;
    Polynomial reduced;
    for (const auto& [powers,c] : f) {
        auto e = powers;
        for (size_t i = 0; i < n; ++i) if (periods[i]) e[i] %= periods[i];
        auto term = reduced.try_emplace(std::move(e),0).first;
        term->second += c;
        if (term->second == 0) reduced.erase(term);
    }
    return reduced;
}
} // namespace

bool member_lex(const Polynomial& f, const Basis& g) {
    // Ordinary sparse reduction is cheaper on small inputs. The batch path
    // never converts exponents to machine integers.
    if (!f.empty() && !g.empty() && std::any_of(f.begin(),f.end(),[](const auto& term) {
            return std::any_of(term.first.begin(),term.first.end(),[](const auto& e) { return e > 64; });
        })) {
        if (std::all_of(g.begin(),g.end(),[](const auto& row) { return row.size() <= 2; }))
            return binomial_member(f,g);
        return primary::member_lex(reduce_cyclic_powers(f,g),g);
    }
    return primary::member_lex(f,g);
}
} // namespace atlas::localization
