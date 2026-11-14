// Finite birational extensions using weighted content of a hypersurface.
#include "normalization.hpp"
#include "primary_decomposition.hpp"
#include "sparse_flint.hpp"
#include <algorithm>
#include <map>

namespace atlas::normalization {
namespace {
using local::Powers;

// Coefficients are indexed by distance from the leading power, so that a_i
// in Y^m + sum_i a_i Y^(m-i) must be divisible by d^i.
using Coefficients = std::map<mpz_class, Polynomial>;

Polynomial weighted_content(const Coefficients& coefficients, size_t width, local::SparseLexRing& ring,
                            Polynomial& linear_factor) {
    if (coefficients.empty()) return {};
    auto shortest = std::min_element(coefficients.begin(),coefficients.end(),
        [](const auto& a, const auto& b) { return a.second.size() < b.second.size(); });
    std::vector<std::pair<Polynomial, mpz_class>> common;
    for (const auto& [p,e] : primary::factors(shortest->second,width)) {
        mpz_class limit = e / shortest->first;
        if (limit > 0) common.emplace_back(p,std::move(limit));
    }
    if (common.empty()) return {};

    // Test v_p(a_i) >= k*i by exact polynomial division. Bisection also
    // handles sparse inputs with exponents greater than a machine word.
    PolyValue coefficient(ring.ctx), divisor(ring.ctx), quotient(ring.ctx), factor(ring.ctx);
    Polynomial result{{Powers(width),1}};
    bool nonconstant = false;
    for (const auto& [p, maximum] : common) {
        mpz_class upper = maximum;
        ring.write(factor.p,p);
        Z factor_degree;
        fmpq_mpoly_total_degree_fmpz(&factor_degree.v,factor.p,ring.ctx);
        mpz_class degree_p;
        fmpz_get_mpz(degree_p.get_mpz_t(),&factor_degree.v);
        for (const auto& row : coefficients) {
            if (&row == &*shortest) continue;
            ring.write(coefficient.p,row.second);
            // Do not expand a trial power larger than the dividend itself.
            Z coefficient_degree;
            fmpq_mpoly_total_degree_fmpz(&coefficient_degree.v,coefficient.p,ring.ctx);
            mpz_class degree_bound;
            fmpz_get_mpz(degree_bound.get_mpz_t(),&coefficient_degree.v);
            degree_bound /= degree_p * row.first;
            upper = std::min(upper,degree_bound);
            mpz_class lower = 0;
            while (lower < upper) {
                mpz_class trial = (lower + upper + 1) / 2;
                ring.write(divisor.p,primary::power(p,trial*row.first,width));
                if (fmpq_mpoly_divides(quotient.p,coefficient.p,divisor.p,ring.ctx)) lower = trial;
                else upper = trial - 1;
            }
            if (upper == 0) break;
        }
        if (upper > 0) {
            if (p.size() > 1 && degree_p == 1 && linear_factor.empty()) linear_factor = p;
            result = primary::multiply(result,primary::power(p,upper,width));
            nonconstant = true;
        }
    }
    return nonconstant ? result : Polynomial{};
}

Polynomial variable(size_t width, size_t i, const mpz_class& exponent = 1) {
    Powers e(width);
    e[i] = exponent;
    return {{std::move(e), 1}};
}
void add(Polynomial& p, const Polynomial& q, const mpq_class& scale) {
    for (const auto& [e,c] : q) {
        auto it = p.try_emplace(e,0).first;
        it->second += scale*c;
        if (it->second == 0) p.erase(it);
    }
}

// A nonconstant leading coefficient a need not be a unit. Keep Y and append
// V=aY/d. Multiplying aY^m+sum c_i Y^(m-i)=0 by a^(m-1)/d^m gives a monic
// equation for V with coefficients a^(i-1)*c_i/d^i. The saturated graph is
// therefore an integral subalgebra of the fraction field of the original ring.
Presentation nonmonic_chart(const Polynomial& f, const Polynomial& a, Polynomial d,
                            const Polynomial& linear, size_t axis, const std::vector<std::string>& names, bool lex) {
    const size_t n = names.size();
    const Polynomial one{{Powers(n),1}};
    Presentation out;
    out.names = names;
    for (size_t j = 0; j < n; ++j) {
        out.images.push_back(variable(n,j));
        out.fractions.push_back({variable(n,j),one});
    }
    auto numerator = primary::multiply(a,variable(n,axis));
    out.fractions.push_back({numerator,d});
    auto relation = f;
    // If d contains an affine form L, use L as a base coordinate. In
    // particular, high powers of a translated divisor then remain sparse.
    if (!linear.empty()) {
        for (size_t j=n; j-- > 0;) {
            auto v = variable(n,j);
            const auto it = linear.find(v.begin()->first);
            if (it == linear.end()) continue;
            out.fractions[j].numerator = linear;
            auto inverse = v;
            add(inverse,linear,-1);
            add(inverse,v,it->second);
            for (auto& term : inverse) term.second /= it->second;
            out.images[j] = inverse;

            // substitute removes the source slot; insert an unused slot at j
            // so the image may use the new base coordinate at that position.
            std::map<mpz_class,Polynomial> cache;
            auto replace = [&](const Polynomial& g) {
                Polynomial lifted;
                for (const auto& [e,c] : g) {
                    auto term=e;
                    term.insert(term.begin()+j,0);
                    lifted.emplace(std::move(term),c);
                }
                return primary::substitute(lifted,j+1,inverse,n,cache);
            };
            relation = replace(relation);
            numerator = replace(numerator);
            d = replace(d);
            break;
        }
    }
    size_t suffix=1;
    std::string new_name;
    do { new_name = "n"+std::to_string(suffix++); }
    while (std::find(names.begin(),names.end(),new_name) != names.end());
    out.names.push_back(new_name);
    auto extend = [](const Polynomial& g) {
        Polynomial lifted;
        for (const auto& [e,c] : g) {
            auto term=e;
            term.emplace_back(0);
            lifted.emplace(std::move(term),c);
        }
        return lifted;
    };
    d = extend(d);
    auto graph = primary::multiply(d,variable(n+1,n));
    add(graph,extend(numerator),-1);
    out.relations = primary::saturate_lex({extend(relation),graph},d,n+1);
    if (!lex) out.relations = primary::reduced(out.relations,n+1,false);
    for (auto& g : out.images) g = extend(g);
    return out;
}
}

std::optional<Presentation> divide_power_content(const Basis& prime,
                                                const std::vector<std::string>& names, bool lex) {
    if (prime.size() != 1 || names.size() < 2) return std::nullopt;
    const auto& f = prime.front();
    const auto n = names.size();
    local::SparseLexRing ring(n);
    for (size_t axis = 0; axis < n; ++axis) {
        mpz_class m = 0;
        for (const auto& [e,c] : f) m = std::max(m,e[axis]);
        if (m < 2) continue;
        const auto lead = variable(n,axis,m);
        Coefficients coefficients;
        for (const auto& [e,c] : f) {
            auto term = e;
            term[axis] = 0;
            coefficients[m-e[axis]].emplace(std::move(term),c);
        }
        auto lc = std::move(coefficients.at(0));
        coefficients.erase(0);
        Polynomial linear_factor;
        auto d = weighted_content(coefficients,n,ring,linear_factor);
        if (d.empty()) continue;
        if (lc.size() != 1 || lc.begin()->first != Powers(n))
            return nonmonic_chart(f,lc,std::move(d),linear_factor,axis,names,lex);
        for (auto& [i,a] : coefficients) for (auto& term : a) term.second /= lc.begin()->second;

        // The fraction T=Y/d satisfies this monic relation and Y=d*T,
        // yielding an integral extension in exactly the same fraction field.
        Polynomial relation = lead;
        PolyValue numerator(ring.ctx), divisor(ring.ctx), quotient(ring.ctx);
        for (const auto& [i,a] : coefficients) {
            ring.write(numerator.p,a);
            ring.write(divisor.p,primary::power(d,i,n));
            if (!fmpq_mpoly_divides(quotient.p,numerator.p,divisor.p,ring.ctx))
                throw std::logic_error("Inexact weighted content division in normalization.");
            for (const auto& [e,c] : ring.read(quotient.p)) {
                auto term = e;
                term[axis] = m-i;
                relation.emplace(std::move(term),c);
            }
        }
        Presentation out{names,primary::reduced({relation},n,lex),{},{}};
        size_t k = 1;
        do { out.names[axis] = "n" + std::to_string(k++); }
        while (std::find(names.begin(),names.end(),out.names[axis]) != names.end());
        const Polynomial one{{Powers(n),1}};
        for (size_t j = 0; j < n; ++j) {
            auto x = variable(n,j);
            out.images.push_back(j == axis ? primary::multiply(x,d) : x);
            out.fractions.push_back({std::move(x),j == axis ? d : one});
        }
        return out;
    }
    return std::nullopt;
}
} // namespace atlas::normalization
