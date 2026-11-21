// Trager factorization of K[x], where K=QQ[theta]/(m). Polynomials are
// stored in QQ[x,theta] lex order, normalized modulo the marked relation.
#include "branch_order.hpp"
#include "local_io.hpp"
#include "polynomial_factors.hpp"
#include "primary_decomposition.hpp"
#include "sparse_flint.hpp"
#include <numeric>
#include <sstream>

namespace atlas {
namespace {
using local::Basis;
using local::Polynomial;
using local::Powers;
using Factors = std::vector<std::pair<Polynomial, mpz_class>>;

mpz_class degree(const Polynomial& f) {
    mpz_class out = -1;
    for (const auto& [e,c] : f) {
        mpz_class total = std::accumulate(e.begin(), e.end()-1, mpz_class(0));
        if (total > out) out = total;
    }
    return out;
}

struct FieldRing {
    size_t width; // ordinary coordinates followed by theta
    local::SparseLexRing ring;
    Polynomial modulus;
    PolyValue relation;
    Polynomial one;

    FieldRing(size_t n, Polynomial m)
        : width(n), ring(n), modulus(std::move(m)), relation(ring.ctx), one{{Powers(n),1}} {
        if (modulus.empty() || degree(modulus) != 0 || modulus.rbegin()->first.back() < 1)
            throw std::invalid_argument("Expected a univariate coefficient-field relation.");
        ring.write(relation.p,modulus);
    }

    Polynomial monic(const Polynomial& f) {
        if (f.empty()) return {};
        // The reduced QQ basis of (m,f) contains the unique monic K-polynomial
        // and m. A nonzero constant of K gives the unit ideal instead.
        auto g = primary::reduced({modulus,f},width,true);
        return g.front() == modulus ? Polynomial{} : g.front();
    }

    std::pair<Polynomial, Polynomial> division(const Polynomial& f, const Polynomial& g) {
        // g is monic over K. Its head is theta-free, so (g,m) is a lex
        // basis with coprime heads. Reduce the quotient modulo m as well.
        PolyValue numerator(ring.ctx), divisor(ring.ctx), quotient(ring.ctx), discarded(ring.ctx), rem(ring.ctx);
        ring.write(numerator.p,f);
        ring.write(divisor.p,g);
        fmpq_mpoly_struct* divisors[] = {divisor.p,relation.p};
        fmpq_mpoly_struct* quotients[] = {quotient.p,discarded.p};
        fmpq_mpoly_divrem_ideal(quotients,rem.p,numerator.p,divisors,2,ring.ctx);
        auto remainder = ring.read(rem.p);
        fmpq_mpoly_divrem(discarded.p,rem.p,quotient.p,relation.p,ring.ctx);
        return {ring.read(rem.p), std::move(remainder)};
    }

    Polynomial quotient(const Polynomial& f, const Polynomial& g) {
        auto [q,r] = division(f,g);
        if (!r.empty()) throw std::logic_error("Inexact coefficient-field polynomial division.");
        return q;
    }

    Polynomial gcd(const Polynomial& f, const Polynomial& g) {
        if (g.empty() || f == g) return f;
        if (f.empty()) return g;
        if (f == one || g == one) return one;
        if (division(f,g).second.empty()) return g;
        if (division(g,f).second.empty()) return f;
        if (width == 2) return primary::reduced({f,g,modulus},width,true).front();
        // (f) intersect (g) in K[x] is principal with generator lcm(f,g).
        // Eliminate t from (t*f, (1-t)*g, m); the surviving nonconstant
        // x-polynomial is this monic lcm. Then f/(lcm/g) is the gcd.
        Basis rows(3);
        for (size_t j = 0; j < 3; ++j) {
            for (const auto& [e,c] : *(j == 0 ? &f : j == 1 ? &g : &modulus)) {
                Powers p(e);
                p.insert(p.begin(), j == 2 ? 0 : 1);
                rows[j].emplace(p,c);
                if (j == 1) { p[0] = 0; rows[j].emplace(std::move(p),-c); }
            }
        }
        for (const auto& row : primary::reduced(rows,width+1,true)) {
            if (row.rbegin()->first.front() != 0) continue;
            Polynomial projected;
            for (const auto& [e,c] : row) projected.emplace(Powers(e.begin()+1,e.end()),c);
            if (degree(projected) > 0) return quotient(f,quotient(projected,g));
        }
        throw std::logic_error("Missing coefficient-field least common multiple.");
    }

    Polynomial translate(const Polynomial& f, const mpz_class& a, bool reverse,
                         std::optional<size_t> axis = std::nullopt) {
        if (!a) return f;
        PolyValue original(ring.ctx), shifted(ring.ctx), term(ring.ctx), q(ring.ctx), rem(ring.ctx);
        ring.write(original.p,f);
        std::vector<PolyValue> coordinates;
        std::vector<fmpq_mpoly_struct*> images;
        Z scale;
        mpz_class power = 1;
        for (size_t i = 0; i < width; ++i) {
            auto& image = coordinates.emplace_back(ring.ctx);
            fmpq_mpoly_gen(image.p,i,ring.ctx);
            if (i+1 == width || (axis && i != *axis)) continue;
            power *= a;
            fmpz_set_mpz(&scale.v, power.get_mpz_t());
            if (reverse) fmpz_neg(&scale.v,&scale.v);
            fmpq_mpoly_gen(term.p,width-1,ring.ctx);
            fmpq_mpoly_scalar_mul_fmpz(term.p,term.p,&scale.v,ring.ctx);
            fmpq_mpoly_add(image.p,image.p,term.p,ring.ctx);
        }
        for (auto& image : coordinates) images.push_back(image.p);
        if (!fmpq_mpoly_compose_fmpq_mpoly(shifted.p,original.p,images.data(),ring.ctx,ring.ctx))
            throw std::runtime_error("Unable to translate the coefficient-field polynomial.");
        fmpq_mpoly_divrem(q.p,rem.p,shifted.p,relation.p,ring.ctx);
        return ring.read(rem.p);
    }

    Factors factor(Polynomial f, std::optional<size_t> section_axis = std::nullopt) {
        Factors factors;
        if (f.empty() || degree(f) == 0) return factors;
        // Remove coordinate factors before translation, including enormous
        // multiplicities on sparse monomials.
        auto minima = f.begin()->first;
        minima.back() = 0;
        for (const auto& [e,c] : f)
            for (size_t j = 0; j+1 < width; ++j) minima[j] = std::min(minima[j],e[j]);
        for (size_t j = 0; j+1 < width; ++j) if (minima[j]) {
            Powers p(width);
            p[j] = 1;
            factors.emplace_back(Polynomial{{p,1}},minima[j]);
        }
        if (!factors.empty()) {
            Polynomial stripped;
            for (const auto& [e,c] : f) {
                Powers p(e);
                for (size_t j = 0; j+1 < width; ++j) p[j] -= minima[j];
                stripped.emplace(std::move(p),c);
            }
            f = std::move(stripped);
        }
        f = monic(f);
        if (f.empty() || f == one) return factors;
        if (degree(f) == 1) { factors.emplace_back(std::move(f),1); return factors; }

        PolyValue original(ring.ctx), derivative(ring.ctx);
        ring.write(original.p,f);
        Polynomial common = section_axis ? one : f;
        for (size_t i = 0; i+1 < width && common != one; ++i) {
            fmpq_mpoly_derivative(derivative.p,original.p,i,ring.ctx);
            if (!fmpq_mpoly_is_zero(derivative.p,ring.ctx))
                common = gcd(common,monic(ring.read(derivative.p)));
        }
        const auto squarefree = quotient(f,common);
        PolyValue shifted(ring.ctx), norm(ring.ctx);
        for (mpz_class a = 0;; ++a) {
            ring.write(shifted.p,translate(squarefree,a,false,section_axis));
            if (!fmpq_mpoly_resultant(norm.p,relation.p,shifted.p,width-1,ring.ctx))
                throw std::runtime_error("Unable to compute the number-field norm.");
            auto rational = primary::factors(ring.read(norm.p),width);
            // Norm = product of all conjugate translates. For squarefree F
            // it is squarefree precisely when the conjugates are disjoint.
            if (std::any_of(rational.begin(),rational.end(),[](const auto& p){ return p.second != 1; })) continue;
            for (const auto& [image,m] : rational) {
                auto prime = gcd(squarefree,monic(translate(image,a,true,section_axis)));
                if (prime == one || prime.empty()) throw std::logic_error("A norm factor has no marked-field branch.");
                if (section_axis) {
                    factors.emplace_back(std::move(prime),1);
                    continue;
                }
                PolyValue divisor(ring.ctx);
                ring.write(divisor.p,prime);
                const auto exponent = factor_multiplicity(original.p,divisor.p,width-1,
                                                           {divisor.p,relation.p},ring.ctx);
                factors.emplace_back(std::move(prime),mpz_class(exponent,10));
            }
            return factors;
        }
    }
};

// Rank the descending power-basis coefficient lists. Their ranks preserve
// the coefficient comparison used by the archived recursive polynomial order.
std::vector<size_t> field_order(const Factors& factors, size_t width) {
    using Coefficients = std::vector<mpq_class>;
    std::vector<std::map<Powers,Coefficients>> grouped(factors.size());
    std::map<Coefficients,size_t> ranks;
    for (size_t i = 0; i < factors.size(); ++i) {
        for (const auto& [e,c] : factors[i].first) {
            auto& coefficients = grouped[i][Powers(e.begin(),e.end()-1)];
            const size_t k = e.back().get_ui();
            coefficients.resize(k+1);
            coefficients[k] = c;
        }
        for (auto& [p,coefficients] : grouped[i]) {
            std::reverse(coefficients.begin(),coefficients.end());
            ranks.emplace(coefficients,0);
        }
    }
    size_t next = 0;
    for (auto& [c,k] : ranks) k = ++next;
    Factors ranked;
    for (size_t i = 0; i < factors.size(); ++i) {
        Polynomial p;
        for (const auto& [e,c] : grouped[i]) p.emplace(e,ranks.at(c));
        ranked.emplace_back(std::move(p),factors[i].second);
    }
    return archived_branch_order(ranked,width-1);
}
} // namespace

std::vector<std::pair<local::Polynomial, mpz_class>> marked_factors(
        const local::Polynomial& f, const local::Polynomial& modulus, size_t width) {
    return FieldRing(width,modulus).factor(f);
}

std::vector<std::pair<local::Polynomial, mpz_class>> marked_section_factors(
        const local::Polynomial& f, const local::Polynomial& modulus, size_t width, size_t axis) {
    return FieldRing(width,modulus).factor(f,axis);
}

local::Polynomial marked_gcd(const local::Basis& rows, const local::Polynomial& modulus, size_t width) {
    FieldRing ring(width,modulus);
    Polynomial common;
    for (const auto& f : rows) {
        if (f.empty()) continue;
        common=ring.gcd(common,ring.monic(f));
        if (common==ring.one) break;
    }
    return common;
}

local::Basis marked_cofactors(const local::Polynomial& a,const local::Polynomial& b,
                              const local::Polynomial& modulus,size_t width) {
    FieldRing ring(width,modulus);
    auto common=ring.gcd(ring.monic(a),ring.monic(b));
    return common==ring.one ? local::Basis{a,b} : local::Basis{ring.quotient(a,common),ring.quotient(b,common)};
}

std::string field_factorization(std::istream& input) {
    std::string size, equation, minimal;
    if (!std::getline(input,size) || !std::getline(input,minimal) || !std::getline(input,equation))
        throw std::invalid_argument("Expected a number-field ring, modulus and polynomial.");
    const size_t width = std::stoul(size);
    if (width < 2) throw std::invalid_argument("Expected a marked polynomial ring.");
    auto factors = marked_factors(local::decode_row(equation,width),local::decode_row(minimal,width),width);
    std::ostringstream lines;
    for (size_t i : field_order(factors,width))
        lines << factors[i].second << '\n' << local::encode_row(factors[i].first) << '\n';
    return lines.str();
}
}

extern "C" int atlas_field_factorization_abi() noexcept { return 1; }
