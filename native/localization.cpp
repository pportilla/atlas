// Extend an ideal to QQ[x]_P, then contract to the polynomial ring.
#include "localization.hpp"
#include "primary_decomposition.hpp"
#include "local_io.hpp"
#include "local_point_basis.hpp"
#include <algorithm>
#include <sstream>

namespace atlas::localization {
namespace {
using local::Powers;

Basis unit_ideal(size_t width) { return {Polynomial{{Powers(width),1}}}; }

Basis prime_basis(const Basis& prime, size_t width) {
    if (prime.empty()) return {};
    if (prime.size() != 1) return primary::reduced(prime,width,true);
    if (prime.front().empty()) return {};
    auto result = prime;
    const mpq_class lc = result.front().rbegin()->second;
    if (lc != 1) for (auto& [e,c] : result.front()) c /= lc;
    return result;
}

// A primitive Laurent exponent difference extends to a unimodular basis.
// The corresponding torus change of variables sends the binomial to a
// unit times t-c. Coprime monomials exclude coordinate factors in QQ[x].
bool primitive_binomial(const Polynomial& g) {
    if (g.size() != 2) return false;
    const auto& a = g.begin()->first;
    const auto& b = g.rbegin()->first;
    mpz_class divisor = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != 0 && b[i] != 0) return false;
        divisor = gcd(divisor,mpz_class(a[i]-b[i]));
    }
    return divisor == 1;
}

// A reduced lex basis whose leading monomials are distinct coordinates
// presents a polynomial ring: each such coordinate is a polynomial in the
// free coordinates. Also recognize a hypersurface over this graph when its
// one residual generator is irreducible in that polynomial ring.
void validate_prime(const Basis& prime, size_t width) {
    if (prime.empty()) return;
    const Polynomial* residual = nullptr;
    bool graph_with_hypersurface = true;
    for (const auto& g : prime) {
        const auto& head = g.rbegin()->first;
        const bool coordinate = std::count(head.begin(),head.end(),0)+1 == static_cast<long>(width)
                             && std::find(head.begin(),head.end(),1) != head.end();
        if (coordinate) continue;
        if (residual) { graph_with_hypersurface = false; break; }
        residual = &g;
    }
    if (graph_with_hypersurface) {
        if (!residual || primitive_binomial(*residual)) return;
        if (*residual != unit_ideal(width).front()) {
            const auto fs = primary::factors(*residual,width);
            if (fs.size() == 1 && fs.front().second == 1) return;
        }
        throw InvalidInput("Localization requires a proper prime ideal over QQ.");
    }
    const auto components = primary::decompose(prime,width,true);
    if (components.size() != 1 || components.front().prime != components.front().primary)
        throw InvalidInput("Localization requires a proper prime ideal over QQ.");
}

bool finite_leading_terms(const Basis& g, size_t width) {
    std::vector<bool> bounded(width);
    for (const auto& f : g) if (!f.empty()) {
        const auto& e = f.rbegin()->first;
        const auto first = std::find_if(e.begin(),e.end(),[](const auto& v) { return v != 0; });
        if (first != e.end() && std::all_of(first+1,e.end(),[](const auto& v) { return v == 0; }))
            bounded[first-e.begin()] = true;
    }
    return std::all_of(bounded.begin(),bounded.end(),[](bool v) { return v; });
}

// A univariate relation h_i(x_i) vanishing at a rational point has the
// same local principal ideal as (x_i-a_i)^m_i, where m_i is its order
// of vanishing. One such relation on every axis isolates the point: the
// enlarged ideal contains a power of its maximal ideal and is primary or
// the unit ideal. This avoids iterating powers of P through its Loewy length.
std::optional<Basis> rational_fiber(const Basis& generators, const Basis& prime, size_t width) {
    if (prime.size() != width || width < 2 ||
        std::all_of(generators.begin(),generators.end(),[](const auto& f){ return f.size() <= 1; }) ||
        !finite_leading_terms(generators,width))
        return std::nullopt;
    const Powers zero(width);
    std::vector<const Polynomial*> coordinates(width);
    for (const auto& p : prime) {
        const auto& e = p.rbegin()->first;
        const auto one = std::find(e.begin(),e.end(),1);
        if (one == e.end() || std::count(e.begin(),e.end(),0)+1 != static_cast<long>(width) ||
            p.size() > 2 || (p.size() == 2 && p.begin()->first != zero)) return std::nullopt;
        coordinates[one-e.begin()] = &p;
    }
    const auto basis = primary::reduced(generators,width,true);
    if (basis.empty() || basis == unit_ideal(width)) return basis;
    Basis enlarged = basis;
    size_t bounded = 0;
    for (const auto& f : basis) {
        const auto& head = f.rbegin()->first;
        const auto axis = std::find_if(head.begin(),head.end(),[](const auto& e){ return e != 0; });
        if (axis == head.end()) continue;
        const size_t i = axis-head.begin();
        if (!std::all_of(f.begin(),f.end(),[&](const auto& term) {
                const auto& e = term.first;
                return std::count(e.begin(),e.end(),0) + (e[i] != 0) == static_cast<long>(width);
            })) continue;
        ++bounded; // Minimal leading monomials imply at most one per axis.
        const auto& coordinate = *coordinates[i];
        mpz_class multiplicity = 0;
        if (coordinate.size() == 1) multiplicity = f.begin()->first[i];
        else {
            // At a nonzero coordinate the first nonvanishing derivative is
            // reached in at most the number of terms (Vandermonde). Work in
            // one variable even if the exponent cannot fit a machine word.
            std::vector<mpq_class> point{-coordinate.begin()->second};
            Polynomial derivative;
            for (const auto& [e,c] : f) derivative.emplace(Powers{e[i]},c);
            while (!local::nonzero_at_point(derivative,point)) {
                ++multiplicity;
                Polynomial next;
                for (const auto& [e,c] : derivative)
                    if (e.front() != 0) next.emplace(Powers{e.front()-1},c*e.front());
                derivative = std::move(next);
            }
        }
        if (multiplicity == 0) return unit_ideal(width);
        if (multiplicity < head[i]) enlarged.push_back(primary::power(coordinate,multiplicity,width));
    }
    if (bounded != width) return std::nullopt;
    if (enlarged.size() == basis.size()) return basis;
    return primary::reduced(enlarged,width,true);
}

// In an Artinian quotient R/I the powers of P descend to a stationary
// ideal. After localization the stationary ideal M satisfies M=P M, hence
// is zero by Nakayama. Since I+P^m is P-primary for every m>=1, at
// stationarity it is already the required contracted ideal.
std::optional<Basis> finite_contraction(const Basis& generators, const Basis& prime, size_t width) {
    if (!finite_leading_terms(prime,width) || !finite_leading_terms(generators,width)) return std::nullopt;
    auto current = prime;  // Every generator is in P on this path.
    for (;;) {
        Basis next = generators;
        for (const auto& p : prime) for (const auto& h : current) {
            auto multiple = primary::multiply(p,h);
            if (std::find(next.begin(),next.end(),multiple) == next.end())
                next.push_back(std::move(multiple));
        }
        next = primary::reduced(next,width,true);
        if (next == current) return current;
        current = std::move(next);
    }
}

Json::Value polynomials(const Basis& basis, const std::vector<std::string>& names) {
    Json::Value list(Json::arrayValue);
    for (const auto& f : basis) list.append(local::format_polynomial(f,names));
    return list;
}

Json::Value ordered_polynomials(const Basis& basis, const std::vector<std::string>& names, bool lex) {
    if (lex || basis.empty()) return polynomials(basis,names);
    if (basis.size() == 1) {
        const auto& g = basis.front();
        const auto lc = g.at(local::global_head(g,false));
        if (lc == 1) return polynomials(basis,names);
        auto result = basis;
        for (auto& [e,c] : result.front()) c /= lc;
        return polynomials(result,names);
    }
    // If all leading monomials stay put, the same reductions certify
    // Buchberger's criterion and reducedness in the requested order.
    if (std::all_of(basis.begin(),basis.end(),[](const Polynomial& g) {
            return local::global_head(g,false) == g.rbegin()->first;
        })) {
        auto result = basis;
        std::sort(result.begin(),result.end(),[](const Polynomial& a, const Polynomial& b) {
            return local::global_less(b.rbegin()->first,a.rbegin()->first,false);
        });
        return polynomials(result,names);
    }
    return polynomials(primary::reduced(basis,names.size(),false),names);
}

// For s outside P, f/s belongs to I R_P exactly when f belongs to its
// contraction. A unit in the localized polynomial ring has numerator outside P.
Json::Value fraction_record(const Fraction& f, const Basis& prime, const Basis& contraction,
                            const std::vector<std::string>& names) {
    Json::Value value;
    value["numerator"] = local::format_polynomial(f.numerator,names);
    value["denominator"] = local::format_polynomial(f.denominator,names);
    value["member"] = member_lex(f.numerator,contraction);
    value["unit"] = !member_lex(f.numerator,prime);
    return value;
}
} // namespace

Basis contract(const Basis& generators, const Basis& prime_lex, size_t width) {
    if (auto fiber = rational_fiber(generators,prime_lex,width)) return *fiber;
    if (auto shortcut = special_contraction(generators,prime_lex,width)) return *shortcut;
    if (auto finite = finite_contraction(generators,prime_lex,width)) return *finite;
    // Elimination can reveal a monomial times a local unit even when none
    // of the original generators has a cancellable factor. Such a monomial
    // is strictly smaller than the leading monomial of its reduced basis
    // row, and hence is not yet in the ideal. Adjoining it preserves the
    // localization and strictly increases the ideal, so iteration terminates.
    auto basis = primary::reduced(generators,width,true);
    Basis enlarged = generators;
    for (const auto& g : basis) if (g.size() > 1) {
        Powers common = g.begin()->first;
        for (const auto& [powers,c] : g)
            for (size_t i = 0; i < width; ++i) common[i] = std::min(common[i],powers[i]);
        if (std::all_of(common.begin(),common.end(),[](const auto& e) { return e == 0; })) continue;
        Polynomial quotient;
        for (const auto& [powers,c] : g) {
            auto e = powers;
            for (size_t i = 0; i < width; ++i) e[i] -= common[i];
            quotient.emplace(std::move(e),c);
        }
        if (!member_lex(quotient,prime_lex)) enlarged.push_back({{std::move(common),1}});
    }
    if (enlarged.size() != generators.size()) return contract(enlarged,prime_lex,width);
    if (auto finite = finite_contraction(basis,prime_lex,width)) return *finite;
    // The same argument applies to polynomial unit factors. No leading
    // monomial of a reduced basis divides a proper divisor of another.
    if (basis != generators)
        if (auto simpler = contract_factored(std::move(basis),prime_lex,width)) return *simpler;
    auto result = unit_ideal(width);
    // In a primary ideal Q an element outside sqrt(Q) is regular modulo Q.
    // Thus Q survives unchanged iff sqrt(Q) is contained in P. Otherwise an
    // element of sqrt(Q) outside P becomes a unit and Q extends to (1).
    for (const auto& c : primary::decompose(generators,width,true))
        if (std::all_of(c.prime.begin(),c.prime.end(),[&](const auto& p) { return member_lex(p,prime_lex); }))
            result = primary::intersection_lex(result,c.primary,width);
    return result;
}

Json::Value compute(const Basis& generators, const Basis& prime, const std::vector<std::string>& names,
                    bool lex, const std::optional<Fraction>& fraction) {
    const auto width = names.size();
    const auto prime_lex = prime_basis(prime,width);
    validate_prime(prime_lex,width);
    if (fraction && member_lex(fraction->denominator,prime_lex))
        throw InvalidInput("A denominator in the prime ideal is not invertible in this localization.");
    const auto contraction = contract(generators,prime_lex,width);
    Json::Value result;
    result["type"] = "localization";
    auto& ring = result["ring"];
    ring["field"] = "QQ";
    ring["order"] = lex ? "lex" : "grevlex";
    ring["variables"] = Json::Value(Json::arrayValue);
    for (const auto& name : names) ring["variables"].append(name);
    result["prime"] = ordered_polynomials(prime_lex,names,lex);
    result["generators"] = ordered_polynomials(contraction,names,lex);
    if (fraction) result["element"] = fraction_record(*fraction,prime_lex,contraction,names);
    return result;
}

std::string operation(std::istream& input) {
    std::string coordinates, order, counts, line;
    if (!std::getline(input,coordinates) || !std::getline(input,order) ||
        (order != "lex" && order != "grevlex") || !std::getline(input,counts))
        throw std::invalid_argument("Expected localization coordinates, monomial order and operand sizes.");
    std::istringstream fields(counts), variables(coordinates);
    std::ptrdiff_t ni, np;
    int has_fraction;
    std::string extra;
    if (!(fields >> ni >> np >> has_fraction) || fields >> extra || ni < 0 || np < 0 ||
        (has_fraction != 0 && has_fraction != 1)) throw std::invalid_argument("Invalid localization operand sizes.");
    std::vector<std::string> names;
    for (std::string v; variables >> v;) names.push_back(std::move(v));
    auto read = [&]() {
        if (!std::getline(input,line)) throw std::invalid_argument("Missing localization operand.");
        return local::decode_row(line,names.size());
    };
    Basis generators, prime;
    for (std::ptrdiff_t i = 0; i < ni; ++i) generators.push_back(read());
    for (std::ptrdiff_t i = 0; i < np; ++i) prime.push_back(read());
    std::optional<Fraction> fraction;
    if (has_fraction) fraction = Fraction{read(),read()};
    if (std::getline(input,line)) throw std::invalid_argument("Unexpected localization operand.");
    Json::Value out;
    try {
        out = compute(generators,prime,names,order == "lex",fraction);
    } catch (const InvalidInput& error) {
        out["error"]["code"] = "invalid_request";
        out["error"]["message"] = error.what();
    }
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    return Json::writeString(writer,out) + "\n";
}
} // namespace atlas::localization
