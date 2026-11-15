// Ordinary powers and contractions at all minimal primes of a radical QQ ideal.
#include "ideal_powers.hpp"
#include "ideal_power_principal.hpp"
#include "local_io.hpp"
#include "primary_decomposition.hpp"
#include "resolution_tangents.hpp"
#include "sparse_flint.hpp"
#include <algorithm>
#include <sstream>

namespace atlas::ideal_powers {
namespace {
using local::Powers;
using local::Polynomial;

Basis unit(size_t width) { return {Polynomial{{Powers(width),1}}}; }

bool constant(const Polynomial& f) {
    return f.size() == 1 && std::all_of(f.begin()->first.begin(),f.begin()->first.end(),
                                        [](const auto& k) { return k == 0; });
}

void require_radical(bool verified) {
    if (!verified) throw InvalidInput("Symbolic powers require a prime or radical ideal over QQ.");
}

Polynomial principal_power(const Polynomial& f, size_t width, bool lex,
                          const mpz_class& exponent, bool symbolic) {
    auto singleton = PrincipalBasis(f,width,lex,exponent,symbolic).materialize();
    return std::move(singleton.front());
}

// For 0 < n and several generators, enumerate the weak compositions of n.
// Only one row per unordered product is needed. All exponents retain GMP width.
Basis power_products(const Basis& rows, size_t width, const mpz_class& exponent) {
    local::SparseLexRing ring(width);
    std::vector<PolyValue> generators, prefix, multiple;
    for (const auto& f : rows) {
        ring.write(generators.emplace_back(ring.ctx).p,f);
        prefix.emplace_back(ring.ctx);
        multiple.emplace_back(ring.ctx);
    }
    Basis products;
    PolyValue result(ring.ctx);
    auto build = [&](auto&& self, size_t i, const mpz_class& degree) -> void {
        if (degree == 0) { products.push_back(ring.read(prefix[i].p)); return; }
        if (i+1 == rows.size()) {
            Z k;
            fmpz_set_mpz(&k.v,degree.get_mpz_t());
            if (!fmpq_mpoly_pow_fmpz(multiple[i].p,generators[i].p,&k.v,ring.ctx))
                throw std::runtime_error("Unable to expand polynomial power.");
            fmpq_mpoly_mul(result.p,prefix[i].p,multiple[i].p,ring.ctx);
            products.push_back(ring.read(result.p));
            return;
        }
        fmpq_mpoly_set(multiple[i].p,prefix[i].p,ring.ctx);
        for (mpz_class remaining = degree;; --remaining) {
            fmpq_mpoly_set(prefix[i+1].p,multiple[i].p,ring.ctx);
            self(self,i+1,remaining);
            if (remaining == 0) break;
            fmpq_mpoly_mul(multiple[i].p,multiple[i].p,generators[i].p,ring.ctx);
        }
    };
    fmpq_mpoly_one(prefix[0].p,ring.ctx);
    build(build,0,exponent);
    return products;
}

// Every non-head term in these graph equations uses only free coordinates.
// In the power products every other term has lower total degree in the
// pivot variables; the products thus already form the reduced basis of I^n.
bool graph_coordinates(const Basis& rows, bool lex) {
    std::vector<size_t> pivots;
    for (const auto& f : rows) {
        const auto& head = local::global_head(f,lex);
        const auto axis = std::find(head.begin(),head.end(),1);
        if (axis == head.end() || std::count(head.begin(),head.end(),0)+1 != static_cast<long>(head.size())
            || f.at(head) != 1) return false;
        pivots.push_back(axis-head.begin());
    }
    auto sorted = pivots;
    std::sort(sorted.begin(),sorted.end());
    if (std::adjacent_find(sorted.begin(),sorted.end()) != sorted.end()) return false;
    for (size_t i=0; i<rows.size(); ++i) for (const auto& [e,c] : rows[i]) {
        if (e[pivots[i]] == 1 && std::count(e.begin(),e.end(),0)+1 == static_cast<long>(e.size())) continue;
        for (auto j : pivots) if (e[j] != 0) return false;
    }
    return true;
}

PowerResult ordinary(const Basis& rows, size_t width, bool lex, const mpz_class& exponent) {
    if (exponent == 0) return unit(width);
    if (rows.empty()) return Basis{};
    if (rows.size() == 1) return Basis{principal_power(rows.front(),width,lex,exponent,false)};
    if (auto packed = packed_power(rows,width,lex,exponent)) return std::move(*packed);
    if (exponent == 1) return primary::reduced(rows,width,lex);
    auto products = power_products(rows,width,exponent);
    if (!graph_coordinates(rows,lex)) return primary::reduced(products,width,lex);
    std::sort(products.begin(),products.end(),[&](const auto& a,const auto& b) {
        return local::global_less(local::global_head(b,lex),local::global_head(a,lex),lex);
    });
    return products;
}

// A nonzero codimension-size Jacobian minor selects a dense smooth open
// subset of this irreducible component. There the prime is a regular
// immersion, and gr_P(R) is locally polynomial over R/P. Its power has
// only P as an associated prime on that open set. Saturation contracts it.
Basis prime_power(const Basis& prime, size_t width, const mpz_class& exponent) {
    if (prime.size() <= 1 || graph_coordinates(prime,true))
        return ordinary(prime,width,true,exponent).materialize();
    const auto height = width-primary::dimension_lex(prime,width);
    auto result = ordinary(prime,width,true,exponent).materialize();
    if (height == width || prime.size() == height) return result;
    local::SparseLexRing ring(width);
    std::vector<PolyValue> divisors, derivatives;
    divisors.reserve(prime.size());
    derivatives.reserve(prime.size()*width);
    std::vector<fmpq_mpoly_struct*> basis, jacobian;
    for (const auto& f : prime) {
        auto p = divisors.emplace_back(ring.ctx).p;
        ring.write(p,f);
        basis.push_back(p);
        for (size_t j=0; j<width; ++j) {
            auto entry = derivatives.emplace_back(ring.ctx).p;
            fmpq_mpoly_derivative(entry,p,j,ring.ctx);
            jacobian.push_back(entry);
        }
    }
    auto minors = reduced_minors(jacobian,prime.size(),height,basis,ring.ctx);
    if (minors.empty()) throw std::logic_error("Missing smooth open subset of a QQ prime ideal.");
    // A low-degree minor generally produces a smaller elimination problem.
    auto best = std::min_element(minors.begin(),minors.end(),[&](const auto& a,const auto& b) {
        return fmpq_mpoly_length(a.p,ring.ctx) < fmpq_mpoly_length(b.p,ring.ctx);
    });
    return primary::saturate_lex(result,ring.read(best->p),width);
}

bool squarefree_heads(const Basis& lex_basis) {
    return std::all_of(lex_basis.begin(),lex_basis.end(),[](const auto& f) {
        const auto& e = f.rbegin()->first;
        return std::all_of(e.begin(),e.end(),[](const auto& k) { return k <= 1; });
    });
}

PowerResult symbolic(const Basis& rows, size_t width, bool lex, const mpz_class& exponent) {
    if (rows.size() == 1) return Basis{principal_power(rows.front(),width,lex,exponent,true)};
    if (graph_coordinates(rows,lex)) return ordinary(rows,width,lex,exponent);
    if (auto split = content_symbolic(rows,width,lex,exponent)) return *split;
    const auto g = primary::reduced(rows,width,true);
    if (g.size() == 1 && constant(g.front())) return unit(width);
    if (g.size() == 1) return Basis{principal_power(g.front(),width,lex,exponent,true)};
    if (std::all_of(g.begin(),g.end(),[](const auto& f){ return f.size() == 1; }))
        return monomial_power(g,width,lex,exponent,true);
    if (!squarefree_heads(g)) require_radical(primary::radical_lex(g,width) == g);
    if (exponent == 0) return unit(width);
    if (exponent == 1) return lex ? g : primary::reduced(rows,width,false);
    const auto height = width-primary::dimension_lex(g,width);
    // Radical complete intersections have unmixed powers; distinct maximal
    // points are pairwise comaximal, so all powers of a radical Artin ideal
    // also have exactly the original associated primes.
    if (height == width || std::min(rows.size(),g.size()) == height)
        return ordinary(rows.size() <= g.size() ? rows : g,width,lex,exponent);
    Basis result = unit(width);
    for (const auto& component : primary::decompose(g,width,true))
        result = primary::intersection_lex(result,prime_power(component.prime,width,exponent),width);
    return lex ? result : primary::reduced(result,width,false);
}
} // namespace

mpz_class parse_exponent(const std::string& text) {
    if (text.empty() || !std::all_of(text.begin(),text.end(),[](char c) { return c >= '0' && c <= '9'; }))
        throw InvalidInput("The ideal power exponent must be a nonnegative integer.");
    return mpz_class(text,10);
}

PowerResult typed_power(const Basis& generators, size_t width, bool lex, const mpz_class& exponent, bool is_symbolic) {
    if (exponent < 0) throw InvalidInput("The ideal power exponent must be a nonnegative integer.");
    const Basis one = unit(width);
    if (!is_symbolic && exponent == 0) return one;
    if (generators.size() == 1 && generators.front().size() > 1)
        return PrincipalBasis(generators.front(),width,lex,exponent,is_symbolic);
    bool has_zero = false, all_monomials = true;
    for (const auto& f : generators) {
        if (constant(f)) return one;
        has_zero |= f.empty();
        if (f.size() > 1) all_monomials = false;
    }
    if (all_monomials) return monomial_power(generators,width,lex,exponent,is_symbolic);
    Basis nonzero;
    if (has_zero)
        for (const auto& f : generators) if (!f.empty()) nonzero.push_back(f);
    const auto& rows = has_zero ? nonzero : generators;
    return is_symbolic ? symbolic(rows,width,lex,exponent) : ordinary(rows,width,lex,exponent);
}

Basis power(const Basis& rows, size_t width, bool lex, const mpz_class& exponent, bool symbolic) {
    return typed_power(rows,width,lex,exponent,symbolic).materialize();
}

std::optional<PowerResult> bounded_ordinary_power(const Basis& rows, size_t width, bool lex, const mpz_class& exponent) {
    if (exponent < 0 || exponent > 4 || width > 4 || rows.size() > 4) return std::nullopt;
    if (exponent == 0) return unit(width);
    bool monomials = true, has_zero = false;
    for (const auto& f : rows) {
        if (f.size() > 16) return std::nullopt;
        monomials &= f.size() <= 1;
        has_zero |= f.empty();
        for (const auto& [e, c] : f) {
            if (mpz_sizeinbase(mpq_numref(c.get_mpq_t()),2) > 128 ||
                mpz_sizeinbase(mpq_denref(c.get_mpq_t()),2) > 128 ||
                std::any_of(e.begin(),e.end(),[](const auto& k) { return k > 16; })) return std::nullopt;
        }
    }
    // At most four monomials to the fourth power have 35 unordered products.
    // For general polynomials the square bound also caps the term products
    // before Buchberger's own coefficient and work limits take effect.
    if (monomials) return typed_power(rows,width,lex,exponent,false);
    if (exponent > 2) return std::nullopt;
    if (rows.size() == 1) return typed_power(rows,width,lex,exponent,false);
    if (has_zero) return std::nullopt;
    if (auto result = packed_power(rows,width,lex,exponent,true))
        return PowerResult(std::move(*result));
    return std::nullopt;
}

Json::Value compute(const Basis& rows, const std::vector<std::string>& names, bool lex,
                    const mpz_class& exponent, bool symbolic) {
    auto result = typed_power(rows,names.size(),lex,exponent,symbolic);
    Json::Value out;
    out["type"] = "ideal";
    auto& ring = out["ring"];
    ring["field"] = "QQ";
    ring["order"] = lex ? "lex" : "grevlex";
    ring["variables"] = Json::Value(Json::arrayValue);
    for (const auto& name : names) ring["variables"].append(name);
    auto& generators = out["generators"];
    generators = Json::Value(Json::arrayValue);
    for (const auto& f : result.expressions(names)) generators.append(f);
    return out;
}

std::string operation(std::istream& in, bool symbolic) {
    std::string variables,order,degree,row;
    if (!std::getline(in,variables) || !std::getline(in,order) ||
        (order != "lex" && order != "grevlex") || !std::getline(in,degree))
        throw InvalidInput("Expected ideal power coordinates, order and exponent.");
    std::istringstream coords(variables);
    std::vector<std::string> names;
    for (std::string name; coords >> name;) names.push_back(std::move(name));
    Json::Value result;
    try {
        const auto exponent = parse_exponent(degree);
        Basis rows;
        while (std::getline(in,row)) rows.push_back(local::decode_row(row,names.size()));
        result = compute(rows,names,order == "lex",exponent,symbolic);
    } catch (const InvalidInput& error) {
        result["error"]["code"] = "invalid_request";
        result["error"]["message"] = error.what();
    }
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    return Json::writeString(writer,result)+"\n";
}
} // namespace atlas::ideal_powers
