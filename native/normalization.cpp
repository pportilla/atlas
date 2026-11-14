// Integral closure of affine QQ domains by the Grauert–Remmert criterion.
#include "normalization.hpp"
#include "local_io.hpp"
#include "primary_decomposition.hpp"
#include "resolution_tangents.hpp"
#include "sparse_flint.hpp"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace atlas::normalization {
namespace {
using local::Powers;
using primary::multiply;

Polynomial one(size_t n) { return {{Powers(n), 1}}; }

Polynomial variable(size_t i, size_t n) {
    Powers e(n);
    e[i] = 1;
    return {{std::move(e), 1}};
}

// Keep a single FLINT conversion of each divisor for a batch of normal forms.
struct QuotientRing {
    local::SparseLexRing ring;
    std::vector<PolyValue> divisors, quotients;
    std::vector<fmpq_mpoly_struct*> divisor_rows, quotient_rows;
    PolyValue input, output;

    QuotientRing(const Basis& lex_basis, size_t n)
        : ring(n), input(ring.ctx), output(ring.ctx) {
        divisors.reserve(lex_basis.size());
        quotients.reserve(lex_basis.size());
        for (const auto& g : lex_basis) {
            auto p = divisors.emplace_back(ring.ctx).p;
            ring.write(p,g);
            divisor_rows.push_back(p);
            quotient_rows.push_back(quotients.emplace_back(ring.ctx).p);
        }
    }

    Polynomial reduce(const Polynomial& g) {
        if (g.empty() || divisors.empty()) return g;
        ring.write(input.p,g);
        fmpq_mpoly_divrem_ideal(quotient_rows.data(),output.p,input.p,
                                divisor_rows.data(),divisor_rows.size(),ring.ctx);
        return ring.read(output.p);
    }

    Polynomial exact_divide(const Polynomial& g, const Polynomial& d) {
        ring.write(input.p,g);
        PolyValue divisor(ring.ctx);
        ring.write(divisor.p,d);
        if (!fmpq_mpoly_divides(output.p,input.p,divisor.p,ring.ctx))
            throw std::logic_error("Inexact division in the normalization ideal quotient.");
        return ring.read(output.p);
    }

    Fraction cancel(Polynomial a, Polynomial b) {
        a = reduce(a);
        b = reduce(b);
        if (b.empty()) throw std::logic_error("Zero denominator on a normalization component.");
        if (a.empty()) return {{}, one(ring.powers.values.size())};
        PolyValue den(ring.ctx), gcd(ring.ctx), other(ring.ctx);
        ring.write(input.p,a);
        ring.write(den.p,b);
        if (!fmpq_mpoly_gcd_cofactors(gcd.p,output.p,other.p,input.p,den.p,ring.ctx))
            throw std::runtime_error("Could not reduce a normalization fraction.");
        a = ring.read(output.p);
        b = ring.read(other.p);
        const mpq_class leading = b.rbegin()->second;
        if (leading != 1) {
            for (auto& term : a) term.second /= leading;
            for (auto& term : b) term.second /= leading;
        }
        return {std::move(a),std::move(b)};
    }
};

Polynomial extend(const Polynomial& f) {
    Polynomial g;
    for (const auto& [e,c] : f) {
        Powers p(e);
        p.emplace_back(0);
        g.emplace(std::move(p),c);
    }
    return g;
}

void extend_all(Basis& rows) {
    for (auto& g : rows) g = extend(g);
}

Presentation identity(const Basis& prime, const std::vector<std::string>& names) {
    Presentation out{names,prime,{},{}};
    const auto denominator = one(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        out.images.push_back(variable(i,names.size()));
        out.fractions.push_back({out.images.back(),denominator});
    }
    return out;
}

Basis jacobian_ideal(const Basis& prime, QuotientRing& A, size_t n, size_t codim) {
    std::vector<PolyValue> derivatives;
    derivatives.reserve(prime.size()*n);
    std::vector<fmpq_mpoly_struct*> entries;
    for (const auto& f : A.divisors) for (size_t i = 0; i < n; ++i) {
        auto p = derivatives.emplace_back(A.ring.ctx).p;
        fmpq_mpoly_derivative(p,f.p,i,A.ring.ctx);
        entries.push_back(p);
    }
    Basis J = prime;
    for (const auto& minor : reduced_minors(entries,prime.size(),codim,A.divisor_rows,A.ring.ctx))
        J.push_back(A.ring.read(minor.p));
    return primary::reduced(J,n,true);
}

void add(Polynomial& sum, const Polynomial& f, const mpq_class& coefficient) {
    for (const auto& [e,c] : f) {
        auto it = sum.try_emplace(e,0).first;
        it->second += coefficient*c;
        if (it->second == 0) sum.erase(it);
    }
}

Fraction evaluate(const Polynomial& f, const std::vector<Fraction>& coordinates,
                  size_t n, QuotientRing& source) {
    // Clear denominators monomial by monomial at the largest power of each
    // coordinate in f. Every operation takes place in the original domain.
    Powers degrees(coordinates.size());
    for (const auto& [e,c] : f) for (size_t i = 0; i < e.size(); ++i)
        degrees[i] = std::max(degrees[i],e[i]);
    auto denominator = one(n);
    std::vector<std::map<mpz_class,Polynomial>> numerators(degrees.size()), denominators(degrees.size());
    auto cached = [&](auto& cache, const auto& p, const auto& k) -> const Polynomial& {
        auto [it,inserted] = cache.try_emplace(k);
        if (inserted) it->second = source.reduce(primary::power(p,k,n));
        return it->second;
    };
    for (size_t i = 0; i < degrees.size(); ++i)
        denominator = source.reduce(multiply(denominator,cached(denominators[i],coordinates[i].denominator,degrees[i])));
    Polynomial numerator;
    for (const auto& [e,c] : f) {
        auto term = one(n);
        for (size_t i = 0; i < e.size(); ++i) {
            if (e[i] != 0)
                term = source.reduce(multiply(term,cached(numerators[i],coordinates[i].numerator,e[i])));
            if (e[i] != degrees[i])
                term = source.reduce(multiply(term,cached(denominators[i],coordinates[i].denominator,degrees[i]-e[i])));
        }
        add(numerator,term,c);
    }
    return source.cancel(std::move(numerator),std::move(denominator));
}

Basis quotient(const Basis& ideal, const Basis& by, QuotientRing& A, size_t n) {
    // C:g = (C intersect (g))/g in the polynomial ring. Intersect these
    // quotients over all generators of J to compute C:J, including I in C.
    Basis result{one(n)};
    for (const auto& g : by) {
        Basis generators;
        for (const auto& f : primary::intersection_lex(ideal,{g},n))
            generators.push_back(A.exact_divide(f,g));
        auto part = primary::reduced(generators,n,true);
        result = primary::intersection_lex(result,part,n);
    }
    return result;
}

std::optional<Fraction> enlargement(const Basis& relations, const Basis& test, size_t n) {
    QuotientRing A(relations,n);
    Basis J;
    for (const auto& f : primary::radical_lex(test,n)) {
        auto r = A.reduce(f);
        if (!r.empty()) J.push_back(std::move(r));
    }
    if (J.empty()) throw std::logic_error("A normalization test ideal vanished on its component.");
    auto degree = [](const Polynomial& f) {
        mpz_class d = 0;
        for (const auto& term : f)
            d = std::max(d,std::accumulate(term.first.begin(),term.first.end(),mpz_class(0)));
        return d;
    };
    std::stable_sort(J.begin(),J.end(),[&](const auto& a,const auto& b) { return degree(a) < degree(b); });
    const auto& g = J.front();
    if (g == one(n)) return std::nullopt;
    Basis principal = relations;
    principal.push_back(g);
    principal = primary::reduced(principal,n,true);
    QuotientRing modulo_g(principal,n);
    Basis product = relations;
    for (const auto& h : J) product.push_back(A.reduce(multiply(g,h)));
    // End_A(J) = (gJ :_A J)/g. A numerator outside gA witnesses strict
    // enlargement. A single such fraction suffices: the integral closure is
    // a finite A-module, so an ascending chain of strict extensions terminates.
    for (const auto& f : quotient(product,J,A,n)) {
        auto r = modulo_g.reduce(f);
        if (!r.empty()) return Fraction{std::move(r),g};
    }
    return std::nullopt;
}

void simplify(Presentation& out, Basis& test_images) {
    // A reduced lex basis with leading term x_i contains a monic graph
    // equation. Eliminate the coordinate and transport the map along it.
    for (size_t r = 0; r < out.relations.size();) {
        const auto& f = out.relations[r];
        const auto& head = f.rbegin()->first;
        const auto i = std::find(head.begin(),head.end(),1);
        if (i == head.end() || std::count(head.begin(),head.end(),0)+1 != static_cast<long>(head.size())) {
            ++r;
            continue;
        }
        size_t axis = i-head.begin();
        Polynomial image;
        for (const auto& [e,c] : f) if (e != head) {
            Powers p(e);
            p.erase(p.begin()+axis);
            image.emplace(std::move(p),-c/f.rbegin()->second);
        }
        const auto n = out.names.size()-1;
        std::map<mpz_class,Polynomial> powers;
        out.relations.erase(out.relations.begin()+r);
        for (auto& g : out.relations) g = primary::substitute(g,axis,image,n,powers);
        for (auto& g : out.images) g = primary::substitute(g,axis,image,n,powers);
        for (auto& g : test_images) g = primary::substitute(g,axis,image,n,powers);
        out.names.erase(out.names.begin()+axis);
        out.fractions.erase(out.fractions.begin()+axis);
    }
}

Presentation compose(const Presentation& step, Presentation normalized,
                     const Basis& source_basis, size_t n) {
    const auto m = normalized.names.size();
    // A specialized presentation can supply multiplication relations that
    // generate the kernel without being a Gröbner basis.
    const auto lex_target = primary::reduced(normalized.relations,m,true);
    QuotientRing source(source_basis,n), target(lex_target,m);
    std::vector<Fraction> target_coordinates;
    for (const auto& p : normalized.images) target_coordinates.push_back({p,one(m)});
    Basis images;
    for (const auto& p : step.images)
        images.push_back(evaluate(p,target_coordinates,m,target).numerator);
    normalized.images = std::move(images);
    for (auto& f : normalized.fractions) {
        const auto num = evaluate(f.numerator,step.fractions,n,source);
        const auto den = evaluate(f.denominator,step.fractions,n,source);
        f = source.cancel(multiply(num.numerator,den.denominator),multiply(num.denominator,den.numerator));
    }
    return normalized;
}

// Eliminate a graph in any coordinate, even when the declared order puts
// nonlinear terms ahead of the monic coordinate. The inverse map is polynomial.
std::optional<Presentation> graph_chart(const Basis& prime, const std::vector<std::string>& names, bool lex) {
    const size_t n = names.size();
    for (const auto& f : prime) for (size_t i = 0; i < n; ++i) {
        Powers x(n);
        x[i] = 1;
        const auto linear = f.find(x);
        if (linear == f.end()) continue;
        if (std::any_of(f.begin(),f.end(),[&](const auto& term) { return term.first[i] != 0 && term.first != x; }))
            continue;
        auto step = identity(prime,names);
        Polynomial image;
        for (const auto& [e,c] : f) if (e != x) {
            auto powers = e;
            powers.erase(powers.begin()+i);
            image.emplace(std::move(powers),-c/linear->second);
        }
        std::map<mpz_class,Polynomial> powers;
        step.relations.clear();
        for (const auto& g : prime) if (&f != &g)
            step.relations.push_back(primary::substitute(g,i,image,n-1,powers));
        step.relations = primary::reduced(step.relations,n-1,lex);
        for (auto& g : step.images) g = primary::substitute(g,i,image,n-1,powers);
        step.names.erase(step.names.begin()+i);
        step.fractions.erase(step.fractions.begin()+i);
        return step;
    }
    return std::nullopt;
}

Presentation normalize_domain(const Basis& prime, const std::vector<std::string>& names, bool lex,
                              bool try_power_content = true) {
    auto canonical = lex ? prime : primary::reduced(prime,names.size(),true);
    const auto dim = primary::dimension_lex(canonical,names.size());
    if (prime.empty() || dim == 0) return identity(prime,names);
    if (auto shortcut = special(prime,names,lex)) return *shortcut;
    if (auto chart = graph_chart(prime,names,lex))
        return compose(*chart,normalize_domain(chart->relations,chart->names,lex,try_power_content),canonical,names.size());
    if (try_power_content) {
        if (auto step = divide_power_content(prime,names,lex))
            return compose(*step,normalize_domain(step->relations,step->names,lex,false),canonical,names.size());
    }

    QuotientRing source(canonical,names.size());
    auto test = jacobian_ideal(canonical,source,names.size(),names.size()-dim);
    const auto singular_dimension = primary::dimension_lex(test,names.size());
    // Regular rings are normal. A complete intersection satisfies Serre S2,
    // so regularity in codimension one already suffices in that case.
    if (singular_dimension == -1 ||
        (canonical.size()==names.size()-dim && singular_dimension<=dim-2)) return identity(prime,names);

    Basis test_images;
    for (const auto& f : test) {
        auto r = source.reduce(f);
        if (!r.empty()) test_images.push_back(std::move(r));
    }
    auto out = identity(canonical,names);
    size_t next = 1;
    while (auto extra = enlargement(out.relations,test,out.names.size())) {
        const auto num = evaluate(extra->numerator,out.fractions,names.size(),source);
        const auto den = evaluate(extra->denominator,out.fractions,names.size(),source);
        out.fractions.push_back(source.cancel(multiply(num.numerator,den.denominator),
                                             multiply(num.denominator,den.numerator)));
        auto name = "n"+std::to_string(next++);
        while (std::find(out.names.begin(),out.names.end(),name)!=out.names.end())
            name = "n"+std::to_string(next++);
        out.names.push_back(name);
        extend_all(out.relations);
        extend_all(out.images);
        extend_all(test_images);
        auto g = extend(extra->denominator);
        Polynomial equation = multiply(g,variable(out.names.size()-1,out.names.size()));
        add(equation,extend(extra->numerator),-1);
        out.relations.push_back(std::move(equation));
        out.relations = primary::saturate_lex(out.relations,g,out.names.size());
        // The finite birational map is an isomorphism off the original test
        // locus. Its inverse image still contains the entire nonnormal locus.
        simplify(out,test_images);
        test = out.relations;
        test.insert(test.end(),test_images.begin(),test_images.end());
    }
    simplify(out,test_images);
    if (!lex) out.relations = primary::reduced(out.relations,out.names.size(),false);
    return out;
}

Json::Value ordered_ring(const std::vector<std::string>& names, bool lex) {
    Json::Value ring;
    ring["field"] = "QQ";
    ring["order"] = lex ? "lex" : "grevlex";
    ring["variables"] = Json::Value(Json::arrayValue);
    for (const auto& v : names) ring["variables"].append(v);
    return ring;
}

Json::Value polynomials(const Basis& g, const std::vector<std::string>& names) {
    Json::Value list(Json::arrayValue);
    for (const auto& f : g) list.append(local::format_polynomial(f,names));
    return list;
}

Json::Value component(const Basis& prime, const Presentation& p, const std::vector<std::string>& names, bool lex) {
    Json::Value part;
    part["prime"] = polynomials(prime,names);
    part["ring"] = ordered_ring(p.names,lex);
    part["relations"] = polynomials(p.relations,p.names);
    part["images"] = polynomials(p.images,p.names);
    part["fractions"] = Json::Value(Json::arrayValue);
    for (const auto& f : p.fractions) {
        Json::Value fraction;
        fraction["numerator"] = local::format_polynomial(f.numerator,names);
        fraction["denominator"] = local::format_polynomial(f.denominator,names);
        part["fractions"].append(std::move(fraction));
    }
    return part;
}

} // namespace

Json::Value compute(const Basis& generators, const std::vector<std::string>& names, bool lex) {
    const auto n = names.size();
    const auto parts = primary::decompose(generators,n,lex);
    Json::Value result;
    result["type"] = "normalization";
    result["ring"] = ordered_ring(names,lex);
    result["input_reduced"] = std::all_of(parts.begin(),parts.end(),[](const auto& p) { return p.primary == p.prime; });
    result["components"] = Json::Value(Json::arrayValue);
    for (size_t i = 0; i < parts.size(); ++i) {
        bool minimal = true;
        if (parts.size() > 1) {
            const auto ordered = lex ? parts[i].prime : primary::reduced(parts[i].prime,n,true);
            QuotientRing quotient_ring(ordered,n);
            for (size_t j = 0; j < parts.size() && minimal; ++j) if (j != i)
                minimal = !std::all_of(parts[j].prime.begin(),parts[j].prime.end(),
                                       [&](const auto& f) { return quotient_ring.reduce(f).empty(); });
        }
        if (minimal)
            result["components"].append(component(parts[i].prime,normalize_domain(parts[i].prime,names,lex),names,lex));
    }
    return result;
}

std::string operation(std::istream& input) {
    std::string header, order, line;
    if (!std::getline(input,header) || !std::getline(input,order) || (order!="lex" && order!="grevlex"))
        throw std::invalid_argument("Expected normalization coordinates and a monomial order.");
    std::istringstream coordinates(header);
    std::vector<std::string> names;
    for (std::string v; coordinates >> v;) names.push_back(std::move(v));
    Basis rows;
    while (std::getline(input,line)) rows.push_back(local::decode_row(line,names.size()));
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    return Json::writeString(writer,compute(rows,names,order=="lex")) + "\n";
}

} // namespace atlas::normalization
