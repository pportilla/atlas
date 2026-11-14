// Primary decomposition over QQ by zero-dimensional extension and contraction.
// Block lex bases implement arithmetic over QQ(U) without building fractions.
#include "primary_decomposition.hpp"
#include "local_monomial.hpp"
#include "local_io.hpp"
#include "local_dimension.hpp"
#include "polynomial_factors.hpp"
#include "sparse_flint.hpp"
#include "field_root_coordinates.hpp"
#include "rational_sparse_json.hpp"
#include <algorithm>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

extern "C" {
void* atlas_groebner(const char*, size_t) noexcept;
void* atlas_groebner_modular(const char*, size_t, long*, double*) noexcept;
const char* atlas_groebner_text(const void*) noexcept;
void atlas_groebner_free(void*) noexcept;
char* atlas_wide_block(const char*, size_t) noexcept;
const char* atlas_groebner_error() noexcept;
void atlas_groebner_release(char*) noexcept;
}

namespace atlas::primary {
namespace {
using Axes = std::vector<size_t>;
using Factors = std::vector<std::pair<Polynomial, mpz_class>>;

Polynomial one(size_t n) { return {{Powers(n), 1}}; }
bool constant(const Polynomial& p) {
    return p.size() == 1 && std::all_of(p.begin()->first.begin(), p.begin()->first.end(),
                                      [](const auto& e) { return e == 0; });
}

void add(Polynomial& f, Powers e, const mpq_class& c) {
    if (c == 0) return;
    auto it = f.emplace(std::move(e), c);
    if (!it.second && (it.first->second += c) == 0) f.erase(it.first);
}

Polynomial product(const Polynomial& a, const Polynomial& b) {
    Polynomial result;
    for (const auto& [u,c] : a) for (const auto& [v,d] : b) {
        Powers e(u);
        for (size_t i = 0; i < e.size(); ++i) e[i] += v[i];
        add(result, std::move(e), c*d);
    }
    return result;
}

thread_local bool prefer_modular = false;
struct RestorePolicy {
    bool previous = prefer_modular;
    ~RestorePolicy() { prefer_modular = previous; }
};

Basis basis(const Basis& rows, size_t n, int order) {
    if (!n) return std::any_of(rows.begin(), rows.end(), constant) ? Basis{one(n)} : Basis{};
    std::ostringstream wire;
    wire << n << ' ' << order << " 0\n";
    for (const auto& row : rows) {
        if (constant(row)) return {one(n)};
        if (!row.empty()) wire << local::encode_row(row) << '\n';
    }
    const auto request = wire.str();
    std::unique_ptr<void, decltype(&atlas_groebner_free)> handle(
        nullptr, atlas_groebner_free);
    if (prefer_modular)
        handle.reset(atlas_groebner_modular(request.data(), request.size(), nullptr, nullptr));
    if (!handle) handle.reset(atlas_groebner(request.data(), request.size()));
    std::string result;
    if (handle) result = atlas_groebner_text(handle.get());
    else {
        std::string error = atlas_groebner_error();
        if (error.find("out of range") == std::string::npos && error != "Unsupported number of variables.")
            throw std::runtime_error(error);
        const auto wide = "basis " + request;
        std::unique_ptr<char, decltype(&atlas_groebner_release)> reply(
            atlas_wide_block(wide.data(), wide.size()), atlas_groebner_release);
        if (!reply) throw std::runtime_error(atlas_groebner_error());
        result = reply.get();
    }
    Basis out;
    std::istringstream stream(result);
    for (std::string line; std::getline(stream, line);)
        if (!line.empty()) out.push_back(local::decode_row(line, n));
    return out;
}

Polynomial remap(const Polynomial& f, const Axes& axes) {
    Polynomial result;
    for (const auto& [e,c] : f) {
        Powers p(axes.size());
        for (size_t i = 0; i < axes.size(); ++i) if (axes[i] < e.size()) p[i] = e[axes[i]];
        result.emplace(std::move(p),c);
    }
    return result;
}

Basis remap(const Basis& g, const Axes& axes) {
    Basis result;
    for (const auto& f : g) result.push_back(remap(f, axes));
    return result;
}

Axes sequence(size_t n) {
    Axes a(n);
    std::iota(a.begin(), a.end(), 0);
    return a;
}
Axes inverse(const Axes& a) {
    Axes b(a.size());
    for (size_t i = 0; i < a.size(); ++i) b[a[i]] = i;
    return b;
}
Axes complement(const Axes& axes, size_t n) {
    Axes result;
    for (size_t i = 0; i < n; ++i)
        if (std::find(axes.begin(), axes.end(), i) == axes.end()) result.push_back(i);
    return result;
}

Polynomial variable(size_t index, size_t n) {
    Powers e(n);
    e[index] = 1;
    return {{std::move(e),1}};
}

// All membership checks during the dimension induction use block lex.
bool member(Polynomial f, const Basis& g) {
    while (!f.empty()) {
        const auto [head, coeff] = *f.rbegin();
        auto divisor = std::find_if(g.begin(), g.end(), [&](const auto& p) {
            return local::monomial_divides(p.rbegin()->first, head);
        });
        if (divisor == g.end()) return false;
        const auto& [lead, lc] = *divisor->rbegin();
        Powers shift(head);
        for (size_t i = 0; i < head.size(); ++i) shift[i] -= lead[i];
        mpq_class scale = -coeff/lc;
        for (const auto& [e,c] : *divisor) {
            Powers shifted(e);
            for (size_t i = 0; i < e.size(); ++i) shifted[i] += shift[i];
            add(f, std::move(shifted), c*scale);
        }
    }
    return true;
}
bool contained(const Basis& generators, const Basis& lex_basis) {
    return std::all_of(generators.begin(), generators.end(), [&](const auto& f) { return member(f, lex_basis); });
}

Basis eliminate_first(const Basis& rows, size_t n, bool lex) {
    Basis result;
    Axes axes = sequence(n+1);
    axes.erase(axes.begin());
    for (const auto& f : basis(rows, n+1, lex ? 1 : 2)) {
        if (std::all_of(f.begin(), f.end(), [](const auto& term) { return term.first.front() == 0; }))
            result.push_back(remap(f, axes));
    }
    return result;
}

Basis saturate(const Basis& g, const Polynomial& h, size_t n) {
    if (constant(h)) return g;
    auto axes = sequence(n);
    axes.insert(axes.begin(), n);
    Basis rows = remap(g, axes);
    Polynomial relation = one(n+1);
    for (const auto& [e,c] : h) {
        Powers p = e;
        p.insert(p.begin(),1);
        relation.emplace(std::move(p), -c);
    }
    rows.push_back(std::move(relation));
    return eliminate_first(rows,n,true);
}

Basis intersection(const Basis& a, const Basis& b, size_t n) {
    if (a == b || b == Basis{one(n)}) return a;
    if (a == Basis{one(n)}) return b;
    Basis rows;
    auto append = [&](const Basis& g, bool with_constant) {
        for (const auto& f : g) {
            Polynomial row;
            for (const auto& [e,c] : f) {
                Powers p(e);
                p.insert(p.begin(),1);
                row.emplace(p,c);
                if (with_constant) { p[0] = 0; row.emplace(std::move(p), -c); }
            }
            rows.push_back(std::move(row));
        }
    };
    append(a,false); append(b,true);
    return eliminate_first(rows,n,true);
}

Basis with(const Basis& g, const Polynomial& f, size_t n) {
    auto rows = g;
    rows.push_back(f);
    return basis(rows,n,1);
}

bool supported_on(const Polynomial& f, const Axes& excluded) {
    for (const auto& [e,c] : f) for (auto i : excluded) if (e[i] != 0) return false;
    return true;
}

// Return a nonzero relation in the selected coordinates, or prove independence
// by elimination. A head outside the selected subring cannot divide a monomial in it.
Polynomial relation(const Basis& g, const Axes& selected, size_t n) {
    auto excluded = complement(selected,n);
    for (const auto& f : g) if (supported_on(f, excluded)) return f;
    bool possible = false;
    for (const auto& f : g) {
        const auto& e = f.rbegin()->first;
        if (std::all_of(excluded.begin(), excluded.end(), [&](size_t i) { return e[i] == 0; })) possible = true;
    }
    if (!possible) return {};
    auto order = excluded;
    order.insert(order.end(), selected.begin(), selected.end());
    auto gb = basis(remap(g,order), n, 1);
    for (const auto& f : gb) {
        const auto& e = f.rbegin()->first;
        if (std::all_of(e.begin(), e.begin()+excluded.size(), [](const auto& k) { return k == 0; }))
            return remap(f,inverse(order));
    }
    return {};
}

struct Extension {
    Axes axes;
    Basis relations;
    size_t dependent;
};

Extension extension(const Basis& g, size_t n) {
    // Greedy gives a maximal independent coordinate set. Every skipped
    // coordinate has a relation already over the previously chosen parameters.
    Axes params;
    Basis rels;
    for (size_t i = n; i-- > 0;) {
        auto trial = params;
        trial.insert(trial.begin(),i);
        auto f = relation(g,trial,n);
        if (f.empty()) params = std::move(trial);
        else rels.push_back(std::move(f));
    }
    auto axes = complement(params,n);
    size_t dep = axes.size();
    axes.insert(axes.end(), params.begin(), params.end());
    return {axes,remap(rels,axes),dep};
}

// Over QQ(U) the projected heads can be redundant even in a reduced QQ
// block basis. Choose one row per minimal head. Division by just these rows
// introduces only inverses of their leading coefficients. This yields the
// same contraction and avoids adding artificial exceptional fibers.
Polynomial denominator(const Basis& g, size_t dep, size_t n) {
    if (dep == n) return one(n);
    std::map<Powers,Polynomial> coefficients;
    for (const auto& f : g) {
        const auto& head = f.rbegin()->first;
        Powers projected(head.begin(), head.begin()+dep);
        Polynomial lc;
        for (auto it = f.rbegin(); it != f.rend(); ++it) {
            const auto& [e,c] = *it;
            if (!std::equal(projected.begin(), projected.end(), e.begin())) break;
            Powers p(e);
            std::fill(p.begin(), p.begin()+dep, 0);
            lc.emplace(std::move(p), c);
        }
        auto [it, inserted] = coefficients.try_emplace(std::move(projected), lc);
        if (!inserted && (constant(lc) || lc.size() < it->second.size()))
            it->second = std::move(lc);
    }
    Polynomial h = one(n);
    for (const auto& [head,lc] : coefficients) {
        if (constant(lc)) continue;
        const bool minimal = std::none_of(coefficients.begin(), coefficients.end(), [&](const auto& pair) {
            return pair.first != head && local::monomial_divides(pair.first, head);
        });
        if (minimal) h = product(h,lc);
    }
    if (h.size() == 1) {
        Powers support = h.begin()->first;
        for (auto& e : support) if (e != 0) e = 1;
        return {{std::move(support), 1}};
    }
    local::SparseLexRing ring(n);
    PolyValue input(ring.ctx), squarefree(ring.ctx);
    ring.write(input.p,h);
    rational_squarefree_part(squarefree.p,input.p,ring.ctx);
    return ring.read(squarefree.p);
}

// For a block lex basis over QQ(U), a pure linear leading power for every
// coordinate before t proves the quotient is generated by t. Its only
// relation is the eliminant of smallest positive degree.
const Polynomial* shape(const Basis& g, size_t t) {
    std::vector<bool> solved(t);
    const Polynomial* eliminant = nullptr;
    for (const auto& f : g) {
        const auto& head = f.rbegin()->first;
        size_t i = 0;
        while (i < t && head[i] == 0) ++i;
        if (i == t) {
            if (head[t] > 0 && (!eliminant || head[t] < eliminant->rbegin()->first[t])) eliminant = &f;
        } else if (head[i] == 1 &&
                   std::all_of(head.begin()+i+1,head.begin()+t+1,[](const auto& k) { return k == 0; })) {
            solved[i] = true;
        }
    }
    return std::all_of(solved.begin(),solved.end(),[](bool b) { return b; }) ? eliminant : nullptr;
}

Factors field_factors(const Polynomial& eliminant, size_t t, size_t n) {
    Factors result;
    for (auto& item : factors(eliminant,n)) {
        if (item.first.rbegin()->first[t] != 0) result.push_back(std::move(item));
    }
    if (result.empty()) throw std::logic_error("Missing minimal polynomial over the parameter field.");
    return result;
}

Basis contract(const Basis& rows, size_t dep, size_t n) {
    return saturate(rows,denominator(rows,dep,n),n);
}

Components in_shape(const Basis& g, const Basis& open, const Polynomial& eliminant, size_t dep, size_t n) {
    auto split = field_factors(eliminant,dep-1,n);
    Components out;
    for (const auto& [p,k] : split) {
        Basis q = split.size()==1 ? open : contract(with(g,power(p,k,n),n),dep,n);
        Basis prime = k==1 ? q : contract(with(g,p,n),dep,n);
        out.push_back({std::move(q),std::move(prime)});
    }
    return out;
}

Factors separating_factors(const Basis& radical, size_t dep, size_t n) {
    if (const auto* f = shape(radical,dep-1)) return field_factors(*f,dep-1,n);
    // A reduced finite algebra over QQ(U) has a primitive element. For any two
    // geometric points only finitely many integer k fail to separate them by
    // x0 + k*x1 + ... + k^(d-1)*x_(d-1). The shape test certifies each attempt.
    Axes lifted = sequence(n);
    lifted.insert(lifted.begin()+dep,n);
    auto extended = remap(radical,lifted);
    for (mpz_class k=1;; ++k) {
        Polynomial linear;
        mpz_class coefficient=1;
        for (size_t i=0;i<dep;++i) {
            Powers e(n);
            e[i] = 1;
            linear.emplace(std::move(e),coefficient);
            coefficient *= k;
        }
        Polynomial eq = remap(linear,lifted);
        add(eq,variable(dep,n+1).begin()->first,-1);
        auto gb = with(extended,eq,n+1);
        if (const auto* f = shape(gb,dep)) {
            auto split = field_factors(*f,dep,n+1);
            std::map<mpz_class,Polynomial> powers;
            for (auto& [p,e] : split) p = substitute(p,dep,linear,n,powers);
            return split;
        }
    }
}

Basis squarefree_rows(const Basis& rows, size_t n) {
    local::SparseLexRing ring(n);
    PolyValue input(ring.ctx), output(ring.ctx);
    Basis result;
    for (const auto& row : rows) {
        if (row.empty()) continue;
        if (constant(row)) return {one(n)};
        ring.write(input.p, row);
        rational_squarefree_part(output.p, input.p, ring.ctx);
        auto f = ring.read(output.p);
        if (std::find(result.begin(), result.end(), f) == result.end())
            result.push_back(std::move(f));
    }
    return result;
}

Basis radical(Basis rows, size_t n, bool lex) {
    std::vector<Basis> pieces;
    while (true) {
        rows = squarefree_rows(rows, n);
        if (rows.empty() || constant(rows.front())) break;
        const bool principal = rows.size() == 1;
        rows = basis(rows, n, 1);
        if (principal || std::all_of(rows.begin(), rows.end(), [](const auto& f) {
            const auto& head = f.rbegin()->first;
            return std::all_of(head.begin(), head.end(), [](const auto& e) { return e <= 1; });
        })) break;

        const auto ext = extension(rows, n);
        auto localized = remap(rows, ext.axes);
        // Each dependent coordinate now satisfies a separable relation over
        // QQ(U). This preserves the zero set also on the exceptional fibres.
        for (auto& sf : squarefree_rows(ext.relations, n))
            localized.push_back(std::move(sf));
        auto block = basis(localized, n, 1);
        const auto h = denominator(block, ext.dependent, n);
        const auto back = inverse(ext.axes);
        if (constant(h)) {
            rows = basis(remap(block, back), n, 1);
            break;
        }
        pieces.push_back(basis(remap(saturate(block, h, n), back), n, 1));
        block.push_back(h);
        rows = remap(block, back);
    }
    for (auto it = pieces.rbegin(); it != pieces.rend(); ++it)
        rows = intersection(rows, *it, n);
    return lex ? rows : basis(rows, n, 0);
}

Components generic_components(const Basis& g, const Basis& open, const Extension& ext, size_t n) {
    const auto dep = ext.dependent;
    if (const auto* f = shape(g,dep-1)) return in_shape(g,open,*f,dep,n);

    // After imposing a separable relation on each coordinate, the resulting
    // finite algebra is a quotient of a tensor product of finite etale
    // algebras. It is reduced and has exactly the original geometric points.
    Basis rows = g;
    local::SparseLexRing ring(n);
    PolyValue input(ring.ctx), squarefree(ring.ctx);
    for (const auto& relation : ext.relations) {
        ring.write(input.p,relation);
        rational_squarefree_part(squarefree.p,input.p,ring.ctx);
        auto sf = ring.read(squarefree.p);
        if (!member(sf,g)) rows.push_back(std::move(sf));
    }
    const auto radical = rows.size() == g.size() ? g : basis(rows,n,1);
    const auto open_radical = radical == g ? open : contract(radical,dep,n);
    auto split = separating_factors(radical,dep,n);
    Components out;
    for (size_t i=0;i<split.size();++i) {
        // On every other Artin local factor the product is nilpotent. On
        // this factor it is a unit, so saturation isolates the entire primary
        // component, including nilpotents that one coordinate may not see.
        Polynomial other = one(n);
        for (size_t j=0;j<split.size();++j) if (i != j) other = product(other,split[j].first);
        auto q = saturate(open,other,n);
        auto p = open == open_radical ? q : saturate(open_radical,other,n);
        out.push_back({std::move(q),std::move(p)});
    }
    return out;
}

Components irredundant(Components input, size_t n) {
    Components result;
    for (auto& item : input) {
        auto it = std::find_if(result.begin(),result.end(),[&](const auto& c) { return c.prime == item.prime; });
        if (it == result.end()) result.push_back(std::move(item));
        else it->primary = intersection(it->primary,item.primary,n);
    }
    for (size_t i = 0; i < result.size();) {
        // Redundancy forces some other radical into this prime. Distinct
        // isolated primes therefore require no intersection tests.
        bool embedded = false;
        for (size_t j=0;j<result.size();++j) if (j!=i && contained(result[j].prime,result[i].prime)) {
            embedded = true;
            break;
        }
        if (embedded) {
            Basis rest{one(n)};
            for (size_t j=0;j<result.size();++j) if (j!=i) rest=intersection(rest,result[j].primary,n);
            if (contained(rest,result[i].primary)) {
                result.erase(result.begin()+i);
                continue;
            }
        }
        ++i;
    }
    return result;
}

// If H avoids the associated primes of J, multiplication by H on R/J
// is injective, hence H*J = (H) intersect J. The quotients by the gcd
// have no common hypersurface component, so this union is irredundant.
std::optional<Components> split_content(const Basis& rows, size_t n, bool lex) {
    local::SparseLexRing ring(n);
    PolyValue gcd(ring.ctx), f(ring.ctx), q(ring.ctx);
    for (const auto& row : rows) {
        if (row.empty()) continue;
        ring.write(f.p,row);
        if (fmpq_mpoly_is_zero(gcd.p,ring.ctx)) fmpq_mpoly_set(gcd.p,f.p,ring.ctx);
        else if (!fmpq_mpoly_gcd(gcd.p,gcd.p,f.p,ring.ctx)) return std::nullopt;
        if (fmpq_mpoly_is_fmpq(gcd.p,ring.ctx)) return std::nullopt;
    }
    Basis quotients;
    for (const auto& row : rows) {
        if (row.empty()) continue;
        ring.write(f.p,row);
        if (!fmpq_mpoly_divides(q.p,f.p,gcd.p,ring.ctx))
            throw std::logic_error("Inexact polynomial content quotient.");
        quotients.push_back(ring.read(q.p));
    }
    auto h = ring.read(gcd.p);
    auto parts = decompose(quotients,n,lex);
    for (const auto& item : parts) {
        if (lex) {
            if (member(h,item.prime)) return std::nullopt;
        } else {
            // Use the requested order for an exact membership test. The
            // wide division kernel operates on arbitrary sparse powers.
            std::ostringstream wire;
            wire << "remainder " << n << " 0 0\n" << local::encode_row(h) << '\n';
            for (const auto& p : item.prime) wire << local::encode_row(p) << '\n';
            auto request = wire.str();
            std::unique_ptr<char, decltype(&atlas_groebner_release)> remainder(
                atlas_wide_block(request.data(),request.size()), atlas_groebner_release);
            if (!remainder) throw std::runtime_error(atlas_groebner_error());
            if (remainder.get()[0] == '\n') return std::nullopt;
        }
    }
    auto principal = decompose({h},n,lex);
    parts.insert(parts.end(),std::make_move_iterator(principal.begin()),std::make_move_iterator(principal.end()));
    return parts;
}

int dimension(const Basis& g, size_t n) {
    std::vector<std::vector<int>> supports;
    for (const auto& f : g) {
        const auto& e = f.rbegin()->first;
        std::vector<int> support;
        for (size_t i = 0; i < n; ++i) if (e[i] != 0) support.push_back(i);
        if (support.empty()) return -1;
        supports.push_back(std::move(support));
    }
    return local::monomial_support_dimension(std::move(supports), n);
}

Components decompose_lex(const Basis& g, size_t n, std::optional<int> unmixed_dimension) {
    // Every spurious component introduced by an exceptional split lies in
    // the support of the input. If the input is unmixed, a branch below its
    // dimension cannot contain an associated prime and may be discarded.
    if (unmixed_dimension && dimension(g,n) < *unmixed_dimension) return {};
    if (auto quick = special_decomposition(g,n,true)) return *quick;
    // A shape-position basis over QQ is already zero-dimensional, so no
    // separate independence or coordinate-relation computations are needed.
    if (const auto* f = shape(g,n-1)) return in_shape(g,g,*f,n,n);
    auto ext = extension(g,n);
    const bool permuted = ext.axes != sequence(n);
    auto block = permuted ? basis(remap(g,ext.axes),n,1) : g;
    auto h = denominator(block,ext.dependent,n);
    auto open = saturate(block,h,n);
    auto result = generic_components(block,open,ext,n);
    if (open != block) {
        // Find m with h^m (I:h^infty) contained in I. This proves exactly
        // I = (I:h^infty) intersect (I+(h^m)); the second ideal strictly
        // contains I because h belongs to the independent parameter ring.
        auto hm = h;
        while (!std::all_of(open.begin(),open.end(),[&](const auto& f) { return member(product(f,hm),block); }))
            hm = product(hm,hm);
        auto rest = decompose_lex(with(block,hm,n),n,unmixed_dimension);
        result.insert(result.end(),std::make_move_iterator(rest.begin()),std::make_move_iterator(rest.end()));
        result = irredundant(std::move(result),n);
    }
    if (permuted) {
        const auto back = inverse(ext.axes);
        for (auto& item : result) {
            const bool prime = item.primary == item.prime;
            item.primary = basis(remap(item.primary,back),n,1);
            item.prime = prime ? item.primary : basis(remap(item.prime,back),n,1);
        }
    }
    return result;
}
} // namespace


bool member_lex(const Polynomial& f, const Basis& lex_basis) { return member(f,lex_basis); }
Basis radical_lex(const Basis& rows, size_t width) { return radical(rows,width,true); }
Basis intersection_lex(const Basis& a, const Basis& b, size_t width) { return intersection(a,b,width); }
Basis saturate_lex(const Basis& rows, const Polynomial& h, size_t width) { return saturate(rows,h,width); }
int dimension_lex(const Basis& rows, size_t width) { return dimension(rows,width); }
Polynomial multiply(const Polynomial& a, const Polynomial& b) { return product(a,b); }

Polynomial substitute(const Polynomial& f, size_t axis, const Polynomial& image,
                      size_t width, std::map<mpz_class, Polynomial>& powers) {
    Polynomial out;
    for (const auto& [e,c] : f) {
        const auto k = e[axis];
        auto it = powers.find(k);
        if (it == powers.end()) it = powers.emplace(k,power(image,k,width)).first;
        Powers p = e;
        p.erase(p.begin()+axis);
        for (const auto& [v,b] : it->second) {
            Powers shifted(p);
            for (size_t i = 0; i < width; ++i) shifted[i] += v[i];
            add(out,std::move(shifted),c*b);
        }
    }
    return out;
}

Basis reduced(const Basis& rows, size_t width, bool lex) { return basis(rows,width,lex ? 1 : 0); }

Basis reduced(const Basis& rows, size_t width, bool lex, bool modular) {
    RestorePolicy policy;
    prefer_modular = modular;
    return reduced(rows, width, lex);
}

Polynomial power(Polynomial f, mpz_class exponent, size_t n) {
    if (exponent < 0) throw std::invalid_argument("Negative polynomial power.");
    Polynomial result = one(n);
    while (exponent > 0) {
        if (mpz_odd_p(exponent.get_mpz_t())) result = product(result,f);
        exponent >>= 1;
        if (exponent > 0) f = product(f,f);
    }
    return result;
}

Factors factors(const Polynomial& f, size_t n) {
    local::SparseLexRing ring(n);
    PolyValue poly(ring.ctx);
    ring.write(poly.p,f);
    struct Factorization {
        fmpq_mpoly_factor_t value;
        const fmpq_mpoly_ctx_struct* ctx;
        explicit Factorization(const fmpq_mpoly_ctx_struct* c) : ctx(c) { fmpq_mpoly_factor_init(value,ctx); }
        ~Factorization() { fmpq_mpoly_factor_clear(value,ctx); }
    } split(ring.ctx);
    ordered_factors(split.value,poly.p,ring.ctx);
    fmpq_mpoly_factor_sort(split.value,ring.ctx);
    Factors result;
    for (slong i=0;i<split.value->num;++i) {
        mpz_class multiplicity;
        fmpz_get_mpz(multiplicity.get_mpz_t(),split.value->exp+i);
        result.emplace_back(ring.read(split.value->poly+i),std::move(multiplicity));
    }
    return result;
}

Components decompose(const Basis& rows, size_t n, bool lex) {
    if (auto quick = special_decomposition(rows,n,lex)) return *quick;
    if (auto split = split_content(rows,n,lex)) return *split;
    if (auto split = split_comaximal(rows,n,lex)) return *split;
    auto g = basis(rows,n,1);
    // A height-r proper ideal generated by r elements in a polynomial ring is
    // a complete intersection. Its quotient is Cohen-Macaulay and unmixed.
    const int dim = dimension(g,n);
    const auto count = std::count_if(rows.begin(), rows.end(), [](const auto& f) { return !f.empty(); });
    std::optional<int> unmixed;
    if (dim >= 0 && n-static_cast<size_t>(dim) == static_cast<size_t>(count)) unmixed = dim;
    auto result = decompose_lex(g,n,unmixed);
    if (!lex) for (auto& item : result) {
        const bool prime = item.primary == item.prime;
        item.primary = basis(item.primary,n,0);
        item.prime = prime ? item.primary : basis(item.prime,n,0);
    }
    return result;
}

namespace {
// A prime over a number field K is geometrically prime precisely when
// X x_K X is irreducible. Over Qbar the Galois group acts transitively on
// the components of X. In the square, diagonal pairs and unequal pairs
// form disjoint orbits as soon as there is more than one component.
// The product is reduced in characteristic zero. Sharing the last variable
// identifies the two copies of the specified coefficient field.
bool geometrically_prime(const Basis& prime, size_t n, bool marked, int dim) {
    const size_t coordinates = n - size_t(marked);
    // Over K a reduced lex basis of a K-rational point has leading
    // monomials x_1,...,x_d in the geometric coordinates. More generally
    // if every such head is a single variable, the remaining coordinates
    // are free and the quotient is a polynomial ring over K.
    bool affine_space = true;
    const Polynomial* equation = nullptr;
    size_t equations = 0;
    for (const auto& f : prime) {
        const auto& e = f.rbegin()->first;
        if (marked && std::all_of(e.begin(),e.end()-1,[](const auto& k){return k==0;})) continue;
        equation = &f;
        ++equations;
        if (local::total_degree(e) != 1) affine_space = false;
    }
    if (affine_space || dim == 0) return affine_space;
    if (equations == 1) {
        // An irreducible primitive polynomial linear in one geometric
        // variable stays irreducible after extending its constant field.
        for (size_t j=0; j<coordinates; ++j) {
            mpz_class degree = 0;
            for (const auto& term : *equation) degree = std::max(degree,term.first[j]);
            if (degree == 1) return true;
        }
    }
    const size_t width = coordinates + n;
    Axes left(width,n), right(width,n);
    for (size_t j=0; j<coordinates; ++j) {
        left[j] = j;
        right[coordinates+j] = j;
    }
    if (marked) left.back() = right.back() = n-1;
    Basis square = remap(prime,left);
    auto second = remap(prime,right);
    square.insert(square.end(),second.begin(),second.end());
    return decompose(square,width,true).size() == 1;
}

// At a nonempty, reduced finite fibre of a coordinate projection of a
// d-dimensional prime to A^d, the fibre equations span the cotangent
// space at each point. Its dimension is at most d, so each point is smooth.
// Such fibres fill a nonempty open by generic separability. The integral
// grid is Zariski dense; increasing boxes avoid a degree or search cap.
Basis smooth_sample(const Basis& prime, size_t n) {
    auto ext = extension(prime,n);
    const Axes parameters(ext.axes.begin()+ext.dependent,ext.axes.end());
    if (parameters.empty()) return prime;
    Basis slice(prime);
    Basis sample;
    for (mpz_class height=0; ; ++height) {
        auto search = [&](auto&& self, size_t j, bool boundary) -> bool {
            if (j == parameters.size()) {
                if (!boundary) return false;
                auto fibre = basis(slice,n,1);
                if (dimension(fibre,n)!=0 || radical_lex(fibre,n)!=fibre) return false;
                sample = std::move(fibre);
                return true;
            }
            slice.push_back(variable(parameters[j],n));
            for (mpz_class value=0; value<=height; ++value) {
                if (value!=0) slice.back()[Powers(n)] = -value;
                if (self(self,j+1,boundary || value==height)) return true;
            }
            slice.pop_back();
            return false;
        };
        if (search(search,0,false)) return sample;
    }
}

// On a finite marked section, split a geometric coordinate before decomposing
// the entire compositum over QQ. The rational elimination polynomial vanishes
// on the original ideal. Trying *every* root and cutting the original ideal
// by each linear equation preserves all correlations between coordinates.
using RootList = std::vector<std::vector<mpq_class>>;
using RootCache = std::map<Polynomial,std::optional<RootList>>;

std::optional<std::vector<mpq_class>> dense_coefficients(const Polynomial& f) {
    const auto& degree=f.rbegin()->first.front();
    if (!mpz_fits_slong_p(degree.get_mpz_t()) ||
        degree >= std::vector<mpq_class>().max_size()) return std::nullopt;
    std::vector<mpq_class> coefficients(degree.get_ui()+1);
    for (const auto& [e,c] : f) coefficients[e.front().get_ui()] = c;
    return coefficients;
}

// A reduced finite lex basis with heads x_i (i != j) and x_j^d
// presents QQ[t]/(h) with t=x_j. In particular this also recognizes a
// separating coordinate followed in the order by fixed rational coordinates.
const Polynomial* finite_shape(const Basis& g, size_t n, size_t& axis) {
    if (g.size()!=n) return nullptr;
    const Polynomial* eliminant=nullptr;
    for (const auto& f : g) {
        const auto& head=f.rbegin()->first;
        const auto i=std::find_if(head.begin(),head.end(),[](const auto& e){return e!=0;});
        if (i==head.end() || std::any_of(i+1,head.end(),[](const auto& e){return e!=0;})) return nullptr;
        if (*i>1) {
            if (eliminant) return nullptr;
            axis=i-head.begin();
            eliminant=&f;
        }
    }
    if (!eliminant) {
        axis=n-1;
        return shape(g,axis);
    }
    return supported_on(*eliminant,complement({axis},n)) ? eliminant : nullptr;
}

bool finite_marked_points(const Basis& g, size_t n, const std::vector<mpq_class>& minimal,
                          RootCache& roots, std::vector<Basis>& points) {
    if (g == Basis{one(n)}) return true;
    size_t axis=n;
    for (const auto& f : g) {
        const auto& head=f.rbegin()->first;
        for (size_t j=0; j+1<n; ++j) {
            if (head[j]>1 && local::total_degree(head)==head[j]) axis=j;
        }
    }
    if (axis==n) {
        // A zero-dimensional lex basis contains pure powers of all axes.
        // Geometric heads of degree one present a single reduced K-point.
        if (!geometrically_prime(g,n,true,0)) return false;
        points.push_back(g);
        return true;
    }
    const auto eliminant=remap(relation(g,{axis},n),{axis});
    if (eliminant.empty()) return false;
    // The rational elimination polynomial may contain extra conjugates for
    // other embeddings of K. Their cuts are empty on this chart. If its
    // roots are not all in K, the unrestricted decomposition still applies.
    for (const auto& [factor,multiplicity] : factors(eliminant,1)) {
        auto it=roots.find(factor);
        if (it==roots.end()) {
            const auto eq=dense_coefficients(factor);
            if (!eq) return false;
            it=roots.emplace(factor,atlas::roots_in_power_basis(minimal,*eq)).first;
        }
        if (!it->second) return false;
        for (const auto& root : *it->second) {
            Polynomial cut=variable(axis,n);
            for (size_t j=0; j<root.size(); ++j) {
                Powers e(n);
                e.back()=j;
                add(cut,e,-root[j]);
            }
            if (!finite_marked_points(with(g,cut,n),n,minimal,roots,points)) return false;
        }
    }
    return true;
}

Json::Value absolute_sections(const Json::Value& input) {
    const size_t n = input["width"].asUInt();
    const bool marked = input["marked"].asBool();
    if (!n || input["operation"].asString()!="decompose")
        throw std::invalid_argument("Expected a nonempty polynomial ring for geometric components.");
    Basis rows;
    for (const auto& row : input["ideal"]) rows.push_back(sparse_json::read_terms(row,n));
    auto gb = basis(rows,n,1);
    const auto modulus = marked ? relation(gb,{n-1},n) : Polynomial{};
    if (marked && modulus.empty()) throw std::invalid_argument("Missing algebraic coefficient-field equation.");
    // Work on reduced support. The caller checks local crossings on the fibre.
    Json::Value primes(Json::arrayValue), fields(Json::arrayValue);
    const int dim=input["dimension"].asInt();
    Components components;
    bool found_points=false;
    if (!marked && dim==0) {
        // In lex shape position the whole coordinate algebra is QQ[t]/(h).
        // Its geometric points are determined by t alone, even if h has
        // repeated factors. Splitting every coordinate eliminant separately
        // would construct algebraic numbers already contained in this field.
        size_t axis;
        if (const auto* h=finite_shape(gb,n,axis)) {
            found_points=true;
            const auto split=field_factors(*h,axis,n);
            for (const auto& [p,multiplicity] : split) {
                if (p.rbegin()->first[axis]>1) {
                    fields.append(sparse_json::term_list(remap(p,{axis})));
                } else {
                    auto point=split.size()==1 && multiplicity==1 ? gb : with(gb,p,n);
                    components.push_back({{},std::move(point)});
                }
            }
        }
    }
    if (marked && dim==0 && dimension(gb,n)==0) {
        const auto minimal=dense_coefficients(remap(modulus,{n-1}));
        std::vector<Basis> points;
        RootCache roots;
        if (minimal) for (const auto& entry : input["root_images"]) {
            const auto equation=sparse_json::read_row(entry[0]);
            RootList images;
            for (const auto& row : entry[1]) images.push_back(sparse_json::read_row(row));
            if (!atlas::certify_power_basis_roots(*minimal,equation,images)) continue;
            Polynomial key;
            for (size_t i=0; i<equation.size(); ++i) add(key,{i},equation[i]/equation.back());
            roots.emplace(std::move(key),std::move(images));
        }
        if (minimal && finite_marked_points(gb,n,*minimal,roots,points)) {
            found_points=true;
            for (auto& point : points) components.push_back({{},std::move(point)});
        }
    }
    if (!found_points && marked && n>1 && gb.size()==2 && !constant(modulus)) {
        // A hypersurface over K needs one Trager factorization, avoiding
        // multivariate primary decomposition of the larger rational lift.
        const auto& f = gb.front()==modulus ? gb.back() : gb.front();
        for (const auto& factor : marked_factors(f,modulus,n))
            components.push_back({{}, basis({factor.first,modulus},n,1)});
    } else if (!found_points) {
        components=decompose(radical_lex(rows,n),n,true);
    }
    for (const auto& part : components) {
        if (dimension(part.prime,n)!=dim) return Json::Value();
        if (geometrically_prime(part.prime,n,marked,dim)) {
            Json::Value encoded(Json::arrayValue);
            for (const auto& f : part.prime) encoded.append(sparse_json::term_list(f));
            primes.append(encoded);
            continue;
        }
        // The normal closure of the coordinates of a smooth closed point
        // makes its conjugates rational. Each geometric component then has
        // a smooth rational point and is defined over the enlarged field.
        const auto sample = smooth_sample(part.prime,n);
        for (size_t j=0; j<n; ++j) {
            const auto eliminant = remap(relation(sample,{j},n),{j});
            if (eliminant.empty()) throw std::logic_error("Missing finite-fibre eliminant.");
            for (const auto& [p,m] : factors(eliminant,1)) {
                if (p.rbegin()->first[0] <= 1) continue;
                fields.append(sparse_json::term_list(p));
            }
        }
    }
    Json::Value result;
    result["primes"] = primes;
    result["splitting_polynomials"] = fields;
    return result;
}
} // namespace

extern "C" const char* atlas_absolute_sections(const char* input) noexcept {
    return sparse_json::invoke(input,absolute_sections);
}

std::string operation(std::istream& input, const std::string& op) {
    std::string names, order;
    if (!std::getline(input,names) || !std::getline(input,order) || (order!="lex" && order!="grevlex"))
        throw std::invalid_argument("Expected QQ coordinates and a monomial order.");
    std::istringstream coordinates(names);
    size_t n=0;
    for (std::string v; coordinates >> v;) ++n;
    Basis generators;
    for (std::string line; std::getline(input,line);) generators.push_back(local::decode_row(line,n));
    std::ostringstream out;
    if (op == "ideal_radical" || op == "ideal_radical_gpu") {
        RestorePolicy policy;
        prefer_modular = op == "ideal_radical_gpu";
        for (const auto& f : radical(std::move(generators), n, order == "lex"))
            out << local::encode_row(f) << '\n';
        return out.str();
    }
    auto result = decompose(generators,n,order=="lex");
    out << result.size() << '\n';
    for (const auto& item : result) {
        out << item.primary.size() << ' ' << item.prime.size() << '\n';
        for (const auto& f : item.primary) out << local::encode_row(f) << '\n';
        for (const auto& f : item.prime) out << local::encode_row(f) << '\n';
    }
    return out.str();
}
} // namespace atlas::primary
