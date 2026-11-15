// Ordinary and symbolic powers using exponent vectors of monomial ideals.
#include "ideal_powers.hpp"
#include "local_monomial.hpp"
#include "primary_decomposition.hpp"
#include <algorithm>
#include <array>
#include <functional>

namespace atlas::ideal_powers {
namespace {
using local::Powers;
using Monomials = std::vector<Powers>;

// Sort by degree so a strict divisor can only occur in an earlier degree.
// In a homogeneous group only equal vectors can be redundant.
Monomials minimal(Monomials rows) {
    if (rows.size() < 2) return rows;
    std::sort(rows.begin(),rows.end());
    rows.erase(std::unique(rows.begin(),rows.end()),rows.end());
    std::map<mpz_class,Monomials> by_degree;
    for (auto& row : rows) {
        auto degree = local::total_degree(row);
        by_degree[std::move(degree)].push_back(std::move(row));
    }
    Monomials result;
    result.reserve(rows.size());
    for (auto& [degree,group] : by_degree) {
        const size_t smaller = result.size();
        for (auto& row : group)
            if (std::none_of(result.begin(),result.begin()+smaller,[&](const auto& g){return local::monomial_divides(g,row);}))
                result.push_back(std::move(row));
    }
    return result;
}

template<class Row, class Integer>
void distribute(std::vector<Row>& result, Row& e, const std::vector<size_t>& support,
                size_t index, Integer deficit) {
    auto& current = e[support[index]];
    const auto saved = current;
    if (index+1 == support.size()) {
        current += deficit;
        result.push_back(e);
    } else {
        for (;;) {
            distribute(result,e,support,index+1,deficit);
            if (deficit == 0) break;
            --deficit;
            ++current;
        }
    }
    current = saved;
}

Monomials products(const Monomials& a, const Monomials& b) {
    Monomials rows;
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = &a == &b ? i : 0; j < b.size(); ++j) {
            auto sum = a[i];
            for (size_t k = 0; k < sum.size(); ++k) sum[k] += b[j][k];
            rows.push_back(std::move(sum));
        }
    }
    return minimal(std::move(rows));
}

bool disjoint_supports(const Monomials& rows, size_t width) {
    std::vector<bool> seen(width);
    for (const auto& e : rows) for (size_t i = 0; i < width; ++i) if (e[i] != 0) {
        if (seen[i]) return false;
        seen[i] = true;
    }
    return true;
}

// When the supports are disjoint, distinct choices of generator
// multiplicities are incomparable under divisibility. Emit them directly.
Monomials disjoint_products(const Monomials& base, unsigned long n) {
    std::vector<std::vector<size_t>> supports(base.size());
    const auto width = base.front().size();
    for (size_t i = 0; i < base.size(); ++i)
        for (size_t k = 0; k < width; ++k)
            if (base[i][k] != 0) supports[i].push_back(k);
    Powers exponents(width);
    Monomials result;
    // A generator touches only its own coordinates. Reuse one exponent
    // vector while distributing the remaining factors among the generators.
    auto emit = [&](auto&& self, size_t i, unsigned long remaining) -> void {
        for (auto count = remaining;; --count) {
            for (auto axis : supports[i]) exponents[axis] = base[i][axis] * count;
            if (i + 1 == base.size()) {
                result.push_back(exponents);
                return;
            }
            self(self, i + 1, remaining - count);
            if (count == 0) break;
        }
    };
    emit(emit, 0, n);
    return result;
}

Monomials ordinary(Monomials base, mpz_class n, bool disjoint) {
    if (base.size() == 1) {
        for (auto& e : base.front()) e *= n;
        return base;
    }
    if (disjoint && n.fits_ulong_p()) return disjoint_products(base,n.get_ui());
    Monomials answer;
    while (n > 0) {
        if (mpz_odd_p(n.get_mpz_t())) answer = answer.empty() ? base : products(answer,base);
        n >>= 1;
        if (n > 0) base = products(base,base);
    }
    return answer;
}

Basis as_basis(Monomials rows, bool lex) {
    std::sort(rows.begin(),rows.end(),[&](const auto& a,const auto& b){return local::global_less(b,a,lex);});
    Basis basis;
    basis.reserve(rows.size());
    for (auto& e : rows) basis.emplace_back().emplace(std::move(e), 1);
    return basis;
}

// A coordinate prime with support S imposes sum(e_i for i in S) >= n.
// Given a generator e of the previous intersection, its minimal multiples
// satisfying the new bound add exactly the deficit, distributed over S.
Monomials symbolic_intersection(const Monomials& rows, size_t width, const mpz_class& n) {
    const auto components = primary::decompose(as_basis(rows,true),width,true);
    Monomials result{Powers(width)};
    for (const auto& component : components) {
        std::vector<size_t> support;
        for (const auto& p : component.prime) {
            const auto& e = p.begin()->first;
            support.push_back(std::find(e.begin(),e.end(),1)-e.begin());
        }
        Monomials next;
        for (auto& e : result) {
            mpz_class deficit = n;
            for (auto axis : support) deficit -= e[axis];
            if (deficit <= 0) next.push_back(std::move(e));
            else distribute(next,e,support,0,deficit);
        }
        result = minimal(std::move(next));
    }
    return result;
}
// For equigenerated monomial ideals, every product has the same degree:
// only duplicate products are redundant. Use the native packed ring when
// the entire product fits in one exponent word.
std::optional<PackedBasis> packed_homogeneous(const Monomials& base,size_t width,bool lex,
                                               const mpz_class& n) {
    if (!width || width>4 || n>64 || base.size()<2) return std::nullopt;
    Ring ring;
    ring.init(width,lex ? 1 : 0);
    const auto degree = local::total_degree(base.front());
    if (degree*n>MAX_DEGREE) return std::nullopt;
    std::vector<u64> heads;
    for (const auto& m : base) {
        if (local::total_degree(m) != degree) return std::nullopt;
        std::array<long,4> exponents{};
        for (size_t k=0;k<width;++k) exponents[k]=m[k].get_si();
        u64 words[2];
        ring.pack(words,exponents.data());
        heads.push_back(words[1]);
    }
    // Keep tuple enumeration bounded for highly redundant presentations.
    // Larger product sets use ideal multiplication with intermediate pruning.
    const auto exponent = n.get_ui();
    size_t count=1;
    for (size_t i=1;i<=exponent;++i) {
        if (base.size()>32768 || count*(base.size()+i-1)/i>32768) return std::nullopt;
        count=count*(base.size()+i-1)/i;
    }
    std::vector<u64> products;
    products.reserve(count);
    auto add = [&](auto&& self,u64 prefix,size_t first,unsigned long left) -> void {
        for (size_t i=first;i<heads.size();++i) {
            const auto next=prefix+heads[i];
            if (left==1) products.push_back(next);
            else self(self,next,i,left-1);
        }
    };
    add(add,0,0,exponent);
    if (lex) std::sort(products.begin(),products.end(),std::greater<>());
    else std::sort(products.begin(),products.end());
    products.erase(std::unique(products.begin(),products.end()),products.end());
    PackedBasis result{ring,{}};
    result.rows.reserve(products.size());
    const u64 total=degree.get_ui()*exponent;
    for (auto word:products) {
        auto& p=result.rows.emplace_back();
        p.m={total,word};
        fmpz_one(&p.c.emplace_back().v);
    }
    return result;
}
} // namespace

PowerResult monomial_power(const Basis& basis,size_t width,bool lex,const mpz_class& n,bool symbolic) {
    Monomials rows;
    for (const auto& f : basis) if (!f.empty()) rows.push_back(f.begin()->first);
    // Nonconstant monomials with mutually disjoint supports cannot divide
    // each other. Reduction can reveal disjointness in other presentations.
    bool disjoint = disjoint_supports(rows, width);
    if (!disjoint) {
        rows = minimal(std::move(rows));
        disjoint = disjoint_supports(rows, width);
    }
    if (symbolic) for (const auto& e : rows)
        if (std::any_of(e.begin(),e.end(),[](const auto& a){return a>1;}))
            throw InvalidInput("Symbolic powers require a radical ideal over QQ.");
    if (n == 0) return Basis{local::Polynomial{{Powers(width),1}}};
    if (rows.empty() || n == 1) return as_basis(std::move(rows),lex);
    // Disjoint supports give a radical complete intersection. Its powers
    // have only minimal associated primes, hence coincide with symbolic powers.
    if (!symbolic || disjoint) {
        if (auto packed = packed_homogeneous(rows,width,lex,n)) return std::move(*packed);
        return as_basis(ordinary(std::move(rows),n,disjoint),lex);
    }
    return as_basis(symbolic_intersection(rows,width,n),lex);
}
} // namespace atlas::ideal_powers
