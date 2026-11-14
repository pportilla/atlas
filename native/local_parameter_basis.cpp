#include "local_parameter_basis.hpp"
#include "local_monomial.hpp"
#include "local_division.hpp"
#include "local_io.hpp"
#include "primary_decomposition.hpp"
#include <algorithm>

namespace atlas::local {
namespace {
void add_scaled(Polynomial& out, const Polynomial& f, const mpq_class& scale) {
    if (scale == 0) return;
    for (const auto& [e,c] : f) {
        auto it = out.try_emplace(e,0).first;
        it->second += scale*c;
        if (it->second == 0) out.erase(it);
    }
}

bool constant(const Polynomial& f) {
    return f.size() == 1 && std::all_of(f.begin()->first.begin(),f.begin()->first.end(),
                                        [](const auto& e) { return e == 0; });
}

// A sparse exact quotient is a factor certificate. Stop speculative
// division when a dense intermediate quotient is developing. The
// original-coordinate standard-basis algorithm remains available.
std::optional<Polynomial> sparse_quotient(const Polynomial& f, const Polynomial& g) {
    Polynomial out, pending = f;
    const auto& [divisor,c] = *g.rbegin();
    const size_t budget = 8*(f.size()+g.size());
    while (!pending.empty() && out.size() <= budget) {
        const auto& [head,b] = *pending.rbegin();
        Powers shift(head.size());
        for (size_t i=0; i<head.size(); ++i) {
            if (head[i] < divisor[i]) return std::nullopt;
            shift[i] = head[i]-divisor[i];
        }
        const mpq_class scale = b/c;
        out.emplace(shift,scale);
        for (const auto& [e,a] : g) {
            Powers k(shift);
            for (size_t i=0; i<k.size(); ++i) k[i] += e[i];
            auto it = pending.try_emplace(std::move(k),0).first;
            it->second -= scale*a;
            if (it->second == 0) pending.erase(it);
        }
    }
    if (!pending.empty()) return std::nullopt;
    return out;
}

mpz_class extract(Polynomial& row, const Polynomial& factor) {
    if (factor.size() == 1) {
        const auto& head = factor.begin()->first;
        std::optional<mpz_class> count;
        for (size_t j=0; j<head.size(); ++j) if (head[j] != 0)
            for (const auto& [e,c] : row) {
                mpz_class k = e[j]/head[j];
                if (!count || k < *count) count = std::move(k);
            }
        if (!count || *count == 0) return 0;
        Polynomial quotient;
        for (const auto& [e,c] : row) {
            Powers key = e;
            for (size_t j=0; j<head.size(); ++j) key[j] -= *count*head[j];
            quotient.emplace(std::move(key),c);
        }
        row = std::move(quotient);
        return *count; // All candidates are monic.
    }
    mpz_class count = 0;
    while (auto quotient = sparse_quotient(row,factor)) {
        ++count;
        row = std::move(*quotient);
        if (constant(row)) break;
    }
    return count;
}

using Factorization = std::map<size_t,mpz_class>;
struct Products {
    Basis factors;
    std::vector<Factorization> rows;
};

// Find products of regular parameters without factoring wide polynomials.
// A smooth, possibly reducible affine polynomial is itself a valid local
// parameter. Every factor used is recovered by exact polynomial division.
std::optional<Products> parameter_products(const Basis& ideal, const PointRow& point) {
    const size_t width = point.size();
    Basis candidates;
    std::vector<bool> units;
    auto candidate = [&](Polynomial g) {
        if (g.empty() || constant(g)) return false;
        const mpq_class leading = g.rbegin()->second;
        for (auto& term : g) term.second /= leading;
        if (std::find(candidates.begin(),candidates.end(),g) != candidates.end()) return false;
        const bool unit = nonzero_at_point(g,point);
        if (!unit && order_at_point(g,point) != 1) return false;
        candidates.push_back(std::move(g));
        units.push_back(unit);
        return true;
    };
    for (const auto& f : ideal) if (f.size() <= 64 && max_total_degree(f) <= 6)
        for (auto& [g,k] : primary::factors(f,width)) candidate(std::move(g));

    bool tried_gcd = false;
    for (;;) {
        size_t known = candidates.size();
        std::vector<Factorization> decompositions;
        bool unresolved = false;
        for (auto row : ideal) {
            Factorization factors;
            for (size_t j=0; j<candidates.size() && !constant(row); ++j) {
                auto k = extract(row,candidates[j]);
                if (k != 0 && !units[j]) factors.emplace(j,std::move(k));
            }
            if (!constant(row) && !nonzero_at_point(row,point)) {
                if (candidate(std::move(row))) factors.emplace(candidates.size()-1,1);
                else unresolved = true;
            }
            decompositions.push_back(std::move(factors));
        }
        if (!unresolved) {
            Products result;
            std::map<size_t,size_t> used;
            for (const auto& factors : decompositions) {
                Factorization row;
                for (const auto& [j,k] : factors) {
                    auto [it,fresh] = used.emplace(j,result.factors.size());
                    if (fresh) result.factors.push_back(candidates[j]);
                    row.emplace(it->second,k);
                }
                result.rows.push_back(std::move(row));
            }
            if (result.factors.size() > width) return std::nullopt;
            return result;
        }
        if (candidates.size() != known) continue;
        if (tried_gcd) return std::nullopt;
        tried_gcd = true;
        for (size_t j=0; j<ideal.size(); ++j) {
            if (max_total_degree(ideal[j]) > 8192) continue;
            for (size_t i=0; i<j; ++i) if (max_total_degree(ideal[i]) <= 8192) {
                auto g = polynomial_gcd({ideal[i],ideal[j]},width);
                if (g) candidate(std::move(*g));
            }
        }
        if (candidates.size() == known) return std::nullopt;
    }
}

struct Parameter {
    PointRow differential;
    Polynomial image;
};

// Eliminate each Jacobian row while applying the same rational row
// operations to its polynomial. After completion the i-th image is
// T_i=X_i+terms of degree at least two in centered coordinates.
bool independent_row(std::vector<std::optional<Parameter>>& pivots, Parameter row) {
    auto subtract = [](Parameter& a, const Parameter& b, const mpq_class scale) {
        for (size_t i=0; i<a.differential.size(); ++i) a.differential[i] -= scale*b.differential[i];
        add_scaled(a.image,b.image,-scale);
    };
    size_t width = pivots.size();
    for (size_t i=0; i<width; ++i)
        if (pivots[i] && row.differential[i] != 0) subtract(row,*pivots[i],row.differential[i]);
    size_t pivot = 0;
    while (pivot < width && row.differential[pivot] == 0) ++pivot;
    if (pivot == width) return false;
    const mpq_class scale = row.differential[pivot];
    for (auto& c : row.differential) c /= scale;
    for (auto& t : row.image) t.second /= scale;
    for (auto& p : pivots) if (p && p->differential[pivot] != 0)
        subtract(*p,row,p->differential[pivot]);
    pivots[pivot] = std::move(row);
    return true;
}
}

std::optional<Basis> independent_parameter_basis(const Basis& generators,
        const std::vector<std::string>& names, const PointRow& point, bool lex, bool prefer_modular) {
    const auto width = point.size();
    if (generators.size() < 2 || generators.size() > 32 || width > 12) return std::nullopt;
    const auto products = parameter_products(generators,point);
    if (!products) return std::nullopt;
    Basis linear;
    std::vector<std::optional<Parameter>> chart(width);
    for (const auto& parameter : products->factors) {
        auto differential = point_jet_below(2,parameter,point);
        PointRow row(width);
        for (size_t j=0; j<width; ++j) {
            Powers e(width);
            e[j]=1;
            auto t=differential.find(e);
            if (t != differential.end()) row[j] = t->second;
        }
        if (!independent_row(chart,{std::move(row),parameter})) return std::nullopt;
        linear.push_back(std::move(differential));
    }
    for (size_t i=0; i<width; ++i) if (!chart[i]) {
        Powers e(width);
        e[i]=1;
        PointRow row(width);
        row[i]=1;
        Polynomial coordinate{{e,1}};
        if (point[i] != 0) coordinate.emplace(Powers(width),-point[i]);
        independent_row(chart,{std::move(row),std::move(coordinate)});
    }

    // The image of each dF_i is exactly F_i, so these homogeneous
    // products pull back to generators of the given germ up to units.
    const Polynomial one{{Powers(width),1}};
    Basis homogeneous;
    for (const auto& factors : products->rows) {
        auto f = one;
        for (const auto& [i,k] : factors)
            f = primary::multiply(f,primary::power(linear[i],k,width));
        homogeneous.push_back(std::move(f));
    }
    auto basis = standard_basis(homogeneous,names,lex,prefer_modular);
    // The formal inverse function theorem gives an isomorphism of
    // completions with identity tangent map. All lowest forms, including
    // the tie order, are preserved; completion preserves the initial ideal.
    Basis result;
    std::vector<std::map<mpz_class,Polynomial>> powers(width);
    for (const auto& f : basis) {
        Polynomial lifted;
        for (const auto& [e,c] : f) {
            auto term = one;
            for (size_t i=0; i<width; ++i) if (e[i] != 0) {
                auto [it,fresh] = powers[i].try_emplace(e[i]);
                if (fresh) it->second = primary::power(chart[i]->image,e[i],width);
                term = primary::multiply(term,it->second);
            }
            add_scaled(lifted,term,c);
        }
        result.push_back(std::move(lifted));
    }
    return result;
}
}
