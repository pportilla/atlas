// Compute the extended tangent quotient as a graded vector space. See KERNEL.md.
#include "ae_graded.hpp"
#include "ae_graded_surface.hpp"
#include "local_monomial.hpp"
#include "local_point_quotient.hpp"
#include "primary_decomposition.hpp"
#include <flint/fmpz.h>
#include <flint/fmpz_mat.h>
#include <algorithm>

namespace atlas::local {
namespace {

// A positive grading identifies finite jets of the formal local quotient with
// the weight pieces of its polynomial counterpart. Any positive null vector
// of these exponent differences suffices. An unsuccessful search just declines.
bool positive_grading(const Basis& map, size_t n) {
    std::vector<Powers> differences;
    bool ordinary = true;
    for (const auto& f : map) {
        if (f.empty()) continue;
        const auto& first = f.begin()->first;
        for (auto it = std::next(f.begin()); it != f.end(); ++it) {
            Powers row(n);
            for (size_t j = 0; j < n; ++j) row[j] = it->first[j] - first[j];
            ordinary &= total_degree(row) == 0;
            differences.push_back(std::move(row));
        }
    }
    if (ordinary) return true;
    fmpz_mat_t equations, kernel;
    fmpz_mat_init(equations, differences.size(), n);
    fmpz_mat_init(kernel, n, n);
    for (size_t i = 0; i < differences.size(); ++i)
        for (size_t j = 0; j < n; ++j)
            fmpz_set_mpz(fmpz_mat_entry(equations, i, j), differences[i][j].get_mpz_t());
    const slong nullity = fmpz_mat_nullspace(kernel, equations);
    // Standard basis columns may have either sign. Their oriented sum handles
    // independent grading blocks without solving a rational feasibility LP.
    Powers weights(n);
    for (slong k = 0; k < nullity; ++k) {
        int orientation = 0;
        for (size_t j = 0; j < n; ++j)
            if ((orientation = fmpz_sgn(fmpz_mat_entry(kernel, j, k))) != 0) break;
        for (size_t j = 0; j < n; ++j) {
            mpz_class value;
            fmpz_get_mpz(value.get_mpz_t(), fmpz_mat_entry(kernel, j, k));
            weights[j] += orientation * value;
        }
    }
    fmpz_mat_clear(kernel);
    fmpz_mat_clear(equations);
    return std::all_of(weights.begin(), weights.end(), [](const auto& w) { return w > 0; });
}

bool has_source(const Powers& e, size_t n) {
    return std::any_of(e.begin(), e.begin() + n, [](const auto& x) { return x != 0; });
}
}

std::optional<Json::Value> ae_graded(const Basis& map, const std::vector<std::string>& names) {
    const size_t n = names.size(), p = map.size(), base = n+p, width = base+p;
    if (n == 2 && p == 3)
        if (auto ranks = ae_graded_surface(map)) return ranks;
    // Lexicographic graph elimination can grow much faster than a small jet.
    if (width > 18) return std::nullopt;
    size_t terms = 0;
    for (const auto& f : map) {
        terms += f.size();
        if (terms > 120) return std::nullopt;
        for (const auto& t : f) if (total_degree(t.first) > 40) return std::nullopt;
    }
    if (!positive_grading(map,n)) return std::nullopt;

    // Square-zero markers z_j encode vectors as sum(v_j*z_j).
    // Lex order x > y > z eliminates all source variables before the targets.
    Basis relations;
    for (size_t j = 0; j < p; ++j) {
        Polynomial graph;
        Powers target(width);
        target[n+j] = 1;
        graph.emplace(std::move(target), 1);
        for (const auto& [exp,c] : map[j]) {
            Powers e = exp;
            e.resize(width);
            graph.emplace(std::move(e), -c);
        }
        relations.push_back(std::move(graph));
    }
    for (size_t i = 0; i < p; ++i) for (size_t j = i; j < p; ++j) {
        Powers e(width);
        ++e[base+i];
        ++e[base+j];
        relations.push_back({{std::move(e), 1}});
    }
    for (size_t axis = 0; axis < n; ++axis) {
        Polynomial derivative;
        for (size_t j = 0; j < p; ++j) for (const auto& [exp,c] : map[j]) {
            if (!exp[axis]) continue;
            Powers e = exp;
            --e[axis];
            e.resize(width);
            e[base+j] = 1;
            derivative.emplace(std::move(e), c * exp[axis]);
        }
        if (!derivative.empty()) relations.push_back(std::move(derivative));
    }

    // Under this elimination order, reduction of a y,z-only polynomial
    // never introduces x. Thus pure-target standard monomials are exactly
    // the image of target fields. Count their complement without subtracting
    // two potentially infinite dimensions.
    std::vector<Powers> scalar;
    std::vector<std::vector<Powers>> heads(p);
    auto initial = packed_lex_heads(relations, width);
    if (!initial) {
        // Intermediate Gröbner terms can exceed the packed exponent range.
        initial.emplace();
        for (const auto& g : primary::reduced(relations, width, true))
            initial->push_back(g.rbegin()->first);
    }
    for (const auto& lead : *initial) {
        size_t marker = p;
        unsigned count = 0;
        for (size_t j = 0; j < p; ++j) if (lead[base+j]) {
            marker = j;
            count += lead[base+j] == 1 ? 1 : 2;
        }
        Powers h(lead.begin(), lead.begin()+base);
        if (count == 0) scalar.push_back(std::move(h));
        else if (count == 1) heads[marker].push_back(std::move(h));
    }
    mpz_class dimension = 0;
    Json::Value result;
    result["method"] = "graded_graph";
    for (auto& component : heads) {
        component.insert(component.end(), scalar.begin(), scalar.end());
        // Partition monomials having source factors by their first nonzero
        // source index i. Set x_0,...,x_(i-1)=0, then divide by x_i.
        for (size_t i = 0; i < n; ++i) {
            std::vector<Powers> section;
            for (const auto& h : component) {
                if (has_source(h,i)) continue;
                Powers e(h.begin()+i,h.end());
                if (e.front() > 0) --e.front();
                section.push_back(std::move(e));
            }
            const auto part = monomial_point_quotient(base-i, section, false).dimension;
            if (!part) {
                result["finite"] = false;
                result["value"] = Json::nullValue;
                return result;
            }
            dimension += *part;
        }
    }
    result["finite"] = true;
    result["value"] = dimension.get_str();
    return result;
}
}
