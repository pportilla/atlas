// Associated graded ring for the point ideal without translating the input.
#include "local_dimension_graph.hpp"
#include "local_dimension.hpp"
#include "primary_decomposition.hpp"
#include <limits>
#include <stdexcept>

namespace atlas::local {
int dimension_from_point_graph(const Basis& generators, const std::vector<mpq_class>& point,
                               bool prefer_modular) {
    const size_t n = point.size();
    std::vector<size_t> original(n);
    size_t coordinates = n;
    for (size_t i = 0; i < n; ++i)
        original[i] = point[i] == 0 ? i : coordinates++;
    if (coordinates >= static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::length_error("Point graph exceeds addressable coordinate indices.");
    // H, U_1,...,U_n, then an additional X_i for each nonzero p_i.
    // Zero-point axes are already their own centered variables. Weight
    // zero on the remaining X and one on H,U.
    // Graph relations identify (U) with the maximal ideal of the point
    // after H=1. Its associated graded quotient computes the local dimension.
    const size_t width = 1+coordinates;
    Basis graph;
    for (const auto& f : generators) if (!f.empty()) {
        Polynomial row;
        mpz_class top = 0;
        for (const auto& [e, c] : f) {
            mpz_class degree = 0;
            for (size_t i = 0; i < n; ++i) if (point[i] == 0) degree += e[i];
            top = std::max(top, degree);
        }
        for (const auto& [e, c] : f) {
            Powers powers(width, 0);
            powers[0] = top;
            for (size_t i = 0; i < n; ++i) {
                powers[1+original[i]] = e[i];
                if (point[i] == 0) powers[0] -= e[i];
            }
            row.emplace(std::move(powers), c);
        }
        graph.push_back(std::move(row));
    }
    for (size_t i = 0; i < n; ++i) if (point[i] != 0) {
        Polynomial relation;
        Powers powers(width, 0);
        powers[0] = powers[1+original[i]] = 1;
        relation.emplace(powers, 1);
        powers[1+original[i]] = 0;
        relation.emplace(powers, -point[i]);
        powers[0] = 0;
        powers[1+i] = 1;
        relation.emplace(std::move(powers), -1);
        graph.push_back(std::move(relation));
    }

    // Under lex with H first, a homogeneous relation selects the largest
    // H exponent, hence smallest U degree. Deleting H from its lead gives
    // the lead of its initial form. Each element of the dehomogenized graph
    // ideal homogenizes into this ideal after multiplication by some H^k;
    // such a shift has the same projected lead.
    const auto basis = primary::reduced(graph, width, true, prefer_modular);
    std::vector<std::vector<int>> supports;
    for (const auto& row : basis) if (!row.empty()) {
        const auto& head = row.rbegin()->first;
        std::vector<int> support;
        for (size_t j = 1; j < width; ++j)
            if (head[j] != 0) support.push_back(static_cast<int>(j-1));
        if (support.empty()) throw std::logic_error("On-scheme point graph produced a unit.");
        supports.push_back(std::move(support));
    }
    return monomial_support_dimension(std::move(supports), static_cast<int>(width-1));
}
}
