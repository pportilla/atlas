#pragma once

#include "local_basis.hpp"
#include <algorithm>
#include <utility>

namespace atlas::local {
// Matches native_polynomial.default_sort_key on QQ expressions. Only the
// active coordinates participate, in ascending name order. Within that
// ring the tuple of (exponents, coefficient) is stored largest exponent first.
struct OrderedRow {
    Polynomial row;
    std::vector<std::string> variables;
    std::vector<std::pair<Powers, mpq_class>> reordered_terms;

    OrderedRow(Polynomial polynomial, const std::vector<std::string>& names, bool names_sorted)
        : row(std::move(polynomial)) {
        std::vector<size_t> active;
        for (size_t i = 0; i < names.size(); ++i)
            if (std::any_of(row.begin(), row.end(), [i](const auto& t) { return t.first[i] != 0; }))
                active.push_back(i);
        if (!names_sorted)
            std::sort(active.begin(), active.end(), [&](size_t i, size_t j) { return names[i] < names[j]; });
        for (size_t i : active) variables.push_back(names[i]);
        if (names_sorted) return;
        for (const auto& [powers, coefficient] : row) {
            Powers restricted;
            for (size_t i : active) restricted.push_back(powers[i]);
            reordered_terms.emplace_back(std::move(restricted), coefficient);
        }
        std::sort(reordered_terms.rbegin(), reordered_terms.rend());
    }

    bool operator<(const OrderedRow& other) const {
        if (variables != other.variables) return variables < other.variables;
        if (!reordered_terms.empty()) return reordered_terms < other.reordered_terms;
        // With the same active variables, all omitted coordinates are zero
        // in both rows. Reverse map iteration already has the required order.
        return std::lexicographical_compare(row.rbegin(), row.rend(), other.row.rbegin(), other.row.rend());
    }
};

}
