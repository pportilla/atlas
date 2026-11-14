#pragma once

#include <algorithm>
#include <cstdint>
#include <gmpxx.h>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

namespace atlas {

// A regular cone sends distinct support exponents to distinct exponents.
// Keep their rational coefficient strings and sort indices in descending
// graded lexicographic order, as in the public strict equation.
template<class Power>
void append_chart_equation(std::string& out, const Power* powers, const std::vector<std::string>& coefficients,
                     const std::vector<std::string>& names) {
    const size_t terms = coefficients.size(), dimension = names.size();
    using Degree = std::conditional_t<std::is_same_v<Power, int32_t>, int64_t, mpz_class>;
    std::vector<Degree> degree(terms);
    std::vector<size_t> order(terms);
    std::iota(order.begin(), order.end(), size_t(0));
    for (size_t i = 0; i < terms; ++i)
        for (size_t j = 0; j < dimension; ++j)
            degree[i] += powers[i*dimension+j];
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (degree[a] != degree[b]) return degree[a] > degree[b];
        const auto* left = powers + a*dimension;
        const auto* right = powers + b*dimension;
        return std::lexicographical_compare(right, right+dimension, left, left+dimension);
    });
    bool first = true;
    for (size_t i : order) {
        const auto& coefficient = coefficients[i];
        if (coefficient == "0") continue;
        const bool negative = coefficient.front() == '-';
        if (negative) out += '-';
        else if (!first) out += '+';
        first = false;
        bool factor = coefficient.compare(negative, std::string::npos, "1") != 0 || degree[i] == 0;
        if (factor) out.append(coefficient, negative, std::string::npos);
        for (size_t j = 0; j < dimension; ++j) {
            const auto& power = powers[i*dimension+j];
            if (power == 0) continue;
            if (factor) out += '*';
            out += names[j];
            if (power != 1) {
                out += '^';
                if constexpr (std::is_same_v<Power, int32_t>) out += std::to_string(power);
                else out += power.get_str();
            }
            factor = true;
        }
    }
    if (first) out += '0';
    out += '\n';
}

} // namespace atlas
