#pragma once
#include <numeric>
#include <set>
#include <stdexcept>

namespace atlas::components {
// A certified inequality against a representative excludes its entire class.
// Lack of a common regular open has no such implication.
template <class Chart, class Stratum, class Compare>
void partition(size_t count, Chart chart, Stratum stratum, Compare compare, size_t *classes) {
    std::iota(classes, classes + count, size_t(0));
    auto root = [&](size_t i) {
        while (classes[i] != i) {
            classes[i] = classes[classes[i]];
            i = classes[i];
        }
        return i;
    };
    for (size_t i = 0; i < count; ++i) {
        std::set<size_t> distinct;
        for (size_t j = 0; j < i; ++j) {
            if (chart(i) == chart(j) || stratum(i) != stratum(j))
                continue;
            const auto first = root(i), second = root(j);
            if (first == second || distinct.count(second))
                continue;
            switch (compare(i, j)) {
            case 1:
                classes[first] = second;
                break;
            case -1:
                distinct.insert(second);
                break;
            case 0:
                break;
            default:
                throw std::runtime_error("Invalid chart-overlap decision.");
            }
        }
    }
    for (size_t i = 0; i < count; ++i)
        classes[i] = root(i);
}
} // namespace atlas::components
