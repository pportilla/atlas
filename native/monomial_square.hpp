#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace atlas {
// The fibre monomial u(x) has the same zero set as its squarefree part.
// A product of two squarefree supports S,T has exponent two on S intersect T
// and exponent one on the rest of their union.
struct MonomialSquare {
    std::vector<uint64_t> supports;
    std::vector<std::pair<uint64_t, uint64_t>> products;

    void add_support(uint64_t s) {
        for (auto old : supports)
            if ((old & s) == old) return;
        supports.erase(std::remove_if(supports.begin(), supports.end(), [&](auto old) {
            return (old & s) == s;
        }), supports.end());
        supports.push_back(s);
    }
    void find_products() {
        products.clear();
        for (size_t i = 0; i < supports.size(); ++i)
            for (size_t j = i; j < supports.size(); ++j)
                products.emplace_back(supports[i] | supports[j], supports[i] & supports[j]);
    }
    bool contains(uint64_t positive, uint64_t repeated) const {
        for (const auto& [support, overlap] : products)
            if (!(support & ~positive) && !(overlap & ~repeated)) return true;
        return false;
    }
};
}
