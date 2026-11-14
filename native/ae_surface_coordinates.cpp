#include "ae_surface_coordinates.hpp"

#include <algorithm>
#include <array>

namespace atlas::local {
namespace {
// Normalize the line to x, or to slope*x+y. The coefficient of the whole
// power is immaterial because each target coordinate can be rescaled.
struct BinaryLinePower {
    mpz_class degree;
    std::optional<mpq_class> slope;  // null denotes the line x
};

std::optional<BinaryLinePower> binary_line_power(const Polynomial& f) {
    if (f.empty() || f.begin()->first.size() != 2) return std::nullopt;
    const auto& [first, coefficient] = *f.begin();
    if (f.size() == 1) {
        if (!first[1] && first[0] > 0) return BinaryLinePower{first[0], std::nullopt};
        if (!first[0] && first[1] > 0) return BinaryLinePower{first[1], mpq_class(0)};
        return std::nullopt;
    }
    // With both terms of the linear form present its d-th power has d+1
    // nonzero terms. Check the given support before making a bounded loop.
    if (first[0] != 0 || first[1] != f.size()-1) return std::nullopt;
    const size_t degree = f.size()-1;
    auto it = std::next(f.begin());
    const mpq_class slope = it->second / (degree * coefficient);
    auto previous = f.begin();
    for (size_t k = 1; it != f.end(); ++it, ++k, ++previous) {
        const auto& [powers, c] = *it;
        if (powers[0] != k || powers[1] != degree-k ||
            k*c != (degree-k+1)*slope*previous->second) return std::nullopt;
    }
    return BinaryLinePower{first[1], slope};
}

}

std::optional<Basis> normalized_surface_axes(const Basis& map) {
    if (map.size() != 3) return std::nullopt;
    std::array<BinaryLinePower, 3> forms;
    for (size_t j = 0; j < forms.size(); ++j) {
        auto line = binary_line_power(map[j]);
        if (!line) return std::nullopt;
        for (size_t i = 0; i < j; ++i)
            if (line->slope == forms[i].slope) return std::nullopt;
        forms[j] = std::move(*line);
    }
    // Two of the lines are a source frame. In that frame the third line has
    // two nonzero coefficients. Rescale the frame and the three targets to
    // obtain (x^a,y^b,(x+y)^c). No tangent dimension is inferred here.
    std::sort(forms.begin(), forms.end(), [](const auto& u, const auto& v) { return u.degree < v.degree; });
    if (forms.back().degree > 64) return std::nullopt;
    Basis result(3);
    result[0].emplace(Powers{forms[0].degree, 0}, 1);
    result[1].emplace(Powers{0, forms[1].degree}, 1);
    const unsigned long c = forms[2].degree.get_ui();
    mpz_class coefficient = 1;
    for (unsigned long k = 0; k <= c; ++k) {
        result[2].emplace(Powers{k, c-k}, coefficient);
        if (k < c) coefficient = coefficient * (c-k) / (k+1);
    }
    return result;
}
}
