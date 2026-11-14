// Exact positive incidence witnesses on the ratio-zero section of a graph chart.
#include "graph_centre_incidence.hpp"
#include "local_io.hpp"
#include "primary_decomposition.hpp"

#include <sstream>

namespace atlas {
namespace {
using local::Basis;
using local::Polynomial;
using local::Powers;
using Point = std::vector<mpq_class>;

Polynomial read_row(std::istream& input, size_t width) {
    std::string row;
    if (!std::getline(input, row)) throw std::invalid_argument("Missing graph incidence equation.");
    return local::decode_row(row, width);
}

bool verified_point(const Basis& equations, const Point& point) {
    try {
        return !local::has_point_unit(equations, point);
    } catch (const std::overflow_error&) {
        // This lift exceeded the evaluator's exact work budget. Other lifts
        // may still be decidable (for example, when more coordinates vanish).
        return false;
    }
}

// Only look for triangular equations of modest size. Anything omitted here
// is still verified in the original centre ideal before accepting a point.
bool small_row(const Polynomial& row) {
    if (row.empty() || row.size() > 64) return false;
    for (const auto& [powers, c] : row) {
        if (mpz_sizeinbase(c.get_num_mpz_t(), 2) > 1024 ||
            mpz_sizeinbase(c.get_den_mpz_t(), 2) > 1024) return false;
        for (const auto& e : powers) if (e > 16) return false;
    }
    return true;
}

struct CentreSearch {
    const Basis& original;
    Point point;
    std::string result;
    size_t branches = 64;

    // A univariate row forces its coordinate to one of its roots. Work
    // only with its rational linear factors and substitute into the rest.
    // If no such row is exposed, make no assertion about the centre.
    void visit(const Basis& rows, const std::vector<size_t>& axes) {
        if (!branches) return;
        --branches;
        if (axes.empty()) {
            if (!verified_point(original, point)) return;
            for (size_t i = 0; i < point.size(); ++i)
                result += (i ? " " : "") + point[i].get_str();
            result += '\n';
            return;
        }
        Polynomial best;
        size_t axis = 0;
        for (const auto& row : rows) {
            size_t used = axes.size();
            bool mixed = false;
            for (const auto& [powers, c] : row) {
                for (size_t i = 0; i < axes.size(); ++i) {
                    if (powers[i] == 0) continue;
                    if (used != axes.size() && used != i) mixed = true;
                    used = i;
                }
            }
            if (mixed || row.empty()) continue;
            if (used == axes.size()) return;  // nonzero constant
            if (!best.empty() && row.rbegin()->first[used] >= best.rbegin()->first[0]) continue;
            best.clear();
            for (const auto& [powers, c] : row) best.emplace(Powers{powers[used]}, c);
            axis = used;
        }
        if (best.empty()) return;
        std::vector<Polynomial> linear;
        if (best.rbegin()->first[0] == 1) linear.push_back(std::move(best));
        else for (const auto& [factor, multiplicity] : primary::factors(best, 1))
            if (factor.rbegin()->first[0] == 1) linear.push_back(factor);
        auto remaining = axes;
        remaining.erase(remaining.begin() + axis);
        for (const auto& f : linear) {
            if (!branches) break;
            auto c = f.find(Powers{0});
            mpq_class root = 0;
            if (c != f.end()) root = -c->second / f.rbegin()->second;
            point[axes[axis]] = root;
            Basis next;
            std::map<mpz_class, Polynomial> powers;
            Polynomial image;
            if (root != 0) image.emplace(Powers(remaining.size()), root);
            for (const auto& row : rows) {
                auto specialized = primary::substitute(row, axis, image, remaining.size(), powers);
                if (small_row(specialized)) next.push_back(std::move(specialized));
            }
            visit(next, remaining);
        }
    }
};
}

std::string graph_centre_points(std::istream& input, size_t width) {
    if (width == 0 || width > 32) return {};
    Basis original, small;
    while (input.peek() != std::char_traits<char>::eof()) {
        original.push_back(read_row(input, width));
        if (small_row(original.back())) small.push_back(original.back());
    }
    std::vector<size_t> axes(width);
    for (size_t i = 0; i < width; ++i) axes[i] = i;
    CentreSearch search{original, Point(width), {}};
    search.visit(small, axes);
    return search.result;
}

std::string graph_centre_incidence(std::istream& input, size_t width, size_t rows,
                                  size_t parent_width, size_t points) {
    if (!parent_width || parent_width > width || width > 32)
        throw std::invalid_argument("Invalid graph chart point dimensions.");
    Basis equations;
    for (size_t i = 0; i < rows; ++i) equations.push_back(read_row(input, width));
    for (size_t j = 0; j < points; ++j) {
        std::string row;
        if (!std::getline(input, row)) throw std::invalid_argument("Missing graph chart point.");
        std::istringstream values(row);
        Point lift(width);  // Original coordinates, followed by zero affine ratios.
        for (size_t i = 0; i < parent_width; ++i) {
            if (!(values >> lift[i]) || lift[i].get_den() == 0)
                throw std::invalid_argument("Invalid graph chart point.");
            lift[i].canonicalize();
        }
        std::string extra;
        if (values >> extra) throw std::invalid_argument("Invalid graph chart point width.");
        if (verified_point(equations, lift)) return "yes\n";
    }
    return "unknown\n";
}
}
