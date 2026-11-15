// Exact positive incidence witnesses on the ratio-zero section of a graph chart.
#include "graph_centre_incidence.hpp"
#include "local_io.hpp"
#include "local_monomial.hpp"
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
    const Basis& units;
    Point point;
    std::string result;
    size_t branches = 64;

    // Each candidate is forced by equations in the original ideal. Once
    // every coordinate is fixed, evaluate *all* original equations and
    // open conditions, including rows omitted from triangular search.
    bool leaf() {
        try {
            if (local::has_point_unit(original, point)) return true;
            for (const auto& u : units)
                if (!local::has_point_unit({u}, point)) return true;
        } catch (const std::overflow_error&) {
            return false;
        }
        for (size_t i = 0; i < point.size(); ++i)
            result += (i ? " " : "") + point[i].get_str();
        result += '\n';
        return true;
    }

    // Return true only if every geometric root was handled. The ordinary
    // incidence caller can also use individual witnesses of a partial search.
    bool visit(const Basis& rows, const std::vector<size_t>& axes, bool reduce_linear) {
        if (!branches) return false;
        --branches;
        if (axes.empty()) return leaf();
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
            if (used == axes.size()) return true;  // nonzero constant, empty branch
            if (!best.empty() && row.rbegin()->first[used] >= best.rbegin()->first[0]) continue;
            best.clear();
            for (const auto& [powers, c] : row) best.emplace(Powers{powers[used]}, c);
            axis = used;
        }
        if (best.empty()) {
            if (!reduce_linear) return false;
            // A change of affine coordinates can hide every univariate row.
            // Row-reduce just the linear constraints, never the dense chart
            // ideal. Keep nonlinear equations available for subsequent fibres.
            Basis linear;
            for (const auto& row : rows)
                if (local::max_total_degree(row) == 1) linear.push_back(row);
            if (linear.size() < 2) return false;
            auto enlarged = primary::reduced(linear, axes.size(), true);
            enlarged.insert(enlarged.end(), rows.begin(), rows.end());
            return visit(enlarged, axes, false);
        }
        std::vector<Polynomial> linear;
        bool complete = true;
        if (best.rbegin()->first[0] == 1) linear.push_back(std::move(best));
        else for (const auto& [factor, multiplicity] : primary::factors(best, 1)) {
            if (factor.rbegin()->first[0] == 1) linear.push_back(factor);
            else complete = false;
        }
        auto remaining = axes;
        remaining.erase(remaining.begin() + axis);
        for (const auto& f : linear) {
            if (!branches) return false;
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
            if (!visit(next, remaining, reduce_linear)) complete = false;
        }
        return complete;
    }
};

bool search_points(CentreSearch& search, bool reduce_linear) {
    Basis small;
    for (const auto& row : search.original)
        if (small_row(row)) small.push_back(row);
    std::vector<size_t> axes(search.point.size());
    for (size_t i = 0; i < axes.size(); ++i) axes[i] = i;
    return search.visit(small, axes, reduce_linear);
}
}

std::string graph_centre_points(std::istream& input, size_t width) {
    if (width == 0 || width > 32) return {};
    Basis original, units;
    while (input.peek() != std::char_traits<char>::eof())
        original.push_back(read_row(input, width));
    CentreSearch search{original, units, Point(width), {}};
    search_points(search, false);
    return search.result;
}

std::string frame_points(std::istream& input, size_t width, size_t equations, size_t unit_count) {
    if (width == 0 || width > 32 || equations > 256 || unit_count > 128) return "unknown\n";
    Basis original, units;
    for (size_t i = 0; i < equations; ++i) original.push_back(read_row(input, width));
    for (size_t i = 0; i < unit_count; ++i) units.push_back(read_row(input, width));
    if (input.peek() != std::char_traits<char>::eof())
        throw std::invalid_argument("Unexpected open-frame point rows.");
    if (std::any_of(units.begin(), units.end(), [](const auto& row) { return row.empty(); }))
        return "points\n";
    CentreSearch search{original, units, Point(width), {}};
    if (!search_points(search, true)) return "unknown\n";
    return "points\n" + search.result;
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
