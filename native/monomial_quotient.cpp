// Sparse standard monomials and Hilbert series of monomial quotients.
// Exponents, degrees, coefficients, and lengths can all exceed a machine word.
#include <gmpxx.h>
#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace atlas {
namespace {

using Integer = mpz_class;
using Row = std::vector<Integer>;
using Rows = std::vector<Row>;
using Polynomial = std::map<Integer, Integer>;

// Upper ends of boxes use -1 for infinity. All input powers are nonnegative.
bool below(const Integer& power, const Integer& end) {
    return end < 0 || power < end;
}

Integer sum(const Row& row) {
    return std::accumulate(row.begin(), row.end(), Integer(0));
}

std::string next_line(std::istream& in) {
    std::string line;
    if (!std::getline(in, line)) throw std::invalid_argument("Missing monomial quotient input.");
    return line;
}

Row read_row(const std::string& line) {
    std::istringstream fields(line);
    Row row;
    for (std::string field; fields >> field;) {
        row.emplace_back(field, 10);
        if (row.back() < 0) throw std::invalid_argument("Expected nonnegative integers.");
    }
    return row;
}

void write_row(std::ostream& out, const Row& row) {
    for (size_t i = 0; i < row.size(); ++i) {
        if (i) out << ' ';
        out << row[i];
    }
    out << '\n';
}

struct Staircase {
    explicit Staircase(size_t width, const Rows& heads) : bounds(width, -1) {
        std::set<Row> mixed;
        for (const auto& row : heads) {
            size_t axis = width, count = 0;
            for (size_t i = 0; i < width; ++i) {
                if (row[i] != 0) { axis = i; ++count; }
            }
            if (!count) { unit = true; return; }
            if (count == 1) {
                if (below(row[axis], bounds[axis])) bounds[axis] = row[axis];
            } else {
                mixed.insert(row);
            }
        }
        for (const auto& row : mixed)
            if (std::equal(row.begin(), row.end(), bounds.begin(), below)) generators.push_back(row);
    }

    Row bounds;
    Rows generators;
    bool unit = false;
};

std::vector<Staircase> components(const Staircase& staircase) {
    // Generators supported on disjoint coordinate sets give tensor factors.
    const size_t width = staircase.bounds.size();
    std::vector<size_t> parent(width);
    std::iota(parent.begin(), parent.end(), 0);
    auto root = [&](size_t i) {
        while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; }
        return i;
    };
    auto first_axis = [](const Row& row) {
        return static_cast<size_t>(std::find_if(row.begin(), row.end(),
                    [](const Integer& x) { return x != 0; }) - row.begin());
    };
    for (const Row& row : staircase.generators) {
        const size_t first = root(first_axis(row));
        for (size_t i = 0; i < width; ++i)
            if (row[i] != 0) parent[root(i)] = first;
    }
    std::map<size_t, std::vector<size_t>> groups;
    for (size_t i = 0; i < width; ++i) groups[root(i)].push_back(i);
    std::map<size_t, size_t> indices;
    std::vector<Staircase> result;
    for (const auto& [group, axes] : groups) {
        indices[group] = result.size();
        result.emplace_back(axes.size(), Rows{});
        for (size_t j = 0; j < axes.size(); ++j)
            result.back().bounds[j] = staircase.bounds[axes[j]];
    }
    for (const Row& row : staircase.generators) {
        const size_t group = root(first_axis(row));
        Row projection;
        for (const size_t axis : groups.at(group)) projection.push_back(row[axis]);
        result[indices.at(group)].generators.push_back(std::move(projection));
    }
    return result;
}

template<class Emit>
void boxes(const Staircase& staircase, Emit emit) {
    const auto& generators = staircase.generators;
    const size_t width = staircase.bounds.size();
    if (width == 2) {
        // Each downward step in the sorted planar staircase gives one strip.
        Integer left = 0, height = staircase.bounds[1];
        for (const Row& row : generators) {
            if (!below(row[1], height)) continue;
            if (left < row[0]) emit(Row{left, 0}, Row{row[0], height});
            left = row[0];
            height = row[1];
        }
        emit(Row{left, 0}, Row{staircase.bounds[0], height});
        return;
    }
    struct Pending { Row lower, upper; size_t start; };
    std::vector<Pending> pending{{Row(width, 0), staircase.bounds, 0}};
    while (!pending.empty()) {
        auto box = std::move(pending.back());
        pending.pop_back();
        size_t i = box.start;
        for (; i < generators.size(); ++i) {
            const auto& row = generators[i];
            if (!std::equal(row.begin(), row.end(), box.upper.begin(), below)) continue;
            // A surviving point has a unique first coordinate below this generator.
            Row corner = box.lower;
            for (size_t j = 0; j < width; ++j) {
                if (corner[j] >= row[j]) continue;
                Row end = box.upper;
                end[j] = row[j];
                pending.push_back({corner, std::move(end), i+1});
                corner[j] = row[j];
            }
            break;
        }
        if (i == generators.size()) emit(box.lower, box.upper);
    }
}

std::string basis(const Staircase& staircase) {
    struct Monomial { Integer degree; Row row; };
    std::vector<Monomial> points;
    boxes(staircase, [&](const Row& lower, const Row& upper) {
        Row point = lower;
        while (true) {
            points.push_back({sum(point), point});
            size_t i = point.size();
            while (i && ++point[i-1] == upper[i-1]) {
                point[i-1] = lower[i-1];
                --i;
            }
            if (!i) break;
        }
    });
    std::sort(points.begin(), points.end(), [](const Monomial& a, const Monomial& b) {
        return a.degree != b.degree ? a.degree < b.degree : a.row > b.row;
    });
    std::ostringstream out;
    out << points.size() << '\n';
    for (const auto& point : points) write_row(out, point.row);
    return out.str();
}

struct SeriesBox { Integer offset; Row lengths; long free; };
struct SeriesFactor { long dimension = -1; Integer bound = 0; std::vector<SeriesBox> boxes; };
struct Series {
    long dimension = 0;
    Integer multiplicity = 1, bound = 0;
    std::vector<SeriesFactor> factors;
};

Series analyze(const Staircase& staircase, bool store_boxes) {
    Series result;
    if (staircase.unit) {
        result.dimension = -1;
        result.multiplicity = 0;
        return result;
    }
    for (const Staircase& component : components(staircase)) {
        if (component.generators.empty() && component.bounds.front() < 0) {
            // A free variable contributes no numerator term.
            ++result.dimension;
            continue;
        }
        SeriesFactor factor;
        Integer multiplicity = 0;
        boxes(component, [&](const Row& lower, const Row& upper) {
            long free = 0;
            Row lengths;
            Integer volume = 1;
            for (size_t i = 0; i < upper.size(); ++i) {
                if (upper[i] < 0) { ++free; continue; }
                Integer length = upper[i] - lower[i];
                volume *= length;
                if (store_boxes) lengths.push_back(std::move(length));
            }
            if (free > factor.dimension) { factor.dimension = free; multiplicity = 0; }
            if (free == factor.dimension) multiplicity += volume;
            if (store_boxes) {
                const Integer offset = sum(lower);
                const Integer bound = offset + sum(lengths);
                factor.bound = std::max(factor.bound, bound);
                factor.boxes.push_back({offset, std::move(lengths), free});
            }
        });
        result.dimension += factor.dimension;
        result.multiplicity *= multiplicity;
        if (store_boxes) {
            result.bound += factor.bound;
            result.factors.push_back(std::move(factor));
        }
    }
    return result;
}

void add(Polynomial& polynomial, const Integer& degree, const Integer& coefficient) {
    if ((polynomial[degree] += coefficient) == 0) polynomial.erase(degree);
}

Polynomial multiply(const Polynomial& a, const Polynomial& b, const Integer& limit) {
    Polynomial result;
    for (const auto& [i, x] : a)
        for (const auto& [j, y] : b) {
            Integer degree = i + j;
            if (degree > limit) break;
            add(result, degree, x*y);
        }
    return result;
}

Integer choose(const Integer& n, unsigned long k) {
    Integer binomial;
    mpz_bin_ui(binomial.get_mpz_t(), n.get_mpz_t(), k);
    return binomial;
}

enum class Expansion { forward, reverse, at_one };

Polynomial numerator(const Series& series, const Integer& limit, Expansion expansion) {
    if (series.dimension < 0 || limit < 0) return {};
    Polynomial result{{0, 1}};
    for (const auto& factor : series.factors) {
        Polynomial piece;
        for (const auto& box : factor.boxes) {
            Polynomial terms;
            if (expansion == Expansion::at_one) {
                // In z=1-t a box has a pole of order free. Bring each box
                // over z^dimension and keep only the requested Taylor terms.
                const long shift = factor.dimension - box.free;
                if (shift > limit) continue;
                // This expansion is only requested up to the ring dimension.
                const unsigned long stop = limit.get_ui() - shift;
                for (unsigned long k = 0; k <= stop && k <= box.offset; ++k)
                    terms[Integer(k) + shift] = (k % 2 ? -1 : 1) * choose(box.offset, k);
                for (const Integer& length : box.lengths) {
                    Polynomial bounded_axis;
                    for (unsigned long k = 0; k <= stop && k < length; ++k)
                        bounded_axis[k] = (k % 2 ? -1 : 1) * choose(length, k+1);
                    terms = multiply(terms, bounded_axis, limit);
                }
            } else {
                Integer offset = box.offset;
                int sign = 1;
                if (expansion == Expansion::reverse) {
                    offset = factor.bound - offset - sum(box.lengths);
                    if (box.lengths.size() % 2) sign = -1;
                }
                if (offset > limit) continue;
                terms[offset] = sign;
                for (const Integer& length : box.lengths)
                    if (offset+length <= limit)
                        terms = multiply(terms, {{0, 1}, {length, -1}}, limit);
            }
            for (const auto& [degree, coefficient] : terms) add(piece, degree, coefficient);
        }
        result = multiply(result, piece, limit);
    }
    return result;
}

Integer coefficient(const Integer& degree, long poles) {
    if (degree < 0) return 0;
    if (!poles) return degree == 0 ? 1 : 0;
    return choose(degree+poles-1, poles-1);
}

Integer series_value(const Polynomial& numerator, const Integer& degree, long poles) {
    Integer value = 0;
    for (const auto& [k, c] : numerator) {
        if (k > degree) break;
        value += c * coefficient(degree-k, poles);
    }
    return value;
}

std::string function_values(const Series& series, long width, const Row& degrees) {
    std::ostringstream out;
    out << series.dimension << ' ' << series.multiplicity << '\n';
    if (degrees.empty()) return out.str();

    auto early = [&](const Integer& n) { return width == 0 || 2*n <= series.bound; };
    Integer early_limit = -1, first_high = -1;
    for (const auto& n : degrees) {
        if (early(n)) early_limit = std::max(n, early_limit);
        else if (first_high < 0 || n < first_high) first_high = n;
    }
    const Polynomial low = numerator(series, early_limit, Expansion::forward);
    Polynomial tail, reversed;
    if (first_high >= 0) {
        tail = numerator(series, series.dimension, Expansion::at_one);
        reversed = numerator(series, series.bound-first_high-width, Expansion::reverse);
    }
    for (const auto& n : degrees) {
        for (const long cumulative : {1L, 0L}) {
            const long poles = width + cumulative;
            Integer value = 0;
            if (early(n)) {
                value = series_value(low, n, poles);
            } else {
                // The polynomial tail omits the finite correction from
                // numerator terms at shifts j >= n+p. Read those in reverse.
                const long reduced_poles = series.dimension + cumulative;
                for (const auto& [k, c] : tail) {
                    if (k >= reduced_poles) break;
                    value += c * coefficient(n, reduced_poles - k.get_si());
                }
                const Integer correction = series_value(reversed, series.bound-n-poles, poles);
                if (poles % 2) value -= correction;
                else value += correction;
            }
            if (!cumulative) out << ' ';
            out << value;
        }
        out << '\n';
    }
    return out.str();
}

std::string homogeneous_heads(std::istream& in, size_t width, bool inspect) {
    const std::string order = next_line(in);
    if (order != "grevlex" && order != "lex") throw std::invalid_argument("Unknown graded ideal order.");
    Rows heads;
    bool need_basis = false, homogeneous = true;
    for (std::string line; std::getline(in, line);) {
        std::istringstream terms(line);
        Row head;
        Integer degree = -1;
        size_t count = 0;
        for (std::string term; std::getline(terms, term, ';');) {
            Row row = read_row(term);
            if (row.size() != width) throw std::invalid_argument("Wrong monomial width.");
            auto total = sum(row);
            if (count && total != degree) homogeneous = false;
            if (!count || (order == "lex" ? head < row :
                    std::lexicographical_compare(row.rbegin(), row.rend(), head.rbegin(), head.rend())))
                head = std::move(row);
            degree = std::move(total);
            ++count;
        }
        // The sole term in a nonzero scalar polynomial has an empty exponent tuple.
        if (!width && line.empty()) { count = 1; degree = 0; }
        if (!count) throw std::invalid_argument("Expected a nonzero generator.");
        if (count == 1 && degree == 0) {
            std::ostringstream out;
            out << "heads\n";
            write_row(out, Row(width, 0));
            return out.str();
        }
        if (count > 1) need_basis = true;
        heads.push_back(std::move(head));
    }
    if (inspect && heads.size() > 1 && need_basis) return "basis\n";
    if (!homogeneous) return "inhomogeneous\n";
    std::ostringstream out;
    out << "heads\n";
    for (const auto& row : heads) write_row(out, row);
    return out.str();
}

} // namespace

std::string monomial_quotient(std::istream& in, const std::string& operation) {
    const long width = std::stol(next_line(in));
    if (width < 0) throw std::invalid_argument("A ring cannot have negative coordinate count.");
    if (operation == "monomial_homogeneous_inspect" || operation == "monomial_homogeneous_heads")
        return homogeneous_heads(in, width, operation == "monomial_homogeneous_inspect");
    Row degrees;
    if (operation == "monomial_function") degrees = read_row(next_line(in));
    Rows heads;
    for (std::string line; std::getline(in, line);) {
        auto row = read_row(line);
        if (row.size() != static_cast<size_t>(width))
            throw std::invalid_argument("Expected one nonnegative exponent per coordinate.");
        heads.push_back(std::move(row));
    }
    const Staircase staircase(width, heads);
    if (operation == "monomial_basis" || operation == "monomial_length") {
        if (staircase.unit) return "0\n";
        if (std::any_of(staircase.bounds.begin(), staircase.bounds.end(),
                        [](const Integer& bound) { return bound < 0; })) return "infinite\n";
        if (operation == "monomial_basis") return basis(staircase);
        return analyze(staircase, false).multiplicity.get_str() + "\n";
    }
    const auto series = analyze(staircase, operation == "monomial_series" || !degrees.empty());
    if (operation == "monomial_function") return function_values(series, width, degrees);
    if (operation != "monomial_series") throw std::invalid_argument("Unknown monomial quotient operation.");
    std::ostringstream out;
    out << series.dimension << ' ' << series.multiplicity << '\n';
    for (const auto& [degree, value] : numerator(series, series.bound, Expansion::forward))
        out << degree << ' ' << value << '\n';
    return out.str();
}
} // namespace atlas
