// Absolute factors of rational polynomials from separable coordinate sections.
#include "algebraic_number.hpp"
#include "common_field.hpp"
#include "component_projection.hpp"
#include "local_io.hpp"
#include "local_monomial.hpp"
#include "polynomial_factors.hpp"
#include "primary_decomposition.hpp"
#include "quotient_ring.hpp"
#include "rational_sparse_json.hpp"

#include <flint/fmpq_poly.h>
#include <algorithm>
#include <limits>
#include <unordered_map>

namespace atlas {
namespace {
using local::Polynomial;
using local::Powers;
using Num = AlgebraicNumber;
using GeometricPolynomial = std::map<Powers, Num>;

struct Univariate {
    fmpq_poly_t p;
    Univariate() { fmpq_poly_init(p); }
    ~Univariate() { fmpq_poly_clear(p); }
    Univariate(const Univariate &) = delete;
};

Num number(const mpq_class &scalar) {
    fmpq_t q;
    fmpq_init(q);
    fmpq_set_mpq(q, scalar.get_mpq_t());
    Num value;
    qqbar_set_fmpq(value.n, q);
    fmpq_clear(q);
    return value;
}

mpq_class rational(const Num &x) {
    fmpq_t q;
    fmpq_init(q);
    qqbar_get_fmpq(q, x.n);
    mpq_class value;
    fmpq_get_mpq(value.get_mpq_t(), q);
    fmpq_clear(q);
    return value;
}

std::vector<Num> roots(const Polynomial &univariate) {
    const auto &degree = univariate.rbegin()->first[0];
    if (!degree.fits_slong_p() || degree >= std::numeric_limits<Json::ArrayIndex>::max())
        throw std::length_error("The explicit root list exceeds the addressable array size.");
    Univariate f;
    fmpq_t q;
    fmpq_init(q);
    for (const auto &[e, c] : univariate) {
        fmpq_set_mpq(q, c.get_mpq_t());
        fmpq_poly_set_coeff_fmpq(f.p, e[0].get_si(), q);
    }
    fmpq_clear(q);
    const auto n = degree.get_si();
    std::vector<Num> values(n);
    auto marks = _qqbar_vec_init(n);
    qqbar_roots_fmpq_poly(marks, f.p, 0);
    for (slong i = 0; i < n; ++i) qqbar_swap(values[i].n, marks + i);
    _qqbar_vec_clear(marks, n);
    return values;
}

GeometricPolynomial geometric(const Polynomial &f) {
    GeometricPolynomial g;
    for (const auto &[e, c] : f) g.emplace(e, number(c));
    return g;
}

mpz_class integer_power(const mpz_class &base, const mpz_class &power) {
    if (power == 0 || base == 1) return 1;
    if (base == 0) return 0;
    mpz_class result = 1, b = base;
    const size_t bits = mpz_sizeinbase(power.get_mpz_t(), 2);
    for (size_t i = 0; i < bits; ++i) {
        if (mpz_tstbit(power.get_mpz_t(), i)) result *= b;
        if (i + 1 < bits) b *= b;
    }
    return result;
}

// Substitute integers for the transverse coordinates; for a marked input
// also substitute the free line parameter for its final coefficient variable.
Polynomial section(const Polynomial &f, size_t axis, const Powers &point, bool marked = false) {
    Polynomial result;
    for (const auto &[e, c] : f) {
        mpq_class term = c;
        for (size_t i = 0; i < point.size() && term != 0; ++i)
            if (i != axis) term *= integer_power(point[i], e[i]);
        Powers power{e[axis]};
        if (marked) power[0] += e.back();
        if (term != 0 && (result[power] += term) == 0) result.erase(power);
    }
    return result;
}

struct Section {
    Powers point;
    Polynomial root_equation;
};

// At a full-degree separable fibre, the absolute factors specialize to
// disjoint nonempty root sets. They all have the same positive degree in
// this coordinate because Gal(Qbar/Q) acts transitively on the absolute
// factors of a rational irreducible. Such fibres form a dense open set.
Section transverse_section(const Polynomial &f, size_t axis, const Powers &degrees) {
    Powers point(degrees.size());
    std::vector<size_t> transverse;
    for (size_t i = 0; i < degrees.size(); ++i)
        if (i != axis && degrees[i] != 0) transverse.push_back(i);
    for (mpz_class radius = 0;; ++radius) {
        std::fill(point.begin(), point.end(), 0);
        do {
            if (radius == 0 ||
                std::any_of(transverse.begin(), transverse.end(), [&](size_t i) { return point[i] == radius; })) {
                auto image = section(f, axis, point);
                if (!image.empty() && image.rbegin()->first[0] == degrees[axis]) {
                    const auto factors = primary::factors(image, 1);
                    if (std::all_of(factors.begin(), factors.end(),
                                    [](const auto &part) { return part.second == 1; })) {
                        const auto best =
                            std::min_element(factors.begin(), factors.end(), [](const auto &a, const auto &b) {
                                return a.first.rbegin()->first < b.first.rbegin()->first;
                            });
                        return {point, best->first};
                    }
                }
            }
            size_t j = 0;
            for (; j < transverse.size(); ++j) {
                auto &digit = point[transverse[j]];
                if (++digit <= radius) break;
                digit = 0;
            }
            if (j == transverse.size()) break;
        } while (true);
    }
}

bool equal_factors(const GeometricPolynomial &a, const GeometricPolynomial &b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](const auto &x, const auto &y) {
               return x.first == y.first && qqbar_equal(x.second.n, y.second.n);
           });
}

std::vector<GeometricPolynomial> conjugate_factors(const Polynomial &marked, const Polynomial &equation) {
    std::map<Powers, Polynomial> coefficients;
    for (const auto &[e, c] : marked) coefficients[Powers(e.begin(), e.end() - 1)].emplace(Powers{e.back()}, c);
    auto marks = roots(equation);
    std::vector<GeometricPolynomial> orbit(marks.size());
    for (const auto &[e, terms] : coefficients) {
        Univariate polynomial;
        fmpq_t q;
        fmpq_init(q);
        for (const auto &[k, c] : terms) {
            fmpq_set_mpq(q, c.get_mpq_t());
            fmpq_poly_set_coeff_fmpq(polynomial.p, k[0].get_si(), q);
        }
        fmpq_clear(q);
        for (size_t i = 0; i < marks.size(); ++i)
            qqbar_evaluate_fmpq_poly(orbit[i][e].n, polynomial.p, marks[i].n);
    }
    std::vector<GeometricPolynomial> distinct;
    for (auto &f : orbit)
        if (std::none_of(distinct.begin(), distinct.end(),
                         [&](const auto &other) { return equal_factors(f, other); }))
            distinct.emplace_back(std::move(f));
    return distinct;
}

std::vector<GeometricPolynomial> binomial(const Polynomial &f) {
    const auto &[a, ca] = *f.rbegin();
    const auto &[b, cb] = *f.begin();
    mpz_class gcd_difference = 0;
    for (size_t i = 0; i < a.size(); ++i) gcd_difference = gcd(gcd_difference, a[i] - b[i]);
    if (gcd_difference == 1) return {geometric(f)};
    Powers u = a, v = b;
    for (auto &e : u) e /= gcd_difference;
    for (auto &e : v) e /= gcd_difference;
    std::vector<GeometricPolynomial> result;
    for (auto &root : roots({{{gcd_difference}, 1}, {{0}, cb / ca}})) {
        qqbar_neg(root.n, root.n);
        result.push_back({{u, Num(1)}, {v, std::move(root)}});
    }
    return result;
}

using Factors = std::vector<std::pair<GeometricPolynomial, mpz_class>>;

Factors split_binomial(const Polynomial &f) {
    const auto &[leading, lc] = *f.rbegin();
    const auto &[other, coefficient] = *f.begin();
    Powers a = leading, b = other;
    Factors result;
    for (size_t j = 0; j < a.size(); ++j) {
        const mpz_class common = std::min(a[j], b[j]);
        if (!common) continue;
        Powers coordinate(a.size());
        coordinate[j] = 1;
        result.emplace_back(GeometricPolynomial{{coordinate, Num(1)}}, common);
        a[j] -= common;
        b[j] -= common;
    }
    for (auto &factor : binomial({{a, 1}, {b, coefficient / lc}})) result.emplace_back(std::move(factor), 1);
    return result;
}

std::vector<GeometricPolynomial> absolute_factors(const Polynomial &f, size_t width) {
    if (f.size() == 1) return {geometric(f)}; // A rational irreducible monomial is a coordinate.
    if (f.size() == 2) return binomial(f);
    Powers degrees(width);
    for (const auto &[e, c] : f)
        for (size_t j = 0; j < width; ++j) degrees[j] = std::max(degrees[j], e[j]);
    size_t axis = width;
    mpz_class common = local::max_total_degree(f);
    size_t active = 0;
    for (size_t j = 0; j < width; ++j)
        if (degrees[j] != 0) {
            ++active;
            common = gcd(common, degrees[j]);
            if (axis == width || degrees[j] < degrees[axis]) axis = j;
        }
    // The number of conjugate factors divides every partial degree.
    if (common == 1) return {geometric(f)};
    if (active == 1) {
        std::vector<GeometricPolynomial> result;
        Powers variable(width);
        variable[axis] = 1;
        for (auto &root : roots(section(f, axis, Powers(width)))) {
            qqbar_neg(root.n, root.n);
            result.push_back({{variable, Num(1)}, {Powers(width), std::move(root)}});
        }
        return result;
    }
    // Keep the packed rank and smoothness shortcuts on small supports.
    // The coordinate-section argument does not require them.
    if (local::max_total_degree(f) <= 8 && width <= 4) {
        const auto quick = components::absolute_rational_factor(local::encode_row(f), width);
        if (quick.isBool() && quick.asBool()) return {geometric(f)};
    }
    const auto slice = transverse_section(f, axis, degrees);
    if (slice.root_equation.rbegin()->first[0] == 1) return {geometric(f)};
    Polynomial input, modulus;
    for (auto [e, c] : f) {
        Powers powers = e;
        powers.push_back(0);
        input.emplace(std::move(powers), std::move(c));
    }
    for (const auto &[e, c] : slice.root_equation) {
        Powers powers(width + 1);
        powers.back() = e[0];
        modulus.emplace(std::move(powers), c);
    }
    // A Galois automorphism fixing the selected intersection point fixes
    // its unique component. Thus this absolute component is already defined
    // over Q(alpha), and is the unique factor there vanishing at the point.
    for (const auto &[factor, power] : marked_section_factors(input, modulus, width + 1, axis)) {
        auto value = section(factor, axis, slice.point, true);
        if (value.empty() || local::polynomial_divides(slice.root_equation, value, 1))
            return conjugate_factors(factor, slice.root_equation);
    }
    throw std::logic_error("No factor through the certified transverse point.");
}

struct NumberHash {
    size_t operator()(const Num &a) const { return qqbar_hash(a.n); }
};
struct NumberEqual {
    bool operator()(const Num &a, const Num &b) const { return qqbar_equal(a.n, b.n); }
};

Factors factor_over_closure(const Polynomial &f, size_t width) {
    // Splitting a binomial is a lattice calculation before any rational
    // factorization. In particular huge coprime reduced powers are harmless.
    if (f.size() == 2) return split_binomial(f);
    Factors result;
    for (const auto &[factor, multiplicity] : primary::factors(f, width))
        for (auto &g : absolute_factors(factor, width)) result.emplace_back(std::move(g), multiplicity);
    return result;
}

Json::Value factor(const Json::Value &q) {
    if (q["operation"].asString() != "factor") throw std::invalid_argument("Unknown absolute factorization operation.");
    const size_t n = q["width"].asUInt();
    const auto order = q["order"].asString();
    if (order != "lex" && order != "grevlex") throw std::invalid_argument("Unknown polynomial order.");
    const bool lex = order == "lex";
    const auto f = local::decode_row(q["polynomial"].asString(), n);
    if (f.empty()) throw std::invalid_argument("The zero polynomial has no irreducible factorization.");
    Json::Value result;
    result["unit"] = f.at(local::global_head(f, lex)).get_str();
    result["field"] = Json::Value();
    result["factors"] = Json::arrayValue;
    if (f.size() == 1 && f.begin()->first == Powers(n)) return result;

    auto factors = factor_over_closure(f, n);
    std::unordered_map<Num, size_t, NumberHash, NumberEqual> indices;
    std::vector<Num> values;
    for (auto &[g, multiplicity] : factors) {
        const auto leading = std::max_element(g.begin(), g.end(), [&](const auto &a, const auto &b) {
                                 return local::global_less(a.first, b.first, lex);
                             })->second;
        for (auto &[e, c] : g) {
            if (!qqbar_is_one(leading.n)) qqbar_div(c.n, c.n, leading.n);
            if (!qqbar_is_rational(c.n) && indices.emplace(c, values.size()).second) values.push_back(c);
        }
    }
    std::vector<std::vector<mpq_class>> images;
    result["field"] = coefficient_field(values, images);
    const bool marked = !values.empty();
    const size_t output_width = n + marked;
    local::Basis relations;
    if (marked) {
        auto &relation = relations.emplace_back();
        Powers exponent(output_width);
        for (const auto &c : result["field"]["coefficients"]) {
            const mpq_class coefficient(c.asString());
            if (coefficient != 0) relation.emplace(exponent, coefficient);
            ++exponent.back();
        }
    }
    quotient_ring::Ring ring(relations, output_width, true);
    Polynomial product{{Powers(output_width), f.at(local::global_head(f, lex))}};
    for (const auto &[g, power] : factors) {
        Polynomial row;
        for (const auto &[e, c] : g) {
            if (values.empty())
                row.emplace(e, rational(c));
            else {
                Powers marked = e;
                marked.push_back(0);
                if (qqbar_is_rational(c.n))
                    row.emplace(std::move(marked), rational(c));
                else {
                    const auto &image = images[indices.at(c)];
                    for (size_t j = 0; j < image.size(); ++j)
                        if (image[j] != 0) {
                            marked.back() = j;
                            row.emplace(marked, image[j]);
                        }
                }
            }
        }
        Json::Value part;
        part["polynomial"] = local::encode_row(row);
        part["multiplicity"] = power.get_str();
        result["factors"].append(std::move(part));
        if (power != 1) row = std::get<Polynomial>(ring.evaluate("power", {row}, power));
        product = std::get<Polynomial>(ring.evaluate("multiply", {product, row}));
    }
    Polynomial original;
    for (const auto &[e, c] : f) {
        Powers powers = e;
        if (marked) powers.push_back(0);
        original.emplace(std::move(powers), c);
    }
    if (product != original)
        throw std::logic_error("The absolute factors do not reconstruct the input in their common field.");
    return result;
}
} // namespace
} // namespace atlas

extern "C" const char *atlas_absolute_factorization(const char *request) noexcept {
    return atlas::sparse_json::invoke(request, atlas::factor);
}
