#include <flint/fexpr.h>
#include "common_field.hpp"
#include "algebraic_number.hpp"
#include "binomial_root_family.hpp"
#include <flint/fmpq.h>
#include <flint/fmpq_poly.h>
#include <flint/fmpz_poly.h>
#include <array>
#include <map>
#include <stdexcept>
#include <vector>

namespace atlas {
using namespace std;
using J = Json::Value;
using Num = AlgebraicNumber;

namespace {
struct RootPolynomial {
    fmpz_poly_t poly;
    RootPolynomial() { fmpz_poly_init(poly); }
    ~RootPolynomial() { fmpz_poly_clear(poly); }
    RootPolynomial(const RootPolynomial&) = delete;
    RootPolynomial& operator=(const RootPolynomial&) = delete;
};

struct PowerBasisElement {
    fmpq_poly_t poly;
    PowerBasisElement() { fmpq_poly_init(poly); }
    ~PowerBasisElement() { fmpq_poly_clear(poly); }
    PowerBasisElement(const PowerBasisElement&) = delete;
    PowerBasisElement& operator=(const PowerBasisElement&) = delete;
};

void rational_coordinates(fmpq_poly_t result, const qqbar_t value) {
    fmpq_t q;
    fmpq_init(q);
    qqbar_get_fmpq(q, value);
    fmpq_poly_set_fmpq(result, q);
    fmpq_clear(q);
}

// Substitute the old primitive element into one of its coordinate polynomials.
// Horner reduction keeps the intermediate degree bounded by twice the field degree.
void change_generator(fmpq_poly_t value, const fmpq_poly_t old_in_new, const fmpq_poly_t modulus) {
    if (fmpq_poly_degree(value) <= 0) return;
    PowerBasisElement image;
    fmpq_t c;
    fmpq_init(c);
    for (slong j = fmpq_poly_length(value); j-- > 0;) {
        fmpq_poly_mul(image.poly, image.poly, old_in_new);
        fmpq_poly_rem(image.poly, image.poly, modulus);
        fmpq_poly_get_coeff_fmpq(c, value, j);
        fmpq_poly_add_fmpq(image.poly, image.poly, c);
    }
    fmpq_clear(c);
    fmpq_poly_swap(value, image.poly);
}

void extend_by_generators(const vector<Num>& values, Num& generator, vector<PowerBasisElement>& coordinates) {
    if (qqbar_is_rational(generator.n)) rational_coordinates(coordinates[0].poly, generator.n);
    else fmpq_poly_set_coeff_si(coordinates[0].poly, 1, 1);

    for (size_t i = 1; i < values.size(); ++i) {
        auto& value = values[i];
        auto& image = coordinates[i].poly;
        if (qqbar_is_rational(value.n)) {
            rational_coordinates(image, value.n);
            continue;
        }
        // For a separating integer k, theta = old + k*value generates the
        // compositum. Once value is expressed in theta, old = theta-k*value
        // gives the exact change of basis for every earlier coordinate.
        for (slong round = 1, prec = 128; ; ++round, prec *= 2) {
            if (qqbar_degree(generator.n) % qqbar_degree(value.n) == 0 &&
                qqbar_express_in_field(image, generator.n, value.n, 0, 0, prec)) break;
            bool found = false;
            for (slong k = 1; k <= round; ++k) {
                Num candidate = generator + Num(k)*value;
                if (!qqbar_express_in_field(image, candidate.n, value.n, 0, 0, prec)) continue;
                PowerBasisElement old_in_new, indeterminate, modulus;
                fmpq_poly_set_coeff_si(indeterminate.poly, 1, 1);
                fmpq_poly_scalar_mul_si(old_in_new.poly, image, k);
                fmpq_poly_sub(old_in_new.poly, indeterminate.poly, old_in_new.poly);
                fmpq_poly_set_fmpz_poly(modulus.poly, QQBAR_POLY(candidate.n));
                for (size_t j = 0; j < i; ++j)
                    change_generator(coordinates[j].poly, old_in_new.poly, modulus.poly);
                generator = std::move(candidate);
                found = true;
                break;
            }
            if (found) break;
            if (prec > WORD_MAX/2) throw runtime_error("Number field precision exceeds a signed word.");
        }
    }
}
} // namespace

J coefficient_field(const vector<Num>& values, vector<vector<mpq_class>>& images) {
    if (values.empty()) return J();
    Num generator = values.front();
    vector<PowerBasisElement> coordinates(values.size());
    J roots(Json::arrayValue);
    auto minimal = QQBAR_POLY(generator.n);
    const auto n = fmpz_poly_degree(minimal);
    if (n >= 3 && static_cast<size_t>(n) == values.size() &&
        _fmpz_poly_hamming_weight(minimal->coeffs, minimal->length) == 2 &&
        all_of(values.begin(), values.end(), [&](const Num& v) { return fmpz_poly_equal(minimal, QQBAR_POLY(v.n)); })) {
        J coefficients(Json::arrayValue);
        for (slong i = 0; i <= n; ++i)
            coefficients.append(flint_string(fmpz_get_str(nullptr, 10, minimal->coeffs+i)));
        qqbar_ptr conjugates = _qqbar_vec_init(n);
        qqbar_conjugates(conjugates, generator.n);
        for (const auto& v : values) {
            slong j = 0;
            while (j < n && !qqbar_equal(conjugates+j, v.n)) ++j;
            J row;
            row["coefficients"] = coefficients;
            row["index"] = Json::Int64(j+1);
            roots.append(std::move(row));
        }
        _qqbar_vec_clear(conjugates, n);
    }
    if (roots.empty() || !binomial_root_family(roots, values, generator, coordinates))
        extend_by_generators(values, generator, coordinates);
    J field;
    auto equation = QQBAR_POLY(generator.n);
    field["degree"] = Json::Int64(fmpz_poly_degree(equation));
    for (slong j = 0; j < fmpz_poly_length(equation); ++j)
        field["coefficients"].append(flint_string(fmpz_get_str(nullptr, 10, equation->coeffs+j)));
    fexpr_t exact, index;
    fexpr_init(exact);
    qqbar_get_fexpr_root_indexed(exact, generator.n);
    field["exact"] = flint_string(fexpr_get_str(exact));
    fexpr_view_arg(index, exact, 1);
    fmpz_t k;
    fmpz_init(k);
    const bool indexed = fexpr_get_fmpz(k, index);
    field["index"] = Json::Int64(fmpz_get_si(k));
    fmpz_clear(k);
    fexpr_clear(exact);
    if (!indexed) throw logic_error("The common field has no indexed algebraic generator.");
    fmpq_t scalar;
    fmpq_init(scalar);
    for (const auto& element : coordinates) {
        auto& row = images.emplace_back(fmpq_poly_length(element.poly));
        for (size_t j = 0; j < row.size(); ++j) {
            fmpq_poly_get_coeff_fmpq(scalar, element.poly, j);
            fmpq_get_mpq(row[j].get_mpq_t(), scalar);
        }
    }
    fmpq_clear(scalar);
    return field;
}

// Present a finite set of exact chart coefficients in one number field.
// Each input uses FLINT's own root order (as in PolynomialRootIndexed).
J common_field(const J& request) {
    const auto& roots = request["field_roots"];
    const auto& rectangles = request["root_rectangles"];
    const bool root_images = request.get("root_images", false).asBool();
    vector<Num> values;
    map<J, vector<Num>> families;
    for (const auto& row : roots) {
        auto [family, inserted] = families.try_emplace(row["coefficients"]);
        if (inserted) {
            RootPolynomial polynomial;
            fmpz_t coefficient;
            fmpz_init(coefficient);
            slong i = 0;
            for (const auto& c : row["coefficients"]) {
                if (!c.isString() || fmpz_set_str(coefficient, c.asCString(), 10)) {
                    fmpz_clear(coefficient);
                    throw invalid_argument("Invalid number field coefficient.");
                }
                fmpz_poly_set_coeff_fmpz(polynomial.poly, i++, coefficient);
            }
            fmpz_clear(coefficient);
            slong degree = fmpz_poly_degree(polynomial.poly);
            if (degree < 1) throw invalid_argument("Expected a nonconstant root polynomial.");
            family->second.resize(degree);
            qqbar_ptr conjugates = _qqbar_vec_init(degree);
            qqbar_roots_fmpz_poly(conjugates, polynomial.poly, 0);
            for (slong j = 0; j < degree; ++j) qqbar_swap(family->second[j].n, conjugates + j);
            _qqbar_vec_clear(conjugates, degree);
        }
        slong index = row["index"].asInt64();
        if (index < 1 || static_cast<size_t>(index) > family->second.size())
            throw invalid_argument("Invalid algebraic root index.");
        values.push_back(family->second[index-1]);
    }
    if (values.empty()) throw invalid_argument("No algebraic constants.");
    if (!rectangles.isNull()) {
        // The public expression syntax numbers roots by disjoint rational
        // isolating rectangles. Exact qqbar comparisons locate this marked
        // embedding, including real roots or roots on a rectangle edge.
        if (values.size() != 1) throw invalid_argument("Expected one marked root.");
        Num real, imaginary;
        qqbar_re_im(real.n, imaginary.n, values[0].n);
        int index = -1;
        for (Json::ArrayIndex j = 0; j < rectangles.size(); ++j) {
            if (rectangles[j].size() != 4) throw invalid_argument("Expected four rectangle bounds.");
            array<Num,4> bounds;
            for (Json::ArrayIndex k = 0; k < bounds.size(); ++k) {
                fmpq_t q;
                fmpq_init(q);
                bool valid = rectangles[j][k].isString() &&
                             fmpq_set_str(q, rectangles[j][k].asCString(), 10) == 0 &&
                             !fmpz_is_zero(fmpq_denref(q));
                if (valid) qqbar_set_fmpq(bounds[k].n, q);
                fmpq_clear(q);
                if (!valid) throw invalid_argument("Invalid root rectangle bound.");
            }
            if (qqbar_cmp_re(real.n, bounds[0].n) >= 0 && qqbar_cmp_re(real.n, bounds[1].n) <= 0 &&
                qqbar_cmp_re(imaginary.n, bounds[2].n) >= 0 && qqbar_cmp_re(imaginary.n, bounds[3].n) <= 0) {
                if (index != -1) throw invalid_argument("The root belongs to several rectangles.");
                index = static_cast<int>(j);
            }
        }
        if (index < 0) throw invalid_argument("The marked root was absent from the isolating rectangles.");
        J result;
        result["root_index"] = index;
        return result;
    }
    Num generator = values.front();
    vector<PowerBasisElement> coordinates(values.size());
    if (!binomial_root_family(roots, values, generator, coordinates))
        extend_by_generators(values, generator, coordinates);
    J out;
    out["minimal_polynomial"] = atlas::flint_string(fmpz_poly_get_str_pretty(QQBAR_POLY(generator.n), "T"));
    out["degree"] = Json::Int64(qqbar_degree(generator.n));
    fexpr_t exact;
    fexpr_init(exact);
    qqbar_get_fexpr_root_indexed(exact, generator.n);
    out["exact"] = atlas::flint_string(fexpr_get_str(exact));
    fexpr_clear(exact);
    map<J,J> images;
    for (size_t i=0; i<values.size(); ++i) {
        const auto& expression = coordinates[i].poly;
        out["values"].append(flint_string(fmpq_poly_get_str_pretty(expression, "T")));
        if (root_images) {
            J row(Json::arrayValue);
            fmpq_t c;
            fmpq_init(c);
            for (slong j = 0; j < fmpq_poly_length(expression); ++j) {
                fmpq_poly_get_coeff_fmpq(c, expression, j);
                row.append(flint_string(fmpq_get_str(nullptr, 10, c)));
            }
            fmpq_clear(c);
            images[roots[static_cast<Json::ArrayIndex>(i)]["coefficients"]].append(std::move(row));
        }
    }
    if (root_images) {
        out["root_images"] = J(Json::arrayValue);
        for (auto& [equation,rows] : images) {
            if (rows.size() + 1 == equation.size()) {
                J family(Json::arrayValue);
                family.append(equation);
                family.append(rows);
                out["root_images"].append(std::move(family));
            }
        }
    }
    return out;
}
} // namespace atlas
