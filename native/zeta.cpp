// Exact univariate reduction of the Denef--Loeser sum.
#include "groebner.hpp"
#include <flint/fmpz_poly.h>
#include <flint/fmpz_poly_factor.h>
#include <flint/fmpz_poly_q.h>
#include <memory>
#include <sstream>

namespace atlas {
namespace {
struct Univariate {
    fmpz_poly_t p;
    Univariate() { fmpz_poly_init(p); }
    ~Univariate() { fmpz_poly_clear(p); }
    Univariate(const Univariate&) = delete;
};

struct RationalFunction {
    fmpz_poly_q_t value;
    RationalFunction() { fmpz_poly_q_init(value); }
    ~RationalFunction() { fmpz_poly_q_clear(value); }
    RationalFunction(const RationalFunction&) = delete;
};

// Each occupied level holds the reduced sum of 2^level nonzero strata.
void add_stratum(std::vector<std::unique_ptr<RationalFunction>>& levels,
                 std::unique_ptr<RationalFunction> term) {
    for (auto& partial : levels) {
        if (!partial) {
            partial = std::move(term);
            return;
        }
        fmpz_poly_q_add_in_place(term->value, partial->value);
        partial.reset();
    }
    levels.push_back(std::move(term));
}

Z integer(std::istream& in) {
    Z a;
    std::string text;
    if (!(in >> text) || fmpz_set_str(&a.v, text.c_str(), 10))
        throw std::invalid_argument("Expected an integral zeta coefficient.");
    return a;
}

void append_integer(std::string& out, const fmpz* a) {
    char* text = fmpz_get_str(nullptr, 10, a);
    out += text;
    flint_free(text);
}

// The signed content, then the primitive factors and their exponents. The
// coefficient lists are ascending, for building the factored display and
// reading the rational roots of the (split) denominator.
void append_factors(std::string& out, const fmpz_poly_t p) {
    fmpz_poly_factor_t f;
    fmpz_poly_factor_init(f);
    fmpz_poly_factor(f, p);
    append_integer(out, &f->c);
    out += " " + std::to_string(f->num) + "\n";
    for (slong j = 0; j < f->num; ++j) {
        out += std::to_string(f->exp[j]);
        for (slong i = 0; i < f->p[j].length; ++i) {
            out += ' ';
            append_integer(out, f->p[j].coeffs + i);
        }
        out += '\n';
    }
    fmpz_poly_factor_clear(f);
}

std::string zeta(std::istream& in) {
    size_t n, strata;
    if (!(in >> n >> strata)) throw std::invalid_argument("Missing zeta input sizes.");
    std::vector<Univariate> factors(n);
    for (auto& linear : factors) {
        const auto order = integer(in), nu = integer(in);
        if (fmpz_sgn(&order.v) <= 0 || fmpz_sgn(&nu.v) <= 0)
            throw std::invalid_argument("Expected positive numerical data.");
        fmpz_poly_set_coeff_fmpz(linear.p, 0, &nu.v);
        fmpz_poly_set_coeff_fmpz(linear.p, 1, &order.v);
    }
    std::vector<std::unique_ptr<RationalFunction>> levels;
    std::vector<size_t> seen(n, strata);
    for (size_t k = 0; k < strata; ++k) {
        const auto chi = integer(in);
        size_t count;
        if (!(in >> count) || count > n) throw std::invalid_argument("Invalid stratum.");
        std::unique_ptr<RationalFunction> term;
        if (!fmpz_is_zero(&chi.v)) {
            term = std::make_unique<RationalFunction>();
            fmpz_poly_set_fmpz(fmpz_poly_q_numref(term->value), &chi.v);
        }
        for (size_t j = 0; j < count; ++j) {
            size_t index;
            if (!(in >> index) || index >= n || seen[index] == k)
                throw std::invalid_argument("Invalid component in stratum.");
            seen[index] = k;
            if (term) {
                auto denominator = fmpz_poly_q_denref(term->value);
                fmpz_poly_mul(denominator, denominator, factors[index].p);
            }
        }
        if (term) {
            fmpz_poly_q_canonicalise(term->value);
            add_stratum(levels, std::move(term));
        }
    }
    RationalFunction sum;
    for (const auto& partial : levels)
        if (partial) fmpz_poly_q_add_in_place(sum.value, partial->value);
    const auto numerator = fmpz_poly_q_numref(sum.value);
    const auto denominator = fmpz_poly_q_denref(sum.value);
    if (!numerator->length || !fmpz_equal(numerator->coeffs, denominator->coeffs))
        return "bad origin\n";
    std::string out;
    for (slong i = 0; i < numerator->length; ++i) {
        if (i) out += ' ';
        append_integer(out, numerator->coeffs + i);
    }
    out += '\n';
    append_factors(out, numerator);
    append_factors(out, denominator);
    return out;
}
}  // namespace
}  // namespace atlas

extern "C" char* atlas_topological(const char* input, size_t length) noexcept {
    try {
        std::istringstream stream(std::string(input, length));
        return atlas::copy_text(atlas::zeta(stream));
    } catch (const std::exception& error) {
        atlas::set_error(error.what());
    } catch (...) {
        atlas::set_error("Unknown zeta-function failure.");
    }
    return nullptr;
}
