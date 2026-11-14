// Exact univariate reduction of the Denef--Loeser sum.
#include "groebner.hpp"
#include <flint/fmpz_poly.h>
#include <flint/fmpz_poly_factor.h>
#include <sstream>

namespace atlas {
namespace {
struct Univariate {
    fmpz_poly_t p;
    Univariate() { fmpz_poly_init(p); }
    ~Univariate() { fmpz_poly_clear(p); }
    Univariate(const Univariate&) = delete;
};

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
    Univariate denominator, numerator, term;
    fmpz_poly_one(denominator.p);
    for (auto& linear : factors) {
        const auto order = integer(in), nu = integer(in);
        if (fmpz_sgn(&order.v) <= 0 || fmpz_sgn(&nu.v) <= 0)
            throw std::invalid_argument("Expected positive numerical data.");
        fmpz_poly_set_coeff_fmpz(linear.p, 0, &nu.v);
        fmpz_poly_set_coeff_fmpz(linear.p, 1, &order.v);
        fmpz_poly_mul(denominator.p, denominator.p, linear.p);
    }
    for (size_t k = 0; k < strata; ++k) {
        const auto chi = integer(in);
        size_t count;
        if (!(in >> count) || count > n) throw std::invalid_argument("Invalid stratum.");
        std::vector<bool> seen(n);
        fmpz_poly_set(term.p, denominator.p);
        for (size_t j = 0; j < count; ++j) {
            size_t index;
            if (!(in >> index) || index >= n || seen[index])
                throw std::invalid_argument("Invalid component in stratum.");
            seen[index] = true;
            fmpz_poly_divexact(term.p, term.p, factors[index].p);
        }
        fmpz_poly_scalar_mul_fmpz(term.p, term.p, &chi.v);
        fmpz_poly_add(numerator.p, numerator.p, term.p);
    }
    if (numerator.p->length == 0 || !fmpz_equal(numerator.p->coeffs, denominator.p->coeffs))
        return "bad origin\n";
    Univariate gcd;
    fmpz_poly_gcd(gcd.p, numerator.p, denominator.p);
    fmpz_poly_divexact(numerator.p, numerator.p, gcd.p);
    fmpz_poly_divexact(denominator.p, denominator.p, gcd.p);
    std::string out;
    for (slong i = 0; i < numerator.p->length; ++i) {
        if (i) out += ' ';
        append_integer(out, numerator.p->coeffs + i);
    }
    out += '\n';
    append_factors(out, numerator.p);
    append_factors(out, denominator.p);
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
