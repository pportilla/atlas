// Exact scalar arithmetic for expressions at the prompt and in saved sessions.
#include <gmpxx.h>
#include <stdexcept>
#include <string>

namespace atlas {
namespace {
mpq_class rational(const std::string& text) {
    mpq_class value(text, 10);
    if (value.get_den() == 0) throw std::domain_error("Division by zero.");
    value.canonicalize();
    return value;
}

mpq_class power(mpq_class base, mpz_class exponent) {
    if (exponent < 0) {
        if (base == 0) throw std::domain_error("Division by zero.");
        mpq_inv(base.get_mpq_t(), base.get_mpq_t());
        exponent = -exponent;
    }
    if (exponent == 0 || base == 1) return 1;
    if (base == 0) return 0;
    if (base == -1) return mpz_odd_p(exponent.get_mpz_t()) ? -1 : 1;
    mpq_class result = 1;
    while (exponent > 0) {
        if (mpz_odd_p(exponent.get_mpz_t())) result *= base;
        exponent >>= 1;
        if (exponent > 0) base *= base;
    }
    return result;
}
} // namespace

std::string scalar_arithmetic(const std::string& op, const std::string& left, const std::string& right) {
    auto a = rational(left);
    const auto b = rational(right);
    if (op == "+") a += b;
    else if (op == "-") a -= b;
    else if (op == "*") a *= b;
    else if (op == "/") {
        if (b == 0) throw std::domain_error("Division by zero.");
        a /= b;
    } else if (op == "^") {
        if (b.get_den() != 1) throw std::invalid_argument("Use an integer exponent.");
        a = power(a, b.get_num());
    } else if (op == "compare") {
        const int order = cmp(a, b);
        return std::to_string((order > 0) - (order < 0)) + "\n";
    } else throw std::invalid_argument("Unknown scalar operation.");
    return a.get_str() + "\n";
}
} // namespace atlas
