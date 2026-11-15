#pragma once

#include "local_basis.hpp"
#include "ideal_power_principal.hpp"
#include "groebner.hpp"
#include <variant>
#include <istream>
#include <json/json.h>
#include <stdexcept>

namespace atlas::ideal_powers {
using local::Basis;
class InvalidInput : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};
mpz_class parse_exponent(const std::string& text);
std::optional<Basis> content_symbolic(const Basis& rows, size_t width, bool lex, const mpz_class& exponent);
struct PackedBasis {
    Ring ring;
    std::vector<Poly> rows;
};

class PowerResult {
    std::variant<Basis,PackedBasis,PrincipalBasis> value_;
public:
    PowerResult(PrincipalBasis value) : value_(std::move(value)) {}
    PowerResult(Basis value) : value_(std::move(value)) {}
    PowerResult(PackedBasis value) : value_(std::move(value)) {}
    bool is_packed() const { return std::holds_alternative<PackedBasis>(value_); }
    Basis materialize() const &;
    Basis materialize() &&;
    std::vector<std::string> expressions(const std::vector<std::string>& names) const;
};
PowerResult monomial_power(const Basis& rows, size_t width, bool lex, const mpz_class& exponent, bool symbolic);
PowerResult typed_power(const Basis& rows, size_t width, bool lex, const mpz_class& exponent, bool symbolic);
// An empty result defers the operation to the cancellable worker.
std::optional<PowerResult> bounded_ordinary_power(const Basis& rows, size_t width, bool lex, const mpz_class& exponent);
std::optional<PackedBasis> packed_power(const Basis& rows, size_t width, bool lex, const mpz_class& exponent,
                                        bool bounded = false);
Basis power(const Basis& rows, size_t width, bool lex, const mpz_class& exponent, bool symbolic);
Json::Value compute(const Basis& rows, const std::vector<std::string>& names, bool lex,
                    const mpz_class& exponent, bool symbolic);
std::string operation(std::istream& input, bool symbolic);
}
