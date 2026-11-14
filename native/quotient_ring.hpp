#pragma once

#include "local_basis.hpp"
#include <istream>
#include <json/json.h>
#include <stdexcept>
#include <variant>

namespace atlas::quotient_ring {
using local::Basis;
using local::Polynomial;
using local::Powers;
using Value = std::variant<Polynomial, bool, std::monostate>;

struct InvalidInput : std::invalid_argument { using std::invalid_argument::invalid_argument; };

class PackedReducer;

class Ring {
    struct Reduced {};
    Ring(Basis basis, size_t width, bool lex, Reduced);
    size_t width_;
    bool lex_;
    Basis basis_;
    std::vector<Powers> heads_;
    bool zero_, binomial_;
    std::unique_ptr<PackedReducer> packed_;
    bool standard(const Powers& powers) const;
    Polynomial direct(Polynomial f) const;
    Polynomial normal(const Polynomial& f) const;
    Polynomial product(const Polynomial& a, const Polynomial& b) const;
    Polynomial power(Polynomial a, mpz_class exponent) const;
    std::optional<Polynomial> inverse(const Polynomial& a) const;
public:
    ~Ring();
    Ring(Ring&&) noexcept;
    Ring(const Basis& relations, size_t width, bool lex);
    static Ring from_basis(Basis basis, size_t width, bool lex);
    const Basis& basis() const { return basis_; }
    Value evaluate(const std::string& action, const Basis& operands, const mpz_class& exponent = 0) const;
    Json::Value render(const Value& value, const std::vector<std::string>& names) const;
};

mpz_class parse_exponent(const std::string& text);
Json::Value compute(const Basis& relations, const std::vector<std::string>& names, bool lex,
                    const std::string& action, const Basis& operands, const mpz_class& exponent = 0);
std::string operation(std::istream& in, bool prepared = false);
}
