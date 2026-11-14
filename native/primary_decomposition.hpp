#pragma once

#include "local_basis.hpp"
#include <istream>
#include <json/value.h>
#include <optional>
#include <string>

namespace atlas::quotient_ring { class Ring; }

namespace atlas::primary {
using local::Basis;
using local::Polynomial;
using local::Powers;
struct Component { Basis primary; Basis prime; };
using Components = std::vector<Component>;

// Bound optional decomposition work, including recursively requested bases
// and intersections. Ordinary primary decomposition runs without this scope.
class CertificateBudget {
    size_t remaining_;
    size_t* previous_;
public:
    explicit CertificateBudget(size_t work);
    ~CertificateBudget();
    CertificateBudget(const CertificateBudget&) = delete;
};

// Shared exact ideal arithmetic. The dimension routine takes a lex Gröbner basis.
bool member_lex(const Polynomial& f, const Basis& lex_basis);
// Deduct the number of divisor terms used. nullopt means the budget ran out.
std::optional<bool> member_lex(const Polynomial& f, const Basis& lex_basis, size_t& remaining);
Basis radical_lex(const Basis& rows, size_t width);
Basis intersection_lex(const Basis& a, const Basis& b, size_t width);
Basis saturate_lex(const Basis& rows, const Polynomial& h, size_t width);
int dimension_lex(const Basis& rows, size_t width);
Polynomial multiply(const Polynomial& a, const Polynomial& b);
Basis reduced(const Basis& rows, size_t width, bool lex);
Basis reduced(const Basis& rows, size_t width, bool lex, bool modular);
// Factors are monic for lex order. Scalar factors are omitted.
std::vector<std::pair<Polynomial, mpz_class>> factors(const Polynomial& f, size_t width);
Polynomial power(Polynomial f, mpz_class exponent, size_t width);
// Replace one coordinate by its image in the ring of result_width variables.
// Reuse the power table for successive inputs under the same substitution.
Polynomial substitute(const Polynomial& f, size_t axis, const Polynomial& image,
                      size_t result_width, std::map<mpz_class, Polynomial>& powers);
std::optional<Components> split_comaximal(const Basis& rows, size_t width, bool lex);
std::optional<Components> special_decomposition(const Basis& rows, size_t width, bool lex);
Components decompose(const Basis& rows, size_t width, bool lex);
using FractionMap = std::vector<std::pair<Polynomial,Polynomial>>;
// Test inside the source coordinate ring, after clearing denominators which
// have already been certified regular at this generic point.
bool vanishes_under_map(const Polynomial& f, const FractionMap& images,
                        const quotient_ring::Ring& source, size_t width);
Json::Value prepare_overlap(const Json::Value& input);
int generic_overlap(const Json::Value& input);
std::string operation(std::istream& input, const std::string& op);
}
