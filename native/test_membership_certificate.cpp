// A matrix truncated by the maximum degree can miss a cofactor or stop at
// a partial set of columns. Every successful test verifies the full identity.
#include "membership_certificate.hpp"
#include <iostream>
#include <random>

using namespace atlas::chart;
using namespace atlas::certificate;

namespace {
void assert_ok(bool value, const char* detail) {
    if (!value) throw std::logic_error(detail);
}
bool exactly_divided(const Ideal& given, const Expr& expected, const Ideal& cofactors) {
    Expr sum;
    if (given.size() != cofactors.size()) return false;
    for (size_t i = 0; i < given.size(); ++i) sum = sum + given[i] * cofactors[i];
    return sum == expected;
}
}  // namespace

int main() {
    const Variables over{{"z", 0}, {"x", 0}, {"x", 1}};
    const Expr z = symbol(over[0]), x = symbol(over[1]), y = symbol(over[2]), one = constant(1);
    const CertificateCaps caps;
    const auto product = x * y - z;
    std::mt19937 rng(12);
    for (unsigned trial = 0; trial < 72; ++trial) {
        auto coefficient = [&]() {
            Q q(static_cast<long>(rng() % 31) - 15, rng() % 17 + 1);
            q.canonicalize();
            return q;
        };
        const Ideal given{scaled(x, coefficient()) + product + constant(coefficient()),
                          scaled(y, coefficient()) + y * z - x, {}};
        const Expr a = scaled(x, coefficient()) + constant(coefficient());
        const Expr b = scaled(z * x, coefficient()) + scaled(y, coefficient());
        const Expr f = a * given[0] + b * given[1];
        auto answer = certify_member(over, given, f, caps);
        assert_ok(answer.certified && exactly_divided(given, f, answer.cofactors), "timed identity with planted cofactors");
        answer.cofactors[0] = answer.cofactors[0] + one;
        assert_ok(!verify_member(over, given, f, answer.cofactors), "changed multiplier accepted");
    }
    const Ideal away{z * z, x * y};
    const auto origin = certify_empty_on_open(over, away, z, caps);
    const auto radical = certify_radical_member(over, away, z, caps);
    assert_ok(origin.certified && radical.certified && origin.auxiliary, "overlapping divisor");
    const Expr adjacent = symbol(*origin.auxiliary) * z - one;
    assert_ok(exactly_divided({away[0], away[1], adjacent}, one, origin.cofactors), "incorrect open certificate");
    assert_ok(!certify_radical_member(over, away, x, caps).certified, "incorrect radical membership");
    assert_ok(!certify_member(over, {x * x - one}, x, caps).certified, "nonmember marked certified");
    assert_ok(certify_unit(over, {x * y - one, x}, caps).certified, "missed unit on line");
    assert_ok(!verify_member(over, {x}, one, {}), "missing multiplier");
    // The exceptional scalar below can make a nonconstant polynomial look
    // constant during liberal reduction modulo the first prime.
    const Z test_prime("4611686018427387847");
    const Expr odd = one + scaled(x, Q(test_prime));
    assert_ok(!certify_unit(over, {odd}, caps).certified, "nilpotent on the full polynomial");
    const Expr near_prime = scaled(x, Q(1, test_prime)) + y;
    const Expr multiple = scaled(y, Q(17) / Q((Z(1) << 140) + 1));
    const Expr target = near_prime * multiple;
    auto test = certify_member(over, {near_prime}, target, caps);
    assert_ok(test.certified && exactly_divided({near_prime}, target, test.cofactors), "test of a constant divisor with a large numerator");
    const Expr wide = normal({over[1]}, {{Powers{Z(1) << 100}, Q(23)}});
    assert_ok(!certify_member(over, {wide}, x * wide, caps).certified, "exact degree exceeded");
    for (unsigned axis = 0; axis < 4; ++axis) {
        auto budget = caps;
        if (axis == 0) budget.max_cols = 2;
        if (axis == 1) budget.max_rank = 0;
        if (axis == 2) budget.max_rows = 1;
        if (axis == 3) budget.max_nonzeros = 3;
        assert_ok(!certify_unit(over, {x * y - one, x}, budget).certified, "linear budget overrun");
    }
    assert_ok(certify_unit(over, {x * y - one, x}, caps).certified, "exact solution when caps allow it");
    std::cout << "Bounded membership certificates pass.\n";
}
