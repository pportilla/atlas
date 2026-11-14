#include "groebner_verify.hpp"

#include <iostream>

using namespace atlas;

static void term(Poly& f, const Ring& ring, long x, long y, long z, long coefficient,
                 long w = 0, long v = 0) {
    const long powers[] = {x, y, z, w, v};
    const size_t offset = f.m.size();
    f.m.resize(offset + ring.stride);
    ring.pack(f.m.data() + offset, powers);
    f.c.emplace_back();
    fmpz_set_si(&f.c.back().v, coefficient);
}

static bool agrees(const Ring& ring, const std::vector<Poly>& basis,
                   const std::vector<Poly>& generators, bool expected, const char* example) {
    Engine reference;
    reference.R = ring;
    bool sparse = reference.is_basis(basis);
    if (sparse) {
        for (Poly f : generators) {
            reference.reduce(f, 0, nullptr, nullptr);
            if (!f.empty()) {
                sparse = false;
                break;
            }
        }
    }
    auto shared = HomogeneousCertificate{}.check(ring, basis, generators);
    if (shared && *shared == expected && sparse == expected) return true;
    std::cerr << "Certificate mismatch for " << example << '\n';
    return false;
}

class GuardedReduction : public Engine {
public:
    void set_limits(size_t bits, double work = 0) {
        bit_limit_ = bits;
        work_limit_ = work;
        work_ = 0;
    }
};

static Poly read_row(const Ring& ring, const std::string& row, Rational* scale = nullptr) {
    const char* at = row.data();
    return parse(ring, at, at + row.size(), scale);
}

static bool check_coefficient_guard() {
    GuardedReduction engine;
    engine.R.init(3, 1);
    // With 17*y-z as divisor, y²+z² reduces to 290*z²/289. An 8-bit
    // limit makes the temporary 290 visible without large test constants.
    engine.compute({read_row(engine.R, "17 0 1 0;-1 0 0 1\n")}, 0);
    auto reduce = [&](const std::string& row, size_t keep = 0) {
        Rational scale;
        auto f = read_row(engine.R, row, &scale);
        engine.reduce(f, keep, &scale.num.v, &scale.den.v);
        std::string result;
        write(engine.R, result, f, &scale.num.v, &scale.den.v);
        return result;
    };
    auto expect_swell = [&](const std::string& row) {
        try { reduce(row); } catch (const Swell&) { return true; }
        return false;
    };

    // x is already settled. Cancelling the joint content must rescale the
    // settled coefficient, active suffix and the rational witness together.
    const std::string row = "-290/13 1 0 0;-1/13 0 2 0;-1/13 0 0 2\n";
    engine.set_limits(8);
    if (reduce(row) != "-290/13 1 0 0;-290/3757 0 0 2\n") return false;
    // With coefficient one on x, the common content is one and the
    // normalized head really does exceed the coefficient limit.
    if (!expect_swell("1 1 0 0;1 0 2 0;1 0 0 2\n")) return false;
    engine.set_limits(8, 1);
    if (!expect_swell(row)) return false;

    // Skip a monomial divisible by y between the kept x and pending z.
    // Its coefficient one must not enter the content of the live terms.
    engine.compute({read_row(engine.R, "1 0 1 0\n")}, 0);
    engine.set_limits(8);
    return reduce("257/3 1 0 0;1/3 0 1 0;-514/3 0 0 1\n", 1) ==
           "257/3 1 0 0;-514/3 0 0 1\n";
}

int main() {
    try {
        if (!check_coefficient_guard()) {
            std::cerr << "Incorrect content scaling or coefficient/work limit during division\n";
            return 1;
        }
    } catch (const Swell&) {
        std::cerr << "Removable content triggered a coefficient limit during division\n";
        return 1;
    }
    Ring ring;
    ring.init(5, 0, true);
    constexpr int degree = 8;
    std::vector<Poly> basis, generators;
    // The generators are all degree-eight monomials. The triangular change
    // x -> 2x + z preserves this ideal and gives nonmonic, overlapping rows.
    for (int i = 0; i <= degree; ++i) {
        for (int j = 0; i + j <= degree; ++j) {
            const int k = degree - i - j;
            generators.emplace_back();
            term(generators.back(), ring, i, j, k, 1);
            basis.emplace_back();
            long binomial = 1;
            for (int t = 0; t <= i; ++t) {
                if (t) binomial = binomial * (i - t + 1) / t;
                term(basis.back(), ring, i - t, j, k + t, binomial * (1L << (i - t)));
            }
        }
    }
    if (!agrees(ring, basis, generators, true, "the degree-eight ideal")) return 1;

    generators.emplace_back();
    term(generators.back(), ring, 0, 0, degree - 1, 1);
    if (!agrees(ring, basis, generators, false, "a generator outside the ideal")) return 1;

    // Three long but sparse chains exceed the shared graph budget. The
    // generators w^d - v^d belong to (w - v) by ordinary sparse reduction.
    auto wide = basis;
    wide.emplace_back();
    term(wide.back(), ring, 0, 0, 0, 1, 1, 0);
    term(wide.back(), ring, 0, 0, 0, -1, 0, 1);
    std::vector<Poly> chains;
    for (int d = 22000; d < 22003; ++d) {
        chains.emplace_back();
        term(chains.back(), ring, 0, 0, 0, 1, d, 0);
        term(chains.back(), ring, 0, 0, 0, -1, 0, d);
    }
    if (HomogeneousCertificate{}.check(ring, wide, chains).has_value()) {
        std::cerr << "Expected the wide certificate to use the sparse verifier\n";
        return 1;
    }
    Engine fallback;
    fallback.R = ring;
    if (!fallback.is_basis(wide)) return 1;
    for (Poly& f : chains) {
        fallback.reduce(f, 0, nullptr, nullptr);
        if (!f.empty()) return 1;
    }

    // Remove z^8, then change yz^7 to yz^7 + z^8. Its S-pair with
    // y^2 z^6 now has a nonzero normal form (a multiple of z^9).
    basis.erase(basis.begin());
    term(basis.front(), ring, 0, 0, degree, 1);
    return agrees(ring, basis, {}, false, "a nonzero S-pair") ? 0 : 1;
}
