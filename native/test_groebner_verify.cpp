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

int main() {
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
