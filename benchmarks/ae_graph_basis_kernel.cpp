// Time the lexicographic Groebner basis of the graph ideal that graded
// elimination (native/ae_graded.cpp) builds for (x^a, y^b, (x+y)^c).
//
//   make -C native -f groebner.mk build/libgroebner.so
//   g++ -O2 -std=c++17 -Inative benchmarks/ae_graph_basis_kernel.cpp -o /tmp/ae_graph_basis -Lnative/build -Wl,-rpath,$PWD/native/build -lgroebner -lflint -lgmp
//   /tmp/ae_graph_basis 3 4 5 compute
//
// The route is `compute` (Engine::compute, as packed_lex_heads calls it),
// `hilbert` (the Hilbert-driven route alone) or `modular` (modular images).
// The output gives the size of the reduced basis, its largest coefficient and
// fingerprints of its leading monomials and of all its terms, so two builds
// can be checked for the same basis.
#include "groebner.hpp"

#include <flint/fmpz.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

using namespace atlas;

namespace {

std::string term(const fmpz_t coefficient, const std::vector<long>& exponents) {
    char* digits = fmpz_get_str(nullptr, 10, coefficient);
    std::ostringstream text;
    text << digits;
    flint_free(digits);
    for (long e : exponents) text << ' ' << e;
    return text.str();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s a b c compute|hilbert|modular\n", argv[0]);
        return 2;
    }
    const int a = std::atoi(argv[1]), b = std::atoi(argv[2]), c = std::atoi(argv[3]);
    const std::string route = argv[4];
    if (a < 1 || b < 1 || c < 1 || (route != "compute" && route != "hilbert" && route != "modular")) {
        std::fprintf(stderr, "Give three positive degrees and a route.\n");
        return 2;
    }
    // Variables x1 > x2 > y1 > y2 > y3 > z1 > z2 > z3, as in ae_graded.
    constexpr int n = 2, p = 3, base = n + p, width = base + p;
    struct Term { long x, y; };  // binomial coefficient C(c, x) for (x+y)^c
    std::vector<std::vector<Term>> map(3);
    map[0].push_back({a, 0});
    map[1].push_back({0, b});
    for (int k = 0; k <= c; ++k) map[2].push_back({k, c - k});
    fmpz_t coefficient;
    fmpz_init(coefficient);
    auto value = [&](int j, const Term& t) {
        if (j == 2) fmpz_bin_uiui(coefficient, c, t.x); else fmpz_one(coefficient);
    };
    std::vector<std::string> rows;
    for (int j = 0; j < p; ++j) {  // y_j - f_j
        std::vector<long> e(width, 0);
        e[n + j] = 1;
        fmpz_one(coefficient);
        std::string row = term(coefficient, e);
        for (const auto& t : map[j]) {
            std::vector<long> f(width, 0);
            f[0] = t.x;
            f[1] = t.y;
            value(j, t);
            fmpz_neg(coefficient, coefficient);
            row += ";" + term(coefficient, f);
        }
        rows.push_back(row);
    }
    for (int i = 0; i < p; ++i) for (int j = i; j < p; ++j) {  // square-zero markers
        std::vector<long> e(width, 0);
        ++e[base + i];
        ++e[base + j];
        fmpz_one(coefficient);
        rows.push_back(term(coefficient, e));
    }
    for (int axis = 0; axis < n; ++axis) {  // sum_j (d f_j / d x_axis) z_j
        std::string row;
        for (int j = 0; j < p; ++j) for (const auto& t : map[j]) {
            const long power = axis == 0 ? t.x : t.y;
            if (!power) continue;
            std::vector<long> f(width, 0);
            f[0] = t.x;
            f[1] = t.y;
            --f[axis];
            f[base + j] = 1;
            value(j, t);
            fmpz_mul_si(coefficient, coefficient, power);
            if (!row.empty()) row += ";";
            row += term(coefficient, f);
        }
        rows.push_back(row);
    }
    fmpz_clear(coefficient);

    Ring ring;
    ring.init(width, 1);
    std::vector<Poly> polys;
    for (const auto& row : rows) {
        const std::string line = row + "\n";
        const char* at = line.data();
        polys.push_back(parse(ring, at, at + line.size(), nullptr));
    }
    Engine engine;
    engine.R = ring;
    const auto start = std::chrono::steady_clock::now();
    bool done = true;
    if (route == "compute") engine.compute(polys, 0);
    else if (route == "hilbert") done = hilbert_basis(ring, polys, engine, 0, 0);
    else done = modular_basis(ring, polys, engine, nullptr, static_cast<size_t>(1) << 30);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (!done) {
        std::printf("route=%s declined after %.3f s\n", route.c_str(), seconds);
        return 1;
    }
    unsigned long long heads = 1469598103934665603ULL, all = heads;
    size_t terms = 0, bits = 0;
    const auto mix = [](unsigned long long& h, unsigned long long v) { h = (h ^ v) * 1099511628211ULL; };
    for (const auto& g : engine.G) {
        for (int i = 0; i < ring.stride; ++i) mix(heads, g.p.m[i]);
        for (size_t t = 0; t < g.p.size(); ++t) {
            ++terms;
            bits = std::max<size_t>(bits, fmpz_bits(&g.p.c[t].v));
            mix(all, fmpz_fdiv_ui(&g.p.c[t].v, 1000000007UL));
            for (int i = 0; i < ring.stride; ++i) mix(all, g.p.m[t * ring.stride + i]);
        }
    }
    std::printf("route=%s seconds=%.3f basis=%zu terms=%zu max_bits=%zu heads=%016llx all=%016llx\n",
                route.c_str(), seconds, engine.G.size(), terms, bits, heads, all);
    return 0;
}
