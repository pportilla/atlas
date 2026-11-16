// Time the production solver from parsed generators to an owned, monic rational basis.
#include "groebner.hpp"
#include <flint/fmpq.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

namespace {

struct Coefficient {
    fmpq_t value;
    Coefficient() { fmpq_init(value); }
    Coefficient(const Coefficient&) = delete;
    Coefficient(Coefficient&& other) noexcept : Coefficient() { fmpq_swap(value, other.value); }
    ~Coefficient() { fmpq_clear(value); }
};

struct MonicPolynomial {
    std::vector<atlas::u64> monomials;
    std::vector<Coefficient> coefficients;

    bool operator==(const MonicPolynomial& other) const {
        if (monomials != other.monomials || coefficients.size() != other.coefficients.size()) return false;
        for (size_t i = 0; i < coefficients.size(); ++i)
            if (!fmpq_equal(coefficients[i].value, other.coefficients[i].value)) return false;
        return true;
    }
};

using Basis = std::vector<MonicPolynomial>;

Basis solve(const atlas::Ring& ring, const std::vector<atlas::Poly>& generators) {
    atlas::Engine engine;
    engine.R = ring;
    engine.compute(generators, 0);
    Basis result;
    result.reserve(engine.G.size());
    for (atlas::Element& element : engine.G) {
        auto& p = element.p;
        auto& monic = result.emplace_back();
        monic.monomials = std::move(p.m);
        monic.coefficients.resize(p.size());
        for (size_t i = 0; i < p.size(); ++i)
            fmpq_set_fmpz_frac(monic.coefficients[i].value, &p.c[i].v, &p.c.front().v);
    }
    return result;
}

void write_basis(const atlas::Ring& ring, const Basis& basis) {
    std::cout << "BASIS " << basis.size() << '\n';
    long exponents[atlas::MAX_VARIABLES];
    for (const MonicPolynomial& polynomial : basis) {
        for (size_t i = 0; i < polynomial.coefficients.size(); ++i) {
            if (i) std::cout << ';';
            char* text = fmpq_get_str(nullptr, 10, polynomial.coefficients[i].value);
            std::cout << text;
            flint_free(text);
            ring.unpack(polynomial.monomials.data() + i * ring.stride, exponents);
            for (int k = 0; k < ring.n; ++k) std::cout << ' ' << exponents[k];
        }
        std::cout << '\n';
    }
    std::cout.flush();
}

void run(const std::string& request) {
    int rows, batch, variables, order, known;
    if (!(std::istringstream(request) >> rows >> batch) || rows < 0 || batch < 1)
        throw std::runtime_error("Expected a generator count and positive batch size.");
    std::string line;
    if (!std::getline(std::cin, line) || !(std::istringstream(line) >> variables >> order >> known) ||
        known != 0 || (order != 0 && order != 1))
        throw std::runtime_error("Expected a lex or grevlex ring with no generators declared to be a basis.");
    atlas::Ring ring;
    ring.init(variables, order);
    std::vector<atlas::Poly> generators;
    generators.reserve(rows);
    for (int i = 0; i < rows; ++i) {
        if (!std::getline(std::cin, line)) throw std::runtime_error("Missing input polynomial.");
        line.push_back('\n');
        const char* begin = line.data();
        generators.push_back(atlas::parse(ring, begin, begin + line.size(), nullptr));
    }

    using Clock = std::chrono::steady_clock;
    solve(ring, generators); // Warm this case and release its basis before timing.
    std::vector<Basis> results(batch);
    const auto start = Clock::now();
    for (Basis& result : results) result = solve(ring, generators);
    const std::chrono::duration<double, std::milli> elapsed = Clock::now() - start;
    for (int i = 0; i + 1 < batch; ++i)
        if (results[i] != results.back()) throw std::runtime_error("Atlas returned inconsistent bases in one batch.");
    std::cout << "TIME " << std::setprecision(12) << elapsed.count() / batch << '\n';
    write_basis(ring, results.back());
}

} // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    try {
        std::string request;
        while (std::getline(std::cin, request)) run(request);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
