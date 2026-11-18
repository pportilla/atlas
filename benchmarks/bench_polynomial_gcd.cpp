// Stock FLINT comparison, preserving the pre-optimization GCD and normalization timing.
#include <flint/fmpq_mpoly.h>
#include <flint/mpoly.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::string line() {
    std::string value;
    if (!std::getline(std::cin, value)) throw std::invalid_argument("Incomplete GCD input.");
    return value;
}

struct Operands {
    fmpq_mpoly_ctx_t ctx;
    fmpq_mpoly_t left, right, expected, result;

    explicit Operands(slong variables) {
        fmpq_mpoly_ctx_init(ctx, variables, ORD_DEGREVLEX);
        fmpq_mpoly_init(left, ctx);
        fmpq_mpoly_init(right, ctx);
        fmpq_mpoly_init(expected, ctx);
        fmpq_mpoly_init(result, ctx);
    }
    Operands(const Operands&) = delete;
    ~Operands() {
        fmpq_mpoly_clear(result, ctx);
        fmpq_mpoly_clear(expected, ctx);
        fmpq_mpoly_clear(right, ctx);
        fmpq_mpoly_clear(left, ctx);
        fmpq_mpoly_ctx_clear(ctx);
    }

    void gcd() {
        if (!fmpq_mpoly_gcd(result, left, right, ctx)) throw std::runtime_error("FLINT GCD failed.");
        if (!fmpq_mpoly_is_zero(result, ctx)) fmpq_mpoly_make_monic(result, result, ctx);
    }
};
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Expected a positive batch size.");
        const std::string argument(argv[1]);
        std::size_t parsed;
        const int batch = std::stoi(argument, &parsed);
        if (parsed != argument.size() || batch < 1) throw std::invalid_argument("Invalid batch size.");

        std::istringstream variables(line());
        std::vector<std::string> names;
        for (std::string name; variables >> name;) names.push_back(name);
        if (names.empty()) throw std::invalid_argument("Expected coordinate names.");
        std::vector<const char*> symbols;
        for (const auto& name : names) symbols.push_back(name.c_str());
        Operands pair(names.size());
        for (auto* operand : {pair.left, pair.right, pair.expected}) {
            if (fmpq_mpoly_set_str_pretty(operand, line().c_str(), symbols.data(), pair.ctx))
                throw std::invalid_argument("FLINT could not parse a polynomial.");
        }

        pair.gcd();
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < batch; ++i) pair.gcd();
        const auto elapsed = std::chrono::steady_clock::now() - start;
        if (!fmpq_mpoly_equal(pair.result, pair.expected, pair.ctx))
            throw std::runtime_error("The GCD differs from the expected monic polynomial.");

        const std::chrono::duration<double, std::nano> ns = elapsed;
        std::cout << std::setprecision(12) << "{\"ns_per_gcd\":" << ns.count() / batch
                  << ",\"terms\":[" << fmpq_mpoly_length(pair.left, pair.ctx)
                  << ',' << fmpq_mpoly_length(pair.right, pair.ctx)
                  << ',' << fmpq_mpoly_length(pair.result, pair.ctx)
                  << "],\"flint_version\":\"" << flint_version << "\"}\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
