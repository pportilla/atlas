// Exact weights of a Bierstone–Milman coefficient presentation on one stratum.
#include "groebner.hpp"
#include <flint/fmpq.h>
#include <set>
#include <sstream>

namespace atlas {
namespace {
struct Weight {
    fmpq_t q;
    Weight() { fmpq_init(q); }
    Weight(const Weight& other) : Weight() { fmpq_set(q, other.q); }
    Weight(Weight&& other) noexcept : Weight() { fmpq_swap(q, other.q); }
    ~Weight() { fmpq_clear(q); }
    void read(std::istream& in) {
        std::string s;
        if (!(in >> s) || fmpq_set_str(q, s.c_str(), 10) || fmpz_is_zero(fmpq_denref(q)))
            throw std::invalid_argument("Invalid coefficient weight.");
        fmpq_canonicalise(q);
    }
    std::string text() const {
        char* value = fmpq_get_str(nullptr, 10, q);
        std::string s(value);
        flint_free(value);
        return s;
    }
    bool operator<(const Weight& other) const { return fmpq_cmp(q, other.q) < 0; }
};
}

std::string centre_weights(std::istream& in, const std::string& operation) {
    size_t width, count;
    if (!(in >> width >> count) || count == 0) throw std::invalid_argument("Expected a nonempty presentation.");
    std::vector<Weight> weights(count);
    std::vector<std::vector<Weight>> exponents(count, std::vector<Weight>(width));
    for (size_t i = 0; i < count; ++i) {
        weights[i].read(in);
        if (fmpq_sgn(weights[i].q) <= 0) throw std::invalid_argument("Coefficient weights must be positive.");
        for (auto& x : exponents[i]) x.read(in);
    }
    std::string out;
    if (operation == "bm_residual") {
        std::vector<Weight> alpha(width);
        for (size_t j = 0; j < width; ++j) {
            for (size_t i = 0; i < count; ++i) {
                Weight ratio;
                fmpq_div(ratio.q, exponents[i][j].q, weights[i].q);
                if (i == 0 || ratio < alpha[j]) fmpq_set(alpha[j].q, ratio.q);
            }
        }
        for (const auto& x : alpha) out += x.text() + " ";
        out += '\n';
        for (size_t i = 0; i < count; ++i) {
            for (size_t j = 0; j < width; ++j) {
                Weight e;
                fmpq_mul(e.q, alpha[j].q, weights[i].q);
                fmpq_sub(e.q, exponents[i][j].q, e.q);
                out += e.text() + " ";
            }
            out += '\n';
        }
        return out;
    }
    Weight threshold;
    if (operation == "bm_requirements") threshold.read(in);
    std::set<Weight> values{Weight()};
    size_t size;
    while (in >> size) {
        std::vector<size_t> subset(size);
        for (auto& j : subset) if (!(in >> j) || j >= width)
            throw std::invalid_argument("Unknown exceptional divisor in this stratum.");
        for (size_t i = 0; i < count; ++i) {
            Weight sum, value;
            for (auto j : subset) fmpq_add(sum.q, sum.q, exponents[i][j].q);
            if (operation == "bm_requirements") {
                fmpq_mul(value.q, threshold.q, weights[i].q);
                fmpq_sub(value.q, value.q, sum.q);
                Z ceiling;
                fmpz_cdiv_q(&ceiling.v, fmpq_numref(value.q), fmpq_denref(value.q));
                char* s = fmpz_get_str(nullptr, 10, &ceiling.v);
                out += std::string(s) + " ";
                flint_free(s);
            } else {
                long order;
                if (!(in >> order) || order < 0) throw std::invalid_argument("Invalid residual order.");
                for (long k = 0; k <= order; ++k) {
                    fmpq_add_si(value.q, sum.q, k);
                    fmpq_div(value.q, value.q, weights[i].q);
                    values.insert(value);
                }
            }
        }
        if (operation == "bm_requirements") out += '\n';
    }
    if (!in.eof()) throw std::invalid_argument("Invalid coefficient presentation.");
    if (operation == "bm_values") for (const auto& x : values) out += x.text() + "\n";
    return out;
}
}
extern "C" int atlas_centre_weights_abi() noexcept { return 1; }
