// Use the existing Atlas expression parser and FLINT pullback in the CLI process.
#include "local_io.hpp"
#include "groebner.hpp"
#include <flint/fmpq_mpoly.h>
#include <algorithm>
#include <sstream>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string_view>

extern "C" {
char* atlas_affine(const char*, size_t) noexcept;
void atlas_groebner_release(char*) noexcept;
const char* atlas_groebner_error() noexcept;
}

namespace atlas::local {
namespace {
std::string affine(const std::string& request) {
    std::unique_ptr<char, decltype(&atlas_groebner_release)> reply(
        atlas_affine(request.data(), request.size()), atlas_groebner_release);
    if (!reply) throw std::runtime_error(atlas_groebner_error());
    return reply.get();
}

std::string ring_line(const std::vector<std::string>& names) {
    std::string text;
    for (const auto& name : names) text += name + " ";
    return text + "\n";
}

}

Polynomial decode_row(std::string_view row, size_t width) {
    Polynomial result;
    while (!row.empty()) {
        const auto end = row.find(';');
        std::istringstream in(std::string{row.substr(0, end)});
        mpq_class coefficient;
        if (!(in >> coefficient) || coefficient.get_den() == 0)
            throw std::invalid_argument("Invalid sparse coefficient.");
        coefficient.canonicalize();
        Powers powers(width);
        for (auto& exponent : powers)
            if (!(in >> exponent) || exponent < 0)
                throw std::invalid_argument("Invalid sparse exponent.");
        std::string extra;
        if (in >> extra) throw std::invalid_argument("Invalid sparse exponent vector.");
        if (coefficient != 0 && !result.emplace(std::move(powers), std::move(coefficient)).second)
            throw std::invalid_argument("Repeated monomial in sparse input.");
        if (end == row.npos) break;
        row.remove_prefix(end + 1);
    }
    return result;
}

std::string encode_row(const Polynomial& f) {
    std::ostringstream row;
    bool first = true;
    for (const auto& [powers, c] : f) {
        if (!first) row << ';';
        row << c;
        for (const auto& e : powers) row << ' ' << e;
        first = false;
    }
    return row.str();
}

namespace {
using ActiveTerms = std::vector<const Polynomial::value_type*>;

std::uint64_t power_mod(std::uint64_t base, unsigned long exponent, std::uint64_t prime) {
    std::uint64_t value = 1;
    while (exponent) {
        if (exponent & 1) value = value * base % prime;
        exponent >>= 1;
        base = base * base % prime;
    }
    return value;
}

std::optional<std::uint64_t> residue(const mpq_class& q, unsigned long prime) {
    const auto den = mpz_fdiv_ui(mpq_denref(q.get_mpq_t()), prime);
    if (!den) return std::nullopt;
    const auto num = mpz_fdiv_ui(mpq_numref(q.get_mpq_t()), prime);
    return den == 1 ? num : num * power_mod(den, prime - 2, prime) % prime;
}

bool nonzero_modulo(const ActiveTerms& terms, const std::vector<mpq_class>& point, unsigned long prime) {
    std::vector<std::optional<std::uint64_t>> coordinates;
    for (const auto& q : point) coordinates.push_back(residue(q, prime));
    std::uint64_t sum = 0;
    for (const auto* term : terms) {
        auto c = residue(term->second, prime);
        if (!c) return false;
        for (std::size_t i = 0; i < point.size(); ++i) {
            const auto e = term->first[i];
            if (!e) continue;
            if (!coordinates[i]) return false;
            if (!*coordinates[i]) { *c = 0; break; }
            *c = *c * power_mod(*coordinates[i], mpz_fdiv_ui(e.get_mpz_t(), prime - 1), prime) % prime;
        }
        sum = (sum + *c) % prime;
    }
    return sum != 0;
}

std::optional<bool> nonzero_at_point(const Polynomial& f, const std::vector<mpq_class>& point) {
    ActiveTerms terms;
    bool positive = false, negative = false;
    // Estimate the size of powers and denominators before forming exact
    // rational terms. A larger estimate selects modular evaluation.
    constexpr unsigned long bit_budget = 65536;
    unsigned long bits = 0;
    auto charge = [&](unsigned long size, const mpz_class& copies) {
        if (size && copies > (bit_budget - bits) / size) bits = bit_budget;
        else bits += size * copies.get_ui();
    };
    for (const auto& term : f) {
        const auto& [powers, c] = term;
        if (powers.size() != point.size()) throw std::invalid_argument("Wrong width for point evaluation.");
        int sign = sgn(c);
        for (std::size_t i = 0; sign && i < powers.size(); ++i) {
            if (!powers[i]) continue;
            if (point[i] == 0) sign = 0;
            else if (point[i] < 0 && mpz_odd_p(powers[i].get_mpz_t())) sign = -sign;
        }
        if (!sign) continue;
        positive |= sign > 0;
        negative |= sign < 0;
        terms.push_back(&term);
        charge(mpz_sizeinbase(mpq_denref(c.get_mpq_t()), 2), 1);
        for (std::size_t i = 0; i < powers.size(); ++i) {
            if (!powers[i] || point[i] == 1 || point[i] == -1) continue;
            charge(mpz_sizeinbase(mpq_numref(point[i].get_mpq_t()), 2), powers[i]);
            charge(mpz_sizeinbase(mpq_denref(point[i].get_mpq_t()), 2), powers[i]);
        }
    }
    if (!positive || !negative) return !terms.empty();
    if (bits == bit_budget) {
        // These primes are below 2^30, so products of residues fit uint64_t.
        // Zero residues or noninvertible denominators cannot establish vanishing.
        for (unsigned long p : {1000000007UL, 1000000009UL, 998244353UL})
            if (nonzero_modulo(terms, point, p)) return true;
        return std::nullopt;
    }
    mpq_class value = 0;
    for (const auto* term : terms) {
        mpq_class c = term->second;
        for (std::size_t i = 0; i < point.size(); ++i) {
            auto e = term->first[i];
            if (!e || point[i] == 1) continue;
            if (point[i] == -1) {
                if (mpz_odd_p(e.get_mpz_t())) c = -c;
            } else {
                mpq_class power;
                mpz_pow_ui(mpq_numref(power.get_mpq_t()), mpq_numref(point[i].get_mpq_t()), e.get_ui());
                mpz_pow_ui(mpq_denref(power.get_mpq_t()), mpq_denref(point[i].get_mpq_t()), e.get_ui());
                c *= power;
            }
        }
        value += c;
    }
    return value != 0;
}
}

Polynomial parse_expression(const std::vector<std::string>& names, std::string source) {
    // The public grammar treats newlines as whitespace. The kernel protocol is line based.
    for (char& ch : source) if (ch == '\n' || ch == '\r') ch = ' ';
    auto row = affine("parse_input\n" + ring_line(names) + source + "\n");
    if (row.empty() || row.back() != '\n') throw std::runtime_error("Incomplete polynomial.");
    row.pop_back();
    return decode_row(row, names.size());
}

std::string format_polynomial(const Polynomial& f, const std::vector<std::string>& names) {
    if (f.empty()) return "0";
    using Entry = Polynomial::value_type;
    std::vector<std::pair<mpz_class, const Entry*>> sorted;
    sorted.reserve(f.size());
    for (const auto& t : f)
        sorted.emplace_back(std::accumulate(t.first.begin(), t.first.end(), mpz_class(0)), &t);
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first > b.first : a.second->first > b.second->first;
    });
    std::string text;
    for (const auto& [degree, entry] : sorted) {
        auto c = entry->second.get_str();
        if (c.front() == '-') { text += '-'; c.erase(0, 1); }
        else if (!text.empty()) text += '+';
        bool factor = c != "1" || degree == 0;
        if (factor) text += c;
        for (size_t i = 0; i < names.size(); ++i) {
            const auto k = entry->first[i];
            if (!k) continue;
            if (factor) text += '*';
            text += names[i];
            if (k > 1) text += "^" + k.get_str();
            factor = true;
        }
    }
    return text;
}

bool has_point_unit(const Basis& generators, const std::vector<mpq_class>& point) {
    if (std::all_of(point.begin(), point.end(), [](const auto& q) { return q == 0; })) {
        const Powers zero(point.size(), 0);
        return std::any_of(generators.begin(), generators.end(), [&](const Polynomial& f) {
            auto c = f.find(zero);
            return c != f.end() && c->second != 0;
        });
    }
    bool undecided = false;
    for (const auto& f : generators) {
        const auto nonzero = nonzero_at_point(f, point);
        if (nonzero.value_or(false)) return true;
        undecided |= !nonzero.has_value();
    }
    if (undecided) throw std::overflow_error("Use general exact rational point evaluation.");
    return false;
}

Basis project_monomials(const Basis& generators, const std::vector<mpq_class>& point) {
    std::vector<std::pair<mpz_class, Powers>> heads;
    for (const auto& f : generators) {
        if (f.empty()) continue;
        if (f.size() != 1 || f.begin()->first.size() != point.size())
            throw std::invalid_argument("Expected monomial generators in the point coordinates.");
        auto head = f.begin()->first;
        if (f.begin()->second == 0) continue;
        for (std::size_t i = 0; i < point.size(); ++i) if (point[i] != 0) head[i] = 0;
        const auto degree = std::accumulate(head.begin(), head.end(), mpz_class(0));
        if (!degree) return {Polynomial{{std::move(head), 1}}};
        heads.emplace_back(degree, std::move(head));
    }
    std::sort(heads.begin(), heads.end());
    Basis result;
    for (auto& head : heads) {
        const bool redundant = std::any_of(result.begin(), result.end(), [&](const Polynomial& f) {
            const auto& divisor = f.begin()->first;
            return std::equal(divisor.begin(), divisor.end(), head.second.begin(), std::less_equal<>());
        });
        if (!redundant) result.push_back(Polynomial{{std::move(head.second), 1}});
    }
    return result;
}

namespace {
// One FLINT ring and one set of substitutions x_i + p_i for the entire batch.
class CoordinateTranslation {
    fmpq_mpoly_ctx_t ring;
    std::vector<fmpq_mpoly_struct> coordinates;
    std::vector<fmpq_mpoly_struct*> images;
    fmpq_mpoly_t source, result;
    fmpq_t coefficient;
    std::vector<Z> storage;
    std::vector<fmpz*> powers;

public:
    explicit CoordinateTranslation(const std::vector<mpq_class>& shifts)
        : coordinates(shifts.size()), images(shifts.size()), storage(shifts.size()) {
        for (auto& e : storage) powers.push_back(&e.v);
        fmpq_mpoly_ctx_init(ring, shifts.size(), ORD_LEX);
        fmpq_mpoly_init(source, ring);
        fmpq_mpoly_init(result, ring);
        fmpq_init(coefficient);
        for (size_t i = 0; i < shifts.size(); ++i) {
            images[i] = &coordinates[i];
            fmpq_mpoly_init(images[i], ring);
            fmpq_mpoly_gen(images[i], i, ring);
            if (shifts[i] != 0) {
                fmpq_set_mpq(coefficient, shifts[i].get_mpq_t());
                fmpq_mpoly_add_fmpq(images[i], images[i], coefficient, ring);
            }
        }
    }

    CoordinateTranslation(const CoordinateTranslation&) = delete;
    CoordinateTranslation& operator=(const CoordinateTranslation&) = delete;

    ~CoordinateTranslation() {
        for (auto p : images) fmpq_mpoly_clear(p, ring);
        fmpq_mpoly_clear(source, ring);
        fmpq_mpoly_clear(result, ring);
        fmpq_clear(coefficient);
        fmpq_mpoly_ctx_clear(ring);
    }

    Polynomial apply(const Polynomial& f) {
        fmpq_mpoly_zero(source, ring);
        fmpq_mpoly_fit_length(source, f.size(), ring);
        // Distinct terms in reverse map order already match FLINT's descending lex order.
        for (auto it = f.rbegin(); it != f.rend(); ++it) {
            fmpq_set_mpq(coefficient, it->second.get_mpq_t());
            for (size_t i = 0; i < powers.size(); ++i)
                fmpz_set_mpz(powers[i], it->first[i].get_mpz_t());
            fmpq_mpoly_push_term_fmpq_fmpz(source, coefficient, powers.data(), ring);
        }
        fmpq_mpoly_reduce(source, ring);
        if (!fmpq_mpoly_compose_fmpq_mpoly(result, source, images.data(), ring, ring))
            throw std::overflow_error("The polynomial pullback exceeds FLINT's limits.");

        Polynomial translated;
        const slong length = fmpq_mpoly_length(result, ring);
        for (slong i = 0; i < length; ++i) {
            // A shift can only decrease the exponents of an input term.
            Powers powers(images.size());
            fmpq_mpoly_get_term_exp_fmpz(this->powers.data(), result, i, ring);
            for (size_t j = 0; j < powers.size(); ++j)
                fmpz_get_mpz(powers[j].get_mpz_t(), this->powers[j]);
            fmpq_mpoly_get_term_coeff_fmpq(coefficient, result, i, ring);
            auto term = translated.try_emplace(translated.begin(), std::move(powers));
            fmpq_get_mpq(term->second.get_mpq_t(), coefficient);
        }
        return translated;
    }
};
}

Basis translate(const Basis& inputs, const std::vector<mpq_class>& shifts) {
    if (inputs.empty() || std::all_of(shifts.begin(), shifts.end(), [](const auto& a) { return a == 0; }))
        return inputs;
    // Check the whole batch before expanding any row. The general point
    // handler can remove unit factors or use a wide sparse representation.
    for (const auto& f : inputs) {
        for (const auto& [e, c] : f) {
            if (e.size() != shifts.size()) throw std::invalid_argument("Wrong coordinate count for translation.");
            unsigned long count = 1;
            for (size_t i = 0; i < e.size(); ++i) {
                if (shifts[i] == 0) continue;
                if (e[i] >= 1000000 / count) throw std::overflow_error("Large coordinate translation.");
                count *= e[i].get_ui() + 1;
            }
        }
    }
    CoordinateTranslation translation(shifts);
    Basis output;
    output.reserve(inputs.size());
    for (const auto& f : inputs) output.push_back(translation.apply(f));
    return output;
}
}
