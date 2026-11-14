// Nonzerodivisors on polynomial graph frames over QQ.
#include "graph_regular.hpp"
#include "local_io.hpp"
#include "local_monomial.hpp"
#include "primary_decomposition.hpp"

#include <algorithm>

namespace atlas {
namespace {
using local::Basis;
using local::Polynomial;
using local::Powers;

constexpr size_t term_budget = 65536;
constexpr size_t byte_budget = 16 * 1024 * 1024;
constexpr size_t coefficient_bits = 16384;

bool scalar(const Polynomial& f) {
    return f.size() == 1 && local::total_degree(f.begin()->first) == 0;
}

size_t height(const mpq_class& c) {
    return std::max(mpz_sizeinbase(c.get_num_mpz_t(), 2), mpz_sizeinbase(c.get_den_mpz_t(), 2));
}

// With one common denominator, all numerators of f are bounded by the
// largest original numerator times that denominator. Products can use these
// same denominators, including when summands have cancelling coefficients.
std::optional<size_t> cleared_height(const Polynomial& f) {
    mpz_class denominator = 1;
    size_t numerator_bits = 0;
    for (const auto& term : f) {
        const auto& c = term.second;
        mpz_lcm(denominator.get_mpz_t(), denominator.get_mpz_t(), c.get_den_mpz_t());
        numerator_bits = std::max(numerator_bits, mpz_sizeinbase(c.get_num_mpz_t(), 2));
        const size_t bits = mpz_sizeinbase(denominator.get_mpz_t(), 2) + numerator_bits;
        if (bits > coefficient_bits) return std::nullopt;
    }
    return mpz_sizeinbase(denominator.get_mpz_t(), 2) + numerator_bits;
}

// Bound the number of raw expanded terms before composing, including terms
// that may cancel. A monomial image can also raise a large rational to a very
// large power, even when its support costs only one term.
bool affordable(const Basis& rows, size_t omitted, size_t axis, const Polynomial& image) {
    auto image_bits = cleared_height(image);
    if (!image_bits) return false;
    // No coefficient growth from powers of a signed monomial.
    if (image.size() == 1 && (image.begin()->second == 1 || image.begin()->second == -1))
        image_bits = 0;
    size_t terms = 0, bytes = 0;
    for (size_t i = 0; i < rows.size(); ++i) if (i != omitted) {
        const auto source_bits = cleared_height(rows[i]);
        if (!source_bits) return false;
        size_t row_terms = 0, growth = 0;
        for (const auto& term : rows[i]) {
            const auto& exponent = term.first[axis];
            size_t count = 1;
            if (exponent != 0) {
                if (image.empty()) continue;
                if (*image_bits) {
                    if (exponent > coefficient_bits / *image_bits) return false;
                    growth = std::max(growth, exponent.get_ui() * *image_bits);
                }
                if (image.size() > 1) {
                    // image.size()^exponent bounds the expanded support.
                    if (exponent > 16) return false;
                    for (size_t power = 0; power < exponent.get_ui(); ++power) {
                        if (count > term_budget / image.size()) return false;
                        count *= image.size();
                    }
                }
            }
            if (count > term_budget - terms) return false;
            terms += count;
            row_terms += count;
        }
        // A common denominator for this row is D_source * D_image^max_power.
        // Adding at most 2^16 contributions can add at most 16 numerator bits.
        // Budget both numerator and denominator, for every possible output.
        const size_t term_bytes = 2 * ((*source_bits + growth + 16) / 8 + 1);
        if (row_terms > (byte_budget - bytes) / term_bytes) return false;
        bytes += row_terms * term_bytes;
    }
    return true;
}

// Eliminate a row a*x + b with scalar a != 0 and b independent of x. This
// gives an isomorphism of quotient rings, including nilpotents. Subsequent
// scalar pivots may become visible only after the earlier substitution.
bool eliminate_graph(Basis& rows, size_t width) {
    for (size_t i = 0; i + 1 < rows.size(); ++i) {
        const auto& f = rows[i];
        for (size_t axis = width; axis-- > 0;) {
            Powers variable(width);
            variable[axis] = 1;
            auto pivot = f.find(variable);
            if (pivot == f.end()) continue;
            if (std::any_of(f.begin(), f.end(), [&](const auto& term) {
                    return term.first[axis] != 0 && term.first != variable;
                })) continue;
            // Dividing every term by a wide scalar can already expand the
            // graph image before it is used in any of the other equations.
            if (height(pivot->second) / 8 + 1 > byte_budget / f.size()) continue;
            Polynomial image;
            for (const auto& [powers, c] : f) if (powers != variable) {
                auto reduced = powers;
                reduced.erase(reduced.begin() + axis);
                image.emplace(std::move(reduced), -c / pivot->second);
            }
            if (!affordable(rows, i, axis, image)) continue;
            rows.erase(rows.begin() + i);
            std::map<mpz_class, Polynomial> powers{{0, {{Powers(width - 1), 1}}}, {1, image}};
            for (auto& row : rows)
                row = primary::substitute(row, axis, image, width - 1, powers);
            return true;
        }
    }
    return false;
}

bool suitable_for_gcd(const Polynomial& f) {
    // Sparse exponents are arbitrary precision on input. Decline large dense
    // degrees before FLINT chooses a gcd algorithm. Its QQ representation uses
    // one denominator per polynomial, so also bound the cleared integer rows.
    if (local::max_total_degree(f) > 4096) return false;
    const auto bits = cleared_height(f);
    return bits && *bits <= 8 * byte_budget / f.size();
}

// A nonzero residue on any fibre of a prime is an exact global nonmembership
// witness. Specialize coordinates that are not the first variable of a lex
// leading monomial; these are natural parameter candidates in triangular
// bases. An empty or uninformative fibre leaves the original prime test.
bool nonzero_on_fibre(const Polynomial& f, const Basis& prime, size_t width, size_t& remaining) {
    std::vector<bool> parameter(width, true);
    for (const auto& row : prime) {
        const auto& powers = row.rbegin()->first;
        auto axis = std::find_if(powers.begin(), powers.end(), [](const auto& e) { return e != 0; });
        if (axis != powers.end()) parameter[axis - powers.begin()] = false;
    }
    const size_t free = std::count(parameter.begin(), parameter.end(), true);
    if (!free) return false;
    // Evaluation at 1 removes these exponents without raising coefficients to
    // powers, including when the input exponents are wider than machine words.
    auto specialize = [&](const Polynomial& row) {
        Polynomial image;
        for (const auto& [powers, c] : row) {
            Powers exponents;
            exponents.reserve(width - free);
            for (size_t i = 0; i < width; ++i)
                if (!parameter[i]) exponents.push_back(powers[i]);
            auto [it, fresh] = image.try_emplace(std::move(exponents), c);
            if (!fresh && (it->second += c) == 0) image.erase(it);
        }
        return image;
    };
    auto value = specialize(f);
    if (value.empty()) return false;
    Basis equations;
    for (const auto& row : prime) {
        auto image = specialize(row);
        if (scalar(image)) return false;
        if (!image.empty()) equations.push_back(std::move(image));
    }
    const auto fibre = primary::reduced(equations, width - free, true);
    if (std::any_of(fibre.begin(), fibre.end(), scalar)) return false;
    return primary::member_lex(value, fibre, remaining) == false;
}

bool nonmember(const Polynomial& f, const Basis& prime, size_t width, size_t& remaining) {
    if (f.empty()) return false;
    if (f.size() >= 64 && nonzero_on_fibre(f, prime, width, remaining)) return true;
    return primary::member_lex(f, prime, remaining) == false;
}

bool reconstructed(const Basis& rows, const primary::Components& components, size_t width) {
    if (components.empty()) return false;
    Basis intersection = components.front().primary;
    for (size_t i = 1; i < components.size(); ++i)
        intersection = primary::intersection_lex(intersection, components[i].primary, width);
    return intersection == primary::reduced(rows, width, true);
}

// In B[t], suppose the equations have the form J, t^k H, where J and H
// are in B and k>0. If J has height equal to its generator count, B/J is
// a complete intersection. Once H is proved regular on B/J, t^k H is
// regular too. The new quotient is thus unmixed, including when J, H or
// t^k carry multiplicities. Its support is covered by V(P,t) and V(P,H)
// for the associated primes P of J. On the latter, t is a free variable.
bool product_section_regular(const Basis& rows, size_t width, size_t axis, size_t section) {
    const size_t n = width - 1;
    std::vector<size_t> axes;
    for (size_t i = 0; i < width; ++i) if (i != axis) axes.push_back(i);
    // Sparse coordinates first give smaller lex elimination blocks. The
    // choice changes the work only, never the component certificate.
    std::vector<size_t> activity(width);
    for (size_t i = 0; i + 1 < rows.size(); ++i) if (i != section)
        for (const auto& term : rows[i])
            for (auto j : axes) activity[j] += term.first[j] != 0;
    std::stable_sort(axes.begin(), axes.end(), [&](size_t a, size_t b) {
        return activity[a] < activity[b];
    });
    auto project = [&](const Powers& powers) {
        Powers image;
        image.reserve(n);
        for (auto j : axes) image.push_back(powers[j]);
        return image;
    };
    Basis base;
    Polynomial divisor;
    for (size_t i = 0; i + 1 < rows.size(); ++i) {
        Polynomial image;
        for (const auto& [e, c] : rows[i]) image.emplace(project(e), c);
        if (i == section) divisor = std::move(image);
        else base.push_back(std::move(image));
    }
    // In V(P,t), only the constant coefficient survives. On V(P,H),
    // viewed in the polynomial ring over a cut prime, any coefficient suffices.
    std::map<mpz_class, Polynomial> coefficients;
    for (const auto& [e, c] : rows.back()) coefficients[e[axis]].emplace(project(e), c);
    if (!coefficients.count(0)) return false;

    primary::CertificateBudget budget(1u << 20);
    const auto base_components = primary::decompose(base, n, true);
    if (!reconstructed(base, base_components, n)) return false;
    const int expected = static_cast<int>(n) - static_cast<int>(base.size());
    size_t remaining = 1u << 19;
    for (const auto& part : base_components) {
        const auto& prime = part.prime;
        if (primary::dimension_lex(prime, n) != expected ||
            !nonmember(divisor, prime, n, remaining) ||
            !nonmember(coefficients.at(0), prime, n, remaining)) return false;
        Basis cut = prime;
        cut.push_back(divisor);
        // Reduce redundant input in a degree order before finding components.
        cut = primary::reduced(cut, n, false);
        if (std::any_of(cut.begin(), cut.end(), scalar)) continue;  // empty cut
        const auto pieces = primary::decompose(cut, n, true);
        if (!reconstructed(cut, pieces, n)) return false;
        for (const auto& piece : pieces) {
            // Over B/P'[t] it suffices that one coefficient is nonzero mod P'.
            if (std::none_of(coefficients.begin(), coefficients.end(), [&](const auto& term) {
                    return nonmember(term.second, piece.prime, n, remaining);
                })) return false;
        }
    }
    return true;
}

bool product_section(const Basis& rows, size_t width) {
    for (size_t axis = 0; axis < width; ++axis) {
        size_t section = rows.size();
        bool eligible = true;
        for (size_t i = 0; i + 1 < rows.size(); ++i) {
            const auto& row = rows[i];
            const auto& exponent = row.begin()->first[axis];
            if (std::any_of(row.begin(), row.end(), [&](const auto& term) {
                    return term.first[axis] != exponent;
                }) || (exponent != 0 && section != rows.size())) {
                eligible = false;
                break;
            }
            if (exponent != 0) section = i;
        }
        if (eligible && section != rows.size())
            return product_section_regular(rows, width, axis, section);
    }
    return false;
}

// Probe a small frame ideal independently of the size of the multiplier.
// In a complete primary decomposition I = intersection(Q_i), multiplication
// by f on R/I is injective iff f is outside every sqrt(Q_i). Embedded primes
// are essential: looking only at irreducible components of V(I) is unsound.
bool regular_on_components(Basis& rows, size_t width) {
    if (width > 5 || rows.size() > 9 || !cleared_height(rows.back())) return false;
    size_t terms = 0;
    for (auto it = rows.begin(); it != rows.end() - 1; ++it) {
        terms += it->size();
        if (terms > (width == 5 ? 64u : 48u) || local::max_total_degree(*it) > 12) return false;
        const auto bits = cleared_height(*it);
        if (!bits || *bits > 512) return false;
    }
    try {
        if (width == 5) return product_section(rows, width);
        Polynomial function = std::move(rows.back());
        rows.pop_back();
        primary::CertificateBudget budget(1u << 18);
        const auto components = primary::decompose(rows, width, true);
        if (!reconstructed(rows, components, width)) return false;
        size_t remaining = 1u << 19;
        for (const auto& component : components)
            if (!nonmember(function, component.prime, width, remaining)) return false;
        return true;
    } catch (const std::runtime_error&) {
        // An unavailable decomposition or basis leaves ordinary saturation in charge.
        return false;
    }
}
}

bool graph_regular(std::istream& input, size_t width, size_t equation_count) {
    if (!width || width > 32 || equation_count > 64) return false;
    Basis rows;
    size_t bytes = 0, terms = 0;
    for (size_t i = 0; i <= equation_count; ++i) {
        std::string line;
        if (!std::getline(input, line)) throw std::invalid_argument("Missing graph regularity row.");
        if (line.size() > byte_budget - bytes) return false;
        bytes += line.size();
        terms += 1 + std::count(line.begin(), line.end(), ';');
        if (terms > term_budget) return false;
        rows.push_back(local::decode_row(line, width));
    }
    for (;;) {
        rows.erase(std::remove_if(rows.begin(), rows.end() - 1,
                                  [](const Polynomial& f) { return f.empty(); }), rows.end() - 1);
        if (rows.back().empty() || std::any_of(rows.begin(), rows.end() - 1, scalar)) return false;
        if (rows.size() == 1) return true; // A nonzero polynomial in a polynomial ring.
        if (!eliminate_graph(rows, width)) break;
        --width;
    }
    // In the UFD QQ[x], (h):g = (h) iff gcd(h,g) is a unit. This also holds
    // for reducible h and repeated factors, with no radicality assumption.
    if (rows.size() != 2) return regular_on_components(rows, width);
    if (scalar(rows.back())) return true;
    if (!suitable_for_gcd(rows.front()) || !suitable_for_gcd(rows.back())) return false;
    auto gcd = local::polynomial_gcd(rows, width);
    return gcd && scalar(*gcd);
}
}
