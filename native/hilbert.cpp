// Hilbert series of monomial ideals, and the Groebner basis of a homogeneous
// ideal whose Hilbert series is known.
//
// For an order that does not compare degrees first (lexicographic,
// elimination) Buchberger's algorithm wastes most of its time on
// S-polynomials that reduce to zero. If the generators are homogenized, the
// Hilbert function of the ideal, read off a Groebner basis for the graded
// reverse lexicographic order, says how many leading monomials each degree
// must have; once they are there, every remaining pair of that degree is
// useless. The result is certified by its own Hilbert series (KERNEL.md,
// section 8), so the bookkeeping here needs no trust.
#include "groebner.hpp"

#include <flint/fmpz_poly.h>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>

namespace atlas {
namespace {

struct Series {
    fmpz_poly_t p;
    Series() { fmpz_poly_init(p); }
    Series(const Series&) = delete;
    Series& operator=(const Series&) = delete;
    ~Series() { fmpz_poly_clear(p); }
};

// Keep the monomials that no other one divides.
void minimalize(std::vector<uint16_t>& g, int m) {
    const size_t count = g.size() / m;
    std::vector<size_t> order(count);
    std::vector<unsigned> degree(count, 0);
    for (size_t i = 0; i < count; ++i) {
        order[i] = i;
        for (int v = 0; v < m; ++v) degree[i] += g[i * m + v];
    }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return degree[a] < degree[b]; });
    std::vector<uint16_t> kept;
    for (size_t i : order) {
        const uint16_t* mono = g.data() + i * m;
        bool needed = true;
        for (size_t k = 0; k < kept.size() && needed; k += m) {
            bool divides = true;
            for (int v = 0; v < m && divides; ++v) divides = kept[k + v] <= mono[v];
            needed = !divides;
        }
        if (needed) kept.insert(kept.end(), mono, mono + m);
    }
    g.swap(kept);
}

// N(M) = N(M + (x)) + t*N(M : x) for a variable x, until the generators are
// pairwise coprime, where N is the product of the 1 - t^deg.
void numerator(fmpz_poly_t out, std::vector<uint16_t>& g, int m, bool minimal = false) {
    if (!minimal) minimalize(g, m);
    const size_t count = g.size() / m;
    if (!count) {
        fmpz_poly_one(out);
        return;
    }
    std::vector<size_t> uses(m, 0);
    bool unit = false;
    for (size_t i = 0; i < count; ++i) {
        bool constant = true;
        for (int v = 0; v < m; ++v)
            if (g[i * m + v]) { ++uses[v]; constant = false; }
        unit = unit || constant;
    }
    if (unit) {
        fmpz_poly_zero(out);
        return;
    }
    const int pivot = static_cast<int>(std::max_element(uses.begin(), uses.end()) - uses.begin());
    if (uses[pivot] < 2) {
        fmpz_poly_one(out);
        Series factor;
        for (size_t i = 0; i < count; ++i) {
            unsigned degree = 0;
            for (int v = 0; v < m; ++v) degree += g[i * m + v];
            fmpz_poly_zero(factor.p);
            fmpz_poly_set_coeff_si(factor.p, 0, 1);
            fmpz_poly_set_coeff_si(factor.p, degree, -1);
            fmpz_poly_mul(out, out, factor.p);
        }
        return;
    }
    std::vector<uint16_t> sum, quotient;
    for (size_t i = 0; i < count; ++i) {
        const uint16_t* mono = g.data() + i * m;
        quotient.insert(quotient.end(), mono, mono + m);
        if (mono[pivot]) --quotient[quotient.size() - m + pivot];
        else sum.insert(sum.end(), mono, mono + m);
    }
    sum.resize(sum.size() + m, 0);
    sum[sum.size() - m + pivot] = 1;
    Series first, second;
    numerator(first.p, sum, m, true);  // minimal: the multiples of the variable are gone
    numerator(second.p, quotient, m);
    fmpz_poly_shift_left(second.p, second.p, 1);
    fmpz_poly_add(out, first.p, second.p);
}

bool equal(const std::vector<Z>& a, const std::vector<Z>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!fmpz_equal(&a[i].v, &b[i].v)) return false;
    return true;
}

}  // namespace

std::vector<Z> hilbert_numerator(std::vector<uint16_t> generators, int m) {
    Series series;
    numerator(series.p, generators, m);
    std::vector<Z> out(static_cast<size_t>(fmpz_poly_length(series.p)));
    for (size_t i = 0; i < out.size(); ++i) fmpz_poly_get_coeff_fmpz(&out[i].v, series.p, static_cast<slong>(i));
    return out;
}

// The coefficient of t^d in numerator/(1-t)^m.
void hilbert_value(fmpz* out, const std::vector<Z>& numerator, int m, long d) {
    Z binomial;
    fmpz_zero(out);
    for (size_t j = 0; j < numerator.size() && static_cast<long>(j) <= d; ++j) {
        fmpz_bin_uiui(&binomial.v, static_cast<ulong>(m - 1 + d - static_cast<long>(j)), static_cast<ulong>(m - 1));
        fmpz_addmul(out, &numerator[j].v, &binomial.v);
    }
}

std::vector<Z> Engine::leading_numerator() const {
    std::vector<uint16_t> heads;
    long e[MAX_VARIABLES];
    for (int g : active_) {
        R.unpack(lead(g), e);
        for (int v = 0; v < R.n; ++v) heads.push_back(static_cast<uint16_t>(e[v]));
    }
    return hilbert_numerator(std::move(heads), R.n);
}

bool Engine::compute_homogeneous(const std::vector<Poly>& generators, const std::vector<Z>& numerator,
                                 size_t bit_limit, double work_limit) {
    G.clear(); B.clear(); waiting_.clear();
    unit = false;
    work_ = 0;
    strategy_ = EAGER;
    bit_limit_ = bit_limit;
    work_limit_ = work_limit;
    rebuild_active();
    std::vector<Poly> input = generators;
    std::vector<size_t> order;
    for (size_t k = 0; k < input.size(); ++k)
        if (!input[k].empty()) order.push_back(k);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return input[a].m[0] < input[b].m[0]; });
    size_t next = 0;
    long degree = -1, missing = 0;  // leading monomials still to be found in this degree
    Z have, want;
    while (!unit) {
        const long of_input = next < order.size() ? static_cast<long>(input[order[next]].m[0]) : LONG_MAX;
        const long of_pair = B.empty() ? LONG_MAX : B.back().sugar;
        const long d = std::min(of_input, of_pair);
        if (d == LONG_MAX) break;
        if (d != degree) {
            degree = d;
            hilbert_value(&have.v, leading_numerator(), R.n, d);
            hilbert_value(&want.v, numerator, R.n, d);
            fmpz_sub(&have.v, &have.v, &want.v);
            if (fmpz_sgn(&have.v) < 0) return false;
            missing = fmpz_fits_si(&have.v) ? fmpz_get_si(&have.v) : LONG_MAX;
        }
        if (missing == 0) {
            // The leading monomials of this degree are complete: whatever is
            // left in it reduces to zero.
            while (next < order.size() && static_cast<long>(input[order[next]].m[0]) == d) ++next;
            while (!B.empty() && B.back().sugar == d) B.pop_back();
            continue;
        }
        Poly h;
        if (of_input <= of_pair) {
            h = std::move(input[order[next++]]);
        } else {
            const Pair pair = B.back();
            B.pop_back();
            h = spoly(pair);
            ++reductions;
        }
        long sugar = d;
        reduce(h, 0, nullptr, nullptr, &sugar);
        if (h.empty()) { ++zero_reductions; continue; }
        primitive(h, nullptr);
        guard(h);
        if (h.size() == 1 && h.m[0] == 0) { unit = true; break; }
        insert(std::move(h), d);
        if (missing != LONG_MAX) --missing;
    }
    finish();
    bit_limit_ = 0;
    work_limit_ = 0;
    return true;
}

bool hilbert_basis(const Ring& R, const std::vector<Poly>& input, Engine& result, size_t bit_limit, double work_limit) {
    if (R.graded || !(R.lex || R.eliminate) || R.n + 1 > MAX_VARIABLES) return false;
    Ring graded, target;
    graded.init(R.n + 1, 0);
    target.init(R.n + 1, R.lex ? 1 : 2, true);
    std::vector<Poly> first, second;
    for (const Poly& f : input) {
        if (f.empty()) continue;
        first.push_back(homogenize(R, graded, f));
        second.push_back(homogenize(R, target, f));
    }
    // The Hilbert series of the homogenized ideal, from its basis for the
    // graded reverse lexicographic order.
    Engine reference;
    reference.R = graded;
    reference.compute(first, 0);
    const std::vector<Z> series = reference.leading_numerator();
    Engine engine;
    engine.R = target;
    try {
        if (!engine.compute_homogeneous(second, series, bit_limit, work_limit)) return false;
    } catch (const Swell&) {
        return false;
    }
    // The certificate: the elements found lie in the ideal, being combinations
    // of the generators, and their leading monomials have its Hilbert series.
    if (!equal(engine.leading_numerator(), series)) return false;
    std::vector<Poly> affine;
    for (const Element& e : engine.G) affine.push_back(dehomogenize(target, R, e.p));
    // Every generator is declared part of a basis: no pair is formed, so this
    // only removes redundant elements and reduces the tails.
    result.R = R;
    result.compute(affine, affine.size());
    return true;
}

}  // namespace atlas

using namespace atlas;

extern "C" {

// The numerator of the Hilbert series of a monomial ideal, for tests. Input:
// a line with the number m of variables, then one monomial per line as m
// exponents. Returns its coefficients, lowest degree first, on one line; the
// caller frees it with atlas_groebner_release.
char* atlas_hilbert_numerator(const char* input, size_t length) noexcept {
    try {
        const std::string text(input, length);
        const char* at = text.c_str();
        char* stop = nullptr;
        const long m = std::strtol(at, &stop, 10);
        if (m < 1 || m > MAX_VARIABLES) throw std::invalid_argument("Bad number of variables.");
        std::vector<uint16_t> generators;
        for (at = stop;; at = stop) {
            const long e = std::strtol(at, &stop, 10);
            if (stop == at) break;
            if (e < 0 || e > static_cast<long>(MAX_DEGREE)) throw std::invalid_argument("Bad exponent.");
            generators.push_back(static_cast<uint16_t>(e));
        }
        if (generators.size() % m) throw std::invalid_argument("Incomplete monomial.");
        std::string out;
        for (const Z& c : hilbert_numerator(std::move(generators), static_cast<int>(m))) {
            char* digits = fmpz_get_str(nullptr, 10, &c.v);
            if (!out.empty()) out += ' ';
            out += digits;
            flint_free(digits);
        }
        out += '\n';
        return copy_text(out);
    } catch (const std::exception& error) {
        set_error(error.what());
    } catch (...) {
        set_error("Unknown failure.");
    }
    return nullptr;
}

// As atlas_groebner, by the Hilbert-driven route alone, without limits. Null
// when the order compares degrees first.
void* atlas_groebner_hilbert(const char* input, size_t length) noexcept {
    try {
        const char* at = input;
        const char* end = input + length;
        char* stop = nullptr;
        const long n = std::strtol(at, &stop, 10);
        const long order = std::strtol(stop, &stop, 10);
        std::strtol(stop, &stop, 10);
        at = stop;
        if (at >= end || *at != '\n' || order < 0 || order > 2) throw std::invalid_argument("Bad header.");
        ++at;
        auto engine = std::make_unique<Engine>();
        engine->R.init(static_cast<int>(n), static_cast<int>(order));
        const Ring R = engine->R;
        std::vector<Poly> polys;
        while (at < end) polys.push_back(parse(R, at, end, nullptr));
        const auto start = std::chrono::steady_clock::now();
        if (!hilbert_basis(R, polys, *engine, 0, 0)) throw std::runtime_error("The Hilbert-driven route does not apply.");
        engine->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        Z one;
        fmpz_one(&one.v);
        for (const Element& e : engine->G) write(engine->R, engine->text, e.p, &one.v, &e.p.c[0].v);
        return engine.release();
    } catch (const std::exception& error) {
        set_error(error.what());
    } catch (...) {
        set_error("Unknown failure.");
    }
    return nullptr;
}

}
