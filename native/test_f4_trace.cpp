// The F4 trace and its replays against the exact engine: the reduced basis
// over Q of the same homogeneous ideal, taken modulo each prime.
#include "f4_trace.hpp"

#include <flint/ulong_extras.h>

#include <iostream>
#include <map>
#include <string>

using namespace atlas;

static std::vector<Poly> ideal(const Ring& R, const Ring& H, const std::string& text) {
    std::vector<Poly> out;
    const char* at = text.data();
    const char* end = at + text.size();
    while (at < end) {
        const Poly f = parse(R, at, end, nullptr);
        if (!f.empty()) out.push_back(homogenize(R, H, f));
    }
    return out;
}

// The generators as they are, in R.
static std::vector<Poly> given(const Ring& R, const std::string& text) {
    std::vector<Poly> out;
    const char* at = text.data();
    const char* end = at + text.size();
    while (at < end) {
        Poly f = parse(R, at, end, nullptr);
        if (!f.empty()) out.push_back(std::move(f));
    }
    return out;
}

static std::string katsura(int n) {
    std::string text;
    for (int m = 0; m <= n; ++m) {
        std::map<std::vector<int>, long> terms;
        for (int i = -n; i <= n; ++i) {
            std::vector<int> e(n + 1, 0);
            if (m == n) {
                e[std::abs(i)] = 1;
            } else {
                if (std::abs(m - i) > n) continue;
                ++e[std::abs(i)];
                ++e[std::abs(m - i)];
            }
            ++terms[e];
        }
        std::vector<int> e(n + 1, 0);
        if (m < n) e[m] = 1;
        --terms[e];
        bool first = true;
        for (const auto& term : terms) {
            if (!term.second) continue;
            text += (first ? "" : ";") + std::to_string(term.second);
            for (int x : term.first) text += " " + std::to_string(x);
            first = false;
        }
        text += "\n";
    }
    return text;
}

// The residues of the exact basis modulo p on the monomials of the trace.
// False when the bases differ in size, leading monomials or supports.
static bool image(const Ring& H, const Engine& exact, const F4Trace& trace, uint32_t p, std::vector<uint32_t>& out) {
    const int S = H.stride;
    if (exact.G.size() != trace.output_lengths.size()) return false;
    out.clear();
    for (size_t k = 0; k < exact.G.size(); ++k) {
        const Poly& g = exact.G[k].p;
        const std::vector<u64>& m = trace.output_monomials[k];
        const u64 lead = fmpz_fdiv_ui(&g.c[0].v, p);
        if (!lead || !H.equal(g.m.data(), m.data())) return false;
        const u64 inverse = n_invmod(lead, p);
        size_t at = 0;
        for (size_t t = 0; t < g.size(); ++t) {
            const u64 residue = fmpz_fdiv_ui(&g.c[t].v, p) * inverse % p;
            if (at < trace.output_lengths[k] && H.equal(g.m.data() + t * S, m.data() + at * S)) {
                out.push_back(static_cast<uint32_t>(residue));
                ++at;
            } else if (residue) {
                return false;  // a term the trace does not have
            }
        }
        if (at != trace.output_lengths[k]) return false;
    }
    return true;
}

static bool agrees(const char* name, int variables, const std::string& text, bool bound, bool proved) {
    Ring R, H;
    R.init(variables, 0);
    H.init(variables + 1, 0, true);
    const std::vector<Poly> generators = ideal(R, H, text);
    Engine exact;
    exact.R = H;
    exact.compute(generators, 0);
    bool ok = true;
    for (unsigned threads : {1U, 4U}) {
        F4Trace trace;
        u64 p = (static_cast<u64>(1) << 31) - 1;
        do { --p; } while (!n_is_prime(p));
        std::vector<uint32_t> expected, residues;
        if (!trace.trace(H, generators, static_cast<uint32_t>(p), ~static_cast<size_t>(0), threads) ||
            !image(H, exact, trace, static_cast<uint32_t>(p), expected) || expected != trace.output_residues) {
            std::cerr << name << ": the trace differs from the exact basis\n";
            return false;
        }
        if (trace.hilbert != bound || (bound && trace.proved_degrees != proved)) {
            std::cerr << name << ": unexpected use of the Hilbert bound\n";
            ok = false;
        }
        for (int k = 0; k < 3; ++k) {
            do { --p; } while (!n_is_prime(p));
            if (!trace.replay(generators, static_cast<uint32_t>(p), residues, threads) ||
                !image(H, exact, trace, static_cast<uint32_t>(p), expected) || expected != residues) {
                std::cerr << name << ": a replay differs from the exact basis\n";
                return false;
            }
        }
    }
    return ok;
}

// The direct trace of generators that are not homogeneous, with its rows
// selected, and its replays (also of every element): the reduced basis of
// the ideal itself. The homogenized trace stops at its missed Hilbert bound;
// with `larger` its basis has more elements than the reduced one.
static bool direct_agrees(const char* name, int variables, const std::string& text, bool larger) {
    Ring R, H;
    R.init(variables, 0);
    H.init(variables + 1, 0, true);
    const std::vector<Poly> generators = given(R, text);
    Engine exact;
    exact.R = R;
    exact.compute(generators, 0);
    F4Trace::Options options;
    options.select_rows = true;
    for (unsigned threads : {1U, 4U}) {
        F4Trace trace;
        u64 p = (static_cast<u64>(1) << 31) - 1;
        do { --p; } while (!n_is_prime(p));
        std::vector<uint32_t> expected, residues, elements;
        if (!trace.trace(R, generators, static_cast<uint32_t>(p), ~static_cast<size_t>(0), threads, options) || trace.homogeneous ||
            !image(R, exact, trace, static_cast<uint32_t>(p), expected) || expected != trace.output_residues) {
            std::cerr << name << ": the direct trace differs from the exact basis\n";
            return false;
        }
        for (int k = 0; k < 3; ++k) {
            do { --p; } while (!n_is_prime(p));
            if (!trace.replay(generators, static_cast<uint32_t>(p), residues, threads, &elements) ||
                !image(R, exact, trace, static_cast<uint32_t>(p), expected) || expected != residues ||
                elements.size() != trace.element_residues().size()) {
                std::cerr << name << ": a direct replay differs from the exact basis\n";
                return false;
            }
        }
    }
    // The homogenized ideal needs more elements, and misses its Hilbert bound.
    F4Trace homogenized;
    F4Trace::Options stop;
    stop.stop_at_missed_bound = true;
    const std::vector<Poly> forms = ideal(R, H, text);
    if (homogenized.trace(H, forms, 2147483629U, ~static_cast<size_t>(0), 1, stop) || !homogenized.missed_bound) {
        std::cerr << name << ": the homogenized trace did not stop at a missed bound\n";
        return false;
    }
    F4Trace full;
    if (!full.trace(H, forms, 2147483629U, ~static_cast<size_t>(0), 1) || full.output_lengths.size() < exact.G.size() ||
        (larger && full.output_lengths.size() == exact.G.size())) {
        std::cerr << name << ": the homogenized basis has " << full.output_lengths.size() << " elements, the reduced basis "
                  << exact.G.size() << "\n";
        return false;
    }
    return true;
}

// The lift of a trace whose elements all have small integer coefficients:
// each residue is taken in (-p/2, p/2).
static std::vector<Poly> small_lift(const std::vector<const std::vector<u64>*>& monomials, const std::vector<uint32_t>& residues,
                                    int stride, uint32_t p) {
    std::vector<Poly> out;
    size_t i = 0;
    for (const std::vector<u64>* m : monomials) {
        Poly f;
        f.m = *m;
        f.c.resize(m->size() / stride);
        for (Z& c : f.c) {
            const uint32_t r = residues[i++];
            fmpz_set_si(&c.v, r > p / 2 ? static_cast<long>(r) - static_cast<long>(p) : static_cast<long>(r));
        }
        out.push_back(std::move(f));
    }
    return out;
}

// The exact certificate of a direct trace: its selected rows reduce to zero
// over Q with the true elements, and not with a changed one.
static bool rows_certify(const char* name, int variables, const std::string& text) {
    Ring R;
    R.init(variables, 0);
    const std::vector<Poly> generators = given(R, text);
    F4Trace trace;
    F4Trace::Options options;
    options.select_rows = true;
    const uint32_t p = 2147483629U;
    if (!trace.trace(R, generators, p, ~static_cast<size_t>(0), 2, options)) return false;
    std::vector<const std::vector<u64>*> terms, outputs;
    for (const Element& e : trace.G) terms.push_back(&e.p.m);
    for (const std::vector<u64>& m : trace.output_monomials) outputs.push_back(&m);
    std::vector<Poly> elements = small_lift(terms, trace.element_residues(), R.stride, p);
    std::vector<Poly> reduced = small_lift(outputs, trace.output_residues, R.stride, p);
    bool ok = true;
    if (!trace.rows_vanish(generators, elements, reduced, 2)) {
        std::cerr << name << ": the rows of the trace do not vanish over Q\n";
        ok = false;
    }
    // A trace without selected rows certifies nothing.
    F4Trace plain;
    if (!plain.trace(R, generators, p, ~static_cast<size_t>(0), 1) || plain.rows_vanish(generators, elements, reduced, 1)) {
        std::cerr << name << ": rows vanished for a trace that selected none\n";
        ok = false;
    }
    // Any change of a coefficient, of an element or of the reduced basis, is seen.
    for (size_t k = 0; k < elements.size(); ++k) {
        Z& c = elements[k].c.back();
        fmpz_add_ui(&c.v, &c.v, 1);
        if (trace.rows_vanish(generators, elements, reduced, 2)) {
            std::cerr << name << ": a wrong element " << k << " was accepted\n";
            ok = false;
        }
        fmpz_sub_ui(&c.v, &c.v, 1);
    }
    for (size_t k = 0; k < reduced.size(); ++k) {
        if (reduced[k].size() < 2) continue;
        Z& c = reduced[k].c.back();
        fmpz_add_ui(&c.v, &c.v, 1);
        if (trace.rows_vanish(generators, elements, reduced, 2)) {
            std::cerr << name << ": a wrong reduced element " << k << " was accepted\n";
            ok = false;
        }
        fmpz_sub_ui(&c.v, &c.v, 1);
    }
    return ok;
}

int main() {
    bool ok = true;
    // Complete intersections: every degree meets the Hilbert bound.
    ok &= agrees("Katsura-4", 5, katsura(4), true, true);
    ok &= agrees("Katsura-6", 7, katsura(6), true, true);
    ok &= agrees("two conics", 2, "1 2 0;1 0 2;-1 0 0\n1 1 1;-2 0 0\n", true, true);
    // At most as many generators as variables, yet no complete intersection
    // after homogenizing: the bound is not met and the elimination decides.
    ok &= agrees("cyclic-4", 4, "1 1 0 0 0;1 0 1 0 0;1 0 0 1 0;1 0 0 0 1\n"
                                "1 1 1 0 0;1 0 1 1 0;1 0 0 1 1;1 1 0 0 1\n"
                                "1 1 1 1 0;1 0 1 1 1;1 1 0 1 1;1 1 1 0 1\n1 1 1 1 1;-1 0 0 0 0\n", true, false);
    ok &= agrees("twisted cubic", 3, "1 1 0 1;-1 0 2 0\n1 0 1 0;-1 2 0 0\n1 0 0 1;-1 1 1 0\n", true, false);
    ok &= agrees("repeated generator", 2, "1 2 0;-3 0 1\n2 2 0;-6 0 1\n1 1 2;7 0 0\n", true, false);
    // More generators than variables: no bound.
    ok &= agrees("overdetermined", 2, "1 3 0;-2 1 1;5 0 0\n1 2 1;-1 0 2;1 1 0\n1 0 3;-1 1 0\n1 2 0;1 0 2;-4 0 0\n", false, false);
    // Many rows for few new elements: random combinations in several blocks.
    ok &= agrees("Katsura-7", 8, katsura(7), true, true);
    // Wide coefficients and exponents.
    ok &= agrees("wide coefficients", 2,
                 "1 2 0;1267650600228229401496703205377 0 1;-1 0 0\n1 1 1;-340282366920938463463374607431768211457 0 0\n",
                 true, true);
    ok &= agrees("wide exponents", 2, "1 40 0;-1 0 37;2 3 3\n1 5 30;-3 0 0\n", true, true);

    // A trace prime that loses rank: modulo 2147483629 the two generators
    // coincide. Every replay shows the larger rank and rejects the trace.
    {
        Ring R, H;
        R.init(2, 0);
        H.init(3, 0, true);
        const std::vector<Poly> generators = ideal(R, H, "1 2 0;1 0 2\n1 2 0;2147483630 0 2\n");
        F4Trace trace;
        std::vector<uint32_t> residues;
        if (!trace.trace(H, generators, 2147483629U, ~static_cast<size_t>(0), 1) || trace.output_lengths.size() != 1 ||
            trace.replay(generators, 2147483587U, residues, 1)) {
            std::cerr << "An unlucky trace prime was not detected by the replay\n";
            ok = false;
        }
        // A prime dividing a leading coefficient is refused.
        const std::vector<Poly> scaled = ideal(R, H, "2147483629 2 0;1 0 1\n1 1 1;-2 0 0\n");
        F4Trace other;
        if (other.trace(H, scaled, 2147483629U, ~static_cast<size_t>(0), 1)) {
            std::cerr << "A prime dividing a leading coefficient was accepted\n";
            ok = false;
        }
        // The budget bounds the recorded structure.
        F4Trace small;
        const std::vector<Poly> k6 = ideal(R, H, "1 2 0;1 0 2;-1 0 0\n1 1 1;-2 0 0\n");
        if (small.trace(H, k6, 2147483587U, 4, 1) || !small.too_large) {
            std::cerr << "The budget was not enforced\n";
            ok = false;
        }
    }
    // Generators that are not homogeneous and whose homogenization is no
    // complete intersection: the direct trace gives the reduced basis itself.
    const std::string cyclic5 = "1 1 0 0 0 0;1 0 1 0 0 0;1 0 0 1 0 0;1 0 0 0 1 0;1 0 0 0 0 1\n"
                                "1 1 1 0 0 0;1 0 1 1 0 0;1 0 0 1 1 0;1 0 0 0 1 1;1 1 0 0 0 1\n"
                                "1 1 1 1 0 0;1 0 1 1 1 0;1 0 0 1 1 1;1 1 0 0 1 1;1 1 1 0 0 1\n"
                                "1 1 1 1 1 0;1 0 1 1 1 1;1 1 0 1 1 1;1 1 1 0 1 1;1 1 1 1 0 1\n1 1 1 1 1 1;-1 0 0 0 0 0\n";
    const std::string cyclic4 = "1 1 0 0 0;1 0 1 0 0;1 0 0 1 0;1 0 0 0 1\n1 1 1 0 0;1 0 1 1 0;1 0 0 1 1;1 1 0 0 1\n"
                                "1 1 1 1 0;1 0 1 1 1;1 1 0 1 1;1 1 1 0 1\n1 1 1 1 1;-1 0 0 0 0\n";
    const std::string twisted = "1 1 0 1;-1 0 2 0\n1 0 1 0;-1 2 0 0\n1 0 0 1;-1 1 1 0\n";
    const std::string reimer4 = "2 2 0 0 0;-2 0 2 0 0;2 0 0 2 0;-2 0 0 0 2;-1 0 0 0 0\n2 3 0 0 0;-2 0 3 0 0;2 0 0 3 0;-2 0 0 0 3;-1 0 0 0 0\n"
                                "2 4 0 0 0;-2 0 4 0 0;2 0 0 4 0;-2 0 0 0 4;-1 0 0 0 0\n2 5 0 0 0;-2 0 5 0 0;2 0 0 5 0;-2 0 0 0 5;-1 0 0 0 0\n";
    ok &= direct_agrees("cyclic-4, direct", 4, cyclic4, false);
    ok &= direct_agrees("twisted cubic, direct", 3, twisted, false);
    ok &= direct_agrees("Reimer-4, direct", 4, reimer4, true);
    // Without stopping, cyclic-5 and Katsura are traced directly as well.
    {
        Ring R;
        R.init(5, 0);
        const std::vector<Poly> generators = given(R, cyclic5);
        Engine exact;
        exact.R = R;
        exact.compute(generators, 0);
        F4Trace trace;
        std::vector<uint32_t> expected;
        if (!trace.trace(R, generators, 2147483629U, ~static_cast<size_t>(0), 2) ||
            !image(R, exact, trace, 2147483629U, expected) || expected != trace.output_residues) {
            std::cerr << "cyclic-5: the direct trace differs from the exact basis\n";
            ok = false;
        }
    }
    ok &= rows_certify("twisted cubic", 3, twisted);
    ok &= rows_certify("cyclic-4", 4, cyclic4);
    ok &= rows_certify("binomials", 3, "1 2 1 0;-1 0 0 3\n1 1 2 0;-1 0 0 2\n1 3 0 0;-1 0 1 1\n1 0 3 0;-1 1 0 1\n");
    if (!ok) return 1;
    std::cout << "F4 trace and replay agree with the exact engine\n";
    return 0;
}
