// Exact reduced Groebner bases and normal forms over Q.
//
// Buchberger's algorithm with the Gebauer-Moeller installation of pairs and
// sugar selection. Polynomials are kept primitive over Z, so every step is
// exact and the reduced basis returned is the unique one for the order.
//
// A caller may declare that the first generators already form a Groebner
// basis for the same order. Their mutual S-polynomials are then not formed.
#include "groebner.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <iterator>
#include <type_traits>
#include <utility>

namespace atlas {

namespace {
// out = ca * x^shift_a * a[ia..] + cb * x^shift_b * b[ib..]; a null factor is one.
// With take, the coefficients of a are moved into out (scaled in place).
template<bool take>
void merge_terms(const Ring& R, Poly& out, std::conditional_t<take, Poly&, const Poly&> a, size_t ia,
                 const fmpz* ca, const u64* shift_a, const Poly& b, size_t ib, const fmpz* cb, const u64* shift_b) {
    const int S = R.stride;
    const size_t na = a.size(), nb = b.size();
    out.clear();
    out.c.reserve(na - std::min(ia, na) + nb - std::min(ib, nb));
    out.m.reserve((na - std::min(ia, na) + nb - std::min(ib, nb)) * S);
    u64 buffer_a[MAX_STRIDE], buffer_b[MAX_STRIDE];
    const u64 *ma = nullptr, *mb = nullptr;
    auto load_a = [&] {
        if (ia >= na) return;
        const u64* raw = a.m.data() + ia * S;
        if (shift_a) { R.multiply(buffer_a, raw, shift_a); ma = buffer_a; } else ma = raw;
    };
    auto load_b = [&] {
        if (ib >= nb) return;
        const u64* raw = b.m.data() + ib * S;
        if (shift_b) { R.multiply(buffer_b, raw, shift_b); mb = buffer_b; } else mb = raw;
    };
    auto emit = [&](const u64* mono, const fmpz* value, const fmpz* factor) {
        out.m.insert(out.m.end(), mono, mono + S);
        out.c.emplace_back();
        if (factor) fmpz_mul(&out.c.back().v, value, factor); else fmpz_set(&out.c.back().v, value);
    };
    // A term of a: taken from a when it is discarded, otherwise copied.
    auto emit_a = [&](const u64* mono) {
        if constexpr (take) {
            out.m.insert(out.m.end(), mono, mono + S);
            out.c.emplace_back();
            fmpz* value = &a.c[ia].v;
            if (ca) fmpz_mul(value, value, ca);
            fmpz_swap(&out.c.back().v, value);
        } else {
            emit(mono, &a.c[ia].v, ca);
        }
    };
    load_a(); load_b();
    if (!shift_a && na - std::min(ia, na) > 8 * (nb - std::min(ib, nb))) {
        // A short b against a long, unshifted a (a reduction step): find the
        // place of each term of b in a by galloping, and pass the run of a
        // before it on as one block. The terms and their order are those of
        // the term-by-term merge below.
        auto first_not_above = [&](const u64* mono) {
            size_t lo = ia, hi = na, probe = ia, step = 1;
            while (probe < na) {
                if (R.compare(a.m.data() + probe * S, mono) > 0) { lo = probe + 1; probe += step; step <<= 1; }
                else { hi = probe; break; }
            }
            while (lo < hi) {
                const size_t middle = lo + (hi - lo) / 2;
                if (R.compare(a.m.data() + middle * S, mono) > 0) lo = middle + 1; else hi = middle;
            }
            return lo;
        };
        auto run = [&](size_t end) {
            if (end <= ia) return;
            out.m.insert(out.m.end(), a.m.begin() + ia * S, a.m.begin() + end * S);
            if constexpr (take) {
                if (ca) for (size_t k = ia; k < end; ++k) fmpz_mul(&a.c[k].v, &a.c[k].v, ca);
                out.c.insert(out.c.end(), std::make_move_iterator(a.c.begin() + ia),
                             std::make_move_iterator(a.c.begin() + end));
            } else {
                for (size_t k = ia; k < end; ++k) {
                    out.c.emplace_back();
                    if (ca) fmpz_mul(&out.c.back().v, &a.c[k].v, ca); else fmpz_set(&out.c.back().v, &a.c[k].v);
                }
            }
            ia = end;
        };
        for (; ib < nb; ++ib, load_b()) {
            run(first_not_above(mb));
            if (ia < na && R.compare(a.m.data() + ia * S, mb) == 0) {
                emit_a(a.m.data() + ia * S);
                fmpz* sum = &out.c.back().v;
                if (cb) fmpz_addmul(sum, &b.c[ib].v, cb); else fmpz_add(sum, sum, &b.c[ib].v);
                if (fmpz_is_zero(sum)) { out.c.pop_back(); out.m.resize(out.m.size() - S); }
                ++ia;
            } else {
                emit(mb, &b.c[ib].v, cb);
            }
        }
        run(na);
        return;
    }
    while (ia < na && ib < nb) {
        const int order = R.compare(ma, mb);
        if (order > 0) { emit_a(ma); ++ia; load_a(); }
        else if (order < 0) { emit(mb, &b.c[ib].v, cb); ++ib; load_b(); }
        else {
            emit_a(ma);
            fmpz* sum = &out.c.back().v;
            if (cb) fmpz_addmul(sum, &b.c[ib].v, cb); else fmpz_add(sum, sum, &b.c[ib].v);
            if (fmpz_is_zero(sum)) { out.c.pop_back(); out.m.resize(out.m.size() - S); }
            ++ia; ++ib; load_a(); load_b();
        }
    }
    for (; ia < na; ++ia, load_a()) emit_a(ma);
    for (; ib < nb; ++ib, load_b()) emit(mb, &b.c[ib].v, cb);
}
} // namespace

void merge(const Ring& R, Poly& out, const Poly& a, size_t ia, const fmpz* ca, const u64* shift_a,
           const Poly& b, size_t ib, const fmpz* cb, const u64* shift_b) {
    merge_terms<false>(R, out, a, ia, ca, shift_a, b, ib, cb, shift_b);
}

void merge(const Ring& R, Poly& out, Poly&& a, size_t ia, const fmpz* ca, const u64* shift_a,
           const Poly& b, size_t ib, const fmpz* cb, const u64* shift_b) {
    merge_terms<true>(R, out, a, ia, ca, shift_a, b, ib, cb, shift_b);
}

void primitive(Poly& f, fmpz* num) {
    if (f.empty()) return;
    Z content;
    for (size_t i = 0; i < f.size() && !fmpz_is_one(&content.v); ++i)
        fmpz_gcd(&content.v, &content.v, &f.c[i].v);
    if (fmpz_sgn(&f.c[0].v) < 0) fmpz_neg(&content.v, &content.v);
    if (fmpz_is_one(&content.v)) return;
    for (size_t i = 0; i < f.size(); ++i) fmpz_divexact(&f.c[i].v, &f.c[i].v, &content.v);
    if (num) fmpz_mul(num, num, &content.v);
}

// The settled terms and the live suffix share one rational scale. Terms
// before 'at' have already been moved or reduced and no longer contribute.
static void remove_reduction_content(Poly& done, Poly& pending, size_t at, fmpz* num) {
    Z content;
    for (const auto& coefficient : done.c) {
        fmpz_gcd(&content.v, &content.v, &coefficient.v);
        if (fmpz_is_one(&content.v)) return;
    }
    for (size_t i = at; i < pending.size(); ++i) {
        fmpz_gcd(&content.v, &content.v, &pending.c[i].v);
        if (fmpz_is_one(&content.v)) return;
    }
    if (fmpz_is_zero(&content.v)) return;
    for (auto& coefficient : done.c)
        fmpz_divexact(&coefficient.v, &coefficient.v, &content.v);
    for (size_t i = at; i < pending.size(); ++i)
        fmpz_divexact(&pending.c[i].v, &pending.c[i].v, &content.v);
    if (num) fmpz_mul(num, num, &content.v);
}

template<bool membership>
bool Engine::reduce_terms(Poly& f, size_t keep, fmpz* num, fmpz* den, long* sugar, long bound) {
    const int S = R.stride;
    Poly done, next;
    size_t at = 0;  // the first term of f not yet examined
    auto settle = [&] {
        done.c.push_back(std::move(f.c[at]));
        done.m.insert(done.m.end(), f.m.begin() + at * S, f.m.begin() + (at + 1) * S);
        ++at;
    };
    while (at < keep && at < f.size()) settle();
    Z scale, negative, common;
    int scalings = 0;
    auto check_growth = [&] {
        guard_work();
        if (scalings < 8 && !over_bit_limit(f, at)) return;
        // A pseudo-division step can inflate all coefficients by the same
        // factor. Apply the bit limit to the head after removing that factor.
        remove_reduction_content(done, f, at, num);
        scalings = 0;
        if (over_bit_limit(f, at)) throw Swell();
    };
    while (at < f.size()) {
        const u64* mono = f.m.data() + at * S;
        const int g = reducer(mono, bound == LONG_MAX ? LONG_MAX : bound - static_cast<long>(mono[0]));
        if (g < 0) {
            // Later reductions produce only smaller monomials, so this term survives.
            if constexpr (membership) return false;
            settle();
            continue;
        }
        const Poly& r = G[g].p;
        if (sugar) *sugar = std::max(*sugar, static_cast<long>(mono[0] - r.m[0]) + G[g].sugar);
        if (r.size() == 1) {
            // A monomial divides this term exactly over Q and contributes no tail.
            ++at;
            if (work_limit_ > 0) ++work_;
            check_growth();
            continue;
        }
        u64 shift[MAX_STRIDE];
        R.divide(shift, mono, r.m.data());
        const fmpz* a = &f.c[at].v;
        const fmpz* b = &r.c[0].v;
        bool one;
        // f <- scale*f - (a/gcd)*x^shift*r, with scale = b/gcd made positive.
        if (fmpz_is_pm1(b)) {
            fmpz_mul(&negative.v, a, b);
            fmpz_neg(&negative.v, &negative.v);
            one = true;
        } else {
            fmpz_gcd(&common.v, a, b);
            fmpz_divexact(&scale.v, b, &common.v);
            fmpz_divexact(&negative.v, a, &common.v);
            if (fmpz_sgn(&scale.v) < 0) fmpz_neg(&scale.v, &scale.v); else fmpz_neg(&negative.v, &negative.v);
            one = fmpz_is_one(&scale.v);
        }
        merge(next, std::move(f), at + 1, one ? nullptr : &scale.v, nullptr, r, 1, &negative.v, shift);
        f.swap(next);
        at = 0;
        if (!one) {
            for (auto& coefficient : done.c)
                fmpz_mul(&coefficient.v, &coefficient.v, &scale.v);
            if (den) fmpz_mul(den, den, &scale.v);
            ++scalings;
        }
        check_growth();
    }
    f.swap(done);
    return f.empty();
}

void Engine::reduce(Poly& f, size_t keep, fmpz* num, fmpz* den, long* sugar, long bound) {
    reduce_terms<false>(f, keep, num, den, sugar, bound);
}

bool Engine::contains(Poly& f) {
    return reduce_terms<true>(f, 0, nullptr, nullptr, nullptr, LONG_MAX);
}

// Ordered division returns the same quotient list as multivariate division
// over K: all lifted K heads are theta-free and appear first, in the order
// of the reduced K-basis. Division by m last only changes coefficients.
std::string Engine::division(Poly& f, Rational& scale) {
    if (f.empty()) return "\n";  // division of the zero polynomial has no quotients
    std::vector<std::string> quotients(G.size());
    const int stride = R.stride;
    Poly done, next;
    size_t at = 0;
    while (at < f.size()) {
        const u64* monomial = f.m.data() + at*stride;
        size_t k = 0;
        while (k < G.size() && !R.divides(G[k].p.m.data(), monomial)) ++k;
        if (k == G.size()) {
            done.c.push_back(std::move(f.c[at]));
            done.m.insert(done.m.end(), monomial, monomial+stride);
            ++at;
            continue;
        }
        const Poly& g = G[k].p;
        Z coefficient;
        fmpz_mul(&coefficient.v, &scale.num.v, &f.c[at].v);
        Poly term;
        term.c.emplace_back();
        fmpz_one(&term.c[0].v);
        term.m.resize(stride);
        R.divide(term.m.data(), monomial, g.m.data());
        // G is output monic. The quotient coefficient on that generator is
        // the current leading coefficient, regardless of primitive scaling.
        std::string word;
        write(R, word, term, &coefficient.v, &scale.den.v);
        word.pop_back();
        if (!quotients[k].empty()) quotients[k] += ';';
        quotients[k] += word;
        Z common, left, right;
        fmpz_gcd(&common.v, &f.c[at].v, &g.c[0].v);
        fmpz_divexact(&left.v, &g.c[0].v, &common.v);
        fmpz_divexact(&right.v, &f.c[at].v, &common.v);
        fmpz_neg(&right.v, &right.v);
        merge(next, std::move(f), at+1, &left.v, nullptr, g, 1, &right.v, term.m.data());
        f.swap(next);
        // Multiplying the held remainder by left is offset in its denominator.
        if (!fmpz_is_one(&left.v)) {
            for (Z& c : done.c) fmpz_mul(&c.v, &c.v, &left.v);
            fmpz_mul(&scale.den.v, &scale.den.v, &left.v);
        }
        at = 0;
    }
    std::string result;
    // Quotients accumulate in Q[x,theta]. Reduce their coefficients modulo
    // m, which is the last basis row for every proper ideal.
    const bool proper = !G.empty() && G.back().p.m[0] > 0;
    Engine coefficients;
    if (proper) {
        coefficients.R = R;
        coefficients.compute({G.back().p}, 1);
    }
    for (size_t k = 0; k < G.size() - static_cast<size_t>(proper); ++k) {
        const std::string word = quotients[k] + "\n";
        if (!proper) { result += word; continue; }
        const char* cursor = word.data();
        Rational factor;
        Poly q = parse(R, cursor, cursor + word.size(), &factor);
        coefficients.reduce(q, 0, &factor.num.v, &factor.den.v);
        write(R, result, q, &factor.num.v, &factor.den.v);
    }
    write(R, result, done, &scale.num.v, &scale.den.v);
    return result;
}

Poly Engine::spoly(const Pair& pair) const {
    const Poly &f = G[pair.i].p, &g = G[pair.j].p;
    u64 shift_f[MAX_STRIDE], shift_g[MAX_STRIDE];
    R.divide(shift_f, pair.lcm, f.m.data());
    R.divide(shift_g, pair.lcm, g.m.data());
    Z common, left, right;
    fmpz_gcd(&common.v, &f.c[0].v, &g.c[0].v);
    fmpz_divexact(&left.v, &g.c[0].v, &common.v);
    fmpz_divexact(&right.v, &f.c[0].v, &common.v);
    fmpz_neg(&right.v, &right.v);
    Poly out;
    merge(out, f, 1, &left.v, shift_f, g, 1, &right.v, shift_g);
    return out;
}

// Add an element known to need no pairs with the ones already present.
void Engine::append(Poly&& h, long sugar) {
    if (strategy_ == NORMAL) sugar = 0;
    Element e;
    e.mask = R.mask(h.m.data());
    e.sugar = sugar;
    e.p = std::move(h);
    G.push_back(std::move(e));
}

// Gebauer-Moeller update (Becker-Weispfenning, Algorithm UPDATE).
void Engine::insert(Poly&& h, long sugar) {
    const int S = R.stride;
    const int t = static_cast<int>(G.size());
    if (strategy_ == NORMAL) sugar = 0;
    u64 head[MAX_STRIDE] = {0};
    std::copy(h.m.begin(), h.m.begin() + S, head);
    const u64 head_mask = R.mask(head);
    const long ecart = sugar - static_cast<long>(head[0]);

    // The pairs of the new element with every active one that still forms pairs.
    candidates_.clear();
    for (size_t a = 0; a < active_.size(); ++a)
        if (G[active_[a]].pairing) candidates_.push_back(a);
    const size_t count = candidates_.size();
    pair_lcm_.resize(count * S);
    pair_mask_.resize(count);
    pair_sugar_.resize(count);
    pair_key_.resize(count);
    for (size_t k = 0; k < count; ++k) {
        const size_t a = candidates_[k];
        const u64* other = active_lead_[a];
        u64* lcm = pair_lcm_.data() + k * S;
        R.lcm(lcm, head, other);
        pair_mask_[k] = head_mask | active_mask_[a];  // the mask of an lcm
        pair_sugar_[k] = std::max(sugar + static_cast<long>(lcm[0] - head[0]),
                                  G[active_[a]].sugar + static_cast<long>(lcm[0] - other[0]));
        // Small lcms first: they make other pairs superfluous. The key also
        // fixes which pair represents a set of pairs with one lcm.
        const u64 weight = static_cast<u64>(std::min<long>(std::max<long>(pair_sugar_[k], 0), 0xFFFFFF));
        pair_key_[k] = (lcm[0] << 48) | (weight << 24) | k;
    }
    std::sort(pair_key_.begin(), pair_key_.end());
    auto other_head = [&](size_t k) { return active_lead_[candidates_[k]]; };
    std::vector<Pair> added;
    for (size_t x = 0; x < count; ++x) {
        const size_t k = pair_key_[x] & 0xFFFFFF;
        const u64* lcm = pair_lcm_.data() + k * S;
        if (lcm[0] == head[0] + other_head(k)[0]) continue;  // coprime heads
        bool keep = true;
        for (size_t y = 0; y < count; ++y) {
            const size_t l = pair_key_[y] & 0xFFFFFF;
            const u64* other = pair_lcm_.data() + l * S;
            if (other[0] > lcm[0]) break;
            if (y == x || (pair_mask_[l] & ~pair_mask_[k]) || !R.divides(other, lcm)) continue;
            // A proper divisor, a coprime pair with the same lcm, or the
            // representative of this lcm (the earlier in the sort).
            if (other[0] != lcm[0] || other[0] == head[0] + other_head(l)[0] || y < x) { keep = false; break; }
        }
        if (!keep) continue;
        Pair pair;
        pair.i = active_[candidates_[k]];
        pair.j = t;
        pair.sugar = pair_sugar_[k];
        pair.mask = pair_mask_[k];
        std::copy(lcm, lcm + S, pair.lcm);
        added.push_back(pair);
    }
    // An old pair is superfluous when the new head divides its lcm and
    // neither of its pairs with the new element has that same lcm.
    u64 with[MAX_STRIDE];
    size_t kept = 0;
    for (size_t k = 0; k < B.size(); ++k) {
        const Pair& pair = B[k];
        bool drop = false;
        if (!(head_mask & ~pair.mask) && R.divides(head, pair.lcm)) {
            R.lcm(with, head, lead(pair.i));
            if (!R.equal(with, pair.lcm)) {
                R.lcm(with, head, lead(pair.j));
                drop = !R.equal(with, pair.lcm);
            }
        }
        if (!drop) {
            if (kept != k) B[kept] = pair;
            ++kept;
        }
    }
    B.resize(kept);
    // An element whose head the new one divides forms no further pairs. It
    // is no longer a reducer either, unless the strategy is lazy and its
    // sugar excess is smaller: it then still reduces what the new one may not.
    size_t live = 0;
    for (size_t a = 0; a < active_.size(); ++a) {
        if (!(head_mask & ~active_mask_[a]) && R.divides(head, active_lead_[a])) {
            G[active_[a]].pairing = false;
            if (strategy_ != LAZY || ecart <= active_ecart_[a]) {
                G[active_[a]].active = false;
                continue;
            }
        }
        active_[live] = active_[a];
        active_mask_[live] = active_mask_[a];
        active_lead_[live] = active_lead_[a];
        active_ecart_[live] = active_ecart_[a];
        ++live;
    }
    active_.resize(live);
    active_mask_.resize(live);
    active_lead_.resize(live);
    active_ecart_.resize(live);

    Element e;
    e.mask = head_mask;
    e.sugar = sugar;
    e.p = std::move(h);
    G.push_back(std::move(e));
    const size_t place = std::upper_bound(active_.begin(), active_.end(), t,
                                          [this](int x, int y) { return before(x, y); }) - active_.begin();
    active_.insert(active_.begin() + place, t);
    active_mask_.insert(active_mask_.begin() + place, head_mask);
    active_lead_.insert(active_lead_.begin() + place, lead(t));
    active_ecart_.insert(active_ecart_.begin() + place, ecart);

    // Next to treat is last: lowest sugar, then smallest lcm, then oldest.
    auto later = [this](const Pair& x, const Pair& y) {
        if (x.sugar != y.sugar) return x.sugar > y.sugar;
        const int order = R.compare(x.lcm, y.lcm);
        if (order) return order > 0;
        return x.i > y.i;
    };
    std::sort(added.begin(), added.end(), later);
    const size_t middle = B.size();
    B.insert(B.end(), added.begin(), added.end());
    std::inplace_merge(B.begin(), B.begin() + middle, B.end(), later);
}

void Engine::finish() {
    const int S = R.stride;
    std::vector<Poly> pending;
    if (unit) {
        Poly one;
        one.c.emplace_back();
        fmpz_one(&one.c.back().v);
        one.m.assign(S, 0);
        pending.push_back(std::move(one));
    } else {
        // Keep a minimal set of heads.
        std::vector<int> minimal;
        for (int a : active_) {
            bool needed = true;
            for (int b : active_)
                if (b != a && R.divides(lead(b), lead(a)) && (!R.equal(lead(b), lead(a)) || b < a)) {
                    needed = false;
                    break;
                }
            if (needed) minimal.push_back(a);
        }
        // Smallest head first. A term of a tail can only be reduced by an
        // element with a smaller head, and when that element's own tail is
        // already reduced, each reducible term costs one step.
        std::sort(minimal.begin(), minimal.end(), [this](int a, int b) { return R.compare(lead(a), lead(b)) < 0; });
        for (int a : minimal) pending.push_back(std::move(G[a].p));
    }
    G.clear();
    B.clear();
    strategy_ = EAGER;
    rebuild_active();
    for (Poly& work : pending) {
        reduce(work, 1, nullptr, nullptr);
        primitive(work, nullptr);
        Element e;
        e.mask = R.mask(work.m.data());
        e.sugar = degree(work);
        e.p = std::move(work);
        G.push_back(std::move(e));
        const int g = static_cast<int>(G.size()) - 1;
        active_.push_back(g);
        active_mask_.push_back(G[g].mask);
        active_lead_.push_back(lead(g));
        active_ecart_.push_back(G[g].sugar - static_cast<long>(lead(g)[0]));
    }
    // The reduced basis, largest head first.
    std::reverse(G.begin(), G.end());
    rebuild_active();
}

void Engine::compute(const std::vector<Poly>& input, size_t known) {
    // Degree orders rarely swell and the eager strategy is the quickest on
    // them; with a lexicographic part the lazy one is the safe start.
    static const int degree_plan[] = {EAGER, NORMAL, LAZY};
    static const int lex_plan[] = {LAZY, NORMAL, EAGER};
    const int* plan = (R.lex || R.eliminate) ? lex_plan : degree_plan;
#ifdef ATLAS_GROEBNER_TRACE
    static int forced[3];
    if (const char* text = std::getenv("ATLAS_GROEBNER_PLAN")) {
        for (int k = 0; k < 3; ++k) forced[k] = text[k] - '0';
        plan = forced;
    }
#endif
    for (int round = 0;; ++round) {
        check_deadline();
        std::vector<Poly> copy = input;
        G.clear(); B.clear(); waiting_.clear();
        unit = false;
        work_ = 0;
        const int cycle = round / 3;
        // No limit from the seventh cycle on, with the first strategy; and none
        // when everything is declared a basis, since no pair is then formed.
        const bool open = cycle >= 6 || known >= input.size();
        // Tighter first limits when degrees are not compared first: the
        // Hilbert route below then takes over early.
        const bool plain = !(R.lex || R.eliminate) || R.graded;
        strategy_ = open ? plan[0] : plan[round % 3];
        bit_limit_ = open ? 0 : static_cast<size_t>(plain ? 256 : 128) << (2 * cycle);
        work_limit_ = open ? 0 : (plain ? 3e5 : 1e5) * static_cast<double>(1ULL << (3 * cycle));
        try {
            attempt(copy, known);
            // Normal forms taken later are not subject to the limits.
            bit_limit_ = 0;
            work_limit_ = 0;
            return;
        } catch (const Swell&) {
            ++restarts;
        }
        // Two routes that do not depend on the strategy, each tried once per
        // cycle with limits that grow like those above. For an order that
        // does not compare degrees first: the homogenized ideal with its
        // Hilbert function, which makes every useless pair disappear. For any
        // order: modular images, which avoid swelling coefficients.
        // The homogenized auxiliary routes implement the QQ orders. The
        // coefficient-block order uses the exact Buchberger strategies.
        if (round % 3 == 0 && R.n < MAX_VARIABLES && !R.coefficient_last) {
            Engine other;
            const long before = restarts;
            if ((R.lex || R.eliminate) && !R.graded &&
                hilbert_basis(R, input, other, static_cast<size_t>(1) << (16 + 2 * cycle),
                              2e7 * static_cast<double>(1ULL << (3 * cycle)))) {
                *this = std::move(other);
                restarts = before;
                route = 2;
                return;
            }
            Engine images;
            if (modular_basis(R, input, images, nullptr, static_cast<size_t>(1) << (22 + 2 * std::min(cycle, 3)))) {
                *this = std::move(images);
                restarts = before;
                route = 1;
                return;
            }
        }
    }
}

bool Engine::is_basis(const std::vector<Poly>& candidate) {
    G.clear(); B.clear(); waiting_.clear();
    unit = false;
    work_ = 0;
    bit_limit_ = 0;
    work_limit_ = 0;
    strategy_ = EAGER;
    rebuild_active();
    std::vector<size_t> order;
    for (size_t k = 0; k < candidate.size(); ++k)
        if (!candidate[k].empty()) order.push_back(k);
    std::sort(order.begin(), order.end(), [&](size_t x, size_t y) {
        return R.compare(candidate[x].m.data(), candidate[y].m.data()) < 0;
    });
    for (size_t k : order) {
        Poly f = candidate[k];
        primitive(f, nullptr);
        if (f.size() == 1 && f.m[0] == 0) unit = true;
        insert(std::move(f), degree(candidate[k]));
    }
    // No element enters any more, so the pairs can be reduced independently
    // of one another. Without limits a reduction only reads the basis.
    std::atomic<size_t> cursor{0};
    std::atomic<bool> failed{false};
    // The workers stop once the caller's deadline passes; see check_deadline.
    check_deadline();
    const Deadline limit = thread_deadline();
    std::atomic<bool> late{false};
    auto worker = [&] {
        for (size_t k; !failed && (k = cursor.fetch_add(1)) < B.size();) {
            if (passed(limit)) {
                late = true;
                failed = true;
                break;
            }
            Poly h = spoly(B[k]);
            reduce(h, 0, nullptr, nullptr);
            if (!h.empty()) failed = true;
        }
    };
    if (!unit) {
        const size_t threads = B.size() < 64 ? 1 : std::min<size_t>(B.size() / 16, std::max(1u, std::thread::hardware_concurrency()));
        std::vector<std::thread> crew;
        for (size_t t = 1; t < threads; ++t) crew.emplace_back(worker);
        worker();
        for (std::thread& t : crew) t.join();
        if (late) expire();
        if (failed) return false;
        reductions += static_cast<long>(B.size());
        zero_reductions += static_cast<long>(B.size());
    }
    B.clear();
    finish();
    return true;
}

// Reduce the head of f for as long as it has a reducer. The reducer with the
// least sugar excess is used; when even that one raises the sugar of f while
// something of lower sugar is pending, f goes back to wait at its new sugar.
Engine::Outcome Engine::head(Poly& f, long& sugar) {
    Poly next;
    Z scale, negative, common;
    int scalings = 0;
    while (!f.empty()) {
        const u64* mono = f.m.data();
        const int g = reducer(mono);
        if (g < 0) return READY;
        const Poly& r = G[g].p;
        u64 shift[MAX_STRIDE];
        R.divide(shift, mono, r.m.data());
        const long need = static_cast<long>(shift[0]) + G[g].sugar;
        if (need > sugar) {
            if (pending_below(need)) {
                primitive(f, nullptr);
                waiting_.push_back(Delayed{std::move(f), need});
                return POSTPONED;
            }
            sugar = need;
        }
        const fmpz* a = &f.c[0].v;
        const fmpz* b = &r.c[0].v;
        fmpz_gcd(&common.v, a, b);
        fmpz_divexact(&scale.v, b, &common.v);
        fmpz_divexact(&negative.v, a, &common.v);
        if (fmpz_sgn(&scale.v) < 0) fmpz_neg(&scale.v, &scale.v); else fmpz_neg(&negative.v, &negative.v);
        const bool one = fmpz_is_one(&scale.v);
        merge(next, std::move(f), 1, one ? nullptr : &scale.v, nullptr, r, 1, &negative.v, shift);
        f.swap(next);
        guard_work();
        if ((!one && ++scalings >= 4) || over_bit_limit(f)) {
            primitive(f, nullptr);
            scalings = 0;
            if (over_bit_limit(f)) throw Swell();
        }
    }
    return ZERO;
}

void Engine::attempt(std::vector<Poly>& input, size_t known) {
    auto constant = [&](const Poly& f) { return f.size() == 1 && f.m[0] == 0; };
    std::vector<size_t> rest;
    for (size_t k = 0; k < input.size() && !unit; ++k) {
        if (input[k].empty()) continue;
        if (constant(input[k])) { unit = true; break; }
        if (k < known) append(std::move(input[k]), degree(input[k]));
        else rest.push_back(k);
    }
    rebuild_active();
    // Small heads first, so that they are available as reducers early.
    std::sort(rest.begin(), rest.end(), [&](size_t x, size_t y) {
        const int order = R.compare(input[x].m.data(), input[y].m.data());
        if (order) return order < 0;
        return input[x].size() != input[y].size() ? input[x].size() < input[y].size() : x < y;
    });
    for (size_t k : rest) {
        if (unit) break;
        check_deadline();
        Poly& f = input[k];
        long sugar = degree(f);
        if (strategy_ == LAZY) {  // the generators enter in the order of their degrees
            waiting_.push_back(Delayed{std::move(f), sugar});
            continue;
        }
        reduce(f, 0, nullptr, nullptr, &sugar);
        if (f.empty()) continue;
        primitive(f, nullptr);
        if (constant(f)) { unit = true; break; }
        insert(std::move(f), sugar);
    }
    while (!unit) {
        check_deadline();
        // Next: the lowest sugar, then the smallest lcm or head; a pair before a waiting polynomial.
        int chosen = -1;
        for (size_t k = 0; k < waiting_.size(); ++k) {
            if (chosen < 0) { chosen = static_cast<int>(k); continue; }
            const Delayed &x = waiting_[k], &y = waiting_[chosen];
            if (x.sugar < y.sugar || (x.sugar == y.sugar && R.compare(x.p.m.data(), y.p.m.data()) < 0))
                chosen = static_cast<int>(k);
        }
        if (chosen >= 0 && !B.empty()) {
            const Pair& pair = B.back();
            const Delayed& d = waiting_[chosen];
            if (pair.sugar < d.sugar || (pair.sugar == d.sugar && R.compare(pair.lcm, d.p.m.data()) <= 0)) chosen = -1;
        }
        Poly h;
        long sugar;
        if (chosen >= 0) {
            h = std::move(waiting_[chosen].p);
            sugar = waiting_[chosen].sugar;
            waiting_.erase(waiting_.begin() + chosen);
        } else if (!B.empty()) {
            const Pair pair = B.back();
            B.pop_back();
            h = spoly(pair);
            ++reductions;
            sugar = pair.sugar;
        } else {
            break;
        }
        if (strategy_ == LAZY) {
            const Outcome outcome = head(h, sugar);
            if (outcome == POSTPONED) continue;
            if (outcome == READY) reduce(h, 1, nullptr, nullptr, nullptr, sugar);
        } else {
            reduce(h, 0, nullptr, nullptr, &sugar);
        }
#ifdef ATLAS_GROEBNER_TRACE
        {
            size_t bits = 0;
            for (size_t i = 0; i < h.size(); ++i) bits = std::max<size_t>(bits, fmpz_bits(&h.c[i].v));
            std::fprintf(stderr, "%ld (strategy %d) sugar %ld: %zu terms, %zu bits; basis %zu, pairs %zu, waiting %zu\n",
                         reductions, strategy_, sugar, h.size(), bits, G.size(), B.size(), waiting_.size());
        }
#endif
        if (h.empty()) { ++zero_reductions; continue; }
        primitive(h, nullptr);
        guard(h);
        if (constant(h)) { unit = true; break; }
        insert(std::move(h), sugar);
    }
    finish();
}

Poly multiply(const Ring& R, const Poly& a, const Poly& b) {
    // Term by term of the shorter factor.
    const Poly& few = a.size() <= b.size() ? a : b;
    const Poly& many = a.size() <= b.size() ? b : a;
    Poly sum, merged;
    for (size_t i = 0; i < few.size(); ++i) {
        merge(R, merged, std::move(sum), 0, nullptr, nullptr, many, 0, &few.c[i].v, few.m.data() + i * R.stride);
        sum.swap(merged);
    }
    return sum;
}

// One factor of a polynomial line: its primitive part, and its content in scale.
static Poly factor(const Ring& R, const char*& at, const char* end, Rational* scale) {
    const int S = R.stride;
    std::vector<Rational> coefficients;
    std::vector<u64> monomials;
    long exponents[MAX_VARIABLES];
    std::string token;
    auto integer = [&](fmpz* out) {
        const char* start = at;
        if (at < end && (*at == '-' || *at == '+')) ++at;
        while (at < end && *at >= '0' && *at <= '9') ++at;
        if (at == start) throw std::invalid_argument("Expected an integer.");
        if (at - start <= 18) {
            fmpz_set_si(out, std::strtol(std::string(start, at).c_str(), nullptr, 10));
        } else {
            token.assign(start, at);
            if (fmpz_set_str(out, token.c_str(), 10)) throw std::invalid_argument("Bad integer.");
        }
    };
    while (at < end && *at != '\n' && *at != '*') {
        coefficients.emplace_back();
        Rational& q = coefficients.back();
        integer(&q.num.v);
        if (at < end && *at == '/') { ++at; integer(&q.den.v); } else fmpz_one(&q.den.v);
        if (fmpz_sgn(&q.den.v) <= 0) throw std::invalid_argument("Bad denominator.");
        for (int v = 0; v < R.n; ++v) {
            if (at >= end || *at != ' ') throw std::invalid_argument("Expected an exponent.");
            ++at;
            long e = 0;
            const char* start = at;
            while (at < end && *at >= '0' && *at <= '9') {
                e = e * 10 + (*at - '0');
                if (e > static_cast<long>(MAX_DEGREE)) throw Limit("Exponent out of range.");
                ++at;
            }
            if (at == start) throw std::invalid_argument("Expected an exponent.");
            exponents[v] = e;
        }
        monomials.resize(monomials.size() + S);
        R.pack(monomials.data() + monomials.size() - S, exponents);
        if (at < end && *at == ';') ++at;
        else if (at < end && *at != '\n' && *at != '*') throw std::invalid_argument("Expected a term separator.");
    }
    const size_t count = coefficients.size();
    std::vector<size_t> order(count);
    for (size_t i = 0; i < count; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t x, size_t y) {
        return R.compare(monomials.data() + x * S, monomials.data() + y * S) > 0;
    });
    Z denominator, factor;
    fmpz_one(&denominator.v);
    for (const Rational& q : coefficients) fmpz_lcm(&denominator.v, &denominator.v, &q.den.v);
    Poly f;
    for (size_t k = 0; k < count; ++k) {
        const size_t i = order[k];
        const u64* mono = monomials.data() + i * S;
        fmpz_divexact(&factor.v, &denominator.v, &coefficients[i].den.v);
        fmpz_mul(&factor.v, &factor.v, &coefficients[i].num.v);
        if (!f.empty() && R.equal(mono, f.m.data() + f.m.size() - S)) {
            fmpz_add(&f.c.back().v, &f.c.back().v, &factor.v);
        } else {
            if (!f.empty() && fmpz_is_zero(&f.c.back().v)) { f.c.pop_back(); f.m.resize(f.m.size() - S); }
            f.m.insert(f.m.end(), mono, mono + S);
            f.c.emplace_back();
            fmpz_set(&f.c.back().v, &factor.v);
        }
    }
    if (!f.empty() && fmpz_is_zero(&f.c.back().v)) { f.c.pop_back(); f.m.resize(f.m.size() - S); }
    Z content;
    fmpz_one(&content.v);
    primitive(f, &content.v);
    if (scale) {
        fmpz_set(&scale->num.v, &content.v);
        fmpz_set(&scale->den.v, &denominator.v);
    }
    return f;
}

Poly parse(const Ring& R, const char*& at, const char* end, Rational* scale) {
    Rational part;
    Poly product = factor(R, at, end, scale ? scale : &part);
    while (at < end && *at == '*') {
        ++at;
        Poly next = factor(R, at, end, &part);
        if (scale) {
            fmpz_mul(&scale->num.v, &scale->num.v, &part.num.v);
            fmpz_mul(&scale->den.v, &scale->den.v, &part.den.v);
        }
        product = multiply(R, product, next);
        Z content;
        fmpz_one(&content.v);
        primitive(product, &content.v);
        if (scale) fmpz_mul(&scale->num.v, &scale->num.v, &content.v);
    }
    if (at < end && *at == '\n') ++at;
    else if (at < end) throw std::invalid_argument("Expected the end of a polynomial.");
    return product;
}

void write(const Ring& R, std::string& out, const Poly& f, const fmpz* num, const fmpz* den) {
    const int S = R.stride;
    long exponents[MAX_VARIABLES];
    Z top, bottom, common;
    for (size_t i = 0; i < f.size(); ++i) {
        if (i) out += ';';
        fmpz_mul(&top.v, &f.c[i].v, num);
        fmpz_gcd(&common.v, &top.v, den);
        fmpz_divexact(&top.v, &top.v, &common.v);
        fmpz_divexact(&bottom.v, den, &common.v);
        if (fmpz_sgn(&bottom.v) < 0) { fmpz_neg(&top.v, &top.v); fmpz_neg(&bottom.v, &bottom.v); }
        char* digits = fmpz_get_str(nullptr, 10, &top.v);
        out += digits;
        flint_free(digits);
        if (!fmpz_is_one(&bottom.v)) {
            digits = fmpz_get_str(nullptr, 10, &bottom.v);
            out += '/';
            out += digits;
            flint_free(digits);
        }
        R.unpack(f.m.data() + i * S, exponents);
        for (int v = 0; v < R.n; ++v) { out += ' '; out += std::to_string(exponents[v]); }
    }
    out += '\n';
}

namespace {
thread_local std::string last_error;
}

void set_error(const char* message) { last_error = message; }

Deadline& thread_deadline() {
    static thread_local Deadline state;
    return state;
}

char* copy_text(const std::string& s) {
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (out) std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
}

}  // namespace atlas

using namespace atlas;

extern "C" {

int atlas_coefficient_block_abi() noexcept { return 1; }

char* atlas_field_division(void* basis, const char* input, size_t length) noexcept {
    try {
        Engine* engine = static_cast<Engine*>(basis);
        if (!engine->R.coefficient_last) throw std::invalid_argument("Expected a coefficient-block basis.");
        const char* end = input + length;
        Rational scale;
        Poly f = parse(engine->R, input, end, &scale);
        return copy_text(engine->division(f, scale));
    } catch (const std::exception& error) {
        set_error(error.what());
    } catch (...) {
        set_error("Unknown number-field division failure.");
    }
    return nullptr;
}

// Input: a header line "n order known" (order 0 grevlex, 1 lex, 2 the first
// variable before the others and grevlex after that; adding 3 places a
// coefficient-generator block after those variables. The first
// `known` generators are declared to be a Groebner basis), then one
// polynomial per line. Returns a basis handle, or null with an error set.
void* atlas_groebner(const char* input, size_t length) noexcept {
    try {
        const char* at = input;
        const char* end = input + length;
        char* stop = nullptr;
        const long n = std::strtol(at, &stop, 10);
        const long order = std::strtol(stop, &stop, 10);
        const long known = std::strtol(stop, &stop, 10);
        at = stop;
        if (at >= end || *at != '\n' || order < 0 || order > 5 || known < 0)
            throw std::invalid_argument("Bad header.");
        ++at;
        auto engine = std::make_unique<Engine>();
        engine->R.init(static_cast<int>(n), static_cast<int>(order));
        std::vector<Poly> polys;
        while (at < end) polys.push_back(parse(engine->R, at, end, nullptr));
        const auto start = std::chrono::steady_clock::now();
        engine->compute(polys, static_cast<size_t>(known));
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

// The reduced basis, monic, largest head first; one polynomial per line.
const char* atlas_groebner_text(const void* basis) noexcept {
    return static_cast<const Engine*>(basis)->text.c_str();
}

// Time spent computing, without parsing and printing; and work counters.
double atlas_groebner_seconds(const void* basis) noexcept {
    return static_cast<const Engine*>(basis)->seconds;
}
// How the basis was obtained: 0 by an exact strategy; when those swelled,
// 1 by modular images or 2 through the Hilbert function of the homogenized
// ideal. Both are certified.
int atlas_groebner_route(const void* basis) noexcept { return static_cast<const Engine*>(basis)->route; }
void atlas_groebner_counts(const void* basis, long* reductions, long* zero) noexcept {
    *reductions = static_cast<const Engine*>(basis)->reductions;
    *zero = static_cast<const Engine*>(basis)->zero_reductions;
}

// Exact normal forms of the given polynomials, one per line; zero is an
// empty line. The caller frees the result with atlas_groebner_release.
char* atlas_groebner_reduce(void* basis, const char* input, size_t length) noexcept {
    try {
        Engine* engine = static_cast<Engine*>(basis);
        const char* at = input;
        const char* end = input + length;
        std::string out;
        while (at < end) {
            Rational scale;
            Poly f = parse(engine->R, at, end, &scale);
            engine->reduce(f, 0, &scale.num.v, &scale.den.v);
            write(engine->R, out, f, &scale.num.v, &scale.den.v);
        }
        return copy_text(out);
    } catch (const std::exception& error) {
        set_error(error.what());
    } catch (...) {
        set_error("Unknown failure.");
    }
    return nullptr;
}

// One query in the basis ring: 1 for a member, 0 for a nonmember, -1 on error.
int atlas_groebner_contains(void* basis, const char* input, size_t length) noexcept {
    try {
        Engine* engine = static_cast<Engine*>(basis);
        const char* at = input;
        const char* end = input + length;
        Poly f = parse(engine->R, at, end, nullptr);
        if (at != end) throw std::invalid_argument("Expected one membership query.");
        return engine->contains(f);
    } catch (const std::exception& error) {
        set_error(error.what());
    } catch (...) {
        set_error("Unknown membership failure.");
    }
    return -1;
}

const char* atlas_groebner_error() noexcept { return last_error.c_str(); }
void atlas_groebner_release(char* text) noexcept { std::free(text); }
void atlas_groebner_free(void* basis) noexcept { delete static_cast<Engine*>(basis); }

}
