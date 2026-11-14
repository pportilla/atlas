// Ideals of a polynomial ring over Q, on top of the Groebner engine:
// saturation, ideal quotients, radical membership, the Jacobian test for the
// smooth locus, normal crossings, the affine charts of the blowup of an ideal
// and the verification of a candidate centre.
//
// Every routine follows engine/blowup_ideal.py step by step and returns the
// same generators, so the two implementations can be compared exactly.
#include "groebner.hpp"
#include "graph_centre_incidence.hpp"
#include "graph_regular.hpp"
#include "local_io.hpp"
#include "primary_decomposition.hpp"
#include "polynomial_graph.hpp"
#include <flint/fmpz_mat.h>
#include <flint/fmpz_poly.h>
#include <flint/fmpz_mpoly_factor.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <memory>
#include <optional>

namespace atlas {
namespace {

// The polynomial p*num/den. p is primitive with a positive leading
// coefficient in the grevlex order and num/den is in lowest terms, so equal
// polynomials have equal representations.
struct Gen {
    Poly p;
    Z num, den;
};
using Ideal = std::vector<Gen>;

bool same(const Poly& a, const Poly& b) {
    if (a.size() != b.size() || a.m != b.m) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!fmpz_equal(&a.c[i].v, &b.c[i].v)) return false;
    return true;
}

bool same(const Gen& a, const Gen& b) {
    return same(a.p, b.p) && fmpz_equal(&a.num.v, &b.num.v) && fmpz_equal(&a.den.v, &b.den.v);
}

bool constant(const Poly& f) { return f.size() == 1 && f.m[0] == 0; }

Poly one(const Ring& R) {
    Poly p;
    p.c.emplace_back();
    fmpz_one(&p.c.back().v);
    p.m.assign(R.stride, 0);
    return p;
}

// p*num/den in the normal representation; a missing factor is one.
Gen normal(Poly p, const fmpz* num = nullptr, const fmpz* den = nullptr) {
    Gen g;
    if (num) fmpz_set(&g.num.v, num); else fmpz_one(&g.num.v);
    if (den) fmpz_set(&g.den.v, den); else fmpz_one(&g.den.v);
    primitive(p, &g.num.v);
    Z common;
    fmpz_gcd(&common.v, &g.num.v, &g.den.v);
    if (!fmpz_is_one(&common.v) && !fmpz_is_zero(&common.v)) {
        fmpz_divexact(&g.num.v, &g.num.v, &common.v);
        fmpz_divexact(&g.den.v, &g.den.v, &common.v);
    }
    if (fmpz_sgn(&g.den.v) < 0) {
        fmpz_neg(&g.num.v, &g.num.v);
        fmpz_neg(&g.den.v, &g.den.v);
    }
    g.p = std::move(p);
    return g;
}

std::vector<Poly> polys(const Ideal& ideal) {
    std::vector<Poly> out;
    out.reserve(ideal.size());
    for (const Gen& g : ideal) out.push_back(g.p);
    return out;
}

// The nonzero generators, each kept once.
Ideal expressions(const Ideal& ideal) {
    Ideal out;
    for (const Gen& g : ideal) {
        if (g.p.empty()) continue;
        bool seen = false;
        for (const Gen& h : out)
            if (same(g, h)) { seen = true; break; }
        if (!seen) out.push_back(g);
    }
    return out;
}

// The image of f under x_v -> y_map[v]; a negative entry sets the variable
// to one. The result is sorted for the target ring but need not be primitive.
Poly convert(const Ring& from, const Ring& to, const Poly& f, const int* map) {
    const int S = from.stride, T = to.stride;
    const size_t count = f.size();
    std::vector<u64> monomials(count * T);
    long in[MAX_VARIABLES], out[MAX_VARIABLES];
    for (size_t i = 0; i < count; ++i) {
        from.unpack(f.m.data() + i * S, in);
        std::fill(out, out + to.n, 0);
        for (int v = 0; v < from.n; ++v)
            if (in[v] && map[v] >= 0) out[map[v]] += in[v];
        to.pack(monomials.data() + i * T, out);
    }
    std::vector<size_t> order(count);
    for (size_t i = 0; i < count; ++i) order[i] = i;
    auto larger = [&](size_t x, size_t y) {
        return to.compare(monomials.data() + x * T, monomials.data() + y * T) > 0;
    };
    if (!std::is_sorted(order.begin(), order.end(), larger)) std::sort(order.begin(), order.end(), larger);
    Poly g;
    auto drop_zero = [&] {
        if (!g.empty() && fmpz_is_zero(&g.c.back().v)) { g.c.pop_back(); g.m.resize(g.m.size() - T); }
    };
    for (size_t k = 0; k < count; ++k) {
        const size_t i = order[k];
        const u64* mono = monomials.data() + i * T;
        if (!g.empty() && to.equal(mono, g.m.data() + g.m.size() - T)) {
            fmpz_add(&g.c.back().v, &g.c.back().v, &f.c[i].v);
        } else {
            drop_zero();
            g.m.insert(g.m.end(), mono, mono + T);
            g.c.push_back(f.c[i]);
        }
    }
    drop_zero();
    return g;
}

Gen convert(const Ring& from, const Ring& to, const Gen& g, const int* map) {
    return normal(convert(from, to, g.p, map), &g.num.v, &g.den.v);
}

std::vector<int> shifted(int n, int by) {
    std::vector<int> map(n);
    for (int v = 0; v < n; ++v) map[v] = v + by;
    return map;
}

// The exponent vector of one variable.
void variable(const Ring& R, u64* m, int v) {
    long e[MAX_VARIABLES] = {0};
    e[v] = 1;
    R.pack(m, e);
}

Poly derivative(const Ring& R, const Poly& f, int v) {
    const int S = R.stride, p = R.position(v);
    const int word = 1 + (p >> 2), shift = 48 - 16 * (p & 3);
    Poly g;
    for (size_t i = 0; i < f.size(); ++i) {
        const u64* mono = f.m.data() + i * S;
        const u64 e = (mono[word] >> shift) & 0xFFFF;
        if (!e) continue;
        const size_t start = g.m.size();
        g.m.insert(g.m.end(), mono, mono + S);
        g.m[start] -= 1;
        g.m[start + word] -= 1ULL << shift;
        g.c.emplace_back();
        fmpz_mul_ui(&g.c.back().v, &f.c[i].v, e);
    }
    return g;
}

// A reduced grevlex basis together with its reducer.
struct Basis {
    Engine engine;
    Basis(const Ring& R, const std::vector<Poly>& input, size_t known = 0) {
        engine.R = R;
        engine.compute(input, known);
    }
    Basis(const Basis&) = delete;
    Basis& operator=(const Basis&) = delete;
    bool unit() const { return engine.unit; }
    bool contains(Poly f) {
        engine.reduce(f, 0, nullptr, nullptr);
        return f.empty();
    }
    // The normal form up to a rational factor.
    Poly remainder(Poly f) {
        engine.reduce(f, 0, nullptr, nullptr);
        primitive(f, nullptr);
        return f;
    }
    std::vector<Poly> elements() const {
        std::vector<Poly> out;
        out.reserve(engine.G.size());
        for (const Element& e : engine.G) out.push_back(e.p);
        return out;
    }
    Ideal monic() const {
        Ideal out;
        for (const Element& e : engine.G) out.push_back(normal(e.p, nullptr, &e.p.c[0].v));
        return out;
    }
};

Ideal unit_ideal(const Ring& R) {
    Ideal out;
    out.push_back(normal(one(R)));
    return out;
}

// Ideal quotients. With a new variable t placed first, an elimination basis
// of t*A + (1-t)*B gives A meet B, and (I meet (g))/g is I : g.
class Quotient {
public:
    explicit Quotient(const Ring& R) : base_(R) {
        wide_.init(R.n + 1, 2);
        // Both rings pack a variable at the same place; t takes the next field.
        t_word_ = 1 + (R.n >> 2);
        t_one_ = 1ULL << (48 - 16 * (R.n & 3));
    }

    // The intersection of two ideals.
    std::vector<Poly> intersection(const std::vector<Poly>& a, const std::vector<Poly>& b) {
        std::vector<const Poly*> pointers;
        for (const Poly& p : b) pointers.push_back(&p);
        return meet(a, pointers);
    }

    // I : (g_1, ..., g_r), as the meet of the I : g_j.
    std::vector<Poly> colon(const std::vector<Poly>& ideal, const std::vector<Poly>& by) {
        std::vector<Poly> result;
        bool whole = true;  // the unit ideal so far
        for (const Poly& g : by) {
            if (g.empty()) continue;
            std::vector<Poly> part;
            for (Poly& multiple : meet(ideal, {&g})) part.push_back(divide(std::move(multiple), g));
            if (unit(part)) continue;
            if (whole) { result = std::move(part); whole = false; continue; }
            std::vector<const Poly*> pointers;
            for (const Poly& p : part) pointers.push_back(&p);
            result = meet(result, pointers);
        }
        if (whole) {
            result.clear();
            result.push_back(one(base_));
        }
        return result;
    }

private:
    Ring base_, wide_;
    int t_word_ = 1;
    u64 t_one_ = 0;

    bool unit(const std::vector<Poly>& ideal) const {
        for (const Poly& f : ideal)
            if (constant(f)) return true;
        return false;
    }
    // Append sign*t^power*f to out, in the wide ring.
    void lift(Poly& out, const Poly& f, int power, int sign) const {
        const int S = base_.stride, W = wide_.stride;
        for (size_t i = 0; i < f.size(); ++i) {
            const size_t start = out.m.size();
            out.m.resize(start + W, 0);
            std::copy(f.m.begin() + i * S, f.m.begin() + (i + 1) * S, out.m.begin() + start);
            if (power) { out.m[start] += 1; out.m[start + t_word_] += t_one_; }
            out.c.push_back(f.c[i]);
            if (sign < 0) fmpz_neg(&out.c.back().v, &out.c.back().v);
        }
    }
    // Generators of (a) meet (b).
    std::vector<Poly> meet(const std::vector<Poly>& a, const std::vector<const Poly*>& b) {
        Engine engine;
        engine.R = wide_;
        std::vector<Poly> input;
        for (const Poly& f : a) {
            if (f.empty()) continue;
            input.emplace_back();
            lift(input.back(), f, 1, 1);
        }
        for (const Poly* g : b) {
            if (g->empty()) continue;
            input.emplace_back();
            lift(input.back(), *g, 1, -1);  // every term with t comes first
            lift(input.back(), *g, 0, 1);
        }
        engine.compute(input, 0);
        const int S = base_.stride, W = wide_.stride;
        std::vector<Poly> result;
        for (Element& e : engine.G) {
            if (Ring::field(e.p.m.data(), wide_.n - 1)) continue;  // the head has t
            Poly f;
            f.c = std::move(e.p.c);
            f.m.resize(f.c.size() * S);
            for (size_t i = 0; i < f.c.size(); ++i)
                std::copy(e.p.m.begin() + i * W, e.p.m.begin() + i * W + S, f.m.begin() + i * S);
            result.push_back(std::move(f));
        }
        return result;
    }
    // f / g up to a rational factor, for g dividing f.
    Poly divide(Poly f, const Poly& g) {
        const Ring& R = base_;
        const int S = R.stride;
        Poly quotient, next;
        Z scale, negative, common;
        u64 shift[MAX_STRIDE];
        while (!f.empty()) {
            if (!R.divides(g.m.data(), f.m.data())) throw std::logic_error("Inexact division in an ideal quotient.");
            R.divide(shift, f.m.data(), g.m.data());
            fmpz_gcd(&common.v, &f.c[0].v, &g.c[0].v);
            fmpz_divexact(&scale.v, &g.c[0].v, &common.v);
            fmpz_divexact(&negative.v, &f.c[0].v, &common.v);
            if (fmpz_sgn(&scale.v) < 0) fmpz_neg(&scale.v, &scale.v); else fmpz_neg(&negative.v, &negative.v);
            const bool one = fmpz_is_one(&scale.v);
            if (!one)
                for (size_t i = 0; i < quotient.size(); ++i) fmpz_mul(&quotient.c[i].v, &quotient.c[i].v, &scale.v);
            quotient.m.insert(quotient.m.end(), shift, shift + S);
            quotient.c.emplace_back();
            fmpz_neg(&quotient.c.back().v, &negative.v);
            merge(R, next, f, 1, one ? nullptr : &scale.v, nullptr, g, 1, &negative.v, shift);
            f.swap(next);
        }
        primitive(quotient, nullptr);
        return quotient;
    }
};

Ideal monic(const std::vector<Poly>& basis) {
    Ideal out;
    for (const Poly& p : basis) out.push_back(normal(p, nullptr, &p.c[0].v));
    return out;
}

// t*f - 1 in a ring whose first variable is t and whose other variables are
// those of R, for f primitive in R.
Poly inverse_relation(const Ring& R, const Ring& T, const Poly& f) {
    const std::vector<int> up = shifted(R.n, 1);
    Poly relation = convert(R, T, f, up.data());
    u64 t[MAX_STRIDE];
    variable(T, t, 0);
    for (size_t i = 0; i < relation.size(); ++i) {
        u64* mono = relation.m.data() + i * T.stride;
        T.multiply(mono, mono, t);  // keeps the order of the terms
    }
    relation.m.resize(relation.m.size() + T.stride, 0);
    relation.c.emplace_back();
    fmpz_set_si(&relation.c.back().v, -1);
    primitive(relation, nullptr);
    return relation;
}

// The reduced grevlex basis of I : by^infinity, for by not zero. With t
// compared first and grevlex after it, the elements without t of the basis
// of I + (t*by - 1) are that basis.
std::vector<Poly> saturation(const Ring& R, const std::vector<Poly>& generators, const Poly& by) {
    if (constant(by) || generators.empty()) return Basis(R, generators).elements();
    Ring E;
    E.init(R.n + 1, 2);
    const std::vector<int> up = shifted(R.n, 1);
    std::vector<Poly> input;
    for (const Poly& g : generators) input.push_back(convert(R, E, g, up.data()));
    input.push_back(inverse_relation(R, E, by));
    Engine engine;
    engine.R = E;
    engine.compute(input, 0);
    std::vector<int> down(R.n + 1);
    down[0] = -1;
    for (int v = 0; v < R.n; ++v) down[v + 1] = v;
    std::vector<Poly> out;
    for (const Element& e : engine.G) {
        if (Ring::field(e.p.m.data(), E.n - 1)) continue;  // the head has t
        out.push_back(convert(E, R, e.p, down.data()));
    }
    return out;
}

// The reduced lexicographic basis of the ideal, largest head first, written
// in the grevlex ring.
Ideal lexicographic(const Ring& R, const std::vector<Poly>& generators) {
    Ring L;
    L.init(R.n, 1);
    const std::vector<int> same_place = shifted(R.n, 0);
    std::vector<Poly> input;
    for (const Poly& g : generators) {
        input.push_back(convert(R, L, g, same_place.data()));
        primitive(input.back(), nullptr);
    }
    Engine engine;
    engine.R = L;
    engine.compute(input, 0);
    Ideal out;
    for (const Element& e : engine.G)
        out.push_back(normal(convert(L, R, e.p, same_place.data()), nullptr, &e.p.c[0].v));
    return out;
}

// Generators of I : by^infinity. With `lex` they are those of
// engine/blowup_ideal.saturate: the reduced lexicographic basis, or the
// grevlex one when by is constant or I has no generators. Without it, always
// the reduced grevlex basis, which costs less.
Ideal saturate(const Ring& R, const Ideal& ideal, const Gen& by, bool lex) {
    if (by.p.empty()) return unit_ideal(R);
    const std::vector<Poly> generators = polys(expressions(ideal));
    const std::vector<Poly> basis = saturation(R, generators, by.p);
    if (!lex || constant(by.p) || generators.empty()) return monic(basis);
    return lexicographic(R, basis);
}

// sqrt(smaller) inside sqrt(larger): each f of the smaller ideal lies in
// larger, or larger + (1 - t*f) is the unit ideal.
bool radical_subset(const Ring& R, const Ideal& smaller, const Ideal& larger) {
    const Ideal small = expressions(smaller);
    Basis basis(R, polys(expressions(larger)));
    std::vector<const Gen*> outside;
    for (const Gen& f : small)
        if (!basis.contains(f.p)) outside.push_back(&f);
    if (outside.empty()) return true;
    if (basis.unit()) return true;
    Ring T;
    T.init(R.n + 1, 0);
    const std::vector<int> up = shifted(R.n, 1);
    // A grevlex basis stays one when a variable is adjoined.
    std::vector<Poly> lifted;
    for (const Poly& g : basis.elements()) lifted.push_back(convert(R, T, g, up.data()));
    for (const Gen* f : outside) {
        std::vector<Poly> input = lifted;
        input.push_back(inverse_relation(R, T, f->p));
        Engine engine;
        engine.R = T;
        engine.compute(input, lifted.size());
        if (!engine.unit) return false;
    }
    return true;
}

// The reduced basis of (selected) : (equations). On the open set where one of
// these multipliers is a unit, the selected equations generate the ideal.
std::vector<Poly> multipliers(const Ring& R, const std::vector<Poly>& selected, const std::vector<Poly>& equations) {
    Basis sub(R, selected);
    std::vector<Poly> others;
    for (const Poly& g : equations)
        if (!sub.contains(g)) others.push_back(g);
    std::vector<Poly> unit;
    unit.push_back(one(R));
    if (others.empty()) return unit;
    Quotient quotient(R);
    return Basis(R, quotient.colon(selected, others)).elements();
}

u64 next_subset(u64 x) {  // the next word with as many bits set
    const u64 low = x & (~x + 1), ripple = x + low;
    return (((ripple ^ x) >> 2) / low) | ripple;
}

// A matrix of polynomials, row after row.
struct Matrix {
    size_t rows = 0;
    int columns = 0;
    std::vector<Poly> entries;
    Poly* row(size_t i) { return entries.data() + i * columns; }
    const Poly* row(size_t i) const { return entries.data() + i * columns; }
};

Matrix jacobian(const Ring& R, const std::vector<Poly>& equations) {
    Matrix J;
    J.rows = equations.size();
    J.columns = R.n;
    J.entries.resize(J.rows * J.columns);
    for (size_t i = 0; i < J.rows; ++i)
        for (int v = 0; v < R.n; ++v) J.row(i)[v] = derivative(R, equations[i], v);
    return J;
}

// The c-by-c minors of a matrix with at least c rows and c columns, for one
// set of c rows at a time, in the lexicographic order of the sets of rows.
class Minors {
public:
    Minors(const Ring& R, Matrix matrix, int c)
        : R_(R), n_(matrix.columns), c_(c), count_(matrix.rows), jacobian_(std::move(matrix.entries)), minors_(c), rows_(c) {
        fmpz_set_si(&minus_.v, -1);
        for (int d = 0; d < c_; ++d) rows_[d] = d;
        fill(0);
    }
    const std::vector<size_t>& rows() const { return rows_; }
    // The nonzero minors of the current rows, by set of columns.
    const std::map<u64, Poly>& current() const { return minors_[c_ - 1]; }
    bool next() {
        int d = c_ - 1;
        while (d >= 0 && rows_[d] == count_ - c_ + d) --d;
        if (d < 0) return false;
        ++rows_[d];
        for (int e = d + 1; e < c_; ++e) rows_[e] = rows_[e - 1] + 1;
        fill(d);
        return true;
    }

private:
    const Ring& R_;
    const int n_, c_;
    const size_t count_;
    std::vector<Poly> jacobian_;
    std::vector<std::map<u64, Poly>> minors_;  // minors_[d]: those of rows_[0..d]
    std::vector<size_t> rows_;
    Z minus_;

    // Recompute the minors from level `first` on, expanding along the last row.
    void fill(int first) {
        const u64 limit = 1ULL << n_;
        for (int d = first; d < c_; ++d) {
            minors_[d].clear();
            const Poly* row = jacobian_.data() + rows_[d] * n_;
            if (d == 0) {
                for (int v = 0; v < n_; ++v)
                    if (!row[v].empty()) minors_[0].emplace(1ULL << v, row[v]);
                continue;
            }
            if (minors_[d - 1].empty()) continue;
            for (u64 set = (1ULL << (d + 1)) - 1; set < limit; set = next_subset(set)) {
                Poly sum, merged;
                int index = 0;
                for (int v = 0; v < n_; ++v) {
                    if (!((set >> v) & 1)) continue;
                    const int place = index++;
                    if (row[v].empty()) continue;
                    const auto found = minors_[d - 1].find(set ^ (1ULL << v));
                    if (found == minors_[d - 1].end()) continue;
                    const Poly term = multiply(R_, row[v], found->second);
                    merge(R_, merged, sum, 0, nullptr, nullptr, term, 0, ((place + d) & 1) ? &minus_.v : nullptr, nullptr);
                    sum.swap(merged);
                }
                if (!sum.empty()) minors_[d].emplace(set, std::move(sum));
            }
        }
    }
};

struct Smooth {
    bool empty = false, smooth = false;
    Ideal bad;  // cuts out the complement of the smooth locus in V(I)
};

// The remainders of the current minors modulo the basis, without repeats.
std::vector<Poly> remainders(Basis& basis, const Minors& minors) {
    std::vector<Poly> rest;
    for (const auto& entry : minors.current()) {
        Poly remainder = basis.remainder(entry.second);
        if (remainder.empty()) continue;
        bool seen = false;
        for (const Poly& other : rest)
            if (same(other, remainder)) { seen = true; break; }
        if (!seen) rest.push_back(std::move(remainder));
    }
    return rest;
}

// Replace the entries of a row by their normal forms modulo the basis, the
// whole row multiplied by one nonzero rational number.
void reduce_row(Basis& basis, Poly* row, int columns) {
    std::vector<Z> num(columns), den(columns);
    Z common, lcm;
    fmpz_one(&lcm.v);
    for (int j = 0; j < columns; ++j) {
        if (row[j].empty()) continue;
        fmpz_one(&num[j].v);
        fmpz_one(&den[j].v);
        basis.engine.reduce(row[j], 0, &num[j].v, &den[j].v);
        if (row[j].empty()) continue;
        fmpz_gcd(&common.v, &num[j].v, &den[j].v);
        fmpz_divexact(&num[j].v, &num[j].v, &common.v);
        fmpz_divexact(&den[j].v, &den[j].v, &common.v);
        fmpz_lcm(&lcm.v, &lcm.v, &den[j].v);
    }
    Z content, factor;
    for (int j = 0; j < columns; ++j) {
        if (row[j].empty()) continue;
        fmpz_divexact(&factor.v, &lcm.v, &den[j].v);
        fmpz_mul(&factor.v, &factor.v, &num[j].v);
        for (size_t i = 0; i < row[j].size(); ++i) {
            fmpz_mul(&row[j].c[i].v, &row[j].c[i].v, &factor.v);
            fmpz_gcd(&content.v, &content.v, &row[j].c[i].v);
        }
    }
    if (fmpz_is_zero(&content.v) || fmpz_is_one(&content.v)) return;
    for (int j = 0; j < columns; ++j)
        for (size_t i = 0; i < row[j].size(); ++i) fmpz_divexact(&row[j].c[i].v, &row[j].c[i].v, &content.v);
}

// Whether the Jacobian matrix of the equations has rank at least c at every
// point of V(I), I being the ideal they generate, given by a Groebner basis:
// the c-by-c minors and I generate the unit ideal.
//
// The ideal generated by I and the minors of a given size does not change
// when an entry is replaced by another one congruent to it modulo I, when a
// row is multiplied by a nonzero number, or when a multiple of a row is added
// to another row. A matrix with a nonzero constant entry u can thus be
// brought to the form [u 0; 0 M'], and its c-by-c minors generate the same
// ideal as the minors of M' of size c - 1.
bool full_rank(const Ring& R, const std::vector<Poly>& generators, const std::vector<Poly>& equations, int c) {
    auto basis = std::make_unique<Basis>(R, generators, generators.size());
    if (basis->unit()) return true;
    Matrix M = jacobian(R, equations);
    std::vector<bool> row_alive(M.rows, true), column_alive(M.columns, true);
    for (size_t i = 0; i < M.rows; ++i) reduce_row(*basis, M.row(i), M.columns);
    Z minus;
    fmpz_set_si(&minus.v, -1);
    while (c > 0) {
        // A constant entry in the row with the fewest other entries.
        size_t pivot_row = M.rows;
        int pivot_column = -1;
        size_t fewest = ~static_cast<size_t>(0);
        for (size_t i = 0; i < M.rows; ++i) {
            if (!row_alive[i]) continue;
            int constant_at = -1;
            size_t filled = 0;
            for (int j = 0; j < M.columns; ++j) {
                if (!column_alive[j] || M.row(i)[j].empty()) continue;
                ++filled;
                if (constant_at < 0 && constant(M.row(i)[j])) constant_at = j;
            }
            if (constant_at >= 0 && filled < fewest) { fewest = filled; pivot_row = i; pivot_column = constant_at; }
        }
        if (pivot_column < 0) break;
        const Poly* pivot = M.row(pivot_row);
        const fmpz* u = &pivot[pivot_column].c[0].v;
        for (size_t k = 0; k < M.rows; ++k) {
            if (!row_alive[k] || k == pivot_row || M.row(k)[pivot_column].empty()) continue;
            Poly* row = M.row(k);
            const Poly factor = row[pivot_column];
            for (int j = 0; j < M.columns; ++j) {
                if (!column_alive[j] || j == pivot_column || pivot[j].empty()) continue;
                const Poly product = multiply(R, factor, pivot[j]);
                Poly merged;
                merge(R, merged, row[j], 0, u, nullptr, product, 0, &minus.v, nullptr);
                row[j].swap(merged);
            }
            // The other entries of the row are multiplied by u as well.
            for (int j = 0; j < M.columns; ++j) {
                if (!column_alive[j] || j == pivot_column || !pivot[j].empty()) continue;
                for (size_t i = 0; i < row[j].size(); ++i) fmpz_mul(&row[j].c[i].v, &row[j].c[i].v, u);
            }
            row[pivot_column].clear();
            reduce_row(*basis, row, M.columns);
        }
        row_alive[pivot_row] = false;
        column_alive[pivot_column] = false;
        --c;
    }
    if (c == 0) return true;
    // What is left, without the rows that vanish.
    Matrix rest;
    for (int j = 0; j < M.columns; ++j)
        if (column_alive[j]) ++rest.columns;
    for (size_t i = 0; i < M.rows; ++i) {
        if (!row_alive[i]) continue;
        bool zero = true;
        for (int j = 0; j < M.columns; ++j)
            if (column_alive[j] && !M.row(i)[j].empty()) zero = false;
        if (zero) continue;
        ++rest.rows;
        for (int j = 0; j < M.columns; ++j)
            if (column_alive[j]) rest.entries.push_back(std::move(M.row(i)[j]));
    }
    if (rest.rows < static_cast<size_t>(c) || rest.columns < c) return false;
    Minors minors(R, std::move(rest), c);
    do {
        const std::vector<Poly> fresh = remainders(*basis, minors);
        if (fresh.empty()) continue;
        std::vector<Poly> input = basis->elements();
        const size_t known = input.size();
        input.insert(input.end(), fresh.begin(), fresh.end());
        basis = std::make_unique<Basis>(R, input, known);
        if (basis->unit()) return true;
    } while (minors.next());
    return false;
}

// How much of the smoothness test is wanted.
//   IDEAL: the ideal of the complement of the smooth locus.
//   EXACT: only whether V(I) is smooth of the given dimension d.
//   RANK: the same, for a caller who knows that every irreducible component
//         of V(I) has dimension at least d. A point is then smooth of
//         dimension d as soon as its tangent space has dimension at most d,
//         that is, as soon as some c-by-c minor of the Jacobian is not zero.
enum class Test { IDEAL, EXACT, RANK };

// The closed complement of the locus where V(ideal) is smooth of the given
// dimension. For c = n - dimension and c of the equations, a c-by-c minor m
// of their Jacobian and a multiplier q in (those equations) : I certify the
// open set D(q*m).
Smooth smooth_locus(const Ring& R, const Ideal& ideal, long dimension, Test test) {
    const int n = R.n;
    const std::vector<Poly> raw = polys(expressions(ideal));
    auto bad = std::make_unique<Basis>(R, raw);
    const std::vector<Poly> generators = bad->elements();
    Smooth result;
    result.empty = bad->unit();
    const long codim = n - dimension;
    bool whole = false;  // the ideal is zero and the dimension is n
    if (!bad->unit() && codim == 0) whole = generators.empty();
    // A shorter presentation can turn the test into one determinant.
    const std::vector<Poly>& equations = raw.size() < generators.size() ? raw : generators;
    if (!bad->unit() && codim > 0 && codim <= n && equations.size() >= static_cast<size_t>(codim)) {
        const int c = static_cast<int>(codim);
        if (test != Test::IDEAL) {
            // Where every c-by-c minor of the whole Jacobian vanishes the
            // tangent space is too large, whatever the multipliers are. This
            // settles most failures without a single ideal quotient.
            const bool full = full_rank(R, generators, equations, c);
            if (!full) return result;
            if (test == Test::RANK) {
                result.smooth = true;
                return result;
            }
        }
        Minors minors(R, jacobian(R, equations), c);
        do {
            // Modulo the ideal found so far a minor and its remainder agree
            // on every point not yet covered.
            const std::vector<Poly> rest = remainders(*bad, minors);
            if (rest.empty()) continue;
            std::vector<Poly> selected;
            for (size_t r : minors.rows()) selected.push_back(equations[r]);
            const std::vector<Poly> factors = multipliers(R, selected, equations);
            std::vector<Poly> input = bad->elements();
            const size_t known = input.size();
            for (const Poly& q : factors)
                for (const Poly& m : rest) {
                    Poly product = multiply(R, q, m);
                    primitive(product, nullptr);
                    input.push_back(std::move(product));
                }
            bad = std::make_unique<Basis>(R, input, known);
        } while (!bad->unit() && minors.next());
    }
    result.smooth = whole || bad->unit();
    if (test == Test::IDEAL) result.bad = whole ? unit_ideal(R) : bad->monic();
    return result;
}

struct Stratum {
    std::vector<int> parts;  // indices of the divisors intersected
    Smooth witness;
};

// Smoothness of the ambient variety and of every intersection of the given
// divisors, in the expected dimension. Strata inside an empty one are skipped.
//
// With `pure` the answer wanted is only whether all of them are smooth, and
// the caller vouches that the ambient variety is irreducible of the given
// dimension. The search stops at the first failure and most strata need the
// rank test alone:
//   - the ambient variety, being irreducible of the right dimension;
//   - a divisor marked in `pure`: each of its components has codimension one;
//   - an intersection S of a stratum with a further divisor E. When this
//     point is reached without a failure, the ambient variety W is smooth,
//     the stratum is smooth of pure dimension D and E has pure codimension
//     one in W. So E is cut out by one equation near each point of W, and
//     every component of S has dimension at least D - 1.
struct Crossings {
    const Ring& R;
    const std::vector<Ideal>& divisors;
    long dimension;
    const std::vector<bool>* pure;  // null: visit every stratum and keep the ideals
    std::vector<Stratum> strata;
    bool ok = true;

    void descend(std::vector<int>& parts, const Ideal& equations, size_t start) {
        Test test = Test::IDEAL;
        if (pure) test = parts.size() == 1 && !(*pure)[parts[0]] ? Test::EXACT : Test::RANK;
        Stratum stratum;
        stratum.parts = parts;
        stratum.witness = smooth_locus(R, equations, dimension - static_cast<long>(parts.size()), test);
        const bool empty = stratum.witness.empty, smooth = stratum.witness.smooth;
        strata.push_back(std::move(stratum));
        if (!smooth) {
            ok = false;
            if (pure) return;
        }
        if (empty) return;
        for (size_t j = start; j < divisors.size(); ++j) {
            Ideal next = equations;
            next.insert(next.end(), divisors[j].begin(), divisors[j].end());
            parts.push_back(static_cast<int>(j));
            descend(parts, next, j + 1);
            parts.pop_back();
            if (!ok && pure) return;
        }
    }
    void run(const Ideal& ambient) {
        std::vector<int> parts;
        descend(parts, expressions(ambient), 0);
    }
};

// Whether ambient + piece = ambient + (h) for h the pivot or a generator of
// the piece. Inside an irreducible ambient variety every component of the
// piece then has codimension one. The ambient ideal is given by a basis.
bool principal(const Ring& R, const Ideal& ambient, const Gen& pivot, const Ideal& piece) {
    const std::vector<Poly> base = polys(ambient);
    std::vector<Poly> all = base;
    for (const Gen& g : piece) all.push_back(g.p);
    Basis whole(R, all, base.size());
    auto generates = [&](const Poly& h) {
        if (h.empty() || !whole.contains(h)) return false;
        std::vector<Poly> input = base;
        input.push_back(h);
        Basis one(R, input, base.size());
        for (const Gen& g : piece)
            if (!one.contains(g.p)) return false;
        return true;
    };
    if (generates(pivot.p)) return true;
    for (const Gen& g : piece)
        if (generates(g.p)) return true;
    return false;
}

// V(pivot) is the union of the pieces, inside V(ambient).
bool cover(const Ring& R, const Ideal& ambient, const Gen& pivot, const std::vector<Ideal>& pieces) {
    Ideal single;
    single.push_back(pivot);
    for (const Ideal& piece : pieces) {
        Ideal larger = ambient;
        larger.insert(larger.end(), piece.begin(), piece.end());
        if (!radical_subset(R, single, larger)) return false;
    }
    // Products of one generator from each piece cut out the union.
    Ideal products;
    std::vector<size_t> choice(pieces.size(), 0);
    bool any = true;
    for (const Ideal& piece : pieces)
        if (piece.empty()) any = false;
    while (any) {
        Poly product = one(R);
        for (size_t k = 0; k < pieces.size() && !product.empty(); ++k)
            product = multiply(R, product, pieces[k][choice[k]].p);
        products.push_back(normal(std::move(product)));
        size_t k = pieces.size();
        while (k > 0 && ++choice[k - 1] == pieces[k - 1].size()) choice[--k] = 0;
        if (k == 0) break;
    }
    Ideal larger = ambient;
    larger.push_back(pivot);
    return radical_subset(R, products, larger);
}

struct Chart {
    Ring ring;  // the coordinates of the base and then the ratios
    Gen pivot;
    Ideal ambient, strict;
    std::vector<Ideal> exceptional;
};

// The affine chart of the blowup of the centre where its i-th generator
// generates the pullback: the closure of the graph of the ratios f_j/f_i.
// The centre keeps its zero generators; they give empty charts. `lex` asks
// for the generators that engine/blowup_ideal.blowup_charts returns.
Chart blowup_chart(const Ring& R, const Ideal& ambient, const Ideal* strict, const Ideal& center,
                   const std::vector<Ideal>& exceptional, int i, bool lex) {
    const int n = R.n, k = static_cast<int>(center.size());
    Chart chart;
    chart.ring.init(n + k - 1, 0);
    const Ring& C = chart.ring;
    const std::vector<int> same_place = shifted(n, 0);
    auto up = [&](const Ideal& ideal) {
        Ideal out;
        for (const Gen& g : ideal) out.push_back(convert(R, C, g, same_place.data()));
        return out;
    };
    chart.pivot = convert(R, C, center[i], same_place.data());
    Ideal graph = up(expressions(ambient));
    Z left, right;
    for (int j = 0; j < k; ++j) {
        if (j == i) continue;
        // f_j - f_i*ratio, cleared of denominators.
        const Gen f = convert(R, C, center[j], same_place.data());
        u64 ratio[MAX_STRIDE];
        variable(C, ratio, n + (j < i ? j : j - 1));
        fmpz_mul(&left.v, &f.num.v, &chart.pivot.den.v);
        fmpz_mul(&right.v, &chart.pivot.num.v, &f.den.v);
        fmpz_neg(&right.v, &right.v);
        Poly relation;
        merge(C, relation, f.p, 0, &left.v, nullptr, chart.pivot.p, 0, &right.v, ratio);
        graph.push_back(normal(std::move(relation)));
    }
    chart.ambient = saturate(C, graph, chart.pivot, lex);
    auto transform = [&](const Ideal& ideal) {
        Ideal total = chart.ambient;
        const Ideal lifted = up(expressions(ideal));
        total.insert(total.end(), lifted.begin(), lifted.end());
        return saturate(C, total, chart.pivot, lex);
    };
    if (strict) chart.strict = transform(*strict);
    for (const Ideal& ideal : exceptional) chart.exceptional.push_back(transform(ideal));
    return chart;
}

// ---- The text interface ----

struct Reader {
    const char* at;
    const char* end;
    std::vector<long> numbers;
    std::string op;

    Reader(const char* input, size_t length) : at(input), end(input + length) {
        const char* stop = static_cast<const char*>(std::memchr(at, '\n', length));
        if (!stop) throw std::invalid_argument("Missing header.");
        const char* space = std::find(at, stop, ' ');
        op.assign(at, space);
        const std::string rest(space, stop);
        const char* cursor = rest.c_str();
        for (;;) {
            char* next = nullptr;
            const long value = std::strtol(cursor, &next, 10);
            if (next == cursor) break;
            numbers.push_back(value);
            cursor = next;
        }
        if (*cursor) throw std::invalid_argument("Bad header.");
        at = stop + 1;
    }
    long number(size_t k) const {
        if (k >= numbers.size()) throw std::invalid_argument("Bad header.");
        return numbers[k];
    }
    size_t count(size_t k) const {
        const long value = number(k);
        if (value < 0) throw std::invalid_argument("Bad header.");
        return static_cast<size_t>(value);
    }
    std::string_view row_text() {
        if (at == end) throw std::invalid_argument("Missing polynomial.");
        const char* stop = std::find(at, end, '\n');
        const std::string_view row(at, stop - at);
        at = stop == end ? stop : stop + 1;
        return row;
    }
    Gen gen(const Ring& R) {
        if (at >= end) throw std::invalid_argument("Missing polynomial.");
        Rational scale;
        Poly p = parse(R, at, end, &scale);
        return normal(std::move(p), &scale.num.v, &scale.den.v);
    }
    Ideal ideal(const Ring& R, size_t size) {
        Ideal out;
        for (size_t k = 0; k < size; ++k) out.push_back(gen(R));
        return out;
    }
    Ideal rest(const Ring& R) {
        Ideal out;
        while (at < end) out.push_back(gen(R));
        return out;
    }
};

void put(const Ring& R, std::string& out, const Ideal& ideal) {
    for (const Gen& g : ideal) write(R, out, g.p, &g.num.v, &g.den.v);
}

Ring grevlex(long n) {
    if (n < 1 || n > MAX_VARIABLES) throw Limit("Unsupported number of variables.");
    Ring R;
    R.init(static_cast<int>(n), 0);
    return R;
}

struct CoordinatePolynomial {
    fmpz_poly_t p;
    CoordinatePolynomial() { fmpz_poly_init(p); }
    ~CoordinatePolynomial() { fmpz_poly_clear(p); }
    CoordinatePolynomial(const CoordinatePolynomial&) = delete;
};

// A proper monomial ideal is finite exactly when it contains a pure power
// of each variable. Its radical then has the coordinates as its reduced basis.
std::string finite_monomial_radical(const Ring& R, const std::vector<Poly>& ideal) {
    u64 pure_powers = 0;
    for (const Poly& f : ideal) {
        if (f.empty()) continue;
        if (constant(f)) {
            std::string out;
            put(R, out, unit_ideal(R));
            return out;
        }
        const u64* m = f.m.data();
        for (int v = 0; v < R.n; ++v) {
            if (Ring::field(m, R.position(v)) == m[0]) {
                pure_powers |= u64(1) << v;
                break;
            }
        }
    }
    if (pure_powers != (u64(1) << R.n) - 1) return "positive_dimension\n";

    std::string out;
    Gen coordinate = normal(one(R));
    for (int v = 0; v < R.n; ++v) {
        variable(R, coordinate.p.m.data(), v);
        write(R, out, coordinate.p, &coordinate.num.v, &coordinate.den.v);
    }
    return out;
}

// Seidenberg's criterion over Q: adjoin the squarefree parts of the
// coordinate eliminants, then compute the reduced basis in the original ring.
std::string op_finite_radical(Reader& in) {
    const Ring R = grevlex(in.number(0));
    std::vector<Poly> input;
    while (in.at < in.end) input.push_back(parse(R, in.at, in.end, nullptr));
    if (std::all_of(input.begin(), input.end(), [](const Poly& f) { return f.size() <= 1; }))
        return finite_monomial_radical(R, input);
    std::vector<Poly> generators = input;
    Ring lex;
    lex.init(R.n, 1);
    for (int v = 0; v < R.n; ++v) {
        int permutation[MAX_VARIABLES];
        for (int j = 0; j < R.n; ++j) permutation[j] = j == v ? R.n-1 : j - (j > v);
        std::vector<Poly> reordered;
        for (const Poly& f : input) reordered.push_back(convert(R, lex, f, permutation));
        Basis basis(lex, reordered);
        if (basis.unit()) {
            std::string out;
            put(R, out, unit_ideal(R));
            return out;
        }
        // In lex order with v last, a pure v leading power implies every term is in Q[v].
        const Poly* eliminant = nullptr;
        for (const auto& e : basis.engine.G) {
            if (e.p.m[0] == Ring::field(e.p.m.data(), lex.position(R.n-1))) {
                eliminant = &e.p;
                break;
            }
        }
        if (!eliminant) return "positive_dimension\n";
        CoordinatePolynomial p, derivative, divisor, squarefree;
        for (size_t i = 0; i < eliminant->size(); ++i)
            fmpz_poly_set_coeff_fmpz(p.p, eliminant->m[i*lex.stride], &eliminant->c[i].v);
        fmpz_poly_derivative(derivative.p, p.p);
        fmpz_poly_gcd(divisor.p, p.p, derivative.p);
        if (fmpz_poly_degree(divisor.p) == 0) continue;
        if (!fmpz_poly_divides(squarefree.p, p.p, divisor.p))
            throw std::logic_error("Inexact squarefree quotient.");
        Poly relation;
        long powers[MAX_VARIABLES] = {};
        for (long degree = fmpz_poly_degree(squarefree.p); degree >= 0; --degree) {
            const fmpz* coefficient = fmpz_poly_get_coeff_ptr(squarefree.p, degree);
            if (fmpz_is_zero(coefficient)) continue;
            powers[v] = degree;
            relation.m.resize(relation.m.size() + R.stride);
            R.pack(relation.m.data() + relation.m.size() - R.stride, powers);
            relation.c.emplace_back();
            fmpz_set(&relation.c.back().v, coefficient);
        }
        generators.push_back(std::move(relation));
    }
    std::string out;
    put(R, out, Basis(R, generators).monic());
    return out;
}

// A prime over the leading monomial ideal must contain at least one
// variable from each leading support. Its least possible height is the
// minimum hitting set size, attained by the corresponding coordinate prime.
// Branch on a smallest support not hit yet. Disjoint supports give a lower
// bound for the remaining height; this avoids enumerating every coordinate
// subset even for coordinate flats in large ambient spaces.
int minimum_height(const std::vector<u64>& supports, u64 selected, int height, int best) {
    u64 branch = 0, disjoint = 0;
    int bound = height;
    for (auto support : supports) {
        if (support & selected) continue;
        if (!branch) branch = support;
        if (!(disjoint & support)) {
            disjoint |= support;
            if (++bound >= best) return best;
        }
    }
    if (!branch) return height;
    while (branch) {
        const u64 bit = branch & (0 - branch);
        best = minimum_height(supports, selected | bit, height + 1, best);
        if (best == bound) break;
        branch ^= bit;
    }
    return best;
}

int dimension_of_basis(const Ring& R, const Basis& basis) {
    if (basis.unit()) return -1;
    std::vector<u64> supports;
    for (const auto& e : basis.engine.G) {
        u64 support = 0;
        for (int j = 0; j < R.n; ++j)
            if (R.field(e.p.m.data(), R.position(j))) support |= u64(1) << j;
        supports.push_back(support);
    }
    std::sort(supports.begin(), supports.end(), [](u64 a, u64 b) {
        const auto count_a = __builtin_popcountll(a), count_b = __builtin_popcountll(b);
        return count_a == count_b ? a < b : count_a < count_b;
    });
    supports.erase(std::unique(supports.begin(), supports.end()), supports.end());
    return R.n - minimum_height(supports, 0, 0, R.n);
}

std::string op_dimension(Reader& in) {
    const Ring R = grevlex(in.number(0));
    return std::to_string(dimension_of_basis(R, Basis(R, polys(in.rest(R))))) + "\n";
}

// Does the residual ideal cut the exceptional fibre to dimension below k?
// If V(base) = V(J) x A^d, every irreducible component is a cylinder.
// A unit coefficient in the free variables makes the residual nonzero
// generically on every component. Otherwise decide by the full dimension.
bool cuts_below_dimension(const Ring& R, Basis& base, int dim, int threshold, const Ideal& residual) {
    if (dim < threshold) return true;
    std::vector<bool> free(R.n, true);
    for (const auto& e : base.engine.G)
        for (size_t i = 0; i < e.p.size(); ++i)
            for (int v = 0; v < R.n; ++v)
                if (Ring::field(e.p.m.data() + i*R.stride, R.position(v))) free[v] = false;
    std::vector<Poly> remaining;
    for (const Gen& g : residual) {
        Poly f = base.remainder(g.p);
        if (f.empty()) continue;
        if (constant(f)) return true;
        if (dim == threshold && std::find(free.begin(), free.end(), true) != free.end()) {
            // A group containing only a pure free-variable term has a
            // nonzero constant coefficient (the grevlex normal form is
            // already reduced modulo J). Other groups need no factorization.
            std::map<std::vector<long>, bool> constant_coefficient;
            long exponent[MAX_VARIABLES];
            for (size_t i = 0; i < f.size(); ++i) {
                R.unpack(f.m.data() + i*R.stride, exponent);
                bool scalar = true;
                std::vector<long> key;
                for (int v = 0; v < R.n; ++v) {
                    if (free[v]) key.push_back(exponent[v]);
                    else if (exponent[v]) scalar = false;
                }
                const auto inserted = constant_coefficient.emplace(std::move(key), scalar);
                if (!inserted.second) inserted.first->second = false;
            }
            for (const auto& entry : constant_coefficient)
                if (entry.second) return true;
        }
        remaining.push_back(std::move(f));
    }
    if (remaining.empty()) return false;
    auto all = base.elements();
    const size_t known = all.size();
    all.insert(all.end(), remaining.begin(), remaining.end());
    return dimension_of_basis(R, Basis(R, all, known)) < threshold;
}

std::string op_residual_dimension_below(Reader& in) {
    const Ring R = grevlex(in.number(0));
    const int threshold = in.number(1);
    Basis base(R, polys(in.ideal(R, in.count(2))));
    const int dim = dimension_of_basis(R, base);
    if (in.op == "residual_dimension_below")
        return cuts_below_dimension(R, base, dim, threshold, in.rest(R)) ? "1\n" : "0\n";
    // A nonempty exceptional divisor in a regular blowup chart is pure of
    // the expected dimension. A quotient of a known pivot power has exact
    // order on every component precisely when its zero locus cuts below it.
    std::string result;
    for (const Gen& q : in.rest(R)) {
        bool certified = dim == threshold && cuts_below_dimension(R, base, dim, threshold, {q});
        result += certified ? "1\n" : "0\n";
    }
    return result;
}

std::string op_verify(Reader& in, bool residual_only = false);

// The reduced singular locus of a hypersurface in characteristic zero.
// Return only a smooth positive-dimensional centre through the given origin.
// Codimension >= 2 rules out every repeated factor, even over Qbar: any
// repeated factor divides f and all its partial derivatives. A zero result
// leaves the ordinary resolution search in charge of multiplicities/units.
std::string op_smooth_reduced_centre(Reader& in) {
    const Ring R = grevlex(in.number(0));
    const Gen f = in.gen(R);
    if (in.at != in.end || f.p.empty()) return {};
    Ideal jacobian{f};
    for (int j = 0; j < R.n; ++j) jacobian.push_back(normal(derivative(R, f.p, j)));
    for (const auto& g : jacobian) {
        if (!g.p.empty() && g.p.m[g.p.m.size() - R.stride] == 0) return {};
    }
    const long dimension = dimension_of_basis(R, Basis(R, polys(jacobian)));
    if (dimension < 1 || dimension > R.n - 2) return {};

    local::Basis generators;
    for (const auto& g : jacobian) {
        if (g.p.empty()) continue;
        std::string row;
        write(R, row, g.p, &g.num.v, &g.den.v);
        generators.push_back(local::decode_row(row, R.n));
    }
    const auto radical = primary::radical_lex(generators, R.n);
    // A complete intersection is enough for this bounded candidate. Other
    // smooth presentations and overcomplete generators retain the fallback.
    if (radical.size() != static_cast<size_t>(R.n - dimension) ||
        R.n + radical.size() > MAX_VARIABLES) return {};
    Ideal centre;
    for (const auto& generator : radical) {
        const std::string row = local::encode_row(generator) + "\n";
        const char* at = row.data();
        Rational scale;
        Poly p = parse(R, at, at + row.size(), &scale);
        centre.push_back(normal(std::move(p), &scale.num.v, &scale.den.v));
    }
    if (!smooth_locus(R, centre, dimension, Test::IDEAL).smooth) return {};
    std::string out;
    put(R, out, centre);
    if (in.op == "smooth_reduced_centre") return out;

    // Reuse the verifier of arbitrary projective ideal blowups, including
    // exact chart saturation and all SNC strata. The centre equations in the
    // projective boundary have homogeneous degree zero in the ratio variables.
    const size_t k = centre.size();
    std::string frame = "verify " + std::to_string(R.n) + " " + std::to_string(k) +
                        " 1 " + std::to_string(k) + "\n";
    write(R, frame, f.p, &f.num.v, &f.den.v);
    frame += out;
    const Ring wide = grevlex(R.n + k);
    const auto places = shifted(R.n,0);
    for (const auto& g : centre) {
        const auto lift = convert(R, wide, g, places.data());
        write(wide, frame, lift.p, &lift.num.v, &lift.den.v);
    }
    // If the entire saturated ideal is the graph of polynomials over the
    // free coordinate axes, its base change to Qbar is affine space. The
    // exceptional divisor on the blowup is its projectivized normal bundle,
    // geometrically irreducible, with every standard chart meeting the
    // origin fibre. Its individual opens have precisely the centre ideal.
    const bool graph_bundle = in.numbers.size() == 2 && in.number(1) == 1 &&
                              local::polynomial_graph(radical, R.n);
    Reader verify(frame.data(),frame.size());
    const std::string proof = op_verify(verify, graph_bundle);
    if (proof.compare(0,5,"pass ") != 0) return {};
    if (graph_bundle) {
        // Use the same monic grevlex exceptional-ideal presentation as
        // geometric component decomposition; the projective pivots stay lex.
        const Ideal display = Basis(R, polys(centre)).monic();
        out += "exceptional " + std::to_string(display.size()) + "\n";
        put(R, out, display);
    }
    return std::string(graph_bundle ? "graph " : "centre ") + std::to_string(k) + "\n" + out + proof;
}

// For standalone saturation, small multipliers are cheap to adjoin directly.
// Centre selection queries graph_regular explicitly because a proof there
// also avoids constructing and testing both sides of a split.
bool saturation_regular(const Reader& in) {
    constexpr size_t minimum_terms = 32;
    constexpr size_t max_packet_bytes = 16 * 1024 * 1024;
    if (!in.count(1) || size_t(in.end - in.at) > max_packet_bytes) return false;
    std::string_view rows(in.at, in.end - in.at);
    if (!rows.empty() && rows.back() == '\n') rows.remove_suffix(1);
    const size_t split = rows.rfind('\n');
    if (split == std::string_view::npos) return false;
    const auto by = rows.substr(split + 1);
    if (size_t(std::count(by.begin(), by.end(), ';')) + 1 < minimum_terms ||
        rows.find('*') != std::string_view::npos) return false;
    std::istringstream stream{std::string(rows)};
    return graph_regular(stream, in.count(0), in.count(1));
}

// "saturate n s": s generators, then the element to saturate by.
std::string op_saturate(Reader& in) {
    const Ring R = grevlex(in.number(0));
    const bool unchanged = saturation_regular(in);
    const Ideal ideal = in.ideal(R, in.count(1));
    const Gen by = in.gen(R);
    std::string out;
    if (unchanged && !constant(by.p))
        put(R, out, lexicographic(R, polys(expressions(ideal))));
    else
        put(R, out, saturate(R, ideal, by, true));
    return out;
}

// The closure of the graph of the identification of two ordinary affine
// charts off their exceptional divisors. Embed their projections in the
// two independent coordinate blocks before clearing denominators in each
// difference. Saturating uses exactly the same basis order as op_saturate.
std::string op_affine_overlap(Reader& in) {
    const Ring left = grevlex(in.number(0)), right = grevlex(in.number(1));
    // Elimination adjoins an inverse to the product ring.
    if (left.n + right.n >= MAX_VARIABLES) throw Limit("Unsupported number of variables.");
    const Ring product = grevlex(left.n + right.n);
    const auto left_places = shifted(left.n, 0), right_places = shifted(right.n, left.n);
    const size_t maps = in.count(2);
    const Ideal left_images = in.ideal(left, maps);
    Ideal graph;
    for (const Gen& image : left_images) {
        const Gen a = convert(left, product, image, left_places.data());
        const Gen b = convert(right, product, in.gen(right), right_places.data());
        Z ca, cb;
        fmpz_mul(&ca.v, &a.num.v, &b.den.v);
        fmpz_mul(&cb.v, &b.num.v, &a.den.v);
        fmpz_neg(&cb.v, &cb.v);
        Poly difference;
        merge(product, difference, a.p, 0, &ca.v, nullptr, b.p, 0, &cb.v, nullptr);
        graph.push_back(normal(std::move(difference)));
    }

    size_t header = 3;
    Poly unit = one(product);
    auto include_divisors = [&](const Ring& ring, const std::vector<int>& places) {
        const size_t divisors = in.count(header++);
        for (size_t j = 0; j < divisors; ++j) {
            const size_t count = in.count(header++);
            Poly pivot;
            for (size_t k = 0; k < count; ++k) {
                Gen g = in.gen(ring);
                if (pivot.empty() && !g.p.empty()) pivot = std::move(g.p);
            }
            if (pivot.empty()) throw std::invalid_argument("The exceptional ideal vanishes on its entire chart.");
            unit = multiply(product, unit, convert(ring, product, pivot, places.data()));
        }
    };
    include_divisors(left, left_places);
    include_divisors(right, right_places);
    if (header != in.numbers.size() || in.at != in.end) throw std::invalid_argument("Bad overlap request.");
    std::string out;
    put(product, out, saturate(product, graph, normal(std::move(unit)), true));
    return out;
}

// "radical n s": s generators of the smaller ideal, then the larger one.
std::string op_radical(Reader& in) {
    const Ring R = grevlex(in.number(0));
    const Ideal smaller = in.ideal(R, in.count(1));
    const Ideal larger = in.rest(R);
    return radical_subset(R, smaller, larger) ? "1\n" : "0\n";
}

void put(const Ring& R, std::string& out, const Smooth& witness) {
    out += witness.empty ? "1 " : "0 ";
    out += witness.smooth ? "1 " : "0 ";
    out += std::to_string(witness.bad.size());
    out += '\n';
    put(R, out, witness.bad);
}

// "smooth n dimension": the generators. Answer: "empty smooth size" and the
// ideal of the complement of the smooth locus.
std::string op_smooth(Reader& in) {
    const Ring R = grevlex(in.number(0));
    const Ideal ideal = in.rest(R);
    std::string out;
    put(R, out, smooth_locus(R, ideal, in.number(1), Test::IDEAL));
    return out;
}

// The lists of an "snc" or "charts" header from position `first`: the size of
// the ambient ideal, of the strict transform (-1: none) and of each divisor.
struct Divisors {
    Ideal ambient, strict;
    bool has_strict = false;
    std::vector<Ideal> exceptional;
    Divisors(Reader& in, const Ring& R, size_t first) {
        ambient = in.ideal(R, in.count(first));
        has_strict = in.number(first + 1) >= 0;
        if (has_strict) strict = in.ideal(R, in.count(first + 1));
        const size_t pieces = in.count(first + 2);
        for (size_t k = 0; k < pieces; ++k) exceptional.push_back(in.ideal(R, in.count(first + 3 + k)));
    }
};

// "snc n dimension a s e k_1 ... k_e". Answer: "ok count", then for each
// stratum "c i_1 ... i_c" (divisor indices, the strict transform first when
// present) followed by its smoothness answer.
std::string op_snc(Reader& in) {
    const Ring R = grevlex(in.number(0));
    const Divisors given(in, R, 2);
    std::vector<Ideal> divisors;
    if (given.has_strict) divisors.push_back(expressions(given.strict));
    for (const Ideal& ideal : given.exceptional) divisors.push_back(expressions(ideal));
    Crossings crossings{R, divisors, in.number(1), nullptr, {}, true};
    crossings.run(given.ambient);
    std::string out = crossings.ok ? "1 " : "0 ";
    out += std::to_string(crossings.strata.size());
    out += '\n';
    for (const Stratum& stratum : crossings.strata) {
        out += std::to_string(stratum.parts.size());
        for (int part : stratum.parts) { out += ' '; out += std::to_string(part); }
        out += '\n';
        put(R, out, stratum.witness);
    }
    return out;
}

// "charts n a s e k_1 ... k_e": those ideals, then the generators of the
// centre. Answer, for each chart: "a s k_1 ... k_e" and the transformed
// ideals in the coordinates of the chart.
std::string op_charts(Reader& in) {
    const Ring R = grevlex(in.number(0));
    const Divisors given(in, R, 1);
    const Ideal center = in.rest(R);
    if (center.empty()) return "";
    if (R.n + static_cast<long>(center.size()) > MAX_VARIABLES) throw Limit("Unsupported number of variables.");
    std::string out;
    for (size_t i = 0; i < center.size(); ++i) {
        const Chart chart = blowup_chart(R, given.ambient, given.has_strict ? &given.strict : nullptr, center,
                                         given.exceptional, static_cast<int>(i), true);
        out += std::to_string(chart.ambient.size());
        out += ' ';
        out += given.has_strict ? std::to_string(chart.strict.size()) : "-1";
        for (const Ideal& ideal : chart.exceptional) { out += ' '; out += std::to_string(ideal.size()); }
        out += '\n';
        put(chart.ring, out, chart.ambient);
        put(chart.ring, out, chart.strict);
        for (const Ideal& ideal : chart.exceptional) put(chart.ring, out, ideal);
    }
    return out;
}

// "verify n k g b_1 ... b_g": the hypersurface f, the k generators of the
// centre, then g boundary ideals in the n coordinates and the k projective
// ones. The candidate is a resolution of f when the centre lies in the
// singular locus and, on every chart of its blowup, the boundary ideals cut
// out the exceptional divisor and make normal crossings with the strict
// transform. Answer: "rejected reason", or "pass strata" and, for each
// chart, "a s" with the ambient ideal and the strict transform.
std::string op_verify(Reader& in, bool residual_only) {
    const Ring R = grevlex(in.number(0));
    const int n = R.n;
    const size_t k = in.count(1);
    if (k < 1 || n + static_cast<long>(k) > MAX_VARIABLES) throw Limit("Unsupported number of variables.");
    const Ring U = grevlex(n + static_cast<long>(k));
    const Gen f = in.gen(R);
    const Ideal center = in.ideal(R, k);
    std::vector<Ideal> boundary;
    for (size_t g = 0; g < in.count(2); ++g) boundary.push_back(in.ideal(U, in.count(3 + g)));

    // Homogeneous in the projective coordinates, so that the ideals agree on overlaps.
    long exponents[MAX_VARIABLES];
    for (const Ideal& ideal : boundary)
        for (const Gen& g : ideal) {
            long degree = -1;
            for (size_t i = 0; i < g.p.size(); ++i) {
                U.unpack(g.p.m.data() + i * U.stride, exponents);
                long d = 0;
                for (size_t j = 0; j < k; ++j) d += exponents[n + j];
                if (degree >= 0 && d != degree) return "rejected homogeneity\n";
                degree = d;
            }
        }
    // The blowup must be an isomorphism over the smooth locus of f.
    Ideal singular;
    singular.push_back(f);
    for (int v = 0; v < n; ++v) singular.push_back(normal(derivative(R, f.p, v)));
    if (!radical_subset(R, singular, center)) return "rejected centre\n";

    Ideal strict;
    strict.push_back(f);
    const Ideal nothing;
    size_t strata = 0;
    std::vector<int> place(n + k);
    // The tests do not depend on the generators chosen for the ideals of a chart.
    for (size_t i = 0; i < k; ++i) {
        const Chart chart = blowup_chart(R, nothing, &strict, center, {}, static_cast<int>(i), false);
        const Ring& C = chart.ring;
        for (int v = 0; v < n; ++v) place[v] = v;
        for (size_t j = 0; j < k; ++j)
            place[n + j] = j == i ? -1 : n + static_cast<int>(j < i ? j : j - 1);
        std::vector<Ideal> pieces;
        for (const Ideal& ideal : boundary) {
            pieces.emplace_back();
            for (const Gen& g : ideal) pieces.back().push_back(convert(U, C, g, place.data()));
        }
        if (!cover(C, chart.ambient, chart.pivot, pieces)) return "rejected cover\n";
        // The chart of a blowup of affine space is irreducible of dimension
        // n. The strict transform of the hypersurface consists of components
        // of the zero set of f in it, which all have codimension one.
        std::vector<Ideal> divisors;
        std::vector<bool> pure;
        divisors.push_back(expressions(chart.strict));
        pure.push_back(true);
        for (const Ideal& piece : pieces) {
            divisors.push_back(expressions(piece));
            pure.push_back(principal(C, chart.ambient, chart.pivot, piece));
        }
        Crossings crossings{C, divisors, n, &pure, {}, true};
        crossings.run(chart.ambient);
        if (!crossings.ok) return "rejected crossings\n";
        strata += crossings.strata.size();
    }
    std::string out = "pass " + std::to_string(strata) + "\n";
    for (size_t i = 0; i < k; ++i) {
        Chart chart = blowup_chart(R, nothing, &strict, center, {}, static_cast<int>(i), true);
        if (residual_only) {
            Basis ambient(chart.ring, polys(chart.ambient));
            auto& residual = chart.strict;
            residual.erase(std::remove_if(residual.begin(),residual.end(),
                [&](const Gen& g){return ambient.contains(g.p);}), residual.end());
        }
        out += std::to_string(chart.ambient.size());
        out += ' ';
        out += std::to_string(chart.strict.size());
        out += '\n';
        put(chart.ring, out, chart.ambient);
        put(chart.ring, out, chart.strict);
    }
    return out;
}


// ---------------------------------------------------------------------------
// The step of the procedure for a smooth surface X in a smooth threefold with
// exceptional divisors (engine/surface.py, which states the procedure and why
// it ends). X is given by equations in affine space; on an open set where a
// Jacobian minor M of c of them in c of the variables is invertible, the other
// two variables are coordinates of X and
//     D_k f = det d(g_1, ..., g_c, f)/d(y, x_k) = M * (df/dx_k on X).
// A principal ideal J of X has radical J : (J + D J); the unions of the curves
// of a divisor of multiplicity at least 1, 2, ... follow by quotients.

// With ATLAS_TRACE set, the time of each part of a step goes to stderr.
struct Timed {
    const char* name;
    std::chrono::steady_clock::time_point start;
    explicit Timed(const char* label) : name(label), start(std::chrono::steady_clock::now()) {
        const char* level = std::getenv("ATLAS_TRACE");
        if (level && level[0] == '2') std::fprintf(stderr, "    [kernel] %s ...\n", name);
    }
    ~Timed() {
        static const char* level = std::getenv("ATLAS_TRACE");
        if (!level) return;
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (seconds >= 0.05 || level[0] == '2') std::fprintf(stderr, "    [kernel] %s %.2fs\n", name, seconds);
    }
};

Poly combine(const Ring& R, const Poly& a, const Poly& b, bool subtract) {
    Z minus;
    fmpz_set_si(&minus.v, -1);
    Poly out;
    merge(R, out, a, 0, nullptr, nullptr, b, 0, subtract ? &minus.v : nullptr, nullptr);
    return out;
}

Poly determinant(const Ring& R, const std::vector<std::vector<Poly>>& m) {
    const size_t c = m.size();
    if (c == 0) return one(R);
    if (c == 1) return m[0][0];
    Poly sum;
    for (size_t j = 0; j < c; ++j) {
        if (m[0][j].empty()) continue;
        std::vector<std::vector<Poly>> minor;
        for (size_t i = 1; i < c; ++i) {
            std::vector<Poly> row;
            for (size_t k = 0; k < c; ++k)
                if (k != j) row.push_back(m[i][k]);
            minor.push_back(std::move(row));
        }
        const Poly below = determinant(R, minor);
        if (below.empty()) continue;
        sum = combine(R, sum, multiply(R, m[0][j], below), j & 1);
    }
    return sum;
}

std::vector<Poly> present(std::vector<Poly> gens) {
    gens.erase(std::remove_if(gens.begin(), gens.end(), [](const Poly& p) { return p.empty(); }), gens.end());
    return gens;
}

bool unit_basis(const Ring& R, const std::vector<Poly>& gens) {
    const std::vector<Poly> input = present(gens);
    if (input.empty()) return false;
    return Basis(R, input).unit();
}

// The polynomial p*num/den, with p primitive.
struct Fraction {
    Poly p;
    Z num, den;
    Fraction() {
        fmpz_one(&num.v);
        fmpz_one(&den.v);
    }
};

void lowest(Fraction& f) {
    Z common;
    fmpz_gcd(&common.v, &f.num.v, &f.den.v);
    if (!fmpz_is_one(&common.v)) {
        fmpz_divexact(&f.num.v, &f.num.v, &common.v);
        fmpz_divexact(&f.den.v, &f.den.v, &common.v);
    }
}

// The normal form of f modulo the basis, exactly.
Fraction residue(Basis& basis, Poly f) {
    Fraction out;
    basis.engine.reduce(f, 0, &out.num.v, &out.den.v);
    out.p = std::move(f);
    lowest(out);
    return out;
}

Fraction times(const Ring& R, const Fraction& a, const Fraction& b) {
    Fraction out;
    if (a.p.empty() || b.p.empty()) return out;
    out.p = multiply(R, a.p, b.p);
    fmpz_mul(&out.num.v, &a.num.v, &b.num.v);
    fmpz_mul(&out.den.v, &a.den.v, &b.den.v);
    primitive(out.p, &out.num.v);
    lowest(out);
    return out;
}

Fraction plus(const Ring& R, const Fraction& a, const Fraction& b, bool subtract) {
    if (b.p.empty()) return a;
    Z ca, cb;
    fmpz_mul(&ca.v, &a.num.v, &b.den.v);
    fmpz_mul(&cb.v, &b.num.v, &a.den.v);
    if (subtract) fmpz_neg(&cb.v, &cb.v);
    Fraction out;
    if (a.p.empty()) {
        out.p = b.p;
        fmpz_set(&out.num.v, &b.num.v);
        if (subtract) fmpz_neg(&out.num.v, &out.num.v);
        fmpz_set(&out.den.v, &b.den.v);
        return out;
    }
    merge(R, out.p, a.p, 0, &ca.v, nullptr, b.p, 0, &cb.v, nullptr);
    fmpz_mul(&out.den.v, &a.den.v, &b.den.v);
    primitive(out.p, &out.num.v);
    lowest(out);
    return out;
}

// The Rabinowitsch test with a unit already reduced modulo the basis.
bool empty_off_basis(const Ring& R, const Basis& basis, const Poly& unit) {
    if (unit.empty() || basis.unit()) return true;
    if (constant(unit) || basis.engine.G.empty()) return false;
    Ring T;
    T.init(R.n + 1, 0);
    const std::vector<int> up = shifted(R.n, 1);
    std::vector<Poly> wide;
    for (const Poly& g : basis.elements()) wide.push_back(convert(R, T, g, up.data()));
    wide.push_back(inverse_relation(R, T, unit));
    return Basis(T, wide, wide.size() - 1).unit();
}

// Congruent units determine the same open set of V(gens).
bool empty_off(const Ring& R, const std::vector<Poly>& gens, const Poly& u) {
    const std::vector<Poly> input = present(gens);
    if (input.empty()) return u.empty();
    Basis basis(R, input);
    return empty_off_basis(R, basis, basis.remainder(u));
}

class Surface {
public:
    Surface(const Ring& R, std::vector<Poly> rows, const std::vector<int>& dependent, Poly minor)
        : R_(R), rows_(present(std::move(rows))), minor_(std::move(minor)), up_(shifted(R.n, 1)) {
        T_.init(R.n + 1, 0);
        std::vector<bool> taken(R.n, false);
        for (int v : dependent) taken[v] = true;
        for (int v = 0; v < R.n; ++v)
            if (!taken[v]) coords_.push_back(v);
        // The vector fields D_k: the signed minors of the Jacobian of the rows
        // in the columns (dependent variables, x_k).
        const size_t c = dependent.size();
        for (int k : coords_) {
            std::vector<int> columns(dependent);
            columns.push_back(k);
            std::vector<std::pair<int, Poly>> field;
            for (size_t j = 0; j < columns.size(); ++j) {
                std::vector<std::vector<Poly>> m;
                for (size_t i = 0; i < c; ++i) {
                    std::vector<Poly> row;
                    for (size_t l = 0; l < columns.size(); ++l)
                        if (l != j) row.push_back(derivative(R_, rows_[i], columns[l]));
                    m.push_back(std::move(row));
                }
                Poly coefficient = determinant(R_, m);
                if (coefficient.empty()) continue;
                if ((c + j) & 1) coefficient = combine(R_, Poly(), coefficient, true);
                field.emplace_back(columns[j], std::move(coefficient));
            }
            fields_.push_back(std::move(field));
        }
        if (!rows_.empty()) reducer_.reset(new Basis(R_, rows_));
    }

    // Whether the ideal has no zero on the open set of the surface.
    bool empty(const std::vector<Poly>& gens) const {
        std::vector<Poly> all = rows_;
        all.insert(all.end(), gens.begin(), gens.end());
        return empty_inside(all);
    }
    // The same for an ideal whose zeros lie on the surface: its equations are
    // not needed. The function u that is invertible on the open set is first
    // reduced modulo the ideal I: I has no zero with u != 0 exactly when
    // I + (t*u - 1) is the unit ideal, and u may be replaced by its normal form.
    bool empty_inside(const std::vector<Poly>& gens) const {
        const std::vector<Poly> input = present(gens);
        if (input.empty()) return false;
        Basis basis(R_, input);
        if (basis.unit()) return true;
        if (constant(minor_)) return false;
        Poly unit = basis.remainder(minor_);
        if (unit.empty()) return true;  // u vanishes on every zero of I
        if (constant(unit)) return false;
        std::vector<Poly> wide = lifted(basis.elements());
        wide.push_back(inverse_relation(R_, T_, unit));
        return Basis(T_, wide).unit();
    }
    bool here(const std::vector<Poly>& gens, const std::vector<std::vector<Poly>>& working) const {
        for (const std::vector<Poly>& w : working) {
            std::vector<Poly> all(gens);
            all.insert(all.end(), w.begin(), w.end());
            if (!empty(all)) return true;
        }
        return false;
    }
    // Whether f is zero on the surface (on the open set X' where the minor u
    // is invertible). Three exact answers, the cheap ones first: f lies in
    // the ideal of the equations; a point of X' where f is not zero exists;
    // f lies in (equations) : u^infinity, by a basis of equations + (t*u - 1).
    bool vanishes(const Poly& f) const {
        if (f.empty()) return true;
        if (rows_.empty()) return false;
        const Poly r = reducer_->remainder(f);
        if (r.empty()) return true;
        if (witness(r)) return false;
        if (!local_) {
            const Timed part("basis of the open set of the surface");
            std::vector<Poly> input = lifted(rows_);
            if (!constant(minor_)) input.push_back(inverse_relation(R_, T_, minor_));
            local_.reset(new Basis(T_, input));
        }
        return local_->contains(convert(R_, T_, r, up_.data()));
    }
    // A point of X' with f != 0. The coordinates of the frame are given
    // integer values a; the points of X' over a are the zeros of the ideal
    // Z = equations + (x_k - a_k) where u != 0, and one of them has f != 0
    // exactly when Z + (t*u*f - 1) is not the unit ideal, u*f taken modulo Z.
    bool witness(const Poly& f) const {
        static const long values[3][4] = {{2, 3, -1, 5}, {-3, 1, 4, -2}, {5, -4, 2, 7}};
        if (coords_.size() > 4) return false;
        for (const long* a : values) {
            std::vector<Poly> input = rows_;
            for (size_t k = 0; k < coords_.size(); ++k) {
                u64 x[MAX_STRIDE];
                variable(R_, x, coords_[k]);
                Poly linear;
                linear.m.insert(linear.m.end(), x, x + R_.stride);
                linear.c.emplace_back();
                fmpz_set_si(&linear.c.back().v, 1);
                linear.m.resize(linear.m.size() + R_.stride, 0);
                linear.c.emplace_back();
                fmpz_set_si(&linear.c.back().v, -a[k]);
                input.push_back(std::move(linear));
            }
            Basis points(R_, input);
            if (points.unit()) continue;
            Poly value = points.remainder(f);
            if (value.empty()) continue;
            if (!constant(minor_)) {
                const Poly unit = points.remainder(minor_);
                if (unit.empty()) continue;
                value = points.remainder(multiply(R_, value, unit));
                if (value.empty()) continue;
            }
            if (constant(value)) return true;
            std::vector<Poly> wide = lifted(points.elements());
            wide.push_back(inverse_relation(R_, T_, value));
            if (!Basis(T_, wide).unit()) return true;
        }
        return false;
    }
    Poly reduced(Poly f) const {
        if (f.empty() || !reducer_) return f;
        return reducer_->remainder(std::move(f));
    }
    // D_k f for the k-th coordinate (0 or 1); exact, not reduced.
    Poly derive(const Poly& f, int which) const {
        Poly sum;
        for (const auto& term : fields_[which]) {
            const Poly d = derivative(R_, f, term.first);
            if (d.empty()) continue;
            sum = combine(R_, sum, multiply(R_, term.second, d), false);
        }
        return sum;
    }
    std::vector<Poly> functions(const std::vector<Poly>& ideal) const {
        std::vector<Poly> out;
        for (const Poly& g : ideal)
            if (!vanishes(g)) out.push_back(g);
        return out;
    }
    std::vector<Poly> derived(const std::vector<Poly>& ideal) const {
        std::vector<Poly> out;
        for (const Poly& g : functions(ideal))
            for (int which = 0; which < 2; ++which) {
                Poly d = reduced(derive(g, which));
                if (!d.empty()) out.push_back(std::move(d));
            }
        return out;
    }
    std::vector<Poly> quotient(const std::vector<Poly>& ideal, const std::vector<Poly>& by) const {
        Quotient q(R_);
        return Basis(R_, present(q.colon(present(ideal), present(by)))).elements();
    }
    std::vector<Poly> radical(const std::vector<Poly>& ideal) const {
        std::vector<Poly> by(ideal);
        for (Poly& d : derived(ideal)) by.push_back(std::move(d));
        return quotient(ideal, by);
    }
    // The residues modulo a basis of the coefficients of the two fields.
    std::vector<std::vector<Fraction>> coefficients(Basis& basis) const {
        std::vector<std::vector<Fraction>> out;
        for (const auto& field : fields_) {
            out.emplace_back();
            for (const auto& term : field) out.back().push_back(residue(basis, term.second));
        }
        return out;
    }
    // The residue of D_k f, from those of the coefficients of D_k and of the
    // partial derivatives of f: nothing large is multiplied.
    Fraction derive(Basis& basis, const std::vector<std::vector<Fraction>>& known, const Poly& f, int which) const {
        Fraction sum;
        for (size_t t = 0; t < fields_[which].size(); ++t) {
            const Poly d = derivative(R_, f, fields_[which][t].first);
            if (d.empty() || known[which][t].p.empty()) continue;
            sum = plus(R_, sum, times(R_, known[which][t], residue(basis, d)), false);
        }
        return sum;
    }
    const std::vector<Poly>& rows() const { return rows_; }
    const Poly& minor() const { return minor_; }
    const Ring& ring() const { return R_; }

private:
    Ring R_, T_;
    std::vector<Poly> rows_;
    Poly minor_;
    std::vector<int> up_, coords_;
    std::vector<std::vector<std::pair<int, Poly>>> fields_;
    std::unique_ptr<Basis> reducer_;
    mutable std::unique_ptr<Basis> local_;

    std::vector<Poly> lifted(const std::vector<Poly>& gens) const {
        std::vector<Poly> out;
        for (const Poly& g : gens) out.push_back(convert(R_, T_, g, up_.data()));
        return out;
    }
};

struct Step {
    bool found = false;
    long kind = 0, top = 0, flag = 0;
    std::vector<long> labels;
    std::vector<Poly> gens;
};

std::vector<Poly> joined(std::vector<Poly> a, const std::vector<Poly>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

// The curves that a divisor E cuts on X are those that X cuts on E, with the
// same multiplicities (both are the scheme X meet E). When E is itself given
// with coordinates (`planes[i]`, for instance a coordinate plane), they are
// computed there, where derivations are plain; otherwise on X.
Step surface_step(const Surface& X, const std::vector<Poly>& equations, const std::vector<Poly>& functions,
                  const std::vector<std::vector<Poly>>& working, const std::vector<std::unique_ptr<Surface>>& planes) {
    const Ring& R = X.ring();
    Step step;
    std::vector<size_t> labels;
    for (size_t i = 0; i < functions.size(); ++i)
        if (X.here({functions[i]}, working)) labels.push_back(i);
    if (labels.empty()) return step;
    auto frame = [&](size_t i) -> const Surface& { return planes[i] ? *planes[i] : X; };
    // For each divisor, the unions of its curves of multiplicity at least 1, 2, ...
    std::vector<std::vector<std::vector<Poly>>> levels(functions.size());
    size_t top = 0;
    for (size_t i : labels) {
        const Timed timed(planes[i] ? "traces on the divisor" : "traces on the surface");
        const Surface& F = frame(i);
        std::vector<Poly> current = planes[i] ? joined(F.rows(), equations) : joined(X.rows(), {functions[i]});
        while (levels[i].size() < 64) {
            std::vector<Poly> reduced = F.radical(current);
            if (!X.here(reduced, working)) break;
            current = F.quotient(current, reduced);
            levels[i].push_back(std::move(reduced));
        }
        if (levels[i].empty()) throw std::logic_error("A divisor through the points has no curve there.");
        top = std::max(top, levels[i].size());
    }
    // (a) the points where the union of the traces is worse than nodal.
    std::vector<Poly> whole = levels[labels[0]][0];
    {
        const Timed timed("union of the traces");
        Quotient q(R);
        for (size_t k = 1; k < labels.size(); ++k)
            whole = Basis(R, present(q.intersection(whole, levels[labels[k]][0]))).elements();
    }
    std::vector<Poly> worse;
    {
        const Timed timed("first derivatives");
        worse = joined(whole, X.derived(whole));
    }
    {
        // The Hessians are only needed at the singular points of the union:
        // modulo their ideal J. A derivation of X maps the ideal of X into
        // itself, so the first derivatives may be reduced modulo it, and the
        // second ones are computed as residues modulo J.
        const Timed timed("Hessians");
        Basis points(R, present(worse));
        if (!points.unit()) {
            const std::vector<std::vector<Fraction>> known = X.coefficients(points);
            std::vector<Poly> hessians;
            for (const Poly& g : X.functions(whole)) {
                const Poly ga = X.reduced(X.derive(g, 0)), gb = X.reduced(X.derive(g, 1));
                // The same factor of ga and of gb appears in both products.
                const Fraction h = plus(R, times(R, X.derive(points, known, ga, 0), X.derive(points, known, gb, 1)),
                                        times(R, X.derive(points, known, ga, 1), X.derive(points, known, gb, 0)), true);
                Poly left = points.remainder(h.p);
                if (!left.empty()) hessians.push_back(std::move(left));
            }
            for (Poly& h : hessians) worse.push_back(std::move(h));
        }
    }
    bool bad_points;
    {
        const Timed timed("points worse than nodes");
        bad_points = X.here(worse, working);
    }
    if (bad_points) {
        step.found = true;
        step.kind = 3;
        step.gens = std::move(worse);
        return step;
    }
    // (b) the curves with the largest multiplicity, or shared by two divisors.
    std::vector<Poly> chosen;
    size_t owner = 0;
    bool have = false;
    if (top >= 2) {
        for (size_t k = labels.size(); k-- > 0;)
            if (levels[labels[k]].size() == top) {
                step.labels = {static_cast<long>(labels[k])};
                chosen = levels[labels[k]][top - 1];
                owner = labels[k];
                have = true;
                break;
            }
    } else {
        for (size_t a = labels.size(); a-- > 0 && !have;)
            for (size_t b = labels.size(); b-- > a + 1 && !have;) {
                const std::vector<Poly>& first = levels[labels[a]][0];
                const std::vector<Poly> outside = saturation(R, present(first), functions[labels[b]]);
                const bool all = outside.size() == 1 && constant(outside[0]);
                std::vector<Poly> inside = all ? first : X.quotient(first, outside);
                if (X.here(inside, working)) {
                    step.labels = {static_cast<long>(labels[a]), static_cast<long>(labels[b])};
                    chosen = std::move(inside);
                    owner = labels[a];
                    have = true;
                }
            }
    }
    if (have) {
        const Timed timed("singular points of the chosen curves");
        std::vector<Poly> singular;
        {
            const Timed part("derivatives of the chosen curves");
            singular = joined(chosen, frame(owner).derived(chosen));
        }
        step.found = true;
        step.kind = 2;
        step.top = static_cast<long>(top);
        // Smooth everywhere is decided without the open set; only otherwise is it needed.
        bool smooth;
        {
            const Timed part("smooth everywhere");
            smooth = unit_basis(R, singular);
        }
        if (!smooth && !X.empty_inside(singular)) {
            step.flag = 1;
            step.gens = std::move(singular);
        } else {
            step.gens = std::move(chosen);
        }
        return step;
    }
    // (c) the nodes where a divisor crosses itself.
    for (size_t k = labels.size(); k-- > 0;) {
        const std::vector<Poly>& trace = levels[labels[k]][0];
        std::vector<Poly> nodes = joined(trace, frame(labels[k]).derived(trace));
        if (X.here(nodes, working)) {
            step.found = true;
            step.kind = 1;
            step.labels = {static_cast<long>(labels[k])};
            step.gens = std::move(nodes);
            return step;
        }
    }
    return step;
}

// "surface n b e f u w s_1 ... s_w": e equations of the surface in n variables,
// the first b of them those of the threefold it lies in; f functions (the
// equations of the divisors, in the order of their labels), u
// functions that are invertible on the part of the surface considered, then
// the w ideals of the points considered, with s_i generators each. The answer
// is "nomodel" when no two variables are coordinates near some of the points;
// otherwise, for each open set used and each step found there, a line
// "candidate kind top flag l labels... count" and the count generators of the
// closure in the chart of the locus to blow up.
std::string op_surface(Reader& in) {
    const Ring R = grevlex(in.number(0));
    if (R.n + 1 > MAX_VARIABLES) throw Limit("Unsupported number of variables.");
    const size_t b = in.count(1), e = in.count(2), f = in.count(3), u = in.count(4), w = in.count(5);
    if (b > e) throw std::invalid_argument("Bad header.");
    const std::vector<Poly> threefold = present(polys(in.ideal(R, b)));
    const std::vector<Poly> equations = joined(threefold, present(polys(in.ideal(R, e - b))));
    const std::vector<Poly> functions = polys(in.ideal(R, f));
    Poly units = one(R);
    for (const Poly& unit : present(polys(in.ideal(R, u)))) units = multiply(R, units, unit);
    std::vector<std::vector<Poly>> remaining;
    for (size_t i = 0; i < w; ++i) {
        std::vector<Poly> ideal = present(polys(in.ideal(R, in.count(6 + i))));
        if (!empty_off(R, joined(equations, ideal), units)) remaining.push_back(std::move(ideal));
    }
    const int n = R.n, c = n - 2;
    if (c < 0 || static_cast<size_t>(c) > equations.size()) throw std::invalid_argument("Not a surface.");
    // A divisor of the threefold with coordinates of its own: two variables, the
    // others determined by its equations with a Jacobian minor that vanishes
    // nowhere on it.
    std::vector<std::unique_ptr<Surface>> planes(functions.size());
    if (static_cast<int>(threefold.size()) == c - 1 && equations.size() == threefold.size() + 1) {
        for (size_t i = 0; i < functions.size(); ++i) {
            if (functions[i].empty()) continue;
            const std::vector<Poly> rows = joined(threefold, {functions[i]});
            std::vector<int> chosen(c);
            for (int d = 0; d < c; ++d) chosen[d] = d;
            for (;;) {
                std::vector<std::vector<Poly>> m;
                for (int p = 0; p < c; ++p) {
                    std::vector<Poly> row;
                    for (int q = 0; q < c; ++q) row.push_back(derivative(R, rows[p], chosen[q]));
                    m.push_back(std::move(row));
                }
                Poly minor = determinant(R, m);
                if (!minor.empty()) {
                    primitive(minor, nullptr);
                    if (unit_basis(R, joined(rows, {minor}))) {
                        planes[i].reset(new Surface(R, rows, chosen, one(R)));
                        break;
                    }
                }
                int d = c - 1;
                while (d >= 0 && chosen[d] == n - c + d) --d;
                if (d < 0) break;
                ++chosen[d];
                for (int q = d + 1; q < c; ++q) chosen[q] = chosen[q - 1] + 1;
            }
        }
    }
    std::string out;
    std::vector<int> dependent(c);
    for (int d = 0; d < c; ++d) dependent[d] = d;
    bool more_dependent = true;
    while (more_dependent && !remaining.empty()) {
        std::vector<size_t> rows(c);
        for (int d = 0; d < c; ++d) rows[d] = d;
        bool more_rows = true;
        while (more_rows && !remaining.empty()) {
            std::vector<std::vector<Poly>> m;
            std::vector<Poly> chosen;
            for (int i = 0; i < c; ++i) {
                std::vector<Poly> row;
                for (int j = 0; j < c; ++j) row.push_back(derivative(R, equations[rows[i]], dependent[j]));
                m.push_back(std::move(row));
                chosen.push_back(equations[rows[i]]);
            }
            Poly minor;
            {
                const Timed part("minor of a frame");
                minor = determinant(R, m);
            }
            if (!minor.empty()) {
                primitive(minor, nullptr);
                std::vector<std::vector<Poly>> hit, rest;
                Ideal smaller;
                smaller.push_back(normal(minor));
                for (const std::vector<Poly>& ideal : remaining) {
                    const Timed part("points inside a frame");
                    Ideal larger;
                    for (const Poly& g : joined(equations, ideal)) larger.push_back(normal(g));
                    (radical_subset(R, smaller, larger) ? rest : hit).push_back(ideal);
                }
                if (!hit.empty()) {
                    Poly invertible = multiply(R, minor, units);
                    primitive(invertible, nullptr);
                    const Surface X(R, chosen, dependent, invertible);
                    Step step = surface_step(X, equations, functions, hit, planes);
                    if (step.found) {
                        // The closure in the chart: the ideal saturated by the invertible
                        // function, taken modulo the ideal first.
                        const Timed closing("closure of the centre");
                        Basis open(R, present(joined(equations, step.gens)));
                        std::vector<Poly> centre = open.elements();
                        if (!open.unit() && !constant(invertible)) {
                            Poly by = open.remainder(invertible);
                            const Timed saturating("saturation of the centre");
                            if (by.empty()) centre = {one(R)};
                            else if (!constant(by)) centre = saturation(R, centre, by);
                        }
                        Basis closed(R, centre);
                        if (!closed.unit()) {
                            const Ideal gens = closed.monic();
                            out += "candidate " + std::to_string(step.kind) + " " + std::to_string(step.top) + " " +
                                   std::to_string(step.flag) + " " + std::to_string(step.labels.size());
                            for (long label : step.labels) out += " " + std::to_string(label);
                            out += " " + std::to_string(gens.size()) + "\n";
                            put(R, out, gens);
                        }
                    }
                    for (std::vector<Poly>& ideal : hit) {
                        const Timed covering("points left outside the frame");
                        ideal.push_back(minor);
                        // Only the points where the units given are invertible count.
                        if (!empty_off(R, joined(equations, ideal), units)) rest.push_back(std::move(ideal));
                    }
                    remaining = std::move(rest);
                }
            }
            // The next set of rows, then of dependent variables, in lexicographic order.
            int d = c - 1;
            while (d >= 0 && rows[d] == equations.size() - c + d) --d;
            if (d < 0) more_rows = false;
            else {
                ++rows[d];
                for (int k = d + 1; k < c; ++k) rows[k] = rows[k - 1] + 1;
            }
        }
        int d = c - 1;
        while (d >= 0 && dependent[d] == n - c + d) --d;
        if (d < 0) more_dependent = false;
        else {
            ++dependent[d];
            for (int k = d + 1; k < c; ++k) dependent[k] = dependent[k - 1] + 1;
        }
    }
    if (!remaining.empty()) return "nomodel\n";
    return out.empty() ? "none\n" : out;
}


// Only zero sets enter the constructible Euler computation. A monomial
// base equation has the same zero set as the product of its distinct
// variables. Every multiple of that squarefree product vanishes there,
// so remove such terms from all other conditions *before* factorization
// or fibre projection. Keep these reductions out of the Jacobian test.
void restrict_to_base_monomials(const Ring& R, std::vector<Poly>& base,
                                std::vector<std::vector<Poly>>& ideals,
                                std::vector<std::vector<Poly>>& removed) {
    std::vector<Poly> divisors, other;
    for (auto& f : base) {
        if (f.size() != 1) { other.push_back(std::move(f)); continue; }
        long powers[MAX_VARIABLES];
        R.unpack(f.m.data(), powers);
        for (int j = 0; j < R.n; ++j) powers[j] = powers[j] != 0;
        R.pack(f.m.data(), powers);
        if (std::any_of(divisors.begin(), divisors.end(), [&](const Poly& p) {
                return R.divides(p.m.data(), f.m.data()); })) continue;
        divisors.erase(std::remove_if(divisors.begin(), divisors.end(), [&](const Poly& p) {
            return R.divides(f.m.data(), p.m.data()); }), divisors.end());
        divisors.push_back(std::move(f));
    }
    base = std::move(other);
    if (divisors.empty()) return;
    auto restrict = [&](std::vector<Poly>& equations) {
        for (auto& f : equations) {
            Poly out;
            for (size_t i = 0; i < f.size(); ++i) {
                auto m = f.m.data() + i*R.stride;
                if (std::any_of(divisors.begin(), divisors.end(), [&](const Poly& p) {
                        return R.divides(p.m.data(), m); })) continue;
                out.m.insert(out.m.end(), m, m+R.stride);
                out.c.push_back(std::move(f.c[i]));
            }
            f.swap(out);
        }
        equations = present(std::move(equations));
    };
    restrict(base);
    for (auto& ideal : ideals) restrict(ideal);
    for (auto& ideal : removed) restrict(ideal);
    for (auto& p : divisors) base.push_back(std::move(p));
}

void append_stratum(std::string& out, const std::vector<size_t>& subset, const Z& chi) {
    out += "stratum " + std::to_string(subset.size());
    for (size_t j : subset) out += " " + std::to_string(j);
    char* decimal = fmpz_get_str(nullptr, 10, &chi.v);
    out += " "; out += decimal; out += '\n';
    flint_free(decimal);
}

// A character of a coordinate torus is a unit. A hypersurface depending
// on fewer than d independent character differences has a torus factor of
// positive dimension, hence Euler characteristic zero. With d independent
// differences and d+1 terms the corresponding isogeny takes the section to
// a hyperplane minus its coordinate hyperplanes. Its Euler characteristic
// is (-1)^(d-1) times the degree of the isogeny (the absolute determinant).
bool orbit_section_euler(const Ring& R, const Poly& section,
                         const std::vector<unsigned>& supports, unsigned zeros,
                         Z& chi) {
    std::vector<int> active;
    for (int j = 0; j < R.n; ++j) if (!(zeros & (1u << j))) active.push_back(j);
    const size_t d = active.size();
    std::vector<const u64*> terms;
    for (size_t i = 0; i < section.size(); ++i)
        if (!(zeros & supports[i])) terms.push_back(section.m.data() + i*R.stride);
    if (terms.empty()) {
        fmpz_set_ui(&chi.v, d == 0);
        return true;
    }
    if (terms.size() == 1 || terms.size() <= d) return true;
    if (terms.size() != d + 1) return false;
    fmpz_mat_t differences;
    fmpz_mat_init(differences, d, d);
    for (size_t row = 0; row < d; ++row)
        for (size_t col = 0; col < d; ++col) {
            const int pos = R.position(active[col]);
            const long difference = long(R.field(terms[row+1], pos)) - long(R.field(terms[0], pos));
            fmpz_set_si(fmpz_mat_entry(differences, row, col), difference);
        }
    fmpz_mat_det(&chi.v, differences);
    fmpz_mat_clear(differences);
    fmpz_abs(&chi.v, &chi.v);
    if (!(d % 2)) fmpz_neg(&chi.v, &chi.v);
    return true;
}

// Partition a monomial base by its exact coordinate zero set. Each
// monomial divisor then contains the entire orbit or is disjoint from it.
// One additional hypersurface may split the orbit into its section and
// its complement. If any section is unresolved, use the general algorithm.
bool monomial_orbit_strata(const Ring& R, const std::vector<Poly>& base,
                           const std::vector<std::vector<Poly>>& ideals, std::string& out) {
    if (R.n > 8 || ideals.size() > size_t(R.n+1)) return false;
    auto support = [&](const u64* exponents) {
        unsigned mask = 0;
        for (int j = 0; j < R.n; ++j)
            if (R.field(exponents, R.position(j))) mask |= 1u << j;
        return mask;
    };
    std::vector<unsigned> required;
    for (const auto& f : base) {
        if (f.size() != 1) return false;
        required.push_back(support(f.m.data()));
    }
    const size_t k = ideals.size();
    std::vector<unsigned> axes(k), terms;
    size_t section = k;
    for (size_t j = 0; j < k; ++j) {
        if (ideals[j].empty()) continue; // The zero ideal contains every orbit.
        if (ideals[j].size() != 1) return false;
        const auto& f = ideals[j][0];
        if (f.size() == 1) {
            axes[j] = support(f.m.data());
        } else {
            if (section != k) return false;
            section = j;
            for (size_t i = 0; i < f.size(); ++i) terms.push_back(support(f.m.data() + i*R.stride));
        }
    }
    std::vector<Z> totals(1u << k);
    const unsigned full = (1u << R.n) - 1;
    for (unsigned zeros = 0; zeros <= full; ++zeros) {
        if (std::any_of(required.begin(), required.end(), [&](unsigned g){ return !(g & zeros); })) continue;
        unsigned subset = 0;
        for (size_t j = 0; j < k; ++j)
            if (ideals[j].empty() || (axes[j] & zeros)) subset |= 1u << j;
        if (zeros == full) fmpz_add_ui(&totals[subset].v, &totals[subset].v, 1);
        if (section < k) {
            Z chi;
            if (!orbit_section_euler(R, ideals[section][0], terms, zeros, chi)) return false;
            const auto with = subset | (1u << section);
            fmpz_add(&totals[with].v, &totals[with].v, &chi.v);
            fmpz_sub(&totals[subset].v, &totals[subset].v, &chi.v);
        }
    }
    std::vector<std::vector<size_t>> subsets;
    for (unsigned mask = 1; mask < totals.size(); ++mask) {
        if (fmpz_is_zero(&totals[mask].v)) continue;
        subsets.emplace_back();
        for (size_t j = 0; j < k; ++j) if (mask & (1u << j)) subsets.back().push_back(j);
    }
    std::sort(subsets.begin(), subsets.end(), [](const auto& a, const auto& b) {
        return a.size() == b.size() ? a < b : a.size() < b.size();
    });
    for (const auto& subset : subsets) {
        unsigned mask = 0;
        for (size_t j : subset) mask |= 1u << j;
        append_stratum(out, subset, totals[mask]);
    }
    return true;
}

// "strata n b k s_1 ... s_k [r t_1 ... t_r]": b equations of a part of a
// chart, then k ideals of closed sets, then r ideals of closed sets that are
// left out of the part. For every set J of the k whose common part there is
// not empty, the Euler characteristic with compact supports of the points of
// the part lying on those of J, on no other one of the k and on none of the
// r. Answer: a line "stratum m j_1 ... j_m chi" for each J where it is not zero.
// A marked coefficient field represents one embedding. Divide each chart's
// geometric contribution by its own degree before adding disjoint pieces.
std::string op_strata_totals(Reader& in) {
    std::istringstream rows(std::string(in.at, in.end));
    std::map<size_t, Z> sums;
    for (std::string row; std::getline(rows, row);) {
        std::istringstream fields(row);
        size_t key;
        std::string degree_text, count_text, extra;
        int sign = 1;
        if (!(fields >> key >> degree_text >> count_text >> sign) || (sign != -1 && sign != 1) || fields >> extra)
            throw std::invalid_argument("Expected a stratum, field degree and Euler characteristic.");
        Z degree, count;
        if (fmpz_set_str(&degree.v, degree_text.c_str(), 10) ||
            fmpz_set_str(&count.v, count_text.c_str(), 10) || fmpz_sgn(&degree.v) <= 0)
            throw std::invalid_argument("Invalid stratum contribution.");
        if (!fmpz_divisible(&count.v, &degree.v)) return "inconsistent\n";
        fmpz_divexact(&count.v, &count.v, &degree.v);
        auto& total = sums[key];
        if (sign == 1) fmpz_add(&total.v, &total.v, &count.v);
        else fmpz_sub(&total.v, &total.v, &count.v);
    }
    std::string out;
    for (const auto& [key, count] : sums) {
        if (fmpz_is_zero(&count.v)) continue;
        char* text = fmpz_get_str(nullptr, 10, &count.v);
        out += std::to_string(key) + " " + text + "\n";
        flint_free(text);
    }
    return out;
}

using DivisorIdeals = std::vector<std::vector<Poly>>;
struct SubsetOrder {
    bool operator()(const std::vector<size_t>& a, const std::vector<size_t>& b) const {
        return a.size() == b.size() ? a < b : a.size() < b.size();
    }
};
using StratumTotals = std::map<std::vector<size_t>, Z, SubsetOrder>;

std::vector<Poly> fibre_factors(const Ring&, const Poly&);

void count_fibre_piece(const Ring& R, Basis& basis, DivisorIdeals ideals,
                       DivisorIdeals removed, StratumTotals& totals) {
    if (basis.unit()) return;
    auto base = basis.elements();
    auto restrict = [&](DivisorIdeals& conditions) {
        for (auto& ideal : conditions) {
            for (auto& polynomial : ideal)
                polynomial = basis.remainder(std::move(polynomial));
            ideal = present(std::move(ideal));
        }
    };
    restrict(ideals);
    restrict(removed);
    restrict_to_base_monomials(R, base, ideals, removed);
    const size_t k = ideals.size();
    Euler euler(R);
    std::vector<std::vector<size_t>> alive = {{}};
    for (size_t size = 1; size <= k; ++size) {
        std::vector<std::vector<size_t>> found;
        std::vector<size_t> subset(size);
        for (size_t i = 0; i < size; ++i) subset[i] = i;
        for (;;) {
            bool faces = true;
            for (size_t drop = 0; drop < size && faces; ++drop) {
                std::vector<size_t> face;
                for (size_t i = 0; i < size; ++i)
                    if (i != drop) face.push_back(subset[i]);
                faces = std::find(alive.begin(), alive.end(), face) != alive.end();
            }
            if (faces) {
                std::vector<Poly> equations = base;
                for (size_t j : subset) equations = joined(std::move(equations), ideals[j]);
                if (!unit_basis(R, equations)) {
                    found.push_back(subset);
                    std::vector<std::vector<Poly>> others = removed;
                    for (size_t j = 0; j < k; ++j)
                        if (std::find(subset.begin(), subset.end(), j) == subset.end()) others.push_back(ideals[j]);
                    const auto chi = euler.characteristic(equations, others);
                    if (!fmpz_is_zero(&chi.v)) {
                        auto& value = totals[subset];
                        fmpz_add(&value.v, &value.v, &chi.v);
                    }
                }
            }
            size_t d = size;
            while (d > 0 && subset[d - 1] == k - size + (d - 1)) --d;
            if (d == 0) break;
            ++subset[d - 1];
            for (size_t i = d; i < size; ++i) subset[i] = subset[i - 1] + 1;
        }
        if (found.empty()) break;
        alive = std::move(found);
    }
}

std::string op_strata(Reader& in) {
    const Ring R = grevlex(in.number(0));
    const size_t b = in.count(1), k = in.count(2);
    std::vector<size_t> sizes(k);
    for (size_t i = 0; i < k; ++i) sizes[i] = in.count(3 + i);
    const size_t r = in.numbers.size() > 3 + k ? in.count(3 + k) : 0;
    std::vector<size_t> left(r);
    for (size_t i = 0; i < r; ++i) left[i] = in.count(4 + k + i);
    std::vector<Poly> base = present(polys(in.ideal(R, b)));
    std::vector<std::vector<Poly>> ideals, removed;
    for (size_t i = 0; i < k; ++i) ideals.push_back(present(polys(in.ideal(R, sizes[i]))));
    for (size_t i = 0; i < r; ++i) removed.push_back(present(polys(in.ideal(R, left[i]))));
    if (k == 0) return "none\n";
    std::string out;
    if (removed.empty() && monomial_orbit_strata(R, base, ideals, out))
        return out.empty() ? "none\n" : out;
    // A fan can try a shared orbit calculation before falling back to general ideal strata.
    if (in.op == "monomial_strata") return "undetermined\n";
    Basis basis(R, base);
    if (basis.unit()) return "none\n";
    const auto equations = basis.elements();
    std::vector<Poly> factors;
    if (equations.size() == 1 && equations[0].m[0] > 1)
        factors = fibre_factors(R, equations[0]);
    StratumTotals totals;
    if (factors.empty() || (factors.size() == 1 && same(factors[0], equations[0]))) {
        count_fibre_piece(R, basis, ideals, removed, totals);
    } else {
        // The first vanished factor names a unique piece of the owned fibre.
        // Include previous factors among the holes to discard all overlaps.
        for (const Poly& factor : factors) {
            Basis piece(R, {factor});
            count_fibre_piece(R, piece, ideals, removed, totals);
            removed.push_back({factor});
        }
    }
    for (const auto& [subset, chi] : totals)
        if (!fmpz_is_zero(&chi.v)) append_stratum(out, subset, chi);
    return out.empty() ? "none\n" : out;
}

// On affine space coordinate hyperplanes are transverse. With one additional
// hypersurface f=0, the only possible failure is where df vanishes on the
// tangent space of the intersection of all coordinate divisors through p.
// The vector fields x_j*d/dx_j for boundary axes, and d/dx_j for all other
// axes, span exactly this tangent space at p (even if an ordinary x_j=0).
std::optional<bool> coordinate_crossings_on_fibre(const Ring& R,
                                                 const std::vector<std::vector<Poly>>& divisors,
                                                 const std::vector<Poly>& fibre,
                                                 bool monomial_only = false) {
    const Poly* hypersurface = nullptr;
    std::vector<bool> boundary(R.n);
    for (const auto& ideal : divisors) {
        if (ideal.size() != 1) return std::nullopt;
        const auto& f = ideal.front();
        if (f.size() == 1 && f.m[0] == 1) {
            int j = 0;
            while (!Ring::field(f.m.data(), R.position(j))) ++j;
            if (boundary[j]) return std::nullopt;
            boundary[j] = true;
        } else {
            if (hypersurface) return std::nullopt;
            hypersurface = &f;
        }
    }
    if (!hypersurface || constant(*hypersurface)) return true;

    // A nonzero scalar modulo the monomial fibre ideal is a unit at every
    // geometric point of that fibre. Its support suffices over QQ. Apply
    // this to f itself or to a derivative tangent to all boundary divisors.
    const bool monomial_fibre = std::all_of(fibre.begin(), fibre.end(),
                                           [](const Poly& g) { return g.size() == 1; });
    auto unit_on_fibre = [&](const Poly& f) {
        if (constant(f)) return true;
        if (!monomial_fibre || f.empty() || f.m[(f.size()-1)*R.stride] != 0) return false;
        for (size_t i = 0; i+1 < f.size(); ++i)
            if (std::none_of(fibre.begin(), fibre.end(), [&](const Poly& g) {
                return R.divides(g.m.data(), f.m.data() + i*R.stride);
            })) return false;
        return true;
    };
    if (unit_on_fibre(*hypersurface)) return true;

    std::vector<Poly> bad = fibre;
    bad.push_back(*hypersurface);
    for (int j = 0; j < R.n; ++j) {
        Poly d = derivative(R, *hypersurface, j);
        if (d.empty()) continue;
        if (boundary[j]) {
            Poly x = one(R);
            variable(R, x.m.data(), j);
            d = multiply(R, x, d);
        }
        if (unit_on_fibre(d)) return true;
        bad.push_back(std::move(d));
    }
    if (monomial_only) return std::nullopt;
    try {
        return unit_basis(R, bad);
    } catch (const Limit&) {
        // The ordinary subset tests can still succeed on smaller ideals.
        return std::nullopt;
    }
}

// Work on each rational factor of a fibre equation. Its zero set is covered
// over the algebraic closure even if a rational factor splits geometrically.
std::vector<Poly> fibre_factors(const Ring& R, const Poly& equation) {
    struct Factorization {
        fmpz_mpoly_ctx_t ctx;
        fmpz_mpoly_t f;
        fmpz_mpoly_factor_t factors;
        explicit Factorization(int n) {
            fmpz_mpoly_ctx_init(ctx, n, ORD_DEGREVLEX);
            fmpz_mpoly_init(f, ctx);
            fmpz_mpoly_factor_init(factors, ctx);
        }
        ~Factorization() {
            fmpz_mpoly_factor_clear(factors, ctx);
            fmpz_mpoly_clear(f, ctx);
            fmpz_mpoly_ctx_clear(ctx);
        }
    } work(R.n);
    long powers[MAX_VARIABLES];
    ulong exponents[MAX_VARIABLES];
    for (size_t i = 0; i < equation.size(); ++i) {
        R.unpack(equation.m.data()+i*R.stride, powers);
        for (int j = 0; j < R.n; ++j) exponents[j] = powers[j];
        fmpz_mpoly_push_term_fmpz_ui(work.f, &equation.c[i].v, exponents, work.ctx);
    }
    fmpz_mpoly_sort_terms(work.f, work.ctx);
    if (!fmpz_mpoly_factor(work.factors, work.f, work.ctx))
        return {equation}; // The unsplit fibre is still an exact cover.
    std::vector<Poly> out;
    for (slong i = 0; i < fmpz_mpoly_factor_length(work.factors, work.ctx); ++i) {
        fmpz_mpoly_factor_get_base(work.f, work.factors, i, work.ctx);
        Poly f;
        for (slong j = 0; j < fmpz_mpoly_length(work.f, work.ctx); ++j) {
            fmpz_mpoly_get_term_exp_si(powers, work.f, j, work.ctx);
            fmpz_mpoly_get_term_coeff_fmpz(&f.c.emplace_back().v, work.f, j, work.ctx);
            f.m.resize(f.c.size()*R.stride);
            R.pack(f.m.data()+(f.c.size()-1)*R.stride, powers);
        }
        const auto identity = shifted(R.n, 0);
        out.push_back(convert(R, R, f, identity.data())); // sort for the kernel
    }
    return out;
}

// V(base) minus the union of the holes is a union of D(product), taking
// one generator per hole. Keep products reduced in the base ring.
bool outside_holes(const Ring& R, Basis& base, const std::vector<std::vector<Poly>>& removed,
                   const Poly& product, size_t i = 0) {
    if (product.empty() || base.unit()) return false;
    if (i == removed.size()) return !empty_off_basis(R, base, product);
    for (const Poly& g : removed[i]) {
        Poly unit = base.remainder(multiply(R, product, base.remainder(g)));
        if (outside_holes(R, base, removed, unit, i+1)) return true;
    }
    return false;
}

struct OwnedCrossings {
    const Ring& R;
    long codimension;
    std::vector<Poly> ambient, fibre;
    std::vector<std::vector<Poly>> divisors, removed;
    std::vector<bool> strict;

    bool regular(const std::vector<Poly>& piece, const std::vector<size_t>& subset) const {
        std::vector<Poly> equations = ambient, base = piece;
        for (size_t j : subset) {
            equations = joined(std::move(equations), divisors[j]);
            if (!strict[j]) base = joined(std::move(base), divisors[j]);
        }
        Basis boundary(R, base);
        if (boundary.unit()) return true;
        std::vector<Poly> locus = boundary.elements();
        for (size_t j : subset)
            if (strict[j])
                for (const Poly& g : divisors[j]) locus.push_back(boundary.remainder(g));
        Basis cut(R, present(locus));
        if (!outside_holes(R, cut, removed, one(R))) return true;
        const long rank = codimension + static_cast<long>(subset.size());
        Matrix matrix = jacobian(R, equations);
        if (rank < 1 || rank > R.n || static_cast<size_t>(rank) > matrix.rows) return false;
        // Differentiate first. Reducing the rows on the boundary before
        // adjoining the strict equation avoids intermediate high-degree bases.
        for (size_t i = 0; i < matrix.rows; ++i) reduce_row(boundary, matrix.row(i), matrix.columns);
        Minors minors(R, std::move(matrix), static_cast<int>(rank));
        do {
            for (Poly g : remainders(boundary, minors)) {
                if (constant(g)) return true;
                locus.push_back(std::move(g));
            }
        } while (minors.next());
        Basis bad(R, present(locus));
        return !outside_holes(R, bad, removed, one(R));
    }

    bool over(const std::vector<Poly>& piece, std::vector<size_t>& subset, size_t first = 0) const {
        for (size_t i = first; i < divisors.size(); ++i) {
            subset.push_back(i);
            if (!regular(piece, subset) || !over(piece, subset, i+1)) return false;
            subset.pop_back();
        }
        return true;
    }

    bool certify() const {
        if (!codimension && ambient.empty() && removed.empty()) {
            if (auto coordinate = coordinate_crossings_on_fibre(R, divisors, fibre)) return *coordinate;
        }
        std::vector<Poly> base = joined(ambient, fibre);
        std::vector<size_t> subset;
        if (fibre.empty()) return over(base, subset);
        const Poly& shortest = *std::min_element(fibre.begin(), fibre.end(),
            [](const Poly& a, const Poly& b) { return a.size() < b.size(); });
        for (const Poly& f : fibre_factors(R, shortest))
            if (!over(joined(base, {f}), subset)) return false;
        return true;
    }
};

// On one coordinate axis the common zeros of the rows are the roots of
// their univariate GCD. A zero GCD means the entire line. The degree cap
// bounds dense polynomial allocation for sparse, high-degree inputs.
bool coordinate_axis_gcd(const Ring& R, int axis, const std::vector<Poly>& rows, fmpz_poly_t gcd) {
    constexpr u64 max_dense_degree = 256;
    const int position = R.position(axis);
    CoordinatePolynomial restricted;
    fmpz_poly_zero(gcd);
    for (const Poly& row : rows) {
        fmpz_poly_zero(restricted.p);
        for (size_t i = 0; i < row.size(); ++i) {
            const u64* m = row.m.data() + i * R.stride;
            if (m[0] != Ring::field(m, position)) continue;
            if (m[0] > max_dense_degree) return false;
            fmpz_poly_set_coeff_fmpz(restricted.p, static_cast<slong>(m[0]), &row.c[i].v);
        }
        fmpz_poly_gcd(gcd, gcd, restricted.p);
        if (fmpz_poly_degree(gcd) == 0) break;
    }
    return true;
}

bool has_coordinate_axis_point(const Ring& R, int axis, const std::vector<Poly>& equations,
                               const std::vector<std::vector<Poly>>& removed) {
    CoordinatePolynomial roots;
    if (!coordinate_axis_gcd(R, axis, equations, roots.p) || fmpz_poly_degree(roots.p) == 0)
        return false;
    if (removed.empty()) return true;
    CoordinatePolynomial hole, common, quotient;
    if (fmpz_poly_degree(roots.p) > 1) {
        // Removing a common root must remove every copy, regardless of the
        // multiplicity with which it occurs in the equations or in a hole.
        fmpz_poly_derivative(hole.p, roots.p);
        fmpz_poly_gcd(common.p, roots.p, hole.p);
        if (fmpz_poly_degree(common.p) > 0) {
            fmpz_poly_divexact(quotient.p, roots.p, common.p);
            fmpz_poly_swap(roots.p, quotient.p);
        }
    }
    for (const auto& generators : removed) {
        if (!coordinate_axis_gcd(R, axis, generators, hole.p) || fmpz_poly_is_zero(hole.p))
            return false;
        // A finite union of proper closed subsets cannot cover the entire
        // line over Qbar. Otherwise retain only roots outside this hole.
        if (fmpz_poly_is_zero(roots.p) || fmpz_poly_degree(hole.p) == 0) continue;
        fmpz_poly_gcd(common.p, roots.p, hole.p);
        if (fmpz_poly_degree(common.p) == 0) continue;
        fmpz_poly_divexact(quotient.p, roots.p, common.p);
        fmpz_poly_swap(roots.p, quotient.p);
        if (fmpz_poly_degree(roots.p) == 0) return false;
    }
    return true;
}

std::string op_open_locus(Reader& in) {
    const size_t holes = in.count(2);
    // The general verifier eliminates redundant holes before expanding opens.
    // Defer large products to it, and bound the depth of this recursive test.
    if (holes > 16) return "unknown\n";
    size_t combinations = 1;
    for (size_t i = 0; i < holes; ++i) {
        const size_t count = in.count(3+i);
        if (count > 128/combinations) return "unknown\n";
        combinations *= std::max<size_t>(1, count);
    }
    const Ring R = grevlex(in.number(0));
    const auto equations = present(polys(in.ideal(R, in.count(1))));
    std::vector<std::vector<Poly>> removed;
    for (size_t i = 0; i < holes; ++i)
        removed.push_back(present(polys(in.ideal(R, in.count(3+i)))));
    if (std::any_of(equations.begin(), equations.end(), constant)) return "no\n";
    auto vanishes_at_origin = [&](const Poly& f) { return f.m[(f.size()-1) * R.stride] != 0; };
    bool origin = std::all_of(equations.begin(), equations.end(), vanishes_at_origin);
    for (const auto& hole : removed) {
        if (hole.empty()) return "no\n";
        if (origin && std::all_of(hole.begin(), hole.end(), vanishes_at_origin)) origin = false;
    }
    if (origin) return "yes\n";
    for (int axis = 0; axis < R.n; ++axis)
        if (has_coordinate_axis_point(R, axis, equations, removed)) return "yes\n";
    Basis closed(R, equations);
    return outside_holes(R, closed, removed, one(R)) ? "yes\n" : "no\n";
}

std::string op_owned_crossings(Reader& in) {
    const Ring R = grevlex(in.number(0));
    const long dimension = in.number(1);
    if (dimension < 0 || dimension > R.n) throw std::invalid_argument("Invalid chart dimension.");
    const size_t parts = in.count(3), holes = in.count(5);
    OwnedCrossings check{R, R.n-dimension, present(polys(in.ideal(R, in.count(2)))),
                         present(polys(in.ideal(R, in.count(4)))), {}, {}, {}};
    for (size_t i = 0; i < parts; ++i) {
        check.strict.push_back(in.count(6+2*i) != 0);
        check.divisors.push_back(present(polys(in.ideal(R, in.count(7+2*i)))));
    }
    for (size_t i = 0; i < holes; ++i)
        check.removed.push_back(present(polys(in.ideal(R, in.count(6+2*parts+i)))));
    return check.certify() ? "yes\n" : "no\n";
}

// "crossings n d a k s_1 ... s_k f r": a chart W = V(ambient) of dimension d in
// n variables (a generators), k divisors with s_i generators each, then f
// generators of the fibre and r of the region. For every set of divisors whose
// intersection is not smooth of the expected codimension at some point over
// the fibre, a line "bad m i_1 ... i_m inside count" and the count generators
// of the ideal of those points; inside tells whether one of them is in the
// region. At a point of k of the divisors their intersection is smooth of
// codimension k in W exactly when the Jacobian matrix of the equations of W
// and of the divisors has rank n - d + k there; the matrix is reduced modulo
// the ideal of the points over the fibre. A set is examined only when all the
// sets with one divisor fewer meet over the fibre.
std::string op_crossings(Reader& in) {
    const Ring R = grevlex(in.number(0));
    const long dimension = in.number(1);
    const size_t a = in.count(2), k = in.count(3);
    std::vector<size_t> sizes(k);
    for (size_t i = 0; i < k; ++i) sizes[i] = in.count(4 + i);
    const size_t f = in.count(4 + k), r = in.count(5 + k);
    const std::vector<Poly> ambient = present(polys(in.ideal(R, a)));
    std::vector<std::vector<Poly>> divisors;
    for (size_t i = 0; i < k; ++i) divisors.push_back(present(polys(in.ideal(R, sizes[i]))));
    const std::vector<Poly> fibre = present(polys(in.ideal(R, f)));
    const std::vector<Poly> region = present(polys(in.ideal(R, r)));
    const long codimension = R.n - dimension;
    if (in.op == "bounded_crossings") {
        if (codimension || !ambient.empty() || !region.empty()) return "unknown\n";
        return coordinate_crossings_on_fibre(R, divisors, fibre, true).value_or(false)
            ? "none\n" : "unknown\n";
    }
    if (!codimension && ambient.empty() && coordinate_crossings_on_fibre(R, divisors, fibre).value_or(false))
        return "none\n";
    std::vector<std::vector<size_t>> alive = {{}};
    std::string out;
    for (size_t size = 1; size <= k; ++size) {
        std::vector<std::vector<size_t>> found;
        std::vector<size_t> subset(size);
        for (size_t i = 0; i < size; ++i) subset[i] = i;
        for (;;) {
            bool faces = true;
            for (size_t drop = 0; drop < size && faces; ++drop) {
                std::vector<size_t> face;
                for (size_t i = 0; i < size; ++i)
                    if (i != drop) face.push_back(subset[i]);
                faces = std::find(alive.begin(), alive.end(), face) != alive.end();
            }
            if (faces) {
                std::vector<Poly> gens = ambient;
                for (size_t j : subset) gens = joined(std::move(gens), divisors[j]);
                std::vector<Poly> locus = joined(gens, fibre);
                if (!unit_basis(R, locus)) {
                    found.push_back(subset);
                    const long rank = codimension + static_cast<long>(size);
                    bool failing = true;
                    if (rank <= R.n) {
                        Basis basis(R, locus);
                        Matrix J = jacobian(R, gens);
                        for (size_t i = 0; i < J.rows; ++i) reduce_row(basis, J.row(i), J.columns);
                        if (static_cast<long>(J.rows) >= rank && rank >= 1) {
                            Minors minors(R, std::move(J), static_cast<int>(rank));
                            do {
                                for (Poly& m : remainders(basis, minors)) locus.push_back(std::move(m));
                            } while (minors.next());
                        }
                        failing = !unit_basis(R, locus);
                    }
                    if (failing) {
                        Basis bad(R, locus);
                        const Ideal generators = bad.monic();
                        const bool inside = !unit_basis(R, joined(locus, region));
                        out += "bad " + std::to_string(size);
                        for (size_t j : subset) out += " " + std::to_string(j);
                        out += std::string(inside ? " 1 " : " 0 ") + std::to_string(generators.size()) + "\n";
                        put(R, out, generators);
                    }
                }
            }
            size_t d = size;
            while (d > 0 && subset[d - 1] == k - size + (d - 1)) --d;
            if (d == 0) break;
            ++subset[d - 1];
            for (size_t i = d; i < size; ++i) subset[i] = subset[i - 1] + 1;
        }
        if (found.empty()) break;
        alive = std::move(found);
    }
    return out.empty() ? "none\n" : out;
}


// ---------------------------------------------------------------------------
// The blowup of a centre x_i = phi_i (i = 1..c), with phi_i polynomials in the
// parameters of the chart (the variables of its ambient equations), written
// out. In the chart of the j-th coordinate, x_j = phi_j + w and
// x_i = phi_i + y_i*w for the others; w and the y_i take the places of the
// x_j and x_i among the variables. A function becomes a polynomial in the new
// coordinates, reduced modulo the ambient equations; its order along the new
// divisor w = 0 is the least exponent of w.

// f with x_v replaced by image/den, times den^(degree of f in x_v): Horner's
// rule on the coefficients of the powers of x_v. `scale`, when given, is
// multiplied by that power of den.
Poly substitute(const Ring& R, const Poly& f, int v, const Poly& image, const fmpz* den, fmpz* scale) {
    if (f.empty()) return f;
    const int S = R.stride, p = R.position(v);
    const int word = 1 + (p >> 2), shift = 48 - 16 * (p & 3);
    std::map<long, Poly> parts;
    long degree = 0;
    for (size_t i = 0; i < f.size(); ++i) {
        const u64* mono = f.m.data() + i * S;
        const long e = static_cast<long>((mono[word] >> shift) & 0xFFFF);
        Poly& part = parts[e];
        const size_t start = part.m.size();
        part.m.insert(part.m.end(), mono, mono + S);
        part.m[start] -= static_cast<u64>(e);
        part.m[start + word] -= static_cast<u64>(e) << shift;
        part.c.emplace_back();
        fmpz_set(&part.c.back().v, &f.c[i].v);
        degree = std::max(degree, e);
    }
    if (degree == 0) return f;
    const bool plain = fmpz_is_one(den);
    Poly sum = parts[degree];
    Z power;
    fmpz_one(&power.v);
    for (long d = degree - 1; d >= 0; --d) {
        sum = multiply(R, sum, image);
        if (!plain) fmpz_mul(&power.v, &power.v, den);
        const auto part = parts.find(d);
        if (part == parts.end()) continue;
        Poly next;
        merge(R, next, sum, 0, nullptr, nullptr, part->second, 0, plain ? nullptr : &power.v, nullptr);
        sum.swap(next);
    }
    if (scale && !plain) fmpz_mul(scale, scale, &power.v);
    return sum;
}

// A completed Buchberger pass with no unit certifies a geometric point.
// At the work limit leave the decision to the usual chart computation.
class IncidenceSliceBasis : public Engine {
public:
    bool has_point(std::vector<Poly>& equations) {
        bit_limit_ = 2048;
        work_limit_ = 150000;
        try {
            attempt(equations, 0);
            return !unit;
        } catch (const Swell&) {
            return false;
        }
    }
};

// Over D(p) the projection from the p-chart is an isomorphism, and the
// strict divisor there is its original divisor. Try coordinate hyperplanes
// through one parent centre point before forming a basis in the graph ring.
std::string op_graph_parent_slice(Reader& in) {
    const Ring R = grevlex(in.number(0));
    if (R.n > 6 || in.count(1) > 32) return "unknown\n";
    const auto rows = polys(in.ideal(R, in.count(1)));
    const Poly pivot = in.gen(R).p;
    const Ideal point = in.ideal(R, R.n);
    size_t terms = pivot.size();
    for (const Poly& row : rows) {
        terms += row.size();
        if (!row.empty() && row.m[0] > 128) return "unknown\n";
    }
    if (pivot.empty() || pivot.m[0] > 128 || terms > 4096) return "unknown\n";
    const Ring extended = grevlex(R.n + 1);
    const auto places = shifted(R.n, 1);
    for (int axis = 0; axis < R.n; ++axis) {
        const Gen& value = point[axis];
        if (!value.p.empty() && !constant(value.p))
            throw std::invalid_argument("Expected a rational centre coordinate.");
        if (fmpz_bits(&value.num.v) + fmpz_bits(&value.den.v) > 128) continue;
        Poly coordinate = value.p;
        if (!coordinate.empty()) fmpz_mul(&coordinate.c.front().v, &coordinate.c.front().v, &value.num.v);
        Poly unit = substitute(R, pivot, axis, coordinate, &value.den.v, nullptr);
        if (unit.empty()) continue;
        std::vector<Poly> equations;
        if (!constant(unit)) equations.push_back(inverse_relation(R, extended, unit));
        bool empty = false;
        for (const Poly& row : rows) {
            Poly section = substitute(R, row, axis, coordinate, &value.den.v, nullptr);
            if (constant(section)) { empty = true; break; }
            if (!section.empty()) equations.push_back(convert(R, extended, section, places.data()));
        }
        if (empty) continue;
        IncidenceSliceBasis basis;
        basis.R = extended;
        if (basis.has_point(equations)) return "yes\n";
    }
    return "unknown\n";
}

// den*x_v*(x_w or 1) + num*shift: the numerator of the image of x_v.
Poly image_of(const Ring& R, int v, int w, const Gen& shift) {
    Poly term;
    u64 mono[MAX_STRIDE], other[MAX_STRIDE];
    variable(R, mono, v);
    if (w >= 0) {
        variable(R, other, w);
        R.multiply(mono, mono, other);
    }
    term.m.assign(mono, mono + R.stride);
    term.c.emplace_back();
    fmpz_set(&term.c.back().v, &shift.den.v);
    if (shift.p.empty()) return term;
    Poly out;
    merge(R, out, term, 0, nullptr, nullptr, shift.p, 0, &shift.num.v, nullptr);
    return out;
}

long order_along(const Ring& R, const Poly& f, int v) {
    const int S = R.stride, p = R.position(v);
    const int word = 1 + (p >> 2), shift = 48 - 16 * (p & 3);
    long least = -1;
    for (size_t i = 0; i < f.size(); ++i) {
        const long e = static_cast<long>((f.m[i * S + word] >> shift) & 0xFFFF);
        if (least < 0 || e < least) least = e;
    }
    return least < 0 ? 0 : least;
}

void lower(const Ring& R, Poly& f, int v, long by) {
    if (!by) return;
    const int S = R.stride, p = R.position(v);
    const int word = 1 + (p >> 2), shift = 48 - 16 * (p & 3);
    for (size_t i = 0; i < f.size(); ++i) {
        f.m[i * S] -= static_cast<u64>(by);
        f.m[i * S + word] -= static_cast<u64>(by) << shift;
    }
}

// "blowup n a c p_1 ... p_c k s_1 ... s_k f r m": a chart in n variables with
// a ambient equations; the centre x_{p_i} = phi_i, the phi_i following the
// ambient equations; k divisors with s_i generators each; f generators of the
// fibre, r of the region, and m functions to carry along exactly (the
// coordinates of the first chart). For each j = 1..c the answer has a line
// "chart j", then for each divisor "divisor order present count" and the
// count generators of its transform (present is 0 when it misses the chart),
// then "fibre count", "region count" and "functions count" with theirs.
std::string op_blowup(Reader& in) {
    const Ring R = grevlex(in.number(0));
    if (R.n + 1 > MAX_VARIABLES) throw Limit("Unsupported number of variables.");
    const size_t a = in.count(1), c = in.count(2);
    std::vector<int> places(c);
    for (size_t i = 0; i < c; ++i) {
        const long p = in.number(3 + i);
        if (p < 0 || p >= R.n) throw std::invalid_argument("Bad coordinate.");
        places[i] = static_cast<int>(p);
    }
    const size_t k = in.count(3 + c);
    std::vector<size_t> sizes(k);
    for (size_t i = 0; i < k; ++i) sizes[i] = in.count(4 + c + i);
    const size_t f = in.count(4 + c + k), r = in.count(5 + c + k), m = in.count(6 + c + k);
    const std::vector<Poly> ambient = present(polys(in.ideal(R, a)));
    const Ideal shifts = in.ideal(R, c);
    std::vector<Ideal> divisors;
    for (size_t i = 0; i < k; ++i) divisors.push_back(in.ideal(R, sizes[i]));
    const Ideal fibre = in.ideal(R, f), region = in.ideal(R, r), functions = in.ideal(R, m);
    std::unique_ptr<Basis> reducer;
    if (!ambient.empty()) reducer.reset(new Basis(R, ambient));
    Quotient quotient(R);
    std::string out;
    for (size_t j = 0; j < c; ++j) {
        const int w = places[j];
        std::vector<Poly> images(c);
        images[j] = image_of(R, w, -1, shifts[j]);
        for (size_t i = 0; i < c; ++i)
            if (i != j) images[i] = image_of(R, places[i], w, shifts[i]);
        // The pivot first: afterwards x_w stands for the new coordinate w.
        auto moved = [&](const Gen& g, bool exact) {
            Z num, den;
            fmpz_set(&num.v, &g.num.v);
            fmpz_set(&den.v, &g.den.v);
            Poly p = substitute(R, g.p, w, images[j], &shifts[j].den.v, &den.v);
            for (size_t i = 0; i < c; ++i)
                if (i != j) p = substitute(R, p, places[i], images[i], &shifts[i].den.v, &den.v);
            if (reducer && !p.empty()) {
                if (exact) {
                    Z rnum, rden;
                    fmpz_one(&rnum.v);
                    fmpz_one(&rden.v);
                    reducer->engine.reduce(p, 0, &rnum.v, &rden.v);
                    fmpz_mul(&num.v, &num.v, &rnum.v);
                    fmpz_mul(&den.v, &den.v, &rden.v);
                } else {
                    p = reducer->remainder(std::move(p));
                }
            }
            return normal(std::move(p), &num.v, &den.v);
        };
        auto ideal = [&](const Ideal& gens, bool exact) {
            Ideal result;
            for (const Gen& g : gens) {
                Gen image = moved(g, exact);
                if (!image.p.empty() || exact) result.push_back(std::move(image));
            }
            return result;
        };
        out += "chart " + std::to_string(j + 1) + "\n";
        for (size_t i = 0; i < k; ++i) {
            std::vector<Poly> gens = polys(ideal(divisors[i], false));
            if (gens.empty()) throw std::invalid_argument("A divisor contains a chart of a blowup.");
            long order = -1;
            for (const Poly& g : gens) {
                const long e = order_along(R, g, w);
                if (order < 0 || e < order) order = e;
            }
            if (order > 0 && gens.size() > 1) {
                Poly power = one(R);
                u64 mono[MAX_STRIDE];
                variable(R, mono, w);
                for (long e = 0; e < order; ++e) R.multiply(power.m.data(), power.m.data(), mono);
                std::vector<Poly> by;
                by.push_back(std::move(power));
                std::vector<Poly> after;
                for (Poly& g : Basis(R, present(quotient.colon(joined(ambient, gens), by))).elements())
                    if (!reducer || !reducer->contains(g)) after.push_back(std::move(g));
                gens = std::move(after);
            } else {
                for (Poly& g : gens) lower(R, g, w, order);
            }
            const bool here = !unit_basis(R, joined(ambient, gens));
            out += "divisor " + std::to_string(order) + (here ? " 1 " : " 0 ") + std::to_string(gens.size()) + "\n";
            put(R, out, monic(gens));
        }
        const Ideal over = ideal(fibre, false), part = ideal(region, false), carried = ideal(functions, true);
        out += "fibre " + std::to_string(over.size()) + "\n";
        put(R, out, over);
        out += "region " + std::to_string(part.size()) + "\n";
        put(R, out, part);
        out += "functions " + std::to_string(carried.size()) + "\n";
        put(R, out, carried);
    }
    return out;
}

std::string graph_strict_parts(const Ring& R, const std::vector<Poly>& ambient,
                              Basis& chart, Basis& exceptional, const Gen& pivot,
                              const Ideal& functions) {
    // Graph charts are saturated by p. In their coordinate ring p
    // is regular, so the residual of a principal (f) stays
    // principal. Use normal forms in A[t]/(t*p-1) to divide.
    Ring E;
    E.init(R.n + 1, 2);
    auto up = shifted(R.n, 1);
    auto down = shifted(E.n, -1);
    std::unique_ptr<Basis> inverse;
    std::string out;
    for (Gen f : functions) {
        if (chart.contains(f.p))
            throw std::invalid_argument("The function vanishes identically on the chart.");
        size_t order = 0;
        while (exceptional.contains(f.p)) {
            if (!inverse) {
                std::vector<Poly> lifted;
                for (const Poly& a : ambient) lifted.push_back(convert(R, E, a, up.data()));
                lifted.push_back(inverse_relation(R, E, pivot.p));
                inverse = std::make_unique<Basis>(E, lifted);
            }
            // The inverse relation uses the primitive polynomial.
            // Its rational content is accounted for in f's scale.
            Poly tf = convert(R, E, f.p, up.data());
            u64 t[MAX_STRIDE];
            variable(E, t, 0);
            for (size_t j = 0; j < tf.size(); ++j) {
                u64* monomial = tf.m.data() + j*E.stride;
                E.multiply(monomial, monomial, t);
            }
            Fraction q = residue(*inverse, std::move(tf));
            if (q.p.empty() || Ring::field(q.p.m.data(), E.position(0)))
                throw std::logic_error("The exceptional quotient is not a chart polynomial.");
            f.p = convert(E, R, q.p, down.data());
            fmpz_mul(&f.num.v, &f.num.v, &q.num.v);
            fmpz_mul(&f.num.v, &f.num.v, &pivot.den.v);
            fmpz_mul(&f.den.v, &f.den.v, &q.den.v);
            fmpz_mul(&f.den.v, &f.den.v, &pivot.num.v);
            ++order;
        }
        out += std::to_string(order) + " 1\n";
        put(R, out, {normal(std::move(f.p), &f.num.v, &f.den.v)});
    }
    return out;
}

// Remove the exceptional part from the divisor of a nonzero function.
// On a smooth chart the ideal of each Cartier divisor is invertible.
// If the residual ideal is contained in that ideal, quotienting by it
// subtracts one copy of the divisor. Repeat until its order is zero.
std::string op_strict_parts(Reader& in) {
    const Ring R = grevlex(in.number(0));
    const auto ambient = polys(in.ideal(R, in.count(1)));
    const size_t count = in.count(2);
    Basis chart(R, ambient);
    std::vector<std::vector<Poly>> divisors;
    std::vector<std::unique_ptr<Basis>> tests;
    for (size_t i = 0; i < count; ++i) {
        const Ideal equations = in.ideal(R, in.count(3 + i));
        divisors.push_back(polys(equations));
        auto test = std::make_unique<Basis>(R, joined(ambient, divisors.back()));
        if (test->unit() || std::all_of(divisors.back().begin(), divisors.back().end(),
                                       [&](const Poly& f) { return chart.contains(f); }))
            throw std::invalid_argument("Expected a nonempty divisor of positive codimension.");
        if (in.op == "graph_strict_part") {
            if (count != 1 || equations.size() != 1)
                throw std::invalid_argument("Expected one principal exceptional divisor.");
            return graph_strict_parts(R, ambient, chart, *test, equations.front(), in.rest(R));
        }
        tests.push_back(std::move(test));
    }
    Quotient quotient(R);
    std::string out;
    for (const Gen& f : in.rest(R)) {
        if (chart.contains(f.p))
            throw std::invalid_argument("A factor vanishes identically on a chart component.");
        std::vector<Poly> residual = joined(ambient, {f.p});
        for (size_t i = 0; i < count; ++i) {
            int order = 0;
            while (std::all_of(residual.begin(), residual.end(),
                               [&](const Poly& g) { return tests[i]->contains(g); })) {
                Basis before(R, residual);
                residual = Basis(R, quotient.colon(residual, divisors[i])).elements();
                if (std::all_of(residual.begin(), residual.end(),
                                [&](const Poly& g) { return before.contains(g); }))
                    throw std::invalid_argument("A factor has no finite order on a divisor.");
                ++order;
            }
            out += std::to_string(order) + " ";
        }
        const Ideal strict = Basis(R, residual).monic();
        out += std::to_string(strict.size()) + "\n";
        put(R, out, strict);
    }
    return out;
}

// Count the owned irreducible components when setting coordinate variables to
// zero leaves one equation in two free coordinates. Other presentations defer.
std::string op_plane_components(Reader& in) {
    const Ring R = grevlex(in.number(0));
    const auto equations = polys(in.ideal(R, in.count(1)));
    auto region = polys(in.ideal(R, in.count(2)));
    std::vector<std::vector<Poly>> removed;
    for (size_t i = 3; i < in.numbers.size(); ++i)
        removed.push_back(polys(in.ideal(R, in.count(i))));
    std::vector<bool> zero(R.n, false);
    long powers[MAX_VARIABLES];
    for (const Poly& f : equations) {
        if (f.size() == 1 && f.m[0] == 1) {
            R.unpack(f.m.data(), powers);
            zero[std::find(powers, powers + R.n, 1L) - powers] = true;
        }
    }
    std::vector<int> free;
    for (int i = 0; i < R.n; ++i) if (!zero[i]) free.push_back(i);
    if (free.size() != 2) return "unsupported\n";
    auto restrict = [&](const Poly& f) {
        Poly out;
        for (size_t j = 0; j < f.size(); ++j) {
            const u64* mon = f.m.data() + j * R.stride;
            R.unpack(mon, powers);
            bool vanishes = false;
            for (int k = 0; k < R.n; ++k) if (zero[k] && powers[k]) vanishes = true;
            if (!vanishes) {
                out.m.insert(out.m.end(), mon, mon + R.stride);
                out.c.push_back(f.c[j]);
            }
        }
        return out;
    };
    Poly equation;
    for (const Poly& g : equations) {
        Poly f = restrict(g);
        if (f.empty()) continue;
        if (!equation.empty()) return "unsupported\n";
        equation = std::move(f);
    }
    if (equation.empty()) return "unsupported\n";
    for (Poly& g : region) g = restrict(g);
    for (auto& z : removed) for (Poly& g : z) g = restrict(g);
    const auto counts = Euler(R).plane_components(equation, free[0], free[1], region, removed);
    return std::to_string(counts.first) + " " + std::to_string(counts.second) + "\n";
}

std::string op_graph_cover(Reader& in) {
    const auto r = grevlex(in.number(0));
    const auto ambient = in.ideal(r, in.count(1));
    const auto pivot = in.gen(r);
    std::vector<Ideal> pieces;
    for (size_t i=0; i<in.count(2); ++i) pieces.push_back(in.ideal(r,in.count(3+i)));
    return cover(r,ambient,pivot,pieces) ? "yes\n" : "no\n";
}

// Orders of the equation and of the relative Jacobian on a smooth graph
// chart. The first d coordinates are the coordinates of the projection.
std::string op_graph_orders(Reader& in) {
    const Ring R = grevlex(in.number(0));
    const int d = static_cast<int>(in.number(1));
    if (d < 1 || d > R.n) throw std::invalid_argument("Bad projection dimension.");
    const auto ambient = polys(in.ideal(R, in.count(2)));
    const auto prime = polys(in.ideal(R, in.count(3)));
    const auto equation = in.gen(R).p;
    Basis chart(R, ambient);
    if (chart.contains(equation)) throw std::invalid_argument("Zero pullback on the chart.");
    auto divisor = joined(ambient, prime);
    Basis first(R, divisor);
    if (first.unit()) throw std::invalid_argument("Empty divisor.");
    std::vector<Poly> generators;
    for (const Poly& g : prime)
        if (!chart.contains(g)) generators.push_back(g);
    if (generators.empty()) throw std::invalid_argument("Missing divisor ideal.");

    const int extra = R.n - d;
    std::vector<Poly> jac;
    if (extra == 0) {
        jac.push_back(one(R));
    } else {
        Matrix relative;
        relative.rows = ambient.size();
        relative.columns = extra;
        if (relative.rows < static_cast<size_t>(extra)) throw std::invalid_argument("Missing graph equations.");
        for (const Poly& g : ambient)
            for (int j = d; j < R.n; ++j)
                relative.entries.push_back(derivative(R, g, j));
        Minors minors(R, std::move(relative), extra);
        do {
            for (const auto& pair : minors.current())
                if (!chart.contains(pair.second)) jac.push_back(pair.second);
        } while (minors.next());
        if (jac.empty()) throw std::invalid_argument("The graph projection is not generically etale.");
    }
    int order = -1, discrepancy = -1;
    auto power = generators;
    for (int k = 1; order < 0 || discrepancy < 0; ++k) {
        Basis test(R, joined(ambient, power));
        if (order < 0 && !test.contains(equation)) order = k - 1;
        if (discrepancy < 0 && std::any_of(jac.begin(), jac.end(), [&](const Poly& h) {
                return !test.contains(h);
            })) discrepancy = k - 1;
        if (order < 0 || discrepancy < 0) {
            power.clear();
            for (const Poly& h : test.elements())
                if (!chart.contains(h))
                    for (const Poly& g : generators) power.push_back(multiply(R, g, h));
        }
    }
    return std::to_string(order) + " " + std::to_string(discrepancy + 1) + "\n";
}

}  // namespace
// Work with a frame on the open set where every specified unit is nonzero.
static std::string frame_locus_grevlex(const Ring& R, const std::vector<Poly>& generators,
                                       const std::vector<Poly>& units, bool closure) {
    Basis basis(R, generators);
    Poly unit = one(R);
    for (const Poly& u : units) {
        if (unit.empty() || basis.unit()) break;
        Poly factor = basis.remainder(u);
        if (constant(factor)) continue;
        if (factor.empty() || constant(unit)) unit = std::move(factor);
        else unit = basis.remainder(multiply(R, unit, factor));
    }
    if (!closure) return empty_off_basis(R, basis, unit) ? "1\n" : "0\n";
    std::string out;
    if (unit.empty() || basis.unit()) put(R, out, unit_ideal(R));
    else if (constant(unit) || generators.empty()) put(R, out, basis.monic());
    else put(R, out, monic(saturation(R, basis.elements(), unit)));
    return out;
}

std::string frame_locus(const Ring& input_ring, const std::vector<Poly>& equations,
                        const std::vector<Poly>& units, bool closure) {
    const Ring R = grevlex(input_ring.n);
    const auto positions = shifted(R.n, 0);
    auto convert_rows = [&](const std::vector<Poly>& rows) {
        std::vector<Poly> result;
        result.reserve(rows.size());
        for (const Poly& f : rows) result.push_back(convert(input_ring, R, f, positions.data()));
        return result;
    };
    return frame_locus_grevlex(R, present(convert_rows(equations)), convert_rows(units), closure);
}

// Expanded affine-frame inputs are already in this ring and term order.
static std::string op_frame_locus(Reader& in) {
    const Ring R = grevlex(in.number(0));
    const auto equations = present(polys(in.ideal(R, in.count(1))));
    const auto units = polys(in.ideal(R, in.count(2)));
    return frame_locus_grevlex(R, equations, units, in.op == "frame_basis");
}

// Sufficient membership tests on the open frame. The frame's own equations
// vanish on every restriction. A nonzero rational scalar vanishes exactly
// when the frame is empty; only a certified nonempty frame rules that out.
static std::string op_frame_membership(Reader& in) {
    const size_t width = in.count(0), count = in.count(1);
    const long nonempty = in.number(2);
    if (nonempty != 0 && nonempty != 1) throw std::invalid_argument("Invalid nonempty flag.");
    std::vector<std::string_view> equations;
    for (size_t i = 0; i < count; ++i) equations.push_back(in.row_text());
    const std::string_view query = in.row_text();
    if (in.at != in.end) throw std::invalid_argument("Extra frame membership rows.");

    // Inputs are expanded sparse rational rows. Most queries have a distinct
    // number of terms, so only decode possible matches, or a possible scalar.
    // The unpacked reader preserves arbitrary exponents and coefficients.
    const auto separators = std::count(query.begin(), query.end(), ';');
    std::optional<local::Polynomial> polynomial;
    for (const auto equation : equations) {
        if (std::count(equation.begin(), equation.end(), ';') != separators) continue;
        if (!polynomial) polynomial = local::decode_row(query, width);
        if (query == equation || *polynomial == local::decode_row(equation, width))
            return "yes\n";
    }
    if (separators != 0) return "unknown\n";
    if (!polynomial) polynomial = local::decode_row(query, width);
    if (polynomial->empty()) return "yes\n";
    const auto& powers = polynomial->begin()->first;
    if (std::all_of(powers.begin(), powers.end(), [](const auto& e) { return e == 0; }))
        return nonempty ? "no\n" : "scalar\n";
    return "unknown\n";
}
}  // namespace atlas

using namespace atlas;

extern "C" {

int atlas_monomial_strata_abi() noexcept { return 1; }
int atlas_frame_locus_abi() noexcept { return 1; }
int atlas_affine_overlap_abi() noexcept { return 1; }
int atlas_smooth_reduced_centre_abi() noexcept { return 1; }

// One request: a header line naming the operation with its integers, then
// one polynomial per line as read by atlas_groebner. The operations are
// described above each op_ function. Returns the answer, to be freed with
// atlas_groebner_release, or null with the error set.
char* atlas_ideal(const char* input, size_t length) noexcept {
    try {
        Reader in(input, length);
        std::string out;
        if (in.op == "strict_parts" || in.op == "graph_strict_part") out = op_strict_parts(in);
        else if (in.op == "residual_dimension_below" || in.op == "certify_exceptional_orders")
            out = op_residual_dimension_below(in);
        else if (in.op == "plane_components") out = op_plane_components(in);
        else if (in.op == "graph_cover") out = op_graph_cover(in);
        else if (in.op == "graph_orders") out = op_graph_orders(in);
        else if (in.op == "dimension") out = op_dimension(in);
        else if (in.op == "saturate") out = op_saturate(in);
        else if (in.op == "saturation_regular") out = saturation_regular(in) ? "yes\n" : "unknown\n";
        else if (in.op == "smooth_reduced_centre" || in.op == "smooth_reduced_graph") out = op_smooth_reduced_centre(in);
        else if (in.op == "affine_overlap") out = op_affine_overlap(in);
        else if (in.op == "finite_radical") out = op_finite_radical(in);
        else if (in.op == "radical") out = op_radical(in);
        else if (in.op == "smooth") out = op_smooth(in);
        else if (in.op == "snc") out = op_snc(in);
        else if (in.op == "charts") out = op_charts(in);
        else if (in.op == "verify") out = op_verify(in);
        else if (in.op == "surface") out = op_surface(in);
        else if (in.op == "strata" || in.op == "monomial_strata") out = op_strata(in);
        else if (in.op == "strata_totals") out = op_strata_totals(in);
        else if (in.op == "crossings" || in.op == "bounded_crossings") out = op_crossings(in);
        else if (in.op == "frame_basis" || in.op == "frame_empty") out = op_frame_locus(in);
        else if (in.op == "frame_membership") out = op_frame_membership(in);
        else if (in.op == "graph_regular") {
            std::istringstream rows(std::string(in.at, in.end));
            out = graph_regular(rows, in.count(0), in.count(1)) ? "yes\n" : "unknown\n";
        }
        else if (in.op == "open_locus") out = op_open_locus(in);
        else if (in.op == "graph_centre_points" || in.op == "graph_centre_incidence") {
            std::istringstream rows(std::string(in.at, in.end));
            if (in.op == "graph_centre_points") out = graph_centre_points(rows, in.count(0));
            else out = graph_centre_incidence(rows, in.count(0), in.count(1), in.count(2), in.count(3));
        }
        else if (in.op == "graph_parent_slice") out = op_graph_parent_slice(in);
        else if (in.op == "owned_crossings") out = op_owned_crossings(in);
        else if (in.op == "blowup") out = op_blowup(in);
        else throw std::invalid_argument("Unknown operation.");
        return copy_text(out);
    } catch (const std::exception& error) {
        set_error(error.what());
    } catch (...) {
        set_error("Unknown failure.");
    }
    return nullptr;
}

// Input: a header line "n s", then s generators of I and, after them, the
// generators of J. Returns the reduced grevlex basis of I : J, one polynomial
// per line; the caller frees it with atlas_groebner_release.
char* atlas_groebner_colon(const char* input, size_t length) noexcept {
    try {
        const char* at = input;
        const char* end = input + length;
        char* stop = nullptr;
        const long n = std::strtol(at, &stop, 10);
        const long first = std::strtol(stop, &stop, 10);
        at = stop;
        if (at >= end || *at != '\n' || first < 0) throw std::invalid_argument("Bad header.");
        ++at;
        if (n < 1 || n + 1 > MAX_VARIABLES) throw Limit("Unsupported number of variables.");
        Ring R;
        R.init(static_cast<int>(n), 0);
        std::vector<Poly> ideal, by;
        for (long k = 0; at < end; ++k) (k < first ? ideal : by).push_back(parse(R, at, end, nullptr));
        Quotient quotient(R);
        Basis reduced(R, quotient.colon(ideal, by));
        std::string out;
        put(R, out, reduced.monic());
        return copy_text(out);
    } catch (const std::exception& error) {
        set_error(error.what());
    } catch (...) {
        set_error("Unknown failure.");
    }
    return nullptr;
}

}

extern "C" int atlas_open_locus_abi() noexcept { return 1; }
extern "C" int atlas_exceptional_orders_abi() noexcept { return 1; }
extern "C" int atlas_graph_centre_incidence_abi() noexcept { return 1; }
extern "C" int atlas_bounded_crossings_abi() noexcept { return 1; }
