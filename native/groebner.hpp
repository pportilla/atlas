// Shared declarations of the exact kernel: packed monomials, integer
// polynomials and the Buchberger engine.
#ifndef ATLAS_GROEBNER_HPP
#define ATLAS_GROEBNER_HPP

#include <flint/fmpz.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace atlas {

using u64 = std::uint64_t;
constexpr u64 HI = 0x8000800080008000ULL;
constexpr int MAX_VARIABLES = 32;
constexpr int MAX_STRIDE = MAX_VARIABLES / 4 + 1;
constexpr u64 MAX_DEGREE = 32767;

struct Limit : std::runtime_error {
    using std::runtime_error::runtime_error;
};
// The current selection strategy is swelling; another one should try.
struct Swell {};

// An owned FLINT integer. Small values live in the word itself.
struct Z {
    fmpz v;
    Z() noexcept : v(0) {}
    Z(const Z& other) : v(0) { fmpz_set(&v, &other.v); }
    Z(Z&& other) noexcept : v(other.v) { other.v = 0; }
    Z& operator=(Z other) noexcept { std::swap(v, other.v); return *this; }
    ~Z() { fmpz_clear(&v); }
};

// Monomials are packed four exponents to a word; the top bit of each 16-bit
// field stays clear, which makes divisibility and lcm word operations. The
// first word of a monomial is its total degree.
struct Ring {
    int n = 0, words = 0, stride = 0, bits = 1;
    bool lex = false;
    bool eliminate = false;  // compare the first variable before grevlex
    bool graded = false;     // compare the total degree before anything else

    // order: 0 grevlex, 1 lex, 2 the first variable and then grevlex. With
    // `by_degree` the total degree is compared first; on homogeneous
    // polynomials in one more, last, variable this is the order induced by
    // the given one on their dehomogenizations.
    void init(int variables, int order, bool by_degree = false) {
        if (variables < 1 || variables > MAX_VARIABLES) throw Limit("Unsupported number of variables.");
        n = variables;
        lex = order == 1;
        eliminate = order == 2;
        graded = by_degree;
        words = (n + 3) / 4;
        stride = words + 1;
        bits = std::min(8, 64 / n);
    }
    // grevlex compares the last variable first, lex the first one.
    int position(int v) const { return lex ? v : n - 1 - v; }
    static u64 field(const u64* m, int p) { return (m[1 + (p >> 2)] >> (48 - 16 * (p & 3))) & 0xFFFF; }

    void pack(u64* m, const long* e) const {
        std::fill(m, m + stride, 0);
        u64 degree = 0;
        for (int v = 0; v < n; ++v) {
            if (e[v] < 0 || static_cast<u64>(e[v]) > MAX_DEGREE) throw Limit("Exponent out of range.");
            degree += static_cast<u64>(e[v]);
            const int p = position(v);
            m[1 + (p >> 2)] |= static_cast<u64>(e[v]) << (48 - 16 * (p & 3));
        }
        if (degree > MAX_DEGREE) throw Limit("Degree out of range.");
        m[0] = degree;
    }
    void unpack(const u64* m, long* e) const {
        for (int v = 0; v < n; ++v) e[v] = static_cast<long>(field(m, position(v)));
    }
    // 1 when a > b in the monomial order, -1 when a < b.
    int compare(const u64* a, const u64* b) const {
        if (graded && a[0] != b[0]) return a[0] > b[0] ? 1 : -1;
        if (eliminate) {
            const u64 ea = field(a, n - 1), eb = field(b, n - 1);
            if (ea != eb) return ea > eb ? 1 : -1;
        }
        if (!lex) {
            if (a[0] != b[0]) return a[0] > b[0] ? 1 : -1;
            for (int i = 1; i < stride; ++i)
                if (a[i] != b[i]) return a[i] < b[i] ? 1 : -1;
            return 0;
        }
        for (int i = 1; i < stride; ++i)
            if (a[i] != b[i]) return a[i] > b[i] ? 1 : -1;
        return 0;
    }
    bool equal(const u64* a, const u64* b) const {
        for (int i = 1; i < stride; ++i)
            if (a[i] != b[i]) return false;
        return true;
    }
    bool divides(const u64* a, const u64* b) const {  // a | b
        for (int i = 1; i < stride; ++i)
            if ((((b[i] | HI) - a[i]) & HI) != HI) return false;
        return true;
    }
    void multiply(u64* r, const u64* a, const u64* b) const {
        r[0] = a[0] + b[0];
        if (r[0] > MAX_DEGREE) throw Limit("Degree out of range.");
        for (int i = 1; i < stride; ++i) r[i] = a[i] + b[i];
    }
    void divide(u64* r, const u64* a, const u64* b) const {  // a / b, b | a
        for (int i = 0; i < stride; ++i) r[i] = a[i] - b[i];
    }
    void lcm(u64* r, const u64* a, const u64* b) const {
        u64 degree = 0;
        for (int i = 1; i < stride; ++i) {
            const u64 ge = ((a[i] | HI) - b[i]) & HI;      // fields where a >= b
            const u64 select = (ge - (ge >> 15)) | ge;     // those fields, all ones
            const u64 w = (a[i] & select) | (b[i] & ~select);
            r[i] = w;
            degree += (w & 0xFFFF) + ((w >> 16) & 0xFFFF) + ((w >> 32) & 0xFFFF) + (w >> 48);
        }
        if (degree > MAX_DEGREE) throw Limit("Degree out of range.");
        r[0] = degree;
    }
    // Divisibility prefilter: g | m implies mask(g) is a subset of mask(m).
    u64 mask(const u64* m) const {
        u64 result = 0;
        int bit = 0;
        for (int p = 0; p < n; ++p) {
            const u64 e = field(m, p);
            for (int k = 1; k <= bits; ++k, ++bit)
                if (e >= static_cast<u64>(k)) result |= 1ULL << bit;
        }
        return result;
    }
};

// Terms in decreasing monomial order; integer coefficients.
struct Poly {
    std::vector<Z> c;
    std::vector<u64> m;
    size_t size() const { return c.size(); }
    bool empty() const { return c.empty(); }
    void clear() { c.clear(); m.clear(); }
    void swap(Poly& other) noexcept { c.swap(other.c); m.swap(other.m); }
};

struct Rational {
    Z num, den;
};

// out = ca * x^shift_a * a[ia..] + cb * x^shift_b * b[ib..]; a null factor is one.
void merge(const Ring& R, Poly& out, const Poly& a, size_t ia, const fmpz* ca, const u64* shift_a,
           const Poly& b, size_t ib, const fmpz* cb, const u64* shift_b);
// Divide f by its content, signed so that the leading coefficient becomes
// positive; num, when given, is multiplied by what was divided out.
void primitive(Poly& f, fmpz* num);
Poly multiply(const Ring& R, const Poly& a, const Poly& b);
// One polynomial, ended by a newline: factors separated by '*', each a list
// of terms "coefficient e1 ... en" separated by ';'. The coefficient is an
// integer or p/q. A factor without terms is zero. The result is primitive;
// with scale, the polynomial read is result*num/den.
Poly parse(const Ring& R, const char*& at, const char* end, Rational* scale);
// The terms of f*num/den in that format, ended by a newline.
void write(const Ring& R, std::string& out, const Poly& f, const fmpz* num, const fmpz* den);

struct Element {
    Poly p;
    u64 mask = 0;
    long sugar = 0;
    bool active = true;   // still a reducer
    bool pairing = true;  // still forms pairs with new elements
};

struct Pair {
    int i, j;
    long sugar;
    u64 mask;
    u64 lcm[MAX_STRIDE];
};

// A polynomial waiting for its turn, with its sugar.
struct Delayed {
    Poly p;
    long sugar;
};

// Buchberger's algorithm with the Gebauer-Moeller installation of pairs.
// After compute, G is the reduced basis: primitive, largest head first.
class Engine {
public:
    Ring R;
    std::vector<Element> G;
    std::vector<Pair> B;  // sorted, the next pair to treat is last
    bool unit = false;
    long reductions = 0, zero_reductions = 0, restarts = 0;
    int route = 0;  // 0: an exact strategy, 1: modular images, 2: the Hilbert function
    double seconds = 0;
    std::string text;

    // The reduced basis is unique, so the strategy only affects the time.
    // When coefficients or work swell under one strategy the next one takes
    // over, and they alternate under wider limits.
    // The first `known` polynomials are declared to be a Groebner basis.
    void compute(const std::vector<Poly>& input, size_t known);
    // Replace f by a scalar multiple of its normal form; the terms before
    // `keep` are left alone. With num and den the exact remainder is f*num/den.
    // With a bound, a reducer is used only when it keeps the sugar within it.
    void reduce(Poly& f, size_t keep, fmpz* num, fmpz* den, long* sugar = nullptr, long bound = LONG_MAX);
    // Buchberger's algorithm for homogeneous generators when the Hilbert
    // series of the ideal they generate is known (its numerator, for the
    // denominator (1-t)^n): in a degree where the leading monomials found
    // already account for the whole ideal, the remaining pairs are dropped.
    // False when the series given cannot be that of the ideal. Throws Swell
    // when a coefficient or the work exceeds a limit (0: none).
    bool compute_homogeneous(const std::vector<Poly>& generators, const std::vector<Z>& numerator,
                             size_t bit_limit, double work_limit);
    // The numerator of the Hilbert series of the quotient by the ideal that
    // the leading monomials of the current reducers generate.
    std::vector<Z> leading_numerator() const;
    // Whether the polynomials are a Groebner basis of the ideal they generate:
    // every S-polynomial that the criteria do not discard reduces to zero.
    // On success G holds the reduced basis of that ideal.
    bool is_basis(const std::vector<Poly>& candidate);

    // EAGER: pairs by sugar, each S-polynomial reduced completely at once.
    // NORMAL: pairs by lcm alone, reducers by smallest head.
    // LAZY: pairs by sugar, and the sugar of a polynomial is its degree in the
    // homogenized computation: a reducer that would raise it is not used on
    // tails, and on a head it sends the polynomial back to wait until every
    // pair of lower sugar has been treated.
    enum Strategy { EAGER = 0, NORMAL = 1, LAZY = 2 };

protected:
    int strategy_ = EAGER;
    size_t bit_limit_ = 0;       // 0: no limit
    double work_limit_ = 0;
    mutable double work_ = 0;
    std::vector<int> active_;    // the reducers, in the order they are tried
    std::vector<u64> active_mask_;
    std::vector<const u64*> active_lead_;
    std::vector<long> active_ecart_;  // sugar minus the degree of the head
    std::vector<size_t> candidates_;                    // scratch for insert
    std::vector<u64> pair_lcm_, pair_mask_, pair_key_;
    std::vector<long> pair_sugar_;
    std::vector<Delayed> waiting_;

    enum Outcome { ZERO, POSTPONED, READY };
    void attempt(std::vector<Poly>& input, size_t known);
    Outcome head(Poly& f, long& sugar);
    bool pending_below(long sugar) const {
        if (!B.empty() && B.back().sugar < sugar) return true;
        for (const Delayed& d : waiting_)
            if (d.sugar < sugar) return true;
        return false;
    }
    void guard(const Poly& f) const {
        if (bit_limit_ && !f.empty() && fmpz_bits(&f.c[0].v) > bit_limit_) throw Swell();
        if (work_limit_ > 0 && work_ > work_limit_) throw Swell();
    }
    const u64* lead(int g) const { return G[g].p.m.data(); }
    long degree(const Poly& f) const {
        u64 d = 0;
        for (size_t i = 0; i < f.size(); ++i) d = std::max(d, f.m[i * R.stride]);
        return static_cast<long>(d);
    }
    // Reducers are tried in this order: small sugar excess, then few terms.
    bool before(int x, int y) const {
        if (strategy_ == NORMAL) {
            const int order = R.compare(lead(x), lead(y));
            return order ? order < 0 : x < y;
        }
        const long ex = G[x].sugar - static_cast<long>(lead(x)[0]), ey = G[y].sugar - static_cast<long>(lead(y)[0]);
        if (ex != ey) return ex < ey;
        if (G[x].p.size() != G[y].p.size()) return G[x].p.size() < G[y].p.size();
        return x < y;
    }
    void rebuild_active() {
        active_.clear(); active_mask_.clear(); active_lead_.clear(); active_ecart_.clear();
        for (size_t g = 0; g < G.size(); ++g)
            if (G[g].active) active_.push_back(static_cast<int>(g));
        std::sort(active_.begin(), active_.end(), [this](int x, int y) { return before(x, y); });
        for (int g : active_) {
            active_mask_.push_back(G[g].mask);
            active_lead_.push_back(lead(g));
            active_ecart_.push_back(G[g].sugar - static_cast<long>(lead(g)[0]));
        }
    }
    // The first reducer of m whose sugar excess is at most `room`. Outside the
    // normal strategy the reducers are sorted by that excess.
    int reducer(const u64* m, long room = LONG_MAX) const {
        const u64 inverse = ~R.mask(m);
        const size_t count = active_.size();
        for (size_t k = 0; k < count; ++k) {
            if (active_ecart_[k] > room) {
                if (strategy_ == NORMAL) continue;
                break;
            }
            if (!(active_mask_[k] & inverse) && R.divides(active_lead_[k], m)) return active_[k];
        }
        return -1;
    }
    void merge(Poly& out, const Poly& a, size_t ia, const fmpz* ca, const u64* shift_a,
               const Poly& b, size_t ib, const fmpz* cb, const u64* shift_b) const {
        atlas::merge(R, out, a, ia, ca, shift_a, b, ib, cb, shift_b);
        if (work_limit_ > 0) work_ += static_cast<double>(out.size()) + 1;  // no write without a limit: see is_basis
    }
    Poly spoly(const Pair& pair) const;
    void append(Poly&& h, long sugar);
    void insert(Poly&& h, long sugar);
    void finish();
};

// f with every term brought to the degree of f by a power of the last
// variable of H, which has one variable more than R; and the polynomial of H
// with that variable set to one.
Poly homogenize(const Ring& R, const Ring& H, const Poly& f);
Poly dehomogenize(const Ring& H, const Ring& R, const Poly& f);

// The numerator N of the Hilbert series N(t)/(1-t)^m of Q[z_1..z_m]/M, for the
// monomial ideal M generated by the monomials with the given exponents (m
// numbers for each). And the dimension of the part of degree d of a quotient
// with that numerator.
std::vector<Z> hilbert_numerator(std::vector<uint16_t> generators, int m);
void hilbert_value(fmpz* out, const std::vector<Z>& numerator, int m, long d);
// The reduced Groebner basis for an order that does not compare degrees
// first, through the homogenized ideal and its Hilbert function (hilbert.cpp).
// False when the route does not apply or exceeds the limits.
bool hilbert_basis(const Ring& R, const std::vector<Poly>& input, Engine& result, size_t bit_limit, double work_limit);

// Euler characteristics with compact supports of constructible sets of an
// affine space over the complex numbers (euler.cpp). One object keeps what it
// has computed, for the sets of one chart.
class Euler {
public:
    explicit Euler(const Ring& R);
    ~Euler();
    Euler(const Euler&) = delete;
    Euler& operator=(const Euler&) = delete;
    // Of V(equations) minus the union of the V(I) for the ideals I given.
    long characteristic(const std::vector<Poly>& equations, const std::vector<std::vector<Poly>>& excluded);

private:
    struct State;
    State* state_;
};

// Thread-local message of the last failure, and a malloc'ed copy of a string.
void set_error(const char* message);
char* copy_text(const std::string& s);

// What the modular route did (modular.cpp).
struct ModularReport {
    long primes = 0, rejected = 0, traces = 0, verifications = 0;
    long instructions = 0, pool = 0, basis = 0;
    double trace_seconds = 0, replay_seconds = 0, lift_seconds = 0, verify_seconds = 0;
    bool gpu = false;
};
// The reduced Groebner basis of the ideal generated by the input, for the
// order of R, computed through modular images and certified over Q. It is
// left in `result`. False when the route gives up, which it does when the
// replay program would exceed `budget` words; `result` is then unusable.
bool modular_basis(const Ring& R, const std::vector<Poly>& input, Engine& result, ModularReport* report = nullptr,
                   size_t budget = static_cast<size_t>(1) << 27);
// Run a replay program for several primes at once. `pools` holds, for each
// prime, `pool` words whose first part the caller has filled with the
// generators; `status` receives 1 for a prime that agrees with the trace.
// Returns 0 when the batch was run.
typedef int (*ReplayBatch)(const uint32_t* code, size_t length, uint32_t pool, uint32_t work, const uint32_t* primes,
                           uint32_t count, uint32_t* pools, uint32_t* status);

}  // namespace atlas

#endif
