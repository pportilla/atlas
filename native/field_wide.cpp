// Unbounded sparse Buchberger and ordered division for product monomial orders.
// Each exponent is a FLINT integer; only the number of actual terms costs space.
#include "flint_polynomial.hpp"
#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <system_error>
#include <thread>
#include <flint/fmpq.h>
#include <map>
#include <set>
#include <sstream>

namespace atlas {
namespace {
using Powers = std::vector<Z>;

struct Coefficient {
    fmpq_t q;
    Coefficient() { fmpq_init(q); }
    Coefficient(const Coefficient& other) : Coefficient() { fmpq_set(q, other.q); }
    Coefficient(Coefficient&& other) noexcept : Coefficient() { fmpq_swap(q, other.q); }
    ~Coefficient() { fmpq_clear(q); }
};

struct Order {
    size_t n;              // number of ordinary x variables
    int kind;              // 0 grevlex, 1 lex, 2 first then grevlex, 3 grlex
    bool coefficient;      // coordinate theta after the ordinary variables
    bool operator()(const Powers& a, const Powers& b) const {
        if (kind == 2) {
            int first = fmpz_cmp(&a[0].v, &b[0].v);
            if (first) return first > 0;
        }
        if (kind == 1) {
            for (size_t j = 0; j < n; ++j) {
                int cmp = fmpz_cmp(&a[j].v, &b[j].v);
                if (cmp) return cmp > 0;
            }
        } else {
            Z grade;
            for (size_t j = 0; j < n; ++j) {
                fmpz_add(&grade.v, &grade.v, &a[j].v);
                fmpz_sub(&grade.v, &grade.v, &b[j].v);
            }
            if (!fmpz_is_zero(&grade.v)) return fmpz_sgn(&grade.v) > 0;
            if (kind == 3) {
                for (size_t j = 0; j < n; ++j) {
                    const int cmp = fmpz_cmp(&a[j].v, &b[j].v);
                    if (cmp) return cmp > 0;
                }
            }
            for (size_t j = n; j-- > 0;) {
                int cmp = fmpz_cmp(&a[j].v, &b[j].v);
                if (cmp) return cmp < 0;
            }
        }
        return coefficient && fmpz_cmp(&a[n].v, &b[n].v) > 0;
    }
};

using Terms = std::map<Powers, Coefficient, Order>;
using Rows = std::vector<Terms>;

bool divides(const Powers& a, const Powers& b) {
    for (size_t j = 0; j < a.size(); ++j) if (fmpz_cmp(&a[j].v, &b[j].v) > 0) return false;
    return true;
}

Powers difference(const Powers& top, const Powers& bottom) {
    Powers m(top.size());
    for (size_t i = 0; i < m.size(); ++i) fmpz_sub(&m[i].v, &top[i].v, &bottom[i].v);
    return m;
}

void add(Terms& p, Powers m, const Coefficient& c) {
    auto entry = p.try_emplace(std::move(m)).first;
    fmpq_add(entry->second.q, entry->second.q, c.q);
    if (fmpq_is_zero(entry->second.q)) p.erase(entry);
}

void add_shifted(Terms& p, const Terms& g, const Powers& shift, const Coefficient& factor) {
    Coefficient c;
    for (const auto& [exponents, value] : g) {
        Powers m(shift.size());
        for (size_t i = 0; i < m.size(); ++i) fmpz_add(&m[i].v, &exponents[i].v, &shift[i].v);
        fmpq_mul(c.q, value.q, factor.q);
        add(p, std::move(m), c);
    }
}

void monic(Terms& p) {
    const Coefficient lead = p.begin()->second;
    for (auto& term : p) fmpq_div(term.second.q, term.second.q, lead.q);
}

// In a basis computation x^a = +/-M can replace all copies in one term at once.
// Every substitution lowers the monomial and preserves its residue modulo the
// reducers. Ordered division still performs each step to collect its quotients.
bool reduce_binomial_power(Terms& p, const Terms& divisor) {
    if (divisor.size() != 2) return false;
    const auto& [lead, lc] = *divisor.begin();
    const auto& [tail, tc] = *divisor.rbegin();
    size_t variable = lead.size();
    for (size_t j = 0; j < lead.size(); ++j) {
        if (fmpz_is_zero(&lead[j].v)) continue;
        if (variable != lead.size() || !fmpz_is_zero(&tail[j].v)) return false;
        variable = j;
    }
    if (variable == lead.size()) return false;
    Coefficient ratio;
    fmpq_div(ratio.q, tc.q, lc.q);
    if (!fmpz_is_one(fmpq_denref(ratio.q)) || !fmpz_is_pm1(fmpq_numref(ratio.q))) return false;

    auto head = p.extract(p.begin());
    auto& powers = head.key();
    Z copies, leftover;
    fmpz_fdiv_qr(&copies.v, &leftover.v, &powers[variable].v, &lead[variable].v);
    powers[variable] = std::move(leftover);
    for (size_t j = 0; j < powers.size(); ++j)
        fmpz_addmul(&powers[j].v, &copies.v, &tail[j].v);
    // x^a = -(tc/lc) M, so equal signs flip the coefficient for odd powers.
    if (fmpq_sgn(ratio.q) > 0 && fmpz_is_odd(&copies.v))
        fmpq_neg(head.mapped().q, head.mapped().q);
    add(p, std::move(powers), head.mapped());
    return true;
}

Terms divide(Terms p, const Rows& reducers, Rows* quotients = nullptr) {
    Terms remainder(p.key_comp());
    while (!p.empty()) {
        auto head = p.begin();
        size_t k = 0;
        while (k < reducers.size() && !divides(reducers[k].begin()->first, head->first)) ++k;
        if (k == reducers.size()) {
            remainder.insert(p.extract(head));
            continue;
        }
        if (!quotients && reduce_binomial_power(p, reducers[k])) continue;
        auto shift = difference(head->first, reducers[k].begin()->first);
        Coefficient factor;
        fmpq_div(factor.q, head->second.q, reducers[k].begin()->second.q);
        if (quotients) add((*quotients)[k], shift, factor);
        fmpq_neg(factor.q, factor.q);
        add_shifted(p, reducers[k], shift, factor);
    }
    return remainder;
}

// Whether divide(p, reducers) is zero. A term set aside for the remainder is
// never touched again, so the first one already decides.
bool reduces_to_zero(Terms p, const Rows& reducers) {
    while (!p.empty()) {
        auto head = p.begin();
        size_t k = 0;
        while (k < reducers.size() && !divides(reducers[k].begin()->first, head->first)) ++k;
        if (k == reducers.size()) return false;
        if (reduce_binomial_power(p, reducers[k])) continue;
        auto shift = difference(head->first, reducers[k].begin()->first);
        Coefficient factor;
        fmpq_div(factor.q, head->second.q, reducers[k].begin()->second.q);
        fmpq_neg(factor.q, factor.q);
        add_shifted(p, reducers[k], shift, factor);
    }
    return true;
}

Terms s_polynomial(const Terms& a, const Terms& b) {
    const auto& u = a.begin()->first;
    const auto& v = b.begin()->first;
    Powers lcm(u.size());
    bool coprime = true;
    for (size_t i = 0; i < u.size(); ++i) {
        if (!fmpz_is_zero(&u[i].v) && !fmpz_is_zero(&v[i].v)) coprime = false;
        fmpz_set(&lcm[i].v, fmpz_cmp(&u[i].v, &v[i].v) >= 0 ? &u[i].v : &v[i].v);
    }
    Terms p(a.key_comp());
    if (coprime) return p;  // Buchberger's product criterion
    Coefficient sign;
    fmpq_one(sign.q);
    add_shifted(p, a, difference(lcm,u), sign);
    fmpq_neg(sign.q, sign.q);
    add_shifted(p, b, difference(lcm,v), sign);
    return p;
}

Rows basis(Rows input) {
    Rows g;
    auto insert = [&](Terms p) {
        if (p.empty()) return;
        p = divide(std::move(p), g);
        if (!p.empty()) { monic(p); g.push_back(std::move(p)); }
    };
    for (auto& p : input) insert(std::move(p));
    for (size_t j = 0; j < g.size(); ++j)
        for (size_t i = 0; i < j; ++i) insert(s_polynomial(g[i], g[j]));
    std::vector<size_t> minimal;
    for (size_t i = 0; i < g.size(); ++i) {
        bool needed = true;
        for (size_t j = 0; j < g.size(); ++j) {
            if (i != j && divides(g[j].begin()->first, g[i].begin()->first) &&
                (j < i || !divides(g[i].begin()->first, g[j].begin()->first))) {
                needed = false;
                break;
            }
        }
        if (needed) minimal.push_back(i);
    }
    Rows reduced;
    for (size_t i : minimal) {
        Terms tail = g[i];
        auto head = tail.extract(tail.begin());
        // Its own head cannot reduce any smaller tail term.
        tail = divide(std::move(tail), g);
        tail.insert(std::move(head));
        reduced.push_back(std::move(tail));
    }
    std::sort(reduced.begin(), reduced.end(), [](const Terms& a, const Terms& b) {
        return a.key_comp()(a.begin()->first, b.begin()->first);
    });
    return reduced;
}

Terms read_row(const std::string& text, Order order) {
    Terms p(order);
    std::istringstream terms(text);
    for (std::string word; std::getline(terms, word, ';');) {
        std::istringstream input(word);
        std::string coefficient;
        Coefficient c;
        if (!(input >> coefficient) || fmpq_set_str(c.q, coefficient.c_str(), 10) ||
            fmpz_is_zero(fmpq_denref(c.q))) throw std::invalid_argument("Invalid rational coefficient.");
        fmpq_canonicalise(c.q);
        Powers exponents(order.n+order.coefficient);
        for (auto& e : exponents) {
            std::string value;
            if (!(input >> value) || fmpz_set_str(&e.v, value.c_str(), 10) || fmpz_sgn(&e.v) < 0)
                throw std::invalid_argument("Invalid polynomial exponent.");
        }
        if (input >> coefficient) throw std::invalid_argument("Unexpected polynomial exponent.");
        add(p, std::move(exponents), c);
    }
    return p;
}

void print_row(std::string& out, const Terms& p) {
    for (auto it = p.begin(); it != p.end(); ++it) {
        if (it != p.begin()) out += ';';
        char* value = fmpq_get_str(nullptr, 10, it->second.q);
        out += value;
        flint_free(value);
        for (const auto& v : it->first) {
            value = fmpz_get_str(nullptr, 10, &v.v);
            out += ' ';
            out += value;
            flint_free(value);
        }
    }
    out += '\n';
}

// With lex order [other chart coordinates, v, base coordinates], a row whose
// leading chart power is v and whose other chart powers are zero is D*v-N.
// The reduced basis is monic, so D already has leading coefficient one.
std::string inverse_fraction(Rows rows, size_t base_size, Order order) {
    if (base_size == 0 || base_size >= order.n || order.kind != 1 || order.coefficient)
        throw std::invalid_argument("Invalid chart inverse ring.");
    const size_t v = order.n - base_size - 1;
    const Order base{base_size, 1, false};
    for (const auto& row : basis(std::move(rows))) {
        const auto& head = row.begin()->first;
        if (!fmpz_is_one(&head[v].v) || std::any_of(head.begin(), head.begin()+v,
                    [](const Z& power) { return !fmpz_is_zero(&power.v); })) continue;
        Terms numerator(base), denominator(base);
        for (const auto& [exponents, value] : row) {
            Powers projected(exponents.begin()+v+1, exponents.end());
            if (fmpz_is_zero(&exponents[v].v)) {
                Coefficient negative;
                fmpq_neg(negative.q, value.q);
                numerator.emplace(std::move(projected), std::move(negative));
            } else denominator.emplace(std::move(projected), value);
        }
        std::string result;
        print_row(result, numerator);
        print_row(result, denominator);
        return result;
    }
    throw std::invalid_argument("The chart map has no rational inverse.");
}

// The first variable is an inverse of the product of the frame units. A
// first-variable elimination order induces grevlex after this contraction.
std::string frame_result(Rows rows, Order order, bool closure) {
    auto reduced = basis(std::move(rows));
    if (!closure) {
        if (reduced.size() != 1) return "0\n";
        const auto& head = reduced.front().begin()->first;
        return std::all_of(head.begin(), head.end(),
                    [](const Z& p) { return fmpz_is_zero(&p.v); }) ? "1\n" : "0\n";
    }
    const Order contracted{order.n-1, 0, false};
    std::string result;
    for (const auto& row : reduced) {
        if (!fmpz_is_zero(&row.begin()->first.front().v)) continue;
        Terms projected(contracted);
        for (const auto& [m, c] : row)
            projected.emplace(Powers(m.begin()+1, m.end()), c);
        print_row(result, projected);
    }
    return result;
}

using VectorField = std::vector<std::pair<size_t, Terms>>;

Terms derivative(const Terms& f, const VectorField& field) {
    Terms out(f.key_comp());
    for (const auto& [axis, coefficient] : field) for (const auto& [e,c] : f) {
        if (fmpz_is_zero(&e[axis].v)) continue;
        Powers head(e);
        fmpz_sub_ui(&head[axis].v,&head[axis].v,1);
        Coefficient scale;
        fmpq_mul_fmpz(scale.q,c.q,&e[axis].v);
        add_shifted(out,coefficient,head,scale);
    }
    return out;
}

Terms lift_inverse(const Terms& p, Order extended, bool multiply_inverse) {
    Terms result(extended);
    for (const auto& [e,c] : p) {
        Powers shifted(e);
        shifted.emplace(shifted.begin());
        if (multiply_inverse) fmpz_one(&shifted.front().v);
        result.emplace(std::move(shifted),c);
    }
    return result;
}

Terms product(const Terms& f, const Terms& g) {
    Terms result(f.key_comp());
    for (const auto& [e,c] : f) add_shifted(result,g,e,c);
    return result;
}

struct FrameRing {
    fmpq_mpoly_ctx_t ctx;
    explicit FrameRing(size_t width) { fmpq_mpoly_ctx_init(ctx, width, ORD_DEGREVLEX); }
    ~FrameRing() { fmpq_mpoly_ctx_clear(ctx); }
};

// Divisors shared read-only, so that several threads may reduce by them.
struct PackedDivisors {
    const fmpq_mpoly_ctx_struct* ctx;
    std::vector<PolyValue> storage;
    std::vector<fmpq_mpoly_struct*> rows;
    std::vector<std::vector<ulong>> heads;
    explicit PackedDivisors(const fmpq_mpoly_ctx_struct* c) : ctx(c) {}
    void add(PolyValue&& polynomial) {
        storage.push_back(std::move(polynomial));
    }
    void prepare() {
        for (auto& row : storage) {
            rows.push_back(row.p);
            auto& head = heads.emplace_back(fmpq_mpoly_ctx_nvars(ctx));
            fmpq_mpoly_get_term_exp_ui(head.data(), row.p, 0, ctx);
        }
    }
    bool irreducible_head(const PolyValue& f) const {
        std::vector<ulong> term(fmpq_mpoly_ctx_nvars(ctx));
        fmpq_mpoly_get_term_exp_ui(term.data(), f.p, 0, ctx);
        return std::none_of(heads.begin(), heads.end(), [&](const auto& head) {
            for (size_t i = 0; i < term.size(); ++i)
                if (head[i] > term[i]) return false;
            return true;
        });
    }
    void reduce(PolyValue& f) const {
        if (rows.empty() || fmpq_mpoly_is_zero(f.p, ctx)) return;
        std::vector<PolyValue> scratch;
        scratch.reserve(rows.size());
        std::vector<fmpq_mpoly_struct*> quotients;
        for (size_t i = 0; i < rows.size(); ++i) quotients.push_back(scratch.emplace_back(ctx).p);
        PolyValue remainder(ctx);
        fmpq_mpoly_divrem_ideal(quotients.data(), remainder.p, f.p, rows.data(), rows.size(), ctx);
        fmpq_mpoly_swap(f.p, remainder.p, ctx);
    }
};

struct PackingLimit {};

// Store exponents in FLINT's graded ring. Bound expansion of packed
// division for high powers, and the common denominator of each sparse row.
PolyValue pack_frame_row(const Terms& f, const FrameRing& ring) {
    PolyValue result(ring.ctx);
    std::vector<ulong> powers(f.key_comp().n);
    Z common;
    fmpz_one(&common.v);
    for (const auto& term : f) {
        fmpz_lcm(&common.v, &common.v, fmpq_denref(term.second.q));
        if (fmpz_bits(&common.v) > 16384 || fmpz_bits(fmpq_numref(term.second.q)) > 16384)
            throw PackingLimit{};
    }
    Z coefficient;
    for (const auto& [e, c] : f) {
        unsigned degree = 0;
        for (size_t i = 0; i < e.size(); ++i) {
            if (fmpz_cmp_ui(&e[i].v, 512 - degree) > 0) throw PackingLimit{};
            powers[i] = fmpz_get_ui(&e[i].v);
            degree += powers[i];
        }
        fmpz_divexact(&coefficient.v, &common.v, fmpq_denref(c.q));
        fmpz_mul(&coefficient.v, &coefficient.v, fmpq_numref(c.q));
        fmpz_mpoly_push_term_fmpz_ui(result.p->zpoly, &coefficient.v, powers.data(), ring.ctx->zctx);
    }
    fmpz_one(fmpq_numref(result.p->content));
    fmpz_set(fmpq_denref(result.p->content), &common.v);
    // Input terms already use the FLINT ring's descending grevlex order.
    fmpq_mpoly_reduce(result.p, ring.ctx);
    return result;
}

Terms unpack_frame_row(const PolyValue& p, Order order) {
    Terms result(order);
    std::vector<ulong> e(order.n);
    for (slong i = 0; i < fmpq_mpoly_length(p.p, p.ctx); ++i) {
        fmpq_mpoly_get_term_exp_ui(e.data(), p.p, i, p.ctx);
        Powers powers(order.n);
        for (size_t j = 0; j < e.size(); ++j) fmpz_set_ui(&powers[j].v, e[j]);
        auto pos = result.try_emplace(result.end(),std::move(powers));
        fmpq_mpoly_get_term_coeff_fmpq(pos->second.q,p.p,i,p.ctx);
    }
    return result;
}

// Remainders modulo a reduced Groebner basis for a graded reverse
// lexicographic order. The remainder modulo a Groebner basis does not depend
// on how the division is carried out, so FLINT's packed division returns the
// remainder of ordered division. Rows past the packing range raise
// PackingLimit, and the caller divides them in the sparse map instead.
class PackedRemainders {
    FrameRing ring;
    PackedDivisors divisors;
    Order order;
public:
    PackedRemainders(const Rows& basis, Order order_) : ring(order_.n), divisors(ring.ctx), order(order_) {
        if (order.kind != 0 || order.coefficient) throw PackingLimit{};
        for (const auto& f : basis) divisors.add(pack_frame_row(f, ring));
        divisors.prepare();
    }
    Terms remainder(const Terms& f) const {
        auto p = pack_frame_row(f, ring);
        divisors.reduce(p);
        return unpack_frame_row(p, order);
    }
    bool zero(const Terms& f) const {
        auto p = pack_frame_row(f, ring);
        divisors.reduce(p);
        return fmpq_mpoly_is_zero(p.p, ring.ctx);
    }
};

// Keep the derivative representatives identical to ordered chart division.
// The saturated basis is needed only to discard globally vanishing jets.
class PackedDerivatives {
    FrameRing ring;
    PackedDivisors reducers, saturated;
    using Field = std::vector<std::pair<size_t, PolyValue>>;
    std::vector<Field> fields;
public:
    PackedDerivatives(size_t n, const Rows& basis, const Rows& open_basis,
                      const std::vector<VectorField>& directions)
        : ring(n), reducers(ring.ctx), saturated(ring.ctx) {
        for (const auto& f : basis) reducers.add(pack_frame_row(f, ring));
        for (const auto& f : open_basis) saturated.add(pack_frame_row(f, ring));
        reducers.prepare();
        saturated.prepare();
        for (const auto& field : directions) {
            fields.emplace_back();
            for (const auto& [axis, f] : field)
                fields.back().emplace_back(axis, pack_frame_row(f, ring));
        }
    }

    Terms apply(const Terms& f, size_t k, bool plain, bool jet) const {
        auto source = pack_frame_row(f, ring);
        PolyValue sum(ring.ctx), part(ring.ctx), product(ring.ctx);
        for (const auto& [axis, coefficient] : fields[k]) {
            fmpq_mpoly_derivative(part.p, source.p, axis, ring.ctx);
            fmpq_mpoly_mul(product.p, part.p, coefficient.p, ring.ctx);
            fmpq_mpoly_add(sum.p, sum.p, product.p, ring.ctx);
        }
        if (!plain) reducers.reduce(sum);
        // Once the leading term survives division, the saturated residue is
        // nonzero. No quotient or representation of that residue is required.
        if (jet && !fmpq_mpoly_is_zero(sum.p, ring.ctx) && !saturated.irreducible_head(sum)) {
            PolyValue test(ring.ctx);
            fmpq_mpoly_set(test.p, sum.p, ring.ctx);
            saturated.reduce(test);
            if (fmpq_mpoly_is_zero(test.p, ring.ctx)) return Terms(f.key_comp());
        }
        return unpack_frame_row(sum, f.key_comp());
    }
};

// Below this many input terms a layer of derivatives is computed in one thread.
constexpr size_t PARALLEL_TERMS = 2048;

// Calls work(0), ..., work(n-1), on several threads when `parallel`.
template<class Work>
void each_index(size_t n, bool parallel, Work&& work) {
    const size_t lanes = parallel ? std::min<size_t>(n, std::thread::hardware_concurrency()) : 1;
    if (lanes <= 1) {
        for (size_t j = 0; j < n; ++j) work(j);
        return;
    }
    std::atomic<size_t> next{0};
    // The failure of the first index, as the sequential loop would meet it.
    std::exception_ptr failure;
    size_t failed = n;
    std::mutex guard;
    auto worker = [&] {
        for (size_t j; (j = next.fetch_add(1, std::memory_order_relaxed)) < n;) {
            try {
                work(j);
            } catch (...) {
                std::lock_guard<std::mutex> lock(guard);
                if (j < failed) {
                    failed = j;
                    failure = std::current_exception();
                }
            }
        }
    };
    std::vector<std::thread> crew;
    crew.reserve(lanes - 1);
    try {
        for (size_t k = 1; k < lanes; ++k) crew.emplace_back([&] {
            worker();
            flint_cleanup();
        });
    } catch (const std::system_error&) {
        // The started threads and this one take every index.
    }
    worker();
    for (auto& t : crew) t.join();
    if (failure) std::rethrow_exception(failure);
}

// Derivatives and coordinate orders on a localized implicit chart. Bind the
// chart's cached reduced bases once for all layers and repeated divisions.
std::string implicit_frame(std::istream& input, bool jet, bool chain) {
    auto line = [&]() {
        std::string value;
        if (!std::getline(input,value)) throw std::invalid_argument("Incomplete implicit frame.");
        return value;
    };
    auto integer = [&]() { return std::stoul(line()); };
    const size_t n = integer();
    if (!n) throw std::invalid_argument("Missing implicit frame coordinates.");
    const Order chart{n,0,false}, inverse{n+1,2,false}, localized{n+1,0,false};
    auto rows = [&](Order order) {
        const size_t count = integer();
        Rows g;
        for (size_t j = 0; j < count; ++j) {
            auto p = read_row(line(),order);
            if (!p.empty()) g.push_back(std::move(p));
        }
        return g;
    };
    const auto reducers = rows(chart);
    Terms f = read_row(line(),chart);
    std::string result;
    if (jet || chain) {
        auto order = integer();
        const auto saturated = jet ? rows(chart) : Rows{};
        const size_t count = jet ? integer() : 1;
        std::vector<VectorField> fields(count);
        std::vector<bool> plain(count,false);
        for (size_t k = 0; k < count; ++k) {
            plain[k] = integer() != 0;
            const auto entries = integer();
            for (size_t j = 0; j < entries; ++j) {
                const auto axis = integer();
                if (axis >= n) throw std::invalid_argument("Invalid derivative coordinate.");
                fields[k].emplace_back(axis, read_row(line(),chart));
            }
        }
        print_row(result,f);
        std::unique_ptr<PackedDerivatives> packed;
        try {
            packed = std::make_unique<PackedDerivatives>(n, reducers, saturated, fields);
        } catch (const PackingLimit&) {}
        auto apply = [&](const Terms& p, size_t k) {
            if (packed) {
                try {
                    return packed->apply(p,k,plain[k],jet);
                } catch (const PackingLimit&) {}
            }
            auto d = derivative(p,fields[k]);
            if (!plain[k]) d = divide(std::move(d),reducers);
            if (jet && !d.empty() && reduces_to_zero(d,saturated)) d.clear();
            return d;
        };
        if (chain) {
            for (size_t i = 0; i < order; ++i) {
                f = apply(f,0);
                print_row(result,f);
            }
            return result;
        }
        std::set<std::string> seen{result};
        Rows layer{std::move(f)};
        for (size_t i = 1; i < order && !layer.empty(); ++i) {
            // The derivatives of a layer are independent; new ones are kept
            // in the order of the sequential loop.
            Rows derived(layer.size() * count, Terms(chart));
            size_t terms = 0;
            for (const auto& p : layer) terms += p.size();
            each_index(derived.size(), terms * count >= PARALLEL_TERMS,
                       [&](size_t j) { derived[j] = apply(layer[j / count], j % count); });
            Rows next;
            for (auto& d : derived) {
                if (d.empty()) continue;
                std::string row;
                print_row(row,d);
                if (seen.insert(row).second) {
                    result += row;
                    next.push_back(std::move(d));
                }
            }
            layer = std::move(next);
        }
        return result;
    }
    const auto unit = read_row(line(),chart);
    const size_t axes = integer();
    // The chart basis and the cuts are reduced Groebner bases for grevlex.
    auto packed = [](const Rows& basis, Order order) -> std::unique_ptr<PackedRemainders> {
        try {
            return std::make_unique<PackedRemainders>(basis, order);
        } catch (const PackingLimit&) {
            return nullptr;
        }
    };
    auto remainder = [](const std::unique_ptr<PackedRemainders>& fast, const Terms& f, const Rows& basis) {
        if (fast) {
            try {
                return fast->remainder(f);
            } catch (const PackingLimit&) {}
        }
        return divide(f, basis);
    };
    auto zero = [](const std::unique_ptr<PackedRemainders>& fast, const Terms& f, const Rows& basis) {
        if (fast) {
            try {
                return fast->zero(f);
            } catch (const PackingLimit&) {}
        }
        return reduces_to_zero(f, basis);
    };
    std::unique_ptr<PackedRemainders> packed_reducers;
    bool reducers_packed = false;
    for (size_t k = 0; k < axes; ++k) {
        const auto status = line();
        if (status == "absent") { result += "0\n"; continue; }
        if (status != "present") throw std::invalid_argument("Invalid divisor cut.");
        const auto cut = rows(chart), open_cut = rows(localized), over = rows(inverse);
        Z order;
        if (f.empty()) throw std::invalid_argument("Zero has no finite coordinate order.");
        if (!reducers_packed) {
            packed_reducers = packed(reducers, chart);
            reducers_packed = true;
        }
        const auto packed_cut = packed(cut, chart);
        const auto packed_open_cut = open_cut.empty() ? nullptr : packed(open_cut, localized);
        while (zero(packed_cut, f, cut) ||
               (!open_cut.empty() && zero(packed_open_cut, lift_inverse(f,localized,false), open_cut))) {
            auto g = remainder(packed_reducers, f, reducers);
            bool found = false;
            for (int retry = 0; retry < 12; ++retry) {
                auto q = divide(lift_inverse(g,inverse,true),over);
                if (q.empty() || fmpz_is_zero(&q.begin()->first.front().v)) {
                    f.clear();
                    for (const auto& [e,c] : q) f.emplace(Powers(e.begin()+1,e.end()),c);
                    fmpz_add_ui(&order.v,&order.v,1);
                    found = true;
                    break;
                }
                g = remainder(packed_reducers, product(g,unit), reducers);
            }
            if (!found || f.empty()) return "quotient_unavailable\n";
        }
        char* count = fmpz_get_str(nullptr,10,&order.v);
        result += count;
        result += '\n';
        flint_free(count);
    }
    print_row(result,f);
    return result;
}

std::string calculate(std::istream& input) {
    std::string header;
    std::getline(input, header);
    std::istringstream fields(header);
    std::string op;
    long n, kind, coefficient;
    if (!(fields >> op >> n >> kind >> coefficient) || n <= 0 || kind < 0 || kind > 3 ||
        (coefficient != 0 && coefficient != 1)) throw std::invalid_argument("Invalid sparse ring header.");
    Order order{static_cast<size_t>(n), static_cast<int>(kind), coefficient != 0};
    Rows rows;
    for (std::string line; std::getline(input,line);) rows.push_back(read_row(line, order));
    std::string result;
    if (op == "inverse_fraction") {
        size_t base_size;
        if (!(fields >> base_size)) throw std::invalid_argument("Missing chart inverse size.");
        return inverse_fraction(std::move(rows), base_size, order);
    }
    if (op == "frame_basis" || op == "frame_empty") {
        if (order.n < 2 || coefficient || (op == "frame_basis" && kind != 2))
            throw std::invalid_argument("Invalid open frame order.");
        return frame_result(std::move(rows), order, op == "frame_basis");
    }
    if (op == "basis") {
        for (auto& p : basis(std::move(rows))) print_row(result, p);
    } else if ((op == "divide" || op == "remainder") && !rows.empty()) {
        Terms f = std::move(rows.front());
        rows.erase(rows.begin());
        for (auto& g : rows) if (g.empty()) throw std::invalid_argument("Cannot divide by zero.");
        Rows quotients(op == "divide" ? rows.size() : 0, Terms(order));
        auto remainder = divide(std::move(f), rows, op == "divide" ? &quotients : nullptr);
        for (auto& q : quotients) print_row(result, q);
        print_row(result, remainder);
    } else throw std::invalid_argument("Invalid sparse polynomial operation.");
    return result;
}
}  // namespace
std::string implicit_frame_operation(std::istream& input, const std::string& operation) {
    return implicit_frame(input,operation == "implicit_jet",operation == "implicit_derivative_chain");
}
}  // namespace atlas

extern "C" char* atlas_wide_block(const char* input, size_t length) noexcept {
    try {
        std::istringstream stream(std::string(input,length));
        return atlas::copy_text(atlas::calculate(stream));
    } catch (const std::exception& error) {
        atlas::set_error(error.what());
    } catch (...) {
        atlas::set_error("Unknown sparse coefficient-block failure.");
    }
    return nullptr;
}

extern "C" int atlas_grlex_abi() noexcept { return 1; }
extern "C" int atlas_wide_remainder_abi() noexcept { return 1; }
