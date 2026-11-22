// Exact left Gröbner bases for the two PBW eliminations in a local b-root test.
#include "monodromy_weyl.hpp"
#include <algorithm>
#include <set>

namespace atlas::monodromy_weyl {
namespace {
using namespace sparse_json;

// Ordered products u^a s^b x^i d^j, with [s,u]=u and [d_i,x_j]=delta_ij.
// In the second elimination the u,s exponents are always zero.
struct Monomial {
    Powers e;
    Z block, parameter, total;
};
struct Order {
    bool operator()(const Monomial& a, const Monomial& b) const {
        int c = cmp(a.block, b.block);
        if (c) return c < 0;
        c = cmp(a.parameter, b.parameter);
        if (c) return c < 0;
        c = cmp(a.total, b.total);
        if (c) return c < 0;
        for (size_t i = a.e.size(); i-- > 0;) {
            c = cmp(a.e[i], b.e[i]);
            if (c) return c > 0;
        }
        return false;
    }
};
using Poly = std::map<Monomial, Q, Order>;

bool divides(const Powers& a, const Powers& b) {
    for (size_t i = 0; i < a.size(); ++i) if (a[i] > b[i]) return false;
    return true;
}
Powers quotient(Powers a, const Powers& b) {
    for (size_t i = 0; i < a.size(); ++i) a[i] -= b[i];
    return a;
}
void add(Poly& p, Monomial m, const Q& c) {
    if (!c) return;
    auto [it, inserted] = p.try_emplace(std::move(m), c);
    if (!inserted) {
        it->second += c;
        if (!it->second) p.erase(it);
    }
}
struct Pair {
    Z degree;
    size_t i, j;
    bool operator<(const Pair& b) const {
        return std::tie(degree, j, i) < std::tie(b.degree, b.j, b.i);
    }
};
class Ring {
public:
    size_t n;
    bool eliminate_u;
    Ring(size_t width, bool shift) : n(width), eliminate_u(shift) {}
    Powers one() const { return Powers(2 + 2*n); }
    Monomial monomial(Powers e) const {
        Z total = 0, block = 0;
        for (size_t i = 0; i < e.size(); ++i) {
            total += e[i];
            if (i >= 2+n) block += e[i];
        }
        if (eliminate_u) block = e[0];
        // First eliminate u, then order by the exponent of s. Moving s
        // after the total degree makes mixed Weyl operators grow rapidly.
        Z parameter = eliminate_u ? e[1] : Z(0);
        return {std::move(e), std::move(block), std::move(parameter), std::move(total)};
    }

    // Normal order d^p x^q by Leibniz, and s^a u^b = u^b(s+b)^a.
    // Expand only axes with a nontrivial commutation.
    void multiply(Poly& out, const Powers& left, const Monomial& right, const Q& scale) const {
        const Powers& r = right.e;
        Powers e = left;
        for (size_t j = 0; j < e.size(); ++j) e[j] += r[j];
        auto expand = [&](auto&& self, size_t i, const Q& coeff) -> void {
            while (i < n && (!left[2+n+i] || !r[2+i])) ++i;
            if (i == n) {
                add(out, monomial(e), coeff);
                if (left[1] && r[0]) {
                    Q c = coeff;
                    for (Z k = 1; k <= left[1]; ++k) {
                        --e[1];
                        c *= (left[1] - k + 1) * r[0];
                        c /= k;
                        add(out, monomial(e), c);
                    }
                    e[1] += left[1];
                }
                return;
            }
            self(self, i+1, coeff);
            const size_t x = 2+i, d = 2+n+i;
            const Z& bound = std::min(left[d], r[x]);
            Q c = coeff;
            for (Z k = 1; k <= bound; ++k) {
                --e[x]; --e[d];
                c *= (left[d] - k + 1) * (r[x] - k + 1);
                c /= k;
                self(self, i+1, c);
            }
            e[x] += bound;
            e[d] += bound;
        };
        expand(expand, 0, scale);
    }
    void multiple(Poly& out, const Powers& m, const Poly& p, const Q& c) const {
        for (const auto& [term, value] : p) multiply(out, m, term, c * value);
    }
    Poly normal_form(Poly p, const std::vector<Poly>& gs) const {
        Poly remainder;
        while (!p.empty()) {
            auto it = std::prev(p.end());
            const auto reducer = std::find_if(gs.begin(), gs.end(), [&](const Poly& g) {
                return divides(g.rbegin()->first.e, it->first.e);
            });
            if (reducer == gs.end()) {
                remainder.insert(p.extract(it));
            } else {
                auto m = quotient(it->first.e, reducer->rbegin()->first.e);
                const Q c = -it->second; // reducers are monic
                multiple(p, m, *reducer, c);
            }
        }
        return remainder;
    }
    // A d-free operator outside the origin's maximal ideal certifies that
    // the localization of the cyclic module at the origin is zero.
    static bool local_unit(const Poly& p) {
        return !p.empty() && p.rbegin()->first.block == 0 && p.begin()->first.total == 0;
    }

    // Completed critical pairs give the Buchberger chain criterion in these
    // PBW algebras: commutation preserves the leading product and lowers
    // every correction term. Degree ordering is only a scheduling choice.
    std::vector<Poly> basis(std::vector<Poly> input, bool stop_at_unit = false) const {
        std::vector<Poly> gs;
        std::vector<std::vector<bool>> done;
        std::set<Pair> queue;
        auto append = [&](Poly p) {
            p = normal_form(std::move(p), gs);
            if (p.empty()) return false;
            if (stop_at_unit && local_unit(p)) {
                gs = {std::move(p)};
                return true;
            }
            const Q lc = p.rbegin()->second;
            if (lc != 1) for (auto& term : p) term.second /= lc;
            const size_t j = gs.size();
            const Powers& b = p.rbegin()->first.e;
            for (size_t i = 0; i < j; ++i) {
                const Powers& a = gs[i].rbegin()->first.e;
                Z deg = 0;
                for (size_t k = 0; k < a.size(); ++k) deg += std::max(a[k], b[k]);
                queue.insert({std::move(deg), i, j});
            }
            done.emplace_back(j, false);
            gs.push_back(std::move(p));
            return false;
        };
        for (auto& g : input) if (append(std::move(g))) return gs;
        while (!queue.empty()) {
            Pair pair = *queue.begin();
            queue.erase(queue.begin());
            const size_t i = pair.i, j = pair.j;
            Powers lcm = gs[i].rbegin()->first.e;
            const auto& b = gs[j].rbegin()->first.e;
            for (size_t k = 0; k < b.size(); ++k) lcm[k] = std::max(lcm[k], b[k]);
            auto finished = [&](size_t a, size_t b) { return done[std::max(a,b)][std::min(a,b)]; };
            bool redundant = false;
            for (size_t k = 0; k < gs.size(); ++k) {
                if (k != i && k != j && finished(i,k) && finished(j,k) &&
                    divides(gs[k].rbegin()->first.e, lcm)) {
                    redundant = true;
                    break;
                }
            }
            if (!redundant) {
                Poly s;
                multiple(s, quotient(lcm, gs[i].rbegin()->first.e), gs[i], 1);
                multiple(s, quotient(lcm, b), gs[j], -1);
                if (append(std::move(s))) return gs;
            }
            done[j][i] = true;
        }
        return gs;
    }
};
Q rational_power(Q base, Z exponent) {
    Q result = 1;
    while (exponent > 0) {
        if (mpz_odd_p(exponent.get_mpz_t())) result *= base;
        exponent >>= 1;
        if (exponent) base *= base;
    }
    return result;
}
} // namespace

struct Annihilator::Data {
    Ring differential;
    Poly f;
    std::vector<Poly> partials, operators;
    explicit Data(size_t width) : differential(width, false), partials(width) {}
};

Annihilator::Annihilator(const Terms& f, std::size_t width) : data(std::make_unique<Data>(width)) {
    Ring shift(width, true);
    std::vector<Poly> graph(width+1);
    auto e = shift.one();
    e[1] = 1;
    add(graph[0], shift.monomial(e), 1);
    for (size_t i = 0; i < width; ++i) {
        e = shift.one();
        e[2+width+i] = 1;
        add(graph[i+1], shift.monomial(e), 1);
    }
    // Eliminate u from <u f+s, d_i + u f_i> to obtain Ann_{D[s]} f^s.
    for (const auto& [powers, coeff] : f) {
        if (powers.size() != width) throw std::invalid_argument("Invalid b-function exponent width.");
        if (!coeff) continue;
        e = shift.one();
        std::copy(powers.begin(), powers.end(), e.begin()+2);
        add(data->f, data->differential.monomial(e), coeff);
        e[0] = 1;
        add(graph[0], shift.monomial(e), coeff);
        for (size_t i = 0; i < width; ++i) if (powers[i]) {
            --e[2+i];
            add(graph[i+1], shift.monomial(e), coeff*powers[i]);
            e[0] = 0;
            add(data->partials[i], data->differential.monomial(e), coeff*powers[i]);
            e[0] = 1;
            ++e[2+i];
        }
    }
    if (data->f.empty()) throw std::invalid_argument("Expected a nonzero polynomial for a local b-function.");
    auto gs = shift.basis(std::move(graph));
    std::sort(gs.begin(), gs.end(), [](const Poly& a, const Poly& b) {
        return Order{}(a.rbegin()->first, b.rbegin()->first);
    });
    // Keep a minimal set of leading monomials in the elimination ideal.
    // The resulting operators are still a Gröbner basis of the annihilator.
    for (auto& p : gs) {
        if (p.rbegin()->first.block != 0) break;
        if (std::none_of(data->operators.begin(), data->operators.end(), [&](const Poly& q) {
                return divides(q.rbegin()->first.e, p.rbegin()->first.e);
            })) data->operators.push_back(std::move(p));
    }
}

Annihilator::~Annihilator() = default;

bool Annihilator::local_root(const mpq_class& alpha) const {
    const Ring& ring = data->differential;
    // (s+1) f_i = d_i f - (f d_i - s f_i), so at alpha != -1 the
    // partial derivatives belong to this left ideal from the outset.
    std::vector<Poly> specialized;
    if (alpha != -1) specialized = data->partials;
    specialized.push_back(data->f);
    for (const auto& p : data->operators) {
        Poly q;
        for (const auto& [m, coeff] : p) {
            auto e = m.e;
            Q c = coeff * rational_power(alpha, e[1]);
            e[1] = 0;
            add(q, ring.monomial(std::move(e)), c);
        }
        if (!q.empty()) specialized.push_back(std::move(q));
    }
    const auto basis = ring.basis(std::move(specialized), true);
    return std::none_of(basis.begin(), basis.end(), Ring::local_unit);
}
} // namespace atlas::monodromy_weyl
