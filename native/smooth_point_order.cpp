// Order and directional vanishing in the completed local ring of a smooth QQ frame.
#include "smooth_point_order.hpp"
#include "local_division.hpp"
#include "local_io.hpp"
#include "local_monomial.hpp"

#include <algorithm>

namespace atlas::local {
namespace {
constexpr unsigned degree_limit = 32;
constexpr size_t height_limit = 16384;

struct Inconclusive {};
struct Budget {
    size_t work = 350000;
    void (*check_time)() = nullptr;
    void spend(size_t n) {
        if (check_time) check_time();
        if (n > work) throw Inconclusive{};
        work -= n;
    }
};

size_t bits(const mpz_class& x) { return mpz_sizeinbase(x.get_mpz_t(), 2); }
size_t height(const mpq_class& q) {
    return std::max(bits(q.get_num()), bits(q.get_den()));
}

void check_height(const mpq_class& value) {
    if (height(value) > height_limit) throw Inconclusive{};
}

// Bound the size of the Taylor translation, including rational powers at
// shifted points and the common denominator needed to add contributions.
Polynomial centered_jet(const Polynomial& f, const std::vector<mpq_class>& p,
                        const std::vector<bool>& fixed, unsigned degree, Budget& budget) {
    mpz_class denominator = 1;
    std::vector<unsigned> max_exponent(p.size());
    for (const auto& [e, c] : f) {
        budget.spend(1);
        if (e.size() != p.size()) throw Inconclusive{};
        mpz_class base_degree = 0;
        for (size_t i = 0; i < p.size(); ++i)
            if (p[i] == 0) base_degree += e[i];
        if (base_degree > degree) continue;
        const unsigned room = degree - base_degree.get_ui();
        size_t box = 1;
        for (size_t i = 0; i < p.size(); ++i) if (p[i] != 0 && e[i] != 0) {
            if (abs(p[i]) != 1) {
                if (e[i] > height_limit / height(p[i])) throw Inconclusive{};
                max_exponent[i] = std::max(max_exponent[i], static_cast<unsigned>(e[i].get_ui()));
            }
            // binomial(e,k) has at most k*bits(e) bits for k <= room.
            unsigned last = e[i] < room ? e[i].get_ui() : room;
            if (last && bits(e[i]) > height_limit / last) throw Inconclusive{};
            if (box > budget.work / (last + 1)) throw Inconclusive{};
            box *= last + 1;
        }
        budget.spend(box);
        check_height(c);
        mpz_lcm(denominator.get_mpz_t(), denominator.get_mpz_t(), c.get_den_mpz_t());
        if (bits(denominator) > height_limit) throw Inconclusive{};
    }
    size_t scale_bits = bits(denominator);
    for (size_t i = 0; i < p.size(); ++i) scale_bits += max_exponent[i] * height(p[i]);
    if (scale_bits > height_limit) throw Inconclusive{};
    auto jet = point_jet_below(degree + 1, f, p);
    for (auto it = jet.begin(); it != jet.end();) {
        bool discard = false;
        for (size_t i = 0; i < fixed.size(); ++i)
            if (fixed[i] && it->first[i] != 0) discard = true;
        if (discard) it = jet.erase(it);
        else { check_height(it->second); ++it; }
    }
    return jet;
}

// Eliminate the linear part of the frame equations over QQ. Every pivot
// coordinate becomes a linear form in free parameters plus terms of degree
// at least two. The allowed pivots exclude all prescribed free parameters.
std::vector<size_t> echelon(Basis& rows, const std::vector<bool>& allowed, Budget& budget) {
    const size_t width = allowed.size();
    std::vector<size_t> pivots;
    for (size_t i = 0; i < width && pivots.size() < rows.size(); ++i) if (allowed[i]) {
        Powers unit(width);
        unit[i] = 1;
        const size_t rank = pivots.size();
        auto found = std::find_if(rows.begin() + rank, rows.end(),
                                  [&](const Polynomial& row) { return row.count(unit); });
        if (found == rows.end()) continue;
        std::swap(rows[rank], *found);
        auto& pivot = rows[rank];
        const mpq_class lead = pivot.at(unit);
        budget.spend(pivot.size());
        for (auto& term : pivot) {
            term.second /= lead;
            check_height(term.second);
        }
        for (size_t j = 0; j < rows.size(); ++j) if (j != rank) {
            auto& row = rows[j];
            auto at = row.find(unit);
            if (at == row.end()) continue;
            const mpq_class factor = at->second;
            budget.spend(pivot.size());
            for (const auto& [m, c] : pivot) {
                auto entry = row.try_emplace(m, 0).first;
                entry->second -= factor * c;
                if (entry->second == 0) row.erase(entry);
                else check_height(entry->second);
            }
        }
        pivots.push_back(i);
    }
    if (pivots.size() != rows.size()) throw Inconclusive{};
    return pivots;
}

struct Monomial {
    unsigned degree, pivot_degree;
    std::vector<unsigned char> powers;
    bool operator<(const Monomial& b) const {
        if (degree != b.degree) return degree < b.degree;
        if (pivot_degree != b.pivot_degree) return pivot_degree > b.pivot_degree;
        return powers < b.powers;
    }
};

Monomial monomial(const Powers& powers, const std::vector<size_t>& pivots) {
    Monomial m{0, 0, {}};
    m.powers.reserve(powers.size());
    for (const auto& e : powers) {
        m.powers.push_back(static_cast<unsigned char>(e.get_ui()));
        m.degree += m.powers.back();
    }
    for (auto i : pivots) m.pivot_degree += m.powers[i];
    return m;
}

// Reduce in QQ[[parameters]]/m^(d+1). A rewrite either raises the total
// degree or strictly decreases the number of pivot factors at that degree.
// Thus the first surviving pure-parameter monomial certifies the exact order.
unsigned jet_order(const Basis& frame, const std::vector<mpq_class>& point,
                   const Polynomial& f, unsigned d, const std::vector<bool>& fixed,
                   const std::vector<bool>& allowed, Budget& budget, Polynomial* full = nullptr) {
    const size_t width = point.size();
    Basis rows;
    for (const auto& row : frame) {
        rows.push_back(centered_jet(row, point, fixed, std::max(1u, d), budget));
        if (rows.back().count(Powers(width))) throw Inconclusive{};
    }
    const auto pivots = echelon(rows, allowed, budget);
    using Terms = std::map<Monomial, mpq_class>;
    std::vector<Terms> tails(pivots.size());
    for (size_t j = 0; j < pivots.size(); ++j) {
        for (const auto& [e, c] : rows[j]) {
            auto m = monomial(e, pivots);
            if (m.degree == 1 && m.powers[pivots[j]]) continue;
            tails[j].emplace(std::move(m), c);
        }
    }
    Terms pending;
    for (auto& [e, c] : centered_jet(f, point, fixed, d, budget))
        pending.emplace(monomial(e, pivots), std::move(c));
    while (!pending.empty()) {
        auto term = pending.extract(pending.begin());
        auto& m = term.key();
        if (m.pivot_degree == 0) {
            if (!full) return m.degree;
            Powers parameters;
            for (size_t i = 0; i < width; ++i)
                if (!allowed[i]) parameters.emplace_back(m.powers[i]);
            full->emplace(std::move(parameters), std::move(term.mapped()));
            continue;
        }
        size_t j = 0;
        while (!m.powers[pivots[j]]) ++j;
        --m.powers[pivots[j]];
        --m.pivot_degree;
        --m.degree;
        const auto& tail = tails[j];
        budget.spend(tail.size() + 1);
        for (const auto& [a, c] : tail) {
            if (m.degree + a.degree > d) break;
            auto n = m;
            n.degree += a.degree;
            n.pivot_degree += a.pivot_degree;
            for (size_t i = 0; i < width; ++i) n.powers[i] += a.powers[i];
            auto entry = pending.try_emplace(std::move(n), 0).first;
            entry->second -= term.mapped() * c;
            if (entry->second == 0) pending.erase(entry);
            else check_height(entry->second);
        }
    }
    return d + 1;
}

// For a small affine polynomial a single full FLINT translation costs less
// than rebuilding its successive jets. Bound both the expanded support and
// rational height before choosing this path.
bool small_affine_translation(const Polynomial& f, const std::vector<mpq_class>& point) {
    if (f.size() > 128 || point.size() > 8) return false;
    for (const auto& q : point) if (height(q) > 64) return false;
    size_t remaining = 8192;
    for (const auto& [e, c] : f) {
        if (e.size() != point.size() || height(c) > 256) return false;
        size_t terms = 1;
        for (size_t i = 0; i < e.size(); ++i) if (point[i] != 0) {
            if (e[i] > 32 || terms > remaining / (e[i].get_ui() + 1)) return false;
            terms *= e[i].get_ui() + 1;
        }
        if (terms > remaining) return false;
        remaining -= terms;
    }
    return true;
}

bool suitable(const Basis& frame, const std::vector<mpq_class>& point) {
    return point.size() <= 24 && frame.size() <= point.size() &&
           std::all_of(point.begin(), point.end(), [](const auto& q) { return height(q) <= height_limit; });
}
} // namespace

std::optional<unsigned> smooth_point_order(const Basis& frame, const std::vector<mpq_class>& point,
                                         const Polynomial& f, unsigned max_degree) {
    if (!suitable(frame, point) || f.empty()) return std::nullopt;
    const unsigned limit = std::min(max_degree, degree_limit);
    if (frame.empty() && small_affine_translation(f, point)) {
        const auto centered = translate({f}, point);
        mpz_class order = limit + 1;
        for (const auto& [e, c] : centered.front()) order = std::min(order, total_degree(e));
        if (order <= limit) return static_cast<unsigned>(order.get_ui());
        return std::nullopt;
    }
    const std::vector<bool> fixed(point.size(), false), allowed(point.size(), true);
    Budget budget;
    try {
        for (unsigned d = std::min(1u, limit);; d = std::min(limit, 2*d + 1)) {
            const unsigned order = jet_order(frame, point, f, d, fixed, allowed, budget);
            if (order <= d) return order;
            if (d == limit) return std::nullopt;
        }
    } catch (const Inconclusive&) {
        return std::nullopt;
    }
}

std::optional<Polynomial> smooth_point_jet(const Basis& frame, const std::vector<mpq_class>& point,
                                           const Polynomial& f, const std::vector<bool>& dependent,
                                           unsigned degree, size_t work, void (*check_time)()) {
    if (!suitable(frame, point) || degree > degree_limit || dependent.size() != point.size() ||
        static_cast<size_t>(std::count(dependent.begin(), dependent.end(), true)) != frame.size())
        return std::nullopt;
    Budget budget{work, check_time};
    Polynomial result;
    try {
        jet_order(frame, point, f, degree, std::vector<bool>(point.size()), dependent, budget, &result);
    } catch (const Inconclusive&) {
        return std::nullopt;
    }
    return result;
}

bool smooth_point_direction_zero(const Basis& frame, const std::vector<mpq_class>& point,
                                 const Polynomial& f, const std::vector<size_t>& coordinates,
                                 size_t axis, unsigned degree) {
    const size_t width = point.size();
    if (!suitable(frame, point) || degree > degree_limit ||
        coordinates.size() != width - frame.size() || axis >= width) return false;
    std::vector<bool> fixed(width, false), allowed(width, true);
    for (auto i : coordinates) {
        if (i >= width || !allowed[i]) return false;
        allowed[i] = false;
        fixed[i] = i != axis;
    }
    if (allowed[axis]) return false;
    Budget budget;
    try {
        return jet_order(frame, point, f, degree, fixed, allowed, budget) > degree;
    } catch (const Inconclusive&) {
        return false;
    }
}
} // namespace atlas::local
