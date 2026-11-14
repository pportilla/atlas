#include "local_basis.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <utility>

extern "C" {
void* atlas_groebner(const char*, size_t) noexcept;
void* atlas_groebner_modular(const char*, size_t, long*, double*) noexcept;
char* atlas_wide_block(const char*, size_t) noexcept;
void atlas_groebner_release(char*) noexcept;
const char* atlas_groebner_text(const void*) noexcept;
const char* atlas_groebner_error() noexcept;
void atlas_groebner_free(void*) noexcept;
}

namespace atlas::local {
namespace {

mpz_class degree(const Powers& e) {
    mpz_class d = 0;
    for (const auto& power : e) d += power;
    return d;
}

bool divides(const Powers& a, const Powers& b) {
    return std::equal(a.begin(), a.end(), b.begin(), std::less_equal<>());
}

mpz_class max_degree(const Polynomial& f) {
    mpz_class d = 0;
    for (const auto& term : f) d = std::max(d, degree(term.first));
    return d;
}

Polynomial monic(const Polynomial& f, const Powers& head) {
    Polynomial result = f;
    const auto& coefficient = f.at(head);
    if (coefficient != 1)
        for (auto& term : result) term.second /= coefficient;
    return result;
}

// Matches native_polynomial.default_sort_key on QQ expressions. Only the
// active coordinates participate, in ascending name order. Within that
// ring the tuple of (exponents, coefficient) is stored largest exponent first.
using SortKey = std::pair<std::vector<std::string>, std::vector<std::pair<Powers, mpq_class>>>;

SortKey sort_key(const Polynomial& row, const std::vector<std::string>& names) {
    std::vector<size_t> active;
    for (size_t i = 0; i < names.size(); ++i)
        if (std::any_of(row.begin(), row.end(), [i](const auto& t) { return t.first[i] != 0; }))
            active.push_back(i);
    std::sort(active.begin(), active.end(), [&](size_t i, size_t j) { return names[i] < names[j]; });
    SortKey key;
    for (size_t i : active) key.first.push_back(names[i]);
    for (const auto& [e, c] : row) {
        Powers restricted;
        for (size_t i : active) restricted.push_back(e[i]);
        key.second.emplace_back(std::move(restricted), c);
    }
    std::sort(key.second.rbegin(), key.second.rend());
    return key;
}

using Heads = std::vector<Powers>;

Heads local_heads(const Basis& rows, bool lex) {
    Heads heads;
    for (const auto& row : rows) heads.push_back(local_head(row, lex));
    return heads;
}

Basis minimal_rows(const Basis& rows, const Heads& heads, const std::vector<std::string>& names) {
    std::map<Powers, std::vector<size_t>> groups;
    for (size_t i = 0; i < heads.size(); ++i) groups[heads[i]].push_back(i);

    Heads ordered;
    for (const auto& group : groups) ordered.push_back(group.first);
    std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
        auto da = degree(a), db = degree(b);
        return da != db ? da < db : a < b;
    });
    Heads minimal;
    std::vector<std::pair<SortKey, Polynomial>> chosen;
    for (const auto& head : ordered) {
        if (std::any_of(minimal.begin(), minimal.end(), [&](const auto& h) { return divides(h, head); }))
            continue;
        minimal.push_back(head);
        std::pair<mpz_class, size_t> best_cost;
        SortKey best_key;
        Polynomial best;
        for (auto i : groups.at(head)) {
            const auto& row = rows[i];
            auto cost = std::make_pair(max_degree(row), row.size());
            if (!best.empty() && best_cost < cost) continue;
            auto candidate = monic(row, head);
            auto key = sort_key(candidate, names);
            if (best.empty() || cost < best_cost || key < best_key) {
                best_cost = cost;
                best_key = std::move(key);
                best = std::move(candidate);
            }
        }
        chosen.emplace_back(std::move(best_key), std::move(best));
    }
    std::sort(chosen.begin(), chosen.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    Basis result;
    for (auto& item : chosen) result.push_back(std::move(item.second));
    return result;
}

// In a homogeneous lift t^(d-|e|)*x^e the largest t exponent selects the
// smallest local degree. Within a t degree the global tie order applies.
Basis global_basis(const Basis& rows, size_t variables, bool lex, bool lift, bool prefer_modular) {
    const auto width = variables + static_cast<size_t>(lift);
    const int order = lex ? 1 : (lift ? 2 : 0);
    std::ostringstream input;
    input << width << ' ' << order << " 0\n";
    for (const auto& row : rows) {
        auto top = max_degree(row);
        bool first = true;
        for (const auto& [e, c] : row) {
            if (!first) input << ';';
            first = false;
            input << c.get_str();
            if (lift) input << ' ' << top - degree(e);
            for (const auto& power : e) input << ' ' << power;
        }
        input << '\n';
    }
    auto request = input.str();
    std::unique_ptr<void, decltype(&atlas_groebner_free)> basis(nullptr, atlas_groebner_free);
    if (prefer_modular)
        basis.reset(atlas_groebner_modular(request.data(), request.size(), nullptr, nullptr));
    if (!basis) basis.reset(atlas_groebner(request.data(), request.size()));
    std::string result;
    if (basis) result = atlas_groebner_text(basis.get());
    else {
        const std::string error = atlas_groebner_error();
        if (error.find("out of range") == std::string::npos && error != "Unsupported number of variables.")
            throw std::runtime_error(error);
        auto wide = "basis " + request;
        std::unique_ptr<char, decltype(&atlas_groebner_release)> reply(
            atlas_wide_block(wide.data(), wide.size()), atlas_groebner_release);
        if (!reply) throw std::runtime_error(atlas_groebner_error());
        result = reply.get();
    }
    std::istringstream output(result);
    Basis projected;
    for (std::string line; std::getline(output, line);) {
        if (line.empty()) continue;
        Polynomial row;
        std::istringstream terms(line);
        for (std::string term; std::getline(terms, term, ';');) {
            std::istringstream fields(term);
            std::string q;
            mpz_class homogenizer;
            if (!(fields >> q) || (lift && !(fields >> homogenizer)))
                throw std::runtime_error("Malformed coefficient in native standard basis.");
            Powers e(variables);
            for (auto& power : e)
                if (!(fields >> power)) throw std::runtime_error("Malformed exponent in native standard basis.");
            // Every basis row is homogeneous when lifted. Dropping its first
            // exponent therefore leaves all its monomials distinct.
            if (!row.emplace(std::move(e), mpq_class(q)).second)
                throw std::runtime_error("Duplicate monomial in native standard basis.");
        }
        projected.push_back(std::move(row));
    }
    return projected;
}

}

Powers local_head(const Polynomial& polynomial, bool lex) {
    const Powers* head = nullptr;
    mpz_class lowest = 0;
    for (const auto& [e, c] : polynomial) {
        if (c == 0) continue;
        auto d = degree(e);
        if (!head || d < lowest || (d == lowest &&
            (lex ? e > *head : std::lexicographical_compare(e.rbegin(), e.rend(), head->rbegin(), head->rend())))) {
            head = &e;
            lowest = d;
        }
    }
    if (!head) throw std::invalid_argument("The zero polynomial has no local head.");
    return *head;
}

Basis standard_basis(const Basis& generators, const std::vector<std::string>& names, bool lex, bool prefer_modular) {
    Basis rows;
    const Powers origin(names.size(), 0);
    const Basis unit = {Polynomial{{origin, 1}}};
    bool homogeneous = true;
    for (const auto& g : generators) {
        Polynomial row;
        for (const auto& [e, c] : g) {
            if (e.size() != names.size()) throw std::invalid_argument("Wrong width for a local polynomial.");
            if (c != 0) row.emplace(e, c);
        }
        if (row.empty()) continue;
        if (row.count(origin)) return unit;
        const auto d = degree(row.begin()->first);
        if (homogeneous)
            for (const auto& term : row)
                if (degree(term.first) != d) { homogeneous = false; break; }
        rows.push_back(std::move(row));
    }
    if (rows.empty()) return {};
    Heads heads = local_heads(rows, lex);
    if (rows.size() == 1) return {monic(rows[0], heads[0])};
    // A row whose head divides all its terms is a monomial times a local
    // unit. If all supplied rows have this property their heads suffice.
    bool associates = true;
    for (size_t i = 0; i < rows.size() && associates; ++i)
        for (const auto& term : rows[i])
            if (!divides(heads[i], term.first)) { associates = false; break; }
    if (associates) return minimal_rows(rows, heads, names);

    Basis certified = global_basis(rows, names.size(), lex, !homogeneous, prefer_modular);
    Heads certified_heads = local_heads(certified, lex);
    for (const auto& h : certified_heads)
        if (h == origin) return unit;
    // When input heads already generate the certified initial ideal, retain
    // the input representatives, including when a saved basis is replayed.
    for (const auto& h : certified_heads)
        if (std::none_of(heads.begin(), heads.end(), [&](const auto& m) { return divides(m, h); }))
            return minimal_rows(certified, certified_heads, names);
    return minimal_rows(rows, heads, names);
}

namespace {

// At fixed degree in two variables, lex and grevlex both prefer the
// monomial with the smaller second exponent. The keys add on monomial
// multiplication, and increase in local reduction order.
constexpr unsigned jet_stride = 128;
constexpr unsigned jet_degree_limit = 40;
static_assert(2 * jet_degree_limit < jet_stride);
using Jet = std::map<unsigned, mpq_class>;

bool jet_divides(unsigned a, unsigned b) {
    const auto ay = a % jet_stride, by = b % jet_stride;
    return ay <= by && a / jet_stride - ay <= b / jet_stride - by;
}

unsigned jet_lcm(unsigned a, unsigned b) {
    const unsigned y = std::max(a % jet_stride, b % jet_stride);
    const unsigned x = std::max(a / jet_stride - a % jet_stride, b / jet_stride - b % jet_stride);
    return (x + y) * jet_stride + y;
}

// In the plane the staircase corners occur just before each change of
// required y power. Return the first degree covered by the whole head ideal.
unsigned staircase_bound(const std::vector<unsigned>& heads) {
    std::vector<std::pair<unsigned, unsigned>> corners;
    for (auto h : heads) {
        const auto y = h % jet_stride;
        corners.emplace_back(h / jet_stride - y, y);
    }
    std::sort(corners.begin(), corners.end());
    if (corners.empty() || corners.front().first != 0) return 0;
    auto last_y = corners.front().second;
    unsigned bound = 0;
    for (const auto& [x, y] : corners) {
        if (y >= last_y) continue;
        bound = std::max(bound, x + last_y - 1);
        last_y = y;
    }
    return last_y == 0 ? bound : 0;
}

void add_jet_multiple(Jet& row, const Jet& reducer, unsigned shift, const mpq_class& scale, unsigned end) {
    for (const auto& [k, c] : reducer) {
        const unsigned m = k + shift;
        if (m >= end) break;
        mpq_class v = scale * c;
        auto [it, added] = row.try_emplace(m, std::move(v));
        if (!added) {
            it->second += v;
            if (!it->second) row.erase(it);
        }
    }
}

struct JetPair {
    unsigned lcm;
    size_t a, b;
    bool operator<(const JetPair& other) const { return lcm > other.lcm; }
};

std::optional<Basis> plane_jet_basis(const Basis& generators, unsigned cap) {
    unsigned end = (cap + 1) * jet_stride;
    std::vector<Jet> rows;
    std::vector<unsigned> heads;
    std::priority_queue<JetPair> pairs;
    size_t work = 0;
    auto insert = [&](Jet row) {
        // Ordinary local leading reduction terminates modulo m^(cap+1).
        // Stop once the head cannot be reduced; each accepted row then
        // enlarges the monomial ideal of the previous leading terms.
        while (!row.empty()) {
            unsigned h = row.begin()->first;
            size_t i = 0;
            while (i < heads.size() && !jet_divides(heads[i], h)) ++i;
            if (i == heads.size()) break;
            work += rows[i].size();
            if (work > 120000) return false;
            const mpq_class c = -row.begin()->second;
            add_jet_multiple(row, rows[i], h - heads[i], c, end);
        }
        if (row.empty()) return true;
        const unsigned h = row.begin()->first;
        const mpq_class c = row.begin()->second;
        if (c != 1)
            for (auto& term : row) term.second /= c;
        for (size_t i = 0; i < heads.size(); ++i) {
            unsigned lcm = jet_lcm(h, heads[i]);
            if (lcm < end && lcm != h + heads[i])
                pairs.push({lcm, i, heads.size()});
        }
        heads.push_back(h);
        rows.push_back(std::move(row));
        if (h / jet_stride == h % jet_stride || h % jet_stride == 0) {
            // Already known heads come from I modulo m^(cap+1), even before
            // all S pairs have been examined. As soon as they cover degree B,
            // Nakayama certifies m^B inside I and higher tails may be dropped.
            const unsigned b = staircase_bound(heads);
            if (b && b <= cap) end = std::min(end, (b + 1) * jet_stride);
        }
        return true;
    };
    for (const auto& f : generators) {
        Jet row;
        for (const auto& [m, c] : f) {
            const auto key = degree(m).get_ui() * jet_stride + m[1].get_ui();
            if (key < end && c != 0) row.emplace(key, c);
        }
        if (!insert(std::move(row))) return std::nullopt;
    }
    while (!pairs.empty()) {
        const auto pair = pairs.top();
        pairs.pop();
        if (pair.lcm >= end) break;
        // The two strictly smaller lcms give a standard representation of
        // this S pair by the Buchberger chain criterion.
        bool redundant = std::any_of(heads.begin(), heads.end(), [&](unsigned h) {
            return jet_divides(h, pair.lcm) && jet_lcm(h, heads[pair.a]) != pair.lcm
                && jet_lcm(h, heads[pair.b]) != pair.lcm;
        });
        if (redundant) continue;
        Jet s;
        add_jet_multiple(s, rows[pair.a], pair.lcm - heads[pair.a], 1, end);
        add_jet_multiple(s, rows[pair.b], pair.lcm - heads[pair.b], -1, end);
        if (!insert(std::move(s))) return std::nullopt;
    }
    // If all degree-cap monomials are initial terms, m^cap is contained in
    // I + m^(cap+1). Nakayama then gives m^cap inside I locally. Truncated
    // rows now lift to elements of I itself and the finite-jet head ideal is
    // the full local head ideal. Otherwise use the ordinary exact basis.
    unsigned bound = staircase_bound(heads);
    if (!bound || bound > cap) return std::nullopt;
    Basis result;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (std::any_of(heads.begin(), heads.end(),
                [&](unsigned h) { return h != heads[i] && jet_divides(h, heads[i]); }))
            continue;
        Polynomial p;
        for (const auto& [k, c] : rows[i]) {
            if (k != heads[i] && k / jet_stride >= bound) break;
            p.emplace(Powers{k / jet_stride - k % jet_stride, k % jet_stride}, c);
        }
        result.push_back(std::move(p));
    }
    return result;
}

}

std::optional<Basis> finite_standard_basis(const Basis& generators, const std::vector<std::string>& names, bool) {
    if (names.size() != 2 || generators.size() != 2) return std::nullopt;
    unsigned long highest = 0;
    bool inhomogeneous = false;
    bool common_x = true, common_y = true;
    for (const auto& row : generators) {
        if (row.size() < 2) return std::nullopt;
        unsigned long low = std::numeric_limits<unsigned long>::max(), high = 0;
        for (const auto& term : row) {
            if (term.first.size() != names.size()) throw std::invalid_argument("Wrong width for a local polynomial.");
            if (term.second == 0) return std::nullopt;
            common_x &= term.first[0] != 0;
            common_y &= term.first[1] != 0;
            const auto d = degree(term.first);
            if (d > jet_degree_limit) return std::nullopt;
            low = std::min(low, d.get_ui());
            high = std::max(high, d.get_ui());
        }
        if (low < 4) return std::nullopt;
        inhomogeneous |= high != low;
        highest = std::max(highest, high);
    }
    // Homogeneous systems and low-order transverse inputs are inexpensive
    // in the ordinary kernel. Reserve the finite jet for high tangencies.
    if (!inhomogeneous || common_x || common_y) return std::nullopt;
    const auto a = local_head(generators[0], true), b = local_head(generators[1], true);
    if (!divides(a, b) && !divides(b, a)) return std::nullopt;
    return plane_jet_basis(generators, 2 * highest);
}

}
