#include "local_basis.hpp"
#include "local_monomial.hpp"
#include "local_io.hpp"
#include "groebner.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <numeric>
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

Poly packed_row(const Ring& ring, const Polynomial& row, Rational* scale) {
    for (const auto& term : row)
        if (term.first.size() != static_cast<size_t>(ring.n))
            throw std::invalid_argument("Wrong width for a local polynomial.");
    auto text = encode_row(row);
    text += '\n';
    const char* at = text.data();
    return parse(ring, at, at + text.size(), scale);
}

Powers unpack_powers(const Ring& ring, const u64* term) {
    long exponents[MAX_VARIABLES];
    ring.unpack(term, exponents);
    return Powers(exponents, exponents + ring.n);
}

Polynomial unpack_row(const Ring& ring, const Poly& row, const fmpz* numerator,
                      const fmpz* denominator) {
    Polynomial result;
    mpq_class scale;
    fmpz_get_mpz(mpq_numref(scale.get_mpq_t()), numerator);
    fmpz_get_mpz(mpq_denref(scale.get_mpq_t()), denominator);
    scale.canonicalize();
    mpz_class integer;
    for (size_t i = 0; i < row.size(); ++i) {
        fmpz_get_mpz(integer.get_mpz_t(), &row.c[i].v);
        result.emplace(unpack_powers(ring, row.m.data() + i * ring.stride), scale * integer);
    }
    return result;
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
struct OrderedRow {
    Polynomial row;
    std::vector<std::string> variables;
    std::vector<std::pair<Powers, mpq_class>> reordered_terms;

    OrderedRow(Polynomial polynomial, const std::vector<std::string>& names, bool names_sorted)
        : row(std::move(polynomial)) {
        std::vector<size_t> active;
        for (size_t i = 0; i < names.size(); ++i)
            if (std::any_of(row.begin(), row.end(), [i](const auto& t) { return t.first[i] != 0; }))
                active.push_back(i);
        if (!names_sorted)
            std::sort(active.begin(), active.end(), [&](size_t i, size_t j) { return names[i] < names[j]; });
        for (size_t i : active) variables.push_back(names[i]);
        if (names_sorted) return;
        for (const auto& [powers, coefficient] : row) {
            Powers restricted;
            for (size_t i : active) restricted.push_back(powers[i]);
            reordered_terms.emplace_back(std::move(restricted), coefficient);
        }
        std::sort(reordered_terms.rbegin(), reordered_terms.rend());
    }

    bool operator<(const OrderedRow& other) const {
        if (variables != other.variables) return variables < other.variables;
        if (!reordered_terms.empty()) return reordered_terms < other.reordered_terms;
        // With the same active variables, all omitted coordinates are zero
        // in both rows. Reverse map iteration already has the required order.
        return std::lexicographical_compare(row.rbegin(), row.rend(), other.row.rbegin(), other.row.rend());
    }
};

Basis sorted_rows(std::vector<OrderedRow> rows) {
    std::sort(rows.begin(), rows.end());
    Basis result;
    result.reserve(rows.size());
    for (auto& entry : rows) result.push_back(std::move(entry.row));
    return result;
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
        auto da = total_degree(a), db = total_degree(b);
        return da != db ? da < db : a < b;
    });
    Heads minimal;
    const bool names_sorted = std::is_sorted(names.begin(), names.end());
    std::vector<OrderedRow> chosen;
    for (const auto& head : ordered) {
        if (std::any_of(minimal.begin(), minimal.end(), [&](const auto& h) { return monomial_divides(h, head); }))
            continue;
        minimal.push_back(head);
        std::pair<mpz_class, size_t> best_cost;
        std::optional<OrderedRow> best;
        for (auto i : groups.at(head)) {
            const auto& row = rows[i];
            auto cost = std::make_pair(max_total_degree(row), row.size());
            if (best && best_cost < cost) continue;
            OrderedRow candidate(monic(row, head), names, names_sorted);
            if (!best || cost < best_cost || candidate < *best) {
                best_cost = cost;
                best = std::move(candidate);
            }
        }
        chosen.push_back(std::move(*best));
    }
    return sorted_rows(std::move(chosen));
}

// In a homogeneous lift t^(d-|e|)*x^e the largest t exponent selects the
// smallest local degree. Within a t degree the global tie order applies.
Basis global_basis(const Basis& rows, size_t variables, bool lex, bool lift, bool prefer_modular) {
    const auto width = variables + static_cast<size_t>(lift);
    const int order = lex ? 1 : (lift ? 2 : 0);
    std::ostringstream input;
    input << width << ' ' << order << " 0\n";
    for (const auto& row : rows) {
        auto top = max_total_degree(row);
        bool first = true;
        for (const auto& [e, c] : row) {
            if (!first) input << ';';
            first = false;
            input << c.get_str();
            if (lift) input << ' ' << top - total_degree(e);
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

// Pack rational rows once for leading-term queries. A lift by total degree
// gives the local order. With lift disabled the same engine uses its global order.
bool compute_packed_basis(Engine& engine, const Basis& rows, size_t width, bool lex,
                          bool lift, bool prefer_modular) {
    if (!width || width + lift > MAX_VARIABLES) return false;
    Ring ring;
    ring.init(static_cast<int>(width + lift), lex ? 1 : (lift ? 2 : 0));
    std::vector<Poly> polys;
    for (const auto& row : rows) if (!row.empty()) {
        const auto top = max_total_degree(row);
        if (top > MAX_DEGREE) return false;
        std::vector<u64> packed(row.size() * ring.stride);
        std::vector<const mpq_class*> coefficients;
        coefficients.reserve(row.size());
        mpz_class denominator = 1;
        for (const auto& [e, q] : row) {
            long powers[MAX_VARIABLES];
            long d = 0;
            for (size_t i = 0; i < width; ++i) {
                powers[i + lift] = e[i].get_si();
                d += powers[i + lift];
            }
            if (lift) powers[0] = top.get_si() - d;
            ring.pack(packed.data() + coefficients.size() * ring.stride, powers);
            mpz_lcm(denominator.get_mpz_t(), denominator.get_mpz_t(), mpq_denref(q.get_mpq_t()));
            coefficients.push_back(&q);
        }
        std::vector<size_t> order(row.size());
        std::iota(order.begin(), order.end(), 0);
        auto monomial = [&](size_t i) { return packed.data() + i * ring.stride; };
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return ring.compare(monomial(a), monomial(b)) > 0;
        });
        Poly poly;
        poly.m.reserve(packed.size());
        poly.c.reserve(row.size());
        mpz_class integer;
        for (const auto i : order) {
            const auto q = coefficients[i]->get_mpq_t();
            mpz_divexact(integer.get_mpz_t(), denominator.get_mpz_t(), mpq_denref(q));
            mpz_mul(integer.get_mpz_t(), integer.get_mpz_t(), mpq_numref(q));
            poly.m.insert(poly.m.end(), monomial(i), monomial(i) + ring.stride);
            poly.c.emplace_back();
            fmpz_set_mpz(&poly.c.back().v, integer.get_mpz_t());
        }
        primitive(poly, nullptr);
        polys.push_back(std::move(poly));
    }
    if (!prefer_modular || !modular_basis(ring, polys, engine)) {
        engine.R = ring;
        engine.compute(polys, 0);
    }
    return true;
}

}

std::optional<std::vector<Powers>> packed_lex_heads(const Basis& rows, size_t width) {
    try {
        Engine engine;
        if (!compute_packed_basis(engine, rows, width, true, false, false)) return std::nullopt;
        std::vector<Powers> heads;
        heads.reserve(engine.G.size());
        for (const auto& g : engine.G) heads.push_back(unpack_powers(engine.R, g.p.m.data()));
        return heads;
    } catch (const Limit&) {
        return std::nullopt;
    }
}

// A dimension query only needs the supports of the leading monomials. Keep
// the rational input and the certified result in memory across this boundary.
std::optional<std::vector<std::vector<int>>> local_leading_supports(
        const Basis& rows, size_t width, bool lex, bool prefer_modular) {
    if (!width || width > MAX_VARIABLES) return std::nullopt;
    // A monomial times a local unit generates the same localized ideal as
    // that monomial. Preserve the shortcut used by the full basis routine.
    std::vector<std::vector<int>> supports;
    bool associates = true;
    for (const auto& row : rows) if (!row.empty()) {
        const auto head = local_head(row, lex);
        if (std::any_of(row.begin(), row.end(), [&](const auto& t) { return !monomial_divides(head, t.first); })) {
            associates = false;
            break;
        }
        std::vector<int> support;
        for (size_t i = 0; i < width; ++i) if (head[i] != 0) support.push_back(i);
        supports.push_back(std::move(support));
    }
    if (associates) return supports;
    supports.clear();
    bool lift = false;
    for (const auto& row : rows) if (!row.empty()) {
        const auto d = total_degree(row.begin()->first);
        for (const auto& term : row) if (total_degree(term.first) != d) { lift = true; break; }
        if (lift) break;
    }
    try {
        Engine engine;
        if (!compute_packed_basis(engine, rows, width, lex, lift, prefer_modular)) return std::nullopt;
        const auto& ring = engine.R;
        for (const auto& g : engine.G) {
            // Every lifted row is homogeneous. Suppressing t turns its
            // leading term into its local leading term in the original axes.
            std::vector<int> support;
            for (size_t i = 0; i < width; ++i)
                if (Ring::field(g.p.m.data(), ring.position(i + lift))) support.push_back(i);
            if (support.empty()) throw std::logic_error("On-scheme local basis contains a unit.");
            supports.push_back(std::move(support));
        }
        return supports;
    } catch (const Limit&) {
        return std::nullopt;
    }
}

HomogeneousReducer::HomogeneousReducer(std::unique_ptr<Engine> engine) : engine_(std::move(engine)) {
    const auto& ring = engine_->R;
    std::vector<unsigned long> bounds(ring.n, 0);
    for (const auto& g : engine_->G) {
        binomial_ |= g.p.size() == 2;
        const u64* head = g.p.m.data();
        for (int i = 0; i < ring.n; ++i) {
            // A pure power has its entire total degree on one axis.
            if (Ring::field(head, ring.position(i)) == head[0]) {
                auto d = static_cast<unsigned long>(head[0]);
                if (!bounds[i] || d < bounds[i]) bounds[i] = d;
                break;
            }
        }
    }
    unsigned long cutoff = 1;
    for (auto bound : bounds) {
        if (!bound) return;
        cutoff += bound - 1;
    }
    cutoff_ = cutoff;
}
HomogeneousReducer::HomogeneousReducer(HomogeneousReducer&&) noexcept = default;
HomogeneousReducer::~HomogeneousReducer() = default;

std::optional<HomogeneousReducer> homogeneous_reducer(const Basis& centered, size_t width,
                                                      bool lex, bool prefer_modular) {
    if (width == 0 || width > MAX_VARIABLES || centered.size() < 2) return std::nullopt;
    size_t count = 0;
    bool nonmonomial = false;
    for (const auto& f : centered) {
        if (f.empty()) continue;
        const auto d = total_degree(f.begin()->first);
        if (d == 0 || d > MAX_DEGREE) return std::nullopt;
        for (const auto& [e, c] : f)
            if (e.size() != width || c == 0 || total_degree(e) != d) return std::nullopt;
        ++count;
        nonmonomial |= f.size() > 1;
    }
    if (count < 2 || !nonmonomial) return std::nullopt;

    Ring ring;
    ring.init(static_cast<int>(width), lex ? 1 : 0);
    try {
        std::vector<Poly> polys;
        polys.reserve(count);
        for (const auto& f : centered)
            if (!f.empty()) polys.push_back(packed_row(ring, f, nullptr));

        auto engine = std::make_unique<Engine>();
        if (prefer_modular) {
            try {
                if (modular_basis(ring, polys, *engine))
                    return HomogeneousReducer(std::move(engine));
            } catch (const Limit&) {
                // The homogenized modular computation can exceed the packed limits.
            }
            engine = std::make_unique<Engine>();
        }
        engine->R = ring;
        engine->compute(polys, 0);
        return HomogeneousReducer(std::move(engine));
    } catch (const Limit&) {
        return std::nullopt;
    }
}

Basis HomogeneousReducer::materialize() const {
    Basis basis;
    basis.reserve(engine_->G.size());
    Z one;
    fmpz_one(&one.v);
    for (const auto& g : engine_->G)
        basis.push_back(unpack_row(engine_->R, g.p, &one.v, &g.p.c.front().v));
    return basis;
}

std::optional<Polynomial> HomogeneousReducer::reduce(const Polynomial& centered_query) const {
    const size_t width = static_cast<size_t>(engine_->R.n);
    // Filter before packing. Even a query outside the packed range can vanish
    // modulo a finite homogeneous ideal. Only allocate a filtered row on demand.
    Polynomial filtered;
    bool dropped = false;
    for (auto it = centered_query.begin(); it != centered_query.end(); ++it) {
        if (it->first.size() != width) throw std::invalid_argument("Wrong width for a local polynomial.");
        const auto d = total_degree(it->first);
        if (cutoff_ && d >= *cutoff_) {
            if (!dropped) filtered.insert(centered_query.begin(), it);
            dropped = true;
        } else {
            // The local binomial reducer can jump over many repeated divisions.
            if ((binomial_ && d >= 128) || d > MAX_DEGREE) return std::nullopt;
            if (dropped) filtered.insert(filtered.end(), *it);
        }
    }
    if (dropped && filtered.empty()) return Polynomial{};
    try {
        Rational scale;
        Poly f = packed_row(engine_->R, dropped ? filtered : centered_query, &scale);
        engine_->reduce(f, 0, &scale.num.v, &scale.den.v);
        return unpack_row(engine_->R, f, &scale.num.v, &scale.den.v);
    } catch (const Limit&) {
        return std::nullopt;
    }
}

bool global_less(const Powers& a, const Powers& b, bool lex) {
    if (lex) return a < b;
    const auto da = total_degree(a), db = total_degree(b);
    if (da != db) return da < db;
    return std::lexicographical_compare(b.rbegin(), b.rend(), a.rbegin(), a.rend());
}

const Powers& global_head(const Polynomial& f, bool lex) {
    if (lex) return f.rbegin()->first;
    return std::max_element(f.begin(), f.end(), [](const auto& a, const auto& b) {
        return global_less(a.first,b.first,false);
    })->first;
}

Powers local_head(const Polynomial& polynomial, bool lex) {
    const Powers* head = nullptr;
    mpz_class lowest = 0;
    for (const auto& [e, c] : polynomial) {
        if (c == 0) continue;
        auto d = total_degree(e);
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
    bool used_finite_jet;
    return standard_basis(generators, names, lex, prefer_modular, used_finite_jet);
}

Basis standard_basis(const Basis& generators, const std::vector<std::string>& names, bool lex,
                     bool prefer_modular, bool& used_finite_jet) {
    auto finite = finite_standard_basis(generators, names, lex);
    used_finite_jet = finite.has_value();
    if (finite) return std::move(*finite);
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
        const auto d = total_degree(row.begin()->first);
        if (homogeneous)
            for (const auto& term : row)
                if (total_degree(term.first) != d) { homogeneous = false; break; }
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
            if (!monomial_divides(heads[i], term.first)) { associates = false; break; }
    if (associates) return minimal_rows(rows, heads, names);

    Basis certified = global_basis(rows, names.size(), lex, !homogeneous, prefer_modular);
    Heads certified_heads = local_heads(certified, lex);
    for (const auto& h : certified_heads)
        if (h == origin) return unit;
    // When input heads already generate the certified initial ideal, retain
    // the input representatives, including when a saved basis is replayed.
    for (const auto& h : certified_heads)
        if (std::none_of(heads.begin(), heads.end(), [&](const auto& m) { return monomial_divides(m, h); }))
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

void make_jet_monic(Jet& row) {
    const mpq_class coefficient = row.begin()->second;
    if (coefficient != 1)
        for (auto& term : row) term.second /= coefficient;
}

std::optional<Basis> plane_jet_basis(const Basis& generators, const std::vector<std::string>& names, unsigned cap) {
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
        make_jet_monic(row);
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
    std::array<Jet, 2> input;
    for (size_t i = 0; i < input.size(); ++i) {
        for (const auto& [m, c] : generators[i]) {
            const auto key = total_degree(m).get_ui() * jet_stride + m[1].get_ui();
            input[i].emplace(key, c);
        }
    }
    // Insertion order can change the tails of a local standard basis.
    // Use the local heads, breaking ties by the monic input polynomials.
    if (input[0].begin()->first == input[1].begin()->first) {
        for (auto& row : input) make_jet_monic(row);
        if (input[1] < input[0]) input[0].swap(input[1]);
    } else if (input[1].begin()->first < input[0].begin()->first) {
        input[0].swap(input[1]);
    }
    for (auto& row : input) {
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
    std::vector<OrderedRow> result;
    const bool names_sorted = std::is_sorted(names.begin(), names.end());
    for (size_t i = 0; i < rows.size(); ++i) {
        if (std::any_of(heads.begin(), heads.end(),
                [&](unsigned h) { return h != heads[i] && jet_divides(h, heads[i]); }))
            continue;
        Polynomial p;
        for (const auto& [k, c] : rows[i]) {
            if (k != heads[i] && k / jet_stride >= bound) break;
            p.emplace(Powers{k / jet_stride - k % jet_stride, k % jet_stride}, c);
        }
        result.emplace_back(std::move(p), names, names_sorted);
    }
    return sorted_rows(std::move(result));
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
            const auto d = total_degree(term.first);
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
    if (!monomial_divides(a, b) && !monomial_divides(b, a)) return std::nullopt;
    return plane_jet_basis(generators, names, 2 * highest);
}

}
