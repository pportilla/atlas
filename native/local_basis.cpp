#include "local_basis.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

extern "C" {
void* atlas_groebner(const char*, size_t) noexcept;
const char* atlas_groebner_text(const void*) noexcept;
const char* atlas_groebner_error() noexcept;
void atlas_groebner_free(void*) noexcept;
}

namespace atlas::local {
namespace {

unsigned long degree(const Powers& e) {
    unsigned long d = 0;
    for (auto power : e) {
        if (power > std::numeric_limits<unsigned long>::max() - d)
            throw std::overflow_error("Local monomial degree exceeds the native range.");
        d += power;
    }
    return d;
}

bool divides(const Powers& a, const Powers& b) {
    return std::equal(a.begin(), a.end(), b.begin(), std::less_equal<unsigned long>());
}

unsigned long max_degree(const Polynomial& f) {
    unsigned long d = 0;
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
        std::pair<unsigned long, size_t> best_cost;
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
Basis global_basis(const Basis& rows, size_t variables, bool lex, bool lift) {
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
            for (auto power : e) input << ' ' << power;
        }
        input << '\n';
    }
    auto request = input.str();
    std::unique_ptr<void, decltype(&atlas_groebner_free)> basis(
        atlas_groebner(request.data(), request.size()), atlas_groebner_free);
    if (!basis) {
        std::string error = atlas_groebner_error();
        if (error.find("out of range") != std::string::npos || error == "Unsupported number of variables.")
            throw std::overflow_error(error);
        throw std::runtime_error(error);
    }
    std::istringstream output(atlas_groebner_text(basis.get()));
    Basis projected;
    for (std::string line; std::getline(output, line);) {
        if (line.empty()) continue;
        Polynomial row;
        std::istringstream terms(line);
        for (std::string term; std::getline(terms, term, ';');) {
            std::istringstream fields(term);
            std::string q;
            unsigned long homogenizer;
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
    unsigned long lowest = 0;
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

Basis standard_basis(const Basis& generators, const std::vector<std::string>& names, bool lex) {
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

    Basis certified = global_basis(rows, names.size(), lex, !homogeneous);
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

}
