// One stage of the blowups of engine/local_blowups.py in one native call.
//
// "stage": for every chart that meets the chosen centre, the blowup of the
// centre. A centre x_i = phi_i(parameters) is blown up by its explicit
// charts (op "blowup" of native/ideal.cpp). Any other centre is first given
// few generators modulo the chart and blown up in graph form (op "charts"):
// on each chart of the blowup every pulled-back divisor is divided by the
// largest power of the new divisor (one parent identity from
// "centre_power_transforms", certified by "certify_exceptional_orders", or
// the strict ideal of "graph_strict_part" / "strict_parts" with a residual
// dimension test), the components still meeting the chart are kept, the new
// divisor gets its numerical data ("exceptional_data"), every ideal gets few
// generators, and the coordinates that a linear chart equation with a
// scalar coefficient determines are eliminated ("pullbacks"). With
// `retains` the blown-up chart stays for its points off the centre.
//
// "failures": for every chart, whether it still owns a point over the fibre
// and, when asked, the loci over the fibre where the divisors fail normal
// crossings ("crossings"), with the part of each in the chart's own region.
//
// Every step calls the native operation that the Python orchestration used
// to call one at a time, with the same request text, in the same order and
// with the same fallbacks, so the charts are those of the step-by-step
// computation. Polynomials are sent in the canonical term order of
// engine/native_polynomial.py. A packed operation that is out of range
// declines the whole call; the caller then takes the arbitrary-width route.
#include "chart_expr.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace {
using V = Json::Value;
using atlas::local::Polynomial;
using atlas::local::Powers;
using Ideal = std::vector<Polynomial>;
using atlas::chart::Declined;
using atlas::chart::MAX_VARIABLES;
using atlas::chart::Raise;
using atlas::chart::Released;
using atlas::chart::Strings;
using atlas::chart::Unavailable;
using atlas::chart::affine_request;
using atlas::chart::binomial_request;
using atlas::chart::ideal_request;
using atlas::chart::level;
using atlas::chart::out_of_range;
using atlas::chart::split_lines;

// ---------------------------------------------------------------- rings

// The coordinates of a chart: printed names and identities order the terms
// as engine/native_polynomial.py does; `sources` say where each coordinate
// comes from (an index of the input chart, or the name of a new one).
struct Ring {
    Strings names;
    std::vector<long long> identities;
    std::vector<V> sources;
    long theta = -1;
    std::vector<size_t> order;  // positions sorted by (name, identity)
    Strings header;             // names in affine expressions
    Strings keys;               // zero-padded ranks for the generator choice

    size_t size() const { return names.size(); }

    void finish() {
        const size_t n = names.size();
        order.resize(n);
        for (size_t i = 0; i < n; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            if (names[a] != names[b]) return names[a] < names[b];
            return identities[a] < identities[b];
        });
        bool valid = std::set<std::string>(names.begin(), names.end()).size() == n;
        for (const auto& name : names) {
            if (name.empty() || !std::isalpha(static_cast<unsigned char>(name[0]))) valid = false;
            for (char c : name)
                if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') valid = false;
        }
        header.clear();
        for (size_t i = 0; i < n; ++i) header.push_back(valid ? names[i] : "atlasVar" + std::to_string(i));
        const size_t width = std::to_string(n).size();
        keys.assign(n, "");
        for (size_t rank = 0; rank < n; ++rank) {
            std::string key = std::to_string(rank);
            keys[order[rank]] = std::string(width > key.size() ? width - key.size() : 0, '0') + key;
        }
    }
    std::string header_line() const {
        std::string out;
        for (size_t i = 0; i < header.size(); ++i) out += (i ? " " : "") + header[i];
        return out;
    }
};

Polynomial decode(const Ring& R, const std::string& row) { return atlas::local::decode_row(row, R.size()); }

Ideal decode_all(const Ring& R, const Strings& rows, size_t from = 0, size_t count = std::string::npos) {
    Ideal out;
    const size_t end = count == std::string::npos ? rows.size() : from + count;
    if (end > rows.size()) throw Raise{"ArithmeticError", "A native answer is shorter than expected."};
    for (size_t i = from; i < end; ++i) out.push_back(decode(R, rows[i]));
    return out;
}

std::vector<const Polynomial::value_type*> sorted_terms(const Ring& R, const Polynomial& f, bool by_degree) {
    std::vector<const Polynomial::value_type*> terms;
    for (const auto& term : f) terms.push_back(&term);
    std::sort(terms.begin(), terms.end(), [&](auto a, auto b) {
        if (by_degree) {
            mpz_class da = 0, db = 0;
            for (const auto& e : a->first) da += e;
            for (const auto& e : b->first) db += e;
            if (da != db) return da > db;
        }
        for (size_t k : R.order)
            if (a->first[k] != b->first[k]) return a->first[k] > b->first[k];
        return false;
    });
    return terms;
}

// The sparse row of engine/native_polynomial.py `row`: terms in descending
// order of their exponents in the sorted coordinates.
std::string row(const Ring& R, const Polynomial& f) {
    std::string out;
    bool first = true;
    for (const auto* term : sorted_terms(R, f, false)) {
        if (!first) out += ';';
        first = false;
        out += term->second.get_str();
        for (const auto& e : term->first) out += " " + e.get_str();
    }
    return out;
}

Strings rows(const Ring& R, const Ideal& ideal) {
    Strings out;
    for (const auto& f : ideal) out.push_back(row(R, f));
    return out;
}

V json_rows(const Ring& R, const Ideal& ideal) {
    V out(Json::arrayValue);
    for (const auto& f : ideal) out.append(row(R, f));
    return out;
}

// The expression of engine/polynomial.py `format_poly` with the ring names.
std::string text(const Ring& R, const Polynomial& f) {
    std::string out;
    bool any = false;
    for (const auto* term : sorted_terms(R, f, true)) {
        std::string coefficient = term->second.get_str();
        const bool negative = coefficient[0] == '-';
        if (negative) coefficient.erase(0, 1);
        Strings factors;
        for (size_t k : R.order) {
            const auto& e = term->first[k];
            if (e == 0) continue;
            factors.push_back(e == 1 ? R.header[k] : R.header[k] + "^" + e.get_str());
        }
        if (coefficient != "1" || factors.empty()) factors.insert(factors.begin(), coefficient);
        out += negative ? "-" : any ? "+" : "";
        for (size_t i = 0; i < factors.size(); ++i) out += (i ? "*" : "") + factors[i];
        any = true;
    }
    return any ? out : "0";
}

Polynomial constant(const mpq_class& c, size_t width) {
    Polynomial f;
    if (c != 0) f.emplace(Powers(width, 0), c);
    return f;
}

Polynomial coordinate(size_t width, size_t axis) {
    Powers e(width, 0);
    e[axis] = 1;
    return {{e, 1}};
}

bool is_unit(const Ideal& basis) {
    if (basis.size() != 1 || basis[0].size() != 1) return false;
    const auto& [e, c] = *basis[0].begin();
    return c == 1 && std::all_of(e.begin(), e.end(), [](const mpz_class& x) { return x == 0; });
}

bool is_number(const Polynomial& f) {
    return f.empty() || (f.size() == 1 &&
        std::all_of(f.begin()->first.begin(), f.begin()->first.end(), [](const mpz_class& x) { return x == 0; }));
}

Polynomial times(const Polynomial& a, const Polynomial& b) {
    Polynomial out;
    for (const auto& [ea, ca] : a)
        for (const auto& [eb, cb] : b) {
            Powers e(ea.size());
            for (size_t i = 0; i < e.size(); ++i) e[i] = ea[i] + eb[i];
            auto& c = out[e];
            c += ca * cb;
        }
    for (auto it = out.begin(); it != out.end();) it = it->second == 0 ? out.erase(it) : std::next(it);
    return out;
}

Polynomial power(const Polynomial& f, const mpz_class& n) {
    Polynomial out = constant(1, f.empty() ? 0 : f.begin()->first.size());
    if (f.empty()) return n == 0 ? out : Polynomial{};
    for (mpz_class k = 0; k < n; ++k) out = times(out, f);
    return out;
}

// The same polynomial with `extra` zero exponents appended.
Polynomial widen(const Polynomial& f, size_t extra) {
    if (!extra) return f;
    Polynomial out;
    for (const auto& [e, c] : f) {
        Powers wider = e;
        wider.resize(e.size() + extra, 0);
        out.emplace(std::move(wider), c);
    }
    return out;
}

Ideal widen(const Ideal& ideal, size_t extra) {
    Ideal out;
    for (const auto& f : ideal) out.push_back(widen(f, extra));
    return out;
}

Ideal joined(Ideal a, const Ideal& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

// Distinct nonzero generators in their first order of appearance.
Ideal distinct(const Ideal& ideal) {
    Ideal out;
    for (const auto& f : ideal)
        if (!f.empty() && std::find(out.begin(), out.end(), f) == out.end()) out.push_back(f);
    return out;
}

std::string integer_text(const V& value) {
    return value.isString() ? value.asString() : value.isUInt64() ? std::to_string(value.asUInt64())
                                                                    : std::to_string(value.asInt64());
}

// ---------------------------------------------------------------- operations

class Call {
public:
    explicit Call(bool packed_present) : packed_present_(packed_present) {}

    // engine/groebner.py: the reduced grevlex basis held by the kernel.
    struct Basis {
        void* handle = nullptr;
        Ideal polys;
        Basis() = default;
        Basis(const Basis&) = delete;
        ~Basis() { if (handle) atlas_groebner_free(handle); }
    };
    using Held = std::shared_ptr<Basis>;

    Held groebner(const Ring& R, const Ideal& ideal) {
        if (R.size() == 0)
            throw Raise{"Unavailable", "Native Groebner computation unavailable for the specified ring or order. "
                                       "Build the current kernel with make groebner."};
        if (R.size() > MAX_VARIABLES) throw Declined{"The basis exceeds the packed variable range."};
        std::string payload = std::to_string(R.size()) + " 0 0\n";
        for (const auto& f : ideal) payload += row(R, f) + "\n";
        void* handle = atlas_groebner(payload.data(), payload.size());
        if (!handle) {
            const std::string message = atlas_groebner_error();
            if (out_of_range(message)) throw Declined{message};
            throw Raise{"Unavailable", "Native Groebner computation unavailable for the specified ring or order. "
                                       "Build the current kernel with make groebner."};
        }
        auto basis = std::make_shared<Basis>();
        basis->handle = handle;
        basis->polys = decode_all(R, split_lines(atlas_groebner_text(handle)));
        return basis;
    }

    // engine/blowup_ideal.py `_gb`: bases of canonical ideals. Like that
    // cache they are kept between calls; a basis depends only on the rows.
    Held held(const Ring& R, const Ideal& ideal) {
        std::string key = std::to_string(R.size()) + "\n";
        for (const auto& f : ideal) key += row(R, f) + "\n";
        {
            std::lock_guard<std::mutex> lock(cache_lock());
            auto& bases = basis_cache();
            auto found = bases.find(key);
            if (found != bases.end()) return found->second;
        }
        auto basis = groebner(R, ideal);
        std::lock_guard<std::mutex> lock(cache_lock());
        auto& bases = basis_cache();
        if (bases.size() >= CACHE_LIMIT) bases.clear();
        bases.emplace(std::move(key), basis);
        return basis;
    }

    bool contains(const Basis& basis, const Ring& R, const Polynomial& f) {
        const std::string query = row(R, f) + "\n";
        const int answer = atlas_groebner_contains(basis.handle, query.data(), query.size());
        if (answer < 0) {
            if (out_of_range(atlas_groebner_error())) throw Declined{atlas_groebner_error()};
            throw Raise{"ArithmeticError", "The native membership test could not be completed."};
        }
        return answer != 0;
    }

    Polynomial remainder(const Basis& basis, const Ring& R, const Polynomial& f) {
        const std::string query = row(R, f) + "\n";
        char* answer = atlas_groebner_reduce(basis.handle, query.data(), query.size());
        if (!answer) {
            if (out_of_range(atlas_groebner_error())) throw Declined{atlas_groebner_error()};
            throw Raise{"ArithmeticError", "The native normal form could not be computed."};
        }
        Released hold{answer};
        std::string text = answer;
        if (!text.empty()) text.pop_back();
        return decode(R, text);
    }

    Ideal ideal_basis(const Ring& R, const Ideal& ideal) { return held(R, distinct(ideal))->polys; }

    bool ideal_subset(const Ring& R, const Ideal& smaller, const Ideal& larger) {
        auto basis = held(R, distinct(larger));
        for (const auto& f : distinct(smaller))
            if (!contains(*basis, R, f)) return false;
        return true;
    }

    // engine/kernel.py `present_outside`.
    std::optional<bool> present_outside(const Ring& R, const Ideal& ideal, const std::vector<Ideal>& removed) {
        if (R.size() < 1 || R.size() > MAX_VARIABLES) return std::nullopt;
        std::string header = "open_locus " + std::to_string(R.size()) + " " + std::to_string(ideal.size()) + " " +
                             std::to_string(removed.size());
        Strings lines = rows(R, ideal);
        for (const auto& hole : removed) {
            header += " " + std::to_string(hole.size());
            for (const auto& g : hole) lines.push_back(row(R, g));
        }
        try {
            const auto answer = ideal_request(header, lines);
            if (answer == Strings{"yes"}) return true;
            if (answer == Strings{"no"}) return false;
        } catch (const Unavailable&) {
        }
        return std::nullopt;
    }

    // engine/blowup_ideal.py `empty_locus`.
    bool empty_locus(const Ring& R, const Ideal& ideal) {
        if (R.size() == 0) throw Raise{"ValueError", "The polynomial ring needs one or more coordinate symbols."};
        const Ideal equations = distinct(ideal);
        if (packed_present_) {
            // engine/blowup_ideal.py `_packed_present`, kept between calls.
            std::string key = std::to_string(R.size()) + "\n";
            for (const auto& f : equations) key += row(R, f) + "\n";
            std::optional<std::optional<bool>> known;
            {
                std::lock_guard<std::mutex> lock(cache_lock());
                auto found = presence_cache().find(key);
                if (found != presence_cache().end()) known = found->second;
            }
            if (!known) {
                known = present_outside(R, equations, {});
                std::lock_guard<std::mutex> lock(cache_lock());
                auto& cache = presence_cache();
                if (cache.size() >= CACHE_LIMIT) cache.clear();
                cache.emplace(std::move(key), *known);
            }
            if (*known) return !**known;
        }
        return is_unit(held(R, equations)->polys);
    }

    // engine/local_blowups.py `_present`.
    bool present(const Ring& R, const Ideal& ideal, const std::vector<Ideal>& removed) {
        if (removed.empty()) return !empty_locus(R, ideal);
        if (auto answer = present_outside(R, ideal, removed)) return *answer;
        V fields;
        fields["width"] = Json::UInt(R.size());
        fields["ideal"] = json_rows(R, ideal);
        fields["holes"] = V(Json::arrayValue);
        for (const auto& hole : removed) fields["holes"].append(json_rows(R, hole));
        return level("present_outside", fields).asBool();
    }

    std::optional<Ideal> colon(const Ring& R, const Ideal& ideal, const Ideal& by) {
        if (R.size() == 0) return std::nullopt;
        if (R.size() >= MAX_VARIABLES) throw Declined{"The quotient exceeds the packed variable range."};
        std::string payload = std::to_string(R.size()) + " " + std::to_string(ideal.size()) + "\n";
        for (const auto& f : ideal) payload += row(R, f) + "\n";
        for (const auto& f : by) payload += row(R, f) + "\n";
        char* answer = atlas_groebner_colon(payload.data(), payload.size());
        if (!answer) throw Declined{atlas_groebner_error()};
        Released hold{answer};
        return decode_all(R, split_lines(answer));
    }

    // engine/local_blowups.py `_generators`: few generators modulo the ambient ideal.
    Ideal generators(const Ring& R, const Ideal& ambient, const Ideal& ideal) {
        if (ideal.empty()) return {};
        if (ideal.size() == 1) {
            const Polynomial remainder_ = ambient.empty() ? ideal[0] : remainder(*held(R, ambient), R, ideal[0]);
            return remainder_.empty() ? Ideal{} : Ideal{remainder_};
        }
        const auto basis = groebner(R, joined(ambient, ideal));
        V fields;
        fields["width"] = Json::UInt(R.size());
        fields["coordinate_keys"] = V(Json::arrayValue);
        for (const auto& key : R.keys) fields["coordinate_keys"].append(key);
        fields["ambient"] = json_rows(R, ambient);
        fields["basis"] = json_rows(R, basis->polys);
        Ideal out;
        for (const auto& index : level("few_generators", fields)) out.push_back(basis->polys.at(index.asUInt()));
        return out;
    }

    // engine/kernel.py `graph_strict_part`.
    std::optional<std::pair<mpz_class, Ideal>> graph_strict_part(const Ring& R, const Ideal& ambient,
                                                                const Polynomial& pivot, const Polynomial& f) {
        if (R.size() + 1 > MAX_VARIABLES) return std::nullopt;
        Strings answer;
        try {
            answer = ideal_request("graph_strict_part " + std::to_string(R.size()) + " " +
                                   std::to_string(ambient.size()) + " 1 1",
                                   rows(R, joined(ambient, {pivot, f})));
        } catch (const Unavailable&) {
            return std::nullopt;
        }
        std::istringstream head(answer.at(0));
        std::string order, size;
        head >> order >> size;
        if (size != "1" || answer.size() != 2)
            throw Raise{"ArithmeticError", "Expected a principal residual ideal on the chart."};
        return std::make_pair(mpz_class(order), decode_all(R, answer, 1, 1));
    }

    // engine/kernel.py `strict_parts` for one principal divisor.
    std::optional<std::vector<std::pair<mpz_class, Ideal>>> strict_parts(const Ring& R, const Ideal& ambient,
                                                                        const Polynomial& pivot,
                                                                        const Ideal& functions) {
        if (R.size() + 1 > MAX_VARIABLES) return std::nullopt;
        Strings answer;
        try {
            answer = ideal_request("strict_parts " + std::to_string(R.size()) + " " + std::to_string(ambient.size()) +
                                   " 1 1", rows(R, joined(joined(ambient, {pivot}), functions)));
        } catch (const Unavailable&) {
            return std::nullopt;
        }
        std::vector<std::pair<mpz_class, Ideal>> out;
        size_t at = 0;
        for (size_t i = 0; i < functions.size(); ++i) {
            std::istringstream head(answer.at(at++));
            std::string order, size;
            head >> order >> size;
            const size_t count = std::stoul(size);
            out.emplace_back(mpz_class(order), decode_all(R, answer, at, count));
            at += count;
        }
        return out;
    }

    // engine/local_blowups.py `_strict_ideal`: order and residual ideal of a
    // pulled-back divisor along the new pivot.
    std::pair<mpz_class, Ideal> strict_ideal(const Ring& R, const Ideal& ambient, const Polynomial& pivot,
                                             const Ideal& generators) {
        Ideal nonzero;
        for (const auto& g : generators)
            if (!ideal_subset(R, {g}, ambient)) nonzero.push_back(g);
        if (!ambient.empty() && nonzero.size() == 1)
            if (auto principal = graph_strict_part(R, ambient, pivot, nonzero[0])) return *principal;
        std::optional<std::vector<std::pair<mpz_class, Ideal>>> native;
        if (!nonzero.empty()) native = strict_parts(R, ambient, pivot, nonzero);
        if (native && !native->empty()) {
            if (native->size() == 1) return native->front();
            std::vector<Strings> orders(1);
            for (const auto& part : *native) orders[0].push_back(part.first.get_str());
            const auto answer = binomial_request("residual_orders " + std::to_string(native->size()), orders);
            std::istringstream values(answer.at(0));
            std::string word;
            values >> word;
            const mpz_class order(word);
            Ideal residual;
            for (const auto& part : *native) {
                values >> word;
                const Polynomial factor = power(pivot, mpz_class(word));
                for (const auto& g : part.second) residual.push_back(times(g, factor));
            }
            return {order, ideal_basis(R, joined(ambient, residual))};
        }
        V fields;
        fields["width"] = Json::UInt(R.size());
        fields["ambient"] = json_rows(R, ambient);
        fields["pivot"] = row(R, pivot);
        fields["ideal"] = json_rows(R, generators);
        fields["limit"] = 64;
        const V order = level("cartier_order", fields);
        if (order.isNull()) throw Raise{"GiveUp", "A divisor vanishes on a whole chart, or to an order beyond the limit."};
        const mpz_class expected(integer_text(order));
        auto quotient = colon(R, joined(ambient, generators), {power(pivot, expected)});
        if (!quotient) throw Raise{"GiveUp", "The native kernel could not compute the residual divisor ideal."};
        return {expected, *quotient};
    }

    // engine/kernel.py `residual_dimension_below`.
    std::optional<bool> residual_dimension_below(const Ring& R, const Ideal& base, const Ideal& residual,
                                                 long threshold) {
        Strings answer;
        try {
            if (R.size() < 1 || R.size() > MAX_VARIABLES) throw Unavailable{};
            answer = ideal_request("residual_dimension_below " + std::to_string(R.size()) + " " +
                                   std::to_string(threshold) + " " + std::to_string(base.size()),
                                   rows(R, joined(base, residual)));
        } catch (const Unavailable&) {
            Strings lines{"wide_dimension_below", R.header_line(), "grevlex", std::to_string(threshold)};
            for (const auto& f : joined(base, residual)) lines.push_back(text(R, f));
            try {
                answer = affine_request(lines);
            } catch (const Unavailable&) {
                return std::nullopt;
            }
        }
        return std::stol(answer.at(0)) != 0;
    }

    void deadline(std::optional<std::chrono::steady_clock::time_point> limit) {
        limit_ = limit;
    }
    void check_time() const {
        if (limit_ && std::chrono::steady_clock::now() > *limit_)
            throw Raise{"GiveUp", "The time allowed for the blowups is over."};
    }

private:
    static constexpr size_t CACHE_LIMIT = 4096;
    static std::mutex& cache_lock() {
        static std::mutex lock;
        return lock;
    }
    static std::map<std::string, Held>& basis_cache() {
        static std::map<std::string, Held> cache;
        return cache;
    }
    static std::map<std::string, std::optional<bool>>& presence_cache() {
        static std::map<std::string, std::optional<bool>> cache;
        return cache;
    }

    bool packed_present_;
    std::optional<std::chrono::steady_clock::time_point> limit_;
};

// ---------------------------------------------------------------- charts

struct Component {
    std::string label;
    V n, nu;  // exact integers as the caller wrote them
    Ideal ideal;
};

struct Chart {
    Ring ring;
    Ideal ambient;
    std::vector<Component> components;
    Ideal fibre, region, coordinates;
    std::vector<Ideal> removed;
    long pivot = -1;  // index of the chart of the blowup, for the history
};

Ring read_ring(const V& data) {
    Ring R;
    for (const auto& name : data["names"]) R.names.push_back(name.asString());
    for (const auto& identity : data["identities"]) R.identities.push_back(identity.asInt64());
    if (R.identities.size() != R.names.size()) throw std::invalid_argument("Each coordinate needs an identity.");
    for (size_t i = 0; i < R.names.size(); ++i) R.sources.emplace_back(Json::UInt(i));
    R.theta = data.get("theta", -1).asInt();
    R.finish();
    return R;
}

Ideal read_ideal(const Ring& R, const V& data) {
    Ideal out;
    for (const auto& row_ : data) out.push_back(decode(R, row_.asString()));
    return out;
}

Chart read_chart(const V& data) {
    Chart c;
    c.ring = read_ring(data);
    c.ambient = read_ideal(c.ring, data["ambient"]);
    for (const auto& item : data["components"])
        c.components.push_back({item[0].asString(), item[1], item[2], read_ideal(c.ring, item[3])});
    c.fibre = read_ideal(c.ring, data["fibre"]);
    c.region = read_ideal(c.ring, data["region"]);
    c.coordinates = read_ideal(c.ring, data["coordinates"]);
    for (const auto& hole : data["removed"]) c.removed.push_back(read_ideal(c.ring, hole));
    return c;
}

V write_chart(const Chart& c, Json::UInt parent) {
    V out;
    out["parent"] = parent;
    out["pivot"] = Json::Int(c.pivot);
    out["variables"] = V(Json::arrayValue);
    for (const auto& source : c.ring.sources) out["variables"].append(source);
    out["ambient"] = json_rows(c.ring, c.ambient);
    out["components"] = V(Json::arrayValue);
    for (const auto& component : c.components) {
        V item(Json::arrayValue);
        item.append(component.label);
        item.append(component.n);
        item.append(component.nu);
        item.append(json_rows(c.ring, component.ideal));
        out["components"].append(item);
    }
    out["fibre"] = json_rows(c.ring, c.fibre);
    out["region"] = json_rows(c.ring, c.region);
    out["coordinates"] = json_rows(c.ring, c.coordinates);
    out["removed"] = V(Json::arrayValue);
    for (const auto& hole : c.removed) out["removed"].append(json_rows(c.ring, hole));
    return out;
}

// engine/binomial_kernel.py `exceptional_data`: (N, nu) of the new divisor.
std::pair<V, V> exceptional_data(long codimension, const std::vector<Strings>& weights) {
    const auto answer = binomial_request("exceptional_data " + std::to_string(codimension) + " " +
                                         std::to_string(weights.size()), weights);
    std::istringstream values(answer.at(0));
    std::string n, nu;
    values >> n >> nu;
    return {V(n), V(nu)};
}

// The map of engine/local_blowups.py `_eliminate`: substitute one coordinate.
class Elimination {
public:
    Elimination(Call& call) : call_(call) {}

    // An equation a*v + b with a nonzero scalar a exhibits the chart as a
    // polynomial graph; substituting v = -b/a is an isomorphism.
    void run(Chart& chart) {
        while (true) {
            std::optional<std::pair<size_t, Polynomial>> found;
            for (const auto& g : chart.ambient) {
                for (size_t v = 0; v < chart.ring.size() && !found; ++v) {
                    if (chart.ring.theta >= 0 && static_cast<long>(v) == chart.ring.theta) continue;
                    if (auto value = solved(chart.ring, g, v)) found.emplace(v, std::move(*value));
                }
                if (found) break;
            }
            if (!found) return;
            substitute(chart, found->first, found->second);
        }
    }

private:
    Call& call_;

    static std::optional<Polynomial> solved(const Ring& R, const Polynomial& g, size_t v) {
        if (g == coordinate(R.size(), v)) return Polynomial{};
        bool uses = false;
        for (const auto& [e, c] : g) if (e[v] != 0) uses = true;
        if (!uses) return std::nullopt;
        Polynomial a, b;
        for (const auto& [e, c] : g) {
            if (e[v] > 1) return std::nullopt;
            Powers rest = e;
            rest[v] = 0;
            (e[v] == 1 ? a : b).emplace(std::move(rest), c);
        }
        if (!is_number(a)) return std::nullopt;
        const mpq_class scale = -1 / a.begin()->second;
        for (auto& [e, c] : b) c *= scale;
        return b;
    }

    void substitute(Chart& chart, size_t v, const Polynomial& value) {
        const Ring& source = chart.ring;
        Ring target;
        for (size_t i = 0; i < source.size(); ++i) {
            if (i == v) continue;
            target.names.push_back(source.names[i]);
            target.identities.push_back(source.identities[i]);
            target.sources.push_back(source.sources[i]);
        }
        target.theta = source.theta < 0 ? -1 : source.theta - (static_cast<long>(v) < source.theta ? 1 : 0);
        target.finish();
        auto narrow = [&](const Polynomial& f) {
            Polynomial out;
            for (const auto& [e, c] : f) {
                Powers p;
                for (size_t i = 0; i < e.size(); ++i) if (i != v) p.push_back(e[i]);
                out.emplace(std::move(p), c);
            }
            return out;
        };
        Ideal functions;
        auto add = [&](const Ideal& ideal) { functions.insert(functions.end(), ideal.begin(), ideal.end()); };
        add(chart.ambient);
        for (const auto& component : chart.components) add(component.ideal);
        add(chart.fibre);
        add(chart.region);
        for (const auto& hole : chart.removed) add(hole);
        add(chart.coordinates);
        Ideal moved;
        if (target.size()) {
            Strings lines{"pullbacks", target.header_line(), source.header_line()};
            const Polynomial image = narrow(value);
            for (size_t i = 0; i < source.size(); ++i) {
                if (i == v) lines.push_back(text(target, image));
                else lines.push_back(target.header[i - (i > v ? 1 : 0)]);
            }
            lines.push_back("0");
            for (const auto& f : functions) lines.push_back(text(source, f));
            moved = decode_all(target, affine_request(lines));
            if (moved.size() != functions.size()) throw Raise{"ArithmeticError", "Expected one pullback per function."};
        } else {
            const mpq_class at = value.empty() ? mpq_class(0) : value.begin()->second;
            for (const auto& f : functions) {
                mpq_class sum = 0;
                for (const auto& [e, c] : f) {
                    mpq_class term = c;
                    for (mpz_class k = 0; k < e[v]; ++k) term *= at;
                    sum += term;
                }
                moved.push_back(constant(sum, 0));
            }
        }
        size_t at = 0;
        auto take = [&](size_t count) {
            Ideal part(moved.begin() + at, moved.begin() + at + count);
            at += count;
            return distinct(part);
        };
        chart.ambient = take(chart.ambient.size());
        for (auto& component : chart.components) component.ideal = take(component.ideal.size());
        chart.fibre = take(chart.fibre.size());
        chart.region = take(chart.region.size());
        for (auto& hole : chart.removed) hole = take(hole.size());
        chart.coordinates.assign(moved.begin() + at, moved.end());
        chart.ring = std::move(target);
    }
};

class Stage {
public:
    Stage(Call& call, const V& request)
        : call_(call), label_(request["label"].asString()), stage_(request["stage"].asInt64()),
          dimension_(request["dimension"].asInt64()) {}

    // engine/local_blowups.py `_blow_up`.
    std::vector<Chart> blow_up(const Chart& chart, const Ideal& centre, long codimension) {
        V fields;
        fields["width"] = Json::UInt(chart.ring.size());
        fields["codimension"] = Json::Int(codimension);
        fields["ambient"] = json_rows(chart.ring, chart.ambient);
        fields["centre"] = json_rows(chart.ring, centre);
        const V linear = level("linear_centre", fields);
        if (linear.isArray() && !linear.empty()) return blow_up_linear(chart, linear, codimension);
        return blow_up_graph(chart, centre, codimension);
    }

private:
    Call& call_;
    std::string label_;
    long long stage_;
    long dimension_;

    std::vector<Chart> blow_up_linear(const Chart& chart, const V& linear, long codimension) {
        const Ring& R = chart.ring;
        const size_t n = R.size();
        std::vector<size_t> places;
        Ideal shifts;
        for (const auto& pair : linear) {
            places.push_back(pair[0].asUInt());
            shifts.push_back(decode(R, pair[1].asString()));
        }
        // engine/local_blowups.py `_child_variables`.
        std::set<std::string> used(R.names.begin(), R.names.end());
        auto fresh = [&](std::string name) {
            while (used.count(name)) name += "_";
            used.insert(name);
            return name;
        };
        std::vector<Ring> rings;
        for (size_t j = 0; j < places.size(); ++j) {
            Ring child = R;
            if (!shifts[j].empty()) {
                const std::string w = fresh(R.names[places[j]] + "_" + std::to_string(stage_));
                child.names[places[j]] = w;
                child.identities[places[j]] = 0;
                child.sources[places[j]] = V(w);
            }
            for (size_t i = 0; i < places.size(); ++i) {
                if (i == j) continue;
                const std::string ratio = fresh("r" + std::to_string(j + 1) + "_" + std::to_string(i + 1));
                child.names[places[i]] = ratio;
                child.identities[places[i]] = 0;
                child.sources[places[i]] = V(ratio);
            }
            child.finish();
            rings.push_back(std::move(child));
        }
        // engine/kernel.py `blowup_linear`.
        if (n + 1 > MAX_VARIABLES) throw Declined{"The linear blowup exceeds the packed variable range."};
        std::string header = "blowup " + std::to_string(n) + " " + std::to_string(chart.ambient.size()) + " " +
                             std::to_string(places.size()) + " ";
        for (size_t i = 0; i < places.size(); ++i) header += (i ? " " : "") + std::to_string(places[i]);
        header += " " + std::to_string(chart.components.size());
        Ideal lines = joined(chart.ambient, shifts);
        for (const auto& component : chart.components) {
            header += " " + std::to_string(component.ideal.size());
            lines = joined(lines, component.ideal);
        }
        Ideal functions = chart.coordinates;
        for (const auto& hole : chart.removed) functions = joined(functions, hole);
        header += " " + std::to_string(chart.fibre.size()) + " " + std::to_string(chart.region.size()) + " " +
                  std::to_string(functions.size());
        lines = joined(joined(joined(lines, chart.fibre), chart.region), functions);
        Strings answer;
        try {
            answer = ideal_request(header, rows(R, lines));
        } catch (const Unavailable& error) {
            throw Declined{error.message};
        }
        std::vector<Chart> children;
        size_t at = 0;
        for (size_t j = 0; at < answer.size(); ++j) {
            if (j >= rings.size()) break;
            const Ring& ring = rings[j];
            ++at;  // "chart j"
            Chart child;
            child.ring = ring;
            child.ambient = chart.ambient;
            std::vector<Strings> weights;
            for (const auto& component : chart.components) {
                std::istringstream head(answer.at(at));
                std::string word, order, here, count;
                head >> word >> order >> here >> count;
                const size_t size = std::stoul(count);
                if (here == "1") child.components.push_back({component.label, component.n, component.nu,
                                                             decode_all(ring, answer, at + 1, size)});
                weights.push_back({integer_text(component.n), integer_text(component.nu), order});
                at += 1 + size;
            }
            auto data = exceptional_data(codimension, weights);
            child.components.push_back({label_, data.first, data.second, {coordinate(n, places[j])}});
            std::vector<Ideal> parts;
            for (int k = 0; k < 3; ++k) {
                std::istringstream head(answer.at(at));
                std::string word, count;
                head >> word >> count;
                const size_t size = std::stoul(count);
                parts.push_back(decode_all(ring, answer, at + 1, size));
                at += 1 + size;
            }
            child.fibre = parts[0];
            child.region = parts[1];
            for (size_t i = 0; i < j; ++i) child.region.push_back(coordinate(n, places[i]));
            const Ideal& moved = parts[2];
            child.coordinates.assign(moved.begin(), moved.begin() + chart.coordinates.size());
            size_t from = chart.coordinates.size();
            for (const auto& hole : chart.removed) {
                Ideal kept;
                for (size_t k = 0; k < hole.size(); ++k)
                    if (!moved.at(from + k).empty()) kept.push_back(moved[from + k]);
                child.removed.push_back(std::move(kept));
                from += hole.size();
            }
            child.pivot = static_cast<long>(j);
            children.push_back(std::move(child));
        }
        return children;
    }

    std::vector<Chart> blow_up_graph(const Chart& chart, const Ideal& centre, long codimension) {
        const Ring& R = chart.ring;
        const size_t n = R.size();
        const Ideal gens = call_.generators(R, chart.ambient, centre);
        std::vector<Strings> points;
        if (codimension == dimension_ && R.theta < 0) {
            if (n >= 1 && n <= MAX_VARIABLES) {
                try {
                    for (const auto& line : ideal_request("graph_centre_points " + std::to_string(n),
                                                          rows(R, joined(centre, chart.ambient)))) {
                        std::istringstream words(line);
                        Strings point;
                        for (std::string word; words >> word;) point.push_back(word);
                        points.push_back(std::move(point));
                    }
                } catch (const Unavailable&) {
                    points.clear();
                }
            }
        }
        // engine/blowup_ideal.py `blowup_charts` with the kernel's op "charts".
        const size_t k = gens.size();
        if (n + k < 1 || n + k > MAX_VARIABLES) throw Declined{"The blowup exceeds the packed variable range."};
        Strings answer;
        try {
            answer = ideal_request("charts " + std::to_string(n) + " " + std::to_string(chart.ambient.size()) +
                                   " -1 0", rows(R, joined(chart.ambient, gens)));
        } catch (const Unavailable& error) {
            throw Declined{error.message};
        }
        struct Open {
            Ring ring;
            Ideal ambient;
            Polynomial pivot;
            size_t index;
        };
        std::vector<Open> opens;
        size_t at = 0;
        for (size_t i = 0; i < k; ++i) {
            Ring ring = R;
            std::set<std::string> taken(R.names.begin(), R.names.end());
            for (size_t j = 0; j < k; ++j) {
                if (j == i) continue;
                std::string name = "r" + std::to_string(i + 1) + "_" + std::to_string(j + 1);
                while (taken.count(name)) name += "_";
                taken.insert(name);
                ring.names.push_back(name);
                ring.identities.push_back(0);
                ring.sources.emplace_back(name);
            }
            ring.finish();
            std::istringstream head(answer.at(at++));
            long a = 0, s = 0;
            head >> a >> s;
            Open open{ring, decode_all(ring, answer, at, a), widen(gens[i], k - 1), i};
            at += a + std::max(s, 0L);
            for (long size; head >> size;) at += size;
            opens.push_back(std::move(open));
        }
        // engine/kernel.py `centre_power_transforms` for the principal divisors.
        std::vector<std::pair<std::string, std::optional<std::pair<mpz_class, Ideal>>>> shared;
        for (const auto& component : chart.components) {
            if (component.ideal.size() != 1) continue;
            Strings lines{"centre_power_transforms", R.header_line(), std::to_string(chart.ambient.size()),
                          std::to_string(k)};
            for (const auto& f : chart.ambient) lines.push_back(text(R, f));
            for (const auto& f : gens) lines.push_back(text(R, f));
            lines.push_back(text(R, component.ideal[0]));
            std::optional<std::pair<mpz_class, Ideal>> value;
            try {
                const auto transforms = affine_request(lines);
                if (transforms != Strings{"none"}) {
                    if (transforms.size() != 1 + k)
                        throw Raise{"ArithmeticError", "Expected one centre-power quotient per graph chart."};
                    Ideal quotients;
                    for (const auto& open : opens) quotients.push_back(decode(open.ring, transforms[open.index + 1]));
                    value.emplace(mpz_class(transforms[0]), std::move(quotients));
                }
            } catch (const Unavailable&) {
            }
            auto existing = std::find_if(shared.begin(), shared.end(),
                                         [&](const auto& item) { return item.first == component.label; });
            if (existing != shared.end()) existing->second = std::move(value);
            else shared.emplace_back(component.label, std::move(value));
        }
        const long threshold = dimension_ - 1;
        std::vector<Chart> children;
        for (const auto& open : opens) {
            call_.check_time();
            const Ring& ring = open.ring;
            if (call_.empty_locus(ring, open.ambient)) continue;
            const Ideal base = joined(open.ambient, {open.pivot});
            std::vector<std::pair<std::string, std::pair<mpz_class, Polynomial>>> candidates;
            for (const auto& [name, value] : shared)
                if (value) candidates.push_back({name, {value->first, value->second[open.index]}});
            std::map<std::string, std::pair<mpz_class, Ideal>> certified;
            if (!candidates.empty() && ring.size() >= 1 && ring.size() <= MAX_VARIABLES) {
                Ideal quotients;
                for (const auto& item : candidates) quotients.push_back(item.second.second);
                try {
                    const auto flags = ideal_request("certify_exceptional_orders " + std::to_string(ring.size()) +
                                                     " " + std::to_string(threshold) + " " +
                                                     std::to_string(base.size()),
                                                     rows(ring, joined(base, quotients)));
                    if (flags.size() != quotients.size())
                        throw Raise{"ArithmeticError", "Expected one exceptional-order certificate per quotient."};
                    for (size_t i = 0; i < flags.size(); ++i)
                        if (std::stol(flags[i]) != 0)
                            certified[candidates[i].first] = {candidates[i].second.first, {candidates[i].second.second}};
                } catch (const Unavailable&) {
                }
            }
            Chart child;
            child.ring = ring;
            child.ambient = open.ambient;
            std::vector<Strings> weights;
            for (const auto& component : chart.components) {
                const Ideal before = widen(component.ideal, k - 1);
                mpz_class order;
                Ideal after;
                auto found = certified.find(component.label);
                if (found != certified.end()) {
                    order = found->second.first;
                    after = found->second.second;
                } else {
                    std::tie(order, after) = call_.strict_ideal(ring, open.ambient, open.pivot, before);
                    const auto proper = call_.residual_dimension_below(ring, base, after, threshold);
                    if (!proper) throw Raise{"GiveUp", "No native certificate for the residual divisor dimension is available."};
                    if (!*proper) throw Raise{"Uneven", "A divisor has different orders along parts of one exceptional divisor."};
                }
                if (divisor_present(open, after, chart, component.ideal, points))
                    child.components.push_back({component.label, component.n, component.nu, after});
                weights.push_back({integer_text(component.n), integer_text(component.nu), order.get_str()});
            }
            auto data = exceptional_data(codimension, weights);
            child.components.push_back({label_, data.first, data.second, {open.pivot}});
            child.region = widen(chart.region, k - 1);
            for (size_t j = 0; j < open.index; ++j) child.region.push_back(coordinate(ring.size(), n + j));
            for (auto& component : child.components)
                component.ideal = call_.generators(ring, open.ambient, component.ideal);
            child.fibre = widen(chart.fibre, k - 1);
            child.coordinates = widen(chart.coordinates, k - 1);
            for (const auto& hole : chart.removed) child.removed.push_back(widen(hole, k - 1));
            child.pivot = static_cast<long>(open.index);
            Elimination(call_).run(child);
            children.push_back(std::move(child));
        }
        return children;
    }

    template<class Open>
    bool divisor_present(const Open& open, const Ideal& after, const Chart& parent, const Ideal& before,
                         const std::vector<Strings>& points) {
        const Ring& ring = open.ring;
        const Ideal equations = joined(open.ambient, after);
        if (!points.empty()) {
            // engine/kernel.py `graph_centre_incidence`.
            if (ring.size() >= 1 && ring.size() <= MAX_VARIABLES) {
                Strings lines = rows(ring, equations);
                for (const auto& point : points) {
                    std::string line;
                    for (size_t i = 0; i < point.size(); ++i) line += (i ? " " : "") + point[i];
                    lines.push_back(line);
                }
                try {
                    if (ideal_request("graph_centre_incidence " + std::to_string(ring.size()) + " " +
                                      std::to_string(equations.size()) + " " + std::to_string(points[0].size()) +
                                      " " + std::to_string(points.size()), lines) == Strings{"yes"})
                        return true;
                } catch (const Unavailable&) {
                }
            }
            // engine/kernel.py `graph_parent_slice`.
            const Ring& R = parent.ring;
            if (R.size() + 1 <= MAX_VARIABLES) {
                const Polynomial pivot = parent_pivot(open, R.size());
                Strings lines = rows(R, joined(joined(parent.ambient, before), {pivot}));
                std::string zeros;
                for (size_t i = 0; i < R.size(); ++i) zeros += " 0";
                for (const auto& value : points[0]) lines.push_back(value + zeros);
                try {
                    if (ideal_request("graph_parent_slice " + std::to_string(R.size()) + " " +
                                      std::to_string(parent.ambient.size() + before.size()), lines) ==
                        Strings{"yes"})
                        return true;
                } catch (const Unavailable&) {
                }
            }
        }
        return !call_.empty_locus(ring, equations);
    }

    template<class Open>
    static Polynomial parent_pivot(const Open& open, size_t width) {
        Polynomial out;
        for (const auto& [e, c] : open.pivot) out.emplace(Powers(e.begin(), e.begin() + width), c);
        return out;
    }
};

V stage(const V& request) {
    Call call(request["packed_present"].asBool());
    if (!request["seconds"].isNull())
        call.deadline(std::chrono::steady_clock::now() +
                      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                          std::chrono::duration<double>(request["seconds"].asDouble())));
    Stage step(call, request);
    const long codimension = request["dimension"].asInt64() - request["size"].asInt64();
    const bool retains = request["retains"].asBool();
    const std::string label = request["label"].asString();
    V following(Json::arrayValue), values(Json::arrayValue);
    const auto& charts = request["charts"];
    for (Json::ArrayIndex i = 0; i < charts.size(); ++i) {
        const V& data = charts[i];
        if (data["centre"].isNull()) {
            V keep;
            keep["keep"] = i;
            following.append(keep);
            continue;
        }
        const Chart chart = read_chart(data);
        const Ideal centre = read_ideal(chart.ring, data["centre"]);
        auto append = [&](const Chart& child) {
            for (const auto& component : child.components)
                if (component.label == label) {
                    V pair(Json::arrayValue);
                    pair.append(integer_text(component.n));
                    pair.append(integer_text(component.nu));
                    values.append(pair);
                }
            following.append(write_chart(child, i));
        };
        if (!retains) {
            for (const auto& child : step.blow_up(chart, centre, codimension)) append(child);
            continue;
        }
        Ideal owned = joined(joined(joined(chart.ambient, chart.fibre), chart.region), centre);
        if (call.present(chart.ring, owned, chart.removed)) {
            for (auto& child : step.blow_up(chart, centre, codimension)) {
                auto found = std::find_if(child.components.begin(), child.components.end(),
                                          [&](const Component& c) { return c.label == label; });
                if (found == child.components.end()) continue;
                child.region = distinct(joined(child.region, found->ideal));
                append(child);
            }
        }
        V kept(Json::arrayValue);
        if (!chart.removed.empty()) {
            V fields;
            fields["width"] = Json::UInt(chart.ring.size());
            fields["ambient"] = json_rows(chart.ring, chart.ambient);
            fields["centre"] = json_rows(chart.ring, centre);
            fields["holes"] = V(Json::arrayValue);
            for (const auto& hole : chart.removed) fields["holes"].append(json_rows(chart.ring, hole));
            kept = level("retained_centres", fields);
        }
        V retained;
        retained["retain"] = i;
        retained["holes"] = kept;
        following.append(retained);
    }
    if (!values.empty()) {
        V fields;
        fields["values"] = values;
        if (level("common_value", fields).isNull())
            throw Raise{"Uneven", "The parts of a centre give the new divisor different numerical data."};
    }
    V result;
    result["charts"] = following;
    return result;
}

// engine/local_blowups.py `_failures` and `_own_failures` for many charts.
V failures(const V& request) {
    Call call(request["packed_present"].asBool());
    const long dimension = request["dimension"].asInt64();
    V answers(Json::arrayValue);
    std::vector<Chart> charts;
    for (const auto& data : request["charts"]) charts.push_back(read_chart(data));
    // Every chart first: is any of its own part over the fibre left?
    std::vector<bool> here;
    for (Json::ArrayIndex i = 0; i < charts.size(); ++i) {
        const Chart& c = charts[i];
        here.push_back(!request["charts"][i]["check"].asBool() ||
                       call.present(c.ring, joined(joined(c.ambient, c.fibre), c.region), c.removed));
    }
    for (Json::ArrayIndex i = 0; i < charts.size(); ++i) {
        const Chart& c = charts[i];
        const V& data = request["charts"][i];
        V answer;
        answer["present"] = bool(here[i]);
        if (!here[i] || !data["crossings"].asBool()) {
            answers.append(answer);
            continue;
        }
        const Ring& R = c.ring;
        std::optional<Strings> found;
        if (R.size() >= 1 && R.size() <= MAX_VARIABLES) {
            std::string header = "crossings " + std::to_string(R.size()) + " " + std::to_string(dimension) + " " +
                                 std::to_string(c.ambient.size()) + " " + std::to_string(c.components.size());
            Ideal lines = c.ambient;
            for (const auto& component : c.components) {
                header += " " + std::to_string(component.ideal.size());
                lines = joined(lines, component.ideal);
            }
            header += " " + std::to_string(c.fibre.size()) + " " + std::to_string(c.region.size());
            try {
                found = ideal_request(header, rows(R, joined(joined(lines, c.fibre), c.region)));
                if (*found == Strings{"unknown"}) found.reset();
            } catch (const Unavailable&) {
                found.reset();
            }
        }
        if (!found) {
            Strings lines{"wide_crossing_loci", R.header_line(), std::to_string(dimension)};
            auto ideal = [&](const Ideal& gens) {
                lines.push_back(std::to_string(gens.size()));
                for (const auto& g : gens) lines.push_back(text(R, g));
            };
            ideal(c.ambient);
            ideal(c.fibre);
            lines.push_back(std::to_string(c.components.size()));
            for (const auto& component : c.components) ideal(component.ideal);
            ideal(c.region);
            try {
                found = affine_request(lines);
            } catch (const Unavailable& error) {
                throw Raise{"Unavailable", error.message};
            }
            if (*found == Strings{"unknown"}) throw Raise{"TypeError", "No crossing loci were computed."};
        }
        // The loci, then the part of each in the chart's own region.
        V loci(Json::arrayValue);
        std::vector<std::pair<std::vector<std::string>, Ideal>> over;  // labels -> generators, last one wins
        std::vector<std::vector<std::string>> inside;
        const Strings& lines = *found;
        for (size_t at = 0; at < lines.size() && lines[at] != "none";) {
            std::istringstream words(lines[at]);
            Strings fields;
            for (std::string word; words >> word;) fields.push_back(word);
            const size_t size = std::stoul(fields.at(1));
            const size_t count = std::stoul(fields.at(3 + size));
            V positions(Json::arrayValue);
            std::vector<std::string> labels;
            for (size_t j = 0; j < size; ++j) {
                const size_t p = std::stoul(fields[2 + j]);
                positions.append(Json::UInt(p));
                labels.push_back(c.components.at(p).label);
            }
            const bool flag = fields[2 + size] == "1";
            const Ideal gens = decode_all(R, lines, at + 1, count);
            V locus(Json::arrayValue);
            locus.append(positions);
            locus.append(flag);
            locus.append(json_rows(R, gens));
            loci.append(locus);
            auto existing = std::find_if(over.begin(), over.end(), [&](const auto& item) { return item.first == labels; });
            if (existing != over.end()) existing->second = gens;
            else over.emplace_back(labels, gens);
            if (flag && std::find(inside.begin(), inside.end(), labels) == inside.end()) inside.push_back(labels);
            at += 1 + count;
        }
        answer["loci"] = loci;
        if (data["own"].asBool()) {
            V own(Json::arrayValue);
            for (const auto& labels : inside) {
                const auto& gens = std::find_if(over.begin(), over.end(),
                                                [&](const auto& item) { return item.first == labels; })->second;
                const Ideal ideal = call.ideal_basis(R, joined(gens, c.region));
                V item(Json::arrayValue), names(Json::arrayValue);
                for (const auto& label : labels) names.append(label);
                item.append(names);
                item.append(call.present(R, ideal, c.removed));
                item.append(json_rows(R, ideal));
                own.append(item);
            }
            answer["own"] = own;
        }
        answers.append(answer);
    }
    return answers;
}

V execute(const V& request) {
    const auto operation = request["operation"].asString();
    try {
        if (operation == "stage") return stage(request);
        if (operation == "failures") return failures(request);
    } catch (const Raise& error) {
        V out;
        out["raise"] = error.kind;
        out["message"] = error.message;
        return out;
    } catch (const Declined& error) {
        V out;
        out["declined"] = error.reason;
        return out;
    } catch (const Unavailable& error) {
        V out;
        out["raise"] = "Unavailable";
        out["message"] = error.message;
        return out;
    }
    throw std::invalid_argument("Unknown canonical blowup operation.");
}
}  // namespace

extern "C" const char* atlas_canonical_blowup(const char* input) noexcept {
    return atlas::sparse_json::invoke(input, execute);
}
