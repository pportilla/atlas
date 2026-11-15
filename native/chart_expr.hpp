// Chart polynomials and native requests for orchestration written in C++.
//
// `Expr` is the value of engine/native_polynomial.py: the nonzero terms of a
// polynomial in its own active coordinates, sorted by (name, identity), so a
// value does not depend on the ring it was computed in. Rows and expression
// texts are written exactly as that module and engine/affine_kernel.py write
// them, and the request wrappers fail the way their Python transports do, so
// a composition of native operations written here sends each operation the
// requests the step-by-step orchestration sent.
#pragma once

#include "local_io.hpp"
#include "rational_sparse_json.hpp"

#include <algorithm>
#include <cctype>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

extern "C" {
char* atlas_affine(const char*, size_t) noexcept;
char* atlas_ideal(const char*, size_t) noexcept;
char* atlas_binomial(const char*, size_t) noexcept;
char* atlas_groebner_colon(const char*, size_t) noexcept;
void* atlas_groebner(const char*, size_t) noexcept;
const char* atlas_groebner_text(const void*) noexcept;
char* atlas_groebner_reduce(void*, const char*, size_t) noexcept;
int atlas_groebner_contains(void*, const char*, size_t) noexcept;
void atlas_groebner_free(void*) noexcept;
void atlas_groebner_release(char*) noexcept;
const char* atlas_groebner_error() noexcept;
const char* atlas_centre_level(const char*) noexcept;
const char* atlas_centre_search(const char*, const char* (*)(const char*)) noexcept;
const char* atlas_point_geometry(const char*) noexcept;
}

namespace atlas {
Json::Value centre_search(const Json::Value& request, std::function<Json::Value(const Json::Value&)> query);
Json::Value centre_level(const Json::Value& request);
size_t expression_size(const std::map<std::vector<mpz_class>, mpq_class>& terms);
}

namespace atlas::chart {
using V = Json::Value;
using Q = mpq_class;
using Z = mpz_class;
using Powers = atlas::local::Powers;
using Strings = std::vector<std::string>;

constexpr size_t MAX_VARIABLES = 32;

// An exception of the Python orchestration at this point; `kind` names its class.
struct Raise {
    std::string kind, message;
};
// A packed operation is out of range: the caller takes the arbitrary-width route.
struct Declined {
    std::string reason;
};
// A native operation returned no answer (native_library.Unavailable).
struct Unavailable {
    std::string message;
};

inline bool out_of_range(const std::string& message) {
    return message == "Degree out of range." || message == "Exponent out of range.";
}

// ---------------------------------------------------------------- values

struct Coord {
    std::string name;
    long long identity = 0;
    bool operator<(const Coord& other) const {
        return name != other.name ? name < other.name : identity < other.identity;
    }
    bool operator==(const Coord& other) const { return name == other.name && identity == other.identity; }
    bool operator!=(const Coord& other) const { return !(*this == other); }
};

using Variables = std::vector<Coord>;

// The data of a value; values share it, as Python shares its immutable objects.
struct ExprData {
    std::vector<Coord> coords;          // active coordinates, sorted
    std::map<Powers, Q> terms;          // nonzero coefficients by exponents over coords
    mutable size_t hash = 0;
};

class Expr {
public:
    Expr() : d_(none()) {}
    Expr(std::vector<Coord> coords, std::map<Powers, Q> terms)
        : d_(std::make_shared<const ExprData>(ExprData{std::move(coords), std::move(terms), 0})) {}

    const std::vector<Coord>& coords() const { return d_->coords; }
    const std::map<Powers, Q>& terms() const { return d_->terms; }
    bool zero() const { return d_->terms.empty(); }
    bool number() const { return d_->coords.empty(); }
    Q scalar() const { return zero() ? Q(0) : d_->terms.begin()->second; }
    bool operator==(const Expr& other) const {
        if (d_ == other.d_) return true;
        if (d_->hash && other.d_->hash && d_->hash != other.d_->hash) return false;
        return d_->coords == other.d_->coords && d_->terms == other.d_->terms;
    }
    bool operator!=(const Expr& other) const { return !(*this == other); }
    bool has(const Coord& c) const { return std::find(coords().begin(), coords().end(), c) != coords().end(); }
    bool is_symbol() const {
        return coords().size() == 1 && terms().size() == 1 && terms().begin()->first[0] == 1 &&
               terms().begin()->second == 1;
    }
    // A hash for maps of values; equal values have equal hashes.
    size_t hash() const {
        if (d_->hash) return d_->hash;
        size_t h = 0x84222325;
        auto mix = [&](size_t v) { h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
        auto integer = [&](const mpz_t z) {
            mix(static_cast<size_t>(mpz_sgn(z) + 1));
            for (size_t i = 0; i < mpz_size(z); ++i) mix(static_cast<size_t>(mpz_getlimbn(z, i)));
        };
        for (const auto& c : coords()) {
            mix(std::hash<std::string>{}(c.name));
            mix(static_cast<size_t>(c.identity));
        }
        for (const auto& [e, c] : terms()) {
            for (const auto& x : e) integer(x.get_mpz_t());
            integer(c.get_num_mpz_t());
            integer(c.get_den_mpz_t());
        }
        d_->hash = h ? h : 1;
        return d_->hash;
    }

private:
    std::shared_ptr<const ExprData> d_;
    static const std::shared_ptr<const ExprData>& none() {
        static const auto zero = std::make_shared<const ExprData>();
        return zero;
    }
};

using Ideal = std::vector<Expr>;

struct ExprHash {
    size_t operator()(const Expr& f) const { return f.hash(); }
};
struct IdealHash {
    size_t operator()(const Ideal& ideal) const {
        size_t h = ideal.size();
        for (const auto& f : ideal) h ^= f.hash() + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

inline Expr constant(const Q& c) {
    std::map<Powers, Q> terms;
    if (c != 0) terms.emplace(Powers{}, c);
    return Expr({}, std::move(terms));
}

inline Expr symbol(const Coord& c) { return Expr({c}, {{Powers{1}, Q(1)}}); }

// engine/native_polynomial.py `from_terms`: keep the active coordinates only.
inline Expr normal(const std::vector<Coord>& coords, const std::map<Powers, Q>& terms) {
    std::vector<size_t> active;
    for (size_t i = 0; i < coords.size(); ++i)
        if (std::any_of(terms.begin(), terms.end(), [&](const auto& t) { return t.second != 0 && t.first[i] != 0; }))
            active.push_back(i);
    std::sort(active.begin(), active.end(), [&](size_t a, size_t b) { return coords[a] < coords[b]; });
    std::vector<Coord> kept;
    for (size_t i : active) kept.push_back(coords[i]);
    std::map<Powers, Q> out;
    for (const auto& [e, c] : terms) {
        if (c == 0) continue;
        Powers p;
        for (size_t i : active) p.push_back(e[i]);
        out[p] += c;
    }
    for (auto it = out.begin(); it != out.end();) it = it->second == 0 ? out.erase(it) : std::next(it);
    return Expr(std::move(kept), std::move(out));
}

// The terms of a value over a list of coordinates containing its own.
inline std::map<Powers, Q> spread(const Expr& f, const std::vector<Coord>& over) {
    std::vector<size_t> at;
    for (const auto& c : f.coords()) {
        auto it = std::find(over.begin(), over.end(), c);
        if (it == over.end()) throw Raise{"ValueError", "Polynomial uses coordinates outside the declared ring."};
        at.push_back(static_cast<size_t>(it - over.begin()));
    }
    std::map<Powers, Q> out;
    for (const auto& [e, c] : f.terms()) {
        Powers p(over.size(), 0);
        for (size_t i = 0; i < at.size(); ++i) p[at[i]] = e[i];
        out.emplace(std::move(p), c);
    }
    return out;
}

inline std::vector<Coord> union_coords(const Expr& a, const Expr& b) {
    std::vector<Coord> out;
    std::set_union(a.coords().begin(), a.coords().end(), b.coords().begin(), b.coords().end(),
                   std::back_inserter(out));
    return out;
}

inline Expr operator+(const Expr& a, const Expr& b) {
    if (b.zero()) return a;
    if (a.zero()) return b;
    const auto coords = union_coords(a, b);
    auto terms = spread(a, coords);
    for (const auto& [e, c] : spread(b, coords)) terms[e] += c;
    return normal(coords, terms);
}

inline Expr scaled(const Expr& a, const Q& c) {
    if (c == 0 || a.zero()) return Expr{};
    if (c == 1) return a;
    std::map<Powers, Q> terms = a.terms();
    for (auto& [e, value] : terms) value *= c;
    return Expr(a.coords(), std::move(terms));
}

inline Expr operator-(const Expr& a) { return scaled(a, Q(-1)); }

inline Expr operator-(const Expr& a, const Expr& b) { return a + (-b); }

inline Expr operator*(const Expr& a, const Expr& b) {
    if (a.zero() || b.zero()) return Expr{};
    if (a.number()) return scaled(b, a.scalar());
    if (b.number()) return scaled(a, b.scalar());
    const auto coords = union_coords(a, b);
    const auto left = spread(a, coords), right = spread(b, coords);
    std::map<Powers, Q> terms;
    for (const auto& [ea, ca] : left)
        for (const auto& [eb, cb] : right) {
            Powers e(coords.size());
            for (size_t i = 0; i < e.size(); ++i) e[i] = ea[i] + eb[i];
            terms[e] += ca * cb;
        }
    return normal(coords, terms);
}

inline Expr power(const Expr& a, const Z& n) {
    Expr out = constant(1);
    for (Z k = 0; k < n; ++k) out = out * a;
    return out;
}

inline Expr product(const std::vector<Expr>& values) {
    Expr out = constant(1);
    for (const auto& f : values) out = out * f;
    return out;
}

// Simultaneous substitution of coordinates by values (engine/native_polynomial.py `xreplace`).
inline Expr substitute(const Expr& f, const std::vector<std::pair<Coord, Expr>>& images) {
    if (std::none_of(images.begin(), images.end(), [&](const auto& p) { return f.has(p.first); })) return f;
    Expr out;
    std::map<std::pair<size_t, Z>, Expr> powers;
    const auto& coords = f.coords();
    for (const auto& [e, c] : f.terms()) {
        Expr term = constant(c);
        for (size_t i = 0; i < coords.size(); ++i) {
            if (e[i] == 0) continue;
            auto found = powers.find({i, e[i]});
            if (found == powers.end()) {
                auto image = std::find_if(images.begin(), images.end(),
                                          [&](const auto& p) { return p.first == coords[i]; });
                const Expr base = image == images.end() ? symbol(coords[i]) : image->second;
                found = powers.emplace(std::make_pair(i, e[i]), power(base, e[i])).first;
            }
            term = term * found->second;
        }
        out = out + term;
    }
    return out;
}

// ---------------------------------------------------------------- rings

// engine/affine_kernel.py `_RingText`: printed names, unless ambiguous.
inline Strings header_names(const Variables& variables) {
    std::set<std::string> distinct;
    bool valid = true;
    for (const auto& v : variables) {
        distinct.insert(v.name);
        if (v.name.empty() || !std::isalpha(static_cast<unsigned char>(v.name[0]))) valid = false;
        for (char c : v.name)
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') valid = false;
    }
    valid = valid && distinct.size() == variables.size();
    Strings out;
    for (size_t i = 0; i < variables.size(); ++i) out.push_back(valid ? variables[i].name : "atlasVar" + std::to_string(i));
    return out;
}

inline std::string header(const Variables& variables) {
    std::string out;
    for (const auto& name : header_names(variables)) out += (out.empty() ? "" : " ") + name;
    return out;
}

inline size_t index(const Variables& variables, const Coord& c) {
    auto it = std::find(variables.begin(), variables.end(), c);
    if (it == variables.end()) throw Raise{"ValueError", "A coordinate is not in the chart."};
    return static_cast<size_t>(it - variables.begin());
}

// The row of engine/native_polynomial.py `row`: terms in the value's own
// (descending) order, exponents in the order of the variables.
inline std::string row(const Variables& variables, const Expr& f) {
    std::vector<size_t> at;
    for (const auto& c : f.coords()) {
        auto it = std::find(variables.begin(), variables.end(), c);
        if (it == variables.end()) throw Raise{"KeyError", c.name};
        at.push_back(static_cast<size_t>(it - variables.begin()));
    }
    std::string out;
    bool first = true;
    for (auto it = f.terms().rbegin(); it != f.terms().rend(); ++it) {
        if (!first) out += ';';
        first = false;
        out += it->second.get_str();
        std::vector<const Z*> exponents(variables.size(), nullptr);
        for (size_t i = 0; i < at.size(); ++i) exponents[at[i]] = &it->first[i];
        for (const auto* e : exponents) out += e ? " " + e->get_str() : " 0";
    }
    return out;
}

inline Strings rows(const Variables& variables, const Ideal& ideal) {
    Strings out;
    for (const auto& f : ideal) out.push_back(row(variables, f));
    return out;
}

inline V json_rows(const Variables& variables, const Ideal& ideal) {
    V out(Json::arrayValue);
    for (const auto& f : ideal) out.append(row(variables, f));
    return out;
}

// The expression of engine/native_polynomial.py `text` (engine/polynomial.py `format_poly`).
inline std::string text(const Variables& variables, const Expr& f) {
    const auto names = header_names(variables);
    Strings own;
    for (const auto& c : f.coords()) own.push_back(names[index(variables, c)]);
    std::vector<const std::pair<const Powers, Q>*> terms;
    for (const auto& term : f.terms()) terms.push_back(&term);
    std::stable_sort(terms.begin(), terms.end(), [](auto a, auto b) {
        Z da = 0, db = 0;
        for (const auto& e : a->first) da += e;
        for (const auto& e : b->first) db += e;
        if (da != db) return da > db;
        return a->first > b->first;
    });
    std::string out;
    for (const auto* term : terms) {
        std::string coefficient = term->second.get_str();
        const bool negative = coefficient[0] == '-';
        if (negative) coefficient.erase(0, 1);
        Strings factors;
        for (size_t i = 0; i < own.size(); ++i) {
            const auto& e = term->first[i];
            if (e == 0) continue;
            factors.push_back(e == 1 ? own[i] : own[i] + "^" + e.get_str());
        }
        if (coefficient != "1" || factors.empty()) factors.insert(factors.begin(), coefficient);
        out += negative ? "-" : out.empty() ? "" : "+";
        for (size_t i = 0; i < factors.size(); ++i) out += (i ? "*" : "") + factors[i];
    }
    return out.empty() ? "0" : out;
}

inline Expr decode(const Variables& variables, const std::string& line) {
    const auto terms = atlas::local::decode_row(line, variables.size());
    std::map<Powers, Q> copy(terms.begin(), terms.end());
    return normal(variables, copy);
}

inline Ideal decode_all(const Variables& variables, const Strings& lines, size_t from = 0,
                        size_t count = std::string::npos) {
    const size_t end = count == std::string::npos ? lines.size() : from + count;
    if (end > lines.size()) throw Raise{"ArithmeticError", "A native answer is shorter than expected."};
    Ideal out;
    for (size_t i = from; i < end; ++i) out.push_back(decode(variables, lines[i]));
    return out;
}

// ---------------------------------------------------------------- requests

inline Strings split_lines(const char* text) {
    Strings lines;
    std::string current;
    bool open = false;
    for (const char* at = text; *at; ++at) {
        if (*at == '\n') {
            lines.push_back(std::move(current));
            current.clear();
            open = false;
        } else {
            current += *at;
            open = true;
        }
    }
    if (open) lines.push_back(std::move(current));
    return lines;
}

struct Released {
    char* text;
    ~Released() { atlas_groebner_release(text); }
};

// engine/native_library.py `ideal_request`.
inline Strings ideal_request(const std::string& head, const Strings& lines) {
    std::string payload = head + "\n";
    for (const auto& line : lines) payload += line + "\n";
    char* answer = atlas_ideal(payload.data(), payload.size());
    if (!answer) throw Unavailable{atlas_groebner_error()};
    Released hold{answer};
    return split_lines(answer);
}

// engine/native_library.py `affine_request`.
inline Strings affine_request(const Strings& lines) {
    std::string payload;
    for (size_t i = 0; i < lines.size(); ++i) payload += (i ? "\n" : "") + lines[i];
    payload += "\n";
    char* answer = atlas_affine(payload.data(), payload.size());
    if (!answer) {
        const std::string message = atlas_groebner_error();
        if (message == "Unsupported number of variables." || out_of_range(message)) throw Unavailable{message};
        throw Raise{"ArithmeticError", message};
    }
    Released hold{answer};
    return split_lines(answer);
}

// engine/binomial_kernel.py `run`.
inline Strings binomial_request(const std::string& operation, const std::vector<Strings>& lines) {
    std::string payload = operation + "\n";
    for (size_t i = 0; i < lines.size(); ++i) {
        if (i) payload += "\n";
        for (size_t j = 0; j < lines[i].size(); ++j) payload += (j ? " " : "") + lines[i][j];
    }
    char* answer = atlas_binomial(payload.data(), payload.size());
    if (!answer) throw Raise{"ValueError", atlas_groebner_error()};
    Released hold{answer};
    return split_lines(answer);
}

inline std::string json_text(const V& value) {
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    return Json::writeString(writer, value);
}

inline V json_value(const std::string& text) {
    V value;
    Json::CharReaderBuilder parser;
    std::istringstream stream(text);
    std::string error;
    if (!Json::parseFromStream(parser, stream, &value, &error)) throw Raise{"ArithmeticError", error};
    return value;
}

// engine/kernel_json.py `kernel_json` for one JSON entry point.
template<class Entry>
V kernel_json(Entry entry, const std::string& operation, V fields) {
    fields["operation"] = operation;
    const V response = json_value(entry(json_text(fields).c_str()));
    if (response.isMember("error")) throw Raise{"InputError", response["error"].asString()};
    return response["result"];
}

// engine/centre_weights.py `level_weights`, called in the kernel directly.
inline V level(const std::string& operation, V fields, const Strings& weights = {}) {
    fields["operation"] = operation;
    fields["weights"] = V(Json::arrayValue);
    for (const auto& w : weights) fields["weights"].append(w);
    try {
        return atlas::centre_level(fields);
    } catch (const std::exception& error) {
        throw Raise{"InputError", error.what()};
    }
}

// engine/centre_weights.py `centre_search`: the native search asks `query`,
// whose exceptions cross the search unchanged.
template<class Query>
V centre_search(const std::string& operation, V fields, Query&& query) {
    fields["operation"] = operation;
    std::function<V(const V&)> handle = [&](const V& task) -> V {
        try {
            return query(task);
        } catch (const std::exception& error) {
            throw Raise{"ValueError", error.what()};
        }
    };
    try {
        return atlas::centre_search(fields, handle);
    } catch (const std::exception& error) {
        throw Raise{"ArithmeticError", error.what()};
    }
}

inline std::string rational_text(const Q& q) { return q.get_str(); }

inline Q fraction(const V& value) {
    if (value.isIntegral()) return Q(value.isUInt64() ? std::to_string(value.asUInt64()) : std::to_string(value.asInt64()));
    Q q(value.asString());
    q.canonicalize();
    return q;
}

}  // namespace atlas::chart
