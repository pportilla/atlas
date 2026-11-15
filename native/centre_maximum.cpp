// The maximum of the invariant of Bierstone and Milman over a closed set of
// a chart, and the locus where it is attained, in one native call: the
// procedure of engine/centres.py `maximum` on the frames of
// engine/centre_frame.py. The definitions and references are in those files.
//
// Every step calls the native operation that the Python procedure called,
// on the same values, in the same order and with the same caches where a
// cache changes a later request, so the value, key and centre are those of
// the step-by-step computation. A packed operation that is out of range, or
// a ring outside the packed kernel, declines the call; the caller then runs
// the arbitrary-width route.
#include "chart_expr.hpp"

#include <chrono>
#include <functional>
#include <unordered_map>
#include <unordered_set>

extern "C" const char* atlas_affine_centre_maximum(const char*) noexcept;

namespace {
using namespace atlas::chart;

size_t combine(size_t h, size_t v) { return h ^ (v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2)); }
size_t hash_of(const Variables& variables) {
    size_t h = variables.size();
    for (const auto& c : variables) h = combine(combine(h, std::hash<std::string>{}(c.name)), size_t(c.identity));
    return h;
}
// A ring with ideals, as a key of the caches that the Python modules keep.
struct RingKey {
    Variables variables;
    std::vector<Ideal> ideals;
    bool operator==(const RingKey& other) const { return variables == other.variables && ideals == other.ideals; }
};
struct RingKeyHash {
    size_t operator()(const RingKey& k) const {
        size_t h = hash_of(k.variables);
        for (const auto& ideal : k.ideals) h = combine(h, IdealHash{}(ideal));
        return h;
    }
};
struct IdealPairHash {
    size_t operator()(const std::pair<Ideal, Ideal>& k) const { return combine(IdealHash{}(k.first), IdealHash{}(k.second)); }
};
struct ExprTextHash {
    size_t operator()(const std::pair<Expr, std::string>& k) const {
        return combine(k.first.hash(), std::hash<std::string>{}(k.second));
    }
};
struct ExprOrderHash {
    size_t operator()(const std::pair<Expr, unsigned>& k) const { return combine(k.first.hash(), k.second); }
};

struct NoContact {
    std::string message;
};

// ---------------------------------------------------------------- invariants

struct Entry {
    bool infinite = false;
    Q q;
    bool operator==(const Entry& other) const { return infinite == other.infinite && (infinite || q == other.q); }
};
using Invariant = std::vector<Entry>;

std::string entry_text(const Entry& e) { return e.infinite ? "inf" : rational_text(e.q); }

V invariant_row(const Invariant& values) {
    V out(Json::arrayValue);
    for (const auto& e : values) out.append(entry_text(e));
    return out;
}

V strings(const Strings& values) {
    V out(Json::arrayValue);
    for (const auto& s : values) out.append(s);
    return out;
}

// ---------------------------------------------------------------- the kernel

struct Settings {
    bool packed_present = true;
    std::optional<std::chrono::steady_clock::time_point> deadline;
    long long dummies = 0;
};
thread_local Settings* settings = nullptr;

Coord dummy(const std::string& name) { return Coord{name, -(++settings->dummies)}; }

void check_time() {
    if (settings->deadline && std::chrono::steady_clock::now() > *settings->deadline)
        throw Raise{"GiveUp", "The time allowed for the blowups is over."};
}

// engine/groebner.py `groebner` and its native basis.
struct Basis {
    void* handle = nullptr;
    Variables gens;
    Ideal exprs;
    Basis() = default;
    Basis(const Basis&) = delete;
    ~Basis() { if (handle) atlas_groebner_free(handle); }

    bool contains(const Expr& f) const {
        const std::string query = row(gens, f) + "\n";
        const int answer = atlas_groebner_contains(handle, query.data(), query.size());
        if (answer < 0) {
            if (out_of_range(atlas_groebner_error())) throw Declined{atlas_groebner_error()};
            throw Raise{"ArithmeticError", "The native membership test could not be completed."};
        }
        return answer != 0;
    }
    Expr remainder(const Expr& f) const {
        const std::string query = row(gens, f) + "\n";
        char* answer = atlas_groebner_reduce(handle, query.data(), query.size());
        if (!answer) {
            if (out_of_range(atlas_groebner_error())) throw Declined{atlas_groebner_error()};
            throw Raise{"ArithmeticError", "The native normal form could not be computed."};
        }
        Released hold{answer};
        std::string text = answer;
        if (!text.empty()) text.pop_back();
        return decode(gens, text);
    }
};
using Held = std::shared_ptr<const Basis>;

Held groebner(const Ideal& ideal, const Variables& gens, const std::string& order = "grevlex", size_t known = 0) {
    static const std::map<std::string, int> orders{{"grevlex", 0}, {"lex", 1}, {"first", 2}};
    if (gens.empty())
        throw Raise{"Unavailable", "Native Groebner computation unavailable for the specified ring or order. "
                                   "Build the current kernel with make groebner."};
    if (gens.size() > MAX_VARIABLES) throw Declined{"The basis exceeds the packed variable range."};
    std::string payload = std::to_string(gens.size()) + " " + std::to_string(orders.at(order)) + " " +
                          std::to_string(known) + "\n";
    for (const auto& f : ideal) payload += row(gens, f) + "\n";
    void* handle = atlas_groebner(payload.data(), payload.size());
    if (!handle) {
        if (out_of_range(atlas_groebner_error())) throw Declined{atlas_groebner_error()};
        throw Raise{"Unavailable", "Native Groebner computation unavailable for the specified ring or order. "
                                   "Build the current kernel with make groebner."};
    }
    auto basis = std::make_shared<Basis>();
    basis->handle = handle;
    basis->gens = gens;
    basis->exprs = decode_all(gens, split_lines(atlas_groebner_text(handle)));
    return basis;
}

// engine/blowup_ideal.py `_expressions`: distinct nonzero generators.
Ideal distinct(const Ideal& ideal) {
    Ideal out;
    for (const auto& f : ideal)
        if (!f.zero() && std::find(out.begin(), out.end(), f) == out.end()) out.push_back(f);
    return out;
}

Ideal joined(Ideal a, const Ideal& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

void require(const Variables& variables) {
    for (size_t i = 0; i < variables.size(); ++i)
        for (size_t j = 0; j < i; ++j)
            if (variables[i] == variables[j]) throw Raise{"ValueError", "Coordinates must be distinct symbols."};
    if (variables.empty()) throw Raise{"ValueError", "The polynomial ring needs one or more coordinate symbols."};
}

// engine/blowup_ideal.py `_gb` (cached) and `ideal_basis`.
Held held(const Variables& variables, const Ideal& ideal) {
    thread_local std::unordered_map<RingKey, Held, RingKeyHash> cache;
    RingKey k{variables, {ideal}};
    auto found = cache.find(k);
    if (found != cache.end()) return found->second;
    auto basis = groebner(ideal, variables);
    if (cache.size() > 4096) cache.clear();
    cache.emplace(std::move(k), basis);
    return basis;
}

Ideal ideal_basis(const Variables& variables, const Ideal& ideal) {
    require(variables);
    return held(variables, distinct(ideal))->exprs;
}

bool ideal_subset(const Variables& variables, const Ideal& smaller, const Ideal& larger) {
    require(variables);
    const auto basis = held(variables, distinct(larger));
    for (const auto& f : distinct(smaller))
        if (!basis->contains(f)) return false;
    return true;
}

// engine/kernel.py `present_outside` without holes.
std::optional<bool> packed_present(const Variables& variables, const Ideal& ideal) {
    if (variables.size() < 1 || variables.size() > MAX_VARIABLES) return std::nullopt;
    try {
        const auto answer = ideal_request("open_locus " + std::to_string(variables.size()) + " " +
                                          std::to_string(ideal.size()) + " 0", rows(variables, ideal));
        if (answer == Strings{"yes"}) return true;
        if (answer == Strings{"no"}) return false;
    } catch (const Unavailable&) {
    }
    return std::nullopt;
}

bool empty_locus(const Variables& variables, const Ideal& ideal) {
    require(variables);
    const Ideal equations = distinct(ideal);
    if (settings->packed_present)
        if (auto present = packed_present(variables, equations)) return !*present;
    const Held basis = held(variables, equations);
    return basis->exprs.size() == 1 && basis->exprs[0] == constant(1);
}

// engine/blowup_ideal.py `radical_subset` (engine/kernel.py `radical_subset`).
bool radical_subset(const Variables& variables, const Ideal& smaller, const Ideal& larger) {
    require(variables);
    if (variables.size() + 1 > MAX_VARIABLES) throw Declined{"The radical test exceeds the packed variable range."};
    try {
        auto small = rows(variables, smaller);
        const size_t count = small.size();
        for (const auto& line : rows(variables, larger)) small.push_back(line);
        return ideal_request("radical " + std::to_string(variables.size()) + " " + std::to_string(count), small) ==
               Strings{"1"};
    } catch (const Unavailable& error) {
        throw Declined{error.message};
    }
}

// engine/blowup_ideal.py `saturate` (engine/kernel.py `saturate`).
Ideal saturate(const Variables& variables, const Ideal& ideal, const Expr& by) {
    require(variables);
    if (variables.size() + 1 > MAX_VARIABLES) throw Declined{"The saturation exceeds the packed variable range."};
    try {
        auto lines = rows(variables, joined(ideal, {by}));
        return decode_all(variables, ideal_request("saturate " + std::to_string(variables.size()) + " " +
                                                   std::to_string(lines.size() - 1), lines));
    } catch (const Unavailable& error) {
        throw Declined{error.message};
    }
}

// engine/affine_kernel.py `frame_function`.
Expr frame_function(const Variables& variables, const Expr& value) {
    if (variables.empty()) return value;
    for (const auto& c : value.coords())
        if (std::find(variables.begin(), variables.end(), c) == variables.end())
            throw Raise{"ValueError", "Polynomial uses coordinates outside the declared ring."};
    return value;
}

Ideal frame_polys(const std::string& operation, const Variables& variables, const std::optional<Variables>& columns,
                  const Ideal& functions) {
    Strings lines{operation, header(variables)};
    if (columns) {
        std::string line;
        for (const auto& c : *columns) line += (line.empty() ? "" : " ") + std::to_string(index(variables, c));
        lines.push_back(line);
    }
    for (const auto& f : functions) lines.push_back(text(variables, f));
    return decode_all(variables, affine_request(lines));
}

std::optional<std::pair<Expr, Expr>> frame_linear(const Variables& variables, const Expr& f, const Coord& k) {
    const auto answer = affine_request({"linear", header(variables), std::to_string(index(variables, k)),
                                        text(variables, f)});
    if (answer == Strings{"none"}) return std::nullopt;
    return std::make_pair(decode(variables, answer.at(0)), decode(variables, answer.at(1)));
}

Expr frame_minor(const Variables& variables, const Ideal& equations, const Variables& columns) {
    return frame_polys("minor", variables, columns, equations).at(0);
}

using Field = std::vector<std::pair<Coord, Expr>>;

Field frame_field(const Variables& variables, const Ideal& equations, const Variables& columns) {
    const auto values = frame_polys("field", variables, columns, equations);
    Field out;
    for (size_t i = 0; i < columns.size() && i < values.size(); ++i)
        if (!values[i].zero()) out.emplace_back(columns[i], values[i]);
    return out;
}

Expr frame_derivative(const Variables& variables, const Expr& f, const Field& field) {
    Variables columns;
    Ideal functions{f};
    for (const auto& [axis, c] : field) {
        columns.push_back(axis);
        functions.push_back(c);
    }
    return frame_polys("derive", variables, columns, functions).at(0);
}

Expr frame_restrict(const Variables& variables, const Expr& f, const Coord& k, const Expr& a, const Expr& b) {
    return frame_polys("restrict", variables, Variables{k}, {f, a, b}).at(0);
}

// engine/affine_kernel.py `frame_locus`: emptiness or the contracted basis on an open frame.
Ideal frame_basis(const Variables& variables, const Ideal& equations, const Ideal& units);
bool frame_empty(const Variables& variables, const Ideal& equations, const Ideal& units) {
    if (variables.empty())
        return std::any_of(equations.begin(), equations.end(), [](const Expr& f) { return !f.zero(); }) ||
               std::any_of(units.begin(), units.end(), [](const Expr& u) { return u.zero(); });
    thread_local std::unordered_map<RingKey, bool, RingKeyHash> cache;
    RingKey k{variables, {equations, units}};
    auto found = cache.find(k);
    if (found != cache.end()) return found->second;
    std::optional<bool> answer;
    if (variables.size() <= MAX_VARIABLES) {
        try {
            const auto result = ideal_request("frame_empty " + std::to_string(variables.size()) + " " +
                                              std::to_string(equations.size()) + " " + std::to_string(units.size()),
                                              rows(variables, joined(equations, units)));
            if (result == Strings{"1"} || result == Strings{"0"}) answer = result == Strings{"1"};
        } catch (const Unavailable&) {
        } catch (const Raise& error) {
            if (error.kind != "KeyError") throw;
        }
    }
    if (!answer) {
        Strings lines{"frame_empty", header(variables), std::to_string(equations.size())};
        for (const auto& f : joined(equations, units)) lines.push_back(text(variables, f));
        answer = affine_request(lines) == Strings{"1"};
    }
    if (cache.size() > 16384) cache.clear();
    cache.emplace(std::move(k), *answer);
    return *answer;
}

Ideal frame_basis(const Variables& variables, const Ideal& equations, const Ideal& units) {
    if (variables.empty()) return frame_empty(variables, equations, units) ? Ideal{constant(1)} : Ideal{};
    if (variables.size() <= MAX_VARIABLES) {
        try {
            return decode_all(variables, ideal_request("frame_basis " + std::to_string(variables.size()) + " " +
                                                       std::to_string(equations.size()) + " " +
                                                       std::to_string(units.size()),
                                                       rows(variables, joined(equations, units))));
        } catch (const Unavailable&) {
        } catch (const Raise& error) {
            if (error.kind != "KeyError") throw;
        }
    }
    Strings lines{"frame_basis", header(variables), std::to_string(equations.size())};
    for (const auto& f : joined(equations, units)) lines.push_back(text(variables, f));
    return decode_all(variables, affine_request(lines));
}

// engine/native_polynomial.py `count_ops` (native/point_geometry.cpp, "expression_size").
long count_ops(const Expr& f) { return static_cast<long>(atlas::expression_size(f.terms())); }

V centre_rows(const Variables& variables, const Ideal& generators) {
    V out(Json::arrayValue);
    for (const auto& g : generators) out.append(row(variables, frame_function(variables, g)));
    return out;
}

V maximum_indices(const std::vector<std::optional<std::pair<Invariant, V>>>& values, size_t birth_width = 0) {
    V fields;
    fields["values"] = V(Json::arrayValue);
    for (const auto& item : values) {
        if (!item) {
            fields["values"].append(V());
            continue;
        }
        V pair(Json::arrayValue);
        pair.append(invariant_row(item->first));
        V key(Json::arrayValue);
        for (const auto& x : item->second) key.append(x.isString() ? x : V(rational_text(fraction(x))));
        pair.append(key);
        fields["values"].append(pair);
    }
    fields["birth_width"] = Json::UInt(birth_width);
    return level("maximum", fields);
}

template<class Present>
std::vector<std::vector<std::string>> divisor_subsets(const std::vector<std::string>& labels, Present&& present,
                                                      bool largest = false) {
    V fields;
    fields["count"] = Json::UInt(labels.size());
    fields["largest"] = largest;
    const V selected = centre_search("divisor_subsets", fields, [&](const V& task) -> V {
        std::vector<std::string> chosen;
        for (const auto& j : task["indices"]) chosen.push_back(labels.at(j.asUInt()));
        return present(chosen);
    });
    std::vector<std::vector<std::string>> out;
    for (const auto& line : selected) {
        std::vector<std::string> chosen;
        for (const auto& j : line) chosen.push_back(labels.at(j.asUInt()));
        out.push_back(std::move(chosen));
    }
    return out;
}

template<class Live>
unsigned jet_order(unsigned initial, Live&& live) {
    V fields;
    fields["initial"] = initial;
    const V order = centre_search("jet_order", fields, [&](const V& task) -> V { return live(task["i"].asUInt()); });
    if (order.isNull()) throw Raise{"GiveUp", "An order is beyond the limit."};
    return order.asUInt();
}

// ---------------------------------------------------------------- frames

// An ordered map from labels: engine/centres.py keeps dict insertion order.
template<class T>
struct Labelled {
    std::vector<std::pair<std::string, T>> items;
    const T* find(const std::string& label) const {
        for (const auto& [l, v] : items) if (l == label) return &v;
        return nullptr;
    }
    void set(const std::string& label, const T& value) {
        for (auto& [l, v] : items) if (l == label) { v = value; return; }
        items.emplace_back(label, value);
    }
};

// engine/centre_frame.py `Frame`.
class Frame : public std::enable_shared_from_this<Frame> {
public:
    Variables variables, coords;
    Ideal rows, units, solved;

    Frame(Variables variables_, Variables coords_, Ideal rows_ = {}, const Ideal& units_ = {}, Ideal solved_ = {})
        : variables(std::move(variables_)), coords(std::move(coords_)), rows(std::move(rows_)),
          solved(std::move(solved_)) {
        for (const auto& u : units_) if (!u.number() || u.zero()) units.push_back(u);
    }

    const Field& field(const Coord& k) {
        auto found = fields_.find(k.name + "#" + std::to_string(k.identity));
        if (found != fields_.end()) return found->second;
        Variables others;
        for (const auto& v : variables)
            if (std::find(coords.begin(), coords.end(), v) == coords.end()) others.push_back(v);
        Field value;
        if (others.empty()) value = {{k, constant(1)}};
        else {
            others.push_back(k);
            value = frame_field(variables, rows, others);
        }
        return fields_.emplace(k.name + "#" + std::to_string(k.identity), std::move(value)).first->second;
    }

    const Basis& basis() {
        if (!basis_) {
            const auto contracted = frame_basis(variables, rows, units);
            basis_ = groebner(contracted, variables, "grevlex", contracted.size());
        }
        return *basis_;
    }

    std::pair<Coord, Held> over(const Coord& x) {
        auto found = inverted_.find(x.name + "#" + std::to_string(x.identity));
        if (found != inverted_.end()) return found->second;
        const Coord t = dummy("over");
        Variables gens{t};
        gens.insert(gens.end(), variables.begin(), variables.end());
        auto value = std::make_pair(t, groebner(joined(rows, {symbol(t) * symbol(x) - constant(1)}), gens, "first"));
        inverted_.emplace(x.name + "#" + std::to_string(x.identity), value);
        return value;
    }

    Expr unit() {
        if (!unit_) {
            Expr value = constant(1);
            for (const auto& u : units) value = reduced(value * reduced(u));
            unit_ = value;
        }
        return *unit_;
    }

    // `second` is absent (False), constant (None) or a basis.
    struct Cut {
        Held first;
        enum { Absent, Constant, Localized } kind;
        Held second;
        Variables coordinates;
    };
    const Cut& cut(const Coord& x) {
        const auto k = x.name + "#" + std::to_string(x.identity);
        auto found = cuts_.find(k);
        if (found != cuts_.end()) return found->second;
        Cut value;
        value.first = groebner(joined(rows, {symbol(x)}), variables);
        Expr u = constant(1);
        for (const auto& factor : units) {
            const Expr reduced_factor = value.first->remainder(frame_function(variables, factor));
            u = value.first->remainder(frame_function(variables, u * reduced_factor));
        }
        if (u.zero()) value.kind = Cut::Absent;
        else if (u.number()) value.kind = Cut::Constant;
        else {
            const Coord s = dummy("unit");
            value.coordinates = {s};
            value.coordinates.insert(value.coordinates.end(), variables.begin(), variables.end());
            value.kind = Cut::Localized;
            value.second = groebner(joined(value.first->exprs, {symbol(s) * u - constant(1)}), value.coordinates);
        }
        return cuts_.emplace(k, std::move(value)).first->second;
    }

    Expr reduced(const Expr& f) {
        if (rows.empty()) return f;
        return reduced_terms(frame_function(variables, f));
    }

    Expr reduced_terms(const Expr& value) {
        if (!reducer_) reducer_ = groebner(rows, variables);
        return reducer_->remainder(value);
    }

    Expr derivative(const Expr& f, const Coord& k) {
        const auto& d = field(k);
        Expr value = frame_derivative(variables, f, d);
        if (d.size() == 1 && d[0].first == k && d[0].second == constant(1)) return value;
        return reduced_terms(value);
    }

    Ideal derivative_chain(const Expr& f, const Coord& k, unsigned degree) {
        if (!degree) return {f};
        if (degree == 1 || rows.empty()) {
            Ideal chain{f};
            for (unsigned i = 0; i < degree; ++i) chain.push_back(derivative(chain.back(), k));
            return chain;
        }
        Strings lines{"implicit_derivative_chain"};
        for (auto& line : implicit_input(f)) lines.push_back(std::move(line));
        lines.push_back(std::to_string(degree));
        for (auto& line : field_lines(k)) lines.push_back(std::move(line));
        const auto answer = affine_request(lines);
        Ideal chain{f};
        for (size_t i = 1; i < answer.size(); ++i) chain.push_back(decode(variables, answer[i]));
        return chain;
    }

    Strings field_lines(const Coord& k) {
        const auto& d = field(k);
        Strings out{std::to_string(int(d.size() == 1 && d[0].first == k && d[0].second == constant(1))),
                    std::to_string(d.size())};
        for (const auto& [axis, c] : d) {
            out.push_back(std::to_string(index(variables, axis)));
            out.push_back(row(variables, c));
        }
        return out;
    }

    bool graph_regular(const Expr& f) {
        if (known_scalar_unit(f)) return true;
        auto found = regular_.find(f);
        if (found != regular_.end()) return found->second;
        const auto answer = ideal_request("graph_regular " + std::to_string(variables.size()) + " " +
                                          std::to_string(rows.size()), atlas::chart::rows(variables, joined(rows, {f})));
        return regular_[f] = answer.at(0) == "yes";
    }

    bool known_scalar_unit(const Expr& f) const { return scalar_units_.count(f) != 0; }

    std::optional<Strings> working_points(const Ideal& locus, const std::vector<Ideal>& working) {
        Strings points;
        auto add = [](Strings& into, const Strings& values) {
            for (const auto& p : values) if (std::find(into.begin(), into.end(), p) == into.end()) into.push_back(p);
        };
        for (const auto& piece : working) {
            const Ideal k = joined(locus, piece);
            auto found = point_supports_.find(k);
            if (found == point_supports_.end()) {
                const Ideal equations = joined(rows, k);
                const auto answer = ideal_request("frame_points " + std::to_string(variables.size()) + " " +
                                                  std::to_string(equations.size()) + " " +
                                                  std::to_string(units.size()),
                                                  atlas::chart::rows(variables, joined(equations, units)));
                std::optional<Strings> support;
                if (answer.at(0) == "points") support = Strings(answer.begin() + 1, answer.end());
                found = point_supports_.emplace(k, support).first;
                if (support && !support->empty()) {
                    known_nonempty_ = true;
                    add(witness_points_, *support);
                }
            }
            if (!found->second) return std::nullopt;
            add(points, *found->second);
        }
        return points;
    }

    std::optional<long> point_order(const Expr& f, const std::string& point) {
        const auto k = std::make_pair(f, point);
        auto found = point_orders_.find(k);
        if (found != point_orders_.end()) return found->second;
        auto lines = atlas::chart::rows(variables, rows);
        const auto count = lines.size();
        lines.push_back(point);
        lines.push_back(row(variables, f));
        const auto answer = ideal_request("smooth_point_order " + std::to_string(variables.size()) + " " +
                                          std::to_string(count) + " 64", lines);
        std::optional<long> value;
        if (answer.at(0) != "unknown") value = std::stol(answer.at(0));
        point_orders_.emplace(k, value);
        return value;
    }

    bool direction_vanishes(const Expr& f, const Coord& k, unsigned degree, const Strings& points) {
        const auto base = atlas::chart::rows(variables, rows);
        const std::string head = "smooth_point_direction " + std::to_string(variables.size()) + " " +
                                 std::to_string(base.size()) + " " + std::to_string(degree) + " " +
                                 std::to_string(index(variables, k));
        std::string free;
        for (const auto& v : coords) free += (free.empty() ? "" : " ") + std::to_string(index(variables, v));
        const auto query = row(variables, f);
        for (const auto& point : points) {
            auto lines = base;
            lines.push_back(point);
            lines.push_back(query);
            lines.push_back(free);
            if (ideal_request(head, lines) != Strings{"yes"}) return false;
        }
        return true;
    }

    Strings contact_open_points(const Strings& points, const Ideal& coordinates) {
        if (coordinates.empty()) return points;
        std::string axes;
        for (const auto& v : coordinates) {
            if (!v.is_symbol()) throw Raise{"ValueError", "Expected a coordinate."};
            axes += (axes.empty() ? "" : " ") + std::to_string(index(variables, v.coords()[0]));
        }
        Strings lines{axes};
        lines.insert(lines.end(), points.begin(), points.end());
        const auto answer = ideal_request("contact_open_points " + std::to_string(variables.size()), lines);
        return Strings(answer.begin() + 1, answer.end());
    }

    std::string membership(const Ideal& equations, const Expr& f, bool known) const {
        return ideal_request("frame_membership " + std::to_string(variables.size()) + " " +
                             std::to_string(equations.size()) + " " + std::to_string(int(known)),
                             atlas::chart::rows(variables, joined(equations, {f}))).at(0);
    }

    bool vanishes(const Expr& f) {
        if (f.zero()) return true;
        if (variables.empty()) return empty({});
        if (rows.empty() && std::none_of(units.begin(), units.end(), [](const Expr& u) { return u.zero(); }))
            return false;
        const Expr& k = f;
        auto found = vanishing_.find(k);
        if (found != vanishing_.end()) return found->second;
        const auto result = membership(rows, f, known_nonempty_);
        if (result != "unknown") {
            const bool answer = result == "yes" || (result == "scalar" && empty({}));
            if (!answer) scalar_units_.insert(f);
            return vanishing_[k] = answer;
        }
        if (!witness_pieces_.empty()) {
            for (const auto& piece : witness_pieces_) working_points({}, {piece});
            witness_pieces_.clear();
        }
        for (const auto& p : Strings(witness_points_))
            if (point_order(f, p)) return vanishing_[k] = false;
        const Expr residue = reduced(f);
        bool answer;
        if (residue.zero()) answer = true;
        else if (!basis_ && !coords.empty() && !units.empty() && witness(residue)) answer = false;
        else answer = basis().contains(residue);
        return vanishing_[k] = answer;
    }

    bool witness(const Expr& f) {
        for (long start : {2L, -3L, 5L}) {
            Ideal fixed;
            for (size_t i = 0; i < coords.size(); ++i)
                fixed.push_back(symbol(coords[i]) - constant(Q(start + static_cast<long>(i))));
            if (!empty_on(fixed, {f})) return true;
        }
        return false;
    }

    Ideal jet(const Expr& f, unsigned order) {
        const auto k = std::make_pair(f, order);
        auto found = jets_.find(k);
        if (found != jets_.end()) return found->second;
        if (order <= 1 || coords.empty() ||
            std::any_of(units.begin(), units.end(), [](const Expr& u) { return u.zero(); }))
            return {f};
        Strings answer;
        if (!rows.empty()) {
            Strings lines{"implicit_jet"};
            for (auto& line : implicit_input(f)) lines.push_back(std::move(line));
            lines.push_back(std::to_string(order));
            for (auto& line : basis_lines(variables, basis())) lines.push_back(std::move(line));
            lines.push_back(std::to_string(coords.size()));
            for (const auto& c : coords)
                for (auto& line : field_lines(c)) lines.push_back(std::move(line));
            answer = affine_request(lines);
        } else {
            std::string axes;
            for (const auto& c : coords) axes += (axes.empty() ? "" : " ") + std::to_string(index(variables, c));
            answer = affine_request({"affine_jet", header(variables), std::to_string(order), axes, text(variables, f)});
        }
        Ideal result{f};
        for (size_t i = 1; i < answer.size(); ++i) result.push_back(decode(variables, answer[i]));
        return jets_.emplace(k, std::move(result)).first->second;
    }

    static Strings basis_lines(const Variables& variables, const Basis& basis) {
        Strings out{std::to_string(basis.exprs.size())};
        for (const auto& f : basis.exprs) out.push_back(row(variables, f));
        return out;
    }

    Strings implicit_input(const Expr& f) {
        if (!reducer_) reducer_ = groebner(rows, variables);
        Strings out{std::to_string(variables.size())};
        for (auto& line : basis_lines(variables, *reducer_)) out.push_back(std::move(line));
        out.push_back(row(variables, f));
        return out;
    }

    bool empty(const Ideal& gens) { return empty_on(gens, {}); }
    bool inside(const Expr& unit_, const Ideal& gens) { return empty_on(gens, {unit_}); }

    bool empty_on(const Ideal& gens, const Ideal& extra) {
        auto k = std::make_pair(gens, extra);
        auto found = empty_.find(k);
        if (found != empty_.end()) return found->second;
        const bool value = frame_empty(variables, joined(rows, gens), joined(units, extra));
        empty_.emplace(std::move(k), value);
        if (!value) known_nonempty_ = true;
        return value;
    }
    std::optional<bool> cached_empty(const Ideal& gens, const Ideal& extra) const {
        auto found = empty_.find(std::make_pair(gens, extra));
        if (found == empty_.end()) return std::nullopt;
        return found->second;
    }

    std::shared_ptr<Frame> restricted(const Expr& z, const Coord& k, const Expr& unit_,
                                      const std::vector<Ideal>& witness_loci, const std::vector<Ideal>& working) {
        Variables kept;
        for (const auto& c : coords) if (c != k) kept.push_back(c);
        auto child = std::make_shared<Frame>(variables, kept, joined(rows, {z}), joined(units, {unit_}), solved);
        child->witness_pieces_ = working;
        for (const auto& gens : witness_loci) {
            if (cached_empty(gens, {unit_}) == std::optional<bool>(false) &&
                membership(joined(rows, gens), z, true) == "yes") {
                child->known_nonempty_ = true;
                break;
            }
        }
        return child;
    }

    using Move = std::function<Expr(const Expr&)>;

    std::pair<std::shared_ptr<Frame>, Move> eliminated(const Expr& z, const Coord& k,
                                                       std::optional<std::pair<Expr, Expr>> coefficients) {
        if (!coefficients) coefficients = frame_linear(variables, z, k);
        if (!coefficients) throw Raise{"TypeError", "The contact equation is not linear."};
        const auto [a, b] = *coefficients;
        const Variables source = variables;
        Move move = [source, k, a = a, b = b](const Expr& f) { return frame_restrict(source, f, k, a, b); };
        Ideal kept_rows;
        for (const auto& g : rows) {
            Expr moved = move(g);
            if (!moved.zero()) kept_rows.push_back(std::move(moved));
        }
        Variables kept_variables, kept_coords;
        for (const auto& v : variables) if (v != k) kept_variables.push_back(v);
        for (const auto& c : coords) if (c != k) kept_coords.push_back(c);
        Ideal moved_units;
        for (const auto& u : units) moved_units.push_back(move(u));
        moved_units.push_back(a);
        return {std::make_shared<Frame>(kept_variables, kept_coords, kept_rows, moved_units, joined(solved, {z})),
                move};
    }

    std::shared_ptr<Frame> changed(const std::vector<std::pair<Coord, Expr>>& swap) const {
        auto move = [&](const Ideal& gens) {
            Ideal out;
            for (const auto& g : gens) out.push_back(substitute(g, swap));
            return out;
        };
        return std::make_shared<Frame>(variables, coords, move(rows), move(units), move(solved));
    }

    bool known_nonempty() const { return known_nonempty_; }

private:
    std::map<std::string, Field> fields_;
    Held basis_, reducer_;
    std::map<std::string, std::pair<Coord, Held>> inverted_;
    std::map<std::string, Cut> cuts_;
    std::optional<Expr> unit_;
    std::unordered_map<std::pair<Expr, unsigned>, Ideal, ExprOrderHash> jets_;
    std::unordered_map<Ideal, std::optional<Strings>, IdealHash> point_supports_;
    std::unordered_map<std::pair<Expr, std::string>, std::optional<long>, ExprTextHash> point_orders_;
    std::vector<Ideal> witness_pieces_;
    Strings witness_points_;
    std::unordered_map<std::pair<Ideal, Ideal>, bool, IdealPairHash> empty_;
    std::unordered_map<Expr, bool, ExprHash> vanishing_, regular_;
    bool known_nonempty_ = false;
    std::unordered_set<Expr, ExprHash> scalar_units_;
};
using FramePtr = std::shared_ptr<Frame>;

// ---------------------------------------------------------------- presentations

using Exponents = Labelled<Q>;

struct Element {
    Exponents exps;
    Expr poly;
    Q weight;
};

// engine/centre_weights.py `Presentation`.
struct Presentation {
    std::vector<std::string> labels;
    Strings lines;

    Presentation(const std::vector<std::string>& labels_, const std::vector<Element>& collection) : labels(labels_) {
        lines.push_back(std::to_string(labels.size()) + " " + std::to_string(collection.size()));
        for (const auto& element : collection) {
            std::string line = rational_text(element.weight);
            for (const auto& h : labels) {
                const Q* e = element.exps.find(h);
                if (!e) throw Raise{"KeyError", h};
                line += " " + rational_text(*e);
            }
            lines.push_back(line);
        }
    }
    Strings request(const std::string& operation, const Strings& extra = {}) const {
        Strings all{operation};
        all.insert(all.end(), lines.begin(), lines.end());
        all.insert(all.end(), extra.begin(), extra.end());
        return affine_request(all);
    }
    std::vector<Exponents> residual() const {
        std::vector<Exponents> out;
        for (const auto& line : request("bm_residual")) {
            std::istringstream words(line);
            Exponents row;
            size_t j = 0;
            for (std::string word; words >> word; ++j) row.items.emplace_back(labels.at(j), fraction(V(word)));
            out.push_back(std::move(row));
        }
        return out;
    }
    std::vector<Q> candidate_values(const std::vector<std::vector<size_t>>& strata,
                                    const std::vector<std::vector<unsigned>>& bounds) const {
        Strings extra;
        for (size_t i = 0; i < strata.size(); ++i) {
            std::string line = std::to_string(strata[i].size());
            for (auto id : strata[i]) line += " " + std::to_string(id);
            for (auto order : bounds[i]) line += " " + std::to_string(order);
            extra.push_back(line);
        }
        std::vector<Q> out;
        for (const auto& line : request("bm_values", extra)) out.push_back(fraction(V(line)));
        return out;
    }
    Q point_maximum(const std::vector<std::pair<std::vector<size_t>, std::vector<long>>>& point_orders) const {
        Strings extra;
        for (const auto& [ids, orders] : point_orders) {
            std::string line = std::to_string(ids.size());
            for (auto id : ids) line += " " + std::to_string(id);
            for (auto order : orders) line += " " + std::to_string(order);
            extra.push_back(line);
        }
        return fraction(V(request("bm_point_maximum", extra).at(0)));
    }
    std::vector<std::vector<Z>> requirements(const std::vector<std::vector<size_t>>& strata, const Q& value) const {
        Strings extra{rational_text(value)};
        for (const auto& ids : strata) {
            std::string line = std::to_string(ids.size());
            for (auto id : ids) line += " " + std::to_string(id);
            extra.push_back(line);
        }
        std::vector<std::vector<Z>> out;
        for (const auto& line : request("bm_requirements", extra)) {
            std::istringstream words(line);
            std::vector<Z> row;
            for (std::string word; words >> word;) row.emplace_back(word);
            out.push_back(std::move(row));
        }
        return out;
    }
};

// ---------------------------------------------------------------- the invariant

struct State {
    FramePtr frame;
    std::vector<Element> coll;
    Ideal extra;
    std::vector<std::pair<std::string, Coord>> fresh;  // the divisors that are not old so far
    Ideal locus;
    std::vector<Ideal> working;
    Invariant prefix;
    std::vector<std::vector<std::pair<Coord, Expr>>> undo;
};

struct Found {
    Invariant value;
    V key;
    Ideal centre;
    State state;
};

struct Context {
    std::map<std::string, std::optional<V>> records;  // invariant rows, None for given divisors
    std::vector<std::string> births;
    Ideal fixed;
};

const Coord* lookup(const std::vector<std::pair<std::string, Coord>>& fresh, const std::string& label) {
    for (const auto& [h, v] : fresh) if (h == label) return &v;
    return nullptr;
}

bool contains(const Ideal& ideal, const Expr& f) { return std::find(ideal.begin(), ideal.end(), f) != ideal.end(); }

std::vector<Found> level_(const State& st, Context& ctx);

// engine/centres.py `_settle`.
Element settle(const Element& element, const Frame& frame, const std::vector<std::pair<std::string, Coord>>& fresh,
               const std::vector<std::string>& through, const std::vector<std::pair<std::string, Coord>>& rest) {
    std::vector<std::string> keys;
    for (const auto& [h, e] : element.exps.items) keys.push_back(h);
    for (const auto& h : through) if (std::find(keys.begin(), keys.end(), h) == keys.end()) keys.push_back(h);
    V fields;
    fields["exponents"] = V(Json::arrayValue);
    for (const auto& h : keys) {
        const Q* e = element.exps.find(h);
        fields["exponents"].append(rational_text(e ? *e : Q(0)));
    }
    fields["through"] = V(Json::arrayValue);
    for (const auto& h : through)
        fields["through"].append(Json::UInt(std::find(keys.begin(), keys.end(), h) - keys.begin()));
    const V updated = level("settle", fields, {rational_text(element.weight)});
    const Z power_(updated["power"].isString() ? updated["power"].asString() : std::to_string(updated["power"].asInt64()));
    Exponents scaled;
    for (size_t i = 0; i < keys.size(); ++i) scaled.items.emplace_back(keys[i], fraction(updated["exponents"][Json::ArrayIndex(i)]));
    Ideal factors;
    for (const auto& h : through) {
        const Q& e = *scaled.find(h);
        mpz_class n;
        mpz_tdiv_q(n.get_mpz_t(), e.get_num_mpz_t(), e.get_den_mpz_t());
        factors.push_back(power(symbol(*lookup(fresh, h)), n));
    }
    Element out;
    for (const auto& [h, e] : scaled.items)
        if (lookup(rest, h) && e != 0) out.exps.items.emplace_back(h, e);
    out.poly = frame_function(frame.variables, power(element.poly, power_) * product(factors));
    out.weight = fraction(updated["weight"]);
    return out;
}

// engine/centres.py `_remove_coordinate_powers`.
std::pair<std::vector<Z>, Expr> remove_coordinate_powers(Frame& frame, const Expr& poly, const Variables& divisors) {
    Strings answer;
    if (!frame.rows.empty() && !frame.known_scalar_unit(poly)) {
        Strings lines{"implicit_coordinates"};
        for (auto& line : frame.implicit_input(poly)) lines.push_back(std::move(line));
        lines.push_back(row(frame.variables, frame.unit()));
        lines.push_back(std::to_string(divisors.size()));
        for (const auto& v : divisors) {
            const auto& cut = frame.cut(v);
            if (cut.kind == Frame::Cut::Absent) {
                lines.push_back("absent");
                continue;
            }
            const auto [t, over] = frame.over(v);
            lines.push_back("present");
            for (auto& line : Frame::basis_lines(frame.variables, *cut.first)) lines.push_back(std::move(line));
            if (cut.kind == Frame::Cut::Constant) lines.push_back("0");
            else for (auto& line : Frame::basis_lines(cut.coordinates, *cut.second)) lines.push_back(std::move(line));
            Variables extended{t};
            extended.insert(extended.end(), frame.variables.begin(), frame.variables.end());
            for (auto& line : Frame::basis_lines(extended, *over)) lines.push_back(std::move(line));
        }
        answer = affine_request(lines);
        if (answer == Strings{"quotient_unavailable"})
            throw Raise{"GiveUp", "A quotient by the equation of a divisor was not found."};
    } else {
        std::string axes;
        for (const auto& v : divisors) axes += (axes.empty() ? "" : " ") + std::to_string(index(frame.variables, v));
        answer = affine_request({"coordinate_residual", header(frame.variables), axes, text(frame.variables, poly)});
    }
    std::vector<Z> orders;
    for (size_t i = 0; i + 1 < answer.size(); ++i) orders.emplace_back(answer[i]);
    return {orders, decode(frame.variables, answer.back())};
}

// engine/centres.py `_contact`: each state on an open set with maximal
// contact is handed to `yield_` with the choice made.
using Choice = std::tuple<size_t, Coord, unsigned>;

void contact(const State& st, Context& ctx, const std::vector<Choice>& banned,
             const std::function<void(const State&, const Choice&)>& yield_) {
    std::vector<Ideal> remaining = st.working;
    State current = st;
    unsigned attempts = 0;
    while (!remaining.empty()) {
        check_time();
        bool yielded = false;
        FramePtr frame = current.frame;
        const auto& fresh = current.fresh;
        V ranks(Json::arrayValue);
        Strings weights;
        for (const auto& element : current.coll) {
            Invariant rank{{false, element.weight}, {false, Q(long(element.exps.items.size()))},
                           {false, Q(count_ops(element.poly))}};
            ranks.append(invariant_row(rank));
            weights.push_back(rational_text(element.weight));
        }
        V fields;
        fields["ranks"] = ranks;
        fields["integral_weight"] = true;
        const V ranked = level("candidate_order", fields, weights);
        for (const auto& position : ranked) {
            const size_t index_ = position.asUInt();
            const Element element = current.coll.at(index_);
            const Expr& poly = element.poly;
            if (frame->vanishes(poly)) continue;
            std::vector<std::string> names;
            for (const auto& [h, e] : element.exps.items) if (lookup(fresh, h)) names.push_back(h);
            V candidate_fields;
            candidate_fields["exponents"] = V(Json::arrayValue);
            for (const auto& h : names) candidate_fields["exponents"].append(rational_text(*element.exps.find(h)));
            const V candidate = level("contact_candidate", candidate_fields, {rational_text(element.weight)});
            Ideal factors;
            for (const auto& i : candidate["support"]) factors.push_back(symbol(*lookup(fresh, names.at(i.asUInt()))));
            const Expr monomial = product(factors);
            const unsigned degree = static_cast<unsigned>(std::stoul(candidate["degree"].isString()
                                                                         ? candidate["degree"].asString()
                                                                         : std::to_string(candidate["degree"].asUInt64())));
            std::optional<Strings> points;
            if (!frame->rows.empty() && candidate["probe_direction"].asBool()) {
                points = frame->working_points({}, remaining);
                if (!points) points = frame->working_points(current.locus, remaining);
            }
            const Variables coords = frame->coords;
            for (const auto& k : coords) {
                if (std::find(banned.begin(), banned.end(), Choice{index_, k, attempts}) != banned.end()) continue;
                bool absent = false;
                std::vector<std::string> absent_labels;
                for (const auto& [h, v] : fresh) if (v == k) { absent = true; absent_labels.push_back(h); }
                if (points) {
                    Ideal open = factors;
                    if (absent) open.push_back(symbol(k));
                    const auto active = frame->contact_open_points(*points, open);
                    if (active.empty() || frame->direction_vanishes(poly, k, degree, active)) continue;
                }
                const Ideal chain = frame->derivative_chain(poly, k, degree);
                const Expr z = chain[chain.size() - 2], top = chain.back();
                if (frame->vanishes(top)) continue;
                const Expr unit = frame_function(frame->variables, top * monomial * (absent ? symbol(k) : constant(1)));
                std::vector<Ideal> hit;
                for (const auto& w : remaining)
                    if (!frame->inside(unit, joined(current.locus, w))) hit.push_back(w);
                if (hit.empty()) continue;
                FramePtr inside;
                Frame::Move move;
                std::optional<std::pair<Expr, Expr>> coefs;
                if (frame->rows.empty()) coefs = frame_linear(frame->variables, z, k);
                if (frame->rows.empty() && coefs) {
                    auto [eliminated, mover] = frame->eliminated(z, k, coefs);
                    move = mover;
                    Ideal units = eliminated->units;
                    units.push_back(move(monomial));
                    if (absent) units.push_back(move(symbol(k)));
                    inside = std::make_shared<Frame>(eliminated->variables, eliminated->coords, eliminated->rows, units,
                                                     eliminated->solved);
                } else {
                    std::vector<Ideal> loci;
                    for (const auto& w : hit) loci.push_back(joined(current.locus, w));
                    inside = frame->restricted(z, k, unit, loci, hit);
                    const Variables target = inside->variables;
                    move = [target](const Expr& f) { return frame_function(target, f); };
                }
                std::vector<Element> coll;
                Strings marks_weights;
                for (const auto& e : current.coll) marks_weights.push_back(rational_text(e.weight));
                const V marks = level("contact", V(Json::objectValue), marks_weights);
                for (size_t i = 0; i < current.coll.size(); ++i) {
                    const Element& item = current.coll[i];
                    const V& row_marks = marks[Json::ArrayIndex(i)];
                    Exponents e;
                    for (const auto& [h, x] : item.exps.items)
                        if (std::find(absent_labels.begin(), absent_labels.end(), h) == absent_labels.end())
                            e.items.emplace_back(h, x);
                    Ideal derivatives;
                    if (item.poly == poly && row_marks.size() <= chain.size()) derivatives = chain;
                    else derivatives = frame->derivative_chain(item.poly, k, row_marks.size() - 1);
                    for (size_t j = 0; j < derivatives.size() && j < row_marks.size(); ++j) {
                        Expr moved = move(derivatives[j]);
                        if (!inside->vanishes(moved)) coll.push_back({e, moved, fraction(row_marks[Json::ArrayIndex(j)])});
                    }
                }
                auto ideal = [&](const Ideal& gens) {
                    Ideal out;
                    for (const auto& g : gens) {
                        Expr moved = move(g);
                        if (!moved.zero()) out.push_back(std::move(moved));
                    }
                    return out;
                };
                for (const auto& g : ideal(current.extra))
                    if (!inside->vanishes(g)) coll.push_back({Exponents{}, g, Q(1)});
                State next = current;
                next.frame = inside;
                next.coll = std::move(coll);
                next.extra.clear();
                next.locus = ideal(current.locus);
                next.fresh.clear();
                for (const auto& [h, v] : current.fresh)
                    if (std::find(absent_labels.begin(), absent_labels.end(), h) == absent_labels.end())
                        next.fresh.emplace_back(h, v);
                next.working.clear();
                for (const auto& w : hit) next.working.push_back(ideal(w));
                yielded = true;
                yield_(next, Choice{index_, k, attempts});
                std::vector<Ideal> following;
                for (const auto& w : remaining)
                    if (std::find(hit.begin(), hit.end(), w) == hit.end()) following.push_back(w);
                for (const auto& w : hit) following.push_back(joined(w, {unit}));
                remaining.clear();
                for (const auto& w : following)
                    if (!frame->empty(joined(current.locus, w))) remaining.push_back(w);
                if (remaining.empty()) return;
            }
        }
        // Some point has no pure derivative of the right order that is not zero:
        // change the complementary coordinates linearly.
        ++attempts;
        Variables free, bases;
        for (const auto& c : frame->coords) {
            if (!contains(ctx.fixed, symbol(c))) free.push_back(c);
            bool divisor = false;
            for (const auto& [h, v] : fresh) if (v == c) divisor = true;
            if (!divisor) bases.push_back(c);
        }
        if (attempts > 8 || bases.empty() || free.size() < 1) throw NoContact{"No hypersurface of maximal contact was found."};
        // With one free coordinate that is the only base every shear is empty.
        // An attempt that yielded nothing, with no choice banned for it, then
        // repeats in every later attempt, which may try only fewer choices.
        if (!yielded && free.size() == 1 && bases == free &&
            std::none_of(banned.begin(), banned.end(), [&](const Choice& c) { return std::get<2>(c) == attempts - 1; }))
            throw NoContact{"No hypersurface of maximal contact was found."};
        V shear_fields;
        shear_fields["attempt"] = attempts;
        shear_fields["free"] = Json::UInt(free.size());
        shear_fields["bases"] = Json::UInt(bases.size());
        const V shear = level("contact_shear", shear_fields);
        const Coord base = bases.at(shear["base"].asUInt());
        std::vector<std::pair<Coord, Expr>> swap, back;
        for (size_t i = 0; i < free.size() && i < shear["coefficients"].size(); ++i) {
            if (free[i] == base) continue;
            const Expr value = symbol(free[i]) + scaled(symbol(base), fraction(shear["coefficients"][Json::ArrayIndex(i)]));
            swap.emplace_back(free[i], value);
            back.emplace_back(free[i], symbol(free[i]) - (value - symbol(free[i])));
        }
        if (swap.empty()) continue;
        auto move = [&](const Ideal& gens) {
            Ideal out;
            for (const auto& g : gens) out.push_back(frame_function(frame->variables, substitute(g, swap)));
            return out;
        };
        State next = current;
        next.frame = frame->changed(swap);
        for (auto& element : next.coll) element.poly = frame_function(frame->variables, substitute(element.poly, swap));
        next.extra = move(current.extra);
        next.locus = move(current.locus);
        next.working.clear();
        for (const auto& w : current.working) next.working.push_back(move(w));
        next.undo.push_back(back);
        current = std::move(next);
        for (auto& w : remaining) w = move(w);
    }
}

std::vector<Found> residue(const State& st, Context& ctx);

// engine/centres.py `_groups`.
std::vector<std::pair<std::vector<Ideal>, std::vector<Element>>> groups(const State& st) {
    Frame& frame = *st.frame;
    if (frame.rows.empty()) {
        std::vector<Element> nonzero;
        for (const auto& item : st.coll) if (!item.poly.zero()) nonzero.push_back(item);
        return {{st.working, nonzero}};
    }
    auto distinct_pieces = [&](const std::vector<Ideal>& ideals, bool check_empty) {
        std::vector<Ideal> pieces;
        for (const auto& w : ideals) if (std::find(pieces.begin(), pieces.end(), w) == pieces.end()) pieces.push_back(w);
        if (check_empty) {
            std::vector<Ideal> kept;
            for (const auto& w : pieces) if (!frame.empty(joined(st.locus, w))) kept.push_back(w);
            pieces = std::move(kept);
        }
        if (pieces.size() < 2) return pieces;
        std::vector<std::pair<Ideal, Ideal>> representatives;
        for (const auto& w : pieces) {
            const Ideal basis = ideal_basis(frame.variables, joined(joined(frame.rows, st.locus), w));
            if (std::none_of(representatives.begin(), representatives.end(),
                             [&](const auto& item) { return item.first == basis; }))
                representatives.emplace_back(basis, w);
        }
        std::vector<Ideal> out;
        for (auto& item : representatives) out.push_back(std::move(item.second));
        return out;
    };
    std::vector<Ideal> ideals;
    auto pack = [&](const std::vector<Ideal>& work) {
        V out(Json::arrayValue);
        for (const auto& w : work) {
            auto found = std::find(ideals.begin(), ideals.end(), w);
            if (found == ideals.end()) {
                ideals.push_back(w);
                found = ideals.end() - 1;
            }
            out.append(Json::UInt(found - ideals.begin()));
        }
        return out;
    };
    V fields;
    fields["work"] = pack(distinct_pieces(st.working, false));
    fields["count"] = Json::UInt(st.coll.size());
    const V found = centre_search("grouped_loci", fields, [&](const V& task) -> V {
        const Expr& f = st.coll.at(task["i"].asUInt()).poly;
        const auto operation = task["query"].asString();
        if (operation == "regular") return frame.graph_regular(f);
        if (operation == "vanishes") return frame.vanishes(f);
        std::vector<Ideal> work;
        for (const auto& i : task["work"]) work.push_back(ideals.at(i.asUInt()));
        if (operation == "working_orders") {
            auto points = frame.working_points({}, work);
            if (!points && !st.locus.empty()) points = frame.working_points(st.locus, work);
            if (!points) return V();
            V orders(Json::arrayValue);
            for (const auto& p : *points) {
                const auto order = frame.point_order(f, p);
                orders.append(order ? V(Json::Int64(*order)) : V());
            }
            return orders;
        }
        if (operation == "live") {
            std::vector<Ideal> live;
            for (const auto& w : work) if (!frame.empty(joined(st.locus, w))) live.push_back(w);
            return pack(live);
        }
        const Ideal support = saturate(frame.variables, frame.rows, f);
        std::vector<Ideal> on_pieces, off_pieces;
        for (const auto& w : work) on_pieces.push_back(joined(w, support));
        for (const auto& w : work)
            for (const auto& g : support)
                off_pieces.push_back(saturate(frame.variables, joined(joined(frame.rows, st.locus), w), g));
        const auto on = distinct_pieces(on_pieces, true), off = distinct_pieces(off_pieces, true);
        V answer;
        answer["on"] = pack(on);
        answer["off"] = pack(off);
        bool overlap = false;
        for (const auto& w : off)
            if (!frame.empty(joined(joined(st.locus, w), support))) { overlap = true; break; }
        answer["overlap"] = overlap;
        return answer;
    });
    if (found.isNull()) throw Raise{"GiveUp", "A function vanishes identically near only part of a locus."};
    std::vector<std::pair<std::vector<Ideal>, std::vector<Element>>> out;
    for (const auto& item : found) {
        std::vector<Ideal> work;
        for (const auto& i : item[0]) work.push_back(ideals.at(i.asUInt()));
        std::vector<Element> selected;
        for (const auto& i : item[1]) selected.push_back(st.coll.at(i.asUInt()));
        out.emplace_back(std::move(work), std::move(selected));
    }
    return out;
}

// engine/centres.py `_residual`.
std::vector<Found> residual(const State& st, Context& ctx) {
    std::vector<Found> out;
    for (auto& [working, coll] : groups(st)) {
        State next = st;
        next.working = working;
        if (coll.empty()) {
            Invariant value = st.prefix;
            value.push_back({true, Q(0)});
            out.push_back({value, V(Json::arrayValue), st.locus, next});
        } else {
            next.coll = coll;
            for (auto& item : residue(next, ctx)) out.push_back(std::move(item));
        }
    }
    return out;
}

// engine/centres.py `_level`.
std::vector<Found> level_(const State& st, Context& ctx) {
    check_time();
    Frame& frame = *st.frame;
    const auto& fresh = st.fresh;
    auto here = [&](const Ideal& gens) {
        for (const auto& w : st.working)
            if (!frame.empty(joined(joined(st.locus, gens), w))) return true;
        return false;
    };
    std::vector<std::string> labels;
    for (const auto& [h, v] : fresh) labels.push_back(h);
    V records(Json::arrayValue);
    for (const auto& h : labels) {
        auto found = ctx.records.find(h);
        if (found == ctx.records.end() || !found->second) {
            records.append(V());
            continue;
        }
        V row(Json::arrayValue);
        for (Json::ArrayIndex i = 0; i < found->second->size() && i < st.prefix.size(); ++i)
            row.append((*found->second)[i]);
        records.append(row);
    }
    V fields;
    fields["records"] = records;
    fields["prefix"] = invariant_row(st.prefix);
    std::vector<std::string> old;
    for (const auto& i : level("old_divisors", fields)) old.push_back(labels.at(i.asUInt()));
    std::vector<std::string> through;
    for (const auto& h : old) if (here({symbol(*lookup(fresh, h))})) through.push_back(h);
    const auto subsets = divisor_subsets(through, [&](const std::vector<std::string>& chosen) {
        Ideal gens;
        for (const auto& h : chosen) gens.push_back(symbol(*lookup(fresh, h)));
        return here(gens);
    }, true);
    std::vector<Found> out;
    for (const auto& chosen : subsets) {
        Ideal locus = st.locus;
        for (const auto& h : chosen) locus.push_back(symbol(*lookup(fresh, h)));
        std::vector<Ideal> working;
        for (const auto& w : st.working) if (!frame.empty(joined(locus, w))) working.push_back(w);
        std::vector<std::pair<std::string, Coord>> rest;
        for (const auto& [h, v] : fresh)
            if (std::find(old.begin(), old.end(), h) == old.end()) rest.emplace_back(h, v);
        std::vector<Element> coll;
        for (const auto& element : st.coll) coll.push_back(settle(element, frame, fresh, chosen, rest));
        State start = st;
        start.coll = coll;
        start.extra.clear();
        for (const auto& h : chosen) start.extra.push_back(symbol(*lookup(fresh, h)));
        start.fresh = rest;
        start.locus = locus;
        start.working = working;
        start.prefix.push_back({false, Q(long(chosen.size()))});
        std::vector<Choice> banned;
        while (true) {
            std::vector<Found> part;
            std::optional<Choice> choice;
            try {
                contact(start, ctx, banned, [&](const State& piece, const Choice& made) {
                    choice = made;
                    for (auto& item : residual(piece, ctx)) part.push_back(std::move(item));
                    choice.reset();
                });
            } catch (const NoContact&) {
                if (!choice || std::find(banned.begin(), banned.end(), *choice) != banned.end()) throw;
                banned.push_back(*choice);
                continue;
            }
            for (auto& item : part) out.push_back(std::move(item));
            break;
        }
    }
    return out;
}

// engine/centres.py `_residue`.
std::vector<Found> residue(const State& st, Context& ctx) {
    check_time();
    Frame& frame = *st.frame;
    const auto& fresh = st.fresh;
    auto here = [&](const Ideal& gens) {
        for (const auto& w : st.working)
            if (!frame.empty(joined(joined(st.locus, gens), w))) return true;
        return false;
    };
    std::vector<std::string> live;
    for (const auto& [h, v] : fresh) if (here({symbol(v)})) live.push_back(h);
    std::vector<Element> normal_;
    for (const auto& element : st.coll) {
        Exponents exps;
        for (const auto& h : live) {
            const Q* e = element.exps.find(h);
            exps.items.emplace_back(h, e ? *e : Q(0));
        }
        Expr poly = element.poly;
        if (!live.empty()) {
            Variables divisors;
            for (const auto& h : live) divisors.push_back(*lookup(fresh, h));
            auto [orders, residual_] = remove_coordinate_powers(frame, poly, divisors);
            poly = residual_;
            V fields;
            fields["orders"] = V(Json::arrayValue);
            for (const auto& o : orders) fields["orders"].append(o.get_str());
            Strings weights;
            for (const auto& [h, e] : exps.items) weights.push_back(rational_text(e));
            const V adjusted = level("add_orders", fields, weights);
            Exponents updated;
            for (size_t i = 0; i < live.size(); ++i) updated.items.emplace_back(live[i], fraction(adjusted[Json::ArrayIndex(i)]));
            exps = std::move(updated);
        }
        normal_.push_back({exps, poly, element.weight});
    }
    const auto presentation = Presentation(live, normal_).residual();
    const Exponents alpha = presentation.at(0);
    std::vector<Element> rest;
    for (size_t i = 0; i < normal_.size(); ++i) rest.push_back({presentation.at(i + 1), normal_[i].poly, normal_[i].weight});
    const Presentation table(live, rest);
    const auto strata = divisor_subsets(live, [&](const std::vector<std::string>& chosen) {
        Ideal gens;
        for (const auto& h : chosen) gens.push_back(symbol(*lookup(fresh, h)));
        return here(gens);
    });
    std::vector<std::vector<size_t>> indices;
    for (const auto& chosen : strata) {
        std::vector<size_t> ids;
        for (const auto& h : chosen) ids.push_back(std::find(live.begin(), live.end(), h) - live.begin());
        indices.push_back(std::move(ids));
    }
    auto conditions = [&](const Q& value) {
        std::vector<Ideal> out;
        const auto required = table.requirements(indices, value);
        for (size_t s = 0; s < strata.size() && s < required.size(); ++s) {
            Ideal gens;
            for (const auto& h : strata[s]) gens.push_back(symbol(*lookup(fresh, h)));
            for (size_t i = 0; i < rest.size() && i < required[s].size(); ++i)
                if (required[s][i] >= 1)
                    for (auto& d : frame.jet(rest[i].poly, static_cast<unsigned>(required[s][i].get_ui()))) gens.push_back(d);
            out.push_back(std::move(gens));
        }
        return out;
    };
    auto point_maximum = [&]() -> std::optional<Q> {
        std::vector<std::pair<std::vector<size_t>, std::vector<long>>> orders;
        for (size_t s = 0; s < strata.size(); ++s) {
            check_time();
            Ideal divisor;
            for (const auto& h : strata[s]) divisor.push_back(symbol(*lookup(fresh, h)));
            const auto points = frame.working_points(joined(st.locus, divisor), st.working);
            if (!points) return std::nullopt;
            for (const auto& point : *points) {
                std::vector<long> row;
                for (const auto& item : rest) {
                    const auto order = frame.point_order(item.poly, point);
                    if (!order) return std::nullopt;
                    row.push_back(*order);
                }
                orders.emplace_back(indices[s], row);
            }
        }
        if (orders.empty()) return std::nullopt;
        return table.point_maximum(orders);
    };
    std::optional<Q> value;
    if (!frame.rows.empty()) value = point_maximum();
    if (!value) {
        std::vector<std::vector<unsigned>> bounds;
        for (const auto& chosen : strata) {
            Ideal divisor;
            for (const auto& h : chosen) divisor.push_back(symbol(*lookup(fresh, h)));
            std::vector<unsigned> orders;
            for (const auto& item : rest)
                orders.push_back(jet_order(0, [&](unsigned k) { return here(joined(divisor, frame.jet(item.poly, k))); }));
            bounds.push_back(std::move(orders));
        }
        const auto values = table.candidate_values(indices, bounds);
        V fields;
        fields["count"] = Json::UInt(values.size());
        const V selected = centre_search("maximum_threshold", fields, [&](const V& task) -> V {
            check_time();
            for (const auto& gens : conditions(values.at(task["i"].asUInt())))
                if (here(gens)) return true;
            return false;
        });
        value = values.at(selected.asUInt());
    }
    if (*value == 0) {
        // The monomial case [BM, (3.2)].
        V fields;
        fields["strata"] = V(Json::arrayValue);
        for (const auto& ids : indices) {
            V row(Json::arrayValue);
            for (auto id : ids) row.append(Json::UInt(id));
            fields["strata"].append(row);
        }
        Strings weights;
        for (const auto& h : live) weights.push_back(rational_text(*alpha.find(h)));
        std::vector<std::vector<std::string>> sets;
        for (const auto& i : level("monomial", fields, weights)) sets.push_back(strata.at(i.asUInt()));
        if (sets.empty()) throw Raise{"GiveUp", "The monomial part of a presentation has order less than one."};
        V birth_fields;
        birth_fields["births"] = strings(ctx.births);
        birth_fields["groups"] = V(Json::arrayValue);
        for (const auto& set : sets) birth_fields["groups"].append(strings(set));
        const V keys = level("birth_keys", birth_fields);
        std::vector<std::optional<std::pair<Invariant, V>>> ranked;
        for (const auto& k : keys) ranked.emplace_back(std::make_pair(Invariant{}, k));
        const size_t chosen = maximum_indices(ranked)[0].asUInt();
        Invariant prefix = st.prefix;
        prefix.push_back({false, Q(0)});
        Ideal centre = st.locus;
        for (const auto& h : sets.at(chosen)) centre.push_back(symbol(*lookup(fresh, h)));
        return {{prefix, keys[Json::ArrayIndex(chosen)], centre, st}};
    }
    V fields;
    fields["threshold"] = V(Json::arrayValue);
    fields["threshold"].append(rational_text(*value));
    Strings weights;
    for (const auto& item : rest) weights.push_back(rational_text(item.weight));
    const V next_weights = level("next_level", fields, weights);
    std::vector<Element> coll;
    for (size_t i = 0; i < rest.size(); ++i)
        coll.push_back({rest[i].exps, rest[i].poly, fraction(next_weights["scaled"][Json::ArrayIndex(i)])});
    if (next_weights.isMember("extra")) coll.push_back({alpha, constant(1), fraction(next_weights["extra"])});
    std::vector<Found> out;
    for (const auto& gens : conditions(*value)) {
        std::vector<Ideal> working;
        for (const auto& w : st.working) if (!frame.empty(joined(joined(st.locus, gens), w))) working.push_back(w);
        if (working.empty()) continue;
        State next = st;
        next.coll = coll;
        next.locus = joined(st.locus, gens);
        next.working = working;
        next.prefix.push_back({false, *value});
        for (auto& item : level_(next, ctx)) out.push_back(std::move(item));
    }
    return out;
}

// engine/centres.py `_principal`.
Expr principal(const Variables& variables, const Ideal& ambient, const Ideal& ideal) {
    V fields;
    fields["width"] = Json::UInt(variables.size());
    fields["ranks"] = V(Json::arrayValue);
    for (const auto& g : ideal) {
        const bool coordinate = g.is_symbol() &&
            std::find(variables.begin(), variables.end(), g.coords()[0]) != variables.end();
        fields["ranks"].append(invariant_row({{false, Q(coordinate ? 0 : 1)}, {false, Q(count_ops(g))}}));
    }
    fields["ambient"] = centre_rows(variables, ambient);
    fields["ideal"] = centre_rows(variables, ideal);
    const V selected = level("principal", fields);
    if (selected.isNull()) throw Raise{"GiveUp", "A divisor is not given by one equation on a chart."};
    return ideal.at(selected.asUInt());
}

Ideal intersection(const Variables& variables, const Ideal& first, const Ideal& second) {
    V fields;
    fields["width"] = Json::UInt(variables.size());
    fields["first"] = centre_rows(variables, first);
    fields["second"] = centre_rows(variables, second);
    Ideal out;
    for (const auto& line : level("locus_intersection", fields)) out.push_back(decode(variables, line.asString()));
    return out;
}

Ideal cover_products(const Variables& variables, const std::vector<Ideal>& holes) {
    V fields;
    fields["width"] = Json::UInt(variables.size());
    fields["holes"] = V(Json::arrayValue);
    for (const auto& hole : holes) fields["holes"].append(centre_rows(variables, hole));
    Ideal out;
    for (const auto& line : level("cover_products", fields)) out.push_back(decode(variables, line.asString()));
    return out;
}

// engine/centres.py `_frames` with `eliminate`.
struct Piece {
    FramePtr frame;
    std::vector<Ideal> live;
    Frame::Move move;
};

std::vector<Piece> frames(const Variables& variables, const Ideal& ambient, unsigned dimension, const Ideal& divisors,
                          const std::vector<Ideal>& working, const Ideal& units) {
    std::vector<Ideal> remaining = working;
    std::vector<Piece> found;
    const std::optional<Expr> gate = units.empty() ? std::nullopt
                                                   : std::optional<Expr>(frame_function(variables, product(units)));
    auto left = [&](const Ideal& w) {
        return !empty_locus(variables, joined(ambient, w)) &&
               (!gate || !radical_subset(variables, {*gate}, joined(ambient, w)));
    };
    const Frame::Move same = [](const Expr& f) { return f; };
    V fields;
    fields["width"] = Json::UInt(variables.size());
    fields["dimension"] = dimension;
    fields["equations"] = Json::UInt(ambient.size());
    fields["divisors"] = V(Json::arrayValue);
    for (const auto& v : variables) fields["divisors"].append(contains(divisors, symbol(v)));
    const V complete = centre_search("frame_cover", fields, [&](const V& task) -> V {
        if (task["query"].asString() == "affine") {
            found.push_back({std::make_shared<Frame>(variables, variables, Ideal{}, units), remaining, same});
            return true;
        }
        Variables dependent;
        for (const auto& j : task["dependent"]) dependent.push_back(variables.at(j.asUInt()));
        Ideal equations;
        for (const auto& j : task["rows"]) equations.push_back(ambient.at(j.asUInt()));
        const Expr minor = frame_minor(variables, equations, dependent);
        if (minor.zero()) return false;
        Ideal factors{minor};
        for (const auto& v : dependent) if (contains(divisors, symbol(v))) factors.push_back(symbol(v));
        factors.push_back(product(units));
        const Expr away = frame_function(variables, product(factors));
        std::vector<Ideal> hit;
        for (const auto& w : remaining) if (!radical_subset(variables, {away}, joined(ambient, w))) hit.push_back(w);
        if (hit.empty()) return false;
        Variables coords;
        for (const auto& v : variables)
            if (std::find(dependent.begin(), dependent.end(), v) == dependent.end()) coords.push_back(v);
        auto frame = std::make_shared<Frame>(variables, coords, equations, Ideal{away});
        std::optional<std::pair<Expr, Expr>> coefs;
        if (task["linear"].asBool()) coefs = frame_linear(variables, equations.at(0), dependent.at(0));
        if (task["linear"].asBool() && coefs) {
            auto [inside, move] = Frame(variables, variables).eliminated(equations.at(0), dependent.at(0), coefs);
            Ideal moved_units = inside->units;
            if (contains(divisors, symbol(dependent.at(0)))) moved_units.push_back(move(symbol(dependent.at(0))));
            for (const auto& u : units) moved_units.push_back(move(u));
            found.push_back({std::make_shared<Frame>(inside->variables, inside->coords, inside->rows, moved_units,
                                                     inside->solved), hit, move});
        } else {
            found.push_back({frame, hit, same});
        }
        std::vector<Ideal> following;
        for (const auto& w : remaining) if (std::find(hit.begin(), hit.end(), w) == hit.end()) following.push_back(w);
        for (const auto& w : hit) following.push_back(joined(w, {away}));
        remaining.clear();
        for (const auto& w : following) if (left(w)) remaining.push_back(w);
        return remaining.empty();
    });
    if (!complete.asBool()) throw Raise{"GiveUp", "No coordinates were found on a chart."};
    return found;
}

// engine/centres.py `_affine_maximum`: the complete native descent on ordinary affine charts.
std::optional<V> affine_maximum(const V& request, const Variables& variables, const Ideal& ambient, unsigned dimension,
                                const Ideal& strict, const std::vector<std::pair<std::string, Ideal>>& divisors,
                                const std::vector<Ideal>& working, const std::vector<Ideal>& away) {
    if (!ambient.empty() || !away.empty() || dimension != variables.size() || dimension > 3 || strict.size() != 1 ||
        divisors.size() > 1 || request["gpu"].asBool())
        return std::nullopt;
    V fields;
    fields["width"] = dimension;
    fields["strict"] = json_rows(variables, strict);
    fields["working"] = V(Json::arrayValue);
    for (const auto& w : working) fields["working"].append(json_rows(variables, w));
    fields["divisor"] = V();
    fields["birth"] = V();
    if (!divisors.empty()) {
        fields["divisor"] = json_rows(variables, divisors[0].second);
        const auto& records = request["records"];
        if (records.isMember(divisors[0].first) && !records[divisors[0].first].isNull())
            fields["birth"] = records[divisors[0].first];
    }
    const V result = kernel_json(atlas_affine_centre_maximum, "maximum", fields);
    if (result.isNull()) return std::nullopt;
    V out;
    out["value"] = V(Json::arrayValue);
    for (const auto& entry : result["value"])
        out["value"].append(entry.isIntegral() ? V(std::to_string(entry.asInt64())) : entry);
    out["key"] = V(Json::arrayValue);
    out["centre"] = V(Json::arrayValue);
    for (const auto& line : result["ideal"]) out["centre"].append(row(variables, decode(variables, line.asString())));
    return out;
}

// engine/centres.py `maximum`.
V maximum(const V& request) {
    Variables original;
    for (Json::ArrayIndex i = 0; i < request["names"].size(); ++i)
        original.push_back({request["names"][i].asString(), request["identities"][i].asInt64()});
    auto ideal_of = [&](const V& lines) {
        Ideal out;
        for (const auto& line : lines) out.push_back(decode(original, line.asString()));
        return out;
    };
    const Ideal chart = ideal_of(request["ambient"]);
    const unsigned dimension = request["dimension"].asUInt();
    Ideal strict = ideal_of(request["strict"]);
    std::vector<std::pair<std::string, Ideal>> divisors;
    for (const auto& item : request["divisors"]) divisors.emplace_back(item[0].asString(), ideal_of(item[1]));
    std::vector<Ideal> working, away;
    for (const auto& w : request["working"]) working.push_back(ideal_of(w));
    for (const auto& z : request["away"]) away.push_back(ideal_of(z));
    if (!settings->deadline || std::chrono::steady_clock::now() <= *settings->deadline) {
        if (auto direct = affine_maximum(request, original, chart, dimension, strict, divisors, working, away)) {
            check_time();
            return *direct;
        }
    }
    Ideal kept;
    for (const auto& g : strict) if (!ideal_subset(original, {g}, chart)) kept.push_back(g);
    strict = kept;
    std::vector<Ideal> live;
    for (const auto& w : working) if (!empty_locus(original, joined(joined(chart, strict), w))) live.push_back(w);
    working = live;
    if (working.empty() || strict.empty()) return V();
    // Every divisor through the set must be a coordinate hyperplane: one that
    // is not becomes one on the graph of its equation.
    std::vector<std::pair<std::string, Coord>> names;
    std::vector<std::pair<Coord, Expr>> graph;
    Variables variables = original;
    Ideal ambient = chart;
    for (const auto& [label, ideal] : divisors) {
        if (std::all_of(working.begin(), working.end(),
                        [&](const Ideal& w) { return empty_locus(original, joined(joined(chart, ideal), w)); }))
            continue;
        Expr g = principal(original, chart, ideal);
        if (!(g.is_symbol() && std::find(original.begin(), original.end(), g.coords()[0]) != original.end())) {
            const Coord name = dummy("divisor_" + label);
            variables.push_back(name);
            graph.emplace_back(name, g);
            ambient.push_back(symbol(name) - g);
            g = symbol(name);
        }
        names.emplace_back(label, g.coords()[0]);
    }
    Context ctx;
    for (const auto& name : request["records"].getMemberNames()) {
        const V& record = request["records"][name];
        ctx.records[name] = record.isNull() ? std::nullopt : std::optional<V>(record);
    }
    for (const auto& label : request["births"]) ctx.births.push_back(label.asString());
    Ideal divisor_names;
    for (const auto& [label, v] : names) divisor_names.push_back(symbol(v));
    ctx.fixed = divisor_names;
    std::vector<Ideal> meeting;
    for (const auto& z : away)
        if (std::any_of(working.begin(), working.end(), [&](const Ideal& w) {
                return !empty_locus(original, joined(joined(joined(chart, strict), z), w));
            }))
            meeting.push_back(z);
    const Ideal covers = cover_products(original, meeting);
    std::vector<std::pair<Ideal, std::vector<Ideal>>> pieces;
    std::vector<Ideal> left = working;
    Ideal seen;
    for (const auto& unit : covers) {
        if (contains(seen, unit)) continue;
        seen.push_back(unit);
        if (unit == constant(1)) {
            pieces.emplace_back(Ideal{}, left);
            break;
        }
        std::vector<Ideal> part;
        for (const auto& w : left)
            if (!radical_subset(original, {unit}, joined(joined(chart, strict), w))) part.push_back(w);
        if (!part.empty()) pieces.emplace_back(Ideal{unit}, part);
        std::vector<Ideal> next;
        for (const auto& w : left) {
            const Ideal extended = joined(w, {unit});
            if (!empty_locus(original, joined(joined(chart, strict), extended))) next.push_back(extended);
        }
        left = std::move(next);
        if (left.empty()) break;
    }
    std::vector<Found> found;
    std::vector<Piece> all;
    for (const auto& [units, part] : pieces)
        for (auto& piece : frames(variables, ambient, dimension, divisor_names, part, units)) all.push_back(std::move(piece));
    for (auto& piece : all) {
        Frame& frame = *piece.frame;
        Ideal own;
        for (const auto& g : strict) {
            Expr moved = piece.move(g);
            if (!moved.zero()) own.push_back(std::move(moved));
        }
        std::vector<Ideal> pieces_live;
        for (const auto& w : piece.live) {
            Ideal moved_piece;
            for (const auto& g : w) {
                Expr moved = piece.move(g);
                if (!moved.zero()) moved_piece.push_back(std::move(moved));
            }
            pieces_live.push_back(std::move(moved_piece));
        }
        std::vector<std::pair<std::string, Coord>> present;
        for (const auto& [label, v] : names)
            if (std::find(frame.variables.begin(), frame.variables.end(), v) != frame.variables.end())
                present.emplace_back(label, v);
        const unsigned order = jet_order(1, [&](unsigned k) {
            Ideal jets;
            for (const auto& g : own) for (auto& d : frame.jet(g, k)) jets.push_back(std::move(d));
            for (const auto& w : pieces_live) if (!frame.empty(joined(jets, w))) return true;
            return false;
        });
        Ideal locus;
        for (const auto& g : own) for (auto& d : frame.jet(g, order)) locus.push_back(std::move(d));
        std::vector<Ideal> live_pieces;
        for (const auto& w : pieces_live) if (!frame.empty(joined(locus, w))) live_pieces.push_back(w);
        State state;
        state.frame = piece.frame;
        for (const auto& g : own) state.coll.push_back({Exponents{}, g, Q(long(order))});
        state.fresh = present;
        state.locus = locus;
        state.working = live_pieces;
        state.prefix = {{false, Q(long(order))}};
        for (auto& item : level_(state, ctx)) found.push_back(std::move(item));
    }
    if (found.empty()) return V();
    std::vector<std::optional<std::pair<Invariant, V>>> ranked;
    for (const auto& item : found) ranked.emplace_back(std::make_pair(item.value, item.key));
    const V indices = maximum_indices(ranked);
    const Found& best = found.at(indices[0].asUInt());
    std::optional<Ideal> centre;
    for (const auto& position : indices) {
        const Found& item = found.at(position.asUInt());
        const Frame& frame = *item.state.frame;
        auto back = [&](Ideal gens) {
            for (auto swap = item.state.undo.rbegin(); swap != item.state.undo.rend(); ++swap)
                for (auto& g : gens) g = frame_function(variables, substitute(g, *swap));
            Ideal out;
            for (const auto& g : gens) {
                Expr moved = frame_function(original, substitute(g, graph));
                if (!moved.zero()) out.push_back(std::move(moved));
            }
            return out;
        };
        // `back` is a substitution, a ring map: the images of any generators
        // of the locus generate its image. The basis in the frame, where the
        // centre is simplest, keeps the generators carried back small.
        Ideal locus = joined(joined(item.centre, frame.rows), frame.solved);
        if (variables.size() <= MAX_VARIABLES) {
            try {
                locus = ideal_basis(variables, locus);
            } catch (const Declined&) {
            }
        }
        Ideal gens = ideal_basis(original, joined(chart, back(locus)));
        for (const auto& unit : back(frame.units)) gens = saturate(original, gens, unit);
        gens = ideal_basis(original, gens);
        centre = !centre || *centre == gens ? gens : intersection(original, *centre, gens);
    }
    V out;
    out["value"] = invariant_row(best.value);
    out["key"] = best.key;
    out["centre"] = json_rows(original, *centre);
    return out;
}


// ---------------------------------------------------------------- surfaces

struct NoModel {
    std::string message;
};

extern "C" const char* atlas_surface_centre_choice(const char*) noexcept;

// The key of a step of engine/surface.py: (kind, top, flag, labels).
struct SurfaceStep {
    V kind, top, flag;
    std::vector<std::string> labels;
    std::string type;  // "curve" or "points"
    Ideal centre;
};

V surface_row(const SurfaceStep& step) {
    V row(Json::arrayValue), head(Json::arrayValue);
    for (const V* value : {&step.kind, &step.top, &step.flag})
        head.append(value->isString() ? *value : V(value->isUInt64() ? std::to_string(value->asUInt64())
                                                                     : std::to_string(value->asInt64())));
    row.append(head);
    row.append(strings(step.labels));
    return row;
}

std::vector<size_t> ordered_parts(const Variables& variables, const std::vector<Ideal>& parts) {
    std::vector<std::pair<Coord, size_t>> sorted;
    for (size_t i = 0; i < variables.size(); ++i) sorted.emplace_back(variables[i], i);
    std::sort(sorted.begin(), sorted.end());
    const size_t width = std::to_string(variables.size()).size();
    std::vector<std::string> keys(variables.size());
    for (size_t rank = 0; rank < sorted.size(); ++rank) {
        std::string key = std::to_string(rank);
        keys[sorted[rank].second] = std::string(width > key.size() ? width - key.size() : 0, '0') + key;
    }
    V fields;
    fields["coordinate_keys"] = strings(keys);
    fields["parts"] = V(Json::arrayValue);
    for (const auto& part : parts) fields["parts"].append(centre_rows(variables, part));
    fields["degrees"] = V(Json::arrayValue);
    std::vector<size_t> out;
    for (const auto& i : level("ordered_parts", fields)) out.push_back(i.asUInt());
    return out;
}

// engine/surface.py `_best`.
std::optional<SurfaceStep> best(const Variables& variables, const std::vector<SurfaceStep>& found) {
    if (found.empty()) return std::nullopt;
    V fields;
    fields["values"] = V(Json::arrayValue);
    for (const auto& step : found) fields["values"].append(surface_row(step));
    const V indices = level("surface_maximum", fields);
    SurfaceStep out = found.at(indices[0].asUInt());
    std::vector<Ideal> centres;
    for (const auto& i : indices) {
        const Ideal& c = found.at(i.asUInt()).centre;
        if (std::find(centres.begin(), centres.end(), c) == centres.end()) centres.push_back(c);
    }
    const auto order = ordered_parts(variables, centres);
    Ideal centre = centres.at(order[0]);
    for (size_t j = 1; j < order.size(); ++j)
        centre = ideal_basis(variables, intersection(variables, centre, centres.at(order[j])));
    out.centre = centre;
    return out;
}

// engine/surface.py `_line_centre`: the native rational-graph certificate in an affine threefold.
std::optional<SurfaceStep> line_centre(const Variables& variables, const Ideal& ambient, const Ideal& strict,
                                       const std::vector<std::pair<std::string, Ideal>>& divisors,
                                       const std::vector<Ideal>& working, const std::vector<Ideal>& away) {
    if (variables.size() != 3 || !ambient.empty() || strict.size() != 1 ||
        std::any_of(divisors.begin(), divisors.end(), [](const auto& d) { return d.second.size() != 1; }))
        return std::nullopt;
    V fields;
    fields["surface"] = row(variables, strict[0]);
    fields["divisors"] = V(Json::arrayValue);
    for (const auto& [label, ideal] : divisors) {
        V item(Json::arrayValue);
        item.append(label);
        item.append(row(variables, ideal[0]));
        fields["divisors"].append(item);
    }
    fields["work"] = V(Json::arrayValue);
    for (const auto& w : working) fields["work"].append(json_rows(variables, w));
    fields["holes"] = V(Json::arrayValue);
    for (const auto& h : away) fields["holes"].append(json_rows(variables, h));
    const V found = kernel_json(atlas_surface_centre_choice, "curve", fields);
    if (found.isNull()) return std::nullopt;
    SurfaceStep step{V(2), found["top"], V(0), {}, "curve", {}};
    for (const auto& label : found["labels"]) step.labels.push_back(label.asString());
    for (const auto& line : found["ideal"]) step.centre.push_back(decode(variables, line.asString()));
    return step;
}

// One model of engine/surface.py `_models`.
struct Model {
    Variables names;
    Ideal units;
    std::vector<std::pair<std::string, Expr>> functions;
    std::vector<Ideal> work;
    Ideal left;
    size_t threefold = 0;
};

bool open_empty(const Variables& variables, const Ideal& units, const Ideal& gens) {
    return frame_empty(variables, gens, units);
}

// engine/surface.py `_models`: each model is handed to `yield_` as it is found.
void models(const Variables& variables, const Ideal& equations, const std::vector<std::pair<std::string, Expr>>& functions,
            const std::vector<Ideal>& working, size_t threefold, const std::function<void(const Model&)>& yield_) {
    struct Pending {
        Variables names;
        std::vector<std::pair<Expr, bool>> tagged;
        std::vector<std::pair<std::string, Expr>> fns;
        Ideal units;
        std::vector<Ideal> work;
    };
    std::vector<Pending> pending;
    {
        Pending first{variables, {}, functions, {}, working};
        for (size_t i = 0; i < equations.size(); ++i)
            if (!equations[i].zero()) first.tagged.emplace_back(equations[i], i < threefold);
        pending.push_back(std::move(first));
    }
    while (!pending.empty()) {
        Pending item = std::move(pending.back());
        pending.pop_back();
        Ideal eqs;
        for (const auto& [e, flag] : item.tagged) eqs.push_back(e);
        if (eqs.empty()) {
            if (item.names.size() != 2) throw NoModel{"A chart does not reduce to a plane."};
            yield_(Model{item.names, item.units, item.fns, item.work, {}, 0});
            continue;
        }
        std::optional<std::pair<Expr, Expr>> selected;
        V fields;
        fields["width"] = Json::UInt(item.names.size());
        fields["costs"] = V(Json::arrayValue);
        for (const auto& [e, flag] : item.tagged) fields["costs"].append(Json::Int64(count_ops(e)));
        const V choice = centre_search("surface_model", fields, [&](const V& task) -> V {
            const Expr& e = item.tagged.at(task["i"].asUInt()).first;
            const Coord& y = item.names.at(task["axis"].asUInt());
            const auto linear = frame_linear(item.names, e, y);
            if (!linear) return V();
            V hit(Json::arrayValue);
            Ideal units = item.units;
            units.push_back(linear->first);
            for (size_t j = 0; j < item.work.size(); ++j)
                if (!open_empty(item.names, units, joined(eqs, item.work[j]))) hit.append(Json::UInt(j));
            if (hit.empty()) return V();
            selected = linear;
            return hit;
        });
        if (!choice.isNull()) {
            const size_t chosen = choice["i"].asUInt();
            const Coord y = item.names.at(choice["axis"].asUInt());
            const auto [a, b] = *selected;
            std::vector<Ideal> hit;
            for (const auto& j : choice["hit"]) hit.push_back(item.work.at(j.asUInt()));
            Variables rest;
            for (const auto& v : item.names) if (v != y) rest.push_back(v);
            const Variables names = item.names;
            auto move = [&](const Expr& f) { return frame_restrict(names, f, y, a, b); };
            auto ideal = [&](const Ideal& gens) {
                Ideal out;
                for (const auto& g : gens) {
                    Expr moved = move(g);
                    if (!moved.zero()) out.push_back(std::move(moved));
                }
                return out;
            };
            Pending next{rest, {}, {}, ideal(item.units), {}};
            for (size_t i = 0; i < item.tagged.size(); ++i) {
                if (i == chosen) continue;
                Expr moved = move(item.tagged[i].first);
                if (!moved.zero()) next.tagged.emplace_back(std::move(moved), item.tagged[i].second);
            }
            for (const auto& [label, f] : item.fns) next.fns.emplace_back(label, move(f));
            next.units.push_back(a);
            for (const auto& w : hit) next.work.push_back(ideal(w));
            pending.push_back(std::move(next));
            std::vector<Ideal> left;
            for (const auto& w : item.work) if (std::find(hit.begin(), hit.end(), w) == hit.end()) left.push_back(w);
            if (!a.number()) for (const auto& w : hit) left.push_back(joined(w, {a}));
            std::vector<Ideal> kept;
            for (const auto& w : left) if (!open_empty(item.names, item.units, joined(eqs, w))) kept.push_back(w);
            if (!kept.empty()) pending.push_back(Pending{item.names, item.tagged, item.fns, item.units, kept});
        } else {
            std::vector<Ideal> work;
            for (const auto& w : item.work) if (!open_empty(item.names, item.units, joined(eqs, w))) work.push_back(w);
            if (!work.empty()) {
                Ideal left;
                size_t first = 0;
                for (const auto& [e, flag] : item.tagged) if (flag) { left.push_back(e); ++first; }
                for (const auto& [e, flag] : item.tagged) if (!flag) left.push_back(e);
                yield_(Model{item.names, item.units, item.fns, work, left, first});
            }
        }
    }
}

// engine/surface.py `_on_model` with the kernel's op "surface".
std::vector<SurfaceStep> on_model(const Model& model) {
    V label_fields;
    std::vector<std::string> names;
    for (const auto& [label, f] : model.functions) names.push_back(label);
    label_fields["labels"] = strings(names);
    std::vector<std::string> labels;
    for (const auto& l : level("label_order", label_fields)) labels.push_back(l.asString());
    Ideal functions;
    for (const auto& label : labels)
        for (const auto& [l, f] : model.functions) if (l == label) { functions.push_back(f); break; }
    const Variables& variables = model.names;
    if (variables.size() + 1 > MAX_VARIABLES) throw Declined{"The surface step exceeds the packed variable range."};
    Ideal lines = joined(joined(model.left, functions), model.units);
    std::string sizes;
    for (const auto& w : model.work) {
        sizes += " " + std::to_string(w.size());
        lines = joined(lines, w);
    }
    Strings answer;
    try {
        answer = ideal_request("surface " + std::to_string(variables.size()) + " " + std::to_string(model.threefold) +
                               " " + std::to_string(model.left.size()) + " " + std::to_string(functions.size()) + " " +
                               std::to_string(model.units.size()) + " " + std::to_string(model.work.size()) + sizes,
                               rows(variables, lines));
    } catch (const Unavailable& error) {
        throw Declined{error.message};
    }
    if (answer == Strings{"nomodel"}) throw NoModel{"No two variables are coordinates of the surface near some point."};
    std::vector<SurfaceStep> out;
    for (size_t at = 0; at < answer.size() && answer[at] != "none";) {
        std::istringstream words(answer[at]);
        Strings fields;
        for (std::string word; words >> word;) fields.push_back(word);
        const size_t count_labels = std::stoul(fields.at(4));
        const size_t count = std::stoul(fields.at(5 + count_labels));
        SurfaceStep step{V(Json::Int64(std::stoll(fields[1]))), V(fields[2]), V(Json::Int64(std::stoll(fields[3]))), {},
                         "", decode_all(variables, answer, at + 1, count)};
        for (size_t j = 0; j < count_labels; ++j) step.labels.push_back(labels.at(std::stoul(fields[5 + j])));
        step.type = step.kind.asInt64() == 2 && step.flag.asInt64() == 0 ? "curve" : "points";
        out.push_back(std::move(step));
        at += 1 + count;
    }
    return out;
}

// engine/surface.py `_step`.
std::optional<SurfaceStep> surface_steps(const Variables& variables, const Ideal& ambient, const Ideal& strict,
                                         const std::vector<std::pair<std::string, Expr>>& functions,
                                         const std::vector<Ideal>& working) {
    std::vector<SurfaceStep> found;
    models(variables, joined(ambient, strict), functions, working, ambient.size(), [&](const Model& model) {
        for (auto& step : on_model(model)) {
            Ideal centre = ideal_basis(variables, joined(joined(ambient, strict), step.centre));
            for (const auto& unit : model.units)
                if (!unit.number()) centre = saturate(variables, centre, unit);
            centre = ideal_basis(variables, centre);
            if (centre.size() == 1 && centre[0] == constant(1)) continue;
            step.centre = centre;
            found.push_back(std::move(step));
        }
    });
    return best(variables, found);
}

// engine/surface.py `step`.
std::optional<SurfaceStep> surface_step(const Variables& variables, const Ideal& ambient, const Ideal& strict,
                                        const std::vector<std::pair<std::string, Ideal>>& divisors,
                                        std::vector<Ideal> working, const std::vector<Ideal>& away) {
    if (auto direct = line_centre(variables, ambient, strict, divisors, working, away)) return direct;
    std::vector<Ideal> live;
    for (const auto& w : working) if (!empty_locus(variables, joined(joined(ambient, strict), w))) live.push_back(w);
    working = live;
    if (working.empty()) return std::nullopt;
    std::vector<Ideal> holes;
    for (const auto& z : away)
        if (std::any_of(working.begin(), working.end(), [&](const Ideal& w) {
                return !empty_locus(variables, joined(joined(joined(ambient, strict), z), w));
            }))
            holes.push_back(z);
    if (!holes.empty()) {
        const Ideal covers = cover_products(variables, holes);
        const Coord t = dummy("off");
        Variables extended{t};
        extended.insert(extended.end(), variables.begin(), variables.end());
        std::vector<SurfaceStep> found;
        std::vector<Ideal> left = working;
        Ideal seen;
        for (const auto& unit : covers) {
            if (contains(seen, unit)) continue;
            seen.push_back(unit);
            auto inner = surface_step(extended, joined(ambient, {symbol(t) * unit - constant(1)}), strict, divisors,
                                      left, {});
            if (inner) {
                Ideal closed;
                const Held eliminated = groebner(inner->centre, extended, "first");
                for (const auto& g : eliminated->exprs)
                    if (!g.has(t)) closed.push_back(g);
                inner->centre = ideal_basis(variables, closed);
                found.push_back(std::move(*inner));
            }
            std::vector<Ideal> next;
            for (const auto& w : left) {
                const Ideal extended_piece = joined(w, {unit});
                if (!empty_locus(variables, joined(joined(ambient, strict), extended_piece))) next.push_back(extended_piece);
            }
            left = std::move(next);
            if (left.empty()) break;
        }
        return best(variables, found);
    }
    std::vector<std::pair<std::string, Expr>> functions;
    for (const auto& [label, ideal] : divisors)
        if (std::any_of(working.begin(), working.end(), [&](const Ideal& w) {
                return !empty_locus(variables, joined(joined(joined(ambient, strict), ideal), w));
            }))
            functions.emplace_back(label, principal(variables, ambient, ideal));
    return surface_steps(variables, ambient, strict, functions, working);
}

V surface(const V& request) {
    Variables variables;
    for (Json::ArrayIndex i = 0; i < request["names"].size(); ++i)
        variables.push_back({request["names"][i].asString(), request["identities"][i].asInt64()});
    auto ideal_of = [&](const V& lines) {
        Ideal out;
        for (const auto& line : lines) out.push_back(decode(variables, line.asString()));
        return out;
    };
    std::vector<std::pair<std::string, Ideal>> divisors;
    for (const auto& item : request["divisors"]) divisors.emplace_back(item[0].asString(), ideal_of(item[1]));
    std::vector<Ideal> working, away;
    for (const auto& w : request["working"]) working.push_back(ideal_of(w));
    for (const auto& z : request["away"]) away.push_back(ideal_of(z));
    const auto found = surface_step(variables, ideal_of(request["ambient"]), ideal_of(request["strict"]), divisors,
                                    working, away);
    if (!found) return V();
    V out;
    out["key"] = surface_row(*found);
    out["kind"] = found->type;
    out["centre"] = json_rows(variables, found->centre);
    return out;
}

V execute(const V& request) {
    Settings local;
    local.packed_present = request["packed_present"].asBool();
    if (!request["seconds"].isNull())
        local.deadline = std::chrono::steady_clock::now() +
                         std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                             std::chrono::duration<double>(request["seconds"].asDouble()));
    Settings* outer = settings;
    settings = &local;
    struct Restore {
        Settings* value;
        ~Restore() { settings = value; }
    } restore{outer};
    auto failure = [](const std::string& kind, const std::string& message) {
        V out;
        out["raise"] = kind;
        out["message"] = message;
        return out;
    };
    try {
        if (request["operation"].asString() == "maximum") {
            V out;
            out["maximum"] = maximum(request);
            return out;
        }
        if (request["operation"].asString() == "surface_step") {
            V out;
            out["step"] = surface(request);
            return out;
        }
    } catch (const NoModel& error) {
        return failure("NoModel", error.message);
    } catch (const NoContact& error) {
        return failure("NoContact", error.message);
    } catch (const Raise& error) {
        return failure(error.kind, error.message);
    } catch (const Unavailable& error) {
        return failure("Unavailable", error.message);
    } catch (const Declined& error) {
        V out;
        out["declined"] = error.reason;
        return out;
    }
    throw std::invalid_argument("Unknown centre maximum operation.");
}
}  // namespace

extern "C" const char* atlas_centre_maximum(const char* input) noexcept {
    return atlas::sparse_json::invoke(input, execute);
}
