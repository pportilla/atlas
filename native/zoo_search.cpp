// One in-memory catalogue, shared by the native session and the browser Wasm build.
// The wire format is one line per entry: id, then tab-separated sparse terms.
// Within each term the signed rational and ordered exponents are separated by spaces.
#include <algorithm>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using std::string;
using std::string_view;
using std::vector;

bool digit(char c) { return c >= '0' && c <= '9'; }
bool letter(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool space(char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }

// Exponents require only addition; rational equality requires only cross products.
// Keeping decimal magnitudes avoids a platform-sized exponent or coefficient limit.
string natural(string_view digits) {
    if (digits.empty() || !std::all_of(digits.begin(), digits.end(), digit))
        throw std::invalid_argument("Expected a nonnegative integer.");
    size_t i = digits.find_first_not_of('0');
    return i == string_view::npos ? "0" : string(digits.substr(i));
}

string add(const string& a, const string& b) {
    string result;
    size_t i = a.size(), j = b.size();
    int carry = 0;
    while (i || j || carry) {
        int sum = carry;
        if (i) sum += a[--i] - '0';
        if (j) sum += b[--j] - '0';
        result.push_back(static_cast<char>('0' + sum % 10));
        carry = sum / 10;
    }
    std::reverse(result.begin(), result.end());
    return result;
}

string multiply(const string& a, const string& b) {
    if (a == "1") return b;
    if (b == "1") return a;
    if (a == "0" || b == "0") return "0";
    string result(a.size() + b.size(), '0');
    for (size_t i = a.size(); i-- > 0;) {
        int carry = 0;
        for (size_t j = b.size(); j-- > 0;) {
            size_t k = i + j + 1;
            int sum = result[k] - '0' + (a[i] - '0') * (b[j] - '0') + carry;
            result[k] = static_cast<char>('0' + sum % 10);
            carry = sum / 10;
        }
        result[i] = static_cast<char>('0' + carry);
    }
    return natural(result);
}

struct Rational {
    bool negative = false;
    string numerator = "1", denominator = "1";

    bool same_magnitude(const Rational& b) const {
        if (denominator == b.denominator) return numerator == b.numerator;
        if (numerator == b.numerator) return false; // Both numerators are nonzero.
        return multiply(numerator, b.denominator) == multiply(b.numerator, denominator);
    }
};

struct Term {
    Rational coefficient;
    vector<string> powers, shape;

    void set_shape() {
        for (const string& p : powers) if (p != "0") shape.push_back(p);
        std::sort(shape.begin(), shape.end());
    }
};

using Terms = vector<Term>;

class Reader {
    string_view source;
    size_t pos = 0;

public:
    explicit Reader(string_view text) : source(text) {}
    char peek() const { return pos < source.size() ? source[pos] : '\0'; }
    void skip_space() { while (space(peek())) ++pos; }
    bool consume(char c) {
        if (peek() != c) return false;
        ++pos;
        return true;
    }

    string number() {
        size_t start = pos;
        while (digit(peek())) ++pos;
        return natural(source.substr(start, pos - start));
    }

    string name() {
        size_t start = pos++;
        while (letter(peek()) || digit(peek()) || peek() == '_') ++pos;
        return string(source.substr(start, pos - start));
    }

    // A comma starts a new signed term; '+' and '-' can also separate terms.
    std::optional<Terms> query(size_t max_width) {
        struct NamedTerm { Rational coefficient; std::map<string, string> powers; };
        vector<NamedTerm> named;
        std::map<string, size_t> variables;
        skip_space();
        while (peek()) {
            NamedTerm term;
            term.coefficient.negative = consume('-');
            if (!term.coefficient.negative) consume('+');
            skip_space();
            for (;;) {
                if (digit(peek())) {
                    string scalar = number();
                    if (scalar == "0") throw std::invalid_argument("Each search term needs a nonzero coefficient.");
                    term.coefficient.numerator = multiply(term.coefficient.numerator, scalar);
                } else if (letter(peek())) {
                    string variable = name();
                    skip_space();
                    string power = "1";
                    bool has_power = consume('^');
                    if (!has_power && source.substr(pos, 2) == "**") {
                        pos += 2;
                        has_power = true;
                    }
                    if (has_power) {
                        skip_space();
                        power = number();
                    }
                    if (power != "0") {
                        auto [it, inserted] = term.powers.emplace(variable, power);
                        if (!inserted) it->second = add(it->second, power);
                        variables.emplace(variable, variables.size());
                    }
                } else {
                    throw std::invalid_argument("Enter a monomial, for example 1/2*x*y^3.");
                }
                skip_space();
                while (consume('/')) {
                    skip_space();
                    string divisor = number();
                    if (divisor == "0") throw std::invalid_argument("A coefficient denominator must be nonzero.");
                    term.coefficient.denominator = multiply(term.coefficient.denominator, divisor);
                    skip_space();
                }
                if (consume('*')) {
                    skip_space();
                    continue;
                }
                if (letter(peek()) || digit(peek())) continue;
                break;
            }
            named.push_back(std::move(term));
            if (!peek()) break;
            if (consume(',')) {
                skip_space();
                if (!peek()) throw std::invalid_argument("Enter another monomial after the comma.");
            } else if (peek() != '+' && peek() != '-') {
                throw std::invalid_argument("Separate monomials with commas and use ^ for a nonnegative exponent.");
            }
        }

        if (variables.size() > max_width) return std::nullopt;
        Terms result;
        for (auto& named_term : named) {
            Term term{std::move(named_term.coefficient), vector<string>(variables.size(), "0"), {}};
            for (auto& [v, e] : named_term.powers) term.powers[variables.at(v)] = std::move(e);
            term.set_shape();
            result.push_back(std::move(term));
        }
        return result;
    }

    Term indexed_term() {
        Term term;
        term.coefficient.negative = consume('-');
        term.coefficient.numerator = number();
        if (consume('/')) term.coefficient.denominator = number();
        if (term.coefficient.numerator == "0" || term.coefficient.denominator == "0")
            throw std::invalid_argument("Invalid indexed coefficient.");
        while (peek()) {
            if (!consume(' ')) throw std::invalid_argument("Invalid indexed power.");
            term.powers.push_back(number());
        }
        term.set_shape();
        return term;
    }
};

struct Entry {
    string id;
    Terms terms;
};

using Options = vector<vector<const Term*>>;

// All term matches share the same injection from query to catalogue coordinates.
// After assigning an axis, every term must retain at least one possible witness.
// Equal shapes ensure that the remaining target coordinates have exponent zero.
bool assign_axes(const Terms& pattern, const Options& options, const vector<size_t>& order,
                 vector<size_t>& image, vector<bool>& occupied, size_t depth) {
    if (depth == order.size()) return true;
    size_t axis = order[depth];
    for (size_t target = 0; target < occupied.size(); ++target) {
        if (occupied[target]) continue;
        image[axis] = target;
        bool viable = true;
        for (size_t i = 0; i < pattern.size() && viable; ++i) {
            viable = std::any_of(options[i].begin(), options[i].end(), [&](const Term* candidate) {
                for (size_t j = 0; j <= depth; ++j) {
                    size_t v = order[j];
                    if (pattern[i].powers[v] != candidate->powers[image[v]]) return false;
                }
                return true;
            });
        }
        if (!viable) continue;
        occupied[target] = true;
        if (assign_axes(pattern, options, order, image, occupied, depth + 1)) return true;
        occupied[target] = false;
    }
    return false;
}

bool matches(const Terms& pattern, const vector<size_t>& order, const Entry& entry, bool flip) {
    size_t width = entry.terms.front().powers.size();
    if (order.size() > width) return false;
    Options options;
    for (const auto& term : pattern) {
        auto& choices = options.emplace_back();
        for (const auto& candidate : entry.terms) {
            if ((term.coefficient.negative != candidate.coefficient.negative) == flip &&
                term.shape == candidate.shape && term.coefficient.same_magnitude(candidate.coefficient))
                choices.push_back(&candidate);
        }
        if (choices.empty()) return false;
    }
    if (pattern.size() == 1) return true;
    vector<bool> occupied(width, false);
    vector<size_t> image(order.size());
    return assign_axes(pattern, options, order, image, occupied, 0);
}

class Catalogue {
    vector<Entry> entries;
    size_t max_width = 0;
    // One posting per record and exponent multiset, in display order.
    std::map<vector<string>, vector<size_t>> postings;
public:
    explicit Catalogue(string_view wire) {
        while (!wire.empty()) {
            size_t newline = wire.find('\n');
            string_view row = wire.substr(0, newline);
            size_t tab = row.find('\t');
            if (tab == 0 || tab == string_view::npos) throw std::invalid_argument("Invalid Resolution Zoo index row.");
            Entry entry{string(row.substr(0, tab)), {}};
            std::set<vector<string>> shapes;
            while (tab != string_view::npos) {
                row.remove_prefix(tab + 1);
                tab = row.find('\t');
                Term term = Reader(row.substr(0, tab)).indexed_term();
                if (!entry.terms.empty() && term.powers.size() != entry.terms.front().powers.size())
                    throw std::invalid_argument("Inconsistent exponent widths in the Resolution Zoo index.");
                max_width = std::max(max_width, term.powers.size());
                shapes.insert(term.shape);
                entry.terms.push_back(std::move(term));
            }
            for (const auto& shape : shapes) postings[shape].push_back(entries.size());
            entries.push_back(std::move(entry));
            if (newline == string_view::npos) break;
            wire.remove_prefix(newline + 1);
        }
    }

    string search(string input) const {
        size_t pos = 0;
        while ((pos = input.find("−", pos)) != string::npos) input.replace(pos, 3, "-");
        auto query = Reader(input).query(max_width);
        if (!query) return {};
        const Terms& pattern = *query;
        string result;
        auto include = [&](const Entry& e) {
            if (!result.empty()) result += '\n';
            result += e.id;
        };
        if (pattern.empty()) {
            for (const auto& e : entries) include(e);
            return result;
        }
        // Start with the least frequent shape in the catalogue.
        const vector<size_t>* candidates = nullptr;
        for (const Term& t : pattern) {
            auto it = postings.find(t.shape);
            if (it == postings.end()) return result;
            if (!candidates || it->second.size() < candidates->size()) candidates = &it->second;
        }
        vector<size_t> order(pattern.front().powers.size()), counts(order.size(), 0);
        std::iota(order.begin(), order.end(), 0);
        for (const Term& t : pattern)
            for (size_t axis : order) if (t.powers[axis] != "0") ++counts[axis];
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return counts[a] > counts[b]; });
        for (size_t i : *candidates)
            if (matches(pattern, order, entries[i], false) || matches(pattern, order, entries[i], true))
                include(entries[i]);
        return result;
    }
};

thread_local string response;
}

// The immutable handle can serve concurrent queries. Returned text is borrowed
// until the next call on the same thread. A null result leaves atlas_zoo_error().
extern "C" {
void* atlas_zoo_create(const char* wire) noexcept {
    try {
        if (!wire) throw std::invalid_argument("Missing Resolution Zoo catalogue.");
        return new Catalogue(wire);
    } catch (const std::exception& error) {
        response = error.what();
        return nullptr;
    }
}
void atlas_zoo_destroy(void* handle) noexcept { delete static_cast<Catalogue*>(handle); }
const char* atlas_zoo_search(const void* handle, const char* query) noexcept {
    try {
        if (!handle || !query) throw std::invalid_argument("Missing Resolution Zoo catalogue or query.");
        response = static_cast<const Catalogue*>(handle)->search(query);
        return response.c_str();
    } catch (const std::exception& error) {
        response = error.what();
        return nullptr;
    }
}
const char* atlas_zoo_error() noexcept { return response.c_str(); }
}
