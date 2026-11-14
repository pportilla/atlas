// Unbounded sparse Buchberger and ordered division for product monomial orders.
// Each exponent is a FLINT integer; only the number of actual terms costs space.
#include "groebner.hpp"
#include <flint/fmpq.h>
#include <map>
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

// Always choose the first divisor whose head divides the current term.
// This gives both the canonical remainder and the public ordered quotients.
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
        auto shift = difference(head->first, reducers[k].begin()->first);
        Coefficient factor;
        fmpq_div(factor.q, head->second.q, reducers[k].begin()->second.q);
        if (quotients) add((*quotients)[k], shift, factor);
        fmpq_neg(factor.q, factor.q);
        add_shifted(p, reducers[k], shift, factor);
    }
    return remainder;
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
    if (op == "basis") {
        for (auto& p : basis(std::move(rows))) print_row(result, p);
    } else if (op == "divide" && !rows.empty()) {
        Terms f = std::move(rows.front());
        rows.erase(rows.begin());
        for (auto& g : rows) if (g.empty()) throw std::invalid_argument("Cannot divide by zero.");
        Rows quotients(rows.size(), Terms(order));
        auto remainder = divide(std::move(f), rows, &quotients);
        for (auto& q : quotients) print_row(result, q);
        print_row(result, remainder);
    } else throw std::invalid_argument("Invalid sparse polynomial operation.");
    return result;
}
}  // namespace
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
