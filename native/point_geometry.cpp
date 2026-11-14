// Sparse rational point evaluation, Taylor forms and exact Jacobian row spaces.
#include <gmpxx.h>
#include <json/json.h>
#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
using Z = mpz_class;
using Q = mpq_class;
using Powers = std::vector<Z>;
using Row = std::vector<Q>;
using Terms = std::map<Powers, Q>;
using Matrix = std::vector<Row>;
using Pivots = std::map<size_t, Row>;

Powers read_exponents(const Json::Value& array, size_t width) {
    if (!array.isArray() || array.size() != width) throw std::invalid_argument("Invalid exponent vector.");
    Powers e;
    for (const auto& item : array) {
        e.emplace_back(item.asString(), 10);
        if (e.back() < 0) throw std::invalid_argument("Negative polynomial exponent.");
    }
    return e;
}
Row read_row(const Json::Value& array) {
    Row row;
    for (const auto& item : array) {
        row.emplace_back(item.asString(), 10);
        if (row.back().get_den() == 0) throw std::invalid_argument("Zero rational denominator.");
        row.back().canonicalize();
    }
    return row;
}
Terms read_terms(const Json::Value& data, size_t width) {
    Terms terms;
    for (const auto& entry : data) {
        Q c(entry[0].asString(), 10);
        if (c.get_den() == 0) throw std::invalid_argument("Zero rational denominator.");
        c.canonicalize();
        if (c != 0) terms[read_exponents(entry[1], width)] += c;
    }
    return terms;
}
template<class Vector>
Json::Value strings(const Vector& values) {
    Json::Value out(Json::arrayValue);
    for (const auto& n : values) out.append(n.get_str());
    return out;
}
Json::Value term_list(const Terms& terms) {
    Json::Value out(Json::arrayValue);
    for (const auto& [e, c] : terms) {
        if (c == 0) continue;
        Json::Value entry(Json::arrayValue);
        entry.append(c.get_str());
        entry.append(strings(e));
        out.append(entry);
    }
    return out;
}
void add(Terms& terms, const Powers& key, const Q& value) {
    if (value == 0) return;
    auto it = terms.try_emplace(key, 0).first;
    it->second += value;
    if (it->second == 0) terms.erase(it);
}
Powers minima(const Terms& terms) {
    Powers out = terms.begin()->first;
    for (const auto& [e, c] : terms)
        for (size_t i = 0; i < e.size(); ++i) out[i] = std::min(out[i], e[i]);
    return out;
}

// A gcd-free basis represents the multiplicative relations of the positive
// coordinates without factoring any input integer into primes.
struct Point {
    Row coordinates;
    Powers bases;
    std::vector<Powers> columns;
    std::vector<size_t> moving, fixed;

    explicit Point(Row point) : coordinates(std::move(point)) {
        std::set<Z> factors;
        for (size_t i = 0; i < coordinates.size(); ++i) {
            const auto& q = coordinates[i];
            if (q == 0) { fixed.push_back(i); continue; }
            moving.push_back(i);
            if (abs(q.get_num()) > 1) factors.insert(abs(q.get_num()));
            if (q.get_den() > 1) factors.insert(q.get_den());
        }
        while (true) {
            bool split = false;
            for (auto i = factors.begin(); i != factors.end() && !split;) {
                const Z a = *i++;
                for (auto j = i; j != factors.end(); ++j) {
                    const Z b = *j, d = gcd(a, b);
                    if (d == 1) continue;
                    factors.erase(a);
                    factors.erase(b);
                    factors.insert(d);
                    for (const auto& n : {a, b}) {
                        Z rest;
                        mpz_remove(rest.get_mpz_t(), n.get_mpz_t(), d.get_mpz_t());
                        if (rest > 1) factors.insert(rest);
                    }
                    split = true;
                    break;
                }
                if (split) break;
            }
            if (!split) break;
        }
        for (const auto& factor : factors) {
            bases.push_back(factor);
            Powers col(coordinates.size());
            for (auto i : moving) {
                const auto& q = coordinates[i];
                Z rest;
                col[i] = mpz_remove(rest.get_mpz_t(), mpq_numref(q.get_mpq_t()), factor.get_mpz_t());
                col[i] -= mpz_remove(rest.get_mpz_t(), mpq_denref(q.get_mpq_t()), factor.get_mpz_t());
            }
            columns.push_back(std::move(col));
        }
    }
    Powers key(const Powers& e) const {
        Powers result(bases.size());
        for (size_t j = 0; j < bases.size(); ++j)
            for (auto i : moving) result[j] += columns[j][i] * e[i];
        return result;
    }
    Q signed_value(const Powers& e, const Q& c) const {
        bool neg = false;
        for (auto i : moving) if (coordinates[i] < 0 && mpz_odd_p(e[i].get_mpz_t())) neg = !neg;
        return neg ? -c : c;
    }
};

Z residual_scale(const Powers& bases, const Powers& e, const Powers& lower) {
    Z scale = 1;
    for (size_t j = 0; j < e.size(); ++j) {
        const Z exponent = e[j] - lower[j];
        if (!exponent.fits_ulong_p()) throw std::overflow_error("Exact rational coefficient exceeds addressable memory.");
        Z power;
        mpz_pow_ui(power.get_mpz_t(), bases[j].get_mpz_t(), exponent.get_ui());
        scale *= power;
    }
    return scale;
}
bool weights_nonzero(const Powers& bases, const Terms& weights) {
    if (weights.empty()) return false;
    const bool positive = weights.begin()->second > 0;
    if (std::all_of(weights.begin(), weights.end(), [&](const auto& term) { return (term.second > 0) == positive; }))
        return true;
    const Powers lower = minima(weights);
    // Only a nonzero residue decides the predicate. In particular a prime
    // dividing a denominator cannot be used to prove a cancellation.
    for (unsigned long p : {1000000007UL, 1000000009UL, 998244353UL}) {
        Z modulus = p, residue = 0;
        bool good = true;
        for (const auto& [e, c] : weights) {
            Z term, inverse;
            if (!mpz_invert(inverse.get_mpz_t(), mpq_denref(c.get_mpq_t()), modulus.get_mpz_t())) {
                good = false;
                break;
            }
            term = (inverse * c.get_num()) % modulus;
            for (size_t i = 0; i < e.size(); ++i) {
                if (e[i] == lower[i]) continue;
                Z power, exponent = e[i] - lower[i];
                mpz_powm(power.get_mpz_t(), bases[i].get_mpz_t(), exponent.get_mpz_t(), modulus.get_mpz_t());
                term = (term * power) % modulus;
            }
            residue += term;
        }
        if (good && residue % modulus != 0) return true;
    }
    Q exact = 0;
    for (const auto& [e, c] : weights) exact += c * residual_scale(bases, e, lower);
    return exact != 0;
}
bool at_point(const Point& p, const Terms& f) {
    Terms weights;
    for (const auto& [e, c] : f) {
        if (std::any_of(p.fixed.begin(), p.fixed.end(), [&](size_t i) { return e[i] != 0; })) continue;
        add(weights, p.key(e), p.signed_value(e, c));
    }
    return weights_nonzero(p.bases, weights);
}

// Scale all coefficient rows by the same nonzero factor. Proportional
// power-vectors require only a vanishing decision for their shared scalar.
Row materialize(const Powers& bases, const std::vector<Terms>& weights) {
    Row direction(weights.size());
    const auto pivot = std::find_if(weights.begin(), weights.end(), [](const auto& w) { return !w.empty(); });
    if (pivot == weights.end()) return direction;
    const auto& key = pivot->begin()->first;
    const Q scale = pivot->begin()->second;
    bool proportional = true;
    for (size_t j = 0; j < weights.size(); ++j) {
        auto match = weights[j].find(key);
        if (match != weights[j].end()) direction[j] = match->second;
        if (!direction[j]) { proportional &= weights[j].empty(); continue; }
        if (pivot->size() != weights[j].size()) { proportional = false; continue; }
        for (const auto& [k, c] : weights[j]) {
            auto w = pivot->find(k);
            if (w == pivot->end() || c * scale != direction[j] * w->second) { proportional = false; break; }
        }
    }
    if (proportional) return weights_nonzero(bases, *pivot) ? direction : Row(weights.size());
    Powers common = minima(*pivot);
    for (const auto& w : weights) {
        if (w.empty()) continue;
        const auto lower = minima(w);
        for (size_t j = 0; j < lower.size(); ++j) common[j] = std::min(common[j], lower[j]);
    }
    std::map<Powers, Z> scales;
    Row row(weights.size());
    for (size_t j = 0; j < row.size(); ++j)
        for (const auto& [e, c] : weights[j]) {
            auto [it, inserted] = scales.try_emplace(e);
            if (inserted) it->second = residual_scale(bases, e, common);
            row[j] += c * it->second;
        }
    return row;
}
std::vector<Terms> differential_weights(const Point& p, const Terms& f) {
    std::vector<Terms> weights(p.coordinates.size());
    for (const auto& [e, coefficient] : f) {
        Z degree = 0;
        for (auto i : p.fixed) degree += e[i];
        if (degree > 1) continue;
        auto key = p.key(e);
        Q c = p.signed_value(e, coefficient);
        if (degree == 1) {
            for (auto i : p.fixed) if (e[i] != 0) add(weights[i], key, c);
        } else {
            for (auto i : p.moving) if (e[i] != 0) add(weights[i], key, c * e[i] / p.coordinates[i]);
        }
    }
    return weights;
}

size_t torus_order(const Point& p, const Terms& f) {
    std::vector<Powers> exponents, keys;
    Row coefficients;
    for (const auto& [e, c] : f) {
        exponents.push_back(e);
        coefficients.push_back(p.signed_value(e, c));
        keys.push_back(p.key(e));
    }
    std::vector<size_t> axes;
    for (auto i : p.moving)
        if (std::any_of(exponents.begin(), exponents.end(), [&](const auto& e){ return e[i] != exponents[0][i]; }))
            axes.push_back(i);
    const auto count = f.size();
    std::map<size_t, Powers> pivots;
    std::vector<Powers> frontier(1, Powers(count, 1));
    // Euler derivatives multiply a monomial by its exponent. New independent
    // moment rows suffice, since their multiples span the next degree.
    for (size_t order = 0; !frontier.empty(); ++order) {
        std::vector<Powers> next;
        for (auto& row : frontier) {
            for (size_t i = 0; i < count; ++i) {
                if (!row[i]) continue;
                auto it = pivots.find(i);
                if (it != pivots.end()) {
                    const auto& a = it->second;
                    Z d = gcd(a[i], row[i]), left = a[i] / d, right = row[i] / d;
                    for (size_t j = i; j < count; ++j) row[j] = left * row[j] - right * a[j];
                    continue;
                }
                Z content = 0;
                for (const auto& v : row) content = gcd(content, v);
                if (content > 1) for (auto& v : row) v /= content;
                pivots.emplace(i, row);
                Terms weights;
                for (size_t j = 0; j < count; ++j) add(weights, keys[j], coefficients[j] * row[j]);
                if (weights_nonzero(p.bases, weights)) return order;
                for (size_t axis : axes) {
                    Powers child(count);
                    for (size_t j = 0; j < count; ++j) child[j] = row[j] * exponents[j][axis];
                    next.push_back(std::move(child));
                }
                break;
            }
        }
        frontier.swap(next);
    }
    throw std::logic_error("The nonzero polynomial has no nonvanishing derivative.");
}
Z order_at_point(const Point& p, const Terms& f) {
    std::map<std::pair<Z, Powers>, Terms> groups;
    for (const auto& [e, c] : f) {
        Z lower = 0;
        Powers fixed;
        for (auto i : p.fixed) { lower += e[i]; fixed.push_back(e[i]); }
        groups[{lower, std::move(fixed)}][e] = c;
    }
    std::optional<Z> best;
    for (const auto& [group, terms] : groups) {
        if (best && group.first >= *best) break;
        Z degree = group.first + torus_order(p, terms);
        if (!best || degree < *best) best = degree;
    }
    if (!best) throw std::invalid_argument("The zero polynomial has no finite order.");
    return *best;
}
using Coefficients = std::map<Powers, Terms>;
void taylor_slice(const Point& p, const Powers& e, const Powers& key, size_t j, unsigned long left,
                  Powers& powers, const Q& coefficient, Coefficients& result) {
    if (!left) { add(result[powers], key, coefficient); return; }
    if (j == p.moving.size()) return;
    const auto axis = p.moving[j];
    Q weight = coefficient;
    for (unsigned long k = 0; k <= left && k <= e[axis]; ++k) {
        powers[axis] = k;
        taylor_slice(p, e, key, j + 1, left - k, powers, weight, result);
        weight *= (e[axis] - k);
        weight /= (k + 1) * p.coordinates[axis];
    }
    powers[axis] = 0;
}
Terms initial_form(const Point& p, const Terms& f) {
    const Z degree = order_at_point(p, f);
    Coefficients coefficients;
    for (const auto& [e, c] : f) {
        Z extra = degree;
        Powers powers(e.size());
        for (auto i : p.fixed) { powers[i] = e[i]; extra -= e[i]; }
        if (extra < 0) continue;
        // Within a fixed-axis slice, distinct torus exponents are separated
        // by moments of degree at most support size minus one.
        if (extra >= f.size()) throw std::logic_error("Sparse point order exceeds its interpolation bound.");
        taylor_slice(p, e, p.key(e), 0, extra.get_ui(), powers, p.signed_value(e, c), coefficients);
    }
    std::vector<Terms> weights;
    for (auto& entry : coefficients) weights.push_back(std::move(entry.second));
    const auto values = materialize(p.bases, weights);
    Terms result;
    size_t i = 0;
    for (const auto& entry : coefficients) add(result, entry.first, values[i++]);
    if (result.empty()) throw std::logic_error("Nonzero initial form expected.");
    const Q lead = result.rbegin()->second;
    for (auto& entry : result) entry.second /= lead;
    return result;
}

void reduce(Row& row, const Pivots& pivots) {
    for (const auto& [col, pivot] : pivots) {
        const Q scale = row[col];
        if (scale != 0) for (size_t j = col; j < row.size(); ++j) row[j] -= scale * pivot[j];
    }
}
void insert(Pivots& pivots, Row row) {
    reduce(row, pivots);
    for (size_t i = 0; i < row.size(); ++i) if (row[i] != 0) {
        const Q scale = row[i];
        for (size_t j = i; j < row.size(); ++j) row[j] /= scale;
        pivots.emplace(i, std::move(row));
        return;
    }
}
Json::Value row_space(const Matrix& rows, size_t width, const Json::Value& witness, bool rank_only) {
    Pivots pivots;
    if (!witness.isNull()) {
        for (const auto& i : witness) insert(pivots, rows.at(i.asUInt()));
        if (pivots.size() != witness.size()) return {};
        if (pivots.size() < width) for (auto row : rows) {
            reduce(row, pivots);
            if (std::any_of(row.begin(), row.end(), [](const Q& c){ return c != 0; })) return {};
        }
    } else {
        for (const auto& row : rows) {
            insert(pivots, row);
            if (pivots.size() == width) break;
        }
    }
    Json::Value out;
    out["rank"] = Json::UInt64(pivots.size());
    out["dimension"] = Json::UInt64(width - pivots.size());
    out["rows"] = out["basis"] = Json::arrayValue;
    if (rank_only) return out;
    for (auto pivot = pivots.rbegin(); pivot != pivots.rend(); ++pivot) {
        for (auto prior = std::next(pivot); prior != pivots.rend(); ++prior) {
            const Q c = prior->second[pivot->first];
            if (c != 0) for (size_t j = pivot->first; j < width; ++j) prior->second[j] -= c * pivot->second[j];
        }
    }
    for (const auto& [col, row] : pivots) out["rows"].append(strings(row));
    for (size_t col = 0; col < width; ++col) {
        if (pivots.count(col)) continue;
        Row direction(width);
        direction[col] = 1;
        for (const auto& [j, row] : pivots) direction[j] = -row[col];
        out["basis"].append(strings(direction));
    }
    return out;
}
Powers primitive(const Row& row) {
    Z den = 1, content = 0;
    for (const auto& q : row) den = lcm(den, q.get_den());
    Powers integers;
    for (const auto& q : row) {
        integers.emplace_back(q.get_num() * (den / q.get_den()));
        content = gcd(content, integers.back());
    }
    if (content > 1) for (auto& v : integers) v /= content;
    return integers;
}

bool full_cotangent_rank(const Point& p, const std::vector<Terms>& polynomials) {
    const size_t width = p.coordinates.size();
    if (width == 0 || polynomials.size() < width) return false;
    if (p.bases.empty()) {
        Pivots pivots;
        for (const auto& f : polynomials) insert(pivots, materialize(p.bases, differential_weights(p, f)));
        return pivots.size() == width;
    }
    // A modular nonsingular minor is a certificate. Skip primes at which a
    // nonzero point coordinate is undefined or reduces to zero.
    for (unsigned long prime : {1000000007UL, 1000000009UL, 998244353UL, 1004535809UL}) {
        const Z mod = prime;
        bool admissible = true;
        Powers point(width);
        for (auto i : p.moving) {
            const auto& q = p.coordinates[i];
            Z inv;
            if (q.get_num() % mod == 0 || !mpz_invert(inv.get_mpz_t(), mpq_denref(q.get_mpq_t()), mod.get_mpz_t())) {
                admissible = false;
                break;
            }
            point[i] = (q.get_num() * inv) % mod;
        }
        if (!admissible) continue;
        std::map<size_t, Powers> pivots;
        for (const auto& f : polynomials) {
            Row coefficients;
            for (const auto& [e, c] : f) coefficients.push_back(c);
            const auto integers = primitive(coefficients);
            Powers row(width);
            size_t term = 0;
            for (const auto& [e, c] : f) {
                Z weight = integers[term++], fixed_degree = 0;
                for (auto i : p.fixed) fixed_degree += e[i];
                if (fixed_degree > 1) continue;
                for (auto i : p.moving) {
                    Z power;
                    mpz_powm(power.get_mpz_t(), point[i].get_mpz_t(), e[i].get_mpz_t(), mod.get_mpz_t());
                    weight = (weight * power) % mod;
                }
                if (fixed_degree == 1) {
                    for (auto i : p.fixed) if (e[i] != 0) row[i] += weight;
                } else {
                    for (auto i : p.moving) row[i] += e[i] * weight;
                }
            }
            for (size_t i = 0; i < width; ++i) {
                Z value = row[i] % mod;
                if (!value) continue;
                auto it = pivots.find(i);
                if (it != pivots.end()) {
                    for (size_t j = i; j < width; ++j) row[j] = (row[j] - value * it->second[j]) % mod;
                } else {
                    Z inv;
                    mpz_invert(inv.get_mpz_t(), value.get_mpz_t(), mod.get_mpz_t());
                    for (size_t j = i; j < width; ++j) row[j] = (row[j] * inv) % mod;
                    pivots.emplace(i, std::move(row));
                    break;
                }
            }
            if (pivots.size() == width) return true;
        }
    }
    return false;
}

struct LocalGenerator { Powers powers; Terms residual; };
Terms one(size_t width) { return {{Powers(width), Q(1)}}; }
LocalGenerator point_generator(const Point& p, const Terms& f) {
    const auto common = minima(f);
    Powers powers = common;
    for (auto i : p.moving) powers[i] = 0;
    Terms residual;
    for (const auto& [exp, c] : f) {
        Powers e = exp;
        for (size_t i = 0; i < e.size(); ++i) e[i] -= common[i];
        residual.emplace(std::move(e), c);
    }
    if (at_point(p, residual)) residual = one(powers.size());
    return {powers, std::move(residual)};
}
bool large_translation(const Point& p, const Terms& f, unsigned limit) {
    for (const auto& [e, c] : f) {
        Z count = 1;
        for (auto i : p.moving) {
            count *= 1 + e[i];
            if (count >= limit) return true;
        }
    }
    return false;
}
Json::Value centered_plan(const Point& p, const std::vector<Terms>& ideal) {
    const size_t width = p.coordinates.size();
    const auto unit = one(width);
    std::vector<LocalGenerator> rows;
    std::set<Powers> monomials;
    Json::Value out;
    out["rows"] = Json::arrayValue;
    for (const auto& f : ideal) {
        if (f.empty()) continue;
        auto g = point_generator(p, f);
        if (g.residual == unit) {
            if (g.powers == Powers(width)) { out["unit"] = true; return out; }
            if (!monomials.insert(g.powers).second) continue;
        }
        rows.push_back(std::move(g));
    }
    bool check_rank = !p.moving.empty() && ideal.size() >= width;
    for (auto& row : rows) {
        if (row.residual != unit && !monomials.empty()) {
            const auto original_count = row.residual.size();
            for (auto it = row.residual.begin(); it != row.residual.end();) {
                bool divisible = std::any_of(monomials.begin(), monomials.end(), [&](const auto& m){
                    for (size_t j = 0; j < width; ++j) if (it->first[j] + row.powers[j] < m[j]) return false;
                    return true;
                });
                if (divisible) it = row.residual.erase(it);
                else ++it;
            }
            if (row.residual.empty()) continue;
            if (original_count != row.residual.size()) {
                auto g = point_generator(p, row.residual);
                for (size_t j = 0; j < width; ++j) row.powers[j] += g.powers[j];
                row.residual = std::move(g.residual);
            }
        }
        if (check_rank && row.residual != unit && large_translation(p, row.residual, 9)) {
            if (full_cotangent_rank(p, ideal)) { out["maximal"] = true; return out; }
            check_rank = false;
        }
        Json::Value item;
        item["powers"] = strings(row.powers);
        item["residual"] = term_list(row.residual);
        out["rows"].append(item);
    }
    return out;
}

Json::Value compute(const Json::Value& r) {
    const std::string op = r["operation"].asString();
    if (op == "row_space" || op == "residues") {
        size_t width = r["width"].asUInt64();
        Matrix rows;
        for (const auto& v : r["rows"]) {
            rows.push_back(read_row(v));
            if (rows.back().size() != width) throw std::invalid_argument("Invalid Jacobian row width.");
        }
        if (op == "row_space") return row_space(rows, width, r["witness"], r["rank_only"].asBool());
        Json::Value out(Json::arrayValue);
        unsigned long prime = r["prime"].asUInt();
        if (prime < 2) throw std::invalid_argument("Invalid Jacobian residue modulus.");
        for (const auto& row : rows) for (const auto& n : primitive(row))
            out.append(static_cast<Json::UInt>(mpz_fdiv_ui(n.get_mpz_t(), prime)));
        return out;
    }
    Point p(read_row(r["point"]));
    const size_t width = p.coordinates.size();
    if (op == "columns") {
        Json::Value out;
        out["bases"] = strings(p.bases);
        out["columns"] = Json::arrayValue;
        for (const auto& c : p.columns) out["columns"].append(strings(c));
        return out;
    }
    if (op == "jets" || op == "centered" || op == "maximal") {
        std::vector<Terms> ideal;
        for (const auto& item : r["ideal"]) ideal.push_back(read_terms(item, width));
        if (op == "centered") return centered_plan(p, ideal);
        if (op == "maximal") return full_cotangent_rank(p, ideal);
        Json::Value rows(Json::arrayValue);
        for (const auto& f : ideal) {
            if (at_point(p, f)) throw std::invalid_argument("The point must lie on the scheme.");
            auto row = materialize(p.bases, differential_weights(p, f));
            if (std::any_of(row.begin(), row.end(), [](const Q& c){ return c != 0; })) rows.append(strings(row));
        }
        return rows;
    }
    auto f = read_terms(r["terms"], width);
    if (op == "nonzero") return at_point(p, f);
    if (op == "order") return order_at_point(p, f).get_str();
    if (op == "initial") return term_list(initial_form(p, f));
    if (op == "jet") return strings(materialize(p.bases, differential_weights(p, f)));
    if (op == "smooth") {
        if (at_point(p, f)) throw std::invalid_argument("The hypersurface must pass through the supplied point.");
        for (const auto& weights : differential_weights(p, f)) if (weights_nonzero(p.bases, weights)) return true;
        return false;
    }
    if (op == "large") return large_translation(p, f, r["limit"].asUInt());
    if (op == "cone_forms") {
        Json::Value result;
        result["heads"] = result["forms"] = Json::arrayValue;
        for (const auto& item : r["ideal"]) {
            const auto polynomial = read_terms(item, width);
            if (polynomial.empty()) continue;
            std::optional<Z> degree;
            Terms initial;
            for (const auto& [e, c] : polynomial) {
                Z sum = 0;
                for (const auto& k : e) sum += k;
                if (!degree || sum < *degree) { initial.clear(); degree = sum; }
                if (sum == *degree) initial[e] = c;
            }
            auto head = initial.rbegin()->first;
            if (r["order"].asString() == "grevlex") for (const auto& [e, c] : initial)
                if (std::lexicographical_compare(e.rbegin(), e.rend(), head.rbegin(), head.rend())) head = e;
            result["heads"].append(strings(head));
            if (r["forms"].asBool()) result["forms"].append(term_list(initial));
        }
        return result;
    }
    throw std::invalid_argument("Unknown point geometry operation.");
}
}  // namespace

extern "C" const char* atlas_point_geometry(const char* input) noexcept {
    static thread_local std::string text;
    try {
        Json::Value request, response;
        Json::CharReaderBuilder parser;
        std::istringstream stream(input);
        std::string error;
        if (!Json::parseFromStream(parser, stream, &request, &error)) throw std::invalid_argument(error);
        response["result"] = compute(request);
        Json::StreamWriterBuilder writer;
        writer["indentation"] = "";
        text = Json::writeString(writer, response);
    } catch (const std::exception& ex) {
        Json::Value error;
        error["error"] = ex.what();
        text = Json::writeString(Json::StreamWriterBuilder(), error);
    }
    return text.c_str();
}
