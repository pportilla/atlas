// Sparse rational point evaluation, Taylor forms and exact Jacobian row spaces.
#include "rational_sparse_json.hpp"
#include "local_dimension.hpp"
#include "local_dimension_graph.hpp"
#include "local_point_basis.hpp"
#include "local_point_quotient.hpp"
#include "local_quotient_jets.hpp"
#include "primary_decomposition.hpp"
#include "local_io.hpp"
#include <flint/fmpq.h>
#include <flint/fmpq_mat.h>
#include <flint/nmod_mat.h>
#include <algorithm>
#include <climits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
using namespace atlas::sparse_json;
using Pivots = std::map<size_t, Row>;

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

// After clearing denominators and removing a common power, evaluate an
// integer sum without constructing integers with as many bits as the powers.
bool wide_sum_nonzero(const Powers& bases, const Terms& weights, const Powers& lower) {
    Z den = 1;
    for (const auto& [e,c] : weights) den = lcm(den, c.get_den());
    if (bases.size() == 1) {
        // If the accumulated low coefficient is not divisible by the full
        // gap power, the higher terms cannot cancel it. mpz_remove tells us how
        // many powers divide it without allocating the gap power.
        Z carry = 0, previous = lower.front();
        for (const auto& [e,c] : weights) {
            const Z gap = e.front()-previous;
            if (carry != 0 && gap != 0) {
                Z rest;
                if (gap > mpz_remove(rest.get_mpz_t(), carry.get_mpz_t(), bases[0].get_mpz_t())) return true;
                Z power;
                mpz_pow_ui(power.get_mpz_t(), bases[0].get_mpz_t(), gap.get_ui());
                mpz_divexact(carry.get_mpz_t(), carry.get_mpz_t(), power.get_mpz_t());
            }
            carry += c.get_num()*(den/c.get_den());
            previous = e.front();
        }
        return carry != 0;
    }
    // Compute a rigorous upper bound on the absolute integer sum. If every
    // integer through 2b divides it, so does binom(2b,b): each prime in
    // that binomial has valuation at most floor(log_p(2b)) by Legendre's
    // formula. The binomial is at least 2^b. This may take time for giant
    // identities but never asserts zero from a fixed number of trials.
    Z bits = 0;
    std::map<Powers,Z> integers;
    for (const auto& [e,c] : weights) {
        Z a = c.get_num()*(den/c.get_den());
        Z term_bits = mpz_sizeinbase(a.get_mpz_t(), 2);
        Powers exponents(e.size());
        for (size_t i = 0; i < e.size(); ++i) {
            exponents[i] = e[i]-lower[i];
            term_bits += exponents[i]*mpz_sizeinbase(bases[i].get_mpz_t(), 2);
        }
        bits = std::max(bits, term_bits);
        integers.emplace(std::move(exponents), a);
    }
    const Z count = weights.size();
    bits += mpz_sizeinbase(count.get_mpz_t(), 2);
    const Z limit = 2*bits;
    for (Z modulus = 2; modulus <= limit; ++modulus) {
        Z sum = 0;
        for (const auto& [e,c] : integers) {
            Z term = c;
            for (size_t i = 0; i < e.size(); ++i) if (e[i] != 0) {
                Z factor;
                mpz_powm(factor.get_mpz_t(), bases[i].get_mpz_t(), e[i].get_mpz_t(), modulus.get_mpz_t());
                term = (term * factor) % modulus;
            }
            sum += term;
        }
        if (sum % modulus != 0) return true;
    }
    return false;
}

bool wide_scale(const Powers& bases, const Powers& e, const Powers& lower) {
    Z bits = 0;
    for (size_t i = 0; i < e.size(); ++i)
        bits += (e[i]-lower[i])*mpz_sizeinbase(bases[i].get_mpz_t(), 2);
    return bits > 65536;
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
    for (const auto& term : weights)
        if (wide_scale(bases, term.first, lower)) return wide_sum_nonzero(bases, weights, lower);
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

struct WidePointRow : std::overflow_error {
    WidePointRow() : std::overflow_error("Use sparse point computation.") {}
};

// Scale all entries of a row by the same nonzero factor. Proportional
// power-vectors require only a vanishing decision for their shared scalar.
Row materialize(const Powers& bases, const std::vector<Terms>& weights, bool bounded = false) {
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
            if (inserted) {
                if (bounded && wide_scale(bases, e, common)) throw WidePointRow();
                it->second = residual_scale(bases, e, common);
            }
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

using PowerRow = std::vector<Terms>;

// Division-free Gaussian elimination on sums of rational powers. A pivot
// is proved nonzero at the point before we use it; identical products in
// (pivot * row - multiple * pivot_row) cancel in the sparse representation.
void accumulate_product(Terms& out, const Terms& left, const Terms& right, int sign) {
    for (const auto& [a,c] : left) for (const auto& [b,d] : right) {
        auto e = a;
        for (size_t i = 0; i < e.size(); ++i) e[i] += b[i];
        add(out, e, sign*c*d);
    }
}

bool insert_power_row(const Point& p, PowerRow row, std::map<size_t,PowerRow>& pivots) {
    for (size_t i = 0; i < row.size(); ++i) {
        if (!weights_nonzero(p.bases, row[i])) continue;
        const auto prior = pivots.find(i);
        if (prior == pivots.end()) {
            pivots.emplace(i, std::move(row));
            return true;
        }
        const auto& pivot = prior->second;
        for (size_t j = i+1; j < row.size(); ++j) {
            Terms difference;
            accumulate_product(difference, pivot[i], row[j], 1);
            accumulate_product(difference, row[i], pivot[j], -1);
            row[j] = std::move(difference);
        }
    }
    return false;
}

int power_row_rank(const Point& p, const std::vector<Terms>& ideal, const Json::Value& witness = {}) {
    std::map<size_t,PowerRow> pivots;
    const size_t width = p.coordinates.size();
    auto insert_row = [&](const Terms& f) { return insert_power_row(p, differential_weights(p,f), pivots); };
    if (!witness.isNull()) {
        for (const auto& index : witness)
            if (!insert_row(ideal.at(index.asUInt()))) return -1;
    }
    for (const auto& f : ideal) {
        if (pivots.size() == width) break;
        if (insert_row(f) && !witness.isNull()) return -1;
    }
    return static_cast<int>(pivots.size());
}

Json::Value modular_power_rows(const Point& p, const std::vector<Terms>& ideal, unsigned prime) {
    if (prime < 2) throw std::invalid_argument("Invalid Jacobian modulus.");
    // Clear the common coefficient denominator and remove the minimum
    // exponent of each numeric base from each row. These row scalings are
    // nonzero over QQ, even when the chosen prime divides a coordinate.
    Json::Value residues(Json::arrayValue);
    const Z mod = prime;
    for (const auto& f : ideal) {
        const auto row = differential_weights(p, f);
        std::optional<Powers> lower;
        Z den = 1;
        for (const auto& entry : row) for (const auto& [e,c] : entry) {
            den = lcm(den, c.get_den());
            if (!lower) lower = e;
            else for (size_t j = 0; j < e.size(); ++j) (*lower)[j] = std::min((*lower)[j], e[j]);
        }
        for (const auto& entry : row) {
            Z sum = 0;
            for (const auto& [e,c] : entry) {
                Z term = c.get_num()*(den/c.get_den());
                for (size_t i = 0; i < e.size(); ++i) if (e[i] != (*lower)[i]) {
                    Z factor, exponent = e[i]-(*lower)[i];
                    mpz_powm(factor.get_mpz_t(), p.bases[i].get_mpz_t(), exponent.get_mpz_t(), mod.get_mpz_t());
                    term = (term*factor) % mod;
                }
                sum += term;
            }
            residues.append(static_cast<Json::UInt>(mpz_fdiv_ui(sum.get_mpz_t(), prime)));
        }
    }
    return residues;
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
Coefficients initial_weights(const Point& p, const Terms& f) {
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
    return coefficients;
}
Terms initial_form(const Point& p, const Terms& f, bool bounded = false) {
    auto coefficients = initial_weights(p,f);
    std::vector<Terms> weights;
    for (auto& entry : coefficients) weights.push_back(std::move(entry.second));
    const auto values = materialize(p.bases, weights, bounded);
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
bool echelon_rows(Pivots& pivots, const Matrix& rows, size_t width, const Json::Value& witness) {
    if (!witness.isNull()) {
        for (const auto& i : witness) insert(pivots, rows.at(i.asUInt()));
        if (pivots.size() != witness.size()) return false;
        if (pivots.size() < width) for (auto row : rows) {
            reduce(row, pivots);
            if (std::any_of(row.begin(), row.end(), [](const Q& c){ return c != 0; })) return false;
        }
    } else {
        for (const auto& row : rows) {
            insert(pivots, row);
            if (pivots.size() == width) break;
        }
    }
    return true;
}

void back_substitute(Pivots& pivots, size_t width) {
    for (auto pivot = pivots.rbegin(); pivot != pivots.rend(); ++pivot) {
        for (auto prior = std::next(pivot); prior != pivots.rend(); ++prior) {
            const Q c = prior->second[pivot->first];
            if (c != 0) for (size_t j = pivot->first; j < width; ++j) prior->second[j] -= c * pivot->second[j];
        }
    }
}

Json::Value nullspace(const Pivots& pivots, size_t width) {
    Json::Value basis(Json::arrayValue);
    for (size_t col = 0; col < width; ++col) {
        if (pivots.count(col)) continue;
        Row direction(width);
        direction[col] = 1;
        for (const auto& [j, row] : pivots) direction[j] = -row[col];
        basis.append(strings(direction));
    }
    return basis;
}

Json::Value row_space(const Matrix& rows, size_t width, const Json::Value& witness, bool rank_only) {
    Pivots pivots;
    if (!echelon_rows(pivots, rows, width, witness)) return {};
    Json::Value out;
    out["rank"] = Json::UInt64(pivots.size());
    out["dimension"] = Json::UInt64(width - pivots.size());
    out["rows"] = out["basis"] = Json::arrayValue;
    if (!rank_only) {
        back_substitute(pivots, width);
        for (const auto& [col, row] : pivots) out["rows"].append(strings(row));
        out["basis"] = nullspace(pivots, width);
    }
    return out;
}

Json::Value determinant_plan(const Matrix& rows, size_t width);
Json::Value certified_determinant(const Json::Value& plan, const Json::Value& candidate, size_t n);

Q square_determinant(const Matrix& rows) {
    if (rows.empty()) return 1;
    const auto n = rows.size();
    fmpq_mat_t matrix;
    fmpq_mat_init(matrix, n, n);
    fmpq_t determinant;
    fmpq_init(determinant);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) fmpq_set_mpq(fmpq_mat_entry(matrix,i,j), rows[i][j].get_mpq_t());
    fmpq_mat_det(determinant, matrix);
    Q value;
    fmpq_get_mpq(value.get_mpq_t(), determinant);
    fmpq_clear(determinant);
    fmpq_mat_clear(matrix);
    return value;
}

Json::Value matrix_analysis(Matrix augmented, size_t width, bool solve, const Json::Value& witness,
                            const Json::Value& candidate) {
    const auto height = augmented.size();
    Pivots pivots;
    if (!echelon_rows(pivots, augmented, width+solve, witness)) return {};
    back_substitute(pivots, width+solve);
    // The last column can be a pivot precisely when the system is inconsistent.
    const bool inconsistent = solve && pivots.erase(width);
    Json::Value out;
    out["row_count"] = Json::UInt64(height);
    out["column_count"] = Json::UInt64(width);
    out["rank"] = Json::UInt64(pivots.size());
    out["determinant"] = out["solution"] = Json::nullValue;
    out["pivot_columns"] = out["image"] = out["rref"] = Json::arrayValue;
    Row particular(width);
    for (auto& [col,row] : pivots) {
        out["pivot_columns"].append(Json::UInt64(col));
        Row image;
        image.reserve(height);
        for (const Row& input : augmented) image.push_back(input[col]);
        out["image"].append(strings(image));
        if (solve) { particular[col] = row.back(); row.pop_back(); }
        out["rref"].append(strings(row));
    }
    if (height > pivots.size()) {
        const auto zero = strings(Row(width));
        while (out["rref"].size() < height) out["rref"].append(zero);
    }
    out["kernel"] = nullspace(pivots, width);
    if (height == width) {
        Json::Value determinant;
        if (!candidate.isNull())
            determinant = certified_determinant(determinant_plan(augmented,width), candidate, width);
        out["determinant_device_verified"] = !determinant.isNull();
        out["determinant"] = pivots.size() != width ? Json::Value("0") :
            determinant.isNull() ? Json::Value(square_determinant(augmented).get_str()) : determinant;
    }
    if (solve) {
        auto& solution = out["solution"];
        solution["status"] = inconsistent ? "inconsistent" : pivots.size() == width ? "unique" : "underdetermined";
        solution["particular"] = inconsistent ? Json::Value() : strings(particular);
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

// Remove all rational row content before testing the signed-word bound.
// This also admits very large coefficients which share a row scale.
Json::Value determinant_plan(const Matrix& rows, size_t width) {
    if (!width || rows.size() != width || width > 6) return {};
    Json::Value plan;
    plan["entries"] = Json::arrayValue;
    Z bound = 1;
    Q scale = 1;
    for (const auto& row : rows) {
        const auto entries = primitive(row.size() == width ? row : Row(row.begin(),row.begin()+width));
        Z norm = 0;
        for (const auto& x : entries) norm += abs(x);
        if (!norm) return {};
        bound *= norm;
        if (!bound.fits_slong_p() || mpz_sizeinbase(bound.get_mpz_t(), 2) > 63) return {};
        const auto first = std::find_if(entries.begin(), entries.end(), [](const Z& x){ return x != 0; });
        scale *= row[first-entries.begin()] / *first;
        for (const auto& x : entries) plan["entries"].append(Json::Int64(x.get_si()));
    }
    plan["bound"] = bound.get_str();
    plan["scale"] = scale.get_str();
    return plan;
}

Json::Value certified_determinant(const Json::Value& plan, const Json::Value& candidate, size_t n) {
    if (!plan.isObject() || !candidate.isString() || plan["entries"].size() != n*n) return {};
    const Z value(candidate.asString()), bound(plan["bound"].asString());
    if (abs(value) > bound) return {};
    Z modulus = 1;
    for (ulong p : {1000000007UL, 1000000009UL, 998244353UL}) {
        nmod_mat_t image;
        nmod_mat_init(image,n,n,p);
        for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) {
            const auto entry = plan["entries"][Json::ArrayIndex(i*n+j)].asInt64();
            const auto residue = entry % static_cast<int64_t>(p);
            nmod_mat_entry(image,i,j) = residue < 0 ? residue+static_cast<int64_t>(p) : residue;
        }
        const ulong det = nmod_mat_det(image);
        nmod_mat_clear(image);
        if (det != mpz_fdiv_ui(value.get_mpz_t(),p)) return {};
        modulus *= p;
        if (modulus > 2*bound) {
            Q exact = value * Q(plan["scale"].asString());
            return exact.get_str();
        }
    }
    return {};
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
enum class CenterKind { rows, unit, maximal };
struct CenteredGenerators {
    CenterKind kind = CenterKind::rows;
    std::vector<LocalGenerator> rows;
};
CenteredGenerators centered_generators(const Point& p, const std::vector<Terms>& ideal) {
    const size_t width = p.coordinates.size();
    const auto unit = one(width);
    std::vector<LocalGenerator> rows;
    std::set<Powers> monomials;
    CenteredGenerators out;
    for (const auto& f : ideal) {
        if (f.empty()) continue;
        auto g = point_generator(p, f);
        if (g.residual == unit) {
            if (g.powers == Powers(width)) return {CenterKind::unit, {}};
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
            if (full_cotangent_rank(p, ideal)) return {CenterKind::maximal, {}};
            check_rank = false;
        }
        out.rows.push_back(std::move(row));
    }
    return out;
}

Json::Value centered_plan(const Point& p, const std::vector<Terms>& ideal) {
    const auto centered = centered_generators(p, ideal);
    Json::Value out;
    out["rows"] = Json::arrayValue;
    if (centered.kind == CenterKind::unit) out["unit"] = true;
    if (centered.kind == CenterKind::maximal) out["maximal"] = true;
    for (const auto& row : centered.rows) {
        Json::Value item;
        item["powers"] = strings(row.powers);
        item["residual"] = term_list(row.residual);
        out["rows"].append(item);
    }
    return out;
}

}  // namespace
namespace atlas {
// The number of operations in the printed form of a polynomial: the cost used
// to rank presentations (engine/native_polynomial.py `count_ops`).
size_t expression_size(const std::map<std::vector<mpz_class>, mpq_class>& terms) {
    size_t cost = terms.empty() ? 0 : terms.size()-1;
    for (const auto& [powers, coefficient] : terms) {
        size_t factors = 0;
        for (const auto& k : powers) {
            if (k > 1) ++cost;
            if (k != 0) ++factors;
        }
        if (factors > 1) cost += factors-1;
        if (!powers.empty() && abs(coefficient) != 1) ++cost;
    }
    return cost;
}
}  // namespace atlas
namespace {
Json::Value compute(const Json::Value& r) {
    const std::string op = r["operation"].asString();
    if (op == "row_space" || op == "residues" || op == "matrix_analysis" || op == "matrix_determinant_plan") {
        size_t width = r["width"].asUInt64();
        const bool solve = op == "matrix_analysis" && r["augmented"].asBool();
        Matrix rows;
        for (const auto& v : r["rows"]) {
            rows.push_back(read_row(v));
            if (rows.back().size() != width+solve) throw std::invalid_argument("Invalid matrix row width.");
        }
        if (op == "matrix_determinant_plan") return determinant_plan(rows,width);
        if (op == "matrix_analysis") return matrix_analysis(std::move(rows), width, solve, r["witness"], r["det_candidate"]);
        if (op == "row_space") return row_space(rows, width, r["witness"], r["rank_only"].asBool());
        Json::Value out(Json::arrayValue);
        unsigned long prime = r["prime"].asUInt();
        if (prime < 2) throw std::invalid_argument("Invalid Jacobian residue modulus.");
        for (const auto& row : rows) for (const auto& n : primitive(row))
            out.append(static_cast<Json::UInt>(mpz_fdiv_ui(n.get_mpz_t(), prime)));
        return out;
    }
    if (op == "expression_size") return Json::UInt64(atlas::expression_size(read_terms(r["terms"], r["width"].asUInt64())));
    if (op == "marked_coefficients") {
        const auto coefficients = read_row(r["coefficients"]);
        Z den = 1;
        for (const Q& c : coefficients) den = lcm(den, c.get_den());
        Powers integers;
        for (const Q& c : coefficients) integers.emplace_back(c.get_num() * (den / c.get_den()));
        return strings(integers);
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
    if (op == "jets" || op == "rank_jets" || op == "sparse_jet_rank" || op == "centered" || op == "independent") {
        std::vector<Terms> ideal;
        for (const auto& item : r["ideal"]) ideal.push_back(read_terms(item, width));
        if (op == "sparse_jet_rank") {
            const int rank = power_row_rank(p, ideal, r["witness"]);
            return rank < 0 ? Json::Value() : Json::Value(rank);
        }
        if (op == "centered") return centered_plan(p, ideal);
        if (op == "independent") {
            if (p.moving.empty()) return atlas::local::independent_at_origin(ideal, width);
            if (ideal.empty() || ideal.size() > width) return false;
            Pivots pivots;
            for (const auto& f : ideal) insert(pivots, materialize(p.bases, differential_weights(p, f)));
            return pivots.size() == ideal.size();
        }
        Json::Value rows(Json::arrayValue);
        for (const auto& f : ideal)
            if (at_point(p, f)) throw std::invalid_argument("The point must lie on the scheme.");
        try {
            for (const auto& f : ideal) {
                auto row = materialize(p.bases, differential_weights(p, f), op == "rank_jets");
                if (std::any_of(row.begin(), row.end(), [](const Q& c){ return c != 0; })) rows.append(strings(row));
            }
        } catch (const WidePointRow&) {
            Json::Value wide;
            wide["residues"] = modular_power_rows(p, ideal, r["prime"].asUInt());
            return wide;
        }
        return rows;
    }
    auto f = read_terms(r["terms"], width);
    if (op == "nonzero") return at_point(p, f);
    if (op == "order") return order_at_point(p, f).get_str();
    if (op == "principal_series_head") {
        // A principal homogeneous initial ideal has Hilbert series determined
        // by this single degree, independently of its leading direction.
        if (!width) throw std::invalid_argument("A principal head needs a variable.");
        Powers head(width);
        head.front() = order_at_point(p, f);
        return strings(head);
    }
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

namespace atlas::local {
bool independent_at_origin(const Basis& equations, size_t width) {
    if (equations.empty() || equations.size() > width) return false;
    Pivots pivots;
    Powers linear(width);
    for (const auto& f : equations) {
        Row differential(width);
        for (size_t axis = 0; axis < width; ++axis) {
            linear[axis] = 1;
            const auto term = f.find(linear);
            if (term != f.end()) differential[axis] = term->second;
            linear[axis] = 0;
        }
        insert(pivots, std::move(differential));
    }
    return pivots.size() == equations.size();
}

std::optional<Polynomial> bounded_initial_at_point(const Polynomial& f, const PointRow& point) {
    try { return initial_form(Point(point), f, true); }
    catch (const WidePointRow&) { return std::nullopt; }
}

int hypersurface_jet_order(const Polynomial& f, const PointRow& point) {
    const Point p(point);
    if (at_point(p,f)) return 0;
    for (const auto& weights : differential_weights(p,f))
        if (weights_nonzero(p.bases,weights)) return 1;
    return 2;
}

Basis simplify_point_generators(const Basis& ideal, const PointRow& point) {
    const auto centered = centered_generators(Point(point),ideal);
    if (centered.kind == CenterKind::unit) return {one(point.size())};
    Basis result;
    if (centered.kind == CenterKind::maximal) {
        for (size_t i=0; i<point.size(); ++i) {
            Powers axis(point.size()); axis[i]=1;
            Polynomial coordinate{{std::move(axis),1}};
            if (point[i] != 0) coordinate.emplace(Powers(point.size()),-point[i]);
            result.push_back(std::move(coordinate));
        }
        return result;
    }
    for (const auto& row : centered.rows) {
        Polynomial f;
        for (const auto& [e,c] : row.residual) {
            auto powers = e;
            for (size_t i=0; i<powers.size(); ++i) powers[i] += row.powers[i];
            f.emplace(std::move(powers),c);
        }
        if (!f.empty()) result.push_back(std::move(f));
    }
    return result;
}

Powers head_at_point(const Polynomial& f, const PointRow& point, bool lex) {
    const Point p(point);
    Polynomial support;
    for (const auto& [e, weights] : initial_weights(p,f))
        if (weights_nonzero(p.bases,weights)) support.emplace(e,1);
    return local_head(support,lex);
}

bool nonzero_at_point(const Polynomial& f, const PointRow& point) {
    return at_point(Point(point), f);
}

mpz_class order_at_point(const Polynomial& f, const PointRow& point) {
    return ::order_at_point(Point(point), f);
}

namespace {
mpq_class evaluated_weights(const Point& point, const Terms& weights) {
    const auto floor = minima(weights);
    mpq_class value = 0;
    for (const auto& [e,c] : weights) value += c * residual_scale(point.bases, e, floor);
    Powers zero(floor.size()), numerator = zero, denominator = zero;
    for (size_t i = 0; i < floor.size(); ++i) {
        if (floor[i] > 0) numerator[i] = floor[i];
        else denominator[i] = -floor[i];
    }
    return value * residual_scale(point.bases, numerator, zero) / residual_scale(point.bases, denominator, zero);
}
}

Polynomial specialize_coordinate(const Polynomial& f, size_t axis, const mpq_class& coordinate) {
    const Point p({coordinate});
    // Group by the remaining monomial before evaluating its coefficient.
    // Exact cancellation can remove terms such as (2^a-4^b) symbolically.
    std::map<Powers, Terms> groups;
    for (const auto& [e,c] : f) {
        const Powers a{e[axis]};
        if (coordinate == 0 && a[0] != 0) continue;
        auto projected = e;
        projected.erase(projected.begin()+axis);
        add(groups[projected], p.key(a), p.signed_value(a,c));
    }
    Polynomial result;
    for (const auto& [e,weights] : groups)
        if (weights_nonzero(p.bases, weights)) result.emplace(e, evaluated_weights(p, weights));
    return result;
}

namespace {
Coefficients point_jet_weights(const Polynomial& f, const Point& p, const mpz_class& cutoff,
                               const Powers& axis_bounds) {
    // Sum contributions to each Taylor coefficient before building the
    // integer powers. Coordinates such as 2 and 1/4 share a numeric base.
    Coefficients sums;
    for (const auto& [e,c] : f) {
        Powers k(e.size(),0);
        mpz_class fixed = 0;
        if (!axis_bounds.empty() && std::any_of(p.fixed.begin(), p.fixed.end(), [&](size_t i) {
            return axis_bounds[i] != 0 && e[i] >= axis_bounds[i];
        })) continue;
        for (auto i : p.fixed) { k[i]=e[i]; fixed+=e[i]; }
        if (fixed >= cutoff) continue;
        auto key=p.key(e);
        const auto signed_coefficient=p.signed_value(e,c);
        // point.key(e) already carries p^e, so each derivative step
        // below only adds its falling-factorial and 1/p_i.
        auto visit = [&](auto&& self, size_t j, const mpz_class& room, const mpq_class& coefficient) -> void {
            if (j==p.moving.size() || room==0) { add(sums[k],key,coefficient); return; }
            const auto axis=p.moving[j];
            mpq_class weight=coefficient;
            mpz_class last=std::min(e[axis],room);
            if (!axis_bounds.empty() && axis_bounds[axis] != 0 && last >= axis_bounds[axis])
                last = axis_bounds[axis] - 1;
            for (mpz_class a=0; a<=last; ++a) {
                k[axis]=a;
                self(self,j+1,room-a,weight);
                weight*=e[axis]-a;
                weight/=(a+1)*p.coordinates[axis];
            }
            k[axis]=0;
        };
        visit(visit,0,cutoff-fixed-1,signed_coefficient);
    }
    return sums;
}
}

Polynomial grouped_point_jet(const Polynomial& f, const PointRow& point, const mpz_class& cutoff,
                             const Powers& axis_bounds) {
    const Point p(point);
    Polynomial result;
    for (const auto& [k,weights] : point_jet_weights(f,p,cutoff,axis_bounds))
        if (weights_nonzero(p.bases, weights)) result.emplace(k, evaluated_weights(p, weights));
    return result;
}

std::vector<std::map<Powers, Polynomial>> symbolic_point_jets(const Basis& ideal, const PointRow& point,
                                                              unsigned cutoff) {
    const Point p(point);
    std::vector<Coefficients> jets;
    jets.reserve(ideal.size());
    for (const auto& f : ideal) {
        auto jet = point_jet_weights(f,p,cutoff,{});
        for (auto it = jet.begin(); it != jet.end();) {
            if (!weights_nonzero(p.bases,it->second)) it = jet.erase(it);
            else ++it;
        }
        jets.push_back(std::move(jet));
    }
    return jets;
}

std::vector<size_t> symbolic_pivot_columns(const PointRow& point, std::vector<std::vector<Polynomial>>&& rows) {
    const Point p(point);
    std::map<size_t,PowerRow> pivots;
    for (auto& row : rows) insert_power_row(p,std::move(row),pivots);
    std::vector<size_t> columns;
    columns.reserve(pivots.size());
    for (const auto& entry : pivots) columns.push_back(entry.first);
    return columns;
}

std::optional<Polynomial> bounded_ideal_jet(const Polynomial& f, const PointRow& point,
                                            const mpz_class& cutoff) {
    const Point p(point);
    auto sums = point_jet_weights(f,p,cutoff,{});
    std::vector<Terms> weights;
    weights.reserve(sums.size());
    for (auto& [e,row] : sums) weights.push_back(std::move(row));
    try {
        const auto coefficients = materialize(p.bases, weights, true);
        Polynomial jet;
        size_t i = 0;
        for (const auto& entry : sums) {
            if (coefficients[i] != 0) jet.emplace(entry.first, coefficients[i]);
            ++i;
        }
        return jet;
    } catch (const WidePointRow&) {
        return std::nullopt;
    }
}

std::pair<Powers, mpq_class> leading_term_at_point(const Polynomial& f, const PointRow& point, bool lex) {
    const Point p(point);
    auto initial = initial_form(p, f);
    auto head = local_head(initial, lex);
    // Extract just the Taylor coefficient indexed by head. Group equal
    // rational powers before evaluating so large common factors cancel.
    Terms weights;
    for (const auto& [e,c] : f) {
        if (std::any_of(p.fixed.begin(), p.fixed.end(), [&](auto i) { return e[i] != head[i]; })) continue;
        auto remaining = e;
        mpq_class factor = c;
        for (auto i : p.moving) {
            if (head[i] > e[i]) { factor = 0; break; }
            // The moving degree of an initial term is < the supported
            // slice's term count by the sparse interpolation bound.
            mpz_class binomial;
            mpz_bin_ui(binomial.get_mpz_t(), e[i].get_mpz_t(), head[i].get_ui());
            factor *= binomial;
            remaining[i] -= head[i];
        }
        if (factor != 0) add(weights, p.key(remaining), p.signed_value(remaining, factor));
    }
    return {std::move(head), evaluated_weights(p, weights)};
}

namespace {
std::optional<DimensionResult> small_hypersurface(const Polynomial& f, const Row& point,
                                                  bool smoothness, int certified_rank) {
    // Evaluate the constant and first derivatives together. Cache p_i^k
    // per used coordinate rather than grouping symbolic products, which is
    // only worthwhile when the exponents or point coefficients are large.
    const auto width = point.size();
    std::vector<unsigned> degrees(width);
    for (const auto& [e, c] : f) for (size_t i = 0; i < width; ++i) {
        if (point[i] == 0) continue;
        if (e[i] > 24) return std::nullopt;
        degrees[i] = std::max(degrees[i], static_cast<unsigned>(e[i].get_ui()));
    }
    std::vector<Row> powers(width);
    for (size_t i = 0; i < width; ++i) if (degrees[i]) {
        if (mpz_sizeinbase(mpq_numref(point[i].get_mpq_t()), 2) > 128 ||
            mpz_sizeinbase(mpq_denref(point[i].get_mpq_t()), 2) > 128)
            return std::nullopt;
        powers[i].push_back(1);
        for (unsigned k = 1; k <= degrees[i]; ++k) powers[i].push_back(powers[i].back()*point[i]);
    }
    const bool jets = smoothness && certified_rank < 0;
    Row linear(jets ? width : 0);
    Q constant = 0;
    for (const auto& [e, c] : f) {
        size_t zeros = 0, axis = 0;
        for (size_t i = 0; i < width; ++i) if (!point[i] && e[i] != 0) {
            zeros += e[i] == 1 ? 1 : 2;
            axis = i;
        }
        if (zeros > (jets ? 1U : 0U)) continue;
        Q term = c;
        for (size_t i = 0; i < width; ++i) if (degrees[i] && e[i] != 0)
            term *= powers[i][e[i].get_ui()];
        if (zeros) linear[axis] += term;
        else {
            constant += term;
            // On nonzero axes compute p_i * df/dx_i; only the nonvanishing
            // of the entire gradient is needed for the hypersurface rank.
            if (jets) for (size_t i = 0; i < width; ++i) if (e[i] != 0) linear[i] += term*e[i];
        }
    }
    if (constant != 0) throw PointOutsideScheme();
    const int rank = jets ? std::any_of(linear.begin(), linear.end(), [](const auto& c) { return c != 0; })
                          : certified_rank;
    const int n = static_cast<int>(width);
    return DimensionResult{n-1, rank < 0 ? -1 : n-rank, rank, rank == 1};
}

std::optional<int> dimension_mod_nilpotent_axes(const Basis& ideal, const CenteredGenerators& centered,
                                               const std::vector<std::string>& names, const Point& p,
                                               bool lex, bool prefer_modular) {
    const auto unit = one(names.size());
    std::vector<bool> nilpotent(names.size());
    for (const auto& row : centered.rows) {
        if (row.residual != unit) continue;
        // Only coordinates with p_i=0 occur in the retained monomial.
        // A pure power in the ideal makes that coordinate nilpotent.
        auto axis = std::find_if(row.powers.begin(), row.powers.end(), [](const auto& e) { return e != 0; });
        if (axis != row.powers.end() && std::none_of(axis+1, row.powers.end(), [](const auto& e) { return e != 0; }))
            nilpotent[axis-row.powers.begin()] = true;
    }
    if (std::none_of(nilpotent.begin(), nilpotent.end(), [](bool v) { return v; })) return std::nullopt;

    std::vector<std::string> variables;
    Row point;
    for (size_t i = 0; i < names.size(); ++i) if (!nilpotent[i]) {
        variables.push_back(names[i]);
        point.push_back(p.coordinates[i]);
    }
    Basis projected;
    for (const auto& f : ideal) {
        Polynomial g;
        for (const auto& [e, c] : f) {
            if (std::any_of(p.fixed.begin(), p.fixed.end(),
                            [&](size_t i) { return nilpotent[i] && e[i] != 0; })) continue;
            Powers powers;
            for (size_t i = 0; i < names.size(); ++i) if (!nilpotent[i]) powers.push_back(e[i]);
            g.emplace(std::move(powers), c);
        }
        if (!g.empty()) projected.push_back(std::move(g));
    }
    return dimension_at_point(projected, variables, point, lex, false, prefer_modular).dimension;
}

// Constant row operations expose generators after cancellation of common
// high-degree terms. The echelon rows span the same QQ vector space, so
// they generate the same ideal and give a sharper height-theorem bound.
Basis echelon_generators(const Basis& ideal) {
    std::map<Powers, Polynomial> pivots;
    for (auto row : ideal) while (!row.empty()) {
        const auto [head, coefficient] = *row.rbegin();
        const auto pivot = pivots.find(head);
        if (pivot == pivots.end()) {
            for (auto& term : row) term.second /= coefficient;
            pivots.emplace(head, std::move(row));
            break;
        }
        for (const auto& [e, c] : pivot->second) add(row, e, -coefficient*c);
    }
    Basis rows;
    for (auto& entry : pivots) rows.push_back(std::move(entry.second));
    return rows;
}

int centered_ideal_dimension(const Basis& rows, const std::vector<std::string>& names,
                             bool lex, bool prefer_modular) {
    const auto width = static_cast<int>(names.size());
    if (auto heads = local_leading_supports(rows, names.size(), lex, prefer_modular))
        return monomial_support_dimension(std::move(*heads), width);
    const auto basis = standard_basis(rows, names, lex, prefer_modular);
    std::vector<std::vector<int>> supports;
    for (const auto& f : basis) {
        const auto head = local_head(f, lex);
        std::vector<int> support;
        for (int i = 0; i < width; ++i) if (head[i] != 0) support.push_back(i);
        if (support.empty()) throw std::logic_error("On-scheme local basis contains a unit.");
        supports.push_back(std::move(support));
    }
    return monomial_support_dimension(std::move(supports), width);
}

int coordinate_plane_dimension(const Point& p, const Basis& rows, int width) {
    // Group by exponents on axes where p_i=0. Each group is a monomial in
    // those axes times a polynomial in the others. A fixed zero factor
    // kills it. If the polynomial vanishes at p, fixing all its varying
    // axes also kills it: the remaining factors are common monomials.
    std::vector<std::vector<int>> supports;
    for (const auto& row : rows) {
        std::map<Powers, Polynomial> groups;
        for (const auto& [e, c] : row) {
            Powers key;
            for (auto i : p.fixed) key.push_back(e[i]);
            groups[key].emplace(e, c);
        }
        for (const auto& [key, group] : groups) {
            std::vector<int> support;
            for (size_t j = 0; j < key.size(); ++j)
                if (key[j] != 0) support.push_back(static_cast<int>(p.fixed[j]));
            Terms weights;
            for (const auto& [e, c] : group) add(weights, p.key(e), p.signed_value(e,c));
            if (weights_nonzero(p.bases, weights)) {
                if (support.empty()) throw PointOutsideScheme();
                supports.push_back(std::move(support));
                continue;
            }
            const auto& first = group.begin()->first;
            for (auto i : p.moving) {
                if (std::any_of(group.begin(), group.end(), [&](const auto& term) { return term.first[i] != first[i]; })) {
                    auto condition = support;
                    condition.push_back(static_cast<int>(i));
                    std::sort(condition.begin(), condition.end());
                    supports.push_back(std::move(condition));
                }
            }
        }
    }
    return monomial_support_dimension(std::move(supports), width);
}

Basis by_degree(const Basis& ideal) {
    // Start gcds with the lowest-degree equations. Two high-degree
    // multiples can be much more expensive to compare than either is
    // with another generator containing the same low-degree factor.
    std::multimap<mpz_class, const Polynomial*> ordered;
    for (const auto& f : ideal) if (!f.empty()) {
        mpz_class degree = 0;
        for (const auto& [e, c] : f) {
            mpz_class term = 0;
            for (const auto& power : e) term += power;
            degree = std::max(degree, term);
        }
        ordered.emplace(std::move(degree), &f);
    }
    Basis rows;
    for (const auto& entry : ordered) rows.push_back(*entry.second);
    return rows;
}

bool contains_factor(const Polynomial& f, const Polynomial& factor, size_t width) {
    // For a linear factor solve for the axis on which f has smallest
    // degree. Substituting along that axis keeps widely separated powers
    // in the other coordinates sparse, even for a failed divisibility test.
    bool linear = true;
    std::vector<mpq_class> coefficients(width);
    for (const auto& [e, c] : factor) {
        size_t axis = width;
        for (size_t i = 0; i < width; ++i) if (e[i] != 0) {
            if (axis != width || e[i] != 1) { linear = false; break; }
            axis = i;
        }
        if (!linear) break;
        if (axis != width) coefficients[axis] = c;
    }
    if (linear) {
        size_t axis = width;
        mpz_class least = 4;
        // Substitution of 0 or ±1 is sparse at every exponent.
        const auto constant = factor.find(Powers(width));
        if (std::count_if(coefficients.begin(), coefficients.end(), [](const auto& c) { return c != 0; }) == 1) {
            auto pivot = std::find_if(coefficients.begin(), coefficients.end(), [](const auto& c) { return c != 0; });
            if (constant == factor.end() || abs(constant->second) == abs(*pivot))
                axis = static_cast<size_t>(pivot-coefficients.begin());
        }
        for (size_t i = 0; i < width; ++i) if (coefficients[i] != 0) {
            mpz_class degree = 0;
            for (const auto& [e, c] : f) degree = std::max(degree, e[i]);
            if (degree < least) { least = degree; axis = i; }
        }
        if (axis != width) {
            Polynomial image;
            for (const auto& [e, c] : factor) if (e[axis] == 0) {
                auto reduced = e;
                reduced.erase(reduced.begin()+axis);
                image.emplace(std::move(reduced), -c/coefficients[axis]);
            }
            std::map<mpz_class, Polynomial> powers;
            return primary::substitute(f, axis, image, width-1, powers).empty();
        }
    }
    return polynomial_divides(factor, f, width);
}

int shared_factor_lower_bound(const Point& p, Basis rows, int width, int upper) {
    // If f and h share g with g(p)=0, replacing (f,h) by (g) gives a
    // larger ideal still through p. Its height bound applies to the
    // original scheme as well. Repeat while this can improve the bound.
    for (size_t i = 0; i < rows.size() && width-static_cast<int>(rows.size()) < upper; ++i) {
        // A low-degree equation can already display the hypersurface
        // factors shared by the high-degree equations. Verify divisibility
        // before attempting a gcd of two polynomials with distant powers.
        const bool low_degree = std::all_of(rows[i].begin(), rows[i].end(), [](const auto& term) {
            mpz_class degree = 0;
            for (const auto& e : term.first) degree += e;
            return degree <= 3;
        });
        if (low_degree) {
            for (const auto& factor : primary::factors(rows[i], width)) {
                if (at_point(p, factor.first)) continue;
                int equations = static_cast<int>(rows.size());
                for (size_t j = i+1; j < rows.size(); ++j) {
                    if (contains_factor(rows[j], factor.first, width)) --equations;
                    if (width-equations == upper) return upper;
                }
            }
        }
        size_t j = i+1;
        while (j < rows.size()) {
            auto g = polynomial_gcd({rows[i], rows[j]}, width);
            if (!g || g->empty() || at_point(p, *g)) { ++j; continue; }
            rows[i] = std::move(*g);
            rows.erase(rows.begin()+j);
            if (width-static_cast<int>(rows.size()) == upper) return upper;
        }
    }
    return width-static_cast<int>(rows.size());
}

std::optional<int> dimension_common_factor(const Basis& ideal, const std::vector<std::string>& names,
        const std::vector<mpq_class>& point, bool lex, bool prefer_modular) {
    // With two equations or in the plane the caller already has the gcd test.
    if (ideal.size() < 3 || names.size() < 3) return std::nullopt;
    const auto common = polynomial_gcd(ideal, names.size());
    if (!common || common->empty() || (common->size() == 1 && common->begin()->first == Powers(names.size())))
        return std::nullopt;
    // The variety of gJ is the union of the hypersurface g=0 and V(J).
    // A nonzero common factor vanishing at p therefore realizes dimension n-1.
    if (!at_point(Point(point), *common)) return static_cast<int>(names.size())-1;
    // Otherwise g is a unit at p. The local ideals gJ and J coincide.
    auto quotient = divide_generators(ideal, *common, names.size());
    return dimension_at_point(quotient, names, point, lex, false, prefer_modular).dimension;
}
}

DimensionResult dimension_at_point(const Basis& ideal, const std::vector<std::string>& names,
                                   const std::vector<mpq_class>& point, bool lex, bool smoothness,
                                   bool prefer_modular, int certified_rank) {
    if (point.size() != names.size() || names.size() > static_cast<size_t>(INT_MAX))
        throw std::invalid_argument("Incorrect local point dimension.");
    const int width = static_cast<int>(names.size());
    const Powers zero(width, 0);
    int count = 0;
    const Polynomial* principal = nullptr;
    for (const auto& f : ideal) {
        if (f.empty()) continue;
        principal = &f;
        ++count;
    }
    if (certified_rank > width || certified_rank > count)
        throw std::invalid_argument("Invalid certified Jacobian rank.");
    if (!count) return ambient_space_at_point(width);
    if (count == 1) {
        auto fast = small_hypersurface(*principal, point, smoothness, certified_rank);
        if (fast) return *fast;
    }
    const Point p(point);
    const bool monomial = std::all_of(ideal.begin(), ideal.end(), [](const auto& f) { return f.size() <= 1; });
    if (monomial) {
        // At a nonzero coordinate its whole monomial factor is a unit.
        // Only the supports on the remaining axes affect Krull dimension;
        // among these monomials only degree-one rows enter the Jacobian.
        std::vector<std::vector<int>> supports;
        std::set<int> linear;
        for (const auto& f : ideal) if (!f.empty()) {
            const auto& e = f.begin()->first;
            std::vector<int> support;
            for (size_t i : p.fixed) if (e[i] != 0) support.push_back(static_cast<int>(i));
            if (support.empty()) throw PointOutsideScheme();
            if (support.size() == 1 && e[support.front()] == 1) linear.insert(support.front());
            supports.push_back(std::move(support));
        }
        const int rank = static_cast<int>(linear.size()), embedding = width-rank;
        const int dimension = monomial_support_dimension(std::move(supports), width);
        return {dimension, embedding, rank, embedding == dimension};
    }

    int rank = certified_rank;
    constexpr unsigned power_chart_threshold = 8;
    // Small translated systems are cheaper to expand once, then evaluate
    // both their constant and linear coefficients directly in centered
    // coordinates. Wide sparse systems keep their factored point evaluation.
    if (!p.moving.empty() && count > 1) {
        unsigned long terms = 0;
        bool small = true;
        for (const auto& f : ideal) {
            for (const auto& [e, c] : f) {
                unsigned long size = 1;
                for (auto i : p.moving) {
                    if (e[i] >= (512-terms)/size) { small = false; break; }
                    size *= 1+e[i].get_ui();
                }
                // A shifted high power can yield an expensive dense local
                // basis even when its expansion is cheap to construct.
                if (size >= power_chart_threshold) small = false;
                if (!small) break;
                terms += size;
            }
            if (!small) break;
        }
        if (small) return dimension_at_point(translate(ideal, point), names, Row(width, 0), lex,
                                             smoothness, prefer_modular, certified_rank);
    }
    for (const auto& f : ideal)
        if (!f.empty() && (p.moving.empty() ? f.count(zero) != 0 : at_point(p, f)))
            throw PointOutsideScheme();
    if (rank < 0 && (smoothness || count > 1)) {
        Pivots pivots;
        try {
            for (const auto& f : ideal) {
                if (f.empty()) continue;
                Row row(width);
                if (p.moving.empty()) {
                    Powers e(width, 0);
                    for (int i = 0; i < width; ++i) {
                        e[i] = 1;
                        const auto linear = f.find(e);
                        if (linear != f.end()) row[i] = linear->second;
                        e[i] = 0;
                    }
                } else row = materialize(p.bases, differential_weights(p, f), true);
                insert(pivots, std::move(row));
                if (pivots.size() == names.size()) break;
            }
            rank = static_cast<int>(pivots.size());
        } catch (const WidePointRow&) {
            rank = power_row_rank(p, ideal);
        }
    }
    const int embedding = rank < 0 ? -1 : width-rank;
    const auto result = [&](int dimension) {
        return DimensionResult{dimension, embedding, rank, embedding == dimension};
    };
    // Krull's height theorem gives width-count <= dim <= embedding.
    // The lower bound for one nonzero equation is attained in a regular domain.
    if (count == 1) return result(width-1);
    if (rank == width || rank == count) return result(embedding);
    if (p.moving.empty() && rank > 0) {
        // Constant row operations preserve the ideal. If the input has
        // exactly rank linearly independent polynomial rows, the height
        // bound applies to this smaller generating set and meets embedding.
        std::map<Powers, Polynomial> independent;
        for (auto row : ideal) {
            while (!row.empty()) {
                const auto key = row.begin()->first;
                const auto coefficient = row.begin()->second;
                auto pivot = independent.find(key);
                if (pivot == independent.end()) {
                    for (auto& term : row) term.second /= coefficient;
                    independent.emplace(key, std::move(row));
                    break;
                }
                for (const auto& [e, c] : pivot->second) add(row, e, -coefficient*c);
            }
            if (independent.size() > static_cast<size_t>(rank)) break;
        }
        if (independent.size() == static_cast<size_t>(rank)) return result(embedding);
    }

    Basis generators;
    const Basis* centered = &ideal;
    if (!p.moving.empty()) {
        const auto plan = centered_generators(p, ideal);
        if (plan.kind == CenterKind::unit) throw PointOutsideScheme();
        if (plan.kind == CenterKind::maximal) return result(0);
        if (plan.rows.size() == 1) return result(width-1);
        // Each on-scheme row in disjoint sets of variables is a nonzero
        // hypersurface in its own affine factor. The product has codimension
        // one per row at the given rational point. Detect this on the
        // factored rows before translating a large power in a moving axis.
        std::vector<bool> used(width);
        bool disjoint = true;
        for (const auto& row : plan.rows) {
            for (int i = 0; i < width; ++i) {
                const bool active = row.powers[i] != 0 || std::any_of(row.residual.begin(), row.residual.end(),
                    [&](const auto& term) { return term.first[i] != 0; });
                if (!active) continue;
                if (used[i]) { disjoint = false; break; }
                used[i] = true;
            }
            if (!disjoint) break;
        }
        if (disjoint) return result(width-static_cast<int>(plan.rows.size()));
        const bool large = std::any_of(plan.rows.begin(), plan.rows.end(),
            [&](const auto& row) { return large_translation(p, row.residual, 4096); });
        if (large || std::any_of(plan.rows.begin(), plan.rows.end(),
            [&](const auto& row) { return large_translation(p, row.residual, power_chart_threshold); })) {
            if (auto d = dimension_power_coordinates(ideal, names, point, lex, prefer_modular))
                return result(*d);
        }
        if (large) {
            // Nilpotent quotients preserve the prime spectrum. Keep the
            // tangent rank from the original ideal, including its nilpotents.
            const auto d = dimension_mod_nilpotent_axes(ideal, plan, names, p, lex, prefer_modular);
            if (d) return result(*d);
        }
        // In a regular local UFD a height-one prime containing the ideal
        // is detected by a common polynomial factor vanishing at p. With
        // two generators, or in the plane, the remaining possible height
        // is exactly two. Work in original coordinates for a large shift.
        if (large && (width == 2 || count == 2)) {
            const auto common = polynomial_gcd(ideal, width);
            if (common) return result(width - (at_point(p, *common) ? 2 : 1));
        }
        if (large) {
            auto rows = echelon_generators(ideal);
            // Initial forms of these generators belong to the tangent ideal.
            // Their quotient dimension is an upper bound; the number of
            // generators gives the complementary lower bound by height.
            Basis forms;
            try {
                for (const auto& row : rows) forms.push_back(initial_form(p, row, true));
            } catch (const WidePointRow&) {
                // The exact tangent coefficients may be enormous. The graph
                // uses only the supplied rational coordinates and coefficients.
                return result(dimension_from_point_graph(rows, point, prefer_modular));
            }
            const int upper = centered_ideal_dimension(forms, names, lex, prefer_modular);
            const int lower = std::max(width-static_cast<int>(rows.size()), coordinate_plane_dimension(p, rows, width));
            if (upper == lower) return result(upper);
            auto ordered = by_degree(ideal);
            if (upper == width-1)
                if (auto d = dimension_common_factor(ordered, names, point, lex, prefer_modular)) return result(*d);
            if (shared_factor_lower_bound(p, std::move(ordered), width, upper) == upper) return result(upper);
            return result(dimension_from_point_graph(rows, point, prefer_modular));
        }
        for (const auto& source : plan.rows) {
            auto residual = translate({source.residual}, point).front();
            if (source.powers == zero) {
                generators.push_back(std::move(residual));
            } else {
                Polynomial scaled;
                for (const auto& [powers, c] : residual) {
                    auto e = powers;
                    for (int i = 0; i < width; ++i) e[i] += source.powers[i];
                    scaled.emplace(std::move(e), c);
                }
                generators.push_back(std::move(scaled));
            }
        }
        centered = &generators;
    }
    // The associated graded ring has the same dimension as the local ring.
    // Its initial monomial ideal has the same dimension again. A coordinate
    // prime contains that ideal precisely when it meets every leading support.
    return result(centered_ideal_dimension(*centered, names, lex, prefer_modular));
}
}  // namespace atlas::local

extern "C" const char* atlas_point_geometry(const char* input) noexcept {
    return atlas::sparse_json::invoke(input, compute);
}
