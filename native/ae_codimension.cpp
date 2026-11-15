#include "ae_codimension.hpp"
#include "ae_graded.hpp"
#include "ae_special.hpp"
#include "local_dimension.hpp"
#include "local_monomial.hpp"
#include "primary_decomposition.hpp"

#include <flint/fmpq.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace atlas::local {
namespace {

// A p-vector of source jets in total-degree order, with position m*p+j.
using Index = unsigned;
using Exponent = std::vector<unsigned>;

// An exact rational. FLINT keeps a small numerator and denominator inside
// the words themselves, so the short coefficients of jet rows need neither
// allocation nor multiprecision gcds.
struct Rational {
    fmpq_t v;
    Rational() noexcept { fmpq_init(v); }
    explicit Rational(const mpq_class& q) { fmpq_init(v); fmpq_set_mpq(v, q.get_mpq_t()); }
    Rational(const Rational& other) { fmpq_init(v); fmpq_set(v, other.v); }
    Rational(Rational&& other) noexcept { *v = *other.v; fmpq_init(other.v); }
    Rational& operator=(Rational other) noexcept { fmpq_swap(v, other.v); return *this; }
    ~Rational() { fmpq_clear(v); }
};

// The nonzero coefficients of a row in increasing column order.
struct Term {
    Index col;
    Rational value;
};
using Row = std::vector<Term>;

void sort_columns(Row& row) {
    std::sort(row.begin(), row.end(), [](const Term& a, const Term& b) { return a.col < b.col; });
}

// Add the terms of equal columns and drop zero sums.
Row collect(Row terms) {
    sort_columns(terms);
    Row row;
    row.reserve(terms.size());
    for (auto& term : terms) {
        if (!row.empty() && row.back().col == term.col) {
            fmpq_add(row.back().value.v, row.back().value.v, term.value.v);
            if (fmpq_is_zero(row.back().value.v)) row.pop_back();
        } else {
            row.push_back(std::move(term));
        }
    }
    return row;
}
constexpr Index missing = std::numeric_limits<Index>::max();

struct JetLimit {};
struct Budget {
    static constexpr size_t max_columns = 10000;
    size_t steps = 12000000;
    void use(size_t count) {
        if (count > steps) throw JetLimit{};
        steps -= count;
    }
};

class Monomials {
    std::map<Exponent, Index> ids_;
    std::vector<Index> parent_, axis_, next_;
public:
    const Index width, cutoff;
    std::vector<Index> degree;

    Monomials(Index n, Index d, Index components) : width(n), cutoff(d) {
        // Each monomial has one construction by appending variables in sorted order.
        // The binomial count is checked before constructing any exponent vectors.
        size_t count = 1;
        for (Index k = 1; k <= d; ++k) {
            count = count * (static_cast<size_t>(n) + k) / k;
            if (count > Budget::max_columns / components || count > 1000000 / n)
                throw JetLimit{};
        }
        std::vector<Exponent> powers;
        powers.reserve(count);
        powers.emplace_back(n);
        degree.push_back(0);
        parent_.push_back(0);
        axis_.push_back(0);
        for (size_t m = 0; m < powers.size() && degree[m] < d; ++m) {
            for (Index axis = axis_[m]; axis < n; ++axis) {
                powers.push_back(powers[m]);
                ++powers.back()[axis];
                degree.push_back(degree[m] + 1);
                parent_.push_back(m);
                axis_.push_back(axis);
            }
        }
        for (Index m = 0; m < powers.size(); ++m) ids_.emplace(powers[m],m);
        next_.resize((std::lower_bound(degree.begin(),degree.end(),d) - degree.begin()) * n);
        for (Index m = 0; degree[m] < d; ++m) {
            for (Index axis = 0; axis < n; ++axis) {
                ++powers[m][axis];
                next_[m*n+axis] = ids_.at(powers[m]);
                --powers[m][axis];
            }
        }
    }

    Index id(const Powers& powers) const {
        Exponent small;
        small.reserve(width);
        unsigned total = 0;
        for (const auto& exponent : powers) {
            if (exponent > cutoff - total) return missing;
            total += exponent.get_ui();
            small.push_back(exponent.get_ui());
        }
        return ids_.at(small);
    }

    Index shift(Index m, Index axis) const { return next_[m*width+axis]; }

    Index product(Index m, Index b) const {
        // Caller checks the total degree before multiplying.
        while (b) {
            m = shift(m,axis_[b]);
            b = parent_[b];
        }
        return m;
    }
};

// Keep the normalized rows by smallest nonzero column. The derivative space
// below is common to both tangent spaces and stays stable under multiplication.
class Echelon {
    std::vector<Row> rows_;
    const Echelon* base_;
    Budget& budget_;
    Row scratch_;
public:
    std::vector<Index> pivots;
    Echelon(size_t width, Budget& budget, const Echelon* base = nullptr)
        : rows_(width), base_(base), budget_(budget) {}

    const Row& row(Index pivot) const { return rows_[pivot]; }
    bool occupied(Index i) const {
        return !rows_[i].empty() || (base_ && !base_->rows_[i].empty());
    }

    // Remove the coefficient of r[at] at a normalized pivot and keep only
    // the terms after it. Every other basis term follows the pivot column,
    // so a merge keeps the column order. Coordinate rows need no rational
    // multiplication.
    void reduce_pivot(Row& r, size_t at, const Row& basis) {
        scratch_.clear();
        if (basis.size() == 1) {
            for (size_t i = at+1; i < r.size(); ++i) scratch_.push_back(std::move(r[i]));
            r.swap(scratch_);
            return;
        }
        const Rational lc = std::move(r[at].value);
        budget_.use(basis.size());
        scratch_.reserve(r.size()-at-1+basis.size()-1);
        size_t i = at+1, j = 1;
        while (i < r.size() || j < basis.size()) {
            if (j == basis.size() || (i < r.size() && r[i].col < basis[j].col)) {
                scratch_.push_back(std::move(r[i++]));
                continue;
            }
            if (i < r.size() && r[i].col == basis[j].col) {
                fmpq_submul(r[i].value.v, lc.v, basis[j].value.v);
                if (!fmpq_is_zero(r[i].value.v)) scratch_.push_back(std::move(r[i]));
                ++i;
            } else {
                Term term{basis[j].col, Rational()};
                fmpq_mul(term.value.v, lc.v, basis[j].value.v);
                fmpq_neg(term.value.v, term.value.v);
                scratch_.push_back(std::move(term));
            }
            ++j;
        }
        r.swap(scratch_);
    }

    // Canonical representatives use only columns outside the Jacobian pivots.
    // They keep target products sparse on nonlinear coordinate changes.
    Row remainder(Row r) {
        Row kept;
        // Terms before i have been kept or dropped. Reduction only adds
        // terms after the pivot column, which are not yet irreducible.
        for (size_t i = 0; i < r.size();) {
            const Row& basis = rows_[r[i].col];
            if (basis.empty()) {
                kept.push_back(std::move(r[i++]));
            } else if (basis.size() == 1) {
                ++i;
            } else {
                reduce_pivot(r,i,basis);
                i = 0;
            }
        }
        return kept;
    }

    void insert(Row r) {
        while (!r.empty()) {
            const Index pivot = r.front().col;
            const Row& basis = rows_[pivot].empty() && base_ ? base_->rows_[pivot] : rows_[pivot];
            if (basis.empty()) {
                const Rational lc = r.front().value;
                if (!fmpq_is_one(lc.v)) for (auto& term : r) fmpq_div(term.value.v, term.value.v, lc.v);
                budget_.use(r.size());
                rows_[pivot] = std::move(r);
                pivots.push_back(pivot);
                return;
            }
            reduce_pivot(r,0,basis);
        }
    }
};

struct Jet {
    const Index p;
    Monomials mons;
    Budget& budget;
    Echelon jacobian;
    std::vector<Row> images;

    Jet(const Basis& f, Index n, Index degree, Budget& work)
        : p(f.size()), mons(n,degree,p), budget(work),
          jacobian(mons.degree.size()*p,work), images(p) {
        std::vector<Row> derivatives(n);
        for (Index j = 0; j < p; ++j) {
            for (const auto& [powers,c] : f[j]) {
                const auto order = total_degree(powers);
                if (order > degree + 1) continue;
                if (order <= degree) images[j].push_back({mons.id(powers),Rational(c)});
                Powers lowered = powers;
                for (Index i = 0; i < n; ++i) if (powers[i] > 0) {
                    --lowered[i];
                    derivatives[i].push_back({mons.id(lowered)*p+j,Rational(c*powers[i])});
                    ++lowered[i];
                }
            }
        }
        // Distinct exponents give distinct columns in each row.
        for (auto& row : images) sort_columns(row);
        for (auto& row : derivatives) {
            sort_columns(row);
            jacobian.insert(std::move(row));
        }
        source_closure(jacobian);
    }

    void source_closure(Echelon& space) {
        for (size_t i = 0; i < space.pivots.size(); ++i) {
            const Row& row = space.row(space.pivots[i]);
            if (mons.degree[row.front().col/p] == mons.cutoff) continue;
            for (Index x = 0; x < mons.width; ++x) {
                Row shifted;
                for (const auto& [col,c] : row) {
                    const Index m = col/p;
                    if (mons.degree[m] == mons.cutoff) break;
                    shifted.push_back({mons.shift(m,x)*p + col%p,c});
                }
                // Multiplication by x is injective on monomials.
                sort_columns(shifted);
                space.insert(std::move(shifted));
            }
        }
    }

    Index first_full_power(const Echelon& space) const {
        // All columns of degrees >= k are pivots, hence their span belongs
        // to this jet subspace by descending triangular substitution.
        for (Index i = p*mons.degree.size(); i > 0; --i) {
            if (!space.occupied(i-1)) return mons.degree[(i-1)/p]+1;
        }
        return 0;
    }

    Index contact_order() {
        Echelon contact(p*mons.degree.size(),budget,&jacobian);
        for (const Row& f : images) for (Index j = 0; j < p; ++j) {
            Row row;
            for (const auto& [m,c] : f) row.push_back({m*p+j,c});
            contact.insert(std::move(row));
        }
        source_closure(contact);
        return first_full_power(contact);
    }

    size_t tangent(Index& first_full) {
        Echelon tangent(p*mons.degree.size(),budget,&jacobian);
        for (Index j = 0; j < p; ++j) {
            Row unit;
            unit.push_back({j,Rational()});
            fmpq_one(unit.back().value.v);
            tangent.insert(jacobian.remainder(std::move(unit)));
        }
        // Rather than enumerate all target monomials, multiply only independent
        // rows by f_1,...,f_p. Any discarded combination has its products in
        // the same closure, and df(R^n) is closed under these multiplications.
        for (size_t i = 0; i < tangent.pivots.size(); ++i) {
            const Row& row = tangent.row(tangent.pivots[i]);
            for (const Row& f : images) {
                Row terms;
                for (const auto& [b,coeff] : f) {
                    if (mons.degree[b]+mons.degree[row.front().col/p] > mons.cutoff) break;
                    for (const auto& [col,c] : row) {
                        if (mons.degree[b]+mons.degree[col/p] > mons.cutoff) break;
                        budget.use(1);
                        terms.push_back({mons.product(col/p,b)*p+col%p,Rational()});
                        fmpq_mul(terms.back().value.v,c.v,coeff.v);
                    }
                }
                tangent.insert(jacobian.remainder(collect(std::move(terms))));
            }
        }
        first_full = first_full_power(tangent);
        return p*mons.degree.size() - jacobian.pivots.size() - tangent.pivots.size();
    }
};

Polynomial determinant(const std::vector<Basis>& rows, size_t i,
                       const std::vector<size_t>& columns) {
    if (columns.size() == 1) return rows[i][columns[0]];
    Polynomial result;
    for (size_t j = 0; j < columns.size(); ++j) {
        const auto& entry = rows[i][columns[j]];
        if (entry.empty()) continue;
        auto sub = columns;
        sub.erase(sub.begin()+j);
        auto minor = determinant(rows,i+1,sub);
        auto term = primary::multiply(entry,minor);
        for (const auto& [e,c] : term) {
            auto& coefficient = result[e];
            if (j%2) coefficient -= c; else coefficient += c;
            if (!coefficient) result.erase(e);
        }
    }
    return result;
}

// The contact cokernel has support V(f, I_p(df)). A positive-dimensional
// support proves infinite extended codimension: modulo (f), target vector
// fields have only their p constant values. Avoid computing large minors in
// this optional test; the general jet certificate remains available.
bool positive_contact_dimension(const Basis& f, const std::vector<std::string>& names) {
    const size_t n = names.size(), p = f.size();
    if (n > 6 || (p <= n && p > 3)) return false;
    for (const auto& g : f)
        if (g.size() > 12 || max_total_degree(g) > 24) return false;
    Basis ideal = f;
    if (p <= n) {
        std::vector<Basis> jac(p,Basis(n));
        for (size_t i = 0; i < p; ++i) for (const auto& [e,c] : f[i]) {
            Powers derivative = e;
            for (size_t j = 0; j < n; ++j) if (e[j] != 0) {
                --derivative[j];
                jac[i][j].emplace(derivative,c*e[j]);
                ++derivative[j];
            }
        }
        std::vector<size_t> cols(p);
        for (size_t j=0;j<p;++j) cols[j]=j;
        while (true) {
            auto minor = determinant(jac,0,cols);
            if (!minor.empty()) ideal.push_back(std::move(minor));
            size_t j=p;
            while (j && cols[j-1] == n-p+j-1) --j;
            if (!j) break;
            ++cols[j-1];
            for (size_t h=j;h<p;++h) cols[h]=cols[h-1]+1;
        }
    }
    return dimension_at_point(ideal,names,std::vector<mpq_class>(n),false,false).dimension > 0;
}

Json::Value answer(bool finite, size_t value, const char* method) {
    Json::Value result(Json::objectValue);
    result["finite"] = finite;
    result["value"] = finite ? Json::Value(std::to_string(value)) : Json::Value();
    result["method"] = method;
    return result;
}

} // namespace

Json::Value ae_codimension(const Basis& f, const std::vector<std::string>& names) {
    if (f.empty() || names.empty())
        throw std::invalid_argument("Provide at least one source coordinate and one map component.");
    const Powers origin(names.size());
    for (const auto& component : f)
        if (component.count(origin)) throw std::invalid_argument("Each map component must vanish at the origin.");
    if (auto special = ae_special(f,names)) return *special;

    Budget budget, probe_budget;
    probe_budget.steps /= 2;
    std::optional<Index> contact;
    bool other_methods_tried = false;
    size_t lower_bound = 0;
    Index last_degree = 0;
    const auto alternate = [&]() -> std::optional<Json::Value> {
        other_methods_tried = true;
        if (auto graded = ae_graded(f,names,contact)) return graded;
        if (!contact && positive_contact_dimension(f,names))
            return answer(false,0,"positive_dimensional_contact_support");
        return std::nullopt;
    };

    Index checkpoint = 1;
    unsigned probes_left = 0;
    for (Index degree = 1;;) {
        const bool at_checkpoint = degree == checkpoint;
        Index d;
        try {
            Jet jet(f,names.size(),degree,at_checkpoint ? budget : probe_budget);
            if (!contact) {
                const Index k = jet.contact_order();
                if (k <= degree) contact = k;
            }
            lower_bound = jet.tangent(d);
            last_degree = degree;
        } catch (const JetLimit&) {
            if (at_checkpoint) break;
            // A failed optional probe cannot spend the ordinary checkpoints'
            // work allowance. Disable further probes and resume that sequence.
            probe_budget.steps = 0;
            degree = checkpoint;
            continue;
        }
        // Write M = R^p/df R^n and N = M/(f*B)^p, B = QQ[[y]].
        // Nakayama on m^k M proves m^k M ⊂ (f) M from the contact jets.
        // M is then finite over B by completeness. The k full A layers imply
        // image(m^d M in N) = image(m^(d+k) M in N) ⊂ (y) image(m^d M).
        // Nakayama over B kills this image, proving the exact colength.
        if (contact && d + *contact <= degree + 1) {
            auto result = answer(true,lower_bound,"extended_tangent_jets");
            result["jet_degree"] = degree;
            result["contact_order"] = *contact;
            return result;
        }
        if (degree >= 4 && !other_methods_tried)
            if (auto result = alternate()) return *result;

        // For wide graphs a doubled jet can be much larger than the first
        // possible certificate at d+k-1. Probe that cutoff, and allow one
        // refinement if a full tail appears. Keep the doubling checkpoints
        // so a gap in a later layer cannot send the next trial past them.
        if (at_checkpoint) {
            checkpoint *= 2;
            probes_left = 2;
        }
        const bool wide_graph = names.size() + 2*f.size() >= 10;
        if (wide_graph && contact && degree >= 8 && probes_left && probe_budget.steps &&
            (at_checkpoint || d <= degree)) {
            degree = std::min(checkpoint, d + *contact - 1);
            --probes_left;
        } else {
            degree = checkpoint;
        }
    }
    if (!other_methods_tried)
        if (auto result = alternate()) return *result;
    Json::Value result(Json::objectValue);
    result["finite"] = Json::nullValue;
    result["value"] = Json::nullValue;
    result["method"] = "extended_tangent_jets";
    result["reason"] = "The exact jet work limit was reached before a finite or infinite codimension was certified.";
    result["lower_bound"] = Json::UInt64(lower_bound);
    result["jet_degree"] = last_degree;
    return result;
}
}
