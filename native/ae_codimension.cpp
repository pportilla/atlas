#include "ae_codimension.hpp"
#include "ae_graded.hpp"
#include "ae_special.hpp"
#include "local_dimension.hpp"
#include "local_monomial.hpp"
#include "primary_decomposition.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace atlas::local {
namespace {

// A p-vector of source jets in total-degree order, with position m*p+j.
using Index = unsigned;
using Row = std::map<Index, mpq_class>;
using Exponent = std::vector<unsigned>;
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

void accumulate(Row& row, Index index, const mpq_class& value) {
    auto [it, inserted] = row.try_emplace(index, value);
    if (!inserted) {
        it->second += value;
        if (it->second == 0) row.erase(it);
    }
}

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
public:
    std::vector<Index> pivots;
    Echelon(size_t width, Budget& budget, const Echelon* base = nullptr)
        : rows_(width), base_(base), budget_(budget) {}

    const Row& row(Index pivot) const { return rows_[pivot]; }
    bool occupied(Index i) const {
        return !rows_[i].empty() || (base_ && !base_->rows_[i].empty());
    }

    // Remove a coefficient at a normalized pivot. Coordinate rows need no
    // rational copy or multiplication.
    void reduce_pivot(Row& r, Row::iterator pivot, const Row& basis) {
        if (basis.size() == 1) {
            r.erase(pivot);
            return;
        }
        const auto lc = std::move(pivot->second);
        r.erase(pivot);
        budget_.use(basis.size());
        for (auto t = std::next(basis.begin()); t != basis.end(); ++t)
            accumulate(r,t->first,-lc*t->second);
    }

    // Canonical representatives use only columns outside the Jacobian pivots.
    // They keep target products sparse on nonlinear coordinate changes.
    Row remainder(Row r) {
        for (auto it = r.begin(); it != r.end();) {
            const Index col = it->first;
            const Row& basis = rows_[col];
            if (basis.empty()) {
                ++it;
            } else if (basis.size() == 1) {
                it = r.erase(it);
            } else {
                // Reduction may have introduced a term before the former
                // successor. Terms preceding col are already irreducible.
                reduce_pivot(r,it,basis);
                it = r.upper_bound(col);
            }
        }
        return r;
    }

    void insert(Row r) {
        while (!r.empty()) {
            const Index pivot = r.begin()->first;
            const Row& basis = rows_[pivot].empty() && base_ ? base_->rows_[pivot] : rows_[pivot];
            if (basis.empty()) {
                const auto lc = r.begin()->second;
                if (lc != 1) for (auto& term : r) term.second /= lc;
                budget_.use(r.size());
                rows_[pivot] = std::move(r);
                pivots.push_back(pivot);
                return;
            }
            reduce_pivot(r,r.begin(),basis);
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
                if (order <= degree) images[j].emplace(mons.id(powers),c);
                Powers lowered = powers;
                for (Index i = 0; i < n; ++i) if (powers[i] > 0) {
                    --lowered[i];
                    derivatives[i].emplace(mons.id(lowered)*p+j,c*powers[i]);
                    ++lowered[i];
                }
            }
        }
        for (auto& row : derivatives) jacobian.insert(std::move(row));
        source_closure(jacobian);
    }

    void source_closure(Echelon& space) {
        for (size_t i = 0; i < space.pivots.size(); ++i) {
            const Row& row = space.row(space.pivots[i]);
            if (mons.degree[row.begin()->first/p] == mons.cutoff) continue;
            for (Index x = 0; x < mons.width; ++x) {
                Row shifted;
                for (const auto& [col,c] : row) {
                    const Index m = col/p;
                    if (mons.degree[m] == mons.cutoff) break;
                    shifted.emplace(mons.shift(m,x)*p + col%p,c);
                }
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
            for (const auto& [m,c] : f) row.emplace(m*p+j,c);
            contact.insert(std::move(row));
        }
        source_closure(contact);
        return first_full_power(contact);
    }

    size_t tangent(Index& first_full) {
        Echelon tangent(p*mons.degree.size(),budget,&jacobian);
        for (Index j = 0; j < p; ++j) tangent.insert(jacobian.remainder({{j,1}}));
        // Rather than enumerate all target monomials, multiply only independent
        // rows by f_1,...,f_p. Any discarded combination has its products in
        // the same closure, and df(R^n) is closed under these multiplications.
        for (size_t i = 0; i < tangent.pivots.size(); ++i) {
            const Row& row = tangent.row(tangent.pivots[i]);
            for (const Row& f : images) {
                Row product;
                for (const auto& [b,coeff] : f) {
                    if (mons.degree[b]+mons.degree[row.begin()->first/p] > mons.cutoff) break;
                    for (const auto& [col,c] : row) {
                        if (mons.degree[b]+mons.degree[col/p] > mons.cutoff) break;
                        budget.use(1);
                        accumulate(product,mons.product(col/p,b)*p+col%p,c*coeff);
                    }
                }
                tangent.insert(jacobian.remainder(std::move(product)));
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
