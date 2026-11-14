#include "local_point_quotient.hpp"
#include "local_quotient_initial.hpp"
#include "local_io.hpp"
#include <algorithm>
#include <stdexcept>

namespace atlas::local {
Basis partial_derivatives(const Polynomial& f, size_t width) {
    Basis partials(width);
    for (const auto& [e,c] : f) {
        for (size_t i=0; i<width; ++i) if (e[i] != 0) {
            auto powers = e;
            --powers[i];
            partials[i].emplace(std::move(powers),c*e[i]);
        }
    }
    return partials;
}

namespace {
bool at_origin(const PointRow& point) {
    return std::all_of(point.begin(),point.end(),[](const auto& p) { return p == 0; });
}

PointQuotient from_ideal(const Basis& ideal, const std::vector<std::string>& names,
                        const PointRow& point, bool lex, bool prefer_modular, bool listing) {
    const size_t width = names.size();
    const bool origin = at_origin(point);
    const Powers zero(width);
    // The simplification acts in the local ring. For example a monomial
    // in moving axes is a unit, and a full-rank Jacobian generates m_p.
    Basis rows;
    if (origin) {
        for (const auto& f : ideal) if (!f.empty()) {
            if (f.count(zero)) return {0,{},{}};
            rows.push_back(f);
        }
    } else {
        rows = simplify_point_generators(ideal,point);
        if (rows.size() == 1 && rows.front().size() == 1 && rows.front().count(zero)) return {0,{},{}};
    }
    // The height theorem precludes finite colength for fewer than n
    // generators of a proper ideal in the regular local ring of dimension n.
    if (rows.size() < width) return {};
    if (std::all_of(rows.begin(),rows.end(),[](const auto& row) { return row.size()==1; })) {
        std::vector<Powers> heads;
        for (const auto& f : rows) heads.push_back(f.begin()->first);
        return monomial_point_quotient(width,heads,listing);
    }
    if (width == 1) {
        std::optional<mpz_class> least;
        for (const auto& f : rows) {
            const auto order = origin ? f.begin()->first.front() : order_at_point(f,point);
            if (!least || order < *least) least = order;
        }
        return monomial_point_quotient(width,{Powers{*least}},listing);
    }
    bool sparse = !origin && use_sparse_point_basis(rows,point);
    if (sparse)
        if (auto heads = regular_point_heads(rows,names,point,lex,prefer_modular))
            return monomial_point_quotient(width,*heads,listing);
    if (!sparse && !origin) rows = translate_unbounded(rows,point);
    const auto basis = sparse ? standard_basis_at_point(rows,names,point,lex,prefer_modular)
                              : standard_basis(rows,names,lex,prefer_modular);
    std::vector<Powers> heads;
    heads.reserve(basis.size());
    for (const auto& f : basis) if (!f.empty())
        heads.push_back(sparse ? head_at_point(f,point,lex) : local_head(f,lex));
    return monomial_point_quotient(width,heads,listing);
}
}

PointQuotient quotient_at_point(const std::string& operation, const Basis& inputs,
        const std::vector<std::string>& names, const PointRow& point, bool lex, bool prefer_modular) {
    const size_t width = names.size();
    if (point.size()!=width) throw std::invalid_argument("Incorrect local point dimension.");
    if (operation=="local-quotient-basis") return from_ideal(inputs,names,point,lex,prefer_modular,true);
    const bool jacobian = operation=="local-jacobian-algebra";
    const bool tjurina = operation=="tjurina-number";
    if (!jacobian && !tjurina && operation!="milnor-number")
        throw std::invalid_argument("Unknown local quotient operation.");
    if (!width) throw std::invalid_argument("A hypersurface germ requires at least one coordinate.");
    if (inputs.size()!=1) throw std::invalid_argument("A hypersurface germ requires exactly one polynomial.");
    const auto& f = inputs.front();
    const auto order = hypersurface_jet_order(f,point);
    if (order == 0) throw std::invalid_argument(at_origin(point) ? "The hypersurface must pass through the origin."
                                                                 : "The hypersurface must pass through the supplied point.");
    if (order == 1) return {0,{},jacobian ? partial_derivatives(f,width) : Basis{}};
    if (f.empty()) return {std::nullopt,{},jacobian ? Basis(width) : Basis{}};
    if (!jacobian && width == 1) {
        // In the DVR, f=u*t^k and df/dt has order k-1 in characteristic
        // zero. The additional Tjurina generator leaves this length intact.
        return {order_at_point(f,point)-1,{},{}};
    }
    auto generators = partial_derivatives(f,width);
    if (tjurina) generators.push_back(f);
    auto result = from_ideal(generators,names,point,lex,prefer_modular,jacobian);
    if (jacobian) result.derivatives = std::move(generators);
    return result;
}
}
