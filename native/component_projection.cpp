// Eliminate graph coordinates over K[parameters] before contracting the shape
// from its generic fibre. A marked K is Q[theta]/m, with theta always last.
#include "component_projection.hpp"
#include "local_io.hpp"
#include "polynomial_factors.hpp"
#include "primary_decomposition.hpp"
#include "quotient_ring.hpp"

#include <algorithm>
#include <numeric>
#include <set>

namespace atlas::components {
namespace {
using local::Basis;
using local::Polynomial;
using local::Powers;
using Axes = std::vector<size_t>;
using J = Json::Value;

struct Ring {
    size_t axes, width;
    Polynomial relation;
    explicit Ring(const J& q) : axes(q["width"].asUInt()), width(axes + !q["relation"].isNull()) {
        if (!axes) throw std::invalid_argument("Expected a nonempty projection ring.");
        if (width != axes) {
            relation = local::decode_row(q["relation"].asString(), width);
            if (relation.empty()) throw std::invalid_argument("Missing marked-field relation.");
        }
    }
    Basis rows(const J& q) const {
        if (!q.isArray()) throw std::invalid_argument("Expected a list of equations.");
        Basis answer;
        for (const auto& row : q) answer.push_back(local::decode_row(row.asString(), width));
        return answer;
    }
};

Polynomial one(size_t width) { return {{Powers(width), 1}}; }

Polynomial permute(const Polynomial& f, const Axes& axes) {
    Polynomial result;
    for (const auto& [e,c] : f) {
        Powers p;
        p.reserve(axes.size());
        for (auto j : axes) p.push_back(e[j]);
        result.emplace(std::move(p),c);
    }
    return result;
}

Basis lex_basis(const Ring& ring, const Basis& generators, const Axes& order) {
    Basis rows;
    for (const auto& f : generators) rows.push_back(permute(f,order));
    if (!ring.relation.empty()) rows.push_back(permute(ring.relation,order));
    auto g = primary::reduced(rows,ring.width,true);
    Axes inverse(order.size());
    for (size_t j=0; j<order.size(); ++j) inverse[order[j]]=j;
    for (auto& f : g) f=permute(f,inverse);
    if (!ring.relation.empty()) g.erase(std::remove(g.begin(),g.end(),ring.relation),g.end());
    return g;
}

bool uses(const Polynomial& f, size_t first, size_t last) {
    return std::any_of(f.begin(),f.end(),[&](const auto& term) {
        const auto& e=term.first;
        return std::any_of(e.begin()+first,e.begin()+last,[](const auto& k){return k!=0;});
    });
}

// Write f = a*x+b with no x in a or b. Reject higher powers of x.
std::optional<std::pair<Polynomial,Polynomial>> linear(const Polynomial& f, size_t x) {
    Polynomial a,b;
    for (const auto& [e,c] : f) {
        if (!e[x]) b.emplace(e,c);
        else if (e[x]==1) {
            auto p=e;
            p[x]=0;
            a.emplace(std::move(p),c);
        } else return {};
    }
    if (a.empty()) return {};
    return std::make_pair(std::move(a),std::move(b));
}

J action(const char* name) {
    J answer;
    answer["action"]=name;
    return answer;
}
} // namespace

J decomposition_step(const J& q) {
    Ring ring(q);
    const auto generators=ring.rows(q["generators"]);
    const mpz_class dimension(q["dimension"].asString(),10);
    if (generators == Basis{one(ring.width)}) return action("empty");
    if (dimension<0 || dimension>ring.axes) return action("undecided");
    if (generators.empty()) return action(dimension.get_ui()==ring.axes ? "ambient" : "undecided");
    if (dimension.get_ui()==ring.axes) return action("undecided");
    // Unique graph pivots give a polynomial algebra over K, of the
    // expected dimension, without recursively substituting each row.
    if (generators.size()==ring.axes-dimension.get_ui()
            && quotient_ring::independent_graph(generators,ring.axes)) return action("prime");

    for (size_t i=0; i<generators.size(); ++i) {
        for (size_t j=0; j<ring.axes; ++j) {
            auto pair=linear(generators[i],j);
            if (!pair || uses(pair->first,0,ring.axes)) continue;
            if (ring.axes==1) return action("prime");
            // A nonzero leading coefficient in the marked field is a unit.
            // Solve the graph equation in that field, retaining the caller's
            // coordinates for the other variables.
            Polynomial value;
            if (pair->first.size()==1 && pair->first.begin()->first==Powers(ring.width)) {
                value=std::move(pair->second);
                for (auto& [e,c] : value) c /= -pair->first.begin()->second;
            } else {
                quotient_ring::Ring coefficients({ring.relation},ring.width,true);
                for (auto& [e,c] : pair->second) c=-c;
                value=std::get<Polynomial>(coefficients.evaluate("divide",{pair->second,pair->first}));
            }
            J answer=action("graph");
            answer["row"]=Json::UInt64(i);
            answer["axis"]=Json::UInt64(j);
            answer["value"]=local::encode_row(value);
            return answer;
        }
    }
    if (generators.size()==1)
        return action(dimension.get_ui()+1==ring.axes ? "hypersurface" : "undecided");
    return action("project");
}

J eligible_factors(const J& q) {
    Ring ring(q);
    const auto factors=ring.rows(q["factors"]);
    if (factors.size()!=q["powers"].size()) throw std::invalid_argument("Incomplete factor multiplicities.");
    const bool generic=!q["axis"].isNull();
    const size_t axis=generic ? q["axis"].asUInt() : 0;
    if (axis>=ring.axes) throw std::invalid_argument("Invalid separating coordinate.");
    J indices(Json::arrayValue);
    for (size_t i=0; i<factors.size(); ++i) {
        if (generic && !uses(factors[i],axis,axis+1)) continue;
        const mpz_class m(q["powers"][Json::ArrayIndex(i)].asString(),10);
        if (m<1) throw std::invalid_argument("Invalid factor multiplicity.");
        if (m!=1 && !q["reduced"].asBool()) return {};
        indices.append(Json::UInt64(i));
    }
    return indices;
}

J projection_shape(const J& q) {
    Ring ring(q);
    const size_t d=q["dimension"].asUInt();
    if (d>=ring.axes) throw std::invalid_argument("Invalid projection dimension.");
    const size_t t=ring.axes-1;
    const auto ideal=ring.rows(q["generators"]);
    Axes order;
    for (size_t j=d; j<=t; ++j) order.push_back(j);
    for (size_t j=0; j<d; ++j) order.push_back(j);
    if (!ring.relation.empty()) order.push_back(ring.axes);
    const auto basis=lex_basis(ring,ideal,order);
    Basis relations;
    for (const auto& g : basis) if (!uses(g,d,t)) relations.push_back(g);
    if (relations.empty()) return {};
    Polynomial eliminant=relations.front();
    if (relations.size()>1) {
        if (ring.relation.empty()) {
            auto common=local::polynomial_gcd(relations,ring.width);
            if (!common) throw std::runtime_error("Unable to compute the projection eliminant gcd.");
            eliminant=std::move(*common);
        } else {
            // Take the gcd in K[u,t]. Over the generic field K(u), pure
            // parameter factors become units and are discarded by eligible_factors.
            eliminant=marked_gcd(relations,ring.relation,ring.width);
        }
    }
    if (!uses(eliminant,t,t+1)) return {};
    J graph(Json::arrayValue);
    std::set<Polynomial> denominators;
    for (size_t x=d; x<t; ++x) {
        auto equation=[&](const Basis& rows) {
            for (const auto& g : rows) {
                if (uses(g,d,x) || uses(g,x+1,t)) continue;
                const auto pair=linear(g,x);
                if (!pair || uses(pair->first,d,ring.axes)) continue;
                graph.append(local::encode_row(g));
                denominators.insert(pair->first);
                return true;
            }
            return false;
        };
        if (equation(basis)) continue;
        if (x+1==t) return {};
        // Move x last among the graph variables to eliminate the others
        // first. A relation d(u)*x-g(u,t) must occur in this projected basis.
        auto reordered=order;
        std::rotate(reordered.begin()+x-d,reordered.begin()+x-d+1,reordered.begin()+t-d);
        if (!equation(lex_basis(ring,ideal,reordered))) return {};
    }
    auto denominator=one(ring.width);
    for (const auto& h : denominators) denominator=primary::multiply(denominator,h);
    J answer;
    answer["graph"]=std::move(graph);
    answer["denominator"]=local::encode_row(denominator);
    answer["eliminant"]=local::encode_row(eliminant);
    return answer;
}
} // namespace atlas::components
