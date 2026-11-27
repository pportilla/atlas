// One-sided proofs for the scheduler of the canonical embedded resolution.
// All fixed points and orbit cuts used below lie over the closed origin.
#include "rational_sparse_json.hpp"
#include "groebner.hpp"
#include "membership_certificate.hpp"
#include "polynomial_factors.hpp"
#include <flint/fmpq_poly.h>
#include "sparse_flint.hpp"
#include <array>
#include <limits>

namespace {
using namespace atlas::sparse_json;
using atlas::local::SparseLexRing;
using FanChart = std::array<Powers, 3>;

struct Univariate {
    fmpq_poly_t p;
    Univariate() { fmpq_poly_init(p); }
    ~Univariate() { fmpq_poly_clear(p); }
    Univariate(const Univariate&)=delete;
    Univariate& operator=(const Univariate&)=delete;
    void set(const Terms& terms) {
        fmpq_t q;
        fmpq_init(q);
        for (const auto& [e,c] : terms) {
            fmpq_set_mpq(q,c.get_mpq_t());
            fmpq_poly_set_coeff_fmpq(p,e[0].get_si(),q);
        }
        fmpq_clear(q);
    }
};

// On a one-dimensional torus orbit, the residual divisor may be smooth
// but tangent to the coordinate boundary. For intrinsic smoothness we
// require a common zero of the equation and all its first derivatives.
// For normal crossings a common geometric zero of the restriction
// and all derivatives tangent to the boundary, with some nonzero normal
// derivative, proves failure of normal crossings.
bool bad_orbit(const Terms& residual, const std::array<bool,3>& boundary, unsigned free, bool intrinsic) {
    std::array<Terms,4> rows;
    for (const auto& [e,c] : residual) {
        for (unsigned k=0; k<4; ++k) {
            if (k<3 && e[k]==0) continue;
            if (e[(free+1)%3] != (k==(free+1)%3) || e[(free+2)%3] != (k==(free+2)%3)) continue;
            const Z degree=e[free]-(k==free);
            if (degree>64) return false;
            auto& value=rows[k][{degree}];
            value+=k<3 ? Q(e[k]*c) : c;
        }
    }
    Univariate common, derivative, squarefree, remainder;
    std::array<Univariate,4> polynomials;
    for (unsigned k=0; k<4; ++k) {
        polynomials[k].set(rows[k]);
        if (intrinsic || !(k<3 && k!=free && boundary[k])) {
            fmpq_poly_gcd(common.p,common.p,polynomials[k].p);
            if (fmpq_poly_degree(common.p)==0) return false;
        }
    }
    // If every tangential equation vanishes identically, any nonzero
    // normal derivative detects a nonempty open subset of the orbit.
    // Otherwise use the squarefree gcd, discarding the excluded root 0.
    if (!fmpq_poly_is_zero(common.p)) {
        fmpq_poly_derivative(derivative.p,common.p);
        fmpq_poly_gcd(derivative.p,derivative.p,common.p);
        fmpq_poly_div(squarefree.p,common.p,derivative.p);
        if (fmpz_is_zero(squarefree.p->coeffs))
            fmpq_poly_shift_right(squarefree.p,squarefree.p,1);
        if (fmpq_poly_degree(squarefree.p)==0) return false;
    }
    // A nonzero common root of all four equations is a singular point of
    // the strict transform itself, irrespective of the toric boundary.
    if (intrinsic) return true;
    for (unsigned k=0;k<3;++k) {
        if (k==free || !boundary[k] || fmpq_poly_is_zero(polynomials[k].p)) continue;
        if (fmpq_poly_is_zero(common.p)) return true;
        fmpq_poly_rem(remainder.p,polynomials[k].p,squarefree.p);
        if (!fmpq_poly_is_zero(remainder.p)) return true;
    }
    return false;
}

bool bad_chart(const Terms& f, const FanChart& rays, bool intrinsic=false) {
    Powers orders(3,-1);
    Terms pullback;
    for (const auto& [e,c] : f) {
        Powers v(3);
        for (unsigned i=0; i<3; ++i) {
            for (unsigned j=0;j<3;++j) v[i]+=rays[i][j]*e[j];
            if (orders[i]<0 || v[i]<orders[i]) orders[i]=v[i];
        }
        pullback.emplace(std::move(v),c);
    }
    // The given charts are unimodular. After removing the monomial
    // boundary, the residual is squarefree since the source is squarefree.
    std::array<bool,3> boundary;
    unsigned dim=3;
    for (unsigned j=0;j<3;++j) if ((boundary[j]=(orders[j]!=0))) --dim;
    unsigned order=4;
    std::array<Q,3> binary_quadratic{};
    const unsigned first=boundary[0] ? 1 : 0;
    Terms residual;
    for (const auto& [v,c] : pullback) {
        Powers e(3);
        Z degree=0;
        bool survives=true;
        for (unsigned j=0;j<3;++j) {
            degree+=(e[j]=v[j]-orders[j]);
            if (boundary[j] && e[j]!=0) survives=false;
        }
        if ((intrinsic || survives) && degree<=3) {
            order=std::min(order,static_cast<unsigned>(degree.get_ui()));
            if (!intrinsic && dim==2 && degree==2) binary_quadratic[e[first].get_ui()]+=c;
        }
        residual.emplace(std::move(e),c);
    }
    // At an SNC point the residual branches restrict to independent
    // smooth parameters on the boundary intersection of dimension dim.
    const unsigned highest_possible_order=intrinsic ? 1 : dim;
    if (order > highest_possible_order) return true;
    if (!intrinsic && dim==2 && order==2) {
        // Two transverse residual branches meeting the boundary have
        // distinct tangent lines on it, hence nonzero binary discriminant.
        const auto& q=binary_quadratic;
        if (q[1]*q[1]==4*q[0]*q[2]) return true;
    }
    for (unsigned free=0;free<3;++free) {
        const auto a=(free+1)%3, b=(free+2)%3;
        if (!intrinsic && !(boundary[a] || boundary[b])) continue;
        bool over_origin=true;
        for (unsigned j=0;j<3;++j) if (!rays[a][j] && !rays[b][j]) over_origin=false;
        if (over_origin && bad_orbit(residual,boundary,free,intrinsic)) return true;
    }
    return false;
}

// At the generic point of the coordinate line the normal space is a
// plane over Q(t). At order two an SNC equation must have a binary
// quadratic initial form with two distinct lines over its algebraic closure.
bool non_snc_axis(const Terms& f, unsigned axis) {
    const unsigned a=(axis+1)%3, b=(axis+2)%3;
    std::array<Terms,3> quadratic;
    for (const auto& [e,c] : f) {
        Z degree=e[a]+e[b];
        if (degree<2) return false;
        if (degree==2) quadratic[e[a].get_ui()].emplace(Powers{e[axis]},c);
    }
    std::array<Univariate,3> q;
    for (unsigned k=0;k<3;++k) q[k].set(quadratic[k]);
    Univariate lhs, rhs;
    fmpq_poly_mul(lhs.p,q[1].p,q[1].p);
    fmpq_poly_mul(rhs.p,q[0].p,q[2].p);
    fmpq_poly_scalar_mul_si(rhs.p,rhs.p,4);
    // This includes the case of multiplicity >2, when all three vanish.
    return fmpq_poly_equal(lhs.p,rhs.p);
}

// Restrict the Newton normal fan to the boundary w_free=0 by projecting
// the exponents onto the other two coordinates. Its polygon plus the
// orthant has a compact edge precisely when no term attains both minima.
// The edge normal gives an original positive ray in that boundary face.
// On the image line, order <2 means that the reduced hypersurface does
// not contain the whole line in its singular locus. This excludes the
// source fan for every regular refinement, even before it is constructed.
bool inadmissible_boundary_ray(const Terms& f) {
    for (unsigned free=0;free<3;++free) {
        const unsigned a=(free+1)%3, b=(free+2)%3;
        unsigned min_a=17, min_b=17, order=33;
        for (const auto& [e,c] : f) {
            const unsigned x=e[a].get_ui(), y=e[b].get_ui();
            min_a=std::min(min_a,x);
            min_b=std::min(min_b,y);
            order=std::min(order,x+y);
        }
        if (order<2 && order!=min_a+min_b) return true;
    }
    return false;
}

Json::Value ordinary_prefix(const Json::Value& rows) {
    if (!atlas::certificate::enabled() || rows.empty() || rows.size()>24) return {};
    const auto f=read_terms(rows,3);
    if (f.empty()) return {};
    for (const auto& [e,c] : f)
        if (!c || std::any_of(e.begin(),e.end(),[](const Z& k){return k>16;})) return {};
    if (!inadmissible_boundary_ray(f)) return {};
    // A non-SNC generic transverse plane section precludes resolution by
    // identity, blowups supported over the origin, and blowups of other
    // coordinate lines: the proper image of the closed non-SNC locus
    // would still contain the origin.
    unsigned axis=0;
    for (;axis<3;++axis) if (non_snc_axis(f,axis)) break;
    if (axis==3) return {};
    SparseLexRing ring(3);
    atlas::PolyValue p(ring.ctx),sf(ring.ctx);
    ring.write(p.p,f);
    atlas::rational_squarefree_part(sf.p,p.p,ring.ctx);
    const auto reduced=ring.read(sf.p);
    // Removing a repeated nonconstant factor strictly decreases the degree
    // in at least one variable. Require the same monomials for this prefix.
    if (f.size()!=reduced.size() || !std::equal(f.begin(),f.end(),reduced.begin(),
        [](const auto& a,const auto& b){return a.first==b.first;})) return {};
    // Also exclude the ordinary blowup of the obstructed line itself.
    FanChart basis{{{1,0,0},{0,1,0},{0,0,1}}};
    for (unsigned pivot : {(axis+1)%3,(axis+2)%3}) {
        auto chart=basis;
        chart[pivot][3-axis-pivot]=1;
        if (bad_chart(reduced,chart)) return term_list(reduced);
    }
    return {};
}

// Irreducible cones H(x,y) + z^2 K(x,y), with binary homogeneous forms
// of degrees d and d-2. A simple root of K disjoint from H gives -H/K
// an odd pole even over Qbar. Thus z^2+H/K is irreducible over Qbar(x,y).
// Coprimality descends that statement to the polynomial ring. The strict
// transform remains a single geometric component under birational blowup.
bool quadratic_cone_prefix(const Json::Value& rows) {
    if (!atlas::certificate::enabled() || rows.empty() || rows.size()>24) return false;
    const auto f=read_terms(rows,3);
    if (f.empty() || f.size()!=rows.size()) return false;
    const auto& first=f.begin()->first;
    const Z degree=first[0]+first[1]+first[2];
    if (degree<4 || degree>16) return false;
    for (const auto& [e,c] : f)
        if (!c || e[0]+e[1]+e[2]!=degree || mpz_sizeinbase(c.get_num_mpz_t(),2)>64 ||
            mpz_sizeinbase(c.get_den_mpz_t(),2)>64) return false;
    for (unsigned z=0;z<3;++z) {
        atlas::check_deadline();
        Terms h,k;
        const unsigned x=(z+1)%3;
        for (const auto& [e,c] : f) {
            if (e[z]!=0 && e[z]!=2) break;
            (e[z]==0 ? h : k).emplace(Powers{e[x]},c);
        }
        if (h.size()+k.size()!=f.size() || !h.count({0}) || !h.count({degree}) ||
            !k.count({degree-2})) continue;
        // y=1 loses no common factor at infinity since both highest
        // coefficients are present. H(0) also excludes the torus boundary.
        Univariate H,K,g,derivative;
        H.set(h); K.set(k);
        if (!fmpq_poly_is_squarefree(K.p)) continue;
        fmpq_poly_gcd(g.p,H.p,K.p);
        if (fmpq_poly_degree(g.p)!=0) continue;
        fmpq_poly_derivative(derivative.p,H.p);
        fmpq_poly_gcd(g.p,H.p,derivative.p);
        if (fmpq_poly_degree(g.p)<1) continue;

        // The punctured z-axis is singular (d-2>=2), excluding all
        // origin-supported modifications and blowups of either other axis.
        // Blowing up this axis gives the strict equation
        // t^2 H(1,s)+z^2 K(1,s), still singular along t=z=0.
        // The original Newton face with weights (1,1,2) in (x,y,z) is H.
        // Its repeated nonzero root obstructs torus transversality in any
        // refinement; the relative Newton policy only repairs point fans.
        return true;
    }
    return false;
}

// A repeated lowest-weight face at a torus point obstructs the original
// relative Newton construction. These positive integral vectors choose
// compact faces. A sign vector and the three logarithmic derivatives
// certify degeneracy over QQ without invoking a torus Groebner basis.
bool degenerate_compact_face(const Terms& f, unsigned axis) {
    unsigned minimum=std::numeric_limits<unsigned>::max();
    std::array<std::array<Q,4>,8> values{};
    for (const auto& [e,c] : f) {
        const unsigned weight=e[axis].get_ui()+2*(e[(axis+1)%3].get_ui()+e[(axis+2)%3].get_ui());
        if (weight>minimum) continue;
        if (weight<minimum) { minimum=weight; values={}; }
        for (unsigned signs=0;signs<8;++signs) {
            Q value=c;
            for (unsigned j=0;j<3;++j)
                if ((signs & (1u<<j)) && mpz_odd_p(e[j].get_mpz_t())) value=-value;
            values[signs][0]+=value;
            for (unsigned j=0;j<3;++j) values[signs][j+1]+=value*e[j];
        }
    }
    return std::any_of(values.begin(),values.end(),[](const auto& v){return v==std::array<Q,4>{};});
}

bool relative_axis_prefix(const Json::Value& rows,bool intrinsic) {
    if (!atlas::certificate::enabled() || rows.empty() || rows.size()>16) return false;
    const auto f=read_terms(rows,3);
    if (f.empty() || f.size()!=rows.size()) return false;
    for (const auto& [e,c] : f)
        if (std::any_of(e.begin(),e.end(),[](const Z& v){return v>8;}) ||
            mpz_sizeinbase(c.get_num_mpz_t(),2)>64 || mpz_sizeinbase(c.get_den_mpz_t(),2)>64) return false;
    // Handle the mixed transverse type: three singular coordinate lines,
    // with an ordinary node generically along two lines and a worse
    // section along the third. More general configurations retain their
    // previous resolution ordering, including existing point repairs.
    unsigned axis=3;
    for (unsigned k=0;k<3;++k) {
        for (const auto& [e,c] : f) if (e[(k+1)%3]+e[(k+2)%3]<2) return false;
        if (non_snc_axis(f,k)) {
            if (axis!=3) return false;
            axis=k;
        }
    }
    if (axis==3 || !degenerate_compact_face(f,axis)) return false;
    // Removing a repeated factor decreases a coordinate degree. Use
    // the original support only after checking squarefreeness.
    SparseLexRing ring(3);
    atlas::PolyValue p(ring.ctx),sf(ring.ctx);
    ring.write(p.p,f);
    atlas::rational_squarefree_part(sf.p,p.p,ring.ctx);
    const auto reduced=ring.read(sf.p);
    if (reduced.size()!=f.size() || !std::equal(f.begin(),f.end(),reduced.begin(),
        [](const auto& a,const auto& b){return a.first==b.first;})) return false;
    // An origin-supported modification and a blowup of any other
    // coordinate line leave the generic singularity unchanged. Check
    // the ordinary blowup of this axis on the origin fibre as well.
    FanChart basis{{{1,0,0},{0,1,0},{0,0,1}}};
    for (unsigned pivot : {(axis+1)%3,(axis+2)%3}) {
        auto chart=basis;
        chart[pivot][3-axis-pivot]=1;
        if (bad_chart(reduced,chart,intrinsic)) return true;
    }
    return false;
}

// A monomial support containing a pure power on every axis defines an
// origin-primary ideal. Its normal fan restricts to the original smooth
// fan on every proper orthant face. Pulling triangulation uses existing
// rays; every subsequent regularizing ray is a nonzero lattice point in
// a half-open fundamental parallelepiped. No such point can lie on a
// smooth coordinate face. Thus the trial cannot change a punctured axis.
Json::Value relative_supports(const Json::Value& supports) {
    Json::Value result(Json::arrayValue);
    for (const auto& support : supports) {
        std::array<bool,3> pure{};
        for (const auto& row : support) {
            const auto e=read_exponents(row,3);
            for (unsigned i=0;i<3;++i)
                if (e[i] && !e[(i+1)%3] && !e[(i+2)%3]) pure[i]=true;
        }
        if (!std::all_of(pure.begin(),pure.end(),[](bool value){return value;})) result.append(support);
    }
    return result;
}

// On the two-dimensional dense orbit of an interior ray, the strict
// equation is its Laurent initial form. The 3-torus maps smoothly and
// surjectively to that orbit with a one-dimensional torus as kernel.
// A repeated nonmonomial factor therefore gives a whole curve where
// the orbit equation and its first derivatives vanish. The Newton
// repair classifies the whole fan before it attempts any point patches.
Json::Value nonisolated_repair_orbit(const Json::Value& rows,Json::Value rays,bool ordinary) {
    if (rows.empty() || rows.size()>256 || rays.size()>128) return false;
    const auto f=read_terms(rows,3);
    for (const auto& [e,c] : f) {
        if (mpz_sizeinbase(c.get_num_mpz_t(),2)>128 || mpz_sizeinbase(c.get_den_mpz_t(),2)>128 ||
            std::any_of(e.begin(),e.end(),[](const Z& x){return x>32;})) return false;
    }
    if (ordinary) {
        // With pure powers of the same lowest degree on all three axes,
        // the Newton polyhedron is that simplex plus the orthant. Its
        // regular normal fan has exactly one exceptional ray.
        unsigned minimum=97;
        for (const auto& [e,c] : f)
            minimum=std::min(minimum,static_cast<unsigned>(e[0].get_ui()+e[1].get_ui()+e[2].get_ui()));
        if (!minimum || minimum==97) return {};
        for (unsigned j=0;j<3;++j) {
            Powers exponent(3);
            exponent[j]=minimum;
            if (!f.count(exponent)) return {};
        }
        Json::Value normal(Json::arrayValue);
        for (unsigned j=0;j<3;++j) normal.append(1);
        rays=Json::Value(Json::arrayValue);
        rays.append(std::move(normal));
    }
    SparseLexRing ring(3);
    for (const auto& row : rays) {
        atlas::check_deadline();
        const auto ray=read_exponents(row,3);
        if (std::any_of(ray.begin(),ray.end(),[](const Z& x){return !x || x>256;})) continue;
        Z minimum=-1;
        Terms face;
        for (const auto& [e,c] : f) {
            const Z weight=ray[0]*e[0]+ray[1]*e[1]+ray[2]*e[2];
            if (minimum>=0 && weight>minimum) continue;
            if (minimum<0 || weight<minimum) { minimum=weight; face.clear(); }
            face.emplace(e,c);
        }
        if (face.size()<3) continue;
        atlas::PolyValue p(ring.ctx),sf(ring.ctx),repeated(ring.ctx);
        ring.write(p.p,face);
        atlas::rational_squarefree_part(sf.p,p.p,ring.ctx);
        if (fmpq_mpoly_divides(repeated.p,p.p,sf.p,ring.ctx) &&
            fmpq_mpoly_length(repeated.p,ring.ctx)>1) return true;
    }
    return false;
}

Json::Value check(const Json::Value& request) {
    if (request["operation"]=="nonisolated_repair_orbit")
        return nonisolated_repair_orbit(request["terms"],request["rays"],request["ordinary"].asBool());
    if (request["operation"]=="quadratic_cone_prefix") return quadratic_cone_prefix(request["terms"]);
    if (request["operation"]=="relative_source_prefix")
        return quadratic_cone_prefix(request["terms"]) || relative_axis_prefix(request["terms"],request["intrinsic"].asBool());
    if (request["operation"]=="relative_supports") return relative_supports(request["supports"]);
    if (request["operation"]=="ordinary_prefix") return ordinary_prefix(request["terms"]);
    // Only scheduler-provided rows from ordinary_prefix, and regular fan
    // charts from the shared Newton fan builder, are accepted by the caller.
    if (request["operation"]!="unresolved_chart") throw std::invalid_argument("Unknown obstruction operation.");
    const auto& cones=request["cones"];
    if (cones.empty() || cones.size()>192 || request["rays"].size()>128) return false;
    const auto f=read_terms(request["terms"],3);
    if (f.empty() || f.size()>24) return false;
    std::vector<Powers> rays;
    for (const auto& row : request["rays"]) rays.push_back(read_exponents(row,3));
    for (const auto& cone : cones) {
        if (cone.size()!=3) return false;
        atlas::check_deadline();
        FanChart chart;
        for (unsigned j=0;j<3;++j) chart[j]=rays.at(cone[j].asUInt());
        if (bad_chart(f,chart,request["intrinsic"].asBool())) return true;
    }
    return false;
}
}
extern "C" const char* atlas_canonical_obstruction(const char* input) {
    return atlas::sparse_json::invoke(input,check);
}

extern "C" int atlas_quadratic_cone_obstruction_abi() noexcept { return 1; }
extern "C" int atlas_strict_fan_obstruction_abi() noexcept { return 1; }
extern "C" int atlas_relative_source_obstruction_abi() noexcept { return 1; }
extern "C" int atlas_nonisolated_repair_orbit_abi() noexcept { return 1; }
