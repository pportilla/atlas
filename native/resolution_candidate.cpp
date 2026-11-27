// Exact, prefix-skipping decoding of the exhaustive resolution certificate syntax.
#include "local_io.hpp"
#include "local_monomial.hpp"
#include "groebner.hpp"
#include "membership_certificate.hpp"
#include "primary_decomposition.hpp"
#include "polynomial_factors.hpp"
#include "quotient_ring.hpp"
#include "rational_sparse_json.hpp"
#include "resolution_tangents.hpp"
#include "sparse_flint.hpp"
#include <array>
#include <numeric>
#include <set>
#include <tuple>

namespace {
using namespace atlas::sparse_json;
struct InvalidCode {};

struct Factors {
    fmpq_mpoly_factor_t v;
    const fmpq_mpoly_ctx_struct* ctx;
    explicit Factors(const fmpq_mpoly_ctx_struct* c) : ctx(c) { fmpq_mpoly_factor_init(v, c); }
    ~Factors() { fmpq_mpoly_factor_clear(v, ctx); }
    Factors(const Factors&) = delete;
};

struct Reader {
    std::string bits;
    size_t at = 0;
    explicit Reader(const Z& code) {
        if (code < 0) throw std::invalid_argument("Expected a nonnegative resolution candidate code.");
        Z sentinel = code+1;
        bits = sentinel.get_str(2).substr(1);
    }
    Z natural() {
        size_t start = at, last = at+(bits.size()-at+1)/2;
        while (at < last && bits[at] != '1') ++at;
        if (at == last) throw InvalidCode{};
        const size_t length = at-start+1;
        Z value(bits.substr(at,length),2);
        at += length;
        return value-1;
    }
    Z integer() {
        Z value = natural();
        if (mpz_odd_p(value.get_mpz_t())) return (value+1)/2;
        return -value/2;
    }
    size_t size(size_t minimum=0) {
        const Z count = natural()+minimum;
        if (count > bits.size()-at) throw InvalidCode{};
        return count.get_ui();
    }
};

std::string polynomial(Reader& reader, size_t variables, size_t projective, size_t field_degree, bool boundary) {
    Terms p;
    Z degree = -1;
    const size_t count = reader.size(1);
    bool mixed = false;
    for (size_t j = 0; j < count; ++j) {
        auto c = reader.integer();
        if (!c) throw InvalidCode{};
        Z index = 0;
        if (field_degree > 1) {
            index = reader.natural();
            if (index >= field_degree) throw InvalidCode{};
        }
        Powers e(variables+projective);
        for (auto& exponent : e) exponent = reader.natural();
        if (boundary) {
            Z total = std::accumulate(e.begin()+variables,e.end(),Z(0));
            if (degree >= 0 && degree != total) mixed = true;
            degree = total;
        }
        if (field_degree > 1) e.push_back(index);
        auto [it,fresh] = p.emplace(e,c);
        if (!fresh && (it->second += c) == 0) p.erase(it);
    }
    // Consume the full polynomial before rejecting a nonhomogeneous one.
    // Prefix skipping must retain the original exhaustive schedule.
    if (p.empty() || mixed) throw InvalidCode{};
    return atlas::local::encode_row(p);
}

Json::Value candidate(Reader& reader, size_t n) {
    Json::Value result;
    Z requested = reader.natural()+1;
    if (requested > 1 && requested > reader.bits.size()-reader.at) throw InvalidCode{};
    const size_t degree = requested.get_ui();
    if (degree > 1) {
        Powers coefficients;
        Terms relation;
        for (size_t k = 0; k < degree; ++k) {
            auto c = reader.integer();
            if (c) relation.emplace(Powers{k},c);
            coefficients.push_back(std::move(c));
        }
        auto root = reader.natural();
        if (root >= degree) throw InvalidCode{};
        relation.emplace(Powers{degree},1);
        const auto factors = atlas::primary::factors(relation,1);
        if (factors.size() != 1 || factors.front().second != 1) throw InvalidCode{};
        coefficients.emplace_back(1);
        std::reverse(coefficients.begin(),coefficients.end());
        result["coefficients"] = strings(coefficients);
        result["index"] = Json::UInt64(root.get_ui());
    }
    Json::Value center(Json::arrayValue), boundary(Json::arrayValue);
    const size_t generators = reader.size(1);
    for (size_t j=0; j<generators; ++j) center.append(polynomial(reader,n,0,1,false));
    const size_t components = reader.size();
    for (size_t j=0; j<components; ++j) {
        Json::Value rows(Json::arrayValue);
        const size_t size = reader.size(1);
        for (size_t k=0; k<size; ++k) rows.append(polynomial(reader,n,generators,degree,true));
        boundary.append(std::move(rows));
    }
    result["center"] = std::move(center);
    result["boundary"] = std::move(boundary);
    return result;
}

// Detect an ordinary coordinate-centre modification already certified by the
// current exhaustive policy. For these data the preceding Newton branch can
// never supply a record; the graph verifier still verifies the ordered candidates.
Json::Value transverse_pencil(const Json::Value& rows) {
    // At most 16 binary coefficients per row and 64 bits per input rational.
    // The Horner/power certificate then operates on O(16 * 64)-bit integers.
    constexpr unsigned width=3, max_degree=15;
    if (!rows.isArray() || rows.size()<4 || rows.size()>2*(max_degree+1)) return {};
    for (const auto& term:rows) {
        if (!term.isArray() || term.size()!=2 || !term[0].isString() || term[0].asString().size()>43 ||
                !term[1].isArray() || term[1].size()!=width) return {};
        for (const auto& p:term[1])
            if (!p.isString() || p.asString().empty() || p.asString().size()>2 ||
                    p.asString().find_first_not_of("0123456789")!=std::string::npos) return {};
    }
    const auto terms=read_terms(rows,width);
    if (terms.empty() || terms.size()!=rows.size()) return {};
    for (const auto& [e,c]:terms)
        if (!c || mpz_sizeinbase(c.get_num_mpz_t(),2)>64 || mpz_sizeinbase(c.get_den_mpz_t(),2)>64 ||
                std::any_of(e.begin(),e.end(),[&](const Z& k){return k>max_degree;})) return {};
    for (unsigned parameter=0;parameter<width;++parameter) {
        std::vector<unsigned> axes;
        for (unsigned j=0;j<width;++j) if (j!=parameter) axes.push_back(j);
        const unsigned a=axes[0], b=axes[1];
        const auto& first=terms.begin()->first;
        const unsigned degree=first[a].get_ui()+first[b].get_ui();
        if (degree<2 || degree>max_degree) continue;
        // F = H(X,Y) + Z*K(X,Y), homogeneous initially in (X,Y).
        // Identify H=c*(X+sY)^d term by term over Q.
        std::vector<Q> h(degree+1), k(degree+1);
        bool homogeneous=true;
        for (const auto& [e,c]:terms) {
            if (e[parameter]>1 || e[a]+e[b]!=degree) {homogeneous=false;break;}
            (e[parameter]==0?h:k)[e[b].get_ui()]=c;
        }
        if (!homogeneous || h[0]==0 || h[1]==0) continue;
        const Q slope=h[1]/(degree*h[0]);
        Q coefficient=h[0], value=0;
        for (unsigned j=0;j<=degree;++j) {
            if (h[j]!=coefficient) {homogeneous=false;break;}
            coefficient *= slope*(degree-j)/(j+1);
            value = value*(-slope)+k[j];  // Evaluate K(-s,1).
        }
        if (!homogeneous || value==0) continue;
        // The source normal fan is the blowup of (X,Y): its polyhedron is
        // x+y>=d in the orthant, since H has both axial terms. On the fibre
        // orbit Z=0 the non-monomial repeated form H is degenerate; this
        // boundary modification cannot enter the finite-point repair path.
        // The only missing axial intercept is Z. Each of the Newton policy's
        // donor shears gives all three intercepts d. Its ordinary exceptional
        // P^2 has a line, meeting the dense torus, along which H' = L^d
        // fails transversality. Finite-point Newton repair rejects that curve.
        // K(-s,1) != 0 also makes H+Z*K a smooth section transverse to the
        // coordinate exceptional divisor. Both axes of the centre are used,
        // and coprimality makes the primitive polynomial linear in Z irreducible.
        Json::Value found(Json::arrayValue);
        found.append(a);found.append(b);
        return found;
    }
    return {};
}

// A fully supported repeated linear tangent has equal axial intercepts d.
// The only coordinate choice is the identity; the Newton polyhedron is the
// simplex of size d plus the orthant. Its normal fan is the smooth ordinary
// blowup. On the dense exceptional torus the equation is L^d; the curve
// L=0 is a positive-dimensional locus of failed crossings. The finite-point
// repair cannot return a Newton chart. For a reduced equation neither the
// identity nor this ordinary blowup resolves it. The origin worker checks
// reducedness before computing any blowups. Every proper coordinate-centre
// candidate misses a pure axial term on a free axis. Later certificate candidates still race a longer canonical solve.
bool repeated_tangent_obstruction(const Json::Value& rows, unsigned n, bool reduced_graph = false) {
    constexpr unsigned max_degree=16, max_rows=128;
    if (n!=3 || !rows.isArray() || rows.size()<6 || rows.size()>max_rows) return false;
    for (const auto& row : rows) {
        if (!row.isArray() || row.size()!=2 || !row[0].isString() || row[0].asString().size()>43 ||
                !row[1].isArray() || row[1].size()!=n) return false;
        for (const auto& e : row[1]) {
            if (!e.isString()) return false;
            const auto digits=e.asString();
            if (digits.empty() || digits.size()>2 || digits.find_first_not_of("0123456789")!=std::string::npos)
                return false;
        }
    }
    const Terms terms=read_terms(rows,n);
    if (terms.size()!=rows.size()) return false;
    unsigned order=max_degree+1, degree=0;
    Terms lowest;
    for (const auto& [e,c] : terms) {
        const unsigned total=std::accumulate(e.begin(),e.end(),0u,
            [](unsigned sum, const Z& power){return sum+power.get_ui();});
        if (!c || total>max_degree || mpz_sizeinbase(c.get_num_mpz_t(),2)>64 ||
                mpz_sizeinbase(c.get_den_mpz_t(),2)>64) return false;
        degree=std::max(degree,total);
        if (total<order) {order=total;lowest.clear();}
        if (total==order) lowest.emplace(e,c);
    }
    // For L using every variable the degree-d expansion has exactly (d+2
    // choose 2) coefficients. Do not construct it without a matching support
    // count: the entire synchronous check is bounded by 128 sparse terms.
    if (order<2 || degree==order || lowest.size()!=(order+1)*(order+2)/2) return false;
    Powers e(n); e[0]=order;
    const auto first=lowest.find(e);
    if (first==lowest.end()) return false;
    const Q leading=first->second;
    Terms linear;
    --e[0];
    for (unsigned j=0; j<n; ++j) {
        ++e[j];
        const auto mixed=lowest.find(e);
        if (mixed==lowest.end()) return false;
        Powers axis(n); axis[j]=1;
        linear.emplace(std::move(axis), j==0 ? Q(1) : Q(mixed->second/(order*leading)));
        --e[j];
    }
    const Terms expected=atlas::primary::power(std::move(linear),order,n);
    for (const auto& [monomial,coefficient] : expected) {
        const auto found=lowest.find(monomial);
        if (found==lowest.end() || found->second!=leading*coefficient) return false;
    }
    if (!reduced_graph) return true;
    // Write F=c*L^d+H in coordinates (L,u,v). A homogeneous binary H
    // has as its singular locus off zero exactly its repeated factor lines.
    // No such line leaves only the origin. Two distinct lines give a
    // singular reduced centre. With one line M=0 of multiplicity m, the
    // M-chart of the blowup (L,M) contains L=M*t and has strict equation
    // c*M^(d-b)*t^d + H/M^b for b=min(d,m). Both summands have order
    // at least two at (t,u,v)=0 since deg(H)>=d+2, so that graph fails.
    if (degree < order + 2) return false;
    std::vector<bool> used(n);
    for (const auto& [e, c] : terms) {
        if (lowest.count(e)) continue;
        if (atlas::local::total_degree(e) != degree) return false;
        for (unsigned j=0; j<n; ++j) if (e[j] != 0) used[j] = true;
    }
    return std::find(used.begin(), used.end(), false) != used.end();
}

// On L=M=0 the irreducible pinch A*L^2+B*M^2*C is singular. If all
// components of the direction of this line are nonzero, at least two
// coordinates remain generically nonzero after any *single* triangular
// translation. A toric subdivision is an isomorphism over those strata
// (cones of dimension at most one). The strict transform is still singular
// along that punctured curve, whose closure meets the origin fibre by
// properness. This excludes both the diagonal fan and each one-axis
// translated fan. A composition of translations needs its own test.
bool inclined_toric_obstruction(const std::string& row) {
    if (!atlas::certificate::enabled() || row.size() > 1024) return false;
    const auto input = atlas::local::decode_row(row, 3);
    if (input.size() > 16) return false;
    std::array<Terms, 2> parts;
    for (const auto& [e, c] : input) {
        const Z degree = e[0]+e[1]+e[2];
        if ((degree != 2 && degree != 3) || mpz_sizeinbase(c.get_num_mpz_t(), 2) > 64 ||
            mpz_sizeinbase(c.get_den_mpz_t(), 2) > 64) return false;
        parts[degree.get_ui()-2].emplace(e, c);
    }
    using Linear = std::array<Q, 3>;
    Linear L{}, M{}, C{};
    atlas::local::SparseLexRing ring(3);
    atlas::PolyValue source(ring.ctx), expanded(ring.ctx);
    for (unsigned j = 0; j < 2; ++j) {
        atlas::check_deadline();
        if (parts[j].empty()) return false;
        ring.write(source.p, parts[j]);
        Factors factors(ring.ctx);
        if (!fmpq_mpoly_factor_squarefree(factors.v, source.p, ring.ctx) ||
            factors.v->num != static_cast<slong>(j+1)) return false;
        for (slong i = 0; i < factors.v->num; ++i) {
            const auto exponent = factors.v->exp+i;
            Linear* form;
            if (fmpz_equal_ui(exponent, 2)) form = j ? &M : &L;
            else if (j && fmpz_is_one(exponent)) form = &C;
            else return false;
            for (const auto& [e, c] : ring.read(factors.v->poly+i)) {
                if (e[0]+e[1]+e[2] != 1) return false;
                (*form)[std::find(e.begin(), e.end(), 1)-e.begin()] = c;
            }
        }
        if (!fmpq_mpoly_factor_expand(expanded.p, factors.v, ring.ctx) ||
            !fmpq_mpoly_equal(expanded.p, source.p, ring.ctx)) return false;
    }
    Q det = 0;
    for (size_t i = 0; i < 3; ++i) {
        const size_t j = (i+1)%3, k = (i+2)%3;
        const Q direction = L[j]*M[k] - L[k]*M[j];
        if (!direction) return false;
        det += C[i]*direction;
    }
    // Independent L,M,C are coordinates over the geometric field. As a
    // polynomial linear in C, A*L^2+B*M^2*C is primitive, hence irreducible.
    return det != 0;
}

// Two binomial squares can contribute parallel, equally spaced triples in
// the support of f. Their normals suggest weights only. Direct substitution
// into f and all three logarithmic derivatives proves the entire curve is
// singular, including when the summands have *different* weighted degrees.
bool singular_curve_from_triples(const Terms& f) {
    using Triple = std::array<long, 3>;
    if (f.size() < 5 || f.size() > 16) return false;
    std::map<Triple, Q> support;
    for (const auto& [e, c] : f)
        support.emplace(Triple{e[0].get_si(), e[1].get_si(), e[2].get_si()}, c);
    std::set<Triple> directions;
    for (auto a = support.begin(); a != support.end(); ++a) {
        for (auto b = std::next(a); b != support.end(); ++b) {
            Triple mid{}, step{};
            for (unsigned j = 0; j < 3; ++j) {
                step[j] = b->first[j] - a->first[j];
                mid[j] = a->first[j] + step[j]/2;
            }
            if (std::any_of(step.begin(), step.end(), [](long v) { return v % 2; }) ||
                !support.count(mid)) continue;
            const long g = std::gcd(std::gcd(step[0], step[1]), step[2]);
            for (auto& v : step) v /= g;
            directions.insert(step);
            if (directions.size() > 12) return false;
        }
    }
    for (auto a = directions.begin(); a != directions.end(); ++a) {
        for (auto b = std::next(a); b != directions.end(); ++b) {
            atlas::check_deadline();
            Triple w{};
            for (unsigned j = 0; j < 3; ++j)
                w[j] = (*a)[(j+1)%3] * (*b)[(j+2)%3] - (*a)[(j+2)%3] * (*b)[(j+1)%3];
            if (w[0] < 0) for (auto& v : w) v = -v;
            if (*std::min_element(w.begin(), w.end()) <= 0 ||
                w[0] == w[1] || w[1] == w[2] || w[2] == w[0]) continue;
            std::map<long, std::array<Q, 4>> coefficients;
            for (const auto& [e, c] : support) {
                auto& values = coefficients[std::inner_product(w.begin(), w.end(), e.begin(), 0L)];
                values[0] += c;
                for (unsigned j = 0; j < 3; ++j) values[j+1] += c * e[j];
            }
            if (!std::all_of(coefficients.begin(), coefficients.end(), [](const auto& row) {
                    return row.second == std::array<Q, 4>{};
                })) continue;
            // Smoothness of the whole intrinsic strict transform excludes
            // intersections of distinct geometric branches. Squarefreeness
            // keeps nonreduced-input handling with the existing dispatcher.
            atlas::local::SparseLexRing ring(3);
            atlas::PolyValue source(ring.ctx);
            ring.write(source.p, f);
            Factors factors(ring.ctx);
            if (!fmpq_mpoly_factor_squarefree(factors.v, source.p, ring.ctx)) return false;
            for (slong i = 0; i < factors.v->num; ++i)
                if (!fmpz_is_one(factors.v->exp+i)) return false;
            return factors.v->num > 0;
        }
    }
    return false;
}

// A singular monomial curve with three distinct positive weights is
// contained in no plane through zero. In any linear coordinates it meets
// the dense torus, and changing a single coordinate still leaves at least
// two nonzero there. A monomial modification is an isomorphism at its
// generic point. Properness brings the closure of the strict singular
// locus back to the origin fibre. This obstructs the unrepaired intrinsic
// fan; starting a repair at a fibre point can subsequently change a curve.
// The scheduler tests the exceptional defect orbits before skipping repairs.
bool singular_torus_curve(const std::string& row) {
    if (!atlas::certificate::enabled() || row.size() > 1600) return false;
    const Terms f = atlas::local::decode_row(row, 3);
    if (f.size() < 3 || f.size() > 24) return false;
    using Triple = std::array<long, 3>;
    Triple intercept{};
    std::vector<Triple> exponents;
    for (const auto& [e, c] : f) {
        if (mpz_sizeinbase(c.get_num_mpz_t(), 2) > 64 || mpz_sizeinbase(c.get_den_mpz_t(), 2) > 64 ||
            std::any_of(e.begin(), e.end(), [](const auto& k) { return k < 0 || k > 12; })) return false;
        Triple a{e[0].get_si(), e[1].get_si(), e[2].get_si()};
        for (unsigned j = 0; j < 3; ++j)
            if (a[j] && !a[(j+1)%3] && !a[(j+2)%3]) intercept[j] = a[j];
        exponents.push_back(a);
    }
    // Even the identity-frame polynomial graph adds only one coordinate.
    // Over the dense original torus its image therefore belongs to strata
    // of codimension at most one, where any fan subdivision is an isomorphism.
    if (singular_curve_from_triples(f)) return true;
    if (intercept[0] && intercept[1] && intercept[2]) return false;
    auto difference = [&](const Triple& b) -> Triple {
        return {b[0]-exponents[0][0], b[1]-exponents[0][1], b[2]-exponents[0][2]};
    };
    const auto first = difference(exponents[1]);
    Triple weight{};
    for (size_t i = 2; i < exponents.size(); ++i) {
        auto v = difference(exponents[i]);
        for (unsigned j = 0; j < 3; ++j)
            weight[j] = first[(j+1)%3]*v[(j+2)%3] - first[(j+2)%3]*v[(j+1)%3];
        if (weight != Triple{}) break;
    }
    if (weight[0] < 0) for (auto& k : weight) k = -k;
    for (unsigned j = 0; j < 3; ++j)
        if (weight[j] <= 0 || weight[j] == weight[(j+1)%3]) return false;
    for (const auto& e : exponents) {
        const auto v = difference(e);
        if (std::inner_product(weight.begin(), weight.end(), v.begin(), 0L)) return false;
    }
    // At a sign vector test f and x_j*df/dx_j. Homogeneity extends each
    // vanishing to the entire parametrized curve sign_j*t^weight_j.
    bool singular = false;
    for (unsigned mask = 0; mask < 8 && !singular; ++mask) {
        std::array<Q, 4> values{};
        for (const auto& [e, c] : f) {
            Q term = c;
            for (unsigned j = 0; j < 3; ++j)
                if ((mask & (1u << j)) && mpz_odd_p(e[j].get_mpz_t())) term = -term;
            values[0] += term;
            for (unsigned j = 0; j < 3; ++j) values[j+1] += term*e[j];
        }
        singular = values == std::array<Q, 4>{};
    }
    if (!singular) return false;

    // Certify geometric irreducibility of a primitive quadratic over the
    // other two variables. A monic discriminant square over Qbar already
    // has its unique monic square root over QQ, by Galois descent.
    for (unsigned axis = 0; axis < 3; ++axis) {
        Q leading = 0;
        Terms linear, constant;
        bool quadratic = true;
        for (const auto& [e, c] : f) {
            auto v = e;
            v[axis] = 0;
            if (e[axis] == 2 && v == Powers(3)) leading = c;
            else if (e[axis] == 1) linear.emplace(std::move(v), c);
            else if (e[axis] == 0) constant.emplace(e, c);
            else { quadratic = false; break; }
        }
        if (!quadratic || leading == 0) continue;
        auto disc = atlas::primary::multiply(linear, linear);
        for (const auto& [e, c] : constant)
            if ((disc[e] -= 4*leading*c) == 0) disc.erase(e);
        if (disc.empty()) continue;
        atlas::local::SparseLexRing ring(3);
        atlas::PolyValue value(ring.ctx);
        ring.write(value.p, disc);
        fmpq_mpoly_make_monic(value.p, value.p, ring.ctx);
        if (!fmpq_mpoly_is_square(value.p, ring.ctx)) return true;
    }
    return false;
}

Json::Value jacobian_support(const Terms& f, size_t n) {
    atlas::local::SparseLexRing ring(n);
    atlas::PolyValue source(ring.ctx), derivative(ring.ctx), squarefree(ring.ctx);
    ring.write(source.p,f);
    using Generators=std::set<Terms>;
    auto insert_part=[&](Generators& rows, const fmpq_mpoly_t p) {
        if (!fmpq_mpoly_is_zero(p,ring.ctx)) {
            atlas::rational_squarefree_part(squarefree.p,p,ring.ctx);
            rows.insert(ring.read(squarefree.p));
        }
    };
    Generators equations;
    insert_part(equations,source.p);
    for (size_t i=0;i<n;++i) {
        fmpq_mpoly_derivative(derivative.p,source.p,i,ring.ctx);
        insert_part(equations,derivative.p);
    }
    // Reduce by newly exposed linear equations, then strip repeated factors
    // again. Each step preserves the common geometric zero set.
    const Terms one{{Powers(n),1}};
    for (size_t step=0;step<n;++step) {
        if (equations.count(one)) { equations={one}; break; }
        atlas::local::Basis linear;
        for (const auto& equation:equations)
            if (std::all_of(equation.begin(),equation.end(),[](const auto& term) {
                return std::accumulate(term.first.begin(),term.first.end(),Z(0))<=1;
            })) linear.push_back(equation);
        if (linear.empty()) break;
        atlas::quotient_ring::Ring quotient(linear,n,false);
        Generators next(quotient.basis().begin(),quotient.basis().end());
        if (next.count(one)) { equations={one}; break; }
        for (const auto& row:equations) {
            const auto remainder=std::get<Terms>(quotient.evaluate("reduce",{row}));
            ring.write(source.p,remainder);
            insert_part(next,source.p);
        }
        if (next==equations) break;
        equations=std::move(next);
    }
    Json::Value rows(Json::arrayValue);
    for (const auto& equation:equations) rows.append(atlas::local::encode_row(equation));
    return rows;
}

Json::Value two_blowups(const Terms& f,size_t n,size_t x,size_t y) {
    atlas::local::SparseLexRing ring(n);
    atlas::PolyValue source(ring.ctx);
    ring.write(source.p,f);
    std::istringstream slopes(atlas::repeated_directions(source.p,x,y,ring.ctx));
    Powers ex(n),ey(n);
    ex.at(x)=1;
    ey.at(y)=1;
    const Terms first{{ex,1}}, second{{ey,1}};
    Json::Value result(Json::arrayValue);
    for (std::string slope;std::getline(slopes,slope);) {
        Terms a=first,b=second;
        if (slope=="vertical") std::swap(a,b);
        else {
            const Q q(slope);
            if (q!=0) b.emplace(ex,-q);
        }
        // For tangent b=0 the composite centre is (a^3,ab,b^2).
        // The two reduced exceptional lines are cut out by U1 and U3
        // over a=b=0. Keep the factors for symbolic callers' display.
        Json::Value item;
        item["parameters"].append(atlas::local::encode_row(a));
        item["parameters"].append(atlas::local::encode_row(b));
        const Terms* parameters[]={&a,&b};
        for (const std::vector<int>& product:std::vector<std::vector<int>>{{0,0,0},{0,1},{1,1}}) {
            Json::Value factors(Json::arrayValue);
            Terms generator{{Powers(n),1}};
            for (int i:product) {
                generator=atlas::primary::multiply(generator,*parameters[i]);
                factors.append(i);
            }
            item["centre"].append(atlas::local::encode_row(generator));
            item["factors"].append(std::move(factors));
        }
        for (int projective:{2,4}) {
            Json::Value ideal(Json::arrayValue);
            for (int i:{0,1,projective}) ideal.append(i);
            item["boundary"].append(std::move(ideal));
        }
        result.append(std::move(item));
    }
    return result;
}

// For f = a*x^2 + b*x + c, substitute x = X - b/(2*a).
// b and c use only the other variables. With b(0)=0 this triangular
// automorphism fixes the origin and has Jacobian determinant one.
Json::Value quadratic_translations(const Terms& f, size_t n) {
    Json::Value candidates(Json::arrayValue);
    for (size_t axis = 0; axis < n; ++axis) {
        Powers square(n);
        square[axis] = 2;
        const auto leading = f.find(square);
        if (leading == f.end()) continue;
        const Q a = leading->second;
        Terms shift, adapted;
        bool quadratic = true;
        for (const auto& [e, coefficient] : f) {
            if (e[axis] == 0 || e == square) {
                adapted.emplace(e, coefficient);
            } else if (e[axis] == 1) {
                auto power = e;
                power[axis] = 0;
                shift.emplace(std::move(power), -coefficient / (2*a));
            } else {
                quadratic = false;
                break;
            }
        }
        // The nonlinear shift may include linear terms, but must fix the origin.
        // Leave purely linear changes to the existing linear-coordinate routes.
        if (!quadratic || shift.size() > 128 || shift.count(Powers(n)) ||
            std::none_of(shift.begin(), shift.end(), [](const auto& term) {
                return std::accumulate(term.first.begin(), term.first.end(), Z(0)) > 1;
            })) continue;
        bool cancelled = false;
        for (const auto& [e, c] : atlas::primary::multiply(shift, shift)) {
            if ((adapted[e] -= a*c) == 0) {
                adapted.erase(e);
                cancelled = true;
            }
        }
        // Search for a hidden square in the given face. If no constant-part
        // monomial cancels, leave the original face to the Newton verifier.
        if (!cancelled) continue;
        Json::Value choice;
        choice["axis"] = Json::UInt64(axis);
        choice["shift"] = atlas::local::encode_row(shift);
        choice["adapted"] = atlas::local::encode_row(adapted);
        candidates.append(std::move(choice));
    }
    return candidates;
}

// Straighten a repeated graph branch of the ramification polynomial of a
// cyclic surface cover. Only the search for a useful frame is bounded.
// Every chosen frame is an exact triangular polynomial automorphism.
Json::Value ramification_translations(const Terms& f, size_t n) {
    Json::Value result(Json::arrayValue);
    if (n!=3 || f.size()>64) return result;
    for (size_t cover=0;cover<n;++cover) {
        Terms branch;
        unsigned pure=0;
        for (const auto& [e,c] : f) {
            const Z degree=std::accumulate(e.begin(),e.end(),Z(0));
            if (e[cover]==0 && degree<=64 && mpz_sizeinbase(c.get_num_mpz_t(),2)<=4096 &&
                mpz_sizeinbase(c.get_den_mpz_t(),2)<=4096) branch.emplace(e,c);
            else if (e[cover]<2 || e[cover]!=degree || ++pure>1) break;
        }
        if (pure!=1 || branch.size()+1!=f.size() || branch.empty()) continue;
        atlas::local::SparseLexRing ring(n);
        atlas::PolyValue source(ring.ctx);
        ring.write(source.p,branch);
        // Square-free products alone often expose a repeated graph; handle
        // only those, so this optional probe never needs complete factoring.
        fmpq_mpoly_factor_t squarefree;
        fmpq_mpoly_factor_init(squarefree,ring.ctx);
        std::vector<Terms> repeated;
        if (fmpq_mpoly_factor_squarefree(squarefree,source.p,ring.ctx))
            for (slong i=0;i<squarefree->num;++i)
                if (fmpz_cmp_ui(squarefree->exp+i,2)>=0)
                    repeated.push_back(ring.read(squarefree->poly+i));
        fmpq_mpoly_factor_clear(squarefree,ring.ctx);
        ring.write(source.p,f);
        for (const auto& factor : repeated) {
            if (factor.size()<2 || factor.count(Powers(n))) continue;
            for (size_t axis=0;axis<n;++axis) {
                Powers unit(n);unit[axis]=1;
                const auto leading=factor.find(unit);
                if (leading==factor.end()) continue;
                Terms shift=factor;
                shift.erase(unit);
                if (std::any_of(shift.begin(),shift.end(),[&](const auto& term){return term.first[axis]!=0;}))
                    continue;  // The branch must be a graph a*y+h(z) with scalar a.
                for (auto& [e,c] : shift) c/=-leading->second;
                if (shift.size()>1) {
                    // Translating a bivariate branch preserves its degree in y.
                    // Bound its new z-degree before expanding a polynomial shift.
                    const size_t z=3-cover-axis;
                    const Z shift_degree=shift.rbegin()->first[z];
                    Z y_degree=0, z_degree=0;
                    for (const auto& [e,c] : branch) {
                        y_degree=std::max(y_degree,e[axis]);
                        z_degree=std::max(z_degree,Z(e[z]+e[axis]*shift_degree));
                    }
                    if ((1+y_degree)*(1+z_degree)>4096) continue;
                }
                Terms image=shift;image[unit]=1;
                atlas::PolyValue adapted(ring.ctx);
                std::vector<atlas::PolyValue> coordinates;
                std::vector<fmpq_mpoly_struct*> images;
                coordinates.reserve(n);
                for (size_t i=0;i<n;++i) {
                    coordinates.emplace_back(ring.ctx);
                    auto p=coordinates.back().p;
                    if (i==axis) ring.write(p,image);
                    else fmpq_mpoly_gen(p,i,ring.ctx);
                    images.push_back(p);
                }
                if (!fmpq_mpoly_compose_fmpq_mpoly(adapted.p,source.p,images.data(),ring.ctx,ring.ctx))
                    continue;
                Json::Value choice;
                choice["axis"]=Json::UInt64(axis);
                choice["shift"]=atlas::local::encode_row(shift);
                choice["adapted"]=atlas::local::encode_row(ring.read(adapted.p));
                result.append(std::move(choice));
            }
        }
    }
    return result;
}

// The pullback of f by x_axis = X_axis + shift, where shift does not involve
// x_axis. Declines an expansion beyond the term budget of the fan engines.
bool translated_polynomial(Terms& adapted, const Terms& f, size_t n, size_t axis, const Terms& shift) {
    Z degree = 0, growth = 0;
    for (const auto& [e, c] : f) degree = std::max(degree, e[axis]);
    for (const auto& [e, c] : shift) growth = std::max(growth, Z(std::accumulate(e.begin(), e.end(), Z(0))));
    if (degree > 128 || degree*growth > 4096) return false;
    atlas::local::SparseLexRing ring(n);
    atlas::PolyValue source(ring.ctx), image(ring.ctx);
    ring.write(source.p, f);
    Powers unit(n);
    unit[axis] = 1;
    Terms graph = shift;
    graph[unit] += 1;
    std::vector<atlas::PolyValue> coordinates;
    std::vector<fmpq_mpoly_struct*> images;
    coordinates.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        coordinates.emplace_back(ring.ctx);
        auto p = coordinates.back().p;
        if (i == axis) ring.write(p, graph);
        else fmpq_mpoly_gen(p, i, ring.ctx);
        images.push_back(p);
    }
    if (!fmpq_mpoly_compose_fmpq_mpoly(image.p, source.p, images.data(), ring.ctx, ring.ctx)) return false;
    if (fmpq_mpoly_length(image.p, ring.ctx) > 4096) return false;
    adapted = ring.read(image.p);
    return true;
}

Json::Value translation_choice(size_t axis, const Terms& shift, const Terms& adapted) {
    Json::Value choice;
    choice["axis"] = Json::UInt64(axis);
    choice["shift"] = atlas::local::encode_row(shift);
    choice["adapted"] = atlas::local::encode_row(adapted);
    return choice;
}

// A Newton fan has a boundary ray over the coordinate subspace L with free
// coordinates I only if f vanishes on L. When L is not in the singular locus,
// f has terms x_j*n(x_I) with j outside I. The substitution
// x_j = X_j + m, with m = (prod_{i in I} X_i)^k, is a triangular automorphism
// with Jacobian determinant one. It fixes every coordinate subspace that
// does not contain L, hence every coordinate subspace of the singular locus,
// and the pullback has the terms n(x_I)*m, so that it no longer vanishes on L.
//
// A term x_j^a*r of f gives the new terms x_j^b*m^(a-b)*r. Their value on a
// normal w of the Newton polyhedron is at least that of the original term
// when w(m) >= w_j. With strict inequality on every normal that does not
// vanish on I the initial forms on those normals are kept, so that the
// substitution changes the polyhedron only towards L. The least such k is
// always proposed. A smaller k with equality on some normal changes the
// face of that normal by the same substitution, which can be degenerate
// there: it is proposed first and marked as a tie for the caller to test.
// The subspaces are treated in order of dimension: one that contains a
// treated subspace has left the hypersurface with it.
Json::Value boundary_translations(const Terms& f, size_t n, const Json::Value& subspaces, const Json::Value& normals) {
    Json::Value candidates(Json::arrayValue);
    std::vector<Powers> rays;
    for (const auto& ray : normals) rays.push_back(read_exponents(ray, n));
    std::vector<std::vector<size_t>> ordered;
    for (const auto& subspace : subspaces) {
        std::vector<size_t> axes;
        for (const auto& index : subspace) {
            if (index.asUInt64() >= n) throw std::invalid_argument("Invalid coordinate subspace.");
            axes.push_back(index.asUInt64());
        }
        std::sort(axes.begin(), axes.end());
        axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
        if (!axes.empty()) ordered.push_back(std::move(axes));
    }
    std::stable_sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) { return a.size() < b.size(); });
    for (const auto& axes : ordered) {
        std::vector<bool> free(n);
        for (size_t i : axes) free[i] = true;
        // The least order on L of the coefficient of each normal coordinate.
        std::vector<bool> seen(n);
        std::vector<Z> order(n);
        bool vanishes = true;
        for (const auto& [e, c] : f) {
            size_t outside = n, count = 0;
            Z degree = 0;
            for (size_t i = 0; i < n; ++i) {
                if (free[i]) degree += e[i];
                else if (e[i] != 0) { count += e[i] == 1 ? 1 : 2; outside = i; }
            }
            if (count == 0) { vanishes = false; break; }
            if (count != 1) continue;
            if (!seen[outside] || degree < order[outside]) order[outside] = degree;
            seen[outside] = true;
        }
        if (!vanishes) continue;
        struct Choice { Z degree, power; size_t axis; bool tie; };
        std::vector<Choice> choices;
        for (size_t j = 0; j < n; ++j) {
            if (!seen[j]) continue;
            Z strict = 1, weak = 1;
            for (const auto& w : rays) {
                Z value = 0;
                for (size_t i : axes) value += w[i];
                if (value <= 0) continue;
                strict = std::max(strict, Z(w[j]/value + 1));
                weak = std::max(weak, Z((w[j] + value - 1)/value));
            }
            const Z size = static_cast<unsigned long>(axes.size());
            if (weak < strict) choices.push_back({order[j] + weak*size, weak, j, true});
            choices.push_back({order[j] + strict*size, strict, j, false});
        }
        std::stable_sort(choices.begin(), choices.end(), [](const Choice& a, const Choice& b) {
            return std::tie(a.axis, a.degree) < std::tie(b.axis, b.degree);
        });
        std::stable_sort(choices.begin(), choices.end(), [&](const Choice& a, const Choice& b) {
            return order[a.axis] < order[b.axis];
        });
        for (const auto& choice : choices) {
            if (choice.power > 64) continue;
            Powers product(n);
            for (size_t i : axes) product[i] = choice.power;
            const Terms shift{{product, 1}};
            Terms adapted;
            if (!translated_polynomial(adapted, f, n, choice.axis, shift)) continue;
            // The new terms on L could cancel against higher powers of x_j.
            if (std::none_of(adapted.begin(), adapted.end(), [&](const auto& term) {
                    for (size_t i = 0; i < n; ++i) if (!free[i] && term.first[i] != 0) return false;
                    return true;
                })) continue;
            Json::Value item = translation_choice(choice.axis, shift, adapted);
            item["tie"] = choice.tie;
            for (size_t i : axes) item["subspace"].append(Json::UInt64(i));
            candidates.append(std::move(item));
        }
        // The first subspace with a substitution decides this step.
        if (!candidates.empty()) break;
    }
    return candidates;
}

// A face polynomial with a repeated factor h is singular along h = 0 in its
// torus, and one with two factors is singular where both vanish. When
// h = c*x_j + m with m free of x_j, the substitution x_j = X_j - m/c turns h
// into the coordinate c*X_j, whose zeros are outside the torus. The face
// polynomial is quasihomogeneous of positive degree, so m vanishes at the
// origin. Repeated factors are proposed first.
Json::Value face_translations(const Terms& f, size_t n, const Terms& face) {
    Json::Value candidates(Json::arrayValue);
    if (face.size() < 3 || face.size() > 4096) return candidates;
    atlas::local::SparseLexRing ring(n);
    atlas::PolyValue source(ring.ctx);
    ring.write(source.p, face);
    Factors factors(ring.ctx);
    if (!fmpq_mpoly_factor(factors.v, source.p, ring.ctx)) return candidates;
    std::set<std::pair<size_t, Terms>> found;
    if (factors.v->num < 2 && (factors.v->num < 1 || fmpz_cmp_ui(factors.v->exp, 2) < 0)) return candidates;
    for (const bool repeated : {true, false})
    for (slong k = 0; k < factors.v->num; ++k) {
        if ((fmpz_cmp_ui(factors.v->exp + k, 2) >= 0) != repeated) continue;
        const Terms factor = ring.read(factors.v->poly + k);
        if (factor.size() < 2 || factor.count(Powers(n))) continue;
        for (size_t axis = 0; axis < n; ++axis) {
            Powers unit(n);
            unit[axis] = 1;
            const auto leading = factor.find(unit);
            if (leading == factor.end()) continue;
            Terms shift = factor;
            shift.erase(unit);
            if (std::any_of(shift.begin(), shift.end(), [&](const auto& term) { return term.first[axis] != 0; }))
                continue;
            for (auto& [e, c] : shift) c /= -leading->second;
            if (!found.emplace(axis, shift).second) continue;
            Terms adapted;
            if (!translated_polynomial(adapted, f, n, axis, shift)) continue;
            candidates.append(translation_choice(axis, shift, adapted));
        }
    }
    return candidates;
}

// The pullback of f by a sequence of triangular substitutions, each given by
// its axis and a shift that does not involve that axis and vanishes at the origin.
Json::Value translation_chain(Terms f, size_t n, const Json::Value& steps) {
    for (const auto& step : steps) {
        const size_t axis = step["axis"].asUInt64();
        if (axis >= n) throw std::invalid_argument("Invalid translated coordinate.");
        const Terms shift = atlas::local::decode_row(step["shift"].asString(), n);
        if (shift.empty() || shift.count(Powers(n)) ||
            std::any_of(shift.begin(), shift.end(), [&](const auto& term) { return term.first[axis] != 0; }))
            return Json::Value();
        Terms adapted;
        if (!translated_polynomial(adapted, f, n, axis, shift)) return Json::Value();
        f = std::move(adapted);
    }
    Json::Value result;
    result["adapted"] = atlas::local::encode_row(f);
    return result;
}

Json::Value decode(const Json::Value& request) {
    const auto n = request["width"].asUInt64();
    const auto op = request["operation"].asString();
    if (op == "boundary_translations")
        return boundary_translations(atlas::local::decode_row(request["row"].asString(), n), n,
                                     request["subspaces"], request["normals"]);
    if (op == "face_translations")
        return face_translations(atlas::local::decode_row(request["row"].asString(), n), n,
                                 atlas::local::decode_row(request["face"].asString(), n));
    if (op == "translation_chain")
        return translation_chain(atlas::local::decode_row(request["row"].asString(), n), n, request["steps"]);
    if (op == "quadratic_translations")
        return quadratic_translations(atlas::local::decode_row(request["row"].asString(), n), n);
    if (op == "ramification_translations")
        return ramification_translations(atlas::local::decode_row(request["row"].asString(), n), n);
    if (op=="jacobian_support" || op=="two_blowups") {
        if (!n) throw std::invalid_argument("Expected a positive number of coordinates.");
        const auto f=atlas::local::decode_row(request["row"].asString(),n);
        if (op=="jacobian_support") return jacobian_support(f,n);
        return two_blowups(f,n,request["first"].asUInt64(),request["second"].asUInt64());
    }
    if (op == "singular_torus_curve") return n == 3 && singular_torus_curve(request["row"].asString());
    if (op == "inclined_toric_obstruction")
        return atlas::certificate::trace("inclined_toric_obstruction",
                                        n == 3 && inclined_toric_obstruction(request["row"].asString()));
    if (op == "reduced_graph_tangent_obstruction") return atlas::certificate::enabled() && n==3 && repeated_tangent_obstruction(request["terms"],3,true);
    if (op == "repeated_tangent_obstruction") return n==3 && repeated_tangent_obstruction(request["terms"],3);
    if (op == "transverse_pencil") return n==3 ? transverse_pencil(request["terms"]) : Json::Value();
    if (op == "first") {
        Z first = (Z(0b1111011) << (n+1))-2;
        return first.get_str(16);
    }
    if (op != "decode") throw std::invalid_argument("Unknown resolution search operation.");
    const Z code(request["code"].asString(),16);
    Reader reader(code);
    Json::Value out;
    try {
        auto data = candidate(reader,n);
        if (reader.at == reader.bits.size()) out["candidate"] = std::move(data);
    } catch (const InvalidCode&) {
        // Every completion of the consumed prefix fails the same test.
    }
    const size_t unread = reader.bits.size()-reader.at;
    Z next = ((((code+1) >> unread)+1) << unread)-1;
    out["next"] = next.get_str(16);
    return out;
}
} // namespace

extern "C" const char* atlas_resolution_candidate(const char* input) noexcept {
    return atlas::sparse_json::invoke(input,decode);
}

extern "C" int atlas_singular_torus_curve_abi() noexcept { return 1; }
extern "C" int atlas_inclined_toric_obstruction_abi() noexcept { return 1; }

extern "C" int atlas_transverse_pencil_abi() noexcept { return 1; }

extern "C" int atlas_reduced_graph_tangent_obstruction_abi() noexcept { return 1; }

extern "C" int atlas_newton_tangent_obstruction_abi() noexcept { return 1; }
