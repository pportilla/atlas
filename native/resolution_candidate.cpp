// Exact, prefix-skipping decoding of the exhaustive resolution certificate syntax.
#include "local_io.hpp"
#include "primary_decomposition.hpp"
#include "polynomial_factors.hpp"
#include "quotient_ring.hpp"
#include "rational_sparse_json.hpp"
#include "resolution_tangents.hpp"
#include "sparse_flint.hpp"
#include <numeric>
#include <set>

namespace {
using namespace atlas::sparse_json;
struct InvalidCode {};

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
bool repeated_tangent_obstruction(const Json::Value& rows, unsigned n) {
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
    return true;
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
        // Use translations tangent to the identity. Linear coordinate changes
        // have their own resolution paths. Bound this trial's square expansion.
        if (!quadratic || shift.empty() || shift.size() > 128 ||
            std::any_of(shift.begin(), shift.end(), [](const auto& term) {
                return std::accumulate(term.first.begin(), term.first.end(), Z(0)) < 2;
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

Json::Value decode(const Json::Value& request) {
    const auto n = request["width"].asUInt64();
    const auto op = request["operation"].asString();
    if (op == "quadratic_translations")
        return quadratic_translations(atlas::local::decode_row(request["row"].asString(), n), n);
    if (op=="jacobian_support" || op=="two_blowups") {
        if (!n) throw std::invalid_argument("Expected a positive number of coordinates.");
        const auto f=atlas::local::decode_row(request["row"].asString(),n);
        if (op=="jacobian_support") return jacobian_support(f,n);
        return two_blowups(f,n,request["first"].asUInt64(),request["second"].asUInt64());
    }
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

extern "C" int atlas_transverse_pencil_abi() noexcept { return 1; }

extern "C" int atlas_newton_tangent_obstruction_abi() noexcept { return 1; }
