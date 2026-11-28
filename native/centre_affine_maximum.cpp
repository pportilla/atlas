// A complete coefficient-presentation descent on ordinary affine charts.
// Each accepted contact is a global affine coordinate change, except the
// last one, which may be an etale univariate coordinate on the whole locus.
#include "local_io.hpp"
#include "local_division.hpp"
#include "membership_certificate.hpp"
#include "groebner.hpp"
#include "local_monomial.hpp"
#include "primary_decomposition.hpp"
#include "rational_sparse_json.hpp"
#include <algorithm>
#include <map>
#include <numeric>
#include <set>

namespace {
using atlas::local::Basis;
using atlas::local::Polynomial;
using atlas::local::Powers;
using atlas::local::max_total_degree;
namespace ideal = atlas::primary;
using V = Json::Value;
using Q = mpq_class;
struct Decline {};

void check(bool condition) { if (!condition) throw Decline{}; }
Polynomial coordinate(unsigned width,unsigned axis) {
    Powers e(width);
    e[axis]=1;
    return {{e,1}};
}
Polynomial derivative(const Polynomial& f,unsigned axis) {
    Polynomial result;
    for (const auto& [e,c] : f) if (e[axis]!=0) {
        auto d=e;
        --d[axis];
        result.emplace(std::move(d),c*e[axis]);
    }
    return result;
}
Basis append(Basis left,const Basis& right) {
    left.insert(left.end(),right.begin(),right.end());
    return left;
}
unsigned ceiling(const Q& value) {
    if (value<=0) return 0;
    mpz_class n;
    mpz_cdiv_q(n.get_mpz_t(),value.get_num_mpz_t(),value.get_den_mpz_t());
    check(n<=10);
    return n.get_ui();
}

struct Row {
    Polynomial f;
    Q weight;
    Q exceptional;
};

struct Descent {
    unsigned width;
    std::vector<unsigned> axes;
    Basis locus;
    std::vector<Basis> working;
    std::vector<Q> prefix;
    int divisor=-1;
    V birth;

    Basis read(const V& rows, size_t max_terms = 32, unsigned max_degree = 8) const {
        check(rows.isArray() && rows.size()<=20);
        Basis result;
        for (const auto& item:rows) {
            const auto s=item.asString();
            check(s.size()<=3000);
            auto f=atlas::local::decode_row(s,width);
            check(f.size()<=max_terms && max_total_degree(f)<=max_degree);
            for(const auto& term:f)
                check(mpz_sizeinbase(term.second.get_num_mpz_t(),2)<=160 &&
                      mpz_sizeinbase(term.second.get_den_mpz_t(),2)<=160);
            if(!f.empty()) result.push_back(std::move(f));
        }
        return result;
    }
    bool unit(const Basis& lex) const {
        return !lex.empty() && max_total_degree(lex.front())==0;
    }
    bool present(const Basis& extra) const {
        auto base=ideal::reduced(append(locus,extra),width,true);
        if(unit(base)) return false;
        for(const auto& w:working)
            if(!unit(ideal::reduced(append(base,w),width,true))) return true;
        return false;
    }
    // All mixed partials of order < k; scalar multiples define the same ideal.
    Basis jets(const Polynomial& f,unsigned k) const {
        Basis out;
        auto visit=[&](const auto& self,const Polynomial& g,unsigned remaining,size_t first)->void {
            if(g.empty() || !remaining) return;
            out.push_back(g);
            if(remaining>1)
                for(size_t j=first;j<axes.size();++j)
                    self(self,derivative(g,axes[j]),remaining-1,j);
        };
        visit(visit,f,k,0);
        return out;
    }
    bool old() const {
        if(birth.isNull()) return true;
        for(size_t j=0;j<std::min<size_t>(birth.size(),prefix.size());++j) {
            auto s=birth[static_cast<unsigned>(j)].asString();
            if(s=="inf") return true;
            check(s.size()<=200);
            Q a;
            check(a.set_str(s,10)==0 && a.get_den()!=0);
            a.canonicalize();
            if(a!=prefix[j]) return a>prefix[j];
        }
        return false;
    }
    V finish() const {
        auto lex=ideal::reduced(locus,width,true);
        check(!unit(lex) && ideal::dimension_lex(lex,width)==0);
        // Outside the supplied working set another frame could have selected
        // a different component. Certify all points of this closed stratum
        // occur in one of the supplied pieces before publishing its closure.
        check(std::any_of(working.begin(),working.end(),[&](const Basis& w) {
            return std::all_of(w.begin(),w.end(),[&](const Polynomial& f){return ideal::member_lex(f,lex);});
        }));
        return result(prefix, ideal::reduced(lex,width,false));
    }

    static V result(const std::vector<Q>& values, const Basis& centre) {
        V out;
        out["value"]=V(Json::arrayValue);
        for(size_t i=0;i<values.size();++i) {
            if(i%2) out["value"].append(static_cast<unsigned>(values[i].get_num().get_ui()));
            else out["value"].append(values[i].get_str());
        }
        out["value"].append("inf");
        out["ideal"]=V(Json::arrayValue);
        for(const auto& f:centre) out["ideal"].append(atlas::local::encode_row(f));
        return out;
    }

    // Keep chart coordinates for equations of the locus. Functions on the
    // contact section contain no eliminated axis, so subsequent ordinary
    // partials are exactly the derivations in the remaining coordinates.
    Polynomial restrict_to(const Polynomial& f,unsigned pivot,const Polynomial& image,
                           std::map<mpz_class,Polynomial>& powers) const {
        auto section=ideal::substitute(f,pivot,image,width-1,powers);
        Polynomial out;
        for(const auto& [e,c]:section) {
            auto v=e;
            v.insert(v.begin()+pivot,0);
            out.emplace(std::move(v),c);
        }
        return out;
    }
    V descend(std::vector<Row> rows) {
        int old_axis=-1;
        if(divisor>=0) {
            // Mixed exceptional strata, non-coordinate divisors and opens
            // continue through the general chart-cover algorithm.
            auto hyperplane=coordinate(width,divisor);
            check(ideal::member_lex(hyperplane,ideal::reduced(locus,width,true)));
            if(old()) {
                old_axis=divisor;
                divisor=-1;
                for(auto& row:rows) if(row.exceptional!=0) {
                    auto multiple=row.exceptional.get_den();
                    // Bound the degree before materializing either power.
                    // In at most three axes this permits at most 286 terms.
                    check(multiple<=8 && max_total_degree(row.f)*multiple+row.exceptional.get_num()<=10);
                    row.f=ideal::multiply(ideal::power(row.f,multiple,width),
                                          ideal::power(hyperplane,row.exceptional.get_num(),width));
                    row.weight*=multiple;
                    row.exceptional=0;
                }
            }
        }
        prefix.emplace_back(old_axis>=0 ? 1 : 0);
        for(const auto& row:rows) {
            if(row.exceptional!=0 || row.weight.get_den()!=1) continue;
            unsigned weight=ceiling(row.weight);
            for(auto axis:axes) {
                if(static_cast<int>(axis)==divisor) continue;
                auto contact=row.f;
                for(unsigned j=1;j<weight;++j) contact=derivative(contact,axis);
                auto top=derivative(contact,axis);
                if(top.empty()) continue;
                if(axes.size()==1) {
                    auto candidate=append(locus,{contact});
                    if(!unit(ideal::reduced(append(candidate,{top}),width,true))) continue;
                    auto lex=ideal::reduced(candidate,width,true);
                    // On a smooth zero-dimensional section each coefficient
                    // vanishes identically near its points iff its value is zero.
                    bool vanishes=true;
                    for(const auto& r:rows) for(const auto& p:jets(r.f,ceiling(r.weight)))
                        vanishes &= ideal::member_lex(p,lex);
                    if(!vanishes) continue;
                    locus=std::move(candidate);
                    return finish();
                }
                // BM coefficient presentations give the same invariant and
                // maximum stratum on every valid maximal-contact section.
                // A scalar top derivative certifies this section on the
                // whole affine chart, even if a preceding direction gives
                // an implicit section on smaller opens. The final etale
                // coordinate cuts a reduced finite ideal in either frame.
                if(max_total_degree(contact)!=1 || max_total_degree(top)!=0) continue;
                Polynomial image;
                for(const auto& [e,c]:contact) if(e[axis]==0) {
                    auto d=e;
                    d.erase(d.begin()+axis);
                    image.emplace(std::move(d),-c/top.begin()->second);
                }
                std::map<mpz_class,Polynomial> powers;
                std::vector<Row> next;
                for(const auto& r:rows) {
                    auto g=r.f;
                    for(unsigned j=0;j<ceiling(r.weight) && !g.empty();++j) {
                        auto p=restrict_to(g,axis,image,powers);
                        if(!p.empty()) next.push_back({std::move(p),r.weight-j,r.exceptional});
                        g=derivative(g,axis);
                    }
                }
                if(old_axis>=0) {
                    auto p=restrict_to(coordinate(width,old_axis),axis,image,powers);
                    if(!p.empty()) next.push_back({std::move(p),1,0});
                }
                locus.push_back(std::move(contact));
                axes.erase(std::find(axes.begin(),axes.end(),axis));
                check(!next.empty()); // Positive-dimensional contact uses the general frame closure.
                return residual(std::move(next));
            }
        }
        throw Decline{};
    }
    V residual(std::vector<Row> rows) {
        Q common=0;
        if(divisor>=0) {
            for(size_t i=0;i<rows.size();++i) {
                auto& r=rows[i];
                auto order=r.f.begin()->first[divisor];
                for(const auto& [e,c]:r.f) order=std::min(order,e[divisor]);
                if(order!=0) {
                    Polynomial normal;
                    for(const auto& [e,c]:r.f) {
                        auto d=e;
                        d[divisor]-=order;
                        normal.emplace(std::move(d),c);
                    }
                    r.f=std::move(normal);
                    r.exceptional+=order;
                }
                Q ratio=r.exceptional/r.weight;
                if(i==0 || ratio<common) common=ratio;
            }
            for(auto& r:rows) r.exceptional-=common*r.weight;
        }
        std::set<Q> grid;
        for(const auto& r:rows) {
            auto bound=max_total_degree(r.f);
            check(bound<=10);
            for(unsigned k=0;k<=bound.get_ui();++k) grid.insert((r.exceptional+k)/r.weight);
        }
        auto conditions=[&](const Q& value) {
            Basis result;
            for(const auto& r:rows)
                result=append(std::move(result),jets(r.f,ceiling(value*r.weight-r.exceptional)));
            return result;
        };
        const std::vector<Q> values(grid.begin(),grid.end());
        size_t lo=0,hi=values.size();
        while(lo<hi) {
            auto mid=lo+(hi-lo)/2;
            if(present(conditions(values[mid]))) lo=mid+1;
            else hi=mid;
        }
        check(lo && values[lo-1]>0); // Monomial endgame uses the full birth ordering.
        auto value=values[lo-1];
        locus=ideal::reduced(append(locus,conditions(value)),width,true);
        prefix.push_back(value);
        for(auto& r:rows) r.weight*=value;
        if(value<1) rows.push_back({{{Powers(width),1}},1-value,common});
        return descend(std::move(rows));
    }
    // Write f = sum f_j(s,e) t^j with f|E=c t^d. The contact equation
    // (d-1)!^-1 ∂_t^(d-1)f=0 gives t = f_{d-1} W near every point of E:
    // the coefficient multiplying t restricts to the nonzero scalar d c.
    std::optional<Q> pure_contact_order(const Polynomial& f, unsigned d, unsigned t, unsigned s,
                                        bool point_only) const {
        const unsigned bound = max_total_degree(f).get_ui(); // read() caps this before allocation.
        const unsigned absent = bound + 1;
        std::vector<unsigned> exceptional(bound + 1, absent), total(bound + 1, absent);
        for (const auto& [e, c] : f) {
            const unsigned j = e[t].get_ui(), boundary = e[divisor].get_ui();
            exceptional[j] = std::min(exceptional[j], boundary);
            total[j] = std::min(total[j], boundary + unsigned(e[s].get_ui()));
        }
        const unsigned m = exceptional[0];
        if (m == absent || (!point_only && m >= d)) return {};
        // For working sets contained in E, ∂_t^d f is a nonzero constant on
        // E. If m<d, the order-d locus there satisfies t=0 and
        // ∂_e^m f|{t=e=0}=m! P(s)=0, where P is the leading coefficient
        // of f_0 in e. Its roots can include nonrational working points.
        Polynomial leading;
        for (const auto& [e, c] : f)
            if (e[t] == 0 && e[divisor] == m) leading[{e[s]}] = c;
        const unsigned a = leading.begin()->first.front().get_ui();
        if (a < d) return {};
        // On E the order of the normalized C_0 at a point is bounded
        // above by the multiplicity of that root of P. Its derivatives
        // through order a-1 must cut out the reduced origin globally.
        if (leading.size() > 1) {
            Basis derivatives{leading};
            for (unsigned i = 1; i < a; ++i) {
                atlas::check_deadline();
                derivatives.push_back(derivative(derivatives.back(), 0));
            }
            const auto common = atlas::local::polynomial_gcd(derivatives, 1);
            if (!common || common->size() != 1 || common->begin()->first.front() != 1) return {};
        }
        // The mark d-i coefficient C_i on contact is the sum of
        // binom(j,i) f_j (f_{d-1} W)^(j-i). Bound every term so no C_i
        // can lower either minimum. For C_0 insist on strict exceptional
        // inequality in the tail: e^-m C_0|E is then exactly P(s).
        for (unsigned i = 0; i < d - 1; ++i) {
            atlas::check_deadline();
            for (unsigned j = i; j <= bound; ++j) {
                if (exceptional[j] == absent || (j != i && exceptional[d - 1] == absent)) continue;
                const unsigned e = exceptional[j] + (j - i) * exceptional[d - 1];
                const unsigned ord = total[j] + (j - i) * total[d - 1];
                if ((i == 0 && j > 0 && e <= m) ||
                    d * e < (d - i) * m || d * ord < (d - i) * (m + a)) return {};
            }
        }
        return Q(a) / d;
    }

    // The degree-d tangent form on the old (or soon old) divisor E needs
    // two independent transverse parameters. After one maximal contact,
    // at least one coefficient on E still has order equal to its mark:
    // otherwise the binary form would be a pure d-th power. This gives
    // residual order one and prevents any common exceptional factor.
    // The two transverse parameters and the mark-one equation of E give
    // three contacts in all, each subsequent residual order being one.
    std::optional<V> transverse_origin(const Polynomial& f) const {
        if (width != 3 || divisor < 0 || working.empty() || f.empty()) return {};
        atlas::check_deadline();
        auto unknown = [](const char* site) -> std::optional<V> {
            atlas::certificate::trace(site, false);
            return {};
        };
        // All working points must be on E and at least one piece must
        // contain the origin. Pure powers suffice even on a thick fibre.
        bool origin_present = false, point_only = true;
        for (const auto& piece : working) {
            std::vector<bool> fixed(width);
            bool contains_origin = true;
            for (const auto& g : piece) {
                if (g.count(Powers(width))) contains_origin = false;
                if (g.size() != 1) continue;
                const auto& e = g.begin()->first;
                for (unsigned i = 0; i < width; ++i)
                    if (e[i] > 0 && e[(i + 1) % 3] == 0 && e[(i + 2) % 3] == 0) fixed[i] = true;
            }
            if (!fixed[divisor]) return unknown("transverse_working_boundary");
            origin_present |= contains_origin;
            point_only &= std::find(fixed.begin(), fixed.end(), false) == fixed.end();
        }
        if (!origin_present) return unknown("transverse_working_origin");
        mpz_class order = atlas::local::total_degree(f.begin()->first);
        for (const auto& [e, c] : f) order = std::min(order, atlas::local::total_degree(e));
        if (order < 2 || order > 8) return unknown("transverse_order_range");
        const unsigned d = order.get_ui(), u = (divisor + 1) % width;
        std::vector<Q> coefficients(d + 1);
        for (const auto& [e, c] : f) if (e[divisor] == 0) {
            // With this condition the order-(d-1) partials on E are
            // globally linear, not merely linear modulo the maximal ideal.
            if (atlas::local::total_degree(e) != order) return unknown("transverse_linear_boundary_derivatives");
            coefficients[e[u].get_ui()] = c;
        }
        // Up to a nonzero factorial these are the two coefficients of
        // each order-(d-1) derivative of the binary boundary form.
        Q a0, b0;
        bool spans = false;
        for (unsigned j = 0; j < d; ++j) {
            Q a = (j + 1) * coefficients[j + 1], b = (d - j) * coefficients[j];
            if (a0 * b != b0 * a) { spans = true; break; }
            if (a != 0 || b != 0) { a0 = a; b0 = b; }
        }
        Q residual = 1;
        if (!spans) {
            unsigned nonzero = 0, exponent = 0;
            for (unsigned j = 0; j <= d; ++j)
                if (coefficients[j] != 0) { ++nonzero; exponent = j; }
            if (nonzero != 1 || (exponent != 0 && exponent != d)) return unknown("transverse_boundary_rank");
            const unsigned v = (u + 1) % width;
            const unsigned t = exponent ? u : v;
            auto value = pure_contact_order(f, d, t, t == u ? v : u, point_only);
            if (!value) return unknown("pure_contact_residual");
            residual = *value;
        }
        Descent history;
        history.birth = birth;
        history.prefix = {Q(d)};
        bool first = history.old();
        history.prefix.push_back(first ? 1 : 0);
        history.prefix.push_back(residual);
        if ((!spans && first) || (!first && !history.old())) return unknown("transverse_divisor_history");
        // For full tangent rank the boundary partials generate both
        // transverse coordinates. For a pure contact the centre is in
        // E at level two: there ∂_t^(d-1)f gives t and the derivatives of
        // the residual through order a-1 generate (s). Its last mark-one
        // contact with E gives residual order one and the reduced origin.
        Basis point;
        for (unsigned i = 0; i < width; ++i) point.push_back(coordinate(width, i));
        atlas::check_deadline();
        atlas::certificate::trace(spans ? "transverse_origin" : "pure_contact_origin", true);
        return result({Q(d), first ? 1 : 0, residual, first ? 0 : 1, 1, 0},
                      ideal::reduced(point, width, false));
    }

    // At a double point on E in a threefold, a nonsingular quadratic
    // section on T(E) supplies two independent contact directions. An old E
    // supplies the third, with mark one. The coefficient presentation has
    // residual orders 1, 1: after the first section its transverse square
    // still has mark two, and after the second the remaining parameter has
    // mark one (or a square of mark two). There is no common E factor.
    // Once all parameters are used the coefficient ideal is zero. Higher
    // Taylor terms cannot affect any of these orders. If E has not become
    // old by level two, higher coefficients along its normal *can* matter.
    std::optional<V> transverse_double(const Polynomial& f) const {
        if (width != 3 || divisor < 0 || working.empty()) return {};
        atlas::check_deadline();
        for (const auto& w : working)
            if (w.size() != width || std::any_of(w.begin(), w.end(), [](const auto& g) {
                    return max_total_degree(g) != 1;
                })) return {};
        const Basis point_ideal = ideal::reduced(working.front(), width, true);
        if (point_ideal.size() != width) return {};
        std::vector<Q> point(width);
        std::vector<bool> fixed(width);
        for (const auto& row : point_ideal) {
            if (row.empty() || max_total_degree(row) != 1) return {};
            auto lead = row.rbegin();
            const auto& e = lead->first;
            auto pos = std::find(e.begin(), e.end(), 1);
            if (pos == e.end() || row.size() > 1 + row.count(Powers(width))) return {};
            size_t axis = pos - e.begin();
            if (fixed[axis]) return {};
            fixed[axis] = true;
            auto at = row.find(Powers(width));
            if (at != row.end()) point[axis] = -at->second / lead->second;
        }
        if (point[divisor] != 0) return {};
        for (const auto& w : working) {
            if (w != working.front() && ideal::reduced(w, width, true) != point_ideal) return {};
        }

        const auto jet = atlas::local::point_jet_below(3, f, point);
        if (jet.count(Powers(width))) return {};
        for (size_t i = 0; i < width; ++i)
            if (jet.count(coordinate(width, i).begin()->first)) return {};
        const size_t x = (divisor + 1) % width, y = (divisor + 2) % width;
        auto term = [&](unsigned a, unsigned b) -> Q {
            Powers e(width);
            e[x] = a;
            e[y] = b;
            auto it = jet.find(e);
            return it == jet.end() ? Q(0) : it->second;
        };
        if (4 * term(2, 0) * term(0, 2) == term(1, 1) * term(1, 1)) return {};

        Descent history;
        history.birth = birth;
        history.prefix = {2};
        bool first = history.old();
        history.prefix.push_back(first ? 1 : 0);
        history.prefix.push_back(1);
        const bool second = !first && history.old();
        if (!first && !second) {
            // If the entire Jacobian ideal is an affine line, f vanishes
            // to order two along that line (in linear coordinates the
            // constant and linear normal coefficients vanish). Its two
            // transverse contact parameters cut out this same smooth line
            // near p. All remaining coefficient functions there are zero.
            // The closure is the entire line: no other component exists.
            // A readily checked family of parallel lines suffices for this
            // proposal. The untouched divisor coordinate is its parameter.
            // Bound the translation box by the input limits above.
            auto shifted = atlas::local::translate({f}, point);
            for (const auto& term : shifted[0])
                if (term.first[x] + term.first[y] < 2) return {};
            Basis jacobian{f};
            for (unsigned i = 0; i < width; ++i) jacobian.push_back(derivative(f, i));
            const auto line = ideal::reduced(jacobian, width, false);
            if (line.size() != 2 || std::any_of(line.begin(), line.end(), [](const Polynomial& g) {
                    return max_total_degree(g) != 1;
                })) return {};
            atlas::check_deadline();
            return result({2, 0, 1, 0}, line);
        }

        // The centre is in the *whole chart*. Its ideal contains E and
        // the first jet of f. Prove these already generate the reduced
        // maximal ideal globally, so lifting the jet answer adds no other
        // component. The opposite inclusion follows from evaluation at p.
        Basis locus_on_boundary{coordinate(width, divisor)};
        auto restrict = [&](Polynomial g) {
            for (auto it = g.begin(); it != g.end();)
                if (it->first[divisor] != 0) it = g.erase(it);
                else ++it;
            if (!g.empty()) locus_on_boundary.push_back(std::move(g));
        };
        restrict(f);
        for (unsigned j = 0; j < width; ++j) restrict(derivative(f, j));
        if (ideal::reduced(locus_on_boundary, width, true) != point_ideal) return {};
        atlas::check_deadline();
        return result({2, first ? 1 : 0, 1, first ? 0 : 1, 1, 0}, ideal::reduced(point_ideal, width, false));
    }

    V run(const V& input) {
        width=input["width"].asUInt();
        check(input["operation"]=="maximum" && width>0 && width<=3 &&
              input["working"].isArray() && input["working"].size()<=8);
        for(const auto& w:input["working"]) working.push_back(read(w));
        if(!input["divisor"].isNull()) {
            auto eq=read(input["divisor"]);
            check(eq.size()==1);
            for(unsigned j=0;j<width;++j) if(eq.front()==coordinate(width,j)) divisor=j;
            check(divisor>=0);
            birth=input["birth"];
            check(birth.isNull() || (birth.isArray() && birth.size()<=16));
        }
        if (divisor >= 0 && atlas::certificate::enabled()) {
            try {
                auto strict = read(input["strict"], 64, 32);
                if (strict.size() == 1) {
                    // The sparse proof and its univariate gcd are bounded by
                    // the input caps and use the caller's deadline.
                    auto point = transverse_origin(strict[0]);
                    atlas::check_deadline();
                    if (point) return *point;
                    if (max_total_degree(strict[0]) <= 12) {
                        // The general double-point closure uses a global
                        // Gröbner basis. Limit that optional computation.
                        atlas::DeadlineScope trial(std::chrono::steady_clock::now() + std::chrono::milliseconds(12));
                        auto result = transverse_double(strict[0]);
                        if (result && !trial.expired()) return *result;
                    }
                }
            } catch (const Decline&) {
                // A proposal outside its caps still gets the original descent.
            } catch (const atlas::Expired&) {
            } catch (const std::runtime_error&) {
                // A bounded basis can decline through the packed C interface.
            }
            atlas::check_deadline();
        }
        atlas::primary::CertificateBudget budget(16000);
        auto strict=read(input["strict"]);
        check(strict.size()==1);
        axes.resize(width);
        std::iota(axes.begin(),axes.end(),0);
        auto& f=strict.front();
        check(present({f}));
        unsigned order=1;
        while(order<=max_total_degree(f).get_ui() && present(jets(f,order+1))) ++order;
        check(order>=2); // Smooth and purely boundary cases use the ordinary chart chooser.
        locus=ideal::reduced(jets(f,order),width,true);
        prefix.emplace_back(order);
        return descend({{f,Q(order),0}});
    }
};
}

extern "C" const char* atlas_affine_centre_maximum(const char* input) noexcept {
    return atlas::sparse_json::invoke(input,[](const V& r)->V {
        try {
            return Descent{}.run(r);
        } catch(const Decline&) { return {}; }
          catch(const std::runtime_error&) { return {}; } // Limited Groebner or coefficient work.
    });
}
