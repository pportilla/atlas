// A complete coefficient-presentation descent on ordinary affine charts.
// Each accepted contact is a global affine coordinate change, except the
// last one, which may be an etale univariate coordinate on the whole locus.
#include "local_io.hpp"
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

    Basis read(const V& rows) const {
        check(rows.isArray() && rows.size()<=20);
        Basis result;
        for (const auto& item:rows) {
            const auto s=item.asString();
            check(s.size()<=3000);
            auto f=atlas::local::decode_row(s,width);
            check(f.size()<=32 && max_total_degree(f)<=8);
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
        V out;
        out["value"]=V(Json::arrayValue);
        for(size_t i=0;i<prefix.size();++i) {
            if(i%2) out["value"].append(static_cast<unsigned>(prefix[i].get_num().get_ui()));
            else out["value"].append(prefix[i].get_str());
        }
        out["value"].append("inf");
        out["ideal"]=V(Json::arrayValue);
        for(const auto& f:ideal::reduced(lex,width,false)) out["ideal"].append(atlas::local::encode_row(f));
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
    V run(const V& input) {
        width=input["width"].asUInt();
        check(input["operation"]=="maximum" && width>0 && width<=3 &&
              input["working"].isArray() && input["working"].size()<=8);
        auto strict=read(input["strict"]);
        check(strict.size()==1);
        for(const auto& w:input["working"]) working.push_back(read(w));
        if(!input["divisor"].isNull()) {
            auto eq=read(input["divisor"]);
            check(eq.size()==1);
            for(unsigned j=0;j<width;++j) if(eq.front()==coordinate(width,j)) divisor=j;
            check(divisor>=0);
            birth=input["birth"];
            check(birth.isNull() || (birth.isArray() && birth.size()<=16));
        }
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
            atlas::primary::CertificateBudget budget(16000);
            return Descent{}.run(r);
        } catch(const Decline&) { return {}; }
          catch(const std::runtime_error&) { return {}; } // Limited Groebner or coefficient work.
    });
}
