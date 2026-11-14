// Regular binomial fans, their admissible coordinate centres and orbit incidences.
// A star subdivision at the sum of a face of a regular cone is a smooth
// blowup. Summing the two original exponent vectors on its coordinate image
// tests containment in the singular locus. The chart criterion and the dense
// orbit determining each intersection are described in engine/binomial_complex.py.
#include "groebner.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <sstream>

namespace {
using Integer = boost::multiprecision::cpp_int;
using Vector = std::vector<Integer>;
using Face = std::vector<size_t>;
using Fan = std::vector<Face>;
using Rays = std::vector<Vector>;

template<class T> T read(std::istream& in) {
    T x;
    if (!(in >> x)) throw std::invalid_argument("Invalid binomial fan data.");
    return x;
}
template<class T> std::vector<T> row(std::istream& in, size_t n) {
    std::vector<T> a(n);
    for (auto& x : a) x = read<T>(in);
    return a;
}
template<class T> void write(std::ostream& out, const std::vector<T>& a) {
    for (auto& x : a) out << ' ' << x;
}

// Dividing a Laurent segment by its primitive character gives the powers of
// one variable. Translation and projection onto pivot coordinates change a
// higher-dimensional face polynomial by a torus unit and a torus isogeny.
std::string newton_support(std::istream& in, size_t n, bool edge) {
    size_t count = read<size_t>(in), rank = read<size_t>(in);
    if (!count || !rank || rank > n || (edge && rank != 1))
        throw std::invalid_argument("Invalid Newton projection dimensions.");
    auto axes = row<size_t>(in, rank);
    if (*std::max_element(axes.begin(), axes.end()) >= n)
        throw std::invalid_argument("Invalid Newton projection coordinate.");
    Rays support;
    for (size_t i = 0; i < count; ++i) support.push_back(row<Integer>(in, n));
    std::ostringstream out;
    if (edge) {
        Vector direction(n);
        Integer length = 0;
        for (size_t j = 0; j < n; ++j) {
            direction[j] = support.back()[j] - support.front()[j];
            length = gcd(length, direction[j]);
        }
        if (!length || !direction[axes.front()]) throw std::invalid_argument("Invalid Newton edge.");
        for (auto& value : direction) value /= length;
        out << length; write(out, direction); out << '\n';
        for (const auto& exponent : support) {
            Integer k = (exponent[axes.front()]-support.front()[axes.front()])/direction[axes.front()];
            for (size_t j = 0; j < n; ++j)
                if (support.front()[j]+k*direction[j] != exponent[j])
                    throw std::invalid_argument("The support does not lie in the stated edge.");
            out << k << '\n';
        }
    } else {
        Vector minimum = support.front();
        for (const auto& e : support) for (auto j : axes) minimum[j] = std::min(minimum[j], e[j]);
        for (const auto& e : support) {
            for (auto j : axes) out << ' ' << e[j]-minimum[j];
            out << '\n';
        }
    }
    return out.str();
}

std::string face_coefficients(std::istream& in) {
    size_t count = read<size_t>(in);
    Vector numerators(count), denominators(count);
    Integer denominator = 1, content = 0;
    for (size_t i = 0; i < count; ++i) {
        numerators[i] = read<Integer>(in); denominators[i] = read<Integer>(in);
        if (denominators[i] <= 0) throw std::invalid_argument("Invalid Newton coefficient denominator.");
        denominator = lcm(denominator, denominators[i]);
        if (boost::multiprecision::msb(denominator) >= 8192) return "denominator\n";
    }
    for (size_t i = 0; i < count; ++i) {
        numerators[i] *= denominator/denominators[i];
        content = gcd(content, numerators[i]);
    }
    if (!content) throw std::invalid_argument("A Newton face must have a nonzero coefficient.");
    std::ostringstream out;
    for (auto& v : numerators) {
        v /= content;
        if (v && boost::multiprecision::msb(Integer(abs(v))) >= 8192) return "coefficient\n";
        out << v << '\n';
    }
    return out.str();
}
template<class F> void subsets(const Face& list, size_t k, F visit) {
    Face chosen;
    auto next = [&](auto&& self, size_t at) -> void {
        if (chosen.size() == k) { visit(chosen); return; }
        const size_t last = list.size() - (k - chosen.size());
        for (size_t i = at; i <= last; ++i) {
            chosen.push_back(list[i]); self(self, i+1); chosen.pop_back();
        }
    };
    if (k <= list.size()) next(next, 0);
}
Face sorted(Face c) { std::sort(c.begin(), c.end()); return c; }

struct Step { Face face, image; Integer a, b; };
struct Choice { Step step; Integer large, small; size_t incidence = 0; };
bool better(const Choice& a, const Choice& b) {
    if (a.large != b.large) return a.large > b.large;
    if (a.small != b.small) return a.small > b.small;
    if (a.incidence != b.incidence) return a.incidence < b.incidence;
    return a.step.face < b.step.face;
}
bool centre(const Fan& cones, const Fan& bad, const Rays& rays, const Vector& orders,
            const Vector& a, const Vector& b, Choice& result) {
    const size_t n=a.size();
    for (size_t k=2; k<=n; ++k) {
        std::map<Face,Choice> candidates;
        for (const auto& c : bad) subsets(c, k, [&](const Face& face) {
            if (candidates.count(face)) return;
            Integer plus=0, minus=0;
            for (auto i : face) {
                if (orders[i]>0) plus+=orders[i];
                else minus-=orders[i];
            }
            if (!plus || !minus) return;
            Step step{face,{},0,0};
            for(size_t j=0;j<n;++j) {
                if(std::any_of(face.begin(),face.end(),[&](size_t i){ return rays[i][j]!=0; })) {
                    step.image.push_back(j); step.a+=a[j]; step.b+=b[j];
                }
            }
            if (step.a<2 || step.b<2) return;
            candidates.emplace(face,Choice{step, std::max(plus,minus), std::min(plus,minus), 0});
        });
        if(candidates.empty()) continue;
        for(const auto& c:cones) subsets(sorted(c),k,[&](const Face& face){
            auto it=candidates.find(face);
            if(it!=candidates.end()) ++it->second.incidence;
        });
        result = std::min_element(candidates.begin(),candidates.end(),[](const auto& x,const auto& y){
            return better(x.second,y.second);
        })->second;
        return true;
    }
    return false;
}
int terminal(const Face& c,const Vector& orders,size_t n) {
    size_t plus=0,minus=0,p=0,m=0;
    for(size_t j=0;j<c.size();++j) {
        if(orders[c[j]]>0) {++plus; p=j;}
        if(orders[c[j]]<0) {++minus; m=j;}
    }
    if(!plus || !minus) return -1; // a unit term
    if(plus==1 && c[p]<n && orders[c[p]]==1) return static_cast<int>(p);
    if(minus==1 && c[m]<n && orders[c[m]]==-1) return static_cast<int>(m);
    return -2;
}
std::string fan(std::istream& in,size_t n) {
    const Integer max_steps=read<Integer>(in), max_cones=read<Integer>(in);
    const auto a=row<Integer>(in,n), b=row<Integer>(in,n);
    Rays rays(n,Vector(n));
    Face first(n);
    Vector orders(n);
    for(size_t i=0;i<n;++i){rays[i][i]=1; first[i]=i; orders[i]=a[i]-b[i];}
    Fan cones{first};
    std::vector<Step> steps;
    while(true) {
        Fan bad;
        std::vector<int> certificates;
        for(auto& c:cones) {
            int cert=terminal(c,orders,n);
            certificates.push_back(cert);
            if(cert==-2) bad.push_back(sorted(c));
        }
        if(bad.empty()) {
            std::ostringstream out;
            out<<rays.size()<<" "<<cones.size()<<" "<<steps.size()<<"\n";
            for(auto& r:rays){ write(out,r); out<<"\n"; }
            for(size_t ci=0;ci<cones.size();++ci) {
                out<<certificates[ci]; write(out,cones[ci]);
                for(auto r:cones[ci]) out<<" "<<orders[r];
                out<<"\n";
            }
            for(auto& s:steps) {
                out<<s.face.size(); write(out,s.face); out<<" "<<s.image.size(); write(out,s.image);
                out<<" "<<s.a<<" "<<s.b<<"\n";
            }
            return out.str();
        }
        Choice selected;
        if(steps.size()>=max_steps || !centre(cones,bad,rays,orders,a,b,selected)) return "none";
        auto& face=selected.step.face;
        if(Integer(cones.size())+Integer(selected.incidence)*(face.size()-1)>max_cones) return "none";
        size_t next=rays.size();
        Vector ray(n); Integer weight=0;
        for(auto i:face){ for(size_t j=0;j<n;++j) ray[j]+=rays[i][j]; weight+=orders[i]; }
        rays.push_back(std::move(ray)); orders.push_back(weight); steps.push_back(selected.step);
        Fan refined;
        for(const auto& c:cones) {
            const auto s=sorted(c);
            if(!std::includes(s.begin(),s.end(),face.begin(),face.end())) {refined.push_back(c);continue;}
            for(auto i:face){auto child=c; *std::find(child.begin(),child.end(),i)=next; refined.push_back(std::move(child));}
        }
        cones.swap(refined);
    }
}

using Piece = std::pair<Face, Face>;
struct ShortFirst {
    bool operator()(const Piece& a,const Piece& b) const {
        return a.first.size()!=b.first.size()?a.first.size()<b.first.size():a<b;
    }
};
struct Seen { size_t witness; bool meets; };
std::string pairings(std::istream& in, size_t n) {
    const size_t nr = read<size_t>(in), nf = read<size_t>(in);
    std::vector<Vector> forms;
    for (size_t j=0; j<nf; ++j) forms.push_back(row<Integer>(in,n));
    std::ostringstream out;
    for (size_t i=0; i<nr; ++i) {
        const auto ray = row<Integer>(in,n);
        for (const auto& f : forms)
            out << ' ' << std::inner_product(ray.begin(),ray.end(),f.begin(),Integer(0));
        out << '\n';
    }
    return out.str();
}
std::string incidence(std::istream& in,size_t n) {
    size_t nr=read<size_t>(in), nc=read<size_t>(in), ne=read<size_t>(in);
    bool intrinsic=read<int>(in)!=0;
    Vector delta=row<Integer>(in,n);
    Face exceptional=row<size_t>(in,ne);
    std::set<size_t> selected(exceptional.begin(),exceptional.end());
    Vector signs;
    Rays rays;
    for(size_t i=0;i<nr;++i) {
        rays.push_back(row<Integer>(in,n));
        signs.push_back(std::inner_product(rays.back().begin(),rays.back().end(),delta.begin(),Integer(0)));
    }
    std::map<Piece,Seen,ShortFirst> pieces;
    using Local=std::map<Face,Face>;
    std::vector<Local> charts(nc);
    for(size_t ci=0;ci<nc;++ci) {
        const Face c=row<size_t>(in,n); Face pos,neg,axes;
        for(auto r:c) {
            if(r>=nr) throw std::invalid_argument("Invalid cone ray.");
            if(signs[r]>0) pos.push_back(r);
            if(signs[r]<0) neg.push_back(r);
            if(selected.count(r)) axes.push_back(r);
        }
        std::sort(axes.begin(),axes.end());
        std::vector<bool> covered(n,false);
        for(auto r:c) if(!intrinsic || (!pos.empty()&&!neg.empty()) || signs[r]==0) {
            for(size_t j=0;j<n;++j) if(rays[r][j]!=0) covered[j]=true;
        }
        bool over=std::all_of(covered.begin(),covered.end(),[](bool b){return b;});
        for(size_t k=1;k<=axes.size();++k) subsets(axes,k,[&](const Face& s){
            Face zeros=s;
            if(intrinsic) {
                bool p=false,m=false;
                for(auto r:s){p=p||signs[r]>0;m=m||signs[r]<0;}
                if(p&&m) throw std::invalid_argument("Nonterminal chart: both binomial terms vanish on an exceptional intersection.");
                if(p||m){
                    const auto& live=p?neg:pos;
                    if(live.empty()) return;
                    if(live.size()!=1 || abs(signs[live[0]])!=1)
                        throw std::invalid_argument("Nonterminal chart: the surviving monomial is not a single linear coordinate.");
                    zeros.insert(std::lower_bound(zeros.begin(),zeros.end(),live[0]),live[0]);
                }
            }
            charts[ci][s]=zeros;
            auto it=pieces.emplace(Piece{s,zeros},Seen{ci,over}).first;
            it->second.meets |= over;
        });
    }
    std::ostringstream out;
    write(out,signs); out<<"\n";
    std::map<Piece,size_t> ids;
    for(const auto& [piece,data]:pieces) if(data.meets) {
        const auto& s=piece.first; const auto& z=piece.second;
        ids[piece]=ids.size();
        out<<"p "<<s.size(); write(out,s); out<<" "<<z.size(); write(out,z);
        if(s.size()>1) {
            auto parent=[&](const Face& subset){out<<" "<<ids.at({subset,charts[data.witness].at(subset)});};
            for(auto r:s) parent({r});
            subsets(s,s.size()-1,parent);
        }
        out<<"\n";
    }
    for(auto& chart:charts){
        out<<"c";
        for(auto& [s,z]:chart) if(s.size()==1){
            auto it=ids.find({s,z}); if(it!=ids.end()) out<<" "<<s[0]<<" "<<it->second;
        }
        out<<"\n";
    }
    return out.str();
}
}
extern "C" char* atlas_binomial(const char* data,size_t length) noexcept {
    try {
        std::istringstream in(std::string(data,length));
        const auto op=read<std::string>(in); size_t n=read<size_t>(in);
        if(n && op=="newton_edge") return atlas::copy_text(newton_support(in,n,true));
        if(n && op=="newton_project") return atlas::copy_text(newton_support(in,n,false));
        if(n && op=="newton_coefficients") return atlas::copy_text(face_coefficients(in));
        if(n<2) throw std::invalid_argument("Invalid binomial dimension.");
        if(op=="fan") return atlas::copy_text(fan(in,n));
        if(op=="incidence") return atlas::copy_text(incidence(in,n));
        if(op=="pairings") return atlas::copy_text(pairings(in,n));
        throw std::invalid_argument("Unknown binomial operation.");
    } catch(const std::exception& e){atlas::set_error(e.what());}
    catch(...){atlas::set_error("Unknown binomial failure.");}
    return nullptr;
}
extern "C" int atlas_newton_support_abi() noexcept { return 1; }
