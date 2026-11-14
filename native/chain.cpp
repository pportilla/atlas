// Resolution of Q(X,Y) + c Z^k for a nondegenerate binary quadratic Q.
// In split coordinates it is uv+z^k. Every coordinate map is monomial in
// u,v,z. Columns of the exponent matrix are the primitive chart rays.
#include <gmpxx.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {
using Triple = std::array<int, 3>;
using Polynomial = std::map<Triple, mpq_class>;
const char* names[] = {"a","b","c"};
const std::array<Triple,3> identity = {{{1,0,0},{0,1,0},{0,0,1}}};

struct Coefficient {
    mpq_class rational = 0;
    mpq_class radical = 0;  // coefficient of theta
};
struct Form {
    int k = 0;
    int x = 0, y = 1, z = 2;
    mpq_class unit = 1, discriminant = 0;
    // X and Y as linear combinations of u and v (Z=z).
    Coefficient xu, xv, yu, yv;
};

Form normalize(const Polynomial& terms) {
    for (int z=0; z<3; ++z) {
        Form f;
        f.z=z; f.x=(z+1)%3; f.y=(z+2)%3;
        mpq_class A=0, B=0, C=0;
        bool valid=true;
        for (const auto& [e, val]: terms) {
            if (e[z]) {
                if (e[f.x] || e[f.y] || e[z]<2 || f.k) { valid=false; break; }
                f.k=e[z]; f.unit=val;
            } else if (e[f.x]==2 && !e[f.y]) A=val;
            else if (e[f.x]==1 && e[f.y]==1) B=val;
            else if (!e[f.x] && e[f.y]==2) C=val;
            else { valid=false; break; }
        }
        if (!valid || !f.k || B*B==4*A*C) continue;
        if (!A && !C) {
            f.xu.rational=1; f.yv.rational=f.unit/B;
            return f;
        }
        if (!A) {std::swap(A,C); std::swap(f.x,f.y);}
        const mpq_class D=B*B-4*A*C;
        // u=X+(B+sqrt(D))/(2A)Y, hv=X+(B-sqrt(D))/(2A)Y.
        const mpq_class h=f.unit/A;
        f.xu={mpq_class(1,2),-B/(2*D)};
        f.xv={h/2,h*B/(2*D)};
        f.yu={0,A/D};
        f.yv={0,-h*A/D};
        if (D>0 && mpz_perfect_square_p(D.get_num_mpz_t())
                && mpz_perfect_square_p(D.get_den_mpz_t())) {
            mpz_class num,den;
            mpz_sqrt(num.get_mpz_t(),D.get_num_mpz_t());
            mpz_sqrt(den.get_mpz_t(),D.get_den_mpz_t());
            const mpq_class squareRoot(num,den);
            for (auto coeff:{&f.xu,&f.xv,&f.yu,&f.yv}) {
                coeff->rational+=coeff->radical*squareRoot; coeff->radical=0;
            }
        } else f.discriminant=D;
        return f;
    }
    return {};
}

std::string coefficient(const Coefficient& v) {
    if (!v.radical) return v.rational.get_str();
    const auto tail=(v.radical==1 ? "theta" : v.radical==-1 ? "-theta" :
                    v.radical.get_str()+"*theta");
    if (!v.rational) return tail;
    return v.rational.get_str()+(v.radical>0?"+":"")+tail;
}
std::string monomial(Triple powers) {
    std::string s;
    for (int p=0;p<3;++p) if (powers[p]) {
        if (!s.empty()) s+="*";
        s+=names[p];
        if (powers[p]>1) s+="^"+std::to_string(powers[p]);
    }
    return s.empty() ? "1" : s;
}
std::string term(const Coefficient& coeff,const std::string& mon) {
    auto c=coefficient(coeff);
    if (c=="0") return "";
    if (c=="1") return mon;
    if (c=="-1") return "-"+mon;
    if (coeff.radical && coeff.rational) c="("+c+")";
    return c+"*"+mon;
}
std::string linear(const Coefficient& a,const std::string& x,
                   const Coefficient& b,const std::string& y) {
    auto one=term(a,x),two=term(b,y);
    if (one.empty()) return two;
    if (two.empty()) return one;
    return one+(two[0]=='-'?"":"+")+two;
}

struct Chart {
    std::string id, parent, step;
    std::array<Triple,3> rays;
    std::array<std::string,3> parentMap;
    bool leaf=true;
    std::vector<int> center;
};

void subdivide(std::vector<Chart>& atlas,int index,const std::vector<int>& axes,
               const std::string& step) {
    atlas[index].leaf=false;
    const auto parent=atlas[index];
    Triple sum{};
    for(int p:axes) for(int j=0;j<3;++j) sum[j]+=parent.rays[p][j];
    for(int p:axes) {
        Chart child=parent;
        child.id=step+"."+names[p]+"."+std::to_string(index);
        child.leaf=true; child.parent=parent.id; child.step=step;
        child.center = axes;
        child.rays[p]=sum;
        for(int q=0;q<3;++q) child.parentMap[q]=names[q];
        for(int q:axes) if(q!=p) child.parentMap[q]=std::string(names[p])+"*"+names[q];
        atlas.push_back(child);
    }
}

std::string render(const Polynomial& p,bool intrinsic) {
    const Form f=normalize(p);
    if (!f.k) return "defer\n";
    const int j=f.k/2;
    std::vector<Chart> atlas={{"N","","",identity,{"a","b","c"},true,{}}};
    int spine=0;
    for(int i=1;i<=j;++i) {
        subdivide(atlas,spine,{0,1,2},"P"+std::to_string(i));
        spine=static_cast<int>(atlas.size())-1;
    }
    // For odd k the last z-chart has S=(ab+c), E=(c). Blow up the
    // projective line cut out by the U direction on E, with ideal (a,c)
    // here. On S it is Cartier; an intrinsic resolution can stop before it.
    if (!intrinsic && f.k%2) {
        int total=static_cast<int>(atlas.size());
        for(int i=0;i<total;++i) if(atlas[i].leaf) {
            std::vector<int> axes;
            for(int q=0;q<3;++q) {
                if(atlas[i].rays[q]==identity[0] || atlas[i].rays[q]==Triple{j,j,1})
                    axes.push_back(q);
            }
            if(axes.size()==2) subdivide(atlas,i,axes,"L");
        }
    }
    std::ostringstream out;
    out<<"A\t"<<f.k<<'\t'<<f.unit<<'\t'<<f.discriminant<<'\t'<<f.k-1<<'\n';
    using Face = std::vector<std::string>;
    std::set<std::pair<size_t, Face>> cells;
    std::set<std::string> curves;
    for(const auto& chart:atlas) {
        out<<"chart\t"<<chart.id<<'\t'<<chart.parent<<'\t'<<chart.step<<'\t'<<chart.leaf<<'\n';
        if (!chart.center.empty()) {
            out<<"center";
            for (int coordinate : chart.center) out<<'\t'<<names[coordinate];
            out<<'\n';
        }
        Face exceptional;
        Triple ue{}, ve{}, ze{}, common{}, left{}, right{};
        for(int q=0;q<3;++q) {
            ue[q]=chart.rays[q][0]; ve[q]=chart.rays[q][1]; ze[q]=chart.rays[q][2];
            common[q]=std::min(ue[q]+ve[q],f.k*ze[q]);
            left[q]=ue[q]+ve[q]-common[q]; right[q]=f.k*ze[q]-common[q];
        }
        std::array<std::string,3> baseMap;
        auto u=monomial(ue),v=monomial(ve);
        baseMap[f.x]=linear(f.xu,u,f.xv,v);
        baseMap[f.y]=linear(f.yu,u,f.yv,v);
        baseMap[f.z]=monomial(ze);
        for(int q=0;q<3;++q) out<<"map\t"<<baseMap[q]<<'\n';
        for(const auto& ray:chart.rays)
            out<<"ray\t"<<ray[0]<<'\t'<<ray[1]<<'\t'<<ray[2]<<'\n';
        for(const auto& expr:chart.parentMap) out<<"parentmap\t"<<expr<<'\n';
        out<<"strict\t"<<monomial(left)<<"+"<<monomial(right)<<'\n';
        out<<"factor\t"<<monomial(common)<<'\n';
        for(int q=0;q<3;++q) {
            if(std::find(identity.begin(),identity.end(),chart.rays[q])!=identity.end()) continue;
            const auto& ray=chart.rays[q];
            int i=ray[1];
            bool finalLine=ray[0]!=ray[1];
            std::string e="E"+std::to_string(i+int(finalLine));
            exceptional.push_back(e);
            out<<"ambient\t"<<e<<'\t'<<names[q]<<'\t'<<common[q]<<'\n';
            if(!intrinsic) continue;
            if(2*i==f.k) {
                out<<"curve\t"<<e<<"\t0\t-2\t"<<names[q]<<'\n';
                if (chart.leaf) curves.insert(e);
            }
            else {
                // On E_i the uv term of the residual binomial vanishes.
                // Its remaining factors are the U and/or V coordinate rays.
                // U=0 is the branch with relatively larger valuation of u.
                for(int h=0;h<3;++h) if(left[h]) {
                    std::string side=chart.rays[h]==identity[0]?"v":"u";
                    out<<"curve\t"<<e<<side<<"\t0\t-2\t"<<names[q]<<'\t'<<names[h]<<'\n';
                    if (chart.leaf) curves.insert(e+side);
                }
            }
        }
        if (!intrinsic && chart.leaf) {
            std::sort(exceptional.begin(), exceptional.end());
            for (size_t a=0; a<exceptional.size(); ++a)
                for (size_t b=a+1; b<exceptional.size(); ++b)
                    cells.insert({1, {exceptional[a],exceptional[b]}});
            if (exceptional.size()==3) cells.insert({2,exceptional});
        }
    }
    if (intrinsic) {
        Face chain;
        for (int i=1; i<(f.k+1)/2; ++i) chain.push_back("E"+std::to_string(i)+"u");
        if (f.k%2==0) chain.push_back("E"+std::to_string(j));
        for (int i=(f.k+1)/2-1; i>=1; --i) chain.push_back("E"+std::to_string(i)+"v");
        if (curves != std::set<std::string>(chain.begin(), chain.end()))
            throw std::logic_error("Curve labels do not cover the exceptional chain.");
        for (size_t i=1; i<chain.size(); ++i) {
            const auto ends=std::minmax(chain[i-1],chain[i]);
            cells.insert({1, {ends.first,ends.second}});
        }
    }
    for (const auto& [dimension, face] : cells) {
        out<<"cell\t"<<dimension;
        for(const auto& vertex : face) out<<'\t'<<vertex;
        out<<'\n';
    }
    return out.str();
}
} // namespace

extern "C" char* chain_resolve(const char* input,int intrinsic) noexcept {
    try {
        std::istringstream src(input?input:"");
        int n;
        if (!(src>>n) || n<2 || n>4) return nullptr;
        Polynomial p;
        for(int i=0;i<n;++i) {
            std::string c;
            Triple e{};
            if(!(src>>c>>e[0]>>e[1]>>e[2]) || c.size()>8192) return nullptr;
            if(*std::max_element(e.begin(),e.end())>128 || *std::min_element(e.begin(),e.end())<0)
                return nullptr;
            mpq_class value;
            if(value.set_str(c,10)!=0 || !value.get_den() || !value) return nullptr;
            value.canonicalize();
            if(!p.emplace(e,value).second) return nullptr;
        }
        const auto result=render(p,intrinsic!=0);
        char* data=static_cast<char*>(std::malloc(result.size()+1));
        if(data) std::memcpy(data,result.c_str(),result.size()+1);
        return data;
    } catch (...) {return nullptr;}
}
extern "C" void chain_free(void* p) {std::free(p);}
