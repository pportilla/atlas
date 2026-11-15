// Count exceptional intersections on the owned part of one affine chart.
// The projective chart's fibre, closed ownership equations and holes are distinct.
#include "component_projection.hpp"
#include "groebner.hpp"
#include "local_io.hpp"
#include "primary_decomposition.hpp"
#include "quotient_ring.hpp"
#include "rational_sparse_json.hpp"
#include <algorithm>
#include <memory>
#include <numeric>
#include <sstream>

extern "C" char* atlas_ideal(const char*,size_t) noexcept;
extern "C" void atlas_groebner_release(char*) noexcept;

namespace atlas::owned_incidence {
namespace {
using namespace local;
using V=Json::Value;
using Rows=std::vector<std::string>;
struct Unsupported {};

Rows strings(const V& list) {
    if (!list.isArray()) throw std::invalid_argument("Expected chart ideal rows.");
    Rows rows;
    for (const auto& item:list) rows.push_back(item.asString());
    return rows;
}
Rows append(Rows a,const Rows& b) { a.insert(a.end(),b.begin(),b.end());return a; }

// Reuse the packed chart verifier's precise semantics for projection to a plane,
// Euler counts and punctured loci. Its sparse row protocol is internal to C++.
std::string packed(const std::string& header,const Rows& rows) {
    std::string wire=header+"\n";
    for (const auto& row:rows) wire+=row+"\n";
    std::unique_ptr<char,decltype(&atlas_groebner_release)> out(atlas_ideal(wire.data(),wire.size()),atlas_groebner_release);
    if (!out) throw Unsupported{};
    return out.get();
}

struct Chart {
    size_t width;
    int dimension;
    Rows ambient,fibre,owned,all_holes;
    std::vector<Rows> removed,labels;
    std::vector<bool> coordinate;
    std::string hole_sizes;
    std::optional<mpz_class> field_degree;
    bool flat_ambient;
    std::vector<V> result;
    std::map<std::string,Polynomial> polys;

    explicit Chart(const V& q) : width(q["width"].asUInt()),dimension(q["dimension"].asInt()),
        ambient(strings(q["ambient"])),fibre(strings(q["fibre"])),owned(append(fibre,strings(q["region"]))),
        flat_ambient(false) {
        if (!width || dimension<0 || static_cast<size_t>(dimension)>width) throw Unsupported{};
        if (!q["degree"].isNull()) {
            field_degree=mpz_class(q["degree"].asString(),10);
            if (*field_degree<=0) throw std::invalid_argument("Invalid marked field degree.");
            flat_ambient=ambient.size()==1 && decode(ambient)==decode({q["relation"].asString()});
        }
        for (const auto& values:q["holes"]) {
            auto ideal=strings(values);
            hole_sizes+=" "+std::to_string(ideal.size());all_holes=append(std::move(all_holes),ideal);
            removed.push_back(std::move(ideal));
        }
        for (const auto& item:q["labels"]) {
            labels.push_back(strings(item));
            bool axis=false;
            if (flat_ambient && labels.back().size()==1) {
                const auto equations=decode(labels.back());
                if (equations.size()==1 && equations[0].size()==1) {
                    const auto& [e,c]=*equations[0].begin();
                    axis=c==1 && e.back()==0 && std::accumulate(e.begin(),e.end(),mpz_class(0))==1;
                }
            }
            coordinate.push_back(axis);
        }
    }

    Basis decode(const Rows& rows) {
        Basis answer;
        for (const auto& row:rows) {
            auto [it,fresh]=polys.try_emplace(row);
            if (fresh) it->second=decode_row(row,width);
            if (!it->second.empty()) answer.push_back(it->second);
        }
        return answer;
    }

    bool present(const Rows& equations) {
        if (width>MAX_VARIABLES) throw Unsupported{};
        const auto header="open_locus "+std::to_string(width)+" "+std::to_string(equations.size())+" "+
            std::to_string(removed.size())+hole_sizes;
        auto value=packed(header,append(equations,all_holes));
        if (value=="yes\n") return true;
        if (value=="no\n") return false;
        throw Unsupported{};
    }
    bool contains(const Basis& basis,const Rows& rows) {
        for (const auto& p:decode(rows)) if (!primary::member_lex(p,basis)) return false;
        return true;
    }
    bool owns(const Basis& basis) {
        if (!contains(basis,owned)) return false;
        return std::none_of(removed.begin(),removed.end(),[&](const auto& hole){return contains(basis,hole);});
    }

    // Drop the independently solved graph coordinates. If a single equation
    // remains, the component test is the product with its hypersurface.
    // No eliminated variable can occur in another graph row or in the residual.
    bool count_hypersurface(const Basis& basis,int d,V& item) {
        if (basis.size()!=width-static_cast<size_t>(d) || basis.empty()) return false;
        const auto residual=quotient_ring::unresolved_graph_rows(basis,width);
        if (residual.size()>1) return false;
        if (residual.empty()) {
            item["here"]=item["rational"]=owns(basis)?"1":"0";
            return true;
        }
        const auto factors=primary::factors(basis[residual.front()],width);
        if (factors.empty()) return false;
        for (const auto& [factor,m]:factors) {
            if (m!=1) return false;
            const auto abs=components::absolute_rational_factor(encode_row(factor),width);
            // A rejected rational irreducible factor splits geometrically:
            // its individual primes cannot all be presented over QQ.
            if (abs.isNull()) return false;
            if (!abs.asBool()) return true;
        }
        if (factors.size()==1) {
            item["here"]=item["rational"]=owns(basis)?"1":"0";
            return true;
        }
        // The general verifier additionally certifies disjointness and
        // coverage before counting the owned components of a split ideal.
        return false;
    }

    V count(const Rows& whole,const std::vector<unsigned>& subset) {
        V item;item["indices"]=V(Json::arrayValue);
        for (auto i:subset) item["indices"].append(i);
        item["here"]=item["rational"]=V();
        const auto lex=primary::reduced(decode(whole),width,true);
        bool over=contains(lex,fibre);
        if (!over) {
            if (width>=MAX_VARIABLES) throw Unsupported{};
            const auto reply=packed("radical "+std::to_string(width)+" "+std::to_string(fibre.size()),append(fibre,whole));
            if (reply!="1\n" && reply!="0\n") throw std::runtime_error("Invalid radical containment response.");
            over=reply=="1\n";
        }
        if (!over) return item;
        const int d=dimension-static_cast<int>(subset.size());
        if (d==0) {
            if (width>=MAX_VARIABLES) throw Unsupported{};
            auto equations=append(whole,owned);
            const auto reply=packed("strata "+std::to_string(width)+" "+std::to_string(equations.size())+
                " 1 0 "+std::to_string(removed.size())+hole_sizes,append(equations,all_holes));
            std::istringstream stream(reply);std::string tag;
            mpz_class total=0,c;
            while (stream>>tag && tag!="none") {
                int n,index;
                if (!(stream>>n) || n!=1 || !(stream>>index>>c) || index!=0)
                    throw std::runtime_error("Invalid point-count response.");
                total=c;
            }
            item["here"]=total.get_str();
            return item;
        }
        if (d<0) return item;
        if (flat_ambient && field_degree && std::all_of(subset.begin(),subset.end(),[&](auto j){return coordinate[j];})) {
            item["here"]=item["rational"]=owns(lex)?field_degree->get_str():"0";
            return item;
        }
        if (d==1) {
            const std::string header="plane_components "+std::to_string(width)+" "+std::to_string(whole.size())+
                " "+std::to_string(owned.size())+hole_sizes;
            const auto answer=packed(header,append(append(whole,owned),all_holes));
            if (answer!="unsupported\n") {
                std::istringstream counts(answer);mpz_class geometric,rational;
                if (!(counts>>geometric>>rational)) throw std::runtime_error("Invalid plane component counts.");
                item["here"]=geometric.get_str();item["rational"]=rational.get_str();return item;
            }
        }
        // An independent graph presents a polynomial ring on its free
        // coordinates. Its generic point belongs to the open iff its entire
        // component lies in the closed ownership set and avoids each hole.
        if (count_hypersurface(lex,d,item)) return item;
        // Remaining smooth strata may split only after a field extension
        // or require a general projection. Use the existing exact verifier.
        item["components_dimension"]=d;
        return item;
    }

    void visit(const Rows& whole,std::vector<unsigned>& subset,unsigned first) {
        for (unsigned j=first;j<labels.size();++j) {
            auto intersection=append(whole,labels[j]);
            if (!present(append(intersection,owned))) continue;
            subset.push_back(j);
            result.push_back(count(intersection,subset));
            visit(intersection,subset,j+1);
            subset.pop_back();
        }
    }

    V run() {
        std::vector<unsigned> subset;visit(ambient,subset,0);
        std::sort(result.begin(),result.end(),[](const auto& a,const auto& b) {
            const V& x=a["indices"];const V& y=b["indices"];
            return x.size()!=y.size()?x.size()<y.size():x<y;
        });
        V ordered(Json::arrayValue);
        for (auto& item:result) ordered.append(std::move(item));
        return ordered;
    }
};
V operation(const V& q) {
    if (q["operation"].asString()=="count_owned") {
        size_t count=0;
        for (const auto& flag:q["flags"]) count+=flag.asBool();
        return std::to_string(count);
    }
    if (q["operation"].asString()!="cells") throw std::invalid_argument("Unknown owned-incidence operation.");
    // A large lex eliminant or factorization may be better handled by the
    // general chart verifier. Bound this optional pass and discard partial counts.
    primary::CertificateBudget budget(16384);
    try {return Chart(q).run();}
    catch(const Unsupported&) {return V();}
    catch(const std::runtime_error&) {return V();}
}
}
}
extern "C" const char* atlas_owned_chart_incidence(const char* input) noexcept {
    return atlas::sparse_json::invoke(input,atlas::owned_incidence::operation);
}
