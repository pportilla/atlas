// Standard smooth opens and graph transforms in QQ[x] and K[x], K=QQ[theta]/m.
// Width and exponents are unbounded by the packed verifier's representation.
#include "local_io.hpp"
#include "primary_decomposition.hpp"
#include "rational_sparse_json.hpp"
#include <algorithm>
#include <memory>
#include <numeric>
#include <sstream>

extern "C" char* atlas_affine(const char*,size_t) noexcept;
extern "C" char* atlas_wide_block(const char*,size_t) noexcept;
extern "C" void atlas_groebner_release(char*) noexcept;
extern "C" const char* atlas_groebner_error() noexcept;

namespace atlas::graph_wide {
namespace {
using namespace local;
using J=Json::Value;
using Indices=std::vector<size_t>;

Basis join(Basis a,const Basis& b) {
    a.insert(a.end(),b.begin(),b.end());
    return a;
}

template<class Visit>
void choices(size_t n,size_t k,Visit visit) {
    if (k>n) return;
    Indices subset(k);
    std::iota(subset.begin(),subset.end(),0);
    for (;;) {
        if (!visit(subset)) return;
        size_t i=k;
        while (i && subset[i-1]==n-k+i-1) --i;
        if (!i) return;
        ++subset[i-1];
        for (;i<k;++i) subset[i]=subset[i-1]+1;
    }
}

struct Chart {
    size_t axes,width;
    Polynomial field,one;
    std::vector<std::string> names;

    explicit Chart(const J& q) : axes(q["width"].asUInt()),width(axes+!q["relation"].isNull()),one{{Powers(width),1}} {
        if (!axes) throw std::invalid_argument("Expected a nonempty chart.");
        if (width>axes) field=decode_row(q["relation"].asString(),width);
        for (size_t i=0;i<width;++i) names.push_back("x"+std::to_string(i));
    }
    Basis rows(const J& data) const {
        Basis g;
        for (const auto& str:data) {
            auto f=decode_row(str.asString(),width);
            if (!f.empty() && std::find(g.begin(),g.end(),f)==g.end()) g.push_back(std::move(f));
        }
        return g;
    }
    J encoded(const Basis& data) const {
        J rows(Json::arrayValue);
        for (const auto& f:data) rows.append(encode_row(f));
        return rows;
    }
    Basis lift(Basis rows) const {
        if (!field.empty()) rows.push_back(field);
        return rows;
    }
    Basis strip(Basis rows) const {
        if (!field.empty()) rows.erase(std::remove(rows.begin(),rows.end(),field),rows.end());
        return rows;
    }
    Basis reduced(const Basis& g) const {
        if (field.empty()) return primary::reduced(g,width,false);
        // Grevlex in ordinary variables, followed by theta. In a proper
        // quotient m is the sole scalar row and vanishes in K[x].
        std::ostringstream input;
        input << "basis " << axes << " 0 1\n";
        for (const auto& f:lift(g)) input << encode_row(f) << '\n';
        const auto wire=input.str();
        std::unique_ptr<char,decltype(&atlas_groebner_release)> out(
            atlas_wide_block(wire.data(),wire.size()),atlas_groebner_release);
        if (!out) throw std::runtime_error(atlas_groebner_error());
        std::istringstream stream(out.get());
        Basis result;
        for (std::string line;std::getline(stream,line);) result.push_back(decode_row(line,width));
        return strip(std::move(result));
    }
    // (F):I is the intersection of (F):g over g in I. Compute the
    // preimage over QQ, including m in F, so the same proof works over K.
    Basis multipliers(const Basis& selected,const Basis& equations) const {
        const auto source=primary::reduced(lift(selected),width,true);
        Basis answer{one};
        for (const auto& g:equations) {
            if (primary::member_lex(g,source)) continue;
            auto meet=primary::intersection_lex(source,{g},width);
            auto quotient=divide_generators(meet,g,width);
            if (answer==Basis{one}) answer=std::move(quotient);
            else answer=primary::intersection_lex(answer,quotient,width);
            if (answer.empty()) break;
        }
        return answer;
    }
    Polynomial minor(const Basis& selected,const Indices& cols) const {
        if (cols.size()==1) {
            Polynomial result;
            for (const auto& [e,c]:selected.front()) {
                auto k=e;
                if (k[cols[0]]==0) continue;
                --k[cols[0]];
                result.emplace(std::move(k),c*e[cols[0]]);
            }
            return result;
        }
        std::ostringstream input;
        input<<"minor\n";
        for (const auto& name:names) input<<name<<" ";
        input<<'\n';
        for (auto j:cols) input<<j<<" ";
        input<<'\n';
        for (const auto& f:selected) input<<format_polynomial(f,names)<<'\n';
        auto wire=input.str();
        std::unique_ptr<char,decltype(&atlas_groebner_release)> out(
            atlas_affine(wire.data(),wire.size()),atlas_groebner_release);
        if (!out) throw std::runtime_error(atlas_groebner_error());
        // The affine protocol returns one newline-terminated row. A zero
        // determinant is the empty row, which must stay empty when decoded.
        std::istringstream stream(out.get());
        std::string row;
        if (!std::getline(stream,row)) throw std::runtime_error("Missing affine Jacobian minor.");
        return decode_row(row,width);
    }
    J smooth(const Basis& raw,const mpz_class& dim) const {
        const auto generators=reduced(raw);
        Basis bad=generators;
        const mpz_class codim=mpz_class(axes)-dim;
        const Basis unit{one};
        if (bad.empty() && codim==0) bad=unit;
        const auto& equations=raw.size()<generators.size()?raw:generators;
        if (bad!=unit && codim>0 && codim<=axes && codim<=equations.size()) {
            const auto c=codim.get_ui();
            choices(equations.size(),c,[&](const Indices& selected_rows) {
                Basis selected;
                for (auto i:selected_rows) selected.push_back(equations[i]);
                Indices active;
                for (size_t j=0;j<axes;++j) {
                    if (std::any_of(selected.begin(),selected.end(),[&](const auto& f) {
                        return std::any_of(f.begin(),f.end(),[&](const auto& entry){return entry.first[j]!=0;});
                    })) active.push_back(j);
                }
                if (active.size()<c) return true;
                const auto known=primary::reduced(lift(bad),width,true);
                Basis minors;
                choices(active.size(),c,[&](const Indices& indices) {
                    Indices columns;
                    for (auto i:indices) columns.push_back(active[i]);
                    auto m=minor(selected,columns);
                    if (!m.empty() && !primary::member_lex(m,known)) minors.push_back(std::move(m));
                    return true;
                });
                if (!minors.empty()) {
                    // D(multiplier * minor) is a standard smooth open. The
                    // bad locus is the complement of their union in V(I).
                    for (const auto& q:multipliers(selected,equations))
                        for (const auto& m:minors) bad.push_back(primary::multiply(q,m));
                    bad=reduced(bad);
                }
                return bad!=unit;
            });
        }
        J result;
        result["expected_dimension"]=dim.get_str();
        result["empty"]=generators==unit;
        result["smooth"]=bad==unit;
        result["bad_ideal"]=encoded(bad);
        return result;
    }
    void descend(J& report,const std::vector<Basis>& divisors,const Basis& equations,
                 const mpz_class& dimension, J& subset,size_t start) const {
        auto witness=smooth(equations,dimension);
        const bool empty=witness["empty"].asBool();
        witness["components"]=subset;
        report["strata"].append(witness);
        if (!witness["smooth"].asBool()) report["bad"].append(report["strata"].size()-1);
        if (empty) return;
        for (size_t j=start;j<divisors.size();++j) {
            subset.append(Json::UInt64(j));
            descend(report,divisors,join(equations,divisors[j]),dimension-1,subset,j+1);
            subset.resize(subset.size()-1);
        }
    }
    Basis saturated(const Basis& g,const Polynomial& pivot) const {
        if (pivot.empty()) return {one};
        if (pivot.size()==1 && pivot.begin()->first==Powers(width)) return reduced(g);
        return strip(primary::saturate_lex(lift(g),pivot,width));
    }
    J standard_open(const J& q) const {
        auto ambient=rows(q["ambient"]);
        const auto& centre=q["centre"];
        const size_t index=q["pivot"].asUInt();
        if (index>=centre.size() || q["ratios"].size()!=centre.size())
            throw std::invalid_argument("Incomplete family of projective ratios.");
        auto pivot=decode_row(centre[Json::ArrayIndex(index)].asString(),width);
        for (Json::ArrayIndex j=0;j<centre.size();++j) if (j!=index) {
            auto eq=decode_row(centre[j].asString(),width);
            auto term=primary::multiply(pivot,decode_row(q["ratios"][j].asString(),width));
            for (const auto& [e,c]:term) {
                auto it=eq.try_emplace(e,0).first;
                if ((it->second-=c)==0) eq.erase(it);
            }
            ambient.push_back(std::move(eq));
        }
        ambient=saturated(ambient,pivot);
        J result;
        result["ambient"]=encoded(ambient);
        if (!q["strict"].isNull()) result["strict"]=encoded(saturated(join(ambient,rows(q["strict"])),pivot));
        result["exceptional"]=J(Json::arrayValue);
        for (const auto& p:q["exceptional"])
            result["exceptional"].append(encoded(saturated(join(ambient,rows(p)),pivot)));
        return result;
    }
};

J run(const J& q) {
    Chart chart(q);
    const std::string op=q["operation"].asString();
    if (op=="standard_open") return chart.standard_open(q);
    if (op=="saturate") return chart.encoded(chart.saturated(chart.rows(q["ideal"]),decode_row(q["by"].asString(),chart.width)));
    if (op=="radical_subset") {
        const auto larger=chart.lift(chart.rows(q["larger"]));
        const auto smaller=chart.rows(q["smaller"]);
        if (smaller.empty()) return true;
        const auto closed=primary::reduced(larger,chart.width,true);
        if (std::all_of(smaller.begin(),smaller.end(),[&](const auto& f){return primary::member_lex(f,closed);}))
            return true;
        // Computing the radical once also avoids a Rabinowitsch inverse
        // with enormous powers for thick monomial structures.
        const auto radical=primary::radical_lex(closed,chart.width);
        return std::all_of(smaller.begin(),smaller.end(),[&](const auto& f){return primary::member_lex(f,radical);});
    }
    const auto dim=mpz_class(q["dimension"].asString(),10);
    if (op=="smooth") return chart.smooth(chart.rows(q["ideal"]),dim);
    if (op!="snc") throw std::invalid_argument("Unknown chart verification operation.");
    std::vector<Basis> divisors;
    for (const auto& p:q["divisors"]) divisors.push_back(chart.rows(p));
    J subset(Json::arrayValue),report;
    report["strata"]=J(Json::arrayValue);
    report["bad"]=J(Json::arrayValue);
    chart.descend(report,divisors,chart.rows(q["ideal"]),dim,subset,0);
    return report;
}
}
}

extern "C" const char* atlas_graph_wide(const char* input) noexcept {
    return atlas::sparse_json::invoke(input,atlas::graph_wide::run);
}
