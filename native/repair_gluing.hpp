#pragma once
#include "local_io.hpp"
#include "primary_decomposition.hpp"
#include "rational_sparse_json.hpp"
#include "quotient_ring.hpp"
#include <flint/fmpq.h>
#include <flint/fmpq_mat.h>
#include <set>

namespace atlas::repair {
using V=Json::Value;
using local::Basis;
using local::Polynomial;
using local::Powers;

inline V array(){ return V(Json::arrayValue); }

// Collect exactly one nonlinear irreducible parameter equation per finite
// chart. Linear or split-linear equations do not define an extension.
inline V chart_relations(const V& request) {
    const auto& charts=request["charts"];
    V covers=array();
    for(unsigned i=0;i<charts.size();++i) {
        const auto& chart=charts[i];
        if(chart["parent"].isNull()) continue;
        const auto j=chart["parent"].asInt();
        if(j<0 || static_cast<unsigned>(j)>=charts.size()) return {};
        const auto& parent=charts[j];
        if(!parent["parent"].isNull() || parent["marked"].asBool() || !parent["ambient"].empty() ||
           !parent["removed"].asBool() || chart["marked"].asBool() || chart["mapping"]!=parent["width"]) return {};
        const unsigned n=chart["width"].asUInt();
        V found;
        for(unsigned k=0;k<chart["ambient"].size();++k) {
            const auto f=local::decode_row(chart["ambient"][k].asString(),n);
            std::set<unsigned> axes;
            for(const auto& [e,c]:f) for(unsigned u=0;u<n;++u) if(e[u]!=0) axes.insert(u);
            if(axes.size()!=1) continue;
            const auto axis=*axes.begin();
            Polynomial uni;
            for(const auto& [e,c]:f) uni.emplace(Powers{e[axis]},c);
            if(uni.rbegin()->first.front()<2) continue;
            // Listing every embedded root and its dense defining polynomial
            // is required by the saved finite-cover presentation.
            const auto factors=primary::factors(uni,1);
            for(const auto& [p,m]:factors) {
                const auto& degree=p.rbegin()->first.front();
                if(degree<=1) continue;
                if(!found.isNull() || factors.size()!=1 || m!=1 || !degree.fits_ulong_p() ||
                   degree>=std::vector<mpz_class>().max_size()) return {};
                mpz_class den=1;
                for(const auto& [e,c]:p) den=lcm(den,c.get_den());
                std::vector<mpz_class> coefficients(degree.get_ui()+1);
                for(const auto& [e,c]:p) coefficients[degree.get_ui()-e.front().get_ui()]=c.get_num()*(den/c.get_den());
                found["sheet"]=i;found["axis"]=axis;found["relation"]=k;
                found["coefficients"]=sparse_json::strings(coefficients);
            }
        }
        if(!found.isNull()) covers.append(found);
    }
    return covers;
}

struct Matrix {
    fmpq_mat_t value;
    explicit Matrix(unsigned size){ fmpq_mat_init(value,size,size); }
    ~Matrix(){ fmpq_mat_clear(value); }
    Matrix(const Matrix&)=delete;
    bool load(const V& rows,bool nonnegative_integer) {
        const auto n=static_cast<unsigned>(fmpq_mat_nrows(value));
        if(rows.size()!=n) return false;
        for(unsigned i=0;i<n;++i) {
            if(rows[i].size()!=n) return false;
            const auto row=sparse_json::read_row(rows[i]);
            for(unsigned j=0;j<n;++j) {
                if(nonnegative_integer && (row[j]<0 || row[j].get_den()!=1)) return false;
                fmpq_set_mpq(fmpq_mat_entry(value,i,j),row[j].get_mpq_t());
            }
        }
        return true;
    }
};

// The ray matrix must be unimodular, the linear map invertible, and
// the saved base map on this sheet must be its indicated monomial sum.
inline V toric_parent(const V& r) {
    const unsigned n=r["rays"].size();
    if(!n || r["width"].asUInt()!=n+1 || r["images"].size()!=n) return {};
    Matrix rays(n),inverse(n),linear(n),change(n);
    if(!rays.load(r["rays"],true) || !linear.load(r["linear"],false) ||
       !fmpq_mat_inv(inverse.value,rays.value) || !fmpq_mat_inv(change.value,linear.value)) return {};
    V inverse_rows=array();
    quotient_ring::Ring sheet({local::decode_row(r["modulus"].asString(),n+1)},n+1,true);
    for(unsigned i=0;i<n;++i) {
        V row=array();
        Polynomial expected;
        for(unsigned j=0;j<n;++j) {
            mpq_class power,coefficient;
            fmpq_get_mpq(power.get_mpq_t(),fmpq_mat_entry(inverse.value,i,j));
            if(power.get_den()!=1) return {};
            row.append(power.get_str());
            fmpq_get_mpq(coefficient.get_mpq_t(),fmpq_mat_entry(linear.value,i,j));
            if(coefficient!=0) {
                Powers e(n+1);
                for(unsigned k=0;k<n;++k) e[k]=mpz_class(r["rays"][k][j].asString(),10);
                expected.emplace(std::move(e),coefficient);
            }
        }
        const auto image=local::decode_row(r["images"][i].asString(),n+1);
        if(std::get<Polynomial>(sheet.evaluate("reduce",{image}))!=expected) return {};
        inverse_rows.append(std::move(row));
    }
    V out;out["inverse"]=std::move(inverse_rows);out["rays"]=r["rays"];out["linear"]=r["linear"];out["names"]=r["names"];
    return out;
}

inline V toric_transition(const V& r) {
    const auto& a=r["source"];const auto& b=r["target"];
    if(a.isNull() || b.isNull() || a["linear"]!=b["linear"] || a["names"]!=b["names"]) return {};
    const unsigned n=a["rays"].size();
    Matrix left(n),back(n),powers(n);
    if(!left.load(a["rays"],true) || !back.load(b["inverse"],false)) return {};
    fmpq_mat_mul(powers.value,left.value,back.value);
    V results=array();
    for(unsigned j=0;j<n;++j) {
        Powers numerator(n),denominator(n);
        for(unsigned i=0;i<n;++i) {
            mpq_class q;fmpq_get_mpq(q.get_mpq_t(),fmpq_mat_entry(powers.value,i,j));
            if(q.get_den()!=1) return {};
            if(q<0) denominator[i]=-q.get_num();else numerator[i]=q.get_num();
        }
        results.append(local::encode_row({{numerator,1}}));results.append(local::encode_row({{denominator,1}}));
    }
    return results;
}

// Each sheet projects onto a single geometric point on its marked field.
// Gather the point sets in parent order, prove coverage and disjointness,
// and close visible centres through the two regular parent transitions.
template<class Operations> V ownership(const V& r,const Operations& ask) {
    const auto& parents=r["parents"];
    const auto& sheets=r["sheets"];
    std::map<unsigned,std::vector<unsigned>> images,additions,by_child;
    std::set<unsigned> encountered;
    std::vector<unsigned> active;
    for(unsigned s=0;s<sheets.size();++s) {
        const auto j=sheets[s]["parent"].asUInt(),child=sheets[s]["child"].asUInt();
        V q;q["sheet"]=s;q["parent"]=j;
        for(auto t:by_child[child]) {
            q["other"]=t;
            if(!ask("disjoint",q).asBool()) return {};
        }
        by_child[child].push_back(s);
        if(ask("outside",q).asBool()) return {};
        if(encountered.insert(j).second) active.push_back(j);
        bool duplicate=false;
        for(auto t:images[j]) {
            q["other"]=t;
            if(ask("disjoint",q).asBool()) continue;
            if(!ask("same",q).asBool()) return {};
            duplicate=true;break;
        }
        if(!duplicate) images[j].push_back(s);
    }
    auto pack=[](const std::vector<unsigned>& rows) {V a=array();for(auto i:rows) a.append(i);return a;};
    for(auto j:active) {
        V q;q["parent"]=j;q["centres"]=pack(images[j]);
        if(!ask("covers",q).asBool()) return {};
    }
    for(const auto& item:parents) for(auto j:active) {
        const unsigned i=item.asUInt();
        if(i==j) continue;
        V q;q["source"]=j;q["target"]=i;
        if(!ask("transition",q).asBool()) return {};
        for(auto s:images[j]) {
            q["sheet"]=s;
            if(!ask("regular",q).asBool()) continue;
            const auto closed=ask("close",q);
            if(closed.isNull()) continue;
            q["closed"]=closed;q["previous"]=pack(additions[i]);
            if(!ask("point",q).asBool() || !ask("new",q).asBool()) return {};
            additions[i].push_back(closed.asUInt());
        }
    }
    V plan;plan["holes"]=array();plan["additions"]=array();
    for(unsigned s=0;s<sheets.size();++s) {
        const auto j=sheets[s]["parent"].asUInt();
        V q;q["sheet"]=s;q["parent"]=j;
        V holes=array();
        for(auto t:images[j]) {
            q["other"]=t;
            if(ask("disjoint",q).asBool()) holes.append(t);
        }
        plan["holes"].append(holes);
    }
    for(const auto& item:parents)plan["additions"].append(pack(additions[item.asUInt()]));
    return plan;
}
}
