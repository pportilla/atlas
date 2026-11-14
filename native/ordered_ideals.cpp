// Ordered ideal operations over QQ. Gröbner tasks are submitted to the caller's
// existing exact CPU/device execution policy; all lifts, contractions, divisions
// and choices of auxiliary rings are made here with unbounded exponents.
#include "rational_sparse_json.hpp"
#include "local_io.hpp"
#include "sparse_flint.hpp"
#include <algorithm>
#include <functional>
#include <numeric>
#include <set>
#include <memory>

extern "C" {
char* atlas_ideal(const char*, size_t) noexcept;
void atlas_groebner_release(char*) noexcept;
const char* atlas_groebner_error() noexcept;
}

namespace {
using namespace atlas::local;
using Value=Json::Value;
using Axes=std::vector<unsigned>;
using Callback=const char*(*)(const char*);
Value array(){return Value(Json::arrayValue);}
Value encode(const Basis& rows) {
    auto out=array();
    for(const auto& f:rows) out.append(encode_row(f));
    return out;
}
Basis decode(const Value& rows,size_t width) {
    Basis out;
    for(const auto& row:rows) out.push_back(decode_row(row.asString(),width));
    return out;
}
Value axes(const Axes& ring) {auto v=array();for(auto i:ring)v.append(i);return v;}
Basis constant(bool unit,size_t width) {return unit ? Basis{Polynomial{{Powers(width),1}}} : Basis{};}
bool nonzero(const Basis& a) {return std::any_of(a.begin(),a.end(),[](const auto& f){return !f.empty();});}
Basis reorder(const Basis& a,const Axes& from,const Axes& to) {
    std::vector<size_t> cols;
    for(auto k:to) cols.push_back(std::find(from.begin(),from.end(),k)-from.begin());
    Basis result;
    for(const auto& f:a) {
        Polynomial g;
        for(const auto& [p,c]:f) {
            Powers q; for(auto k:cols) q.push_back(k<p.size()?p[k]:mpz_class(0));
            g.emplace(std::move(q),c);
        }
        result.push_back(std::move(g));
    }
    return result;
}
Basis contract(const Basis& a,size_t count) {
    Basis result;
    for(const auto& f:a) {
        bool survived=true; Polynomial g;
        for(const auto& [p,c]:f) {
            auto end=p.begin()+count;
            if (std::any_of(p.begin(),end,[](const auto& x){return x!=0;})) {survived=false;break;}
            g.emplace(Powers(end,p.end()),c);
        }
        if(survived) result.push_back(std::move(g));
    }
    return result;
}
std::optional<Basis> divide(const Basis& rows,const Polynomial& by,size_t width) {
    SparseLexRing ring(width);
    atlas::PolyValue d(ring.ctx),f(ring.ctx),q(ring.ctx);
    ring.write(d.p,by);
    Basis out;
    for(const auto& row:rows) {
        ring.write(f.p,row);
        if(!fmpq_mpoly_divides(q.p,f.p,d.p,ring.ctx)) return {};
        out.push_back(ring.read(q.p));
    }
    return out;
}
struct OrderedIdeals {
    std::string order;
    Axes ring;
    Callback callback;
    Value call(const Value& task) const {
        Json::StreamWriterBuilder w;w["indentation"]="";
        auto payload=Json::writeString(w,task);
        const char* response=callback(payload.c_str());
        if (!response) throw std::runtime_error("The requested basis or stage failed.");
        Value answer;std::string error;Json::CharReaderBuilder r;std::istringstream stream(response);
        if(!Json::parseFromStream(r,stream,&answer,&error)) throw std::runtime_error(error);
        return answer;
    }
    void begin(const std::string& name,std::initializer_list<std::pair<std::string,size_t>> fields) const {
        Value event;event["event"]="begin";event["label"]=name;
        for(const auto& [key,n]:fields) event["counts"][key]=Json::UInt64(n);
        call(event);
    }
    void end() const {Value event;event["event"]="end";call(event);}
    Basis basis(const std::string& label,const Axes& vars,const Basis& a,const std::string& ordering) const {
        if(vars.empty()) return constant(nonzero(a),0);
        Value event;event["event"]="basis";event["label"]=label;event["ring"]=axes(vars);
        event["generators"]=encode(a);event["order"]=ordering;
        return decode(call(event),vars.size());
    }
    Basis eliminate_step(const Axes& vars,const Basis& ideal,size_t removed,const std::string& label) const {
        const auto gb=basis(label+"_basis",vars,ideal,order=="lex"?"lex":"first");
        begin(label+"_contract",{{"generators",gb.size()},{"variables_removed",removed},{"surviving_variables",vars.size()-removed}});
        auto result=contract(gb,removed);
        end();return result;
    }
    Basis intersect(const Basis& left,const Basis& right,const std::string& label) const {
        if(ring.empty()) return constant(nonzero(left)&&nonzero(right),0);
        Axes vars=ring;vars.insert(vars.begin(),ring.size());
        begin(label+"_lift",{{"left_generators",left.size()},{"right_generators",right.size()}});
        Basis lifted;
        for(const Basis* side:{&left,&right}) for(const auto& f:*side) if(!f.empty()) {
            Polynomial g;
            for(const auto& [p,c]:f) {
                auto q=p;q.insert(q.begin(),1);
                g.emplace(q,side==&left?c:-c);
                if(side==&right) {q[0]=0;g.emplace(std::move(q),c);}
            }
            lifted.push_back(std::move(g));
        }
        end();return eliminate_step(vars,lifted,1,label);
    }
    Basis quotient(const Basis& a,Basis by) const {
        by.erase(std::remove_if(by.begin(),by.end(),[](const auto& f){return f.empty();}),by.end());
        if(by.empty()||ring.empty()) return constant(by.empty()||nonzero(a),ring.size());
        if(by.size()==1&&by.front().size()==1) {
            begin("quotient_divide_inputs",{{"input_generators",a.size()}});
            auto divided=divide(a,by.front(),ring.size());end();
            if(divided) return basis("quotient_basis",ring,*divided,order);
        }
        Basis result;
        for(size_t i=0;i<by.size();++i) {
            auto label="quotient_by_"+std::to_string(i+1);
            auto shared=intersect(a,{by[i]},label);
            begin(label+"_divide",{{"generators",shared.size()}});
            const bool monomial=by[i].size()==1;
            if(monomial) by[i].begin()->second=1;
            auto part=divide(shared,by[i],ring.size());end();
            if(!part) throw std::logic_error("Inexact principal ideal quotient.");
            if(part->empty()) return {};
            if(by.size()==1) return monomial?*part:basis("quotient_basis",ring,*part,order);
            result=i==0?std::move(*part):intersect(result,*part,"quotient_intersection_"+std::to_string(i+1));
        }
        return result;
    }
    bool saturation_regular(const Basis& ideal, const Polynomial& f) const {
        // Reuse the existing positive proof exactly. An undecided answer
        // always continues to the exact unbounded elimination below.
        begin("saturation_regularity",{{"generators",ideal.size()}});
        std::ostringstream packet;
        packet << "saturation_regular " << ring.size() << ' ' << ideal.size() << '\n';
        for (const auto& row : ideal) packet << encode_row(row) << '\n';
        packet << encode_row(f) << '\n';
        auto input=packet.str();
        std::unique_ptr<char,decltype(&atlas_groebner_release)> proof(
            atlas_ideal(input.data(), input.size()),atlas_groebner_release);
        if (!proof) throw std::runtime_error(atlas_groebner_error());
        end();
        return std::string_view(proof.get())=="yes\n";
    }
    Value apply(const Value& request) const {
        const auto left=decode(request["left"],ring.size());
        const auto op=request["operation"].asString(); Axes output=ring;Basis result;
        if(op=="eliminate") {
            Axes removed;for(const auto& x:request["removed"]) removed.push_back(x.asUInt());
            std::set<unsigned> all(removed.begin(),removed.end());
            if(removed.empty()||all.size()!=removed.size()||*all.rbegin()>=ring.size()) throw std::invalid_argument("Invalid elimination variables.");
            result=left;
            for(size_t i=0;i<removed.size();) {
                const bool block=order=="lex";Axes prefix=block?removed:Axes{removed[i]};
                Axes kept;
                for(auto v:output) if(std::find(prefix.begin(),prefix.end(),v)==prefix.end()) kept.push_back(v);
                Axes vars=prefix;vars.insert(vars.end(),kept.begin(),kept.end());
                std::string label="elimination";
                if(!block&&removed.size()>1) label+="_"+std::to_string(i+1);
                result=eliminate_step(vars,reorder(result,output,vars),prefix.size(),label);
                output=std::move(kept);i+=prefix.size();
            }
        } else if(op=="intersection") result=intersect(left,decode(request["right"],ring.size()),"intersection");
        else if(op=="quotient") result=quotient(left,decode(request["right"],ring.size()));
        else if(op=="saturate") {
            const auto f=decode_row(request["by"].asString(),ring.size());
            if(f.empty()) result=constant(true,ring.size());
            else if((f.size()==1 && f.begin()->first==Powers(ring.size())) ||
                    (request["native_ring"].asBool() && saturation_regular(left,f)))
                result=basis("saturation_basis",ring,left,order);
            else {
                Axes vars=ring;vars.insert(vars.begin(),ring.size());
                begin("adjoin_inverse",{{"generators",left.size()}});
                auto lifted=reorder(left,ring,vars);
                Polynomial inverse{{Powers(vars.size()),-1}};
                for(const auto& [p,c]:f) {auto q=p;q.insert(q.begin(),1);inverse.emplace(std::move(q),c);}
                lifted.push_back(std::move(inverse));end();result=eliminate_step(vars,lifted,1,"saturation");
            }
        } else throw std::invalid_argument("Unknown ordered ideal operation.");
        Value out;out["ring"]=axes(output);out["generators"]=encode(result);return out;
    }
};
}
// The only ideals of QQ are (0) and (1); the empty scheme has dimension -1.
extern "C" const char* atlas_scalar_ideal(const char* input) noexcept {
    return atlas::sparse_json::invoke(input,[](const Value& request)->Value {
        bool unit=nonzero(decode(request["ideal"],0));
        const auto operation=request["operation"].asString();
        if(operation=="basis") return encode(constant(unit,0));
        if(operation=="dimension") return unit ? -1 : 0;
        auto f=decode_row(request["query"].asString(),0);
        if(operation=="membership") return unit || f.empty();
        if(operation=="remainder") return unit ? "" : encode_row(f);
        throw std::invalid_argument("Unknown scalar ideal operation.");
    });
}
extern "C" const char* atlas_ordered_ideal(const char* input,Callback dispatch) noexcept {
    return atlas::sparse_json::invoke(input,[&](const Value& request){
        Axes ring(request["width"].asUInt()); std::iota(ring.begin(),ring.end(),0);
        return OrderedIdeals{request["order"].asString(),ring,dispatch}.apply(request);
    });
}
