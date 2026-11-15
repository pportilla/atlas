// The geometric selection rules for surface models and coefficient
// presentations. The callback evaluates queries in the supplied chart's
// existing ring and frame, including localization and its execution policy.
#include "geometry_query.hpp"
#include "rational_sparse_json.hpp"
#include "sparse_flint.hpp"
#include "local_io.hpp"
#include "quotient_ring.hpp"
#include "primary_decomposition.hpp"
#include <algorithm>
#include <numeric>
#include <set>

namespace {
using V=Json::Value;
using Callback=const char*(*)(const char*);
V arr(){return V(Json::arrayValue);}
V row(std::initializer_list<V> data){V a=arr();for(const auto& v:data)a.append(v);return a;}

struct Search : atlas::GeometryQuery {
    V indexed(const std::string& op,unsigned i) const { V q;q["i"]=i;return call(op,q); }
    V step(unsigned kind,const V& top,bool point,const V& labels,const V& ideal) const {
        V result;
        result["type"]=(kind==2 && !point)?"curve":"points";
        result["kind"]=kind;result["top"]=top;result["point"]=point;result["labels"]=labels;result["ideal"]=ideal;
        return result;
    }
    V model(const V& r) const {
        std::vector<unsigned> order(r["costs"].size());
        std::iota(order.begin(),order.end(),0);
        std::stable_sort(order.begin(),order.end(),[&](auto i,auto j){return r["costs"][i].asUInt64()<r["costs"][j].asUInt64();});
        for (auto i:order) for (unsigned axis=0;axis<r["width"].asUInt();++axis) {
            V q;q["i"]=i;q["axis"]=axis;
            auto opens=call("linear_open",q);
            if (!opens.isNull()) {q["hit"]=std::move(opens);return q;}
        }
        return {};
    }
    V labels_and_curves(const V& r,bool implicit) const {
        // Nonordinary singularities precede nonreduced or shared curves;
        // the singularities of each single reduced trace come last.
        if (auto bad=call("nonordinary");!bad.isNull()) return step(3,"0",false,arr(),bad);
        const auto& labels=r["labels"];
        V top="1",chosen,selected_labels=arr();
        if (!implicit) {
            auto match=call("curve_choice");
            if (!match.isNull()) {
                top=match["top"];
                selected_labels=match["labels"];
                chosen=match["ideal"];
            }
        } else {
            unsigned maximum=0,owner=0;
            for (unsigned i=0;i<labels.size();++i) {
                unsigned depth=0;
                while(depth<64 && indexed("next_level",i).asBool()) ++depth;
                if (depth>=maximum) {maximum=depth;owner=i;}
            }
            top=std::to_string(maximum);
            if (maximum>=2) {
                selected_labels.append(owner);
                chosen=indexed("highest_curve",owner);
            } else {
                for (unsigned a=labels.size();a-- && chosen.isNull();)
                    for (unsigned b=labels.size();b-- > a+1;) {
                        V q;q["a"]=a;q["b"]=b;
                        chosen=call("shared_curve",q);
                        if (!chosen.isNull()) {selected_labels=row({a,b});break;}
                    }
            }
        }
        if (!chosen.isNull()) {
            V q;q["ideal"]=chosen;
            auto singular=call("curve_singular",q);
            return step(2,top,!singular.isNull(),selected_labels,singular.isNull()?chosen:singular);
        }
        for (unsigned i=labels.size();i--;) {
            auto nodes=indexed("nodes",i);
            if (!nodes.isNull()) return step(1,"0",false,row({i}),nodes);
        }
        return {};
    }
    V subsets(const V& r) const {
        unsigned n=r["count"].asUInt();
        V found=arr();
        auto visit=[&](auto&& self,unsigned first,unsigned left,V& chosen) -> void {
            if (!left) {
                V q;q["indices"]=chosen;
                if (call("nonempty",q).asBool()) found.append(chosen);
                return;
            }
            for (unsigned j=first;j<=n-left;++j) {
                chosen.append(j);self(self,j+1,left-1,chosen);chosen.resize(chosen.size()-1);
            }
        };
        V chosen=arr();
        if (r["largest"].asBool()) {
            for (unsigned size=n;size;--size) {visit(visit,0,size,chosen);if (!found.empty()) return found;}
            found.append(chosen);
        } else for (unsigned size=0;size<=n;++size) visit(visit,0,size,chosen);
        return found;
    }
    V strata(const V& r) const {
        using Subset=std::vector<unsigned>;
        const unsigned n=r["count"].asUInt();
        std::set<Subset> alive{Subset{}};
        V result=arr();
        for (unsigned k=1; k<=n && !alive.empty(); ++k) {
            std::set<Subset> next;
            auto visit=[&](auto&& self,unsigned first,unsigned left,Subset& chosen)->void {
                if (left) {
                    for (unsigned j=first;j<=n-left;++j) {
                        chosen.push_back(j);self(self,j+1,left-1,chosen);chosen.pop_back();
                    }
                    return;
                }
                for (size_t j=0;j<chosen.size();++j) {
                    auto face=chosen;face.erase(face.begin()+j);
                    if (!alive.count(face)) return;
                }
                V q;q["indices"]=arr();
                for (auto i : chosen) q["indices"].append(i);
                if (!call("nonempty",q).asBool()) return;
                next.insert(chosen);result.append(q["indices"]);
            };
            Subset chosen;
            visit(visit,0,k,chosen);
            alive=std::move(next);
        }
        return result;
    }
    V linear_blowup(const V& r) const {
        using namespace atlas::local;
        const unsigned width=r["width"].asUInt();
        const auto& axes=r["axes"];
        auto decode=[&](const V& rows) {Basis b;for(const auto& f:rows)b.push_back(decode_row(f.asString(),width));return b;};
        auto encode=[](const Basis& rows) {auto out=arr();for(const auto& f:rows)out.append(encode_row(f));return out;};
        const auto ambient=decode(r["ambient"]), shifts=decode(r["shifts"]);
        if (shifts.size()!=axes.size()) throw std::invalid_argument("Inconsistent linear blowup centre.");
        for(const auto& axis:axes) if(axis.asUInt()>=width)throw std::invalid_argument("Invalid blowup pivot.");
        SparseLexRing ring(width);
        atlas::quotient_ring::Ring reduced(ambient,width,false);
        auto charts=arr();
        for(unsigned j=0;j<axes.size();++j) {
            const unsigned pivot=axes[j].asUInt();
            std::vector<atlas::PolyValue> maps;
            maps.reserve(width);
            std::vector<fmpq_mpoly_struct*> images;
            atlas::PolyValue source(ring.ctx),value(ring.ctx),tmp(ring.ctx);
            for(unsigned k=0;k<width;++k) {
                maps.emplace_back(ring.ctx);fmpq_mpoly_gen(maps.back().p,k,ring.ctx);
                images.push_back(maps.back().p);
            }
            for(unsigned i=0;i<axes.size();++i) {
                auto axis=axes[i].asUInt();
                if (i!=j) fmpq_mpoly_mul(images[axis],images[axis],images[pivot],ring.ctx);
            }
            for(unsigned i=0;i<axes.size();++i) {
                auto axis=axes[i].asUInt();ring.write(tmp.p,shifts[i]);
                fmpq_mpoly_add(images[axis],images[axis],tmp.p,ring.ctx);
            }
            auto move=[&](const V& rows,bool keep_zero) {
                Basis out;
                for (const auto& f:decode(rows)) {
                    ring.write(source.p,f);
                    if(!fmpq_mpoly_compose_fmpq_mpoly(value.p,source.p,images.data(),ring.ctx,ring.ctx))
                        throw std::runtime_error("The sparse chart pullback could not be represented.");
                    auto g=ring.read(value.p);
                    if (!ambient.empty()) g=std::get<Polynomial>(reduced.evaluate("reduce",{g}));
                    if(keep_zero || !g.empty())out.push_back(std::move(g));
                }
                return out;
            };
            V chart;chart["divisors"]=arr();
            for(const auto& d:r["divisors"]) {
                auto moved=move(d,false);
                if(moved.empty()) {chart["decline"]=true;return chart;}
                auto order=moved.front().begin()->first[pivot];
                for(const auto& g:moved) for(const auto& [e,c]:g)order=std::min(order,e[pivot]);
                Basis after;
                if(order==0) after=std::move(moved);
                else if(moved.size()==1) {
                    Polynomial residual;
                    for(const auto& [e,c]:moved.front()) {auto a=e;a[pivot]-=order;residual.emplace(std::move(a),c);}
                    after.push_back(std::move(residual));
                } else {
                    Powers e(width);e[pivot]=order;
                    V q;q["pivot"]=j;q["moved"]=encode(moved);q["power"]=encode_row(Polynomial{{e,1}});
                    after=decode(call("residual_generators",q));
                }
                auto ideal=ambient;ideal.insert(ideal.end(),after.begin(),after.end());
                const bool present=atlas::primary::reduced(ideal,width,false)!=Basis{{{Powers(width),1}}};
                chart["divisors"].append(row({order.get_str(),present,encode(after)}));
            }
            chart["functions"]=encode(move(r["functions"],true));
            chart["fibre"]=encode(move(r["fibre"],false));
            chart["region"]=encode(move(r["region"],false));
            charts.append(std::move(chart));
        }
        return charts;
    }

    // Coordinate choices prefer the fewest dependent boundary coordinates,
    // with lexicographic ties. Stop once all remaining open slices are covered.
    V frame_cover(const V& r) const {
        const auto width=r["width"].asUInt(),dimension=r["dimension"].asUInt(),equations=r["equations"].asUInt();
        if (dimension>width) return false;
        const auto count=width-dimension;
        if (!count) return call("affine");
        if (count>equations) return false;
        using Tuple=std::vector<unsigned>;
        auto combinations=[&](unsigned total,const auto& consume) {
            Tuple values;
            auto visit=[&](auto&& self,unsigned first,unsigned left)->bool {
                if (!left) return consume(values);
                for(unsigned j=first;j<=total-left;++j) {
                    values.push_back(j);
                    if (self(self,j+1,left-1)) return true;
                    values.pop_back();
                }
                return false;
            };
            return visit(visit,0,count);
        };
        std::vector<std::pair<unsigned,Tuple>> choices;
        combinations(width,[&](const Tuple& axes) {
            unsigned cost=0;
            for(auto j:axes) if(r["divisors"][j].asBool()) ++cost;
            choices.emplace_back(cost,axes);
            return false;
        });
        std::sort(choices.begin(),choices.end());
        V q;
        q["linear"]=count==1 && equations==1;
        for(const auto& choice:choices) {
            q["dependent"]=arr();
            for(auto j:choice.second) q["dependent"].append(j);
            if(combinations(equations,[&](const Tuple& rows) {
                q["rows"]=arr();
                for(auto j:rows) q["rows"].append(j);
                return call("open",q).asBool();
            })) return true;
        }
        return false;
    }

    V grouped_loci(const V& r) const {
        V result=arr();
        bool compatible=true;
        auto visit=[&](auto&& self,const V& work,unsigned index,const V& kept)->void {
            if (!compatible) return;
            if (index==r["count"].asUInt()) {result.append(row({work,kept}));return;}
            V q;q["work"]=work;q["i"]=index;
            V more=kept;more.append(index);
            bool regular = call("regular",q).asBool();
            if (!regular) {
                // A finite order in a smooth local ring proves that f is a
                // nonzero germ there. Covering every working point suffices
                // for this grouping, including if f kills a disjoint component.
                const auto orders = call("working_orders", q);
                regular = orders.isArray() && std::all_of(orders.begin(), orders.end(),
                    [](const V& order) { return order.isUInt(); });
            }
            if (regular) {
                auto live=call("live",q); if (!live.empty()) self(self,live,index+1,more);
            } else if (call("vanishes",q).asBool()) self(self,work,index+1,kept);
            else {
                auto pieces=call("split",q);
                if (pieces["overlap"].asBool()) {compatible=false;return;}
                if (!pieces["on"].empty()) self(self,pieces["on"],index+1,more);
                if (!pieces["off"].empty()) self(self,pieces["off"],index+1,kept);
            }
        };
        visit(visit,r["work"],0,arr());
        return compatible?result:V();
    }

    V incidence_count(const V& r) const {
        const int codim=r["count"].asInt(),dim=r["dimension"].asInt()-codim;
        V result;result["here"]=V();result["rational"]=V();
        if (!call("over_fibre").asBool()) return result;
        if (dim==0) result["here"]=call("points");
        else if (r["flat"].asBool()) result["here"]=call("owned").asBool()?r["degree"]:V("0");
        else if (dim>0) {
            if (dim==1) {
                const auto plane=call("plane");
                if (!plane.isNull()) {result["here"]=plane[0];result["rational"]=plane[1];return result;}
            }
            V q;q["dimension"]=dim;
            const auto components=call("components",q);
            if (!components.isNull()) {
                unsigned count=0;
                for (const auto& owned : components) if (owned.asBool()) ++count;
                result["here"]=std::to_string(count);
            }
        }
        if (dim>0) result["rational"]=result["here"];
        return result;
    }

    V run(const V& r) const {
        const auto op=r["operation"].asString();
        if (op=="frame_cover") return frame_cover(r);
        if (op=="grouped_loci") return grouped_loci(r);
        if (op=="incidence_count") return incidence_count(r);
        if (op=="linear_blowup") return linear_blowup(r);
        if (op=="strata") return strata(r);
        if (op=="surface_model") return model(r);
        if (op=="surface_plan") return labels_and_curves(r,r["implicit"].asBool());
        if (op=="divisor_subsets") return subsets(r);
        if (op=="jet_order") {
            auto k=r["initial"].asUInt();
            while (indexed("live_jet",k+1).asBool()) if (++k>64) return {};
            return k;
        }
        if (op=="maximum_threshold") {
            unsigned count=r["count"].asUInt();
            if (!count) throw std::invalid_argument("The residual order grid is empty.");
            unsigned low=0,high=count-1;
            while (low<high) {
                auto mid=low+(high-low+1)/2;
                if (indexed("live_threshold",mid).asBool()) low=mid;
                else high=mid-1;
            }
            return low;
        }
        throw std::invalid_argument("Unknown centre selection request.");
    }
};
}
namespace atlas {
// The same searches for a caller in C++.
Json::Value centre_search(const Json::Value& request, std::function<Json::Value(const Json::Value&)> query) {
    Search search{{nullptr, std::move(query)}};
    return search.run(request);
}
}
extern "C" const char* atlas_centre_search(const char* input, Callback callback) noexcept {
    return atlas::sparse_json::invoke(input,[&](const V& r){return Search{callback}.run(r);});
}
