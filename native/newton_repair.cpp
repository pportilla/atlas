// Decide the orbit cuts in a Newton repair and the chart of its separating
// parameter. Face singularity, lattice volume and Euler values come from the
// same certified kernels used to build and independently verify the atlas.
#include "rational_sparse_json.hpp"
#include "local_io.hpp"
#include "primary_decomposition.hpp"
#include "sparse_flint.hpp"
#include <algorithm>
#include <iterator>
#include <set>

namespace {
using Value = Json::Value;
using Indices = std::vector<unsigned>;
using atlas::local::Basis;
using atlas::local::Powers;
using atlas::local::Polynomial;

Value array() { return Value(Json::arrayValue); }

Indices indices(const Value& values) {
    Indices result;
    for (const auto& v : values) result.push_back(v.asUInt());
    return result;
}

Value list(const Indices& indices) {
    auto result=array();
    for (auto v : indices) result.append(v);
    return result;
}

Indices common_face(const std::vector<Indices>& minima, const Indices& rays) {
    if (rays.empty()) throw std::invalid_argument("A repair orbit needs a ray.");
    auto face=minima.at(rays.front());
    for (size_t i=1; i<rays.size() && !face.empty(); ++i) {
        Indices next;
        const auto& other=minima.at(rays[i]);
        std::set_intersection(face.begin(),face.end(),other.begin(),other.end(),std::back_inserter(next));
        face=std::move(next);
    }
    return face;
}

Value orbit_faces(const Value& request) {
    const unsigned width=request["width"].asUInt();
    std::vector<std::vector<Indices>> minima;
    for (const auto& factor : request["minima"]) {
        auto& masks=minima.emplace_back();
        for (const auto& row : factor) {
            auto mask=indices(row);
            if (!std::is_sorted(mask.begin(),mask.end())) throw std::invalid_argument("Unordered Newton term indices.");
            masks.push_back(std::move(mask));
        }
    }
    std::map<Indices, unsigned> face_ids;
    Value result, faces=array(), orbits=array();
    bool refines=true;
    for (const auto& row : request["orbits"]) {
        const auto rays=indices(row["rays"]);
        Value orbit;
        orbit["faces"]=array();
        for (size_t i=0; i<minima.size(); ++i) {
            auto face=common_face(minima[i],rays);
            if (i) { orbit["faces"].append(list(face)); continue; }
            const auto [it,inserted]=face_ids.emplace(face,face_ids.size());
            if (inserted) faces.append(list(face));
            orbit["face_id"]=it->second;
            refines &= !face.empty();
        }
        Indices free, fixed, exceptional, exceptional_axes;
        for (unsigned r : rays) if (r>=width) exceptional.push_back(r);
        const auto cone=indices(request["cones"][row["chart"].asUInt()]);
        for (unsigned i=0; i<cone.size(); ++i) {
            (std::find(rays.begin(),rays.end(),cone[i])==rays.end()?free:fixed).push_back(i);
            if (cone[i]>=width) exceptional_axes.push_back(i);
        }
        orbit["free"]=list(free);
        orbit["fixed"]=list(fixed);
        orbit["exceptional"]=list(exceptional);
        orbit["exceptional_axes"]=list(exceptional_axes);
        orbits.append(std::move(orbit));
    }
    result["faces"]=std::move(faces);
    result["orbits"]=std::move(orbits);
    result["refines"]=refines;
    return result;
}

Value orbit_plan(const Value& request) {
    const bool count=request["count_strata"].asBool();
    const auto& rows=request["orbits"];
    std::vector<Indices> faces, degenerate;
    for (const auto& face : request["faces"]) faces.push_back(indices(face));
    for (unsigned i : indices(request["degenerate"]))
        if (!faces.at(i).empty()) degenerate.push_back(faces[i]);
    Value plans=array();
    for (const auto& row : rows) {
        const auto& face=faces.at(row["face_id"].asUInt());
        const auto& parts=row["faces"];
        const unsigned dimension=row["free"].size();
        Value plan;
        if (!count && std::find(degenerate.begin(),degenerate.end(),face)==degenerate.end()) {
            plan["action"]="skip";
        } else if (face.empty()) {
            // An orbit inside two distinct strict components, or one cut by
            // another nonmonomial strict factor, requires a different atlas.
            std::vector<unsigned> inside;
            bool meets_other=false;
            for (unsigned j=0; j<parts.size(); ++j) {
                if (parts[j].empty()) inside.push_back(j);
                if (dimension && parts[j].size()>1) meets_other=true;
            }
            if (inside.size()!=1 || meets_other) plan["action"]="inadmissible";
            else {
                plan["action"]="contained";
                plan["part"]=inside[0];
                plan["chi"]=dimension ? 0 : 1;
            }
        } else if (!dimension) {
            plan["action"]="point";
        } else if (count && (request["all_subfaces"].asBool() || face.size()<=2) &&
                   std::none_of(degenerate.begin(),degenerate.end(),[&](const auto& subface){
                       return std::includes(face.begin(),face.end(),subface.begin(),subface.end()); })) {
            plan["action"]="volume";
            plan["volume_parts"]=array();
            for (unsigned j=0; j<parts.size(); ++j) if (parts[j].size()>dimension) plan["volume_parts"].append(j);
            plan["strict_sign"]=dimension%2 ? 1 : -1;
            plan["off_sign"]=-plan["strict_sign"].asInt();
        } else {
            plan["action"]="exact";
        }
        plans.append(std::move(plan));
    }
    return plans;
}

// For a radical zero-dimensional ideal on (x_1,...,x_n,t), t is separating
// iff its reduced lex basis gives a single eliminant and one x_i-q_i(t) per
// original coordinate. Return basis indices so the caller retains its rows.
Value rur_graph(const Value& request) {
    const unsigned n=request["axes"].asUInt(), width=n+1;
    std::vector<unsigned> eliminants;
    Indices solved(n,request["rows"].size());
    for (unsigned i=0; i<request["rows"].size(); ++i) {
        const auto f=atlas::local::decode_row(request["rows"][i].asString(),width);
        std::set<unsigned> used;
        for (const auto& [p,c] : f) for (unsigned j=0; j<n; ++j) if (p[j]!=0) used.insert(j);
        if (used.empty()) { eliminants.push_back(i); continue; }
        if (used.size()!=1) continue;
        const auto j=*used.begin();
        Powers x(width);
        x[j]=1;
        const auto it=f.find(x);
        if (it==f.end() || it->second!=1 || solved[j]!=request["rows"].size()) continue;
        if (std::all_of(f.begin(),f.end(),[&](const auto& t){return t.first==x || t.first[j]==0;})) solved[j]=i;
    }
    if (eliminants.size()!=1 || std::find(solved.begin(),solved.end(),request["rows"].size())!=solved.end()) return {};
    eliminants.insert(eliminants.end(),solved.begin(),solved.end());
    return list(eliminants);
}

Value supported_rows(const Value& request) {
    const unsigned width=request["width"].asUInt();
    const auto permitted=indices(request["columns"]);
    std::set<unsigned> axes(permitted.begin(),permitted.end());
    for (auto j : axes) if (j>=width) throw std::invalid_argument("Invalid repair projection column.");
    auto result=array();
    for (unsigned i=0; i<request["rows"].size(); ++i) {
        const auto f=atlas::local::decode_row(request["rows"][i].asString(),width);
        std::set<unsigned> used;
        for (const auto& [e,c] : f) for (unsigned j=0; j<width; ++j) if (e[j]!=0) used.insert(j);
        if (request["exact"].asBool() ? used==axes : std::includes(axes.begin(),axes.end(),used.begin(),used.end()))
            result.append(i);
    }
    return result;
}

// The singular scheme on the coordinate torus is the Jacobian ideal
// saturated by the product of its coordinates. The contracted lex basis
// also determines whether only finitely many centres can be repaired.
Value singular_orbit(const Value& request) {
    const unsigned n=request["width"].asUInt();
    if (!n) throw std::invalid_argument("Expected a positive-dimensional Newton orbit.");
    const auto original=atlas::local::decode_row(request["equation"].asString(),n);
    // A Laurent monomial is invertible on this orbit. Divide it out before
    // saturation, also when its exponents exceed the packed basis limits.
    Polynomial f;
    if (!original.empty()) {
        Powers minimum=original.begin()->first;
        for (const auto& [exponents,c] : original) for (unsigned i=0; i<n; ++i)
            minimum[i]=std::min(minimum[i],exponents[i]);
        for (const auto& [exponents,c] : original) {
            auto e=exponents;
            for (unsigned i=0; i<n; ++i) e[i]-=minimum[i];
            f.emplace(std::move(e),c);
        }
    }
    atlas::local::SparseLexRing ring(n);
    atlas::PolyValue input(ring.ctx), derivative(ring.ctx);
    ring.write(input.p,f);
    Basis equations{f};
    for (unsigned i=0; i<n; ++i) {
        fmpq_mpoly_derivative(derivative.p,input.p,i,ring.ctx);
        equations.push_back(ring.read(derivative.p));
    }
    const Polynomial invert{{Powers(n,1),1}};
    const auto ideal=atlas::primary::saturate_lex(equations,invert,n);
    const auto dim=atlas::primary::dimension_lex(ideal,n);
    Value result;
    result["kind"] = dim<0 ? "empty" : dim==0 ? "points" : "positive_dimension";
    result["ideal"] = array();
    for (const auto& g : ideal) result["ideal"].append(atlas::local::encode_row(g));
    return result;
}

Value branch_euler(const Value& request) {
    // Open orbit minus the strict cut (the positive-dimensional orbit has
    // Euler characteristic zero); each strict branch minus its repair points.
    // The shared signed integer accumulator performs all subsequent addition.
    auto result = array();
    const auto& branches=request["branches"];
    const auto& deleted=request["deleted"];
    if (branches.size()!=deleted.size()) throw std::invalid_argument("Inconsistent Newton branch counts.");
    auto add=[&](int branch, const Value& value, int sign) {
        if (value.isNull()) return;
        if (mpz_class(value.asString(),10)==0) return;
        auto row=array();
        row.append(branch); row.append(value); row.append(sign);
        result.append(std::move(row));
    };
    add(-1,request["cut"],-1);
    for (unsigned i=0; i<branches.size(); ++i) {
        add(i,deleted[i],-1);
        add(i,branches[i],1);
    }
    return result;
}

Value repair(const Value& request) {
    const auto name=request["operation"].asString();
    if (name=="singular_orbit") return singular_orbit(request);
    if (name=="branch_euler") return branch_euler(request);
    if (name=="orbit_faces") return orbit_faces(request);
    if (name=="orbit_plan") return orbit_plan(request);
    if (name=="rur_graph") return rur_graph(request);
    if (name=="supported_rows") return supported_rows(request);
    if (name=="orders_supported") {
        const auto orders=atlas::sparse_json::read_row(request["orders"]);
        const unsigned n=request["width"].asUInt();
        return orders.size()>n &&
            std::all_of(orders.begin(),orders.begin()+n,[](const auto& d){return d==0;}) &&
            std::all_of(orders.begin()+n,orders.end(),[](const auto& d){return d>=1;});
    }
    throw std::invalid_argument("Unknown Newton repair operation.");
}
}
extern "C" const char* atlas_newton_repair(const char* input) noexcept {
    return atlas::sparse_json::invoke(input, repair);
}
