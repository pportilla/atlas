// Incidences of exceptional intersections and compatible branch labels in saved atlases.
#include "rational_sparse_json.hpp"
#include "local_io.hpp"
#include "polynomial_graph.hpp"
#include <algorithm>
#include <cctype>
#include <functional>
#include <iterator>
#include <set>

namespace {
using atlas::sparse_json::Z;
using Names = std::vector<std::string>;
using Key = std::vector<unsigned>;
using Value = Json::Value;

Names names(const Value& row) {
    Names result;
    for (const auto& x : row) result.push_back(x.asString());
    return result;
}
template<class T>
Value array(const std::vector<T>& row) {
    Value result(Json::arrayValue);
    for (const auto& x : row) result.append(x);
    return result;
}
std::string join(const Names& row, const std::string& delimiter) {
    std::string result;
    for (size_t i = 0; i < row.size(); ++i) {
        if (i) result += delimiter;
        result += row[i];
    }
    return result;
}
Value cell(const std::string& id, const Names& vertices, const Names& faces) {
    Value result;
    result["id"] = id;
    result["dimension"] = Json::UInt(vertices.size()-1);
    result["vertices"] = array(vertices);
    result["faces"] = array(faces);
    return result;
}

struct Intersections {
    Names labels;
    std::vector<Key> ordered;
    std::map<Key,Value> counts;
    std::map<Key,std::string> fixed;
    explicit Intersections(const Value& request) : labels(names(request["labels"])) {
        std::map<std::string,unsigned> indices;
        for (unsigned i = 0; i < labels.size(); ++i) indices.emplace(labels[i],i);
        auto key = [&](const Value& row) {
            Key subset;
            for (const auto& name : row) subset.push_back(indices.at(name.asString()));
            std::sort(subset.begin(),subset.end());
            return subset;
        };
        for (const auto& item : request["counts"]) {
            auto subset = key(item[0]);
            ordered.push_back(subset);
            counts[subset] = item[1];
        }
        for (const auto& item : request["fixed"]) fixed.emplace(key(item[0]),item[1].asString());
    }
    Names vertices(const Key& key) const {
        Names result;
        for (auto i : key) result.push_back(labels.at(i));
        return result;
    }
    std::string face_name(const Key& key) const {
        const auto found = fixed.find(key);
        if (found != fixed.end()) return found->second;
        auto list = vertices(key);
        return list.size() == 1 ? list.front() : "C"+join(list,"-")+".1";
    }
    Value cells() const {
        Value result;
        result["edges"] = Value(Json::arrayValue);
        result["reason"] = Value();
        auto sorted = ordered;
        std::stable_sort(sorted.begin(),sorted.end(),[](const Key& a, const Key& b) {
            return a.size() != b.size() ? a.size() < b.size() : a < b;
        });
        for (const auto& key : sorted) {
            if (fixed.count(key)) continue;
            const auto& count = counts.at(key);
            auto labels = vertices(key);
            std::string reason;
            if (count.isNull()) reason = "The number of components of "+join(labels," ∩ ")+" was not determined.";
            else if (key.size()==1 && count.asString()!="1")
                reason = labels[0]+" has "+count.asString()+" components; the cells are not attached to each of them.";
            if (reason.empty() && key.size()!=1) {
                Names faces;
                for (size_t i = 0; i < key.size(); ++i) {
                    auto face = key;
                    face.erase(face.begin()+i);
                    auto size = counts.find(face);
                    if (!fixed.count(face) && (size==counts.end() || size->second.isNull() || size->second.asString()!="1")) {
                        reason = "An intersection with several components is a face of "+join(labels," ∩ ")+".";
                        break;
                    }
                    faces.push_back(face_name(face));
                }
                if (reason.empty()) {
                    const Z n(count.asString());
                    for (Z i=1; i<=n; ++i) result["edges"].append(cell("C"+join(labels,"-")+"."+i.get_str(),labels,faces));
                }
            }
            if (!reason.empty()) { result["reason"] = reason; result["edges"] = Value(); return result; }
        }
        return result;
    }
};

// For a curve or surface, a single rational orbit opposite an unsplit
// component has the same number of edges to every conjugate endpoint.
Value boundary(const Value& request, const Intersections& data) {
    bool intrinsic = request["intrinsic"].asBool();
    int dimension = request["dimension"].asInt() - int(intrinsic);
    Value result;
    result["dimension"] = dimension;
    result["vertices"] = Value(Json::arrayValue);
    std::vector<Names> ids(data.labels.size());
    for (unsigned j=0; j<data.labels.size(); ++j) {
        const auto& name = data.labels[j];
        const auto& row = data.counts.at(Key{j});
        Z count = row.isNull() ? 0 : Z(row.asString());
        if (intrinsic && count>1)
            for (Z i=1; i<=count; ++i) ids[j].push_back(name+"."+i.get_str());
        else ids[j].push_back(name);
        for (const auto& id : ids[j]) {
            Value vertex;
            vertex["id"] = id;
            vertex["dimension"] = dimension-1;
            result["vertices"].append(vertex);
        }
    }
    if (!intrinsic || dimension>2) {
        const auto ordinary = data.cells();
        result["edges"] = ordinary["edges"]; result["reason"] = ordinary["reason"];
        return result;
    }
    result["edges"] = Value(Json::arrayValue);
    result["reason"] = Value();
    for (unsigned j=0; j<data.labels.size(); ++j) if (data.counts.at(Key{j}).isNull()) {
        result["reason"] = "The number of components of "+data.labels[j]+" was not determined.";
        return result;
    }
    for (const auto& key : data.ordered) {
        if (key.size()!=2) continue;
        unsigned a=key[0], b=key[1], split=ids[a].size()>1 ? a : b;
        const Z pairs=Z(ids[a].size())*ids[b].size();
        const auto& row = data.counts.at(key);
        Z count=row.isNull() ? 0 : Z(row.asString());
        if (row.isNull() || count%pairs!=0 || (pairs>1 && ((ids[a].size()>1 && ids[b].size()>1) ||
                !request["orbits"].get(data.labels[split],false).asBool() || request["marked"].asBool()))) {
            result["reason"] = "The incidences between the components of "+data.labels[a]+" and "+data.labels[b]+
                               " were not determined.";
            break;
        }
        Z multiplicity = count/pairs;
        for (const auto& u : ids[a]) for (const auto& v : ids[b]) for (Z i=1; i<=multiplicity; ++i) {
            auto label = "C"+u+"-"+v+"."+i.get_str();
            result["edges"].append(cell(label,{u,v},{u,v}));
        }
    }
    return result;
}

struct Graph {
    Names labels;
    std::map<std::string,size_t> indices;
    std::vector<std::map<size_t,unsigned>> edges;
    std::vector<std::vector<unsigned>> degrees;
    bool graph = true;
    explicit Graph(const Value& input) {
        for (const auto& v : input["vertices"]) {
            auto label = v["id"].asString();
            indices[label] = labels.size();
            labels.push_back(label);
        }
        edges.resize(labels.size());
        for (const auto& c : input["cells"]) {
            const auto& v = c["vertices"];
            if (v.size()!=2 || c.get("dimension",1).asInt()!=1) { graph=false; return; }
            auto a=indices.at(v[0].asString()), b=indices.at(v[1].asString());
            ++edges[a][b];
            ++edges[b][a];
        }
        for (const auto& neighbors : edges) {
            std::vector<unsigned> values;
            for (const auto& [key,n] : neighbors) values.push_back(n);
            std::sort(values.begin(),values.end());
            degrees.push_back(std::move(values));
        }
    }
    unsigned multiplicity(size_t a, size_t b) const {
        auto it = edges[a].find(b);
        return it==edges[a].end() ? 0 : it->second;
    }
};
bool branch(const std::string& name,const std::string& parent) {
    if (name==parent) return true;
    auto prefix=parent+".";
    return name.size()>prefix.size() && name.compare(0,prefix.size(),prefix)==0 &&
        std::all_of(name.begin()+prefix.size(),name.end(),[](unsigned char c){return std::isdigit(c);});
}

Value aliases(const Value& old, const Value& found) {
    Graph source(old), target(found);
    const size_t n=source.labels.size();
    if (!source.graph || !target.graph || n!=target.labels.size()) return Value();
    std::vector<std::vector<size_t>> candidates(n);
    for (size_t i=0; i<n; ++i) {
        auto name=source.labels[i];
        auto same=target.indices.find(name);
        if (same!=target.indices.end()) { candidates[i]={same->second}; continue; }
        const auto& vertex=old["vertices"][Json::UInt(i)];
        auto parents=names(vertex["ambient_components"]);
        if (parents.empty() && vertex.isMember("ambient_divisor")) parents.push_back(vertex["ambient_divisor"].asString());
        if (parents.empty() && !name.empty() && (name.back()=='u' || name.back()=='v')) parents.push_back(name.substr(0,name.size()-1));
        for (const auto& [other,j] : target.indices) for (const auto& p : parents)
            if (branch(other,p)) { candidates[i].push_back(j); break; }
        if (candidates[i].empty()) return Value();
    }
    std::vector<size_t> chosen(n,n);
    std::vector<bool> used(n);
    if (old["cells"].empty() && found["cells"].empty()) {
        std::map<std::vector<size_t>,std::vector<size_t>> groups;
        for (size_t i=0; i<n; ++i) groups[candidates[i]].push_back(i);
        for (const auto& [targets,keys] : groups) {
            if (keys.size()!=targets.size()) return Value();
            for (size_t i=0; i<keys.size(); ++i) {
                if (used[targets[i]]) return Value();
                chosen[keys[i]]=targets[i];
                used[targets[i]]=true;
            }
        }
    } else {
        std::function<bool(size_t)> match = [&](size_t matched) {
            if (matched==n) return true;
            size_t key=n, best=0;
            for (size_t i=0; i<n; ++i) if (chosen[i]==n) {
                size_t adjacent=0;
                for (const auto& [j,count] : source.edges[i]) adjacent += chosen[j]!=n;
                if (key==n || candidates[i].size()<candidates[key].size() ||
                    (candidates[i].size()==candidates[key].size() && adjacent>best)) { key=i; best=adjacent; }
            }
            for (auto target_id : candidates[key]) {
                if (used[target_id] || source.degrees[key]!=target.degrees[target_id] ||
                    source.multiplicity(key,key)!=target.multiplicity(target_id,target_id)) continue;
                bool valid=true;
                for (size_t j=0; j<n && valid; ++j) if (chosen[j]!=n)
                    valid=source.multiplicity(key,j)==target.multiplicity(target_id,chosen[j]);
                if (!valid) continue;
                chosen[key]=target_id; used[target_id]=true;
                if (match(matched+1)) return true;
                chosen[key]=n; used[target_id]=false;
            }
            return false;
        };
        if (!match(0)) return Value();
    }
    Value result(Json::objectValue);
    for (size_t i=0;i<n;++i) result[target.labels[chosen[i]]]=source.labels[i];
    return result;
}

Value binomial_layout(const Value& request) {
    std::map<int,unsigned> counts, positions, ray_names;
    const auto& pieces=request["pieces"];
    for (const auto& piece : pieces) if (piece[0].size()==1) ++counts[piece[0][0].asInt()];
    unsigned next=1;
    for (const auto& ray : request["exceptional"]) ray_names.emplace(ray.asInt(),next++);
    Names ids;
    unsigned cells=0;
    Value out(Json::arrayValue);
    for (const auto& piece : pieces) {
        const auto& subset=piece[0];
        Value record;
        record["generic_zero_rays"]=piece[1];
        if (subset.size()==1) {
            int ray=subset[0].asInt();
            const auto serial=++positions[ray];
            std::string id="E"+std::to_string(ray_names.at(ray));
            if (counts[ray]>1) id+="."+std::to_string(serial);
            record["id"]=id;
            record["ray_index"]=ray;
            record["dimension"]=request["width"].asInt()-1-int(request["intrinsic"].asBool());
        } else {
            record["id"]="C"+std::to_string(++cells);
            record["dimension"]=subset.size()-1;
            for (unsigned i=2; i<=3; ++i) {
                Names list;
                for (const auto& id : piece[i]) list.push_back(ids.at(id.asUInt()));
                record[i==2 ? "vertices" : "faces"]=array(list);
            }
        }
        ids.push_back(record["id"].asString());
        out.append(std::move(record));
    }
    return out;
}

// Identify common free axes from coordinate/fibre supports. An axis must be
// the sole identical input coordinate and absent from every other equation.
Value product_axes(const Value& charts, int dimension) {
    std::set<std::string> free;
    bool first=true;
    for (const auto& chart : charts) {
        if (chart["coordinates"].size()!=unsigned(dimension)) return Value();
        auto dependencies=names(chart["dependencies"]);
        std::set<std::string> blocked(dependencies.begin(),dependencies.end()), possible, keys;
        for (const auto& map : chart["coordinates"]) {
            auto key=map[0].asString();
            if (!keys.insert(key).second) return Value();
            if (map[1].isNull()) continue;
            auto axis=map[1].asString();
            if (blocked.count(axis)) continue;
            bool in_fibre=false, appears_elsewhere=false;
            for (const auto& f : chart["fibre"]) {
                if (!f[0].isNull() && f[0].asString()==axis) {in_fibre=true; continue;}
                for (const auto& name : f[1]) if (name.asString()==axis) appears_elsewhere=true;
            }
            for (const auto& g : chart["coordinates"]) if (g[0]!=map[0])
                for (const auto& name : g[2]) if (name.asString()==axis) appears_elsewhere=true;
            if (in_fibre && !appears_elsewhere) possible.insert(key);
        }
        if (first) {free=std::move(possible); first=false;}
        else {
            std::set<std::string> both;
            std::set_intersection(free.begin(),free.end(),possible.begin(),possible.end(),std::inserter(both,both.end()));
            free=std::move(both);
        }
        if (free.empty()) return Value();
    }
    if (dimension-static_cast<int>(free.size())<2) return Value();
    return array(Names(free.begin(),free.end()));
}

Value product_projection(const Value& request) {
    std::set<int> directions;
    for (const auto& axis : request["axes"]) directions.insert(axis.asInt());
    Value kept(Json::arrayValue);
    const auto& images = request["images"];
    for (Json::ArrayIndex i = 0; i < images.size(); ++i) {
        const auto& image = images[i];
        const auto& expected = image["expected"];
        if (!expected.isNull()) {
            if (image["coordinate"] != expected) return {};
        } else {
            for (const auto& axis : image["depends"])
                if (directions.count(axis.asInt())) return {};
            kept.append(i);
        }
    }
    return kept;
}

Value product_torus(const Value& request) {
    const auto& local=request["local"];
    const auto& base=request["base"];
    auto dimension=request["dimension"].asUInt();
    auto restrict_matrix=[&](const Value& matrix,const Value& row_indices) {
        Value result(Json::arrayValue);
        if (matrix.size()!=dimension) return Value();
        std::set<unsigned> rows,columns;
        for (const auto& j : row_indices) rows.insert(j.asUInt());
        for (const auto& j : base) columns.insert(j.asUInt());
        for (unsigned i=0; i<dimension; ++i) {
            if (matrix[i].size()!=dimension) return Value();
            Value row(Json::arrayValue);
            for (unsigned j=0; j<dimension; ++j) {
                if (rows.count(i)!=columns.count(j) && matrix[i][j].asBool()) return Value();
                if (rows.count(i) && columns.count(j)) row.append(j);
            }
            if (rows.count(i)) {
                Value slice(Json::arrayValue); slice.append(i); slice.append(std::move(row));
                result.append(std::move(slice));
            }
        }
        return result;
    };
    auto rays=restrict_matrix(request["rays"],local);
    if (rays.isNull()) return Value();
    auto linear=restrict_matrix(request["linear"],base);
    if (linear.isNull()) return Value();
    Value result(Json::arrayValue);
    result.append(std::move(rays)); result.append(std::move(linear));
    return result;
}

// The standard opens have already been reconstructed as the saturated graph
// of this ideal. A polynomial graph through the origin is affine space in
// its free coordinates. Its exceptional prime is the projectivized normal
// bundle, including when the centre is a coordinate subspace. Match its ideal
// and label on every open before using the geometric incidence of this bundle.
using Equations = std::set<atlas::local::Polynomial>;

Equations up_to_scalar(atlas::local::Basis rows) {
    Equations equations;
    for (auto& f : rows) {
        if (f.empty()) return {};
        const mpq_class scalar = f.begin()->second;
        for (auto& term : f) term.second /= scalar;
        equations.insert(std::move(f));
    }
    return equations;
}

Value coordinate_exceptional(const Value& request, bool polynomial_rows) {
    const Names base = names(request["variables"]), centre = names(request["centre"]);
    const std::set<std::string> variables(base.begin(), base.end()), generators(centre.begin(), centre.end());
    const auto& charts = request["charts"];
    if (generators.size() < 2 || generators.size() != centre.size() || charts.size() != centre.size() ||
        generators.size() > variables.size()) return {};
    Equations centre_equations;
    if (polynomial_rows) {
        // This optional certificate has a bounded sparse transport; larger
        // archives still use the full graph incidence calculation.
        if (base.size() > 16 || std::any_of(centre.begin(), centre.end(),
            [](const std::string& row) { return row.size() > 8192; })) return {};
        atlas::local::Basis basis;
        const atlas::local::Powers origin(base.size(), 0);
        for (const auto& row : centre) {
            auto equation = atlas::local::decode_row(row, base.size());
            if (equation.empty() || equation.count(origin)) return {};
            basis.push_back(std::move(equation));
        }
        if (!atlas::local::polynomial_graph(basis, base.size())) return {};
        centre_equations = up_to_scalar(std::move(basis));
    } else if (!std::includes(variables.begin(), variables.end(), generators.begin(), generators.end())) {
        return {};
    }
    std::string label;
    for (const auto& chart : charts) {
        const auto& boundary = chart["exceptional"];
        if (boundary.size() != 1) return {};
        const auto& component = boundary[0];
        const Names equations = names(component["ideal"]);
        if (std::set<std::string>(equations.begin(), equations.end()) != generators) {
            // The exceptional prime may use monic grevlex generators while
            // the original projective centre has monic lex generators.
            if (!polynomial_rows || equations.size() != centre.size()) return {};
            atlas::local::Basis basis;
            for (const auto& row : equations) {
                if (row.size() > 8192) return {};
                basis.push_back(atlas::local::decode_row(row, base.size()));
            }
            if (up_to_scalar(std::move(basis)) != centre_equations) return {};
        }
        const auto id = component["id"].asString();
        if (id.size() < 2 || id.front() != 'E' ||
            !std::all_of(id.begin()+1, id.end(), [](char c) { return c >= '0' && c <= '9'; }) ||
            (!label.empty() && id != label)) return {};
        label = id;
    }
    return label;
}

Value execute(const Value& r) {
    const auto op=r["operation"].asString();
    if (op=="dimension") return r["ambient"].asInt() - r["removed"].asInt();
    if (op=="product_dimensions") {
        Value result(Json::arrayValue);
        for (const auto& x : r["base"]) result.append(x.asInt()+r["free"].asInt());
        return result;
    }
    if (op=="binomial_layout") return binomial_layout(r);
    if (op=="product_axes") return product_axes(r["charts"],r["dimension"].asInt());
    if (op=="product_projection") return product_projection(r);
    if (op=="product_torus") return product_torus(r);
    if (op=="coordinate_exceptional") return coordinate_exceptional(r, false);
    if (op=="polynomial_graph_exceptional") return coordinate_exceptional(r, true);
    if (op=="cells") return Intersections(r).cells();
    if (op=="boundary") return boundary(r,Intersections(r));
    if (op=="aliases") return aliases(r["old"],r["found"]);
    throw std::invalid_argument("Unknown atlas incidence operation.");
}
}
extern "C" const char* atlas_incidence(const char* input) { return atlas::sparse_json::invoke(input,execute); }

extern "C" int atlas_coordinate_exceptional_abi() noexcept { return 1; }

extern "C" int atlas_polynomial_graph_exceptional_abi() noexcept { return 1; }
