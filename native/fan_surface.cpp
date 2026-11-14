// Curves on the strict transform in a verified regular three-dimensional fan.
#include "local_io.hpp"
#include <json/json.h>
#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>

extern "C" {
char* atlas_affine(const char*, size_t) noexcept;
char* atlas_ideal(const char*, size_t) noexcept;
void atlas_groebner_release(char*) noexcept;
const char* atlas_groebner_error() noexcept;
}

namespace {
using namespace atlas::local;
using Strings = std::vector<std::string>;
using Value = Json::Value;
using Operation = char*(*)(const char*, size_t) noexcept;

struct Unavailable : std::runtime_error { using std::runtime_error::runtime_error; };

Strings request(Operation op, const Strings& rows) {
    std::string input;
    for (const auto& row : rows) input += row + '\n';
    std::unique_ptr<char,decltype(&atlas_groebner_release)> answer(op(input.data(),input.size()), atlas_groebner_release);
    if (!answer) {
        const std::string message = atlas_groebner_error();
        if (op == atlas_ideal || message == "Unsupported number of variables." ||
            message == "Degree out of range." || message == "Exponent out of range.")
            throw Unavailable(message);
        throw std::runtime_error(message);
    }
    std::istringstream stream(answer.get());
    Strings lines;
    for (std::string s; std::getline(stream,s);) lines.push_back(std::move(s));
    return lines;
}
std::string axis_row(unsigned i) {
    Powers powers(3); powers.at(i) = 1;
    return encode_row({{powers,1}});
}
using Factor = std::pair<int,Polynomial>;
std::optional<std::vector<Factor>> factors(const Polynomial& strict, unsigned axis) {
    Polynomial on_divisor;
    for (const auto& [powers,c] : strict) if (!powers.at(axis)) on_divisor.emplace(powers,c);
    if (on_divisor.empty()) return {};
    std::vector<Factor> found;
    if (on_divisor.size()==1) {
        for (unsigned i=0;i<3;++i) if (on_divisor.begin()->first[i]!=0)
            found.emplace_back(i,decode_row(axis_row(i),3));
        return found;
    }
    const auto rows = request(atlas_affine,{"geometric_factors","x y z",format_polynomial(on_divisor,{"x","y","z"})});
    for (size_t i=1;i<rows.size();i+=2) {
        int weight, absolute;
        std::istringstream header(rows[i]);
        if (!(header>>weight>>absolute) || i+1>=rows.size()) throw std::runtime_error("Invalid factor certificate.");
        auto factor = decode_row(rows[i+1],3);
        int coord=-1;
        if (factor.size()==1) for (unsigned j=0;j<3;++j) {
            Powers unit(3); unit[j]=1;
            if (factor.begin()->first==unit) coord=int(j);
        }
        if (coord<0 && !absolute && request(atlas_ideal,{"plane_components 3 2 0",axis_row(axis),rows[i+1]})!=Strings{"1 1"})
            return {};
        found.emplace_back(coord,std::move(factor));
    }
    return found;
}

std::string powers_row(const Powers& powers) {
    std::ostringstream out;
    for (const auto& x : powers) out << ' ' << x;
    return out.str();
}
Polynomial laurent_factor(const Strings& rays, const Polynomial& f) {
    Strings rows{"lattice_support","3"};
    rows.insert(rows.end(),rays.begin(),rays.end());
    for (const auto& [powers,c] : f) rows.push_back(powers_row(powers));
    auto characters=request(atlas_affine,rows);
    if (characters.size()!=f.size()) throw std::runtime_error("Incomplete Laurent character map.");
    Polynomial normalized;
    auto it=f.begin();
    for (const auto& character : characters) {
        Powers powers(3); std::istringstream in(character);
        if (!(in>>powers[0]>>powers[1]>>powers[2])) throw std::runtime_error("Invalid lattice character.");
        normalized.emplace(std::move(powers),it++->second);
    }
    const auto unit=normalized.begin()->second;
    for (auto& [powers,c] : normalized) c/=unit;
    return normalized;
}

// Coordinate curves are the orbit closure of an unordered pair of rays.
// Other curves are determined by their parent ray and a normalized Laurent
// equation on its dense torus. Both descriptions agree across all charts.
struct Curve {
    int first,second;
    Polynomial equation;
    bool operator<(const Curve& b) const {
        return std::tie(first,second,equation)<std::tie(b.first,b.second,b.equation);
    }
};
struct Surface {
    std::map<Curve,unsigned> curves;
    std::map<std::string,std::vector<unsigned>> groups;
    std::vector<std::string> parents;
    std::map<std::pair<unsigned,unsigned>,mpz_class> counts;
    std::map<std::pair<std::string,unsigned>,std::optional<std::vector<Factor>>> factor_cache;

    bool chart(const Value& chart, const Strings& ray_rows) {
        std::array<int,3> rays;
        Strings local_rays;
        for (unsigned i=0;i<3;++i) {
            rays[i]=chart["rays"][i].asInt();
            local_rays.push_back(ray_rows.at(rays[i]));
        }
        const auto row=chart["strict"].asString();
        const auto strict=decode_row(row,3);
        Strings ideals,owned;
        for (const auto& f : chart["owned"]) owned.push_back(f.asString());
        std::vector<unsigned> pieces;
        for (const auto& component : chart["boundary"]) {
            const auto i=component[0].asUInt();
            const auto name=component[1].asString();
            auto [entry,fresh]=factor_cache.try_emplace({row,i});
            if (fresh) entry->second=factors(strict,i);
            if (!entry->second) return false;
            for (const auto& [coordinate,factor] : *entry->second) {
                Curve key{rays.at(i),-1,{}};
                if (coordinate<0) key.equation=laurent_factor(local_rays,factor);
                else {
                    key.second=rays.at(coordinate);
                    if (key.first>key.second) std::swap(key.first,key.second);
                }
                auto [curve,added]=curves.emplace(std::move(key),parents.size());
                auto id=curve->second;
                if (added) {parents.push_back(name); groups[name].push_back(id);}
                if (parents.at(id)!=name || std::find(pieces.begin(),pieces.end(),id)!=pieces.end()) return false;
                pieces.push_back(id);
                ideals.push_back(axis_row(i)); ideals.push_back(encode_row(factor));
            }
        }
        std::string sizes=std::to_string(pieces.size());
        for (size_t i=0;i<pieces.size();++i) sizes+=" 2";
        Strings crossings{"crossings 3 2 1 "+sizes+" "+std::to_string(owned.size())+" 0",row};
        crossings.insert(crossings.end(),ideals.begin(),ideals.end());
        crossings.insert(crossings.end(),owned.begin(),owned.end());
        if (request(atlas_ideal,crossings)!=Strings{"none"}) return false;
        Strings strata{"strata 3 "+std::to_string(1+owned.size())+" "+sizes+" 0",row};
        strata.insert(strata.end(),owned.begin(),owned.end());
        strata.insert(strata.end(),ideals.begin(),ideals.end());
        for (const auto& result : request(atlas_ideal,strata)) {
            if (result=="none") break;
            std::istringstream fields(result); std::string tag; unsigned size;
            if (!(fields>>tag>>size) || tag!="stratum") return false;
            if (size==2) {
                unsigned a,b; mpz_class chi;
                if (!(fields>>a>>b>>chi)) return false;
                counts[std::minmax(pieces.at(a),pieces.at(b))]+=chi;
            }
        }
        return true;
    }
    Value complex() const {
        Value result;
        result["status"]="complete";
        result["convention"]="Exceptional divisor on the smooth strict transform in the germ over the origin over Qbar.";
        auto& vertices=result["vertices"]=Value(Json::arrayValue);
        auto& cells=result["cells"]=Value(Json::arrayValue);
        std::vector<std::string> names(parents.size());
        std::map<std::string,std::string> parent;
        for (const auto& [label,group] : groups) {
            for (size_t i=0;i<group.size();++i) {
                auto& name=names[group[i]];
                name=label+(group.size()>1 ? "."+std::to_string(i+1) : "");
                if (!parent.emplace(name,label).second) return {};
                Value vertex;vertex["id"]=name;vertex["dimension"]=1;vertices.append(std::move(vertex));
            }
        }
        using Pair=std::pair<std::string,std::string>;
        std::map<Pair,mpz_class> pairs,serial;
        bool branches_meet=false;
        for (const auto& [pair,count] : counts) if (count!=0) {
            const auto [a,b]=pair;
            branches_meet |= parents[a]==parents[b];
            pairs[std::minmax(names[a],names[b])]=count;
        }
        for (const auto& [names,count] : pairs) {
            Pair key=branches_meet ? names : Pair(std::minmax(parent.at(names.first),parent.at(names.second)));
            for (mpz_class i=0;i<count;++i) {
                auto& number=++serial[key]; Value cell;
                cell["id"]="C"+key.first+"-"+key.second+"."+number.get_str();cell["dimension"]=1;
                cell["vertices"].append(names.first);cell["vertices"].append(names.second);cell["faces"]=cell["vertices"];
                cells.append(std::move(cell));
            }
        }
        return result;
    }
};
}
namespace atlas {
Json::Value fan_surface_complex(const Json::Value& input) {
    Strings rays;
    for (const auto& row : input["rays"]) {
        std::string ray;
        if (row.size()!=3) return {};
        for (const auto& x : row) ray+=" "+x.asString();
        rays.push_back(std::move(ray));
    }
    Surface surface;
    try {
        for (const auto& chart : input["charts"]) {
            if (chart["rays"].size()!=3 || !surface.chart(chart,rays)) return {};
        }
    } catch (const Unavailable&) {
        // Preserve the general chart gluer for packed exponent overflows or
        // a geometric certificate this kernel cannot establish.
        return {};
    }
    return surface.complex();
}
}
