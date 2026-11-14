// Construct a verified regular monomial modification and its owned origin fibres.
// Every operation here is bounded. An uncertain crossing or orbit integral
// returns null and leaves the general chart verifier available to the caller.
#include "chart_equation.hpp"
#include "rational_sparse_json.hpp"

#include <memory>
#include <set>

extern "C" {
char* atlas_affine(const char*, size_t) noexcept;
char* atlas_ideal(const char*, size_t) noexcept;
void atlas_groebner_release(char*) noexcept;
}
namespace atlas {
std::string fan_cover(std::istream&, bool);
Json::Value torus_section_plan(const Json::Value&);
}

namespace {
using Value = Json::Value;
using Row = std::vector<int>;
using Rows = std::vector<Row>;
using Strings = std::vector<std::string>;

Value array() { return Value(Json::arrayValue); }

template<class Range>
std::string join(const Range& values) {
    std::ostringstream out;
    bool first = true;
    for (const auto& value : values) {
        if (!first) out << ' ';
        out << value;
        first = false;
    }
    return out.str();
}

Value json_row(const Row& values) {
    Value row = array();
    for (auto x : values) row.append(x);
    return row;
}

Strings read_lines(const std::string& text) {
    std::istringstream stream(text);
    Strings out;
    for (std::string line; std::getline(stream,line);) out.push_back(std::move(line));
    return out;
}

using Operation = char*(*)(const char*,size_t) noexcept;
Strings operate(Operation operation, const std::string& input) {
    std::unique_ptr<char, decltype(&atlas_groebner_release)> result(
        operation(input.data(),input.size()),atlas_groebner_release);
    if (!result) return {};
    return read_lines(result.get());
}

Row bits(unsigned mask, int n) {
    Row row(n);
    for (int j=0;j<n;++j) row[j] = (mask >> j) & 1U;
    return row;
}

unsigned support(const Row& row) {
    unsigned mask = 0;
    for (size_t j=0;j<row.size();++j) if (row[j]) mask |= 1U << j;
    return mask;
}

struct Input {
    int n = 0;
    Strings names, coords;
    Rows rays, cones, exponents;
    std::string equation;
    Value frame = array(), centres = array();

    bool read(const Value& request) {
        n = request["variables"].size();
        const auto& fan = request["fan"];
        if (n<3 || n>4 || !request["terms"].isArray() || request["terms"].empty() ||
                request["terms"].size()>3 || !fan["cones"].isArray() || fan["cones"].empty() ||
                fan["cones"].size()>4 || !fan["rays"].isArray() || fan["rays"].size()<unsigned(n) ||
                fan["rays"].size()>unsigned(n)*fan["cones"].size()) return false;
        for (const auto& name : request["variables"]) {
            if (!name.isString()) return false;
            auto v = name.asString();
            if (v.empty() || v.size()>24 || !std::isalpha(static_cast<unsigned char>(v[0])) ||
                    !std::all_of(v.begin()+1,v.end(),[](unsigned char c){return std::isalnum(c) || c=='_';}) ||
                    std::find(names.begin(),names.end(),v)!=names.end()) return false;
            names.push_back(v);
            coords.push_back("u"+std::to_string(names.size()));
            frame.append(json_row(bits(1U << (names.size()-1),n)));
        }
        for (const auto& data : fan["rays"]) {
            Row ray;
            if (!data.isArray() || data.size()!=unsigned(n)) return false;
            for (const auto& x : data) {
                if (!x.isInt() || x.asInt()<0 || x.asInt()>3) return false;
                ray.push_back(x.asInt());
            }
            if (!support(ray) || (rays.size()<unsigned(n) && json_row(ray)!=frame[unsigned(rays.size())]) ||
                    std::find(rays.begin(),rays.end(),ray)!=rays.end()) return false;
            rays.push_back(std::move(ray));
        }
        for (const auto& data : fan["cones"]) {
            Row cone;
            if (!data.isArray() || data.size()!=unsigned(n)) return false;
            for (const auto& x : data) {
                if (!x.isUInt() || x.asUInt()>=rays.size()) return false;
                cone.push_back(x.asInt());
            }
            cones.push_back(std::move(cone));
        }
        if (std::any_of(request["terms"].begin(),request["terms"].end(),[&](const Value& term) {
                return !term.isArray() || term.size()!=2 || !term[0].isString() ||
                       term[0].asString().size()>42 || !term[1].isArray() || term[1].size()!=unsigned(n) ||
                       std::any_of(term[1].begin(),term[1].end(),[](const Value& k) {
                           return !k.isString() || k.asString().size()!=1 || k.asString()[0]<'0' || k.asString()[0]>'3';
                       });
            })) return false;
        auto terms = atlas::sparse_json::read_terms(request["terms"],n);
        std::vector<int32_t> powers;
        Strings coefficients;
        for (const auto& [e,c] : terms) {
            if (!c || mpz_sizeinbase(c.get_num_mpz_t(),2)>64 || mpz_sizeinbase(c.get_den_mpz_t(),2)>64) return false;
            Row row;
            for (const auto& x : e) {row.push_back(x.get_si()); powers.push_back(x.get_si());}
            exponents.push_back(std::move(row));
            coefficients.push_back(c.get_str());
        }
        if (terms.empty()) return false;
        atlas::append_chart_equation(equation,powers.data(),coefficients,names);
        equation.pop_back();
        // The closure of a boundary ray maps to the coordinate subspace of
        // its positive entries. f belongs to its squared ideal precisely when
        // every nonzero support monomial has order at least two in these axes.
        std::set<unsigned> seen;
        for (size_t i=n;i<rays.size();++i) {
            const auto mask = support(rays[i]);
            if (mask==(1U<<n)-1 || !seen.insert(mask).second) continue;
            Row axes;
            for (int j=0;j<n;++j) if (mask & (1U<<j)) axes.push_back(j);
            for (const auto& e : exponents) {
                int order = 0;
                for (int j : axes) order += e[j];
                if (order<2) return false;
            }
            centres.append(json_row(axes));
        }
        return true;
    }
};

struct Chart {
    Rows monomials, region;
    Strings owned;
    Value region_text = array();
    std::string masks;

    Chart(const Input& input, const Row& cone, const std::string& excluded) {
        const int n = input.n;
        for (int j=0;j<n;++j) {
            Row powers;
            for (int ray : cone) powers.push_back(input.rays[ray][j]);
            monomials.push_back(std::move(powers));
        }
        std::istringstream parsed(excluded);
        for (unsigned mask; parsed>>mask;) region.push_back(bits(mask,n));
        // Public regions sort generators by number and then by ascending
        // positions of their factors. Preserve this ordering on replay.
        std::sort(region.begin(),region.end(),[](const Row& a, const Row& b) {
            auto degree = [](const Row& e){return std::accumulate(e.begin(),e.end(),0);};
            return degree(a)!=degree(b) ? degree(a)<degree(b) : a>b;
        });
        for (const auto& row : region) {
            std::string text;
            atlas::append_chart_equation(text,row.data(),{"1"},input.coords);
            text.pop_back();
            region_text.append(text);
        }
        auto include = [&](const Row& powers) {
            auto row = "1 "+join(powers);
            if (std::find(owned.begin(),owned.end(),row)==owned.end()) owned.push_back(row);
            masks += std::to_string(support(powers))+" ";
        };
        for (const auto& row : monomials) include(row);
        for (const auto& row : region) include(row);
    }
};

// Keep the same ideal operations used when independently re-reading an
// archived chart. Only the sufficient, finite monomial proofs are allowed.
bool verify_owned(int n, const Chart& chart, const Strings& ideals, bool strict, const std::string& jet,
                  bool integrals, Value& strata) {
    std::string sizes;
    for (size_t j=0;j<ideals.size();++j) sizes += " 1";
    auto append = [](std::string& out,const Strings& rows){for (const auto& row:rows) out += row+"\n";};
    auto crossed = ideals;
    if (strict) crossed.back() = jet;
    std::string input = "bounded_crossings "+std::to_string(n)+" "+std::to_string(n)+" 0 "+
        std::to_string(ideals.size())+sizes+" "+std::to_string(chart.owned.size())+" 0\n";
    append(input,crossed); append(input,chart.owned);
    if (operate(atlas_ideal,input)!=Strings{"none"}) return false;
    if (!integrals) return true;
    input = "monomial_strata "+std::to_string(n)+" "+std::to_string(chart.owned.size())+" "+
        std::to_string(ideals.size())+sizes+" 0\n";
    append(input,chart.owned); append(input,ideals);
    const auto rows = operate(atlas_ideal,input);
    if (rows.empty() || rows[0]=="undetermined") return false;
    for (const auto& row : rows) {
        if (row=="none") continue;
        std::istringstream fields(row);
        std::string tag,chi;
        unsigned count;
        if (!(fields>>tag>>count) || tag!="stratum" || count>ideals.size()) return false;
        Value pair = array(), indices = array();
        for (unsigned k=0;k<count;++k) {
            unsigned index;
            if (!(fields>>index) || index>=ideals.size()) return false;
            indices.append(index);
        }
        if (!(fields>>chi)) return false;
        pair.append(indices); pair.append(chi); strata.append(pair);
    }
    return true;
}

Value construct(const Value& request) {
    Input input;
    if (!input.read(request)) return {};
    const auto& [n,names,coords,rays,cones,exponents,equation,frame,centres] = input;
    std::ostringstream fan;
    fan << n << ' ' << rays.size() << '\n';
    for (const auto& ray : rays) fan << join(ray) << '\n';
    for (const auto& cone : cones) fan << join(cone) << '\n';
    std::istringstream cover(fan.str());
    const auto ownership = read_lines(atlas::fan_cover(cover,true));
    if (ownership.size()!=cones.size()+1 || ownership[0]!="complete") return {};

    std::vector<Chart> charts;
    std::ostringstream pullback;
    pullback << "fan_charts_with_projection\n" << join(names) << '\n' << join(coords) << '\n' << equation << '\n';
    for (int i=0;i<n;++i) pullback << join(bits(1U<<i,n)) << '\n';
    for (size_t i=0;i<cones.size();++i) {
        charts.emplace_back(input,cones[i],ownership[i+1]);
        for (int ray:cones[i]) pullback << join(rays[ray]) << '\n';
        pullback << charts.back().masks << '\n';
    }
    const auto rows = operate(atlas_affine,pullback.str());
    if (rows.size() != (n+4)*cones.size()) return {};
    Value plan, result;
    plan["dimension"]=n; plan["common_source"]=true; plan["intrinsic"]=false;
    plan["rays"]=request["fan"]["rays"];
    result["centres"]=centres; result["charts"]=array();
    for (size_t i=0;i<cones.size();++i) {
        const auto offset=i*(n+4);
        std::istringstream orders(rows[offset]);
        int strict;
        if (!(orders>>strict)) return {};
        Value verified, section, strata=array(), exceptional=array(), projection=array();
        section["rays"]=json_row(cones[i]);
        section["boundary"]=array();
        Strings ideals;
        for (int j=0;j<n;++j) {
            projection.append(rows[offset+4+j]);
            int ray=cones[i][j];
            if (ray<n) continue;
            int order,nu;
            if (!(orders>>order>>nu)) return {};
            exceptional.append(json_row({j,order,nu}));
            Value boundary=array();
            boundary.append(j); boundary.append("E"+std::to_string(ray-n+1));
            section["boundary"].append(boundary);
            ideals.push_back("1 "+join(bits(1U<<j,n)));
        }
        if (strict) ideals.push_back(rows[offset+2]);
        if (!verify_owned(n,charts[i],ideals,strict,rows[offset+3],request["integrals"].asBool(),strata)) return {};
        plan["charts"].append(section);
        verified["strict"]=strict ? Value(rows[offset+1]) : Value();
        verified["exceptional"]=exceptional;
        verified["projection"]=projection;
        verified["region"]=charts[i].region_text;
        verified["strata"]=strata;
        result["charts"].append(verified);
    }
    result["dual_complex"]=atlas::torus_section_plan(plan)["complex"];
    return result["dual_complex"].isNull() ? Value() : result;
}
} // namespace

extern "C" const char* atlas_monomial_resolution(const char* input) noexcept {
    return atlas::sparse_json::invoke(input,construct);
}
