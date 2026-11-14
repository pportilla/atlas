// Glue components meeting the origin by regular functions on the boundary.
// A product of squarefree coordinate eliminants with free affine coordinates
// gives one geometric component per tuple of roots. A coordinate ratio on the
// original space that extends to a chart determines that root on all overlaps.
#include "local_io.hpp"
#include "primary_decomposition.hpp"
#include "rational_sparse_json.hpp"
#include <flint/fmpq.h>
#include <flint/fmpq_poly.h>
#include <flint/qqbar.h>
#include <numeric>
#include <set>

namespace {
using namespace atlas::local;
namespace ideal = atlas::primary;
using Value = Json::Value;
using Key = std::vector<std::string>;
using Ratio = std::pair<std::string,std::string>;
using Axes = std::vector<size_t>;
using Base = std::map<Ratio,Polynomial>;
using Counts = std::map<Key,Value>;

bool contains(const Basis& basis,const Basis& equations) {
    return std::all_of(equations.begin(),equations.end(),[&](const auto& f){return ideal::member_lex(f,basis);});
}
void append(Basis& a,const Basis& b) { a.insert(a.end(),b.begin(),b.end()); }

struct Chart {
    Key coordinates;
    size_t n;
    Basis ambient, region, fibre;
    std::vector<Basis> removed;
    std::map<std::string,Basis> divisors;
    std::map<std::string,Polynomial> images;
    std::map<size_t,Ratio> known_ratio;
    std::map<std::string,bool> regular;
    Basis read(const Value& rows) const {
        Basis a;
        for (const auto& row:rows) a.push_back(parse_expression(coordinates,row.asString()));
        return a;
    }
    explicit Chart(const Value& chart) {
        for (const auto& x:chart["coordinates"]) coordinates.push_back(x.asString());
        n=coordinates.size();
        ambient=ideal::reduced(read(chart["ambient_ideal"]),n,true);
        region=read(chart["region"]);
        fibre=read(chart["fibre"]);
        for (const auto& row:chart["removed"]) removed.push_back(read(row));
        for (const auto& k:chart["components"].getMemberNames()) divisors[k]=read(chart["components"][k]);
        for (const auto& k:chart["map"].getMemberNames()) images[k]=parse_expression(coordinates,chart["map"][k].asString());
    }
    // In the ambient coordinate ring a regular denominator gives equality in
    // the total quotient ring. Its zero set may contain the entire section.
    std::optional<Ratio> ratio(size_t axis) {
        if (known_ratio.count(axis)) return known_ratio.at(axis);
        Powers unit(n); unit[axis]=1;
        for (const auto& [den,g]:images) {
            if (g.empty()) continue;
            auto multiple=ideal::multiply(g,{{unit,1}});
            for (const auto& [num,f]:images) {
                if (num==den) continue;
                auto relation=multiple;
                for (const auto& [e,c]:f)
                    if ((relation[e]-=c)==0) relation.erase(e);
                if (!ideal::member_lex(relation,ambient)) continue;
                if (!regular.count(den)) regular[den]=contains(ambient,ideal::saturate_lex(ambient,g,n));
                if (regular.at(den)) return known_ratio[axis]={num,den};
            }
        }
        return {};
    }
};

// Recognize, from a reduced lexicographic basis, exact coordinate cylinders:
// linear graph variables, free variables, and independent squarefree
// univariate equations. The proof specifies the quotient ring, including all
// fibres. Each eliminant is normalized before comparing it on other charts.
struct Section {
    Base base;
    std::map<Ratio,size_t> axes;
    Basis basis;
    int dimension;
};
std::optional<Section> section(Chart& chart,Basis basis,int dimension) {
    Section result{{},{},std::move(basis),dimension};
    std::set<size_t> pivots;
    for (const auto& f:result.basis) {
        const auto& head=f.rbegin()->first;
        auto first=std::find_if(head.begin(),head.end(),[](const auto& k){return k!=0;});
        if (first==head.end()) return {};
        size_t axis=first-head.begin();
        if (!pivots.insert(axis).second) return {};
        auto degree=std::accumulate(head.begin(),head.end(),mpz_class(0));
        if (degree==1) continue;
        Polynomial p;
        for (const auto& [e,c]:f) {
            if (std::accumulate(e.begin(),e.end(),mpz_class(0))!=e[axis]) return {};
            p[{e[axis]}]=c/f.rbegin()->second;
        }
        auto key=chart.ratio(axis);
        if (!key || result.base.count(*key)) return {};
        result.base[*key]=p;
        result.axes[*key]=axis;
    }
    if (result.base.empty() || static_cast<int>(chart.n-pivots.size())!=dimension) return {};
    return result;
}

// -1 some sheets, 0 no sheets, 1 all sheets. A proper closed subset of
// an irreducible d-fold has dimension < d. Every sheet here has dimension d.
int includes_components(const Basis& section,const Basis& equations,int d,size_t n) {
    if (contains(section,equations)) return 1;
    auto rows=section;
    append(rows,equations);
    return ideal::dimension_lex(ideal::reduced(rows,n,true),n)<d ? 0 : -1;
}
int owned(const Section& s,const Chart& chart,const Basis& orbit,bool require_region=true) {
    Basis basis=s.basis;
    if (!orbit.empty()) {
        Axes axes;
        for (const auto& [_,axis]:s.axes) axes.push_back(axis);
        for (const auto& f:orbit) {
            Polynomial moved;
            for (const auto& [e,c]:f) {
                Powers row(chart.n);
                for (size_t i=0;i<e.size();++i) row[axes[i]]=e[i];
                moved.emplace(std::move(row),c);
            }
            basis.push_back(std::move(moved));
        }
        basis=ideal::reduced(basis,chart.n,true);
    }
    int owner=require_region?includes_components(basis,chart.region,s.dimension,chart.n):1;
    if (!owner) return 0;
    for (const auto& hole:chart.removed) {
        int removed=includes_components(basis,hole,s.dimension,chart.n);
        if (removed==1) return 0;
        if (removed<0) owner=-1;
    }
    return owner;
}

struct Sheets {
    Base base;
    std::vector<Basis> orbits;
    size_t degree=0;
    bool initialize(const Base& parameters) {
        mpz_class size=1;
        for (const auto& [_,p]:parameters) {
            for (const auto& [f,m]:ideal::factors(p,1)) if (m!=1) return false;
            size*=p.rbegin()->first.front();
        }
        if (!mpz_fits_ulong_p(size.get_mpz_t()) || size>Value::maxUInt)
            throw std::length_error("Too many geometric components to serialize.");
        degree=size.get_ui(); base=parameters;
        return true;
    }
    // On a rational orbit every geometrically defined ownership condition
    // over QQ is constant under Galois. Decompose only when the whole base
    // has different owners. This includes correlations between distinct roots.
    void split_orbits() {
        if (!orbits.empty()) return;
        Basis equations;
        for (const auto& [_,p]:base) {
            Polynomial f;
            for (const auto& [e,c]:p) {
                Powers row(base.size()); row[equations.size()]=e.front();
                f.emplace(std::move(row),c);
            }
            equations.push_back(std::move(f));
        }
        for (const auto& comp:ideal::decompose(equations,base.size(),true)) orbits.push_back(comp.prime);
    }
};

// For a deletion that selects only some rational orbits, evaluate each
// orbit's ideal on the ordered root tuple. This is the same exact
// (real part, imaginary part) ordering used by the public algebraic fields.
struct Roots {
    slong count;
    qqbar_ptr values;
    std::vector<size_t> order;
    explicit Roots(const Polynomial& polynomial)
        : count(polynomial.rbegin()->first.front().get_si()), values(_qqbar_vec_init(count)), order(count) {
        fmpq_poly_t p;
        fmpq_poly_init(p);
        fmpq_t c;
        fmpq_init(c);
        for (const auto& [e,a]:polynomial) {
            fmpq_set_mpq(c,a.get_mpq_t());
            fmpq_poly_set_coeff_fmpq(p,e.front().get_si(),c);
        }
        qqbar_roots_fmpq_poly(values,p,0);
        fmpq_poly_clear(p);
        fmpq_clear(c);
        std::iota(order.begin(),order.end(),0);
        std::sort(order.begin(),order.end(),[&](size_t a,size_t b){
            int real=qqbar_cmp_re(values+a,values+b);
            return real ? real<0 : qqbar_cmp_im(values+a,values+b)<0;
        });
    }
    Roots(const Roots&)=delete;
    ~Roots(){_qqbar_vec_clear(values,count);}
};

bool on_orbit(const Basis& equations,const std::vector<qqbar_srcptr>& tuple) {
    qqbar_t value,term,power;
    qqbar_init(value);qqbar_init(term);qqbar_init(power);
    fmpq_t c;fmpq_init(c);
    fmpz_t exponent;fmpz_init(exponent);
    bool vanishing=true;
    for (const auto& f:equations) {
        qqbar_zero(value);
        for (const auto& [e,a]:f) {
            fmpq_set_mpq(c,a.get_mpq_t());qqbar_set_fmpq(term,c);
            for (size_t k=0;k<tuple.size();++k) if (e[k]!=0) {
                fmpz_set_mpz(exponent,e[k].get_mpz_t());
                qqbar_pow_fmpz(power,tuple[k],exponent);qqbar_mul(term,term,power);
            }
            qqbar_add(value,value,term);
        }
        if (!qqbar_is_zero(value)) {vanishing=false;break;}
    }
    fmpz_clear(exponent);fmpq_clear(c);
    qqbar_clear(value);qqbar_clear(term);qqbar_clear(power);
    return vanishing;
}

std::vector<size_t> orbit_indices(Sheets& sheets) {
    sheets.split_orbits();
    std::vector<std::unique_ptr<Roots>> roots;
    for (const auto& [_,p]:sheets.base) roots.push_back(std::make_unique<Roots>(p));
    std::vector<size_t> result;
    for (size_t i=0;i<sheets.degree;++i) {
        std::vector<qqbar_srcptr> tuple;
        size_t index=i;
        for (const auto& list:roots) {
            tuple.push_back(list->values+list->order[index%list->count]);index/=list->count;
        }
        auto match=std::find_if(sheets.orbits.begin(),sheets.orbits.end(),
                                [&](const Basis& p){return on_orbit(p,tuple);});
        if (match==sheets.orbits.end()) throw std::logic_error("An exact sheet has no rational orbit.");
        result.push_back(match-sheets.orbits.begin());
    }
    return result;
}
Key without(Key key,size_t i) { key.erase(key.begin()+i);return key; }
std::string cell_id(const Key& key,size_t i,bool split) {
    if (key.size()==1) return key.front()+(split?"."+std::to_string(i+1):"");
    std::string id="C";
    for (const auto& k:key) { if (id!="C") id+="-"; id+=k; }
    return id+"."+std::to_string(i+1);
}

Value execute(const Value& input) {
    Counts counts;
    std::set<Key> split;
    for (const auto& item:input["counts"]) {
        Key key;
        for (const auto& k:item[0]) key.push_back(k.asString());
        std::sort(key.begin(),key.end());
        counts[key]=item[1];
        if (item[1].isNull() || item[1].asString()!="1") split.insert(key);
    }
    if (split.empty()) return {};
    for (const auto& [key,count]:counts) {
        for (const auto& face:split)
            if (std::includes(key.begin(),key.end(),face.begin(),face.end())) {split.insert(key);break;}
        if (!split.count(key) && (count.isNull() || count.asString()!="1")) return {};
        if (key.size()>1) for (size_t i=0;i<key.size();++i) if (!counts.count(without(key,i))) return {};
    }
    int dimension=input["dimension"].asInt();
    std::map<Key,Sheets> parameters;
    struct Piece { size_t chart;Section section;int owner; };
    std::map<Key,std::vector<Piece>> pieces;
    std::vector<Chart> charts;
    for (const auto& row:input["charts"]) {
        if (!row["coefficient_field"].isNull()) return {};
        auto& chart=charts.emplace_back(row);
        if (chart.images.size()!=static_cast<size_t>(dimension)) return {};
        Basis map;
        for (const auto& [_,f]:chart.images) map.push_back(f);
        if (std::multiset<Polynomial>(map.begin(),map.end())!=std::multiset<Polynomial>(chart.fibre.begin(),chart.fibre.end()))
            return {};
        for (const auto& key:split) {
            Basis equations=chart.ambient;
            bool present=true;
            for (const auto& k:key) {
                if (!chart.divisors.count(k)) {present=false;break;}
                append(equations,chart.divisors.at(k));
            }
            if (!present) continue;
            auto basis=ideal::reduced(equations,chart.n,true);
            if (ideal::dimension_lex(basis,chart.n)<0) continue;
            // A chart entirely deleted on this intersection supplies no sheet.
            if (std::any_of(chart.removed.begin(),chart.removed.end(),[&](const auto& h){return contains(basis,h);}))
                continue;
            if (!contains(basis,chart.fibre)) return {};
            auto s=section(chart,std::move(basis),dimension-static_cast<int>(key.size()));
            if (!s) return {};
            auto& sheets=parameters[key];
            if (sheets.base.empty() && !sheets.initialize(s->base)) return {};
            if (s->base!=sheets.base) return {};
            int owns=owned(*s,chart,{});
            if (owns<0) sheets.split_orbits();
            pieces[key].push_back({charts.size()-1,std::move(*s),owns});
        }
    }
    for (const auto& key:split) {
        auto& sheets=parameters[key];
        if (!sheets.degree) return {};
        if (!counts.at(key).isNull() && counts.at(key).asString()!=std::to_string(sheets.degree)) return {};
        std::vector<unsigned> owners(std::max<size_t>(1,sheets.orbits.size()));
        for (const auto& part:pieces[key]) {
            for (size_t i=0;i<owners.size();++i) {
                int yes=part.owner<0 ? owned(part.section,charts[part.chart],sheets.orbits.at(i)) : part.owner;
                if (yes<0) return {};
                owners[i]+=yes;
            }
        }
        if (!std::all_of(owners.begin(),owners.end(),[](auto n){return n==1;})) return {};
    }
    Value result;
    result["status"]="complete";
    result["convention"]="Ambient exceptional divisor in the germ over the origin over Qbar.";
    result["vertices"]=Value(Json::arrayValue);result["cells"]=Value(Json::arrayValue);
    // Give every coordinate polynomial a common abstract ordering of its
    // complex roots. Restricting a tuple to the coordinates of a face gives
    // the actual face attachment; no choices of approximate roots are needed.
    auto project=[&](const Base& base,size_t index,const Key& face)->std::optional<size_t> {
        if (!split.count(face)) return 0;
        const auto& target=parameters.at(face).base;
        if (!std::includes(base.begin(),base.end(),target.begin(),target.end())) return {};
        size_t projected=0,place=1;
        for (const auto& [ratio,p]:base) {
            size_t radix=p.rbegin()->first.front().get_ui();
            if (target.count(ratio)) {
                projected+=place*(index%radix); place*=radix;
            }
            index/=radix;
        }
        return projected;
    };
    auto& certificate=result["finite_sheet_certificate"];
    certificate["bases"]=Value(Json::arrayValue);
    std::map<Polynomial,std::string> families;
    std::set<std::string> reserved;
    for (const auto& chart:charts) reserved.insert(chart.coordinates.begin(),chart.coordinates.end());
    auto family_for=[&](const Polynomial& p) {
        auto pos=families.find(p);
        if (pos!=families.end()) return pos->second;
        std::string name="sheetRoot"+std::to_string(families.size()+1);
        // Keep every generated constant distinct from a chart variable.
        while (std::any_of(reserved.begin(),reserved.end(),[&](const auto& v){
            return v.compare(0,name.size()+1,name+"_")==0;
        })) name+="x";
        auto& family=result["root_families"][name];
        family["polynomial"]=format_polynomial(p,{"T"});
        family["degree"]=Json::UInt64(p.rbegin()->first.front().get_ui());
        family["ordering"]="Increasing real part, then imaginary part.";
        families.emplace(p,name);
        return name;
    };
    for (const auto& [key,s]:parameters) {
        Value entry;
        for (const auto& label:key) entry["intersection"].append(label);
        entry["degree"]=Json::UInt64(s.degree);
        for (const auto& [ratio,p]:s.base) {
            Value row;
            row["numerator"]=ratio.first;row["denominator"]=ratio.second;
            row["polynomial"]=format_polynomial(p,{"T"});
            row["root_family"]=family_for(p);
            entry["parameters"].append(row);
        }
        certificate["bases"].append(std::move(entry));
    }
    // Record geometric chart ideals and chart membership in addition to
    // ownership. A later chart can meet a sheet whose generic point is
    // owned by an earlier chart. A deleted centre can remove that sheet
    // entirely from the open chart, which needs a separate visibility test.
    certificate["chart_sections"]=Value(Json::arrayValue);
    for (size_t ci=0;ci<charts.size();++ci) {
        Value entry;
        entry["coordinates"]=Value(Json::arrayValue);
        for (const auto& v:charts[ci].coordinates) entry["coordinates"].append(v);
        entry["divisors"]=Value(Json::objectValue);
        certificate["chart_sections"].append(std::move(entry));
    }
    for (auto& [key,s]:parameters) if (key.size()==1) {
        const auto& label=key.front();
        std::vector<size_t> tuple_orbits;
        for (size_t ci=0;ci<charts.size();++ci) if (charts[ci].divisors.count(label))
            certificate["chart_sections"][Json::ArrayIndex(ci)]["divisors"][label]=Value(Json::arrayValue);
        for (const auto& part:pieces.at(key)) {
            const Chart& chart=charts[part.chart];
            auto& records=certificate["chart_sections"][Json::ArrayIndex(part.chart)]["divisors"][label];
            int visible=owned(part.section,chart,{},false);
            if (!visible) continue;
            std::vector<int> visible_orbits;
            if (visible<0) {
                if (tuple_orbits.empty()) tuple_orbits=orbit_indices(s);
                for (const auto& prime:s.orbits) {
                    int yes=owned(part.section,chart,prime,false);
                    if (yes<0) return {};
                    visible_orbits.push_back(yes);
                }
            }
            for (size_t i=0;i<s.degree;++i) {
                if (visible<0 && !visible_orbits.at(tuple_orbits[i])) continue;
                Value sheet;
                sheet["id"]=cell_id(key,i,true);
                size_t index=i;
                for (const auto& [ratio,p]:s.base) {
                    size_t radix=p.rbegin()->first.front().get_ui();
                    auto root=family_for(p)+"_"+std::to_string(index%radix+1);
                    sheet["restrictions"].append(chart.coordinates[part.section.axes.at(ratio)]+"-"+root);
                    index/=radix;
                }
                records.append(std::move(sheet));
            }
        }
    }
    for (const auto& [key,_]:counts) {
        bool separated=split.count(key);
        const Base base=separated?parameters.at(key).base:Base{};
        for (size_t i=0;i<(separated?parameters.at(key).degree:1);++i) {
            Value cell;
            cell["id"]=cell_id(key,i,separated);
            if (separated) {
                size_t index=i;
                for (const auto& [ratio,p]:base) {
                    size_t radix=p.rbegin()->first.front().get_ui();
                    Value marking;
                    marking["numerator"]=ratio.first;marking["denominator"]=ratio.second;
                    marking["root"]=family_for(p)+"_"+std::to_string(index%radix+1);
                    cell["sheet_parameters"].append(std::move(marking));
                    index/=radix;
                }
            }
            if (key.size()==1) {
                cell["dimension"]=dimension-1;
                result["vertices"].append(cell);
            } else {
                cell["dimension"]=Json::UInt64(key.size()-1);
                for (size_t j=0;j<key.size();++j) {
                    Key vertex{key[j]},face=without(key,j);
                    auto v=project(base,i,vertex), f=project(base,i,face);
                    if (!v || !f) return {};
                    cell["vertices"].append(cell_id(vertex,*v,split.count(vertex)));
                    cell["faces"].append(cell_id(face,*f,split.count(face)));
                }
                result["cells"].append(cell);
            }
        }
    }
    return result;
}
}
extern "C" const char* atlas_finite_sheets(const char* input) noexcept {
    return atlas::sparse_json::invoke(input,execute);
}
