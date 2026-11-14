// Replay a Newton modification of a hypersurface on a smooth polynomial graph.
// The two initial equations are tested on the tori by logarithmic Jacobian
// minors. For a transverse pair of Newton polytopes P,Q in dimension d,
// chi(V(P,Q)) = (-1)^d (Vol(Cayley(P,Q))-Vol(P)-Vol(Q)). All volumes are
// lattice-normalized: slicing the Cayley polytope integrates the mixed-volume
// polynomial and gives each mixed term coefficient one, including pure terms.
#include "groebner.hpp"
#include "local_io.hpp"
#include "rational_sparse_json.hpp"
#include <algorithm>
#include <map>
#include <limits>
#include <numeric>
#include <set>

extern "C" char* atlas_lattice_volume(const char*, size_t) noexcept;
extern "C" void atlas_groebner_release(char*) noexcept;
namespace atlas { std::string fan_cover(std::istream&,bool); }

namespace {
using atlas::local::Polynomial;
using atlas::local::Powers;
using Z = mpz_class;
using Q = mpq_class;
using JsonValue = Json::Value;
using Face = std::vector<int>;
using PointSet = std::vector<Powers>;

void add(Polynomial& f, const Powers& e, const Q& c) {
    auto [pos, fresh] = f.try_emplace(e, c);
    if (!fresh) pos->second += c;
    if (pos->second == 0) f.erase(pos);
}

Polynomial product(const Polynomial& f, const Polynomial& g) {
    Polynomial result;
    for (const auto& [e,c] : f) for (const auto& [h,b] : g) {
        Powers p(e);
        for (size_t i = 0; i < p.size(); ++i) p[i] += h[i];
        add(result, p, c*b);
    }
    return result;
}

JsonValue integer(const Z& a) {
    if (mpz_fits_slong_p(a.get_mpz_t())) return Json::Int64(a.get_si());
    return a.get_str();
}
std::vector<std::string> names(const JsonValue& array) {
    std::vector<std::string> out;
    for (const auto& v : array) out.push_back(v.asString());
    return out;
}

struct Graph {
    int n;
    std::vector<std::string> vars, chart_vars;
    Polynomial g, f;
    Graph(const JsonValue& input) : vars(names(input["variables"])) {
        if (vars.empty() || vars.size() >= 8) throw std::invalid_argument("Invalid graph dimension.");
        n = vars.size() + 1;
        auto all = vars;
        all.push_back(input["auxiliary"].asString());
        if (std::set<std::string>(all.begin(),all.end()).size() != all.size())
            throw std::invalid_argument("Duplicate graph coordinate.");
        for (int i=0; i<n; ++i) chart_vars.push_back("c"+std::to_string(i+1));
        g = atlas::local::parse_expression(all, input["graph"].asString());
        f = atlas::local::parse_expression(all, input["divisor"].asString());
        Powers axis(n); axis.back() = 1;
        if (g.find(axis)==g.end() || g.at(axis)!=1 || g.count(Powers(n)) || f.empty())
            throw std::invalid_argument("Expected a monic graph through the origin and a divisor.");
        Polynomial sub;
        for (const auto& [powers,c] : g) if (powers != axis) {
            if (powers.back()!=0) throw std::invalid_argument("Expected a graph linear in its auxiliary variable.");
            sub[Powers(powers.begin(),powers.end()-1)] = -c;
        }
        // Compare F|G to the input equation, without a local change of analytic coordinates.
        std::map<Z,Polynomial> cache{{0,{{Powers(n-1),1}}}};
        Polynomial base;
        for (const auto& [powers,c] : f) {
            Z k = powers.back();
            if (!cache.count(k)) {
                Polynomial power = sub, value=cache.at(0);
                for (Z j=k; j>0; j/=2) {
                    if (mpz_odd_p(j.get_mpz_t())) value=product(value,power);
                    if (j>1) power=product(power,power);
                }
                cache.emplace(k,std::move(value));
            }
            for (const auto& [p,b] : cache.at(k)) {
                Powers shifted=p;
                for (int i=0;i<n-1;++i) shifted[i]+=powers[i];
                add(base,shifted,c*b);
            }
        }
        if (base!=atlas::local::parse_expression(vars,input["polynomial"].asString()))
            throw std::invalid_argument("The graph divisor restricts to a different polynomial.");
    }
};

struct Fan {
    int n;
    PointSet rays;
    std::vector<Face> cones;
    Fan(const JsonValue& fan,int width) : n(width) {
        std::ostringstream fan_input;
        fan_input<<n<<" "<<fan["rays"].size()<<"\n";
        for (const auto& row : fan["rays"]) {
            Powers r=atlas::sparse_json::read_exponents(row,n);
            const size_t i=rays.size();
            if (i<static_cast<size_t>(n)) {
                Powers axis(n);axis[i]=1;
                if (r!=axis) throw std::invalid_argument("Expected coordinate axes first.");
            } else if (std::find(r.begin(),r.end(),0)!=r.end())
                throw std::invalid_argument("The modification can affect points away from the origin.");
            for (const auto& c:r) fan_input<<c<<" ";
            fan_input<<"\n"; rays.push_back(std::move(r));
        }
        for (const auto& row:fan["cones"]) {
            if (!row.isArray() || row.size()!=static_cast<unsigned>(n))
                throw std::invalid_argument("Wrong number of generators of a maximal cone.");
            Face ids;
            for (const auto& i:row) { ids.push_back(i.asInt()); fan_input<<ids.back()<<" "; }
            fan_input<<"\n";cones.push_back(std::move(ids));
        }
        std::istringstream in(fan_input.str());
        if (atlas::fan_cover(in,false).rfind("complete\n",0)!=0)
            throw std::invalid_argument("The fan does not regularly subdivide the full orthant.");
    }
};

struct Support {
    std::vector<std::pair<Powers,Q>> terms;
    PointSet weights;
    Powers minima;
    std::vector<Face> ray_faces;
    Support(const Polynomial& f,const Fan& fan) : terms(f.begin(),f.end()) {
        for (const Powers& ray:fan.rays) {
            Powers row;
            for (const auto& [e,c]:terms)
                row.push_back(std::inner_product(ray.begin(),ray.end(),e.begin(),Z(0)));
            minima.push_back(*std::min_element(row.begin(),row.end()));
            Face face;
            for (size_t j=0;j<terms.size();++j) if (row[j]==minima.back()) face.push_back(j);
            ray_faces.push_back(std::move(face));
            weights.push_back(std::move(row));
        }
    }
    Face initial(const Face& rays) const {
        // Each initial term must minimize every ray. Start from the
        // smallest singleton face to avoid scanning the entire polynomial.
        const int pivot = *std::min_element(rays.begin(), rays.end(), [&](int a, int b) {
            return ray_faces[a].size() < ray_faces[b].size();
        });
        Face subset;
        for (int j:ray_faces[pivot])
            if (std::all_of(rays.begin(),rays.end(),[&](int r){
                return r==pivot || std::binary_search(ray_faces[r].begin(),ray_faces[r].end(),j);
            })) subset.push_back(j);
        if (subset.empty()) throw std::invalid_argument("Cone crosses two initial vertices.");
        return subset;
    }
    Polynomial strict(const Face& cone) const {
        Polynomial result;
        for (size_t j=0;j<terms.size();++j) {
            Powers p;
            for (int r:cone) p.push_back(weights[r][j]-minima[r]);
            result[p]=terms[j].second;
        }
        return result;
    }
    PointSet projected(const Face& terms,const Face& directions) const {
        PointSet out;
        for (int j:terms) {
            Powers p;for (int i:directions) p.push_back(weights[i][j]-minima[i]);
            out.push_back(std::move(p));
        }
        return out;
    }
};

Polynomial log_derivative(const Polynomial& f,int i) {
    Polynomial out;for (const auto& [e,c]:f) if (e[i]!=0) out[e]=c*e[i];
    return out;
}

// The orbit charts trivialize a torus factor; testing the original face
// polynomials gives the same ranks by the smooth pullback from the orbit.
bool smooth_torus(const Polynomial& g,const Polynomial* f,int n) {
    if (g.size()==1 || (f && f->size()==1)) return true;
    std::vector<Polynomial> equations{g};
    if (f) equations.push_back(*f);
    std::vector<Polynomial> dg,df;
    for (int i=0;i<n;++i) {
        dg.push_back(log_derivative(g,i));
        if (f) df.push_back(log_derivative(*f,i)); else equations.push_back(dg.back());
    }
    if (f) for (int i=0;i<n;++i) for (int j=0;j<i;++j) {
        auto minor=product(dg[i],df[j]);
        for (const auto& [e,c]:product(dg[j],df[i])) add(minor,e,-c);
        if (!minor.empty()) equations.push_back(std::move(minor));
    }
    atlas::Ring ring;ring.init(n+1,0);
    std::vector<atlas::Poly> system;
    auto append=[&](const Polynomial& equation){
        if (equation.empty()) return;
        auto text=atlas::local::encode_row(equation)+"\n";const char* at=text.data();
        system.push_back(atlas::parse(ring,at,at+text.size(),nullptr));
    };
    for (const auto& equation:equations) {
        if (equation.size()==1) return true;
        // Primitive monomials are units on the torus. Strip their gcd before Buchberger.
        if (equation.empty()) continue;
        Powers low=equation.begin()->first;
        for (const auto& [e,c]:equation) for (int i=0;i<n;++i) low[i]=std::min(low[i],e[i]);
        Polynomial normalized;
        for (const auto& [e,c]:equation) {
            Powers p=e;for (int i=0;i<n;++i) p[i]-=low[i];
            p.push_back(0);normalized[p]=c;
        }
        append(normalized);
    }
    append({{Powers(n+1,1),1},{Powers(n+1),-1}});
    atlas::Engine engine;engine.R=ring;engine.compute(system,0);return engine.unit;
}

Z volume(const PointSet& points,int d) {
    if (points.size()<=static_cast<size_t>(d)) return 0;
    if (d==0) return 1;
    std::ostringstream out;out<<d<<"\n";
    for (const auto& p:points) {for (const auto& i:p) out<<i<<" ";out<<"\n";}
    auto text=out.str();
    char* answer=atlas_lattice_volume(text.data(),text.size());
    if (!answer) throw std::runtime_error("Failed to determine the chart's lattice volume.");
    Z result(answer);atlas_groebner_release(answer);return result;
}

std::string label(int r) {return "E"+std::to_string(r+1);}

struct Section {
    size_t count = 1;
    Powers character;
    Polynomial root_polynomial;
    std::string family;
};

// Every subface of an initial graph face above an exceptional cone occurs
// over the origin: perturb a strictly positive exposing weight slightly.
// The fan cover and orbit Jacobian tests in evaluate therefore check *full*
// Newton nondegeneracy, including the faces at infinity. In the saturated
// character lattice, toric Lefschetz (Danilov–Khovanskii, Prop. 3.9) gives
// connectedness if Newton dimension >= 2. Smoothness gives irreducibility.
// In dimension one each simple nonzero root in the primitive character is
// a component. Transversality on all deeper orbits rules out components
// entirely contained in the boundary of the dense orbit.
Section classify_section(const Support& g, const Face& terms) {
    Section section;
    if (terms.size() < 2) {
        section.count = 0;
        return section;
    }
    const auto& low = g.terms[terms.front()].first;
    const auto& high = g.terms[terms.back()].first;
    Powers direction;
    Z length = 0;
    for (size_t i = 0; i < low.size(); ++i) {
        direction.push_back(high[i] - low[i]);
        mpz_gcd(length.get_mpz_t(), length.get_mpz_t(), direction.back().get_mpz_t());
    }
    for (auto& e : direction) e /= length;
    const auto pivot = std::find_if(direction.begin(), direction.end(), [](const Z& e){ return e != 0; })
                     - direction.begin();
    Polynomial roots;
    for (int j : terms) {
        const auto& [p, coefficient] = g.terms[j];
        Z offset = p[pivot] - low[pivot];
        if (offset % direction[pivot] != 0) return section;
        offset /= direction[pivot];
        for (size_t i = 0; i < p.size(); ++i)
            if (p[i] - low[i] != offset * direction[i]) return section;
        roots[{offset}] = coefficient;
    }
    if (length > std::numeric_limits<Json::ArrayIndex>::max())
        throw std::length_error("The explicit geometric dual complex exceeds the JSON array size.");
    section.count = length.get_ui();
    section.character = std::move(direction);
    section.root_polynomial = std::move(roots);
    section.family = "edge" + std::to_string(terms.front()) + "e" + std::to_string(terms.back());
    return section;
}

struct GraphIncidence {
    const Support& g;
    std::map<Face, Section> sections;     // common Newton face -> geometric components
    std::map<Face, Face> intersections;   // exceptional cone -> common Newton face
    explicit GraphIncidence(const Support& support) : g(support) {}

    const Section& section(const Face& terms) {
        auto pos = sections.find(terms);
        if (pos == sections.end()) pos = sections.emplace(terms, classify_section(g, terms)).first;
        return pos->second;
    }
    void observe(const Face& orbit, const Face& terms) {
        section(terms);
        intersections.emplace(orbit, terms);
    }
    std::string id(const Face& cone, size_t index) const {
        if (cone.size() == 1) {
            const auto& section = sections.at(intersections.at(cone));
            return label(cone.front()) + (section.count > 1 ? "." + std::to_string(index) : "");
        }
        std::string name = "C";
        for (int r : cone) name += (name.size() > 1 ? "-" : "") + label(r);
        return name + "." + std::to_string(index);
    }
    std::string restriction(const Face& smaller, const Face& terms, size_t index) const {
        const Face& target = intersections.at(smaller);
        if (sections.at(target).count == 1) return id(smaller, 1);
        // A positive-dimensional subface of a segment is the same segment.
        // Equality of term indices compares its character *and* coefficients
        // in the common source lattice of every chart, so root k stays root k.
        if (terms != target) throw std::logic_error("Incompatible split Newton faces.");
        return id(smaller, index);
    }
    JsonValue complex(const Fan& fan, const Support& f) const {
        JsonValue out;
        out["status"] = "complete";
        out["convention"] = "Ambient exceptional divisor on the smooth graph over the origin over Qbar.";
        out["vertices"] = JsonValue(Json::arrayValue);
        out["cells"] = JsonValue(Json::arrayValue);
        for (const auto& [terms, section] : sections) if (section.count > 1) {
            auto& family = out["root_families"][section.family];
            family["direction"] = atlas::sparse_json::strings(section.character);
            family["polynomial"] = atlas::local::format_polynomial(section.root_polynomial, {"T"});
            family["degree"] = Json::UInt64(section.count);
        }
        for (size_t size = 1; size <= static_cast<size_t>(fan.n); ++size) {
            for (const auto& [cone, terms] : intersections) {
                if (cone.size() != size) continue;
                const auto& section = sections.at(terms);
                for (size_t i = 1; i <= section.count; ++i) {
                    JsonValue row;
                    row["id"] = id(cone, i);
                    if (section.count > 1) {
                        row["root"] = section.family + "_" + std::to_string(i);
                        row["root_family"] = section.family;
                        row["root_index"] = Json::UInt64(i);
                        if (size == 1) row["ray_divisor"] = label(cone.front());
                    }
                    if (size == 1) {
                        const int r = cone.front();
                        row["dimension"] = fan.n - 2;
                        row["ray_index"] = r;
                        row["total_multiplicity"] = integer(f.minima[r]);
                        row["nu"] = integer(std::accumulate(fan.rays[r].begin(),fan.rays[r].end(),Z(0))-g.minima[r]);
                        out["vertices"].append(std::move(row));
                    } else {
                        row["dimension"] = Json::UInt(size-1);
                        for (int r : cone) row["vertices"].append(restriction({r}, terms, i));
                        for (size_t j = 0; j < size; ++j) {
                            Face sub = cone;
                            sub.erase(sub.begin() + j);
                            row["faces"].append(restriction(sub, terms, i));
                        }
                        out["cells"].append(std::move(row));
                    }
                }
            }
        }
        return out;
    }
};

// On a chart where a ray's section splits, the coordinate hyperplane is
// the union of the sheets. Its individual sheet ideal also specifies the
// value of the same primitive torus character used by the incidence map.
void append_divisors(JsonValue& rows, const JsonValue& divisor, const Section& section,
                     const Face& cone, const Fan& fan, const Graph& graph) {
    if (section.count == 1) {
        rows.append(divisor);
        return;
    }
    Powers positive, negative;
    for (int r : cone) {
        const auto& ray = fan.rays[r];
        Z v = std::inner_product(ray.begin(), ray.end(), section.character.begin(), Z(0));
        positive.push_back(v > 0 ? v : Z(0));
        negative.push_back(v < 0 ? -v : Z(0));
    }
    const auto numerator = atlas::local::format_polynomial({{positive,1}}, graph.chart_vars);
    const auto denominator = atlas::local::format_polynomial({{negative,1}}, graph.chart_vars);
    for (size_t i = 1; i <= section.count; ++i) {
        JsonValue piece = divisor;
        piece["id"] = divisor["id"].asString() + "." + std::to_string(i);
        piece["ray_divisor"] = divisor["id"];
        auto root = section.family + "_" + std::to_string(i);
        auto equation = numerator + "-" + root + "*" + denominator;
        piece["ideal"].append(divisor["equation"]);
        piece["ideal"].append(equation);
        piece["equation"] = divisor["equation"].asString() + ", " + equation;
        rows.append(std::move(piece));
    }
}

JsonValue evaluate(const JsonValue& input) {
    Graph graph(input);const int n=graph.n;
    const auto operation=input["operation"].asString();
    if (operation=="support") {
        std::set<Powers> sum;
        for (const auto& [e,c]:graph.g) for (const auto& [h,b]:graph.f) {
            Powers p=e;for(int i=0;i<n;++i) p[i]+=h[i];sum.insert(std::move(p));
        }
        JsonValue result(Json::arrayValue);
        for (const auto& p:sum) result.append(atlas::sparse_json::strings(p));
        return result;
    }
    if (operation!="charts" && operation!="strata" && operation!="incidence")
        throw std::invalid_argument("Unknown graph operation.");
    Fan fan(input["fan"],n);Support g(graph.g,fan),f(graph.f,fan);
    bool validate=input.isMember("saved_charts"), charts=operation=="charts" || validate;
    if (validate && !input["saved_charts"].isArray())
        throw std::invalid_argument("The saved charts must be a list.");
    GraphIncidence incidence(g);
    bool need_incidence=operation!="strata" || validate;
    size_t chart_number=0;
    JsonValue out(Json::objectValue),chart_rows(Json::arrayValue);
    std::set<Face> seen, graph_checked;
    using Pair=std::pair<Face,Face>;
    std::set<Pair> pair_checked;
    std::map<std::pair<Pair,int>,std::pair<Z,Z>> characteristics;
    std::map<Face,Z> contributions; std::set<int> used;
    for (size_t at=0;at<fan.cones.size();++at) {
        const Face& cone=fan.cones[at];
        f.initial(cone);g.initial(cone);
        bool meets_fibre=false;
        for (int mask=1;mask<(1<<n);++mask) {
            Face orbit,remaining;
            for (int i=0;i<n;++i) ((mask>>i)&1?orbit:remaining).push_back(cone[i]);
            std::sort(orbit.begin(),orbit.end());
            if (orbit.back()<n) continue;
            Face a=g.initial(orbit);
            if (a.size()==1) continue;
            meets_fibre=true;
            if (!seen.insert(orbit).second) continue;
            Face b=f.initial(orbit);
            auto get=[](const Support& s,const Face& part){
                Polynomial out;for (int id:part) out.insert(s.terms[id]);return out;
            };
            if (graph_checked.insert(a).second && !smooth_torus(get(g,a),nullptr,n))
                throw std::invalid_argument("Singular graph face on an orbit over the origin.");
            if (b.size()>1 && pair_checked.insert({a,b}).second) {
                auto second=get(f,b);
                if (!smooth_torus(get(g,a),&second,n)) throw std::invalid_argument("The divisor fails the orbit Jacobian test.");
            }
            if (need_incidence && orbit.front()>=n) incidence.observe(orbit,a);
            if (operation!="strata") continue;
            const int d=remaining.size();
            auto [it,fresh]=characteristics.try_emplace({{a,b},d});
            if (fresh) {
                PointSet p=g.projected(a,remaining),q=f.projected(b,remaining);
                Z vp=volume(p,d),vq=volume(q,d), mixed=0;
                if (d>=2) {
                    PointSet cayley;
                    for (Powers e:p) {e.push_back(0);cayley.push_back(std::move(e));}
                    for (Powers e:q) {e.push_back(1);cayley.push_back(std::move(e));}
                    mixed=volume(cayley,d+1)-vp-vq;
                }
                it->second={(d%2==0?-vp:vp),(d%2==0?mixed:-mixed)};
            }
            auto [ambient,intersection]=it->second;
            Face exceptional; for(int i:orbit) if(i>=n) {exceptional.push_back(i);used.insert(i);}
            contributions[exceptional]+=ambient-intersection;
            exceptional.push_back(-1);contributions[exceptional]+=intersection;
        }
        if (charts && meets_fibre) {
            JsonValue c(Json::objectValue);c["id"]="N"+std::to_string(at+1);
            for (const auto& var:graph.chart_vars) c["coordinates"].append(var);
            for (int i:cone) c["ray_indices"].append(i);
            for (int j=0;j<n-1;++j) {
                Powers e;for(int i:cone) e.push_back(fan.rays[i][j]);
                c["map"][graph.vars[j]]=atlas::local::format_polynomial({{e,1}},graph.chart_vars);
            }
            c["ambient_ideal"].append(atlas::local::format_polynomial(g.strict(cone),graph.chart_vars));
            c["strict_transform"]=atlas::local::format_polynomial(f.strict(cone),graph.chart_vars);
            c["exceptional"]=JsonValue(Json::arrayValue);
            JsonValue unsplit(Json::arrayValue);
            for (int i=0;i<n;++i) if (cone[i]>=n && g.initial({cone[i]}).size()>1) {
                int r=cone[i];JsonValue e;e["id"]=label(r);e["equation"]=graph.chart_vars[i];
                e["total_multiplicity"]=integer(f.minima[r]);
                e["nu"]=integer(std::accumulate(fan.rays[r].begin(),fan.rays[r].end(),Z(0))-g.minima[r]);
                if (validate) unsplit.append(e);
                append_divisors(c["exceptional"],e,incidence.section(g.initial({r})),cone,fan,graph);
            }
            if (validate) {
                const auto& saved=input["saved_charts"][Json::ArrayIndex(chart_number)];
                if (c!=saved) {
                    // Older graph records store the single parent equation
                    // for every rational ray divisor, even when it splits.
                    auto legacy=c;
                    legacy["exceptional"]=std::move(unsplit);
                    if (legacy!=saved)
                        throw std::invalid_argument("The saved charts disagree with the verified toric graph modification.");
                }
                ++chart_number;
            }
            if (operation=="charts") chart_rows.append(std::move(c));
        }
    }
    if (validate && chart_number!=input["saved_charts"].size())
        throw std::invalid_argument("The saved charts disagree with the verified toric graph modification.");
    if (operation=="incidence") return incidence.complex(fan,f);
    out["orbits_over_origin_meeting_graph"]=Json::UInt64(seen.size());
    out["graph_face_tests"]=Json::UInt64(graph_checked.size());
    out["complete_intersection_face_tests"]=Json::UInt64(pair_checked.size());
    if (operation=="charts") {
        out["charts"]=std::move(chart_rows);
        out["dual_complex"]=incidence.complex(fan,f);
    }
    else {
        for (int i:used) {
            JsonValue row;row["component"]=label(i);row["multiplicity"]=integer(f.minima[i]);
            row["nu"]=integer(std::accumulate(fan.rays[i].begin(),fan.rays[i].end(),Z(0))-g.minima[i]);
            out["numerical_data"].append(row);
        }
        JsonValue strict;strict["component"]="strict";strict["multiplicity"]=1;strict["nu"]=1;out["numerical_data"].append(strict);
        for (const auto& [ids,chi]:contributions) if (chi!=0) {
            JsonValue row;for(int id:ids)row["components"].append(id<0?"strict":label(id));
            row["euler_characteristic"]=integer(chi);out["strata"].append(row);
        }
    }
    return out;
}
}

extern "C" const char* atlas_toric_graph(const char* input) noexcept {
    return atlas::sparse_json::invoke(input,evaluate);
}
