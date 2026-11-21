// Reconstruct and check standard opens of a projective ideal blowup over QQ.
// A marked number field is presented by its defining equation in the last
// coordinate. Free base coordinates may be suppressed when computing orders.
#include "local_io.hpp"
#include "local_point_quotient.hpp"
#include "primary_decomposition.hpp"
#include "polynomial_factors.hpp"
#include "rational_sparse_json.hpp"
#include "sparse_flint.hpp"
#include <algorithm>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>

extern "C" char* atlas_affine(const char*, size_t) noexcept;
extern "C" char* atlas_ideal(const char*, size_t) noexcept;
extern "C" const char* atlas_groebner_error() noexcept;
extern "C" void atlas_groebner_release(char*) noexcept;

namespace atlas::graph_replay {
using namespace local;
using J = Json::Value;
using Names = std::vector<std::string>;

void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
Polynomial one(size_t width) { return {{Powers(width), 1}}; }
Polynomial axis(size_t width, size_t i) {
    Powers e(width);
    e.at(i) = 1;
    return {{std::move(e), 1}};
}
Basis joined(Basis a, const Basis& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}
bool unit(const Basis& basis) {
    return basis.size() == 1 && !basis[0].empty() && basis[0] == one(basis[0].begin()->first.size());
}
Names names(const J& j) {
    require(j.isArray(), "Expected an array of coordinate or equation strings.");
    Names values;
    for (const auto& x : j) {
        require(x.isString(), "Expected a coordinate or equation string.");
        values.push_back(x.asString());
    }
    return values;
}

// A packed operation declines only for its representation limits. Invalid
// geometry and other kernel errors must propagate with their original reason.
std::optional<std::string> packed(const std::string& input) {
    std::unique_ptr<char, decltype(&atlas_groebner_release)> answer(
        atlas_ideal(input.data(),input.size()),atlas_groebner_release);
    if (answer) return std::string(answer.get());
    const std::string error=atlas_groebner_error();
    if (error=="Unsupported number of variables." || error.find("out of range")!=std::string::npos) return {};
    throw std::invalid_argument(error);
}

struct ChartRing {
    Names variables;
    size_t n;
    std::map<std::string, Polynomial> decoded;
    explicit ChartRing(const J& coords) : variables(names(coords)), n(variables.size()) {
        require(n && std::set<std::string>(variables.begin(), variables.end()).size() == n,
                "Invalid graph coordinates.");
    }
    const Polynomial& read(const J& raw) {
        require(raw.isString(), "Expected a chart equation.");
        const auto text = raw.asString();
        auto it = decoded.find(text);
        if (it == decoded.end()) it = decoded.emplace(text, parse_expression(variables, text)).first;
        return it->second;
    }
    Basis rows(const J& input) {
        require(input.isArray(), "Expected an ideal of chart equations.");
        Basis r;
        for (const auto& g : input) r.push_back(read(g));
        return r;
    }
    J texts(const Basis& b, bool sort=false) const {
        Names strings;
        for (const auto& g : b) strings.push_back(format_polynomial(g, variables));
        if (sort) std::sort(strings.begin(), strings.end());
        J result(Json::arrayValue);
        for (const auto& text : strings) result.append(text);
        return result;
    }
    Basis reduced(const Basis& rows) const { return primary::reduced(rows,n,true); }
    Basis saturated(const Basis& rows, const Polynomial& pivot) const {
        if (pivot.empty()) return {one(n)};
        if (pivot.size()==1 && pivot.begin()->first == Powers(n)) return reduced(rows);
        return primary::saturate_lex(rows,pivot,n);
    }
    bool equal(const Basis& a, const Basis& b) const {
        return a == b || reduced(a) == reduced(b);
    }
    bool radical_member(const Polynomial& f, const Basis& equations) const {
        return f.empty() || unit(saturated(equations,f));
    }
    // On the closed origin fibre V(pivot) must be exactly the union of the
    // listed exceptional components. A complement is open where at least
    // one generator of each component is invertible.
    bool cover(const Basis& closed, const Polynomial& pivot, const std::vector<Basis>& pieces) const {
        std::ostringstream wire;
        wire << "graph_cover " << n << ' ' << closed.size() << ' ' << pieces.size();
        for (const auto& p : pieces) wire << ' ' << p.size();
        wire << '\n';
        for (const auto& g : closed) wire << encode_row(g) << '\n';
        wire << encode_row(pivot) << '\n';
        for (const auto& p : pieces) for (const auto& g : p) wire << encode_row(g) << '\n';
        if (auto result=packed(wire.str())) {
            require(*result=="yes\n" || *result=="no\n", "Incomplete graph cover proof.");
            return *result=="yes\n";
        }
        for (const auto& p : pieces)
            if (!radical_member(pivot,joined(closed,p))) return false;
        auto target = joined(closed,{pivot});
        auto visit = [&](auto&& self, size_t i, const Polynomial& product) -> bool {
            if (i==pieces.size()) return radical_member(product,target);
            for (const auto& g : pieces[i])
                if (!g.empty() && !self(self,i+1,primary::multiply(product,g))) return false;
            return true;
        };
        return visit(visit,0,one(n));
    }
};

// Compact the completely unused projection coordinates. The omitted factors
// are affine space with identity projection, which changes neither order.
std::optional<std::pair<mpz_class,mpz_class>> packed_orders(
    size_t n, size_t base, const Basis& ambient, const Basis& prime, const Polynomial& total) {
    std::set<size_t> active;
    auto select = [&](const Polynomial& p) {
        for (const auto& [e,c] : p) for (size_t i=0; i<base; ++i) if (e[i]!=0) active.insert(i);
    };
    for (const auto& p : ambient) select(p);
    for (const auto& p : prime) select(p);
    select(total);
    if (active.empty()) active.insert(0);
    const size_t dimension = active.size();
    for (size_t i=base; i<n; ++i) active.insert(i);
    std::ostringstream wire;
    wire << "graph_orders " << active.size() << ' ' << dimension << ' ' << ambient.size() << ' ' << prime.size() << '\n';
    auto put = [&](const Polynomial& p) {
        bool first=true;
        for (const auto& [e,c] : p) {
            if (!first) wire << ';';
            first=false; wire << c.get_str();
            for (auto j : active) wire << ' ' << e[j];
        }
        wire << '\n';
    };
    for (const auto& p : ambient) put(p);
    for (const auto& p : prime) put(p);
    put(total);
    const auto answer=packed(wire.str());
    if (!answer) return {};
    std::istringstream values(*answer);
    mpz_class order,nu;
    if (!(values>>order>>nu)) throw std::runtime_error("Incomplete graph divisor orders.");
    return std::pair<mpz_class,mpz_class>{order,nu};
}

Basis relative_minors(const ChartRing& chart, size_t base, const Basis& ambient, const Basis& gb) {
    if (base==chart.n) return {one(chart.n)};
    const size_t extra=chart.n-base;
    require(ambient.size()>=extra,"Missing graph equations.");
    // Restrict the determinant to the relative columns. Call the shared FLINT
    // Jacobian minor evaluator for each selection of rows.
    std::vector<size_t> selected(extra);
    std::iota(selected.begin(),selected.end(),0);
    Basis result;
    do {
        std::ostringstream wire;
        wire << "minor\n";
        for (const auto& v : chart.variables) wire << v << ' ';
        wire << '\n';
        for (size_t j=base;j<chart.n;++j) wire << j << ' ';
        wire << '\n';
        for (auto i : selected) wire << format_polynomial(ambient[i],chart.variables) << '\n';
        const auto input=wire.str();
        std::unique_ptr<char,decltype(&atlas_groebner_release)> out(
            atlas_affine(input.data(),input.size()),atlas_groebner_release);
        if (!out) throw std::invalid_argument(atlas_groebner_error());
        std::string line(out.get());
        if (!line.empty() && line.back()=='\n') line.pop_back();
        auto minor=decode_row(line,chart.n);
        if (!primary::member_lex(minor,gb)) result.push_back(std::move(minor));
        size_t k=extra;
        while (k && selected[k-1]==ambient.size()-extra+k-1) --k;
        if (!k) break;
        ++selected[k-1];
        for (;k<extra;++k) selected[k]=selected[k-1]+1;
    } while (true);
    require(!result.empty(),"The graph projection is not generically etale.");
    return result;
}

std::pair<mpz_class,mpz_class> orders(const ChartRing& chart, size_t base, const Basis& ambient,
                                      const Basis& prime, const Polynomial& total) {
    if (auto packed=packed_orders(chart.n,base,ambient,prime,total)) return *packed;
    const auto gb=chart.reduced(ambient);
    require(!primary::member_lex(total,gb),"Zero pullback on the chart.");
    const auto divisor=chart.reduced(joined(ambient,prime));
    require(!unit(divisor),"Empty divisor.");
    Basis generators;
    for (const auto& g:prime) if (!primary::member_lex(g,gb)) generators.push_back(g);
    require(!generators.empty(),"Missing divisor ideal.");
    const auto minors=relative_minors(chart,base,ambient,gb);
    auto valuation = [&](const Basis& values) -> mpz_class {
        // Height-one Cartier ideals on these smooth charts have ordinary
        // powers equal to symbolic powers. For a principal ideal, binary
        // search handles even exponents beyond machine precision.
        if (generators.size()==1) {
            auto present = [&](const mpz_class& k) {
                auto test=chart.reduced(joined(ambient,{primary::power(generators[0],k,chart.n)}));
                return std::all_of(values.begin(),values.end(),[&](const auto& f){return primary::member_lex(f,test);});
            };
            mpz_class lo=0,hi=1;
            while (present(hi)) {lo=hi; hi*=2;}
            while (lo+1<hi) {mpz_class mid=(lo+hi)/2; if (present(mid)) lo=mid; else hi=mid;}
            return lo;
        }
        Basis power=divisor;
        for (mpz_class k=0;;++k) {
            if (std::any_of(values.begin(),values.end(),[&](const auto& f){return !primary::member_lex(f,power);})) return k;
            Basis next=ambient;
            for (const auto& f:power) if (!primary::member_lex(f,gb))
                for (const auto& g:generators) next.push_back(primary::multiply(f,g));
            power=chart.reduced(next);
        }
    };
    return {valuation({total}),valuation(minors)+1};
}

J open(const J& r, bool omitted) {
    ChartRing chart(r["coords"]);
    const auto base=r["base"].asUInt();
    require(base && base<=chart.n,"Invalid graph projection dimension.");
    const auto centre=chart.rows(r["centre"]), projective=chart.rows(r["projective"]);
    require(centre.size()==projective.size() && !centre.empty(),"Wrong number of projective coordinates.");
    auto pivot=std::find(projective.begin(),projective.end(),one(chart.n));
    require(pivot!=projective.end() && std::count(projective.begin(),projective.end(),*pivot)==1,
            "Expected one unit in the projective graph.");
    const size_t position=pivot-projective.begin();
    bool marked=!r["relation"].isNull();
    require(chart.n==base+centre.size()-1+marked,"Wrong graph chart dimension.");
    Basis axes;
    for (size_t i=0;i<base;++i) axes.push_back(axis(chart.n,i));
    Basis ratios;
    for (const auto& g:projective) if (g!=*pivot) ratios.push_back(g);
    Basis expected;
    for (size_t i=base;i<chart.n-marked;++i) expected.push_back(axis(chart.n,i));
    std::sort(ratios.begin(),ratios.end()); std::sort(expected.begin(),expected.end());
    require(ratios==expected,"Invalid graph ratio coordinates.");
    // The defining centre is an ideal of the base. Even a sealed archive
    // must not replace a centre generator by a ratio-dependent function.
    for (const auto& g:centre) for (const auto& term:g)
        for (size_t i=base;i<chart.n-marked;++i)
            require(term.first[i]==0, "The centre generator uses a projective ratio.");
    const auto& by=centre[position];
    Basis graph;
    for (size_t j=0;j<centre.size();++j) if (j!=position) {
        auto relation=centre[j];
        for (const auto& [e,c]:primary::multiply(by,projective[j])) {
            auto at=relation.try_emplace(e,0).first;
            if ((at->second-=c)==0) relation.erase(at);
        }
        graph.push_back(std::move(relation));
    }
    if (marked) graph.push_back(chart.read(r["relation"]));
    const auto reconstructed=chart.saturated(graph,by);
    if (omitted) {
        require(unit(chart.reduced(joined(reconstructed,axes))),
                "The graph omits an affine open meeting the origin fibre.");
        return {};
    }
    require(chart.rows(r["projection"])==axes,"The projective graph does not project to its base coordinates.");
    require(chart.read(r["pivot"])==by,"The pivot differs from its centre generator.");
    auto ambient=chart.rows(r["ambient"]);
    if (marked) ambient.push_back(chart.read(r["relation"]));
    require(chart.equal(ambient,reconstructed),"The ambient ideal differs from the saturated projective graph.");
    const auto closed=joined(ambient,axes);
    const auto& f=chart.read(r["equation"]);
    std::vector<Basis> pieces;
    J out;
    out["ambient"]=chart.texts(ambient);
    out["position"]=Json::UInt(position);
    out["region"]=chart.texts(Basis(projective.begin(),pivot));
    out["exceptional"]=J(Json::arrayValue);
    out["strict"]=J(Json::arrayValue);
    for (const auto& p:r["exceptional"]) {
        auto piece=chart.rows(p);
        auto [n,nu]=orders(chart,base,ambient,piece,f);
        J item;
        item["orders"].append(n.get_str()); item["orders"].append(nu.get_str());
        item["ideal"]=chart.texts(piece);
        out["exceptional"].append(std::move(item));
        pieces.push_back(std::move(piece));
    }
    require(chart.cover(closed,by,pieces),"The exceptional prime ideals do not cover the pivot over the origin.");
    auto strict = [&](const Polynomial& g) {return chart.saturated(joined(ambient,{g}),by);};
    const auto factors = chart.rows(r["factors"]);
    Basis total;
    if (factors.empty()) {
        Polynomial resolved=f;
        if (r.isMember("base_unit")) {
            const auto multiplier=chart.read(r["base_unit"]);
            resolved=chart.read(r["resolved_equation"]);
            for (const auto& term:multiplier)
                require(std::all_of(term.first.begin()+base,term.first.end(),[](const auto& e){return e==0;}),
                        "A removed unit must use the original base coordinates.");
            require(multiplier.count(Powers(chart.n)) && primary::multiply(resolved,multiplier)==f,
                    "The removed factor must be a unit at the origin and recover the original polynomial.");
        }
        total=strict(resolved);
        out["strict"].append(unit(total)?J():chart.texts(total,true));
    } else {
        Polynomial support=one(chart.n);
        for (const auto& factor:factors) {
            auto g=strict(factor);
            out["strict"].append(unit(g)?J():chart.texts(g,true));
            support=primary::multiply(support,factor);
        }
        if (r.isMember("reduced_strict"))
            require(chart.equal(joined(ambient,chart.rows(r["reduced_strict"])),strict(support)),
                    "The displayed reduced strict transform differs from the saturated pullback.");
        total=strict(f);
    }
    require(chart.equal(joined(ambient,chart.rows(r["strict"])),total),
            "The displayed strict transform differs from the saturated pullback.");
    return out;
}

using Subset = std::vector<unsigned>;

J indices(const Subset& key) {
    J values(Json::arrayValue);
    for (auto j:key) values.append(j);
    return values;
}
Subset subset(const J& key) {
    Subset values;
    for (const auto& j:key) values.push_back(j.asUInt());
    return values;
}

J strata_tasks(ChartRing& ring, const J& r) {
    const unsigned n=r["dimension"].asUInt(), boundaries=r["count"].asUInt();
    const bool intrinsic=r["intrinsic"].asBool();
    require(n>0 && r["charts"].isArray(),"Invalid graph stratum data.");
    const unsigned dimension=n-intrinsic;
    const auto relation=ring.rows(r["relation"]);
    J answer(Json::arrayValue);
    for (unsigned k=1;k<=std::min(dimension,boundaries);++k) {
        Subset key(k);
        std::iota(key.begin(),key.end(),0u);
        while (true) {
            J task;
            task["subset"]=indices(key);
            task["dimension"]=dimension-k;
            task["charts"]=J(Json::arrayValue);
            for (unsigned ci=0;ci<r["charts"].size();++ci) {
                const auto& c=r["charts"][ci];
                require(c["pieces"].size()==boundaries,"Inconsistent number of chart boundary ideals.");
                auto eq=ring.rows(c[intrinsic?"strict":"ambient"]);
                for (auto j:key) eq=joined(std::move(eq),ring.rows(c["pieces"][j]));
                // A chart missing the owned fibre can still connect two
                // visible localizations. Skip only geometrically empty strata.
                if (!unit(ring.reduced(joined(eq,relation)))) task["charts"].append(ci);
            }
            answer.append(std::move(task));
            size_t j=k;
            while (j && key[j-1]==boundaries-k+j-1) --j;
            if (!j) break;
            ++key[j-1];
            for (;j<k;++j) key[j]=key[j-1]+1;
        }
    }
    return answer;
}

struct VisibleComponent {
    std::string name;
    std::map<unsigned,Basis> charts;
};

J stratum_faces(ChartRing& ring, const J& r) {
    const auto relation=ring.rows(r["relation"]);
    const auto fibre=joined(relation,ring.rows(r["fibre"]));
    const unsigned chart_count=r["charts"].asUInt();
    std::map<Subset,std::vector<VisibleComponent>> families;
    J result;
    result["vertices"]=J(Json::arrayValue);
    result["cells"]=J(Json::arrayValue);
    result["divisors"]=J(Json::arrayValue);
    for (unsigned ci=0;ci<chart_count;++ci) result["divisors"].append(J(Json::arrayValue));
    for (const auto& stratum:r["strata"]) {
        const Subset key=subset(stratum["subset"]);
        require(!key.empty(),"Unexpected empty boundary subset.");
        auto& family=families[key];
        for (const auto& group:stratum["groups"]) {
            if (group.empty()) continue;
            VisibleComponent component;
            bool visible=false;
            unsigned first=group[0]["chart"].asUInt();
            for (const auto& p:group) {
                const unsigned ci=p["chart"].asUInt();
                require(ci<chart_count,"Unknown graph chart.");
                auto equations=ring.rows(p["ideal"]);
                if (!visible && !unit(ring.reduced(joined(fibre,equations)))) visible=true;
                component.charts[ci]=std::move(equations);
            }
            if (!visible) continue;
            if (key.size()==1) {
                component.name="E"+std::to_string(result["vertices"].size()+1);
                J vertex;
                vertex["id"]=component.name; vertex["dimension"]=stratum["dimension"];
                result["vertices"].append(std::move(vertex));
                for (const auto& p:group) {
                    J label;
                    label["id"]=component.name; label["ideal"]=p["display"];
                    result["divisors"][p["chart"].asUInt()].append(std::move(label));
                }
            } else {
                const auto child=ring.reduced(joined(relation, component.charts.at(first)));
                auto parent = [&](const Subset& face) {
                    const auto found=families.find(face);
                    require(found!=families.end(),"Missing parent stratum.");
                    std::string name;
                    for (const auto& candidate:found->second) {
                        const auto open=candidate.charts.find(first);
                        if (open==candidate.charts.end()) continue;
                        if (std::all_of(open->second.begin(),open->second.end(),
                                        [&](const auto& f){return primary::member_lex(f,child);})) {
                            if (!name.empty()) throw std::invalid_argument("Stratum incidence is ambiguous.");
                            name=candidate.name;
                        }
                    }
                    require(!name.empty(),"Stratum incidence is ambiguous.");
                    return name;
                };
                component.name="C"+std::to_string(result["cells"].size()+1);
                J cell;
                cell["id"]=component.name; cell["dimension"]=Json::UInt(key.size()-1);
                for (auto j:key) cell["vertices"].append(parent(Subset{j}));
                // Combinations of size k-1 in original lexicographic order.
                for (size_t i=key.size();i-- > 0;) {
                    auto face=key; face.erase(face.begin()+i);
                    cell["faces"].append(parent(face));
                }
                result["cells"].append(std::move(cell));
            }
            family.push_back(std::move(component));
        }
    }
    return result;
}

J execute(const J& r) {
    const auto op=r["operation"].asString();
    if (op=="open" || op=="omitted") return open(r,op=="omitted");
    ChartRing chart(r["coords"]);
    if (op=="verify_centre") {
        auto centre=chart.rows(r["centre"]);
        if (!r["relation"].isNull()) centre.push_back(chart.read(r["relation"]));
        const auto ideal=chart.reduced(centre);
        if (unit(ideal)) return true;
        SparseLexRing ring(chart.n);
        PolyValue equation(ring.ctx), support(ring.ctx);
        ring.write(equation.p, chart.read(r["equation"]));
        rational_squarefree_part(support.p,equation.p,ring.ctx);
        const auto reduced=ring.read(support.p);
        auto singular=partial_derivatives(reduced,chart.n);
        singular.push_back(reduced);
        for (const auto& f:singular)
            require(primary::member_lex(f,ideal) || chart.radical_member(f,ideal),
                    "The graph centre meets the smooth locus or lies outside the reduced hypersurface.");
        return true;
    }
    if (op=="strata_tasks") return strata_tasks(chart,r);
    if (op=="stratum_faces") return stratum_faces(chart,r);
    if (op=="intersection") {
        const auto relation=chart.rows(r["relation"]);
        const auto a=chart.reduced(joined(relation,chart.rows(r["left"])));
        const auto b=chart.reduced(joined(relation,chart.rows(r["right"])));
        J out(Json::arrayValue);
        for (const auto& f:primary::intersection_lex(a,b,chart.n))
            if (std::find(relation.begin(),relation.end(),f)==relation.end()) out.append(encode_row(f));
        return out;
    }
    if (op=="basis") return chart.texts(chart.saturated(chart.rows(r["ideal"]),chart.read(r["pivot"])),true);
    if (op=="cover") {
        std::vector<Basis> pieces;
        for (const auto& p:r["pieces"]) pieces.push_back(chart.rows(p));
        return chart.cover(chart.rows(r["ambient"]),chart.read(r["pivot"]),pieces);
    }
    if (op=="equal_ideals") {
        for (const auto& pair:r["pairs"]) {
            require(pair.isArray() && pair.size()==2,"Expected two ideals for comparison.");
            if (!chart.equal(chart.rows(pair[0]),chart.rows(pair[1]))) return false;
        }
        return true;
    }
    if (op=="component_coverage" || op=="disjoint_components") {
        const Basis relation=chart.rows(r["relation"]);
        std::vector<Basis> components;
        for (const auto& p:r["pieces"])
            components.push_back(chart.reduced(joined(relation,chart.rows(p))));
        if (op=="disjoint_components" || !r["reduced"].asBool()) {
            for (size_t i=0;i<components.size();++i) for (size_t j=0;j<i;++j)
                if (!unit(chart.reduced(joined(components[i],components[j])))) return false;
        }
        if (op=="disjoint_components") return true;
        if (components.empty()) return false;
        const Basis original=chart.reduced(joined(relation,chart.rows(r["ambient"])));
        for (const auto& component:components) for (const auto& f:original)
            if (!primary::member_lex(f,component)) return false;
        auto intersection=components.front();
        for (size_t j=1;j<components.size();++j)
            intersection=primary::intersection_lex(intersection,components[j],chart.n);
        for (const auto& f:intersection)
            if (!primary::member_lex(f,original) &&
                (!r["reduced"].asBool() || !chart.radical_member(f,original))) return false;
        return true;
    }
    if (op=="component_overlap") {
        // Glue two standard opens by proportionality of their homogeneous
        // ratio vectors. The left pivot is 1 and the right pivot is 1, so
        // these equations also force the transition denominator to be a unit.
        const auto a=chart.rows(r["left_ratios"]), b=chart.rows(r["right_ratios"]);
        const size_t pivot=r["right_pivot"].asUInt();
        require(a.size()==b.size() && pivot<a.size() && b[pivot]==one(chart.n),
                "Invalid projective overlap coordinates.");
        auto equations=chart.rows(r["pieces"]);
        for (size_t j=0; j<a.size(); ++j) if (j!=pivot) {
            auto f=a[j];
            for (const auto& [e,c]:primary::multiply(a[pivot],b[j])) {
                auto it=f.try_emplace(e,0).first;
                if ((it->second-=c)==0) f.erase(it);
            }
            equations.push_back(std::move(f));
        }
        return !unit(chart.reduced(equations));
    }
    if (op=="orders") {
        const size_t base=r["base"].asUInt();
        require(base && base<=chart.n,"Invalid graph projection dimension.");
        const auto [n,nu]=orders(chart,base,chart.rows(r["ambient"]),chart.rows(r["prime"]),chart.read(r["equation"]));
        J result(Json::arrayValue);
        result.append(n.get_str()); result.append(nu.get_str()); return result;
    }
    throw std::invalid_argument("Unknown graph replay operation.");
}
}
extern "C" const char* atlas_graph_replay(const char* input) {
    return atlas::sparse_json::invoke(input,atlas::graph_replay::execute);
}
