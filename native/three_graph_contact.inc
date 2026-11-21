// Polynomial sections and quadratic translates over a reduced plane contact divisor.
// Uses the very same plane charts, blowups and exact algebraic centres as Resolver.
namespace {
using GraphParts = vector<Poly>;
using ContactOrders = vector<Exp>;

struct GraphFactorization {
    fmpq_mpoly_factor_t factors;
    const fmpq_mpoly_ctx_struct* ctx;
    explicit GraphFactorization(const fmpq_mpoly_ctx_struct* c): ctx(c) { fmpq_mpoly_factor_init(factors,ctx); }
    ~GraphFactorization() { fmpq_mpoly_factor_clear(factors,ctx); }
};

struct ContactChart {
    GraphParts strict;
    Axes axes{};
    int pivot = -1;
};

Num contact_scalar(const string& value, const Resolver& solver) {
    if (!value.empty() && value.front()=='a') return solver.algebraics.at(stoul(value.substr(1))-1);
    fmpq_t q;
    fmpq_init(q);
    if (fmpq_set_str(q,value.c_str(),10) || fmpz_is_zero(fmpq_denref(q))) {
        fmpq_clear(q);
        throw Fail("Invalid plane chart centre.");
    }
    fmpq_canonicalise(q);
    Num n;
    qqbar_set_fmpq(n.n,q);
    fmpq_clear(q);
    return n;
}

// The centre on a noninitial chart is on its new exceptional coordinate axis.
// Return strict transforms at the centre and discard axes missing it.
ContactChart at_contact_point(const ContactChart& chart, const J& point, const Resolver& solver) {
    ContactChart result=chart;
    if (chart.pivot < 0) return result;
    const int axis=1-chart.pivot;
    Num a=contact_scalar(point[axis].asString(),solver);
    if (!a.zero()) {
        result.axes[axis]=0;
        for (auto& g:result.strict) g=shift(g,axis,a);
    }
    return result;
}

J trace_contact(const GraphParts& differences, const J& resolution, const Resolver& solver) {
    map<string,ContactChart> charts;
    charts.emplace("origin",ContactChart{differences});
    const size_t pairs=differences.size();
    vector<ContactOrders> valuations(1,ContactOrders(pairs));
    vector<Exp> discrepancies{1};
    vector<Exp> open_chi{0};
    J result;
    result["divisors"]=arr(); result["edges"]=arr(); result["branches"]=arr();
    for (const auto& step:resolution["steps"]) {
        auto parent=at_contact_point(charts.at(step["parent_chart"].asString()),step["center"],solver);
        const int i=static_cast<int>(valuations.size());
        const int u=parent.axes[0], v=parent.axes[1];
        const Exp nu=add_powers(discrepancies[u],discrepancies[v]);
        ContactOrders m(pairs), orders(pairs);
        for (size_t j=0;j<pairs;++j) {
            m[j]=order(parent.strict[j]);
            orders[j]=add_powers(m[j],add_powers(valuations[u][j],valuations[v][j]));
        }
        discrepancies.push_back(nu);
        valuations.push_back(orders);
        open_chi.push_back(2);
        for (int pivot=0;pivot<2;++pivot) {
            ContactChart child;
            child.pivot=pivot;
            child.axes=parent.axes;
            child.axes[pivot]=i;
            for (size_t j=0;j<pairs;++j) child.strict.push_back(blow(parent.strict[j],pivot,m[j]));
            charts.emplace(step["id"].asString()+(pivot?".v":".u"),move(child));
        }
    }
    for (auto [a,b]:solver.edges) {
        --open_chi[a]; --open_chi[b];
        result["edges"].append(arr({Resolver::E(a),Resolver::E(b)}));
    }
    for (const auto& branch:resolution["branches"]) {
        auto point=at_contact_point(charts.at(branch["chart"].asString()),branch["point"],solver);
        int index=point.axes[point.pivot];
        int pair=-1;
        for (size_t j=0;j<pairs;++j) {
            if (point.strict[j].count({0,0})) continue;
            if (pair!=-1 || order(point.strict[j])!=1) throw Fail("Contact divisor is not separated.");
            pair=j;
        }
        if (pair==-1 || point.axes[1-point.pivot]!=0) throw Fail("Missing transverse contact branch.");
        --open_chi[index];
        J item; item["divisor"]=Resolver::E(index); item["pair"]=pair;
        result["branches"].append(item);
    }
    for (size_t i=1;i<valuations.size();++i) {
        auto sorted=valuations[i];
        sort(sorted.begin(),sorted.end());
        if (!sorted[0] || sorted[0]!=sorted[1]) throw Fail("Inconsistent graph contact orders.");
        J row;
        row["id"]=Resolver::E(i);
        row["nu"]=Json::Int64(discrepancies[i]); row["chi"]=Json::Int64(open_chi[i]);
        if (pairs==3) {
            row["d"]=Json::Int64(sorted[0]); row["e"]=Json::Int64(sorted[2]);
        }
        row["orders"]=arr();
        for (auto n:valuations[i]) row["orders"].append(Json::Int64(n));
        result["divisors"].append(row);
    }
    return result;
}

Poly contact_product(const GraphParts& differences) {
    Poly product{{{0,0},Num(1)}};
    for (const auto& g:differences) {
        Poly next;
        for (const auto& [a,c]:product) for (const auto& [b,d]:g)
            put(next,{add_powers(a.first,b.first),add_powers(a.second,b.second)},c*d);
        product=move(next);
    }
    return product;
}

using atlas::local::Polynomial;
using atlas::local::Powers;

struct FiberDegree { int axis, degree; };
vector<FiberDegree> graph_fibers(const J& terms, Polynomial& f, bool& wide) {
    array<mpz_class,3> degrees{};
    for (const auto& term:terms) {
        if (!term.isArray() || term.size()!=4) throw invalid_argument("Invalid graph term.");
        Powers powers(3);
        for (int axis=0;axis<3;++axis) {
            powers[axis]=mpz_class(term[axis].asString(),10);
            if (powers[axis]<0) throw invalid_argument("Negative polynomial exponent.");
            if (powers[axis]>2048) wide=true;
            degrees[axis]=max(degrees[axis],powers[axis]);
        }
        // The graph factors through the origin each contribute one to the multiplicity.
        if (powers[0]+powers[1]+powers[2]<3) return {};
        mpq_class coefficient(term[3].asString(),10);
        if (coefficient.get_den()==0) throw invalid_argument("Invalid graph coefficient.");
        coefficient.canonicalize();
        if (coefficient!=0) f.emplace(move(powers),move(coefficient));
    }
    vector<FiberDegree> fibers;
    for (int axis=0;axis<3;++axis) {
        Powers monomial(3);
        monomial[axis]=degrees[axis];
        if ((degrees[axis]==3 || degrees[axis]==4 || degrees[axis]==6) && f.count(monomial))
            fibers.push_back({axis,static_cast<int>(degrees[axis].get_ui())});
    }
    return fibers;
}

optional<vector<Polynomial>> distinct_graph_factors(const Polynomial& f, atlas::local::SparseLexRing& ring) {
    atlas::PolyValue equation(ring.ctx);
    ring.write(equation.p,f);
    GraphFactorization decomposition(ring.ctx);
    if (!fmpq_mpoly_factor(decomposition.factors,equation.p,ring.ctx)) return nullopt;
    const auto count=decomposition.factors->num;
    if (count!=3 && count!=4) return nullopt;
    vector<Polynomial> factors(count);
    for (slong j=0;j<count;++j) {
        if (!fmpz_is_one(decomposition.factors->exp+j)) return nullopt;
        factors[j]=ring.read(decomposition.factors->poly+j);
        if (factors[j].count(Powers(3))) return nullopt;
    }
    sort(factors.begin(),factors.end());
    return factors;
}

// On this axis every factor must be a nonzero rational multiple of z-root(x,y).
optional<GraphParts> graph_roots(const vector<Polynomial>& factors, int fiber,
                                Axes axes, atlas::local::SparseLexRing& ring) {
    Powers linear(3);
    linear[fiber]=1;
    GraphParts roots(factors.size());
    for (size_t j=0;j<factors.size();++j) {
        const auto& g=factors[j];
        auto leading=g.find(linear);
        if (leading==g.end()) return nullopt;
        for (const auto& [p,c]:g) {
            if (p==linear) continue;
            if (p[fiber]!=0) return nullopt;
            mpq_class coefficient=-c/leading->second;
            Num n;
            fmpq_set_mpq(ring.coefficient,coefficient.get_mpq_t());
            qqbar_set_fmpq(n.n,ring.coefficient);
            put(roots[j],{p[axes[0]].get_si(),p[axes[1]].get_si()},n);
        }
    }
    return roots;
}

J nearby_graph_strata(const string& fiber, const J& roots, const J& differences) {
    J strata=arr();
    vector<string> equations(roots.size());
    for (Json::ArrayIndex j=0;j<roots.size();++j) {
        const string root=roots[j].asString();
        equations[j]=root=="0" ? fiber : fiber+"-("+root+")";
        J smooth;
        smooth["model"]="smooth component";
        smooth["component"]="graph "+to_string(j+1);
        smooth["equation"]=equations[j];
        smooth["multiplicity"]=1;
        smooth["eigenvalue_orders"]=arr({1});
        smooth["factors"]=arr({arr({"1","-1"})});
        strata.append(smooth);
    }
    Json::ArrayIndex k=0;
    for (Json::ArrayIndex i=0;i<roots.size();++i) for (Json::ArrayIndex j=i+1;j<roots.size();++j) {
        J node;
        node["model"]="nearby transverse node";
        node["component"]="intersection of graphs "+to_string(i+1)+" and "+to_string(j+1);
        node["equations"]=arr({equations[i],differences[k++]});
        node["eigenvalue_orders"]=arr({1});
        node["factors"]=arr();
        strata.append(node);
    }
    return strata;
}

J resolve_contact_base(Resolver& solver, const GraphParts& contact) {
    auto product=contact_product(contact);
    solver.embedded=true;
    auto resolution=solver.run(product);
    J result=trace_contact(contact,resolution,solver);
    const array<string_view,2> names{solver.names[0],solver.names[1]};
    result["base_polynomial"]=solver.format(product,names);
    result["base_resolution"]=move(resolution);
    return result;
}

J graph_certificate(const J& req, int fiber, Axes axes, const GraphParts& roots) {
    GraphParts differences;
    for (size_t i=0;i<roots.size();++i) for (size_t j=i+1;j<roots.size();++j) {
        auto difference=roots[i];
        for (const auto& [p,c]:roots[j]) put(difference,p,Num(-1)*c);
        differences.push_back(move(difference));
    }
    Resolver solver;
    solver.names={req["variables"][axes[0]].asString(),req["variables"][axes[1]].asString()};
    J result=resolve_contact_base(solver,differences);
    result["fiber_variable"]=req["variables"][fiber];
    result["base_variables"]=arr({solver.names[0],solver.names[1]});
    const array<string_view,2> names{solver.names[0],solver.names[1]};
    result["roots"]=arr(); result["differences"]=arr();
    for (const auto& root:roots) result["roots"].append(solver.format(root,names));
    for (const auto& diff:differences) result["differences"].append(solver.format(diff,names));
    result["nearby_strata"]=nearby_graph_strata(result["fiber_variable"].asString(),
                                               result["roots"],result["differences"]);
    if (roots.size()==4) result["graph_count"]=4;
    if (req.isMember("saved_contact") && req["saved_contact"]!=result)
        throw invalid_argument("Saved graph contact data disagree with the polynomial.");
    result["integration"]=roots.size()==3 ? atlas::three_graph_zetas(result) : atlas::four_graph_zetas(result);
    return result;
}

#include "quadratic_translate_contact.inc"

J polynomial_graph_contact(const J& req) {
    if (req["variables"].size()!=3) return J();
    Polynomial f;
    bool wide=false;
    auto fibers=graph_fibers(req["terms"],f,wide);
    if (fibers.empty()) return J();
    atlas::local::SparseLexRing ring(3);
    // In a wide sextic, a common high-degree translation can disappear
    // completely from the discriminant. Factor only after removing it.
    optional<vector<Polynomial>> factors;
    if (!wide) {
        factors=distinct_graph_factors(f,ring);
        if (!factors) return J();
    }
    for (auto [fiber,degree]:fibers) {
        Axes axes{};
        int count=0;
        for (int j=0;j<3;++j) if (j!=fiber) axes[count++]=j;
        try {
            if (degree==6) {
                Polynomial shift;
                if (wide) {
                    auto centered=remove_wide_fiber_shift(f,fiber,shift,ring);
                    if (!centered) continue;
                    factors=distinct_graph_factors(*centered,ring);
                    if (!factors) continue;
                }
                if (auto model=quadratic_translates(*factors,fiber,shift,ring))
                    return quadratic_contact_certificate(req,f,fiber,axes,*model,ring);
            } else if (!wide) {
                if (auto roots=graph_roots(*factors,fiber,axes,ring))
                    return graph_certificate(req,fiber,axes,*roots);
            }
        } catch (const NonReducedPlane&) {
            // Another projection may have a reduced contact divisor.
            continue;
        }
    }
    return J();
}
} // namespace
