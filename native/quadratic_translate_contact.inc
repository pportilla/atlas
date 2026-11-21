// A sextic in the fibre coordinate with three distinct factors (z-c_i)^2-D.
// All polynomial comparisons take place over QQ, before resolving the reduced
// branch divisor in the base. The terms supplied on the wire may be expanded.
using atlas::PolyValue;
using atlas::local::SparseLexRing;

bool exceeds_contact_degree(const Powers& powers) {
    return any_of(powers.begin(),powers.end(),[](const mpz_class& e){return e>2048;});
}

// Recover the mean of the six roots from the fifth
// coefficient. Horner substitution in the fibre uses only six products,
// including when a base exponent is larger than a native word.
optional<Polynomial> remove_wide_fiber_shift(const Polynomial& f, int fiber,
                                             Polynomial& shift, SparseLexRing& ring) {
    array<Polynomial,7> coefficients;
    for (const auto& [p,c]:f) {
        auto base=p;
        base[fiber]=0;
        coefficients[p[fiber].get_ui()].emplace(move(base),c);
    }
    if (coefficients[6].size()!=1 || !coefficients[6].count(Powers(3))) return nullopt;
    const mpq_class slope=-6*coefficients[6].begin()->second;
    for (const auto& [p,c]:coefficients[5]) shift.emplace(p,c/slope);
    if (shift.empty()) return nullopt;

    Polynomial translated=shift;
    Powers z(3); z[fiber]=1;
    translated[z]=1;
    PolyValue linear(ring.ctx), value(ring.ctx), coefficient(ring.ctx);
    ring.write(linear.p,translated);
    ring.write(value.p,coefficients[6]);
    for (int k=5;k>=0;--k) {
        // Keep the optional factorization probe within a sparse-work budget.
        if (fmpq_mpoly_length(value.p,ring.ctx)>8192 ||
            fmpq_mpoly_length(value.p,ring.ctx)*translated.size()>250000) return nullopt;
        fmpq_mpoly_mul(value.p,value.p,linear.p,ring.ctx);
        ring.write(coefficient.p,coefficients[k]);
        fmpq_mpoly_add(value.p,value.p,coefficient.p,ring.ctx);
    }
    if (fmpq_mpoly_length(value.p,ring.ctx)>8192) return nullopt;
    auto result=ring.read(value.p);
    for (const auto& term:result) if (exceeds_contact_degree(term.first)) return nullopt;
    return result;
}

struct QuadraticTranslates {
    vector<Polynomial> centres;
    Polynomial radicand;
};

// For z²+l*z+k recover c=-l/2 and D=c²-k. Restore the common
// translation after comparing the three radicands in the centered coordinates.
optional<QuadraticTranslates> quadratic_translates(const vector<Polynomial>& factors, int fiber,
                                                   const Polynomial& shift, SparseLexRing& ring) {
    if (factors.size()!=3) return nullopt;
    Powers quadratic(3); quadratic[fiber]=2;
    QuadraticTranslates model;
    PolyValue cpoly(ring.ctx), kpoly(ring.ctx);
    for (const auto& g:factors) {
        const auto leading=g.find(quadratic);
        if (leading==g.end()) return nullopt;
        Polynomial centre, constant;
        for (const auto& [p,c]:g) {
            if (p==quadratic) continue;
            if (p[fiber]>1) return nullopt;
            if (p[fiber]==0) constant.emplace(p,c/leading->second);
            else {
                auto base=p; base[fiber]=0;
                centre.emplace(move(base),-c/(2*leading->second));
            }
        }
        if (centre.count(Powers(3))) return nullopt;
        ring.write(cpoly.p,centre);
        ring.write(kpoly.p,constant);
        fmpq_mpoly_mul(cpoly.p,cpoly.p,cpoly.p,ring.ctx);
        fmpq_mpoly_sub(cpoly.p,cpoly.p,kpoly.p,ring.ctx);
        auto d=ring.read(cpoly.p);
        if (d.empty() || d.count(Powers(3))) return nullopt;
        if (model.centres.empty()) model.radicand=move(d);
        else if (model.radicand!=d) return nullopt;
        for (const auto& [p,c]:shift) {
            centre[p]+=c;
            if (centre[p]==0) centre.erase(p);
        }
        model.centres.push_back(move(centre));
    }
    return model;
}

Poly as_plane_polynomial(const Polynomial& f, Axes axes, SparseLexRing& ring) {
    Poly base;
    for (const auto& [p,c]:f) {
        if (exceeds_contact_degree(p)) throw Fail("Plane contact degree exceeds the native probe budget.");
        Num a;
        fmpq_set_mpq(ring.coefficient,c.get_mpq_t());
        qqbar_set_fmpq(a.n,ring.coefficient);
        put(base,{p[axes[0]].get_si(),p[axes[1]].get_si()},a);
    }
    return base;
}

// A collision occurs at D=0 for the two roots of one quadratic. Between
// distinct quadratics it occurs at d=0 or d²-4D=0, where d=c_i-c_j.
GraphParts quadratic_contact_factors(const Poly& d, const GraphParts& differences) {
    GraphParts contacts{d};
    contacts.insert(contacts.end(),differences.begin(),differences.end());
    for (const Poly& delta:differences) {
        auto crossing=contact_product({delta,delta});
        for (const auto& [p,c]:d) put(crossing,p,Num(-4)*c);
        if (crossing.empty()) throw NonReducedPlane();
        contacts.push_back(move(crossing));
    }
    return contacts;
}

string pretty(const PolyValue& polynomial, const J& variables, SparseLexRing& ring) {
    const char* names[]={variables[0].asCString(),variables[1].asCString(),variables[2].asCString()};
    return atlas::flint_string(fmpq_mpoly_get_str_pretty(polynomial.p,names,ring.ctx));
}

// The reduced contact divisor is smooth away from the origin in a small
// punctured neighbourhood. Its distinct branches do not intersect there.
// On d=0 either root of the first quadratic gives a transverse node. On
// d²=4D only the fibre midpoint is a common root. Along D=0 the nonzero
// differential of D makes the covering hypersurface smooth.
J nearby_quadratic_strata(const J& proof) {
    J strata=arr();
    const J& factors=proof["quadratic_factors"];
    const J& centres=proof["centres"];
    for (Json::ArrayIndex i=0;i<factors.size();++i) {
        J smooth;
        smooth["model"]="smooth component";
        smooth["component"]="quadratic "+to_string(i+1);
        smooth["equation"]=factors[i];
        smooth["multiplicity"]=1;
        smooth["eigenvalue_orders"]=arr({1});
        smooth["factors"]=arr({arr({"1","-1"})});
        strata.append(smooth);
    }
    Json::ArrayIndex pair=0;
    for (Json::ArrayIndex i=0;i<3;++i) for(Json::ArrayIndex j=i+1;j<3;++j,++pair) {
        const string name="intersection of quadratics "+to_string(i+1)+" and "+to_string(j+1);
        J node;
        node["model"]="nearby transverse node";
        node["component"]=name+" at coinciding centres";
        node["eigenvalue_orders"]=arr({1});
        node["factors"]=arr();
        node["equations"]=arr({proof["contact_factors"][1+pair],factors[i]});
        strata.append(node);
        node["component"]=name+" at the midpoint";
        string midpoint="2*"+proof["fiber_variable"].asString()+"-("+centres[i].asString()+")-("+centres[j].asString()+")";
        node["equations"]=arr({proof["contact_factors"][4+pair],midpoint});
        strata.append(node);
    }
    return strata;
}

J quadratic_contact_certificate(const J& req, const Polynomial& input, int fiber, Axes axes,
                                const QuadraticTranslates& model, SparseLexRing& ring) {
    Powers pure(3); pure[fiber]=6;
    const auto& unit=input.at(pure);
    PolyValue product(ring.ctx), z(ring.ctx), centre(ring.ctx), d(ring.ctx), factor(ring.ctx);
    ring.write(product.p,Polynomial{{Powers(3),unit}});
    fmpq_mpoly_gen(z.p,fiber,ring.ctx);
    ring.write(d.p,model.radicand);
    J centres=arr(), factors=arr();
    for (const auto& c:model.centres) {
        ring.write(centre.p,c);
        centres.append(pretty(centre,req["variables"],ring));
        fmpq_mpoly_sub(factor.p,z.p,centre.p,ring.ctx);
        fmpq_mpoly_mul(factor.p,factor.p,factor.p,ring.ctx);
        fmpq_mpoly_sub(factor.p,factor.p,d.p,ring.ctx);
        factors.append(pretty(factor,req["variables"],ring));
        fmpq_mpoly_mul(product.p,product.p,factor.p,ring.ctx);
    }
    ring.write(factor.p,input);
    if (!fmpq_mpoly_equal(product.p,factor.p,ring.ctx))
        throw invalid_argument("The quadratic factors do not reconstruct the input polynomial.");

    GraphParts differences;
    for (size_t i=0;i<3;++i) for (size_t j=i+1;j<3;++j) {
        ring.write(centre.p,model.centres[i]);
        ring.write(factor.p,model.centres[j]);
        fmpq_mpoly_sub(centre.p,centre.p,factor.p,ring.ctx);
        if (fmpq_mpoly_is_zero(centre.p,ring.ctx)) throw NonReducedPlane();
        differences.push_back(as_plane_polynomial(ring.read(centre.p),axes,ring));
    }
    auto contacts=quadratic_contact_factors(as_plane_polynomial(model.radicand,axes,ring),differences);
    Resolver solver;
    solver.names={req["variables"][axes[0]].asString(),req["variables"][axes[1]].asString()};
    J result=resolve_contact_base(solver,contacts);
    const array<string_view,2> names{solver.names[0],solver.names[1]};
    result["contact_model"]="quadratic-translates";
    result["fiber_variable"]=req["variables"][fiber];
    result["base_variables"]=arr({solver.names[0],solver.names[1]});
    result["centres"]=move(centres);
    result["radicand"]=pretty(d,req["variables"],ring);
    result["quadratic_factors"]=move(factors);
    result["contact_factors"]=arr();
    for (const auto& g:contacts) result["contact_factors"].append(solver.format(g,names));
    result["nearby_strata"]=nearby_quadratic_strata(result);
    // The rational unit is immaterial to both zetas. It was checked in the
    // reconstruction above, including when replaying a saved certificate.
    if (req.isMember("saved_contact") && req["saved_contact"]!=result)
        throw invalid_argument("Saved quadratic contact data disagree with the polynomial.");
    result["integration"]=atlas::quadratic_translate_zetas(result);
    return result;
}
