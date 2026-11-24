// Included with the graph verifier's Support and Fan types in scope.
namespace graph_intrinsic {
using Number = atlas::AlgebraicNumber;
using Laurent = std::map<Powers, Number>;
using FacePair = std::pair<Face, Face>;

Z pairing(const Powers& a, const Powers& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), Z(0));
}
void subtract(Powers& a, const Powers& b, const Z& multiple) {
    for (size_t i=0; i<a.size(); ++i) a[i] -= multiple*b[i];
}
Number algebraic(const Q& rational) {
    fmpq_t value;
    fmpq_init(value);
    fmpq_set_str(value, rational.get_str().c_str(), 10);
    Number result;
    qqbar_set_fmpq(result.n, value);
    fmpq_clear(value);
    return result;
}
Number power(const Number& a, const Z& exponent) {
    if (exponent==0) return Number(1);
    atlas::Z e;
    fmpz_set_str(&e.v, exponent.get_str().c_str(), 10);
    Number result;
    qqbar_pow_fmpz(result.n, a.n, &e.v);
    return result;
}
Laurent terms(const Support& support, const Face& indices) {
    Laurent result;
    for (int i:indices) result.emplace(support.terms[i].first, algebraic(support.terms[i].second));
    return result;
}
PointSet differences(const Laurent& polynomial) {
    PointSet out;
    const auto& base=polynomial.begin()->first;
    for (auto i=std::next(polynomial.begin()); i!=polynomial.end(); ++i) {
        Powers e=i->first;
        subtract(e,base,1);
        out.push_back(std::move(e));
    }
    return out;
}
long rank(const PointSet& rows) {
    if (rows.empty()) return 0;
    fmpz_mat_t matrix;
    fmpz_mat_init(matrix,rows.size(),rows.front().size());
    for (size_t i=0;i<rows.size();++i)
        for (size_t j=0;j<rows[i].size();++j)
            fmpz_set_str(fmpz_mat_entry(matrix,i,j), rows[i][j].get_str().c_str(),10);
    const long value=fmpz_mat_rank(matrix);
    fmpz_mat_clear(matrix);
    return value;
}

struct Character {
    Powers direction;
    Powers dual;  // integral retraction onto this primitive character
    Number value;
};
using Sheet = std::vector<Character>;
using Sheets = std::vector<Sheet>;

std::vector<Character> binomial_roots(const Laurent& polynomial) {
    const auto& [p,a]=*polynomial.begin();
    const auto& [q,b]=*std::next(polynomial.begin());
    Powers u=q, dual(q.size());
    subtract(u,p,1);
    Z gcd=0;
    for (size_t j=0;j<u.size();++j) {
        Z next, left, right;
        mpz_gcdext(next.get_mpz_t(),left.get_mpz_t(),right.get_mpz_t(),gcd.get_mpz_t(),u[j].get_mpz_t());
        for (size_t i=0;i<j;++i) dual[i]*=left;
        dual[j]=right;
        gcd=std::move(next);
    }
    if (gcd>std::numeric_limits<Json::ArrayIndex>::max())
        throw std::length_error("The intrinsic graph section exceeds the JSON array size.");
    for (auto& c:u) c/=gcd;
    const unsigned long degree=gcd.get_ui();
    Number target, first, unity;
    qqbar_div(target.n,a.n,b.n);
    qqbar_neg(target.n,target.n);
    qqbar_root_ui(first.n,target.n,degree);
    qqbar_root_of_unity(unity.n,1,degree);
    std::vector<Character> result;
    for (unsigned long i=0;i<degree;++i) {
        result.push_back({u,dual,first});
        if (i+1<degree) first=first*unity;
    }
    std::sort(result.begin(),result.end(),[](const auto& a,const auto& b){
        int compare=qqbar_cmp_re(a.value.n,b.value.n);
        return compare ? compare<0 : qqbar_cmp_im(a.value.n,b.value.n)<0;
    });
    return result;
}

// The splitting Z^n=Z*u + ker(dual) specializes exactly at every root,
// including cancellations among terms on the same projected character.
Laurent restrict_to(const Laurent& polynomial, const Character& root) {
    Laurent out;
    for (const auto& [e,c]:polynomial) {
        Powers projected=e;
        Z coefficient=pairing(e,root.dual);
        subtract(projected,root.direction,coefficient);
        auto value=c*power(root.value,coefficient);
        auto [pos,new_term]=out.try_emplace(projected,value);
        if (!new_term) pos->second=pos->second+value;
        if (pos->second.zero()) out.erase(pos);
    }
    return out;
}

Sheets classify(const Laurent& a, const Laurent& b) {
    if (a.size()<2 || b.size()<2) return {};
    if (a.size()==2 || b.size()==2) {
        const bool first=a.size()==2;
        Sheets result;
        for (const auto& root:binomial_roots(first?a:b)) {
            auto section=restrict_to(first?b:a,root);
            if (section.empty())
                throw std::invalid_argument("Dependent initial equations on a marked torus.");
            if (section.size()==1) continue;
            if (section.size()==2) {
                for (auto& other:binomial_roots(section)) result.push_back({root,std::move(other)});
            } else if (rank(differences(section))>=2) {
                result.push_back({root});
            } else {
                throw std::invalid_argument("Intrinsic graph section needs a general univariate root family.");
            }
        }
        return result;
    }
    // On the smooth projective toric compactification all orbit pairs are
    // transverse. The Koszul resolution and Batyrev–Borisov vanishing for
    // O(-P), O(-Q), O(-P-Q) give H^0(O_CI)=C when each Newton dimension is
    // at least 2 and their sum has dimension >=3. For a hypersurface the
    // same argument works in Newton dimension >=2. Smooth and connected
    // implies irreducible, including after restoring any free torus factors.
    auto p=differences(a),q=differences(b);
    if (rank(p)>=2 && rank(q)>=2) {
        p.insert(p.end(),q.begin(),q.end());
        if (rank(p)>=3) return {Sheet{}};
    }
    throw std::invalid_argument("Intrinsic graph section needs a finite Laurent-system certificate.");
}

bool contains(const Sheet& parent, const Sheet& child) {
    for (const auto& condition:parent) {
        Powers remaining=condition.direction;
        Number value(1);
        for (const auto& component:child) {
            Z exponent=pairing(remaining,component.dual);
            subtract(remaining,component.direction,exponent);
            value=value*power(component.value,exponent);
        }
        if (std::any_of(remaining.begin(),remaining.end(),[](const Z& v){return v!=0;}) ||
            !qqbar_equal(value.n,condition.value.n)) return false;
    }
    return true;
}

struct Mark {
    std::string symbol;
    JsonValue description;
};
Mark describe(const Number& value) {
    auto polynomial=QQBAR_POLY(value.n);
    const slong degree=fmpz_poly_degree(polynomial);
    auto roots=_qqbar_vec_init(degree);
    qqbar_roots_fmpz_poly(roots,polynomial,0);
    slong index=0;
    while (index<degree && !qqbar_equal(value.n,roots+index)) ++index;
    _qqbar_vec_clear(roots,degree);
    if (index==degree) throw std::logic_error("A marked graph root is missing from its minimal polynomial.");
    std::string family="ciRoot", coefficients;
    for (slong i=0;i<=degree;++i) {
        auto c=atlas::flint_string(fmpz_get_str(nullptr,10,fmpz_poly_get_coeff_ptr(polynomial,i)));
        family+=(c.front()=='-'?"m"+c.substr(1):"p"+c);
        if (i) coefficients+=", ";
        coefficients+=c;
    }
    JsonValue meta;
    meta["polynomial"]=atlas::flint_string(fmpz_poly_get_str_pretty(polynomial,"T"));
    meta["degree"]=Json::UInt64(degree);
    meta["root_index"]=Json::UInt64(index+1);
    meta["ordering"]="FLINT qqbar_roots_fmpz_poly";
    meta["exact"]="PolynomialRootIndexed(List("+coefficients+"), "+std::to_string(index+1)+")";
    return {family+"_"+std::to_string(index+1),std::move(meta)};
}
} // namespace graph_intrinsic

struct IntrinsicGraphIncidence {
    using Pair=graph_intrinsic::FacePair;
    using Sheets=graph_intrinsic::Sheets;
    using Sheet=graph_intrinsic::Sheet;
    const Support& g;
    const Support& f;
    std::map<Pair,Sheets> sections;
    std::map<Face,Pair> intersections;
    std::map<std::pair<Pair,Pair>,std::vector<size_t>> restrictions;
    std::vector<std::pair<graph_intrinsic::Number,graph_intrinsic::Mark>> marks;

    IntrinsicGraphIncidence(const Support& g,const Support& f) : g(g),f(f) {}

    const Sheets& section(const Pair& pair) {
        auto it=sections.find(pair);
        if (it==sections.end()) {
            auto value=graph_intrinsic::classify(graph_intrinsic::terms(g,pair.first),
                                                 graph_intrinsic::terms(f,pair.second));
            it=sections.emplace(pair,std::move(value)).first;
        }
        return it->second;
    }
    void observe(const Face& orbit,const Face& a,const Face& b) {
        Pair pair{a,b};
        section(pair);
        intersections.emplace(orbit,std::move(pair));
    }
    const graph_intrinsic::Mark& mark(const graph_intrinsic::Number& value) {
        for (const auto& [constant,name]:marks)
            if (qqbar_equal(value.n,constant.n)) return name;
        marks.emplace_back(value,graph_intrinsic::describe(value));
        return marks.back().second;
    }
    std::string id(const Face& cone,size_t index) const {
        const auto count=sections.at(intersections.at(cone)).size();
        if (cone.size()==1) return label(cone.front())+(count>1?"."+std::to_string(index+1):"");
        std::string base="C";
        for (int r:cone) base+=(base.size()>1?"-":"")+label(r);
        return base+"."+std::to_string(index+1);
    }
    std::string restriction(const Face& cone,const Pair& child,size_t index) {
        const Pair& parent=intersections.at(cone);
        if (sections.at(parent).size()==1) return id(cone,0);
        if (parent==child) return id(cone,index);
        const auto key=std::make_pair(parent,child);
        auto it=restrictions.find(key);
        if (it==restrictions.end()) {
            std::vector<size_t> indices;
            const auto& targets=sections.at(parent);
            for (const auto& sheet:sections.at(child)) {
                size_t match=targets.size();
                for (size_t j=0;j<targets.size();++j) {
                    if (!graph_intrinsic::contains(targets[j],sheet)) continue;
                    if (match!=targets.size()) throw std::invalid_argument("Ambiguous intrinsic graph sheet attachment.");
                    match=j;
                }
                if (match==targets.size()) throw std::invalid_argument("Missing intrinsic graph sheet attachment.");
                indices.push_back(match);
            }
            it=restrictions.emplace(key,std::move(indices)).first;
        }
        return id(cone,it->second.at(index));
    }
    JsonValue complex(const Fan& fan) {
        JsonValue out;
        out["status"]="complete";
        out["convention"]="Exceptional divisor on the resolved strict transform over the origin over Qbar.";
        out["vertices"]=JsonValue(Json::arrayValue);
        out["cells"]=JsonValue(Json::arrayValue);
        for (const auto& [cone,pair]:intersections) {
            const auto& pieces=sections.at(pair);
            for (size_t i=0;i<pieces.size();++i) {
                JsonValue row;
                row["id"]=id(cone,i);
                if (pieces.size()>1) for (const auto& condition:pieces[i]) {
                    JsonValue equation;
                    equation["direction"]=atlas::sparse_json::strings(condition.direction);
                    equation["value"]=mark(condition.value).symbol;
                    row["characters"].append(std::move(equation));
                }
                if (cone.size()==1) {
                    int r=cone.front();
                    row["dimension"]=fan.n-3;
                    row["ray_index"]=r;
                    row["total_multiplicity"]=integer(f.minima[r]);
                    if (pieces.size()>1) row["ray_divisor"]=label(r);
                    out["vertices"].append(std::move(row));
                } else {
                    row["dimension"]=Json::UInt64(cone.size()-1);
                    for (int r:cone) row["vertices"].append(restriction({r},pair,i));
                    for (size_t j=0;j<cone.size();++j) {
                        Face sub=cone;
                        sub.erase(sub.begin()+j);
                        row["faces"].append(restriction(sub,pair,i));
                    }
                    out["cells"].append(std::move(row));
                }
            }
        }
        for (const auto& pair:marks) out["root_families"][pair.second.symbol]=pair.second.description;
        return out;
    }
    void append_chart_divisors(JsonValue& rows,const JsonValue& divisor,const Face& cone,
                              const Fan& fan,const Graph& graph,int r) {
        const auto& pieces=section({g.initial({r}),f.initial({r})});
        if (pieces.size()==1) { rows.append(divisor);return; }
        for (size_t i=0;i<pieces.size();++i) {
            JsonValue piece=divisor;
            piece["ray_divisor"]=divisor["id"];
            piece["id"]=divisor["id"].asString()+"."+std::to_string(i+1);
            piece["ideal"].append(divisor["equation"]);
            auto equations=divisor["equation"].asString();
            std::vector<bool> invertible(cone.size());
            for (const auto& condition:pieces[i]) {
                Powers positive(cone.size()), negative(cone.size());
                bool plus=false,minus=false;
                for (size_t j=0;j<cone.size();++j) {
                    Z v=graph_intrinsic::pairing(fan.rays[cone[j]],condition.direction);
                    if (v>0) {positive[j]=v;if(!invertible[j]) plus=true;}
                    if (v<0) {negative[j]=-v;if(!invertible[j]) minus=true;}
                }
                // Clearing denominators is saturated here: on the preceding
                // equations one entire side is a unit. The new equation then
                // makes every remaining coordinate in its support invertible.
                if (plus && minus)
                    throw std::invalid_argument("Mixed character signs need a saturated graph sheet ideal.");
                for (size_t j=0;j<cone.size();++j)
                    invertible[j]=invertible[j] || positive[j]!=0 || negative[j]!=0;
                auto equation=atlas::local::format_polynomial({{positive,1}},graph.chart_vars)
                    +"-"+mark(condition.value).symbol+"*"
                    +atlas::local::format_polynomial({{negative,1}},graph.chart_vars);
                piece["ideal"].append(equation);
                equations+=", "+equation;
            }
            piece["equation"]=equations;
            rows.append(std::move(piece));
        }
    }
};
