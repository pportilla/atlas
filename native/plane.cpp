// Exact point blowups of plane curve germs over the algebraic closure of Q.
#include <flint/fexpr.h>
#include <flint/qqbar.h>
#include <flint/fmpq_mpoly.h>
#include <flint/fmpq_poly.h>
#include <flint/fmpz_poly.h>
#include <flint/gr_poly.h>
#include <flint/gr_vec.h>
#include <flint/fmpz_vec.h>
#include <json/json.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>
#include <sys/resource.h>
using namespace std;
using J = Json::Value;
using Exp = slong;
using Key = pair<Exp,Exp>;
using Fail = runtime_error;

// This worker handles exactly one request. Quick mode retains the original
// exploration bounds; both modes check the actual exponent representation.
bool quick = true;
Exp add_powers(Exp a, Exp b) {
    if (a<0 || b<0 || a>numeric_limits<Exp>::max()-b)
        throw Fail("Plane exponent or multiplicity exceeds a signed FLINT word.");
    return a+b;
}
Exp multiply_powers(Exp a, Exp b) {
    if (a && b>numeric_limits<Exp>::max()/a)
        throw Fail("Plane factor exponent exceeds a signed FLINT word.");
    return a*b;
}

struct Num {
    qqbar_t n;
    Num(long i=0) { qqbar_init(n); qqbar_set_si(n,i); }
    Num(const Num& x) { qqbar_init(n); qqbar_set(n,x.n); }
    Num(Num&& x) noexcept { qqbar_init(n); qqbar_swap(n,x.n); }
    Num& operator=(Num x) { qqbar_swap(n,x.n); return *this; }
    ~Num() { qqbar_clear(n); }
    bool zero() const { return qqbar_is_zero(n); }
};
Num operator+(const Num& a,const Num& b) { Num r; qqbar_add(r.n,a.n,b.n); return r; }
Num operator*(const Num& a,const Num& b) { Num r; qqbar_mul(r.n,a.n,b.n); return r; }
Num inverse(const Num& a) { Num r; qqbar_inv(r.n,a.n); return r; }
string taken(char* p) { string r(p); flint_free(p); return r; }
J arr(initializer_list<J> xs={}) { J j(Json::arrayValue); for(const auto& x:xs) j.append(x); return j; }
using Poly=map<Key,Num>;
using Pair=array<Poly,2>;
using Factors=vector<pair<Poly,Exp>>;
using Axes=array<int,2>;
using Coord=array<Num,2>;
void put(Poly& p,Key k,const Num& a) {
    if(a.zero()) return;
    auto [it,created]=p.try_emplace(k,a);
    if(!created) { it->second=it->second+a; if(it->second.zero()) p.erase(it); }
    Exp degree=add_powers(k.first,k.second);
    if(quick && (p.size()>8192 || degree>2048))
        throw Fail("Chart expansion exceeded the budget (8192 terms or degree 2048).");
}
Exp order(const Poly& f) {
    if(f.empty()) throw Fail("Unexpected zero polynomial in a plane chart.");
    Exp m=numeric_limits<Exp>::max();
    for(auto& [k,c]: f) m=min(m,add_powers(k.first,k.second));
    return m;
}
// Substitute parent=(u,u*v) or (u*v,v), dividing by exceptional^m.
Poly blow(const Poly& f,int pivot,Exp m) {
    Poly g;
    for(auto& [k,c]:f) {
        Exp s=add_powers(k.first,k.second)-m;
        put(g,pivot==0?Key{s,k.second}:Key{k.first,s},c);
    }
    return g;
}
// Translate one coordinate of the parent chart to a geometric center.
Poly shift(const Poly& f,int axis,const Num& a) {
    if(a.zero()) return f;
    Poly g;
    for(auto& [k,c]: f) {
        Exp e=axis?k.second:k.first;
        Num coeff=c;
        for(Exp j=e;j>=0;--j) {
            put(g,axis?Key{k.first,j}:Key{j,k.second},coeff);
            if(j){coeff=coeff*a;qqbar_mul_ui(coeff.n,coeff.n,j);qqbar_div_ui(coeff.n,coeff.n,e-j+1);}
        }
    }
    return g;
}

// Combine monomial factors as the two chart coordinates. Long chains at a
// coordinate point then produce compact pullback equations, while translations
// at other points keep the exact polynomial unit factors.
void append_factor(Factors& factors,Poly factor,Exp power) {
    if(factor.size()!=1 || !qqbar_is_one(factor.begin()->second.n)) {
        factors.emplace_back(move(factor),power);
        return;
    }
    auto [a,b]=factor.begin()->first;
    for(auto [degree,key]:{pair{a,Key{1,0}},pair{b,Key{0,1}}}) {
        if(!degree) continue;
        Exp order=multiply_powers(degree,power);
        auto item=find_if(factors.begin(),factors.end(),[&](const auto& old) {
            return old.first.size()==1 && old.first.begin()->first==key &&
                   qqbar_is_one(old.first.begin()->second.n);
        });
        if(item==factors.end()) factors.emplace_back(Poly{{key,Num(1)}},order);
        else item->second=add_powers(item->second,order);
    }
}

struct Root {
    Num value;
    bool simple;
};

vector<Root> algebraic_roots(const vector<pair<Exp,Num>>& terms) {
    gr_ctx_t ctx; gr_ctx_init_complex_qqbar(ctx);
    gr_poly_t p; gr_poly_init(p,ctx);
    int status=0;
    for(auto& [power,c]:terms) {
        if(power==numeric_limits<Exp>::max())
            throw Fail("Root polynomial length exceeds a signed FLINT word.");
        status |= gr_poly_set_coeff_scalar(p,power,c.n,ctx);
    }
    gr_vec_t roots; gr_vec_init(roots,0,ctx);
    fmpz_vec_t mult; fmpz_vec_init(mult,0);
    if(!status) status=gr_poly_roots(roots,mult,p,0,ctx);
    vector<Root> out;
    if(!status) for(slong i=0;i<gr_vec_length(roots,ctx);++i){
        Num r; qqbar_set(r.n,static_cast<qqbar_srcptr>(gr_vec_entry_ptr(roots,i,ctx)));
        out.push_back({move(r),fmpz_is_one(fmpz_vec_entry_srcptr(mult,i))!=0});
    }
    fmpz_vec_clear(mult); gr_vec_clear(roots,ctx); gr_poly_clear(p,ctx); gr_ctx_clear(ctx);
    if(status) throw Fail("The exact algebraic root computation did not finish.");
    return out;
}

vector<Root> ordered_roots(const vector<pair<Exp,Num>>& terms) {
    if(terms.empty()) throw invalid_argument("The zero polynomial does not define a finite root family.");
    vector<Root> out;
    if(terms.size()==1){
        Exp power=terms[0].first;
        if(power) out.push_back({Num(0),power==1});
        return out;
    }
    if(terms.size()==2){
        const auto* low=&terms.front();
        const auto* high=&terms.back();
        if(low->first>high->first) swap(low,high);
        Exp width=high->first-low->first;
        // After removing the zero root, these binomials are T^width ± 1.
        // The negative target uses odd numerators with denominator 2*width.
        if(width>0 && high->first<WORD_MAX && width<=WORD_MAX/2){
            bool negative_target=qqbar_equal(low->second.n,high->second.n);
            Num opposite;
            qqbar_neg(opposite.n,low->second.n);
            if(negative_target || qqbar_equal(opposite.n,high->second.n)){
                if(low->first) out.push_back({Num(0),low->first==1});
                for(Exp k=0;k<width;++k){
                    Num root;
                    qqbar_root_of_unity(root.n,negative_target ? 2*k+1 : k,
                                       negative_target ? 2*width : width);
                    out.push_back({move(root),true});
                }
            }
        }
    }
    if(out.empty()) out=algebraic_roots(terms);
    sort(out.begin(),out.end(),[](const Root& a,const Root& b){
        int s=qqbar_cmp_re(a.value.n,b.value.n);
        return s<0 || (!s && qqbar_cmp_im(a.value.n,b.value.n)<0);
    });
    return out;
}

vector<Root> directions(const Poly& f,Exp m) {
    vector<pair<Exp,Num>> terms;
    for(const auto& [k,c]:f) if(add_powers(k.first,k.second)==m) terms.emplace_back(k.second,c);
    return ordered_roots(terms);
}

// PolynomialRootIndexed uses FLINT's conjugate order, independently of the
// real/imaginary order in which the blowup visits its tangent directions.
vector<string> indexed_labels(const vector<Num>& values) {
    map<string,vector<size_t>> families;
    vector<string> labels(values.size());
    for(size_t i=0;i<values.size();++i){
        if(qqbar_is_rational(values[i].n)){
            fmpq_t q; fmpq_init(q); qqbar_get_fmpq(q,values[i].n);
            labels[i]=taken(fmpq_get_str(nullptr,10,q)); fmpq_clear(q);
        }else{
            families[taken(fmpz_poly_get_str(QQBAR_POLY(values[i].n)))].push_back(i);
        }
    }
    for(const auto& group:families){
        const auto& members=group.second;
        const auto& first=values[members.front()];
        slong degree=qqbar_degree(first.n);
        string prefix="PolynomialRootIndexed(List(";
        for(slong i=0;i<=degree;++i){
            if(i) prefix+=", ";
            prefix+=taken(fmpz_get_str(nullptr,10,QQBAR_COEFFS(first.n)+i));
        }
        prefix+="), ";
        if(members.size()==static_cast<size_t>(degree)){
            auto order=members;
            sort(order.begin(),order.end(),[&](size_t a,size_t b){
                return qqbar_cmp_root_order(values[a].n,values[b].n)<0;
            });
            auto repeated=adjacent_find(order.begin(),order.end(),[&](size_t a,size_t b){
                return qqbar_equal(values[a].n,values[b].n);
            });
            // Distinct roots of one minimal polynomial exhaust the family
            // when their count is its degree. FLINT uses this same ordering.
            if(repeated==order.end()){
                for(size_t j=0;j<order.size();++j) labels[order[j]]=prefix+to_string(j+1)+")";
                continue;
            }
        }
        auto conjugates=_qqbar_vec_init(degree);
        qqbar_conjugates(conjugates,first.n);
        for(size_t i:members){
            slong j=0;
            while(j<degree && !qqbar_equal(values[i].n,conjugates+j)) ++j;
            if(j==degree){
                _qqbar_vec_clear(conjugates,degree);
                throw Fail("An algebraic coefficient is absent from its FLINT conjugates.");
            }
            labels[i]=prefix+to_string(j+1)+")";
        }
        _qqbar_vec_clear(conjugates,degree);
    }
    return labels;
}

J ordered_root_labels(const J& rows) {
    if(!rows.isArray()) throw invalid_argument("Root polynomial terms must be an array.");
    vector<pair<Exp,Num>> terms;
    for(const auto& row:rows) {
        if(!row.isArray() || row.size()!=2 || !row[0].isUInt64() ||
           row[0].asUInt64()>=static_cast<Json::UInt64>(WORD_MAX) || !row[1].isString())
            throw invalid_argument("Expected a nonnegative exponent and a rational coefficient per root polynomial term.");
        fmpq_t q; fmpq_init(q);
        if(fmpq_set_str(q,row[1].asCString(),10)!=0 || fmpz_is_zero(fmpq_denref(q))) {
            fmpq_clear(q);
            throw invalid_argument("Invalid root polynomial coefficient.");
        }
        fmpq_canonicalise(q);
        Num c; qqbar_set_fmpq(c.n,q); fmpq_clear(q);
        if(!c.zero()) terms.emplace_back(row[0].asInt64(),c);
    }
    vector<Num> values;
    for(auto& entry:ordered_roots(terms)) values.push_back(move(entry.value));
    J result=arr();
    for(const auto& label:indexed_labels(values)) result.append(label);
    return result;
}

bool reduced_at_origin(const Poly& f) {
    fmpq_mpoly_ctx_t ctx; fmpq_mpoly_ctx_init(ctx,2,ORD_DEGLEX);
    fmpq_mpoly_t a,b,d,g; for(auto* p:{a,b,d,g}) fmpq_mpoly_init(p,ctx);
    fmpq_t q; fmpq_init(q);
    for(auto& [k,c]:f){
        qqbar_get_fmpq(q,c.n); ulong e[2]={(ulong)k.first,(ulong)k.second};
        fmpq_mpoly_set_coeff_fmpq_ui(a,q,e,ctx);
    }
    fmpq_mpoly_derivative(d,a,0,ctx);
    bool ok=fmpq_mpoly_gcd(b,a,d,ctx);
    fmpq_mpoly_derivative(d,a,1,ctx);
    ok=ok && fmpq_mpoly_gcd(g,b,d,ctx);
    ulong z[2]={0,0}; fmpq_mpoly_get_coeff_fmpq_ui(q,g,z,ctx); bool reduced=!fmpq_is_zero(q);
    fmpq_clear(q); for(auto* p:{a,b,d,g}) fmpq_mpoly_clear(p,ctx); fmpq_mpoly_ctx_clear(ctx);
    if(!ok) throw Fail("Exact square-free test failed.");
    return reduced;
}

struct Resolver {
    array<string,2> names;
    bool embedded;
    J charts=arr(),steps=arr(),branches=arr(),numbers=J(Json::objectValue);
    vector<Num> algebraics;
    vector<int> weights{0};
    vector<Exp> mults{0};
    set<pair<int,int>> edges;
    string scalar(const Num& x) {
        if(qqbar_is_rational(x.n)){
            fmpq_t q;fmpq_init(q);qqbar_get_fmpq(q,x.n);
            string s=taken(fmpq_get_str(nullptr,10,q));fmpq_clear(q);return s;
        }
        size_t i=0;
        for(;i<algebraics.size();++i) if(qqbar_equal(x.n,algebraics[i].n)) break;
        string id="a"+to_string(i+1);
        if(i==algebraics.size()){
            algebraics.push_back(x);
            numbers[id]["approximation"]=taken(qqbar_get_str_nd(x.n,10));
        }
        return id;
    }
    string format(const Poly& f) {
        string out;
        for(auto& [k,c]:f){
            string a=scalar(c), word;
            bool neg=a[0]=='-'; if(neg) a.erase(0,1);
            if(a!="1" || k==Key{0,0}) word=a;
            for(auto [k1,v]:{pair{k.first,"u"},pair{k.second,"v"}}){
                if(!k1) continue;
                if(!word.empty()) word+="*";
                word+=v;if(k1>1) word+="^"+to_string(k1);
            }
            out+=(neg?"-":(out.empty()?"":"+"))+word;
        }
        return out.empty()?"0":out;
    }
    static string E(int i){return "E"+to_string(i);}
    J point(const Coord& c){return arr({scalar(c[0]),scalar(c[1])});}
    bool finished(const Poly& f,Axes ax) {
        if(order(f)!=1) return false;
        if(!embedded || (!ax[0] && !ax[1])) return true;
        // At an exceptional crossing even a transverse strict branch would be a triple point.
        if(ax[0] && ax[1]) return false;
        return f.count(ax[0]?Key{0,1}:Key{1,0});
    }
    struct Task {
        Poly equation;
        Pair mapping;
        Axes axes;
        string parent;
        Coord center;
        J domain;
        Factors factors;
    };
    void visit(Task& task,vector<Task>& todo) {
        auto& [f,base,ax,parent,center,domain,ancestors]=task;
        Exp m=order(f);
        if(quick && steps.size()>=192) throw Fail("Point-blowup limit (192) reached.");
        if(weights.size()>=static_cast<size_t>(numeric_limits<int>::max()))
            throw Fail("The chart identifiers exceed their representation.");
        int e=static_cast<int>(weights.size());weights.push_back(-1);
        mults.push_back(add_powers(m,add_powers(mults[ax[0]],mults[ax[1]])));
        if(ax[0] && ax[1]) edges.erase(minmax(ax[0],ax[1]));
        J s;
        s["id"]="B"+to_string(e); s["parent_chart"]=parent;s["center"]=point(center);
        s["strict_multiplicity"]=Json::Int64(m);s["exceptional_id"]=E(e);
        s["total_multiplicity"]=Json::Int64(mults[e]);
        for(int a:ax) if(a){--weights[a]; edges.insert(minmax(a,e));}
        steps.append(s);

        array<Poly,2> g;array<Pair,2> maps;array<Axes,2> axes{ax,ax};
        array<J,2> cs;array<string,2> ids;array<Factors,2> factors;
        array<J,2> fiber_points{arr(),arr()};
        for(int i=0;i<2;i++){
            ids[i]="B"+to_string(e)+(i?".v":".u");
            g[i]=blow(f,i,m); axes[i][i]=e;
            maps[i]={blow(base[0],i,0),blow(base[1],i,0)};
            auto& c=cs[i];c["id"]=ids[i];c["coordinates"]=arr({"u","v"});
            c["parent_chart"]=parent;c["step"]=s["id"];c["base_domain"]=domain;
            c["strict_transform"]=format(g[i]);c["exceptional"]=arr();
            for(auto& [factor,power]:ancestors) append_factor(factors[i],blow(factor,i,0),power);
            append_factor(factors[i],Poly{{i==0?Key{1,0}:Key{0,1},Num(1)}},m);
            string total;
            for(auto& [factor,power]:factors[i]) total+="("+format(factor)+")^"+to_string(power)+"*";
            c["ambient_total_transform"]=total+"("+format(g[i])+")";
            c["ambient_components"]=arr();
            c["exclude_points"]=arr(); c["map_to_parent"]=J(Json::objectValue);
            for(int j=0;j<2;j++){
                string v=j?"v":"u";
                string origin=scalar(center[j]);
                c["map_to_parent"][parent=="origin"?names[j]:v]=(center[j].zero()?"":origin+"+")+(i==j?v:"u*v");
                c["map"][names[j]]=format(maps[i][j]);
                if(axes[i][j]){
                    J d;d["id"]=E(axes[i][j]);d["equation"]=v;
                    d["total_multiplicity"]=Json::Int64(mults[axes[i][j]]); c["ambient_components"].append(d);if(embedded)c["exceptional"].append(d);
                }
            }
        }
        struct Child {int i; Num a; Poly f; Axes ax; Coord pt;};
        vector<Child> pending;
        auto scan=[&](int i,const Num& a,bool simple){
            Coord pt{};pt[1-i]=a;
            auto ax1=axes[i];if(!a.zero()) ax1[1-i]=0;
            fiber_points[i].append(point(pt));
            bool overlap=(i==0 && !a.zero());
            Coord other{};if(overlap){other[0]=inverse(a);fiber_points[1].append(point(other));}
            // A simple root of the restriction to the new exceptional line
            // gives a smooth transverse branch. At an embedded corner the
            // strict branch and the two exceptional axes form a triple point.
            bool terminal=simple && (!embedded || !ax1[1-i]);
            Poly h;
            if(!terminal){
                h=shift(g[i],1-i,a);
                terminal=finished(h,ax1);
            }
            if(terminal){
                if(branches.size()==numeric_limits<Json::ArrayIndex>::max())
                    throw Fail("The branch count exceeds the output representation.");
                J branch;branch["id"]="P"+to_string(static_cast<size_t>(branches.size())+1);
                branch["chart"]=ids[i];branch["point"]=point(pt);branch["ambient_components"]=arr();
                for(int v:ax1) if(v) branch["ambient_components"].append(E(v));
                branches.append(branch);
                if(!embedded){
                    auto insert_point=[&](int pivot,const Coord& coord){
                        J d;d["id"]=branch["id"];d["ideal"]=arr();string desc;
                        for(int k=0;k<2;k++){
                            string eq=k?"v":"u";if(!coord[k].zero())eq+="-("+scalar(coord[k])+")";
                            d["ideal"].append(eq);if(k)desc+=", ";desc+=eq;
                        }
                        d["equation"]=desc;cs[pivot]["exceptional"].append(d);
                    };
                    insert_point(i,pt);if(overlap)insert_point(1,other);
                }
                return;
            }
            cs[i]["exclude_points"].append(point(pt));
            if(overlap)cs[1]["exclude_points"].append(point(other));
            pending.push_back({i,a,std::move(h),ax1,pt});
        };
        for(const auto& root:directions(f,m)) scan(0,root.value,root.simple);
        // At infinity the tangent polynomial is f_m(u,1); its linear
        // coefficient is the coefficient of x*y^(m-1).
        if(!f.count({0,m})) scan(1,Num(0),f.count({1,m-1}));
        for(auto& c:cs) charts.append(move(c));
        for(auto it=pending.rbegin();it!=pending.rend();++it){
            auto& child=*it;
            auto i=child.i;J d;
            d["parent"]=ids[i];d["exclude_sibling_points"]=arr();
            auto here=point(child.pt);
            for(auto& pt:fiber_points[i])
                if(pt!=here) d["exclude_sibling_points"].append(pt);
            d["invert_parent_coordinates"]=arr();
            if(!child.a.zero() && axes[i][1-i]) d["invert_parent_coordinates"].append(i?"u":"v");
            Pair translated={shift(maps[i][0],1-i,child.a),shift(maps[i][1],1-i,child.a)};
            Factors translated_factors;
            for(auto& [factor,power]:factors[i])
                append_factor(translated_factors,shift(factor,1-i,child.a),power);
            todo.push_back({move(child.f),move(translated),child.ax,ids[i],move(child.pt),
                            move(d),move(translated_factors)});
        }
    }
    J run(const Poly& f){
        if(!reduced_at_origin(f)) throw invalid_argument("The plane curve germ is nonreduced: a repeated factor vanishes at the origin.");
        if(order(f)==1) throw Fail("Expected a singular plane curve.");
        Pair identity={Poly{{{1,0},Num(1)}},Poly{{{0,1},Num(1)}}};
        vector<Task> todo;
        todo.push_back({f,move(identity),{0,0},"origin",{},J("ambient germ at the origin"),{}});
        while(!todo.empty()) {
            Task task=move(todo.back());
            todo.pop_back();
            visit(task,todo);
        }
        J d;d["status"]="complete";d["vertices"]=arr();d["cells"]=arr();
        if(embedded) {
            d["convention"]="Ambient exceptional curves over the origin, over the algebraic closure of Q.";
            for(size_t i=1;i<weights.size();i++){
                J v;v["id"]=E(i);v["self_intersection"]=weights[i];v["total_multiplicity"]=Json::Int64(mults[i]);
                d["vertices"].append(v);
            }
            for(auto [i,j]:edges){J edge;edge["vertices"]=arr({E(i),E(j)});d["cells"].append(edge);}
        }else{
            d["convention"]="Reduced inverse image of the singular point on the smooth curve; one point per geometric branch.";
            for(auto b:branches){b["label"]=b["id"];d["vertices"].append(b);}
        }
        J r;r["status"]="resolved";r["engine"]="C++ / FLINT plane point blowups";
        r["summary"]=to_string(steps.size())+(steps.size()==1?" point blowup; ":" point blowups; ")+to_string(branches.size())+(branches.size()==1?" geometric branch":" geometric branches")+" above the origin.";
        r["charts"]=move(charts);r["steps"]=move(steps);r["dual_complex"]=move(d);
        const auto labels=indexed_labels(algebraics);
        for(size_t i=0;i<labels.size();++i) numbers["a"+to_string(i+1)]["exact"]=labels[i];
        r["branches"]=move(branches);r["algebraic_numbers"]=move(numbers);
        r["notes"]=arr({"Each chart excludes the listed points, which later blowups replace. A child inherits the parent base domain, excluding the other strict-transform points on the fiber and any listed coordinate divisors. Multiplicities refer to the total transform in the ambient plane."});
        return r;
    }
};
// Present a finite set of exact chart coefficients in one number field.
// Each input uses FLINT's own root order (as in PolynomialRootIndexed).
J coefficient_field(const J& roots, const J& rectangles) {
    vector<Num> values;
    for (const auto& row : roots) {
        fmpz_poly_t polynomial;
        fmpz_poly_init(polynomial);
        fmpz_t coefficient;
        fmpz_init(coefficient);
        slong i = 0;
        for (const auto& c : row["coefficients"]) {
            if (fmpz_set_str(coefficient, c.asCString(), 10))
                throw invalid_argument("Invalid number field coefficient.");
            fmpz_poly_set_coeff_fmpz(polynomial, i++, coefficient);
        }
        fmpz_clear(coefficient);
        slong degree = fmpz_poly_degree(polynomial);
        slong index = row["index"].asInt64();
        if (index < 1 || index > degree)
            throw invalid_argument("Invalid algebraic root index.");
        qqbar_ptr conjugates = _qqbar_vec_init(degree);
        qqbar_roots_fmpz_poly(conjugates, polynomial, 0);
        values.emplace_back();
        qqbar_set(values.back().n, conjugates + index - 1);
        _qqbar_vec_clear(conjugates, degree);
        fmpz_poly_clear(polynomial);
    }
    if (values.empty()) throw invalid_argument("No algebraic constants.");
    if (!rectangles.isNull()) {
        // The public expression syntax numbers roots by disjoint rational
        // isolating rectangles. Exact qqbar comparisons locate this marked
        // embedding, including real roots or roots on a rectangle edge.
        if (values.size() != 1) throw invalid_argument("Expected one marked root.");
        Num real, imaginary;
        qqbar_re_im(real.n, imaginary.n, values[0].n);
        int index = -1;
        for (Json::ArrayIndex j = 0; j < rectangles.size(); ++j) {
            if (rectangles[j].size() != 4) throw invalid_argument("Expected four rectangle bounds.");
            array<Num,4> bounds;
            for (Json::ArrayIndex k = 0; k < bounds.size(); ++k) {
                fmpq_t q;
                fmpq_init(q);
                bool valid = fmpq_set_str(q, rectangles[j][k].asCString(), 10) == 0;
                if (valid) qqbar_set_fmpq(bounds[k].n, q);
                fmpq_clear(q);
                if (!valid) throw invalid_argument("Invalid root rectangle bound.");
            }
            if (qqbar_cmp_re(real.n, bounds[0].n) >= 0 && qqbar_cmp_re(real.n, bounds[1].n) <= 0 &&
                qqbar_cmp_re(imaginary.n, bounds[2].n) >= 0 && qqbar_cmp_re(imaginary.n, bounds[3].n) <= 0) {
                if (index != -1) throw invalid_argument("The root belongs to several rectangles.");
                index = static_cast<int>(j);
            }
        }
        if (index < 0) throw invalid_argument("The marked root was absent from the isolating rectangles.");
        J result;
        result["root_index"] = index;
        return result;
    }
    Num generator = values.front();
    fmpq_poly_t expression;
    fmpq_poly_init(expression);
    for (size_t i = 1; i < values.size(); ++i) {
        // Only finitely many integers k fail to separate the embeddings of
        // Q(generator, value) by generator + k*value. For a good k, increasing
        // precision finds both field inclusions. FLINT verifies each exactly.
        for (slong round = 1, prec = 128; ; ++round, prec *= 2) {
            bool found = false;
            for (slong k = 0; k <= round; ++k) {
                Num candidate = generator + Num(k)*values[i];
                if (qqbar_express_in_field(expression, candidate.n, values[i].n, 0, 0, prec) &&
                    qqbar_express_in_field(expression, candidate.n, generator.n, 0, 0, prec)) {
                    generator = std::move(candidate);
                    found = true;
                    break;
                }
            }
            if (found) break;
            if (prec > WORD_MAX/2) throw Fail("Number field precision exceeds a signed word.");
        }
    }
    J out;
    out["minimal_polynomial"] = taken(fmpz_poly_get_str_pretty(QQBAR_POLY(generator.n), "T"));
    out["degree"] = Json::Int64(qqbar_degree(generator.n));
    fexpr_t exact;
    fexpr_init(exact);
    qqbar_get_fexpr_root_indexed(exact, generator.n);
    out["exact"] = taken(fexpr_get_str(exact));
    fexpr_clear(exact);
    for (const auto& value : values) {
        for (slong prec = 128; !qqbar_express_in_field(expression, generator.n, value.n, 0, 0, prec); prec *= 2)
            if (prec > WORD_MAX/2) throw Fail("Number field precision exceeds a signed word.");
        out["values"].append(taken(fmpq_poly_get_str_pretty(expression, "T")));
    }
    fmpq_poly_clear(expression);
    return out;
}
int main(){
    J req,out;
    try{
        cin>>req;
        quick=!req.get("exhaustive",false).asBool();
        // Apply the optional quick-mode memory budget only to this helper.
        struct rlimit limit;
        if (quick && getrlimit(RLIMIT_AS,&limit)==0) {
            limit.rlim_cur=min<rlim_t>(limit.rlim_cur,rlim_t(1)<<30);
            if(setrlimit(RLIMIT_AS,&limit)!=0) throw Fail("Could not bound the exact-arithmetic worker.");
        }
        if (req.isMember("ordered_roots")) {
            out=ordered_root_labels(req["ordered_roots"]);
        } else if (req.isMember("field_roots")) {
            out=coefficient_field(req["field_roots"], req["root_rectangles"]);
        } else {
            Resolver solver;solver.names={req["variables"][0].asString(),req["variables"][1].asString()};
            solver.embedded=req["mode"].asString()=="embedded";
            Poly f;
            for(auto& t:req["terms"]){
                Num n;fmpq_t q;fmpq_init(q);
                if(fmpq_set_str(q,t[2].asCString(),10)!=0)throw Fail("Invalid internal coefficient.");
                fmpq_canonicalise(q);qqbar_set_fmpq(n.n,q);fmpq_clear(q);
                auto exponent=[](const J& v)->Exp {
                    if(!v.isUInt64() || v.asUInt64()>static_cast<Json::UInt64>(numeric_limits<Exp>::max()))
                        throw Fail("Plane input exponent exceeds a signed FLINT word.");
                    return static_cast<Exp>(v.asUInt64());
                };
                put(f,{exponent(t[0]),exponent(t[1])},n);
            }
            out=solver.run(f);
        }
    }catch(const invalid_argument& e){out["error"]=e.what();out["kind"]="input";}
    catch(const exception& e){out["error"]=e.what();out["kind"]="limit";}
    Json::StreamWriterBuilder w;w["indentation"]="";
    cout<<Json::writeString(w,out)<<"\n";return 0;
}
