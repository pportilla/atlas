// Jung's method for isolated surface singularities f(x,y,z)=0 at the origin.
//
// 1. A linear change of coordinates, found by a fixed search, makes f regular
//    of order m = ord f in the fibre coordinate t, with the other roots of
//    f(0,0,t) simple, so the discriminant is the branch locus times a unit.
// 2. The discriminant of f with respect to t is the resultant Res_t(f, f_t).
// 3. The plane Resolver (plane_resolver.hpp) resolves the discriminant curve.
// 4. Over every crossing of the total transform, f is quasi-ordinary. Its roots
//    split along a tree of monomials (jung_local.hpp); every orbit of roots is
//    a normalized branch, a cyclic quotient singularity given by its lattice.
// 5. Hirzebruch-Jung strings resolve the quotients. The curves covering each
//    exceptional line get their genera by Riemann-Hurwitz and their
//    self-intersections from the projection formula; the canonical cycle
//    solves the adjunction equations.
#include "jung_local.hpp"
#include <flint/fmpq_mpoly_factor.h>
#include <flint/fmpq_poly.h>
#include <flint/fmpq_mat.h>
#include <flint/gr_mat.h>
#include <chrono>
#include <functional>
#include <optional>

namespace jung {

struct Declined : std::invalid_argument { using std::invalid_argument::invalid_argument; };

// ---------------------------------------------------------------- rational polynomials
struct MPoly {
    fmpq_mpoly_t p; const fmpq_mpoly_ctx_struct* ctx;
    explicit MPoly(const fmpq_mpoly_ctx_t c) : ctx(c) { fmpq_mpoly_init(p,c); }
    MPoly(const MPoly& o) : ctx(o.ctx) { fmpq_mpoly_init(p,ctx); fmpq_mpoly_set(p,o.p,ctx); }
    MPoly& operator=(const MPoly& o) { fmpq_mpoly_set(p,o.p,ctx); return *this; }
    ~MPoly() { fmpq_mpoly_clear(p,ctx); }
    slong length() const { return fmpq_mpoly_length(p,ctx); }
};

struct Term3 { long i,j,k; Num c; };

static string q_string(const fmpq_t q) { return atlas::flint_string(fmpq_get_str(nullptr,10,q)); }

static string mpoly_string(const MPoly& f, const vector<string>& names) {
    vector<const char*> v; for (const auto& s:names) v.push_back(s.c_str());
    return atlas::flint_string(fmpq_mpoly_get_str_pretty(f.p,v.data(),f.ctx));
}

static long total_degree_min(const MPoly& f) {
    long m=-1; ulong e[3];
    for (slong i=0;i<f.length();++i) {
        fmpq_mpoly_get_term_exp_ui(e,f.p,i,f.ctx);
        long d=static_cast<long>(e[0]+e[1]+e[2]);
        if (m<0 || d<m) m=d;
    }
    return m;
}

// Values f_k(d) of the homogeneous components at a direction.
struct QVal { fmpq_t v; QVal() { fmpq_init(v); } QVal(const QVal& o) { fmpq_init(v); fmpq_set(v,o.v); }
              ~QVal() { fmpq_clear(v); } };
static map<long,QVal> components_at(const MPoly& f, const array<long,3>& d) {
    map<long,QVal> out; ulong e[3];
    fmpq_t c; fmpz_t pw; fmpq_init(c); fmpz_init(pw);
    for (slong i=0;i<f.length();++i) {
        fmpq_mpoly_get_term_exp_ui(e,f.p,i,f.ctx);
        fmpq_mpoly_get_term_coeff_fmpq(c,f.p,i,f.ctx);
        for (int a=0;a<3;++a) { fmpz_set_si(pw,d[a]); fmpz_pow_ui(pw,pw,e[a]); fmpq_mul_fmpz(c,c,pw); }
        fmpq_add(out[static_cast<long>(e[0]+e[1]+e[2])].v,out[static_cast<long>(e[0]+e[1]+e[2])].v,c);
    }
    fmpq_clear(c); fmpz_clear(pw);
    for (auto it=out.begin();it!=out.end();) it=fmpq_is_zero(it->second.v)?out.erase(it):next(it);
    return out;
}

// ---------------------------------------------------------------- local charts
static Poly swap_poly(const Poly& f) { Poly g; for (const auto& [k,c]:f) g.emplace(Key{k.second,k.first},c); return g; }

// psi(u) with h(u,psi(u)) = 0, psi(0)=0, where h is smooth and transverse to u=0.
static vector<Num> implicit_branch(const Poly& h, long N) {
    auto it=h.find({0,1});
    if (it==h.end()) throw Fail("Internal error: the discriminant branch is not transverse to its exceptional curve.");
    const Num h01(it->second);
    vector<Num> psi(N+1);
    for (long k=1;k<=N;++k) {
        // coefficient of u^k in h(u, psi(u)) with psi known below degree k
        long jmax=0; for (const auto& [key,c]:h) jmax=max<long>(jmax,key.second);
        vector<vector<Num>> pw{vector<Num>(N+1)}; pw[0][0]=Num(1);
        for (long j=1;j<=jmax;++j) {
            vector<Num> next(N+1);
            for (long a=0;a<=k;++a) if (!pw[j-1][a].zero()) for (long b=1;a+b<=k;++b) if (!psi[b].zero())
                nadd_to(next[a+b],pw[j-1][a]*psi[b]);
            pw.push_back(move(next));
        }
        Num value(0);
        for (const auto& [key,c]:h) if (key.first<=k) nadd_to(value,Num(c)*pw[key.second][k-key.first]);
        psi[k]=nneg(ndiv(value,h01));
    }
    return psi;
}

// Coordinates (p,q) of the base as series in U=u^(1/D), W=w^(1/D).
static pair<Series,Series> local_chart(const Pair& map, const Poly* strict, long D, const Box& B) {
    const Box raw{B.a/D,B.b/D};
    vector<Num> psi;
    if (strict) psi=implicit_branch(*strict,raw.a);
    long jmax=0;
    for (const auto& f:map) for (const auto& [k,c]:f) jmax=max<long>(jmax,k.second);
    // powers of v = w + psi(u) as series in (u,w)
    vector<Series> vp{s_one()};
    Series v; add_term(v,{0,1},Num(1));
    for (size_t k=1;k<psi.size();++k) add_term(v,{static_cast<long>(k),0},psi[k]);
    for (long j=1;j<=jmax;++j) vp.push_back(s_mul(vp.back(),v,raw));
    array<Series,2> out;
    for (int a=0;a<2;++a) {
        Series s;
        for (const auto& [k,c]:map[a]) {
            if (k.first>raw.a) continue;
            for (const auto& [kk,cc]:vp[k.second]) {
                E2 key{k.first+kk.first,kk.second};
                if (in_box(key,raw)) add_term(s,key,Num(c)*cc);
            }
        }
        for (const auto& [k,c]:s) out[a].emplace(E2{k.first*D,k.second*D},c);
    }
    return {out[0],out[1]};
}

struct Local {
    unique_ptr<Node> tree;
    vector<Branch> branches;
    vector<Quotient> quotients;
    Box box;
    Series P,Q;
};

static Local analyse_once(const vector<Term3>& f, long m, long D, const Pair& map, const Poly* strict, const Box& B) {
    Local L; L.box=B;
    tie(L.P,L.Q)=local_chart(map,strict,D,B);
    long pi=0,qj=0;
    for (const auto& t:f) { pi=max(pi,t.i); qj=max(qj,t.j); }
    vector<Series> Pp{s_one()}, Qp{s_one()};
    for (long i=1;i<=pi;++i) Pp.push_back(s_mul(Pp.back(),L.P,B));
    for (long j=1;j<=qj;++j) Qp.push_back(s_mul(Qp.back(),L.Q,B));
    std::map<E2,Series> products;
    TS F;
    for (const auto& t:f) {
        auto it=products.find({t.i,t.j});
        if (it==products.end()) it=products.emplace(E2{t.i,t.j},s_mul(Pp[t.i],Qp[t.j],B)).first;
        UPoly mono(t.k+1); mono[t.k]=t.c;
        ts_add_product(F,mono,{0,0},it->second,B);
    }
    auto [W,unit]=hensel(F,m,B);
    L.tree=split(move(W),m,B);
    L.branches=collect_branches(*L.tree,D);
    for (const auto& br:L.branches) L.quotients.push_back(quotient(br));
    return L;
}

static Local analyse(const vector<Term3>& f, long m, long D, const Pair& map, const Poly* strict, Box B) {
    for (int attempt=0;;++attempt) {
        try { return analyse_once(f,m,D,map,strict,B); }
        catch (const Precision&) {
            if (attempt>=6) throw Fail("The quasi-ordinary roots were not separated within the precision limit.");
            B.a=2*B.a+D; B.b=(B.b?2*B.b+D:0);
        }
    }
}

// Over a generic point of a discriminant branch w=0 the surface must be
// smooth: the roots that agree on w=0 form one orbit with exponent 1/n in w.
static bool smooth_over_branch(const Node& v, long D, long above=0) {
    if (v.n==1) return true;
    const long here=above+v.delta.second;   // w-exponent of the differences split here
    if (here>0) {
        if (here*v.n!=D) return false;
        std::function<bool(const Node&)> flat=[&](const Node& x) {
            for (const auto& k:x.kid) if (k && (k->delta.second!=0 || !flat(*k))) return false;
            return true;
        };
        return flat(v);
    }
    for (const auto& k:v.kid) if (k && !smooth_over_branch(*k,D,here)) return false;
    return true;
}

// ---------------------------------------------------------------- the resolution graph
struct Vertex {
    string id, kind, over;
    long genus=0, self=0;
    long multiplicity=0;   // ramification index (cover) or 0
    long degree=0;         // degree over the exceptional line (cover)
    J extra;
};

struct Graph {
    vector<Vertex> v;
    std::map<pair<size_t,size_t>,long> edges;   // i<j -> number of intersection points
    void edge(size_t a, size_t b) { if (a==b) throw Fail("Internal error: a curve meets itself."); edges[minmax(a,b)]++; }
};


// Canonical cycle K = sum k_i E_i with K.E_j = 2g_j - 2 - E_j^2; K^2 and chi.
static J invariants(const Graph& G, vector<string>* coefficients=nullptr) {
    const slong n=static_cast<slong>(G.v.size());
    J out;
    long chi=0;
    for (const auto& x:G.v) chi+=2-2*x.genus;
    for (const auto& [e,c]:G.edges) chi-=c;
    out["euler_characteristic"]=Json::Int64(chi);
    if (n==0) { out["K2"]="0"; out["canonical_cycle"]=J(Json::arrayValue); out["gorenstein"]=true; return out; }
    fmpq_mat_t M,X,R; fmpq_mat_init(M,n,n); fmpq_mat_init(X,n,1); fmpq_mat_init(R,n,1);
    for (slong i=0;i<n;++i) {
        fmpq_set_si(fmpq_mat_entry(M,i,i),G.v[i].self,1);
        fmpq_set_si(fmpq_mat_entry(R,i,0),2*G.v[i].genus-2-G.v[i].self,1);
    }
    for (const auto& [e,c]:G.edges) {
        fmpq_set_si(fmpq_mat_entry(M,e.first,e.second),c,1);
        fmpq_set_si(fmpq_mat_entry(M,e.second,e.first),c,1);
    }
    if (!fmpq_mat_solve(X,M,R)) { fmpq_mat_clear(M); fmpq_mat_clear(X); fmpq_mat_clear(R);
        throw Fail("Internal error: the intersection matrix is singular."); }
    fmpq_t K2,t; fmpq_init(K2); fmpq_init(t);
    J coeffs(Json::arrayValue);
    bool integral=true;
    for (slong i=0;i<n;++i) {
        fmpq_mul(t,fmpq_mat_entry(X,i,0),fmpq_mat_entry(R,i,0));
        fmpq_add(K2,K2,t);
        coeffs.append(q_string(fmpq_mat_entry(X,i,0)));
        if (!fmpz_is_one(fmpq_denref(fmpq_mat_entry(X,i,0)))) integral=false;
        if (coefficients) coefficients->push_back(q_string(fmpq_mat_entry(X,i,0)));
    }
    out["K2"]=q_string(K2);
    out["canonical_cycle"]=coeffs;
    out["gorenstein"]=integral;
    // Negative definiteness: the exceptional set of a resolution must have one.
    fmpq_mat_t Mm; fmpq_mat_init(Mm,n,n); fmpq_mat_neg(Mm,M);
    bool definite=true;
    for (slong k=1;k<=n && definite;++k) {
        fmpq_mat_t S; fmpq_mat_init(S,k,k);
        for (slong i=0;i<k;++i) for (slong j=0;j<k;++j) fmpq_set(fmpq_mat_entry(S,i,j),fmpq_mat_entry(Mm,i,j));
        fmpq_t d; fmpq_init(d); fmpq_mat_det(d,S);
        if (fmpq_sgn(d)<=0) definite=false;
        fmpq_clear(d); fmpq_mat_clear(S);
    }
    out["negative_definite"]=definite;
    fmpq_mat_clear(Mm);
    fmpq_clear(K2); fmpq_clear(t); fmpq_mat_clear(M); fmpq_mat_clear(X); fmpq_mat_clear(R);
    return out;
}

// Contract smooth rational (-1)-curves while the configuration stays a good
// resolution: at most two intersection points, with distinct curves.
static Graph minimal_good(Graph G, J& contracted) {
    while (true) {
        size_t victim=G.v.size();
        vector<pair<size_t,long>> nb;
        for (size_t i=0;i<G.v.size() && victim==G.v.size();++i) {
            if (G.v[i].genus!=0 || G.v[i].self!=-1) continue;
            vector<pair<size_t,long>> here; long points=0;
            for (const auto& [e,c]:G.edges) {
                if (e.first==i) here.emplace_back(e.second,c);
                else if (e.second==i) here.emplace_back(e.first,c);
                if (e.first==i || e.second==i) points+=c;
            }
            if (points>2) continue;
            if (here.size()==1 && here[0].second>1) continue;
            victim=i; nb=here;
        }
        if (victim==G.v.size()) return G;
        contracted.append(G.v[victim].id);
        for (const auto& [j,c]:nb) G.v[j].self+=c;
        if (nb.size()==2) G.edge(nb[0].first,nb[1].first);
        Graph H;
        vector<size_t> index(G.v.size());
        for (size_t i=0;i<G.v.size();++i) if (i!=victim) { index[i]=H.v.size(); H.v.push_back(G.v[i]); }
        for (const auto& [e,c]:G.edges) {
            if (e.first==victim || e.second==victim) continue;
            H.edges[minmax(index[e.first],index[e.second])]+=c;
        }
        G=move(H);
    }
}

static J graph_json(const Graph& G) {
    J out, vs(Json::arrayValue), es(Json::arrayValue);
    for (const auto& x:G.v) {
        J v=x.extra.isNull()?J(Json::objectValue):x.extra;
        v["id"]=x.id; v["kind"]=x.kind; v["over"]=x.over;
        v["genus"]=Json::Int64(x.genus); v["self_intersection"]=Json::Int64(x.self);
        vs.append(v);
    }
    for (const auto& [e,c]:G.edges) {
        J edge; edge["vertices"]=arr({G.v[e.first].id,G.v[e.second].id}); edge["points"]=Json::Int64(c);
        es.append(edge);
    }
    out["vertices"]=vs; out["edges"]=es;
    return out;
}

// ---------------------------------------------------------------- the method
struct Special {
    string kind;              // "corner" or "branch"
    int a=0, b=0;             // exceptional ids of u=0 and w=0 (b=0 on a branch)
    size_t chart=0;
    Coord point;
    bool swap=false;          // the exceptional axis of a branch point is v=0
    Poly strict;              // straightened discriminant branch (branch points)
    Local local;
    // Polynomials in the chart's coordinates, moved to the local coordinates.
    Pair localize(Pair map) const {
        for (int a=0;a<2;++a) if (!point[a].zero()) for (auto& g:map) g=shift(g,a,point[a]);
        if (swap) for (auto& g:map) g=swap_poly(g);
        return map;
    }
    const Poly* branch() const { return kind=="branch"?&strict:nullptr; }
};

using CPoly = map<Key,Num>;
template<class N> static void cput(map<Key,N>& p, Key k, const N& c) {
    if (c.zero()) return;
    auto [it,created]=p.try_emplace(k,c);
    if (!created) { it->second=it->second+c; if (it->second.zero()) p.erase(it); }
}
template<class N> static map<Key,N> poly_mul(const map<Key,N>& a, const map<Key,N>& b) {
    map<Key,N> r;
    for (const auto& [ka,ca]:a) for (const auto& [kb,cb]:b) cput(r,{ka.first+kb.first,ka.second+kb.second},ca*cb);
    return r;
}

// Coordinates of an ancestor chart as polynomials in a chart's coordinates.
static Pair coordinates_in(const Resolver& plane, size_t c, size_t ancestor) {
    Pair X{Poly{{{1,0},PNum(1)}},Poly{{{0,1},PNum(1)}}};
    size_t cur=c;
    while (cur!=ancestor) {
        const auto& ch=plane.chart_data[cur];
        if (ch.parent<0) throw Fail("Internal error: a chart is not below the creation chart of its curve.");
        Pair P;
        const Poly uv=poly_mul(X[0],X[1]);
        if (ch.pivot==0) { P[0]=X[0]; P[1]=uv; } else { P[0]=uv; P[1]=X[1]; }
        for (int a=0;a<2;++a) if (!ch.center[a].zero()) put(P[a],{0,0},ch.center[a]);
        X=move(P); cur=static_cast<size_t>(ch.parent);
    }
    return X;
}

static bool below(const Resolver& plane, size_t c, size_t ancestor) {
    for (long cur=static_cast<long>(c); cur>=0; cur=plane.chart_data[cur].parent)
        if (static_cast<size_t>(cur)==ancestor) return true;
    return false;
}

static long binomial(long n, long k) { long r=1; for (long i=1;i<=k;++i) r=r*(n-k+i)/i; return r; }

struct Component {
    int curve=0; long e=1, f=0;
    vector<array<size_t,3>> points;   // (special, local point, side)
};

static J run(const J& req) {
    const auto started=std::chrono::steady_clock::now();
    if (!req["variables"].isArray() || req["variables"].size()!=3)
        throw invalid_argument("Jung's method needs a surface in three variables.");
    vector<string> names;
    for (const auto& v:req["variables"]) names.push_back(v.asString());
    fmpq_mpoly_ctx_t ctx; fmpq_mpoly_ctx_init(ctx,3,ORD_LEX);
    J record;
    {
    MPoly f(ctx);
    for (const auto& t:req["terms"]) {
        if (!t.isArray() || t.size()!=4) throw invalid_argument("Expected three exponents and a coefficient per term.");
        ulong e[3];
        for (int a=0;a<3;++a) {
            if (!t[a].isUInt64() || t[a].asUInt64()>(1u<<20)) throw invalid_argument("Exponent out of range for Jung's method.");
            e[a]=t[a].asUInt64();
        }
        fmpq_t c; fmpq_init(c);
        if (fmpq_set_str(c,t[3].asCString(),10)) { fmpq_clear(c); throw invalid_argument("Invalid rational coefficient."); }
        fmpq_canonicalise(c);
        MPoly term(ctx); fmpq_mpoly_set_coeff_fmpq_ui(term.p,c,e,ctx); fmpq_mpoly_add(f.p,f.p,term.p,ctx);
        fmpq_clear(c);
    }
    if (fmpq_mpoly_is_zero(f.p,ctx)) throw invalid_argument("The zero polynomial does not define a surface.");
    // Factors that do not vanish at the origin are units of the local ring.
    {
        fmpq_mpoly_factor_t fac; fmpq_mpoly_factor_init(fac,ctx);
        if (!fmpq_mpoly_factor(fac,f.p,ctx)) throw Fail("Rational factorization failed.");
        MPoly germ(ctx); fmpq_mpoly_one(germ.p,ctx);
        long through=0;
        for (slong i=0;i<fmpq_mpoly_factor_length(fac,ctx);++i) {
            MPoly g(ctx); fmpq_mpoly_factor_get_base(g.p,fac,i,ctx);
            fmpq_t c0; fmpq_init(c0); ulong z[3]={0,0,0}; fmpq_mpoly_get_coeff_fmpq_ui(c0,g.p,z,ctx);
            const bool vanishes=fmpq_is_zero(c0); fmpq_clear(c0);
            if (!vanishes) continue;
            const slong ex=fmpq_mpoly_factor_get_exp_si(fac,i,ctx);
            if (ex>1) { fmpq_mpoly_factor_clear(fac,ctx);
                throw Declined("The surface germ is nonreduced: a repeated factor vanishes at the origin, so the singularity is not isolated."); }
            ++through; fmpq_mpoly_mul(germ.p,germ.p,g.p,ctx);
        }
        fmpq_mpoly_factor_clear(fac,ctx);
        if (through==0) throw invalid_argument("The origin does not lie on the surface.");
        if (through>1) throw Declined("The surface germ is reducible at the origin; its components meet along a curve, so the singularity is not isolated.");
        f=germ;
    }
    const long m=total_degree_min(f);
    record["input"]=mpoly_string(f,names);
    record["order"]=Json::Int64(m);
    if (m==1) {
        record["status"]="smooth";
        record["summary"]="The surface is smooth at the origin; no exceptional curves.";
        J g; g["vertices"]=J(Json::arrayValue); g["edges"]=J(Json::arrayValue);
        record["graph"]=g; record["minimal_good_resolution"]=g;
        record["invariants"]=invariants(Graph{});
        record["invariants"]["geometric_genus"]=Json::Int64(0);
        fmpq_mpoly_ctx_clear(ctx);
        return record;
    }

    // ---- 1. projection: fibre direction d with f_m(d) != 0, minimal degree in t,
    // and the remaining roots of f(t d) simple, then fewest terms.
    vector<array<long,3>> directions{{0,0,1},{0,1,0},{1,0,0}};
    for (long bound=1;bound<=2;++bound) {
        vector<array<long,3>> layer;
        for (long a=-bound;a<=bound;++a) for (long b=-bound;b<=bound;++b) for (long c=-bound;c<=bound;++c) {
            array<long,3> d{a,b,c};
            long mx=max({labs(a),labs(b),labs(c)});
            if (mx!=bound) continue;
            long nz=(a!=0)+(b!=0)+(c!=0); if (nz<2 && bound==1) continue;
            // first nonzero entry positive, primitive
            long first=a?a:(b?b:c); if (first<0) continue;
            if (std::gcd(std::gcd(labs(a),labs(b)),labs(c))!=1) continue;
            layer.push_back(d);
        }
        stable_sort(layer.begin(),layer.end(),[](const auto& x,const auto& y){
            long sx=labs(x[0])+labs(x[1])+labs(x[2]), sy=labs(y[0])+labs(y[1])+labs(y[2]);
            return sx<sy;
        });
        directions.insert(directions.end(),layer.begin(),layer.end());
    }
    struct Choice { array<long,3> d; long degree; slong terms; int fibre; };
    std::optional<Choice> best;
    fmpq_mpoly_ctx_t nctx; fmpq_mpoly_ctx_init(nctx,3,ORD_LEX);   // variables (p, q, t)
    std::unique_ptr<MPoly> ftilde;
    auto transform=[&](const array<long,3>& d, int fibre) {
        // x_fibre = d_fibre t, x_j = p + d_j t, x_k = q + d_k t
        vector<int> base; for (int a=0;a<3;++a) if (a!=fibre) base.push_back(a);
        fmpq_mpoly_struct* images[3]; vector<unique_ptr<MPoly>> keep;
        for (int a=0;a<3;++a) {
            keep.push_back(make_unique<MPoly>(nctx));
            MPoly& g=*keep.back();
            fmpq_mpoly_gen(g.p,2,nctx); fmpq_mpoly_scalar_mul_si(g.p,g.p,d[a],nctx);
            if (a!=fibre) { MPoly x(nctx); fmpq_mpoly_gen(x.p,a==base[0]?0:1,nctx); fmpq_mpoly_add(g.p,g.p,x.p,nctx); }
            images[a]=g.p;
        }
        auto out=make_unique<MPoly>(nctx);
        if (!fmpq_mpoly_compose_fmpq_mpoly(out->p,f.p,images,ctx,nctx)) throw Fail("Coordinate change failed.");
        return out;
    };
    for (const auto& d:directions) {
        auto comps=components_at(f,d);
        auto it=comps.find(m);
        if (it==comps.end()) continue;
        long degree=comps.rbegin()->first;
        fmpq_poly_t q0; fmpq_poly_init(q0);
        for (const auto& [k,v]:comps) fmpq_poly_set_coeff_fmpq(q0,k-m,v.v);
        const bool simple=fmpq_poly_is_squarefree(q0);
        fmpq_poly_clear(q0);
        if (!simple) continue;
        int fibre=2; while (d[fibre]==0) --fibre;
        if (best && degree>best->degree) continue;
        auto g=transform(d,fibre);
        if (!best || degree<best->degree || g->length()<best->terms) {
            best=Choice{d,degree,g->length(),fibre};
            ftilde=move(g);
        }
    }
    if (!best) throw Declined("No projection direction with small integer entries makes the surface regular of its order with simple remaining roots.");
    const auto d=best->d; const int fibre=best->fibre;
    vector<int> base; for (int a=0;a<3;++a) if (a!=fibre) base.push_back(a);
    const bool permutation=(d[base[0]]==0 && d[base[1]]==0 && d[fibre]==1);
    vector<string> nnames{names[base[0]],names[base[1]],names[fibre]};
    if (!permutation) for (auto& s:nnames) s+="'";
    {
        J pr; pr["fibre_direction"]=arr({Json::Int64(d[0]),Json::Int64(d[1]),Json::Int64(d[2])});
        pr["coordinates"]=arr({nnames[0],nnames[1],nnames[2]});
        J sub(Json::objectValue);
        for (int a=0;a<3;++a) {
            string s;
            auto piece=[&](long c,const string& v){
                if (!c) return;
                string w=(c==1?"":c==-1?"-":to_string(c)+"*")+v;
                if (!s.empty() && w[0]!='-') s+="+";
                s+=w;
            };
            if (a!=fibre) piece(1,nnames[a==base[0]?0:1]);
            piece(d[a],nnames[2]);
            sub[names[a]]=s;
        }
        pr["substitution"]=sub;
        pr["degree_in_fibre"]=Json::Int64(best->degree);
        pr["equation"]=mpoly_string(*ftilde,nnames);
        record["projection"]=pr;
    }

    // ---- 2. discriminant
    MPoly ft(nctx), disc(nctx);
    fmpq_mpoly_derivative(ft.p,ftilde->p,2,nctx);
    if (!fmpq_mpoly_resultant(disc.p,ftilde->p,ft.p,2,nctx)) throw Fail("The discriminant resultant failed.");
    if (fmpq_mpoly_is_zero(disc.p,nctx)) throw Declined("The discriminant vanishes identically: the surface is not reduced.");
    MPoly dred(nctx); fmpq_mpoly_one(dred.p,nctx);
    long nmax=1;
    J dfactors(Json::arrayValue);
    {
        fmpq_mpoly_factor_t fac; fmpq_mpoly_factor_init(fac,nctx);
        if (!fmpq_mpoly_factor(fac,disc.p,nctx)) throw Fail("Discriminant factorization failed.");
        for (slong i=0;i<fmpq_mpoly_factor_length(fac,nctx);++i) {
            MPoly g(nctx); fmpq_mpoly_factor_get_base(g.p,fac,i,nctx);
            fmpq_t c0; fmpq_init(c0); ulong z[3]={0,0,0}; fmpq_mpoly_get_coeff_fmpq_ui(c0,g.p,z,nctx);
            const bool vanishes=fmpq_is_zero(c0); fmpq_clear(c0);
            if (!vanishes) continue;
            const slong ex=fmpq_mpoly_factor_get_exp_si(fac,i,nctx);
            nmax=max<long>(nmax,ex);
            fmpq_mpoly_mul(dred.p,dred.p,g.p,nctx);
            J fj; fj["factor"]=mpoly_string(g,nnames); fj["multiplicity"]=Json::Int64(ex); dfactors.append(fj);
        }
        fmpq_mpoly_factor_clear(fac,nctx);
    }
    {
        J dj; dj["factors_through_origin"]=dfactors;
        dj["terms"]=Json::Int64(fmpq_mpoly_length(disc.p,nctx));
        record["discriminant"]=dj;
    }

    // ---- 3. embedded resolution of the reduced discriminant germ
    Poly delta;
    {
        ulong e[3]; fmpq_t c; fmpq_init(c);
        for (slong i=0;i<fmpq_mpoly_length(dred.p,nctx);++i) {
            fmpq_mpoly_get_term_exp_ui(e,dred.p,i,nctx);
            fmpq_mpoly_get_term_coeff_fmpq(c,dred.p,i,nctx);
            if (e[2]) throw Fail("Internal error: the discriminant involves the fibre coordinate.");
            put(delta,{static_cast<Exp>(e[0]),static_cast<Exp>(e[1])},from_fmpq(c).num());
        }
        fmpq_clear(c);
    }
    if (order(delta)<2) throw Declined("The discriminant is smooth at the origin, which happens only for a smooth or non-normal surface.");
    quick=false;
    Resolver plane; plane.names={nnames[0],nnames[1]}; plane.embedded=true; plane.keep_charts=true;
    J plane_record=plane.run(delta);
    const int curves=static_cast<int>(plane.weights.size())-1;
    {
        J pj; pj["blowups"]=Json::Int64(plane.steps.size());
        pj["summary"]=plane_record["summary"];
        J es(Json::arrayValue);
        for (int e=1;e<=curves;++e) {
            J v; v["id"]="E"+to_string(e); v["self_intersection"]=plane.weights[e];
            v["discriminant_multiplicity"]=Json::Int64(plane.mults[e]); es.append(v);
        }
        pj["curves"]=es;
        pj["record"]=plane_record;
        record["discriminant_resolution"]=pj;
    }

    // ---- 4. quasi-ordinary analysis at every crossing
    long D=1; for (long k=2;k<=m;++k) D=std::lcm(D,k);
    vector<Term3> terms;
    {
        ulong e[3]; fmpq_t c; fmpq_init(c);
        for (slong i=0;i<ftilde->length();++i) {
            fmpq_mpoly_get_term_exp_ui(e,ftilde->p,i,nctx);
            fmpq_mpoly_get_term_coeff_fmpq(c,ftilde->p,i,nctx);
            terms.push_back({static_cast<long>(e[0]),static_cast<long>(e[1]),static_cast<long>(e[2]),from_fmpq(c)});
        }
        fmpq_clear(c);
    }
    // A first guess; every node certifies its precision and the box doubles when it does not.
    auto box_for=[&](long alpha,long beta){ return Box{D*(alpha+2),beta<0?0:D*(beta+2)}; };
    vector<Special> specials;
    for (size_t c=0;c<plane.chart_data.size();++c) {
        const auto& ch=plane.chart_data[c];
        if (!ch.axes[0] || !ch.axes[1]) continue;
        bool excluded=false;
        for (const auto& pt:ch.excluded) if (pt[0].zero() && pt[1].zero()) excluded=true;
        if (excluded) continue;
        Special s; s.kind="corner"; s.a=ch.axes[0]; s.b=ch.axes[1]; s.chart=c; s.point={PNum(0),PNum(0)};
        s.local=analyse(terms,m,D,ch.map,nullptr,box_for(plane.mults[s.a],plane.mults[s.b]));
        specials.push_back(move(s));
    }
    {
        set<pair<int,int>> found;
        for (const auto& s:specials) found.insert(minmax(s.a,s.b));
        if (found!=plane.edges || found.size()!=specials.size())
            throw Fail("Internal error: the crossing charts do not match the dual graph of the discriminant resolution.");
    }
    for (const auto& br:plane.branch_data) {
        const auto& ch=plane.chart_data[br.chart];
        const int i=(ch.axes[0]==br.axis)?0:1;
        if (ch.axes[i]!=br.axis || (ch.axes[1-i] && br.point[1-i].zero())) throw Fail("Internal error: a discriminant branch meets a crossing.");
        Special s; s.kind="branch"; s.a=br.axis; s.b=0; s.chart=br.chart; s.point=br.point; s.swap=(i==1);
        s.strict=s.localize(Pair{ch.strict,Poly{}})[0];
        s.local=analyse(terms,m,D,s.localize(ch.map),&s.strict,box_for(plane.mults[s.a],1));
        if (!smooth_over_branch(*s.local.tree,D))
            throw Declined("The surface is singular along a curve over a branch of the discriminant, so the singularity is not isolated.");
        specials.push_back(move(s));
    }

    // Generic points of the exceptional lines: the sheets of the covers.
    vector<std::map<long,long>> sheets(curves+1);   // e -> number of sheets
    vector<Local> generic_local(curves+1);
    vector<long> generic_v(curves+1,0);
    J generic(Json::arrayValue);
    for (int e=1;e<=curves;++e) {
        size_t c=0; while (c<plane.chart_data.size() && plane.chart_data[c].id!="B"+to_string(e)+".u") ++c;
        if (c==plane.chart_data.size()) throw Fail("Internal error: missing creation chart.");
        const auto& ch=plane.chart_data[c];
        long vstar=0;
        for (long cand=1;;cand=(cand>0?-cand:1-cand)) {
            Num val(0), x(cand);
            for (const auto& [k,cf]:ch.strict) if (k.first==0) nadd_to(val,Num(cf)*npow(x,k.second));
            if (!val.zero()) { vstar=cand; break; }
        }
        Pair map=ch.map; for (auto& g:map) g=shift(g,1,PNum(vstar));
        Local L=analyse(terms,m,D,map,nullptr,box_for(plane.mults[e],-1));
        J gj; gj["curve"]="E"+to_string(e); gj["chart"]=ch.id; gj["v"]=Json::Int64(vstar);
        J sh(Json::arrayValue);
        for (const auto& q:L.quotients) {
            if (q.k2!=1) throw Fail("Internal error: a generic sheet is ramified along the base.");
            sheets[e][q.k1]++;
            sh.append(Json::Int64(q.k1));
        }
        gj["sheet_ramification"]=sh;
        generic.append(gj);
        generic_local[e]=move(L); generic_v[e]=vstar;
    }

    // ---- 5. covers of the exceptional lines and their components
    vector<vector<array<size_t,2>>> on(curves+1);   // (special, side)
    for (size_t s=0;s<specials.size();++s) {
        on[specials[s].a].push_back({s,0});
        if (specials[s].b) on[specials[s].b].push_back({s,1});
    }
    long global_tests=0;
    vector<Component> comps;
    std::map<array<size_t,3>,size_t> owner;
    // Components of the cover over E_e that local data leave open. At a
    // generic point (0,v*) of the creation chart the sheets are orbits of
    // roots. A set S of sheets is a union of components exactly when the
    // product P_S of its root factors is single valued near E_e; its
    // coefficient of u^j is then a polynomial of degree <= j in v. The Taylor
    // coefficients at v* of degrees j+1..K vanish for such S. For other S, a
    // nonzero norm of degree < K forbids that (K bounds the degree, counting
    // the poles at points of E_e that were blown up). The sheets of S stay
    // apart from the others below u^T_S, so the coefficients up to u^T_S
    // decide. The exact factor, pulled back to every crossing, vanishes at a
    // local root exactly when that local point lies on the component.
    auto global_components=[&](int e, long eps) {
        size_t cu=plane.chart_data.size(), cv=cu;
        for (size_t c=0;c<plane.chart_data.size();++c) {
            if (plane.chart_data[c].id=="B"+to_string(e)+".u") cu=c;
            if (plane.chart_data[c].id=="B"+to_string(e)+".v") cv=c;
        }
        // Blowups centred on the strict transform of E_e bound the poles that
        // conjugate coefficients can have at the points of E_e.
        long Ne=0;
        for (size_t c=0;c<plane.chart_data.size();++c) {
            const auto& ch=plane.chart_data[c];
            if (ch.pivot!=0 || ch.parent<0) continue;
            const auto& par=plane.chart_data[ch.parent];
            for (int k=0;k<2;++k) if (par.axes[k]==e && ch.center[k].zero()) ++Ne;
        }
        auto sheets_of=[&](const Local& L, vector<Path>& roots, vector<vector<size_t>>& members) {
            roots.clear(); members.clear();
            leaf_paths(*L.tree,{},roots);
            vector<long> sheet(roots.size(),-1);
            for (size_t i=0;i<roots.size();++i) {
                if (sheet[i]>=0) continue;
                members.push_back({});
                Path p=roots[i];
                while (true) {
                    size_t j=0; while (j<roots.size() && roots[j]!=p) ++j;
                    if (j==roots.size()) throw Fail("Internal error: lost a root under monodromy.");
                    if (sheet[j]>=0) break;
                    sheet[j]=static_cast<long>(members.size()-1); members.back().push_back(j);
                    p=monodromy(p,0,D);
                }
            }
        };
        vector<Path> roots; vector<vector<size_t>> members;
        sheets_of(generic_local[e],roots,members);
        vector<size_t> cls;
        for (size_t k=0;k<members.size();++k) if (static_cast<long>(members[k].size())==eps) cls.push_back(k);
        const size_t f=cls.size();
        if (f>20) throw Declined("Too many sheets over E"+to_string(e)+" for the global monodromy test.");
        auto separation=[&](unsigned long mask) {
            vector<bool> in(roots.size(),false);
            for (size_t b=0;b<f;++b) if (mask>>b&1) for (size_t r:members[cls[b]]) in[r]=true;
            long T=0;
            for (size_t r=0;r<roots.size();++r) {
                if (in[r]) continue;
                long sum=0;
                for (size_t q=0;q<roots.size();++q) if (in[q]) sum+=contact(roots[r],roots[q]).first;
                T=max(T,(sum+D-1)/D);
            }
            return T;
        };
        const Pair gmap=[&]{ Pair mp=plane.chart_data[cu].map; for (auto& g:mp) g=shift(g,1,PNum(generic_v[e])); return mp; }();
        // Roots at the generic point, recomputed when a test needs more precision.
        Box have{0,0};
        vector<Series> value;
        Local H;     // owns the nodes that the root paths point to
        auto ensure=[&](long T, long K) {
            if (have.a>=D*T && have.b>=D*K) return;
            const Box target{max(have.a,D*T),max(have.b,D*K)};
            const Box need=needed_box(*generic_local[e].tree,target);
            H=analyse(terms,m,D,gmap,nullptr,Box{need.a+D,need.b});
            vector<Path> hr; vector<vector<size_t>> hm;
            sheets_of(H,hr,hm);
            if (hr.size()!=roots.size()) throw Fail("Internal error: the generic roots changed with precision.");
            value.assign(hr.size(),Series());
            for (size_t b=0;b<f;++b) for (size_t r:hm[cls[b]]) {
                auto [y,known]=path_root(hr[r],&target);
                if (known.a<target.a || known.b<target.b) throw Fail("Internal error: generic roots lack precision (known "+
                    to_string(known.a)+","+to_string(known.b)+" target "+to_string(target.a)+","+to_string(target.b)+
                    " box "+to_string(H.box.a)+","+to_string(H.box.b)+" need "+to_string(need.a)+","+to_string(need.b)+").");
                value[r]=move(y);
            }
            roots=hr; members=hm; have=target;
        };
        vector<Component> found;
        vector<size_t> remaining(f); std::iota(remaining.begin(),remaining.end(),0);
        while (!remaining.empty()) {
            unsigned long chosen=0; vector<Series> factor;
            long chosenT=0;
            const size_t R=remaining.size();
            // The complement of an invariant set in the (invariant) remaining
            // sheets is invariant, so some curve has at most half of them. A
            // set accepted at the smallest size is a single curve.
            for (size_t size=1;2*size<=R && !chosen;++size) {
                vector<unsigned long> masks;
                long Tsize=0;
                const long d=binomial(f,size);
                for (unsigned long sub=0;sub<(1ul<<R);++sub) {
                    if (static_cast<size_t>(__builtin_popcountl(sub))!=size) continue;
                    unsigned long mask=0;
                    for (size_t b=0;b<R;++b) if (sub>>b&1) mask|=1ul<<remaining[b];
                    masks.push_back(mask); Tsize=max(Tsize,separation(mask));
                }
                // Product of the root factors of S and the first Taylor
                // coefficient of degree > j (up to K) that does not vanish.
                auto product=[&](unsigned long mask, long T, long K) {
                    const Box B{D*T,D*K};
                    vector<Series> P{s_one()};    // prod (z - root)
                    for (size_t b=0;b<f;++b) if (mask>>b&1) for (size_t r:members[cls[b]]) {
                        vector<Series> Q(P.size()+1);
                        const Series root=s_truncate(value[r],B);
                        for (size_t l=0;l<P.size();++l) {
                            Q[l+1]=s_add(Q[l+1],P[l]);
                            Q[l]=s_add(Q[l],s_scale(s_mul(P[l],root,B),Num(-1)));
                        }
                        P=move(Q);
                    }
                    return P;
                };
                auto single_valued=[&](const vector<Series>& P, long bound) {
                    for (const auto& coeff:P) for (const auto& [k,c]:coeff) {
                        if (k.first%D || k.second%D) return false;
                        const long j=k.first/D, deg=k.second/D;
                        if (deg>j && deg<=bound*j) return false;
                    }
                    return true;
                };
                // A cheap pass first: a nonzero coefficient a few degrees above
                // j already proves that S is not a union of curves.
                ensure(Tsize,Tsize+4);
                vector<unsigned long> survivors;
                for (unsigned long mask:masks) {
                    const long T=separation(mask);
                    if (single_valued(product(mask,T,T+4),1+4)) survivors.push_back(mask);
                }
                if (survivors.empty()) continue;
                ensure(Tsize,d*Tsize*(1+Ne)+1);
                for (unsigned long mask:survivors) {
                    const long T=separation(mask), K=d*T*(1+Ne);
                    auto P=product(mask,T,K);
                    if (single_valued(P,d*(1+Ne))) { chosen=mask; factor=move(P); chosenT=T; break; }
                }
            }
            Component C; C.curve=e; C.e=eps;
            if (!chosen) {   // no invariant part of at most half: the rest is one component
                for (size_t b:remaining) chosen|=1ul<<b;
                chosenT=-1;
            }
            C.f=__builtin_popcountl(chosen);
            vector<size_t> rest;
            for (size_t b:remaining) if (!(chosen>>b&1)) rest.push_back(b);
            auto taken=[&](const array<size_t,3>& pt) {
                for (const auto& F:found) for (const auto& q:F.points) if (q==pt) return true;
                return false;
            };
            if (chosenT<0) {
                for (const auto& [s2,side]:on[e]) {
                    const auto& qs=specials[s2].local.quotients;
                    for (size_t p=0;p<qs.size();++p)
                        if ((side?qs[p].k2:qs[p].k1)==eps && !taken({s2,p,side})) C.points.push_back({s2,p,side});
                }
            } else {
                // The exact factor: coefficients are polynomials in (u,v) of the creation chart.
                vector<CPoly> A(factor.size());
                const CPoly linear{{{0,1},Num(1)},{{0,0},Num(-generic_v[e])}};
                for (size_t l=0;l<factor.size();++l) for (const auto& [k,c]:factor[l]) {
                    const long j=k.first/D, deg=k.second/D;
                    if (j>chosenT || deg>j) continue;
                    CPoly term{{{j,0},c}};
                    for (long r=0;r<deg;++r) term=poly_mul(term,linear);
                    for (const auto& [kk,cc]:term) cput(A[l],kk,cc);
                }
                for (const auto& [s2,side]:on[e]) {
                    Special& S=specials[s2];
                    const size_t anc=below(plane,S.chart,cu)?cu:cv;
                    if (!below(plane,S.chart,anc)) throw Fail("Internal error: a crossing of E"+to_string(e)+" is outside its charts.");
                    vector<CPoly> Al=A;
                    if (anc==cv) for (auto& g:Al) { CPoly h; for (const auto& [kk,cc]:g) cput(h,{kk.first-kk.second,kk.first},cc); g=move(h); }
                    // Precision: D*T along E_e; across, the w-order of W'(root)
                    // bounds the leading term of the value at a point off C.
                    vector<Path> lroots; leaf_paths(*S.local.tree,{},lroots);
                    long across=D;
                    for (const auto& L1:lroots) {
                        long sum=0;
                        for (const auto& L2:lroots) if (L1!=L2) { const E2 c=contact(L1,L2); sum+=side?c.first:c.second; }
                        across=max(across,sum+D);
                    }
                    const Box R=side?Box{across,D*chosenT}:Box{D*chosenT,across};
                    const Box need=needed_box(*S.local.tree,R);
                    if (need.a>S.local.box.a || need.b>S.local.box.b)
                        S.local=analyse(terms,m,D,S.localize(plane.chart_data[S.chart].map),S.branch(),
                                        Box{max(need.a,S.local.box.a)+D,max(need.b,S.local.box.b)+D});
                    const auto& qs=S.local.quotients;
                    auto [Ue,Ve]=local_chart(S.localize(coordinates_in(plane,S.chart,anc)),S.branch(),D,R);
                    long pu=0,pv=0;
                    for (const auto& g:Al) for (const auto& [kk,cc]:g) { pu=max(pu,kk.first); pv=max(pv,kk.second); }
                    vector<Series> Up{s_one()}, Vp{s_one()};
                    for (long r=1;r<=pu;++r) Up.push_back(s_mul(Up.back(),Ue,R));
                    for (long r=1;r<=pv;++r) Vp.push_back(s_mul(Vp.back(),Ve,R));
                    vector<Series> Acoef(Al.size());
                    for (size_t l=0;l<Al.size();++l) for (const auto& [kk,cc]:Al[l])
                        Acoef[l]=s_add(Acoef[l],s_scale(s_mul(Up[kk.first],Vp[kk.second],R),cc));
                    long degree=0;
                    for (size_t p=0;p<qs.size();++p) {
                        if ((side?qs[p].k2:qs[p].k1)!=eps) continue;
                        auto [zeta,known]=path_root(S.local.branches[p].path,&R);
                        if (known.a<R.a || known.b<R.b) throw Fail("Internal error: local roots lack precision.");
                        Series val=Acoef.back();
                        for (size_t l=Acoef.size()-1;l-->0;) val=s_add(s_mul(val,zeta,R),Acoef[l]);
                        bool vanishes=true;
                        for (const auto& [kk,cc]:val) if ((side?kk.second:kk.first)<=D*chosenT) vanishes=false;
                        if (vanishes) { C.points.push_back({s2,p,side}); degree+=side?qs[p].r2:qs[p].r1; }
                    }
                    if (degree!=C.f) throw Fail("Internal error: the component over E"+to_string(e)+" does not meet a crossing in its degree.");
                }
            }
            found.push_back(move(C));
            remaining=rest;
        }
        return found;
    };
    // A cyclic cover t^m + g(p,q): the group of m-th roots of unity acts on
    // the roots, transitively and compatibly with every local monodromy. The
    // branches at a crossing p form one orbit Z/l_p (l_p branches) and the
    // curves over E form Z/c_E with c_E = gcd of l_p over the crossings of E;
    // the branch with label g lies on the curve g mod c_E. Labels start at an
    // arbitrary branch at each crossing: since the dual graph of the plane
    // resolution is a tree, every choice gives the same graph.
    bool cyclic=(best->degree==m);
    for (const auto& t:terms) if (t.k!=0 && t.k!=m) cyclic=false;
    std::map<size_t,vector<long>> group_labels;   // special -> label of each branch
    auto branch_labels=[&](size_t s2)->const vector<long>& {
        auto it=group_labels.find(s2);
        if (it!=group_labels.end()) return it->second;
        const auto& brs=specials[s2].local.branches;
        vector<long> lab(brs.size(),-1);
        const Num z=croot_of_unity(1,static_cast<ulong>(m));
        size_t b=0; long g=0;
        while (lab[b]<0) {
            lab[b]=g++;
            const Path image=scale_path(brs[b].path,z);
            size_t next=brs.size();
            for (size_t c=0;c<brs.size() && next==brs.size();++c)
                for (const auto& mem:brs[c].members) if (mem==image) { next=c; break; }
            if (next==brs.size()) throw Fail("Internal error: a branch is missing under the covering group.");
            b=next;
        }
        for (long l:lab) if (l<0) throw Fail("Internal error: the covering group is not transitive on the branches.");
        return group_labels.emplace(s2,move(lab)).first->second;
    };
    for (int e=1;e<=curves;++e) {
        if (cyclic && sheets[e].size()==1) {
            const long eps=sheets[e].begin()->first, count=sheets[e].begin()->second;
            long cE=0;
            for (const auto& [s2,side]:on[e]) cE=std::gcd(cE,static_cast<long>(specials[s2].local.branches.size()));
            if (cE<1 || count%cE) throw Fail("Internal error: the cyclic cover over E"+to_string(e)+" has inconsistent orbits.");
            const size_t first=comps.size();
            for (long k=0;k<cE;++k) { Component C; C.curve=e; C.e=eps; C.f=count/cE; comps.push_back(C); }
            for (const auto& [s2,side]:on[e]) {
                const auto& lab=branch_labels(s2);
                const auto& qs=specials[s2].local.quotients;
                for (size_t p=0;p<qs.size();++p) {
                    if ((side?qs[p].k2:qs[p].k1)!=eps) throw Fail("Internal error: conjugate branches with different ramification.");
                    comps[first+lab[p]%cE].points.push_back({s2,p,side});
                }
            }
            for (size_t ci=first;ci<comps.size();++ci) for (const auto& pt:comps[ci].points) owner[pt]=ci;
            continue;
        }
        for (const auto& [eps,count]:sheets[e]) {
            vector<vector<pair<size_t,long>>> at;   // per special point on e: (local point, r)
            vector<size_t> sides;
            bool connected=(count==1), trivial=true;
            for (const auto& [s,side]:on[e]) {
                vector<pair<size_t,long>> here; long total=0;
                const auto& qs=specials[s].local.quotients;
                for (size_t p=0;p<qs.size();++p) {
                    const long k=side?qs[p].k2:qs[p].k1, r=side?qs[p].r2:qs[p].r1;
                    if (k!=eps) continue;
                    here.emplace_back(p,r); total+=r;
                    if (r!=1) trivial=false;
                }
                if (total!=count) throw Fail("Internal error: the local covering degrees over E"+to_string(e)+" do not match its generic sheets.");
                if (here.size()==1) connected=true;
                at.push_back(here); sides.push_back(side);
            }
            if (connected) {
                Component C; C.curve=e; C.e=eps; C.f=count;
                for (size_t i=0;i<on[e].size();++i) for (const auto& [p,r]:at[i]) C.points.push_back({on[e][i][0],p,on[e][i][1]});
                comps.push_back(C);
            } else if (trivial && m==2) {
                // The double cover is Galois: the two sections are exchanged
                // by the covering involution together with their points.
                for (long k=0;k<count;++k) {
                    Component C; C.curve=e; C.e=eps; C.f=1;
                    for (size_t i=0;i<on[e].size();++i) C.points.push_back({on[e][i][0],at[i][k].first,on[e][i][1]});
                    comps.push_back(C);
                }
            } else if (on[e].size()<=1 && trivial) {
                // Sections through one crossing: every assignment gives the same graph.
                for (long k=0;k<count;++k) {
                    Component C; C.curve=e; C.e=eps; C.f=1;
                    for (size_t i=0;i<on[e].size();++i) C.points.push_back({on[e][i][0],at[i][k].first,on[e][i][1]});
                    comps.push_back(C);
                }
            } else {
                for (auto& C:global_components(e,eps)) comps.push_back(move(C));
                global_tests++;
            }
            for (size_t ci=0;ci<comps.size();++ci) if (comps[ci].curve==e && comps[ci].e==eps)
                for (const auto& pt:comps[ci].points) owner[pt]=ci;
        }
    }

    // ---- 6. the resolution graph
    Graph G;
    struct Origin { bool cover; size_t comp, s, p, i; };
    vector<Origin> origin;    // where each vertex of G comes from
    for (size_t ci=0;ci<comps.size();++ci) {
        const auto& C=comps[ci];
        long ram=0, adj=0;
        for (const auto& pt:C.points) {
            const auto& q=specials[pt[0]].local.quotients[pt[1]];
            ram+=(pt[2]?q.r2:q.r1)-1;
            if (!q.w.empty()) adj+=pt[2]?q.w.back().second:q.w.front().first;
        }
        const long chi2=-2*C.f+ram;    // 2g-2
        if (chi2%2 || chi2<-2) throw Fail("Internal error: Riemann-Hurwitz gives no genus.");
        const long num=C.f*plane.weights[C.curve]-adj;
        if (num%C.e) throw Fail("Internal error: a covering curve has a fractional self-intersection.");
        Vertex v; v.id="C"+to_string(ci+1); v.kind="cover"; v.over="E"+to_string(C.curve);
        v.genus=chi2/2+1; v.self=num/C.e; v.multiplicity=C.e; v.degree=C.f;
        v.extra["ramification_index"]=Json::Int64(C.e); v.extra["degree"]=Json::Int64(C.f);
        G.v.push_back(v); origin.push_back({true,ci,0,0,0});
    }
    J points(Json::arrayValue);
    for (size_t s=0;s<specials.size();++s) {
        const auto& S=specials[s];
        J sj; sj["kind"]=S.kind; sj["chart"]=plane.chart_data[S.chart].id;
        sj["point"]=arr({plane.scalar(S.point[0]),plane.scalar(S.point[1])});
        sj["divisors"]=S.b?arr({"E"+to_string(S.a),"E"+to_string(S.b)}):arr({"E"+to_string(S.a),"discriminant"});
        J lps(Json::arrayValue);
        for (size_t p=0;p<S.local.quotients.size();++p) {
            const auto& q=S.local.quotients[p];
            J lp; lp["sheets"]=Json::Int64(q.degree);
            lp["ramification"]=arr({Json::Int64(q.k1),Json::Int64(q.k2)});
            lp["cyclic_quotient"]=arr({Json::Int64(q.n),Json::Int64(q.q)});
            J lat(Json::arrayValue);
            for (const auto& g:q.basis) lat.append(arr({Json::Int64(g.first),Json::Int64(g.second)}));
            lp["loop_lattice"]=lat;
            const size_t left=owner.at({s,p,0});
            size_t prev=left;
            J chain(Json::arrayValue);
            for (size_t i=0;i<q.w.size();++i) {
                Vertex v; v.id="H"+to_string(G.v.size()+1); v.kind="string"; v.over="P"+to_string(s+1);
                v.genus=0; v.self=-q.b[i];
                v.extra["orders"]=arr({Json::Int64(q.w[i].first),Json::Int64(q.w[i].second)});
                G.v.push_back(v); chain.append(v.id); origin.push_back({false,0,s,p,i});
                G.edge(prev,G.v.size()-1); prev=G.v.size()-1;
            }
            if (S.b) G.edge(prev,owner.at({s,p,1}));
            lp["string"]=chain;
            lp["curves"]=S.b?arr({G.v[left].id,G.v[owner.at({s,p,1})].id}):arr({G.v[left].id});
            lps.append(lp);
        }
        sj["points"]=lps;
        points.append(sj);
    }
    record["crossings"]=points;
    record["generic_sheets"]=generic;
    record["global_monodromy_tests"]=Json::Int64(global_tests);
    record["cyclic_cover"]=cyclic;
    record["graph"]=graph_json(G);
    record["invariants"]=invariants(G);

    // ---- 7. geometric genus p_g = dim O_X / I with I = {g : v_D(g) >= -a_D},
    // K = sum a_D D the canonical cycle (div of the pulled back Gorenstein
    // form). A string curve D at a local point is the toric divisor of a ray
    // w: v_D(g) >= c means no term of g at that point with <w,alpha> < c. For a
    // covering curve C with ray n and adjacent ray w', the terms of level
    // <n,alpha> = c0 restricted to C form a section of O(-c0 C)|_C, of degree
    // -c0 C^2; if it vanishes at the corner to order > -c0 C^2 it is zero. So
    // terms with <n,alpha> < c and <w',alpha> <= -<n,alpha> C^2 decide. With
    // N0 = max c_D, m^N0 lies in I, so p_g is the rank of these conditions on
    // the monomials p^i q^j t^c (c < m, i+j+c < N0).
    long pg=0;
    {
        vector<string> kc; invariants(G,&kc);
        struct Cond { E2 ray; long c; bool cover; E2 adj; long self; };
        std::map<pair<size_t,size_t>,vector<Cond>> at;
        long N0=0;
        for (size_t i=0;i<G.v.size();++i) {
            fmpq_t a; fmpq_init(a);
            if (fmpq_set_str(a,kc[i].c_str(),10) || !fmpz_is_one(fmpq_denref(a))) { fmpq_clear(a);
                throw Fail("Internal error: the canonical cycle of a hypersurface singularity is not integral."); }
            const long c=-fmpz_get_si(fmpq_numref(a)); fmpq_clear(a);
            if (c<=0) continue;
            N0=max(N0,c);
            const auto& o=origin[i];
            if (o.cover) {
                const auto& pt=comps[o.comp].points.front();
                const auto& q=specials[pt[0]].local.quotients[pt[1]];
                const E2 ray=pt[2]?E2{0,q.k2}:E2{q.k1,0};
                const E2 adj=pt[2]?(q.w.empty()?E2{q.k1,0}:q.w.back()):(q.w.empty()?E2{0,q.k2}:q.w.front());
                at[{pt[0],pt[1]}].push_back({ray,c,true,adj,G.v[i].self});
            } else {
                const auto& q=specials[o.s].local.quotients[o.p];
                at[{o.s,o.p}].push_back({q.w[o.i],c,false,{0,0},0});
            }
        }
        if (N0>0) {
            vector<array<long,3>> mono;
            for (long c=0;c<m;++c) for (long i=0;i+c<N0;++i) for (long j=0;i+j+c<N0;++j) mono.push_back({i,j,c});
            vector<vector<Num>> rows;
            auto dot=[](const E2& a,const E2& b){ return a.first*b.first+a.second*b.second; };
            for (const auto& [where,conds]:at) {
                Special& S=specials[where.first];
                Box R{0,0};
                for (const auto& cd:conds) {
                    const long top=cd.c*D;
                    if (!cd.cover) { R.a=max(R.a,(top-1)/cd.ray.first); R.b=max(R.b,(top-1)/cd.ray.second); continue; }
                    const long across=-top*cd.self;   // bound for <adj,E>
                    if (cd.ray.first) { R.a=max(R.a,(top-1)/cd.ray.first); R.b=max(R.b,across/cd.adj.second); }
                    else { R.b=max(R.b,(top-1)/cd.ray.second); R.a=max(R.a,across/cd.adj.first); }
                }
                const Box need=needed_box(*S.local.tree,R);
                if (need.a>S.local.box.a || need.b>S.local.box.b)
                    S.local=analyse(terms,m,D,S.localize(plane.chart_data[S.chart].map),S.branch(),
                                    Box{max(need.a,S.local.box.a)+D,max(need.b,S.local.box.b)+D});
                auto [Ps,Qs]=local_chart(S.localize(plane.chart_data[S.chart].map),S.branch(),D,R);
                auto [zeta,known]=path_root(S.local.branches[where.second].path,&R);
                if (known.a<R.a || known.b<R.b) throw Fail("Internal error: a local root lacks precision for the geometric genus.");
                vector<Series> Pp{s_one()}, Qp{s_one()}, Zp{s_one()};
                for (long k=1;k<N0;++k) { Pp.push_back(s_mul(Pp.back(),Ps,R)); Qp.push_back(s_mul(Qp.back(),Qs,R)); }
                for (long k=1;k<m;++k) Zp.push_back(s_mul(Zp.back(),zeta,R));
                vector<Series> values;
                for (const auto& [i,j,c]:mono) values.push_back(s_mul(s_mul(Pp[i],Qp[j],R),Zp[c],R));
                set<E2> forbidden;
                for (const auto& v:values) for (const auto& [E,coef]:v) for (const auto& cd:conds) {
                    const long level=dot(cd.ray,E);
                    if (level>=cd.c*D) continue;
                    if (cd.cover && dot(cd.adj,E)>-level*cd.self) continue;
                    forbidden.insert(E);
                }
                for (const auto& E:forbidden) {
                    vector<Num> row;
                    for (const auto& v:values) { auto it=v.find(E); row.push_back(it==v.end()?Num(0):it->second); }
                    rows.push_back(move(row));
                }
            }
            if (!rows.empty()) {
                const slong r=static_cast<slong>(rows.size()), c=static_cast<slong>(mono.size());
                bool rational=true;
                for (const auto& row:rows) for (const auto& x:row) if (!x.rat) rational=false;
                if (rational) {
                    fmpq_mat_t A; fmpq_mat_init(A,r,c);
                    for (slong i=0;i<r;++i) for (slong j=0;j<c;++j) fmpq_set(fmpq_mat_entry(A,i,j),rows[i][j].q);
                    fmpq_mat_t B; fmpq_mat_init(B,r,c); pg=fmpq_mat_rref(B,A); fmpq_mat_clear(B); fmpq_mat_clear(A);
                } else {
                    gr_ctx_t ctx2; gr_ctx_init_complex_qqbar(ctx2);
                    gr_mat_t A; gr_mat_init(A,r,c,ctx2);
                    int status=0;
                    for (slong i=0;i<r;++i) for (slong j=0;j<c;++j) {
                        qqbar_t x; qqbar_init(x); rows[i][j].to_qqbar(x);
                        status|=gr_set_other(gr_mat_entry_ptr(A,i,j,ctx2),x,ctx2,ctx2);
                        qqbar_clear(x);
                    }
                    slong rank=0;
                    status|=gr_mat_rank(&rank,A,ctx2);
                    gr_mat_clear(A,ctx2); gr_ctx_clear(ctx2);
                    if (status) throw Fail("The exact rank for the geometric genus did not finish.");
                    pg=rank;
                }
            }
        }
    }
    {
        fmpq_t k2,mu; fmpq_init(k2); fmpq_init(mu);
        fmpq_set_str(k2,record["invariants"]["K2"].asCString(),10);
        fmpq_set_si(mu,12*pg+record["invariants"]["euler_characteristic"].asInt64()-1,1);
        fmpq_add(mu,mu,k2);
        record["invariants"]["geometric_genus"]=Json::Int64(pg);
        record["invariants"]["milnor_number_from_laufer"]=q_string(mu);
        fmpq_clear(k2); fmpq_clear(mu);
    }
    J contracted(Json::arrayValue);
    Graph Gmin=minimal_good(G,contracted);
    J mj=graph_json(Gmin);
    mj["contracted"]=contracted;
    mj["invariants"]=invariants(Gmin);
    mj["invariants"]["geometric_genus"]=Json::Int64(pg);
    mj["invariants"]["milnor_number_from_laufer"]=record["invariants"]["milnor_number_from_laufer"];
    record["minimal_good_resolution"]=mj;
    const auto& inv=mj["invariants"];
    record["status"]="resolved";
    record["summary"]=to_string(Gmin.v.size())+" exceptional curves in the minimal good resolution ("+
        to_string(G.v.size())+" from Jung's method), K^2 = "+inv["K2"].asString()+", Euler characteristic "+
        to_string(inv["euler_characteristic"].asInt64())+", geometric genus "+to_string(pg)+".";
    const auto labels=indexed_labels(plane.algebraics);
    J numbers(Json::objectValue);
    for (size_t i=0;i<labels.size();++i) {
        numbers["a"+to_string(i+1)]["exact"]=labels[i];
        numbers["a"+to_string(i+1)]["approximation"]=atlas::flint_string(qqbar_get_str_nd(plane.algebraics[i].n,10));
    }
    record["algebraic_numbers"]=numbers;
    fmpq_mpoly_ctx_clear(nctx);
    }
    fmpq_mpoly_ctx_clear(ctx);
    record["engine"]="C++ / FLINT Jung method";
    record["elapsed_ms"]=Json::Int64(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now()-started).count());
    return record;
}
} // namespace jung

static int program() {
    J req,out;
    try {
        std::cin>>req;
        out=jung::run(req);
    } catch (const jung::Declined& e) { out=J(); out["error"]=e.what(); out["kind"]="declined"; }
    catch (const std::invalid_argument& e) { out=J(); out["error"]=e.what(); out["kind"]="input"; }
    catch (const std::exception& e) { out=J(); out["error"]=e.what(); out["kind"]="limit"; }
    Json::StreamWriterBuilder w; w["indentation"]="";
    std::cout<<Json::writeString(w,out)<<"\n";
    return 0;
}

extern "C" __attribute__((visibility("default"))) int atlas_jung_main(int, char**) noexcept {
    const int status=program();
    std::cout.flush();
    fflush(nullptr);
    return status;
}

#ifndef ATLAS_HELPER_LIBRARY
int main(int argc, char** argv) { return atlas_jung_main(argc, argv); }
#endif
