// Quasi-ordinary roots over a normal crossing point and the normalized
// branches they define.
//
// Over a point where the discriminant is u^a w^b times a unit, the roots of
// the Weierstrass polynomial are fractional power series whose differences
// are monomials times units (Abhyankar-Jung). After a Tschirnhausen shift the
// roots share the factor U^delta, delta the smallest exponent of a
// difference; delta is the componentwise minimum of e/k over the terms U^e of
// the coefficients of t^(n-k), and it is attained. Dividing by U^delta leaves
// roots with at least two distinct constant terms, so Hensel's lemma splits
// the polynomial. Repeating gives a tree whose nodes carry the exponents of
// root differences. Every exponent of a root lies in Z^2 + sum Z*delta along
// its path, so an orbit of roots under the local monodromy is a normalized
// branch Spec C[sigma^v cap M] with M = Z^2 + sum Z*delta (a cyclic quotient
// singularity), and the children of a node are permuted by the roots of unity
// of order [L+Z*delta : L].
#pragma once
#include "jung_series.hpp"
#include <flint/fmpz_mat.h>
#include <memory>
#include <functional>

namespace jung {

struct Precision : Fail { using Fail::Fail; };

struct Node {
    long n=1;
    E2 delta{0,0};
    vector<Num> c;          // constant terms of the children
    vector<long> mult;      // their multiplicities
    vector<unique_ptr<Node>> kid;   // null for a single root
    TS poly;                // polynomial after the shift and the division by U^delta
    Series shift;           // Tschirnhausen shift
    Box box;                // box of poly
    Box in_box;             // box of the polynomial this node received
};

// Determinant of the lattice spanned by integer vectors (all of rank 2 here).
inline long lattice_det(const vector<E2>& gens) {
    fmpz_mat_t A,H; fmpz_mat_init(A,gens.size(),2); fmpz_mat_init(H,gens.size(),2);
    for (size_t i=0;i<gens.size();++i) {
        fmpz_set_si(fmpz_mat_entry(A,i,0),gens[i].first);
        fmpz_set_si(fmpz_mat_entry(A,i,1),gens[i].second);
    }
    fmpz_mat_hnf(H,A);
    fmpz_t d; fmpz_init(d);
    fmpz_mul(d,fmpz_mat_entry(H,0,0),fmpz_mat_entry(H,1,1));
    fmpz_submul(d,fmpz_mat_entry(H,0,1),fmpz_mat_entry(H,1,0));
    fmpz_abs(d,d);
    long r=fmpz_get_si(d);
    fmpz_clear(d); fmpz_mat_clear(A); fmpz_mat_clear(H);
    if (r==0) throw Fail("Internal error: degenerate exponent lattice.");
    return r;
}

// Split the monic polynomial P (P(0,0,t)=t^n) into its tree of roots.
inline unique_ptr<Node> split(TS P, long n, Box B) {
    auto node=make_unique<Node>();
    node->n=n; node->in_box=B;
    if (n==1) return node;
    // Tschirnhausen: remove the mean of the roots.
    Series mean=s_scale(ts_coefficient(P,n-1),Num(-1));
    mean=s_scale(mean,ndiv_si(Num(1),n));
    if (!mean.empty()) P=ts_taylor_shift(P,mean,B);
    node->shift=mean;
    // delta = componentwise minimum of e/k over terms of the coefficient of t^(n-k).
    long an=-1, ad=1, bn=-1, bd=1;  // fractions a=an/ad, b=bn/bd
    for (const auto& [k,poly]:P) for (long e=0;e<n && e<static_cast<long>(poly.size());++e) {
        if (poly[e].zero()) continue;
        const long kk=n-e;
        if (an<0 || k.first*ad<an*kk) { an=k.first; ad=kk; }
        if (bn<0 || k.second*bd<bn*kk) { bn=k.second; bd=kk; }
    }
    if (an<0) throw Precision("The box does not separate the roots.");
    if (an%ad || bn%bd) throw Precision("A root-difference exponent is not on the scaled lattice.");
    const E2 delta{an/ad,bn/bd};
    bool attained=false;
    for (const auto& [k,poly]:P) for (long e=0;e<n && e<static_cast<long>(poly.size());++e)
        if (!poly[e].zero() && k.first==(n-e)*delta.first && k.second==(n-e)*delta.second) attained=true;
    if (!attained) throw Precision("The smallest root difference is not attained in the box.");
    if (n*delta.first>B.a || n*delta.second>B.b) throw Precision("The box is too small to certify a root difference.");
    if (delta.first==0 && delta.second==0) throw Fail("Internal error: the roots are not separated at the origin.");
    node->delta=delta;
    const Box nb{B.a-n*delta.first,B.b-n*delta.second};
    TS Q;
    for (const auto& [k,poly]:P) for (long e=0;e<static_cast<long>(poly.size());++e) {
        if (poly[e].zero()) continue;
        if (e==n) { if (k!=E2{0,0}) throw Fail("Internal error: non-monic quasi-ordinary factor."); }
        const long kk=n-e;
        E2 key{k.first-kk*delta.first,k.second-kk*delta.second};
        if (key.first<0 || key.second<0) throw Fail("Internal error: a coefficient is not divisible by its root monomial.");
        if (!in_box(key,nb)) continue;
        UPoly mono(e+1); mono[e]=poly[e];
        ts_add(Q,key,mono);
    }
    node->box=nb;
    auto it0=Q.find({0,0});
    if (it0==Q.end()) throw Fail("Internal error: lost the leading coefficient.");
    for (auto& [root,mult]:u_roots(it0->second)) { node->c.push_back(root); node->mult.push_back(mult); }
    if (node->c.size()<2) throw Fail("Internal error: the Tschirnhausen form did not split.");
    for (size_t j=0;j<node->c.size();++j) {
        if (node->mult[j]==1) { node->kid.emplace_back(); continue; }
        TS shifted;
        for (const auto& [k,poly]:Q) ts_add(shifted,k,u_shift(poly,node->c[j]));
        auto [A,rest]=hensel(shifted,node->mult[j],nb);
        node->kid.push_back(split(move(A),node->mult[j],nb));
    }
    node->poly=move(Q);
    return node;
}

// ---------------------------------------------------------------- roots
inline Series ts_eval(const TS& P, const Series& y, const Box& B) {
    const long n=ts_degree(P);
    if (n<0) return {};
    Series r=ts_coefficient(P,n);
    for (long k=n-1;k>=0;--k) r=s_add(s_mul(r,y,B),ts_coefficient(P,k));
    return s_truncate(r,B);
}
inline TS ts_derivative(const TS& P) {
    TS d;
    for (const auto& [k,p]:P) {
        UPoly q;
        for (size_t i=1;i<p.size();++i) q.push_back(nmul_si(p[i],static_cast<long>(i)));
        u_trim(q); if (!q.empty()) d.emplace(k,q);
    }
    return d;
}
// The simple root of P whose constant term is c, by Newton's method.
inline Series newton_root(const TS& P, const Num& c, const Box& B) {
    Series y; add_term(y,{0,0},c);
    const TS dP=ts_derivative(P);
    for (int it=0;it<80;++it) {
        Series val=ts_eval(P,y,B);
        if (val.empty()) return y;
        Series corr=s_mul(val,s_inverse(ts_eval(dP,y,B),B),B);
        y=s_add(y,s_scale(corr,Num(-1)));
    }
    throw Fail("Internal error: Newton's method did not converge in the box ring.");
}
inline Series monomial_times(const Series& y, const E2& d) {
    Series r; for (const auto& [k,c]:y) r.emplace(E2{k.first+d.first,k.second+d.second},c); return r;
}
using Path = vector<pair<const Node*,size_t>>;
// The root at the end of a path from the top node, truncated to the box in
// which it is exact: B_top minus (n-1)*delta over the nodes of the path. With
// a target box, Newton's method runs only to the precision the target needs.
inline pair<Series,Box> path_root(const Path& path, const Box* target=nullptr) {
    auto [nu,j]=path.back();
    if (nu->kid[j]) throw Fail("Internal error: a root path must end at a simple root.");
    Box work=nu->box;
    if (target) {
        E2 shift{0,0};
        for (const auto& step:path) { shift.first+=step.first->delta.first; shift.second+=step.first->delta.second; }
        work.a=min(work.a,max(0L,target->a-shift.first));
        work.b=min(work.b,max(0L,target->b-shift.second));
    }
    Series y=newton_root(ts_truncate(nu->poly,work),nu->c[j],work);
    for (size_t k=path.size();k-->0;) {
        const Node* v=path[k].first;
        if (k+1<path.size()) add_term(y,{0,0},v->c[path[k].second]);
        y=s_add(v->shift,monomial_times(y,v->delta));
    }
    // Node k's variable is exact below in_box - (n-1)*delta in its own
    // coordinates, which the ancestors' monomials shift by their deltas.
    Box known=path.front().first->in_box;
    E2 above{0,0};
    for (const auto& [v,i]:path) {
        known.a=min(known.a,v->in_box.a-(v->n-1)*v->delta.first+above.first);
        known.b=min(known.b,v->in_box.b-(v->n-1)*v->delta.second+above.second);
        above.first+=v->delta.first; above.second+=v->delta.second;
    }
    if (target) { known.a=min(known.a,target->a); known.b=min(known.b,target->b); }
    return {s_truncate(y,known),known};
}
// Box that the top polynomial needs: every node must certify its delta
// (B >= n*delta after its ancestors' divisions) and a root must stay exact
// up to the target after the losses (n-1)*delta.
inline Box needed_box(const Node& top, const Box& target) {
    Box need=target;
    std::function<void(const Node&,E2,E2)> walk=[&](const Node& v, E2 used, E2 lost) {
        if (v.n==1) return;
        const E2 u2{used.first+v.n*v.delta.first,used.second+v.n*v.delta.second};
        const E2 l2{lost.first+(v.n-1)*v.delta.first,lost.second+(v.n-1)*v.delta.second};
        need.a=max({need.a,u2.first,target.a+l2.first}); need.b=max({need.b,u2.second,target.b+l2.second});
        for (size_t j=0;j<v.kid.size();++j) if (v.kid[j]) walk(*v.kid[j],u2,l2);
    };
    walk(top,{0,0},{0,0});
    return need;
}
inline void leaf_paths(const Node& v, Path path, vector<Path>& out) {
    for (size_t j=0;j<v.c.size();++j) {
        Path p=path; p.emplace_back(&v,j);
        if (!v.kid[j]) out.push_back(p); else leaf_paths(*v.kid[j],p,out);
    }
}
// Exponent of the difference of two roots: the deltas of the nodes from the
// top down to their last common node add up (each delta is relative to the
// variable of its node, already divided by its ancestors' monomials).
inline E2 contact(const Path& a, const Path& b) {
    E2 sum{0,0};
    for (size_t k=0;k<a.size() && k<b.size();++k) {
        sum.first+=a[k].first->delta.first; sum.second+=a[k].first->delta.second;
        if (a[k].first!=b[k].first || a[k].second!=b[k].second) return sum;
    }
    throw Fail("Internal error: two root paths coincide.");
}
// The image of a root path under U -> omega U (side 0) or W -> omega W (side 1).
inline Path monodromy(const Path& p, int side, long D) {
    const Num omega=croot_of_unity(1,static_cast<ulong>(D));
    Path out; const Node* v=p.front().first;
    long ex=0;   // exponent of the monomial in front of this node's constants
    for (size_t k=0;k<p.size();++k) {
        const Node* orig=p[k].first;
        ex+=side?orig->delta.second:orig->delta.first;
        const Num target=orig->c[p[k].second]*npow(omega,static_cast<ulong>(((ex%D)+D)%D));
        size_t j=0; while (j<v->c.size() && !nequal(v->c[j],target)) ++j;
        if (j==v->c.size()) throw Fail("Internal error: the monodromy image of a root is missing.");
        out.emplace_back(v,j);
        if (k+1<p.size()) v=v->kid[j].get();
    }
    return out;
}
// ---------------------------------------------------------------- branches
// The loop a*gamma_u + b*gamma_w acts on a root as U -> omega^a U,
// W -> omega^b W (omega = exp(2 pi i/D)). A normalized branch is an orbit of
// roots, the toric surface of the cone R^2_{>=0} in the lattice N of loops
// that fix one of its roots: N = Hom(M,Z) for M = Z^2 + Z supp(root), and
// [Z^2:N] is the number of roots in the orbit.
struct Branch {
    Path path;                  // a root of the orbit
    long roots=0;               // size of the orbit
    long k1=1, k2=1;            // orders of gamma_u and gamma_w on that root
    vector<E2> lattice;         // generators of N
    vector<Path> members;       // all roots of the orbit
};

// The roots of t^m + g are permuted by t -> z t (z^m = 1). Every constant on a
// root's path is multiplied by z: the node of the image has the conjugate
// polynomial z^n P(y/z), whose shift is z times the original shift.
inline Path scale_path(const Path& p, const Num& z) {
    Path out; const Node* v=p.front().first;
    for (size_t k=0;k<p.size();++k) {
        const Num target=p[k].first->c[p[k].second]*z;
        size_t j=0; while (j<v->c.size() && !nequal(v->c[j],target)) ++j;
        if (j==v->c.size()) throw Fail("Internal error: the covering group does not permute the roots.");
        out.emplace_back(v,j);
        if (k+1<p.size()) v=v->kid[j].get();
    }
    return out;
}

inline Path act(Path p, long a, long b, long D) {
    for (long i=0;i<a;++i) p=monodromy(p,0,D);
    for (long i=0;i<b;++i) p=monodromy(p,1,D);
    return p;
}

inline vector<Branch> collect_branches(const Node& top, long D) {
    vector<Path> leaves; leaf_paths(top,{},leaves);
    vector<bool> seen(leaves.size(),false);
    vector<Branch> out;
    auto index=[&](const Path& p) {
        for (size_t j=0;j<leaves.size();++j) if (leaves[j]==p) return j;
        throw Fail("Internal error: the monodromy image of a root is not a root.");
    };
    for (size_t i=0;i<leaves.size();++i) {
        if (seen[i]) continue;
        Branch B; B.path=leaves[i];
        vector<size_t> queue{i}; seen[i]=true;
        for (size_t q=0;q<queue.size();++q) for (int side=0;side<2;++side) {
            const size_t j=index(monodromy(leaves[queue[q]],side,D));
            if (!seen[j]) { seen[j]=true; queue.push_back(j); }
        }
        B.roots=static_cast<long>(queue.size());
        for (size_t q:queue) B.members.push_back(leaves[q]);
        for (B.k1=1; act(B.path,B.k1,0,D)!=B.path; ++B.k1) {}
        for (B.k2=1; act(B.path,0,B.k2,D)!=B.path; ++B.k2) {}
        B.lattice={{B.k1,0},{0,B.k2}};
        for (long a=0;a<B.k1;++a) for (long b=0;b<B.k2;++b)
            if ((a||b) && act(B.path,a,b,D)==B.path) B.lattice.push_back({a,b});
        out.push_back(move(B));
    }
    return out;
}

// The cyclic quotient singularity of a branch and the Hirzebruch-Jung string
// resolving it. n1=(k1,0) and n2=(0,k2) are the primitive generators of the
// two rays in N. The string's rays w_1..w_r satisfy w_{i-1}+w_{i+1} = b_i w_i
// with n/q = [b_1,...,b_r]; w = (a,b) has order a along u=0 and b along w=0.
struct Quotient {
    long degree=0, k1=0, k2=0, r1=0, r2=0, n=1, q=0;
    vector<long> b;
    vector<E2> w;   // w_1..w_r in Z^2
    E2 basis[2];    // a basis of N
};

inline long ext_gcd(long a, long b, long& x, long& y) {
    if (b==0) { x=(a>=0)?1:-1; y=0; return a>=0?a:-a; }
    long x1,y1; long g=ext_gcd(b,a%b,x1,y1);
    x=y1; y=x1-(a/b)*y1; return g;
}

inline Quotient quotient(const Branch& br) {
    Quotient Q;
    Q.k1=br.k1; Q.k2=br.k2;
    Q.degree=lattice_det(br.lattice);
    if (Q.degree!=br.roots) throw Fail("Internal error: a branch orbit does not match the index of its lattice.");
    if (Q.degree%Q.k1 || Q.degree%Q.k2 || (Q.k1*Q.k2)%Q.degree)
        throw Fail("Internal error: inconsistent branch lattice.");
    Q.r1=Q.degree/Q.k1; Q.r2=Q.degree/Q.k2;
    Q.n=Q.k1*Q.k2/Q.degree;
    fmpz_mat_t A,H; fmpz_mat_init(A,br.lattice.size(),2); fmpz_mat_init(H,br.lattice.size(),2);
    for (size_t i=0;i<br.lattice.size();++i) {
        fmpz_set_si(fmpz_mat_entry(A,i,0),br.lattice[i].first); fmpz_set_si(fmpz_mat_entry(A,i,1),br.lattice[i].second);
    }
    fmpz_mat_hnf(H,A);
    const E2 b1{fmpz_get_si(fmpz_mat_entry(H,0,0)),fmpz_get_si(fmpz_mat_entry(H,0,1))};
    const E2 b2{fmpz_get_si(fmpz_mat_entry(H,1,0)),fmpz_get_si(fmpz_mat_entry(H,1,1))};
    fmpz_mat_clear(A); fmpz_mat_clear(H);
    Q.basis[0]=b1; Q.basis[1]=b2;
    if (Q.n==1) return Q;
    const long det=b1.first*b2.second-b1.second*b2.first;
    auto coords=[&](const E2& v) {
        const long x=v.first*b2.second-v.second*b2.first, y=b1.first*v.second-b1.second*v.first;
        if (x%det || y%det) throw Fail("Internal error: vector outside the lattice of the branch.");
        return E2{x/det,y/det};
    };
    const E2 n1{Q.k1,0}, n2{0,Q.k2};
    const E2 x=coords(n1);
    long y1,y2;
    if (ext_gcd(x.first,-x.second,y2,y1)!=1) throw Fail("Internal error: the ray generator is not primitive.");
    E2 f2{y1*b1.first+y2*b2.first,y1*b1.second+y2*b2.second};
    const long dd=n1.first*f2.second-n1.second*f2.first;
    long s=(n2.first*f2.second-n2.second*f2.first), t=(n1.first*n2.second-n1.second*n2.first);
    if (dd==0 || s%dd || t%dd) throw Fail("Internal error: the cone basis is not unimodular.");
    s/=dd; t/=dd;
    if (t<0) { t=-t; f2={-f2.first,-f2.second}; }
    const long k=(s>0)?(s+t-1)/t:-((-s)/t);
    const long sp=s-k*t;
    f2={f2.first+k*n1.first,f2.second+k*n1.second};
    if (t!=Q.n || sp>0 || sp<=-t) throw Fail("Internal error: wrong cyclic quotient normal form.");
    Q.q=-sp;
    if (std::gcd(Q.n,Q.q)!=1) throw Fail("Internal error: the quotient type is not coprime.");
    long aa=Q.n, bb=Q.q;
    while (bb>0) { long c=(aa+bb-1)/bb; Q.b.push_back(c); long nb=c*bb-aa; aa=bb; bb=nb; }
    E2 prev=n1, cur=f2;
    for (long bi:Q.b) {
        Q.w.push_back(cur);
        E2 next{bi*cur.first-prev.first,bi*cur.second-prev.second};
        prev=cur; cur=next;
    }
    if (cur!=n2) throw Fail("Internal error: the Hirzebruch-Jung string does not reach the second ray.");
    return Q;
}
} // namespace jung
