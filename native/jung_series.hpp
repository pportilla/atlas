// Truncated power series in two monomial variables over the algebraic numbers,
// and polynomials in one more variable t with such coefficients.
//
// Exponents are integers scaled by a common denominator D, so U = u^(1/D) and
// W = w^(1/D). A box (a,b) keeps the terms U^i W^j with i<=a and j<=b; these
// form the quotient ring by the ideal (U^(a+1), W^(b+1)), so every product and
// every Hensel step below is exact in that ring.
#pragma once
#include "jung_coef.hpp"
#include <numeric>

namespace jung {
using E2 = pair<long,long>;
struct Box { long a=0, b=0; };

inline bool in_box(const E2& k, const Box& B) { return k.first<=B.a && k.second<=B.b; }
using PNum = ::Num;          // the plane resolver's algebraic numbers
using Num = Coef;            // coefficients of series in this resolver
inline Num nsub(const Num& a, const Num& b) { return a-b; }
inline Num nneg(const Num& a) { return -a; }
inline Num ndiv(const Num& a, const Num& b) { return a/b; }
inline Num ndiv_si(const Num& a, long k) { return a/Num(k); }
inline Num nmul_si(const Num& a, long k) { return cmul_si(a,k); }
inline bool nequal(const Num& a, const Num& b) { return a==b; }
inline Num npow(const Num& a, ulong k) { return cpow(a,k); }
inline void nadd_to(Num& a, const Num& b) { a+=b; }
inline Num from_fmpq(const fmpq_t q) { return Num(q); }

// ---------------------------------------------------------------- series
using Series = map<E2,Num>;

inline void add_term(Series& s, const E2& k, const Num& c) {
    if (c.zero()) return;
    auto [it,created]=s.try_emplace(k,c);
    if (!created) { it->second+=c; if (it->second.zero()) s.erase(it); }
}
inline Series s_add(Series x, const Series& y) { for (const auto& [k,c]:y) add_term(x,k,c); return x; }
inline Series s_scale(const Series& x, const Num& c) {
    Series r; if (c.zero()) return r;
    for (const auto& [k,v]:x) r.emplace(k,v*c);
    return r;
}
inline Series s_mul(const Series& x, const Series& y, const Box& B) {
    Series r;
    for (const auto& [kx,cx]:x) {
        if (kx.first>B.a || kx.second>B.b) continue;
        for (const auto& [ky,cy]:y) {
            if (kx.first+ky.first>B.a) break;
            if (kx.second+ky.second>B.b) continue;
            add_term(r,{kx.first+ky.first,kx.second+ky.second},cx*cy);
        }
    }
    return r;
}
inline Series s_truncate(const Series& x, const Box& B) {
    Series r; for (const auto& [k,c]:x) if (in_box(k,B)) r.emplace(k,c);
    return r;
}
inline Series s_one() { return Series{{{0,0},Num(1)}}; }
inline Num s_constant(const Series& s) { auto it=s.find({0,0}); return it==s.end()?Num(0):it->second; }

// Inverse of a unit series by Newton's iteration h <- h(2 - x h); the order
// of 1 - x h at least doubles at each step, so it ends within the box.
inline Series s_inverse(const Series& x, const Box& B) {
    const Num c0=s_constant(x);
    if (c0.zero()) throw Fail("Internal error: inverting a non-unit series.");
    Series h{{{0,0},Num(1)/c0}};
    for (int it=0;it<64;++it) {
        Series e=s_mul(x,h,B);              // x h = 1 + error
        add_term(e,{0,0},Num(-1));
        if (e.empty()) return h;
        h=s_add(h,s_scale(s_mul(h,e,B),Num(-1)));
    }
    throw Fail("Internal error: series inversion did not converge.");
}

// ---------------------------------------------------------------- univariate polynomials
using UPoly = vector<Num>;
inline void u_trim(UPoly& p) { while (!p.empty() && p.back().zero()) p.pop_back(); }
inline UPoly u_mul(const UPoly& a, const UPoly& b) {
    if (a.empty() || b.empty()) return {};
    UPoly r(a.size()+b.size()-1);
    for (size_t i=0;i<a.size();++i) if (!a[i].zero())
        for (size_t j=0;j<b.size();++j) if (!b[j].zero()) r[i+j]+=a[i]*b[j];
    u_trim(r); return r;
}
inline UPoly u_add(UPoly a, const UPoly& b) {
    if (a.size()<b.size()) a.resize(b.size());
    for (size_t i=0;i<b.size();++i) a[i]+=b[i];
    u_trim(a); return a;
}
inline UPoly u_scale(const UPoly& a, const Num& c) {
    UPoly r; r.reserve(a.size()); for (const auto& x:a) r.push_back(x*c); u_trim(r); return r;
}
// p(t+c)
inline UPoly u_shift(UPoly p, const Num& c) {
    if (c.zero() || p.size()<2) return p;
    const long n=static_cast<long>(p.size());
    for (long i=0;i<n;++i) for (long j=n-2;j>=i;--j) p[j]+=p[j+1]*c;
    u_trim(p); return p;
}
// Inverse of b (b(0) != 0) modulo t^n.
inline UPoly u_inverse_mod(const UPoly& b, long n) {
    if (b.empty() || b[0].zero()) throw Fail("Internal error: inverting a polynomial without constant term.");
    UPoly r(n);
    const Num inv=Num(1)/b[0];
    for (long k=0;k<n;++k) {
        Num s=(k==0)?Num(1):Num(0);
        for (long i=1;i<=k && i<static_cast<long>(b.size());++i) s=nsub(s,b[i]*r[k-i]);
        r[k]=s*inv;
    }
    u_trim(r); return r;
}

// Exact roots with multiplicities over the algebraic numbers.
inline vector<pair<Num,long>> u_roots(const UPoly& p) {
    gr_ctx_t ctx; gr_ctx_init_complex_qqbar(ctx);
    gr_poly_t g; gr_poly_init(g,ctx);
    int status=0;
    for (size_t i=0;i<p.size();++i) { qqbar_t c; qqbar_init(c); p[i].to_qqbar(c);
        status|=gr_poly_set_coeff_scalar(g,static_cast<slong>(i),c,ctx); qqbar_clear(c); }
    gr_vec_t roots; gr_vec_init(roots,0,ctx);
    fmpz_vec_t mult; fmpz_vec_init(mult,0);
    if (!status) status=gr_poly_roots(roots,mult,g,0,ctx);
    vector<pair<Num,long>> out;
    if (!status) for (slong i=0;i<gr_vec_length(roots,ctx);++i) {
        Num r; r.set(static_cast<qqbar_srcptr>(gr_vec_entry_ptr(roots,i,ctx)));
        out.emplace_back(move(r),fmpz_get_si(fmpz_vec_entry_srcptr(mult,i)));
    }
    fmpz_vec_clear(mult); gr_vec_clear(roots,ctx); gr_poly_clear(g,ctx); gr_ctx_clear(ctx);
    if (status) throw Fail("The exact algebraic root computation did not finish.");
    return out;
}

// ---------------------------------------------------------------- polynomials in t over series
// Stored as a series of polynomials: key U^i W^j -> polynomial in t.
using TS = map<E2,UPoly>;

inline void ts_add(TS& s, const E2& k, const UPoly& p) {
    if (p.empty()) return;
    auto [it,created]=s.try_emplace(k,p);
    if (!created) { it->second=u_add(move(it->second),p); if (it->second.empty()) s.erase(it); }
}
inline long ts_degree(const TS& s) {
    long d=-1; for (const auto& [k,p]:s) d=max(d,static_cast<long>(p.size())-1); return d;
}
inline Series ts_coefficient(const TS& s, long e) {
    Series r;
    for (const auto& [k,p]:s) if (e<static_cast<long>(p.size()) && !p[e].zero()) r.emplace(k,p[e]);
    return r;
}
inline TS ts_truncate(const TS& s, const Box& B) {
    TS r; for (const auto& [k,p]:s) if (in_box(k,B)) r.emplace(k,p); return r;
}
// Multiply a polynomial in t (series coefficients) by a series.
inline void ts_add_product(TS& out, const UPoly& p, const E2& shift, const Series& x, const Box& B) {
    for (const auto& [k,c]:x) {
        E2 key{k.first+shift.first,k.second+shift.second};
        if (!in_box(key,B)) continue;
        ts_add(out,key,u_scale(p,c));
    }
}
// p(t + s) for a series s, by Taylor expansion.
inline TS ts_taylor_shift(const TS& p, const Series& s, const Box& B) {
    if (s.empty()) return p;
    const long n=ts_degree(p);
    vector<Series> powers{s_one()};
    for (long l=1;l<=n;++l) powers.push_back(s_mul(powers.back(),s,B));
    TS out;
    for (const auto& [k,poly]:p) {
        for (long l=0;l<static_cast<long>(poly.size());++l) {
            if (powers[l].empty()) break;
            // l-th divided derivative: sum_i binom(i,l) p_i t^(i-l)
            UPoly d;
            for (long i=l;i<static_cast<long>(poly.size());++i) {
                fmpz_t bin; fmpz_init(bin); fmpz_bin_uiui(bin,i,l);
                d.push_back(cmul_fmpz(poly[i],bin)); fmpz_clear(bin);
            }
            u_trim(d);
            if (!d.empty()) ts_add_product(out,d,k,powers[l],B);
        }
    }
    return out;
}

inline long gcd_exponents(const TS& p) {
    long g=0;
    for (const auto& [k,poly]:p) g=std::gcd(g,std::gcd(k.first,k.second));
    return g;
}

// Hensel lifting of P = A*B in the box ring, where P(0,0,t) = t^n * B0(t) with
// B0(0) != 0. A is monic of degree n with A(0,0,t)=t^n. The lift proceeds by
// total degree; every correction solves t^n*b + B0*a = e with deg a < n.
inline pair<TS,TS> hensel(const TS& P, long n, const Box& B) {
    auto it0=P.find({0,0});
    if (it0==P.end()) throw Fail("Internal error: Hensel input without constant fibre.");
    const UPoly& P0=it0->second;
    for (long i=0;i<n;++i) if (i<static_cast<long>(P0.size()) && !P0[i].zero())
        throw Fail("Internal error: Hensel factor is not t^n at the origin.");
    UPoly B0(P0.begin()+min<long>(n,P0.size()),P0.end());
    const UPoly s=u_inverse_mod(B0,n);
    UPoly An(n+1); An[n]=Num(1);
    TS A{{{0,0},An}}, Bf{{{0,0},B0}};
    if (n==0) return {A,P};
    const long step=gcd_exponents(P);
    if (step==0) return {A,Bf};
    map<long,vector<E2>> degA, degB;     // keys of positive degree
    map<long,vector<E2>> degP;
    for (const auto& [k,p]:P) if (k.first+k.second>0) degP[k.first+k.second].push_back(k);
    const long top=B.a+B.b;
    for (long d=step; d<=top; d+=step) {
        TS e;
        auto ip=degP.find(d);
        if (ip!=degP.end()) for (const auto& k:ip->second) e[k]=P.at(k);
        for (const auto& [d1,keys1]:degA) {
            if (d1>=d) break;
            auto ib=degB.find(d-d1);
            if (ib==degB.end()) continue;
            for (const auto& ka:keys1) for (const auto& kb:ib->second) {
                E2 k{ka.first+kb.first,ka.second+kb.second};
                if (!in_box(k,B)) continue;
                ts_add(e,k,u_scale(u_mul(A.at(ka),Bf.at(kb)),Num(-1)));
            }
        }
        for (auto& [k,poly]:e) {
            if (poly.empty()) continue;
            UPoly a=u_mul(poly,s); if (static_cast<long>(a.size())>n) a.resize(n); u_trim(a);
            UPoly rest=u_add(poly,u_scale(u_mul(B0,a),Num(-1)));
            for (long i=0;i<n && i<static_cast<long>(rest.size());++i)
                if (!rest[i].zero()) throw Fail("Internal error: inexact Hensel division.");
            UPoly b(rest.size()>static_cast<size_t>(n)?rest.begin()+n:rest.end(),rest.end());
            u_trim(b);
            if (!a.empty()) { A[k]=a; degA[d].push_back(k); }
            if (!b.empty()) { Bf[k]=b; degB[d].push_back(k); }
        }
    }
    return {A,Bf};
}
} // namespace jung
