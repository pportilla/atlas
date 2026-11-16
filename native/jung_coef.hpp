// Exact coefficients for the Jung resolver: rationals in FLINT fmpq, other
// algebraic numbers in qqbar. Rational arithmetic stays in fmpq; a qqbar
// result that is rational returns to fmpq.
#pragma once
#include "plane_resolver.hpp"

namespace jung {
struct Coef {
    bool rat=true;
    fmpq_t q;
    qqbar_t a;
    Coef(long v=0) { fmpq_init(q); fmpq_set_si(q,v,1); qqbar_init(a); }
    Coef(int v) : Coef(static_cast<long>(v)) {}
    Coef(const Coef& o) : rat(o.rat) { fmpq_init(q); qqbar_init(a); if (rat) fmpq_set(q,o.q); else qqbar_set(a,o.a); }
    Coef(Coef&& o) noexcept : rat(o.rat) { fmpq_init(q); qqbar_init(a); fmpq_swap(q,o.q); qqbar_swap(a,o.a); }
    Coef& operator=(Coef o) { rat=o.rat; fmpq_swap(q,o.q); qqbar_swap(a,o.a); return *this; }
    ~Coef() { fmpq_clear(q); qqbar_clear(a); }
    explicit Coef(const ::Num& n) { fmpq_init(q); qqbar_init(a); set(n.n); }
    explicit Coef(const fmpq_t v) { fmpq_init(q); qqbar_init(a); fmpq_set(q,v); }
    void set(const qqbar_t v) {
        if (qqbar_is_rational(v)) { rat=true; qqbar_get_fmpq(q,v); }
        else { rat=false; qqbar_set(a,v); }
    }
    void normalize() { if (!rat && qqbar_is_rational(a)) { rat=true; qqbar_get_fmpq(q,a); } }
    void to_qqbar(qqbar_t out) const { if (rat) qqbar_set_fmpq(out,q); else qqbar_set(out,a); }
    ::Num num() const { ::Num r; to_qqbar(r.n); return r; }
    bool zero() const { return rat ? fmpq_is_zero(q) : qqbar_is_zero(a); }
    bool one() const { return rat ? fmpq_is_one(q) : qqbar_is_one(a); }
};

template<class Rat, class Alg>
inline Coef binary(const Coef& x, const Coef& y, Rat rat, Alg alg) {
    Coef r;
    if (x.rat && y.rat) { rat(r.q,x.q,y.q); return r; }
    qqbar_t u,v; qqbar_init(u); qqbar_init(v);
    x.to_qqbar(u); y.to_qqbar(v);
    r.rat=false; alg(r.a,u,v); r.normalize();
    qqbar_clear(u); qqbar_clear(v);
    return r;
}
inline Coef operator+(const Coef& x, const Coef& y) {
    if (x.zero()) return y;
    if (y.zero()) return x;
    return binary(x,y,[](fmpq_t r,const fmpq_t a,const fmpq_t b){fmpq_add(r,a,b);},
                      [](qqbar_t r,const qqbar_t a,const qqbar_t b){qqbar_add(r,a,b);});
}
inline Coef operator-(const Coef& x, const Coef& y) {
    return binary(x,y,[](fmpq_t r,const fmpq_t a,const fmpq_t b){fmpq_sub(r,a,b);},
                      [](qqbar_t r,const qqbar_t a,const qqbar_t b){qqbar_sub(r,a,b);});
}
inline Coef operator*(const Coef& x, const Coef& y) {
    if (x.zero() || y.zero()) return Coef(0);
    return binary(x,y,[](fmpq_t r,const fmpq_t a,const fmpq_t b){fmpq_mul(r,a,b);},
                      [](qqbar_t r,const qqbar_t a,const qqbar_t b){qqbar_mul(r,a,b);});
}
inline Coef operator/(const Coef& x, const Coef& y) {
    if (y.zero()) throw Fail("Internal error: division by zero.");
    return binary(x,y,[](fmpq_t r,const fmpq_t a,const fmpq_t b){fmpq_div(r,a,b);},
                      [](qqbar_t r,const qqbar_t a,const qqbar_t b){qqbar_div(r,a,b);});
}
inline Coef operator-(const Coef& x) { Coef r=x; if (r.rat) fmpq_neg(r.q,r.q); else qqbar_neg(r.a,r.a); return r; }
inline void operator+=(Coef& x, const Coef& y) {
    if (y.zero()) return;
    if (x.rat && y.rat) { fmpq_add(x.q,x.q,y.q); return; }
    x=x+y;
}
inline bool operator==(const Coef& x, const Coef& y) {
    if (x.rat && y.rat) return fmpq_equal(x.q,y.q);
    if (x.rat!=y.rat) return false;    // normalized: a rational is never stored as qqbar
    return qqbar_equal(x.a,y.a);
}
inline Coef cpow(const Coef& x, ulong k) {
    Coef r;
    if (x.rat) { fmpq_pow_si(r.q,x.q,static_cast<slong>(k)); return r; }
    r.rat=false; qqbar_pow_ui(r.a,x.a,k); r.normalize(); return r;
}
inline Coef cmul_si(const Coef& x, long k) {
    Coef r=x;
    if (r.rat) fmpq_mul_si(r.q,r.q,k); else { qqbar_mul_si(r.a,r.a,k); r.normalize(); }
    return r;
}
inline Coef cmul_fmpz(const Coef& x, const fmpz_t k) {
    Coef r=x;
    if (r.rat) fmpq_mul_fmpz(r.q,r.q,k); else { qqbar_mul_fmpz(r.a,r.a,k); r.normalize(); }
    return r;
}
inline Coef croot_of_unity(ulong k, ulong n) {
    Coef r; qqbar_t z; qqbar_init(z); qqbar_root_of_unity(z,static_cast<slong>(k),n); r.set(z); qqbar_clear(z); return r;
}
} // namespace jung
