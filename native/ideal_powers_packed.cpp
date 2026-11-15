// Keep the ordinary power products in the same packed integer representation
// as Buchberger reduction. The caller handles wider rings with FLINT.
#include "ideal_powers.hpp"
#include "groebner.hpp"
#include "local_io.hpp"
#include <numeric>

namespace atlas::ideal_powers {
std::optional<std::vector<Poly>> echelon_products(const Ring&,const std::vector<Poly>&,size_t,size_t);
std::optional<std::vector<Poly>> plane_products(const Ring&, const std::vector<Poly>&, const std::vector<Poly>&, size_t);
namespace {
using local::Polynomial;
using local::Powers;

class PowerBasisEngine : public Engine {
public:
    // The interactive caller can finish one exact Buchberger pass. Coefficient
    // swell or excessive reduction work sends the calculation to a worker.
    bool compute_products(std::vector<Poly>& products, bool bounded) {
        if (!bounded) {
            compute(products, 0);
            return true;
        }
        strategy_ = R.lex ? LAZY : EAGER;
        bit_limit_ = 512;
        work_limit_ = 10000;
        try {
            attempt(products, 0);
            return true;
        } catch (const Swell&) {
            return false;
        }
    }
};

Poly pack(const Polynomial& f, const Ring& ring) {
    const size_t length = f.size();
    std::vector<u64> monomials(ring.stride*length);
    std::vector<const mpq_class*> coefficients;
    coefficients.reserve(length);
    mpz_class denominator = 1;
    for (const auto& [e,c] : f) {
        long powers[MAX_VARIABLES];
        for (int i=0; i<ring.n; ++i) {
            if (e[i] > MAX_DEGREE) throw Limit("Degree out of range.");
            powers[i] = e[i].get_si();
        }
        ring.pack(monomials.data()+ring.stride*coefficients.size(),powers);
        if (mpz_cmp_ui(mpq_denref(c.get_mpq_t()),1) != 0)
            mpz_lcm(denominator.get_mpz_t(),denominator.get_mpz_t(),mpq_denref(c.get_mpq_t()));
        coefficients.push_back(&c);
    }
    auto head = [&](size_t i) { return monomials.data()+ring.stride*i; };
    std::vector<size_t> order(length);
    std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&](auto a,auto b) { return ring.compare(head(a),head(b)) > 0; });
    Poly p;
    p.m.reserve(monomials.size());
    p.c.reserve(length);
    mpz_class scaled;
    for (auto i : order) {
        p.m.insert(p.m.end(),head(i),head(i)+ring.stride);
        const auto q = coefficients[i]->get_mpq_t();
        if (denominator == 1) fmpz_set_mpz(&p.c.emplace_back().v,mpq_numref(q));
        else {
            mpz_divexact(scaled.get_mpz_t(),denominator.get_mpz_t(),mpq_denref(q));
            mpz_mul(scaled.get_mpz_t(),scaled.get_mpz_t(),mpq_numref(q));
            fmpz_set_mpz(&p.c.emplace_back().v,scaled.get_mpz_t());
        }
    }
    if (!fmpz_is_one(&p.c.front().v)) primitive(p,nullptr);
    return p;
}

// When every input head is a squarefree quadratic, recover simple monic
// factors by polynomial division. Distinct coordinate heads give a
// triangular change of variables. It preserves all leading monomials, so
// the transformed monomial ideal has the same reduced-basis pivots.
bool triangular_products(const Ring& ring, const std::vector<Poly>& input) {
    if (ring.n>4 || ring.lex || input.size()<2) return false;
    for (const auto& f : input) if (f.empty() || f.m[0]!=2 || !fmpz_is_one(&f.c.front().v)) return false;
    std::map<int,Poly> forms;
    for (const auto& f : input) {
        int a=-1,b=-1;
        for(int j=0;j<ring.n;++j) {
            auto e=Ring::field(f.m.data(),ring.position(j));
            if(e>1) return false;
            if(e==1) {if(a<0)a=j;else b=j;}
        }
        if(b<0) return false;
        // Terms divisible by the first pivot variable are exactly
        // that variable times the affine form with the later pivot.
        Poly divisor;
        long exp[MAX_VARIABLES]={}; exp[a]=1;u64 variable[MAX_STRIDE]; ring.pack(variable,exp);
        for(size_t k=0;k<f.size();++k) {
            auto m=f.m.data()+k*ring.stride;
            if(!Ring::field(m,ring.position(a)))continue;
            u64 q[MAX_STRIDE];ring.divide(q,m,variable);
            divisor.m.insert(divisor.m.end(),q,q+ring.stride);
            divisor.c.push_back(f.c[k]);
        }
        Poly cofactor, residual=f;
        while(!residual.empty()) {
            if(!ring.divides(divisor.m.data(),residual.m.data()))return false;
            u64 q[MAX_STRIDE];ring.divide(q,residual.m.data(),divisor.m.data());
            cofactor.m.insert(cofactor.m.end(),q,q+ring.stride);
            cofactor.c.push_back(residual.c.front());
            Z negative;fmpz_neg(&negative.v,&residual.c.front().v);
            Poly next;
            merge(ring,next,residual,1,nullptr,nullptr,divisor,1,&negative.v,q);
            residual=std::move(next);
        }
        for(auto entry : {std::make_pair(a,&cofactor),std::make_pair(b,&divisor)}) {
            auto [at,new_form]=forms.try_emplace(entry.first);
            if(new_form) at->second=std::move(*entry.second);
            else {
                const auto& g=at->second;const auto& candidate=*entry.second;
                if(g.m!=candidate.m)return false;
                for(size_t k=0;k<g.size();++k) if(!fmpz_equal(&g.c[k].v,&candidate.c[k].v))return false;
            }
        }
    }
    return true;
}

Polynomial unpack(const Poly& p, const Ring& ring) {
    Polynomial f;
    long powers[MAX_VARIABLES];
    mpz_class lc;
    fmpz_get_mpz(lc.get_mpz_t(),&p.c.front().v);
    for (size_t j=0;j<p.size();++j) {
        ring.unpack(p.m.data()+j*ring.stride,powers);
        auto& q = f.try_emplace(f.begin(),Powers(powers,powers+ring.n),0)->second;
        fmpz_get_mpz(mpq_numref(q.get_mpq_t()),&p.c[j].v);
        if (lc != 1) {
            mpz_set(mpq_denref(q.get_mpq_t()),lc.get_mpz_t());
            q.canonicalize();
        }
    }
    return f;
}
} // namespace

Basis PowerResult::materialize() && {
    if (auto* rows = std::get_if<Basis>(&value_)) return std::move(*rows);
    return static_cast<const PowerResult&>(*this).materialize();
}

Basis PowerResult::materialize() const & {
    if (auto* principal = std::get_if<PrincipalBasis>(&value_)) return principal->materialize();
    if (auto* rows = std::get_if<Basis>(&value_)) return *rows;
    const auto& basis = std::get<PackedBasis>(value_);
    Basis rows;
    rows.reserve(basis.rows.size());
    for (const auto& p : basis.rows) rows.push_back(unpack(p,basis.ring));
    return rows;
}

std::vector<std::string> PowerResult::expressions(const std::vector<std::string>& names) const {
    if (auto* principal = std::get_if<PrincipalBasis>(&value_)) return principal->expressions(names);
    std::vector<std::string> result;
    if (auto* basis = std::get_if<Basis>(&value_)) {
        for (const auto& f : *basis) result.push_back(local::format_polynomial(f,names));
        return result;
    }
    const auto& basis = std::get<PackedBasis>(value_);
    const auto& ring = basis.ring;
    mpq_class coefficient;
    for (const auto& p : basis.rows) {
        // The user-facing expression order is total degree then lex.
        std::vector<size_t> order(p.size());
        std::iota(order.begin(),order.end(),0);
        auto term = [&](size_t i) { return p.m.data()+ring.stride*i; };
        std::sort(order.begin(),order.end(),[&](size_t i,size_t j) {
            auto a=term(i), b=term(j);
            if (a[0]!=b[0]) return a[0]>b[0];
            for (int axis=0; axis<ring.n; ++axis) {
                const auto u=Ring::field(a,ring.position(axis)),v=Ring::field(b,ring.position(axis));
                if (u!=v) return u>v;
            }
            return false;
        });
        std::string text;
        for (size_t i : order) {
            fmpz_get_mpz(mpq_numref(coefficient.get_mpq_t()),&p.c[i].v);
            fmpz_get_mpz(mpq_denref(coefficient.get_mpq_t()),&p.c.front().v);
            coefficient.canonicalize();
            if (coefficient < 0) { text+='-'; coefficient=-coefficient; }
            else if(!text.empty()) text+='+';
            bool factor=coefficient!=1 || !term(i)[0];
            if(factor) text+=coefficient.get_str();
            for(int axis=0;axis<ring.n;++axis) {
                auto e=Ring::field(term(i),ring.position(axis));
                if(!e) continue;
                if(factor) text+='*';
                text+=names[axis];
                if(e>1) text+='^'+std::to_string(e);
                factor=true;
            }
        }
        result.push_back(std::move(text));
    }
    return result;
}

std::optional<PackedBasis> packed_power(const Basis& rows, size_t width, bool lex, const mpz_class& exponent,
                                        bool bounded) {
    if (width > MAX_VARIABLES || exponent > MAX_DEGREE) return std::nullopt;
    try {
        PowerBasisEngine engine;
        engine.R.init(static_cast<int>(width),lex ? 1 : 0);
        std::vector<Poly> generators, products;
        for (const auto& f : rows) generators.push_back(pack(f,engine.R));
        // Reuse each suffix-free prefix when enumerating nondecreasing
        // tuples of generators; this visits each unordered product once.
        const auto degree = exponent.get_ui();
        auto extend = [&](auto&& self, const Poly& prefix, size_t first, unsigned long left) -> void {
            for (size_t i=first; i<generators.size(); ++i) {
                auto product = multiply(engine.R,prefix,generators[i]);
                if (left == 1) products.push_back(std::move(product));
                else self(self,product,i,left-1);
            }
        };
        if (degree == 1) products = std::move(generators);
        else if (degree <= 64) {
            for (size_t i=0; i<generators.size(); ++i) extend(extend,generators[i],i,degree-1);
        } else return std::nullopt;
        if (auto dense = plane_products(engine.R,generators,products,degree)) {
            return PackedBasis{engine.R,std::move(*dense)};
        }
        if (triangular_products(engine.R,generators)) {
            if (auto echelon=echelon_products(engine.R,products,0,0)) return PackedBasis{engine.R,std::move(*echelon)};
        }
        if (!engine.compute_products(products,bounded)) return std::nullopt;
        PackedBasis out{engine.R,{}};
        out.rows.reserve(engine.G.size());
        for (auto& g : engine.G) out.rows.push_back(std::move(g.p));
        return out;
    } catch (const Limit&) {
        return std::nullopt;
    }
}
} // namespace atlas::ideal_powers
