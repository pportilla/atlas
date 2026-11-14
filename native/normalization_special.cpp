// Closed-form normalizations over QQ. A declined case goes through the general kernel.
#include "normalization.hpp"
#include "primary_decomposition.hpp"
#include "sparse_flint.hpp"
#include <flint/fmpz_mat.h>
#include <algorithm>

namespace atlas::normalization {
namespace {
using local::Powers;

Polynomial scalar(size_t width, const mpq_class& value = 1) {
    return value == 0 ? Polynomial{} : Polynomial{{Powers(width), value}};
}

Polynomial variable(size_t width, size_t i, mpz_class exponent = 1) {
    Powers e(width);
    e[i] = std::move(exponent);
    return {{std::move(e), 1}};
}

void add(Polynomial& p, const Polynomial& q, const mpq_class& scale = 1) {
    for (const auto& [e,c] : q) {
        auto entry = p.try_emplace(e, 0).first;
        entry->second += scale*c;
        if (entry->second == 0) p.erase(entry);
    }
}

std::string fresh(const std::vector<std::string>& names) {
    for (size_t i = 1;; ++i) {
        auto name = "n" + std::to_string(i);
        if (std::find(names.begin(), names.end(), name) == names.end()) return name;
    }
}

Presentation identity(const Basis& p, const std::vector<std::string>& names) {
    Presentation out{names,p,{},{}};
    for (size_t i = 0; i < names.size(); ++i) {
        out.images.push_back(variable(names.size(),i));
        out.fractions.push_back({out.images.back(),scalar(names.size())});
    }
    return out;
}

struct IntegerMatrix {
    fmpz_mat_t data;
    IntegerMatrix(size_t n, size_t m) { fmpz_mat_init(data,n,m); }
    ~IntegerMatrix() { fmpz_mat_clear(data); }
    IntegerMatrix(const IntegerMatrix&) = delete;
};

// A positive rank-one monomial curve has normalization QQ[t]. The primitive
// weight vector gives the images, and Bezout gives t in its function field.
std::optional<Presentation> monomial_curve(const Basis& prime, const std::vector<std::string>& names) {
    const size_t n = names.size();
    std::vector<bool> active(n);
    for (const auto& row : prime) {
        if (row.size() != 2 || row.begin()->second != -row.rbegin()->second) return std::nullopt;
        for (const auto& [e,c] : row)
            for (size_t j = 0; j < n; ++j) if (e[j] != 0) active[j] = true;
    }
    std::vector<size_t> axes;
    for (size_t j = 0; j < n; ++j) if (active[j]) axes.push_back(j);
    if (axes.size() < 2 || prime.size() < axes.size()-1) return std::nullopt;

    IntegerMatrix matrix(prime.size(),axes.size()), nullspace(axes.size(),axes.size());
    for (size_t i = 0; i < prime.size(); ++i) {
        const auto& row = prime[i];
        for (size_t j = 0; j < axes.size(); ++j) {
            mpz_class diff = row.begin()->first[axes[j]] - row.rbegin()->first[axes[j]];
            fmpz_set_mpz(fmpz_mat_entry(matrix.data,i,j),diff.get_mpz_t());
        }
    }
    if (fmpz_mat_nullspace(nullspace.data,matrix.data) != 1) return std::nullopt;
    Powers weights(axes.size()), bezout(axes.size());
    mpz_class common;
    for (size_t j = 0; j < axes.size(); ++j) {
        fmpz_get_mpz(weights[j].get_mpz_t(),fmpz_mat_entry(nullspace.data,j,0));
        if (!weights[j] || (weights[j] > 0) != (weights.front() > 0)) return std::nullopt;
        mpz_gcd(common.get_mpz_t(),common.get_mpz_t(),weights[j].get_mpz_t());
    }
    if (weights.front() < 0) common = -common;
    for (auto& w : weights) w /= common;
    mpz_class gcd = 0;
    for (size_t j = 0; j < axes.size() && gcd != 1; ++j) {
        mpz_class next, a, b;
        mpz_gcdext(next.get_mpz_t(),a.get_mpz_t(),b.get_mpz_t(),gcd.get_mpz_t(),weights[j].get_mpz_t());
        for (size_t k = 0; k < j; ++k) bezout[k] *= a;
        bezout[j] = b;
        gcd = std::move(next);
    }

    Powers num(n), den(n);
    Presentation out;
    out.names.push_back(fresh(names));
    for (size_t j = 0; j < n; ++j) if (!active[j]) out.names.push_back(names[j]);
    const size_t m = out.names.size();
    out.images.resize(n);
    for (size_t j = 0; j < axes.size(); ++j) {
        const auto i = axes[j];
        if (bezout[j] < 0) den[i] = -bezout[j]; else num[i] = bezout[j];
        out.images[i] = variable(m,0,weights[j]);
    }
    out.fractions.push_back({{{std::move(num),1}}, {{std::move(den),1}}});
    size_t next = 1;
    for (size_t j = 0; j < n; ++j) if (!active[j]) {
        out.images[j] = variable(m,next++);
        out.fractions.push_back({variable(n,j),scalar(n)});
    }
    return out;
}

// Eliminate a coordinate with a constant, nonzero linear coefficient if it
// occurs only in that one term of the isolated target equation.
void simplify_graph(Presentation& out) {
    const size_t n = out.names.size();
    const auto& f = out.relations.front();
    for (size_t i = 0; i < n; ++i) {
        Powers v(n);
        v[i] = 1;
        const auto linear = f.find(v);
        if (linear == f.end() || std::any_of(f.begin(),f.end(),[&](const auto& term) {
                return term.first[i] != 0 && term.first != v;
            })) continue;
        Polynomial image;
        for (const auto& [e,c] : f) if (e != v) {
            auto powers = e;
            powers.erase(powers.begin()+i);
            image.emplace(std::move(powers),-c/linear->second);
        }
        std::map<mpz_class,Polynomial> cache;
        for (auto& g : out.images) g = primary::substitute(g,i,image,n-1,cache);
        out.names.erase(out.names.begin()+i);
        out.fractions.erase(out.fractions.begin()+i);
        out.relations.clear();
        return;
    }
}

// Complete the square y'^2 = d^2 g. For square-free g, the monic hypersurface
// z^2=g is Cohen-Macaulay and regular in codimension one, hence normal.
std::optional<Presentation> quadratic_cover(const Basis& prime, const std::vector<std::string>& names) {
    if (prime.size() != 1) return std::nullopt;
    const auto& f = prime.front();
    const size_t n = names.size();
    for (size_t i = 0; i < n; ++i) {
        Powers square(n);
        square[i] = 2;
        const auto leading = f.find(square);
        if (leading == f.end()) continue;
        Polynomial half_linear, constant;
        bool quadratic = true;
        for (const auto& [e,c] : f) {
            if (e == square) continue;
            if (e[i] > 1) { quadratic = false; break; }
            if (e[i] == 0) constant.emplace(e,c/leading->second);
            else {
                Powers p(e);
                p[i] = 0;
                half_linear.emplace(std::move(p), c/(2*leading->second));
            }
        }
        if (!quadratic) continue;
        Polynomial discriminant = primary::multiply(half_linear,half_linear);
        add(discriminant,constant,-1);
        if (discriminant.empty()) continue;
        // The supplied ideal is prime, so its nonzero discriminant is a nonsquare.
        auto d = scalar(n);
        for (const auto& [p,e] : primary::factors(discriminant,n))
            if (e > 1) d = primary::multiply(d,primary::power(p,e/2,n));
        if (d == scalar(n)) return identity(prime,names);
        local::SparseLexRing ring(n);
        PolyValue a(ring.ctx), divisor(ring.ctx), quotient(ring.ctx);
        ring.write(a.p,discriminant);
        ring.write(divisor.p,primary::multiply(d,d));
        if (!fmpq_mpoly_divides(quotient.p,a.p,divisor.p,ring.ctx))
            throw std::logic_error("Inexact square-factor division during normalization.");
        auto out = identity(prime,names);
        out.names[i] = fresh(names);
        auto relation = variable(n,i,2);
        add(relation,ring.read(quotient.p),-1);
        out.relations = {std::move(relation)};
        auto y = variable(n,i);
        out.images[i] = primary::multiply(d,y);
        add(out.images[i],half_linear,-1);
        add(y,half_linear);
        out.fractions[i] = {std::move(y),std::move(d)};
        simplify_graph(out);
        return out;
    }
    return std::nullopt;
}
// For L=K(W), W^m=F, over the polynomial ring S with field of fractions K,
// the integral closure has S-basis W^j/D_j, D_j=prod p^floor(j*v_p(F)/m).
// After adjoining roots of unity the projections onto the homogeneous parts
// preserve integrality. Each part is integral iff its m-th power lies in S.
// Faithfully flat scalar extension gives the same criterion over QQ.

Polynomial drop(const Polynomial& f, size_t i) {
    Polynomial out;
    for (const auto& [e,c] : f) {
        auto v = e;
        v.erase(v.begin()+i);
        out.emplace(std::move(v),c);
    }
    return out;
}

Polynomial resize(const Polynomial& f, size_t width) {
    Polynomial out;
    for (const auto& [e,c] : f) {
        auto v = e;
        v.resize(width);
        out.emplace(std::move(v),c);
    }
    return out;
}

std::optional<Presentation> cyclic_cover(const Basis& prime, const std::vector<std::string>& names) {
    if (prime.size() != 1) return std::nullopt;
    const size_t n = names.size();
    const auto& f = prime.front();
    for (size_t i = 0; i < n; ++i) {
        mpz_class degree = 0;
        for (const auto& term : f) degree = std::max(degree,term.first[i]);
        // The multiplication table has quadratic size; use the general method
        // instead for large degrees. Binomial curves already have a direct map.
        if (degree <= 2 || degree > 16) continue;
        const size_t m = degree.get_ui();
        Powers leading(n);
        leading[i] = m;
        auto it = f.find(leading);
        if (it == f.end()) continue;
        if (std::any_of(f.begin(),f.end(),[&](const auto& t) {
                return t.first[i] == m && t.first != leading;
            })) continue;
        const mpq_class lc = it->second;
        Polynomial shift;
        for (const auto& [e,c] : f) if (e[i] == m-1) {
            Powers v(e);
            v[i] = 0;
            shift.emplace(std::move(v),c/(m*lc));
        }
        // The next coefficient must be binomial(m,2)*shift^2. Check its
        // leading term before expanding any power of a possibly dense shift.
        if (!shift.empty()) {
            auto head = shift.rbegin()->first;
            for (auto& k : head) k *= 2;
            head[i] = m-2;
            const auto c = f.find(head);
            if (c == f.end() || c->second != lc*(m*(m-1)/2)*shift.rbegin()->second*shift.rbegin()->second)
                continue;
        }
        Polynomial w = variable(n,i);
        add(w,shift);
        auto radicand = primary::power(w,m,n);
        add(radicand,f,-1/lc);
        if (radicand.empty() || std::any_of(radicand.begin(),radicand.end(),[i](const auto& t) {
                return t.first[i] != 0;
            })) continue;

        const auto factors = primary::factors(radicand,n);
        if (std::all_of(factors.begin(),factors.end(),[](const auto& p) {return p.second==1;}))
            return identity(prime,names);
        const mpq_class unit = radicand.rbegin()->second;  // lex factors are monic
        // A single repeated branch with gcd(m,e)=1 admits a smaller
        // presentation. If W^m=c*p^e, set t=p^a W^b with am+be=1.
        if (factors.size()==1) {
            const auto& [p,e] = factors.front();
            mpz_class gcd,a,b;
            mpz_gcdext(gcd.get_mpz_t(),a.get_mpz_t(),b.get_mpz_t(),degree.get_mpz_t(),e.get_mpz_t());
            if (gcd==1) {
                auto coeff = [&](const mpz_class& exponent) -> mpq_class {
                    if (exponent >= 0) return primary::power(scalar(0,unit),exponent,0).begin()->second;
                    return 1/primary::power(scalar(0,unit),-exponent,0).begin()->second;
                };
                auto out = identity(prime,names);
                out.names[i] = fresh(names);
                auto rel = variable(n,i,degree);
                add(rel,p,-coeff(b));
                out.relations = {std::move(rel)};
                out.images[i].clear();
                add(out.images[i],variable(n,i,e),coeff(a));
                add(out.images[i],shift,-1);
                Fraction t{scalar(n),scalar(n)};
                for (const auto& v : {std::make_pair(&p,a),std::make_pair(static_cast<const Polynomial*>(&w),b)}) {
                    auto& side = v.second >= 0 ? t.numerator : t.denominator;
                    side = primary::multiply(side,primary::power(*v.first,abs(v.second),n));
                }
                out.fractions[i] = std::move(t);
                simplify_graph(out);
                return out;
            }
        }
        // Use W^1/D_1 as the first new variable. Keep a higher basis
        // element as a generator only if it cannot be written as a product
        // of earlier ones up to a rational unit.
        auto coefficient = [&](size_t j, size_t k) {
            Polynomial c = scalar(n,j+k >= m ? unit : mpq_class(1));
            for (const auto& [p,e] : factors) {
                mpz_class extra = (j+k)*e/m - j*e/m - k*e/m;
                if (extra != 0) c = primary::multiply(c,p);
            }
            return c;
        };
        std::vector<size_t> factor_of(m),axis(m);
        Presentation out;
        for (size_t h = 0; h < n; ++h) if (h != i) {
            out.names.push_back(names[h]);
            out.fractions.push_back({variable(n,h),scalar(n)});
        }
        auto used_names = names;
        for (size_t j = 1; j < m; ++j) {
            for (size_t k = 1; k <= j/2; ++k) {
                if (coefficient(k,j-k) == scalar(n)) { factor_of[j] = k; break; }
            }
            if (factor_of[j]) continue;
            axis[j] = out.names.size();
            auto name = fresh(used_names);
            out.names.push_back(name);
            used_names.push_back(std::move(name));
            Polynomial d = scalar(n);
            for (const auto& [p,e] : factors)
                d = primary::multiply(d,primary::power(p,j*e/m,n));
            out.fractions.push_back({primary::power(w,j,n),std::move(d)});
        }
        const size_t width = out.names.size();
        auto base = [&](const Polynomial& g) { return resize(drop(g,i),width); };
        Basis element{scalar(width)};
        for (size_t j = 1; j < m; ++j) {
            auto k = factor_of[j];
            element.push_back(k ? primary::multiply(element[k],element[j-k]) : variable(width,axis[j]));
        }
        out.images.resize(n);
        for (size_t h = 0; h < n; ++h) if (h != i) out.images[h] = variable(width,h-(h>i));
        out.images[i] = primary::multiply(base(out.fractions[axis[1]].denominator),element[1]);
        add(out.images[i],base(shift),-1);
        for (size_t j = 1; j < m; ++j) for (size_t k = j; k < m; ++k) {
            // Multiplication by the algebra generators already spans the table.
            if (factor_of[j] && factor_of[k]) continue;
            auto rel = primary::multiply(element[j],element[k]);
            add(rel,primary::multiply(base(coefficient(j,k)),element[(j+k)%m]),-1);
            if (!rel.empty()) out.relations.push_back(std::move(rel));
        }
        // These relations reduce every product to the m basis elements,
        // which are independent in K(W) since W^m-F is irreducible.
        return out;
    }
    return std::nullopt;
}

// A linear change of plane coordinates may hide a monic cyclic cover. If the
// homogeneous part of highest degree is c*(x+s*y)^d, straighten that form.
// y stays fixed, so the inverse and both maps involve polynomial substitution.
std::optional<Presentation> shear_cover(const Basis& prime, const std::vector<std::string>& names) {
    if (names.size() != 2 || prime.size() != 1) return std::nullopt;
    const auto& f = prime.front();
    mpz_class d = 0;
    for (const auto& term : f) {
        mpz_class degree = term.first[0] + term.first[1];
        if (degree > d) d = std::move(degree);
    }
    // Keep speculative substitutions sparse. The general kernel handles the
    // cases where expansion of a high-degree linear power is unattractive.
    if (d < 2 || d > 64) return std::nullopt;
    auto a = f.find({d,0}), b = f.find({d-1,1});
    if (a == f.end() || b == f.end()) return std::nullopt;
    const mpq_class slope = b->second / (a->second*d);
    mpq_class expected = a->second;
    for (unsigned long k = 1; k <= d.get_ui(); ++k) {
        expected *= slope*(d-k+1)/k;
        auto term = f.find({d-k,k});
        if (term == f.end() || term->second != expected) return std::nullopt;
    }
    auto shear = [](const Polynomial& f, const mpq_class& s) {
        // Embed (x,y) in (dummy,y,X), then replace dummy with X+s*y.
        Polynomial lifted;
        for (const auto& [e,c] : f) lifted.emplace(Powers{e[0],e[1],0},c);
        Polynomial image = variable(2,1);
        add(image,variable(2,0),s);
        std::map<mpz_class,Polynomial> cache;
        auto in_y_X = primary::substitute(lifted,0,image,2,cache);
        Polynomial in_X_y;
        for (const auto& [e,c] : in_y_X) in_X_y.emplace(Powers{e[1],e[0]},c);
        return in_X_y;
    };
    const Basis lifted{shear(f,-slope)};
    auto out = monomial_curve(lifted,names);
    if (!out) out = quadratic_cover(lifted,names);
    if (!out) out = cyclic_cover(lifted,names);
    if (!out) return std::nullopt;
    add(out->images[0],out->images[1],-slope);
    for (auto& fraction : out->fractions) {
        fraction.numerator = shear(fraction.numerator,slope);
        fraction.denominator = shear(fraction.denominator,slope);
    }
    return out;
}

} // namespace

std::optional<Presentation> special(const Basis& prime, const std::vector<std::string>& names, bool) {
    if (auto curve = monomial_curve(prime,names)) return curve;
    if (auto quadratic = quadratic_cover(prime,names)) return quadratic;
    if (auto cover = cyclic_cover(prime,names)) return cover;
    return shear_cover(prime,names);
}
} // namespace atlas::normalization
