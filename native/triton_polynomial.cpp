// Exact polynomial coefficients on Triton. GMP keeps exponents, denominators
// and carry propagation unbounded. FLINT supplies the ordinary CPU path.
#include "local_io.hpp"
#include "sparse_flint.hpp"
#include "quotient_ring.hpp"
#include <algorithm>
#include <cstdint>
#include <sstream>

namespace atlas {
namespace {
using local::Polynomial;
using local::Powers;
using Integers = std::vector<mpz_class>;
enum Work { Add, OuterProduct, PointwiseProduct, DenseConvolution };
using Callback = int (*)(int, int, int, int, int, int, const int64_t*, const int64_t*, int64_t*);
struct Declined {};
constexpr size_t max_words = size_t(1) << 22;
constexpr int digit_bits = 15;

size_t bits(const mpz_class& z) { return mpz_sizeinbase(z.get_mpz_t(), 2); }
size_t largest(const Integers& values) {
    size_t result = 1;
    for (const auto& z : values) result = std::max(result, bits(z));
    return result;
}

std::vector<int64_t> pack(const Integers& z, size_t width, bool single) {
    std::vector<int64_t> packed(z.size() * width);
    for (size_t i = 0; i < z.size(); ++i) {
        auto* words = packed.data() + i * width;
        if (single) {
            words[0] = mpz_get_si(z[i].get_mpz_t());
        } else {
            size_t used = 0;
            // GMP exports the magnitude, low limb first. Keeping the sign in
            // each digit lets additions and products share one carry pass.
            mpz_export(words, &used, -1, sizeof(*words), 0, 64-digit_bits, z[i].get_mpz_t());
            if (z[i] < 0) for (size_t k=0; k<used; ++k) words[k] = -words[k];
        }
    }
    return packed;
}

Integers calculate(Callback callback, Work op, const Integers& a, const Integers& b) {
    size_t xbits = largest(a), ybits = largest(b);
    bool single = op == Add ? std::max(xbits, ybits) <= 61 : xbits+ybits <= 62;
    size_t aw = single ? 1 : (xbits + digit_bits-1) / digit_bits;
    size_t bw = single ? 1 : (ybits + digit_bits-1) / digit_bits;
    if (op == Add) aw = bw = std::max(aw,bw);
    size_t ow = op == Add || single ? std::max(aw,bw) : aw+bw-1;
    size_t count = a.size();
    if (op == OuterProduct) {
        if (count > max_words / b.size()) throw Declined{};
        count *= b.size();
    } else if (op == DenseConvolution) count += b.size()-1;
    // The cap also bounds each wide convolution accumulator by
    // max_words * (2^15-1)^2 < 2^53.
    if (a.size() > max_words/aw || b.size() > max_words/bw || count > max_words/ow)
        throw Declined{};
    auto aa = pack(a,aw,single), bb = pack(b,bw,single);
    std::vector<int64_t> out(count*ow);
    if (!callback(op, a.size(), b.size(), aw, bw, ow, aa.data(), bb.data(), out.data()))
        throw Declined{};
    Integers result(count);
    for (size_t j=0; j<count; ++j) {
        auto& z = result[j];
        z = static_cast<long>(out[(j+1)*ow-1]);
        for (size_t k=ow-1; k>0; --k) {
            z <<= digit_bits;
            z += static_cast<long>(out[j*ow+k-1]);
        }
    }
    return result;
}

mpz_class denominator(const Polynomial& f) {
    mpz_class d = 1;
    for (const auto& term : f)
        mpz_lcm(d.get_mpz_t(),d.get_mpz_t(),mpq_denref(term.second.get_mpq_t()));
    return d;
}

mpz_class numerator(const mpq_class& q, const mpz_class& d) {
    if (q.get_den() == d) return q.get_num();
    return (d / q.get_den()) * q.get_num();
}

void insert(Polynomial& p, Powers e, const mpz_class& c, const mpz_class& d) {
    if (c == 0) return;
    mpq_class q(c,d);
    q.canonicalize();
    auto [it, fresh] = p.try_emplace(std::move(e), q);
    if (!fresh) {
        it->second += q;
        if (it->second == 0) p.erase(it);
    }
}

Polynomial sum(Callback cb, const Polynomial& a, const Polynomial& b, bool subtract=false) {
    if (b.empty()) return a;
    mpz_class d;
    mpz_lcm(d.get_mpz_t(),denominator(a).get_mpz_t(),denominator(b).get_mpz_t());
    std::vector<Powers> exponents;
    Integers left,right;
    auto i = a.begin(), j = b.begin();
    while (i != a.end() || j != b.end()) {
        auto e = i == a.end() ? j->first : j == b.end() ? i->first : std::min(i->first,j->first);
        left.emplace_back(i != a.end() && i->first == e ? numerator((i++)->second,d) : mpz_class(0));
        right.emplace_back(j != b.end() && j->first == e ? numerator((j++)->second,d) : mpz_class(0));
        if (subtract) right.back() = -right.back();
        exponents.push_back(std::move(e));
    }
    Integers values=calculate(cb,Add,left,right);
    Polynomial answer;
    for (size_t k=0; k<values.size(); ++k) insert(answer,std::move(exponents[k]),values[k],d);
    return answer;
}

bool consecutive(const Polynomial& p) {
    return p.begin()->first.size() == 1 && p.rbegin()->first[0] - p.begin()->first[0] == p.size()-1;
}

Polynomial multiply(Callback cb, const Polynomial& a, const Polynomial& b) {
    if (a.empty() || b.empty()) return {};
    mpz_class da=denominator(a), db=denominator(b), d=da*db;
    Integers left, right;
    for (const auto& term:a) left.emplace_back(numerator(term.second,da));
    for (const auto& term:b) right.emplace_back(numerator(term.second,db));
    Polynomial answer;
    if (consecutive(a) && consecutive(b) && std::min(a.size(),b.size()) > 1 &&
        largest(left)+largest(right)+bits(mpz_class(std::min(a.size(),b.size()))) <= 62) {
        // Above this size FLINT's subquadratic multiplication also avoids a
        // long-running device loop for each coefficient.
        if (a.size()>max_words/b.size()) throw Declined{};
        auto values=calculate(cb,DenseConvolution,left,right);
        Powers e{a.begin()->first[0]+b.begin()->first[0]};
        for (const auto& v:values) { insert(answer,e,v,d); ++e[0]; }
        return answer;
    }
    auto values=calculate(cb,OuterProduct,left,right);
    size_t pos=0;
    for (const auto& x:a) for (const auto& y:b) {
        Powers e=x.first;
        for (size_t i=0;i<e.size();++i) e[i]+=y.first[i];
        insert(answer,std::move(e),values[pos++],d);
    }
    return answer;
}

Polynomial constant(size_t n, const mpq_class& c) {
    return c == 0 ? Polynomial{} : Polynomial{{Powers(n),c}};
}

Polynomial power(Callback cb, Polynomial f, mpz_class exponent, size_t n) {
    auto answer=constant(n,1);
    if (!exponent) return answer;
    if (f.empty()) return {};
    if (f.size()==1 && (f.begin()->second==1 || f.begin()->second==-1)) {
        auto e=f.begin()->first;
        for (auto& v:e) v*=exponent;
        return {{std::move(e),mpz_odd_p(exponent.get_mpz_t()) ? f.begin()->second : 1}};
    }
    bool first=true;
    while (exponent>0) {
        if (mpz_odd_p(exponent.get_mpz_t())) {
            answer=first ? f : multiply(cb,answer,f);
            first=false;
        }
        exponent >>= 1;
        if (exponent>0) f=multiply(cb,f,f);
    }
    return answer;
}

Polynomial derivative(Callback cb, const Polynomial& f, size_t j) {
    auto d=denominator(f);
    std::vector<Powers> exponents;
    Integers coeffs,orders;
    for (const auto& [powers,c]:f) if (powers[j]>0) {
        exponents.push_back(powers);
        --exponents.back()[j];
        coeffs.emplace_back(numerator(c,d));
        orders.push_back(powers[j]);
    }
    if (coeffs.empty()) return {};
    auto values=calculate(cb,PointwiseProduct,coeffs,orders);
    Polynomial answer;
    for (size_t i=0;i<values.size();++i) insert(answer,std::move(exponents[i]),values[i],d);
    return answer;
}

// A monomial substitution leaves one candidate term per source term. Reuse
// each distinct scalar power and apply it to every coefficient in one batch.
// Collisions (including evaluation into QQ) are collected exactly in GMP.
mpq_class rational_power(mpq_class x, mpz_class e) {
    if (!e || x==1) return 1;
    if (x==-1) return mpz_odd_p(e.get_mpz_t()) ? -1 : 1;
    mpq_class value=1;
    while (e>0) {
        if (mpz_odd_p(e.get_mpz_t())) value*=x;
        e >>= 1;
        if (e>0) x*=x;
    }
    return value;
}

Polynomial monomial_substitution(Callback cb, const Polynomial& f, const local::Basis& images, size_t n) {
    auto d=denominator(f);
    std::vector<const Powers*> sources;
    Integers values;
    for (const auto& [e,c]:f) {
        bool vanishes=false;
        for (size_t j=0;j<e.size();++j) if (images[j].empty() && e[j]!=0) { vanishes=true; break; }
        if (!vanishes) {
            sources.push_back(&e);
            values.emplace_back(numerator(c,d));
        }
    }
    if (values.empty()) return {};
    std::vector<Powers> out(values.size(),Powers(n));
    for (size_t j=0;j<images.size();++j) {
        if (images[j].empty()) continue;
        const auto& [monomial, scalar] = *images[j].begin();
        Integers exponents;
        for (auto e:sources) exponents.push_back((*e)[j]);
        for (size_t k=0;k<n;++k) if (monomial[k]!=0) {
            const auto terms=monomial[k]==1 ? exponents :
                    calculate(cb,PointwiseProduct,exponents,Integers(exponents.size(),monomial[k]));
            for (size_t i=0;i<out.size();++i) out[i][k]+=terms[i];
        }
        if (scalar==1) continue;
        std::map<mpz_class, mpq_class> weights;
        mpz_class shared=1;
        for (const auto& e:exponents) {
            auto [it,fresh]=weights.try_emplace(e);
            if (fresh) {
                it->second=rational_power(scalar,e);
                mpz_lcm(shared.get_mpz_t(),shared.get_mpz_t(),mpq_denref(it->second.get_mpq_t()));
            }
        }
        Integers factors;
        for (const auto& e:exponents) factors.emplace_back(numerator(weights.at(e),shared));
        values=calculate(cb,PointwiseProduct,values,factors);
        d*=shared;
    }
    Polynomial answer;
    for (size_t i=0;i<out.size();++i) insert(answer,std::move(out[i]),values[i],d);
    return answer;
}

Polynomial compose(Callback cb, const Polynomial& f, const local::Basis& images, size_t n) {
    if (std::all_of(images.begin(),images.end(),[](const auto& image){return image.size()<=1;}))
        return monomial_substitution(cb,f,images,n);
    std::vector<std::map<mpz_class, Polynomial>> cached(images.size());
    Polynomial answer;
    for (const auto& [e,c]:f) {
        auto term=constant(n,c);
        for (size_t j=0;j<e.size() && !term.empty();++j) if (e[j]!=0) {
            auto [it,inserted]=cached[j].try_emplace(e[j]);
            if (inserted) it->second=power(cb,images[j],e[j],n);
            term=multiply(cb,term,it->second);
        }
        answer=answer.empty() ? std::move(term) : sum(cb,answer,term);
    }
    return answer;
}

Polynomial cpu(const std::string& op, size_t n, size_t m, const Polynomial& f,
               const local::Basis& rest, const mpz_class& param) {
    local::SparseLexRing source(n),target(m);
    PolyValue a(source.ctx),b(source.ctx),out(target.ctx);
    source.write(a.p,f);
    if (op=="substitute") {
        std::vector<PolyValue> images;
        std::vector<fmpq_mpoly_struct*> args;
        images.reserve(rest.size());
        for (const auto& g:rest) {
            images.emplace_back(target.ctx);
            target.write(images.back().p,g);
            args.push_back(images.back().p);
        }
        if (!fmpq_mpoly_compose_fmpq_mpoly(out.p,a.p,args.data(),source.ctx,target.ctx))
            throw std::runtime_error("The polynomial composition exceeds available storage.");
    } else if (op=="differentiate") {
        fmpq_mpoly_derivative(out.p,a.p,param.get_ui(),source.ctx);
    } else if (op=="power") {
        Z exponent;
        fmpz_set_mpz(&exponent.v,param.get_mpz_t());
        if (!fmpq_mpoly_pow_fmpz(out.p,a.p,&exponent.v,source.ctx))
            throw std::runtime_error("The polynomial power exceeds available storage.");
    } else {
        source.write(b.p,rest.front());
        if (op=="add") fmpq_mpoly_add(out.p,a.p,b.p,source.ctx);
        else if (op=="subtract") fmpq_mpoly_sub(out.p,a.p,b.p,source.ctx);
        else if (op=="multiply") fmpq_mpoly_mul(out.p,a.p,b.p,source.ctx);
        else if (op=="divide") fmpq_mpoly_scalar_div_fmpq(out.p,a.p,b.p->content,source.ctx);
    }
    return target.read(out.p);
}

// If every coordinate maps to a monomial over K, transform its exponent
// vector on the host and multiply all resulting power-basis coefficients in
// one device batch. Computing scalar powers in K also handles huge exponents.
std::optional<Polynomial> marked_monomial_substitution(Callback cb, const Polynomial& f,
        const local::Basis& images, const Polynomial& relation, size_t width) {
    std::vector<Powers> monomials;
    local::Basis scalars;
    for (const auto& image:images) {
        Powers powers(width);
        Polynomial coefficient;
        for (const auto& [e,c]:image) {
            auto axes=e;
            axes.back()=0;
            if (!coefficient.empty() && axes!=powers) return std::nullopt;
            powers=std::move(axes);
            coefficient.emplace(Powers{e.back()},c);
        }
        monomials.push_back(std::move(powers));
        scalars.push_back(std::move(coefficient));
    }
    Polynomial minimal;
    for (const auto& [e,c]:relation) minimal.emplace(Powers{e.back()},c);
    quotient_ring::Ring field({minimal},1,true);
    const auto one=constant(1,1);
    std::vector<std::map<mpz_class,Polynomial>> powers(images.size());
    struct Term { Powers e; mpq_class source, weight; };
    std::vector<Term> terms;
    const auto da=denominator(f);
    mpz_class db=1;
    for (const auto& [e,c]:f) {
        Powers axes(width);
        Polynomial weight{{Powers{e.back()},1}};
        for (size_t j=0; j<images.size() && !weight.empty(); ++j) if (e[j]!=0) {
            for (size_t k=0;k<width-1;++k) axes[k]+=e[j]*monomials[j][k];
            if (scalars[j]==one) continue;
            auto [it,fresh]=powers[j].try_emplace(e[j]);
            if (fresh) it->second=std::get<Polynomial>(field.evaluate("power",{scalars[j]},e[j]));
            weight=weight==one ? it->second : std::get<Polynomial>(field.evaluate("multiply",{weight,it->second}));
        }
        for (const auto& [v,factor]:weight) {
            if (cb && terms.size()==max_words) throw Declined{};
            axes.back()=v[0];
            terms.push_back({axes,c,factor});
            mpz_lcm(db.get_mpz_t(),db.get_mpz_t(),mpq_denref(factor.get_mpq_t()));
        }
    }
    Polynomial result;
    if (terms.empty()) return result;
    Integers lhs,rhs;
    for (const auto& term:terms) {
        lhs.emplace_back(numerator(term.source,da));
        rhs.emplace_back(numerator(term.weight,db));
    }
    if (cb) lhs=calculate(cb,PointwiseProduct,lhs,rhs);
    else for (size_t j=0;j<terms.size();++j) lhs[j]*=rhs[j];
    const mpz_class d=da*db;
    for (size_t j=0;j<terms.size();++j) insert(result,std::move(terms[j].e),lhs[j],d);
    return result;
}

// The last exponent is the scalar generator. Reduce after each multiplication
// so powering and composition keep representatives in the field's power basis.
Polynomial marked_operation(const std::string& op, Callback callback, size_t width,
                            const Polynomial& relation, const Polynomial& input,
                            const local::Basis& operands, const mpz_class& parameter) {
    if (!width || relation.empty() || relation.rbegin()->first.back()<2)
        throw std::invalid_argument("Expected a defining polynomial for the coefficient field.");
    auto scalar=[](const Polynomial& f) {
        for (const auto& [e,c]:f)
            if (std::any_of(e.begin(),e.end()-1,[](const auto& power){return power!=0;})) return false;
        return true;
    };
    if (!scalar(relation)) throw std::invalid_argument("The field relation must use only the scalar generator.");
    quotient_ring::Ring ring({relation},width,true);
    const auto& field_degree=relation.rbegin()->first.back();
    auto reduce=[&](const Polynomial& value) {
        if (std::all_of(value.begin(),value.end(),[&](const auto& term){return term.first.back()<field_degree;}))
            return value;
        return std::get<Polynomial>(ring.evaluate("reduce",{value}));
    };
    const auto f=reduce(input);
    local::Basis rest;
    for (const auto& value:operands) rest.push_back(reduce(value));
    if (op=="differentiate") {
        if (parameter>=width-1) throw std::invalid_argument("Choose a polynomial coordinate to differentiate.");
        return callback ? derivative(callback,f,parameter.get_ui()) : cpu(op,width,width,f,{},parameter);
    }
    if (op=="divide" && (rest.front().empty() || !scalar(rest.front())))
        throw std::invalid_argument("Division requires a nonzero element of the coefficient field.");
    if (!callback && op!="substitute") {
        local::Basis args{f};
        args.insert(args.end(),rest.begin(),rest.end());
        return std::get<Polynomial>(ring.evaluate(op,args,parameter));
    }
    auto product=[&](const Polynomial& a,const Polynomial& b) {
        if (!callback) return std::get<Polynomial>(ring.evaluate("multiply",{a,b}));
        return reduce(multiply(callback,a,b));
    };
    auto power_mod=[&](Polynomial base, mpz_class k) {
        if (!callback) return std::get<Polynomial>(ring.evaluate("power",{base},k));
        Polynomial value=constant(width,1);
        bool first=true;
        while (k>0) {
            if (mpz_odd_p(k.get_mpz_t())) {
                value=first ? base : product(value,base);
                first=false;
            }
            k >>= 1;
            if (k!=0) base=product(base,base);
        }
        return value;
    };
    if (op=="multiply") return product(f,rest.front());
    if (op=="divide") return product(f,std::get<Polynomial>(ring.evaluate("inverse",{rest.front()})));
    if (op=="add" || op=="subtract") return sum(callback,f,rest.front(),op=="subtract");
    if (op=="power") return power_mod(f,parameter);
    if (auto result=marked_monomial_substitution(callback,f,rest,relation,width)) return std::move(*result);
    // Coefficients are fixed by substitution. Group the field terms of each
    // coefficient before multiplying by the images of its geometric monomial.
    std::map<Powers,Polynomial> coefficients;
    for (const auto& [e,c]:f) {
        Powers axes=e, field_power(width);
        field_power.back()=axes.back();
        axes.pop_back();
        coefficients[axes].emplace(std::move(field_power),c);
    }
    std::vector<std::map<mpz_class,Polynomial>> powers(rest.size());
    Polynomial answer;
    for (auto& [e,term]:coefficients) {
        for (size_t j=0; j<rest.size() && !term.empty(); ++j) if (e[j]!=0) {
            auto [it,fresh]=powers[j].try_emplace(e[j]);
            if (fresh) it->second=power_mod(rest[j],e[j]);
            term=product(term,it->second);
        }
        if (answer.empty()) answer=std::move(term);
        else if (callback) answer=sum(callback,answer,term);
        else answer=std::get<Polynomial>(ring.evaluate("add",{answer,term}));
    }
    return answer;
}

} // namespace

extern "C" char* atlas_polynomial_arithmetic(const char* payload,size_t length,Callback callback) noexcept {
    try {
        std::istringstream in(std::string(payload,length));
        std::string head,op,parameter, line;
        std::getline(in,head);
        std::istringstream header(head);
        size_t n,m;
        if (!(header>>op>>n>>m) || n>static_cast<size_t>(LONG_MAX) || m>static_cast<size_t>(LONG_MAX) ||
            (op!="substitute" && n!=m)) throw std::invalid_argument("Invalid polynomial arithmetic dimensions.");
        const bool marked=op.rfind("marked-",0)==0;
        Polynomial relation;
        if (marked) {
            op=op.substr(7);
            if (m!=n || n==0 || !std::getline(in,line))
                throw std::invalid_argument("A marked operation needs a field relation and a common ring.");
            relation=local::decode_row(line,n);
        }
        header>>parameter;
        if (!std::getline(in,line)) throw std::invalid_argument("Missing polynomial operand.");
        auto f=local::decode_row(line,n);
        local::Basis rest;
        while (std::getline(in,line)) rest.push_back(local::decode_row(line,m));
        mpz_class p=0;
        if (op=="power" || op=="differentiate" || op=="scalar-power") {
            if (!rest.empty() || parameter.empty() || mpz_set_str(p.get_mpz_t(),parameter.c_str(),10) ||
                (op!="scalar-power" && p<0) || (op=="differentiate" && p>=n) || (op=="scalar-power" && n))
                throw std::invalid_argument("Invalid exponent or derivative coordinate.");
            if (op=="scalar-power") {
                if (p<0) {
                    if (f.empty()) throw std::domain_error("Division by zero.");
                    f=constant(0,1/f.begin()->second);
                    p=-p;
                }
                op="power";
            }
        } else if (op=="substitute") {
            if (rest.size()!=n-static_cast<size_t>(marked)) throw std::invalid_argument("Expected one image per source coordinate.");
        } else if ((op!="add" && op!="subtract" && op!="multiply" && op!="divide" &&
                    (op!="compare" || n!=0)) || rest.size()!=1)
            throw std::invalid_argument("Invalid polynomial arithmetic operation or operand count.");
        if (!marked && op=="divide" && (rest[0].size()!=1 || rest[0].begin()->first!=Powers(n)))
            throw std::invalid_argument("Division requires a nonzero rational scalar.");
        const bool comparison=op=="compare";
        if (comparison) op="subtract";
        Polynomial answer;
        if (marked) answer=marked_operation(op,callback,n,relation,f,rest,p);
        else if (!callback) answer=cpu(op,n,m,f,rest,p);
        else if (op=="add" || op=="subtract") answer=sum(callback,f,rest.front(),op=="subtract");
        else if (op=="multiply") answer=multiply(callback,f,rest.front());
        else if (op=="divide") answer=multiply(callback,f,constant(n,1/rest.front().begin()->second));
        else if (op=="power") answer=power(callback,std::move(f),p,n);
        else if (op=="differentiate") answer=derivative(callback,f,p.get_ui());
        else answer=compose(callback,f,rest,m);
        if (comparison && !answer.empty()) answer=constant(0,sgn(answer.begin()->second));
        return copy_text(local::encode_row(answer));
    } catch (const Declined&) { set_error("Triton coefficient batch declined."); }
      catch (const std::exception& error) { set_error(error.what()); }
      catch (...) { set_error("Polynomial arithmetic failed."); }
    return nullptr;
}
} // namespace atlas
