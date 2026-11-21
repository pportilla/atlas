// Bulk residue arithmetic, simple Hensel lifting and bounded rational reconstruction.
#include "rational_sparse_json.hpp"
#include <cstdint>
#include <limits>

namespace {
using Z = mpz_class;
using Words = std::vector<int64_t>;
using Batch = int (*)(int, int, int64_t, int64_t, int64_t, int, int, const int64_t*, int64_t*);
enum Action { Reduce, Add, Subtract, Multiply, Divide, Inverse, Reconstruct, Hensel, CRT };

Z residue(const Z& a, const Z& m) {
    if (a>=0 && a<m) return a;
    Z r;
    mpz_mod(r.get_mpz_t(), a.get_mpz_t(), m.get_mpz_t());
    return r;
}
Z reciprocal(const Z& a, const Z& m) {
    Z inv;
    if (!mpz_invert(inv.get_mpz_t(), a.get_mpz_t(), m.get_mpz_t()))
        throw std::invalid_argument("The denominator must be invertible modulo the modulus.");
    return inv;
}
std::vector<Z> integers(const Json::Value& array) {
    std::vector<Z> out;
    for (const auto& value : array) out.emplace_back(value.asString(), 10);
    return out;
}

Json::Value calculate(const Json::Value& request, Batch batch) {
    static const std::map<std::string,Action> actions{
        {"reduce",Reduce},{"add",Add},{"subtract",Subtract},{"multiply",Multiply},
        {"divide",Divide},{"inverse",Inverse},{"reconstruct",Reconstruct},{"hensel",Hensel},{"crt",CRT}};
    auto entry = actions.find(request["action"].asString());
    if (entry == actions.end()) throw std::invalid_argument("Unknown modular operation.");
    const auto op = entry->second;
    Z m(request["modulus"].asString(), 10), n=m;
    if (m<2) throw std::invalid_argument("The modulus must be at least two.");
    if (op==CRT) {
        n=Z(request["other_modulus"].asString(),10);
        if (n<2 || gcd(n,m)!=1) throw std::invalid_argument("CRT requires coprime moduli of at least two.");
    }
    const Z bound=op==Reconstruct ? Z(sqrt(Z((m-1)/2))) : Z(0);
    const Z crt_inverse=op==CRT ? reciprocal(m,n) : Z(0);
    const auto inputs=integers(request["values"]), other=integers(request["other"]);
    const size_t count=inputs.size();
    if (((op>=Add && op<=Divide) || op==CRT) && count!=other.size())
        throw std::invalid_argument("Supply equally long operand arrays.");
    const Z result_modulus=op==Hensel ? Z(m*m) : op==CRT ? Z(m*n) : m;
    // Each signed-word product of reduced residues, including a CRT or Hensel lift,
    // fits below 2^62. Wider moduli keep the same arbitrary-precision CPU semantics.
    bool eligible=batch && count && count<=std::numeric_limits<int>::max() && m<=2147483647 && n<=2147483647;
    const int inputs_per_entry = (op==Reduce || op==Reconstruct) ? 1 : (op==Hensel || op==CRT) ? 3 : 2;
    const int outputs_per_entry = (op>=Divide && op!=CRT) ? 2 : 1;
    Words packed, answers;
    if (eligible) {
        packed.resize(count*inputs_per_entry);
        answers.resize(count*outputs_per_entry);
    }
    Json::Value result, values(Json::arrayValue);
    auto operands = [&](size_t i, Z& a, Z& b, Z& c) {
        a=residue(inputs[i],m); b=1; c=0;
        if (op>=Add && op<=Divide) b=residue(other[i],m);
        if (op==Inverse) { b=a; a=1; }
        if (op==CRT) { b=crt_inverse; c=residue(other[i]-a,n); }
        if (op==Hensel) {
            Z f=0, derivative=0;
            for (auto j=other.rbegin(); j!=other.rend(); ++j) {
                derivative=residue(derivative*a+f,m);
                f=residue(f*a+*j,result_modulus);
            }
            if (f%m!=0) throw std::invalid_argument("Every Hensel input must be a root modulo the modulus.");
            b=derivative; c=residue(-(f/m),m);
        }
    };
    auto reconstruction = [&](const Z& a, Z p, Z q) -> Json::Value {
        if (q<0) { p=-p; q=-q; }
        if (q==0 || q>bound || abs(p)>bound || gcd(p,q)!=1 || residue(p-a*q,m)!=0)
            return Json::Value();
        mpq_class value(p,q);
        return value.get_str();
    };

    if (eligible) {
        for (size_t i=0;i<count;++i) {
            Z a,b,c;
            operands(i,a,b,c);
            if (op==Reduce) {
                if (!inputs[i].fits_slong_p()) { eligible=false; break; }
                a=inputs[i]; // The GPU performs the reduction of the original signed input.
            }
            packed[i]=a.get_si();
            if (inputs_per_entry>=2) packed[count+i]=b.get_si();
            if (inputs_per_entry==3) packed[2*count+i]=c.get_si();
        }
    }
    const bool ran=eligible && batch(op,static_cast<int>(count),m.get_si(),n.get_si(),bound.get_si(),
                                     inputs_per_entry,outputs_per_entry,packed.data(),answers.data());
    for (size_t i=0;i<count;++i) {
        if (ran) {
            if (op==Reconstruct) values.append(reconstruction(Z(packed[i]), Z(answers[i]), Z(answers[count+i])));
            else {
                if (outputs_per_entry==2 && !answers[count+i]) throw std::invalid_argument("The denominator must be invertible modulo the modulus.");
                values.append(std::to_string(answers[i]));
            }
            continue;
        }
        Z a,b,c,value;
        operands(i,a,b,c);
        switch (op) {
        case Reduce: value=a; break;
        case Add: value=residue(a+b,m); break;
        case Subtract: value=residue(a-b,m); break;
        case Multiply: value=residue(a*b,m); break;
        case Inverse: case Divide: value=residue(a*reciprocal(b,m),m); break;
        case Hensel: value=a+m*residue(c*reciprocal(b,m),m); break;
        case CRT: value=a+m*residue(c*b,n); break;
        case Reconstruct: {
            Z old=m,r=a,old_t=0,t=1;
            while (r>bound) {
                Z q=old/r, next_r=old-q*r, next_t=old_t-q*t;
                old=r; r=next_r; old_t=t; t=next_t;
            }
            values.append(reconstruction(a,r,t));
            continue;
        }
        }
        values.append(value.get_str());
    }
    result["values"]=std::move(values);
    result["modulus"]=result_modulus.get_str();
    result["device_batch"]=ran;
    return result;
}
} // namespace

extern "C" const char* atlas_modular_batch(const char* request, Batch callback) noexcept {
    return atlas::sparse_json::invoke(request,[&](const Json::Value& data){ return calculate(data,callback); });
}
