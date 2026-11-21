// Arithmetic in F_p[T]/(g) with exact FLINT primality and irreducibility checks.
#include "rational_sparse_json.hpp"
#include <flint/fmpz.h>
#include <flint/fmpz_mod.h>
#include <flint/fmpz_mod_poly.h>
#include <flint/fmpz_mod_poly_factor.h>
#include <algorithm>
#include <charconv>
#include <climits>

namespace {
using Z=mpz_class;
using Row=std::vector<Z>;
using Batch=int (*)(int,int,int,int,int64_t,int64_t,int,int,const int64_t*,int64_t*);

struct Integer {
    fmpz_t value;
    Integer() { fmpz_init(value); }
    Integer(const Integer&) = delete;
    ~Integer() { fmpz_clear(value); }
};
struct Field {
    fmpz_mod_ctx_t ctx;
    explicit Field(const fmpz_t p) { fmpz_mod_ctx_init(ctx,p); }
    Field(const Field&) = delete;
    ~Field() { fmpz_mod_ctx_clear(ctx); }
};
struct Poly {
    fmpz_mod_poly_t value;
    const Field& field;
    explicit Poly(const Field& f) : field(f) { fmpz_mod_poly_init(value,field.ctx); }
    Poly(const Poly&) = delete;
    ~Poly(){ fmpz_mod_poly_clear(value,field.ctx); }
    void set(const Json::Value& row) {
        Integer c;
        fmpz_mod_poly_zero(value,field.ctx);
        for (Json::ArrayIndex j=0;j<row.size();++j) {
            Z z(row[j].asString(),10);
            fmpz_set_mpz(c.value,z.get_mpz_t());
            fmpz_mod_poly_set_coeff_fmpz(value,j,c.value,field.ctx);
        }
    }
    Z coeff(slong i) const {
        Integer c; Z out;
        fmpz_mod_poly_get_coeff_fmpz(c.value,value,i,field.ctx);
        fmpz_get_mpz(out.get_mpz_t(),c.value);
        return out;
    }
    slong length() const { return fmpz_mod_poly_length(value,field.ctx); }
    Json::Value encode() const {
        Json::Value result(Json::arrayValue);
        for (slong i=0;i<length();++i) result.append(coeff(i).get_str());
        return result;
    }
};

Json::Value calculate(const Json::Value& request, Batch batch) {
    static const std::map<std::string,int> actions{{"reduce",0},{"add",1},{"subtract",2},
                                                    {"multiply",3},{"divide",4},{"inverse",5}};
    auto it=actions.find(request["action"].asString());
    if (it==actions.end()) throw std::invalid_argument("Unknown finite extension field operation.");
    int action=it->second;
    Z characteristic(request["modulus"].asString(),10);
    Integer prime;
    fmpz_set_mpz(prime.value,characteristic.get_mpz_t());
    if (characteristic<2 || !fmpz_is_prime(prime.value))
        throw std::invalid_argument("A finite extension field requires a prime characteristic.");
    Field field(prime.value);
    Poly relation(field), a(field), b(field), out(field), inv(field);
    relation.set(request["field_polynomial"]);
    const slong d=relation.length()-1;
    if (d<1 || relation.coeff(d)!=1 || !fmpz_mod_poly_is_irreducible(relation.value,field.ctx))
        throw std::invalid_argument("Supply a monic irreducible field polynomial of positive degree.");
    const auto& values=request["values"]; const auto& other=request["other"];
    const size_t size=values.size();
    bool binary=action>=1 && action<=4;
    if (binary && other.size()!=size) throw std::invalid_argument("Supply equally long operand arrays.");
    const slong columns=action==0 ? 2*d-1 : (binary ? 2*d : d);
    bool eligible=batch && size>0 && d<=8 && characteristic<=INT_MAX &&
                  size<=static_cast<size_t>(INT_MAX/(columns+d)) && size*(columns+d)<=8*1024*1024;
    Z exponent=0;
    if (eligible && action>=4) {
        mpz_pow_ui(exponent.get_mpz_t(),characteristic.get_mpz_t(),d);
        exponent-=2;
        if (mpz_sizeinbase(exponent.get_mpz_t(),2)>62) eligible=false;
    }
    if (eligible && action==0) {
        for (const auto& row:values) if (row.size()>columns) eligible=false;
    }
    std::vector<int64_t> packed, output;
    if (eligible) {
        // Structure-of-arrays layout gives adjacent lanes adjacent coefficients.
        packed.resize(size*columns+d); output.resize(size*d);
        for (slong j=0;j<d;++j) packed[size*columns+j]=relation.coeff(j).get_si();
        const int64_t p=characteristic.get_si();
        auto pack=[&](const Json::Value& row,size_t i,slong start,slong width) {
            bool nonzero=false;
            if (row.size()<=width) {
                for (Json::ArrayIndex j=0;j<row.size();++j) {
                    const auto text=row[j].asString();
                    int64_t c;
                    auto parsed=std::from_chars(text.data(),text.data()+text.size(),c);
                    if (parsed.ec!=std::errc() || parsed.ptr!=text.data()+text.size()) {
                        const Z z(text,10);
                        c=mpz_fdiv_ui(z.get_mpz_t(),p);
                    } else {
                        c%=p;
                        if (c<0) c+=p;
                    }
                    packed[size*(start+j)+i]=c;
                    nonzero|=c!=0;
                }
            } else {
                a.set(row);
                fmpz_mod_poly_rem(a.value,a.value,relation.value,field.ctx);
                for (slong j=0;j<a.length();++j) packed[size*(start+j)+i]=a.coeff(j).get_si();
                nonzero=a.length()!=0;
            }
            return nonzero;
        };
        for (Json::ArrayIndex i=0;i<values.size();++i) {
            const bool nonzero=pack(values[i],i,0,action==0?columns:d);
            if (!nonzero && action==5) throw std::invalid_argument("Division by zero in the finite field.");
            if (binary && !pack(other[i],i,d,d) && action==4)
                throw std::invalid_argument("Division by zero in the finite field.");
        }
    }
    bool ran=eligible && batch(characteristic==2 ? action+6 : action,d,static_cast<int>(size),columns,characteristic.get_si(),exponent.get_si(),
                               static_cast<int>(mpz_sizeinbase(exponent.get_mpz_t(),2)),static_cast<int>(packed.size()),packed.data(),output.data());
    Json::Value result, encoded(Json::arrayValue);
    for (Json::ArrayIndex i=0;i<values.size();++i) {
        if (ran) {
            Json::Value row(Json::arrayValue);
            slong len=d;
            while (len && output[(len-1)*size+i]==0) --len;
            for (slong j=0;j<len;++j) row.append(std::to_string(output[j*size+i]));
            encoded.append(std::move(row));
            continue;
        }
        a.set(values[i]); fmpz_mod_poly_rem(a.value,a.value,relation.value,field.ctx);
        if (binary) { b.set(other[i]); fmpz_mod_poly_rem(b.value,b.value,relation.value,field.ctx); }
        if (action==0) { encoded.append(a.encode()); continue; }
        if (action==1) fmpz_mod_poly_add(out.value,a.value,b.value,field.ctx);
        if (action==2) fmpz_mod_poly_sub(out.value,a.value,b.value,field.ctx);
        if (action==3) fmpz_mod_poly_mulmod(out.value,a.value,b.value,relation.value,field.ctx);
        if (action>=4) {
            const Poly& divisor=action==5?a:b;
            if (!divisor.length()) throw std::invalid_argument("Division by zero in the finite field.");
            if (!fmpz_mod_poly_invmod(inv.value,divisor.value,relation.value,field.ctx))
                throw std::logic_error("Inversion failed in a verified finite field.");
            if (action==5) { encoded.append(inv.encode()); continue; }
            fmpz_mod_poly_mulmod(out.value,a.value,inv.value,relation.value,field.ctx);
        }
        encoded.append(out.encode());
    }
    result["values"]=std::move(encoded); result["device_batch"]=ran;
    result["field_polynomial"]=relation.encode(); result["modulus"]=characteristic.get_str();
    return result;
}
} // namespace

extern "C" const char* atlas_finite_field_batch(const char* input,Batch callback) noexcept {
    return atlas::sparse_json::invoke(input,[&](const Json::Value& request){return calculate(request,callback);});
}
