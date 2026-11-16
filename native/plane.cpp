// Exact point blowups of plane curve germs over the algebraic closure of Q.
// GMP first, so that FLINT declares its GMP conversions (sparse_flint.hpp).
#include <gmpxx.h>
#include <flint/fmpq_mpoly_factor.h>
#include "sparse_flint.hpp"
#include "three_graph_zetas.hpp"
#include "four_graph_zetas.hpp"
#include "quadratic_translate_zetas.hpp"
#include <optional>
#include <string_view>
#include "plane_resolver.hpp"
#include "three_graph_contact.inc"

// Address space already mapped by this process, in bytes. Library start-up
// reservations (for example BLAS thread buffers) and, in a forked solver, the
// parent's mappings count here but not against the calculation's budget.
static rlim_t mapped_bytes(){
    FILE* statm=fopen("/proc/self/statm","r");
    unsigned long pages=0;
    if(statm){
        if(fscanf(statm,"%lu",&pages)!=1) pages=0;
        fclose(statm);
    }
    return rlim_t(pages)*rlim_t(sysconf(_SC_PAGESIZE));
}

static int program(){
    J req,out;
    try{
        cin>>req;
        quick=!req.get("exhaustive",false).asBool();
        // Apply the optional quick-mode memory budget only to this helper:
        // at most 1 GiB beyond the address space mapped at its start.
        struct rlimit limit;
        if (quick && getrlimit(RLIMIT_AS,&limit)==0) {
            const rlim_t budget=mapped_bytes()+(rlim_t(1)<<30);
            if (limit.rlim_cur==RLIM_INFINITY || limit.rlim_cur>budget) limit.rlim_cur=budget;
            if(setrlimit(RLIMIT_AS,&limit)!=0) throw Fail("Could not bound the exact-arithmetic worker.");
        }
        if (req.isMember("contact_zetas") || req.isMember("three_graphs")) {
            out=polynomial_graph_contact(req);
        } else if (req.isMember("ordered_roots")) {
            out=ordered_root_labels(req["ordered_roots"]);
        } else if (req.isMember("field_roots")) {
            out=atlas::common_field(req);
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

// The plane program's entry, also run in a forked copy of a solver that
// already loaded this library.
extern "C" __attribute__((visibility("default"))) int atlas_plane_main(int, char**) noexcept {
    const int status=program();
    cout.flush();
    fflush(nullptr);
    return status;
}

#ifndef ATLAS_HELPER_LIBRARY
int main(int argc, char** argv){ return atlas_plane_main(argc, argv); }
#endif
