#include "geometry_query.hpp"
#include "repair_gluing.hpp"

extern "C" const char* atlas_repair_geometry(const char* input) noexcept {
    return atlas::sparse_json::invoke(input,[](const Json::Value& r)->Json::Value {
        const auto op=r["operation"].asString();
        if(op=="relations") return atlas::repair::chart_relations(r);
        if(op=="parent") return atlas::repair::toric_parent(r);
        if(op=="transition") return atlas::repair::toric_transition(r);
        if(op=="point") return r["dimension"].asInt()==0 && mpz_class(r["count"].asString(),10)==1;
        throw std::invalid_argument("Unknown repair chart request.");
    });
}


extern "C" const char* atlas_repair_ownership(const char* input, atlas::GeometryQuery::Callback callback) noexcept {
    return atlas::sparse_json::invoke(input,[&](const Json::Value& r) {
        atlas::GeometryQuery host{callback};
        return atlas::repair::ownership(r,[&](const std::string& op,const Json::Value& q){return host.call(op,q);});
    });
}
