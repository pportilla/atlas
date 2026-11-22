#include "geometry_query.hpp"
#include "repair_gluing.hpp"
#include <algorithm>

namespace {
using atlas::local::Basis;
using atlas::local::Polynomial;
using atlas::local::Powers;

// Give anonymous sheet components a reproducible traversal. All ideals and
// maps use local coordinate columns. Replace the sole positional reference
// by its verified parent presentation before ordering the charts.
Json::Value chart_order(const Json::Value& charts) {
    std::vector<Json::Value> signatures;
    std::vector<unsigned> order;
    for (unsigned i = 0; i < charts.size(); ++i) {
        auto chart = charts[i];
        const int parent = chart["parent"].asInt();
        chart.removeMember("parent");
        if (parent >= 0) {
            auto base = charts[parent];
            base.removeMember("parent");
            chart["parent"] = std::move(base);
        }
        signatures.push_back(std::move(chart));
        order.push_back(i);
    }
    std::sort(order.begin(), order.end(), [&](unsigned i, unsigned j) { return signatures[i] < signatures[j]; });
    Json::Value result(Json::arrayValue);
    for (auto i : order) result.append(i);
    return result;
}

// Sort monomials by their columns in the common linear change of coordinates.
// The ordered exponent vectors, transposed, are this chart's rays.
Json::Value parent_cone(const Json::Value& images) {
    const unsigned n = images.size();
    std::map<Powers, std::vector<mpq_class>> columns;
    for (unsigned i = 0; i < n; ++i)
        for (const auto& [exponent, c] : atlas::local::decode_row(images[i].asString(), n))
            columns.try_emplace(exponent, n).first->second[i] = c;
    if (columns.size() != n || !n) return {};
    std::map<std::vector<mpq_class>, Powers> ordered;
    for (const auto& [exponent, column] : columns)
        if (!ordered.emplace(column, exponent).second) return {};
    Json::Value result;
    for (const auto& [column, exponent] : ordered) {
        result["linear"].append(atlas::sparse_json::strings(column));
        for (unsigned j = 0; j < n; ++j) result["rays"][j].append(exponent[j].get_str());
    }
    return result;
}

// On the orbit a coordinate is either zero or a unit. An owner must
// contain the entire orbit. A proper, nonempty part cannot be certified.
enum class Ownership { whole, disjoint, partial };
Ownership orbit_region(const Basis& equations, const std::vector<bool>& zero) {
    Basis nonzero;
    for (const auto& equation : equations) {
        Polynomial restriction;
        for (const auto& term : equation) {
            bool vanishes = false;
            for (size_t i = 0; i < zero.size(); ++i)
                if (zero[i] && term.first[i] != 0) { vanishes = true; break; }
            if (!vanishes) restriction.insert(term);
        }
        if (restriction.size() == 1) return Ownership::disjoint;
        if (!restriction.empty()) nonzero.push_back(std::move(restriction));
    }
    if (nonzero.empty()) return Ownership::whole;
    // Saturate by the product of the invertible coordinates. Its unit
    // ideal is equivalent to an empty common zero locus on the orbit.
    Powers units(zero.size());
    for (size_t i = 0; i < zero.size(); ++i)
        if (!zero[i]) units[i] = 1;
    const auto saturated = atlas::primary::saturate_lex(nonzero, {{units, 1}}, zero.size());
    const Polynomial one{{Powers(zero.size()), 1}};
    return atlas::primary::member_lex(one, saturated) ? Ownership::disjoint : Ownership::partial;
}

bool parent_partition(const Json::Value& charts) {
    std::map<std::vector<unsigned>, bool> owners;
    for (const auto& chart : charts) {
        const auto& rays = chart["rays"];
        Basis region;
        for (const auto& row : chart["region"])
            region.push_back(atlas::local::decode_row(row.asString(), rays.size()));
        for (const auto& face : chart["faces"]) {
            std::vector<unsigned> ids;
            std::vector<bool> zero(rays.size(), false);
            for (const auto& axis : face) {
                const unsigned i = axis.asUInt();
                zero.at(i) = true;
                ids.push_back(rays[i].asUInt());
            }
            std::sort(ids.begin(), ids.end());
            bool& owned = owners[ids];
            switch (orbit_region(region, zero)) {
                case Ownership::partial: return false;
                case Ownership::disjoint: break;
                case Ownership::whole:
                    if (owned) return false;
                    owned = true;
                    break;
            }
        }
    }
    return !owners.empty() && std::all_of(owners.begin(), owners.end(), [](const auto& e) { return e.second; });
}
}

extern "C" const char* atlas_repair_geometry(const char* input) noexcept {
    return atlas::sparse_json::invoke(input,[](const Json::Value& r)->Json::Value {
        const auto op=r["operation"].asString();
        if(op=="chart_order") return chart_order(r["charts"]);
        if(op=="parent_cone") return parent_cone(r["images"]);
        if(op=="parent_partition") return parent_partition(r["charts"]);
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
