// Orbit sections of a verified regular fan in the nonnegative orthant.
#include "rational_sparse_json.hpp"

#include <algorithm>
#include <map>
#include <set>

namespace atlas { Json::Value fan_surface_complex(const Json::Value&); }

namespace {
using Value = Json::Value;
using Face = std::vector<unsigned>;
using atlas::sparse_json::Z;

struct FaceOrder {
    bool operator()(const Face& a, const Face& b) const {
        return a.size() != b.size() ? a.size() < b.size() : a < b;
    }
};

Value array() { return Value(Json::arrayValue); }
Value indices(const Face& face) {
    Value result = array();
    for (auto index : face) result.append(index);
    return result;
}
Face indices(const Value& row) {
    Face result;
    for (const auto& entry : row) result.push_back(entry.asUInt());
    return result;
}
unsigned support(const Value& coordinates, unsigned n) {
    if (coordinates.size() != n) throw std::invalid_argument("Invalid torus coordinate dimension.");
    unsigned mask = 0;
    for (unsigned j = 0; j < n; ++j)
        if (coordinates[j].asString() != "0") mask |= 1U << j;
    return mask;
}

struct Section {
    Z count;
    unsigned character;  // zero for an unsplit section without a primitive character
    bool operator==(const Section& other) const {
        return count == other.count && character == other.character;
    }
};
using Sections = std::map<Face, Section, FaceOrder>;

// A squarefree segment splits by primitive character; a smooth higher-rank
// torus hypersurface is connected by toric Lefschetz.
Z section_count(unsigned rank, const Value& degree) {
    return rank == 1 ? Z(degree.asString(), 10) : Z(rank != 0);
}

Value cells(const Sections& sections, const Value& labels, unsigned n, bool intrinsic,
            const Value* newton_families = nullptr) {
    Value result;
    result["vertices"] = array();
    result["cells"] = array();
    result["status"] = "complete";
    result["convention"] = std::string(intrinsic ? "Exceptional divisor on the smooth strict transform"
                                                : "Ambient exceptional divisor") +
                           " in the germ over the origin over Qbar.";
    std::map<Face, std::vector<std::string>> ids;
    for (const auto& [face, section] : sections) {
        const bool vertex = face.size() == 1;
        std::string base = vertex ? "" : "C";
        for (size_t j = 0; j < face.size(); ++j) {
            if (j) base += "-";
            if (newton_families)
                base += vertex ? "E" + std::to_string(face[j] - n + 1) : std::to_string(face[j]);
            else base += labels[face[j]].asString();
        }
        for (Z root = 1; root <= section.count; ++root) {
            Value item;
            auto name = base;
            if (!vertex || section.count > 1) name += "." + root.get_str();
            item["id"] = name;
            if (newton_families && section.character)
                item["root"] = (*newton_families)[section.character].asString() + "_" + root.get_str();
            if (vertex) {
                if (newton_families) item["ray_index"] = face.front();
                else item["dimension"] = n - 1 - unsigned(intrinsic);
                result["vertices"].append(std::move(item));
            } else {
                // There is a unique restriction to a connected section. On a
                // disconnected segment, the same root index is required on
                // both sides of the restriction in the common primitive character.
                auto restrict_to = [&](const Face& subface, Value& target) {
                    const auto found = ids.find(subface);
                    if (found == ids.end()) return false;
                    const auto& choices = found->second;
                    if (choices.size() == 1) target.append(choices.front());
                    else if (section.character && choices.size() == section.count &&
                             sections.at(subface).character == section.character)
                        target.append(choices[root.get_ui()-1]);
                    else return false;
                    return true;
                };
                for (auto v : face)
                    if (!restrict_to({v}, item["vertices"])) return {};
                // Omitting the last coordinate first gives lexicographic
                // facet order in the original ordered list of labels.
                for (size_t j = face.size(); j-- > 0;) {
                    auto facet = face;
                    facet.erase(facet.begin() + j);
                    if (!restrict_to(facet, item["faces"])) return {};
                }
                item["dimension"] = unsigned(face.size() - 1);
                result["cells"].append(std::move(item));
            }
            ids[face].push_back(std::move(name));
        }
    }
    return result;
}

// The producer already has the certified Newton face at every orbit. Use the
// same restrictions as saved-chart replay, retaining the historical ray IDs.
Value newton_cells(const Value& request) {
    const auto n = request["dimension"].asUInt();
    const bool intrinsic = request["intrinsic"].asBool();
    const auto& proofs = request["proofs"];
    Value families = array();
    families.append("");
    for (const auto& proof : proofs) families.append(proof[2]);
    Sections sections;
    for (const auto& orbit : request["orbits"]) {
        Face face = indices(orbit[0]);
        if (face.empty() || !std::is_sorted(face.begin(),face.end()))
            throw std::invalid_argument("Invalid Newton cone face.");
        if (face.front() < n) continue;
        const auto id = orbit[1].asUInt();
        if (id >= proofs.size()) throw std::invalid_argument("Missing Newton face certificate.");
        const auto& proof = proofs[id];
        const auto rank = proof[0].asUInt();
        sections.emplace(std::move(face), Section{intrinsic ? section_count(rank,proof[1]) : Z(1),
                                                  intrinsic && rank == 1 ? id+1 : 0});
    }
    auto result = cells(sections, {}, n, intrinsic, &families);
    if (result.isNull()) throw std::invalid_argument("Incompatible nested Newton edge labels.");
    return result;
}

struct Chart {
    Face rays;
    std::map<unsigned, unsigned> boundary;
    std::vector<unsigned> terms;
};

Value plan(const Value& request) {
    const auto n = request["dimension"].asUInt();
    if (n < 2 || n > 8) return {};
    const auto full = (1U << n) - 1;
    const bool intrinsic = request["intrinsic"].asBool();
    const bool common = request["common_source"].asBool();
    std::vector<unsigned> images;
    std::vector<bool> coordinate_ray;
    for (const auto& row : request["rays"]) {
        const auto image = support(row, n);
        images.push_back(image);
        coordinate_ray.push_back(image && !(image & (image - 1)) &&
            std::any_of(row.begin(), row.end(), [](const Value& x){ return x.asString() == "1"; }));
    }

    Value labels = array();
    std::map<std::string, unsigned> label_ids;
    std::map<unsigned, unsigned> ray_labels, label_rays;
    std::vector<Chart> charts;
    bool relative = false;
    unsigned equation = 0;
    for (const auto& item : request["charts"]) {
        if (!common) {
            unsigned current = item["equation"].asUInt();
            if (!charts.empty() && current != equation) return {};
            equation = current;
        }
        Chart chart;
        chart.rays = indices(item["rays"]);
        if (chart.rays.size() != n) return {};
        for (const auto& pair : item["boundary"]) {
            unsigned axis = pair[0].asUInt();
            if (axis >= n) return {};
            const auto ray = chart.rays[axis];
            const auto label = pair[1].asString();
            auto [it, fresh] = label_ids.emplace(label, labels.size());
            const auto id = it->second;
            if (fresh) labels.append(label);
            if (ray_labels.emplace(ray, id).first->second != id ||
                label_rays.emplace(id, ray).first->second != ray ||
                !chart.boundary.emplace(axis, id).second) return {};
            relative |= intrinsic && images.at(ray) != full;
        }
        for (unsigned j = 0; j < n; ++j)
            if (coordinate_ray.at(chart.rays[j]) == bool(chart.boundary.count(j))) return {};
        for (const auto& powers : item["powers"]) chart.terms.push_back(support(powers, n));
        charts.push_back(std::move(chart));
    }

    Sections ambient;
    std::set<Face> seen;
    Value checks = array(), orbits = array();
    for (unsigned ci = 0; ci < charts.size(); ++ci) {
        const auto& chart = charts[ci];
        for (unsigned mask = 1; mask <= full; ++mask) {
            Face cone_face, boundary;
            unsigned image = 0;
            for (unsigned j = 0; j < n; ++j) if (mask & (1U << j)) {
                cone_face.push_back(chart.rays[j]);
                image |= images[chart.rays[j]];
                auto label = chart.boundary.find(j);
                if (label != chart.boundary.end()) boundary.push_back(label->second);
            }
            if (boundary.empty()) continue;
            const bool boundary_face = boundary.size() == cone_face.size();
            std::sort(boundary.begin(), boundary.end());
            if (!intrinsic) {
                // Every orbit closure of the ambient chart contains its fixed
                // point over the origin, including faces of relative fans.
                if (boundary_face) ambient.emplace(std::move(boundary), Section{1, 0});
                continue;
            }
            if (common) {
                std::sort(cone_face.begin(), cone_face.end());
                if (!seen.insert(std::move(cone_face)).second) continue;
            }
            Value terms = array();
            for (unsigned i = 0; i < chart.terms.size(); ++i)
                if (!(mask & chart.terms[i])) terms.append(i);
            if (terms.empty()) return {};
            Value check = array();
            check.append(ci);
            check.append(relative && boundary_face);  // require absolute irreducibility
            check.append(std::move(terms));
            checks.append(std::move(check));
            Value orbit = array();
            orbit.append(indices(boundary));
            orbit.append(boundary_face);
            orbit.append(relative && image == full);
            orbits.append(std::move(orbit));
        }
    }
    Value result;
    if (!intrinsic) {
        result["complex"] = cells(ambient, labels, n, false);
    } else {
        result["checks"] = std::move(checks);
        auto& incidence = result["incidence"];
        incidence["dimension"] = n;
        incidence["labels"] = std::move(labels);
        incidence["relative"] = relative;
        incidence["orbits"] = std::move(orbits);
    }
    return result;
}

Value glue(const Value& request) {
    const auto& answers = request["sections"];
    const auto& incidence = request["incidence"];
    const auto& orbits = incidence["orbits"];
    if (answers.size() != orbits.size()) throw std::invalid_argument("Missing torus section certificate.");
    Sections sections;
    std::set<Face> visible;
    for (unsigned i = 0; i < orbits.size(); ++i) {
        const auto& orbit = orbits[i];
        Face face = indices(orbit[0]);
        Z count(answers[i][0].asString(), 10);
        if (count < 0) return {};
        Section section{std::move(count), answers[i][1].asUInt()};
        if (orbit[1].asBool() && !(sections.emplace(face, section).first->second == section)) return {};
        if (section.count && orbit[2].asBool()) {
            // A nonempty orbit on the origin fibre witnesses every exceptional
            // subface, with every root of its segment represented.
            for (unsigned mask = 1; mask < (1U << face.size()); ++mask) {
                Face subset;
                for (size_t j = 0; j < face.size(); ++j)
                    if (mask & (1U << j)) subset.push_back(face[j]);
                visible.insert(std::move(subset));
            }
        }
    }
    if (incidence["relative"].asBool()) {
        for (auto it = sections.begin(); it != sections.end();) {
            if (!visible.count(it->first)) it = sections.erase(it);
            else ++it;
        }
    }
    return cells(sections, incidence["labels"], incidence["dimension"].asUInt(), true);
}

// All primitive character roots on a terminal binomial are geometrically
// disjoint. Each cell restricts to the same root on each boundary face.
Value binomial_sheets(const Value& request) {
    const Z degree(request["degree"].asString(),10);
    Value result;
    auto& items=result["layout"]=array();
    for (const auto& base : request["layout"]) {
        Value sheet=array();
        for (Z root=1;root<=degree;++root) {
            const auto index=root.get_str();
            Value item;
            item["id"]=base["id"].asString()+"."+index;
            item["root"]="alpha1_"+index;
            item["root_index"]=index;
            for (const auto* field : {"vertices","faces"}) if (base.isMember(field)) {
                item[field]=array();
                for (const auto& name : base[field]) item[field].append(name.asString()+"."+index);
            }
            sheet.append(std::move(item));
        }
        items.append(std::move(sheet));
    }
    return result;
}

// A row of the base coordinate map is a linear combination of the cone
// characters. Both the rays and the input coefficients may be unbounded.
bool projection_matches(const Value& input) {
    using namespace atlas::sparse_json;
    const auto& rays = input["rays"];
    const auto& linear = input["linear"];
    const auto& images = input["images"];
    const size_t n = rays.size();
    if (linear.size() != n || images.size() != n) return false;
    std::vector<Powers> columns(n, Powers(n));
    for (Json::ArrayIndex i = 0; i < n; ++i) {
        const auto row = read_exponents(rays[i], n);
        for (size_t j = 0; j < n; ++j) columns[j][i] = row[j];
    }
    for (Json::ArrayIndex i = 0; i < n; ++i) {
        const auto coefficients = read_row(linear[i]);
        if (coefficients.size() != n) return false;
        Terms expected;
        for (size_t j = 0; j < n; ++j) {
            if (coefficients[j] != 0) expected[columns[j]] += coefficients[j];
        }
        if (expected != read_terms(images[i], n)) return false;
    }
    return true;
}

Value compute(const Value& request) {
    const auto op = request["operation"].asString();
    if (op == "projection") return projection_matches(request);
    if (op == "binomial_sheets") return binomial_sheets(request);
    if (op == "surface") return atlas::fan_surface_complex(request);
    if (op == "newton") return newton_cells(request);
    if (op == "plan") return plan(request);
    if (op == "glue") return glue(request);
    if (op == "section") {
        if (!request["nondegenerate"].asBool()) return {};
        const auto rank = request["rank"].asUInt();
        // A monomial has no torus zero. A smooth section of Newton rank at
        // least two is connected by toric Lefschetz. A squarefree segment
        // has one component per nonzero root in its primitive character.
        return section_count(rank, request["degree"]).get_str();
    }
    throw std::invalid_argument("Unknown torus section operation.");
}
}  // namespace

namespace atlas {
Json::Value torus_section_plan(const Json::Value& request) { return plan(request); }
}

extern "C" const char* atlas_torus_sections(const char* input) noexcept {
    return atlas::sparse_json::invoke(input, compute);
}
