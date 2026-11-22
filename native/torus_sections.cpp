// Orbit sections of a verified regular fan in the nonnegative orthant.
#include "rational_sparse_json.hpp"
#include "local_io.hpp"

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
    Z count;  // components meeting the dense torus
    unsigned character;  // zero for an unsplit section without a primitive character
    Face coordinates{};  // additional components contained in coordinate divisors
    bool operator==(const Section& other) const {
        return count == other.count && character == other.character && coordinates == other.coordinates;
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
        const Z total = section.count + section.coordinates.size();
        for (Z root = 1; root <= total; ++root) {
            Value item;
            auto name = base;
            if (!vertex || total > 1) name += "." + root.get_str();
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
                    const auto& parent = sections.at(subface);
                    // A coordinate component is the orbit closure with that
                    // extra ray. A child's generic point belongs to it exactly
                    // when the child has the same extra (nonexceptional) ray.
                    if (root > section.count) {
                        const Z offset = root - section.count - 1;
                        const auto ray = section.coordinates.at(offset.get_ui());
                        const auto it = std::find(parent.coordinates.begin(), parent.coordinates.end(), ray);
                        if (it != parent.coordinates.end()) {
                            target.append(choices.at(parent.count.get_ui() + (it-parent.coordinates.begin())));
                            return true;
                        }
                    }
                    if (parent.count == 1) target.append(choices.front());
                    else if (root <= section.count && section.character && parent.count == section.count &&
                             parent.character == section.character)
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

// A single coordinate section suffices if every other affine exceptional
// section is empty. On E_i, a nonzero constant has no zeros and a pure
// coordinate power cuts out one reduced affine space through the fixed
// point. This includes a strict graph u_E = c*v^d tangent to E.
Value single_coordinate_section(const std::vector<Chart>& charts, const Value& labels, unsigned n) {
    Sections sections;
    for (const auto& chart : charts) for (const auto& [axis,label] : chart.boundary) {
        const unsigned exceptional = 1U << axis;
        auto survives = [exceptional](unsigned term) { return !(term & exceptional); };
        auto term = std::find_if(chart.terms.begin(), chart.terms.end(), survives);
        if (term == chart.terms.end() || std::any_of(term+1, chart.terms.end(), survives)) return {};
        if (!*term) continue;
        if ((*term & (*term-1)) || !sections.empty()) return {};
        sections.emplace(Face{label}, Section{1,0});
    }
    if (sections.empty()) return {};
    return cells(sections, labels, n, true);
}

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

    const bool closed_sections = request.get("closed_sections", false).asBool();
    if (closed_sections && (!intrinsic || !common)) return {};
    if (intrinsic && common && relative && !closed_sections) {
        Value result;
        result["complex"] = single_coordinate_section(charts, labels, n);
        if (!result["complex"].isNull()) return result;
    }
    // In an embedded SNC atlas with internal rays, every pure exceptional
    // intersection V(F) lies over the origin. Its strict section is a
    // smooth divisor or empty. A component either meets the dense torus
    // or equals a toric prime divisor V(F+r) contained in the section.
    // The extra ray r must be nonexceptional by normal crossings. Higher
    // codimension orbits cannot supply an additional divisor component.
    // For relative fans an exceptional divisor can miss the origin fibre.
    // Use every orbit to witness visibility, including when the caller
    // needs chart-local ideals from an embedded SNC certificate.
    std::map<Face, Face> coordinate_components;
    Value contained_fibre_orbits = array();
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
            if (closed_sections && !relative && !boundary_face && boundary.size()+1 != cone_face.size()) continue;
            if (common) {
                std::sort(cone_face.begin(), cone_face.end());
                if (!seen.insert(std::move(cone_face)).second) continue;
            }
            Value terms = array();
            for (unsigned i = 0; i < chart.terms.size(); ++i)
                if (!(mask & chart.terms[i])) terms.append(i);
            if (closed_sections && !boundary_face && terms.empty()) {
                if (relative && request.get("defer_relative_coordinates", false).asBool()) return {};
                Face extra;
                for (unsigned j = 0; j < n; ++j)
                    if ((mask & (1U << j)) && !chart.boundary.count(j)) extra.push_back(chart.rays[j]);
                if (extra.size() == 1) coordinate_components[boundary].push_back(extra.front());
                if (relative && image == full) {
                    Value pair = array();
                    pair.append(indices(boundary));
                    pair.append(indices(extra));
                    contained_fibre_orbits.append(std::move(pair));
                }
                continue;
            }
            if (closed_sections && !relative && !boundary_face) continue;
            if (terms.empty()) return {};
            Value check = array();
            check.append(ci);
            check.append((relative || closed_sections) && boundary_face);  // require absolute irreducibility
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
        incidence["contained_fibre_orbits"] = std::move(contained_fibre_orbits);
        for (auto& [face, rays] : coordinate_components) {
            // Order coordinate components by their original coordinate axis.
            std::sort(rays.begin(), rays.end(), [&](unsigned a, unsigned b) { return images[a] < images[b]; });
            Value pair = array();
            pair.append(indices(face));
            pair.append(indices(rays));
            incidence["coordinate_components"].append(std::move(pair));
        }
    }
    return result;
}

// On the torus of an exceptional ray, a monomial is a unit. Removing
// the common monomial gives the closure of its hypersurface in this chart.
atlas::local::Polynomial torus_closure(const atlas::local::Polynomial& strict, unsigned axis) {
    atlas::local::Powers minimum;
    for (const auto& [powers, coefficient] : strict) {
        if (powers.at(axis)) continue;
        if (minimum.empty()) minimum = powers;
        else for (size_t i = 0; i < powers.size(); ++i)
            minimum[i] = std::min(minimum[i], powers[i]);
    }
    atlas::local::Polynomial reduced;
    for (const auto& [powers, coefficient] : strict) {
        if (powers.at(axis)) continue;
        auto exponent = powers;
        for (size_t i = 0; i < powers.size(); ++i) exponent[i] -= minimum[i];
        reduced.emplace(std::move(exponent), coefficient);
    }
    return reduced;
}

// Numerator and denominator of the primitive character in a chart basis.
std::pair<std::string, std::string> chart_character(const Value& direction, const Value& rays,
                                                     const Face& cone, const std::vector<std::string>& names) {
    using namespace atlas::sparse_json;
    const auto n = names.size();
    Powers positive(n), negative(n);
    for (size_t i = 0; i < n; ++i) {
        const auto ray = read_exponents(rays[cone[i]], n);
        Z pairing = 0;
        for (unsigned k = 0; k < n; ++k) pairing += ray[k] * Z(direction[k].asString());
        if (pairing > 0) positive[i] = pairing;
        else negative[i] = -pairing;
    }
    return {atlas::local::format_polynomial({{positive, 1}}, names),
            atlas::local::format_polynomial({{negative, 1}}, names)};
}

Value chart_sections(const Value& request, const Sections& sections, const Value& labels) {
    using namespace atlas::sparse_json;
    Value result = array();
    std::map<std::string, unsigned> by_name;
    for (unsigned i = 0; i < labels.size(); ++i) by_name.emplace(labels[i].asString(), i);
    for (const auto& chart : request["charts"]) {
        const auto rays = indices(chart["rays"]);
        std::vector<std::string> names;
        for (const auto& name : chart["coordinates"]) names.push_back(name.asString());
        if (names.size() != rays.size()) return {};
        const auto coefficients = read_row(chart["coefficients"]);
        Terms strict;
        for (unsigned j = 0; j < chart["powers"].size(); ++j)
            strict.emplace(read_exponents(chart["powers"][j], names.size()), coefficients.at(j));
        Value local = array();
        for (const auto& pair : chart["boundary"]) {
            const auto axis = pair[0].asUInt();
            const auto label = pair[1].asString();
            const auto found = sections.find({by_name.at(label)});
            // A relative exceptional section whose closure misses the
            // origin fibre was removed by the orbit visibility test.
            if (found == sections.end()) continue;
            const auto& section = found->second;
            const auto& coords = section.coordinates;
            const Z total = section.count + coords.size();
            auto append = [&](const Z& index, std::string equation) {
                Value item;
                item["id"] = total == 1 ? label : label + "." + index.get_str();
                item["ambient_divisor"] = label;
                item["ideal"].append(names.at(axis));
                item["ideal"].append(std::move(equation));
                local.append(std::move(item));
            };
            if (section.count == 1) {
                const auto closure = torus_closure(strict, axis);
                if (closure.size() < 2) return {};
                append(1, atlas::local::format_polynomial(closure, names));
            } else if (section.count > 1) {
                const auto& direction = request["directions"][section.character];
                if (direction.size() != names.size()) return {};
                const auto [a,b] = chart_character(direction, request["rays"], rays, names);
                const auto family = "-alpha" + std::to_string(section.character) + "_";
                for (Z i = 1; i <= section.count; ++i) append(i, a + family + i.get_str() + "*" + b);
            }
            for (size_t j = 0; j < coords.size(); ++j) {
                const auto at = std::find(rays.begin(), rays.end(), coords[j]);
                if (at != rays.end()) append(section.count + j + 1, names.at(at-rays.begin()));
            }
        }
        result.append(std::move(local));
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
    for (const auto& pair : incidence["coordinate_components"]) {
        const auto it = sections.find(indices(pair[0]));
        if (it == sections.end()) return {};
        it->second.coordinates = indices(pair[1]);
    }
    if (incidence["relative"].asBool()) {
        // A contained orbit witnesses a dense-torus component precisely when
        // no coordinate component covers it. Near the fibre the section on
        // each exceptional face is smooth, so different components are disjoint.
        for (const auto& pair : incidence["contained_fibre_orbits"]) {
            const Face face = indices(pair[0]), extra = indices(pair[1]);
            for (unsigned mask = 1; mask < (1U << face.size()); ++mask) {
                Face subset;
                for (size_t j = 0; j < face.size(); ++j) if (mask & (1U << j)) subset.push_back(face[j]);
                if (visible.count(subset)) continue;
                const auto& section = sections.at(subset);
                if (std::any_of(extra.begin(), extra.end(), [&](unsigned r) {
                        return std::find(section.coordinates.begin(), section.coordinates.end(), r) != section.coordinates.end();
                    })) continue;
                if (section.count != 1) return {};
                visible.insert(std::move(subset));
            }
        }
        for (auto it = sections.begin(); it != sections.end();) {
            // Every coordinate orbit closure contains a fixed point over zero.
            if (!visible.count(it->first)) it->second.count = 0;
            if (!it->second.count && it->second.coordinates.empty()) it = sections.erase(it);
            else ++it;
        }
    }
    auto result = cells(sections, incidence["labels"], incidence["dimension"].asUInt(), true);
    if (result.isNull() || !request.isMember("charts")) return result;
    result["chart_sections"] = chart_sections(request, sections, incidence["labels"]);
    if (result["chart_sections"].isNull()) return {};
    return result;
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
