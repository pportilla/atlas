// Verify the charts of a regular monomial modification in one call.
//
// For every maximal cone, in the caller's order, this runs the native steps of
// the per-chart fan verifier: the supports owned by the chart, the pullback of
// the equation with its projection (fan_charts_with_projection), the region
// generators, the exceptional numerical data shared by all charts, and the
// crossings of the total transform on the owned origin fibre. The boundary
// complex then comes from the torus-section plan. A fan with N charts costs one
// call instead of three foreign calls and their text transport per chart.
//
// Each chart reports the first failing check in the order of the per-chart
// verifier. Work stops after that chart, which the caller would not pass.
#include "rational_sparse_json.hpp"

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
char* atlas_affine(const char*, size_t) noexcept;
char* atlas_ideal(const char*, size_t) noexcept;
void atlas_groebner_release(char*) noexcept;
const char* atlas_groebner_error() noexcept;
}
namespace atlas { Json::Value torus_section_plan(const Json::Value&); }

namespace {
using Value = Json::Value;
using Strings = std::vector<std::string>;
using Operation = char* (*)(const char*, size_t) noexcept;

// A native operation declined. The caller raises the exception its own
// transport raises for the same answer.
struct Declined {
    std::string kind, message;
};

Strings call(Operation operation, const std::string& input, bool affine) {
    std::unique_ptr<char, decltype(&atlas_groebner_release)> answer(operation(input.data(), input.size()),
                                                                   atlas_groebner_release);
    if (!answer) {
        std::string message = atlas_groebner_error();
        const bool unavailable = !affine || message == "Unsupported number of variables." ||
                                 message == "Degree out of range." || message == "Exponent out of range.";
        throw Declined{unavailable ? "unavailable" : "error", message};
    }
    Strings lines;
    std::istringstream text(answer.get());
    for (std::string line; std::getline(text, line);) lines.push_back(std::move(line));
    return lines;
}

std::string join(const Strings& values) {
    std::string out;
    for (const auto& value : values) {
        if (!out.empty()) out += ' ';
        out += value;
    }
    return out;
}

Strings strings(const Value& array) {
    Strings out;
    for (const auto& value : array) out.push_back(value.asString());
    return out;
}

std::string indicator_row(const std::vector<bool>& support) {
    std::string row = "1";
    for (bool bit : support) row += bit ? " 1" : " 0";
    return row;
}

std::string monomial_text(const std::vector<bool>& support, const Strings& coords) {
    std::string text;
    for (size_t j = 0; j < support.size(); ++j) {
        if (!support[j]) continue;
        if (!text.empty()) text += '*';
        text += coords[j];
    }
    return text.empty() ? "1" : text;
}

struct Chart {
    Strings coords, labels;
    std::vector<int> cone;
    std::vector<std::vector<bool>> excluded;
};

Value verify(const Value& request) {
    const Strings names = strings(request["names"]);
    const size_t n = names.size();
    const std::string equation = request["equation"].asString();
    const size_t block = std::max<Json::UInt>(1, request["block"].asUInt());
    const bool bounded = request["bounded"].asBool(), complex = request["complex"].asBool();
    std::vector<Strings> rays;
    for (const auto& ray : request["rays"]) {
        rays.push_back(strings(ray));
        if (rays.back().size() != n) throw std::invalid_argument("Expected one entry per coordinate in a ray.");
    }
    std::vector<bool> axis(rays.size());
    for (size_t r = 0; r < rays.size(); ++r) {
        size_t ones = 0, zeros = 0;
        for (const auto& x : rays[r]) ones += x == "1", zeros += x == "0";
        axis[r] = ones == 1 && zeros + 1 == n;
    }
    std::vector<Chart> charts;
    for (const auto& data : request["charts"]) {
        Chart chart;
        chart.coords = strings(data["coordinates"]);
        chart.labels = strings(data["labels"]);
        for (const auto& id : data["cone"]) {
            if (!id.isUInt() || id.asUInt() >= rays.size()) throw std::invalid_argument("Invalid fan ray index.");
            chart.cone.push_back(id.asInt());
        }
        for (const auto& face : data["excluded"]) {
            std::vector<bool> support(n);
            for (const auto& j : face) {
                if (!j.isUInt() || j.asUInt() >= n) throw std::invalid_argument("Invalid excluded face.");
                support[j.asUInt()] = true;
            }
            chart.excluded.push_back(std::move(support));
        }
        if (chart.coords.size() != n || chart.cone.size() != n || chart.labels.size() != n)
            throw std::invalid_argument("Expected a full-dimensional chart.");
        charts.push_back(std::move(chart));
    }

    Value result, out(Json::arrayValue);
    std::string frame;
    for (const auto& row : request["frame"]) frame += join(strings(row)) + "\n";
    const size_t width = 4 + n;
    std::map<int, std::string> by_ray;
    std::map<std::string, std::pair<std::pair<std::string, std::string>, int>> data;
    Value plan_charts(Json::arrayValue);
    bool stopped = false;
    for (size_t start = 0; start < charts.size() && !stopped;) {
        // Consecutive charts on the same coordinates share one request.
        size_t stop = start + 1;
        while (stop < charts.size() && stop - start < block && charts[stop].coords == charts[start].coords) ++stop;
        std::string input = "fan_charts_with_projection\n" + join(names) + "\n" + join(charts[start].coords) + "\n" +
                            equation + "\n" + frame;
        for (size_t i = start; i < stop; ++i) {
            for (int r : charts[i].cone) input += join(rays[r]) + "\n";
            // The support of each coordinate image (a column of the cone),
            // then of each excluded face, as monomial_support_masks gives it.
            std::string masks;
            for (size_t k = 0; k < n; ++k) {
                unsigned long mask = 0;
                for (size_t j = 0; j < n; ++j)
                    if (rays[charts[i].cone[j]][k] != "0") mask |= 1UL << j;
                masks += std::to_string(mask) + " ";
            }
            for (const auto& face : charts[i].excluded) {
                unsigned long mask = 0;
                for (size_t j = 0; j < n; ++j)
                    if (face[j]) mask |= 1UL << j;
                masks += std::to_string(mask) + " ";
            }
            masks.pop_back();
            input += masks + "\n";
        }
        Strings rows;
        try {
            rows = call(atlas_affine, input, true);
            if (rows.size() != width * (stop - start))
                throw Declined{"error", "The native fan operation returned incomplete chart data."};
        } catch (const Declined& declined) {
            result["failure"]["chart"] = Json::UInt64(start);
            result["failure"]["kind"] = declined.kind;
            result["failure"]["message"] = declined.message;
            break;
        }
        for (size_t i = start; i < stop; ++i) {
            const auto& chart = charts[i];
            const auto* fan = rows.data() + width * (i - start);
            Value verified;
            Value projection(Json::arrayValue), region(Json::arrayValue), orders(Json::arrayValue);
            for (size_t j = 0; j < n; ++j) projection.append(fan[4 + j]);
            for (const auto& face : chart.excluded) region.append(monomial_text(face, chart.coords));
            std::istringstream header(fan[0]);
            std::string token;
            header >> token;
            const bool strict = token == "1";
            verified["projection"] = projection;
            verified["region"] = region;
            verified["strict"] = strict;
            verified["strict_text"] = fan[1];
            verified["status"] = "ok";
            // Exceptional positions in cone order, with their global labels.
            std::vector<size_t> exceptional;
            for (size_t j = 0; j < n; ++j)
                if (!axis[chart.cone[j]]) exceptional.push_back(j);
            std::string status = "ok";
            Value boundary(Json::arrayValue);
            for (size_t j : exceptional) {
                std::string order, nu;
                if (!(header >> order >> nu)) throw std::invalid_argument("The fan orders do not match the cone.");
                Value pair(Json::arrayValue);
                pair.append(order);
                pair.append(nu);
                orders.append(pair);
                const int ray = chart.cone[j];
                const auto& label = chart.labels[j];
                if (status != "ok") continue;
                auto seen = by_ray.emplace(ray, label).first;
                auto known = data.find(label);
                if (seen->second != label || (known != data.end() && known->second.second != ray)) {
                    status = "label";
                    continue;
                }
                auto entry = data.emplace(label, std::make_pair(std::make_pair(order, nu), ray)).first;
                if (entry->second != std::make_pair(std::make_pair(order, nu), ray)) status = "data";
                Value item(Json::arrayValue);
                item.append(Json::UInt64(j));
                item.append(label);
                boundary.append(item);
            }
            if (header >> token) throw std::invalid_argument("The fan orders do not match the cone.");
            verified["orders"] = orders;
            if (status == "ok") {
                // The exceptional coordinates, then the strict transform's
                // value and first derivatives on the owned supports.
                Strings ideals, owned;
                for (size_t j : exceptional) {
                    std::vector<bool> support(n);
                    support[j] = true;
                    ideals.push_back(indicator_row(support));
                }
                if (strict) ideals.push_back(fan[3]);
                auto own = [&](const std::string& row) {
                    if (std::find(owned.begin(), owned.end(), row) == owned.end()) owned.push_back(row);
                };
                for (size_t k = 0; k < n; ++k) {
                    std::string row = "1";
                    for (size_t j = 0; j < n; ++j) row += " " + rays[chart.cone[j]][k];
                    own(row);
                }
                for (const auto& face : chart.excluded) own(indicator_row(face));
                std::string query = std::string(bounded ? "bounded_crossings " : "crossings ") + std::to_string(n) +
                                    " " + std::to_string(n) + " 0 " + std::to_string(ideals.size());
                for (size_t k = 0; k < ideals.size(); ++k) query += " 1";
                query += " " + std::to_string(owned.size()) + " 0\n";
                for (const auto& row : ideals) query += row + "\n";
                for (const auto& row : owned) query += row + "\n";
                try {
                    const auto loci = call(atlas_ideal, query, false);
                    if (loci == Strings{"unknown"}) status = "unknown";
                    else if (!loci.empty() && loci[0] != "none") status = "defect";
                } catch (const Declined& declined) {
                    status = "unavailable";
                    verified["message"] = declined.message;
                }
            }
            verified["status"] = status;
            out.append(verified);
            if (status != "ok") {
                stopped = true;
                break;
            }
            if (complex) {
                Value section;
                section["rays"] = Value(Json::arrayValue);
                for (int r : chart.cone) section["rays"].append(r);
                section["boundary"] = boundary;
                plan_charts.append(section);
            }
        }
        start = stop;
    }
    result["charts"] = out;
    if (complex && !stopped && !result.isMember("failure")) {
        // The ray indices are the caller's order of first appearance, as in
        // the section plan of the per-chart verifier.
        Value plan;
        plan["dimension"] = Json::UInt64(n);
        plan["intrinsic"] = false;
        plan["common_source"] = true;
        plan["rays"] = Value(Json::arrayValue);
        for (const auto& ray : rays) {
            Value entries(Json::arrayValue);
            for (const auto& x : ray) entries.append(x);
            plan["rays"].append(entries);
        }
        plan["charts"] = plan_charts;
        result["plan"] = atlas::torus_section_plan(plan);
    }
    return result;
}
} // namespace

extern "C" int atlas_fan_verification_abi() noexcept { return 1; }

extern "C" const char* atlas_fan_verification(const char* input) noexcept {
    return atlas::sparse_json::invoke(input, verify);
}
