// Timings on prepared equations. Send the complete candidate jet to Singular for comparison.
#include "point_jet.hpp"
#include <chrono>
#include <iostream>

using namespace atlas::certificate;
using namespace atlas::chart;

int main() {
    Json::Value data, result(Json::objectValue);
    std::cin >> data;
    Variables vars, dependent;
    Strings names;
    for (const auto& x : data["variables"]) {
        names.push_back(x.asString());
        vars.push_back({names.back(), 0});
    }
    for (const auto& pos : data["dependent"]) dependent.push_back(vars.at(pos.asUInt()));
    auto expression = [&](const Json::Value& x) {
        if (data["encoded_rows"].asBool()) return decode(vars, x.asString());
        return normal(vars, atlas::local::parse_expression(names, x.asString()));
    };
    auto coefficients = [](const Json::Value& row) {
        std::vector<Q> field;
        for (const auto& c : row) field.emplace_back(c.asString());
        return field;
    };
    GaloisOrbit point;
    point.mu = coefficients(data["mu"]);
    for (const auto& row : data["point"]) point.coordinates.push_back(coefficients(row));
    Ideal frame;
    for (const auto& row : data["rows"]) frame.push_back(expression(row));
    auto f = expression(data["f"]);
    unsigned degree = data["degree"].asUInt();
    const CertificateCaps caps;
    const int repeats = data["repeats"].asInt();
    auto compute = [&] { return frame_jet(vars, frame, dependent, f, point, degree, caps); };
    auto jet = compute();
    if (!jet) throw std::runtime_error("jet declined");
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) jet = compute();
    auto elapsed = std::chrono::steady_clock::now() - start;
    result["us"] = std::chrono::duration<double,std::micro>(elapsed).count()/repeats;
    result["terms"] = Json::arrayValue;
    for (const auto& [p, coefficient] : jet->coefficients) {
        Json::Value term;
        term["powers"] = Json::arrayValue;
        for (unsigned e : p) term["powers"].append(e);
        for (const auto& q : coefficient) term["coefficient"].append(q.get_str());
        result["terms"].append(std::move(term));
    }
    std::cout << result << '\n';
}
