// Time prepared finite open loci and return every orbit for independent comparison.
#include "finite_locus.hpp"
#include <chrono>
#include <iostream>

int main() {
    using namespace atlas::chart;
    Json::Value input, result;
    std::cin >> input;
    Strings names;
    Variables vars;
    for (const auto& name : input["variables"]) {
        names.push_back(name.asString());
        vars.push_back({names.back(), 0});
    }
    auto read = [&](const Json::Value& rows) {
        Ideal out;
        for (const auto& item : rows) out.push_back(normal(vars, atlas::local::parse_expression(names, item.asString())));
        return out;
    };
    auto equations = read(input["equations"]), units = read(input["units"]);
    const int repeats = input["repeats"].asInt();
    if (repeats <= 0) return 1;
    auto compute = [&] { return atlas::certificate::finite_locus(vars, equations, units, {}); };
    auto locus = compute();
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        locus = compute();
        if (!locus.certified) throw std::runtime_error("finite locus declined");
    }
    result["us"] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count()/repeats;
    result["orbits"] = Json::arrayValue;
    auto coefficients = [](const std::vector<Q>& row) {
        Json::Value out(Json::arrayValue);
        for (const auto& c : row) out.append(c.get_str());
        return out;
    };
    for (const auto& orbit : locus.orbits) {
        Json::Value entry;
        entry["mu"] = coefficients(orbit.mu);
        entry["coordinates"] = Json::arrayValue;
        for (const auto& coordinate : orbit.coordinates) entry["coordinates"].append(coefficients(coordinate));
        result["orbits"].append(entry);
    }
    std::cout << result << '\n';
}
