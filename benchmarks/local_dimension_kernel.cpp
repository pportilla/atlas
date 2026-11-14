// Time the same CPU entry point as the standalone CLI on a pre-parsed QQ ideal.
#include "local_dimension.hpp"
#include "local_io.hpp"
#include <json/json.h>
#include <chrono>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>

static std::vector<std::string> fields(const std::string& line) {
    std::vector<std::string> result;
    std::istringstream stream(line);
    std::string item;
    while (std::getline(stream, item, ',')) result.push_back(item);
    return result;
}

int main(int argc, char** argv) {
    using namespace atlas::local;
    try {
        if (argc != 4) throw std::invalid_argument("Expected repetitions, operation and order.");
        const int repetitions = std::stoi(argv[1]);
        const std::string operation = argv[2], order = argv[3];
        if (repetitions < 1 || (order != "lex" && order != "grevlex") ||
            (operation != "local-dimension" && operation != "local-smoothness"))
            throw std::invalid_argument("Invalid timing arguments.");
        std::string variables, coordinates, generator;
        if (!std::getline(std::cin, variables) || !std::getline(std::cin, coordinates))
            throw std::invalid_argument("Expected variables and point.");
        const auto names = fields(variables);
        std::vector<mpq_class> point;
        for (const auto& coordinate : fields(coordinates)) {
            mpq_class q(coordinate, 10);
            if (q.get_den() == 0) throw std::invalid_argument("Zero denominator.");
            q.canonicalize();
            point.push_back(q);
        }
        if (coordinates.empty()) point.resize(names.size());
        if (point.size() != names.size()) throw std::invalid_argument("Point arity mismatch.");
        Basis ideal;
        while (std::getline(std::cin, generator)) {
            auto f = parse_expression(names, generator);
            if (!f.empty()) ideal.push_back(std::move(f));
        }
        const bool smoothness = operation == "local-smoothness";
        auto run = [&]() -> std::optional<DimensionResult> {
            try { return dimension_at_point(ideal, names, point, order == "lex", smoothness); }
            catch (const PointOutsideScheme&) { return std::nullopt; }
        };
        auto result = run();
        using Clock = std::chrono::steady_clock;
        auto start = Clock::now();
        for (int i = 0; i < repetitions; ++i) result = run();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
        Json::Value out;
        out["elapsed_ns"] = Json::Int64(elapsed);
        if (!result) out["off_scheme"] = true;
        else {
            out["local_dimension"] = result->dimension;
            if (smoothness) {
                out["value"] = result->smooth;
                out["embedding_dimension"] = result->embedding_dimension;
                out["jacobian_rank"] = result->jacobian_rank;
            } else out["value"] = result->dimension;
        }
        Json::StreamWriterBuilder format;
        format["indentation"] = "";
        std::cout << Json::writeString(format, out) << '\n';
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
