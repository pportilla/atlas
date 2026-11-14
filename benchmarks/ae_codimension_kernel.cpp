// Measure parsing, the complete kernel, or the graded elimination on the same map.
#include "ae_codimension.hpp"
#include "ae_graded.hpp"
#include "local_io.hpp"
#include <chrono>
#include <iostream>
#include <json/json.h>
#include <sstream>
#include <stdexcept>

int main(int argc, char** argv) {
    using namespace atlas::local;
    try {
        if (argc != 3) throw std::invalid_argument("Supply repetitions and mode (parse, solve, solve-once, graded).");
        const int repetitions = std::stoi(argv[1]);
        const std::string mode = argv[2];
        const bool single = mode == "solve-once";
        if (repetitions < 1 || (single && repetitions != 1) ||
            (!single && mode != "parse" && mode != "solve" && mode != "graded"))
            throw std::invalid_argument("Invalid timing arguments.");
        std::string line;
        std::getline(std::cin, line);
        std::istringstream ring(line);
        std::vector<std::string> names, components;
        while (std::getline(ring, line, ',')) names.push_back(line);
        while (std::getline(std::cin, line)) components.push_back(line);
        auto parse = [&] {
            Basis map;
            for (const auto& f : components) map.push_back(parse_expression(names, f));
            return map;
        };
        auto map = parse();
        Json::Value reference;
        Json::Value result;
        auto run = [&] {
            if (mode == "parse") map = parse();
            else if (single || mode == "solve") result = ae_codimension(map, names);
            else {
                const auto graded = ae_graded(map, names);
                if (!graded) throw std::invalid_argument("This map did not admit graded elimination.");
                result = *graded;
            }
        };
        if (!single) {
            reference = ae_codimension(map, names);
            run();
        }
        using Clock = std::chrono::steady_clock;
        const auto start = Clock::now();
        for (int i = 0; i < repetitions; ++i) run();
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start);
        if (mode == "parse") result = ae_codimension(map, names);
        if (!single && (result["finite"] != reference["finite"] || result["value"] != reference["value"]))
            throw std::logic_error("Timed result disagrees with the public solver.");
        result["elapsed_ns"] = Json::Int64(elapsed.count());
        Json::StreamWriterBuilder format;
        format["indentation"] = "";
        std::cout << Json::writeString(format, result) << '\n';
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
