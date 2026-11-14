// Time the public C++ arithmetic on the same pre-parsed polynomials as Singular.
#include "quotient_ring.hpp"
#include "local_io.hpp"
#include <chrono>
#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
    try {
        if (argc != 6) throw std::invalid_argument("Expected action, order, exponent, batch size and scope.");
        const std::string action(argv[1]), order(argv[2]), scope(argv[5]);
        const mpz_class exponent(argv[3], 10);
        const int batch = std::stoi(argv[4]);
        if ((scope != "prepared" && scope != "setup") || batch <= 0)
            throw std::invalid_argument("Invalid timing arguments.");
        std::string line;
        std::getline(std::cin, line);
        std::istringstream variables(line);
        std::vector<std::string> names;
        for (std::string name; variables >> name;) names.push_back(name);
        std::getline(std::cin, line);
        const size_t count = std::stoul(line);
        atlas::local::Basis relations, operands;
        while (std::getline(std::cin, line)) {
            auto& destination = relations.size() < count ? relations : operands;
            destination.push_back(atlas::local::decode_row(line, names.size()));
        }
        using atlas::quotient_ring::Ring;
        Ring ready(relations, names.size(), order == "lex");
        auto result = ready.evaluate(action, operands, exponent);
        auto start = std::chrono::steady_clock::now();
        if (scope == "prepared") {
            for (int i = 0; i < batch; ++i)
                result = ready.evaluate(action, operands, exponent);
        } else {
            for (int i = 0; i < batch; ++i) {
                Ring ring(relations, names.size(), order == "lex");
                result = ring.evaluate(action, operands, exponent);
            }
        }
        auto elapsed = std::chrono::steady_clock::now() - start;
        Json::StreamWriterBuilder writer;
        writer["indentation"] = "";
        std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() << '\n'
                  << Json::writeString(writer, ready.render(result, names)) << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
