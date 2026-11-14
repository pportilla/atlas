// Compare ideal-valued computations on pre-parsed polynomials in both systems.
#include "ideal_powers.hpp"
#include "local_io.hpp"
#include <chrono>
#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::invalid_argument("Expected operation, order, exponent and batch size.");
        const std::string op(argv[1]), order(argv[2]);
        const mpz_class exponent(argv[3],10);
        const int batch = std::stoi(argv[4]);
        if ((op != "ideal-power" && op != "symbolic-power") || (order != "lex" && order != "grevlex") || batch <= 0)
            throw std::invalid_argument("Invalid timing arguments.");
        std::string line;
        std::getline(std::cin,line);
        std::istringstream variables(line);
        std::vector<std::string> names;
        for (std::string v; variables >> v;) names.push_back(v);
        atlas::local::Basis rows;
        while (std::getline(std::cin,line)) rows.push_back(atlas::local::decode_row(line,names.size()));
        auto run = [&] {
            return atlas::ideal_powers::typed_power(rows,names.size(),order == "lex",exponent,op == "symbolic-power");
        };
        auto result = run();
        const auto start = std::chrono::steady_clock::now();
        for (int i=0; i<batch; ++i) result = run();
        const auto elapsed = std::chrono::steady_clock::now() - start;
        std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() << '\n';
        for (const auto& text : result.expressions(names)) std::cout << text << '\n';
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
