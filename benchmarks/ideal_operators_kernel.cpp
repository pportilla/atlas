// Time the production arithmetic on parsed ideals. Read the same sparse rows as the wire benchmark.
#include "prepared_ideal.hpp"
#include "local_io.hpp"
#include <chrono>
#include <iostream>
#include <sstream>

extern "C" void atlas_groebner_release(char*) noexcept;

namespace {
std::string line() {
    std::string text;
    if (!std::getline(std::cin, text)) throw std::invalid_argument("Incomplete ideal request.");
    return text;
}

using Handle = std::unique_ptr<void, decltype(&atlas_ideal_release)>;

Handle owned(void* p) {
    if (!p) {
        const char* error = atlas_ideal_error();
        throw std::runtime_error(*error ? error : "The calculation exceeded the in-process budget.");
    }
    return Handle(p, atlas_ideal_release);
}

Handle prepare(size_t width, const std::string& rows, size_t count) {
    return owned(atlas_ideal_prepare(width, rows.c_str(), count));
}

template<class Calculate>
Handle timed(unsigned batch, Calculate calculate) {
    auto result = owned(calculate());
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < batch; ++i) result = owned(calculate());
    const auto elapsed = std::chrono::steady_clock::now() - start;
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() << '\n';
    return result;
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Expected the batch size as the only argument.");
        const int batch = std::stoi(argv[1]);
        if (batch < 1) throw std::invalid_argument("The batch size must be positive.");
        const std::string operation = line();
        std::istringstream variables(line());
        std::vector<std::string> names;
        for (std::string name; variables >> name;) names.push_back(name);
        const auto order = line();
        if (order != "lex" && order != "grevlex") throw std::invalid_argument("Unknown order.");

        Handle result(nullptr, atlas_ideal_release);
        if (operation == "ideal_power") {
            const auto exponent = std::stoul(line());
            std::string rows, row;
            size_t count = 0;
            while (std::getline(std::cin, row)) {
                rows += row + '\n';
                ++count;
            }
            auto left = prepare(names.size(), rows, count);
            result = timed(batch, [&] { return atlas_ideal_power_small(left.get(), exponent, order == "lex"); });
        } else {
            if (operation != "ideal_sum" && operation != "ideal_product") throw std::invalid_argument("Unknown operation.");
            std::istringstream counts(line());
            int left_count, right_count;
            if (!(counts >> left_count >> right_count) || left_count < 0 || right_count < 0)
                throw std::invalid_argument("Expected two nonnegative generator counts.");
            auto read = [&](int count) {
                std::string rows;
                for (int i = 0; i < count; ++i) rows += line() + '\n';
                return prepare(names.size(), rows, count);
            };
            const auto left = read(left_count);
            const auto right = read(right_count);
            const bool product = operation == "ideal_product";
            result = timed(batch, [&] { return atlas_ideal_binary(left.get(), right.get(), product); });
        }
        std::unique_ptr<char, decltype(&atlas_groebner_release)> text(atlas_ideal_rows(result.get()), atlas_groebner_release);
        if (!text) throw std::runtime_error(atlas_ideal_error());
        std::istringstream rows(text.get());
        for (std::string row; std::getline(rows, row);) {
            auto f = atlas::local::decode_row(row, names.size());
            std::cout << atlas::local::format_polynomial(f, names) << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
