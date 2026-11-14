// Sparse local algebra operations shared by the dispatcher and the native CLI.
#include "local_io.hpp"
#include "local_division.hpp"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace atlas::local {
namespace {
mpz_class degree(const Powers& powers) {
    return std::accumulate(powers.begin(), powers.end(), mpz_class(0));
}

const Powers& global_head(const Polynomial& f, bool lex) {
    if (lex) return f.rbegin()->first;
    return std::max_element(f.begin(), f.end(), [](const auto& a, const auto& b) {
        const auto da = degree(a.first), db = degree(b.first);
        if (da != db) return da < db;
        return std::lexicographical_compare(b.first.rbegin(), b.first.rend(), a.first.rbegin(), a.first.rend());
    })->first;
}

std::string basis_rows(const Basis& basis) {
    std::string out;
    for (const auto& f : basis) out += encode_row(f) + "\n";
    return out;
}

std::string line(std::istream& in) {
    std::string value;
    if (!std::getline(in, value)) throw std::invalid_argument("Incomplete local operation request.");
    return value;
}

std::vector<mpq_class> read_point(std::istream& in, size_t width) {
    std::istringstream fields(line(in));
    std::vector<mpq_class> result(width);
    for (auto& coordinate : result) {
        if (!(fields >> coordinate) || coordinate.get_den() == 0)
            throw std::invalid_argument("Invalid local point coordinate.");
        coordinate.canonicalize();
    }
    std::string extra;
    if (fields >> extra) throw std::invalid_argument("Incorrect local point dimension.");
    return result;
}
}

std::string operations(std::istream& in, const std::string& op) {
    std::istringstream names_line(line(in));
    std::vector<std::string> names;
    for (std::string name; names_line >> name;) names.push_back(std::move(name));
    const auto mode = line(in);
    const bool lex = mode == "lex";
    const auto width = names.size();
    const auto point = (op == "local_project" || op == "local_query_jet") ?
        read_point(in, width) : std::vector<mpq_class>();
    Basis rows;
    for (std::string row; std::getline(in, row);) rows.push_back(decode_row(row, width));
    if (op == "local_project") return basis_rows(project_monomials(rows, point));
    if (op == "local_basis" || op == "local_basis_gpu")
        return basis_rows(standard_basis(rows, names, lex, op == "local_basis_gpu"));
    if (op == "local_query_jet" || op == "local_normal_form") {
        if (rows.empty()) throw std::invalid_argument("A query polynomial is required.");
        auto query = std::move(rows.back());
        rows.pop_back();
        if (op == "local_query_jet") {
            auto jet = finite_query_jet(rows, query, point, lex);
            return jet ? encode_row(*jet) + "\n" : "";
        }
        auto answer = normal_form(rows, query, width, lex);
        return std::string(answer.weak ? "weak\n" : "standard\n") +
               encode_row(answer.unit) + "\n" + encode_row(answer.remainder) + "\n";
    }
    if (op == "local_partials") {
        if (rows.size() != 1) throw std::invalid_argument("Expected one polynomial for derivatives.");
        Basis partials(width);
        for (const auto& [powers, c] : rows.front())
            for (size_t i = 0; i < width; ++i)
                if (powers[i] != 0) {
                    auto e = powers;
                    --e[i];
                    partials[i].emplace(std::move(e), c * powers[i]);
                }
        return basis_rows(partials);
    }
    if (op == "local_heads" || op == "local_global_heads" || op == "local_associate_heads") {
        Basis result;
        for (const auto& f : rows) {
            if (f.empty()) continue;
            auto head = op == "local_global_heads" ? global_head(f, lex) : local_head(f, lex);
            if (op == "local_associate_heads")
                for (const auto& term : f)
                    if (!std::equal(head.begin(), head.end(), term.first.begin(), std::less_equal<>()))
                        return "";
            result.push_back({{std::move(head), 1}});
        }
        return basis_rows(result);
    }
    throw std::invalid_argument("Unknown local operation.");
}
}
