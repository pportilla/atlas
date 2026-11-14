// Use the existing Atlas expression parser and FLINT pullback in the CLI process.
#include "local_io.hpp"
#include <algorithm>
#include <charconv>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>

extern "C" {
char* atlas_affine(const char*, size_t) noexcept;
void atlas_groebner_release(char*) noexcept;
const char* atlas_groebner_error() noexcept;
}

namespace atlas::local {
namespace {
std::string affine(const std::string& request) {
    std::unique_ptr<char, decltype(&atlas_groebner_release)> reply(
        atlas_affine(request.data(), request.size()), atlas_groebner_release);
    if (!reply) throw std::runtime_error(atlas_groebner_error());
    return reply.get();
}

std::string ring_line(const std::vector<std::string>& names) {
    std::string text;
    for (const auto& name : names) text += name + " ";
    return text + "\n";
}

Polynomial decode(std::string_view row, size_t width) {
    Polynomial result;
    while (!row.empty()) {
        const auto end = row.find(';');
        auto term = row.substr(0, end);
        const auto pos = term.find(' ');
        if (pos == term.npos) throw std::runtime_error("Invalid sparse coefficient.");
        mpq_class coefficient(std::string(term.substr(0, pos)), 10);
        const char* p = term.data() + pos;
        const char* limit = term.data() + term.size();
        Powers powers(width);
        unsigned long total = 0;
        for (auto& e : powers) {
            while (p != limit && *p == ' ') ++p;
            auto parsed = std::from_chars(p, limit, e);
            if (parsed.ec != std::errc() || e > std::numeric_limits<unsigned long>::max() - total)
                throw std::overflow_error("Wide exponent requires the general kernel.");
            total += e;
            p = parsed.ptr;
        }
        if (p != limit) throw std::runtime_error("Invalid sparse exponent vector.");
        result.emplace(std::move(powers), std::move(coefficient));
        if (end == row.npos) break;
        row.remove_prefix(end + 1);
    }
    return result;
}
}

Polynomial parse_expression(const std::vector<std::string>& names, std::string source) {
    // The public grammar treats newlines as whitespace. The kernel protocol is line based.
    for (char& ch : source) if (ch == '\n' || ch == '\r') ch = ' ';
    auto row = affine("parse_input\n" + ring_line(names) + source + "\n");
    if (row.empty() || row.back() != '\n') throw std::runtime_error("Incomplete polynomial.");
    row.pop_back();
    return decode(row, names.size());
}

std::string format_polynomial(const Polynomial& f, const std::vector<std::string>& names) {
    if (f.empty()) return "0";
    using Entry = Polynomial::value_type;
    std::vector<std::pair<unsigned long, const Entry*>> sorted;
    sorted.reserve(f.size());
    for (const auto& t : f)
        sorted.emplace_back(std::accumulate(t.first.begin(), t.first.end(), 0UL), &t);
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first > b.first : a.second->first > b.second->first;
    });
    std::string text;
    for (const auto& [degree, entry] : sorted) {
        auto c = entry->second.get_str();
        if (c.front() == '-') { text += '-'; c.erase(0, 1); }
        else if (!text.empty()) text += '+';
        bool factor = c != "1" || degree == 0;
        if (factor) text += c;
        for (size_t i = 0; i < names.size(); ++i) {
            const auto k = entry->first[i];
            if (!k) continue;
            if (factor) text += '*';
            text += names[i];
            if (k > 1) text += "^" + std::to_string(k);
            factor = true;
        }
    }
    return text;
}

Basis translate(const Basis& inputs, const std::vector<mpq_class>& shifts,
                const std::vector<std::string>& names) {
    if (inputs.empty() || std::all_of(shifts.begin(), shifts.end(), [](const auto& a) { return a == 0; }))
        return inputs;
    auto ring = ring_line(names);
    std::string request = "pullbacks\n" + ring + ring;
    for (size_t i = 0; i < names.size(); ++i)
        request += names[i] + (shifts[i] == 0 ? "" : "+(" + shifts[i].get_str() + ")") + "\n";
    request += "0\n";
    for (const auto& f : inputs) {
        // Defer enormous full expansions to the general point handler, which can
        // remove unit factors or use a wide sparse representation before translating.
        for (const auto& [e, c] : f) {
            unsigned long count = 1;
            for (size_t i = 0; i < e.size(); ++i) {
                if (shifts[i] == 0) continue;
                if (e[i] >= 1000000 / count) throw std::overflow_error("Large coordinate translation.");
                count *= e[i] + 1;
            }
        }
        request += format_polynomial(f, names) + "\n";
    }
    std::istringstream rows(affine(request));
    Basis output;
    for (std::string row; std::getline(rows, row);) output.push_back(decode(row, names.size()));
    if (output.size() != inputs.size()) throw std::runtime_error("Incomplete coordinate translation.");
    return output;
}
}
