// Chart polynomial products and substitutions above the FLINT threshold agree
// with term-by-term arithmetic, for integer, fractional and mixed coefficients.
#include "chart_expr.hpp"

#include <iostream>
#include <random>

using namespace atlas::chart;

namespace {

std::map<Powers, Q> sum_terms(const std::vector<Coord>& over, const Expr& a, const Expr& b) {
    auto terms = spread(a, over);
    for (const auto& [e, c] : spread(b, over)) terms[e] += c;
    return terms;
}

Expr reference_product(const Expr& a, const Expr& b) {
    const auto over = union_coords(a, b);
    std::map<Powers, Q> terms;
    for (const auto& [ea, ca] : spread(a, over))
        for (const auto& [eb, cb] : spread(b, over)) {
            Powers e(over.size());
            for (size_t i = 0; i < e.size(); ++i) e[i] = ea[i] + eb[i];
            terms[e] += ca * cb;
        }
    return normal(over, terms);
}

Expr reference_substitute(const Expr& f, const std::vector<std::pair<Coord, Expr>>& images) {
    Expr sum;
    for (const auto& [e, c] : f.terms()) {
        Expr term = constant(c);
        for (size_t i = 0; i < f.coords().size(); ++i) {
            Expr base = symbol(f.coords()[i]);
            for (const auto& [v, image] : images)
                if (v == f.coords()[i]) base = image;
            for (Z k = 0; k < e[i]; ++k) term = reference_product(term, base);
        }
        sum = normal(union_coords(sum, term), sum_terms(union_coords(sum, term), sum, term));
    }
    return sum;
}

}  // namespace

int main() {
    std::mt19937_64 random(20261115);
    const std::vector<Coord> pool = {{"a", 0}, {"b", 0}, {"b", 7}, {"c", 0}, {"d", 3}};
    int kind = 0;  // 0 mixed, 1 integers, 2 fractions
    auto value = [&](size_t terms, long degree, size_t width) {
        std::vector<Coord> coords(pool.begin(), pool.begin() + static_cast<long>(width));
        std::map<Powers, Q> m;
        for (size_t t = 0; t < terms; ++t) {
            Powers e(coords.size());
            for (auto& x : e) x = static_cast<long>(random() % static_cast<unsigned long>(degree + 1));
            Q c(static_cast<long>(random() % 2001) - 1000,
                kind == 1 ? 1L : static_cast<long>(random() % 40) + (kind == 2 ? 2L : 1L));
            c.canonicalize();
            m[e] += c;
        }
        return normal(coords, m);
    };
    int failures = 0, checks = 0;
    for (int round = 0; round < 90; ++round) {
        kind = round % 3;
        const Expr a = value(20 + random() % 40, 5, 2 + random() % 3);
        const Expr b = value(20 + random() % 40, 5, 2 + random() % 3);
        ++checks;
        if (a * b != reference_product(a, b)) {
            ++failures;
            std::cerr << "product differs in round " << round << "\n";
        }
        const Expr f = value(30 + random() % 30, 3, 3 + random() % 2);
        const auto& cs = f.coords();
        std::vector<std::pair<Coord, Expr>> images;
        images.emplace_back(cs[0], symbol(cs[0]) + scaled(symbol(cs[1]), Q(-3, 2)));
        images.emplace_back(cs[1], value(3, 2, 3));
        ++checks;
        if (substitute(f, images) != reference_substitute(f, images)) {
            ++failures;
            std::cerr << "substitution differs in round " << round << "\n";
        }
    }
    std::cout << "chart expressions: " << checks - failures << "/" << checks << " agree\n";
    return failures == 0 ? 0 : 1;
}
