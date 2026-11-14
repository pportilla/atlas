// Primary decomposition shortcuts for monomial, principal, graph and comaximal ideals over QQ.
#include "primary_decomposition.hpp"
#include "sparse_flint.hpp"
#include <algorithm>
#include <iterator>
#include <map>
#include <numeric>

namespace atlas::primary {
namespace {
using local::Powers;
using Monomials = std::vector<Powers>;

bool divides(const Powers& a, const Powers& b) {
    return std::equal(a.begin(), a.end(), b.begin(), std::less_equal<>());
}

// Insert only minimal generators of a monomial ideal.
void insert_generator(Monomials& generators, Powers row) {
    for (const auto& g : generators) if (divides(g, row)) return;
    generators.erase(std::remove_if(generators.begin(), generators.end(),
                                    [&](const auto& g) { return divides(row, g); }), generators.end());
    generators.push_back(std::move(row));
}

Monomials intersect_monomials(const Monomials& a, const Monomials& b) {
    Monomials result;
    for (const auto& u : a) for (const auto& v : b) {
        Powers lcm(u.size());
        std::transform(u.begin(), u.end(), v.begin(), lcm.begin(),
                       [](const auto& x, const auto& y) { return std::max(x, y); });
        insert_generator(result, std::move(lcm));
    }
    return result;
}

// A zero bound denotes infinity. The box contains all exponent vectors strictly
// below every finite bound. Maximal boxes correspond to irredundant irreducibles.
bool contained(const Powers& a, const Powers& b) {
    for (size_t i = 0; i < a.size(); ++i)
        if (b[i] > 0 && (a[i] == 0 || a[i] > b[i])) return false;
    return true;
}

void insert_box(Monomials& boxes, Powers corner) {
    for (const auto& b : boxes) if (contained(corner, b)) return;
    boxes.erase(std::remove_if(boxes.begin(), boxes.end(), [&](const auto& b) { return contained(b, corner); }),
                boxes.end());
    boxes.push_back(std::move(corner));
}

// Descending order of leading monomials in the output ring.
bool greater_monomial(const Powers& a, const Powers& b, bool lex) {
    if (lex) return a > b;
    const auto da = std::accumulate(a.begin(), a.end(), mpz_class(0));
    const auto db = std::accumulate(b.begin(), b.end(), mpz_class(0));
    if (da != db) return da > db;
    return std::lexicographical_compare(a.rbegin(), a.rend(), b.rbegin(), b.rend());
}

Basis monomial_basis(Monomials rows, bool lex) {
    std::sort(rows.begin(), rows.end(), [&](const auto& a, const auto& b) { return greater_monomial(a, b, lex); });
    Basis out;
    for (auto& powers : rows) out.push_back({{std::move(powers), 1}});
    return out;
}

Components decompose_monomials(const Basis& input, size_t width, bool lex) {
    Monomials boxes{Powers(width)};
    for (const auto& f : input) {
        if (f.empty()) continue;
        const auto& g = f.begin()->first;
        Monomials next;
        for (const auto& box : boxes) {
            bool removed = true;
            for (size_t i = 0; i < width; ++i)
                if (box[i] != 0 && g[i] >= box[i]) { removed = false; break; }
            if (!removed) {
                insert_box(next, box);
                continue;
            }
            for (size_t i = 0; i < width; ++i) if (g[i] > 0) {
                auto child = box;
                child[i] = g[i];
                insert_box(next, std::move(child));
            }
        }
        boxes = std::move(next);
    }

    // Intersections of primary ideals with the same radical stay primary.
    std::map<std::vector<size_t>, Monomials> groups;
    for (const auto& box : boxes) {
        std::vector<size_t> support;
        Monomials irreducible;
        for (size_t i = 0; i < width; ++i) if (box[i] > 0) {
            support.push_back(i);
            Powers e(width);
            e[i] = box[i];
            irreducible.push_back(std::move(e));
        }
        auto [it, fresh] = groups.try_emplace(support, irreducible);
        if (!fresh) it->second = intersect_monomials(it->second, irreducible);
    }
    Components result;
    for (auto& [support, primary] : groups) {
        Monomials prime;
        for (size_t i : support) {
            Powers e(width);
            e[i] = 1;
            prime.push_back(std::move(e));
        }
        result.push_back({monomial_basis(std::move(primary), lex), monomial_basis(std::move(prime), lex)});
    }
    return result;
}

struct Graph {
    size_t row;
    size_t variable;
    Polynomial image;  // x_variable in the ring with that coordinate removed.
    Polynomial prime_equation;
    bool thick;
};

Powers without_variable(Powers e, size_t i) {
    e.erase(e.begin() + i);
    return e;
}

bool graph_coordinate(const Polynomial& f, size_t i, size_t width) {
    Powers variable(width);
    variable[i] = 1;
    return f.count(variable) && std::none_of(f.begin(), f.end(), [&](const auto& term) {
        return term.first[i] != 0 && term.first != variable;
    });
}

std::optional<Graph> find_graph(const Basis& generators, size_t width) {
    for (size_t r = 0; r < generators.size(); ++r) {
        const auto& f = generators[r];
        for (size_t i = 0; i < width; ++i) {
            if (!graph_coordinate(f, i, width)) continue;
            // Substitution into a large power of a sum could make the input
            // much larger. Leave those cases to the general elimination path.
            if (f.size() > 2 && std::any_of(generators.begin(), generators.end(), [&](const auto& g) {
                    return std::any_of(g.begin(), g.end(), [&](const auto& t) { return t.first[i] > 2; });
                })) continue;
            Powers variable(width);
            variable[i] = 1;
            Polynomial image;
            for (const auto& [e, c] : f) if (e != variable)
                image.emplace(without_variable(e, i), -c / f.at(variable));
            return Graph{r, i, std::move(image), f, false};
        }
    }
    return std::nullopt;
}

Basis lift_graph(const Basis& basis, const Polynomial& equation, size_t i, size_t width, bool lex) {
    Basis lifted{equation};
    for (const auto& f : basis) {
        Polynomial g;
        for (const auto& [e, c] : f) {
            auto term = e;
            term.insert(term.begin() + i, 0);
            g.emplace(std::move(term), c);
        }
        lifted.push_back(std::move(g));
    }
    return reduced(lifted, width, lex);
}

Components decompose_graph(const Basis& generators, const Graph& graph, size_t width, bool lex) {
    const Polynomial one{{Powers(width - 1), 1}};
    std::map<mpz_class, Polynomial> powers{{0, one}, {1, graph.image}};
    Basis restricted;
    for (size_t r = 0; r < generators.size(); ++r) if (r != graph.row) {
        auto g = substitute(generators[r], graph.variable, graph.image, width - 1, powers);
        if (!g.empty()) restricted.push_back(std::move(g));
    }
    // For a thick graph the coordinate is otherwise absent. Setting t=x-h,
    // adjoining t modulo t^e extends every component: the extension is free
    // over the base, and (R/Q)[t]/t^e has nilpotent zero divisors if R/Q does.
    Components result;
    for (const auto& part : decompose(restricted, width - 1, lex)) {
        Basis prime = lift_graph(part.prime, graph.prime_equation, graph.variable, width, lex);
        Basis primary = !graph.thick && part.primary == part.prime ? prime
            : lift_graph(part.primary, generators[graph.row], graph.variable, width, lex);
        result.push_back({std::move(primary), std::move(prime)});
    }
    return result;
}

} // namespace

std::optional<Components> split_comaximal(const Basis& rows, size_t width, bool lex) {
    // Examine small generators first. An irreducible or a repeated single
    // factor cannot give a CRT split.
    std::vector<size_t> candidates;
    for (size_t i = 0; i < rows.size(); ++i) if (rows[i].size() > 1) candidates.push_back(i);
    std::stable_sort(candidates.begin(), candidates.end(), [&](size_t i, size_t j) {
        return rows[i].size() < rows[j].size();
    });
    const Basis unit{Polynomial{{Powers(width), 1}}};
    std::vector<std::pair<Polynomial, mpz_class>> split;
    size_t index = 0;
    for (size_t i : candidates) {
        auto fs = factors(rows[i], width);
        if (fs.size() == 1 && fs.front().second > 1) {
            const auto& f = fs.front().first;
            for (size_t j = 0; j < width; ++j) {
                if (!graph_coordinate(f, j, width)) continue;
                bool independent = true;
                for (size_t k = 0; independent && k < rows.size(); ++k) if (k != i)
                    independent = std::none_of(rows[k].begin(), rows[k].end(),
                                               [&](const auto& term) { return term.first[j] != 0; });
                if (independent) return decompose_graph(rows, {i, j, {}, f, true}, width, lex);
            }
        }
        if (fs.size() < 2) continue;
        bool comaximal = true;
        for (size_t j = 1; comaximal && j < fs.size(); ++j)
            for (size_t k = 0; comaximal && k < j; ++k)
                comaximal = reduced({fs[j].first, fs[k].first}, width, false) == unit;
        if (comaximal) {
            index = i;
            split = std::move(fs);
            break;
        }
    }
    if (split.empty()) return std::nullopt;

    // In R/I these factors multiply to zero and their powers generate
    // pairwise comaximal ideals. CRT splits R/I into their quotient rings;
    // primary components from different branches have disjoint supports.
    local::SparseLexRing ring(width);
    atlas::PolyValue chosen(ring.ctx), other(ring.ctx), numerator(ring.ctx), quotient(ring.ctx);
    Components result;
    for (const auto& [factor, exponent] : split) {
        Basis branch{power(factor, exponent, width)};
        ring.write(chosen.p, branch.front());
        ring.write(numerator.p, rows[index]);
        if (!fmpq_mpoly_divides(other.p, numerator.p, chosen.p, ring.ctx))
            throw std::logic_error("Factor did not divide the CRT generator.");
        for (size_t k = 0; k < rows.size(); ++k) if (k != index && !rows[k].empty()) {
            ring.write(numerator.p, rows[k]);
            if (fmpq_mpoly_divides(quotient.p, numerator.p, chosen.p, ring.ctx)) continue;
            // The cofactor is a unit modulo the chosen power, so cancel its
            // common factors from every remaining generator of this branch.
            atlas::PolyValue gcd(ring.ctx);
            if (!fmpq_mpoly_gcd(gcd.p, other.p, numerator.p, ring.ctx))
                throw std::runtime_error("Polynomial gcd did not finish.");
            if (fmpq_mpoly_is_fmpq(gcd.p, ring.ctx)) branch.push_back(rows[k]);
            else {
                if (!fmpq_mpoly_divides(quotient.p, numerator.p, gcd.p, ring.ctx))
                    throw std::logic_error("Inexact CRT generator cancellation.");
                branch.push_back(ring.read(quotient.p));
            }
        }
        auto parts = decompose(branch, width, lex);
        result.insert(result.end(), std::make_move_iterator(parts.begin()), std::make_move_iterator(parts.end()));
    }
    return result;
}

std::optional<Components> special_decomposition(const Basis& generators, size_t width, bool lex) {
    if (std::all_of(generators.begin(), generators.end(), [](const auto& f) { return f.size() <= 1; }))
        return decompose_monomials(generators, width, lex);

    const Polynomial* principal = nullptr;
    for (const auto& f : generators) if (!f.empty()) {
        if (principal) {
            if (auto graph = find_graph(generators, width))
                return decompose_graph(generators, *graph, width, lex);
            return std::nullopt;
        }
        principal = &f;
    }
    Components result;
    // An irreducible in QQ[x] is prime, and its positive powers are primary.
    for (auto& [p, exponent] : factors(*principal, width)) {
        // The factorizer already made p lex-monic. A singleton is its own
        // reduced basis once normalized for the requested monomial order.
        if (!lex) {
            auto head = std::max_element(p.begin(), p.end(), [](const auto& a, const auto& b) {
                return greater_monomial(b.first, a.first, false);
            });
            const auto lc = head->second;
            if (lc != 1) for (auto& term : p) term.second /= lc;
        }
        Basis primary{exponent == 1 ? p : power(p, exponent, width)};
        result.push_back({std::move(primary), {std::move(p)}});
    }
    return result;
}
} // namespace atlas::primary
