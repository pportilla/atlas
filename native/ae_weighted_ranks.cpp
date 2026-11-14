#include "ae_weighted_ranks.hpp"
#include "small_prime_rank.hpp"
#include <flint/fmpz.h>
#include <flint/fmpz_mat.h>
#include <algorithm>
#include <cstdint>
#include <unordered_map>

namespace atlas::local {
namespace {

// These bounds limit an optional direct calculation. Unfinished ranks must
// be handed to the general solver, including zero pieces without a tail proof.
constexpr int degree_limit = 96;
constexpr size_t matrix_limit = 180000;
constexpr size_t coefficient_bits = 4096;
constexpr size_t monomial_limit = 12000;
constexpr ulong prime = 65521;
struct WorkLimit {};

// Positive weights and the degree bound ensure each exponent fits in a byte.
// Within those bounds, multiplying two monomials is carry-free packed addition.
static_assert(degree_limit < 256);
using Packed = std::uint64_t;
using IntegerPolynomial = std::unordered_map<Packed,mpz_class>;
Packed variable(int axis) { return Packed{1} << (8*axis); }
Packed pack(const Powers& powers) {
    Packed m = 0;
    for (size_t i = 0; i < powers.size(); ++i) m += powers[i].get_ui()*variable(i);
    return m;
}

// Appending a variable no smaller than the last gives each monomial one parent.
struct Monomials {
    struct Term { Packed powers; int last; };
    const std::vector<int>& weights;
    std::vector<std::vector<Term>> layers;
    std::unordered_map<Packed,int> column;
    size_t count = 1;

    explicit Monomials(const std::vector<int>& w) : weights(w) {
        const Packed one = 0;
        layers.push_back({{one,0}});
        column.emplace(one,0);
    }

    const std::vector<Term>& layer(int degree) {
        static const std::vector<Term> empty;
        if (degree < 0) return empty;
        if (degree > degree_limit) throw WorkLimit{};
        for (int d = layers.size(); d <= degree; ++d) {
            auto& terms = layers.emplace_back();
            for (int i = 0; i < static_cast<int>(weights.size()); ++i) {
                if (weights[i] > d) continue;
                for (const auto& parent : layers[d-weights[i]]) if (parent.last <= i) {
                    if (++count > monomial_limit) throw WorkLimit{};
                    const auto powers = parent.powers + variable(i);
                    column.emplace(powers,terms.size());
                    terms.push_back({powers,i});
                }
            }
        }
        return layers[degree];
    }
};

struct Table {
    nmod_mat_t residues;
    fmpz_mat_t exact;
    Table(int rows, int cols) {
        nmod_mat_init(residues,rows,cols,prime);
        fmpz_mat_init(exact,rows,cols);
    }
    ~Table() {
        nmod_mat_clear(residues);
        fmpz_mat_clear(exact);
    }
    void set(int row, int col, const mpz_class& value) {
        fmpz_set_mpz(fmpz_mat_entry(exact,row,col),value.get_mpz_t());
        nmod_mat_entry(residues,row,col) = mpz_fdiv_ui(value.get_mpz_t(),prime);
    }
    int corank(int rows, int cols) {
        const slong lower = modular_rank_in_place<prime>(residues);
        if (lower == std::min(rows,cols)) return cols-lower;
        return cols-fmpz_mat_rank(exact);
    }
};

struct WeightedRanks {
    std::vector<int> weights, degrees;
    std::vector<IntegerPolynomial> map;
    struct Derivative { int component; Packed shift; mpz_class coefficient; };
    std::vector<std::vector<Derivative>> jacobian;
    Monomials source, target;
    std::unordered_map<Packed,IntegerPolynomial> images;
    size_t remaining_cells = 1200000;
    size_t remaining_products = 80000;

    WeightedRanks(const Basis& input, std::vector<int> w, std::vector<int> d)
        : weights(std::move(w)), degrees(std::move(d)), map(input.size()),
          jacobian(weights.size()), source(weights), target(degrees) {
        // Nonzero independent target scalings preserve the extended tangent
        // quotient. Clear denominators so every rank row is integral.
        for (size_t j = 0; j < map.size(); ++j) {
            const auto& component = input[j];
            mpz_class den = 1;
            for (const auto& [e,c] : component) {
                if (mpz_sizeinbase(c.get_den_mpz_t(),2) > coefficient_bits ||
                    mpz_sizeinbase(c.get_num_mpz_t(),2) > coefficient_bits) throw WorkLimit{};
                den = lcm(den,c.get_den());
                if (mpz_sizeinbase(den.get_mpz_t(),2) > coefficient_bits) throw WorkLimit{};
            }
            for (const auto& [e,c] : component) {
                mpz_class coefficient = c.get_num() * (den / c.get_den());
                if (mpz_sizeinbase(coefficient.get_mpz_t(),2) > coefficient_bits) throw WorkLimit{};
                const Packed monomial = pack(e);
                map[j].emplace(monomial,coefficient);
                for (size_t i = 0; i < e.size(); ++i) {
                    if (e[i]) jacobian[i].push_back({static_cast<int>(j), monomial - variable(i),
                                                     coefficient * e[i]});
                }
            }
        }
        images.emplace(0,IntegerPolynomial{{0,1}});
    }

    // Each weight is visited in order. A target monomial whose image is
    // needed here already had its lower-weight parent at an earlier piece.
    const IntegerPolynomial& image(const Monomials::Term& monomial) {
        if (const auto it = images.find(monomial.powers); it != images.end()) return it->second;
        const auto parent = monomial.powers - variable(monomial.last);
        const auto& previous = images.at(parent);
        const auto& factor = map[monomial.last];
        if (previous.size() > remaining_products/factor.size()) throw WorkLimit{};
        remaining_products -= previous.size()*factor.size();
        IntegerPolynomial product;
        for (const auto& [e,c] : previous) for (const auto& [f,d] : factor)
            mpz_addmul(product[e+f].get_mpz_t(),c.get_mpz_t(),d.get_mpz_t());
        for (auto it = product.begin(); it != product.end();) {
            if (it->second == 0) {
                it = product.erase(it);
            } else {
                if (mpz_sizeinbase(it->second.get_mpz_t(),2) > coefficient_bits) throw WorkLimit{};
                ++it;
            }
        }
        return images.emplace(monomial.powers,std::move(product)).first->second;
    }

    int piece(int weight) {
        int cols = 0, rows = 0;
        std::vector<int> offset;
        for (int degree : degrees) {
            offset.push_back(cols);
            cols += source.layer(weight+degree).size();
            rows += target.layer(weight+degree).size();
        }
        for (int w : weights) rows += source.layer(weight+w).size();
        if (!rows || !cols) return cols;
        const auto cells = static_cast<size_t>(rows)*cols;
        if (cells > matrix_limit || cells > remaining_cells) throw WorkLimit{};
        remaining_cells -= cells;
        Table matrix(rows,cols);
        int row = 0;
        // Each degree-w tangent relation is either x^a times a column of
        // df (|a|_w = w+weight(x_i)), or a pulled back target monomial.
        for (size_t i = 0; i < weights.size(); ++i) {
            for (const auto& m : source.layer(weight+weights[i])) {
                for (const auto& derivative : jacobian[i])
                    matrix.set(row,offset[derivative.component]+source.column.at(derivative.shift+m.powers),
                               derivative.coefficient);
                ++row;
            }
        }
        for (size_t j = 0; j < map.size(); ++j) {
            for (const auto& m : target.layer(weight+degrees[j])) {
                for (const auto& [e,c] : image(m))
                    matrix.set(row,offset[j]+source.column.at(e),c);
                ++row;
            }
        }
        return matrix.corank(rows,cols);
    }
};

} // namespace

std::optional<Json::Value> ae_weighted_ranks(const Basis& map, const Powers& grading,
                                            unsigned contact_order) {
    if (grading.empty() || map.empty() || grading.size() > sizeof(Packed) ||
        map.size() > sizeof(Packed) || !contact_order || contact_order > degree_limit)
        return std::nullopt;
    std::vector<int> weights, degrees;
    for (const auto& w : grading) {
        if (w <= 0 || w > degree_limit) return std::nullopt;
        weights.push_back(w.get_ui());
    }
    for (const auto& f : map) {
        if (f.empty()) return std::nullopt;
        mpz_class degree = 0;
        for (const auto& term : f) {
            const auto& powers = term.first;
            if (powers.size() != weights.size()) return std::nullopt;
            mpz_class current = 0;
            for (size_t i = 0; i < weights.size(); ++i) {
                if (powers[i] < 0) return std::nullopt;
                current += powers[i]*weights[i];
            }
            if (current <= 0 || current > degree_limit || (degree && degree != current))
                return std::nullopt;
            degree = std::move(current);
        }
        degrees.push_back(degree.get_ui());
    }
    const int largest_degree = *std::max_element(degrees.begin(),degrees.end());
    const int largest_generator = (static_cast<int>(contact_order)-1) *
        *std::max_element(weights.begin(),weights.end()) - *std::min_element(degrees.begin(),degrees.end());
    if (largest_generator + largest_degree > degree_limit) return std::nullopt;
    try {
        WeightedRanks ranks(map,std::move(weights),std::move(degrees));
        int dimension = 0, zeros = 0;
        for (int w = -largest_degree; w+largest_degree <= degree_limit; ++w) {
            const int part = ranks.piece(w);
            dimension += part;
            zeros = part ? 0 : zeros+1;
            // Above largest_generator, M_w = sum f_j M_(w-d_j) by the
            // contact certificate. This descends to the normal quotient.
            if (w >= largest_generator && zeros >= largest_degree) {
                Json::Value result(Json::objectValue);
                result["finite"] = true;
                result["value"] = std::to_string(dimension);
                result["method"] = "weighted_tangent_ranks";
                result["last_weight"] = w;
                result["contact_order"] = contact_order;
                return result;
            }
        }
    } catch (const WorkLimit&) {
        return std::nullopt;
    }
    return std::nullopt;
}

}
