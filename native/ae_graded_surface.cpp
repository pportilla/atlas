#include "ae_graded_surface.hpp"
#include "ae_surface_coordinates.hpp"
#include <flint/fmpz.h>
#include <flint/fmpz_mat.h>
#include <flint/nmod_mat.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>

namespace atlas::local {
namespace {

// The degree bounds apply only to this optional rank calculation.
constexpr int max_degree = 40;
struct Projection {
    int a, b, c;
    std::vector<mpz_class> h; // coefficients of x^i y^(c-i)
};

std::optional<Projection> projection(const Basis& map) {
    int x = -1, y = -1;
    for (int j = 0; j < 3; ++j) {
        if (map[j].size() != 1) continue;
        const auto& e = map[j].begin()->first;
        if (e[0] > 0 && !e[1]) x = j;
        if (e[1] > 0 && !e[0]) y = j;
    }
    if (x < 0 || y < 0) return std::nullopt;
    const auto& a = map[x].begin()->first[0];
    const auto& b = map[y].begin()->first[1];
    const auto& h = map[3-x-y];
    if (h.empty() || a > max_degree || b > max_degree) return std::nullopt;
    const mpz_class c = h.begin()->first[0]+h.begin()->first[1];
    if (c <= 0 || c > max_degree) return std::nullopt;
    // Independent nonzero target scalings make the projection monic and h integral.
    mpz_class den = 1;
    for (const auto& [e,q] : h) {
        if (e[0]+e[1] != c) return std::nullopt;
        mpz_lcm(den.get_mpz_t(),den.get_mpz_t(),q.get_den_mpz_t());
    }
    std::vector<mpz_class> coeffs(c.get_ui()+1);
    mpz_class content = 0;
    for (const auto& [e,q] : h) {
        auto& coeff = coeffs[e[0].get_ui()];
        coeff = (den/q.get_den())*q.get_num();
        mpz_gcd(content.get_mpz_t(),content.get_mpz_t(),coeff.get_mpz_t());
    }
    if (content > 1) for (auto& q : coeffs) q /= content;
    for (const auto& q : coeffs)
        if (mpz_sizeinbase(q.get_mpz_t(),2) > 4096) return std::nullopt;
    return Projection{static_cast<int>(a.get_ui()),static_cast<int>(b.get_ui()),
                      static_cast<int>(c.get_ui()),std::move(coeffs)};
}

struct Matrix {
    fmpz_mat_t m;
    Matrix(int rows, int cols) { fmpz_mat_init(m,rows,cols); }
    ~Matrix() { fmpz_mat_clear(m); }
};

constexpr unsigned long prime = 65521;
using Residues = std::vector<unsigned long>;
struct ModularMatrix {
    nmod_mat_t m;
    ModularMatrix(int rows, int cols) { nmod_mat_init(m,rows,cols,prime); }
    ~ModularMatrix() { nmod_mat_clear(m); }
};

Residues residues(const std::vector<mpz_class>& f) {
    Residues r;
    for (const auto& c : f) r.push_back(mpz_fdiv_ui(c.get_mpz_t(),prime));
    return r;
}

using Binary = std::vector<mpz_class>;
Binary product(const Binary& lhs, const Binary& rhs) {
    Binary result(lhs.size()+rhs.size()-1);
    for (size_t j=0; j<rhs.size(); ++j) if (rhs[j])
        for (size_t i=0; i<lhs.size(); ++i) if (lhs[i])
            mpz_addmul(result[i+j].get_mpz_t(), lhs[i].get_mpz_t(), rhs[j].get_mpz_t());
    return result;
}

struct Pullback { int front, x, power; };

struct TangentRanks {
    const Projection& f;
    std::vector<Binary> powers{{1}};
    std::vector<Residues> powers_mod{{1}};
    std::array<Binary,2> derivative;
    std::array<Residues,2> derivative_mod;
    std::vector<std::array<Residues,2>> carried_products;
    explicit TangentRanks(const Projection& map) : f(map) {
        derivative[0].resize(f.c);
        derivative[1].resize(f.c);
        for (int i=0; i<f.c; ++i) {
            derivative[0][i] = (i+1)*f.h[i+1];
            derivative[1][i] = (f.c-i)*f.h[i];
        }
        for (int j=0; j<2; ++j) derivative_mod[j] = residues(derivative[j]);
        cache_carried_products();
    }

    // Dividing a whole pullback by its Jacobian pivot leaves a monomial
    // shift of -h^k * dh/dx or -h^k * dh/dy. Reuse it across weights.
    void cache_carried_products() {
        auto& products = carried_products.emplace_back();
        const auto& power = powers_mod.back();
        const int degree = static_cast<int>(power.size())-1;
        for (int j=0; j<2; ++j) {
            auto& carried = products[j];
            carried.resize(degree+f.c);
            for (int i=0; i<degree+f.c; ++i) {
                // At most max_degree products of two residues fit in 64 bits.
                std::uint64_t sum = 0;
                for (int l=std::max(0,i-degree); l<=std::min(i,f.c-1); ++l)
                    sum += std::uint64_t(prime-derivative_mod[j][l]) * power[i-l];
                carried[i] = sum % prime;
            }
        }
    }

    // Coefficients written are identical over QQ and over a prime field. Use
    // canonical residues first, materializing the integer matrix only if an
    // upper rank bound still has to be proved.
    template<bool modular, class Table>
    void fill(Table& mat, const std::vector<Pullback>& rows, const std::array<int,3>& degrees,
              int first, int base) const {
        for (size_t r=0; r<rows.size(); ++r) {
            const auto& row = rows[r];
            const int j = row.front, x = row.x;
            const auto& coeff = [&]() -> const auto& {
                if constexpr (modular) return powers_mod[row.power]; else return powers[row.power];
            }();
            if (j==2) {
                // The Euler relation leaves only a monomial in x^a,y^b.
                if constexpr (modular) nmod_mat_entry(mat.m,r,base+x) = 1;
                else fmpz_one(fmpz_mat_entry(mat.m,r,base+x));
                continue;
            }
            // df/dx has first component a*x^(a-1), df/dy has second
            // component b*y^(b-1). They have disjoint pivots. For a target
            // row H in either position perform the monomial division, carrying
            // the eliminated terms into the third position via the derivative.
            const int pivot_degree = j==0 ? f.a-1 : f.b-1;
            if constexpr (modular) {
                const int y = degrees[j]-x-static_cast<int>(coeff.size())+1;
                if ((j==0 ? x : y) >= pivot_degree) {
                    const auto& carried = carried_products[row.power][j];
                    const int shift = j==0 ? x-pivot_degree : x;
                    std::copy(carried.begin(),carried.end(),&nmod_mat_entry(mat.m,r,base+shift));
                    continue;
                }
            }
            Binary carried;
            if constexpr (!modular) carried.resize(std::max(0,degrees[2]+1));
            for (int i=0; i<static_cast<int>(coeff.size()); ++i) if (coeff[i]) {
                const int power_x = x+i;
                const int exponent = j==0 ? power_x : degrees[j]-power_x;
                if (exponent < pivot_degree) {
                    const int col = j==0 ? exponent : first+exponent;
                    if constexpr (modular)
                        nmod_mat_entry(mat.m,r,col) = coeff[i]*(pivot_degree+1)%prime;
                    else {
                        auto entry = fmpz_mat_entry(mat.m,r,col);
                        fmpz_set_mpz(entry,coeff[i].get_mpz_t());
                        fmpz_mul_ui(entry,entry,pivot_degree+1);
                    }
                } else {
                    const int shift = power_x-(j==0 ? pivot_degree : 0);
                    for (int l=0; l<f.c; ++l) if (derivative[j][l]) {
                        if constexpr (modular) {
                            auto& entry = nmod_mat_entry(mat.m,r,base+shift+l);
                            entry = (entry + (prime-derivative_mod[j][l])*coeff[i])%prime;
                        } else {
                            mpz_submul(carried[shift+l].get_mpz_t(),derivative[j][l].get_mpz_t(),
                                       coeff[i].get_mpz_t());
                        }
                    }
                }
            }
            if constexpr (!modular)
                for (size_t i=0; i<carried.size(); ++i)
                    fmpz_set_mpz(fmpz_mat_entry(mat.m,r,base+i),carried[i].get_mpz_t());
        }
    }

    std::optional<int> piece(int w) {
        const std::array<int,3> degrees{w+f.a,w+f.b,w+f.c};
        const int first = std::clamp(degrees[0]+1,0,f.a-1);
        const int second = std::clamp(degrees[1]+1,0,f.b-1);
        const int base = first+second;
        const int cols = base+std::max(0,degrees[2]+1);
        std::vector<Pullback> rows;
        for (int j=0; j<3; ++j) {
            // Cayley-Hamilton on the free QQ[x^a,y^b]-module QQ[x,y]
            // bounds the powers of h needed to span the target algebra.
            for (int k=0; k<f.a*f.b && k*f.c<=degrees[j]; ++k) {
                const int d = degrees[j]-k*f.c;
                // Euler's homogeneous derivative identity gives the target
                // relation (a*x^a, b*y^b, c*h)*q=0 in the Jacobian
                // quotient for every target polynomial q. Eliminate the
                // third-position rows divisible by h using this relation.
                if (j == 2 && k) break;
                for (int i=0; i<=d; i+=f.a) if ((d-i)%f.b==0)
                    rows.push_back({j,i,k});
            }
        }
        if (rows.empty()) return cols;
        if (rows.size()*cols > 600000) return std::nullopt;
        for (const auto& generator : rows)
            while (static_cast<int>(powers.size()) <= generator.power) {
                powers.push_back(product(powers.back(),f.h));
                powers_mod.push_back(residues(powers.back()));
                cache_carried_products();
            }
        ModularMatrix mod(rows.size(),cols);
        fill<true>(mod,rows,degrees,first,base);
        const int lower = nmod_mat_rank(mod.m);
        // An invertible minor modulo a prime is invertible over QQ.
        if (lower == std::min(static_cast<int>(rows.size()),cols)) return cols-lower;
        Matrix mat(rows.size(),cols);
        fill<false>(mat,rows,degrees,first,base);
        return cols-fmpz_mat_rank(mat.m);
    }
};
}

std::optional<Json::Value> ae_graded_surface(const Basis& map) {
    if (map.size()!=3) return std::nullopt;
    auto f = projection(map);
    if (!f) {
        if (auto normal = normalized_surface_axes(map)) f = projection(*normal);
        if (!f) return std::nullopt;
    }
    // Restrict vectors to either coordinate line. In a target position
    // whose full gradient vanishes on that line only target pullbacks
    // survive, with orders generated by the nonzero restricted components.
    const bool missing_on_x = f->b > 1 && std::gcd(f->a, f->h.back() ? f->c : 0) > 1;
    const bool missing_on_y = f->a > 1 && std::gcd(f->b, f->h.front() ? f->c : 0) > 1;
    if (missing_on_x || missing_on_y) {
        Json::Value out;
        out["finite"] = false;
        out["value"] = Json::nullValue;
        out["method"] = "ramified_axis";
        return out;
    }
    // Common projection degrees often yield infinite rank along another
    // line. Let graph elimination decide this case without sampling jets.
    if (std::gcd(f->a,f->b) > 1) return std::nullopt;
    TangentRanks tangent(*f);
    const int largest_generator = f->a+f->b-2-std::min({f->a,f->b,f->c});
    int dimension = 0, zeros = 0;
    // Small projection modules should return promptly to graph elimination
    // when the zero-tail test is unsuccessful. Both bounds only select an
    // algorithm; neither yields a finite or infinite dimension.
    const int work_limit = std::min(160,f->a*f->b*f->c + std::max(f->a,f->b));
    for (int w = -std::max({f->a,f->b,f->c}); w <= work_limit; ++w) {
        const auto piece = tangent.piece(w);
        if (!piece) break;
        dimension += *piece;
        zeros = *piece ? 0 : zeros+1;
        // Beyond the largest weight of x^i y^j e_l (i<a, j<b),
        // N_w=x^a N_(w-a)+y^b N_(w-b). This block proves the full tail.
        if (w >= largest_generator && zeros >= std::max(f->a,f->b)) {
            Json::Value out;
            out["finite"] = true;
            out["value"] = std::to_string(dimension);
            out["method"] = "graded_tangent_ranks";
            out["last_weight"] = w;
            return out;
        }
    }
    return std::nullopt;
}
}
