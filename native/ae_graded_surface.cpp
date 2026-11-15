#include "ae_graded_surface.hpp"
#include "ae_surface_coordinates.hpp"
#include "small_prime_rank.hpp"
#include <flint/fmpz.h>
#include <flint/fmpz_mat.h>
#include <flint/nmod_mat.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <stdexcept>

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

    int rank_without(const std::vector<int>& unit_columns) {
        // Keep monomials in degree order for fraction-free elimination.
        // Arbitrary column permutations can make its pivot integers larger.
        int kept = 0;
        auto unit = unit_columns.begin();
        for (int col = 0; col < fmpz_mat_ncols(m); ++col) {
            if (unit != unit_columns.end() && col == *unit) {
                ++unit;
            } else {
                if (kept != col) fmpz_mat_swap_cols(m,nullptr,kept,col);
                ++kept;
            }
        }
        fmpz_mat_t remaining;
        fmpz_mat_window_init(remaining,m,0,0,fmpz_mat_nrows(m),kept);
        const int rank = fmpz_mat_rank(remaining);
        fmpz_mat_window_clear(remaining);
        return rank;
    }
};

// Two primes below 2^16. The pieces are ranked modulo the larger one. The
// smaller one is tried on the unproved pieces only if no tail is proved.
constexpr unsigned long first_prime = 65521, second_prime = 65519;
using Residues = std::vector<unsigned long>;
template<unsigned long prime>
struct ModularMatrix {
    nmod_mat_t m;
    ModularMatrix(int rows, int cols) { nmod_mat_init(m,rows,cols,prime); }
    ~ModularMatrix() { nmod_mat_clear(m); }

    int rank_without(const std::vector<int>& unit_columns) {
        int end = nmod_mat_ncols(m);
        for (auto i = unit_columns.rbegin(); i != unit_columns.rend(); ++i)
            nmod_mat_swap_cols(m,nullptr,*i,--end);
        nmod_mat_t remaining;
        nmod_mat_window_init(remaining,m,0,0,nmod_mat_nrows(m),end);
        const int rank = modular_rank_in_place<prime>(remaining);
        nmod_mat_window_clear(remaining);
        return rank;
    }
};

template<unsigned long prime>
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

// The residues of h, of its two partial derivatives and of the powers of h
// modulo one prime. Reduction modulo the prime is a ring map, so the powers
// are the residues of the integer powers whether or not those were formed.
template<unsigned long prime>
struct ModularImage {
    int c;
    Residues h;
    std::array<Residues,2> derivative;
    std::vector<Residues> powers{{1}};
    std::vector<std::array<Residues,2>> carried_products;

    ModularImage(const Projection& f, const std::array<Binary,2>& exact_derivative)
        : c(f.c), h(residues<prime>(f.h)) {
        for (int j=0; j<2; ++j) derivative[j] = residues<prime>(exact_derivative[j]);
        cache_carried_products();
    }

    // Dividing a whole pullback by its Jacobian pivot leaves a monomial
    // shift of -h^k * dh/dx or -h^k * dh/dy. Reuse it across weights.
    void cache_carried_products() {
        auto& products = carried_products.emplace_back();
        const auto& power = powers.back();
        const int degree = static_cast<int>(power.size())-1;
        for (int j=0; j<2; ++j) {
            auto& carried = products[j];
            carried.resize(degree+c);
            for (int i=0; i<degree+c; ++i) {
                // At most max_degree products of two residues fit in 64 bits.
                std::uint64_t sum = 0;
                for (int l=std::max(0,i-degree); l<=std::min(i,c-1); ++l)
                    sum += std::uint64_t(prime-derivative[j][l]) * power[i-l];
                carried[i] = sum % prime;
            }
        }
    }

    void extend(int power) {
        while (static_cast<int>(powers.size()) <= power) {
            const auto& last = powers.back();
            Residues next(last.size()+c, 0);
            for (size_t i=0; i<last.size(); ++i) if (last[i])
                for (int l=0; l<=c; ++l)
                    next[i+l] = (next[i+l] + last[i]*h[l]) % prime;
            powers.push_back(std::move(next));
            cache_carried_products();
        }
    }
};

struct TangentRanks {
    const Projection& f;
    std::vector<Binary> powers{{1}};
    std::array<Binary,2> derivative;
    ModularImage<first_prime> first_image;
    ModularImage<second_prime> second_image;
    explicit TangentRanks(const Projection& map)
        : f(map), derivative(derivatives(map)), first_image(map,derivative), second_image(map,derivative) {}

    static std::array<Binary,2> derivatives(const Projection& f) {
        std::array<Binary,2> result;
        result[0].resize(f.c);
        result[1].resize(f.c);
        for (int i=0; i<f.c; ++i) {
            result[0][i] = (i+1)*f.h[i+1];
            result[1][i] = (f.c-i)*f.h[i];
        }
        return result;
    }

    // Coefficients written are identical over QQ and over a prime field. Use
    // canonical residues first, materializing the integer matrix only if an
    // upper rank bound still has to be proved.
    template<unsigned long prime>
    void fill_modular(ModularMatrix<prime>& mat, const ModularImage<prime>& image,
                      const std::vector<Pullback>& rows, const std::array<int,3>& degrees,
                      int first, int base) const {
        for (size_t r=0; r<rows.size(); ++r) {
            const auto& row = rows[r];
            const int j = row.front, x = row.x;
            const auto& coeff = image.powers[row.power];
            // df/dx has first component a*x^(a-1), df/dy has second
            // component b*y^(b-1). They have disjoint pivots. For a target
            // row H in either position perform the monomial division, carrying
            // the eliminated terms into the third position via the derivative.
            const int pivot_degree = j==0 ? f.a-1 : f.b-1;
            const int y = degrees[j]-x-static_cast<int>(coeff.size())+1;
            if ((j==0 ? x : y) >= pivot_degree) {
                const auto& carried = image.carried_products[row.power][j];
                const int shift = j==0 ? x-pivot_degree : x;
                std::copy(carried.begin(),carried.end(),&nmod_mat_entry(mat.m,r,base+shift));
                continue;
            }
            for (int i=0; i<static_cast<int>(coeff.size()); ++i) if (coeff[i]) {
                const int power_x = x+i;
                const int exponent = j==0 ? power_x : degrees[j]-power_x;
                if (exponent < pivot_degree) {
                    const int col = j==0 ? exponent : first+exponent;
                    nmod_mat_entry(mat.m,r,col) = coeff[i]*(pivot_degree+1)%prime;
                } else {
                    const int shift = power_x-(j==0 ? pivot_degree : 0);
                    for (int l=0; l<f.c; ++l) if (derivative[j][l]) {
                        auto& entry = nmod_mat_entry(mat.m,r,base+shift+l);
                        entry = (entry + (prime-image.derivative[j][l])*coeff[i])%prime;
                    }
                }
            }
        }
    }

    void fill_exact(Matrix& mat, const std::vector<Pullback>& rows, const std::array<int,3>& degrees,
                    int first, int base) const {
        for (size_t r=0; r<rows.size(); ++r) {
            const auto& row = rows[r];
            const int j = row.front, x = row.x;
            const auto& coeff = powers[row.power];
            const int pivot_degree = j==0 ? f.a-1 : f.b-1;
            Binary carried(std::max(0,degrees[2]+1));
            for (int i=0; i<static_cast<int>(coeff.size()); ++i) if (coeff[i]) {
                const int power_x = x+i;
                const int exponent = j==0 ? power_x : degrees[j]-power_x;
                if (exponent < pivot_degree) {
                    const int col = j==0 ? exponent : first+exponent;
                    auto entry = fmpz_mat_entry(mat.m,r,col);
                    fmpz_set_mpz(entry,coeff[i].get_mpz_t());
                    fmpz_mul_ui(entry,entry,pivot_degree+1);
                } else {
                    const int shift = power_x-(j==0 ? pivot_degree : 0);
                    for (int l=0; l<f.c; ++l) if (derivative[j][l])
                        mpz_submul(carried[shift+l].get_mpz_t(),derivative[j][l].get_mpz_t(),
                                   coeff[i].get_mpz_t());
                }
            }
            for (size_t i=0; i<carried.size(); ++i)
                fmpz_set_mpz(fmpz_mat_entry(mat.m,r,base+i),carried[i].get_mpz_t());
        }
    }

    // The rows and columns of the weight-w piece.
    struct Layout {
        std::array<int,3> degrees;
        int first, base, cols, remaining_cols;
        std::vector<int> unit_columns;
        std::vector<Pullback> rows;
    };

    Layout layout(int w) const {
        Layout piece;
        auto& degrees = piece.degrees;
        degrees = {w+f.a,w+f.b,w+f.c};
        piece.first = std::clamp(degrees[0]+1,0,f.a-1);
        const int second = std::clamp(degrees[1]+1,0,f.b-1);
        piece.base = piece.first+second;
        piece.cols = piece.base+std::max(0,degrees[2]+1);
        // Euler's identity removes third-position rows divisible by h. The
        // remaining third-position rows are distinct coordinate vectors
        // x^(ai)y^(bj). Quotient out their coordinates over either field.
        for (int i=0; i<=degrees[2]; i+=f.a) if ((degrees[2]-i)%f.b==0)
            piece.unit_columns.push_back(piece.base+i);
        piece.remaining_cols = piece.cols-static_cast<int>(piece.unit_columns.size());
        if (!piece.remaining_cols) return piece;
        for (int j=0; j<2; ++j) {
            // Cayley-Hamilton on the free QQ[x^a,y^b]-module QQ[x,y]
            // bounds the powers of h needed to span the target algebra.
            for (int k=0; k<f.a*f.b && k*f.c<=degrees[j]; ++k) {
                const int d = degrees[j]-k*f.c;
                for (int i=0; i<=d; i+=f.a) if ((d-i)%f.b==0)
                    piece.rows.push_back({j,i,k});
            }
        }
        return piece;
    }

    void extend_powers(int power) {
        while (static_cast<int>(powers.size()) <= power)
            powers.push_back(product(powers.back(),f.h));
    }

    template<unsigned long prime>
    int modular_rank(const Layout& piece, ModularImage<prime>& image) {
        for (const auto& generator : piece.rows) image.extend(generator.power);
        ModularMatrix<prime> mod(piece.rows.size(),piece.cols);
        fill_modular(mod,image,piece.rows,piece.degrees,piece.first,piece.base);
        return mod.rank_without(piece.unit_columns);
    }

    // The dimension of the weight-w piece of the normal quotient, when it is
    // proved by a modular rank: rank modulo a prime never exceeds the
    // rational rank, so reaching the number of rows or of remaining columns
    // proves the rational rank. Otherwise 'exact' is false and 'value' is only
    // an upper bound. Null when the matrix exceeds the memory bound.
    struct Piece { int value; bool exact; };
    std::optional<Piece> modular_piece(int w) {
        const auto piece = layout(w);
        if (!piece.remaining_cols) return Piece{0,true};
        if (piece.rows.empty()) return Piece{piece.remaining_cols,true};
        if (piece.rows.size()*piece.cols > 600000) return std::nullopt;
        // An invertible minor modulo a prime is invertible over QQ.
        const int full = std::min(static_cast<int>(piece.rows.size()),piece.remaining_cols);
        const int lower = modular_rank(piece,first_image);
        return Piece{piece.remaining_cols-lower,lower == full};
    }

    // Rank an unproved piece modulo the second prime.
    void retry(int w, Piece& known) {
        const auto piece = layout(w);
        const int full = std::min(static_cast<int>(piece.rows.size()),piece.remaining_cols);
        const int lower = modular_rank(piece,second_image);
        known.value = std::min(known.value,piece.remaining_cols-lower);
        known.exact = lower == full;
    }

    // The exact dimension of a piece that modular_piece left unproved.
    int exact_piece(int w) {
        const auto piece = layout(w);
        for (const auto& generator : piece.rows) extend_powers(generator.power);
        Matrix mat(piece.rows.size(),piece.cols);
        fill_exact(mat,piece.rows,piece.degrees,piece.first,piece.base);
        return piece.remaining_cols-mat.rank_without(piece.unit_columns);
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
    const int first_weight = -std::max({f->a,f->b,f->c});
    // Beyond the largest weight of x^i y^j e_l (i<a, j<b),
    // N_w=x^a N_(w-a)+y^b N_(w-b), so max(a,b) zero pieces prove the full
    // tail. Zero pieces are proved by modular ranks alone. Only the exact
    // total needs rational ranks, and only once the tail has been proved:
    // a germ of infinite codimension never pays for them here. The last
    // weight follows the size of the projection module, and modular_piece
    // bounds each matrix. Neither bound yields a finite or infinite dimension.
    const int work_limit = f->a*f->b*f->c + std::max(f->a,f->b);
    const int tail = std::max(f->a,f->b);
    std::vector<TangentRanks::Piece> pieces;
    // The first weight whose window of proved zero pieces proves the tail.
    auto proved_tail = [&]() -> std::optional<int> {
        int zeros = 0;
        for (size_t i = 0; i < pieces.size(); ++i) {
            const int w = first_weight+static_cast<int>(i);
            zeros = pieces[i].exact && !pieces[i].value ? zeros+1 : 0;
            if (w >= largest_generator && zeros >= tail) return w;
        }
        return std::nullopt;
    };
    std::optional<int> proof;
    int zeros = 0;
    for (int w = first_weight; w <= work_limit; ++w) {
        const auto piece = tangent.modular_piece(w);
        if (!piece) break;
        pieces.push_back(*piece);
        zeros = piece->exact && !piece->value ? zeros+1 : 0;
        if (w >= largest_generator && zeros >= tail) {
            proof = w;
            break;
        }
    }
    if (!proof) {
        // A prime can lower the ranks of every piece. Before declining,
        // rank the unproved pieces modulo the second prime.
        for (size_t i = 0; i < pieces.size(); ++i)
            if (!pieces[i].exact) tangent.retry(first_weight+static_cast<int>(i),pieces[i]);
        proof = proved_tail();
        if (!proof) return std::nullopt;
    }
    // Replace each unproved piece by its rational value, in order, and stop
    // at the first proved tail. An unlucky prime can only have postponed the
    // modular proof, never advanced it.
    int dimension = 0;
    zeros = 0;
    for (int w = first_weight; w <= *proof; ++w) {
        const auto& known = pieces[w-first_weight];
        const int part = known.exact ? known.value : tangent.exact_piece(w);
        dimension += part;
        zeros = part ? 0 : zeros+1;
        if (w >= largest_generator && zeros >= tail) {
            Json::Value out;
            out["finite"] = true;
            out["value"] = std::to_string(dimension);
            out["method"] = "graded_tangent_ranks";
            out["last_weight"] = w;
            return out;
        }
    }
    throw std::logic_error("A modular proof of the zero tail has no rational counterpart.");
}
}
