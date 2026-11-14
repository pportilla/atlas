// Regular subdivisions of the positive orthant and disjoint affine chart regions.
// See KERNEL.md, Common torus charts and incidence.
#include "groebner.hpp"
#include <flint/fmpq.h>
#include <flint/fmpz_mat.h>
#include <map>
#include <set>
#include <sstream>

namespace atlas {
namespace {
using Face = std::vector<size_t>;
struct Matrix {
    fmpz_mat_t m;
    explicit Matrix(long n) { fmpz_mat_init(m, n, n); }
    ~Matrix() { fmpz_mat_clear(m); }
};
struct FanFraction {
    fmpq_t q;
    FanFraction() { fmpq_init(q); }
    ~FanFraction() { fmpq_clear(q); }
};
}

std::string fan_cover(std::istream& in) {
    long n, ray_count;
    if (!(in >> n >> ray_count) || n < 1 || n > 8 || ray_count < 0)
        throw std::invalid_argument("Invalid fan dimensions.");
    std::vector<std::vector<Z>> rays(ray_count, std::vector<Z>(n));
    std::vector<Z> sums(ray_count);
    for (long i = 0; i < ray_count; ++i) {
        for (auto& x : rays[i]) {
            std::string value;
            if (!(in >> value) || fmpz_set_str(&x.v, value.c_str(), 10) || fmpz_sgn(&x.v) < 0)
                return "invalid\n";
            fmpz_add(&sums[i].v, &sums[i].v, &x.v);
        }
        if (fmpz_is_zero(&sums[i].v)) return "invalid\n";
    }
    FanFraction volume, weight;
    Matrix matrix(n);
    Z determinant;
    std::map<Face, long> boundary;
    std::set<Face> seen;
    const unsigned full = (1U << n) - 1;
    std::string result = "complete\n";
    while (in >> std::ws && in.peek() != std::char_traits<char>::eof()) {
        Face cone(n);
        for (auto& id : cone) if (!(in >> id) || id >= rays.size()) return "invalid\n";
        Face sorted = cone;
        std::sort(sorted.begin(), sorted.end());
        if (seen.count(sorted)) return "invalid\n";
        fmpq_one(weight.q);
        for (long i = 0; i < n; ++i) {
            const auto id = sorted[i];
            fmpz_mul(fmpq_denref(weight.q), fmpq_denref(weight.q), &sums[id].v);
            for (long j = 0; j < n; ++j) fmpz_set(fmpz_mat_entry(matrix.m,i,j), &rays[id][j].v);
        }
        fmpz_mat_det(&determinant.v, matrix.m);
        if (!fmpz_is_pm1(&determinant.v)) return "invalid\n";
        fmpq_add(volume.q, volume.q, weight.q);
        for (long i = 0; i < n; ++i) {
            Face facet = sorted;
            facet.erase(facet.begin()+i);
            boundary[facet] += (i % 2 ? -1 : 1) * fmpz_sgn(&determinant.v);
        }
        // A face is shared with an earlier cone iff its sorted ray set has
        // already occurred. Its complement is the open overlap's unit set.
        std::vector<Face> faces(full+1);
        std::vector<bool> shared(full+1);
        for (unsigned mask = 0; mask <= full; ++mask) {
            auto& face = faces[mask];
            for (long j = 0; j < n; ++j) if (mask & (1U << j)) face.push_back(cone[j]);
            std::sort(face.begin(), face.end());
            shared[mask] = seen.count(face);
        }
        for (unsigned mask = 0; mask < full; ++mask) if (shared[mask]) {
            bool maximal = true;
            for (long j = 0; j < n; ++j) if (!(mask & (1U << j)) && shared[mask | (1U << j)]) {
                maximal = false;
                break;
            }
            if (maximal) result += std::to_string(full ^ mask) + " ";
        }
        result += '\n';
        for (auto& face : faces) seen.insert(std::move(face));
    }
    if (!fmpq_is_one(volume.q)) return "invalid\n";
    for (const auto& item : boundary) if (item.second) {
        bool on_boundary = false;
        for (long j = 0; j < n; ++j) if (std::all_of(item.first.begin(), item.first.end(), [&](size_t i) {
                return fmpz_is_zero(&rays[i][j].v);
            })) { on_boundary = true; break; }
        if (!on_boundary) return "invalid\n";
    }
    return result;
}
}
extern "C" int atlas_fan_cover_abi() noexcept { return 1; }
