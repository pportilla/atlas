// Normalized convex hull volume in the ambient integer lattice.
#include "groebner.hpp"
#include <flint/fmpz_mat.h>
#include <map>
#include <set>
#include <sstream>

namespace {
using atlas::Z;
using Row = std::vector<Z>;
using Face = std::vector<size_t>;
struct Matrix {
    fmpz_mat_t m;
    Matrix(slong rows, slong columns) { fmpz_mat_init(m, rows, columns); }
    ~Matrix() { fmpz_mat_clear(m); }
    Matrix(const Matrix&) = delete;
};

Face without(const Face& face, size_t i) {
    Face result = face;
    result.erase(result.begin() + i);
    return result;
}

struct Hyperplane {
    Row normal;
    Z offset;
    Z height(const Row& point) const {
        Z value;
        fmpz_neg(&value.v, &offset.v);
        for (size_t j = 0; j < point.size(); ++j)
            fmpz_addmul(&value.v, &normal[j].v, &point[j].v);
        return value;
    }
};

Hyperplane plane(const Face& face, const std::vector<Row>& points, const Row& sum) {
    const size_t n = face.size();
    const Row& base = points[face[0]];
    Hyperplane h{Row(n), {}};
    Matrix minor(n - 1, n - 1);
    for (size_t k = 0; k < n; ++k) {
        for (size_t i = 1; i < n; ++i)
            for (size_t j = 0, column = 0; j < n; ++j) if (j != k) {
                fmpz_sub(fmpz_mat_entry(minor.m, i-1, column++), &points[face[i]][j].v, &base[j].v);
            }
        fmpz_mat_det(&h.normal[k].v, minor.m);
        if (k % 2) fmpz_neg(&h.normal[k].v, &h.normal[k].v);
        fmpz_addmul(&h.offset.v, &base[k].v, &h.normal[k].v);
    }
    Z sign;
    fmpz_mul_si(&sign.v, &h.offset.v, -static_cast<slong>(n + 1));
    for (size_t j = 0; j < n; ++j) fmpz_addmul(&sign.v, &sum[j].v, &h.normal[j].v);
    if (fmpz_sgn(&sign.v) < 0) {
        fmpz_neg(&h.offset.v, &h.offset.v);
        for (auto& v : h.normal) fmpz_neg(&v.v, &v.v);
    }
    return h;
}

Z volume(const std::vector<Row>& points, size_t n) {
    Z total;
    if (points.size() <= n) return total;
    // The pivot columns of the transpose select d independent edge vectors.
    Matrix edges(n, points.size()-1), echelon(n, points.size()-1);
    for (size_t i = 1; i < points.size(); ++i)
        for (size_t j = 0; j < n; ++j)
            fmpz_sub(fmpz_mat_entry(edges.m, j, i-1), &points[i][j].v, &points[0][j].v);
    Z denominator;
    if (fmpz_mat_rref(echelon.m, &denominator.v, edges.m) != static_cast<slong>(n)) return total;
    Face simplex{0};
    Matrix initial(n, n);
    for (size_t j = 0, column = 0; j < n; ++j, ++column) {
        while (fmpz_is_zero(fmpz_mat_entry(echelon.m, j, column))) ++column;
        simplex.push_back(1 + column);
        for (size_t k = 0; k < n; ++k)
            fmpz_set(fmpz_mat_entry(initial.m, j, k), fmpz_mat_entry(edges.m, k, column));
    }
    fmpz_mat_det(&total.v, initial.m);
    fmpz_abs(&total.v, &total.v);
    if (points.size() == simplex.size()) return total;
    Row centroid_sum(n);
    for (auto i : simplex)
        for (size_t j = 0; j < n; ++j)
            fmpz_add(&centroid_sum[j].v, &centroid_sum[j].v, &points[i][j].v);
    std::map<Face, Hyperplane> facets;
    for (size_t i = 0; i < simplex.size(); ++i) {
        Face f = without(simplex, i);
        facets.emplace(f, plane(f, points, centroid_sum));
    }
    for (size_t i = 0; i < points.size(); ++i) {
        if (std::binary_search(simplex.begin(), simplex.end(), i)) continue;
        std::set<Face> horizon;
        for (auto it = facets.begin(); it != facets.end();) {
            Z h = it->second.height(points[i]);
            if (fmpz_sgn(&h.v) >= 0) { ++it; continue; }
            // Cones over the visible simplices have disjoint interiors.
            // The unscaled cofactor normal gives their normalized volumes.
            fmpz_sub(&total.v, &total.v, &h.v);
            for (size_t k = 0; k < n; ++k) {
                Face ridge = without(it->first, k);
                if (!horizon.insert(ridge).second) horizon.erase(ridge);
            }
            it = facets.erase(it);
        }
        for (Face f : horizon) {
            f.insert(std::lower_bound(f.begin(), f.end(), i), i);
            facets.emplace(f, plane(f, points, centroid_sum));
        }
    }
    return total;
}
}
extern "C" char* atlas_lattice_volume(const char* input, size_t size) noexcept {
    try {
        std::istringstream in(std::string(input, size));
        long n;
        if (!(in >> n) || n < 0) throw std::invalid_argument("Expected a nonnegative lattice dimension.");
        if (n == 0) {
            size_t count;
            if (!(in >> count)) throw std::invalid_argument("Expected the number of zero-dimensional points.");
            return atlas::copy_text(count ? "1" : "0");
        }
        std::vector<Row> points;
        std::string value;
        while (in >> value) {
            Row p(n);
            for (long j = 0; j < n; ++j) {
                if ((j && !(in >> value)) || fmpz_set_str(&p[j].v, value.c_str(), 10))
                    throw std::invalid_argument("Invalid lattice point.");
            }
            points.push_back(std::move(p));
        }
        const Z answer = volume(points, n);
        char* decimal = fmpz_get_str(nullptr, 10, &answer.v);
        char* result = atlas::copy_text(decimal);
        flint_free(decimal);
        return result;
    } catch (const std::exception& e) {
        atlas::set_error(e.what());
    } catch (...) {
        atlas::set_error("Unknown lattice-volume failure.");
    }
    return nullptr;
}
