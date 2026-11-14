// Smooth refinements of a 3-dimensional positive orthant with one prescribed
// interior ray. Input: one primitive positive integral weight triple on stdin.
// Output: zero-based ray indices, with cones oriented to have determinant +1.
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
using Integer = std::int64_t;
using Ray = std::array<Integer, 3>;
using Triangle = std::array<int, 3>;
using Face = std::pair<int, int>;
constexpr Integer max_weight = 1024;
// With coordinates <= 10^6, even the sum of six absolute triple products
// fits in a signed 64-bit integer. We reject growth before storing a new ray.
constexpr Integer max_coordinate = 1000000;
constexpr int max_rays = 16000;
constexpr int max_cone_records = 80000;

Integer determinant(const Ray& a, const Ray& b, const Ray& c) {
    return a[0] * (b[1]*c[2] - b[2]*c[1])
         + a[1] * (b[2]*c[0] - b[0]*c[2])
         + a[2] * (b[0]*c[1] - b[1]*c[0]);
}
Integer gcd3(const Ray& r) {
    return std::gcd(r[0], std::gcd(r[1], r[2]));
}
Integer residue(Integer value, Integer modulus) {
    const Integer r = value % modulus;
    return r >= 0 ? r : r + modulus;
}
Face face(int a, int b) { return {std::min(a,b), std::max(a,b)}; }

struct Cone {
    Triangle rays;
    int index;  // lattice index (absolute determinant)
    bool active = true;
};
struct Candidate {
    Ray point;
    Ray numerators;  // coefficients in the input cone, scaled by its determinant
};

class Fan {
    std::vector<Ray> rays_;
    std::vector<Cone> cones_;
    std::map<Face, std::set<int>> incidence_;
    // Largest determinant first; lower insertion index breaks ties.
    std::priority_queue<std::pair<int,int>> singular_;

    int insert_cone(Triangle t) {
        if (cones_.size() >= max_cone_records)
            throw std::runtime_error("The toric subdivision exceeded its cone budget.");
        Integer d = determinant(rays_[t[0]], rays_[t[1]], rays_[t[2]]);
        if (d < 0) {
            std::swap(t[1], t[2]);
            d = -d;
        }
        if (d < 1 || d > max_weight)
            throw std::logic_error("Unexpected lattice index in the toric subdivision.");
        const int id = static_cast<int>(cones_.size());
        cones_.push_back({t, static_cast<int>(d)});
        for (int i = 0; i < 3; ++i)
            incidence_[face(t[i], t[(i+1)%3])].insert(id);
        if (d > 1) singular_.push({static_cast<int>(d), -id});
        return id;
    }

    void retire(int id) {
        Cone& cone = cones_[id];
        cone.active = false;
        for (int i = 0; i < 3; ++i) {
            auto it = incidence_.find(face(cone.rays[i], cone.rays[(i+1)%3]));
            it->second.erase(id);
            if (it->second.empty()) incidence_.erase(it);
        }
    }

    Candidate lattice_point(const Cone& cone) const {
        const auto& t = cone.rays;
        const Integer d = cone.index;
        Candidate best{};
        // Prefer the smallest sum of replacement lattice indices, then the
        // smallest maximum. Enumerating these three cyclic subgroups suffices
        // to find a point: their generators span Z^3 / <r0,r1,r2>.
        using Score = std::tuple<Integer, Integer, Ray>;
        Score best_score{std::numeric_limits<Integer>::max(), 0, {}};
        for (int axis = 0; axis < 3; ++axis) {
            Ray e{};
            e[axis] = 1;
            Ray generator = {
                residue(determinant(e, rays_[t[1]], rays_[t[2]]), d),
                residue(determinant(rays_[t[0]], e, rays_[t[2]]), d),
                residue(determinant(rays_[t[0]], rays_[t[1]], e), d)
            };
            const Integer period = d / std::gcd(d, gcd3(generator));
            for (Integer k = 1; k < period; ++k) {
                Ray numerators{}, p{};
                for (int j = 0; j < 3; ++j) {
                    numerators[j] = k * generator[j] % d;
                    for (int i = 0; i < 3; ++i)
                        p[i] += numerators[j] * rays_[t[j]][i];
                }
                for (auto& coordinate : p) {
                    if (coordinate % d) throw std::logic_error("Nonintegral parallelepiped point.");
                    coordinate /= d;
                }
                const Integer common = gcd3(p);
                for (int j = 0; j < 3; ++j) {
                    p[j] /= common;
                    numerators[j] /= common; // adj(B) p is integral after primitivization
                }
                Score score{numerators[0] + numerators[1] + numerators[2],
                            *std::max_element(numerators.begin(), numerators.end()), p};
                if (score < best_score) {
                    best_score = score;
                    best = {p, numerators};
                }
            }
        }
        if (gcd3(best.point) != 1)
            throw std::logic_error("Nontrivial lattice quotient yielded no primitive ray.");
        for (Integer coordinate : best.point) {
            if (coordinate <= 0)
                throw std::logic_error("Unexpected ray on the smooth original boundary.");
            if (coordinate > max_coordinate)
                throw std::runtime_error("The toric subdivision exceeded its coordinate budget.");
        }
        return best;
    }

    void subdivide(int id) {
        const Cone target = cones_[id];
        const Candidate p = lattice_point(target);
        if (rays_.size() >= max_rays)
            throw std::runtime_error("The toric subdivision exceeded its ray budget.");
        const int new_id = static_cast<int>(rays_.size());
        rays_.push_back(p.point);
        std::vector<int> support, affected;
        for (int j = 0; j < 3; ++j)
            if (p.numerators[j]) support.push_back(target.rays[j]);
        if (support.size() == 3) {
            affected.push_back(id);
        } else if (support.size() == 2) {
            const auto& adjacent = incidence_.at(face(support[0],support[1]));
            if (adjacent.size() != 2)
                throw std::logic_error("An interior face must have two incident cones.");
            affected.assign(adjacent.begin(), adjacent.end());
        } else {
            throw std::logic_error("Primitive generators have no smaller point on their ray.");
        }

        // A point in a shared face has identical coefficients in either cone.
        // Replacing each generator of its support produces the full star
        // subdivision. In every affected cone each child index equals the old
        // index times a coefficient strictly between zero and one. Consequently
        // the multiset of nonunit indices strictly decreases at every step.
        for (const int old_id : affected) {
            const Cone old = cones_[old_id];
            retire(old_id);
            for (int j = 0; j < 3; ++j) {
                if (std::find(support.begin(),support.end(),old.rays[j]) == support.end()) continue;
                Triangle child = old.rays;
                child[j] = new_id;
                const int child_id = insert_cone(child);
                if (cones_[child_id].index >= old.index)
                    throw std::logic_error("Star subdivision failed to lower a lattice index.");
            }
        }
    }

public:
    explicit Fan(const Ray& w) : rays_{{1,0,0},{0,1,0},{0,0,1},w} {
        insert_cone({3,1,2});
        insert_cone({0,3,2});
        insert_cone({0,1,3});
    }
    void resolve() {
        while (!singular_.empty()) {
            const int id = -singular_.top().second;
            singular_.pop();
            if (cones_[id].active) subdivide(id);
        }
    }
    void write_json(std::ostream& out) const {
        out << "{\"rays\":[";
        for (std::size_t i = 0; i < rays_.size(); ++i) {
            const auto& r = rays_[i];
            out << (i ? ",[" : "[") << r[0] << ',' << r[1] << ',' << r[2] << ']';
        }
        out << "],\"cones\":[";
        bool first = true;
        for (const Cone& c : cones_) {
            if (!c.active) continue;
            const auto& t = c.rays;
            out << (first ? "[" : ",[") << t[0] << ',' << t[1] << ',' << t[2] << ']';
            first = false;
        }
        out << "]}\n";
    }
};
}

int main() {
    try {
        Ray w{};
        std::string extra;
        if (!(std::cin >> w[0] >> w[1] >> w[2]) || (std::cin >> extra))
            throw std::invalid_argument("Expected exactly three integer weights.");
        for (Integer n : w) {
            if (n < 1 || n > max_weight)
                throw std::invalid_argument("Weights must lie between 1 and 1024.");
        }
        if (gcd3(w) != 1) throw std::invalid_argument("Weights must form a primitive lattice ray.");
        Fan fan(w);
        fan.resolve();
        fan.write_json(std::cout);
        return 0;
    } catch (const std::exception& e) {
        std::cout << "{\"error\":\"" << e.what() << "\"}\n";
        return 1;
    }
}
