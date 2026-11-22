// Resolve a reduced central arrangement in A^3 by blowing up the vertex and
// the disjoint strict transforms of its multiple intersection lines. The
// blowup at the vertex is the line bundle O(-1) on P^2, so subsequent charts
// are ordinary blowups of direction points, with the radial coordinate fixed.
#include "chart_expr.hpp"
#include "branch_order.hpp"
#include "local_monomial.hpp"
#include "primary_decomposition.hpp"
#include <array>

namespace {
using namespace atlas::chart;
using Triple = std::array<Q, 3>;
const Variables coordinates{{"r", 0}, {"u", 1}, {"v", 2}};
const Expr r = symbol(coordinates[0]), u = symbol(coordinates[1]), v = symbol(coordinates[2]);
const Expr one = constant(1);
V array() { return V(Json::arrayValue); }
V list(std::initializer_list<V> items) {
    V out = array();
    for (const auto &item : items)
        out.append(item);
    return out;
}
std::string show(const Expr &f) { return text(coordinates, f); }
std::string sheet(int i) { return "S" + std::to_string(i + 1); }

struct Flat {
    Triple direction;
    std::set<int> planes;
    std::string divisor;
};

// Reduced projective coordinates identify one direction even at infinity.
Triple intersection(const Triple &a, const Triple &b) {
    Triple p{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    auto first = std::find_if(p.begin(), p.end(), [](const Q &x) { return x != 0; });
    if (first == p.end())
        throw std::logic_error("Duplicate monic plane factors.");
    Q pivot = *first;
    for (Q &x : p)
        x /= pivot;
    return p;
}

V divisor(const std::string &name, int order, int nu) {
    V row;
    row["component"] = name;
    row["multiplicity"] = order;
    row["nu"] = nu;
    return row;
}

struct Arrangement {
    V names;
    std::vector<Triple> planes;
    std::vector<Flat> flats;
    bool intrinsic;
    Q unit;
    V charts = array(), atlas = array(), steps = array();
    V vertices = array(), cells = array(), data = array(), strata = array();

    void vertex(const std::string &name, const std::string &ambient) {
        V row;
        row["id"] = name;
        row["dimension"] = intrinsic ? 1 : 2;
        if (intrinsic)
            row["ambient_divisor"] = ambient;
        vertices.append(row);
    }
    void edge(const std::string &a, const std::string &b) {
        V row;
        row["id"] = "C" + std::to_string(cells.size() + 1);
        row["dimension"] = 1;
        row["vertices"] = row["faces"] = list({a, b});
        cells.append(row);
    }
    void stratum(const V &labels, int chi) {
        V row;
        row["components"] = labels;
        row["euler_characteristic"] = chi;
        strata.append(row);
    }

    V nearby(const atlas::local::Polynomial &polynomial) const {
        V result = array();
        for (const auto &flat : flats) {
            const int k = flat.planes.size();
            if (k < 3) continue;
            const auto &p = flat.direction;
            // At every nonzero point of this line the nonincident linear
            // factors are units. Fix the first nonzero coordinate at 1.
            // Restricting the k incident forms to this complementary plane
            // preserves their distinct projective directions.
            const auto pivot = std::find_if(p.begin(), p.end(), [](const Q &a) { return a != 0; }) - p.begin();
            const auto shifted = atlas::local::translate_unbounded({polynomial}, {p[0], p[1], p[2]});
            atlas::local::Polynomial slice;
            for (const auto &[exponents, coefficient] : shifted.front()) {
                if (exponents[pivot] != 0) continue;
                auto powers = exponents;
                powers.erase(powers.begin() + pivot);
                slice.emplace(std::move(powers), coefficient);
            }
            V row;
            row["model"] = "ordinary multiple point";
            row["point"] = atlas::sparse_json::strings(p);
            row["parameter_axis"] = static_cast<int>(pivot);
            row["multiplicity"] = k;
            for (int i = 0; i < 3; ++i) {
                if (i == pivot) continue;
                row["slice_axes"].append(i);
                Powers x(3), t(3);
                x[i] = t[pivot] = 1;
                atlas::local::Polynomial normal{{x, 1}};
                if (p[i] != 0) normal.emplace(t, -p[i]);
                row["normal_coordinates"].append(atlas::sparse_json::term_list(normal));
            }
            row["germ"] = atlas::sparse_json::term_list(slice);
            // One blowup of the transverse plane gives a P^1 punctured at
            // k points, with multiplicity k in A'Campo's formula.
            row["factors"] = list({list({std::to_string(k), std::to_string(k - 2)})});
            result.append(std::move(row));
        }
        return result;
    }
    void geometry() {
        const int count = planes.size();
        V origin;
        origin["id"] = "E1";
        origin["type"] = "blowup";
        origin["center"] = "origin";
        origin["ideal"] = names;
        steps.append(origin);
        data.append(divisor("E1", count, 3));
        if (!intrinsic)
            vertex("E1", "");
        for (int i = 0; i < count; ++i) {
            if (intrinsic)
                vertex("E1." + sheet(i), "E1");
            V row = divisor(sheet(i), 1, 1);
            atlas::local::Polynomial f;
            for (int j = 0; j < 3; ++j) {
                Powers e(3);
                e[j] = 1;
                if (planes[i][j] != 0)
                    f[e] = planes[i][j];
            }
            row["equation"] = atlas::local::format_polynomial(
                f, {names[0].asString(), names[1].asString(), names[2].asString()});
            data.append(row);
        }

        // On E1 each projective line loses its distinct intersection points.
        // The complement has Euler characteristic 3-2d+sum_p(k_p-1).
        int complement = 3 - 2 * count;
        std::vector<int> punctures(count);
        for (auto &flat : flats) {
            const int k = flat.planes.size();
            complement += k - 1;
            for (int i : flat.planes)
                ++punctures[i];
            if (!intrinsic && k == 2) {
                stratum(list({"E1", sheet(*flat.planes.begin()), sheet(*flat.planes.rbegin())}), 1);
                continue;
            }
            flat.divisor = "E" + std::to_string(steps.size() + 1);
            V step;
            step["id"] = flat.divisor;
            step["type"] = "blowup";
            step["center"] = "strict transform of the line in the indicated projective direction";
            step["direction"] = atlas::sparse_json::strings(flat.direction);
            steps.append(step);
            data.append(divisor(flat.divisor, k, 2));
            // Blowing up this point of P^2 inserts a P^1 with k distinct
            // marked tangent directions. The entire origin fibre lies on E1.
            stratum(list({"E1", flat.divisor}), 2 - k);
            if (!intrinsic) {
                vertex(flat.divisor, "");
                edge("E1", flat.divisor);
            }
            for (int i : flat.planes) {
                stratum(list({"E1", flat.divisor, sheet(i)}), 1);
                if (intrinsic) {
                    const auto branch = flat.divisor + "." + sheet(i);
                    vertex(branch, flat.divisor);
                    edge("E1." + sheet(i), branch);
                }
            }
        }
        stratum(list({"E1"}), complement);
        for (int i = 0; i < count; ++i)
            stratum(list({"E1", sheet(i)}), 2 - punctures[i]);
    }

    void chart(int pivot, const Flat *centre, int patch) {
        std::array<int, 2> axes;
        int at = 0;
        for (int i = 0; i < 3; ++i)
            if (i != pivot)
                axes[at++] = i;
        std::array<Expr, 3> direction, tangent;
        direction[pivot] = one;
        if (centre) {
            tangent[axes[patch]] = one;
            tangent[axes[1 - patch]] = v;
            for (int j : axes)
                direction[j] = constant(centre->direction[j] / centre->direction[pivot]) + u * tangent[j];
        } else {
            direction[axes[0]] = u;
            direction[axes[1]] = v;
        }
        V owned, raw;
        owned["arrangement_chart"]["pivot"] = names[pivot];
        if (centre) {
            owned["arrangement_chart"]["line_divisor"] = centre->divisor;
            owned["arrangement_chart"]["tangent_pivot"] = names[axes[patch]];
        }
        owned["coordinates"] = raw["coordinates"] = list({"r", "u", "v"});
        raw["id"] = "A" + std::to_string(charts.size() + 1);
        for (int i = 0; i < 3; ++i)
            owned["map"][names[i].asString()] = show(r * direction[i]);
        raw["map"] = owned["map"];
        owned["ambient_ideal"] = array();
        owned["fibre"] = list({"r"});
        owned["removed"] = array();
        for (const auto &other : flats) {
            if (other.divisor.empty() || &other == centre || other.direction[pivot] == 0)
                continue;
            V ideal = array();
            for (int j : axes)
                ideal.append(show(direction[j] - constant(other.direction[j] / other.direction[pivot])));
            owned["removed"].append(ideal);
        }
        raw["blown_up"] = array();
        for (const auto &ideal : owned["removed"]) {
            V excluded;
            excluded["ideal"] = ideal;
            raw["blown_up"].append(excluded);
        }
        // A parent owns the first nonzero homogeneous coordinate. Each
        // replacement patch owns only its newly inserted exceptional line.
        owned["region"] = array();
        if (centre) {
            owned["region"].append("u");
            if (patch == 1)
                owned["region"].append("v");
        } else {
            for (int i = 0; i < pivot; ++i)
                owned["region"].append(show(direction[i]));
        }
        owned["components"] = array();
        raw["exceptional"] = array();
        auto component = [&](const std::string &label, int n, int nu, const Expr &equation, bool exceptional) {
            V c;
            c["id"] = label;
            c["multiplicity"] = n;
            c["nu"] = nu;
            c["ideal"] = list({show(equation)});
            owned["components"].append(c);
            if (exceptional) {
                V e;
                e["id"] = label;
                e["total_multiplicity"] = n;
                e["equation"] = c["ideal"][0];
                raw["exceptional"].append(e);
            }
        };
        component("E1", planes.size(), 3, r, true);
        if (centre)
            component(centre->divisor, centre->planes.size(), 2, u, true);
        if (intrinsic) {
            raw["ambient_components"] = raw["exceptional"];
            raw["exceptional"] = array();
        }
        Expr strict = constant(unit);
        for (int i = 0; i < int(planes.size()); ++i) {
            // At a centre on this plane the linear equation pulls back to u
            // times its tangent direction. Every other equation stays a unit
            // on the new fibre and is retained on its affine open.
            const auto &values = (centre && centre->planes.count(i)) ? tangent : direction;
            Expr f;
            for (int j = 0; j < 3; ++j)
                f = f + scaled(values[j], planes[i][j]);
            strict = strict * f;
            if (!f.number()) {
                component(sheet(i), 1, 1, f, false);
                if (intrinsic) {
                    for (const auto &ambient : raw["ambient_components"]) {
                        const auto parent = ambient["id"].asString();
                        if (parent != "E1" && !centre->planes.count(i))
                            continue;
                        V branch;
                        branch["id"] = parent + "." + sheet(i);
                        branch["ambient_divisor"] = parent;
                        branch["ideal"] = list({ambient["equation"], show(f)});
                        branch["equation"] = ambient["equation"].asString() + ", " + show(f);
                        raw["exceptional"].append(branch);
                    }
                }
            }
        }
        raw["strict_transform"] = show(strict);
        charts.append(raw);
        atlas.append(owned);
    }
    V build() {
        geometry();
        for (int pivot = 0; pivot < 3; ++pivot) {
            chart(pivot, nullptr, 0);
            for (const auto &flat : flats) {
                if (flat.divisor.empty() || flat.direction[pivot] == 0)
                    continue;
                // One parent chart already contains the whole replacement
                // over this centre. The radial coordinate covers its entire intersection line.
                if (std::any_of(flat.direction.begin(), flat.direction.begin() + pivot,
                                [](const Q &x) { return x != 0; }))
                    continue;
                chart(pivot, &flat, 0);
                chart(pivot, &flat, 1);
            }
        }
        V result;
        result["charts"] = charts;
        result["atlas"] = atlas;
        result["steps"] = steps;
        auto &complex = result["dual_complex"];
        complex["status"] = "complete";
        complex["convention"] =
            intrinsic
                ? "Exceptional divisor on the smooth strict transform in the germ over the origin over Qbar."
                : "Ambient exceptional divisor in the germ over the origin over Qbar.";
        complex["vertices"] = vertices;
        complex["cells"] = cells;
        result["numerical_data"] = data;
        result["strata"] = strata;
        return result;
    }
};

using Factors = std::vector<std::pair<atlas::local::Polynomial, Z>>;

Factors rational_hyperplanes(const atlas::local::Polynomial& f, size_t width, int minimum,
                            Q* primitive_unit = nullptr) {
    if (f.empty())
        return {};
    const Z d = atlas::local::total_degree(f.begin()->first);
    // Bound the number of charts and the size of the expanded chart equations.
    if (d < minimum || d > 12)
        return {};
    for (const auto &term : f)
        if (atlas::local::total_degree(term.first) != d)
            return {};
    auto factors = atlas::primary::factors(f, width, primitive_unit);
    if (Z(factors.size()) != d)
        return {};
    for (const auto &[factor, multiplicity] : factors) {
        if (multiplicity != 1)
            return {};
        for (const auto &[e, c] : factor) {
            if (atlas::local::total_degree(e) != 1)
                return {};
        }
    }
    return factors;
}

V analyze_arrangement(const atlas::local::Polynomial& f, const V& names, bool intrinsic, bool nearby) {
    const auto factors = rational_hyperplanes(f, 3, 4);
    if (factors.empty()) return {};
    Arrangement a{names, {}, {}, intrinsic, f.rbegin()->second};
    for (const auto &[factor, multiplicity] : factors) {
        Triple coefficients{};
        for (const auto &[e, c] : factor) {
            coefficients[std::find(e.begin(), e.end(), 1) - e.begin()] = c;
        }
        a.planes.push_back(coefficients);
    }
    std::sort(a.planes.begin(), a.planes.end());
    std::map<Triple, std::set<int>> incidence;
    for (int i = 0; i < int(a.planes.size()); ++i)
        for (int j = 0; j < i; ++j) {
            auto &flat = incidence[intersection(a.planes[i], a.planes[j])];
            flat.insert(i);
            flat.insert(j);
        }
    if (incidence.size() == 1)
        return {}; // Rank two: a plane-curve cylinder.
    for (const auto &[direction, planes] : incidence)
        a.flats.push_back({direction, planes, {}});
    return nearby ? a.nearby(f) : a.build();
}

V resolve_arrangement(const V &request) {
    if (request["operation"] == "hyperplanes") {
        std::vector<std::string> names;
        for (const auto& name : request["variables"]) names.push_back(name.asString());
        if (names.size() != 4) return {};
        auto f = atlas::sparse_json::read_terms(request["terms"], names.size());
        Q unit;
        const auto factors = rational_hyperplanes(f, names.size(), 3, &unit);
        if (factors.empty()) return {};
        V result;
        result["unit_at_origin"] = unit.get_str();
        for (auto& [e,c] : f) c /= unit;
        result["reduced_equation"] = atlas::local::format_polynomial(f,names);
        for (const auto i : atlas::archived_branch_order(factors,names.size())) {
            V row;
            row["equation"] = atlas::local::format_polynomial(factors[i].first,names);
            row["multiplicity"] = 1;
            result["factors"].append(row);
        }
        return result;
    }
    if (request["variables"].size() != 3) return {};
    return analyze_arrangement(atlas::sparse_json::read_terms(request["terms"], 3),
                               request["variables"], request["intrinsic"].asBool(), false);
}
} // namespace

namespace atlas {
Json::Value arrangement_nearby_strata(const local::Polynomial& f) {
    return analyze_arrangement(f, {}, false, true);
}
}

extern "C" const char *atlas_plane_arrangement(const char *input) noexcept {
    return atlas::sparse_json::invoke(input, resolve_arrangement);
}
