// Choose the canonical smooth-surface centre on a chart where the strict
// transform is a rational graph and the reduced boundary consists of lines.
#include "surface_centre_choice.hpp"
#include "chart_kernel_rows.hpp"
#include "local_monomial.hpp"
#include "primary_decomposition.hpp"
#include "rational_sparse_json.hpp"

#include <algorithm>
#include <array>
#include <memory>

extern "C" char* atlas_ideal(const char*, size_t) noexcept;
extern "C" void atlas_groebner_release(char*) noexcept;

namespace atlas {
namespace {
using namespace chart_rows;
using Ideals = std::vector<Basis>;
constexpr size_t width = 3;

std::optional<std::pair<Polynomial, Polynomial>> linear_parts(const Polynomial& f, size_t axis) {
    Polynomial a, b;
    for (const auto& [p,c] : f) {
        if (!p[axis]) b.emplace(p,c);
        else if (p[axis] == 1) {
            auto e = p;
            e[axis] = 0;
            a.emplace(std::move(e), c);
        } else return {};
    }
    return std::make_pair(std::move(a),std::move(b));
}

Polynomial difference(Polynomial f, const Polynomial& g) {
    for (const auto& [p,c] : g)
        if ((f[p] -= c) == 0) f.erase(p);
    return f;
}

// An affine-linear factor defines one line. An irreducible univariate factor
// defines distinct parallel lines over Qbar, by separability. Two independent
// directions give only ordinary nodes, with at most two branches per point.
using Direction = std::array<mpq_class,width>;
std::optional<Direction> direction(const Polynomial& f) {
    auto axes = support({f},width);
    Direction d{};
    if (axes.size() == 1) d[axes.front()] = 1;
    else if (local::max_total_degree(f) == 1 && !axes.empty()) {
        for (auto j : axes)
            d[j] = f.at(var(j,width).begin()->first);
        const mpq_class scale = d[axes.front()];
        for (auto& c : d) c /= scale;
    } else return {};
    return d;
}

struct Owned {
    const Polynomial& surface;
    Ideals work, holes;

    // Use the same geometric open-locus predicate as the ordinary chart
    // verifier. A packed/branching limit cannot supply a positive certificate.
    std::optional<bool> meets(const Basis& extra) const {
        bool undecided = false;
        for (const auto& slice : work) {
            std::ostringstream in;
            in << "open_locus " << width << ' ' << 1+extra.size()+slice.size() << ' ' << holes.size();
            for (const auto& h : holes) in << ' ' << h.size();
            in << '\n' << local::encode_row(surface) << '\n';
            auto append = [&](const Basis& rows) {
                for (const auto& f : rows) in << local::encode_row(f) << '\n';
            };
            append(extra);
            append(slice);
            for (const auto& h : holes) append(h);
            const auto input = in.str();
            std::unique_ptr<char,decltype(&atlas_groebner_release)> answer(
                atlas_ideal(input.data(),input.size()),atlas_groebner_release);
            if (answer && std::string_view(answer.get()) == "yes\n") return true;
            if (!answer || std::string_view(answer.get()) != "no\n") undecided = true;
        }
        return undecided ? std::optional<bool>{} : std::optional<bool>{false};
    }
};

using Orders = std::map<std::string,mpz_class>;
using Traces = std::map<Polynomial,Orders>;

struct Choice {
    mpz_class multiplicity;
    std::vector<std::string> labels;
    const Polynomial* curve = nullptr;
    bool unique = false;
};

// Compare the canonical key on every factor of the full graph open. A single
// winning Q-line meeting the owned working set attains that same maximum on
// the working set. Its closure is unchanged by any further nonempty open.
Choice best_curve(const Traces& traces) {
    Choice best;
    for (const auto& [p,orders] : traces) {
        mpz_class top = 0;
        std::vector<std::string> labels;
        for (const auto& [label,m] : orders) {
            if (m > top || (m == top && m > 1)) {
                top = m;
                labels = {label};
            } else if (m == top) labels.push_back(label);
        }
        if (top == 1 && labels.size() != 2) continue;
        if (std::tie(top,labels) > std::tie(best.multiplicity,best.labels))
            best = {top,labels,&p,true};
        else if (top == best.multiplicity && labels == best.labels) best.unique = false;
    }
    return best;
}

V try_projection(size_t axis, const Polynomial& a, const Polynomial& b,
                 const std::map<std::string,Polynomial>& divisors, const Owned& owned) {
    const bool constant = local::max_total_degree(a) == 0;
    if (!constant && owned.meets({a}) != false) return {};
    std::set<Direction> directions;
    Traces traces;
    for (const auto& [label,h] : divisors) {
        const auto parts = linear_parts(h,axis);
        if (!parts) return {};
        const auto& [c,d] = *parts;
        // On a!=0, u=-b/a. Discard all irreducible factors of the
        // denominator from the numerator, since they are units there.
        auto value = c.empty() ? d : difference(primary::multiply(a,d),primary::multiply(b,c));
        if (value.empty()) return {};  // This divisor contains the surface.
        for (const auto& [p,m] : primary::factors(value,width)) {
            if (!constant && local::polynomial_divides(p,a,width)) continue;
            const auto dir = direction(p);
            if (!dir) return {};
            directions.insert(*dir);
            if (directions.size() > 2) return {};
            auto& orders = traces[p];
            orders[label] = m;
            // A triple shared component is outside the pair rule used by
            // the general implicit surface kernel.
            if (orders.size() > 2) return {};
        }
    }
    const auto best = best_curve(traces);
    if (!best.unique || local::max_total_degree(*best.curve) != 1 ||
        owned.meets({*best.curve}) != true) return {};
    // The reduced trace is nodal, the preferred multiplicity/label locus is
    // one smooth line, and the eligible open meets it. Saturation removes
    // all vertical components produced by clearing the graph denominator.
    Basis centre{owned.surface,*best.curve};
    if (!constant) centre = primary::saturate_lex(centre,a,width);
    V out;
    out["top"] = best.multiplicity.get_str();
    out["labels"] = array();
    for (const auto& label : best.labels) out["labels"].append(label);
    centre = primary::reduced(centre,width,false);
    // The general rule checks the winning curve for singular points even
    // beyond the current working locus. A globally affine line passes that
    // check in every chart model, including over a=0 outside our projection.
    if (centre.size()!=2 || !std::all_of(centre.begin(),centre.end(),
        [](const auto& g){return local::max_total_degree(g)==1;})) return {};
    if (!constant) {
        // Another eligible coordinate model may extend beyond a!=0. Rule
        // out even an extra vertical component of the winning trace there;
        // otherwise its closed chart centre need not be this same line.
        Basis vertical{a};
        for (const auto& label : best.labels) vertical.push_back(divisors.at(label));
        if (Owned{owned.surface,{vertical},{centre}}.meets({}) != false) return {};
    }
    out["ideal"] = rows(centre);
    return out;
}
} // namespace

Json::Value surface_centre_choice(const Json::Value& q) {
    auto f = local::decode_row(q["surface"].asString(),width);
    Owned owned{f,{},{}};
    for (const auto& rs : q["work"]) owned.work.push_back(parse(rs,width));
    for (const auto& rs : q["holes"]) owned.holes.push_back(parse(rs,width));
    std::map<std::string,Polynomial> divisors;
    for (const auto& d : q["divisors"])
        if (!divisors.emplace(d[0].asString(), local::decode_row(d[1].asString(),width)).second) return {};
    try {
        primary::CertificateBudget budget(16384);
        for (size_t k=0; k<width; ++k) {
            const auto parts = linear_parts(f,k);
            if (!parts || parts->first.empty()) continue;
            auto answer = try_projection(k,parts->first,parts->second,divisors,owned);
            if (!answer.isNull()) return answer;
        }
    } catch (const std::runtime_error& e) {
        const std::string_view error(e.what());
        if (error.find("limit") == std::string_view::npos && error.find("out of range") == std::string_view::npos)
            throw;
    }
    return {};
}
} // namespace atlas

extern "C" const char* atlas_surface_centre_choice(const char* input) {
    return atlas::sparse_json::invoke(input,atlas::surface_centre_choice);
}
