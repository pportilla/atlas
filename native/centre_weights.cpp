// Exact weights of a Bierstone–Milman coefficient presentation on one stratum.
#include "groebner.hpp"
#include <flint/fmpq.h>
#include <set>
#include <sstream>

namespace atlas {
namespace {
struct Weight {
    fmpq_t q;
    Weight() { fmpq_init(q); }
    Weight(const Weight& other) : Weight() { fmpq_set(q, other.q); }
    Weight(Weight&& other) noexcept : Weight() { fmpq_swap(q, other.q); }
    ~Weight() { fmpq_clear(q); }
    void read(std::istream& in) {
        std::string s;
        if (!(in >> s) || fmpq_set_str(q, s.c_str(), 10) || fmpz_is_zero(fmpq_denref(q)))
            throw std::invalid_argument("Invalid coefficient weight.");
        fmpq_canonicalise(q);
    }
    std::string text() const {
        char* value = fmpq_get_str(nullptr, 10, q);
        std::string s(value);
        flint_free(value);
        return s;
    }
    bool operator<(const Weight& other) const { return fmpq_cmp(q, other.q) < 0; }
};
}

std::string centre_weights(std::istream& in, const std::string& operation) {
    size_t width, count;
    if (!(in >> width >> count) || count == 0) throw std::invalid_argument("Expected a nonempty presentation.");
    std::vector<Weight> weights(count);
    std::vector<std::vector<Weight>> exponents(count, std::vector<Weight>(width));
    for (size_t i = 0; i < count; ++i) {
        weights[i].read(in);
        if (fmpq_sgn(weights[i].q) <= 0) throw std::invalid_argument("Coefficient weights must be positive.");
        for (auto& x : exponents[i]) x.read(in);
    }
    std::string out;
    if (operation == "bm_residual") {
        std::vector<Weight> alpha(width);
        for (size_t j = 0; j < width; ++j) {
            for (size_t i = 0; i < count; ++i) {
                Weight ratio;
                fmpq_div(ratio.q, exponents[i][j].q, weights[i].q);
                if (i == 0 || ratio < alpha[j]) fmpq_set(alpha[j].q, ratio.q);
            }
        }
        for (const auto& x : alpha) out += x.text() + " ";
        out += '\n';
        for (size_t i = 0; i < count; ++i) {
            for (size_t j = 0; j < width; ++j) {
                Weight e;
                fmpq_mul(e.q, alpha[j].q, weights[i].q);
                fmpq_sub(e.q, exponents[i][j].q, e.q);
                out += e.text() + " ";
            }
            out += '\n';
        }
        return out;
    }
    Weight threshold;
    if (operation == "bm_requirements") threshold.read(in);
    std::set<Weight> values{Weight()};
    size_t size;
    while (in >> size) {
        std::vector<size_t> subset(size);
        for (auto& j : subset) if (!(in >> j) || j >= width)
            throw std::invalid_argument("Unknown exceptional divisor in this stratum.");
        for (size_t i = 0; i < count; ++i) {
            Weight sum, value;
            for (auto j : subset) fmpq_add(sum.q, sum.q, exponents[i][j].q);
            if (operation == "bm_requirements") {
                fmpq_mul(value.q, threshold.q, weights[i].q);
                fmpq_sub(value.q, value.q, sum.q);
                Z ceiling;
                fmpz_cdiv_q(&ceiling.v, fmpq_numref(value.q), fmpq_denref(value.q));
                char* s = fmpz_get_str(nullptr, 10, &ceiling.v);
                out += std::string(s) + " ";
                flint_free(s);
            } else {
                long order;
                if (!(in >> order) || order < 0) throw std::invalid_argument("Invalid residual order.");
                for (long k = 0; k <= order; ++k) {
                    fmpq_add_si(value.q, sum.q, k);
                    fmpq_div(value.q, value.q, weights[i].q);
                    values.insert(value);
                }
            }
        }
        if (operation == "bm_requirements") out += '\n';
    }
    if (!in.eof()) throw std::invalid_argument("Invalid coefficient presentation.");
    if (operation == "bm_values") for (const auto& x : values) out += x.text() + "\n";
    return out;
}
}
extern "C" int atlas_centre_weights_abi() noexcept { return 1; }

// Weight updates used when passing to maximal contact and to the next level.
#include "rational_sparse_json.hpp"
namespace {
using namespace atlas::sparse_json;
// Prefixes of the BM local invariant compare lexicographically over
// the nonnegative rationals with an additional greatest value (infinity).
int compare_invariants(const Json::Value& a, const Json::Value& b) {
    for (Json::ArrayIndex j=0; j<std::min(a.size(),b.size()); ++j) {
        const auto x = a[j].asString(), y = b[j].asString();
        if (x == y) continue;
        if (x == "inf" || y == "inf") return x == "inf" ? 1 : -1;
        const int sign = cmp(Q(x,10),Q(y,10));
        if (sign) return sign;
    }
    return (a.size() > b.size()) - (a.size() < b.size());
}

Json::Value level_weights(const Json::Value& request) {
    const auto op = request["operation"].asString();
    const auto weights = read_row(request["weights"]);
    if (op == "frame_order") {
        const auto& choices = request["choices"];
        const auto& divisors = request["divisors"];
        std::vector<std::pair<size_t,Json::ArrayIndex>> scores;
        for (Json::ArrayIndex i=0; i<choices.size(); ++i) {
            size_t count = 0;
            for (const auto& j : choices[i]) count += divisors[j.asUInt()].asBool();
            scores.emplace_back(count,i);
        }
        std::sort(scores.begin(),scores.end());
        Json::Value result(Json::arrayValue);
        for (const auto& score : scores) result.append(score.second);
        return result;
    }
    if (op == "old_divisors") {
        Json::Value result(Json::arrayValue);
        const auto& prefix = request["prefix"];
        const auto& records = request["records"];
        for (Json::ArrayIndex i=0; i<records.size(); ++i)
            if (records[i].isNull() || compare_invariants(records[i],prefix)>0) result.append(i);
        return result;
    }
    if (op == "maximum") {
        const auto& values = request["values"];
        Json::Value result(Json::arrayValue);
        for (Json::ArrayIndex i=0; i<values.size(); ++i) {
            int order = 1;
            if (!result.empty()) {
                const auto& best = values[result[0].asUInt()];
                order = compare_invariants(values[i][0],best[0]);
                if (!order) order = compare_invariants(values[i][1],best[1]);
            }
            if (order > 0) result.clear();
            if (order >= 0) result.append(i);
        }
        return result;
    }
    if (op == "contact_shear") {
        const auto attempt = request["attempt"].asUInt64(), free = request["free"].asUInt64();
        const auto bases = request["bases"].asUInt64();
        if (!attempt || !bases) throw std::invalid_argument("Missing contact shear direction.");
        Json::Value result, coefficients(Json::arrayValue);
        result["base"] = (attempt-1)%bases;
        for (Json::UInt64 i=0; i<free; ++i) {
            const int magnitude = static_cast<int>((attempt%3+i%3)%3)+1;
            coefficients.append(i%2 ? -magnitude : magnitude);
        }
        result["coefficients"] = std::move(coefficients);
        return result;
    }
    if (op == "settle") {
        const auto exponents = read_row(request["exponents"]);
        Z power = 1;
        for (const auto& j : request["through"]) power = lcm(power, exponents.at(j.asUInt()).get_den());
        Json::Value result;
        result["power"] = power.get_str();
        Row scaled;
        for (const auto& e : exponents) scaled.emplace_back(e * power);
        result["exponents"] = strings(scaled);
        Q weight = weights.at(0) * power;
        result["weight"] = weight.get_str();
        return result;
    }
    if (op == "contact") {
        Json::Value result(Json::arrayValue);
        for (const auto& w : weights) {
            Json::Value marks(Json::arrayValue);
            for (Q mark = w; mark > 0; mark -= 1) marks.append(mark.get_str());
            result.append(marks);
        }
        return result;
    }
    if (op == "add_orders") {
        const auto orders = read_row(request["orders"]);
        if (weights.size() != orders.size()) throw std::invalid_argument("Invalid coordinate valuation vector.");
        Row out;
        for (size_t j = 0; j < weights.size(); ++j) out.emplace_back(weights[j] + orders[j]);
        return strings(out);
    }
    if (op == "next_level") {
        const Q value = read_row(request["threshold"]).at(0);
        Json::Value out;
        Row scaled;
        for (const auto& w : weights) scaled.emplace_back(w * value);
        out["scaled"] = strings(scaled);
        if (value < 1) { Q complement = 1 - value; out["extra"] = complement.get_str(); }
        return out;
    }
    if (op == "monomial") {
        // Only nonempty strata enter. Pick inclusion-minimal subsets whose
        // exceptional weights sum to one or more (BM section 3.2).
        std::vector<std::set<unsigned>> chosen;
        Json::Value out(Json::arrayValue);
        for (Json::ArrayIndex k = 0; k < request["strata"].size(); ++k) {
            std::set<unsigned> ids;
            Q sum;
            for (const auto& i : request["strata"][k]) {
                unsigned j = i.asUInt();
                if (ids.insert(j).second) sum += weights.at(j);
            }
            if (sum < 1) continue;
            bool minimal = true;
            for (const auto& prior : chosen)
                if (std::includes(ids.begin(), ids.end(), prior.begin(), prior.end())) { minimal = false; break; }
            if (minimal) { chosen.push_back(std::move(ids)); out.append(k); }
        }
        return out;
    }
    throw std::invalid_argument("Unknown centre weight operation.");
}
} // namespace
extern "C" const char* atlas_centre_level(const char* input) noexcept {
    return atlas::sparse_json::invoke(input, level_weights);
}
