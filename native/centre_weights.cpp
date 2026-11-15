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
    Weight maximum;
    size_t size;
    while (in >> size) {
        std::vector<size_t> subset(size);
        for (auto& j : subset) if (!(in >> j) || j >= width)
            throw std::invalid_argument("Unknown exceptional divisor in this stratum.");
        Weight minimum;
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
                if (operation == "bm_point_maximum") {
                    // All orders in this row are attained at the same point.
                    fmpq_add_si(value.q, sum.q, order);
                    fmpq_div(value.q, value.q, weights[i].q);
                    if (i == 0 || value < minimum) fmpq_set(minimum.q, value.q);
                    continue;
                }
                for (long k = 0; k <= order; ++k) {
                    fmpq_add_si(value.q, sum.q, k);
                    fmpq_div(value.q, value.q, weights[i].q);
                    values.insert(value);
                }
            }
        }
        if (operation == "bm_point_maximum" && maximum < minimum) fmpq_set(maximum.q, minimum.q);
        if (operation == "bm_requirements") out += '\n';
    }
    if (!in.eof()) throw std::invalid_argument("Invalid coefficient presentation.");
    if (operation == "bm_point_maximum") return maximum.text() + "\n";
    if (operation == "bm_values") for (const auto& x : values) out += x.text() + "\n";
    return out;
}
}
extern "C" int atlas_centre_weights_abi() noexcept { return 1; }

// Weight updates used when passing to maximal contact and to the next level.
#include "rational_sparse_json.hpp"
#include "ideal_arithmetic.hpp"
#include "local_io.hpp"
#include "primary_decomposition.hpp"
#include "local_monomial.hpp"
#include "ordered_polynomial.hpp"
#include <numeric>
namespace {
using namespace atlas::sparse_json;
// Prefixes of the BM local invariant compare lexicographically over
// the nonnegative rationals with an additional greatest value (infinity).
int compare_invariants(const Json::Value& a, const Json::Value& b, unsigned birth_width = 0) {
    const auto asize = std::max(a.size(), birth_width), bsize = std::max(b.size(), birth_width);
    const auto length = std::min(std::min(asize,bsize),std::max(a.size(),b.size()));
    for (Json::ArrayIndex j=0; j<length; ++j) {
        const auto x = j < a.size() ? a[j].asString() : "0";
        const auto y = j < b.size() ? b[j].asString() : "0";
        if (x == y) continue;
        if (x == "inf" || y == "inf") return x == "inf" ? 1 : -1;
        const int sign = cmp(Q(x,10),Q(y,10));
        if (sign) return sign;
    }
    return (asize > bsize) - (asize < bsize);
}

using Names = std::vector<std::string>;

Names names(const Json::Value& row) {
    Names out;
    for (const auto& name : row) out.push_back(name.asString());
    return out;
}

// A finite centre of nonconstant order splits according to the labels of
// the divisors through each geometric piece. Keep the largest signature;
// if all are equal, take only the first rational component on each chart.
Json::Value point_signatures(const Json::Value& request) {
    using Signature = std::pair<size_t,Names>;
    std::vector<std::vector<Signature>> charts;
    Signature best;
    for (const auto& group : request["charts"]) {
        auto& parts = charts.emplace_back();
        for (const auto& row : group) {
            auto labels = names(row);
            std::sort(labels.begin(),labels.end());
            auto& signature = parts.emplace_back(labels.size(),std::move(labels));
            best = std::max(best,signature);
        }
    }
    Json::Value kept(Json::arrayValue);
    bool all = true, multiple = false;
    for (const auto& parts : charts) {
        Json::Value indices(Json::arrayValue);
        for (Json::ArrayIndex i=0; i<parts.size(); ++i)
            if (parts[i]==best) indices.append(i);
        all &= indices.size()==parts.size();
        multiple |= indices.size()>1;
        kept.append(std::move(indices));
    }
    if (all || request["single"].asBool()) {
        if (!multiple) return {};
        for (auto& row : kept) if (row.size()>1) row.resize(1);
    }
    return kept;
}

atlas::local::Basis equations(const Json::Value& rows, size_t width) {
    atlas::local::Basis basis;
    for (const auto& row : rows) basis.push_back(atlas::local::decode_row(row.asString(),width));
    return basis;
}

using atlas::local::Basis;
using atlas::local::Polynomial;
using atlas::local::Powers;
using atlas::primary::member_lex;
using atlas::primary::reduced;

bool subset(const Basis& ideal, const Basis& lex_basis) {
    return std::all_of(ideal.begin(),ideal.end(),[&](const auto& f){ return member_lex(f,lex_basis); });
}

Json::Value ordered_parts(const Json::Value& request) {
    const auto coordinates = names(request["coordinate_keys"]);
    const bool sorted = std::is_sorted(coordinates.begin(),coordinates.end());
    const auto& parts = request["parts"];
    const auto& display_rank = request["display_rank"];
    auto degrees = read_row(request["degrees"]);
    std::vector<std::vector<atlas::local::OrderedRow>> keys(parts.size());
    if (display_rank.isNull()) {
        for (Json::ArrayIndex i=0; i<parts.size(); ++i)
            for (auto& f : equations(parts[i],coordinates.size()))
                keys[i].emplace_back(std::move(f),coordinates,sorted);
    }
    std::vector<Json::ArrayIndex> order(parts.size());
    std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](auto i, auto j) {
        if (!degrees.empty() && degrees.at(i)!=degrees.at(j)) return degrees[i]<degrees[j];
        if (!display_rank.isNull()) return display_rank[i].asUInt()<display_rank[j].asUInt();
        return keys[i]<keys[j];
    });
    Json::Value out(Json::arrayValue);
    for (auto i : order) out.append(i);
    return out;
}

// Keep the previously chosen presentation of a centre modulo the ambient
// ideal. Remove redundant basis elements of largest total degree first.
Json::Value few_generators(const Json::Value& request) {
    const auto n = request["width"].asUInt();
    const auto ambient = equations(request["ambient"],n), candidates = equations(request["basis"],n);
    const auto ambient_basis = reduced(ambient,n,true);
    Json::Value sorting(request), groups(Json::arrayValue), degrees(Json::arrayValue);
    std::vector<bool> keep;
    for (Json::ArrayIndex i=0; i<candidates.size(); ++i) {
        auto group = Json::Value(Json::arrayValue);
        group.append(request["basis"][i]);
        groups.append(std::move(group));
        const mpz_class rank=-atlas::local::max_total_degree(candidates[i]);
        degrees.append(rank.get_str());
        keep.push_back(!member_lex(candidates[i],ambient_basis));
    }
    sorting["parts"] = std::move(groups);
    sorting["degrees"] = std::move(degrees);
    for (const auto& index : ordered_parts(sorting)) {
        const size_t i=index.asUInt();
        if (!keep[i]) continue;
        auto others=ambient;
        for (size_t j=0; j<candidates.size(); ++j) if (j!=i && keep[j]) others.push_back(candidates[j]);
        if (others.size()>ambient.size() && member_lex(candidates[i],reduced(others,n,true))) keep[i]=false;
    }
    Json::Value out(Json::arrayValue);
    for (Json::ArrayIndex i=0; i<keep.size(); ++i) if (keep[i]) out.append(i);
    return out;
}

Json::Value linear_centre(const Json::Value& request) {
    const size_t n=request["width"].asUInt();
    const auto centre=equations(request["centre"],n), ambient=equations(request["ambient"],n);
    std::vector<bool> parameters(n,false);
    for (const auto& f : ambient) for (const auto& [powers,c] : f)
        for (size_t j=0; j<n; ++j) if (powers[j]!=0) parameters[j]=true;
    std::vector<size_t> columns;
    for (size_t j=0; j<n; ++j) if (!parameters[j]) columns.push_back(j);
    const auto free=columns.size();
    for (size_t j=0; j<n; ++j) if (parameters[j]) columns.push_back(j);
    Basis inputs;
    for (const auto* ideal : {&ambient,&centre}) for (const auto& f : *ideal) {
        Polynomial g;
        for (const auto& [powers,c] : f) {
            Powers p;
            for (auto j : columns) p.push_back(powers[j]);
            g.emplace(std::move(p),c);
        }
        inputs.push_back(std::move(g));
    }
    const auto lex=reduced(inputs,n,true);
    Basis generated=ambient;
    Json::Value solved(Json::arrayValue);
    for (size_t j=0; j<free; ++j) {
        Powers variable(n);
        variable[j]=1;
        for (const auto& f : lex) {
            const auto linear=f.find(variable);
            if (linear==f.end()) continue;
            bool graph=true;
            Polynomial shift, equation;
            for (const auto& [powers,c] : f) {
                if (powers!=variable && std::any_of(powers.begin(),powers.begin()+free,[](const auto& e){return e!=0;})) {
                    graph=false;
                    break;
                }
                Powers original(n);
                for (size_t k=0; k<n; ++k) original[columns[k]]=powers[k];
                equation.emplace(original,c);
                if (powers!=variable) shift.emplace(std::move(original),-c/linear->second);
            }
            if (!graph) continue;
            Json::Value pair(Json::arrayValue);
            pair.append(static_cast<Json::UInt>(columns[j]));
            pair.append(atlas::local::encode_row(shift));
            solved.append(std::move(pair));
            generated.push_back(std::move(equation));
            break;
        }
    }
    if (solved.size()!=request["codimension"].asUInt() || !subset(centre,reduced(generated,n,true))) return {};
    return solved;
}

Json::Value strict_surface(const Json::Value& request) {
    // A rank failure involving strict components alone rules out this
    // phase. Otherwise exactly one strict component must meet a bad locus.
    for (const auto& failure : request["strict_labels"])
        if (std::all_of(failure.begin(),failure.end(),[](const auto& strict){return strict.asBool();})) return {};
    const auto n=request["width"].asUInt();
    const auto ambient=equations(request["ambient"],n);
    const auto& strict=request["strict"];
    const Basis unit{{{Powers(n),1}}};
    std::vector<Basis> failures;
    for (const auto& rows : request["failures"]) failures.push_back(equations(rows,n));
    Json::Value selected;
    for (Json::ArrayIndex i=0; i<strict.size(); ++i) {
        auto chart=equations(strict[i]["ideal"],n);
        chart.insert(chart.end(),ambient.begin(),ambient.end());
        for (const auto& failure : failures) {
            auto locus=chart;
            locus.insert(locus.end(),failure.begin(),failure.end());
            if (reduced(locus,n,false)==unit) continue;
            if (!selected.isNull()) return {};
            selected=strict[i]["index"];
            break;
        }
    }
    return selected;
}

Json::Value level_weights(const Json::Value& request) {
    const auto op = request["operation"].asString();
    const auto weights = read_row(request["weights"]);
    if (op == "ordered_parts") return ordered_parts(request);
    if (op == "few_generators") return few_generators(request);
    if (op == "linear_centre") return linear_centre(request);
    if (op == "principal") {
        const size_t n=request["width"].asUInt();
        auto ambient=equations(request["ambient"],n);
        const auto ideal=equations(request["ideal"],n), ambient_basis=reduced(ambient,n,true);
        const auto& ranks=request["ranks"];
        std::vector<Json::ArrayIndex> order;
        for (Json::ArrayIndex i=0; i<ideal.size(); ++i) if (!member_lex(ideal[i],ambient_basis)) order.push_back(i);
        std::stable_sort(order.begin(),order.end(),[&](auto i,auto j){return compare_invariants(ranks[i],ranks[j])<0;});
        if (order.size()==1) return order.front();
        ambient.emplace_back();
        for (auto i : order) {
            ambient.back()=ideal[i];
            if (subset(ideal,reduced(ambient,n,true))) return i;
        }
        return {};
    }
    if (op == "candidate_order") {
        const auto& ranks = request["ranks"];
        std::vector<Json::ArrayIndex> indices;
        for (Json::ArrayIndex i=0; i<ranks.size(); ++i) {
            if (request["integral_weight"].asBool() && (weights.at(i)<1 || weights.at(i).get_den()!=1)) continue;
            indices.push_back(i);
        }
        std::stable_sort(indices.begin(),indices.end(),[&](auto a, auto b) {
            return compare_invariants(ranks[a],ranks[b]) < 0;
        });
        Json::Value result(Json::arrayValue);
        for (auto i : indices) result.append(i);
        return result;
    }
    if (op == "label_order") {
        auto labels = names(request["labels"]);
        std::sort(labels.begin(),labels.end());
        labels.erase(std::unique(labels.begin(),labels.end()),labels.end());
        Json::Value result(Json::arrayValue);
        if (request["pairs"].asBool()) {
            // Compare (first, second) descending, with first < second.
            for (auto a=labels.rbegin(); a!=labels.rend(); ++a)
                for (auto b=labels.rbegin(); b!=a; ++b) {
                    Json::Value pair(Json::arrayValue);
                    pair.append(*a); pair.append(*b);
                    result.append(std::move(pair));
                }
        } else {
            if (request["descending"].asBool()) std::reverse(labels.begin(),labels.end());
            for (const auto& label : labels) result.append(label);
        }
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
    if (op == "maximum" || op == "surface_maximum") {
        const auto& values = request["values"];
        Json::Value result(Json::arrayValue);
        for (Json::ArrayIndex i=0; i<values.size(); ++i) {
            if (values[i].isNull()) continue;
            int order = 1;
            if (!result.empty()) {
                const auto& best = values[result[0].asUInt()];
                order = compare_invariants(values[i][0],best[0]);
                if (!order && op == "maximum")
                    order = compare_invariants(values[i][1],best[1],request["birth_width"].asUInt());
                if (!order && op == "surface_maximum") {
                    const auto labels = names(values[i][1]), previous = names(best[1]);
                    order = (labels > previous) - (labels < previous);
                }
            }
            if (order > 0) result.clear();
            if (order >= 0) result.append(i);
        }
        return result;
    }
    if (op == "birth_keys") {
        Json::Value result(Json::arrayValue);
        for (const auto& group : request["groups"]) {
            const auto list = names(group);
            std::set<std::string> present(list.begin(),list.end());
            Json::Value key(Json::arrayValue);
            for (const auto& label : request["births"]) key.append(static_cast<int>(present.count(label.asString())));
            result.append(std::move(key));
        }
        return result;
    }
    if (op == "direct_stratum") {
        // Prefer more failing components, then their first positions in the
        // ordered charts. Labels can appear on more than one chart.
        std::map<std::string, unsigned> positions;
        for (const auto& label : request["labels"]) positions.emplace(label.asString(), positions.size());
        std::vector<unsigned> best;
        Json::Value selected;
        for (const auto& subset : request["subsets"]) {
            std::vector<unsigned> rank;
            for (const auto& label : subset) rank.push_back(positions.at(label.asString()));
            if (selected.isNull() || rank.size()>best.size() || (rank.size()==best.size() && rank<best)) {
                selected = subset;
                best = std::move(rank);
            }
        }
        return selected;
    }
    if (op == "point_signatures") return point_signatures(request);
    if (op == "canonical_phase") {
        const auto& charts = request["charts"];
        Json::Value surfaces(Json::arrayValue), selected(Json::arrayValue);
        bool singular = false;
        for (const auto& chart : charts) {
            auto surface = chart.isNull() ? Json::Value() : strict_surface(chart);
            if (!chart.isNull() && surface.isNull()) singular = true;
            surfaces.append(std::move(surface));
        }
        auto phase = request["phase"].asString();
        if (phase == "order" && request["dimension"].asInt()==3 && !singular) phase = "surface";
        for (Json::ArrayIndex i=0; i<charts.size(); ++i)
            if (!charts[i].isNull() && (phase!="order" || !singular || surfaces[i].isNull())) selected.append(i);
        Json::Value result;
        result["phase"] = phase;
        result["charts"] = std::move(selected);
        result["surfaces"] = std::move(surfaces);
        return result;
    }
    if (op == "direct_locus" || op == "split_curve") {
        // Prefer the isolated or singular points when present; a curve
        // splits only if both its part in the divisor and its complement exist.
        const auto& parts = request["parts"];
        const bool present = std::any_of(parts.begin(),parts.end(),[](const auto& p){ return p.asBool(); });
        return present && (op == "direct_locus" || request["outside"].asBool());
    }
    if (op == "present_outside") {
        // The complement of V(H) is covered by D(h) for all h in H.
        // On each such open, saturation gives its closure. A further open
        // meets this closure precisely when it meets the original open.
        const auto n=request["width"].asUInt();
        const Basis unit{{{Powers(n),1}}};
        auto base=reduced(equations(request["ideal"],n),n,true);
        if (base==unit) return false;
        std::vector<Basis> holes;
        for (const auto& rows : request["holes"]) {
            auto hole=equations(rows,n);
            if (std::find(holes.begin(),holes.end(),hole)==holes.end()) holes.push_back(std::move(hole));
        }
        auto visit=[&](auto&& self, const Basis& closure, size_t k) -> bool {
            if (k==holes.size()) return true;
            for (const auto& g : holes[k]) {
                if (member_lex(g,closure)) continue;
                const auto next=atlas::primary::saturate_lex(closure,g,n);
                if (next!=unit && self(self,next,k+1)) return true;
            }
            return false;
        };
        return visit(visit,base,0);
    }
    if (op == "common_value") {
        const auto& values=request["values"];
        if (!values.size()) return {};
        for (const auto& value : values) if (value!=values[0]) return {};
        return values[0];
    }
    if (op == "cover_products" || op == "component_ideal_product") {
        const bool cover = op == "cover_products";
        const size_t width=request["width"].asUInt();
        Basis products{{{Powers(width),1}}};
        for (const auto& rows:request[cover ? "holes" : "factors"]) {
            const auto generators=equations(rows,width);
            if (!cover) {
                products = atlas::ideal_arithmetic::generator_products(products, generators, width);
                continue;
            }
            std::set<Polynomial> seen;
            Basis next;
            for(const auto& p:products) for(const auto& g:generators) {
                auto product=atlas::primary::multiply(p,g);
                if(seen.insert(product).second) next.push_back(std::move(product));
            }
            products=std::move(next);
        }
        Json::Value result(Json::arrayValue);
        for(const auto& p:products) result.append(atlas::local::encode_row(p));
        return result;
    }
    if (op == "contact_candidate") {
        const auto& w=weights.at(0);
        if (w.get_den()!=1 || w<=0) throw std::invalid_argument("Invalid contact weight.");
        Json::Value result;
        result["degree"] = w.get_num().get_str();
        result["probe_direction"] = w >= 3;
        result["support"]=Json::Value(Json::arrayValue);
        const auto exponents=read_row(request["exponents"]);
        for(unsigned i=0;i<exponents.size();++i) if(exponents[i]>0) result["support"].append(i);
        return result;
    }
    if (op == "retained_centres") {
        // Removing the new centre already removes each older closed locus
        // contained in it. Preserve the other holes in their original order.
        const size_t width = request["width"].asUInt();
        const auto centre = equations(request["centre"],width), ambient = equations(request["ambient"],width);
        const auto& holes = request["holes"];
        const atlas::local::Basis unit{{{atlas::local::Powers(width),1}}};
        Json::Value kept(Json::arrayValue);
        for (Json::ArrayIndex i=0; i<holes.size(); ++i) {
            auto base = equations(holes[i],width);
            base.insert(base.end(),ambient.begin(),ambient.end());
            const auto basis = atlas::primary::reduced(base,width,true);
            const bool contained = std::all_of(centre.begin(),centre.end(),[&](const auto& g) {
                return atlas::primary::member_lex(g,basis) || atlas::primary::saturate_lex(basis,g,width)==unit;
            });
            if (!contained) kept.append(i);
        }
        return kept;
    }
    if (op == "cartier_order") {
        const size_t width = request["width"].asUInt();
        auto ambient = equations(request["ambient"],width);
        const auto divisors = equations(request["ideal"],width);
        const auto pivot = atlas::local::decode_row(request["pivot"].asString(),width);
        auto power = pivot;
        ambient.emplace_back();
        for (unsigned order=0; ; ++order) {
            ambient.back() = power;
            const auto basis = atlas::primary::reduced(ambient,width,true);
            for (const auto& f : divisors)
                if (!atlas::primary::member_lex(f,basis)) return order;
            if (order==request["limit"].asUInt()) return {};
            power = atlas::primary::multiply(power,pivot);
        }
    }
    if (op == "locus_intersection") {
        const size_t width = request["width"].asUInt();
        const auto a = equations(request["first"],width), b = equations(request["second"],width);
        auto result = atlas::primary::intersection_lex(a,b,width);
        // intersection_lex may return an unchanged operand for a unit ideal.
        if (a==b || result==a || result==b) result = atlas::primary::reduced(result,width,true);
        Json::Value rows(Json::arrayValue);
        for (const auto& f : result) rows.append(atlas::local::encode_row(f));
        return rows;
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
