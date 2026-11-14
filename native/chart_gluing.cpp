// Incidence of prime strata of arbitrary affine charts. The chart transports
// service requests to the existing absolute decomposition, Euler and SNC
// kernels. All geometric decisions and the ordered face map are made here.
#include "component_partition.hpp"
#include "local_io.hpp"
#include "primary_decomposition.hpp"
#include "quotient_ring.hpp"
#include "rational_sparse_json.hpp"
#include <algorithm>
#include <functional>
#include <optional>
#include <set>

namespace {
using namespace atlas::local;
namespace primary = atlas::primary;
using V = Json::Value;
using Names = std::vector<std::string>;
using Section = std::pair<unsigned, std::string>;
using Piece = std::pair<Section, Basis>;
using Group = std::vector<Piece>;
using Groups = std::vector<Group>;
using Callback = const char *(*)(const char *);
using Ring = atlas::quotient_ring::Ring;

V array() {
    return V(Json::arrayValue);
}
template <class T> V list(const std::vector<T> &items) {
    V out = array();
    for (const auto &item : items)
        out.append(item);
    return out;
}
V rows(const Basis &ideal) {
    V out = array();
    for (const auto &f : ideal)
        out.append(encode_row(f));
    return out;
}
Basis parse(const V &encoded, size_t width) {
    Basis out;
    for (const auto &row : encoded)
        out.push_back(decode_row(row.asString(), width));
    return out;
}
Basis append(Basis a, const Basis &b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}
template <class T> std::vector<T> unique(const std::vector<T> &v) {
    std::vector<T> out;
    std::set<T> used;
    for (const auto &x : v)
        if (used.insert(x).second)
            out.push_back(x);
    return out;
}
Polynomial axis(size_t i, size_t width) {
    Powers p(width);
    p.at(i) = 1;
    return {{p, 1}};
}
Polynomial unit(size_t width) {
    return {{Powers(width), 1}};
}
Polynomial mapped(const Polynomial &f, const std::vector<size_t> &axes, size_t width) {
    Polynomial out;
    for (const auto &[p, c] : f) {
        Powers q(width);
        for (size_t j = 0; j < p.size(); ++j)
            q.at(axes.at(j)) += p[j];
        out[q] += c;
        if (out[q] == 0)
            out.erase(q);
    }
    return out;
}
Polynomial difference(Polynomial a, const Polynomial &b) {
    for (const auto &[p, c] : b) {
        a[p] -= c;
        if (a[p] == 0)
            a.erase(p);
    }
    return a;
}
bool zero(const Basis &b) {
    return std::all_of(b.begin(), b.end(), [](const auto &f) { return f.empty(); });
}
using Axes = std::vector<size_t>;
Axes interval(size_t start, size_t count) {
    Axes out(count);
    std::iota(out.begin(), out.end(), start);
    return out;
}
struct Chart {
    unsigned width;
    Basis ambient, owned, projections, repair_map;
    std::vector<Basis> holes;
    std::vector<std::pair<std::string, Basis>> strict, boundary;
    int theta, parent;
    // Each prime can be used as a parent face and in many removed-open tests.
    std::map<Basis, Ring> rings;
    std::map<Basis, Basis> grevlex;
    explicit Chart(const V &x)
        : width(x["width"].asUInt()), ambient(parse(x["ambient"], width)), owned(parse(x["owned"], width)),
          projections(parse(x["maps"], width)), repair_map(parse(x["repair"], width)),
          theta(x["theta"].asInt()), parent(x["parent"].asInt()) {
        for (const auto &hole : x["holes"])
            holes.push_back(parse(hole, width));
        for (const auto &f : x["components"])
            (f[2].asBool() ? strict : boundary).emplace_back(f[0].asString(), parse(f[1], width));
    }
    Ring &quotient(const Basis &ideal) {
        return rings.try_emplace(ideal, ideal, width, true).first->second;
    }
    bool contains(const Basis &ideal, const Basis &equations) {
        if (equations.empty())
            return true;
        const auto &basis = quotient(ideal).basis();
        return std::all_of(equations.begin(), equations.end(),
                           [&](const auto &f) { return primary::member_lex(f, basis); });
    }
    const Basis &canonical(const Basis &ideal) {
        auto it = grevlex.find(ideal);
        if (it != grevlex.end())
            return it->second;
        return grevlex.emplace(ideal, primary::reduced(ideal, width, false)).first->second;
    }
    bool retained(const Basis &prime) {
        return std::none_of(holes.begin(), holes.end(),
                            [&](const auto &hole) { return contains(prime, hole); });
    }
};

template <class Visit> void subsets(const Names &keys, size_t size, Visit visit) {
    Names current;
    auto next = [&](auto &&self, size_t pos) -> void {
        if (current.size() == size) {
            visit(current);
            return;
        }
        for (size_t i = pos; i + (size - current.size()) <= keys.size(); ++i) {
            current.push_back(keys[i]);
            self(self, i + 1);
            current.pop_back();
        }
    };
    next(next, 0);
}
template <class P> const Basis *lookup(const P &values, const std::string &name) {
    for (const auto &[label, ideal] : values)
        if (label == name)
            return &ideal;
    return nullptr;
}
std::string join(const Names &keys, const std::string &delimiter) {
    std::string out;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (i)
            out += delimiter;
        out += keys[i];
    }
    return out;
}
struct Incomplete {};
struct Driver {
    Callback callback;
    std::vector<Chart> charts;
    int dimension;
    bool intrinsic, absolute;
    V vertices = array(), cells = array();
    using Pair = std::pair<unsigned, unsigned>;
    std::map<Pair, V> transitions, diagonals;
    std::map<Pair, primary::FractionMap> fraction_maps;
    std::map<Piece, std::vector<std::optional<Polynomial>>> images;
    Driver(const V &q, Callback cb)
        : callback(cb), dimension(q["dimension"].asInt()), intrinsic(q["intrinsic"].asBool()),
          absolute(q["absolute"].asBool()) {
        charts.reserve(q["charts"].size());
        for (const auto &x : q["charts"])
            charts.emplace_back(x);
    }
    V call(const char *op, unsigned i, V fields = V(Json::objectValue)) const {
        fields["op"] = op;
        fields["i"] = i;
        Json::StreamWriterBuilder writer;
        writer["indentation"] = "";
        const auto text = Json::writeString(writer, fields);
        const char *response = callback(text.c_str());
        if (!response)
            throw std::runtime_error("Chart computation interrupted.");
        V answer;
        std::string error;
        Json::CharReaderBuilder parser;
        std::istringstream input(response);
        if (!Json::parseFromStream(parser, input, &answer, &error))
            throw std::runtime_error(error);
        return answer;
    }
    bool present(unsigned i, const Basis &ideal) {
        V q;
        q["ideal"] = rows(ideal);
        return call("present", i, q).asBool();
    }
    mpz_class euler(unsigned i, const Basis &ideal) {
        V q;
        q["ideal"] = rows(ideal);
        return mpz_class(call("euler", i, q).asString());
    }
    V prime_call(unsigned i, const Basis &ideal, int dim, const char *op, bool reduced) {
        V q;
        q["ideal"] = rows(ideal);
        q["dimension"] = dim;
        q["reduced"] = reduced;
        return call(op, i, q);
    }
    V branches(unsigned i, const Basis &ideal, int dim, std::map<Basis, V> &known) {
        auto found = known.find(ideal);
        if (found != known.end())
            return found->second;
        auto direct = prime_call(i, ideal, dim, "geometric", true);
        if (!direct.isNull())
            return known.emplace(ideal, std::move(direct)).first->second;
        auto &c = charts[i];
        for (const auto &f : ideal) {
            V q;
            q["equation"] = encode_row(f);
            const auto factors = call("factors", i, q);
            bool repeated = false;
            std::vector<Basis> refinements;
            for (const auto &p : factors)
                if (mpz_class(p[1].asString()) > 1)
                    repeated = true;
            if (factors.size() < 2 && !repeated)
                continue;
            for (const auto &p : factors)
                refinements.push_back(c.canonical(append(ideal, {decode_row(p[0].asString(), c.width)})));
            // Any identical refinement fails strict enlargement, preventing cycles.
            if (std::find(refinements.begin(), refinements.end(), ideal) != refinements.end())
                continue;
            bool complete = true;
            V result = array();
            std::set<Basis> seen;
            for (const auto &refined : unique(refinements)) {
                auto pieces = branches(i, refined, dim, known);
                if (pieces.isNull()) {
                    complete = false;
                    break;
                }
                for (const auto &prime : pieces)
                    if (seen.insert(parse(prime, c.width)).second)
                        result.append(prime);
            }
            if (complete)
                return known.emplace(ideal, std::move(result)).first->second;
        }
        known.emplace(ideal, V());
        return {};
    }
    V primes(unsigned i, const Basis &ideal, int dim, bool branch) {
        V found;
        if (branch && charts[i].theta < 0) {
            std::map<Basis, V> cache;
            found = branches(i, charts[i].canonical(ideal), dim, cache);
        } else
            found = prime_call(i, ideal, dim, "geometric", branch);
        if (found.isNull() && (!branch || absolute))
            found = prime_call(i, ideal, dim, "absolute", branch);
        if (branch && found.isNull())
            prime_call(i, ideal, dim, "split", true);
        return found;
    }
    const V &transition(unsigned i, unsigned j) {
        const Pair pair{i, j};
        auto it = transitions.find(pair);
        if (it != transitions.end())
            return it->second;
        V q;
        q["j"] = j;
        auto record = call("transition", i, q);
        if (record.isObject() && record.isMember("map")) {
            auto &fractions = fraction_maps[pair];
            for (const auto &entry : record["map"]) {
                auto f = parse(entry, charts[i].width);
                if (f.size() != 2)
                    throw std::invalid_argument("Invalid coordinate fraction.");
                fractions.emplace_back(std::move(f[0]), std::move(f[1]));
            }
            if (fractions.size() != charts[j].width)
                throw std::invalid_argument("Invalid number of coordinate fractions.");
        }
        V holes = record.isObject() ? record["holes"] : record;
        return transitions.emplace(pair, std::move(holes)).first->second;
    }
    bool vanishes(unsigned i, unsigned j, const Basis &source, const Basis &target) {
        if (target.empty())
            return true;
        const auto found = fraction_maps.find({i, j});
        if (found != fraction_maps.end()) {
            auto &ring = charts[i].quotient(source);
            return std::all_of(target.begin(), target.end(), [&](const auto &f) {
                return primary::vanishes_under_map(f, found->second, ring, charts[i].width);
            });
        }
        // Custom transition adapters can provide only a substitution callback.
        // Its remainders are transported back here for the same vanishing test.
        V q;
        q["j"] = j;
        q["source"] = rows(source);
        q["target"] = rows(target);
        return zero(parse(call("transport", i, q), charts[i].width));
    }
    const auto &image_constants(const Piece &piece) {
        auto found = images.find(piece);
        if (found != images.end())
            return found->second;
        auto &chart = charts.at(piece.first.first);
        Axes order = interval(0, chart.width);
        if (chart.theta >= 0) {
            order.erase(order.begin() + chart.theta);
            order.push_back(chart.theta);
        }
        Axes destination(chart.width);
        for (size_t i = 0; i < order.size(); ++i)
            destination[order[i]] = i;
        Basis equations;
        for (const auto &f : piece.second)
            equations.push_back(mapped(f, destination, chart.width));
        auto &ring = chart.quotient(equations);
        std::vector<std::optional<Polynomial>> values;
        for (const auto &f : chart.repair_map) {
            const auto image =
                std::get<Polynomial>(ring.evaluate("reduce", {mapped(f, destination, chart.width)}));
            Polynomial scalar;
            bool constant = true;
            for (const auto &[p, c] : image) {
                auto end = p.end() - int(chart.theta >= 0);
                if (std::any_of(p.begin(), end, [](const auto &e) { return e != 0; })) {
                    constant = false;
                    break;
                }
                scalar.emplace(Powers(end, p.end()), c);
            }
            values.push_back(constant ? std::optional<Polynomial>(std::move(scalar)) : std::nullopt);
        }
        return images.emplace(piece, std::move(values)).first->second;
    }
    int compare(const Piece &p, const Piece &r) {
        unsigned i = p.first.first, j = r.first.first;
        auto &a = charts[i];
        auto &b = charts[j];
        if (a.parent >= 0 && a.parent == b.parent) {
            const auto &left = image_constants(p);
            const auto &right = image_constants(r);
            for (size_t k = 0; k < std::min(left.size(), right.size()); ++k)
                if ((left[k] || right[k]) && left[k] != right[k])
                    return -1;
        }
        const V &forward = transition(j, i);
        if (!forward.isNull()) {
            for (const auto &hole : forward)
                if (b.contains(r.second, parse(hole, b.width)))
                    return 0;
            if (!vanishes(j, i, r.second, p.second))
                return 0;
            const V &backward = transition(i, j);
            if (!backward.isNull()) {
                for (const auto &row : backward) {
                    const auto hole = parse(row, a.width);
                    if (a.contains(p.second, hole) || vanishes(j, i, r.second, hole))
                        return 0;
                }
                return vanishes(i, j, p.second, r.second) ? 1 : 0;
            }
        }
        const Pair key{j, i};
        auto found = diagonals.find(key);
        if (found == diagonals.end()) {
            V q;
            q["j"] = i;
            auto diagonal = call("overlap", j, q);
            diagonal["left_width"] = b.width;
            diagonal["right_width"] = a.width;
            diagonal["maps"] = primary::prepare_overlap(diagonal);
            diagonal.removeMember("graph");
            found = diagonals.emplace(key, std::move(diagonal)).first;
        }
        auto &request = found->second;
        request["left_prime"] = rows(r.second);
        request["right_prime"] = rows(p.second);
        return primary::generic_overlap(request);
    }
    // Keep input/first-representative order even when union roots change.
    std::vector<std::vector<size_t>> group_indices(const std::vector<Piece> &pieces) {
        if (pieces.empty())
            return {};
        std::vector<size_t> classes(pieces.size());
        atlas::components::partition(
            pieces.size(), [&](size_t k) { return pieces[k].first.first; },
            [&](size_t k) -> const auto & { return pieces[k].first.second; },
            [&](size_t p, size_t r) { return compare(pieces[p], pieces[r]); }, classes.data());
        std::vector<std::vector<size_t>> groups;
        std::vector<std::set<Section>> sections;
        std::map<size_t, size_t> order;
        for (size_t k = 0; k < pieces.size(); ++k) {
            auto [it, fresh] = order.emplace(classes[k], groups.size());
            if (fresh) {
                groups.emplace_back();
                sections.emplace_back();
            }
            if (!sections[it->second].insert(pieces[k].first).second)
                throw Incomplete{};
            groups[it->second].push_back(k);
        }
        groups.erase(std::remove_if(groups.begin(), groups.end(),
                                    [&](const auto &group) {
                                        return std::none_of(group.begin(), group.end(), [&](size_t k) {
                                            const auto &[s, f] = pieces[k];
                                            return present(s.first, append(f, charts[s.first].owned));
                                        });
                                    }),
                     groups.end());
        return groups;
    }
    Groups groups(const Group &pieces) {
        Groups visible;
        for (const auto &indices : group_indices(pieces)) {
            Group group;
            for (auto k : indices)
                group.push_back(pieces[k]);
            visible.push_back(std::move(group));
        }
        return visible;
    }
    void vertex(const std::string &name, const Names &ambient = {}) {
        V out;
        out["id"] = name;
        out["dimension"] = dimension - 1;
        if (ambient.size() > 1)
            out["ambient_components"] = list(ambient);
        vertices.append(std::move(out));
    }
    std::string cell(const Names &key, const Names &ids, const Names &faces, const mpz_class &index) {
        const auto name = "C" + join(key, "-") + "." + index.get_str();
        V out;
        out["id"] = name;
        out["dimension"] = Json::UInt(key.size() - 1);
        out["vertices"] = list(ids);
        out["faces"] = list(faces);
        cells.append(std::move(out));
        return name;
    }
    void ambient_complex();
    void intrinsic_complex();
    V complex() {
        if (dimension < 0)
            throw Incomplete{};
        if (intrinsic && dimension >= 2)
            intrinsic_complex();
        else
            ambient_complex();
        V result;
        result["vertices"] = vertices;
        result["cells"] = cells;
        result["status"] = "complete";
        result["convention"] = std::string(intrinsic ? "Exceptional divisor on the smooth strict transform"
                                                     : "Ambient exceptional divisor") +
                               " in the germ over the origin over Qbar.";
        return result;
    }
};

// Build all strata first so a required coefficient extension is discovered
// before forming any (potentially very large) closed overlap.
void Driver::ambient_complex() {
    std::vector<std::map<Names, Group>> strata(dimension + 1);
    std::set<Names> keys;
    for (const auto &c : charts) {
        Names names;
        for (const auto &[name, ideal] : c.boundary)
            names.push_back(name);
        std::sort(names.begin(), names.end());
        for (size_t n = 1; n <= std::min<size_t>(names.size(), dimension); ++n)
            subsets(names, n, [&](const Names &key) { keys.insert(key); });
    }
    std::vector<Names> ordered(keys.begin(), keys.end());
    std::stable_sort(ordered.begin(), ordered.end(),
                     [](const auto &a, const auto &b) { return a.size() < b.size(); });
    for (const auto &key : ordered) {
        auto &pieces = strata[key.size()][key];
        for (unsigned i = 0; i < charts.size(); ++i) {
            auto &c = charts[i];
            auto base = c.ambient;
            bool supported = true;
            for (const auto &name : key) {
                const auto *ideal = lookup(c.boundary, name);
                if (!ideal) {
                    supported = false;
                    break;
                }
                base = append(std::move(base), *ideal);
            }
            if (!supported)
                continue;
            auto section = [&](const std::string &label, const Basis &ideal) {
                const auto equations = append(base, ideal);
                if (!c.holes.empty() && !present(i, equations))
                    return;
                const auto found = primes(i, equations, dimension - int(key.size()), false);
                if (found.isNull())
                    throw Incomplete{};
                for (const auto &prime : found) {
                    auto local = append(c.ambient, parse(prime, c.width));
                    if (c.retained(local))
                        pieces.emplace_back(Section{i, label}, std::move(local));
                }
            };
            if (intrinsic)
                for (const auto &[label, ideal] : c.strict)
                    section(label, ideal);
            else
                section("", {});
        }
    }
    using Named = std::vector<std::pair<std::string, Group>>;
    std::map<Names, Named> faces;
    for (const auto &level : strata)
        for (const auto &[key, local] : level) {
            auto visible = groups(local);
            unsigned index = 0;
            for (auto &group : visible) {
                ++index;
                std::string id;
                if (key.size() == 1) {
                    id = key[0] + (visible.size() > 1 ? "." + std::to_string(index) : "");
                    vertex(id);
                } else {
                    const auto &[s, prime] = group.front();
                    auto containing = [&](const Names &smaller) {
                        Names hits;
                        for (const auto &[name, pieces] : faces[smaller])
                            for (const auto &[section, face] : pieces)
                                if (section == s && charts[s.first].contains(prime, face))
                                    hits.push_back(name);
                        if (hits.size() != 1)
                            throw std::runtime_error("A component has no unique containing face.");
                        return hits.front();
                    };
                    Names ids, boundary;
                    for (const auto &name : key)
                        ids.push_back(containing({name}));
                    subsets(key, key.size() - 1,
                            [&](const Names &face) { boundary.push_back(containing(face)); });
                    id = cell(key, ids, boundary, index);
                }
                faces[key].emplace_back(std::move(id), std::move(group));
            }
        }
}

void Driver::intrinsic_complex() {
    Group pieces;
    std::map<Piece, std::set<std::string>> labels;
    for (unsigned i = 0; i < charts.size(); ++i) {
        auto &c = charts[i];
        for (const auto &[strict, hypersurface] : c.strict) {
            const auto carrier = append(c.ambient, hypersurface);
            V query;
            query["ideal"] = rows(carrier);
            query["dimension"] = dimension;
            auto singular = call("smooth", i, query);
            if (!singular.isNull() && present(i, append(parse(singular, c.width), c.owned)))
                throw Incomplete{};
            std::vector<Basis> local;
            std::set<Basis> seen;
            for (const auto &[label, divisor] : c.boundary) {
                const auto equations = append(carrier, divisor);
                if (!c.holes.empty() && !present(i, equations))
                    continue;
                const auto found = primes(i, equations, dimension - 1, true);
                if (found.isNull())
                    throw Incomplete{};
                for (const auto &p : found) {
                    const auto whole = unique(append(c.ambient, parse(p, c.width)));
                    if (!c.retained(whole))
                        continue;
                    const auto &ideal = c.canonical(whole);
                    Piece piece{{i, strict}, ideal};
                    if (labels.emplace(piece, std::set<std::string>{}).second)
                        pieces.push_back(piece);
                    labels[piece].insert(label);
                    if (seen.insert(ideal).second)
                        local.push_back(ideal);
                }
            }
            // Modulo the smooth carrier, replacing a generator by its residue
            // preserves its intersection ideal and conormal span there.
            Ring reducer(carrier, c.width, false);
            V snc;
            snc["carrier"] = rows(carrier);
            snc["branches"] = array();
            snc["dimension"] = dimension;
            for (const auto &ideal : local) {
                Basis residual;
                for (const auto &f : ideal)
                    if (std::find(carrier.begin(), carrier.end(), f) == carrier.end()) {
                        auto r = std::get<Polynomial>(reducer.evaluate("reduce", {f}));
                        if (!r.empty())
                            residual.push_back(std::move(r));
                    }
                snc["branches"].append(rows(unique(residual)));
            }
            if (!call("snc", i, snc).asBool())
                throw Incomplete{};
        }
    }

    std::map<Names, Groups> by_labels;
    std::set<std::string> reserved, used;
    for (auto &group : groups(pieces)) {
        std::set<std::string> ambient;
        for (const auto &piece : group)
            ambient.insert(labels.at(piece).begin(), labels.at(piece).end());
        by_labels[Names(ambient.begin(), ambient.end())].push_back(std::move(group));
        reserved.insert(ambient.begin(), ambient.end());
    }
    using Named = std::vector<std::pair<std::string, Basis>>;
    std::map<std::pair<Section, Names>, Named> faces;
    std::map<Section, Named> local;
    std::vector<Section> section_order;
    for (const auto &[ambient, components] : by_labels)
        for (size_t j = 0; j < components.size(); ++j) {
            auto id = join(ambient, "&");
            if (components.size() > 1)
                id += "." + std::to_string(j + 1);
            while (used.count(id) || (reserved.count(id) && (components.size() != 1 || ambient.size() != 1)))
                id += "_";
            used.insert(id);
            vertex(id, ambient);
            for (const auto &[s, p] : components[j]) {
                if (!local.count(s))
                    section_order.push_back(s);
                local[s].emplace_back(id, p);
                faces[{s, {id}}].emplace_back(id, p);
            }
        }
    for (int size = 2; size <= dimension; ++size) {
        std::map<Names, Group> intersections;
        for (const auto &s : section_order) {
            const auto &divisors = local.at(s);
            if (divisors.size() < size_t(size))
                continue;
            Names names;
            for (const auto &[id, p] : divisors)
                names.push_back(id);
            std::sort(names.begin(), names.end());
            subsets(names, size, [&](const Names &key) {
                Basis ideal;
                for (const auto &id : key)
                    ideal = append(std::move(ideal), *lookup(divisors, id));
                intersections[key].emplace_back(s, unique(ideal));
            });
        }
        for (const auto &[key, sections] : intersections) {
            std::vector<Names> subkeys;
            subsets(key, size - 1, [&](const Names &sub) { subkeys.push_back(sub); });
            mpz_class index = 0;
            if (size == dimension) {
                // The incident faces partition every owned zero-dimensional
                // scheme. Euler sums count parallel top cells with fixed faces.
                std::map<Names, mpz_class> totals;
                for (const auto &[s, equations] : sections) {
                    auto owned = append(equations, charts[s.first].owned);
                    const auto expected = euler(s.first, owned);
                    if (expected == 0)
                        continue;
                    mpz_class observed = 0;
                    Names names;
                    auto count = [&](auto &&self, size_t k, Basis extra) -> void {
                        if (k == subkeys.size()) {
                            const auto n =
                                extra.empty() ? expected : euler(s.first, append(owned, unique(extra)));
                            if (n != 0) {
                                totals[names] += n;
                                observed += n;
                            }
                            return;
                        }
                        for (const auto &[id, face] : faces[{s, subkeys[k]}]) {
                            auto added = extra;
                            for (const auto &f : face)
                                if (std::find(equations.begin(), equations.end(), f) == equations.end())
                                    added.push_back(f);
                            names.push_back(id);
                            self(self, k + 1, std::move(added));
                            names.pop_back();
                        }
                    };
                    count(count, 0, {});
                    if (observed != expected)
                        throw Incomplete{};
                }
                for (const auto &[boundary, n] : totals)
                    for (mpz_class k = 0; k < n; ++k)
                        cell(key, key, boundary, ++index);
            } else {
                Group local_primes;
                for (const auto &[s, equations] : sections) {
                    auto &c = charts[s.first];
                    if (!present(s.first, equations))
                        continue;
                    const auto found = primes(s.first, equations, dimension - size, true);
                    if (found.isNull())
                        throw Incomplete{};
                    for (const auto &p : found) {
                        auto ideal = unique(append(c.ambient, parse(p, c.width)));
                        if (c.retained(ideal))
                            local_primes.emplace_back(s, std::move(ideal));
                    }
                }
                for (const auto &group : groups(local_primes)) {
                    Names boundary;
                    for (const auto &sub : subkeys) {
                        std::optional<std::string> incident;
                        for (const auto &[s, prime] : group) {
                            Names hits;
                            for (const auto &[id, face] : faces[{s, sub}])
                                if (charts[s.first].contains(prime, face))
                                    hits.push_back(id);
                            if (hits.size() != 1 || (incident && hits[0] != *incident))
                                throw Incomplete{};
                            incident = hits.front();
                        }
                        boundary.push_back(*incident);
                    }
                    const auto id = cell(key, key, boundary, ++index);
                    for (const auto &[s, prime] : group)
                        faces[{s, key}].emplace_back(id, prime);
                }
            }
        }
    }
}

// Exact character splitting on coordinate flats. If M is primitive, M²=d
// has the two geometrically integral branches M=±sqrt(d) on the marked sheet.
struct Character {
    Basis fixed;
    Polynomial monomial;
    mpq_class ratio;
};
std::optional<Character> character(const Basis &input, size_t width) {
    Character out;
    std::set<Polynomial> fixed;
    std::vector<size_t> indices;
    for (size_t i = 0; i < width; ++i) {
        auto v = axis(i, width);
        if (std::find(input.begin(), input.end(), v) != input.end()) {
            indices.push_back(i);
            fixed.insert(v);
            out.fixed.push_back(std::move(v));
        }
    }
    if (indices.empty() || indices.size() == width)
        return {};
    std::set<Polynomial> other;
    for (const auto &f : input)
        if (!f.empty() && !fixed.count(f))
            other.insert(f);
    if (other.size() != 1)
        return {};
    Polynomial restricted;
    for (const auto &[p, c] : *other.begin())
        if (std::all_of(indices.begin(), indices.end(), [&](size_t k) { return p[k] == 0; }))
            restricted[p] = c;
    if (restricted.size() != 2 || restricted.begin()->first != Powers(width))
        return {};
    const auto &[p, leading] = *restricted.rbegin();
    mpz_class gcd = 0;
    for (const auto &k : p)
        mpz_gcd(gcd.get_mpz_t(), gcd.get_mpz_t(), k.get_mpz_t());
    if (gcd != 2)
        return {};
    auto powers = p;
    for (auto &k : powers)
        k /= 2;
    out.monomial.emplace(std::move(powers), 1);
    out.ratio = -restricted.begin()->second / leading;
    return out;
}
std::optional<mpq_class> rational_root(const mpq_class &q) {
    if (q < 0 || !mpz_perfect_square_p(q.get_num_mpz_t()) || !mpz_perfect_square_p(q.get_den_mpz_t()))
        return {};
    mpz_class a, b;
    mpz_sqrt(a.get_mpz_t(), q.get_num_mpz_t());
    mpz_sqrt(b.get_mpz_t(), q.get_den_mpz_t());
    return mpq_class(a, b);
}
V character_field(const V &request) {
    for (const auto &c : request["charts"])
        if (!c["ambient"].empty() || c["theta"].asInt() >= 0)
            return {};
    for (const auto &item : request["charts"]) {
        Chart c(item);
        for (const auto &[strict, carrier] : c.strict)
            for (const auto &[id, divisor] : c.boundary) {
                const auto s = character(append(carrier, divisor), c.width);
                if (!s || rational_root(s->ratio))
                    continue;
                const auto &d = s->ratio;
                V out;
                out["d"] = d.get_str();
                auto relation = Polynomial{{Powers{2}, 1}, {Powers{0}, -d}};
                out["relation"] = encode_row(relation);
                out["embedding"] = "PolynomialRootIndexed(List(" + mpz_class(-d.get_num()).get_str() +
                                   ", 0, " + d.get_den().get_str() + "), 1)";
                return out;
            }
    }
    return {};
}
V prime_plan(const V &q) {
    const auto width = q["width"].asUInt();
    auto equations = parse(q["ideal"], width);
    const int theta = q["theta"].asInt(), dimension = q["dimension"].asInt();
    V result;
    Axes free;
    for (size_t i = 0; i < width; ++i)
        if (int(i) != theta)
            free.push_back(i);
    if (theta >= 0) {
        const auto relation = decode_row(q["relation"].asString(), width);
        equations.erase(std::remove(equations.begin(), equations.end(), relation), equations.end());
        for (const auto &f : equations)
            for (const auto &[p, c] : f)
                if (p[theta] != 0)
                    return result;
        if (!q["character"].isNull()) {
            const auto s = character(equations, width);
            if (s && int(free.size()) - int(s->fixed.size()) - 1 == dimension) {
                const auto r = rational_root(s->ratio / mpq_class(q["character"].asString()));
                if (r) {
                    result["primes"] = array();
                    for (int sign : {-1, 1}) {
                        auto f = s->monomial;
                        auto p = Powers(width);
                        p[theta] = 1;
                        f[p] += sign * (*r);
                        result["primes"].append(rows(append(s->fixed, {f})));
                    }
                    return result;
                }
            }
        }
    }
    Axes active;
    for (size_t k : free) {
        bool used = false;
        for (const auto &f : equations)
            for (const auto &[p, c] : f)
                if (p[k] != 0) {
                    used = true;
                    break;
                }
        if (used)
            active.push_back(k);
    }
    if (active.empty()) {
        if (!zero(equations))
            result["primes"] = array();
        else if (dimension == int(free.size())) {
            result["primes"] = array();
            result["primes"].append(array());
        }
        return result;
    }
    result["axes"] = array();
    for (size_t i : active)
        result["axes"].append(Json::UInt(i));
    result["dimension"] = dimension - int(free.size() - active.size());
    result["ideal"] = rows(equations);
    return result;
}

V diagonal(const V &request) {
    Chart a(request["a"]), b(request["b"]);
    const size_t width = a.width + b.width;
    const auto left = interval(0, a.width), right = interval(a.width, b.width);
    Basis graph;
    auto multiplier = unit(width);
    for (const auto &f : a.ambient)
        graph.push_back(mapped(f, left, width));
    for (const auto &f : b.ambient)
        graph.push_back(mapped(f, right, width));
    if (!request["transition"].isNull()) {
        for (const auto &relation : request["transition"]) {
            auto n = mapped(decode_row(relation[0].asString(), a.width), left, width);
            auto h = mapped(decode_row(relation[1].asString(), b.width), right, width);
            auto d = mapped(decode_row(relation[2].asString(), a.width), left, width);
            graph.push_back(difference(primary::multiply(h, d), n));
            multiplier = primary::multiply(multiplier, d);
        }
    } else {
        const bool same = request["descended"].asBool() && a.parent >= 0 && a.parent == b.parent;
        const auto &x = same ? a.repair_map : a.projections;
        const auto &y = same ? b.repair_map : b.projections;
        if (x.size() != y.size())
            throw std::invalid_argument("The chart maps use different base coordinates.");
        for (size_t i = 0; i < x.size(); ++i)
            graph.push_back(difference(mapped(x[i], left, width), mapped(y[i], right, width)));
    }
    if (a.theta >= 0) {
        if (b.theta < 0)
            throw std::invalid_argument("The chart fields do not agree.");
        graph.push_back(difference(axis(left[a.theta], width), axis(right[b.theta], width)));
    }
    V removed = array();
    auto include = [&](Chart &c, const Axes &axes) {
        for (const auto &[id, divisor] : c.boundary) {
            auto pivot = std::find_if(divisor.begin(), divisor.end(),
                                      [&](const auto &f) { return !c.contains(c.ambient, {f}); });
            if (pivot == divisor.end())
                throw std::invalid_argument("The exceptional ideal vanishes on its entire chart.");
            multiplier = primary::multiply(multiplier, mapped(*pivot, axes, width));
        }
        for (const auto &hole : c.holes) {
            Basis equations;
            for (const auto &f : hole)
                equations.push_back(mapped(f, axes, width));
            removed.append(rows(equations));
        }
    };
    include(a, left);
    include(b, right);
    V out;
    out["graph"] = rows(primary::saturate_lex(graph, multiplier, width));
    out["removed"] = removed;
    return out;
}
bool radical_contains(const Basis &ideal, const Basis &tested, size_t width) {
    const auto basis = primary::reduced(ideal, width, true);
    return std::all_of(tested.begin(), tested.end(), [&](const auto &f) {
        if (primary::member_lex(f, basis))
            return true;
        return primary::saturate_lex(basis, f, width) == Basis{unit(width)};
    });
}
bool faithful(const V &request) {
    auto graph = parse(request["graph"], 2 * request["width"].asUInt());
    const auto n = request["width"].asUInt();
    Basis diagonal;
    for (size_t i = 0; i < n; ++i)
        diagonal.push_back(difference(axis(i, 2 * n), axis(n + i, 2 * n)));
    if (!radical_contains(graph, diagonal, 2 * n))
        return false;
    const auto axes = interval(0, n);
    Axes fold = axes;
    fold.insert(fold.end(), axes.begin(), axes.end());
    for (auto &f : graph)
        f = mapped(f, fold, n);
    return radical_contains(parse(request["ambient"], n), graph, n);
}
V self_plan(const V &request) {
    Chart chart(request["chart"]);
    Basis geometric;
    for (size_t i = 0; i < chart.width; ++i)
        if (int(i) != chart.theta)
            geometric.push_back(axis(i, chart.width));
    const bool link = request["descended"].asBool() && chart.parent >= 0;
    const bool parent = link && chart.repair_map == geometric;
    const auto &images = link && !parent ? chart.repair_map : chart.projections;
    bool complete = std::all_of(geometric.begin(), geometric.end(), [&](const auto &f) {
        return std::find(images.begin(), images.end(), f) != images.end();
    });
    V out;
    out["identity"] = complete;
    out["ordinary_parent"] = parent;
    if (!complete && link)
        out["inverse"] = rows(images);
    return out;
}

V preflight(const V &q) {
    const int dim = q["dimension"].asInt();
    const bool intrinsic = q["intrinsic"].asBool();
    const int carrier = dim - int(intrinsic);
    V out;
    out["valid"] = true;
    out["carrier"] = carrier;
    out["check"] = array();
    out["restricted"] = intrinsic && dim >= 3;
    unsigned i = 0;
    for (const auto &chart : q["charts"]) {
        if (chart["ambient"].empty() && int(chart["width"].asUInt()) != dim)
            out["valid"] = false;
        if (!out["restricted"].asBool()) {
            if (intrinsic) {
                for (const auto &component : chart["components"])
                    if (component[2].asBool()) {
                        V row = array();
                        row.append(i);
                        row.append(component[1]);
                        out["check"].append(std::move(row));
                    }
            } else {
                V row = array();
                row.append(i);
                row.append(array());
                out["check"].append(std::move(row));
            }
        }
        ++i;
    }
    return out;
}
V exercise(const V &request, Callback callback) {
    Driver runner(request, callback);
    try {
        if (request["operation"] == "groups") {
            Group pieces;
            for (const auto &p : request["pieces"]) {
                const auto i = p[0].asUInt();
                pieces.emplace_back(Section{i, p[1].asString()}, parse(p[2], runner.charts.at(i).width));
            }
            V out = array();
            for (const auto &group : runner.group_indices(pieces)) {
                V ids = array();
                for (auto k : group)
                    ids.append(Json::UInt64(k));
                out.append(std::move(ids));
            }
            return out;
        }
        return runner.complex();
    } catch (const Incomplete &) {
        return {};
    }
}
} // namespace

extern "C" const char *atlas_chart_gluing(const char *input, Callback callback) noexcept {
    return atlas::sparse_json::invoke(input, [&](const V &request) { return exercise(request, callback); });
}

extern "C" const char *atlas_chart_geometry(const char *input) noexcept {
    return atlas::sparse_json::invoke(input, [](const V &request) -> V {
        auto op = request["operation"].asString();
        if (op == "preflight")
            return preflight(request);
        if (op == "diagonal")
            return diagonal(request);
        if (op == "self")
            return self_plan(request);
        if (op == "faithful")
            return faithful(request);
        if (op == "character_field")
            return character_field(request);
        if (op == "prime_plan")
            return prime_plan(request);
        throw std::invalid_argument("Unknown affine chart geometry operation.");
    });
}
