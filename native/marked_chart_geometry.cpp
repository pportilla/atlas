// Absolute sections of marked affine charts and certification of rational transitions.
#include "chart_kernel_rows.hpp"
#include "primary_decomposition.hpp"
#include "quotient_ring.hpp"
#include "rational_sparse_json.hpp"
#include "polynomial_factors.hpp"
#include "branch_order.hpp"
#include "sparse_flint.hpp"
#include <algorithm>
#include <limits>
#include <optional>
extern "C" {
char *atlas_affine(const char *, size_t) noexcept;
char *atlas_field_coefficient(const char *, size_t) noexcept;
void atlas_groebner_release(char *) noexcept;
const char *atlas_groebner_error() noexcept;
}
namespace {
using namespace atlas::chart_rows;
namespace primary = atlas::primary;
using Ring = atlas::quotient_ring::Ring;
using FractionMap = primary::FractionMap;
using Z = mpz_class;
using Q = mpq_class;
using Callback = const char *(*)(const char *);
struct Calls {
    Callback callback;
    V operator()(const std::string &op, V q) const {
        q["op"] = op;
        Json::StreamWriterBuilder writer;
        writer["indentation"] = "";
        auto text = Json::writeString(writer, q);
        auto raw = callback(text.c_str());
        if (!raw)
            throw std::runtime_error("Marked chart task interrupted.");
        V value;
        std::string error;
        std::istringstream is(raw);
        Json::CharReaderBuilder parser;
        if (!Json::parseFromStream(parser, is, &value, &error))
            throw std::runtime_error(error);
        return value;
    }
};
V dense(const Polynomial &f, bool integers) {
    if (f.empty())
        return array();
    const Z degree = f.rbegin()->first[0];
    if (degree >= std::numeric_limits<Json::ArrayIndex>::max())
        throw std::overflow_error("Marked root polynomial exceeds the addressable dense length.");
    auto n = degree.get_ui();
    V out = array();
    for (size_t i = 0; i <= n; ++i) {
        auto it = f.find(Powers{n - i});
        const Q c = it == f.end() ? Q(0) : it->second;
        if (integers && c.get_den() != 1)
            throw std::invalid_argument("Expected a primitive integral minimal polynomial.");
        out.append(c.get_str());
    }
    return out;
}
Polynomial primitive(Polynomial f) {
    if (f.empty())
        return f;
    Z lcm = 1, gcd = 0;
    for (const auto &[p, c] : f) {
        mpz_lcm(lcm.get_mpz_t(), lcm.get_mpz_t(), c.get_den_mpz_t());
        mpz_gcd(gcd.get_mpz_t(), gcd.get_mpz_t(), c.get_num_mpz_t());
    }
    Q scale(lcm, gcd);
    if (f.rbegin()->second < 0)
        scale = -scale;
    for (auto &[p, c] : f)
        c *= scale;
    return f;
}
std::vector<Polynomial> factors(const Polynomial &f) {
    auto all = primary::factors(f, 1);
    for (auto &[g, m] : all)
        g = primitive(std::move(g));
    auto order = atlas::archived_branch_order(all, 1);
    std::vector<Polynomial> result;
    for (auto i : order)
        result.push_back(std::move(all[i].first));
    return result;
}
std::string literal(const V &coefficients, unsigned index) {
    std::string s = "PolynomialRootIndexed(List(";
    for (size_t j = coefficients.size(); j > 0; --j) {
        if (j != coefficients.size())
            s += ", ";
        s += coefficients[Json::UInt(j - 1)].asString();
    }
    return s + "), " + std::to_string(index) + ")";
}
V mark(const V &q) {
    V data = q["description"];
    unsigned degree = q["degree"].asUInt();
    const auto f = atlas::local::decode_row(q["relation"].asString(), 1);
    auto irreducible = primary::factors(f, 1);
    if (irreducible.size() != 1 || irreducible[0].second != 1)
        return {};
    auto coefficients = dense(primitive(std::move(irreducible[0].first)), true);
    if (coefficients.size() != degree + 1)
        return {};
    if (!data.isNull() && data != coefficients)
        return {};
    const Z index(q["index"].asString());
    if (index < 1 || index > degree)
        return {};
    return literal(coefficients, index.get_ui());
}
V normalized(const V &q) {
    size_t n = q["width"].asUInt(), theta = q["theta"].asUInt();
    int dim = q["dimension"].asInt();
    auto eq = parse(q["ideal"], n);
    const auto m = atlas::local::decode_row(q["relation"].asString(), n);
    Ring ring({m}, n, true);
    Basis equations;
    for (const auto &f : eq) {
        if (f == m)
            continue;
        auto r = std::get<Polynomial>(ring.evaluate("reduce", {f}));
        if (!r.empty())
            equations.push_back(std::move(r));
    }
    auto active = support(equations, n);
    const bool rational = std::find(active.begin(), active.end(), theta) == active.end();
    active.erase(std::remove(active.begin(), active.end(), theta), active.end());
    dim -= int(n - 1 - active.size());
    V out;
    out["axes"] = indices(active);
    active.push_back(theta);
    out["dimension"] = dim;
    out["ideal"] = rows(project(equations, active));
    out["rational"] = rational;
    if (active.size() == 1 || dim < 0) {
        auto input = project(equations, active);
        input.push_back(project(m, active));
        auto basis = primary::reduced(input, active.size(), true);
        out["final"] = true;
        if (basis == Basis{one(active.size())})
            out["result"] = array();
        else if (active.size() == 1 && dim == 0) {
            out["result"] = array();
            out["result"].append(array());
        } else
            out["result"] = V();
    }
    return out;
}
struct RootSolver {
    Calls call;
    std::string embedding;
    std::map<std::string, V> images;
    std::map<std::pair<std::string, Polynomial>, bool> on_sheet;
    V coordinates(const std::string &root) {
        auto it = images.find(root);
        if (it != images.end())
            return it->second;
        V q;
        q["root"] = root;
        q["mark"] = embedding;
        return images.emplace(root, call("coordinates", q)).first->second;
    }
    bool satisfies(const std::string &root, const Polynomial &f) {
        auto key = std::make_pair(root, f);
        auto it = on_sheet.find(key);
        if (it != on_sheet.end())
            return it->second;
        V q;
        q["root"] = root;
        q["mark"] = embedding;
        q["equation"] = atlas::local::encode_row(f);
        return on_sheet.emplace(key, call("satisfies", q).asBool()).first->second;
    }
};
V finite_sections(const Basis &eq, const Polynomial &m, size_t n, int dim, RootSolver &solver) {
    Basis g = primary::reduced(eq, n, true);
    if (g == Basis{one(n)})
        return array();
    std::set<size_t> pivots;
    bool triangular = true;
    for (const auto &f : g) {
        const auto &p = f.rbegin()->first;
        auto it = std::find_if(p.begin(), p.end() - 1, [](const auto &e) { return e != 0; });
        if (it == p.end() - 1)
            continue;
        if (std::accumulate(p.begin(), p.end(), Z(0)) != 1) {
            triangular = false;
            break;
        }
        pivots.insert(it - p.begin());
    }
    if (triangular && int(n - 1 - pivots.size()) == dim) {
        Basis pieces;
        for (const auto &f : g)
            if (support({f}, n) != Axes{n - 1})
                pieces.push_back(f);
        auto out = array();
        out.append(rows(pieces));
        return out;
    }
    for (const auto &f : g) {
        auto used = support({f}, n);
        bool marked = used.size() == 2 && used.back() == n - 1;
        if (!marked && (used.size() != 1 || used[0] == n - 1))
            continue;
        size_t k = used[0];
        Polynomial slice = project(f, {k, n - 1});
        if (slice.rbegin()->first[0] < 2)
            continue;
        Polynomial norm;
        if (marked) {
            auto relations = primary::reduced({project(m, {n - 1, k}), project(f, {n - 1, k})}, 2, true);
            for (const auto &h : relations)
                if (support({h}, 2) == Axes{1}) {
                    norm = project(h, {1});
                    break;
                }
            if (norm.empty())
                continue;
        } else
            norm = project(f, {k});
        Basis roots;
        for (const auto &factor : factors(norm)) {
            if (factor.rbegin()->first[0] == 1) {
                roots.push_back(lift(factor, {k}, n));
                continue;
            }
            auto coeff = dense(factor, true);
            for (unsigned i = 1; i < coeff.size(); ++i) {
                const auto root = literal(coeff, i);
                auto image = solver.coordinates(root);
                if (image.isNull()) {
                    if (marked && !solver.satisfies(root, slice))
                        continue;
                    return {};
                }
                auto linear = var(k, n);
                unsigned j = 0;
                for (const auto &a : image) {
                    Powers p(n);
                    p.back() = j++;
                    Q c(a.asString());
                    if (c != 0)
                        linear[p] = -c;
                }
                roots.push_back(std::move(linear));
            }
        }
        V result = array();
        for (const auto &root : roots) {
            auto sub = g;
            sub.push_back(root);
            auto part = finite_sections(sub, m, n, dim, solver);
            if (part.isNull())
                return {};
            for (const auto &leaf : part)
                result.append(leaf);
        }
        return result;
    }
    return {};
}
std::optional<Polynomial> edge_polynomial(const Polynomial &f) {
    if (f.size() < 2)
        return {};
    const auto &low = f.begin()->first;
    const auto &high = f.rbegin()->first;
    Powers d(low.size());
    Z gcd = 0;
    for (size_t i = 0; i < low.size(); ++i) {
        d[i] = high[i] - low[i];
        mpz_gcd(gcd.get_mpz_t(), gcd.get_mpz_t(), d[i].get_mpz_t());
    }
    for (auto &k : d)
        k /= gcd;
    auto axis = std::find_if(d.begin(), d.end(), [](const auto &k) { return k != 0; }) - d.begin();
    Polynomial h;
    for (const auto &[p, c] : f) {
        Z diff = p[axis] - low[axis];
        if (diff % d[axis] != 0)
            return {};
        Z k = diff / d[axis];
        for (size_t i = 0; i < d.size(); ++i)
            if (low[i] + k * d[i] != p[i])
                return {};
        h.emplace(Powers{k}, c);
    }
    return h;
}
struct Proposals {
    RootSolver &solver;
    std::vector<std::string> finite;
    std::set<std::vector<std::string>> tried;
    V result;
    bool add(const Polynomial &norm, bool coordinate, const Polynomial *slice = nullptr) {
        for (const auto &factor : factors(norm)) {
            if (factor.rbegin()->first[0] <= 1)
                continue;
            const auto coeff = dense(factor, true);
            std::vector<std::string> roots;
            for (unsigned j = 1; j < coeff.size(); ++j) {
                auto root = literal(coeff, j);
                if (!slice || solver.satisfies(root, *slice))
                    roots.push_back(std::move(root));
            }
            auto append = [](auto &a, const auto &b) {
                for (const auto &x : b)
                    if (std::find(a.begin(), a.end(), x) == a.end())
                        a.push_back(x);
            };
            if (coordinate)
                append(finite, roots);
            auto proposal = finite;
            append(proposal, roots);
            if (proposal.empty() || !tried.insert(proposal).second)
                continue;
            V q;
            q["roots"] = array();
            for (const auto &r : proposal)
                q["roots"].append(r);
            if (solver.call("attempt", q).asBool()) {
                result = q["roots"];
                return true;
            }
        }
        return false;
    }
};
V propose(const V &q, RootSolver &solver) {
    size_t n = q["width"].asUInt();
    int theta = q["theta"].asInt();
    const auto equations = parse(q["ideal"], n);
    const auto active = support(equations, n);
    if (active.empty())
        return {};
    auto g = lift(primary::reduced(project(equations, active), active.size(), true), active, n);
    Proposals proposals{solver, {}, {}, {}};
    for (const auto &f : g) {
        auto used = support({f}, n);
        if (std::find(used.begin(), used.end(), size_t(theta)) != used.end())
            continue;
        auto edge = edge_polynomial(f);
        if (edge && proposals.add(*edge, true))
            return proposals.result;
    }
    for (const auto &f : g) {
        auto used = support({f}, n);
        bool marked = std::find(used.begin(), used.end(), size_t(theta)) != used.end();
        used.erase(std::remove(used.begin(), used.end(), size_t(theta)), used.end());
        for (auto k : used)
            for (int value : {0, 1, -1}) {
                Polynomial h;
                for (const auto &[p, c] : f) {
                    Q a = c;
                    for (auto j : used)
                        if (j != k && p[j] != 0) {
                            if (value == 0) {
                                a = 0;
                                break;
                            }
                            if (value == -1 && mpz_odd_p(p[j].get_mpz_t()))
                                a = -a;
                        }
                    if (a != 0) {
                        Powers e = {p[k], marked ? p[theta] : Z(0)};
                        h[e] += a;
                        if (h[e] == 0)
                            h.erase(e);
                    }
                }
                auto variables = support({h}, 2);
                if (std::find(variables.begin(), variables.end(), 0) == variables.end())
                    continue;
                if (variables == Axes{0, 1}) {
                    auto m = atlas::local::decode_row(q["relation"].asString(), n);
                    auto norm =
                        primary::reduced({project(m, {size_t(theta), k}), project(h, {1, 0})}, 2, true);
                    for (const auto &row : norm)
                        if (support({row}, 2) == Axes{1} &&
                            proposals.add(project(row, {1}), used.size() == 1, &h))
                            return proposals.result;
                } else if (proposals.add(project(h, {0}), used.size() == 1))
                    return proposals.result;
                if (used.size() == 1)
                    break;
            }
    }
    return {};
}
// Substitute a rational coordinate tuple into several polynomials, clearing
// a single common denominator for each pair. This preserves their quotient.
Basis compose(const Basis &input, const FractionMap &images, size_t width, const Ring &ring) {
    Powers degree(images.size());
    for (const auto &f : input)
        for (const auto &[p, c] : f)
            for (size_t k = 0; k < images.size(); ++k)
                degree[k] = std::max(degree[k], p[k]);
    std::map<std::pair<size_t, Z>, Polynomial> powers;
    auto power = [&](size_t k, const Z &e) -> const Polynomial & {
        auto key = std::make_pair(k, e);
        auto it = powers.find(key);
        if (it != powers.end())
            return it->second;
        const auto &f = k % 2 ? images[k / 2].second : images[k / 2].first;
        return powers.emplace(std::move(key), std::get<Polynomial>(ring.evaluate("power", {f}, e)))
            .first->second;
    };
    Basis out;
    for (const auto &f : input) {
        Polynomial image;
        for (const auto &[p, c] : f) {
            Polynomial term{{Powers(width), c}};
            for (size_t k = 0; k < p.size() && !term.empty(); ++k) {
                if (p[k] != 0)
                    term = std::get<Polynomial>(ring.evaluate("multiply", {term, power(2 * k, p[k])}));
                if (degree[k] != p[k] && images[k].second != one(width))
                    term = std::get<Polynomial>(
                        ring.evaluate("multiply", {term, power(2 * k + 1, degree[k] - p[k])}));
            }
            for (const auto &[q, a] : term) {
                image[q] += a;
                if (image[q] == 0)
                    image.erase(q);
            }
        }
        out.push_back(std::move(image));
    }
    return out;
}
Basis cofactor(const Polynomial &a, const Polynomial &b, const Polynomial &m, int theta, size_t width) {
    if (b.empty())
        throw std::runtime_error("A transition denominator vanished on the marked sheet.");
    if (a.empty())
        return {{}, one(width)};
    auto unused = support({b}, width);
    unused.erase(std::remove(unused.begin(), unused.end(), size_t(theta)), unused.end());
    if (unused.empty())
        return {a, b};
    if (theta >= 0) {
        auto axes = ids(width);
        axes.erase(axes.begin() + theta);
        axes.push_back(theta);
        return lift(atlas::marked_cofactors(project(a, axes), project(b, axes), project(m, axes), width),
                    axes, width);
    }
    atlas::local::SparseLexRing ring(width);
    atlas::PolyValue f(ring.ctx), g(ring.ctx), common(ring.ctx), quotient(ring.ctx);
    ring.write(f.p, a);
    ring.write(g.p, b);
    if (!fmpq_mpoly_gcd(common.p, f.p, g.p, ring.ctx))
        throw std::runtime_error("FLINT could not compute the polynomial gcd.");
    fmpq_mpoly_make_monic(common.p, common.p, ring.ctx);
    fmpq_one(fmpq_mpoly_content_ref(common.p, ring.ctx));
    Basis result;
    for (auto x : {f.p, g.p}) {
        if (!fmpq_mpoly_divides(quotient.p, x, common.p, ring.ctx))
            throw std::runtime_error("Inexact primitive gcd division.");
        result.push_back(ring.read(quotient.p));
    }
    return result;
}
FractionMap pairs(const V &data, size_t n) {
    FractionMap out;
    for (const auto &pair : data) {
        auto f = parse(pair, n);
        out.emplace_back(f.at(0), f.at(1));
    }
    return out;
}
V repair(const V &q) {
    const auto op = q["operation"].asString();
    const size_t n = q["width"].asUInt();
    const int theta = q["theta"].asInt();
    const auto m = theta >= 0 ? atlas::local::decode_row(q["relation"].asString(), n) : Polynomial{};
    const Basis relations = m.empty() ? Basis{} : Basis{m};
    if (op == "cofactors") {
        auto pair = parse(q["pair"], n);
        return rows(cofactor(pair.at(0), pair.at(1), m, theta, n));
    }
    if (op == "faithful_inverse") {
        auto images = parse(q["mapping"], n);
        if (theta >= 0)
            images.push_back(var(theta, n));
        if (parse(q["ambient"], n) != relations || images.size() != n ||
            (theta >= 0 && size_t(theta + 1) != n))
            return {};
        auto names = ids(n);
        std::string axes;
        std::vector<std::string> variables;
        for (auto k : names) {
            variables.push_back("v" + std::to_string(k));
            axes += variables.back() + " ";
        }
        std::string query = "inverse\n" + axes + "\n" + axes + "\n";
        for (const auto &f : images)
            query += atlas::local::format_polynomial(f, variables) + "\n";
        std::unique_ptr<char, decltype(&atlas_groebner_release)> answer(
            atlas_affine(query.c_str(), query.size()), atlas_groebner_release);
        if (!answer)
            return {};
        std::istringstream input(answer.get());
        V encoded = array();
        FractionMap map;
        for (const auto &f : images)
            map.emplace_back(f, one(n));
        Ring sheet(relations, n, true);
        for (size_t j = 0; j < n; ++j) {
            std::string a, b;
            if (!std::getline(input, a) || !std::getline(input, b))
                throw std::runtime_error("Incomplete affine coordinate inverse.");
            Basis pair{atlas::local::decode_row(a, n), atlas::local::decode_row(b, n)};
            const auto value = compose(pair, map, n, sheet);
            if (value[1].empty() ||
                std::get<Polynomial>(sheet.evaluate("multiply", {var(j, n), value[1]})) != value[0])
                return {};
            encoded.append(rows(pair));
        }
        return encoded;
    }
    if (op == "transfer") {
        auto mapping = pairs(q["map"], n);
        const auto f = atlas::local::decode_row(q["equation"].asString(), mapping.size());
        auto carrier = q["prime"].isNull() ? relations : parse(q["prime"], n);
        Ring ring(carrier, n, false);
        return rows(compose({f}, mapping, n, ring));
    }
    Ring sheet(relations, n, true);
    auto input = parse(q["mapping"], n);
    if (theta >= 0)
        input.push_back(var(theta, n));
    FractionMap images;
    for (const auto &f : input)
        images.emplace_back(f, one(n));
    auto specialize = [&](const V &inverse, const FractionMap &map) -> std::optional<FractionMap> {
        FractionMap result;
        for (const auto &pair : pairs(inverse, map.size())) {
            auto value = compose({pair.first, pair.second}, map, n, sheet);
            if (value[1].empty())
                return {};
            auto reduced = cofactor(value[0], value[1], m, theta, n);
            result.emplace_back(std::move(reduced[0]), std::move(reduced[1]));
        }
        return result;
    };
    if (!q["parent"].isNull()) {
        auto mid = specialize(q["parent"], images);
        if (!mid)
            return {};
        images = std::move(*mid);
    }
    auto fractions = specialize(q["inverse"], images);
    if (!fractions)
        return {};
    V out;
    out["fractions"] = array();
    out["holes"] = array();
    std::set<Polynomial> seen;
    for (const auto &[a, b] : *fractions) {
        out["fractions"].append(rows({a, b}));
        const auto used = support({b}, n);
        if (std::any_of(used.begin(), used.end(), [&](size_t i) { return int(i) != theta; }) &&
            seen.insert(b).second)
            out["holes"].append(atlas::local::encode_row(b));
    }
    return out;
}

V field_operation(const V &q) {
    const auto op = q["operation"].asString();
    if (op == "larger") {
        for (const auto &d : q["degrees"])
            if (d.asUInt() >= q["degree"].asUInt())
                return false;
        return q["degree"].asUInt() > 1;
    }
    if (op == "on_sheet") {
        auto f = atlas::local::decode_row(q["equation"].asString(), 2);
        std::string expression = "q 0";
        for (const auto &[p, c] : f) {
            expression += " q " + c.get_str();
            if (p[0] != 0)
                expression += " " + q["root_program"].asString() + " pow " + p[0].get_str() + " *";
            if (p[1] != 0)
                expression += " " + q["mark_program"].asString() + " pow " + p[1].get_str() + " *";
            expression += " +";
        }
        auto request = "q 0\n" + expression + "\n";
        std::unique_ptr<char, decltype(&atlas_groebner_release)> result(
            atlas_field_coefficient(request.c_str(), request.size()), atlas_groebner_release);
        if (!result)
            throw std::runtime_error(atlas_groebner_error());
        return std::string(result.get()) == "\n" || std::string(result.get()) == "0\n";
    }
    if (op == "faithful_inverse" || op == "transition" || op == "transfer" || op == "cofactors")
        return repair(q);
    if (op == "normalize")
        return normalized(q);
    if (op == "mark")
        return mark(q);
    if (op == "integers") {
        auto f = atlas::local::decode_row(q["f"].asString(), 1);
        V out = array();
        for (const auto &h : factors(f))
            if (h.rbegin()->first[0] > 1)
                out.append(dense(h, true));
        return out;
    }
    if (op == "minimal")
        return dense(atlas::local::decode_row(q["f"].asString(), 1), false);
    throw std::invalid_argument("Unknown marked chart operation.");
}
} // namespace
extern "C" const char *atlas_marked_chart(const char *input) noexcept {
    return atlas::sparse_json::invoke(input, field_operation);
}
extern "C" const char *atlas_marked_sections(const char *input, Callback callback) noexcept {
    return atlas::sparse_json::invoke(input, [&](const V &q) -> V {
        RootSolver solver{{callback}, q["mark"].asString(), {}, {}};
        auto op = q["operation"].asString();
        if (op == "propose")
            return propose(q, solver);
        if (op != "sections")
            throw std::invalid_argument("Unknown marked section task.");
        size_t n = q["width"].asUInt();
        auto m = atlas::local::decode_row(q["relation"].asString(), n);
        auto eq = parse(q["ideal"], n);
        eq.insert(eq.begin(), m);
        return finite_sections(eq, m, n, q["dimension"].asInt(), solver);
    });
}
