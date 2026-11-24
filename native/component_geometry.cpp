// Sparse geometric factor certificates and the ordered projection schedule.
#include "rational_sparse_json.hpp"
#include "component_projection.hpp"
#include "primary_decomposition.hpp"
#include "component_partition.hpp"
#include "local_io.hpp"
#include "local_point_quotient.hpp"

#include <flint/fmpq_mpoly_factor.h>
#include <flint/fmpz_vec.h>
#include <algorithm>
#include <memory>
#include <numeric>
#include <set>

extern "C" char* atlas_affine(const char*, size_t) noexcept;
extern "C" char* atlas_ideal(const char*, size_t) noexcept;
extern "C" const char* atlas_groebner_error() noexcept;
extern "C" void atlas_groebner_release(char*) noexcept;

namespace atlas::components {
using namespace sparse_json;
namespace {

Z degree(const Powers& powers) { return std::accumulate(powers.begin(), powers.end(), Z(0)); }

struct Support {
    std::vector<Powers> rows;
    Powers lead, bounds;
    Z degree = -1;
    explicit Support(const Json::Value& data) {
        if (!data.isArray() || data.empty()) throw std::invalid_argument("Expected a nonempty support.");
        bounds.resize(data[0].size());
        for (const auto& powers : data) {
            auto e = read_exponents(powers, bounds.size());
            Z total = components::degree(e);
            if (total > degree || (total == degree && e > lead)) {
                degree = total;
                lead = e;
            }
            for (size_t j = 0; j < e.size(); ++j) bounds[j] = std::max(bounds[j], e[j]);
            rows.push_back(std::move(e));
        }
    }
};

template<class Visit>
bool enumerate(Powers& row, const Powers& bounds, size_t j, const Z& remaining, Visit& visit) {
    if (j == row.size()) return visit(row);
    const Z limit = std::min(bounds[j], remaining);
    for (row[j] = 0; row[j] <= limit; ++row[j]) {
        const Z rest = remaining-row[j];
        if (!enumerate(row, bounds, j+1, rest, visit)) return false;
    }
    return true;
}

Json::Value factor_supports(const Support& p, const Z& effort) {
    Json::Value layouts(Json::arrayValue), result;
    const Z max_unknowns = 6+2*effort, max_splits = 32*effort;
    Z checked = 0;
    auto split = [&](const Powers& alpha) {
        Z left_degree = degree(alpha);
        if (!left_degree) return true;
        if (++checked > max_splits) return false;
        Powers beta = p.lead;
        for (size_t j=0; j<beta.size(); ++j) beta[j] -= alpha[j];
        Json::Value sides(Json::arrayValue);
        Z count = 0;
        for (const Powers* head : {&alpha, static_cast<const Powers*>(&beta)}) {
            Json::Value terms(Json::arrayValue);
            terms.append(strings(*head));
            Z top_degree = degree(*head);
            auto add = [&](const Powers& term) {
                if (degree(term) == top_degree && term >= *head) return true;
                if (++count > max_unknowns) return false;
                terms.append(strings(term));
                return true;
            };
            Powers term(head->size());
            if (!enumerate(term, p.bounds, 0, top_degree, add)) return false;
            sides.append(std::move(terms));
        }
        layouts.append(std::move(sides));
        return true;
    };
    Powers alpha(p.bounds.size());
    result["complete"] = enumerate(alpha, p.lead, 0, p.degree/2, split);
    result["lead"] = strings(p.lead);
    result["layouts"] = std::move(layouts);
    return result;
}

using IntMatrix = std::vector<Powers>;
IntMatrix identity(size_t n) {
    IntMatrix matrix(n, Powers(n));
    for (size_t i=0; i<n; ++i) matrix[i][i]=1;
    return matrix;
}
Json::Value matrix_rows(const IntMatrix& matrix) {
    Json::Value rows(Json::arrayValue);
    for (const auto& row : matrix) {
        Json::Value entries(Json::arrayValue);
        for (size_t j=0; j<row.size(); ++j) {
            if (row[j] == 0) continue;
            Json::Value term(Json::arrayValue);
            term.append(static_cast<Json::UInt64>(j));
            term.append(row[j].get_str());
            entries.append(std::move(term));
        }
        rows.append(std::move(entries));
    }
    return rows;
}

Json::Value projection(size_t n, Z code) {
    if (!n || code < 0) throw std::invalid_argument("Invalid projection index.");
    auto matrix = identity(n), inverse = matrix;
    if (code > 0 && code < n) {
        size_t k = code.get_ui();
        std::rotate(matrix.rbegin(), matrix.rbegin()+k, matrix.rend());
        std::rotate(inverse.begin(), inverse.begin()+k, inverse.end());
    } else if (n > 1 && code == n) {
        Z power=1;
        for (size_t j=0; j+1<n; ++j) {
            power *= 2;
            matrix.back()[j] = power;
            inverse.back()[j] = -power;
        }
    } else if (code > n && n > 1) {
        code -= n;
        const Z alphabet = Z(n)*(n-1)*2;
        while (code > 0) {
            --code;
            Z index=code%alphabet;
            code/=alphabet;
            int sign = mpz_odd_p(index.get_mpz_t()) ? -1 : 1;
            index/=2;
            size_t i=Z(index/(n-1)).get_ui(), j=Z(index%(n-1)).get_ui();
            if (j>=i) ++j;
            for (size_t k=0; k<n; ++k) {
                matrix[i][k] += sign*matrix[j][k];
                inverse[k][j] -= sign*inverse[k][i];
            }
        }
    }
    Json::Value pair(Json::arrayValue);
    pair.append(matrix_rows(matrix));
    pair.append(matrix_rows(inverse));
    return pair;
}

Json::Value factor_conditions(const Json::Value& q) {
    const size_t n = q["width"].asUInt(), field = q["marked"].asBool() ? 1 : 0;
    const auto f = read_terms(q["polynomial"], n+field);
    const auto& sides = q["sides"];
    if (sides.size() != 2 || sides[0].empty() || sides[1].empty())
        throw std::invalid_argument("Expected two factor supports.");
    const size_t unknowns = sides[0].size()+sides[1].size()-2;
    const Powers a = read_exponents(sides[0][0], n), b = read_exponents(sides[1][0], n);
    Powers leading(n);
    for (size_t j = 0; j < n; ++j) leading[j] = a[j]+b[j];

    // Work in Q[unknowns,theta]. The Python caller appends the marked
    // field relation to the resulting equations before testing existence.
    std::map<Powers, Terms> equations;
    Terms leading_coefficient;
    for (const auto& [e,c] : f) {
        Powers x(e.begin(), e.begin()+n), parameter(unknowns+field);
        if (field) parameter.back() = e.back();
        equations[x][parameter] -= c;
        if (x == leading) leading_coefficient[parameter] += c;
    }
    if (leading_coefficient.empty()) throw std::invalid_argument("Missing leading coefficient.");
    size_t variable = 0;
    std::vector<std::vector<std::pair<Powers,Terms>>> factors(2);
    for (Json::ArrayIndex side=0; side<2; ++side) {
        for (Json::ArrayIndex i=0; i<sides[side].size(); ++i) {
            Powers parameter(unknowns+field);
            Terms coefficient;
            if (i==0 && side==1) coefficient = leading_coefficient;
            else {
                if (i) parameter[variable++] = 1;
                coefficient[parameter] = 1;
            }
            factors[side].emplace_back(read_exponents(sides[side][i], n), std::move(coefficient));
        }
    }
    // Each coefficient of A*B-f in the geometric variables must vanish.
    for (const auto& [ma, ca] : factors[0]) {
        for (const auto& [mb, cb] : factors[1]) {
            Powers x(n);
            for (size_t j=0; j<n; ++j) x[j] = ma[j]+mb[j];
            auto& equation = equations[x];
            for (const auto& [ea, va] : ca) for (const auto& [eb, vb] : cb) {
                Powers e(unknowns+field);
                for (size_t k=0; k<e.size(); ++k) e[k] = ea[k]+eb[k];
                equation[e] += va*vb;
            }
        }
    }
    Json::Value out(Json::arrayValue);
    for (const auto& [monomial, equation] : equations) {
        auto row = term_list(equation);
        if (!row.empty()) out.append(std::move(row));
    }
    return out;
}

// A rational map presents each coordinate as a numerator and denominator.
// Clear the same denominator in both members of a fraction before composing.
Json::Value homogenize_fraction(const Json::Value& q) {
    const size_t width = q["width"].asUInt();
    if (q["pair"].size() != 2) throw std::invalid_argument("Expected a numerator and a denominator.");
    std::vector<Terms> pair;
    Powers degrees(width);
    for (const auto& source : q["pair"]) {
        pair.push_back(read_terms(source, width));
        for (const auto& [e, c] : pair.back()) {
            if (c == 0) continue;
            for (size_t i = 0; i < width; ++i) degrees[i] = std::max(degrees[i], e[i]);
        }
    }
    Json::Value result(Json::arrayValue);
    for (const auto& source : pair) {
        Terms polynomial;
        for (const auto& [e, c] : source) {
            auto powers = e;
            for (size_t i = 0; i < width; ++i) powers.push_back(degrees[i] - e[i]);
            polynomial.emplace(std::move(powers), c);
        }
        result.append(term_list(polynomial));
    }
    return result;
}

// A projectively smooth hypersurface of dimension at least one is
// geometrically irreducible. For a quadric, the exact test is rank > 2
// over the marked coefficient field. Include its minimal polynomial in
// every ideal while differentiating only in the geometric coordinates.
Json::Value projective_factor(const Json::Value& q) {
    const size_t variables = q["width"].asUInt();
    const bool marked = !q["relation"].isNull();
    const size_t width = variables + marked;
    const auto f = local::decode_row(q["polynomial"].asString(),width);
    const Z d(q["degree"].asString(),10), effort(q["effort"].asString(),10);
    if (f.empty() || !variables) return {};
    const bool homogenize = !q["homogeneous"].asBool();
    const size_t geometric = variables + homogenize, n = width + homogenize;
    if (d>2 && (d>3+effort || Z(geometric)>3+effort)) return {};
    auto lift = [&](const local::Polynomial& g, bool equation) {
        local::Polynomial result;
        for (const auto& [e,c] : g) {
            auto powers = e;
            if (homogenize) {
                const Z offset = equation ? d-std::accumulate(e.begin(),e.begin()+variables,Z(0)) : Z(0);
                if (offset<0) throw std::invalid_argument("Incorrect projective degree.");
                powers.insert(powers.begin()+variables,offset);
            }
            result.emplace(std::move(powers),c);
        }
        return result;
    };
    const auto h = lift(f,true);
    auto rows = local::partial_derivatives(h,n);
    rows.resize(geometric);
    if (marked) rows.push_back(lift(local::decode_row(q["relation"].asString(),width),false));
    const auto gb = primary::reduced(rows,n,true);
    const int dimension = primary::dimension_lex(gb,n);
    if (d==2) return static_cast<int>(geometric)-dimension>2;
    if (geometric<3 || dimension!=0) return {};
    // A nonzero point on this homogeneous affine cone supplies a line
    // through the origin, contradicting its computed zero dimension.
    return true;
}

// A coefficient-comparison system has a factor over the algebraic closure
// exactly when its ideal, with the field relation, is proper.
Json::Value coefficient_test(const Terms& f, const Support& support, const Terms& modulus, const Z& effort) {
    const size_t n=support.bounds.size();
    const bool marked=!modulus.empty();
    const auto layouts=factor_supports(support,effort);
    Json::Value source;
    source["polynomial"]=term_list(f);
    source["width"]=Json::UInt64(n);
    source["marked"]=marked;
    for (const auto& sides : layouts["layouts"]) {
        source["sides"]=sides;
        const size_t width=sides[0].size()+sides[1].size()-2+marked;
        const auto conditions=factor_conditions(source);
        if (!width) {
            if (conditions.empty()) return false;
            continue;
        }
        local::Basis equations;
        for (const auto& row : conditions) equations.push_back(read_terms(row,width));
        if (marked) {
            Terms relation;
            for (const auto& [e,c] : modulus) {
                Powers powers(width);
                powers.back()=e.back();
                relation.emplace(std::move(powers),c);
            }
            equations.push_back(std::move(relation));
        }
        if (primary::reduced(equations,width,true)!=local::Basis{Terms{{Powers(width),1}}}) return false;
    }
    return layouts["complete"].asBool() ? Json::Value(true) : Json::Value();
}

Json::Value support_certificate(const Support& p) {
    if (p.degree==1) return true;
    if (std::count_if(p.bounds.begin(),p.bounds.end(),[](const Z& x){return x!=0;})==1) return false;
    if (std::find(p.bounds.begin(),p.bounds.end(),1)!=p.bounds.end()) return true;
    return {};
}

// Write f = a*x^2 + b*x + c in one of its variables, with a, b, c free of x.
// A factor of f free of x divides gcd(a,b,c). Otherwise f has two factors of
// degree one in x over the algebraic closure, which happens exactly when the
// discriminant b^2 - 4ac is a square there. A square-free factorization over
// Q stays square-free over the closure, so that a nonconstant factor of odd
// multiplicity excludes a square. Then f is absolutely irreducible.
struct RationalPolynomial {
    fmpq_mpoly_t p;
    const fmpq_mpoly_ctx_struct* ctx;
    RationalPolynomial(const fmpq_mpoly_ctx_t c, const Terms& f) : ctx(c) {
        fmpq_mpoly_init(p, ctx);
        const slong width = fmpq_mpoly_ctx_nvars(ctx);
        fmpz* powers = _fmpz_vec_init(width);
        std::vector<fmpz*> entries(width);
        for (slong j = 0; j < width; ++j) entries[j] = powers + j;
        fmpq_t coefficient;
        fmpq_init(coefficient);
        for (const auto& [e, c] : f) {
            for (slong j = 0; j < width; ++j) fmpz_set_mpz(powers + j, e[j].get_mpz_t());
            fmpq_set_mpq(coefficient, c.get_mpq_t());
            fmpq_mpoly_push_term_fmpq_fmpz(p, coefficient, entries.data(), ctx);
        }
        fmpq_clear(coefficient);
        _fmpz_vec_clear(powers, width);
        fmpq_mpoly_sort_terms(p, ctx);
        fmpq_mpoly_combine_like_terms(p, ctx);
    }
    RationalPolynomial(const RationalPolynomial&) = delete;
    ~RationalPolynomial() { fmpq_mpoly_clear(p, ctx); }
};

bool quadratic_certificate(const Terms& f, size_t width) {
    if (f.size() > 4096) return false;
    fmpq_mpoly_ctx_t ctx;
    fmpq_mpoly_ctx_init(ctx, width, ORD_LEX);
    bool certified = false;
    for (size_t axis = 0; axis < width && !certified; ++axis) {
        Terms a, b, c;
        bool quadratic = true;
        for (const auto& [e, coefficient] : f) {
            if (e[axis] > 2) { quadratic = false; break; }
            Powers rest = e;
            rest[axis] = 0;
            (e[axis] == 2 ? a : e[axis] == 1 ? b : c).emplace(std::move(rest), coefficient);
        }
        if (!quadratic || a.empty() || c.empty()) continue;
        RationalPolynomial A(ctx, a), B(ctx, b), C(ctx, c), content(ctx, {}), discriminant(ctx, {}), product(ctx, {});
        if (!fmpq_mpoly_gcd(content.p, A.p, C.p, ctx)) continue;
        if (!b.empty() && !fmpq_mpoly_gcd(content.p, content.p, B.p, ctx)) continue;
        if (!fmpq_mpoly_is_fmpq(content.p, ctx)) continue;
        fmpq_mpoly_mul(discriminant.p, B.p, B.p, ctx);
        fmpq_mpoly_mul(product.p, A.p, C.p, ctx);
        fmpq_mpoly_scalar_mul_si(product.p, product.p, 4, ctx);
        fmpq_mpoly_sub(discriminant.p, discriminant.p, product.p, ctx);
        if (fmpq_mpoly_is_fmpq(discriminant.p, ctx)) continue;
        fmpq_mpoly_factor_t factors;
        fmpq_mpoly_factor_init(factors, ctx);
        if (fmpq_mpoly_factor_squarefree(factors, discriminant.p, ctx))
            for (slong i = 0; i < factors->num; ++i)
                certified = certified || (fmpz_is_odd(factors->exp + i) && !fmpq_mpoly_is_fmpq(factors->poly + i, ctx));
        fmpq_mpoly_factor_clear(factors, ctx);
    }
    fmpq_mpoly_ctx_clear(ctx);
    return certified;
}

Json::Value absolute_factor(const Json::Value& q) {
    const size_t n=q["width"].asUInt();
    const bool marked=!q["relation"].isNull();
    const auto input=local::decode_row(q["polynomial"].asString(),n+marked);
    if (input.empty()) return {};
    const auto relation=marked ? local::decode_row(q["relation"].asString(),n+1) : Terms{};
    if (marked && relation.empty()) throw std::invalid_argument("Missing marked-field relation.");
    const Z effort(q["effort"].asString(),10);
    const bool coefficients_only=q["operation"].asString()=="coefficient_test";
    std::vector<size_t> axes;
    for (size_t j=0; j<n; ++j) {
        if (coefficients_only || std::any_of(input.begin(),input.end(),[j](const auto& term){return term.first[j]!=0;}))
            axes.push_back(j);
    }
    if (axes.empty()) return {};
    Terms f, geometric;
    for (const auto& [e,c] : input) {
        Powers k;
        for (size_t j : axes) k.push_back(e[j]);
        geometric[k]=1;
        if (marked) k.push_back(e.back());
        f.emplace(std::move(k),c);
    }
    Json::Value points(Json::arrayValue);
    for (const auto& [e,c] : geometric) points.append(strings(e));
    const Support support(points);
    if (coefficients_only) return coefficient_test(f,support,relation,effort);
    if (q["primitive"].asBool()) return true;
    const auto quick=support_certificate(support);
    if (!quick.isNull()) return quick;
    std::vector<std::string> names;
    std::string header;
    for (size_t j=0; j<axes.size(); ++j) {
        names.push_back("x"+std::to_string(j));
        header+=names.back()+" ";
    }
    if (!q["support_checked"].asBool()) {
        const auto primitive_input="primitive_support\n"+header+"\n"+local::format_polynomial(geometric,names)+"\n";
        std::unique_ptr<char,decltype(&atlas_groebner_release)> primitive(
            atlas_affine(primitive_input.data(),primitive_input.size()),atlas_groebner_release);
        if (!primitive) throw std::runtime_error(atlas_groebner_error());
        if (std::string(primitive.get())=="1\n") return true;
    }
    if (!marked && quadratic_certificate(f, axes.size())) return true;
    if (axes.size()==2 && !marked) {
        const auto input="plane_components 2 1 0\n"+local::encode_row(f)+"\n";
        std::unique_ptr<char,decltype(&atlas_groebner_release)> reply(atlas_ideal(input.data(),input.size()),atlas_groebner_release);
        if (reply) {
            std::istringstream counts(reply.get());
            long absolute, rational;
            if (counts>>absolute>>rational) return absolute==1;
        }
    } else {
        auto request=q;
        request["width"]=Json::UInt64(axes.size());
        request["polynomial"]=local::encode_row(f);
        request["degree"]=support.degree.get_str();
        request["homogeneous"]=std::all_of(support.rows.begin(),support.rows.end(),
                [&](const auto& e){return degree(e)==support.degree;});
        if (marked) {
            Terms compact;
            for (const auto& [e,c] : relation) {
                Powers k(axes.size()+1);
                k.back()=e.back();
                compact.emplace(std::move(k),c);
            }
            request["relation"]=local::encode_row(compact);
        }
        const auto answer=projective_factor(request);
        if (!answer.isNull()) return answer;
    }
    return coefficient_test(f,support,relation,effort);
}

Json::Value run(const Json::Value& q) {
    const auto op = q["operation"].asString();
    if (op == "absolute_factor" || op == "coefficient_test") return absolute_factor(q);
    if (op == "projection_count") {
        const size_t width=q["width"].asUInt();
        const Z effort(q["effort"].asString(),10);
        return Z(width<=1 ? Z(1) : Z(width)+1+effort).get_str();
    }
    if (op=="linear_images" || op=="projection_rows") {
        const size_t n=q["width"].asUInt();
        Json::Value pairs(Json::arrayValue);
        if (op=="projection_rows") pairs=projection(n,Z(q["index"].asString(),10));
        else pairs.append(q["matrix"]);
        Json::Value out(Json::arrayValue);
        for (const auto& matrix : pairs) {
            if (matrix.size()!=n) throw std::invalid_argument("Incorrect projection matrix dimensions.");
            Json::Value rows(Json::arrayValue);
            for (const auto& row : matrix) {
                Terms f;
                for (const auto& entry : row) {
                    const size_t j=entry[0].asUInt();
                    if (j>=n) throw std::invalid_argument("Invalid projection axis.");
                    Q c(entry[1].asString(),10);
                    if (c.get_den()==0) throw std::invalid_argument("Invalid projection coefficient.");
                    c.canonicalize();
                    if (c==0) continue;
                    Powers e(n);
                    e[j]=1;
                    f[std::move(e)]+=c;
                }
                rows.append(local::encode_row(f));
            }
            out.append(std::move(rows));
        }
        return out;
    }
    if (op == "decomposition_step") return decomposition_step(q);
    if (op == "projection_shape") return projection_shape(q);
    if (op == "eligible_factors") return eligible_factors(q);
    if (op == "projective_factor") return projective_factor(q);
    if (op == "prepare_overlap") return primary::prepare_overlap(q);
    if (op == "generic_overlap") return primary::generic_overlap(q);
    if (op == "homogenize_fraction") return homogenize_fraction(q);
    if (op == "factor_conditions") return factor_conditions(q);
    if (op == "projection") return projection(q["width"].asUInt(), Z(q["index"].asString(), 10));
    Support p(q["support"]);
    if (op == "factor_supports") return factor_supports(p, Z(q["effort"].asString(), 10));
    if (op == "finite_graph") {
        const auto width = p.bounds.size();
        if (!width) throw std::invalid_argument("Expected a coefficient parameter.");
        std::set<size_t> pivots;
        int eliminant = -1;
        for (size_t i = 0; i < p.rows.size(); ++i) {
            const auto& head = p.rows[i];
            const auto pivot = std::find_if(head.begin(), head.end()-1, [](const auto& power) { return power != 0; });
            if (pivot == head.end()-1) {
                if (q["require_eliminant"].asBool()) {
                    if (eliminant >= 0 || head.back() < 2) return Json::Value();
                    eliminant = static_cast<int>(i);
                }
            } else {
                if (degree(head) != 1) return Json::Value();
                pivots.insert(static_cast<size_t>(pivot-head.begin()));
            }
        }
        if (width-1-pivots.size() != q["dimension"].asUInt64()) return Json::Value();
        if (q["require_eliminant"].asBool() && eliminant < 0) return Json::Value();
        return Json::Value(eliminant);
    }
    if (op == "absolute_factor_support") return support_certificate(p);
    if (op == "profile") {
        Json::Value r;
        r["degrees"] = strings(p.bounds);
        r["degree"] = p.degree.get_str();
        r["homogeneous"] = std::all_of(p.rows.begin(), p.rows.end(), [&](const auto& e){return degree(e)==p.degree;});
        return r;
    }
    if (op == "homogenize") {
        Json::Value rows(Json::arrayValue);
        for (auto& e : p.rows) {
            e.push_back(p.degree-degree(e));
            rows.append(strings(e));
        }
        return rows;
    }
    throw std::invalid_argument("Unknown geometric component operation.");
}
} // namespace
Json::Value absolute_rational_factor(const std::string& row, size_t width) {
    Json::Value q;
    q["operation"]="absolute_factor";
    q["width"]=Json::UInt64(width);
    q["polynomial"]=row;
    q["effort"]="1";
    return absolute_factor(q);
}
} // namespace atlas::components

extern "C" const char* atlas_component_geometry(const char* input) noexcept {
    return atlas::sparse_json::invoke(input, atlas::components::run);
}

// Chart intersections are computed by the caller's exact ideal backend.
// Only a certified inequality rules out an entire previously glued class.
extern "C" int atlas_partition_components(size_t count, const size_t* charts, const size_t* strata,
                                          int (*compare)(size_t, size_t), size_t* classes) noexcept {
    try {
        if (count) atlas::components::partition(count, [&](size_t i){return charts[i];},
                                                [&](size_t i){return strata[i];}, compare, classes);
        return 0;
    } catch (...) {
        return -1;
    }
}
