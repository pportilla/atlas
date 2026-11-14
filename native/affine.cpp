// Pullbacks, Jacobians and transitions of affine charts over Q. See KERNEL.md §9.5.
#include "groebner.hpp"
#include <flint/mpoly.h>
#include <flint/fmpq_mpoly.h>
#include <flint/fmpq_mpoly_factor.h>
#include <flint/fmpz_mat.h>
#include <flint/gr.h>
#include <flint/gr_mat.h>
#include <sstream>
#include <cctype>
#include <unordered_set>

namespace atlas {
std::string fan_cover(std::istream& in);
std::string centre_weights(std::istream& in, const std::string& operation);
namespace {

std::string line(std::istream& in) {
    std::string s;
    if (!std::getline(in, s)) throw std::invalid_argument("Missing affine-map input.");
    return s;
}

struct IntegerMatrix {
    fmpz_mat_t matrix;
    IntegerMatrix(slong rows, slong columns) { fmpz_mat_init(matrix, rows, columns); }
    ~IntegerMatrix() { fmpz_mat_clear(matrix); }
    IntegerMatrix(const IntegerMatrix&) = delete;
};

// Rows of R are the rays in chart order; the exponents of the pullback
// of a character a are the column b=R*a. Recover a without any floating
// point or fixed-width conversion, even for negative Laurent exponents.
std::string lattice(std::istream& in, const std::string& operation) {
    const slong n = std::stol(line(in));
    if (n <= 0) throw std::invalid_argument("Expected a positive lattice dimension.");
    const slong split = operation == "lattice_valuations" ? std::stol(line(in)) : 0;
    std::vector<std::vector<Z>> rows;
    for (std::string row; std::getline(in, row);) {
        std::istringstream entries(row);
        std::vector<Z> values;
        for (std::string entry; entries >> entry;) {
            values.emplace_back();
            if (fmpz_set_str(&values.back().v, entry.c_str(), 10))
                throw std::invalid_argument("Expected an integer lattice coordinate.");
        }
        if (static_cast<slong>(values.size()) != n) throw std::invalid_argument("Wrong lattice vector length.");
        rows.push_back(std::move(values));
    }
    const slong count = static_cast<slong>(rows.size());
    if (operation == "lattice_valuations") {
        if (split < 1 || split >= count) throw std::invalid_argument("Expected rays and a nonempty support.");
        IntegerMatrix rays(split, n), support(n, count-split), values(split, count-split);
        for (slong i = 0; i < count; ++i)
            for (slong j = 0; j < n; ++j) {
                auto entry = i < split ? fmpz_mat_entry(rays.matrix, i, j)
                                       : fmpz_mat_entry(support.matrix, j, i-split);
                fmpz_set(entry, &rows[i][j].v);
            }
        fmpz_mat_mul(values.matrix, rays.matrix, support.matrix);
        std::string out;
        for (slong i = 0; i < split; ++i) {
            for (slong j = 0; j < count-split; ++j) {
                if (j) out += ' ';
                char* text = fmpz_get_str(nullptr, 10, fmpz_mat_entry(values.matrix, i, j));
                out += text;
                flint_free(text);
            }
            out += '\n';
        }
        return out;
    }
    if (operation == "lattice_dimension" || operation == "lattice_pivots") {
        if (!count) throw std::invalid_argument("Expected nonempty Laurent support.");
        IntegerMatrix differences(count-1, n);
        for (slong i = 1; i < count; ++i)
            for (slong j = 0; j < n; ++j)
                fmpz_sub(fmpz_mat_entry(differences.matrix, i-1, j), &rows[i][j].v, &rows[0][j].v);
        if (operation == "lattice_dimension")
            return std::to_string(fmpz_mat_rank(differences.matrix)) + "\n";
        // The leading columns of the rational row space are unique. The
        // common denominator of this exact RREF does not affect its pivots.
        IntegerMatrix echelon(count-1, n);
        Z denominator;
        const slong rank = fmpz_mat_rref(echelon.matrix, &denominator.v, differences.matrix);
        std::string pivots;
        slong column = 0;
        for (slong row = 0; row < rank; ++row) {
            while (column < n && fmpz_is_zero(fmpz_mat_entry(echelon.matrix, row, column))) ++column;
            if (column == n) throw std::logic_error("Invalid lattice row echelon form.");
            pivots += std::to_string(column++) + "\n";
        }
        return pivots;
    }
    if (count < n) throw std::invalid_argument("Expected a square matrix of rays.");
    IntegerMatrix rays(n, n), inverse(n, n);
    for (slong i = 0; i < n; ++i)
        for (slong j = 0; j < n; ++j)
            fmpz_set(fmpz_mat_entry(rays.matrix, i, j), &rows[i][j].v);
    Z determinant, denominator;
    fmpz_mat_det(&determinant.v, rays.matrix);
    if (!fmpz_is_pm1(&determinant.v)) throw std::invalid_argument("Expected a unimodular matrix of rays.");
    if (!fmpz_mat_inv(inverse.matrix, &denominator.v, rays.matrix))
        throw std::logic_error("Unimodular ray matrix has no inverse.");
    fmpz_mat_scalar_divexact_fmpz(inverse.matrix, inverse.matrix, &denominator.v);
    IntegerMatrix local(n, count-n), characters(n, count-n);
    for (slong k = n; k < count; ++k)
        for (slong j = 0; j < n; ++j)
            fmpz_set(fmpz_mat_entry(local.matrix, j, k-n), &rows[k][j].v);
    fmpz_mat_mul(characters.matrix, inverse.matrix, local.matrix);
    if (operation == "lattice_support" && count > n) {
        // Coordinatewise translation uniquely removes the Laurent monomial
        // ambiguity while preserving all differences and term coefficients.
        for (slong j = 0; j < n; ++j) {
            Z minimum;
            fmpz_set(&minimum.v, fmpz_mat_entry(characters.matrix, j, 0));
            for (slong k = 1; k < count-n; ++k) {
                const auto entry = fmpz_mat_entry(characters.matrix, j, k);
                if (fmpz_cmp(entry, &minimum.v) < 0) fmpz_set(&minimum.v, entry);
            }
            for (slong k = 0; k < count-n; ++k) {
                auto entry = fmpz_mat_entry(characters.matrix, j, k);
                fmpz_sub(entry, entry, &minimum.v);
            }
        }
    }
    std::string result;
    for (slong k = 0; k < count-n; ++k) {
        for (slong j = 0; j < n; ++j) {
            char* value = fmpz_get_str(nullptr, 10, fmpz_mat_entry(characters.matrix, j, k));
            if (j) result += ' ';
            result += value;
            flint_free(value);
        }
        result += '\n';
    }
    return result;
}

// Each side of a map has its own polynomial ring, even when the names coincide.
struct Variables {
    fmpq_mpoly_ctx_t ctx;
    Ring packed;
    int n;
    std::vector<std::string> names;
    std::vector<const char*> pointers;

    explicit Variables(const std::string& variables, ordering_t ordering = ORD_LEX) {
        std::istringstream stream(variables);
        for (std::string name; stream >> name;) names.push_back(name);
        for (const auto& name : names) pointers.push_back(name.c_str());
        n = static_cast<int>(names.size());
        if (n <= 0) throw Limit("Unsupported number of variables.");
        if (n <= MAX_VARIABLES) packed.init(n, 1);
        fmpq_mpoly_ctx_init(ctx, n, ordering);
    }
    explicit Variables(int n) : n(n) {
        packed.init(n, 1);
        fmpq_mpoly_ctx_init(ctx, n, ORD_LEX);
    }
    ~Variables() { fmpq_mpoly_ctx_clear(ctx); }
    Variables(const Variables&) = delete;
};

struct Polynomial {
    fmpq_mpoly_t p;
    const Variables& ring;

    explicit Polynomial(const Variables& r) : ring(r) { fmpq_mpoly_init(p, ring.ctx); }
    Polynomial(Polynomial&& other) noexcept : Polynomial(other.ring) {
        fmpq_mpoly_swap(p, other.p, ring.ctx);
    }
    Polynomial(const Polynomial&) = delete;
    ~Polynomial() { fmpq_mpoly_clear(p, ring.ctx); }
};

Polynomial read(std::istream& in, Variables& ring) {
    Polynomial f(ring);
    auto s = line(in);
    if (fmpq_mpoly_set_str_pretty(f.p, s.c_str(), ring.pointers.data(), ring.ctx))
        throw std::invalid_argument("Expected a polynomial over Q in the specified coordinates.");
    return f;
}

// FLINT stores a rational polynomial as a rational content times a primitive
// integer polynomial. Lexicographic order agrees with that of the packed ring.
Poly packed(const Polynomial& f) {
    Poly result;
    const auto& R = f.ring.packed;
    if (!R.n) throw Limit("Unsupported number of variables.");
    long powers[MAX_VARIABLES];
    for (slong i = 0; i < fmpq_mpoly_length(f.p, f.ring.ctx); ++i) {
        if (!fmpq_mpoly_term_exp_fits_si(f.p, i, f.ring.ctx)) throw Limit("Exponent out of range.");
        fmpq_mpoly_get_term_exp_si(powers, f.p, i, f.ring.ctx);
        result.m.resize(result.m.size() + R.stride);
        R.pack(result.m.data() + result.m.size() - R.stride, powers);
        result.c.emplace_back();
        fmpz_mpoly_get_term_coeff_fmpz(&result.c.back().v, f.p->zpoly, i, f.ring.ctx->zctx);
    }
    return result;
}

void emit(std::string& text, const Polynomial& f) {
    // The affine operations work in FLINT's sparse rings, including exponents
    // and variable counts beyond the Groebner kernel's packed monomials.
    const auto& ring = f.ring;
    if (ring.packed.n && fmpq_mpoly_total_degree_fits_si(f.p, ring.ctx)
            && fmpq_mpoly_total_degree_si(f.p, ring.ctx) <= static_cast<slong>(MAX_DEGREE)) {
        const auto c = f.p->content;
        write(ring.packed, text, packed(f), fmpq_numref(c), fmpq_denref(c));
        return;
    }
    std::vector<Z> exponents(ring.n);
    std::vector<fmpz*> positions;
    for (auto& e : exponents) positions.push_back(&e.v);
    fmpq_t coefficient;
    fmpq_init(coefficient);
    for (slong i = 0; i < f.p->zpoly->length; ++i) {
        if (i) text += ';';
        fmpq_mpoly_get_term_coeff_fmpq(coefficient, f.p, i, ring.ctx);
        char* value = fmpq_get_str(nullptr, 10, coefficient);
        text += value;
        flint_free(value);
        fmpq_mpoly_get_term_exp_fmpz(positions.data(), f.p, i, ring.ctx);
        for (auto e : positions) {
            text += ' ';
            char* number = fmpz_get_str(nullptr, 10, e);
            text += number;
            flint_free(number);
        }
    }
    fmpq_clear(coefficient);
    text += '\n';
}

// Saved record grammar, matching polynomial.parse_expression. No evaluation
// of identifiers, calls or general rational functions is permitted.
struct InputBudget {
    slong degree, terms, bits, products;
};

class RecordParser {
    const std::string input;
    Variables& ring;
    const InputBudget* budget;
    size_t at = 0;

    char peek() {
        while (at < input.size() && std::isspace(static_cast<unsigned char>(input[at]))) ++at;
        return at < input.size() ? input[at] : '\0';
    }

    bool power() {
        return peek() == '^' || input.compare(at, 2, "**") == 0;
    }

    Z integer(bool exponent = false) {
        const size_t begin = at;
        while (at < input.size() && input[at] >= '0' && input[at] <= '9') ++at;
        if (begin == at) throw std::invalid_argument("Expected a nonnegative integer.");
        if (budget && at-begin > (exponent ? 3U : 256U))
            throw Limit(exponent ? "Input budget: Quick parsing accepts exponents up to " +
                        std::to_string(budget->degree) + "." : "Input budget: Integer literal is too long.");
        Z n;
        fmpz_set_str(&n.v, input.substr(begin, at-begin).c_str(), 10);
        if (budget && exponent && fmpz_cmp_si(&n.v, budget->degree) > 0)
            throw Limit("Input budget: Quick parsing accepts exponents up to " + std::to_string(budget->degree) + ".");
        return n;
    }

    void checked(const Polynomial& f) const {
        const slong length = fmpq_mpoly_length(f.p, ring.ctx);
        bool fits = length <= budget->terms && fmpq_mpoly_total_degree_fits_si(f.p, ring.ctx)
                    && fmpq_mpoly_total_degree_si(f.p, ring.ctx) <= budget->degree;
        fmpq_t c;
        fmpq_init(c);
        for (slong i = 0; fits && i < length; ++i) {
            fmpq_mpoly_get_term_coeff_fmpq(c, f.p, i, ring.ctx);
            fits = std::max(fmpz_bits(fmpq_numref(c)), fmpz_bits(fmpq_denref(c))) <=
                   static_cast<ulong>(budget->bits);
        }
        fmpq_clear(c);
        if (!fits) throw Limit("Input budget: Expanded polynomial exceeds the degree, term, or coefficient limit.");
    }

    void multiply(Polynomial& a, const Polynomial& b) const {
        if (budget) {
            const slong length = fmpq_mpoly_length(a.p, ring.ctx);
            if (length && fmpq_mpoly_length(b.p, ring.ctx) > budget->products / length)
                throw Limit("Input budget: Polynomial product exceeds the expansion budget.");
        }
        fmpq_mpoly_mul(a.p, a.p, b.p, ring.ctx);
        if (budget) checked(a);
    }

    Polynomial expr(int binding, int depth) {
        if (budget && depth > 64) throw Limit("Input budget: Expression nesting is too deep.");
        if (depth > 1000) throw Limit("The chart expression is nested too deeply.");
        Polynomial result(ring);
        const char token = peek();
        if (token == '(' || token == '-' || token == '+') {
            ++at;
            auto argument = expr(token == '(' ? 0 : 25, depth+1);
            if (token == '(') {
                if (peek() != ')') throw std::invalid_argument("Missing closing parenthesis.");
                ++at;
            }
            fmpq_mpoly_swap(result.p, argument.p, ring.ctx);
            if (token == '-') fmpq_mpoly_neg(result.p, result.p, ring.ctx);
        } else if (token >= '0' && token <= '9') {
            const auto n = integer();
            fmpq_mpoly_set_fmpz(result.p, &n.v, ring.ctx);
        } else if (std::isalpha(static_cast<unsigned char>(token))) {
            const size_t begin = at++;
            while (at < input.size() && (std::isalnum(static_cast<unsigned char>(input[at])) || input[at] == '_')) ++at;
            const auto v = std::find(ring.names.begin(), ring.names.end(), input.substr(begin, at-begin));
            if (v == ring.names.end()) throw std::invalid_argument("Unknown chart variable.");
            fmpq_mpoly_gen(result.p, v-ring.names.begin(), ring.ctx);
        } else {
            throw std::invalid_argument("Expected a number, a chart variable or a parenthesis.");
        }

        for (;;) {
            const char op = peek();
            const int level = power() ? 30 : op == '*' || op == '/' ? 20 : op == '+' || op == '-' ? 10 : -1;
            if (level < binding) return result;
            ++at;
            if (level == 30) {
                if (op == '*') ++at;
                peek();
                const auto exponent = integer(true);
                if (power()) throw std::invalid_argument("Parenthesize chained powers explicitly.");
                if (budget) {
                    // Match the public expansion schedule: the product budget
                    // applies before each multiply, even when terms cancel.
                    Polynomial value(ring);
                    fmpq_mpoly_one(value.p, ring.ctx);
                    ulong n = fmpz_get_ui(&exponent.v);
                    while (n) {
                        if (n & 1) multiply(value, result);
                        n /= 2;
                        if (n) multiply(result, result);
                    }
                    fmpq_mpoly_swap(value.p, result.p, ring.ctx);
                } else if (!fmpq_mpoly_pow_fmpz(result.p, result.p, &exponent.v, ring.ctx))
                    throw Limit("Chart exponent exceeds FLINT's limits.");
                continue;
            }
            auto right = expr(level+1, depth+1);
            if (op == '+') fmpq_mpoly_add(result.p, result.p, right.p, ring.ctx);
            else if (op == '-') fmpq_mpoly_sub(result.p, result.p, right.p, ring.ctx);
            else if (op == '*') { multiply(result, right); continue; }
            else {
                if (fmpq_mpoly_is_zero(right.p, ring.ctx) || !fmpq_mpoly_is_fmpq(right.p, ring.ctx))
                    throw std::invalid_argument("Division needs a nonzero rational constant.");
                fmpq_t denominator;
                fmpq_init(denominator);
                fmpq_mpoly_get_fmpq(denominator, right.p, ring.ctx);
                fmpq_mpoly_scalar_div_fmpq(result.p, result.p, denominator, ring.ctx);
                fmpq_clear(denominator);
            }
            if (budget) checked(result);
        }
    }

public:
    RecordParser(std::string s, Variables& r, const InputBudget* b = nullptr)
        : input(std::move(s)), ring(r), budget(b) {}
    Polynomial run() {
        auto result = expr(0, 0);
        peek();
        if (at != input.size()) throw std::invalid_argument("Unexpected chart token, use * for multiplication.");
        return result;
    }
};

Polynomial compose(const Polynomial& f, const Variables& chart,
                   const std::vector<fmpq_mpoly_struct*>& images) {
    Polynomial result(chart);
    if (!fmpq_mpoly_compose_fmpq_mpoly(result.p, f.p, images.data(), f.ring.ctx, chart.ctx))
        throw Limit("The polynomial pullback exceeds FLINT's limits.");
    return result;
}

void reduce_mod(Polynomial& f, const Polynomial& modulus) {
    if (fmpq_mpoly_is_zero(modulus.p, modulus.ring.ctx)) return;
    Polynomial q(f.ring), rem(f.ring);
    fmpq_mpoly_divrem(q.p, rem.p, f.p, modulus.p, f.ring.ctx);
    fmpq_mpoly_swap(f.p, rem.p, f.ring.ctx);
}

Polynomial jacobian(const Variables& chart, const std::vector<Polynomial>& map,
                    const std::vector<int>& columns) {
    const auto n = static_cast<int>(columns.size());
    if (n != static_cast<int>(map.size())) throw std::invalid_argument("Expected a square Jacobian minor.");
    gr_ctx_t coefficients;
    gr_ctx_init_fmpq_mpoly(coefficients, chart.n, ORD_LEX);
    gr_mat_t matrix;
    gr_mat_init(matrix, n, n, coefficients);
    Polynomial det(chart);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            auto entry = static_cast<fmpq_mpoly_struct*>(gr_mat_entry_ptr(matrix, i, j, coefficients));
            fmpq_mpoly_derivative(entry, map[i].p, columns[j], chart.ctx);
        }
    const int status = gr_mat_det(det.p, matrix, coefficients);
    gr_mat_clear(matrix, coefficients);
    gr_ctx_clear(coefficients);
    if (status != GR_SUCCESS) throw Limit("The Jacobian determinant could not be computed.");
    return det;
}

Z order(const Polynomial& f, int variable) {
    if (fmpq_mpoly_is_zero(f.p, f.ring.ctx)) throw std::invalid_argument("The zero polynomial has no finite order.");
    Z minimum;
    std::vector<Z> powers(f.ring.n);
    std::vector<fmpz*> entries;
    for (auto& e : powers) entries.push_back(&e.v);
    for (slong i = 0; i < fmpq_mpoly_length(f.p, f.ring.ctx); ++i) {
        fmpq_mpoly_get_term_exp_fmpz(entries.data(), f.p, i, f.ring.ctx);
        if (!i || fmpz_cmp(&minimum.v, entries[variable]) > 0) minimum = powers[variable];
        if (fmpz_is_zero(&minimum.v)) break;
    }
    return minimum;
}

// On a free frame the coordinate valuation is the minimum exponent over
// the support. Dividing by all coordinate powers at once gives the
// residual polynomial and avoids one exact division per order.
Polynomial coordinate_residual(const Polynomial& f, const std::vector<int>& variables, std::string& orders) {
    const auto& ring = f.ring;
    Polynomial factor(ring), residual(ring);
    std::vector<Z> powers(ring.n);
    std::vector<fmpz*> entries;
    for (auto& e : powers) entries.push_back(&e.v);
    for (int j : variables) {
        powers[j] = order(f, j);
        char* value = fmpz_get_str(nullptr, 10, &powers[j].v);
        orders += std::string(value) + "\n";
        flint_free(value);
    }
    fmpq_mpoly_one(factor.p, ring.ctx);
    fmpq_mpoly_set_term_exp_fmpz(factor.p, 0, entries.data(), ring.ctx);
    if (!fmpq_mpoly_divides(residual.p, f.p, factor.p, ring.ctx))
        throw std::logic_error("Inconsistent coordinate orders.");
    return residual;
}

Polynomial pole(const Variables& chart, const std::vector<Polynomial>& fractions) {
    Polynomial product(chart), gcd(chart), a(chart), b(chart);
    fmpq_mpoly_one(product.p, chart.ctx);
    for (size_t i = 0; i < fractions.size(); i += 2) {
        if (fmpq_mpoly_is_zero(fractions[i + 1].p, chart.ctx))
            throw std::invalid_argument("A transition denominator vanished identically.");
        if (!fmpq_mpoly_gcd_cofactors(gcd.p, a.p, b.p, fractions[i].p, fractions[i + 1].p, chart.ctx))
            throw Limit("A transition denominator could not be reduced.");
        fmpq_mpoly_mul(product.p, product.p, b.p, chart.ctx);
    }
    // Remove repeated prime factors, in characteristic zero.
    fmpq_mpoly_set(gcd.p, product.p, chart.ctx);
    for (int j = 0; j < chart.n && !fmpq_mpoly_is_fmpq(gcd.p, chart.ctx); ++j) {
        fmpq_mpoly_derivative(a.p, product.p, j, chart.ctx);
        if (!fmpq_mpoly_gcd(b.p, gcd.p, a.p, chart.ctx)) throw Limit("A polynomial gcd failed.");
        fmpq_mpoly_swap(b.p, gcd.p, chart.ctx);
    }
    if (!fmpq_mpoly_divides(b.p, product.p, gcd.p, chart.ctx)) throw Limit("Inexact squarefree quotient.");
    fmpq_mpoly_make_monic(b.p, b.p, chart.ctx);
    return b;
}

// Eliminate all chart variables except u_k, ordered last among the chart
// variables. The eliminated ideal is generated by D(x)u_k - N(x).
void inverse_coordinate(std::string& answer, const Variables& base,
                        const std::vector<Polynomial>& map, int k) {
    const int n = base.n;
    Variables wide(2 * n);
    Engine graph;
    graph.R = wide.packed;
    std::vector<slong> permutation(n);
    for (int j = 0, next = 0; j < n; ++j) permutation[j] = j == k ? n - 1 : next++;
    std::vector<Poly> generators;
    for (int i = 0; i < n; ++i) {
        Polynomial lifted(wide), x(wide);
        fmpq_mpoly_compose_fmpq_mpoly_gen(lifted.p, map[i].p, permutation.data(), map[i].ring.ctx, wide.ctx);
        fmpq_mpoly_gen(x.p, n + i, wide.ctx);
        fmpq_mpoly_sub(lifted.p, lifted.p, x.p, wide.ctx);
        generators.push_back(packed(lifted));
    }
    graph.compute(generators, 0);
    for (const auto& row : graph.G) {
        const Poly& f = row.p;
        long powers[MAX_VARIABLES];
        graph.R.unpack(f.m.data(), powers);
        if (std::any_of(powers, powers + n - 1, [](long e) { return e != 0; })) continue;
        if (powers[n - 1] != 1) continue;
        Poly numerator, denominator;
        for (size_t t = 0; t < f.size(); ++t) {
            graph.R.unpack(f.m.data() + t * graph.R.stride, powers);
            Poly& part = powers[n - 1] ? denominator : numerator;
            part.m.resize(part.m.size() + base.packed.stride);
            base.packed.pack(part.m.data() + part.m.size() - base.packed.stride, powers + n);
            part.c.push_back(f.c[t]);
            if (!powers[n - 1]) fmpz_neg(&part.c.back().v, &part.c.back().v);
        }
        Z one;
        fmpz_one(&one.v);
        write(base.packed, answer, numerator, &one.v, &denominator.c.front().v);
        write(base.packed, answer, denominator, &one.v, &denominator.c.front().v);
        return;
    }
    throw std::invalid_argument("The chart map has no rational inverse.");
}

// For an irreducible factor over Q with r geometric components, its Newton
// polytope is r times their common Newton polytope (Galois conjugacy).
// Thus r divides every vertex coordinate. Cyclic lex extrema are vertices;
// affinely independent supports consist entirely of vertices.
bool primitive_vertices(const Polynomial& f) {
    const auto n = f.ring.n;
    const auto count = fmpq_mpoly_length(f.p, f.ring.ctx);
    std::vector<std::vector<Z>> terms(count, std::vector<Z>(n));
    std::vector<fmpz*> pointers(n);
    for (slong i = 0; i < count; ++i) {
        for (int j = 0; j < n; ++j) pointers[j] = &terms[i][j].v;
        fmpq_mpoly_get_term_exp_fmpz(pointers.data(), f.p, i, f.ring.ctx);
    }
    Z divisor;
    auto accumulate = [&](const std::vector<Z>& term) {
        for (const auto& e : term) fmpz_gcd(&divisor.v, &divisor.v, &e.v);
    };
    for (int j = 0; j < n; ++j) {
        auto less = [&](const auto& a, const auto& b) {
            for (int i = 0; i < n; ++i) {
                const int k = (i+j) % n;
                const int sign = fmpz_cmp(&a[k].v, &b[k].v);
                if (sign) return sign < 0;
            }
            return false;
        };
        auto extrema = std::minmax_element(terms.begin(), terms.end(), less);
        accumulate(*extrema.first);
        accumulate(*extrema.second);
        if (fmpz_is_one(&divisor.v)) return true;
    }
    if (count > n+1) return false;
    for (const auto& term : terms) accumulate(term);
    if (!fmpz_is_one(&divisor.v)) return false;
    IntegerMatrix differences(count-1, n);
    for (slong i = 1; i < count; ++i)
        for (int j = 0; j < n; ++j)
            fmpz_sub(fmpz_mat_entry(differences.matrix, i-1, j), &terms[i][j].v, &terms[0][j].v);
    return fmpz_mat_rank(differences.matrix) == count-1;
}

// Primitive rational factors and their orders. Use integral primitive
// representatives so archived branch labels retain their previous ordering.
std::string factorization(const Polynomial& f, bool geometric) {
    const auto ctx = f.ring.ctx;
    struct Factors {
        fmpq_mpoly_factor_t value;
        const fmpq_mpoly_ctx_struct* ctx;
        explicit Factors(const fmpq_mpoly_ctx_struct* c) : ctx(c) { fmpq_mpoly_factor_init(value, c); }
        ~Factors() { fmpq_mpoly_factor_clear(value, ctx); }
    } factors(ctx);
    if (!fmpq_mpoly_factor(factors.value, f.p, ctx) || !fmpq_mpoly_factor_make_integral(factors.value, ctx))
        throw Limit("Rational factorization did not finish.");
    Polynomial constant(f.ring);
    fmpq_t coefficient;
    fmpq_init(coefficient);
    fmpq_mpoly_factor_get_constant_fmpq(coefficient, factors.value, ctx);
    fmpq_mpoly_set_fmpq(constant.p, coefficient, ctx);
    fmpq_clear(coefficient);
    std::string result;
    emit(result, constant);
    for (slong i = 0; i < fmpq_mpoly_factor_length(factors.value, ctx); ++i) {
        Polynomial factor(f.ring);
        fmpq_mpoly_factor_get_base(factor.p, factors.value, i, ctx);
        result += std::to_string(fmpq_mpoly_factor_get_exp_si(factors.value, i, ctx));
        if (geometric) result += primitive_vertices(factor) ? " 1" : " 0";
        result += '\n';
        emit(result, factor);
    }
    return result;
}

// On D(a), substitution v=-b/a is represented by
// sum_j c_j (-b)^j a^(degree-j). Only the occupied powers are visited.
Polynomial restrict_linear(const Polynomial& f, slong v, const Polynomial& a, const Polynomial& b) {
    const auto& ring = f.ring;
    if (fmpq_mpoly_is_zero(a.p, ring.ctx)) throw std::invalid_argument("The linear coefficient is zero.");
    struct Univariate {
        fmpq_mpoly_univar_t p;
        const Variables& ring;
        explicit Univariate(const Variables& r) : ring(r) { fmpq_mpoly_univar_init(p, r.ctx); }
        ~Univariate() { fmpq_mpoly_univar_clear(p, ring.ctx); }
    } terms(ring);
    fmpq_mpoly_to_univar(terms.p, f.p, v, ring.ctx);
    Polynomial value(ring), negative(ring), factor(ring), term(ring), q(ring);
    fmpq_mpoly_neg(negative.p, b.p, ring.ctx);
    if (!terms.p->length || fmpz_is_zero(terms.p->exps)) {
        fmpq_mpoly_set(value.p, f.p, ring.ctx);
        return value;
    }
    Z gap;
    for (slong j = 0; j < terms.p->length; ++j) {
        fmpz_sub(&gap.v, terms.p->exps, terms.p->exps+j);
        if (!fmpq_mpoly_pow_fmpz(term.p, negative.p, terms.p->exps+j, ring.ctx)
                || !fmpq_mpoly_pow_fmpz(factor.p, a.p, &gap.v, ring.ctx))
            throw Limit("The restricted polynomial could not be represented.");
        fmpq_mpoly_mul(term.p, term.p, factor.p, ring.ctx);
        fmpq_mpoly_mul(term.p, term.p, terms.p->coeffs+j, ring.ctx);
        fmpq_mpoly_add(value.p, value.p, term.p, ring.ctx);
    }
    if (fmpq_mpoly_is_zero(value.p, ring.ctx) || fmpq_mpoly_is_fmpq(a.p, ring.ctx)) return value;
    // Remove the full a-adic order. Repeated squares avoid one division
    // per unit of multiplicity on sparse, large-degree restrictions.
    std::vector<Polynomial> powers;
    fmpq_mpoly_set(factor.p, a.p, ring.ctx);
    while (fmpq_mpoly_divides(q.p, value.p, factor.p, ring.ctx)) {
        fmpq_mpoly_swap(q.p, value.p, ring.ctx);
        powers.emplace_back(ring);
        fmpq_mpoly_swap(powers.back().p, factor.p, ring.ctx);
        fmpq_mpoly_mul(factor.p, powers.back().p, powers.back().p, ring.ctx);
    }
    for (auto it = powers.rbegin(); it != powers.rend(); ++it)
        if (fmpq_mpoly_divides(q.p, value.p, it->p, ring.ctx)) fmpq_mpoly_swap(q.p, value.p, ring.ctx);
    return value;
}

// FLINT's primitive Buchberger algorithm computes over Q by clearing
// content. Its exponent vectors have no packed-degree or arity restriction.
struct PolynomialVector {
    fmpz_mpoly_vec_t v;
    const fmpz_mpoly_ctx_struct* ctx;
    explicit PolynomialVector(const Variables& ring) : ctx(ring.ctx->zctx) {
        fmpz_mpoly_vec_init(v, 0, ctx);
    }
    ~PolynomialVector() { fmpz_mpoly_vec_clear(v, ctx); }
    PolynomialVector(const PolynomialVector&) = delete;
};

// For an initial monomial ideal, a minimum coordinate cover of its
// leading supports gives the least height of a prime containing it.
using Supports = std::vector<std::vector<int>>;
int cover_height(const Supports& supports, int n, int height, int best) {
    if (supports.empty()) return height;
    std::vector<bool> used(n);
    int bound = height;
    for (const auto& row : supports) {
        if (std::any_of(row.begin(), row.end(), [&](int j) { return used[j]; })) continue;
        for (int j : row) used[j] = true;
        if (++bound >= best) return best;
    }
    for (int j : supports.front()) {
        Supports rest;
        for (const auto& row : supports)
            if (!std::binary_search(row.begin(), row.end(), j)) rest.push_back(row);
        best = cover_height(rest, n, height+1, best);
        if (best == bound) break;
    }
    return best;
}

std::string wide_basis(std::istream& in, const std::string& operation) {
    const auto names = line(in), order = line(in);
    if (order != "lex" && order != "grevlex") throw std::invalid_argument("Unknown monomial order.");
    Variables ring(names, order == "lex" ? ORD_LEX : ORD_DEGREVLEX);
    std::string out;
    if (operation == "wide_division") {
        auto f = read(in, ring);
        std::vector<Polynomial> divisors, quotients;
        while (in.peek() != std::char_traits<char>::eof()) {
            divisors.push_back(read(in, ring));
            if (fmpq_mpoly_is_zero(divisors.back().p, ring.ctx))
                throw std::invalid_argument("Cannot divide by zero.");
            quotients.emplace_back(ring);
        }
        std::vector<fmpq_mpoly_struct*> d, q;
        for (size_t i = 0; i < divisors.size(); ++i) {
            d.push_back(divisors[i].p);
            q.push_back(quotients[i].p);
        }
        Polynomial remainder(ring);
        if (d.empty()) fmpq_mpoly_set(remainder.p, f.p, ring.ctx);
        else fmpq_mpoly_divrem_ideal(q.data(), remainder.p, f.p, d.data(), d.size(), ring.ctx);
        for (const auto& quotient : quotients) emit(out, quotient);
        emit(out, remainder);
    } else {
        PolynomialVector input(ring), basis(ring), reduced(ring);
        while (in.peek() != std::char_traits<char>::eof()) {
            const auto f = read(in, ring);
            if (!fmpq_mpoly_is_zero(f.p, ring.ctx))
                fmpz_mpoly_vec_append(input.v, f.p->zpoly, ring.ctx->zctx);
        }
        if (!input.v->length) return operation == "wide_dimension" ? std::to_string(ring.n) + "\n" : out;
        fmpz_mpoly_buchberger_naive(basis.v, input.v, ring.ctx->zctx);
        fmpz_mpoly_vec_autoreduction_groebner(reduced.v, basis.v, ring.ctx->zctx);
        Supports supports;
        std::vector<Z> powers(ring.n);
        std::vector<fmpz*> entries;
        for (auto& e : powers) entries.push_back(&e.v);
        for (slong i = 0; i < reduced.v->length; ++i) {
            Polynomial f(ring);
            auto p = fmpz_mpoly_vec_entry(reduced.v, i);
            if (fmpz_mpoly_is_zero(p, ring.ctx->zctx)) continue;
            if (operation == "wide_dimension") {
                fmpz_mpoly_get_term_exp_fmpz(entries.data(), p, 0, ring.ctx->zctx);
                std::vector<int> row;
                for (int j = 0; j < ring.n; ++j) if (!fmpz_is_zero(entries[j])) row.push_back(j);
                if (row.empty()) return "-1\n";
                supports.push_back(std::move(row));
                continue;
            }
            fmpz_mpoly_set(f.p->zpoly, p, ring.ctx->zctx);
            fmpq_one(f.p->content);
            fmpq_mpoly_make_monic(f.p, f.p, ring.ctx);
            emit(out, f);
        }
        if (operation == "wide_dimension") {
            std::sort(supports.begin(), supports.end(), [](const auto& a, const auto& b) {
                return a.size() == b.size() ? a < b : a.size() < b.size();
            });
            return std::to_string(ring.n - cover_height(supports, ring.n, 0, ring.n)) + "\n";
        }
    }
    return out;
}

std::string affine(std::istream& in) {
    const auto operation = line(in);
    if (operation == "bm_residual" || operation == "bm_requirements" || operation == "bm_values")
        return centre_weights(in, operation);
    if (operation == "fan_cover") return fan_cover(in);
    if (operation == "wide_basis" || operation == "wide_division" || operation == "wide_dimension")
        return wide_basis(in, operation);
    if (operation == "lattice_valuations" || operation == "lattice_characters" || operation == "lattice_support" ||
        operation == "lattice_dimension" || operation == "lattice_pivots")
        return lattice(in, operation);
    Variables chart(line(in));
    if (operation == "factor" || operation == "geometric_factors")
        return factorization(read(in, chart), operation == "geometric_factors");
    if (operation == "decode" || operation == "parse_input") {
        std::string result;
        emit(result, RecordParser(line(in), chart).run());
        return result;
    }
    if (operation == "parse_quick") {
        InputBudget budget;
        std::istringstream bounds(line(in));
        if (!(bounds >> budget.degree >> budget.terms >> budget.bits >> budget.products) ||
            std::min({budget.degree, budget.terms, budget.bits, budget.products}) < 0)
            throw std::invalid_argument("Invalid polynomial input budget.");
        std::string result;
        emit(result, RecordParser(line(in), chart, &budget).run());
        return result;
    }
    if (operation == "frame_empty" || operation == "frame_basis") {
        std::vector<Poly> equations, units;
        const int count = std::stoi(line(in));
        for (int i = 0; i < count; ++i) equations.push_back(packed(read(in, chart)));
        while (in.peek() != std::char_traits<char>::eof()) units.push_back(packed(read(in, chart)));
        if (!chart.packed.n) throw Limit("Unsupported number of variables.");
        return frame_locus(chart.packed, equations, units, operation == "frame_basis");
    }
    if (operation == "function") {
        std::string result;
        emit(result, read(in, chart));
        return result;
    }
    if (operation == "coordinate_residual") {
        std::vector<int> variables;
        std::istringstream indices(line(in));
        for (int j; indices >> j;) {
            if (j < 0 || j >= chart.n) throw std::invalid_argument("Invalid divisor coordinate.");
            variables.push_back(j);
        }
        std::string result;
        auto residual = coordinate_residual(read(in, chart), variables, result);
        emit(result, residual);
        return result;
    }
    if (operation == "affine_jet") {
        const long order = std::stol(line(in));
        if (order < 1) throw std::invalid_argument("Expected a positive jet order.");
        std::vector<slong> columns;
        std::istringstream indices(line(in));
        for (slong j; indices >> j;) {
            if (j < 0 || j >= chart.n) throw std::invalid_argument("Invalid jet coordinate.");
            columns.push_back(j);
        }
        std::vector<Polynomial> layer;
        layer.push_back(read(in, chart));
        std::string result;
        emit(result, layer.front());
        std::unordered_set<std::string> seen{result};
        for (long k = 1; k < order && !layer.empty(); ++k) {
            std::vector<Polynomial> next;
            for (const auto& f : layer) for (auto j : columns) {
                Polynomial d(chart);
                fmpq_mpoly_derivative(d.p, f.p, j, chart.ctx);
                if (fmpq_mpoly_is_zero(d.p, chart.ctx)) continue;
                std::string row;
                emit(row, d);
                if (seen.insert(row).second) {
                    result += row;
                    next.push_back(std::move(d));
                }
            }
            layer = std::move(next);
        }
        return result;
    }
    if (operation == "linear" || operation == "restrict" || operation == "divide_coordinate") {
        const slong v = std::stol(line(in));
        if (v < 0 || v >= chart.n) throw std::invalid_argument("Invalid coordinate.");
        const auto f = read(in, chart);
        std::string result;
        if (operation == "linear") {
            Z degree;
            fmpq_mpoly_degree_fmpz(&degree.v, f.p, v, chart.ctx);
            if (!fmpz_is_one(&degree.v)) return "none\n";
            Polynomial coeff(chart);
            for (ulong e : {1UL, 0UL}) {
                fmpq_mpoly_get_coeff_vars_ui(coeff.p, f.p, &v, &e, 1, chart.ctx);
                emit(result, coeff);
            }
        } else if (operation == "restrict") {
            const auto a = read(in, chart), b = read(in, chart);
            emit(result, restrict_linear(f, v, a, b));
        } else {
            Polynomial x(chart), q(chart);
            fmpq_mpoly_gen(x.p, v, chart.ctx);
            if (!fmpq_mpoly_divides(q.p, f.p, x.p, chart.ctx)) return "none\n";
            emit(result, q);
        }
        return result;
    }
    if (operation == "field" || operation == "derive" || operation == "minor") {
        std::vector<int> columns;
        std::istringstream indices(line(in));
        for (int j; indices >> j;) {
            if (j < 0 || j >= chart.n) throw std::invalid_argument("Invalid field coordinate.");
            columns.push_back(j);
        }
        std::string result;
        if (operation == "derive") {
            auto f = read(in, chart);
            Polynomial sum(chart), term(chart);
            for (int j : columns) {
                const auto coefficient = read(in, chart);
                fmpq_mpoly_derivative(term.p, f.p, j, chart.ctx);
                fmpq_mpoly_mul(term.p, term.p, coefficient.p, chart.ctx);
                fmpq_mpoly_add(sum.p, sum.p, term.p, chart.ctx);
            }
            emit(result, sum);
        } else {
            std::vector<Polynomial> rows;
            while (in.peek() != std::char_traits<char>::eof()) rows.push_back(read(in, chart));
            if (operation == "minor") {
                emit(result, jacobian(chart, rows, columns));
            } else {
                for (size_t j = 0; j < columns.size(); ++j) {
                    auto kept = columns;
                    kept.erase(kept.begin() + j);
                    auto value = jacobian(chart, rows, kept);
                    if ((rows.size() + j) & 1) fmpq_mpoly_neg(value.p, value.p, chart.ctx);
                    emit(result, value);
                }
            }
        }
        return result;
    }
    Variables base(line(in));
    std::vector<Polynomial> map;
    std::vector<fmpq_mpoly_struct*> images;
    for (int i = 0; i < base.n; ++i) map.push_back(read(in, chart));
    for (auto& image : map) images.push_back(image.p);
    std::string result;
    if (operation == "inverse") {
        if (base.n != chart.n) throw std::invalid_argument("Expected a square chart map.");
        bool identity = true;
        for (int i = 0; i < chart.n; ++i)
            identity = identity && fmpq_mpoly_is_gen(map[i].p, i, chart.ctx);
        if (identity) {
            Polynomial one(base), x(base);
            fmpq_mpoly_one(one.p, base.ctx);
            for (int i = 0; i < base.n; ++i) {
                fmpq_mpoly_gen(x.p, i, base.ctx);
                emit(result, x);
                emit(result, one);
            }
        } else {
            for (int i = 0; i < chart.n; ++i) inverse_coordinate(result, base, map, i);
        }
        return result;
    }
    if (operation == "pole") {
        std::vector<Polynomial> fractions;
        while (in.peek() != std::char_traits<char>::eof()) fractions.push_back(compose(read(in, base), chart, images));
        if (fractions.size() != 2 * map.size()) throw std::invalid_argument("Expected one fraction per coordinate.");
        emit(result, pole(chart, fractions));
        return result;
    }
    Polynomial modulus = read(in, chart);  // zero over Q, m(theta) on a finite cover
    if (operation == "pullbacks") {
        while (in.peek() != std::char_traits<char>::eof()) {
            Polynomial image = compose(read(in, base), chart, images);
            reduce_mod(image, modulus);
            emit(result, image);
        }
        return result;
    }
    if (operation != "chart" || base.n != chart.n)
        throw std::invalid_argument("Expected a square affine chart.");
    std::vector<int> columns;
    for (int j = 0; j < chart.n; ++j) columns.push_back(j);
    Polynomial strict = compose(read(in, base), chart, images), jac = jacobian(chart, map, columns);
    reduce_mod(strict, modulus);
    reduce_mod(jac, modulus);
    if (fmpq_mpoly_is_zero(strict.p, chart.ctx) || fmpq_mpoly_is_zero(jac.p, chart.ctx))
        throw std::invalid_argument("Expected a dominant chart and a nonzero pullback.");
    std::istringstream divisors(line(in));
    for (int v; divisors >> v;) {
        if (v < 0 || v >= chart.n) throw std::invalid_argument("Invalid divisor coordinate.");
        const auto N = order(strict, v);
        auto nu = order(jac, v);
        fmpz_add_ui(&nu.v, &nu.v, 1);
        char* n_text = fmpz_get_str(nullptr, 10, &N.v);
        char* nu_text = fmpz_get_str(nullptr, 10, &nu.v);
        result += std::string(n_text) + " " + nu_text + "\n";
        flint_free(n_text);
        flint_free(nu_text);
        if (!fmpz_is_zero(&N.v)) {
            Polynomial factor(chart), quotient(chart);
            fmpq_mpoly_gen(factor.p, v, chart.ctx);
            if (!fmpq_mpoly_pow_fmpz(factor.p, factor.p, &N.v, chart.ctx)
                || !fmpq_mpoly_divides(quotient.p, strict.p, factor.p, chart.ctx)) throw Limit("Inexact strict transform.");
            fmpq_mpoly_swap(strict.p, quotient.p, chart.ctx);
        }
    }
    emit(result, strict);
    return result;
}
}  // namespace
}  // namespace atlas

extern "C" int atlas_lattice_abi() noexcept { return 1; }
extern "C" int atlas_valuations_abi() noexcept { return 1; }
extern "C" int atlas_geometric_factors_abi() noexcept { return 1; }
extern "C" int atlas_wide_abi() noexcept { return 1; }
extern "C" int atlas_wide_dimension_abi() noexcept { return 1; }
extern "C" int atlas_jet_abi() noexcept { return 1; }
extern "C" int atlas_residual_abi() noexcept { return 1; }

extern "C" char* atlas_affine(const char* input, size_t length) noexcept {
    try {
        std::istringstream stream(std::string(input, length));
        return atlas::copy_text(atlas::affine(stream));
    } catch (const std::exception& e) {
        atlas::set_error(e.what());
    } catch (...) {
        atlas::set_error("Unknown affine-map failure.");
    }
    return nullptr;
}

extern "C" int atlas_input_abi() noexcept { return 1; }
