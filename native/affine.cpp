// Pullbacks, Jacobians and transitions of affine charts over Q. See KERNEL.md §9.5.
#include "groebner.hpp"
#include "polynomial_factors.hpp"
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
std::string monomial_quotient(std::istream&, const std::string&);
namespace local { std::string operations(std::istream&, const std::string&); }
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
    explicit Variables(int n, ordering_t ordering = ORD_LEX) : n(n) {
        packed.init(n, 1);
        fmpq_mpoly_ctx_init(ctx, n, ordering);
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

std::string equation_text(const Polynomial& equation, std::vector<const char*>& names) {
    char* raw = fmpq_mpoly_get_str_pretty(equation.p, names.data(), equation.ring.ctx);
    std::string text(raw);
    flint_free(raw);
    text.erase(std::remove(text.begin(), text.end(), ' '), text.end());
    return text;
}

// The public order is descending total degree, then lexicographic order.
std::string chart_equation_text(const Polynomial& strict, const Variables& graded,
                                const std::vector<slong>& axes, std::vector<const char*>& names) {
    Polynomial equation(graded);
    fmpq_mpoly_compose_fmpq_mpoly_gen(equation.p, strict.p, axes.data(), strict.ring.ctx, graded.ctx);
    return equation_text(equation, names);
}

// The squarefree supports generate the radical of the owned monomial ideal.
// A term in its square is a multiple of a pair of supports A,B: every
// exponent on A|B is positive, and every exponent on A&B is at least two.
// Its value and first derivatives therefore vanish everywhere on that locus.
Polynomial crossing_jet(const Polynomial& strict, const std::string& supports) {
    const auto& ring = strict.ring;
    if (ring.n >= 64) throw Limit("Too many axes for a monomial support mask.");
    std::istringstream input(supports);
    std::vector<uint64_t> masks;
    for (uint64_t m; input >> m;) {
        if (m >> ring.n) throw std::invalid_argument("Invalid owned monomial support.");
        if (std::none_of(masks.begin(), masks.end(), [m](uint64_t a){ return (a & m) == a; })) {
            masks.erase(std::remove_if(masks.begin(), masks.end(), [m](uint64_t a){ return (a & m) == m; }),
                        masks.end());
            masks.push_back(m);
        }
    }
    if (!input.eof() || masks.empty()) throw std::invalid_argument("Expected owned monomial supports.");
    std::vector<std::pair<uint64_t,uint64_t>> square;
    for (size_t i = 0; i < masks.size(); ++i)
        for (size_t j = i; j < masks.size(); ++j)
            square.emplace_back(masks[i] | masks[j], masks[i] & masks[j]);

    Polynomial jet(ring);
    std::vector<Z> powers(ring.n);
    std::vector<fmpz*> exponents;
    for (auto& e : powers) exponents.push_back(&e.v);
    Z coefficient;
    for (slong i = 0; i < strict.p->zpoly->length; ++i) {
        fmpq_mpoly_get_term_exp_fmpz(exponents.data(), strict.p, i, ring.ctx);
        uint64_t positive = 0, repeated = 0;
        for (int j = 0; j < ring.n; ++j) {
            if (!fmpz_is_zero(exponents[j])) {
                positive |= uint64_t(1) << j;
                if (!fmpz_is_one(exponents[j])) repeated |= uint64_t(1) << j;
            }
        }
        if (std::any_of(square.begin(), square.end(), [&](const auto& divisor) {
                return !(divisor.first & ~positive) && !(divisor.second & ~repeated);
            })) continue;
        fmpz_mpoly_get_term_coeff_fmpz(&coefficient.v, strict.p->zpoly, i, ring.ctx->zctx);
        fmpz_mpoly_push_term_fmpz_fmpz(jet.p->zpoly, &coefficient.v, exponents.data(), ring.ctx->zctx);
    }
    // Filtering preserves order and distinctness. Restore rational content
    // even if the remaining integer coefficients acquire a common factor.
    fmpq_set(jet.p->content, strict.p->content);
    fmpq_mpoly_reduce(jet.p, ring.ctx);
    return jet;
}

// For a regular cone R and a character a, the pullback exponent is R*a.
// Distinct characters stay distinct. On an exceptional ray r the minimum
// pairing is the order, and nu is sum(r) by the logarithmic volume form.
std::string fan_charts(std::istream& in, bool projections) {
    Variables base(line(in));
    Variables chart(line(in));
    const int n = base.n;
    if (n != chart.n || n > 8) throw std::invalid_argument("Expected fan charts of dimension at most eight.");
    const auto source = read(in, base);
    const slong terms = fmpq_mpoly_length(source.p, base.ctx);
    if (!terms) throw std::invalid_argument("Expected a nonzero fan equation.");
    Variables graded(n, ORD_DEGLEX);
    std::vector<slong> axes(n);
    for (int j = 0; j < n; ++j) axes[j] = j;
    std::vector<std::vector<Z>> linear;
    if (projections) {
        for (int i = 0; i < n; ++i) {
            std::istringstream row(line(in));
            std::string entry;
            linear.emplace_back(n);
            for (auto& c : linear.back()) {
                if (!(row >> entry) || fmpz_set_str(&c.v, entry.c_str(), 10))
                    throw std::invalid_argument("Expected an integral fan frame.");
            }
            if (row >> entry) throw std::invalid_argument("Wrong fan frame row length.");
        }
    }
    IntegerMatrix support(n, terms), rays(n, n), values(n, terms);
    std::vector<fmpz*> exponent(n);
    for (slong k = 0; k < terms; ++k) {
        for (int j = 0; j < n; ++j) exponent[j] = fmpz_mat_entry(support.matrix, j, k);
        fmpq_mpoly_get_term_exp_fmpz(exponent.data(), source.p, k, base.ctx);
    }
    std::string result;
    while (in.peek() != std::char_traits<char>::eof()) {
        std::vector<Z> nu(n);
        for (int i = 0; i < n; ++i) {
            std::istringstream row(line(in));
            std::string entry;
            for (int j = 0; j < n; ++j) {
                auto r = fmpz_mat_entry(rays.matrix, i, j);
                if (!(row >> entry) || fmpz_set_str(r, entry.c_str(), 10) || fmpz_sgn(r) < 0)
                    throw std::invalid_argument("Expected a nonnegative fan ray.");
                fmpz_add(&nu[i].v, &nu[i].v, r);
            }
            if (row >> entry) throw std::invalid_argument("Wrong fan ray length.");
        }
        Z det;
        fmpz_mat_det(&det.v, rays.matrix);
        if (!fmpz_is_pm1(&det.v)) throw std::invalid_argument("Expected a regular fan cone.");
        fmpz_mat_mul(values.matrix, rays.matrix, support.matrix);
        std::string orders;
        for (int i = 0; i < n; ++i) {
            if (fmpz_is_one(&nu[i].v)) continue;  // A standard axis is not exceptional.
            Z minimum;
            fmpz_set(&minimum.v, fmpz_mat_entry(values.matrix, i, 0));
            for (slong k = 1; k < terms; ++k) {
                const auto value = fmpz_mat_entry(values.matrix, i, k);
                if (fmpz_cmp(value, &minimum.v) < 0) fmpz_set(&minimum.v, value);
            }
            for (const fmpz* value : {&minimum.v, &nu[i].v}) {
                char* decimal = fmpz_get_str(nullptr, 10, value);
                orders += ' ';
                orders += decimal;
                flint_free(decimal);
            }
            for (slong k = 0; k < terms; ++k) {
                auto v = fmpz_mat_entry(values.matrix, i, k);
                fmpz_sub(v, v, &minimum.v);
            }
        }
        Polynomial strict(chart);
        for (slong k = 0; k < terms; ++k) {
            for (int j = 0; j < n; ++j) exponent[j] = fmpz_mat_entry(values.matrix, j, k);
            fmpz_mpoly_push_term_fmpz_fmpz(strict.p->zpoly, source.p->zpoly->coeffs + k,
                                           exponent.data(), chart.ctx->zctx);
        }
        fmpz_mpoly_sort_terms(strict.p->zpoly, chart.ctx->zctx);
        fmpq_set(strict.p->content, source.p->content);
        fmpq_mpoly_reduce(strict.p, chart.ctx);
        result += fmpq_mpoly_is_fmpq(strict.p, chart.ctx) ? '0' : '1';
        result += orders + "\n" + chart_equation_text(strict, graded, axes, chart.pointers) + "\n";
        emit(result, strict);
        emit(result, crossing_jet(strict, line(in)));
        // Projection i is sum_j L_ij u^(R[:,j]). Regularity makes the
        // columns distinct, so only nonzero frame entries need a term.
        for (const auto& row : linear) {
            Polynomial projection(graded);
            for (int j = 0; j < n; ++j) {
                if (fmpz_is_zero(&row[j].v)) continue;
                for (int k = 0; k < n; ++k) exponent[k] = fmpz_mat_entry(rays.matrix, k, j);
                fmpq_mpoly_push_term_fmpz_fmpz(projection.p, &row[j].v, exponent.data(), graded.ctx);
            }
            fmpq_mpoly_sort_terms(projection.p, graded.ctx);
            result += equation_text(projection, chart.pointers) + "\n";
        }
    }
    return result;
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

    void atom(Polynomial& result, char token) {
        if (token >= '0' && token <= '9') {
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
    }

    void apply_power(Polynomial& result) {
        at += input[at] == '*' ? 2 : 1;
        peek();
        const auto exponent = integer(true);
        if (power()) throw std::invalid_argument("Parenthesize chained powers explicitly.");
        if (budget) {
            // The product budget applies before each multiply, even when terms cancel.
            Polynomial value(ring);
            fmpq_mpoly_one(value.p, ring.ctx);
            ulong n = fmpz_get_ui(&exponent.v);
            while (n) {
                if (n & 1) multiply(value, result);
                n /= 2;
                if (n) multiply(result, result);
            }
            fmpq_mpoly_swap(value.p, result.p, ring.ctx);
        } else if (!fmpq_mpoly_pow_fmpz(result.p, result.p, &exponent.v, ring.ctx)) {
            throw Limit("Chart exponent exceeds FLINT's limits.");
        }
    }

    struct Operator {
        char symbol;
        int binding;
    };

    static constexpr int prefix_binding = 25;
    static constexpr int sum_binding = 10;

    void apply(Operator op, std::vector<Polynomial>& values) {
        auto& right = values.back();
        if (op.binding == prefix_binding) {
            if (op.symbol == '-') fmpq_mpoly_neg(right.p, right.p, ring.ctx);
            return;
        }
        auto& left = values[values.size()-2];
        switch (op.symbol) {
        case '+': fmpq_mpoly_add(left.p, left.p, right.p, ring.ctx); break;
        case '-': fmpq_mpoly_sub(left.p, left.p, right.p, ring.ctx); break;
        case '*': multiply(left, right); break;
        case '/': {
            if (fmpq_mpoly_is_zero(right.p, ring.ctx) || !fmpq_mpoly_is_fmpq(right.p, ring.ctx))
                throw std::invalid_argument("Division needs a nonzero rational constant.");
            fmpq_t denominator;
            fmpq_init(denominator);
            fmpq_mpoly_get_fmpq(denominator, right.p, ring.ctx);
            fmpq_mpoly_scalar_div_fmpq(left.p, left.p, denominator, ring.ctx);
            fmpq_clear(denominator);
            break;
        }
        }
        if (budget && op.symbol != '*') checked(left);
        values.pop_back();
    }

public:
    RecordParser(std::string s, Variables& r, const InputBudget* b = nullptr)
        : input(std::move(s)), ring(r), budget(b) {}
    Polynomial run() {
        std::vector<Operator> pending;
        std::vector<Polynomial> values;
        bool needs_atom = true;
        for (;;) {
            const char token = peek();
            if (needs_atom) {
                if (budget && pending.size() > 64)
                    throw Limit("Input budget: Expression nesting is too deep.");
                if (token == '(' || token == '-' || token == '+') {
                    pending.push_back({token, token == '(' ? 0 : prefix_binding});
                    ++at;
                    continue;
                }
                values.emplace_back(ring);
                atom(values.back(), token);
                needs_atom = false;
                continue;
            }
            if (power()) {
                apply_power(values.back());
                continue;
            }

            const int level = token == '*' || token == '/' ? 20 : token == '+' || token == '-' ? sum_binding : 0;
            while (!pending.empty() && pending.back().binding > level) {
                if (!budget && level == sum_binding && pending.back().binding == sum_binding+1) {
                    // Keep a large partial sum on the stack while smaller summands
                    // combine. Each deferred sum is over twice the size of its right
                    // operand, so a flat run needs only logarithmically many levels.
                    auto& right = values.back();
                    const auto& left = values[values.size()-2];
                    if (fmpq_mpoly_length(left.p, ring.ctx)/2 > fmpq_mpoly_length(right.p, ring.ctx)) {
                        // a - b + c becomes a + ((-b) + c) before reassociation.
                        if (pending.back().symbol == '-') {
                            fmpq_mpoly_neg(right.p, right.p, ring.ctx);
                            pending.back().symbol = '+';
                        }
                        break;
                    }
                }
                apply(pending.back(), values);
                pending.pop_back();
            }
            if (level) {
                // The right operand ends at the next operator of equal or lower precedence.
                pending.push_back({token, level+1});
                ++at;
                needs_atom = true;
            } else if (!pending.empty()) {
                if (token != ')') throw std::invalid_argument("Missing closing parenthesis.");
                pending.pop_back();
                ++at;
            } else {
                if (at != input.size()) throw std::invalid_argument("Unexpected chart token, use * for multiplication.");
                return std::move(values.back());
            }
        }
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

// In characteristic zero, gcd(f, all partials of f) removes exactly one
// copy of each prime factor of f. The quotient is its monic squarefree part.
Polynomial squarefree_part(const Polynomial& f) {
    const auto& chart = f.ring;
    Polynomial common(chart), derivative(chart), result(chart);
    if (fmpq_mpoly_is_zero(f.p, chart.ctx))
        throw std::invalid_argument("Expected a nonzero polynomial for its squarefree part.");
    fmpq_mpoly_set(common.p, f.p, chart.ctx);
    for (int j = 0; j < chart.n && !fmpq_mpoly_is_fmpq(common.p, chart.ctx); ++j) {
        fmpq_mpoly_derivative(derivative.p, f.p, j, chart.ctx);
        if (!fmpq_mpoly_gcd(result.p, common.p, derivative.p, chart.ctx)) throw Limit("A polynomial gcd failed.");
        fmpq_mpoly_swap(result.p, common.p, chart.ctx);
    }
    if (!fmpq_mpoly_divides(result.p, f.p, common.p, chart.ctx)) throw Limit("Inexact squarefree quotient.");
    fmpq_mpoly_make_monic(result.p, result.p, chart.ctx);
    return result;
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
    return squarefree_part(product);
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
std::string factorization(const Polynomial& f, const std::string& operation) {
    const auto ctx = f.ring.ctx;
    struct Factors {
        fmpq_mpoly_factor_t value;
        const fmpq_mpoly_ctx_struct* ctx;
        explicit Factors(const fmpq_mpoly_ctx_struct* c) : ctx(c) { fmpq_mpoly_factor_init(value, c); }
        ~Factors() { fmpq_mpoly_factor_clear(value, ctx); }
    } factors(ctx);
    std::string result;
    if (operation == "ordered_factorization") {
        result = ordered_factors(factors.value, f.p, ctx);
        result += '\n';
    } else if (!fmpq_mpoly_factor(factors.value, f.p, ctx) ||
               !fmpq_mpoly_factor_make_integral(factors.value, ctx)) {
        throw Limit("Rational factorization did not finish.");
    }
    Polynomial constant(f.ring);
    fmpq_t coefficient;
    fmpq_init(coefficient);
    fmpq_mpoly_factor_get_constant_fmpq(coefficient, factors.value, ctx);
    fmpq_mpoly_set_fmpq(constant.p, coefficient, ctx);
    fmpq_clear(coefficient);
    emit(result, constant);
    for (slong i = 0; i < fmpq_mpoly_factor_length(factors.value, ctx); ++i) {
        Polynomial factor(f.ring);
        fmpq_mpoly_factor_get_base(factor.p, factors.value, i, ctx);
        char* multiplicity = fmpz_get_str(nullptr, 10, factors.value->exp + i);
        result += multiplicity;
        flint_free(multiplicity);
        if (operation == "geometric_factors") result += primitive_vertices(factor) ? " 1" : " 0";
        result += '\n';
        emit(result, factor);
    }
    return result;
}

struct Univariate {
    fmpq_mpoly_univar_t p;
    const Variables& ring;
    Univariate(const Polynomial& f, slong v) : ring(f.ring) {
        fmpq_mpoly_univar_init(p, ring.ctx);
        fmpq_mpoly_to_univar(p, f.p, v, ring.ctx);
    }
    Univariate(const Univariate&) = delete;
    ~Univariate() { fmpq_mpoly_univar_clear(p, ring.ctx); }
};

// Res_v(a*v+b,f) = sum_j c_j (-b)^j a^(degree-j).
// Only the occupied powers of v are visited.
Polynomial linear_resultant(const Univariate& terms, const Polynomial& a, const Polynomial& b) {
    const auto& ring = terms.ring;
    if (fmpq_mpoly_is_zero(a.p, ring.ctx)) throw std::invalid_argument("The linear coefficient is zero.");
    Polynomial value(ring), negative(ring), factor(ring), term(ring);
    fmpq_mpoly_neg(negative.p, b.p, ring.ctx);
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
    return value;
}

Polynomial restrict_linear(const Polynomial& f, slong v, const Polynomial& a, const Polynomial& b) {
    const auto& ring = f.ring;
    Univariate terms(f, v);
    auto value = linear_resultant(terms, a, b);
    if (!terms.p->length || fmpz_is_zero(terms.p->exps)) return value;
    if (fmpq_mpoly_is_zero(value.p, ring.ctx) || fmpq_mpoly_is_fmpq(a.p, ring.ctx)) return value;
    // Remove the full a-adic order. Repeated squares avoid one division
    // per unit of multiplicity on sparse, large-degree restrictions.
    std::vector<Polynomial> powers;
    Polynomial factor(ring), q(ring);
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

Polynomial resultant(const Polynomial& f, const Polynomial& g, slong v) {
    const auto& ring = f.ring;
    Polynomial value(ring);
    if (fmpq_mpoly_is_zero(f.p, ring.ctx) || fmpq_mpoly_is_zero(g.p, ring.ctx)) return value;
    Univariate left(f, v), right(g, v);
    // Both inputs vanish at v=0.
    if (!fmpz_is_zero(left.p->exps + left.p->length - 1) &&
        !fmpz_is_zero(right.p->exps + right.p->length - 1)) return value;
    if (left.p->length == 1 || right.p->length == 1 ||
        fmpz_is_one(left.p->exps) || fmpz_is_one(right.p->exps)) {
        const bool swap = left.p->length != 1 &&
                          (right.p->length == 1 || !fmpz_is_one(left.p->exps));
        const auto& a = swap ? right : left;
        const auto& h = swap ? left : right;
        if (a.p->length == 1) {
            // Res_v(c*v^m,h) = c^deg(h) * h(0)^m.
            if (!fmpq_mpoly_pow_fmpz(value.p, a.p->coeffs, h.p->exps, ring.ctx))
                throw Limit("The resultant coefficient power could not be represented.");
            if (!fmpz_is_zero(a.p->exps)) {
                Polynomial constant_power(ring);
                if (!fmpq_mpoly_pow_fmpz(constant_power.p, h.p->coeffs + h.p->length - 1,
                                         a.p->exps, ring.ctx))
                    throw Limit("The resultant constant-term power could not be represented.");
                fmpq_mpoly_mul(value.p, value.p, constant_power.p, ring.ctx);
            }
        } else {
            Polynomial leading(ring), constant(ring);
            fmpq_mpoly_univar_get_term_coeff(leading.p, a.p, 0, ring.ctx);
            fmpq_mpoly_univar_get_term_coeff(constant.p, a.p, 1, ring.ctx);
            auto evaluated = linear_resultant(h, leading, constant);
            fmpq_mpoly_swap(value.p, evaluated.p, ring.ctx);
        }
        if (swap && fmpz_is_odd(a.p->exps) && fmpz_is_odd(h.p->exps))
            fmpq_mpoly_neg(value.p, value.p, ring.ctx);
        return value;
    }
    if (!fmpq_mpoly_resultant(value.p, f.p, g.p, v, ring.ctx))
        throw Limit("FLINT could not compute the polynomial resultant.");
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

// Supports joined through any common coordinate must be searched together.
// On disjoint coordinate sets minimum cover sizes add.
int component_cover_height(Supports& supports, int n) {
    if (supports.size() < 2) return static_cast<int>(supports.size());
    std::vector<int> parent(n, -1);
    auto root = [&](int v) {
        while (parent[v] >= 0) {
            if (parent[parent[v]] >= 0) parent[v] = parent[parent[v]];
            v = parent[v];
        }
        return v;
    };
    for (const auto& row : supports) {
        int component = root(row.front());
        for (int v : row) {
            int other = root(v);
            if (component == other) continue;
            if (parent[component] > parent[other]) std::swap(component, other);
            parent[component] += parent[other];
            parent[other] = component;
        }
    }

    // Preserve coordinate order while renumbering each component. The search
    // then allocates its scratch rows in that component's coordinates alone.
    std::vector<int> local(n), widths(n);
    for (int v = 0; v < n; ++v) local[v] = widths[root(v)]++;
    std::vector<Supports> groups(n);
    for (auto& row : supports) {
        const int component = root(row.front());
        for (int& v : row) v = local[v];
        groups[component].push_back(std::move(row));
    }
    int total = 0;
    for (int i = 0; i < n; ++i) {
        auto& group = groups[i];
        if (group.empty()) continue;
        std::sort(group.begin(), group.end(), [](const auto& a, const auto& b) {
            return a.size() == b.size() ? a < b : a.size() < b.size();
        });
        total += cover_height(group, widths[i], 0, widths[i]);
    }
    return total;
}

std::string ordered_polynomials(std::istream& in, const std::string& operation) {
    const auto names = line(in), order = line(in);
    if (order != "lex" && order != "grevlex") throw std::invalid_argument("Unknown monomial order.");
    Variables ring(names, order == "lex" ? ORD_LEX : ORD_DEGREVLEX);
    if (operation == "ordered_factorization") return factorization(read(in, ring), operation);
    std::string out;
    if (operation == "polynomial_gcd" || operation == "gcd_from_lcm") {
        const auto a = read(in, ring), b = read(in, ring);
        Polynomial result(ring);
        if (operation == "polynomial_gcd") {
            if (!fmpq_mpoly_gcd(result.p, a.p, b.p, ring.ctx))
                throw Limit("FLINT could not compute the polynomial gcd.");
        } else {
            const auto lcm = read(in, ring);
            Polynomial cofactor(ring);
            // (a) intersect (b) = (lcm). Dividing twice avoids expanding a*b.
            if (fmpq_mpoly_is_zero(lcm.p, ring.ctx) || fmpq_mpoly_is_zero(a.p, ring.ctx) ||
                !fmpq_mpoly_divides(cofactor.p, lcm.p, a.p, ring.ctx) ||
                !fmpq_mpoly_divides(result.p, b.p, cofactor.p, ring.ctx))
                throw std::logic_error("Inexact division while recovering a gcd from its principal intersection.");
        }
        if (!fmpq_mpoly_is_zero(result.p, ring.ctx))
            fmpq_mpoly_make_monic(result.p, result.p, ring.ctx);
        emit(out, result);
        return out;
    }
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
        const bool dimension = operation == "wide_dimension" || operation == "basis_dimension";
        if (!input.v->length) return dimension ? std::to_string(ring.n) + "\n" : out;
        // basis_dimension receives a Gröbner basis for the specified order.
        // Its leading supports determine dimension without computing a second basis.
        auto rows = input.v;
        if (operation != "basis_dimension") {
            fmpz_mpoly_buchberger_naive(basis.v, input.v, ring.ctx->zctx);
            fmpz_mpoly_vec_autoreduction_groebner(reduced.v, basis.v, ring.ctx->zctx);
            rows = reduced.v;
        }
        Supports supports;
        std::vector<Z> powers(ring.n);
        std::vector<fmpz*> entries;
        for (auto& e : powers) entries.push_back(&e.v);
        for (slong i = 0; i < rows->length; ++i) {
            Polynomial f(ring);
            auto p = fmpz_mpoly_vec_entry(rows, i);
            if (fmpz_mpoly_is_zero(p, ring.ctx->zctx)) continue;
            if (dimension) {
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
        if (dimension) {
            return std::to_string(ring.n - component_cover_height(supports, ring.n)) + "\n";
        }
    }
    return out;
}

std::string affine(std::istream& in) {
    const auto operation = line(in);
    if (operation.rfind("local_", 0) == 0) return local::operations(in, operation);
    if (operation.rfind("monomial_", 0) == 0) return monomial_quotient(in, operation);
    if (operation == "bm_residual" || operation == "bm_requirements" || operation == "bm_values")
        return centre_weights(in, operation);
    if (operation == "fan_cover") return fan_cover(in);
    if (operation == "fan_charts" || operation == "fan_charts_with_projection")
        return fan_charts(in, operation == "fan_charts_with_projection");
    if (operation == "ordered_factorization") return ordered_polynomials(in, operation);
    if (operation == "polynomial_gcd" || operation == "gcd_from_lcm" ||
        operation == "wide_basis" || operation == "wide_division" ||
        operation == "wide_dimension" || operation == "basis_dimension")
        return ordered_polynomials(in, operation);
    if (operation == "lattice_valuations" || operation == "lattice_characters" || operation == "lattice_support" ||
        operation == "lattice_dimension" || operation == "lattice_pivots")
        return lattice(in, operation);
    Variables chart(line(in));
    if (operation == "squarefree_part") {
        std::string result;
        emit(result, squarefree_part(read(in, chart)));
        return result;
    }
    if (operation == "factor_multiplicity") {
        const slong axes = std::stol(line(in));
        const auto value = read(in, chart), divisor = read(in, chart);
        std::vector<Polynomial> basis;
        while (in.peek() != std::char_traits<char>::eof()) basis.push_back(read(in, chart));
        std::vector<fmpq_mpoly_struct*> rows;
        for (auto& row : basis) rows.push_back(row.p);
        return factor_multiplicity(value.p, divisor.p, axes, rows, chart.ctx) + "\n";
    }
    if (operation == "resultant") {
        const slong v = std::stol(line(in));
        if (v < 0 || v >= chart.n) throw std::invalid_argument("Invalid resultant variable.");
        const auto f = read(in, chart), g = read(in, chart);
        std::string result;
        emit(result, resultant(f, g, v));
        return result;
    }
    if (operation == "factor" || operation == "geometric_factors")
        return factorization(read(in, chart), operation);
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
    if (in.peek() != std::char_traits<char>::eof()) emit(result, crossing_jet(strict, line(in)));
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
extern "C" int atlas_crossing_jet_abi() noexcept { return 1; }
extern "C" int atlas_fan_charts_abi() noexcept { return 1; }
extern "C" int atlas_fan_projection_abi() noexcept { return 1; }
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
