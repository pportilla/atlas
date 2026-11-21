// Exact conversion of marked algebraic constants to a primitive power basis.
#include <gmpxx.h>
#include "groebner.hpp"
#include "field_root_coordinates.hpp"
#include "common_field.hpp"
#include "rational_sparse_json.hpp"
#include <flint/fmpq.h>
#include <flint/fmpz_factor.h>
#include <flint/fmpz_poly.h>
#include <flint/fmpz_poly_factor.h>
#include <flint/qqbar.h>
#include <flint/fmpq_poly.h>
#include <flint/ulong_extras.h>
#include <sstream>
#include <map>
#include <set>

namespace atlas {
namespace {
struct Number {
    qqbar_t n;
    Number() { qqbar_init(n); }
    Number(const Number& other) { qqbar_init(n); qqbar_set(n,other.n); }
    Number(Number&& other) noexcept { qqbar_init(n); qqbar_swap(n,other.n); }
    Number& operator=(Number&& other) noexcept { qqbar_swap(n,other.n); return *this; }
    ~Number() { qqbar_clear(n); }
};

struct Fraction {
    fmpq_t value;
    Fraction() { fmpq_init(value); }
    ~Fraction() { fmpq_clear(value); }
    void read(std::istream& in) {
        std::string token;
        if (!(in >> token) || fmpq_set_str(value,token.c_str(),10) || fmpz_is_zero(fmpq_denref(value)))
            throw std::invalid_argument("Invalid rational coefficient.");
        fmpq_canonicalise(value);
    }
};

std::string text(const fmpq_t value) {
    char* raw = fmpq_get_str(nullptr, 10, value);
    std::string result(raw);
    flint_free(raw);
    return result;
}

struct Polynomial {
    fmpq_poly_t p;
    Polynomial() { fmpq_poly_init(p); }
    ~Polynomial() { fmpq_poly_clear(p); }
};

struct Integer {
    fmpz_t n;
    Integer() { fmpz_init(n); }
    ~Integer() { fmpz_clear(n); }
    std::string str() const {
        char* buffer = fmpz_get_str(nullptr,10,n);
        std::string result(buffer);
        flint_free(buffer);
        return result;
    }
};

struct IntegralPolynomial {
    fmpz_poly_t p;
    IntegralPolynomial() { fmpz_poly_init(p); }
    ~IntegralPolynomial() { fmpz_poly_clear(p); }
};

struct Factors {
    fmpz_factor_t data;
    Factors() { fmpz_factor_init(data); }
    ~Factors() { fmpz_factor_clear(data); }
};

struct Roots {
    slong degree;
    qqbar_ptr r;
    explicit Roots(slong n) : degree(n), r(_qqbar_vec_init(n)) {}
    ~Roots() { _qqbar_vec_clear(r,degree); }
};

mpz_class power(const mpz_class& base, const mpz_class& exponent) {
    Integer b, e, result;
    fmpz_set_mpz(b.n, base.get_mpz_t());
    fmpz_set_mpz(e.n, exponent.get_mpz_t());
    if (!fmpz_pow_fmpz(result.n, b.n, e.n))
        throw Limit("A displayed rational power exceeds the available precision.");
    mpz_class out;
    fmpz_get_mpz(out.get_mpz_t(), result.n);
    return out;
}

void prime_powers(std::map<mpz_class, mpq_class>& powers, const mpz_class& value, const mpq_class& e) {
    if (value == 1) return;
    Integer n;
    fmpz_set_mpz(n.n, value.get_mpz_t());
    Factors factors;
    fmpz_factor(factors.data, n.n);
    for (slong j = 0; j < factors.data->num; ++j) {
        mpz_class prime;
        fmpz_get_mpz(prime.get_mpz_t(), factors.data->p+j);
        powers[prime] += e*factors.data->exp[j];
    }
}

void normalize_power(mpq_class base, mpq_class exponent, mpq_class& scalar,
                     std::map<mpq_class, mpz_class>& radicals) {
    mpz_class whole;
    mpz_fdiv_q(whole.get_mpz_t(), exponent.get_num_mpz_t(), exponent.get_den_mpz_t());
    if (whole > 0)
        scalar *= mpq_class(power(base.get_num(), whole), power(base.get_den(), whole));
    else if (whole < 0)
        scalar *= mpq_class(power(base.get_den(), -whole), power(base.get_num(), -whole));
    exponent -= whole;
    // Integral powers, including the large integer part of a rational
    // exponent, need no prime factorization of their base.
    if (exponent == 0 || base == 1) return;

    std::map<mpz_class, mpq_class> powers;
    prime_powers(powers, base.get_num(), exponent);
    prime_powers(powers, base.get_den(), -exponent);
    std::map<mpz_class, std::map<mpz_class, mpz_class>> groups;
    for (const auto& [p, e] : powers) {
        mpz_class remainder;
        mpz_fdiv_qr(whole.get_mpz_t(), remainder.get_mpz_t(), e.get_num_mpz_t(), e.get_den_mpz_t());
        if (whole > 0) scalar *= power(p, whole);
        else if (whole < 0) scalar /= power(p, -whole);
        if (remainder != 0) groups[e.get_den()][p] = remainder;
    }
    for (const auto& [denominator, exponents] : groups) {
        // Preserve equal-denominator factors of an individual radical,
        // such as 12^(2/5), before combining equal powers across atoms.
        mpz_class common;
        for (const auto& [p, k] : exponents) mpz_gcd(common.get_mpz_t(), common.get_mpz_t(), k.get_mpz_t());
        auto it = radicals.try_emplace(mpq_class(common, denominator), 1).first;
        for (const auto& [p, k] : exponents) it->second *= power(p, k/common);
    }
}

void combine_radicals(mpq_class& scalar, std::map<mpq_class,mpz_class>& radicals) {
    std::vector<std::pair<mpz_class, mpq_class>> pending;
    for (const auto& [e, b] : radicals) pending.emplace_back(b,e);
    for (size_t i=0; i<pending.size(); ++i) {
        for (size_t j=0; j<i; ++j) {
            mpz_class g;
            mpz_gcd(g.get_mpz_t(), pending[i].first.get_mpz_t(), pending[j].first.get_mpz_t());
            if (g == 1) continue;
            pending[i].first /= g;
            pending[j].first /= g;
            mpq_class e = pending[i].second + pending[j].second;
            pending.emplace_back(g, e);
        }
    }
    radicals.clear();
    for (const auto& [b, e] : pending) if (b != 1) normalize_power(b, e, scalar, radicals);
}

mpq_class read_rational(std::istream& row) {
    Fraction f;
    f.read(row);
    mpq_class q;
    fmpq_get_mpq(q.get_mpq_t(),f.value);
    return q;
}

std::string display_terms(std::istream& input) {
    std::string line;
    if (!std::getline(input,line)) throw std::invalid_argument("Missing marked display operation.");
    if (line == "combine") {
        std::map<size_t,mpq_class> coefficients;
        while (std::getline(input,line)) {
            std::istringstream row(line);
            size_t i;
            if (!(row >> i)) throw std::invalid_argument("Missing marked display leaf.");
            auto c = read_rational(row);
            coefficients[i] += c * read_rational(row);
        }
        std::ostringstream out;
        for (const auto& [i,c] : coefficients) if (c != 0) out << i << ' ' << c << '\n';
        return out.str();
    }
    if (line != "monomial") throw std::invalid_argument("Unknown marked display operation.");
    mpq_class phase = 0, angle = 0, scalar = 1;
    std::map<mpq_class,mpz_class> radicals;
    std::ostringstream leaves;
    while (std::getline(input,line)) {
        std::istringstream row(line);
        std::string kind;
        size_t i;
        if (!(row >> kind >> i)) throw std::invalid_argument("Invalid display atom.");
        mpq_class n = read_rational(row);
        if (kind == "i") phase += n/2;
        else if (kind == "exp") angle += n*read_rational(row);
        else if (kind == "rational") {
            mpq_class base = read_rational(row), exponent = n*read_rational(row);
            if (base == 0) throw std::invalid_argument("Expected a nonzero radical base.");
            if (base < 0) { base = -base; phase += exponent; }
            normalize_power(base,exponent,scalar,radicals);
        } else if (kind == "power") {
            mpq_class exponent = n*read_rational(row);
            if (-1 < exponent && exponent < 1) leaves << "power " << i << ' ' << exponent << '\n';
            else leaves << "reciprocal " << i << ' ' << abs(exponent) << '\n';
        } else if (kind == "atom") leaves << "atom " << i << ' ' << n << '\n';
        else throw std::invalid_argument("Unknown display atom.");
    }
    combine_radicals(scalar,radicals);
    mpz_class periods;
    mpq_class half_angle = angle/2;
    mpz_fdiv_q(periods.get_mpz_t(),half_angle.get_num_mpz_t(),half_angle.get_den_mpz_t());
    angle -= 2*periods;
    if (angle.get_den() <= 2) phase += angle;
    else leaves << "exp " << (angle > 1 ? angle-2 : angle) << '\n';
    mpz_fdiv_q(periods.get_mpz_t(),phase.get_num_mpz_t(),phase.get_den_mpz_t());
    phase -= periods;
    if (mpz_odd_p(periods.get_mpz_t())) scalar = -scalar;
    if (phase == mpq_class(1,2)) leaves << "i\n";
    else if (phase != 0) {
        auto it = radicals.try_emplace(phase,1).first;
        it->second = -it->second;
    }
    for (const auto& [e,b] : radicals) leaves << "radical " << e << ' ' << b << '\n';
    return scalar.get_str() + "\n" + leaves.str();
}

std::string rational_product(std::istream& in) {
    mpq_class scalar(1);
    std::map<mpq_class, mpz_class> radicals;
    for (std::string line; std::getline(in, line);) {
        if (line.empty()) continue;
        std::istringstream row(line);
        Fraction b, e;
        b.read(row);
        e.read(row);
        if (fmpq_sgn(b.value) <= 0) throw std::invalid_argument("Expected a positive radical base.");
        mpq_class base, exponent;
        fmpq_get_mpq(base.get_mpq_t(), b.value);
        fmpq_get_mpq(exponent.get_mpq_t(), e.value);
        normalize_power(base, exponent, scalar, radicals);
    }

    combine_radicals(scalar, radicals);
    std::ostringstream result;
    result << scalar << '\n';
    for (const auto& [e, b] : radicals) if (b != 1) result << e << ' ' << b << '\n';
    return result.str();
}

// Decompose an integer using the trial bound of the archived radical
// printer. Fully factor any unresolved cofactor. A previously hidden square
// can then produce a shorter radical than the historical bounded search.
bool radical_part(const fmpz_t input, Integer& outside, Integer& inside) {
    Integer value, power, prime;
    fmpz_abs(value.n,input);
    fmpz_one(outside.n);
    fmpz_one(inside.n);
    if (fmpz_is_square(value.n)) {
        fmpz_sqrt(outside.n,value.n);
        return true;
    }
    static const std::vector<ulong> primes = [] {
        std::vector<ulong> list;
        n_primes_t iterator;
        n_primes_init(iterator);
        for (ulong p = n_primes_next(iterator); p <= (1UL<<15); p = n_primes_next(iterator))
            list.push_back(p);
        n_primes_clear(iterator);
        return list;
    }();
    auto include = [&](ulong exponent) {
        if (exponent > 1) {
            fmpz_pow_ui(power.n,prime.n,exponent/2);
            fmpz_mul(outside.n,outside.n,power.n);
        }
        if (exponent % 2) fmpz_mul(inside.n,inside.n,prime.n);
    };
    for (ulong p : primes) {
        if (fmpz_cmp_ui(value.n,p*p) < 0) break;
        if (fmpz_divisible_ui(value.n,p)) {
            fmpz_set_ui(prime.n,p);
            include(fmpz_remove(value.n,value.n,prime.n));
        }
    }
    if (fmpz_is_one(value.n)) return true;
    if (fmpz_is_square(value.n)) {
        fmpz_sqrt(power.n,value.n);
        fmpz_mul(outside.n,outside.n,power.n);
        return true;
    }
    if (fmpz_is_prime(value.n)) {
        fmpz_mul(inside.n,inside.n,value.n);
        return true;
    }
    const int exponent = fmpz_is_perfect_power(prime.n,value.n);
    if (exponent && fmpz_is_prime(prime.n)) {
        include(exponent);
        return true;
    }
    Factors full;
    fmpz_factor(full.data,value.n);
    bool same_spelling = true;
    for (slong j = 0; j < full.data->num; ++j) {
        fmpz_set(prime.n,full.data->p+j);
        include(full.data->exp[j]);
        if (full.data->exp[j] >= 2) same_spelling = false;
    }
    return same_spelling;
}

std::string rational_radical(std::istream& in) {
    Fraction value;
    value.read(in);
    Integer num_out, num_in, den_out, den_in, radicand;
    const bool numerator_same = radical_part(fmpq_numref(value.value),num_out,num_in);
    const bool denominator_same = radical_part(fmpq_denref(value.value),den_out,den_in);
    // sqrt(a/b) = sqrt(a)*sqrt(b)/b. The input is reduced, so its
    // numerator and denominator have no common nontrivial squarefree part.
    fmpz_mul(radicand.n,num_in.n,den_in.n);
    fmpz_mul(num_out.n,num_out.n,den_out.n);
    if (fmpq_sgn(value.value) < 0) fmpz_neg(radicand.n,radicand.n);
    Integer denominator;
    fmpz_set(denominator.n,fmpq_denref(value.value));
    std::string sum;
    in >> std::ws;
    if (in.peek() != std::char_traits<char>::eof()) {
        Fraction scale, factor;
        scale.read(in);
        fmpq_set_fmpz_frac(factor.value, num_out.n, denominator.n);
        fmpq_mul(factor.value, factor.value, scale.value);
        fmpz_set(num_out.n, fmpq_numref(factor.value));
        fmpz_set(denominator.n, fmpq_denref(factor.value));
        in >> std::ws;
        if (in.peek() != std::char_traits<char>::eof()) {
            Fraction offset;
            offset.read(in);
            fmpq_add(offset.value, offset.value, factor.value);
            sum = " " + text(offset.value);
        }
    }
    return num_out.str() + " " + denominator.str() + " " + radicand.str() +
           (numerator_same && denominator_same ? " 1" : " 0") + sum + "\n";
}

void read_polynomial(std::istream& in, Polynomial& p) {
    Fraction q;
    slong degree;
    if (!(in >> degree) || degree <= 0) throw std::invalid_argument("Invalid defining polynomial degree.");
    for (slong j = 0; j <= degree; ++j) {
        q.read(in);
        fmpq_poly_set_coeff_fmpq(p.p,j,q.value);
    }
    if (fmpq_poly_degree(p.p) != degree) throw std::invalid_argument("Invalid defining polynomial.");
}

// Preserve the positive scaling used by historical CRootOf records. For
// three or more terms it is the largest d with d^(n-j) dividing a_j for
// every nonleading coefficient. Its p-adic exponent is the minimum of
// floor(v_p(a_j)/(n-j)). The binomial convention only extracts an exact
// n-th root of the absolute constant, even when smaller divisors exist.
std::string root_normalization(std::istream& in) {
    Polynomial p;
    read_polynomial(in,p);
    IntegralPolynomial z;
    fmpq_poly_get_numerator(z.p,p.p);
    fmpz_poly_primitive_part(z.p,z.p);
    if (fmpz_sgn(fmpz_poly_lead(z.p)) < 0) fmpz_poly_neg(z.p,z.p);
    const slong n = fmpz_poly_degree(z.p);
    auto coeff = [&](slong j) { return fmpz_poly_get_coeff_ptr(z.p,j); };
    Integer scale, divisor, temp;
    fmpz_one(scale.n);
    if (fmpz_cmpabs(coeff(0),coeff(n)) > 0) {
        slong count = 0;
        for (slong j = 0; j < n; ++j) {
            if (!fmpz_is_zero(coeff(j))) {
                ++count;
                fmpz_gcd(divisor.n,divisor.n,coeff(j));
            }
        }
        if (count == 1) {
            if (!fmpz_root(scale.n,divisor.n,n)) fmpz_one(scale.n);
        } else {
            Factors factors;
            fmpz_factor(factors.data,divisor.n);
            for (slong i = 0; i < factors.data->num; ++i) {
                const auto prime = factors.data->p + i;
                ulong power = factors.data->exp[i];
                for (slong j = 0; j < n && power; ++j) {
                    if (!fmpz_is_zero(coeff(j)))
                        power = std::min(power, static_cast<ulong>(fmpz_remove(temp.n,coeff(j),prime)/(n-j)));
                }
                fmpz_pow_ui(temp.n,prime,power);
                fmpz_mul(scale.n,scale.n,temp.n);
            }
        }
    }
    std::string out = scale.str();
    for (slong j = n; j >= 0; --j) {
        if (fmpz_is_zero(coeff(j))) out += " 0";
        else {
            fmpz_pow_ui(divisor.n,scale.n,n-j);
            fmpz_divexact(temp.n,coeff(j),divisor.n);
            out += " " + temp.str();
        }
    }
    return out + "\n";
}

// CRootOf orders the upper-half-plane roots of one irreducible factor by
// the lower-left corners of the first Collins--Krandick rectangles holding
// one such root, then gives the negative conjugate first. It does not sort
// the exact real parts. Reproduce this numbering by classifying FLINT's
// already-known exact roots. The normalized equation must be the same one
// used by CRootOf; positive scaling preserves FLINT's ordering.
struct RootBox {
    Number left, bottom, right, top;
    std::vector<slong> roots;
};

std::string display_order(std::istream& in) {
    Polynomial p;
    read_polynomial(in,p);
    IntegralPolynomial z;
    fmpq_poly_get_numerator(z.p,p.p);
    if (!fmpz_poly_is_squarefree(z.p))
        throw std::invalid_argument("Root display requires a squarefree defining polynomial.");
    const slong n = fmpz_poly_degree(z.p);
    Roots roots(n), real_parts(n), imaginary_parts(n);
    qqbar_roots_fmpz_poly(roots.r,z.p,0);
    std::vector<slong> order;
    RootBox initial;
    for (slong i = 0; i < n; ++i) {
        qqbar_re_im(real_parts.r+i,imaginary_parts.r+i,roots.r+i);
        if (qqbar_is_real(roots.r+i)) order.push_back(i);
        else if (qqbar_sgn_im(roots.r+i) > 0) initial.roots.push_back(i);
    }
    std::sort(order.begin(),order.end(),[&](slong i,slong j) {
        return qqbar_cmp_re(roots.r+i,roots.r+j) < 0;
    });

    // Initial box [-B,B] x [0,B], B=2*max(|a_i/a_n|).
    Integer largest, coefficient;
    for (slong j = 0; j <= n; ++j) {
        fmpz_poly_get_coeff_fmpz(coefficient.n,z.p,j);
        if (fmpz_cmpabs(coefficient.n,largest.n) > 0) fmpz_abs(largest.n,coefficient.n);
    }
    Fraction B;
    fmpq_set_fmpz_frac(B.value,largest.n,fmpz_poly_lead(z.p));
    fmpq_abs(B.value,B.value);
    fmpq_mul_2exp(B.value,B.value,1);
    qqbar_set_fmpq(initial.right.n,B.value);
    qqbar_set(initial.top.n,initial.right.n);
    qqbar_neg(initial.left.n,initial.right.n);

    std::vector<RootBox> todo, isolated;
    if (!initial.roots.empty()) todo.push_back(std::move(initial));
    while (!todo.empty()) {
        RootBox box = std::move(todo.back());
        todo.pop_back();
        Number width,height,mid;
        qqbar_sub(width.n,box.right.n,box.left.n);
        qqbar_sub(height.n,box.top.n,box.bottom.n);
        const bool vertical = qqbar_cmp_re(width.n,height.n) > 0;
        qqbar_add(mid.n,vertical ? box.left.n : box.bottom.n,vertical ? box.right.n : box.top.n);
        qqbar_mul_2exp_si(mid.n,mid.n,-1);
        RootBox lo,hi;
        for (RootBox* child : {&lo,&hi}) {
            qqbar_set(child->left.n,box.left.n);
            qqbar_set(child->right.n,box.right.n);
            qqbar_set(child->bottom.n,box.bottom.n);
            qqbar_set(child->top.n,box.top.n);
        }
        qqbar_set(vertical ? lo.right.n : lo.top.n,mid.n);
        qqbar_set(vertical ? hi.left.n : hi.bottom.n,mid.n);
        for (auto i : box.roots) {
            const int cmp = qqbar_cmp_re((vertical ? real_parts.r : imaginary_parts.r)+i,mid.n);
            // Subdivision counts exclude the south and east boundaries and
            // include the north and west boundaries, including their corner.
            if (vertical ? cmp < 0 : cmp <= 0) lo.roots.push_back(i);
            else hi.roots.push_back(i);
        }
        for (RootBox* child : {&lo,&hi})
            if (child->roots.size() == 1) isolated.push_back(std::move(*child));
            else if (!child->roots.empty()) todo.push_back(std::move(*child));
    }
    std::sort(isolated.begin(),isolated.end(),[](const RootBox& a,const RootBox& b) {
        const int x = qqbar_cmp_re(a.left.n,b.left.n);
        return x < 0 || (x == 0 && qqbar_cmp_re(a.bottom.n,b.bottom.n) < 0);
    });
    for (const auto& box : isolated) {
        const slong i = box.roots[0];
        Number conjugate;
        qqbar_conj(conjugate.n,roots.r+i);
        slong j = 0;
        while (j < n && !qqbar_equal(roots.r+j,conjugate.n)) ++j;
        if (j == n) throw std::logic_error("Missing conjugate of a polynomial root.");
        order.push_back(j);
        order.push_back(i);
    }
    if (static_cast<slong>(order.size()) != n) throw std::logic_error("Invalid algebraic root count.");
    std::vector<slong> positions(n);
    for (slong j = 0; j < n; ++j) positions[order[j]] = j;
    std::ostringstream out;
    for (slong j = 0; j < n; ++j) out << (j ? " " : "") << positions[j];
    return out.str() + "\n";
}

// Historical CRootOf indices are specified by isolating rectangles.
// Compare exact qqbar real/imaginary parts with their rational bounds.
Number boxed_root(std::istream& in) {
    Fraction q;
    Polynomial p;
    read_polynomial(in,p);
    const slong degree = fmpq_poly_degree(p.p);
    Number bounds[4];
    for (auto& b : bounds) { q.read(in); qqbar_set_fmpq(b.n,q.value); }
    Roots roots(degree);
    qqbar_roots_fmpq_poly(roots.r,p.p,0);
    slong index = -1;
    for (slong j = 0; j < degree; ++j) {
        Number re,im;
        qqbar_re_im(re.n,im.n,roots.r+j);
        if (qqbar_cmp_re(re.n,bounds[0].n) >= 0 && qqbar_cmp_re(re.n,bounds[1].n) <= 0 &&
            qqbar_cmp_re(im.n,bounds[2].n) >= 0 && qqbar_cmp_re(im.n,bounds[3].n) <= 0) {
            if (index >= 0) throw std::invalid_argument("The root rectangle is not isolating.");
            index = j;
        }
    }
    if (index < 0) throw std::invalid_argument("The root rectangle contains no root.");
    Number result;
    qqbar_set(result.n,roots.r+index);
    return result;
}

std::string integral_coefficients(const fmpz_poly_t p) {
    std::ostringstream out;
    Integer c;
    for (slong j = 0; j <= fmpz_poly_degree(p); ++j) {
        fmpz_poly_get_coeff_fmpz(c.n,p,j);
        out << (j ? " " : "") << c.str();
    }
    return out.str();
}

struct RootFactor {
    slong index, multiplicity, node_count;
    std::vector<slong> exponents;
};

// CRootOf groups the nonreal pairs by the historical PurePoly expression
// ordering: first the number of expression nodes, then number of terms,
// descending powers with coefficients as tie-breakers, then multiplicity.
Number indexed_root(std::istream& in) {
    const std::string normalized = root_normalization(in);
    slong wanted;
    if (!(in >> wanted)) throw std::invalid_argument("Missing algebraic root index.");
    std::istringstream row(normalized);
    Integer scale, c;
    std::string word;
    row >> word;
    fmpz_set_str(scale.n,word.c_str(),10);
    std::vector<std::string> coefficients;
    while (row >> word) coefficients.push_back(word);
    const slong degree = static_cast<slong>(coefficients.size())-1;
    if (wanted < 0) wanted += degree;
    if (wanted < 0 || wanted >= degree) throw std::invalid_argument("Invalid algebraic root index.");
    IntegralPolynomial z;
    for (slong j = 0; j <= degree; ++j) {
        fmpz_set_str(c.n,coefficients[degree-j].c_str(),10);
        fmpz_poly_set_coeff_fmpz(z.p,j,c.n);
    }
    struct Factorization {
        fmpz_poly_factor_t f;
        Factorization() { fmpz_poly_factor_init(f); }
        ~Factorization() { fmpz_poly_factor_clear(f); }
    } factors;
    fmpz_poly_factor(factors.f,z.p);
    std::vector<RootFactor> groups;
    for (slong i = 0; i < factors.f->num; ++i) {
        const auto p = factors.f->p+i;
        RootFactor factor{i,factors.f->exp[i],0,{}};
        for (slong j = fmpz_poly_degree(p); j >= 0; --j) {
            auto a = fmpz_poly_get_coeff_ptr(p,j);
            if (fmpz_is_zero(a)) continue;
            factor.exponents.push_back(j);
            factor.node_count += (j >= 2 ? 3 : 1) + (j && !fmpz_is_one(a) ? 2 : 0);
        }
        factor.node_count += factor.exponents.size() > 1;
        groups.push_back(std::move(factor));
    }
    std::sort(groups.begin(),groups.end(),[&](const RootFactor& a,const RootFactor& b) {
        if (a.node_count != b.node_count) return a.node_count < b.node_count;
        if (a.exponents.size() != b.exponents.size()) return a.exponents.size() < b.exponents.size();
        for (size_t j = 0; j < a.exponents.size(); ++j) {
            const auto e = a.exponents[j];
            if (e != b.exponents[j]) return e < b.exponents[j];
            const int cmp = fmpz_cmp(fmpz_poly_get_coeff_ptr(factors.f->p+a.index,e),
                                      fmpz_poly_get_coeff_ptr(factors.f->p+b.index,e));
            if (cmp) return cmp < 0;
        }
        return a.multiplicity < b.multiplicity;
    });
    struct Root { Number value; slong multiplicity; };
    std::vector<Root> real, complex;
    for (const auto& group : groups) {
        const auto p = factors.f->p+group.index;
        const auto n = fmpz_poly_degree(p);
        Roots roots(n);
        qqbar_roots_fmpz_poly(roots.r,p,0);
        std::istringstream src(std::to_string(n)+" "+integral_coefficients(p));
        std::istringstream display(display_order(src));
        std::vector<slong> order(n);
        for (slong j = 0, k; j < n; ++j) { display >> k; order.at(k) = j; }
        for (slong k : order) {
            Root root{Number{},group.multiplicity};
            qqbar_set(root.value.n,roots.r+k);
            (qqbar_is_real(root.value.n) ? real : complex).push_back(std::move(root));
        }
    }
    std::sort(real.begin(),real.end(),[](const Root& a,const Root& b) {
        return qqbar_cmp_re(a.value.n,b.value.n)<0;
    });
    for (const auto* list : {&real,&complex}) {
        for (const auto& root : *list) {
            if (wanted < root.multiplicity) {
                Number value;
                qqbar_mul_fmpz(value.n,root.value.n,scale.n);
                return value;
            }
            wanted -= root.multiplicity;
        }
    }
    throw std::logic_error("No matching algebraic root.");
}

Number evaluate(std::istream& in);

std::string root_position(std::istream& in, bool flint_index = false) {
    std::string line;
    std::getline(in,line);
    std::istringstream generator(line);
    const Number alpha = evaluate(generator);
    Polynomial p;
    read_polynomial(in,p);
    const slong degree = fmpq_poly_degree(p.p);
    Number value;
    qqbar_evaluate_fmpq_poly(value.n,p.p,alpha.n);
    if (!qqbar_is_zero(value.n)) throw std::invalid_argument("Generator is not a root of its declared polynomial.");
    slong real = 0, smaller = 0;
    Roots roots(degree);
    qqbar_roots_fmpq_poly(roots.r,p.p,0);
    for (slong j = 0; j < degree; ++j) {
        if (flint_index && qqbar_equal(roots.r+j,alpha.n)) return std::to_string(j) + "\n";
        if (qqbar_is_real(roots.r+j)) {
            ++real;
            if (qqbar_cmp_re(roots.r+j,alpha.n) < 0) ++smaller;
        }
    }
    if (flint_index) throw std::logic_error("Missing marked generator among its polynomial roots.");
    // Historical CRootOf numbering puts sorted real roots first and each
    // conjugate pair in lower/upper half-plane order. With multiple pairs
    // their relative order depends on the isolating rectangles.
    if (qqbar_is_real(alpha.n)) return std::to_string(smaller) + "\n";
    if (degree-real == 2) return std::to_string(real + (qqbar_sgn_im(alpha.n) > 0)) + "\n";
    // For ambiguous complex-pair order the caller sends literals selected
    // by the original isolators. Equality here is exact in Qbar.
    slong index;
    while (in >> index) {
        std::getline(in,line);
        std::istringstream candidate(line);
        const Number root = evaluate(candidate);
        if (qqbar_equal(root.n,alpha.n)) return std::to_string(index) + "\n";
    }
    return "boxes " + std::to_string(real) + "\n";
}

Number evaluate(std::istream& in) {
    std::vector<Number> s;
    std::string code;
    while (in >> code) {
        Fraction q;
        if (code == "q") {
            q.read(in); s.emplace_back(); qqbar_set_fmpq(s.back().n,q.value);
        } else if (code == "i") {
            s.emplace_back(); qqbar_i(s.back().n);
        } else if (code == "root") {
            s.push_back(boxed_root(in));
        } else if (code == "indexedroot") {
            s.push_back(indexed_root(in));
        } else if (code == "flintroot") {
            Polynomial p;
            read_polynomial(in,p);
            const slong degree = fmpq_poly_degree(p.p);
            slong index;
            if (!(in >> index) || index < 0 || index >= degree)
                throw std::invalid_argument("Invalid indexed algebraic root.");
            Roots roots(degree);
            qqbar_roots_fmpq_poly(roots.r,p.p,0);
            s.emplace_back();
            qqbar_set(s.back().n,roots.r+index);
        } else if (code == "pow" && !s.empty()) {
            q.read(in);
            if (qqbar_is_zero(s.back().n) && fmpq_sgn(q.value) < 0) throw std::invalid_argument("Division by zero.");
            qqbar_pow_fmpq(s.back().n,s.back().n,q.value);
        } else if ((code == "+" || code == "*") && s.size() >= 2) {
            const Number y(std::move(s.back())); s.pop_back();
            if (code == "+") qqbar_add(s.back().n,s.back().n,y.n);
            else qqbar_mul(s.back().n,s.back().n,y.n);
        } else throw std::invalid_argument("Invalid algebraic coefficient expression.");
    }
    if (s.size() != 1) throw std::invalid_argument("Empty algebraic coefficient expression.");
    return std::move(s.back());
}

bool in_power_basis(fmpq_poly_t result, qqbar_srcptr alpha, qqbar_srcptr value) {
    const slong d = qqbar_degree(alpha);
    if (d % qqbar_degree(value)) return false;
    // At fixed precision, failure to recognize a power basis is inconclusive.
    // Try increasing precision. To certify nonmembership, find a primitive
    // linear combination alpha + j*value whose degree exceeds [K:Q].
    // All but finitely many integers j separate the distinct embeddings
    // of the compositum in characteristic zero.
    for (slong j = 1, prec = 128;; ++j, prec *= 2) {
        if (qqbar_express_in_field(result,alpha,value,0,0,prec)) {
            if (!qqbar_equal_fmpq_poly_val(value,result,alpha))
                throw std::logic_error("Uncertified field coefficient.");
            break;
        }
        Number sum;
        qqbar_mul_si(sum.n,value,j);
        qqbar_add(sum.n,sum.n,alpha);
        if (qqbar_degree(sum.n) > d) return false;
        if (prec > WORD_MAX/2) throw Limit("Field expression precision limit.");
    }
    return true;
}

std::string express(std::istream& in) {
    std::string line;
    if (!std::getline(in,line)) throw std::invalid_argument("Missing field generator.");
    std::istringstream gen(line); const Number alpha = evaluate(gen);
    if (!std::getline(in,line)) throw std::invalid_argument("Missing algebraic coefficient.");
    std::istringstream expression(line); const Number value = evaluate(expression);
    Polynomial result;
    if (!in_power_basis(result.p, alpha.n, value.n)) return "outside\n";
    Fraction c;
    std::string out;
    for (slong j = 0; j < fmpq_poly_length(result.p); ++j) {
        fmpq_poly_get_coeff_fmpq(c.value,result.p,j);
        char* text = fmpq_get_str(nullptr,10,c.value);
        if (j) out += ' ';
        out += text;
        flint_free(text);
    }
    return out + "\n";
}

// Return the primitive integral minimal polynomial and the exact FLINT mark.
slong marked_root_index(const Number& value) {
    const auto polynomial = QQBAR_POLY(value.n);
    const slong degree = fmpz_poly_degree(polynomial);
    Roots roots(degree);
    qqbar_roots_fmpz_poly(roots.r, polynomial, 0);
    for (slong j=0; j<degree; ++j)
        if (qqbar_equal(roots.r+j, value.n)) return j+1;
    throw std::logic_error("No matching embedding of the minimal polynomial.");
}

std::string describe(std::istream& in) {
    const Number value = evaluate(in);
    return std::to_string(marked_root_index(value)) + " " + integral_coefficients(QQBAR_POLY(value.n)) + "\n";
}

Json::Value arithmetic_field(const Json::Value& input) {
    std::istringstream program(input["program"].asString());
    const Number value=evaluate(program);
    const auto minimal=QQBAR_POLY(value.n);
    const slong degree=fmpz_poly_degree(minimal), index=marked_root_index(value);
    if (degree<2) throw std::invalid_argument("Use QQ for a rational coefficient field.");
    if ((input.isMember("degree") && mpz_class(input["degree"].asString())!=degree) ||
        (input.isMember("root_index") && mpz_class(input["root_index"].asString())!=index))
        throw std::invalid_argument("The marked degree or root index disagrees with the embedding.");
    if (input.isMember("relation")) {
        const auto terms = sparse_json::read_terms(input["relation"], 1);
        if (terms.empty() || terms.rbegin()->first[0]!=degree)
            throw std::invalid_argument("The supplied relation is not the minimal polynomial of the marked root.");
        Integer coefficient;
        mpz_class leading;
        fmpz_get_mpz(leading.get_mpz_t(), fmpz_poly_lead(minimal));
        const mpq_class scale = terms.rbegin()->second/leading;
        for (slong j=0;j<=degree;++j) {
            fmpz_poly_get_coeff_fmpz(coefficient.n, minimal, j);
            mpz_class z;
            fmpz_get_mpz(z.get_mpz_t(), coefficient.n);
            const auto term=terms.find({j});
            if ((term==terms.end() ? mpq_class(0) : term->second) != scale*z)
                throw std::invalid_argument("The supplied relation is not the minimal polynomial of the marked root.");
        }
    }
    Json::Value out;
    out["degree"]=Json::Int64(degree);
    out["root_index"]=Json::Int64(index);
    out["coefficients"]=integral_coefficients(minimal);
    return out;
}
void load_power_basis(Polynomial& p, const std::vector<mpq_class>& coeffs) {
    Fraction coefficient;
    for (size_t j=0; j<coeffs.size(); ++j) {
        fmpq_set_mpq(coefficient.value, coeffs[j].get_mpq_t());
        fmpq_poly_set_coeff_fmpq(p.p, j, coefficient.value);
    }
}
}  // namespace

bool certify_power_basis_roots(const std::vector<mpq_class>& minimal,
        const std::vector<mpq_class>& equation, const std::vector<std::vector<mpq_class>>& roots) {
    if (minimal.size() < 2 || minimal.back() == 0 || equation.size() < 2 ||
        equation.back() == 0 || equation.size()-1 != roots.size()) return false;
    Polynomial modulus;
    load_power_basis(modulus, minimal);
    std::set<std::vector<mpq_class>> distinct;
    for (const auto& root : roots) {
        // The zero element is encoded as an empty row. Requiring reduced,
        // trimmed rows makes a direct equality check exact in the field.
        if (root.size() >= minimal.size() || (!root.empty() && root.back() == 0) ||
            !distinct.insert(root).second) return false;
        Polynomial value, remainder;
        load_power_basis(value, root);
        Fraction c;
        for (auto it=equation.rbegin(); it!=equation.rend(); ++it) {
            fmpq_poly_mul(remainder.p, remainder.p, value.p);
            fmpq_poly_rem(remainder.p, remainder.p, modulus.p);
            fmpq_set_mpq(c.value, it->get_mpq_t());
            fmpq_poly_add_fmpq(remainder.p, remainder.p, c.value);
        }
        if (!fmpq_poly_is_zero(remainder.p)) return false;
    }
    return true;
}

std::optional<std::vector<std::vector<mpq_class>>> roots_in_power_basis(
        const std::vector<mpq_class>& minimal, const std::vector<mpq_class>& equation) {
    if (minimal.size() < 2 || equation.size() < 2)
        throw std::invalid_argument("Expected nonconstant field and root polynomials.");
    Polynomial m, f;
    load_power_basis(m, minimal);
    load_power_basis(f, equation);
    Roots embeddings(fmpq_poly_degree(m.p)), roots(fmpq_poly_degree(f.p));
    qqbar_roots_fmpq_poly(embeddings.r, m.p, 0);
    qqbar_roots_fmpq_poly(roots.r, f.p, 0);
    std::vector<std::vector<mpq_class>> coordinates;
    // All roots are included, so choosing any one embedding gives the
    // same set of formal roots in the abstract coefficient field.
    for (slong i=0; i<roots.degree; ++i) {
        Polynomial image;
        if (!in_power_basis(image.p, embeddings.r, roots.r+i)) return std::nullopt;
        std::vector<mpq_class> row(fmpq_poly_length(image.p));
        Fraction c;
        for (size_t j=0; j<row.size(); ++j) {
            fmpq_poly_get_coeff_fmpq(c.value, image.p, j);
            fmpq_get_mpq(row[j].get_mpq_t(), c.value);
        }
        coordinates.push_back(std::move(row));
    }
    return coordinates;
}

}  // namespace atlas

extern "C" const char* atlas_common_field(const char* input) noexcept {
    return atlas::sparse_json::invoke(input, atlas::common_field);
}

extern "C" char* atlas_field_coefficient(const char* input, size_t length) noexcept {
    try {
        std::istringstream stream(std::string(input,length));
        return atlas::copy_text(atlas::express(stream));
    } catch (const std::exception& error) {
        atlas::set_error(error.what());
    } catch (...) {
        atlas::set_error("Unknown algebraic coefficient failure.");
    }
    return nullptr;
}

extern "C" char* atlas_field_radical(const char* input, size_t length) noexcept {
    try {
        std::istringstream stream(std::string(input,length));
        return atlas::copy_text(atlas::rational_radical(stream));
    } catch (const std::exception& error) {
        atlas::set_error(error.what());
    } catch (...) {
        atlas::set_error("Unknown rational radical display failure.");
    }
    return nullptr;
}

// The optional scale and offset are returned in exact rational arithmetic.
extern "C" char* atlas_field_scaled_radical(const char* input, size_t length) noexcept {
    return atlas_field_radical(input, length);
}

extern "C" char* atlas_field_low_degree_root(const char* input, size_t length) noexcept {
    try {
        std::istringstream stream(std::string(input,length));
        int degree, index;
        if (!(stream >> degree >> index) || degree < 1 || degree > 2 || index < 0 || index >= degree)
            throw std::invalid_argument("Expected a linear or quadratic marked root.");
        atlas::Fraction a, b, c, det, offset, scale;
        a.read(stream);
        b.read(stream);
        if (fmpq_is_zero(a.value)) throw std::invalid_argument("The leading coefficient vanishes.");
        fmpq_div(offset.value, b.value, a.value);
        fmpq_neg(offset.value, offset.value);
        if (degree == 1) return atlas::copy_text(atlas::text(offset.value));
        c.read(stream);
        fmpq_div_2exp(offset.value, offset.value, 1);
        fmpq_mul(det.value, a.value, c.value);
        fmpq_mul_2exp(det.value, det.value, 2);
        fmpq_mul(c.value, b.value, b.value);
        fmpq_sub(det.value, c.value, det.value);
        fmpq_inv(scale.value, a.value);
        fmpq_div_2exp(scale.value, scale.value, 1);
        if (index == 0) fmpq_neg(scale.value, scale.value);
        std::istringstream radical(atlas::text(det.value) + " " + atlas::text(scale.value));
        return atlas::copy_text(atlas::text(offset.value) + " " + atlas::rational_radical(radical));
    } catch (const std::exception& error) { atlas::set_error(error.what()); }
    catch (...) { atlas::set_error("Failed to compute a low degree marked root."); }
    return nullptr;
}

extern "C" char* atlas_field_rational_product(const char* input, size_t length) noexcept {
    try {
        std::istringstream stream(std::string(input,length));
        return atlas::copy_text(atlas::rational_product(stream));
    } catch (const std::exception& error) {
        atlas::set_error(error.what());
    } catch (...) {
        atlas::set_error("Rational product normalization failed.");
    }
    return nullptr;
}

extern "C" char* atlas_field_integer_factors(const char* input, size_t length) noexcept {
    try {
        atlas::Integer value, prime;
        std::istringstream stream(std::string(input,length));
        std::string text, extra;
        if (!(stream >> text) || stream >> extra || text.find_first_not_of("0123456789") != std::string::npos
                || fmpz_set_str(value.n,text.c_str(),10) || fmpz_sgn(value.n) <= 0)
            throw std::invalid_argument("Expected a positive integer.");
        atlas::Factors factors;
        fmpz_factor(factors.data,value.n);
        std::ostringstream out;
        for (slong j = 0; j < factors.data->num; ++j) {
            fmpz_set(prime.n,factors.data->p+j);
            out << prime.str() << ' ' << factors.data->exp[j] << '\n';
        }
        return atlas::copy_text(out.str());
    } catch (const std::exception& error) { atlas::set_error(error.what()); }
    catch (...) { atlas::set_error("Integer radical factorization failed."); }
    return nullptr;
}

extern "C" char* atlas_field_position(const char* input, size_t length) noexcept {
    try {
        std::istringstream stream(std::string(input,length));
        return atlas::copy_text(atlas::root_position(stream));
    } catch (const std::exception& error) {
        atlas::set_error(error.what());
    } catch (...) {
        atlas::set_error("Unknown field root position.");
    }
    return nullptr;
}

extern "C" char* atlas_field_flint_index(const char* input, size_t length) noexcept {
    try {
        std::istringstream stream(std::string(input,length));
        return atlas::copy_text(atlas::root_position(stream,true));
    } catch (const std::exception& error) {
        atlas::set_error(error.what());
    } catch (...) {
        atlas::set_error("Unknown field root index failure.");
    }
    return nullptr;
}

extern "C" char* atlas_field_normalize(const char* input, size_t length) noexcept {
    try {
        std::istringstream stream(std::string(input,length));
        return atlas::copy_text(atlas::root_normalization(stream));
    } catch (const std::exception& error) {
        atlas::set_error(error.what());
    } catch (...) {
        atlas::set_error("Unknown field root normalization failure.");
    }
    return nullptr;
}


extern "C" char* atlas_field_display_order(const char* input, size_t length) noexcept {
    try {
        std::istringstream stream(std::string(input,length));
        return atlas::copy_text(atlas::display_order(stream));
    } catch (const std::exception& error) {
        atlas::set_error(error.what());
    } catch (...) {
        atlas::set_error("Unknown algebraic root ordering failure.");
    }
    return nullptr;
}

extern "C" char* atlas_field_description(const char* input, size_t length) noexcept {
    try {
        std::istringstream stream(std::string(input,length));
        return atlas::copy_text(atlas::describe(stream));
    } catch (const std::exception& error) { atlas::set_error(error.what()); }
    catch (...) { atlas::set_error("Unknown marked coefficient failure."); }
    return nullptr;
}

extern "C" char* atlas_field_display_terms(const char* input, size_t length) noexcept {
    try {
        std::istringstream stream(std::string(input,length));
        return atlas::copy_text(atlas::display_terms(stream));
    } catch (const std::exception& error) { atlas::set_error(error.what()); }
    catch (...) { atlas::set_error("Unknown marked display normalization failure."); }
    return nullptr;
}

extern "C" const char* atlas_arithmetic_field(const char* input) noexcept {
    return atlas::sparse_json::invoke(input, atlas::arithmetic_field);
}
