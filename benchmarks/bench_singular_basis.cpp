// Compare Singular's native std (pure_gb, redSB) from parsed QQ input to a monic basis.
// Build with c++ -O3 -std=c++17 ... $(pkg-config --cflags --libs Singular) -lgmpxx -lgmp.
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <gmpxx.h>
#include <Singular/libsingular.h>

namespace {

ring make_ring(int variables, int order) {
    std::vector<std::string> names;
    for (int i = 0; i < variables; ++i) names.push_back("x" + std::to_string(i));
    std::vector<char*> symbols;
    for (std::string& name : names) symbols.push_back(name.data());

    // Match the interpreter's QQ,(x0,...),dp or lp, including its component block.
    auto orders = static_cast<rRingOrder_t*>(omAlloc0(3 * sizeof(rRingOrder_t)));
    auto first = static_cast<int*>(omAlloc0(3 * sizeof(int)));
    auto last = static_cast<int*>(omAlloc0(3 * sizeof(int)));
    orders[0] = order == 1 ? ringorder_lp : ringorder_dp;
    first[0] = 1;
    last[0] = variables;
    orders[1] = ringorder_C;
    return rDefault(0, variables, symbols.data(), 3, orders, first, last);
}

poly parse_polynomial(const std::string& row, ring r) {
    poly result = nullptr;
    std::istringstream terms(row);
    std::string term;
    while (std::getline(terms, term, ';')) {
        std::istringstream source(term);
        std::string coefficient;
        if (!(source >> coefficient)) throw std::runtime_error("Missing coefficient.");
        mpq_class q(coefficient, 10);
        if (mpz_sgn(q.get_den_mpz_t()) == 0) throw std::runtime_error("Zero denominator.");
        q.canonicalize();
        number num = n_InitMPZ(q.get_num_mpz_t(), r->cf);
        number den = n_InitMPZ(q.get_den_mpz_t(), r->cf);
        number value = n_Div(num, den, r->cf);
        n_Delete(&num, r->cf);
        n_Delete(&den, r->cf);
        poly monomial = p_NSet(value, r);
        for (int i = 1; i <= r->N; ++i) {
            long exponent;
            if (!(source >> exponent) || exponent < 0 || static_cast<unsigned long>(exponent) > r->bitmask)
                throw std::runtime_error("Missing or out-of-range exponent.");
            if (monomial) p_SetExp(monomial, i, exponent, r);
        }
        if (source >> coefficient) throw std::runtime_error("Extra exponent.");
        if (monomial) {
            p_Setm(monomial, r);
            result = p_Add_q(result, monomial, r);
        }
    }
    return result;
}

ideal solve(ideal generators, ring r) {
    // These are the calls made by jjSTD for a plain ideal with no isHomog
    // attribute. id_Norm is simplify(ideal,1)'s native in-place normalization.
    intvec* weights = nullptr;
    ideal result = kStd2(generators, r->qideal, testHomog, &weights);
    delete weights;
    if (!result || errorreported) throw std::runtime_error("libSingular kStd2 failed.");
    idSkipZeroes(result);
    id_Norm(result, r);
    return result;
}

std::vector<poly> ordered_basis(ideal result, ring r) {
    std::vector<poly> basis;
    for (int i = 0; i < IDELEMS(result); ++i) {
        poly p = result->m[i];
        if (p) {
            if (!n_IsOne(p_GetCoeff(p, r), r->cf)) throw std::runtime_error("Expected a monic generator.");
            basis.push_back(p);
        }
    }
    std::sort(basis.begin(), basis.end(), [r](poly a, poly b) { return p_LmCmp(a, b, r) > 0; });
    return basis;
}

void write_basis(const std::vector<poly>& basis, ring r) {
    std::cout << "BASIS " << basis.size() << '\n';
    for (poly polynomial : basis) {
        for (poly term = polynomial; term; term = pNext(term)) {
            if (term != polynomial) std::cout << ';';
            StringSetS("");
            n_Write(p_GetCoeff(term, r), r->cf, FALSE);
            char* coefficient = StringEndS();
            std::cout << coefficient;
            omFree(coefficient);
            for (int i = 1; i <= r->N; ++i) std::cout << ' ' << p_GetExp(term, i, r);
        }
        std::cout << '\n';
    }
    std::cout.flush();
}

void run(const std::string& request) {
    int rows, batch, variables, order, known;
    if (!(std::istringstream(request) >> rows >> batch) || rows < 0 || batch < 1)
        throw std::runtime_error("Expected a generator count and positive batch size.");
    std::string line;
    if (!std::getline(std::cin, line) || !(std::istringstream(line) >> variables >> order >> known)
        || variables < 1 || known != 0 || (order != 0 && order != 1))
        throw std::runtime_error("Expected a lex or grevlex ring with no known basis.");
    ring r = make_ring(variables, order);
    rChangeCurrRing(r);
    ideal generators = idInit(std::max(1, rows), 1);
    for (int i = 0; i < rows; ++i) {
        if (!std::getline(std::cin, line)) throw std::runtime_error("Missing input polynomial.");
        generators->m[i] = parse_polynomial(line, r);
    }
    ideal warm = solve(generators, r);
    id_Delete(&warm, r);
    std::vector<ideal> results(batch);
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    for (ideal& result : results) result = solve(generators, r);
    const std::chrono::duration<double, std::milli> elapsed = Clock::now() - start;
    auto basis = ordered_basis(results.back(), r);
    for (int i = 0; i + 1 < batch; ++i) {
        auto other = ordered_basis(results[i], r);
        if (!std::equal(other.begin(), other.end(), basis.begin(), basis.end(),
                        [r](poly a, poly b) { return p_EqualPolys(a, b, r); }))
            throw std::runtime_error("libSingular returned inconsistent bases in one batch.");
    }
    std::cout << "TIME " << std::setprecision(12) << elapsed.count() / batch << '\n';
    write_basis(basis, r);
    for (ideal& result : results) id_Delete(&result, r);
    id_Delete(&generators, r);
    rChangeCurrRing(nullptr);
    rDelete(r);
}

} // namespace

int main(int argc, char** argv) {
    try {
        // siInit uses the Singular executable path to locate its installed resources.
        if (argc != 2) throw std::runtime_error("Pass the path to the installed Singular executable.");
        siInit(argv[1]);
        si_opt_1 |= Sy_bit(OPT_REDSB);
        si_opt_2 |= Sy_bit(V_PURE_GB);
        std::string request;
        while (std::getline(std::cin, request)) run(request);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
