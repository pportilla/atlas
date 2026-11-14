// An ordinary multiple point is resolved by the blowup at the origin.
// The C ABI accepts sparse monomials over Q, and returns one UTF-8 line per
// result field. Allocation and deallocation both take place in this library.
#include <gmpxx.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {
using Powers = std::vector<int>;
struct Term { Powers powers; mpq_class coeff; int degree; };
using Terms = std::vector<Term>;
using RankKernel = int (*)(const uint32_t*, int, int, uint32_t, int, int*);
struct Accelerator { RankKernel kernel; int device; int minimum_columns; };

// Each prime fits in 31 bits, so a product fits in a signed 64-bit integer.
using Scalar = long long;
constexpr Scalar primes[] = {1000000007LL, 1000000009LL};
Scalar power(Scalar base, Scalar exponent, Scalar prime) {
    Scalar result = 1;
    for (; exponent; exponent >>= 1) {
        if (exponent & 1) result = result * base % prime;
        base = base * base % prime;
    }
    return result;
}

std::vector<Scalar> reduce_coefficients(const Terms& terms, Scalar prime) {
    std::vector<Scalar> coefficients;
    for (const Term& t : terms) {
        const Scalar den = mpz_fdiv_ui(t.coeff.get_den_mpz_t(), prime);
        if (!den) return {};
        const Scalar num = mpz_fdiv_ui(t.coeff.get_num_mpz_t(), prime);
        coefficients.push_back(num * power(den, prime-2, prime) % prime);
    }
    return coefficients;
}

void trim(std::vector<Scalar>& f) {
    while (!f.empty() && !f.back()) f.pop_back();
}

// A binary form defining d different geometric directions has at most a
// simple zero at infinity, and its dehomogenization is square-free. Degree-
// preserving reduction and a unit gcd certify this over Q.
std::string binary_certificate(const Terms& tangent, int degree) {
    int affine_degree = 0;
    for (const auto& t : tangent)
        affine_degree = std::max(affine_degree, t.powers[1]);
    if (degree-affine_degree > 1) return {};
    for (Scalar prime : primes) {
        auto coefficients = reduce_coefficients(tangent, prime);
        if (coefficients.empty()) continue;
        std::vector<Scalar> f(affine_degree+1);
        for (size_t i=0; i<tangent.size(); ++i) f[tangent[i].powers[1]] = coefficients[i];
        if (!f.back()) continue;
        std::vector<Scalar> g(affine_degree);
        for (int i=1; i<=affine_degree; ++i) g[i-1] = f[i]*i % prime;
        trim(g);
        while (!g.empty()) {
            const Scalar inverse = power(g.back(), prime-2, prime);
            while (f.size() >= g.size()) {
                const auto shift = f.size()-g.size();
                const Scalar scale = f.back()*inverse % prime;
                for (size_t i=0; i<g.size(); ++i) {
                    auto& entry = f[i+shift];
                    entry -= scale*g[i] % prime;
                    if (entry<0) entry += prime;
                }
                trim(f);
            }
            std::swap(f,g);
        }
        if (f.size()==1)
            return "binary " + std::to_string(prime) + " " + std::to_string(affine_degree);
    }
    return {};
}

void monomials(int degree, int pos, Powers& monomial, std::vector<Powers>& out) {
    if (pos == static_cast<int>(monomial.size()) - 1) {
        monomial[pos] = degree;
        out.push_back(monomial);
        return;
    }
    for (int i = 0; i <= degree; ++i) {
        monomial[pos] = i;
        monomials(degree-i, pos+1, monomial, out);
    }
}

template<Scalar prime>
bool spans_prime(std::vector<std::vector<Scalar>>& matrix, std::vector<int>& minor) {
    const int rows = static_cast<int>(matrix.size());
    const int cols = static_cast<int>(matrix.front().size());
    int rank = 0;
    std::vector<int> indices(rows);
    std::iota(indices.begin(),indices.end(),0);
    for (int col = 0; col < cols; ++col) {
        int pivot = rank;
        while (pivot < rows && !matrix[pivot][col]) ++pivot;
        if (pivot == rows) return false;
        std::swap(matrix[pivot], matrix[rank]);
        std::swap(indices[pivot], indices[rank]);
        const Scalar inverse = power(matrix[rank][col], prime-2, prime);
        for (int i = rank+1; i < rows; ++i) {
            if (!matrix[i][col]) continue;
            const Scalar scale = matrix[i][col] * inverse % prime;
            for (int j = col+1; j < cols; ++j) {
                matrix[i][j] -= scale * matrix[rank][j] % prime;
                if (matrix[i][j] < 0) matrix[i][j] += prime;
            }
        }
        ++rank;
    }
    minor.assign(indices.begin(),indices.begin()+cols);
    return true;
}

bool spans(std::vector<std::vector<Scalar>>& matrix, Scalar prime, std::vector<int>& minor) {
    return prime == primes[0] ? spans_prime<primes[0]>(matrix,minor)
                              : spans_prime<primes[1]>(matrix,minor);
}

std::string rank_certificate(std::vector<std::vector<Scalar>>& matrix, Scalar prime, Accelerator gpu) {
    auto start = std::chrono::steady_clock::now();
    int rows = static_cast<int>(matrix.size()), cols = static_cast<int>(matrix[0].size());
    std::vector<int> minor(cols);
    std::string backend = "cpu";
    bool verified = false;
    if (gpu.kernel && cols >= gpu.minimum_columns) {
        std::vector<uint32_t> flat;
        flat.reserve(size_t(rows)*cols);
        for (const auto& row : matrix) flat.insert(flat.end(), row.begin(), row.end());
        if (gpu.kernel(flat.data(),rows,cols,prime,gpu.device,minor.data())==1) {
            // Recheck the smaller square minor on the CPU. Even a driver
            // arithmetic error cannot yield a false smoothness certificate.
            std::vector<std::vector<Scalar>> witness;
            for (int i : minor) {
                if (i<0 || i>=rows) break;
                witness.push_back(matrix[i]);
            }
            std::vector<int> unused;
            verified = static_cast<int>(witness.size())==cols && spans(witness,prime,unused);
        }
        backend = verified ? "opencl" : "cpu_after_opencl";
    }
    if (!verified && !spans(matrix,prime,minor)) return {};
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now()-start).count();
    std::ostringstream proof;
    proof << backend << ' ' << micros << ' ' << rows << ' ';
    for (size_t i=0; i<minor.size(); ++i) {
        if (i) proof << ',';
        proof << minor[i];
    }
    return proof.str();
}

// The degree-D part of the Jacobian ideal is spanned by all degree-(D-d+1)
// monomial multiples of the partials. A full-rank minor mod p lifts to a
// nonzero minor over Q. Hence the gradient cannot vanish at any projective
// point over the algebraic closure. Rank failure never certifies singularity.
std::string jacobian_certificate(const Terms& tangent, int n, int d, Accelerator gpu, std::string& reason) {
    bool diagonal = static_cast<int>(tangent.size()) == n;
    for (const Term& t : tangent)
        diagonal = diagonal && std::count(t.powers.begin(), t.powers.end(), d) == 1;
    if (diagonal) return "diagonal";
    // If all partials vanish at a coordinate point, projective smoothness
    // already fails; no CPU/GPU rank job can prove it.
    for (int j=0; j<n; ++j) {
        if (std::none_of(tangent.begin(),tangent.end(),
                [=](const Term& t){ return t.powers[j] >= d-1; })) {
            reason = "coordinate " + std::to_string(j);
            return {};
        }
    }
    if (n==2) {
        reason = "binary";
        return binary_certificate(tangent,d);
    }

    const int degree = n*(d-2) + 1;
    // Bound the dimension before materializing any monomials.
    int count = 1;
    for (int i = 1; i < n; ++i) {
        count = count*(degree+i)/i;
        if (count > 512) {
            // The partial binomial product above only guards allocation. Report
            // the exact full matrix dimensions, even when they exceed 64 bits.
            mpz_class full_columns, full_rows;
            mpz_bin_uiui(full_columns.get_mpz_t(),degree+n-1,n-1);
            mpz_bin_uiui(full_rows.get_mpz_t(),degree-d+n,n-1);
            full_rows *= n;
            reason = "budget " + std::to_string(degree) + " " + full_columns.get_str()
                     + " " + full_rows.get_str() + " 512";
            return {};
        }
    }
    Powers p(n);
    std::vector<Powers> basis, factors;
    monomials(degree, 0, p, basis);
    monomials(degree-d+1, 0, p, factors);
    std::map<Powers,int> index;
    for (int i = 0; i < count; ++i) index[basis[i]] = i;
    std::ostringstream tested_primes;
    for (const Scalar prime : primes) {
        const auto coefficients = reduce_coefficients(tangent, prime);
        if (coefficients.empty()) continue;
        tested_primes << ' ' << prime;
        std::vector<std::vector<Scalar>> matrix;
        for (int k = 0; k < n; ++k) {
            for (const auto& factor : factors) {
                matrix.emplace_back(count, 0);
                auto& row = matrix.back();
                for (size_t i = 0; i < tangent.size(); ++i) {
                    const auto& exponents = tangent[i].powers;
                    if (!exponents[k]) continue;
                    for (int j=0; j<n; ++j) p[j] = factor[j] + exponents[j];
                    --p[k];
                    row[index.at(p)] = coefficients[i] * exponents[k] % prime;
                }
            }
        }
        auto proof = rank_certificate(matrix,prime,gpu);
        if (!proof.empty()) {
            return "jacobian " + std::to_string(prime) + " " + std::to_string(degree) + " " + std::to_string(count) + " " + proof;
        }
    }
    reason = "modular" + tested_primes.str();
    return {};
}

std::string equation(const Terms& terms, int chart=-1, int order=0) {
    std::ostringstream out;
    bool first = true;
    for (const auto& term : terms) {
        auto powers = term.powers;
        if (chart >= 0) powers[chart] = term.degree-order;
        const bool constant = std::none_of(powers.begin(), powers.end(), [](int x){ return x != 0; });
        if (term.coeff < 0) out << (first ? "-" : " - ");
        else if (!first) out << " + ";
        mpq_class magnitude = abs(term.coeff);
        bool factor = magnitude != 1 || constant;
        if (factor) out << magnitude;
        for (size_t i=0; i<powers.size(); ++i) {
            if (!powers[i]) continue;
            if (factor) out << "*";
            out << 'u' << i+1;
            if (powers[i] != 1) out << '^' << powers[i];
            factor = true;
        }
        first = false;
    }
    return out.str();
}

std::string compute(const char* input, Accelerator gpu) {
    std::istringstream stream(input);
    int n=0, count=0;
    if (!(stream >> n >> count) || n < 1 || n > 32 || count < 1 || count > 8192)
        return "defer";
    Terms terms;
    int order = std::numeric_limits<int>::max();
    for (int i=0; i<count; ++i) {
        std::string coefficient;
        if (!(stream >> coefficient) || coefficient.size() > 16384) return "defer";
        mpq_class value;
        if (value.set_str(coefficient,10) != 0 || value.get_den() == 0) return "defer";
        value.canonicalize();
        if (!value) return "defer";
        Powers exp(n);
        for (int& e:exp) if (!(stream >> e) || e<0 || e>10000) return "defer";
        const int degree = std::accumulate(exp.begin(), exp.end(), 0);
        order = std::min(order,degree);
        terms.push_back({exp,value,degree});
    }
    // The input contract is canonical sparse form with distinct monomials.
    // Check this here too, since duplicate terms could invalidate the rank test.
    std::sort(terms.begin(), terms.end(), [](auto& a,auto& b){return a.powers>b.powers;});
    if (std::adjacent_find(terms.begin(),terms.end(),[](auto& a,auto& b){return a.powers==b.powers;})!=terms.end())
        return "defer";
    if (order < 1) return "defer";
    if (order == 1) return "smooth\n" + equation(terms);
    if (n < 2) return "defer";
    Terms tangent;
    for (const auto& t : terms) if (t.degree == order) tangent.push_back(t);
    std::string reason;
    const auto cert = jacobian_certificate(tangent,n,order,gpu,reason);
    if (cert.empty()) return "defer\n" + std::to_string(order) + "\n" + reason + "\n" + equation(tangent);
    std::ostringstream report;
    report << "ordinary\n" << order << '\n' << cert << '\n' << equation(tangent);
    for (int k=0; k<n; ++k)
        report << '\n' << equation(terms,k,order) << '\n' << equation(tangent,k,order);
    return report.str();
}
} // namespace

extern "C" void* resolution_ordinary(const char* input, RankKernel kernel, int device, int minimum_columns) noexcept {
    if (!input) return nullptr;
    try {
        const std::string report = compute(input,{kernel,device,minimum_columns});
        char* result = static_cast<char*>(std::malloc(report.size()+1));
        if (result) std::memcpy(result,report.c_str(),report.size()+1);
        return result;
    } catch (...) {
        return nullptr;
    }
}
extern "C" void resolution_free(void* result) { std::free(result); }
