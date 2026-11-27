// Packed and arbitrary-precision exponent substitutions for toric charts.
#include "chart_equation.hpp"
#include <array>
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <gmpxx.h>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
template<int N>
void transform(const int32_t* exps,const int32_t* rays,const int32_t* cones,
               const int32_t* orders,int terms,int charts,int32_t* out) noexcept {
    for (int chart=0;chart<charts;++chart) {
        const auto* cone=cones+N*chart;
        std::array<const int32_t*,N> directions;
        for (int axis=0;axis<N;++axis) directions[axis]=rays+N*cone[axis];
        for (int term=0;term<terms;++term) {
            const auto* e=exps+N*term;
            auto* dest=out+N*(chart*terms+term);
            for (int axis=0;axis<N;++axis) {
                const auto* r=directions[axis];
                int32_t value=0;
                for (int j=0;j<N;++j) value+=e[j]*r[j];
                dest[axis]=value-orders[cone[axis]];
            }
        }
    }
}
}

extern "C" __attribute__((visibility("default"))) void resolution_chart_cpu_nd(const int32_t* exps,const int32_t* rays,
                                        const int32_t* cones,const int32_t* orders,
                                        int terms,int charts,int dimension,int32_t* out) noexcept {
    using Kernel = decltype(&transform<2>);
    static const Kernel table[]={transform<2>,transform<3>,transform<4>,transform<5>,
                                 transform<6>,transform<7>,transform<8>};
    if(dimension>=2 && dimension<=8)
        table[dimension-2](exps,rays,cones,orders,terms,charts,out);
}

// The original three-dimensional ABI is retained for external callers.
extern "C" __attribute__((visibility("default"))) void resolution_chart_cpu(const int32_t* exps,const int32_t* rays,
                                     const int32_t* cones,const int32_t* orders,
                                     int terms,int charts,int32_t* out) noexcept {
    transform<3>(exps,rays,cones,orders,terms,charts,out);
}
extern "C" __attribute__((visibility("default"))) int resolution_chart_valid(const int32_t* output, int length) noexcept {
    for(int i=0;i<length;++i) if(output[i]<0) return 0;
    return 1;
}

// Nonnegative dot products and their partial sums fit if this upper bound fits.
extern "C" __attribute__((visibility("default"))) bool resolution_chart_packed(const int32_t* exponents, const int32_t* rays,
                                        int terms, int ray_count, int dimension) noexcept {
    if (dimension < 2 || dimension > 8 || terms < 1 || ray_count < 1) return false;
    int32_t largest_ray = *std::max_element(rays, rays + size_t(ray_count) * dimension);
    if (!largest_ray) return true;
    const int64_t limit = std::numeric_limits<int32_t>::max() / largest_ray;
    for (int i = 0; i < terms; ++i) {
        int64_t degree = 0;
        for (int j = 0; j < dimension; ++j) degree += *exponents++;
        if (degree > limit) return false;
    }
    return true;
}

namespace {
using Integers = std::vector<mpz_class>;
using Digits = std::vector<int32_t>;
using ChartDevice = int (*)(const int32_t*, const int32_t*, const int32_t*, const int32_t*, int32_t*);

Integers read_integers(std::istream& input, size_t count) {
    Integers values(count);
    for (auto& value : values)
        if (!(input >> value) || value < 0)
            throw std::invalid_argument("Expected a nonnegative integer chart table.");
    return values;
}

constexpr unsigned limb_bits = 14;

struct Plane { mp_bitcnt_t shift; Digits digits; };

std::vector<Plane> limb_planes(const Integers& values) {
    mpz_class occupied = 0;
    for (const auto& value : values) occupied |= value;
    std::vector<Plane> result;
    mp_bitcnt_t shift = 0;
    while (occupied > 0) {
        // Jump directly to the next occupied base-2^14 digit.
        const auto skip = mpz_scan1(occupied.get_mpz_t(), 0) / limb_bits * limb_bits;
        shift += skip;
        Plane plane{shift, Digits(values.size())};
        for (size_t i = 0; i < values.size(); ++i) {
            // Reading a fixed 14-bit slice avoids allocating value >> shift
            // for every coordinate, including high zeros.
            unsigned digit = 0;
            for (unsigned bit = 0; bit < limb_bits; ++bit)
                digit |= unsigned(mpz_tstbit(values[i].get_mpz_t(), shift + bit)) << bit;
            plane.digits[i] = static_cast<int32_t>(digit);
        }
        result.push_back(std::move(plane));
        occupied >>= skip + limb_bits;
        shift += limb_bits;
    }
    return result;
}

enum class Verification { checked, unavailable, mismatch };

Verification certify_device(const Integers& exponents, const Integers& rays, const Integers& reference,
                            size_t terms, size_t ray_count, size_t dim, ChartDevice run) {
    const size_t groups = (ray_count+dim-1) / dim;
    Digits indices(groups*dim), zero_orders(ray_count), output(groups*dim*terms);
    for (size_t i = 0; i < indices.size(); ++i)
        indices[i] = static_cast<int32_t>(std::min(i, ray_count-1));
    Integers accumulated(output.size());
    const auto right = limb_planes(rays);
    for (const auto& x : limb_planes(exponents)) {
        for (const auto& y : right) {
            // With at most eight coordinates the digit dot product fits int32.
            if (!run(x.digits.data(), y.digits.data(), indices.data(), zero_orders.data(), output.data()))
                return Verification::unavailable;
            const auto shift = x.shift + y.shift;
            for (size_t i = 0; i < output.size(); ++i) {
                if (output[i] < 0) return Verification::mismatch;
                if (output[i]) accumulated[i] += mpz_class(output[i]) << shift;
            }
        }
    }
    for (size_t i = 0; i < ray_count; ++i) {
        const size_t offset = i/dim * terms*dim + i%dim;
        for (size_t j = 0; j < terms; ++j)
            if (accumulated[offset + j*dim] != reference[i*terms + j])
                return Verification::mismatch;
    }
    return Verification::checked;
}

std::vector<std::string> read_words(std::istream& in, size_t count) {
    std::vector<std::string> words(count);
    for (auto& word : words) if (!(in >> word)) throw std::invalid_argument("Incomplete chart equation data.");
    return words;
}

char* owned_text(const std::string& text) {
    char* result = static_cast<char*>(std::malloc(text.size()+1));
    if (result) std::memcpy(result, text.c_str(), text.size()+1);
    return result;
}
} // namespace

// Read the transformed buffer produced by the selected CPU or GPU kernel.
extern "C" __attribute__((visibility("default"))) char* resolution_chart_equations(const char* input, const int32_t* powers,
                                          size_t terms, size_t charts, size_t dimension) noexcept {
    try {
        if (!input || !powers || !terms || !charts || dimension < 2 || dimension > 8 ||
            terms > 6000000 / dimension / charts)
            throw std::invalid_argument("Invalid packed chart dimensions.");
        std::istringstream in(input);
        const auto names = read_words(in, dimension);
        const auto coefficients = read_words(in, terms);
        std::string result;
        for (size_t i = 0; i < charts; ++i)
            atlas::append_chart_equation(result, powers+i*terms*dimension, coefficients, names);
        return owned_text(result);
    } catch (const std::exception& e) {
        return owned_text(std::string("error ") + e.what());
    } catch (...) {
        return nullptr;
    }
}

// Reuse the exact ray/term valuations from the fan. A requested device
// recomputes them with packed digit products; certify its entire result
// before exposing the transformed exponents. Returned text uses resolution_free.
extern "C" __attribute__((visibility("default"))) char* resolution_chart_wide(const char* input, size_t terms, size_t ray_count,
                                       size_t dimension, ChartDevice run) noexcept {
    try {
        if (!input || !terms || !ray_count || !dimension)
            throw std::invalid_argument("Expected nonempty chart tables.");
        if (ray_count > std::numeric_limits<size_t>::max() / terms)
            throw std::length_error("The chart table size exceeds the address space.");
        std::istringstream in(input);
        const auto orders = read_integers(in, ray_count);
        auto values = read_integers(in, ray_count*terms);
        Verification verification = Verification::checked;
        if (run) {
            if (dimension < 2 || dimension > 8 || ray_count > 100000 ||
                    terms > 6000000 / dimension / ((ray_count+dimension-1)/dimension))
                throw std::invalid_argument("The ray table exceeds the chart device batch size.");
            const auto exponents = read_integers(in, terms*dimension);
            const auto rays = read_integers(in, ray_count*dimension);
            verification = certify_device(exponents, rays, values, terms, ray_count, dimension, run);
        }
        in >> std::ws;
        const bool equations = !in.eof();
        std::ostringstream out;
        switch (verification) {
            case Verification::checked: out << "ok\n"; break;
            case Verification::unavailable: out << "unavailable\n"; break;
            case Verification::mismatch: out << "mismatch\n"; break;
        }
        for (size_t i = 0; i < ray_count; ++i) {
            for (size_t j = 0; j < terms; ++j) {
                auto& value = values[i*terms+j];
                if (value < orders[i])
                    throw std::invalid_argument("A ray order exceeds the valuation of an input monomial.");
                value -= orders[i];
                if (!equations) {
                    if (j) out << ' ';
                    out << value;
                }
            }
            if (!equations) out << '\n';
        }
        if (!equations) return owned_text(out.str());

        std::string operation;
        size_t charts;
        if (!(in >> operation >> charts) || operation != "equations" || !charts ||
            terms > std::numeric_limits<size_t>::max()/dimension)
            throw std::invalid_argument("Incorrect chart equation table.");
        const auto names = read_words(in, dimension);
        const auto coefficients = read_words(in, terms);
        Integers powers(terms*dimension);
        std::string result = out.str();
        for (size_t i = 0; i < charts; ++i) {
            for (size_t j = 0; j < dimension; ++j) {
                size_t ray;
                if (!(in >> ray) || ray >= ray_count) throw std::invalid_argument("Invalid chart ray index.");
                for (size_t k = 0; k < terms; ++k)
                    powers[k*dimension+j] = values[ray*terms+k];
            }
            atlas::append_chart_equation(result, powers.data(), coefficients, names);
        }
        in >> std::ws;
        if (!in.eof()) throw std::invalid_argument("Incorrect chart table dimensions.");
        return owned_text(result);
    } catch (const std::exception& e) {
        return owned_text(std::string("error ") + e.what());
    } catch (...) {
        return nullptr;
    }
}
