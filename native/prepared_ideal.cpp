// Immutable operands for repeated arithmetic in a live QQ polynomial ring.
#include "prepared_ideal.hpp"
#include "ideal_arithmetic.hpp"
#include "ideal_powers.hpp"
#include "local_io.hpp"
#include "sparse_flint.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace atlas::ideal_arithmetic {
namespace {
using local::Basis;
using Power = ideal_powers::PowerResult;

struct Ring {
    size_t width;
    fmpq_mpoly_ctx_t ctx;
    explicit Ring(size_t n) : width(n) { fmpq_mpoly_ctx_init(ctx, n, ORD_LEX); }
    ~Ring() { fmpq_mpoly_ctx_clear(ctx); }
};
using RingPtr = std::shared_ptr<const Ring>;

struct Generator {
    RingPtr ring;
    PolyValue value;
    explicit Generator(RingPtr parent) : ring(std::move(parent)), value(ring->ctx) {}
};
using PolyPtr = std::shared_ptr<const Generator>;

struct Generators {
    std::vector<PolyPtr> polys;
    size_t terms = 0;
    bool unit = false;
    // Bounds the bits needed for coefficients and exponents of one term.
    size_t term_bits = 0;

    GeneratorsKind kind() const {
        if (unit) return GeneratorsKind::unit;
        return polys.empty() ? GeneratorsKind::zero : GeneratorsKind::general;
    }

    void append(PolyPtr f) {
        terms += fmpq_mpoly_length(f->value.p, f->ring->ctx);
        polys.push_back(std::move(f));
    }
};

Generators pack(const RingPtr& ring, const Basis& rows) {
    local::SparseLexRing converter(ring->width);
    Generators result;
    result.polys.reserve(rows.size());
    for (const auto& row : rows) {
        if (row.empty()) continue;
        auto f = std::make_shared<Generator>(ring);
        if (unit_generator(row)) {
            fmpq_mpoly_one(f->value.p, ring->ctx);
            return {{std::move(f)}, 1, true, 1};
        }
        converter.write(f->value.p, row);
        const auto* p = f->value.p;
        const auto coeff_bits = fmpz_bits(fmpq_numref(p->content)) + fmpz_bits(fmpq_denref(p->content)) +
                                std::abs(fmpz_mpoly_max_bits(p->zpoly));
        result.term_bits = std::max<size_t>(result.term_bits, coeff_bits + p->zpoly->bits);
        result.append(std::move(f));
    }
    return result;
}

// Input ideals retain their declared rows for powers and replay. Their packed
// form omits zero generators and represents the unit ideal by (1).
class Ideal {
    enum class Source { sparse, packed, power };
    Source source;
    mutable Generators generators;
    mutable Basis rows;
    const std::optional<Power> power;
    mutable std::once_flag to_packed, to_sparse;
public:
    RingPtr ring;
    // Accumulated input size used to limit work in the prompt.
    size_t operand_bytes;
    Ideal(RingPtr r, Basis v, size_t bytes) : source(Source::sparse), rows(std::move(v)),
        ring(std::move(r)), operand_bytes(bytes) {}
    Ideal(RingPtr r, Generators v, size_t bytes) : source(Source::packed), generators(std::move(v)),
        ring(std::move(r)), operand_bytes(bytes) {}
    Ideal(RingPtr r, Power v) : source(Source::power), power(std::move(v)),
        ring(std::move(r)), operand_bytes(0) {}

    const Generators& packed() const {
        if (source != Source::packed)
            std::call_once(to_packed, [&]{ generators = pack(ring, sparse()); });
        return generators;
    }
    const Basis& sparse() const {
        if (source != Source::sparse) std::call_once(to_sparse, [&]{
            if (source == Source::power) {
                rows = power->materialize();
            } else {
                local::SparseLexRing converter(ring->width);
                Basis decoded;
                decoded.reserve(generators.polys.size());
                for (const auto& f : generators.polys) decoded.push_back(converter.read(f->value.p));
                rows = std::move(decoded);
            }
        });
        return rows;
    }
    size_t size() const { return source == Source::packed ? generators.polys.size() : sparse().size(); }
    std::string expressions(const std::vector<std::string>& names) const {
        std::string text;
        if (source == Source::power && power->is_packed()) {
            // The packed result is immutable. Its formatter and sparse()
            // can both read its integer coefficients and monomials concurrently.
            for (const auto& f : power->expressions(names)) { text += f; text += '\n'; }
        } else {
            for (const auto& f : sparse()) { text += local::format_polynomial(f,names); text += '\n'; }
        }
        return text;
    }
};

std::unique_ptr<Ideal> binary(const Ideal& a, const Ideal& b, bool product) {
    if (a.ring->width != b.ring->width) throw std::invalid_argument("Different numbers of ideal coordinates.");
    const size_t bytes = a.operand_bytes + b.operand_bytes;
    if (a.ring->width > 32 || bytes > 32768) return {};
    const auto& left = a.packed();
    const auto& right = b.packed();
    const auto action = combine(product,left.kind(),right.kind());
    if (action == Combination::left || action == Combination::right) {
        const auto& selected = action == Combination::left ? left : right;
        return std::make_unique<Ideal>(a.ring, selected, selected.polys.empty() ? 0 : bytes);
    }
    Generators result;
    if (action == Combination::concatenate) {
        if (left.polys.size() + right.polys.size() > 128 || left.terms + right.terms > 1024)
            return {};
        result.polys = concatenate_generators(left.polys,right.polys.begin(),right.polys.end());
        result.terms = left.terms + right.terms;
        result.term_bits = std::max(left.term_bits, right.term_bits);
    } else {
        // For a given exponent, at most min(#terms in f, #terms in g)
        // coefficient products can contribute. The 1024-product limit
        // accounts for at most ten extra bits when adding those products.
        result.term_bits = left.term_bits + right.term_bits + 10;
        if (left.polys.size() > 128 / right.polys.size() ||
            left.terms > 1024 / right.terms || result.term_bits > 8192) return {};
        result.polys.reserve(left.polys.size() * right.polys.size());
        multiply_generator_pairs(left.polys,right.polys,[&](const auto& f, const auto& g) {
            auto h = std::make_shared<Generator>(a.ring);
            fmpq_mpoly_mul(h->value.p, f->value.p, g->value.p, a.ring->ctx);
            result.append(std::move(h));
        });
    }
    return std::make_unique<Ideal>(a.ring, std::move(result), bytes);
}

thread_local std::string error;
template<typename F>
auto guarded(F f) noexcept -> decltype(f()) {
    try {
        error.clear();
        return f();
    } catch (const std::exception& e) { error = e.what(); }
    catch (...) { error = "Ideal arithmetic failed."; }
    return {};
}

Ideal& handle(void* p) {
    if (!p) throw std::invalid_argument("Missing prepared ideal.");
    return *static_cast<Ideal*>(p);
}

char* copy_text(const std::string& text) {
    auto* out = static_cast<char*>(std::malloc(text.size()+1));
    if (!out) throw std::bad_alloc();
    std::memcpy(out,text.c_str(),text.size()+1);
    return out;
}
} // namespace

} // namespace atlas::ideal_arithmetic

using namespace atlas::ideal_arithmetic;

extern "C" void* atlas_ideal_prepare(size_t width, const char* rows, size_t count) noexcept {
    return guarded([&]() -> void* {
        if (!rows) throw std::invalid_argument("Missing ideal input.");
        std::string_view text(rows);
        const size_t bytes = text.size();
        if (width > 32 || count > 128 || bytes > 32768) return nullptr;
        atlas::local::Basis values;
        values.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            size_t end = text.find('\n');
            if (end == text.npos && i + 1 != count) throw std::invalid_argument("Missing ideal generator.");
            values.push_back(atlas::local::decode_row(text.substr(0,end), width));
            text = end == text.npos ? std::string_view{} : text.substr(end+1);
        }
        if (!text.empty()) throw std::invalid_argument("Extra ideal generators.");
        auto result = std::make_unique<Ideal>(std::make_shared<Ring>(width), std::move(values), bytes);
        (void)result->packed();
        return result.release();
    });
}
extern "C" void* atlas_ideal_binary(void* a, void* b, int product) noexcept {
    return guarded([&]{ return binary(handle(a),handle(b),product != 0).release(); });
}
extern "C" void* atlas_ideal_power_small(void* p, unsigned long exponent, int lex) noexcept {
    return guarded([&]() -> void* {
        const auto& a = handle(p);
        auto result = atlas::ideal_powers::bounded_ordinary_power(a.sparse(),a.ring->width,lex != 0,mpz_class(exponent));
        if (!result) return nullptr;
        return new Ideal(a.ring,std::move(*result));
    });
}
extern "C" size_t atlas_ideal_size(void* p) noexcept {
    return guarded([&]{ return handle(p).size(); });
}
extern "C" char* atlas_ideal_rows(void* p) noexcept {
    return guarded([&]{
        std::string text;
        for (const auto& f : handle(p).sparse()) { text += atlas::local::encode_row(f); text += '\n'; }
        return copy_text(text);
    });
}
extern "C" char* atlas_ideal_expressions(void* p, const char* coordinate_names) noexcept {
    return guarded([&]{
        const auto& ideal = handle(p);
        if (!coordinate_names) throw std::invalid_argument("Missing ideal coordinate names.");
        std::istringstream input(coordinate_names);
        std::vector<std::string> names;
        for (std::string name; input >> name;) names.push_back(std::move(name));
        if (names.size() != ideal.ring->width) throw std::invalid_argument("Wrong number of ideal coordinate names.");
        return copy_text(ideal.expressions(names));
    });
}
extern "C" void atlas_ideal_release(void* p) noexcept { delete static_cast<Ideal*>(p); }
extern "C" const char* atlas_ideal_error() noexcept { return error.c_str(); }
