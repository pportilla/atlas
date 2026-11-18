// Immutable operands for repeated arithmetic in a live QQ polynomial ring.
#include "prepared_ideal.hpp"
#include "ideal_arithmetic.hpp"
#include "ideal_powers.hpp"
#include "local_io.hpp"
#include "polynomial_gcd.hpp"
#include "sparse_flint.hpp"
#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace atlas::ideal_arithmetic {
namespace {
using local::Basis;
using Power = ideal_powers::PowerResult;

struct Ring {
    size_t width;
    bool named_parent;
    // The converter reuses exponent and coefficient scratch in this context.
    // FLINT arithmetic itself only reads the context and needs no lock.
    mutable std::mutex conversion_mutex;
    mutable std::optional<local::SparseLexRing> converter;
    mutable std::once_flag init_graded;
    mutable std::optional<local::SparseFlintRing> graded;
    const fmpq_mpoly_ctx_struct* ctx;
    explicit Ring(size_t n, bool named = false) : width(n), named_parent(named),
        ctx(n <= 32 ? converter.emplace(n).ctx : nullptr) {}

    local::SparseFlintRing& ordered(bool lex) const {
        if (lex) return *converter;
        std::call_once(init_graded,[&]{ graded.emplace(width,ORD_DEGREVLEX); });
        return *graded;
    }
};
using RingPtr = std::shared_ptr<const Ring>;

struct Generator {
    RingPtr ring;
    PolyValue value;
    explicit Generator(RingPtr parent) : ring(std::move(parent)), value(ring->ctx) {}
};
using PolyPtr = std::shared_ptr<const Generator>;

struct PackedGenerator {
    PolyPtr value;
    size_t term_bits = 0;
    bool unit = false;
};

PackedGenerator pack_generator(const RingPtr& ring, const local::Polynomial& row,
                               local::SparseLexRing& converter) {
    if (row.empty()) return {};
    auto f = std::make_shared<Generator>(ring);
    if (unit_generator(row)) {
        fmpq_mpoly_one(f->value.p, ring->ctx);
        return {std::move(f), 1, true};
    }
    converter.write(f->value.p, row);
    const auto* p = f->value.p;
    const auto bits = fmpz_bits(fmpq_numref(p->content)) + fmpz_bits(fmpq_denref(p->content)) +
                      std::abs(fmpz_mpoly_max_bits(p->zpoly)) + p->zpoly->bits;
    return {std::move(f), static_cast<size_t>(bits), false};
}

// Most large session rows contain only machine integers. Write their terms
// straight into FLINT's primitive integer part; the general reader handles
// other coefficients, exponent sizes and input spellings.
bool read_integer_row(PolyValue& out, std::string_view row, size_t width) {
    auto* poly = out.p->zpoly;
    const auto* ctx = out.ctx->zctx;
    const char* pos = row.data();
    const char* end = pos + row.size();
    std::vector<ulong> powers(width);
    fmpz_mpoly_fit_length(poly,1 + std::count(pos,end,';'),ctx);
    while (pos < end) {
        slong coeff;
        const auto c = std::from_chars(pos,end,coeff);
        if (c.ec != std::errc() || !coeff) return false;
        pos = c.ptr;
        for (auto& power : powers) {
            if (pos == end || *pos++ != ' ') return false;
            const auto e = std::from_chars(pos,end,power);
            if (e.ec != std::errc()) return false;
            pos = e.ptr;
        }
        fmpz_mpoly_push_term_si_ui(poly,coeff,powers.data(),ctx);
        if (pos != end && *pos++ != ';') return false;
    }
    fmpz_mpoly_sort_terms(poly,ctx);
    // A repeated exponent must be diagnosed by the normal sparse reader.
    if (!fmpz_mpoly_is_canonical(poly,ctx)) return false;
    fmpq_one(out.p->content);
    fmpq_mpoly_reduce(out.p,out.ctx);
    return true;
}

// The sparse polynomial is exact, including zero and the scale of constants.
// Its packed generator can be shared by every ideal containing this value.
// Over-budget values retain their sparse transport for the isolated worker.
struct PreparedPolynomial {
    RingPtr ring;
    mutable size_t row_bytes = 0;
    mutable std::optional<std::string> worker_row;
    mutable local::Polynomial sparse;
    mutable PackedGenerator packed;
    // A computed polynomial owns its already normalized FLINT value. Building
    // an ideal or exporting the row may also require the older representations.
    std::optional<PolyValue> result;
    mutable std::once_flag for_ideal, for_lex, for_graded, check_gcd_size;
    mutable std::unique_ptr<PolyValue> lex_value, graded_value;
    mutable bool live_gcd = false;

    PreparedPolynomial(RingPtr parent, std::string_view text) : ring(std::move(parent)), row_bytes(text.size()) {
        if (ring->width > 32 || row_bytes > 4096 || std::count(text.begin(),text.end(),';') >= 512) {
            worker_row.emplace(text);
        } else {
            sparse = local::decode_row(text,ring->width);
            std::lock_guard lock(ring->conversion_mutex);
            packed = pack_generator(ring,sparse,*ring->converter);
        }
    }

    PreparedPolynomial(RingPtr parent, PolyValue answer) : ring(std::move(parent)), result(std::move(answer)) {}

    void prepare_for_ideal() const {
        if (result) std::call_once(for_ideal,[&]{
            auto& converter = ring->ordered(fmpq_mpoly_ctx_ord(result->ctx) == ORD_LEX);
            std::lock_guard lock(ring->conversion_mutex);
            sparse = converter.read(result->p);
            auto text = local::encode_row(sparse);
            row_bytes = text.size();
            if (row_bytes > 4096 || sparse.size() > 512) worker_row.emplace(std::move(text));
            else packed = pack_generator(ring,sparse,*ring->converter);
        });
    }

    bool fits_live_gcd(bool lex) const {
        std::call_once(check_gcd_size,[&]{
            if (!ring->ctx || (!result && row_bytes > (1u << 20))) return;
            // General rational rows can accumulate a large common denominator
            // during packing. Do that conversion in the isolated worker.
            if (!result && worker_row && row_bytes > 32768 && worker_row->find('/') != std::string::npos) return;
            const auto& f = result ? *result : packed.value ? packed.value->value : gcd_operand(lex);
            const auto* p = f.p;
            // A short sparse row can still induce a dense univariate GCD of
            // enormous degree. Use the actual degree and primitive coefficient
            // sizes once per immutable input, even when it is itself a GCD.
            if (p->zpoly->length > 16384 || p->zpoly->bits > FLINT_BITS ||
                !fmpq_mpoly_total_degree_fits_si(p,f.ctx) || fmpq_mpoly_total_degree_si(p,f.ctx) > 128)
                return;
            const slong bits = fmpz_mpoly_max_bits(p->zpoly);
            live_gcd = bits >= -4096 && bits <= 4096 &&
                       fmpz_bits(fmpq_numref(p->content)) <= 4096 &&
                       fmpz_bits(fmpq_denref(p->content)) <= 4096;
        });
        return live_gcd;
    }

    bool is_zero() const {
        if (result) return fmpq_mpoly_is_zero(result->p,result->ctx);
        return worker_row ? local::decode_row(*worker_row,ring->width).empty() : sparse.empty();
    }

    const PolyValue& gcd_operand(bool lex) const {
        auto& converter = ring->ordered(lex);
        if (result && result->ctx == converter.ctx) return *result;
        // The ideal generator of a nonzero constant is 1. Its rational
        // associate gives exactly the same monic GCD as the original value.
        if (!result && lex && packed.value) return packed.value->value;
        auto& value = lex ? lex_value : graded_value;
        std::call_once(lex ? for_lex : for_graded,[&]{
            auto f = std::make_unique<PolyValue>(converter.ctx);
            if (result || packed.value) {
                const PolyValue& source = result ? *result : packed.value->value;
                std::vector<slong> images(ring->width);
                std::iota(images.begin(),images.end(),0);
                fmpq_mpoly_compose_fmpq_mpoly_gen(f->p,source.p,images.data(),source.ctx,f->ctx);
            } else if (worker_row && !read_integer_row(*f,*worker_row,ring->width)) {
                const auto decoded = local::decode_row(*worker_row,ring->width);
                std::lock_guard lock(ring->conversion_mutex);
                converter.write(f->p,decoded);
            }
            value = std::move(f);
        });
        return *value;
    }
};
using PreparedPtr = std::shared_ptr<const PreparedPolynomial>;

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

    void add_generator(const PackedGenerator& f) {
        if (f.unit) {
            polys.assign(1,f.value);
            terms = term_bits = 1;
            unit = true;
        } else if (f.value) {
            term_bits = std::max(term_bits, f.term_bits);
            append(f.value);
        }
    }
};

Generators pack(const RingPtr& ring, const Basis& rows) {
    std::lock_guard lock(ring->conversion_mutex);
    Generators result;
    result.polys.reserve(rows.size());
    for (const auto& row : rows) {
        result.add_generator(pack_generator(ring,row,*ring->converter));
        if (result.unit) break;
    }
    return result;
}

// Input ideals retain their declared rows for powers and replay. Their packed
// form omits zero generators and represents the unit ideal by (1).
class Ideal {
    enum class Source { sparse, declared, packed, power };
    Source source;
    mutable Generators generators;
    mutable Basis rows;
    std::vector<PreparedPtr> declared;
    const std::optional<Power> power;
    mutable std::once_flag to_packed, to_sparse;
public:
    RingPtr ring;
    // Accumulated input size used to limit work in the prompt.
    size_t operand_bytes;
    Ideal(RingPtr r, Basis v, size_t bytes) : source(Source::sparse), rows(std::move(v)),
        ring(std::move(r)), operand_bytes(bytes) {}
    Ideal(RingPtr r, std::vector<PreparedPtr> v, size_t bytes) : source(Source::declared),
        declared(std::move(v)), ring(std::move(r)), operand_bytes(bytes) {
        generators.polys.reserve(declared.size());
        for (const auto& f : declared) {
            generators.add_generator(f->packed);
            if (generators.unit) break;
        }
    }
    Ideal(RingPtr r, Generators v, size_t bytes) : source(Source::packed), generators(std::move(v)),
        ring(std::move(r)), operand_bytes(bytes) {}
    Ideal(RingPtr r, Power v) : source(Source::power), power(std::move(v)),
        ring(std::move(r)), operand_bytes(0) {}

    const Generators& packed() const {
        if (source != Source::packed && source != Source::declared)
            std::call_once(to_packed, [&]{ generators = pack(ring,sparse()); });
        return generators;
    }
    const Basis& sparse() const {
        if (source != Source::sparse) std::call_once(to_sparse, [&]{
            if (source == Source::declared) {
                Basis decoded;
                decoded.reserve(declared.size());
                for (const auto& f : declared) decoded.push_back(f->sparse);
                rows = std::move(decoded);
            } else if (source == Source::power) {
                rows = power->materialize();
            } else {
                std::lock_guard lock(ring->conversion_mutex);
                Basis decoded;
                decoded.reserve(generators.polys.size());
                for (const auto& f : generators.polys) decoded.push_back(ring->converter->read(f->value.p));
                rows = std::move(decoded);
            }
        });
        return rows;
    }
    size_t size() const {
        if (source == Source::declared) return declared.size();
        return source == Source::packed ? generators.polys.size() : sparse().size();
    }
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
    if (a.ring->named_parent && b.ring->named_parent && a.ring != b.ring)
        throw std::invalid_argument("The ideals belong to different parent rings.");
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

extern "C" void* atlas_ideal_ring(size_t width) noexcept {
    return guarded([&]{ return new RingPtr(std::make_shared<Ring>(width,true)); });
}
extern "C" void atlas_ideal_ring_release(void* p) noexcept { delete static_cast<RingPtr*>(p); }
extern "C" void* atlas_ideal_polynomial(void* ring, const char* row) noexcept {
    return guarded([&]{
        if (!ring || !row) throw std::invalid_argument("Missing polynomial parent or value.");
        return new PreparedPtr(std::make_shared<PreparedPolynomial>(*static_cast<RingPtr*>(ring),row));
    });
}
extern "C" void atlas_ideal_polynomial_release(void* p) noexcept { delete static_cast<PreparedPtr*>(p); }
extern "C" void* atlas_ideal_polynomial_gcd(void* left, void* right, int lex) noexcept {
    return guarded([&]() -> void* {
        if (!left || !right) throw std::invalid_argument("Missing polynomial GCD operand.");
        const auto& a = **static_cast<PreparedPtr*>(left);
        const auto& b = **static_cast<PreparedPtr*>(right);
        if (a.ring != b.ring) throw std::invalid_argument("The polynomials belong to different parent rings.");
        if (!a.fits_live_gcd(lex) || !b.fits_live_gcd(lex)) return nullptr;
        const auto& f = a.gcd_operand(lex), &g = b.gcd_operand(lex);
        atlas::PolyValue answer(f.ctx);
        atlas::monic_polynomial_gcd(answer.p,f.p,g.p,f.ctx);
        return new PreparedPtr(std::make_shared<PreparedPolynomial>(a.ring,std::move(answer)));
    });
}
extern "C" int atlas_ideal_polynomial_is_zero(void* p) noexcept {
    const int zero = guarded([&]{
        if (!p) throw std::invalid_argument("Missing polynomial.");
        return int((**static_cast<PreparedPtr*>(p)).is_zero());
    });
    return error.empty() ? zero : -1;
}
extern "C" char* atlas_ideal_polynomial_row(void* p) noexcept {
    return guarded([&]{
        if (!p) throw std::invalid_argument("Missing polynomial.");
        const auto& f = **static_cast<PreparedPtr*>(p);
        f.prepare_for_ideal();
        return f.worker_row ? copy_text(*f.worker_row) : copy_text(atlas::local::encode_row(f.sparse));
    });
}
extern "C" void* atlas_ideal_from_polynomials(void* parent, void* const* polys, size_t count) noexcept {
    return guarded([&]() -> void* {
        if (!parent || (count && !polys)) throw std::invalid_argument("Missing ideal parent or generators.");
        const auto& ring = *static_cast<RingPtr*>(parent);
        size_t terms = 0, bytes = 0;
        bool eligible = count <= 128 && ring->width <= 32;
        for (size_t i = 0; i < count; ++i) {
            if (!polys[i]) throw std::invalid_argument("Missing polynomial generator.");
            const auto& f = *static_cast<PreparedPtr*>(polys[i]);
            if (f->ring != ring) throw std::invalid_argument("The generator belongs to a different parent ring.");
            if (eligible) {
                f->prepare_for_ideal();
                terms += f->sparse.size();
                bytes += f->row_bytes;
                eligible = !f->worker_row && terms <= 512 && bytes <= 4096;
            }
        }
        if (!eligible) return nullptr;
        std::vector<PreparedPtr> declared;
        declared.reserve(count);
        for (size_t i = 0; i < count; ++i) declared.push_back(*static_cast<PreparedPtr*>(polys[i]));
        return new Ideal(ring,std::move(declared),bytes+count);
    });
}
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
