// Reduced Groebner bases over Q through modular images, certified.
//
// 1. The generators are homogenized with a new last variable.
// 2. For one prime the homogeneous basis is computed by Buchberger's
//    algorithm, degree by degree. Every S-polynomial that does not reduce to
//    zero leaves behind the list of row operations that produced its
//    remainder: the replay program. It contains no pair that turned out to be
//    useless and no search for reducers.
// 3. The program is run for further primes, independently of one another
//    (CPU threads, or a batch handed to a GPU). A prime whose run does not
//    show the leading monomials and the zero coefficients of the trace is
//    dropped.
// 4. The coefficients are lifted by the Chinese remainder theorem and
//    rational reconstruction.
// 5. The lifted basis G is certified over Q: G is a Groebner basis, the
//    generators reduce to zero by G, and the leading monomials of G are those
//    of the basis for the trace prime, which was computed in full. For a
//    homogeneous ideal this implies that G is its Groebner basis (Arnold,
//    J. Symbolic Comput. 35 (2003)); the proof is in KERNEL.md, section 7.
// 6. Setting the new variable to one gives a Groebner basis of the original
//    ideal for the original order; it is then reduced.
//
// Steps 2 and 3 first use the F4 trace of f4_trace.cpp (KERNEL.md, section
// 7.5): one matrix per degree, randomized elimination checked against the
// Hilbert function, replayed matrix by matrix for the other primes. When it
// does not apply (the solve asks for a device, the order does not compare
// degrees first, a generator is constant) or gives up, the Buchberger trace
// below runs instead.
//
// Generators that are not homogeneous and whose homogenizations miss the
// Hilbert bound may skip step 1, when the homogenized ideal would need a much
// larger basis: they are then traced as they are, and the lifted elements are
// certified through the rows of the trace (KERNEL.md 7.6).
//
// Nothing here is probabilistic: an unlucky prime, or an unlucky random
// combination in the F4 trace, can only cost time.
#include "f4_trace.hpp"
#include "groebner.hpp"
#include "groebner_verify.hpp"
#include "modular_replay.hpp"

#include <flint/fmpq.h>
#include <flint/nmod.h>
#include <flint/ulong_extras.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <system_error>
#include <thread>

namespace atlas {
namespace {

// FLINT owns the large numerator and denominator allocated during rational
// reconstruction. Release each pair even when constructing the basis throws.
struct LiftedRational {
    fmpq value;
    LiftedRational() { fmpq_init(&value); }
    ~LiftedRational() { fmpq_clear(&value); }
    LiftedRational(const LiftedRational&) = delete;
    LiftedRational& operator=(const LiftedRational&) = delete;
};

struct Slot {
    uint32_t offset = 0, length = 0;
};

using TraceScreen = int (*)(uint32_t, uint32_t, uint32_t, const uint32_t*, const uint32_t*,
                            const uint32_t*, const uint32_t*, const uint32_t*, uint32_t*);
thread_local TraceScreen trace_screen = nullptr;
using TraceProfile = void (*)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                              double, double, double);
thread_local TraceProfile trace_profile = nullptr;
thread_local uint64_t cpu_trace_batches = 0, cpu_trace_pairs = 0;
thread_local uint64_t certified_bases = 0, certification_attempts = 0;
thread_local uint64_t lift_ns = 0, verification_ns = 0;
#include "trace_rows.inc"

// The coefficients of f modulo p, divided by the leading one. False when p
// divides the leading coefficient.
bool monic_image(const Poly& f, uint32_t p, uint32_t* out) {
    const u64 lead = fmpz_fdiv_ui(&f.c[0].v, p);
    if (!lead) return false;
    const u64 inverse = n_invmod(lead, p);
    out[0] = 1;
    for (size_t t = 1; t < f.size(); ++t) out[t] = static_cast<uint32_t>(fmpz_fdiv_ui(&f.c[t].v, p) * inverse % p);
    return true;
}

// Buchberger's algorithm modulo one prime for homogeneous generators. The
// engine it derives from keeps the monomials of the basis, the pairs and the
// Gebauer-Moeller criteria; the coefficients live in the pool.
class Tracer : public Engine {
public:
    std::vector<uint32_t> code;
    std::vector<uint32_t> pool;
    uint32_t work_size = 0;
    std::vector<Slot> inputs;                      // the generators, at the start of the pool
    std::vector<Slot> outputs;                     // the reduced basis
    std::vector<std::vector<u64>> output_monomials;

    bool too_long = false;  // the program outgrew the budget
    bool device_screened = false;  // a device completed a layer of this trace

    // False when the prime divides a leading coefficient of a generator, or
    // when the program exceeds `budget` words.
    bool trace(const Ring& ring, const std::vector<Poly>& generators, uint32_t prime, size_t budget) {
        R = ring;
        p_ = prime;
        strategy_ = EAGER;
        rebuild_active();
        uint32_t total = 0;
        for (const Poly& f : generators) {
            inputs.push_back(Slot{total, static_cast<uint32_t>(f.size())});
            total += static_cast<uint32_t>(f.size());
        }
        pool.resize(total);
        for (size_t k = 0; k < generators.size(); ++k)
            if (!monic_image(generators[k], p_, pool.data() + inputs[k].offset)) return false;
        // The generators enter in the order of their degrees, each before the
        // pairs of its degree.
        std::vector<size_t> order(generators.size());
        for (size_t k = 0; k < order.size(); ++k) order[k] = k;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return generators[a].m[0] < generators[b].m[0];
        });
        size_t next = 0;
        long screened_degree = -1;
        while (!unit) {
            check_deadline();
            if (code.size() + pool.size() > budget) { too_long = true; return false; }
            const bool input_ready = next < order.size();
            if (!input_ready && B.empty()) break;
            if (input_ready && (B.empty() || static_cast<long>(generators[order[next]].m[0]) <= B.back().sugar)) {
                const size_t k = order[next++];
                begin_block();
                add_row(OP_ADD, inputs[k], generators[k].m.data(), 0, nullptr);
                finish_block(static_cast<long>(generators[k].m[0]), false);
                continue;
            }
            if (B.back().sugar != screened_degree) {
                screened_degree = B.back().sugar;
                try {
                    screen_degree(budget);
                } catch (const std::bad_alloc&) {
                    // A layer spans more monomials than an individual pair.
                    // Insufficient memory for the optional batch leaves the
                    // original pair queue available to the scalar trace.
                }
                if (B.empty() || B.back().sugar != screened_degree) continue;
            }
            const Pair pair = B.back();
            B.pop_back();
            ++reductions;
            u64 shift[MAX_STRIDE];
            begin_block();
            R.divide(shift, pair.lcm, G[pair.i].p.m.data());
            add_row(OP_ADD, slot_[pair.i], G[pair.i].p.m.data(), 1, shift);
            R.divide(shift, pair.lcm, G[pair.j].p.m.data());
            add_row(OP_SUB, slot_[pair.j], G[pair.j].p.m.data(), 1, shift);
            if (!finish_block(pair.sugar, false)) ++zero_reductions;
        }
        reduce_basis();
        return true;
    }

private:
    uint32_t p_ = 0;
    std::vector<Slot> slot_;  // by index in G
    std::vector<u64> work_;

    // The block under construction.
    struct Row {
        uint32_t op, offset, count;
        uint32_t column;  // RED: the arena index of the monomial reduced
        size_t terms;     // start of its monomials in row_terms_
    };
    std::vector<Row> rows_;
    std::vector<uint32_t> row_terms_;  // arena indices of the monomials of the rows
    std::vector<u64> arena_;           // the monomials of the block
    std::vector<char> pivot_;
    std::vector<uint32_t> table_;      // open addressing: arena index + 1, or 0
    std::vector<uint32_t> heap_;       // the monomials not yet examined, largest on top
    std::vector<uint32_t> columns_;    // the monomials examined, in decreasing order
    std::vector<uint32_t> position_;

    bool smaller(uint32_t a, uint32_t b) const {
        return R.compare(arena_.data() + static_cast<size_t>(a) * R.stride, arena_.data() + static_cast<size_t>(b) * R.stride) < 0;
    }
    size_t hash(const u64* mono) const {
        u64 h = 0;
        for (int i = 1; i < R.stride; ++i) h = (h ^ mono[i]) * 0x9E3779B97F4A7C15ULL;
        // Fold the high exponent fields into the low table-index bits.
        // Otherwise monomials differing only in those fields repeatedly
        // follow the same linear-probing chain.
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        return static_cast<size_t>(h ^ (h >> 33));
    }
    void begin_block() {
        rows_.clear();
        row_terms_.clear();
        arena_.clear();
        pivot_.clear();
        heap_.clear();
        columns_.clear();
        if (table_.empty()) table_.resize(1024);
        std::fill(table_.begin(), table_.end(), 0);
    }
    // The arena index of the monomial, entered as a column if it is new.
    uint32_t column(const u64* mono) {
        const int S = R.stride;
        size_t mask = table_.size() - 1;
        size_t at = hash(mono) & mask;
        for (; table_[at]; at = (at + 1) & mask)
            if (R.equal(arena_.data() + static_cast<size_t>(table_[at] - 1) * S, mono)) return table_[at] - 1;
        const uint32_t fresh = static_cast<uint32_t>(arena_.size() / S);
        arena_.insert(arena_.end(), mono, mono + S);
        pivot_.push_back(0);
        table_[at] = fresh + 1;
        if (2 * (static_cast<size_t>(fresh) + 1) > table_.size()) {  // keep the table at most half full
            table_.assign(table_.size() * 2, 0);
            mask = table_.size() - 1;
            for (uint32_t id = 0; id <= fresh; ++id) {
                size_t slot = hash(arena_.data() + static_cast<size_t>(id) * S) & mask;
                while (table_[slot]) slot = (slot + 1) & mask;
                table_[slot] = id + 1;
            }
        }
        heap_.push_back(fresh);
        std::push_heap(heap_.begin(), heap_.end(), [this](uint32_t a, uint32_t b) { return smaller(a, b); });
        return fresh;
    }
    // A row: the terms of a polynomial from `first` on, times a monomial.
    void add_row(uint32_t op, Slot slot, const u64* monomials, uint32_t first, const u64* shift, uint32_t reduced = 0) {
        const int S = R.stride;
        Row row{op, slot.offset + first, slot.length - first, reduced, row_terms_.size()};
        u64 mono[MAX_STRIDE];
        for (uint32_t t = first; t < slot.length; ++t) {
            const u64* source = monomials + static_cast<size_t>(t) * S;
            if (shift) R.multiply(mono, source, shift);
            row_terms_.push_back(column(shift ? mono : source));
        }
        rows_.push_back(row);
    }

#include "trace_screen.inc"
    // Reduce the rows entered so far, emit the block, run it for the trace
    // prime and insert the remainder. False when the remainder is zero; the
    // block is then dropped. With `keep`, the first column is not reduced and
    // the remainder is stored as an element of the reduced basis instead.
    bool finish_block(long sugar, bool keep) {
        const int S = R.stride;
        // Every monomial that a current reducer divides is reduced, largest
        // first; a reducer only brings in smaller monomials.
        while (!heap_.empty()) {
            std::pop_heap(heap_.begin(), heap_.end(), [this](uint32_t a, uint32_t b) { return smaller(a, b); });
            const uint32_t id = heap_.back();
            heap_.pop_back();
            columns_.push_back(id);
            if (keep && columns_.size() == 1) continue;
            u64 mono[MAX_STRIDE];
            std::copy(arena_.begin() + static_cast<size_t>(id) * S, arena_.begin() + static_cast<size_t>(id + 1) * S, mono);
            const int g = reducer(mono);
            if (g < 0) continue;
            u64 shift[MAX_STRIDE];
            R.divide(shift, mono, G[g].p.m.data());
            pivot_[id] = 1;
            add_row(OP_RED, slot_[g], G[g].p.m.data(), 1, shift, id);
        }
        // A generator or basis element that needs no reduction and has no
        // zero coefficient is used where it lies.
        const bool plain = rows_.size() == 1 && rows_[0].op == OP_ADD;
        std::vector<uint32_t>& position = position_;
        position.resize(arena_.size() / S);
        uint32_t width = 0;
        for (uint32_t id : columns_) position[id] = width++;
        const size_t start = code.size();
        code.push_back(OP_BEGIN);
        code.push_back(width);
        for (const Row& row : rows_) {
            code.push_back(row.op);
            if (row.op == OP_RED) code.push_back(position[row.column]);
            code.push_back(row.offset);
            code.push_back(row.count);
            for (uint32_t t = 0; t < row.count; ++t) code.push_back(position[row_terms_[row.terms + t]]);
        }
        if (work_.size() < width) work_.resize(width);
        execute_replay_block(code.data(), start, code.size(), p_, pool.data(), work_.data());
        std::vector<uint32_t> present, absent;
        Poly h;
        for (uint32_t id : columns_) {
            if (pivot_[id]) continue;
            if (work_[position[id]] % p_) {
                present.push_back(position[id]);
                h.m.insert(h.m.end(), arena_.begin() + static_cast<size_t>(id) * S, arena_.begin() + static_cast<size_t>(id + 1) * S);
            } else {
                absent.push_back(position[id]);
            }
        }
        if (present.empty()) {
            code.resize(start);
            return false;
        }
        Slot slot;
        if (plain && absent.empty()) {
            code.resize(start);
            slot = Slot{rows_[0].offset, rows_[0].count};
        } else {
            work_size = std::max(work_size, width);
            const size_t tail = code.size();
            if (!absent.empty()) {
                code.push_back(OP_CHECK);
                code.push_back(static_cast<uint32_t>(absent.size()));
                code.insert(code.end(), absent.begin(), absent.end());
            }
            slot = Slot{static_cast<uint32_t>(pool.size()), static_cast<uint32_t>(present.size())};
            pool.resize(pool.size() + present.size());
            code.push_back(OP_STORE);
            code.push_back(slot.offset);
            code.push_back(slot.length);
            code.insert(code.end(), present.begin(), present.end());
            execute_replay_block(code.data(), tail, code.size(), p_, pool.data(), work_.data());
        }
        if (keep) {
            outputs.push_back(slot);
            output_monomials.push_back(std::move(h.m));
            return true;
        }
        if (h.m[0] == 0) unit = true;  // a constant
        h.c.resize(slot.length);
        slot_.push_back(slot);
        insert(std::move(h), sugar);
        return true;
    }
    // The elements with minimal leading monomials, each with its tail reduced
    // by the others.
    void reduce_basis() {
        std::vector<int> minimal;
        for (int a : active_) {
            bool needed = true;
            for (int b : active_)
                if (b != a && R.divides(lead(b), lead(a)) && (!R.equal(lead(b), lead(a)) || b < a)) {
                    needed = false;
                    break;
                }
            if (needed) minimal.push_back(a);
        }
        std::sort(minimal.begin(), minimal.end(), [this](int a, int b) { return R.compare(lead(a), lead(b)) > 0; });
        for (Element& e : G) e.active = false;
        for (int a : minimal) G[a].active = true;
        rebuild_active();
        for (int a : minimal) {
            begin_block();
            add_row(OP_ADD, slot_[a], G[a].p.m.data(), 0, nullptr);
            finish_block(0, true);
        }
    }
};

// The standalone C API can set a process default. A solve may override it
// (including selecting the CPU) for its own thread and restore it on exit.
std::atomic<ReplayBatch> default_batch{nullptr};
// The default hook is a dispatcher: it declines unless its caller's scope or
// ATLAS_GROEBNER_GPU asks for a device.
std::atomic<bool> default_dispatches{false};
thread_local ReplayBatch scoped_batch = nullptr;
thread_local int has_scoped_batch = 0;
thread_local uint64_t cpu_batches = 0;
thread_local uint64_t cpu_primes = 0;

ReplayBatch active_batch() {
    return has_scoped_batch ? scoped_batch : default_batch.load();
}

using CRTPairs = int (*)(const uint32_t*, const uint32_t*, uint32_t, uint32_t, uint64_t*);
thread_local CRTPairs crt_pairs = nullptr;

// Each device lane merges two prime images.  The host combines the resulting
// double-width residues with the previous arbitrary-precision CRT modulus.
// The last prime remains independent for rational reconstruction.
void merge_paired_images(const std::vector<uint32_t>& primes, const std::vector<std::vector<uint32_t>>& images,
                         size_t& used, Z& modulus, std::vector<Z>& lifted) {
    if (!crt_pairs || !active_batch() || sizeof(ulong) < sizeof(uint64_t)) return;
    const size_t pairs = (primes.size() - 1 - used) / 2;
    const size_t width = lifted.size();
    if (!pairs || !width || width > static_cast<size_t>(INT_MAX) / (2 * pairs)) return;
    std::vector<uint32_t> packed, parameters;
    std::vector<uint64_t> combined;
    try {
        packed.reserve(2 * pairs * width);
        parameters.reserve(3 * pairs);
        combined.resize(pairs * width);
    } catch (const std::bad_alloc&) { return; }
    for (size_t k = used; k < used + 2*pairs; k += 2) {
        const auto p = primes[k], q = primes[k+1];
        parameters.insert(parameters.end(), {p, q, static_cast<uint32_t>(n_invmod(p % q, q))});
        for (auto image : {k, k+1}) packed.insert(packed.end(), images[image].begin(), images[image].end());
    }
    if (crt_pairs(packed.data(), parameters.data(), pairs, width, combined.data()) != 0) return;
    for (size_t pair = 0; pair < pairs; ++pair) {
        const auto p = primes[used], q = primes[used+1];
        const uint64_t product = static_cast<uint64_t>(p) * q;
        const auto* values = combined.data() + pair*width;
        // Reject a damaged result before changing the accumulated residues.
        // Earlier pairs already accepted here remain valid.
        for (size_t i = 0; i < width; ++i)
            if (values[i] >= product || values[i] % p != images[used][i] ||
                values[i] % q != images[used+1][i]) return;
        if (used == 0) {
            for (size_t i = 0; i < width; ++i) fmpz_set_ui(&lifted[i].v, values[i]);
        } else {
            nmod_t ring;
            nmod_init(&ring, product);
            const ulong inverse = n_invmod(fmpz_fdiv_ui(&modulus.v, product), product);
            for (size_t i = 0; i < width; ++i) {
                const auto old = fmpz_fdiv_ui(&lifted[i].v, product);
                const auto correction = nmod_mul(nmod_sub(values[i], old, ring), inverse, ring);
                fmpz_addmul_ui(&lifted[i].v, &modulus.v, correction);
            }
        }
        fmpz_mul_ui(&modulus.v, &modulus.v, product);
        used += 2;
    }
}

// Whether this solve asked for a device: a trace screen or a batch hook of its
// own scope, a hook installed directly as the default, or the environment
// opt-in that a dispatcher honours outside a scope.
bool device_requested() {
    if (trace_screen) return true;
    if (has_scoped_batch) return scoped_batch != nullptr;
    if (!default_batch.load()) return false;
    if (!default_dispatches.load()) return true;
    const char* setting = std::getenv("ATLAS_GROEBNER_GPU");
    return setting && *setting;
}

// Merge new prime images into the existing CRT residues and reconstruct
// rational coefficients. The last image is held out as a witness.
bool reconstruct_coefficients(const std::vector<uint32_t>& primes, const std::vector<std::vector<uint32_t>>& images,
                              uint32_t trace_prime, size_t& used, Z& modulus, std::vector<Z>& lifted,
                              std::vector<LiftedRational>& rational) {
    const size_t coefficients = lifted.size();
    const uint32_t witness = primes.back();
    merge_paired_images(primes, images, used, modulus, lifted);
    // The CRT inverse is shared by every coefficient of this image.
    // For r in [0,M), r + M ((image - r) / M mod p) is the unique
    // representative in [0,Mp). Save M and its inverse once per prime.
    const size_t first_prime = used;
    std::vector<Z> prefix_moduli;
    std::vector<uint32_t> inverses;
    prefix_moduli.reserve(primes.size() - used - 1);
    for (; used + 1 < primes.size(); ++used) {
        prefix_moduli.push_back(modulus);
        inverses.push_back(n_invmod(fmpz_fdiv_ui(&modulus.v, primes[used]), primes[used]));
        fmpz_mul_ui(&modulus.v, &modulus.v, primes[used]);
    }
    std::atomic<bool> stable{true};
    // The workers stop once the caller's deadline passes; see check_deadline.
    const Deadline limit = thread_deadline();
    std::atomic<bool> stop{false};
    auto halt = [&](size_t i) {
        if (limit.active && (i & 255) == 0 && passed(limit)) stop.store(true, std::memory_order_relaxed);
        return stop.load(std::memory_order_relaxed);
    };
    auto lift = [&](size_t begin, size_t end) {
        for (size_t k = first_prime; k < used; ++k) {
            const uint32_t p = primes[k], inverse = inverses[k - first_prime];
            const fmpz* product = &prefix_moduli[k - first_prime].v;
            for (size_t i = begin; i < end; ++i) {
                if (halt(i)) return;
                const uint32_t old = fmpz_fdiv_ui(&lifted[i].v, p);
                const uint32_t difference = images[k][i] + p - old;
                const u64 correction = static_cast<u64>(difference) * inverse % p;
                fmpz_addmul_ui(&lifted[i].v, product, correction);
            }
        }
        for (size_t i = begin; i < end && stable.load(std::memory_order_relaxed); ++i) {
            if (halt(i)) return;
            if (!fmpq_reconstruct_fmpz(&rational[i].value, &lifted[i].v, &modulus.v)) {
                stable.store(false, std::memory_order_relaxed);
                break;
            }
            const u64 den = fmpz_fdiv_ui(fmpq_denref(&rational[i].value), witness);
            const u64 num = fmpz_fdiv_ui(fmpq_numref(&rational[i].value), witness);
            // Arnold's criterion needs the trace prime not to divide a denominator.
            if (!den || static_cast<u64>(images.back()[i]) * den % witness != num ||
                !fmpz_fdiv_ui(fmpq_denref(&rational[i].value), trace_prime))
                stable.store(false, std::memory_order_relaxed);
        }
    };
    // Assign adjacent coefficients to each worker. Their residues and
    // rational outputs are disjoint through every prime in this round.
    const size_t lanes = std::max<size_t>(1, std::min<size_t>(std::thread::hardware_concurrency(),
                                                              (coefficients + 2047) / 2048));
    std::atomic<size_t> next{0};
    const size_t chunk = (coefficients + lanes - 1) / lanes;
    auto worker = [&] {
        for (size_t lane; (lane = next.fetch_add(1, std::memory_order_relaxed)) < lanes;) {
            const size_t begin = lane * chunk;
            lift(begin, std::min(coefficients, begin + chunk));
        }
    };
    std::vector<std::thread> crew;
    crew.reserve(lanes - 1);
    try {
        for (size_t k = 1; k < lanes; ++k) crew.emplace_back([&] {
            worker();
            flint_cleanup();
        });
    } catch (const std::system_error&) {
        // Any started workers and the calling thread finish all chunks.
    }
    worker();
    for (auto& t : crew) t.join();
    if (stop.load()) expire();
    return stable;
}

double since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// Which trace the last modular basis on this thread used: 1 Buchberger, 2 F4
// on the homogenized generators, 3 F4 on the generators themselves.
thread_local int last_modular_route = 0;
std::atomic<int> f4_route_enabled{1};

// Lift the images: `basis` receives, for each polynomial, the integer
// multiple of its rational lift by the common denominator, on the given
// monomials. False while the lift is not stable.
bool lift_images(const std::vector<uint32_t>& primes, std::vector<std::vector<uint32_t>>& images, uint32_t trace_prime,
                 size_t& used, Z& modulus, std::vector<Z>& lifted, const std::vector<const std::vector<u64>*>& monomials,
                 const std::vector<uint32_t>& lengths, ModularReport& stats, std::vector<Poly>& basis) {
    const auto clock = std::chrono::steady_clock::now();
    std::vector<LiftedRational> rational(lifted.size());
    const bool stable = reconstruct_coefficients(primes, images, trace_prime, used, modulus, lifted, rational);
    // The images merged into `lifted` are no longer needed; the witness stays.
    for (size_t k = 0; k < used && k + 1 < images.size(); ++k) std::vector<uint32_t>().swap(images[k]);
    basis.clear();
    if (stable) {
        size_t i = 0;
        Z common, factor;
        for (size_t g = 0; g < lengths.size(); ++g) {
            const uint32_t length = lengths[g];
            fmpz_one(&common.v);
            for (uint32_t t = 0; t < length; ++t) fmpz_lcm(&common.v, &common.v, fmpq_denref(&rational[i + t].value));
            Poly f;
            f.m = *monomials[g];
            f.c.resize(length);
            for (uint32_t t = 0; t < length; ++t) {
                fmpz_divexact(&factor.v, &common.v, fmpq_denref(&rational[i + t].value));
                fmpz_mul(&f.c[t].v, &factor.v, fmpq_numref(&rational[i + t].value));
            }
            i += length;
            basis.push_back(std::move(f));
        }
    }
    const auto lifting = std::chrono::steady_clock::now() - clock;
    stats.lift_seconds += std::chrono::duration<double>(lifting).count();
    lift_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(lifting).count();
    return stable;
}

// The certificate of 7.3 for a lifted homogeneous basis. On success the
// reduced basis for the given order is left in `result`.
bool certify_homogeneous(const Ring& R, const Ring& H, const std::vector<Poly>& generators, const std::vector<Poly>& basis,
                         Engine& result, ModularReport& stats) {
    const auto clock = std::chrono::steady_clock::now();
    ++stats.verifications;
    ++certification_attempts;
    std::optional<bool> proved = HomogeneousCertificate{}.check(H, basis, generators);
    if (!proved.has_value()) {
        Engine check;
        check.R = H;
        proved = check.is_basis(basis);
        for (size_t k = 0; k < generators.size() && *proved; ++k) {
            check_deadline();
            Poly f = generators[k];
            check.reduce(f, 0, nullptr, nullptr);
            proved = f.empty();
        }
    }
    const auto verifying = std::chrono::steady_clock::now() - clock;
    stats.verify_seconds += std::chrono::duration<double>(verifying).count();
    verification_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(verifying).count();
    if (!*proved) return false;
    std::vector<Poly> affine;
    for (const Poly& f : basis) affine.push_back(dehomogenize(H, R, f));
    // Every generator is declared part of a basis: no pair is formed,
    // so this only removes redundant elements and reduces the tails.
    result.R = R;
    result.compute(affine, affine.size());
    ++certified_bases;
    return true;
}

// The certificate of 7.6 for a direct trace: `lifted` holds the elements of
// the trace and then its reduced basis. The selected rows of every matrix
// reduce to zero over Q, so all of them lie in the ideal; the reduced basis
// is a Groebner basis; the generators reduce to zero by it.
bool certify_direct(const Ring& R, const std::vector<Poly>& generators, const F4Trace& tracer, std::vector<Poly>& lifted,
                    Engine& result, ModularReport& stats, unsigned threads) {
    const auto clock = std::chrono::steady_clock::now();
    ++stats.verifications;
    ++certification_attempts;
    const size_t elements = tracer.G.size();
    std::vector<Poly> outputs(std::make_move_iterator(lifted.begin() + elements), std::make_move_iterator(lifted.end()));
    lifted.resize(elements);
    bool proved = tracer.rows_vanish(generators, lifted, outputs, threads);
    if (proved) {
        result.R = R;
        proved = result.is_basis(outputs);
        for (size_t k = 0; k < generators.size() && proved; ++k) {
            check_deadline();
            Poly f = generators[k];
            result.reduce(f, 0, nullptr, nullptr);
            proved = f.empty();
        }
    }
    const auto verifying = std::chrono::steady_clock::now() - clock;
    stats.verify_seconds += std::chrono::duration<double>(verifying).count();
    verification_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(verifying).count();
    if (proved) ++certified_bases;
    return proved;
}

// Lift the images of a homogeneous basis and certify the result over Q.
// -1: not yet stable, 0: stable but not proved, 1: proved, with the reduced
// basis in `result`.
int lift_and_certify(const Ring& R, const Ring& H, const std::vector<Poly>& generators, const std::vector<uint32_t>& primes,
                     std::vector<std::vector<uint32_t>>& images, uint32_t trace_prime, size_t& used, Z& modulus,
                     std::vector<Z>& lifted, const std::vector<std::vector<u64>>& output_monomials,
                     const std::vector<uint32_t>& lengths, Engine& result, ModularReport& stats) {
    std::vector<const std::vector<u64>*> monomials;
    for (const std::vector<u64>& m : output_monomials) monomials.push_back(&m);
    std::vector<Poly> basis;
    if (!lift_images(primes, images, trace_prime, used, modulus, lifted, monomials, lengths, stats, basis)) return -1;
    return certify_homogeneous(R, H, generators, basis, result, stats);
}

// Whether the F4 traces serve this problem.
bool f4_applies(const Ring& R, const std::vector<Poly>& generators) {
    if (!f4_route_enabled.load(std::memory_order_relaxed) || device_requested()) return false;
    // For an order that does not compare degrees first most rows of a degree
    // reduce to zero and many primes are needed: the Buchberger replay, which
    // repeats only the useful reductions, is the cheaper one there.
    if (R.lex || R.eliminate || generators.empty()) return false;
    for (const Poly& f : generators)
        if (f.m[0] == 0) return false;  // a constant: the ideal is the unit ideal
    return true;
}

u64 previous_prime(u64 p) {
    do { --p; } while (!n_is_prime(p));
    return p;
}

// The trace the F4 route starts from, and its prime (KERNEL.md 7.6): of the
// homogenized generators in H, or with `direct` of the given ones in R.
// Homogeneous generators, and those meeting the Hilbert bound in every
// degree, are traced homogenized. For the others both traces are made at one
// prime, the homogenized one within twice the structure of the direct one,
// and the one with less to lift and certify is kept. Null when the traces
// exceed the budget.
std::unique_ptr<F4Trace> first_trace(const Ring& R, const Ring& H, const std::vector<Poly>& given,
                                     const std::vector<Poly>& homogenized, size_t budget, unsigned threads, bool& direct,
                                     u64& prime) {
    bool mixed = false;
    for (const Poly& f : given)
        for (size_t t = 1; t < f.size() && !mixed; ++t) mixed = f.m[t * R.stride] != f.m[0];
    direct = false;
    prime = (static_cast<u64>(1) << 31) - 1;
    F4Trace::Options bounded;
    bounded.stop_at_missed_bound = mixed;
    while (true) {
        prime = previous_prime(prime);
        auto trace = std::make_unique<F4Trace>();
        if (trace->trace(H, homogenized, static_cast<uint32_t>(prime), budget, threads, bounded)) return trace;
        if (trace->too_large) return nullptr;
        if (trace->missed_bound) break;
    }
    // A degree missed the bound: the homogenized ideal may need many more
    // elements than the given one.
    F4Trace::Options selecting;
    selecting.select_rows = true;
    auto own = std::make_unique<F4Trace>();
    while (!own->trace(R, given, static_cast<uint32_t>(prime), budget, threads, selecting)) {
        if (own->too_large) return nullptr;
        prime = previous_prime(prime);
        own = std::make_unique<F4Trace>();
    }
    auto full = std::make_unique<F4Trace>();
    const size_t allowance = std::min(budget, 2 * own->words + (static_cast<size_t>(1) << 16));
    if (full->trace(H, homogenized, static_cast<uint32_t>(prime), allowance, threads)) {
        // The direct route lifts every element of its trace, the other one its
        // basis alone. It pays when that basis is larger than the basis of
        // the ideal, unless the elements of the direct trace outweigh that
        // basis by far, as they do when the ideal is the unit ideal.
        const size_t basis = own->output_residues.size(), elements = own->element_residues().size();
        const size_t inflated = full->output_residues.size();
        if (2 * inflated < 3 * basis || elements > 16 * basis) return full;
    }
    direct = true;
    return own;
}

// Steps 3 to 5 from an F4 trace, and step 2 again when its prime turns out
// unlucky. `generators` are the homogenized ones in T = H, certified as in 7.3
// (KERNEL.md 7.5), or with `direct` the given ones in T = R, certified through
// the rows of the trace (7.6). `ready`, when given, is the trace for `prime`.
// False when the route gives up; nothing is left in `result` then.
bool f4_modular_basis(const Ring& R, const Ring& H, const std::vector<Poly>& generators, bool direct, Engine& result,
                      ModularReport& stats, size_t budget, std::unique_ptr<F4Trace> ready, u64 prime) {
    const Ring& T = direct ? R : H;
    F4Trace::Options options;
    options.select_rows = direct;
    const unsigned threads = std::max(1U, std::thread::hardware_concurrency());
    auto previous = previous_prime;
    u64 trace_prime = ready ? prime : (static_cast<u64>(1) << 31) - 1;
    for (int attempt = 0; attempt < 3; ++attempt) {
        auto clock = std::chrono::steady_clock::now();
        std::unique_ptr<F4Trace> found = std::move(ready);
        while (!found) {
            trace_prime = previous(trace_prime);
            found = std::make_unique<F4Trace>();
            if (found->trace(T, generators, static_cast<uint32_t>(trace_prime), budget, threads, options)) break;
            if (found->too_large) {
                stats.trace_seconds += since(clock);
                return false;
            }
            found.reset();
        }
        const F4Trace& tracer = *found;
        ++stats.traces;
        stats.trace_seconds += since(clock);
        cpu_trace_batches += static_cast<uint64_t>(tracer.matrices);
        cpu_trace_pairs += static_cast<uint64_t>(tracer.lanes);
        stats.instructions = static_cast<long>(tracer.words);
        stats.pool = 0;
        stats.basis = static_cast<long>(tracer.output_lengths.size());
        // What is lifted: the reduced basis and, for the direct trace, every
        // element it inserted before it.
        std::vector<const std::vector<u64>*> monomials;
        std::vector<uint32_t> lengths;
        std::vector<uint32_t> first;
        if (direct) {
            for (const Element& e : tracer.G) {
                monomials.push_back(&e.p.m);
                lengths.push_back(static_cast<uint32_t>(e.p.size()));
            }
            first = tracer.element_residues();
        }
        for (const std::vector<u64>& m : tracer.output_monomials) monomials.push_back(&m);
        lengths.insert(lengths.end(), tracer.output_lengths.begin(), tracer.output_lengths.end());
        first.insert(first.end(), tracer.output_residues.begin(), tracer.output_residues.end());
        std::vector<uint32_t> primes{static_cast<uint32_t>(trace_prime)};
        std::vector<std::vector<uint32_t>> images;
        images.push_back(std::move(first));
        std::vector<Z> lifted(images[0].size());
        Z modulus;
        fmpz_one(&modulus.v);
        size_t used = 0, rejected = 0;
        int unproved = 0;
        u64 next_prime = trace_prime;
        // Batches of 2, 4, 8 and 16 primes as in the Buchberger route, then
        // of 16 or one per thread: a replay costs as much as the trace's
        // elimination, so few primes should be replayed in vain.
        // The elements of a direct trace can have far larger coefficients than
        // the basis: past 255 primes the homogenized trace takes over.
        const size_t most = direct ? 255 : 4096;
        for (size_t batch = 2; primes.size() <= most; batch = batch < 16 ? 2 * batch : std::max<size_t>(16, threads)) {
            check_deadline();
            if (direct && primes.size() + batch > most + 1) return false;
            clock = std::chrono::steady_clock::now();
            std::vector<uint32_t> fresh(batch);
            for (uint32_t& p : fresh) p = static_cast<uint32_t>(next_prime = previous(next_prime));
            size_t active = 0;
            for (uint32_t p : fresh) {
                bool leads = true;
                for (const Poly& f : generators) leads = leads && fmpz_fdiv_ui(&f.c[0].v, p) != 0;
                active += leads;
            }
            ++cpu_batches;
            cpu_primes += active;
            std::vector<std::vector<uint32_t>> residues(batch);
            std::vector<char> agrees(batch, 0);
            std::atomic<size_t> cursor{0};
            // A batch smaller than the host shares the threads among its primes.
            const unsigned inner = static_cast<unsigned>(std::max<size_t>(1, threads / batch));
            std::mutex lock;
            std::exception_ptr error;
            // The workers stop at the next prime once the caller's deadline passes.
            const Deadline limit = thread_deadline();
            std::atomic<bool> late{false};
            auto worker = [&] {
                try {
                    for (size_t b; (b = cursor.fetch_add(1, std::memory_order_relaxed)) < batch;) {
                        if (passed(limit)) {
                            late = true;
                            cursor.store(batch);
                            break;
                        }
                        std::vector<uint32_t> outputs;
                        agrees[b] = tracer.replay(generators, fresh[b], outputs, inner, direct ? &residues[b] : nullptr);
                        residues[b].insert(residues[b].end(), outputs.begin(), outputs.end());
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> guard(lock);
                    if (!error) error = std::current_exception();
                    cursor.store(batch);
                }
            };
            std::vector<std::thread> crew;
            try {
                for (size_t w = 1; w < std::min<size_t>(batch, threads); ++w) crew.emplace_back([&] {
                    worker();
                    flint_cleanup();
                });
            } catch (const std::system_error&) {
                // The calling thread replays the remaining primes.
            }
            worker();
            for (std::thread& t : crew) t.join();
            if (late) expire();
            if (error) std::rethrow_exception(error);
            stats.replay_seconds += since(clock);
            for (size_t b = 0; b < batch; ++b) {
                if (!agrees[b]) { ++rejected; ++stats.rejected; continue; }
                primes.push_back(fresh[b]);
                images.push_back(std::move(residues[b]));
            }
            stats.primes = static_cast<long>(primes.size());
            // When most primes disagree with the trace, its prime was the unlucky one.
            if (rejected > 2 && rejected >= primes.size()) break;
            if (primes.size() < 2) continue;
            std::vector<Poly> basis;
            if (!lift_images(primes, images, static_cast<uint32_t>(trace_prime), used, modulus, lifted, monomials, lengths,
                             stats, basis))
                continue;
            if (direct ? certify_direct(R, generators, tracer, basis, result, stats, threads)
                       : certify_homogeneous(R, H, generators, basis, result, stats))
                return true;
            // A stable lift that is not a basis: more primes correct a chance
            // reconstruction; after that the trace is the suspect.
            if (++unproved == 2) break;
        }
    }
    return false;
}

}  // namespace

// f with every term brought to the degree of f by a power of the new last
// variable, in the ring H.
Poly homogenize(const Ring& R, const Ring& H, const Poly& f) {
    const int S = R.stride, T = H.stride;
    u64 top = 0;
    for (size_t i = 0; i < f.size(); ++i) top = std::max(top, f.m[i * S]);
    std::vector<u64> monomials(f.size() * T);
    long e[MAX_VARIABLES];
    for (size_t i = 0; i < f.size(); ++i) {
        R.unpack(f.m.data() + i * S, e);
        e[R.n] = static_cast<long>(top - f.m[i * S]);
        H.pack(monomials.data() + i * T, e);
    }
    std::vector<size_t> order(f.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return H.compare(monomials.data() + a * T, monomials.data() + b * T) > 0;
    });
    Poly g;
    for (size_t i : order) {
        g.m.insert(g.m.end(), monomials.begin() + i * T, monomials.begin() + (i + 1) * T);
        g.c.push_back(f.c[i]);
    }
    primitive(g, nullptr);
    return g;
}

// The polynomial with the last variable set to one, in the ring R.
Poly dehomogenize(const Ring& H, const Ring& R, const Poly& f) {
    const int S = H.stride, T = R.stride;
    std::vector<u64> monomials(f.size() * T);
    long e[MAX_VARIABLES];
    for (size_t i = 0; i < f.size(); ++i) {
        H.unpack(f.m.data() + i * S, e);
        R.pack(monomials.data() + i * T, e);
    }
    std::vector<size_t> order(f.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return R.compare(monomials.data() + a * T, monomials.data() + b * T) > 0;
    });
    Poly g;
    for (size_t i : order) {
        g.m.insert(g.m.end(), monomials.begin() + i * T, monomials.begin() + (i + 1) * T);
        g.c.push_back(f.c[i]);
    }
    primitive(g, nullptr);
    return g;
}

bool modular_basis(const Ring& R, const std::vector<Poly>& input, Engine& result, ModularReport* report, size_t budget) {
    ModularReport local;
    ModularReport& stats = report ? *report : local;
    if (R.n + 1 > MAX_VARIABLES) return false;
    Ring H;
    H.init(R.n + 1, R.lex ? 1 : R.eliminate ? 2 : 0, true);
    std::vector<Poly> generators;
    for (const Poly& f : input)
        if (!f.empty()) generators.push_back(homogenize(R, H, f));
    try {
        if (f4_applies(R, generators)) {
            std::vector<Poly> given;
            for (const Poly& f : input)
                if (!f.empty()) given.push_back(f);
            const auto clock = std::chrono::steady_clock::now();
            bool direct = false;
            u64 prime = 0;
            std::unique_ptr<F4Trace> trace = first_trace(R, H, given, generators, budget,
                                                         std::max(1U, std::thread::hardware_concurrency()), direct, prime);
            stats.trace_seconds += since(clock);
            if (trace && f4_modular_basis(R, H, direct ? given : generators, direct, result, stats, budget, std::move(trace), prime)) {
                last_modular_route = direct ? 3 : 2;
                return true;
            }
            // The direct route gave up: the homogenized trace to its end.
            if (direct && f4_modular_basis(R, H, generators, false, result, stats, budget, nullptr, 0)) {
                last_modular_route = 2;
                return true;
            }
        }
    } catch (const std::bad_alloc&) {
        // The matrices of a degree did not fit: the Buchberger trace reduces
        // one pair at a time.
    }
    last_modular_route = 1;

    u64 trace_prime = (static_cast<u64>(1) << 31) - 1;  // primes are taken downwards from 2^31
    auto previous = [](u64 p) {
        do { --p; } while (!n_is_prime(p));
        return p;
    };
    for (int attempt = 0; attempt < 4; ++attempt) {
        // The trace: a full computation modulo one prime.
        auto clock = std::chrono::steady_clock::now();
        std::unique_ptr<Tracer> found;
        do {
            if (found && found->too_long) return false;
            trace_prime = previous(trace_prime);
            found = std::make_unique<Tracer>();
        } while (!found->trace(H, generators, static_cast<uint32_t>(trace_prime), budget));
        const Tracer& tracer = *found;
        ++stats.traces;
        stats.trace_seconds += since(clock);
        stats.instructions = static_cast<long>(tracer.code.size());
        stats.pool = static_cast<long>(tracer.pool.size());
        stats.basis = static_cast<long>(tracer.outputs.size());
        const uint32_t pool_size = static_cast<uint32_t>(tracer.pool.size());
        size_t coefficients = 0;
        for (const Slot& slot : tracer.outputs) coefficients += slot.length;

        // Images of the reduced basis: one row of residues per accepted prime.
        std::vector<uint32_t> primes{static_cast<uint32_t>(trace_prime)};
        std::vector<std::vector<uint32_t>> images(1);
        for (const Slot& slot : tracer.outputs)
            images[0].insert(images[0].end(), tracer.pool.begin() + slot.offset, tracer.pool.begin() + slot.offset + slot.length);
        std::vector<Z> lifted(coefficients);  // residues modulo `modulus` of the primes lifted so far
        Z modulus;
        fmpz_one(&modulus.v);
        size_t used = 0;  // images[0..used) are in `lifted`
        u64 next_prime = trace_prime;
        size_t rejected = 0;
        for (size_t batch = 2;; batch *= 2) {
            check_deadline();
            if (primes.size() > 4096) return false;
            // Replay for a batch of new primes.
            clock = std::chrono::steady_clock::now();
            std::vector<uint32_t> fresh(batch);
            for (uint32_t& p : fresh) p = static_cast<uint32_t>(next_prime = previous(next_prime));
            std::vector<uint32_t> pools(batch * static_cast<size_t>(pool_size));
            std::vector<uint32_t> status(batch, 1);
            auto seed_inputs = [&] {
                for (size_t b = 0; b < batch; ++b)
                    for (size_t k = 0; k < generators.size() && status[b]; ++k)
                        if (!monic_image(generators[k], fresh[b], pools.data() + b * pool_size + tracer.inputs[k].offset))
                            status[b] = 0;
            };
            seed_inputs();
            bool done = false;
            const ReplayBatch batch_hook = active_batch();
            // After a device-built trace, the first few replay primes make
            // smaller batches than the trace layer itself. Native threads
            // avoid retransferring the full program for batches below 16 or
            // that fit one host wave (up to 32 independent primes).
            const size_t host_lanes = std::min<size_t>(32, std::thread::hardware_concurrency());
            if (batch_hook && !(tracer.device_screened && (batch < 16 || batch <= host_lanes)) && !tracer.code.empty()) {
                std::vector<uint32_t> answer(batch, 0);
                if (batch_hook(tracer.code.data(), tracer.code.size(), pool_size, tracer.work_size, fresh.data(),
                               static_cast<uint32_t>(batch), pools.data(), answer.data()) == 0) {
                    for (size_t b = 0; b < batch; ++b) status[b] = status[b] && answer[b];
                    done = true;
                    stats.gpu = true;
                } else {
                    // A failed device readback may have overwritten the host
                    // buffer. Restore every generator image before CPU replay.
                    // All other slots are written by the replay before use.
                    seed_inputs();
                }
            }
            if (!done && !tracer.code.empty()) {
                const size_t active = std::count(status.begin(), status.end(), 1u);
                replay_on_host(tracer.code, pool_size, tracer.work_size, fresh, pools, status);
                ++cpu_batches;
                cpu_primes += active;
            }
            for (size_t b = 0; b < batch; ++b) {
                if (!status[b]) { ++rejected; ++stats.rejected; continue; }
                primes.push_back(fresh[b]);
                images.emplace_back();
                const uint32_t* pool = pools.data() + b * pool_size;
                for (const Slot& slot : tracer.outputs)
                    images.back().insert(images.back().end(), pool + slot.offset, pool + slot.offset + slot.length);
            }
            stats.replay_seconds += since(clock);
            stats.primes = static_cast<long>(primes.size());
            // When most primes disagree with the trace, its prime was the unlucky one.
            if (rejected > 2 && rejected >= primes.size()) break;
            if (primes.size() < 2) continue;

            // Lift all images but the last, which serves as a witness, and
            // certify the result over Q.
            std::vector<uint32_t> lengths;
            for (const Slot& slot : tracer.outputs) lengths.push_back(slot.length);
            if (lift_and_certify(R, H, generators, primes, images, static_cast<uint32_t>(trace_prime), used, modulus, lifted,
                                 tracer.output_monomials, lengths, result, stats) == 1)
                return true;
        }
    }
    return false;
}

}  // namespace atlas

using namespace atlas;

extern "C" {

void atlas_groebner_swap_crt_pairs(CRTPairs* hook) noexcept { std::swap(crt_pairs, *hook); }

// Which trace the last modular basis on this thread used: 0 none, 1 the
// Buchberger trace, 2 the F4 trace of the homogenized generators, 3 the F4
// trace of the generators themselves.
int atlas_groebner_modular_route() noexcept { return last_modular_route; }

// Allow (nonzero) or disallow the F4 trace in every thread; used to compare
// the two traces. Allowed by default.
void atlas_groebner_use_f4(int allowed) noexcept { f4_route_enabled.store(allowed != 0); }

// Optional per-thread benchmark observer. A completed layer reports degree,
// prime, dimensions, CSR edge count, rank, device use and the wall times for
// symbolic preparation, numeric reduction (including transfers), and echelon.
void atlas_groebner_swap_trace_profile(TraceProfile* hook) noexcept { std::swap(trace_profile, *hook); }

// A solve can accelerate trace construction on this thread and restore its
// previous setting. A null hook uses the native row reducer for large layers.
void atlas_groebner_swap_trace_screen(TraceScreen* hook) noexcept { std::swap(trace_screen, *hook); }

void atlas_groebner_cpu_trace_counts(uint64_t* batches, uint64_t* pairs) noexcept {
    *batches = cpu_trace_batches;
    *pairs = cpu_trace_pairs;
}

// Same shape and field arithmetic as the optional device kernels. Trace primes
// are at most 31 bits and all input coefficients are canonical residues.
int atlas_groebner_trace_rows_cpu(uint32_t prime, uint32_t jobs, uint32_t width,
                                  const uint32_t* initial, const uint32_t* offsets, const uint32_t* pivots,
                                  const uint32_t* targets, const uint32_t* coefficients, uint32_t* output) noexcept {
    if (prime < 2 || prime > 0x7fffffffu) return 1;
    if (!jobs || !width) return 0;
    try {
        return trace_rows_cpu(prime, jobs, width, initial, offsets, pivots, targets, coefficients, output);
    } catch (...) {
        return 1;
    }
}

// A function that runs replay programs for several primes at once (a GPU),
// or null for the threads of the host.
void atlas_groebner_use_batch(ReplayBatch hook) noexcept {
    default_dispatches.store(false);
    default_batch.store(hook);
}

// As atlas_groebner_use_batch, for a hook that declines unless its caller's
// scope or ATLAS_GROEBNER_GPU asks for a device. A solve that asked for none
// then keeps the F4 trace.
void atlas_groebner_use_dispatcher(ReplayBatch hook) noexcept {
    default_dispatches.store(hook != nullptr);
    default_batch.store(hook);
}

// Host implementation of the optional batch callback. The producer supplies
// the same validated trace, buffer dimensions and primes as to a device.
int atlas_groebner_replay_host(const uint32_t* code, size_t length, uint32_t pool_size,
                               uint32_t work_size, const uint32_t* primes, uint32_t count,
                               uint32_t* pools, uint32_t* status) noexcept {
    try {
        std::vector<uint64_t> work(work_size);
        for (uint32_t i = 0; i < count; ++i)
            status[i] = execute_replay_block(code, 0, length, primes[i],
                                              pools + static_cast<size_t>(i) * pool_size, work.data());
        return 0;
    } catch (...) {
        return 1;
    }
}

// Install a thread-local override and return its previous state in place.
// An active null hook selects the host, even if the process has a default.
void atlas_groebner_swap_batch(ReplayBatch* hook, int* active) noexcept {
    std::swap(scoped_batch, *hook);
    std::swap(has_scoped_batch, *active);
}

void atlas_groebner_cpu_replays(uint64_t* batches, uint64_t* primes) noexcept {
    *batches = cpu_batches;
    *primes = cpu_primes;
}

// Monotone per-calling-thread counters. Include unsuccessful lift and check
// attempts, and count a basis only once its certified affine output is ready.
void atlas_groebner_certification_counts(uint64_t* counts) noexcept {
    counts[0] = certified_bases;
    counts[1] = certification_attempts;
    counts[2] = lift_ns;
    counts[3] = verification_ns;
}

// Snapshot all thread-local activity at a stage boundary in a single call.
void atlas_groebner_activity_counts(uint64_t* counts) noexcept {
    atlas_groebner_cpu_replays(counts, counts + 1);
    atlas_groebner_cpu_trace_counts(counts + 2, counts + 3);
    atlas_groebner_certification_counts(counts + 4);
}

// A CPU request sets both thread-local hooks to null and restores both on exit.
// Read the counters at the same boundary to avoid separate foreign calls.
void atlas_groebner_swap_cpu_scope(ReplayBatch* batch, int* active, TraceScreen* trace,
                                   uint64_t* counts) noexcept {
    atlas_groebner_swap_batch(batch, active);
    atlas_groebner_swap_trace_screen(trace);
    atlas_groebner_activity_counts(counts);
}

// A forked copy of a solver runs a helper program's entry in place of that
// program. Give its only thread the state of a fresh process: no replay
// hooks and no recorded activity.
void atlas_groebner_reset_fork_child() noexcept {
    crt_pairs = nullptr;
    default_batch.store(nullptr);
    default_dispatches.store(false);
    scoped_batch = nullptr;
    has_scoped_batch = 0;
    trace_screen = nullptr;
    trace_profile = nullptr;
    cpu_batches = cpu_primes = 0;
    cpu_trace_batches = cpu_trace_pairs = 0;
    certified_bases = certification_attempts = 0;
    lift_ns = verification_ns = 0;
}

// For benchmarks and tests of a batch backend: trace the problem, then replay
// it for `count` primes on the host's threads and through the hook. seconds
// receives the trace, the host replay and the hook replay. Returns the number
// of primes on which the two disagree, or -1 when there is no hook or it
// failed, -2 on an error.
long atlas_groebner_replay_compare(const char* input, size_t length, uint32_t count, double* seconds, long* sizes) noexcept {
    try {
        const char* at = input;
        const char* end = input + length;
        char* stop = nullptr;
        const long n = std::strtol(at, &stop, 10);
        const long order = std::strtol(stop, &stop, 10);
        std::strtol(stop, &stop, 10);
        at = stop;
        if (at >= end || *at != '\n' || order < 0 || order > 2) throw std::invalid_argument("Bad header.");
        ++at;
        Ring R, H;
        R.init(static_cast<int>(n), static_cast<int>(order));
        H.init(R.n + 1, static_cast<int>(order), true);
        std::vector<Poly> generators;
        while (at < end) {
            const Poly f = parse(R, at, end, nullptr);
            if (!f.empty()) generators.push_back(homogenize(R, H, f));
        }
        u64 prime = (static_cast<u64>(1) << 31) - 1;
        auto previous = [](u64 p) {
            do { --p; } while (!n_is_prime(p));
            return p;
        };
        auto clock = std::chrono::steady_clock::now();
        std::unique_ptr<Tracer> tracer;
        do {
            prime = previous(prime);
            tracer = std::make_unique<Tracer>();
        } while (!tracer->trace(H, generators, static_cast<uint32_t>(prime), ~static_cast<size_t>(0)));
        seconds[0] = since(clock);
        const uint32_t pool_size = static_cast<uint32_t>(tracer->pool.size());
        sizes[0] = static_cast<long>(tracer->code.size());
        sizes[1] = pool_size;
        sizes[2] = tracer->work_size;
        std::vector<uint32_t> primes(count);
        for (uint32_t& p : primes) p = static_cast<uint32_t>(prime = previous(prime));
        std::vector<uint32_t> host(static_cast<size_t>(count) * pool_size), device;
        std::vector<uint32_t> host_status(count, 1), device_status(count, 0);
        for (size_t b = 0; b < count; ++b)
            for (size_t k = 0; k < generators.size(); ++k)
                if (!monic_image(generators[k], primes[b], host.data() + b * pool_size + tracer->inputs[k].offset)) host_status[b] = 0;
        device = host;
        clock = std::chrono::steady_clock::now();
        replay_on_host(tracer->code, pool_size, tracer->work_size, primes, host, host_status);
        seconds[1] = since(clock);
        const ReplayBatch batch_hook = active_batch();
        if (!batch_hook) return -1;
        clock = std::chrono::steady_clock::now();
        if (batch_hook(tracer->code.data(), tracer->code.size(), pool_size, tracer->work_size, primes.data(), count,
                       device.data(), device_status.data()) != 0)
            return -1;
        seconds[2] = since(clock);
        long differ = 0;
        for (size_t b = 0; b < count; ++b) {
            const bool same_status = (host_status[b] != 0) == (device_status[b] != 0);
            const bool same_pool = !host_status[b] || std::equal(host.begin() + b * pool_size, host.begin() + (b + 1) * pool_size,
                                                                device.begin() + b * pool_size);
            differ += !(same_status && same_pool);
        }
        return differ;
    } catch (const std::exception& error) {
        set_error(error.what());
    } catch (...) {
        set_error("Unknown failure.");
    }
    return -2;
}

// The F4 trace alone, for benchmarks: the reduced basis of a graded reverse
// lexicographic problem modulo the first usable prime below 2^31, then
// `replays` further primes, one per thread on `threads` threads (0: all).
// The trace is the one the modular route takes (first_trace), and the
// seconds include the traces made to choose it. seconds: the trace, the
// replays together. counts: basis
// elements, terms, matrices, combinations reduced, dependent ones, recorded
// words, whether every degree met the Hilbert bound, replays that agree,
// the trace prime.
// `basis`, when not null, receives the basis modulo the trace prime, one
// element per line as "residue e1 ... en" terms separated by ';', to be
// released with atlas_groebner_release. Returns 0, or -1 with the error set.
int atlas_groebner_f4_profile(const char* input, size_t length, uint32_t replays, uint32_t threads,
                              double* seconds, long* counts, char** basis) noexcept {
    try {
        const char* at = input;
        const char* end = input + length;
        char* stop = nullptr;
        const long n = std::strtol(at, &stop, 10);
        const long order = std::strtol(stop, &stop, 10);
        std::strtol(stop, &stop, 10);
        at = stop;
        if (at >= end || *at != '\n' || order != 0) throw std::invalid_argument("Bad header.");
        ++at;
        Ring R, H;
        R.init(static_cast<int>(n), 0);
        H.init(R.n + 1, 0, true);
        std::vector<Poly> homogenized, given;
        while (at < end) {
            Poly f = parse(R, at, end, nullptr);
            if (f.empty()) continue;
            if (f.m[0] == 0) throw std::invalid_argument("A constant generator.");
            homogenized.push_back(homogenize(R, H, f));
            given.push_back(std::move(f));
        }
        if (given.empty()) throw std::invalid_argument("No generator.");
        const unsigned workers = threads ? threads : std::max(1U, std::thread::hardware_concurrency());
        auto previous = previous_prime;
        auto clock = std::chrono::steady_clock::now();
        bool direct = false;
        u64 prime = 0;
        std::unique_ptr<F4Trace> tracer = first_trace(R, H, given, homogenized, ~static_cast<size_t>(0), workers, direct, prime);
        if (!tracer) throw std::runtime_error("The trace is too large.");
        const Ring& T = direct ? R : H;
        const std::vector<Poly>& generators = direct ? given : homogenized;
        seconds[0] = since(clock);
        counts[0] = static_cast<long>(tracer->output_lengths.size());
        counts[1] = static_cast<long>(tracer->output_residues.size());
        counts[2] = tracer->matrices;
        counts[3] = tracer->lanes;
        counts[4] = tracer->dependent;
        counts[5] = static_cast<long>(tracer->words);
        counts[6] = tracer->hilbert && tracer->proved_degrees;
        counts[8] = static_cast<long>(prime);
        std::vector<uint32_t> primes(replays);
        for (uint32_t& p : primes) p = static_cast<uint32_t>(prime = previous(prime));
        std::atomic<size_t> cursor{0};
        std::atomic<long> agreeing{0};
        clock = std::chrono::steady_clock::now();
        std::mutex lock;
        std::exception_ptr error;
        auto worker = [&] {
            try {
                std::vector<uint32_t> residues, elements;
                for (size_t b; (b = cursor.fetch_add(1)) < primes.size();)
                    agreeing += tracer->replay(generators, primes[b], residues, 1, direct ? &elements : nullptr);
            } catch (...) {
                std::lock_guard<std::mutex> guard(lock);
                if (!error) error = std::current_exception();
                cursor.store(primes.size());
            }
        };
        std::vector<std::thread> crew;
        for (unsigned w = 1; w < std::min<size_t>(workers, primes.size()); ++w) crew.emplace_back([&] {
            worker();
            flint_cleanup();
        });
        worker();
        for (std::thread& t : crew) t.join();
        if (error) std::rethrow_exception(error);
        seconds[1] = since(clock);
        counts[7] = agreeing.load();
        if (basis) {
            std::string text;
            long e[MAX_VARIABLES];
            size_t i = 0;
            for (size_t g = 0; g < tracer->output_lengths.size(); ++g) {
                for (uint32_t t = 0; t < tracer->output_lengths[g]; ++t, ++i) {
                    if (t) text += ';';
                    text += std::to_string(tracer->output_residues[i]);
                    T.unpack(tracer->output_monomials[g].data() + static_cast<size_t>(t) * T.stride, e);
                    for (int v = 0; v < R.n; ++v) text += ' ' + std::to_string(e[v]);
                }
                text += '\n';
            }
            *basis = copy_text(text);
        }
        return 0;
    } catch (const std::exception& error) {
        set_error(error.what());
    } catch (...) {
        set_error("Unknown failure.");
    }
    return -1;
}

// As atlas_groebner, through modular images. `report` receives, when not
// null: primes used, primes rejected, traces, verifications, instructions,
// basis elements, and the seconds spent on the trace, the replays, the
// lifting and the certification (four doubles after six longs).
void* atlas_groebner_modular(const char* input, size_t length, long* counts, double* seconds) noexcept {
    try {
        const char* at = input;
        const char* end = input + length;
        char* stop = nullptr;
        const long n = std::strtol(at, &stop, 10);
        const long order = std::strtol(stop, &stop, 10);
        std::strtol(stop, &stop, 10);  // a declared prefix is of no use here
        at = stop;
        if (at >= end || *at != '\n' || order < 0 || order > 2) throw std::invalid_argument("Bad header.");
        ++at;
        auto engine = std::make_unique<Engine>();
        engine->R.init(static_cast<int>(n), static_cast<int>(order));
        std::vector<Poly> polys;
        while (at < end) polys.push_back(parse(engine->R, at, end, nullptr));
        ModularReport report;
        const auto start = std::chrono::steady_clock::now();
        const Ring R = engine->R;
        if (!modular_basis(R, polys, *engine, &report)) throw std::runtime_error("The modular route gave up.");
        engine->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        Z one;
        fmpz_one(&one.v);
        for (const Element& e : engine->G) write(engine->R, engine->text, e.p, &one.v, &e.p.c[0].v);
        if (counts) {
            counts[0] = report.primes; counts[1] = report.rejected; counts[2] = report.traces;
            counts[3] = report.verifications; counts[4] = report.instructions; counts[5] = report.basis;
            counts[6] = report.gpu;
        }
        if (seconds) {
            seconds[0] = report.trace_seconds; seconds[1] = report.replay_seconds;
            seconds[2] = report.lift_seconds; seconds[3] = report.verify_seconds;
        }
        return engine.release();
    } catch (const std::exception& error) {
        set_error(error.what());
    } catch (...) {
        set_error("Unknown failure.");
    }
    return nullptr;
}

}
