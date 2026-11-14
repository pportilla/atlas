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
// Nothing here is probabilistic: an unlucky prime can only cost time.
#include "groebner.hpp"
#include "groebner_verify.hpp"
#include "modular_replay.hpp"

#include <flint/fmpq.h>
#include <flint/ulong_extras.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
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
thread_local ReplayBatch scoped_batch = nullptr;
thread_local int has_scoped_batch = 0;
thread_local uint64_t cpu_batches = 0;
thread_local uint64_t cpu_primes = 0;

ReplayBatch active_batch() {
    return has_scoped_batch ? scoped_batch : default_batch.load();
}

// Merge new prime images into the existing CRT residues and reconstruct
// rational coefficients. The last image is held out as a witness.
bool reconstruct_coefficients(const std::vector<uint32_t>& primes, const std::vector<std::vector<uint32_t>>& images,
                              uint32_t trace_prime, size_t& used, Z& modulus, std::vector<Z>& lifted,
                              std::vector<LiftedRational>& rational) {
    const size_t coefficients = lifted.size();
    const uint32_t witness = primes.back();
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
    auto lift = [&](size_t begin, size_t end) {
        for (size_t k = first_prime; k < used; ++k) {
            const uint32_t p = primes[k], inverse = inverses[k - first_prime];
            const fmpz* product = &prefix_moduli[k - first_prime].v;
            for (size_t i = begin; i < end; ++i) {
                const uint32_t old = fmpz_fdiv_ui(&lifted[i].v, p);
                const uint32_t difference = images[k][i] + p - old;
                const u64 correction = static_cast<u64>(difference) * inverse % p;
                fmpz_addmul_ui(&lifted[i].v, product, correction);
            }
        }
        for (size_t i = begin; i < end && stable.load(std::memory_order_relaxed); ++i) {
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
    return stable;
}

double since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
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

            // Lift all images but the last, which serves as a witness.
            clock = std::chrono::steady_clock::now();
            std::vector<LiftedRational> rational(coefficients);
            const bool stable = reconstruct_coefficients(primes, images, trace_prime, used, modulus, lifted, rational);
            std::vector<Poly> basis;
            if (stable) {
                size_t i = 0;
                Z common, factor;
                for (size_t g = 0; g < tracer.outputs.size(); ++g) {
                    const uint32_t length = tracer.outputs[g].length;
                    fmpz_one(&common.v);
                    for (uint32_t t = 0; t < length; ++t) fmpz_lcm(&common.v, &common.v, fmpq_denref(&rational[i + t].value));
                    Poly f;
                    f.m = tracer.output_monomials[g];
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
            if (!stable) continue;

            // Certification over Q.
            clock = std::chrono::steady_clock::now();
            ++stats.verifications;
            ++certification_attempts;
            std::optional<bool> proved = HomogeneousCertificate{}.check(H, basis, generators);
            if (!proved.has_value()) {
                Engine check;
                check.R = H;
                proved = check.is_basis(basis);
                for (size_t k = 0; k < generators.size() && *proved; ++k) {
                    Poly f = generators[k];
                    check.reduce(f, 0, nullptr, nullptr);
                    proved = f.empty();
                }
            }
            const auto verifying = std::chrono::steady_clock::now() - clock;
            stats.verify_seconds += std::chrono::duration<double>(verifying).count();
            verification_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(verifying).count();
            if (!*proved) continue;
            std::vector<Poly> affine;
            for (const Poly& f : basis) affine.push_back(dehomogenize(H, R, f));
            // Every generator is declared part of a basis: no pair is formed,
            // so this only removes redundant elements and reduces the tails.
            result.R = R;
            result.compute(affine, affine.size());
            ++certified_bases;
            return true;
        }
    }
    return false;
}

}  // namespace atlas

using namespace atlas;

extern "C" {

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
void atlas_groebner_use_batch(ReplayBatch hook) noexcept { default_batch.store(hook); }

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
