// F4 modulo a prime with randomized, Hilbert-checked elimination, and its
// replay modulo further primes. The algorithm and its proofs are in KERNEL.md,
// section 7.5; modular.cpp lifts and certifies the result.
#include "f4_trace.hpp"

#include <flint/fmpz.h>
#include <flint/ulong_extras.h>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <mutex>
#include <system_error>
#include <thread>

namespace atlas {
namespace {

__extension__ typedef unsigned __int128 u128;

// Residues modulo a prime p < 2^31. Sums of products stay below p^2 < 2^62,
// so an accumulator absorbs one product before a conditional subtraction.
struct Field {
    uint32_t p;
    u64 p2, barrett;
    explicit Field(uint32_t prime) : p(prime), p2(static_cast<u64>(prime) * prime), barrett(~static_cast<u64>(0) / prime) {}
    uint32_t reduce(u64 a) const {  // a mod p, for any a
        const u64 q = static_cast<u64>((static_cast<u128>(a) * barrett) >> 64);
        u64 r = a - q * p;
        while (r >= p) r -= p;
        return static_cast<uint32_t>(r);
    }
    uint32_t inverse(uint32_t a) const { return static_cast<uint32_t>(n_invmod(a, p)); }
};

u64 splitmix(u64& state) {
    u64 z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// The monomials are added word by word, so this hash of a product is the sum
// of the hashes of its factors.
constexpr u64 HASH_WEIGHTS[MAX_STRIDE] = {0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL, 0x165667B19E3779F9ULL,
                                          0xD6E8FEB86659FD93ULL, 0xFF51AFD7ED558CCDULL, 0xC4CEB9FE1A85EC53ULL,
                                          0x94D049BB133111EBULL, 0xBF58476D1CE4E5B9ULL, 0x2545F4914F6CDD1DULL};
u64 additive_hash(const u64* m, int stride) {
    u64 h = 0;
    for (int i = 0; i < stride; ++i) h += m[i] * HASH_WEIGHTS[i];
    return h;
}

// The lane kernels. acc holds L lanes per column, acc[c*L + l], L a multiple
// of four. A factor is a residue in a 64-bit lane. On x86-64 an AVX2 version
// is chosen when the processor has it.
using AddRow = void (*)(u64*, int, const u64*, const uint32_t*, const uint32_t*, size_t, u64);
using Sweep = void (*)(u64*, int, uint32_t, uint32_t, const uint32_t*, const uint32_t*, const uint32_t*,
                       const uint32_t* const*, uint32_t, u64, u64);

// The lanes' factors for column c: minus the residue of each lane, which is
// cleared. False when every lane is zero there.
inline bool lane_factors(u64* ac, int L, u64* factor, uint32_t p, u64 barrett) {
    u64 any = 0;
    for (int l = 0; l < L; ++l) {
        const u64 a = ac[l];
        const u64 q = static_cast<u64>((static_cast<u128>(a) * barrett) >> 64);
        u64 r = a - q * p;
        r = r >= p ? r - p : r;
        factor[l] = r ? p - r : 0;
        any |= r;
        ac[l] = 0;
    }
    return any != 0;
}

// acc[cols[k]*L + l] += factor[l] * cf[k] for every term k of a row.
void add_row_plain(u64* acc, int L, const u64* factor, const uint32_t* cf, const uint32_t* cols, size_t len, u64 p2) {
    for (size_t k = 0; k < len; ++k) {
        u64* dst = acc + static_cast<size_t>(cols[k]) * L;
        const u64 b = cf[k];
        for (int l = 0; l < L; ++l) {
            const u64 t = dst[l] + factor[l] * b;
            dst[l] = t >= p2 ? t - p2 : t;
        }
    }
}

// Forward elimination of all lanes by the pivot rows, columns [from, nA).
// Pivot row c is monic; its tail is subtracted with the lane's coefficient
// in column c, which is then cleared.
void sweep_plain(u64* acc, int L, uint32_t from, uint32_t nA, const uint32_t* src, const uint32_t* start,
                 const uint32_t* cols, const uint32_t* const* coefficients, uint32_t p, u64 p2, u64 barrett) {
    u64 factor[32];
    for (uint32_t c = from; c < nA; ++c) {
        if (!lane_factors(acc + static_cast<size_t>(c) * L, L, factor, p, barrett)) continue;
        add_row_plain(acc, L, factor, coefficients[src[c]] + 1, cols + start[c] + 1, start[c + 1] - start[c] - 1, p2);
    }
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
void add_row_avx2(u64* acc, int L, const u64* factor, const uint32_t* cf, const uint32_t* cols, size_t len, u64 p2) {
    const __m256i square = _mm256_set1_epi64x(static_cast<long long>(p2));
    const __m256i below = _mm256_set1_epi64x(static_cast<long long>(p2 - 1));
    const int V = L / 4;
    __m256i f[8];
    for (int v = 0; v < V; ++v) f[v] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(factor + 4 * v));
    for (size_t k = 0; k < len; ++k) {
        __m256i* dst = reinterpret_cast<__m256i*>(acc + static_cast<size_t>(cols[k]) * L);
        const __m256i b = _mm256_set1_epi64x(cf[k]);
        for (int v = 0; v < V; ++v) {
            // Factors and coefficients are below 2^31: the low halves multiply exactly.
            __m256i t = _mm256_add_epi64(_mm256_loadu_si256(dst + v), _mm256_mul_epu32(f[v], b));
            t = _mm256_sub_epi64(t, _mm256_and_si256(_mm256_cmpgt_epi64(t, below), square));
            _mm256_storeu_si256(dst + v, t);
        }
    }
}

__attribute__((target("avx2")))
void sweep_avx2(u64* acc, int L, uint32_t from, uint32_t nA, const uint32_t* src, const uint32_t* start,
                const uint32_t* cols, const uint32_t* const* coefficients, uint32_t p, u64 p2, u64 barrett) {
    u64 factor[32];
    for (uint32_t c = from; c < nA; ++c) {
        if (!lane_factors(acc + static_cast<size_t>(c) * L, L, factor, p, barrett)) continue;
        add_row_avx2(acc, L, factor, coefficients[src[c]] + 1, cols + start[c] + 1, start[c + 1] - start[c] - 1, p2);
    }
}

const bool HAS_AVX2 = __builtin_cpu_supports("avx2");
const AddRow lanes_add_row = HAS_AVX2 ? add_row_avx2 : add_row_plain;
const Sweep lanes_sweep = HAS_AVX2 ? sweep_avx2 : sweep_plain;
#else
const AddRow lanes_add_row = add_row_plain;
const Sweep lanes_sweep = sweep_plain;
#endif

// Open addressing table of the monomials of one matrix.
class Columns {
public:
    explicit Columns(int stride) : S(stride) { slots_.assign(1 << 12, 0); }
    size_t size() const { return hash_.size(); }
    const u64* monomial(uint32_t id) const { return arena_.data() + static_cast<size_t>(id) * S; }
    uint32_t get(const u64* m, u64 h) {
        size_t mask = slots_.size() - 1;
        for (size_t at = slot(h) & mask;; at = (at + 1) & mask) {
            const uint32_t s = slots_[at];
            if (!s) {
                const uint32_t id = static_cast<uint32_t>(hash_.size());
                arena_.insert(arena_.end(), m, m + S);
                hash_.push_back(h);
                slots_[at] = id + 1;
                if (2 * hash_.size() > slots_.size()) grow();
                return id;
            }
            if (hash_[s - 1] == h && std::equal(m, m + S, monomial(s - 1))) return s - 1;
        }
    }

private:
    int S;
    std::vector<u64> arena_, hash_;
    std::vector<uint32_t> slots_;
    static size_t slot(u64 h) {
        h ^= h >> 31;
        h *= 0xBF58476D1CE4E5B9ULL;
        return static_cast<size_t>(h ^ (h >> 32));
    }
    void grow() {
        slots_.assign(slots_.size() * 2, 0);
        const size_t mask = slots_.size() - 1;
        for (uint32_t id = 0; id < hash_.size(); ++id) {
            size_t at = slot(hash_[id]) & mask;
            while (slots_[at]) at = (at + 1) & mask;
            slots_[at] = id + 1;
        }
    }
};

// Rows of the reduced echelon form on the columns without a pivot row,
// indexed by their first column.
struct Echelon {
    uint32_t nB = 0;
    std::vector<int> row_of;                 // column -> row, or -1
    std::vector<std::vector<uint32_t>> rows;  // dense on the nB columns, monic
    std::vector<u64> work;
    explicit Echelon(uint32_t width) : nB(width), row_of(width, -1), work(width) {}

    // Reduce v by the rows so far; a nonzero remainder becomes a new row.
    bool insert(const uint32_t* v, const Field& F) {
        for (uint32_t c = 0; c < nB; ++c) work[c] = v[c];
        for (uint32_t c = 0; c < nB; ++c) {
            const uint32_t a = F.reduce(work[c]);
            work[c] = 0;
            if (!a) continue;
            if (row_of[c] < 0) {
                const u64 inverse = F.inverse(a);
                std::vector<uint32_t> row(nB, 0);
                row[c] = 1;
                for (uint32_t q = c + 1; q < nB; ++q) row[q] = F.reduce(F.reduce(work[q]) * inverse);
                row_of[c] = static_cast<int>(rows.size());
                rows.push_back(std::move(row));
                return true;
            }
            const uint32_t* r = rows[row_of[c]].data();
            const u64 factor = F.p - a;
            for (uint32_t q = c + 1; q < nB; ++q) {
                const u64 t = work[q] + factor * r[q];
                work[q] = t >= F.p2 ? t - F.p2 : t;
            }
        }
        return false;
    }
    // Clear the columns of the other leading entries in every row; the leading
    // columns come back in increasing order.
    std::vector<uint32_t> reduce(const Field& F) {
        std::vector<uint32_t> leads;
        for (uint32_t c = 0; c < nB; ++c)
            if (row_of[c] >= 0) leads.push_back(c);
        for (size_t a = leads.size(); a-- > 0;) {
            std::vector<uint32_t>& row = rows[row_of[leads[a]]];
            for (uint32_t q = leads[a]; q < nB; ++q) work[q] = row[q];
            for (size_t b = a + 1; b < leads.size(); ++b) {
                const uint32_t c = leads[b], v = F.reduce(work[c]);
                work[c] = 0;
                if (!v) continue;
                const uint32_t* r = rows[row_of[c]].data();
                const u64 factor = F.p - v;
                for (uint32_t q = c + 1; q < nB; ++q) {
                    const u64 t = work[q] + factor * r[q];
                    work[q] = t >= F.p2 ? t - F.p2 : t;
                }
            }
            for (uint32_t q = leads[a]; q < nB; ++q) row[q] = F.reduce(work[q]);
            std::fill(work.begin(), work.end(), 0);
        }
        return leads;
    }
};

constexpr int MAX_LANES = 32;

// The lanes of a block for a matrix with `columns` columns: at most 32, and
// at most 64 MiB of accumulators.
int lane_width(size_t columns, long wanted) {
    long lanes = std::min<long>(MAX_LANES, (wanted + 3) / 4 * 4);
    const size_t limit = (static_cast<size_t>(64) << 20) / (8 * std::max<size_t>(columns, 1));
    lanes = std::min<long>(lanes, static_cast<long>(limit / 4 * 4));
    return static_cast<int>(std::max<long>(lanes, 4));
}

// Reduce `count` random combinations of the rows to reduce (a block of lanes)
// and return their parts on the columns without a pivot row, lane by lane.
void reduce_block(const F4Trace::Step& st, const std::vector<const uint32_t*>& coefficients, const Field& F,
                  int L, u64 seed, uint32_t from, std::vector<u64>& acc, std::vector<uint32_t>& out) {
    const size_t nc = static_cast<size_t>(st.nA) + st.nB;
    acc.assign(nc * L, 0);
    u64 factor[MAX_LANES];
    u64 state = seed;
    const F4Trace::Rows& todo = st.todo;
    for (size_t i = 0; i < todo.size(); ++i) {
        for (int l = 0; l < L; ++l) factor[l] = F.reduce(splitmix(state) >> 2);
        lanes_add_row(acc.data(), L, factor, coefficients[todo.src[i]], todo.cols.data() + todo.start[i],
                      todo.start[i + 1] - todo.start[i], F.p2);
    }
    lanes_sweep(acc.data(), L, from, st.nA, st.pivots.src.data(), st.pivots.start.data(), st.pivots.cols.data(),
                coefficients.data(), F.p, F.p2, F.barrett);
    out.assign(static_cast<size_t>(L) * st.nB, 0);
    for (uint32_t c = 0; c < st.nB; ++c)
        for (int l = 0; l < L; ++l) out[static_cast<size_t>(l) * st.nB + c] = F.reduce(acc[(st.nA + static_cast<size_t>(c)) * L + l]);
}

// Run job(0..count-1) on up to `threads` threads. An exception of a job is
// rethrown here once every thread has stopped.
template<class Job>
void run_parallel(size_t count, unsigned threads, Job job) {
    std::atomic<size_t> cursor{0};
    std::mutex lock;
    std::exception_ptr error;
    // The workers stop at the next job once the caller's deadline passes; see check_deadline.
    const Deadline limit = thread_deadline();
    auto worker = [&] {
        try {
            for (size_t k; (k = cursor.fetch_add(1, std::memory_order_relaxed)) < count;) {
                if (passed(limit)) throw Expired();
                job(k);
            }
        } catch (...) {
            std::lock_guard<std::mutex> guard(lock);
            if (!error) error = std::current_exception();
            cursor.store(count);
        }
    };
    std::vector<std::thread> crew;
    const size_t workers = std::min<size_t>(count, std::max(1U, threads));
    try {
        for (size_t w = 1; w < workers; ++w) crew.emplace_back(worker);
    } catch (const std::system_error&) {
        // Thread creation failed: the threads started and this one do every job.
    }
    worker();
    for (std::thread& t : crew) t.join();
    if (error) {
        try {
            std::rethrow_exception(error);
        } catch (const Expired&) {
            expire();  // on the calling thread, which owns the deadline
        }
    }
}

// Reduce the rows to reduce numbered [begin, end) themselves by the pivot
// rows, a block of lanes at a time, and return their parts on the columns
// without a pivot row. With `tails`, the leading term of each row is left
// out: its tail is reduced.
std::vector<std::vector<uint32_t>> reduce_rows(const F4Trace::Step& st, const std::vector<const uint32_t*>& coefficients,
                                               const Field& F, unsigned threads, bool tails, size_t begin, size_t end) {
    const size_t m = end - begin, nc = static_cast<size_t>(st.nA) + st.nB;
    const int L = lane_width(nc, static_cast<long>(std::min<size_t>(m, MAX_LANES)));
    const size_t blocks = (m + L - 1) / L;
    std::vector<std::vector<uint32_t>> out(m, std::vector<uint32_t>(st.nB));
    run_parallel(blocks, threads, [&](size_t b) {
        std::vector<u64> acc(nc * L, 0);
        uint32_t from = st.nA;
        for (int l = 0; l < L && b * L + l < m; ++l) {
            const size_t i = begin + b * L + l;
            const uint32_t* cf = coefficients[st.todo.src[i]];
            for (uint32_t k = st.todo.start[i] + (tails ? 1 : 0); k < st.todo.start[i + 1]; ++k) {
                const uint32_t c = st.todo.cols[k];
                acc[static_cast<size_t>(c) * L + l] = cf[k - st.todo.start[i]];
                from = std::min(from, c);
            }
        }
        lanes_sweep(acc.data(), L, from, st.nA, st.pivots.src.data(), st.pivots.start.data(), st.pivots.cols.data(),
                    coefficients.data(), F.p, F.p2, F.barrett);
        for (int l = 0; l < L && b * L + l < m; ++l)
            for (uint32_t c = 0; c < st.nB; ++c)
                out[b * L + l][c] = F.reduce(acc[(st.nA + static_cast<size_t>(c)) * L + l]);
    });
    return out;
}

// The reduced echelon form of one matrix on its columns without a pivot row.
// `target` is the number of rows it should have (-1: unknown). A matrix with
// few rows to reduce is reduced row by row. Otherwise random combinations of
// those rows are reduced, `target` of them, or block after block until one
// is dependent on the earlier ones. With `exceed`, one combination more than
// the target is tried: if it is independent too, the result has more rows
// than the target, which shows the caller that the rank is larger. With
// `select`, as many rows to reduce as the form has rows are also found whose
// reduced parts are independent: `useful`.
struct Elimination {
    std::vector<uint32_t> leads;
    std::vector<std::vector<uint32_t>> rows;
    std::vector<uint32_t> useful;
    long lanes = 0, dependent = 0;
};
Elimination eliminate(const F4Trace::Step& st, const std::vector<const uint32_t*>& coefficients, const Field& F,
                      long target, bool exceed, bool select, unsigned threads, u64 seed) {
    Elimination result;
    Echelon E(st.nB);
    const long m = static_cast<long>(st.todo.size());
    const long cap = std::min<long>(m, st.nB);  // no matrix has a larger rank
    if (m <= (target >= 0 ? target + 8 : 2 * MAX_LANES)) {
        const std::vector<std::vector<uint32_t>> rows = reduce_rows(st, coefficients, F, threads, false, 0, st.todo.size());
        for (size_t i = 0; i < rows.size(); ++i) {
            ++result.lanes;
            if (E.insert(rows[i].data(), F)) result.useful.push_back(static_cast<uint32_t>(i));
            else ++result.dependent;
        }
    } else {
        uint32_t from = st.nA;
        for (uint32_t c : st.todo.cols) from = std::min(from, c);
        const size_t nc = static_cast<size_t>(st.nA) + st.nB;
        const long wanted = target >= 0 ? std::min(target + (exceed ? 1 : 0), cap) : cap;
        long found = 0;
        for (int round = 0; found < wanted; ++round) {
            // With a target, one block per thread until it is met; without, one
            // block, then two, four, ... as long as every lane is independent.
            const size_t most = target >= 0 ? std::max(1U, threads)
                                            : std::min<size_t>(std::max(1U, threads), static_cast<size_t>(1) << std::min(round, 8));
            std::vector<int> widths;
            for (long left = wanted - found + (target >= 0 ? 0 : 1); left > 0 && widths.size() < most;) {
                const int L = lane_width(nc, left);
                widths.push_back(L);
                left -= L;
            }
            std::vector<std::vector<uint32_t>> parts(widths.size());
            std::vector<std::vector<u64>> accs(widths.size());
            run_parallel(widths.size(), threads, [&](size_t b) {
                reduce_block(st, coefficients, F, widths[b], seed + 0x51ED27F3ULL * (round * 64 + b + 1), from, accs[b], parts[b]);
            });
            bool dependent = false;
            for (size_t b = 0; b < widths.size() && found < wanted; ++b)
                for (int l = 0; l < widths[b] && found < wanted; ++l) {
                    ++result.lanes;
                    if (E.insert(parts[b].data() + static_cast<size_t>(l) * st.nB, F)) ++found;
                    else { ++result.dependent; dependent = true; }
                }
            // A random combination of the rows that depends on the earlier ones
            // shows, with probability at least 1 - 1/p, that they span every row.
            // With a target that must be met the combinations are tried once more.
            if (dependent && (target < 0 || exceed || round >= 1)) break;
        }
        if (select) {
            // The combinations gave the rank. Rows are now reduced themselves,
            // a few blocks at a time, until that many are independent.
            Echelon chosen(st.nB);
            const size_t chunk = static_cast<size_t>(MAX_LANES) * std::max(1U, threads);
            for (size_t begin = 0; begin < st.todo.size() && static_cast<long>(result.useful.size()) < found; begin += chunk) {
                const size_t end = std::min(st.todo.size(), begin + chunk);
                const std::vector<std::vector<uint32_t>> rows = reduce_rows(st, coefficients, F, threads, false, begin, end);
                for (size_t i = 0; i < rows.size() && static_cast<long>(result.useful.size()) < found; ++i)
                    if (chosen.insert(rows[i].data(), F)) result.useful.push_back(static_cast<uint32_t>(begin + i));
            }
        }
    }
    result.leads = E.reduce(F);
    for (uint32_t c : result.leads) result.rows.push_back(std::move(E.rows[E.row_of[c]]));
    return result;
}

// A small matrix is reduced faster by one thread than by several started for it.
unsigned workers(const F4Trace::Step& st, unsigned threads) {
    return st.pivots.cols.size() + st.todo.cols.size() < (static_cast<size_t>(1) << 17) ? 1 : threads;
}

bool monic_residues(const Poly& f, const Field& F, std::vector<uint32_t>& out) {
    out.resize(f.size());
    const uint32_t lead = static_cast<uint32_t>(fmpz_fdiv_ui(&f.c[0].v, F.p));
    if (!lead) return false;
    const u64 inverse = F.inverse(lead);
    out[0] = 1;
    for (size_t t = 1; t < f.size(); ++t) out[t] = F.reduce(fmpz_fdiv_ui(&f.c[t].v, F.p) * inverse);
    return true;
}

// The coefficients of the numerator of a Hilbert series N(t)/(1-t)^n, without
// trailing zeros.
std::vector<Z> trimmed(std::vector<Z> numerator) {
    while (!numerator.empty() && fmpz_is_zero(&numerator.back().v)) numerator.pop_back();
    return numerator;
}
bool same(const std::vector<Z>& a, const std::vector<Z>& b) {
    if (a.size() != b.size()) return false;
    for (size_t k = 0; k < a.size(); ++k)
        if (!fmpz_equal(&a[k].v, &b[k].v)) return false;
    return true;
}

}  // namespace

bool F4Trace::trace(const Ring& ring, const std::vector<Poly>& generators, uint32_t prime, size_t budget, unsigned threads) {
    return trace(ring, generators, prime, budget, threads, Options());
}

bool F4Trace::trace(const Ring& ring, const std::vector<Poly>& generators, uint32_t prime, size_t budget, unsigned threads,
                    const Options& options) {
    R = ring;
    p_ = prime;
    threads_ = std::max(1U, threads);
    strategy_ = EAGER;
    rebuild_active();
    const Field F(prime);
    const int S = R.stride;
    ninputs_ = generators.size();
    coef_.resize(ninputs_);
    hash_.resize(ninputs_);
    for (size_t k = 0; k < ninputs_; ++k) {
        if (generators[k].empty() || generators[k].m[0] == 0) return false;  // constants are left to the other route
        if (!monic_residues(generators[k], F, coef_[k])) return false;
        for (size_t t = 0; t < generators[k].size(); ++t) hash_[k].push_back(additive_hash(generators[k].m.data() + t * S, S));
    }
    std::vector<size_t> order(ninputs_);
    for (size_t k = 0; k < ninputs_; ++k) order[k] = k;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return generators[a].m[0] < generators[b].m[0]; });

    // Homogeneous generators give homogeneous elements: pairs are taken by
    // degree. Otherwise they are taken by the degree of their lcm (the normal
    // strategy), so that an element of lower degree is used at once.
    homogeneous = true;
    for (const Poly& f : generators)
        for (size_t t = 1; t < f.size() && homogeneous; ++t) homogeneous = f.m[t * S] == f.m[0];
    // Hilbert bound: prod (1 - t^{d_k}) over (1-t)^n, valid for at most n
    // homogeneous generators.
    hilbert = homogeneous && ninputs_ <= static_cast<size_t>(R.n);
    selected_ = options.select_rows;
    if (options.stop_at_missed_bound && !hilbert) {
        missed_bound = true;
        return false;
    }
    std::vector<Z> bound;
    if (hilbert) {
        std::vector<Z> num(1);
        fmpz_one(&num[0].v);
        for (const Poly& f : generators) {
            const size_t d = f.m[0];
            std::vector<Z> next(num.size() + d);
            for (size_t k = 0; k < num.size(); ++k) {
                fmpz_add(&next[k].v, &next[k].v, &num[k].v);
                fmpz_sub(&next[k + d].v, &next[k + d].v, &num[k].v);
            }
            num = std::move(next);
        }
        bound = trimmed(std::move(num));
    }
    auto pointers = [&] {
        std::vector<const uint32_t*> ptr(coef_.size());
        for (size_t k = 0; k < coef_.size(); ++k) ptr[k] = coef_[k].data();
        return ptr;
    };

    size_t next_generator = 0;
    while (true) {
        check_deadline();
        const bool more_generators = next_generator < order.size();
        if (B.empty() && !more_generators) break;
        u64 d = more_generators ? generators[order[next_generator]].m[0] : ~static_cast<u64>(0);
        for (const Pair& pair : B) d = std::min<u64>(d, pair.lcm[0]);
        std::vector<Pair> pairs;
        size_t kept = 0;
        for (size_t k = 0; k < B.size(); ++k) {
            if (B[k].lcm[0] == d) pairs.push_back(B[k]);
            else B[kept++] = B[k];
        }
        B.resize(kept);
        std::vector<size_t> inputs;
        while (next_generator < order.size() && generators[order[next_generator]].m[0] == d) inputs.push_back(order[next_generator++]);

        // Once a degree has missed its bound the ideal is no complete
        // intersection: later degrees would miss theirs too, and the
        // combinations are counted by their dependence alone.
        long target = -1;
        if (hilbert && proved_degrees) {
            std::vector<uint16_t> heads;
            long e[MAX_VARIABLES];
            for (int g : active_) {
                R.unpack(lead(g), e);
                for (int v = 0; v < R.n; ++v) heads.push_back(static_cast<uint16_t>(e[v]));
            }
            const std::vector<Z> current = trimmed(hilbert_numerator(heads, R.n));
            if (same(current, bound)) {
                // The leading monomials have the Hilbert series of the bound:
                // the elements so far are a Groebner basis (KERNEL.md 7.5).
                B.clear();
                break;
            }
            Z have, least;
            hilbert_value(&have.v, current, R.n, static_cast<long>(d));
            hilbert_value(&least.v, bound, R.n, static_cast<long>(d));
            fmpz_sub(&have.v, &have.v, &least.v);
            if (fmpz_sgn(&have.v) < 0 || !fmpz_fits_si(&have.v)) {
                hilbert = false;
            } else {
                target = fmpz_get_si(&have.v);
                if (target == 0) continue;  // this degree is complete
            }
        }

        // Symbolic preprocessing: the rows of the pairs, one pivot row per lcm,
        // the generators of this degree, and a pivot row for every column a
        // current reducer divides.
        Columns table(S);
        struct Pending { uint32_t src; std::vector<uint32_t> ids; };
        std::vector<Pending> pivot_rows, todo_rows;
        std::vector<int> pivot_of;
        auto expand = [&](uint32_t src, const u64* shift, u64 shift_hash) {
            Pending row{src, {}};
            const size_t len = length(src, generators);
            const u64* m = terms(src, generators);
            row.ids.reserve(len);
            u64 mono[MAX_STRIDE];
            for (size_t t = 0; t < len; ++t) {
                const u64* term = m + t * S;
                if (shift) R.multiply(mono, term, shift);
                const uint32_t id = table.get(shift ? mono : term, shift_hash + hash_[src][t]);
                if (id >= pivot_of.size()) pivot_of.resize(id + 1, -1);
                row.ids.push_back(id);
            }
            return row;
        };
        std::sort(pairs.begin(), pairs.end(), [&](const Pair& a, const Pair& b) { return R.compare(a.lcm, b.lcm) > 0; });
        for (size_t a = 0; a < pairs.size();) {
            size_t b = a;
            std::vector<int> members;
            while (b < pairs.size() && R.equal(pairs[b].lcm, pairs[a].lcm)) {
                members.push_back(pairs[b].i);
                members.push_back(pairs[b].j);
                ++b;
            }
            std::sort(members.begin(), members.end());
            members.erase(std::unique(members.begin(), members.end()), members.end());
            int best = members[0];
            for (int g : members)
                if (G[g].p.size() < G[best].p.size()) best = g;
            for (int g : members) {
                u64 shift[MAX_STRIDE];
                R.divide(shift, pairs[a].lcm, lead(g));
                Pending row = expand(static_cast<uint32_t>(ninputs_ + g), shift, additive_hash(shift, S));
                if (g == best) {
                    pivot_of[row.ids[0]] = static_cast<int>(pivot_rows.size());
                    pivot_rows.push_back(std::move(row));
                } else {
                    todo_rows.push_back(std::move(row));
                }
            }
            a = b;
        }
        for (size_t k : inputs) todo_rows.push_back(expand(static_cast<uint32_t>(k), nullptr, 0));
        for (uint32_t id = 0; id < table.size(); ++id) {
            if (pivot_of[id] >= 0) continue;
            const u64* mono = table.monomial(id);
            const int g = reducer(mono);
            if (g < 0) continue;
            u64 shift[MAX_STRIDE];
            R.divide(shift, mono, lead(g));
            Pending row = expand(static_cast<uint32_t>(ninputs_ + g), shift, additive_hash(shift, S));
            pivot_of[id] = static_cast<int>(pivot_rows.size());
            pivot_rows.push_back(std::move(row));
        }
        // Columns: those with a pivot row first, each part in decreasing order.
        std::vector<uint32_t> A, Bc;
        for (uint32_t id = 0; id < table.size(); ++id) (pivot_of[id] >= 0 ? A : Bc).push_back(id);
        auto larger = [&](uint32_t a, uint32_t b) { return R.compare(table.monomial(a), table.monomial(b)) > 0; };
        std::sort(A.begin(), A.end(), larger);
        std::sort(Bc.begin(), Bc.end(), larger);
        std::vector<uint32_t> position(table.size());
        for (uint32_t c = 0; c < A.size(); ++c) position[A[c]] = c;
        for (uint32_t c = 0; c < Bc.size(); ++c) position[Bc[c]] = static_cast<uint32_t>(A.size()) + c;
        Step st;
        st.nA = static_cast<uint32_t>(A.size());
        st.nB = static_cast<uint32_t>(Bc.size());
        auto emit = [&](Rows& rows, const Pending& row) {
            rows.src.push_back(row.src);
            for (uint32_t id : row.ids) rows.cols.push_back(position[id]);
            rows.start.push_back(static_cast<uint32_t>(rows.cols.size()));
        };
        for (uint32_t id : A) emit(st.pivots, pivot_rows[pivot_of[id]]);
        for (const Pending& row : todo_rows) emit(st.todo, row);
        words += st.pivots.cols.size() + st.todo.cols.size() + 2 * (st.pivots.size() + st.todo.size());
        if (words > budget || static_cast<size_t>(st.nA) + st.nB > (static_cast<size_t>(1) << 22)) {
            too_large = true;
            return false;
        }

        Elimination found = eliminate(st, pointers(), F, target, false, options.select_rows, workers(st, threads_),
                                      (static_cast<u64>(prime) << 20) ^ (matrices * 0x9E3779B1ULL + d));
        ++matrices;
        lanes += found.lanes;
        dependent += found.dependent;
        if (target >= 0 && static_cast<long>(found.leads.size()) < target) {
            proved_degrees = false;
            if (options.stop_at_missed_bound) {
                missed_bound = true;
                return false;
            }
        }
        if (options.select_rows) {
            if (found.useful.size() != found.leads.size()) return false;  // not with a usable prime
            st.useful = std::move(found.useful);
        }
        st.first = static_cast<uint32_t>(G.size());
        // More new rows than the bound allows cannot happen (KERNEL.md 7.5);
        // should it, the bound is not used again.
        if (target >= 0 && static_cast<long>(found.leads.size()) > target) hilbert = proved_degrees = false;

        // The new elements: their leading monomials have no reducer.
        for (size_t k = 0; k < found.leads.size(); ++k) {
            const std::vector<uint32_t>& row = found.rows[k];
            st.lead.push_back(st.nA + found.leads[k]);
            Poly h;
            std::vector<uint32_t> c;
            std::vector<u64> hashes;
            for (uint32_t q = found.leads[k]; q < st.nB; ++q) {
                if (!row[q]) continue;
                st.support.push_back(q);
                const u64* mono = table.monomial(Bc[q]);
                h.m.insert(h.m.end(), mono, mono + S);
                hashes.push_back(additive_hash(mono, S));
                c.push_back(row[q]);
            }
            st.support_start.push_back(static_cast<uint32_t>(st.support.size()));
            h.c.resize(c.size());
            coef_.push_back(std::move(c));
            hash_.push_back(std::move(hashes));
            if (h.m[0] == 0) unit = true;  // a constant: the unit ideal
            const long degree = static_cast<long>(h.m[0]);
            insert(std::move(h), degree);
        }
        steps_.push_back(std::move(st));
        if (unit) {
            B.clear();
            break;
        }
    }

    // The reduced basis: the elements with minimal leading monomials, each
    // tail reduced by the others. Without homogeneity a new leading monomial
    // may divide another of the same matrix.
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
    std::sort(minimal.begin(), minimal.end(), [&](int a, int b) { return R.compare(lead(a), lead(b)) > 0; });
    for (Element& e : G) e.active = false;
    for (int a : minimal) G[a].active = true;
    rebuild_active();
    Columns table(S);
    std::vector<int> pivot_of;
    struct Pending { uint32_t src; std::vector<uint32_t> ids; };
    std::vector<Pending> pivot_rows, todo_rows;
    auto expand = [&](uint32_t src, const u64* shift, u64 shift_hash) {
        Pending row{src, {}};
        const size_t len = length(src, generators);
        const u64* m = terms(src, generators);
        u64 mono[MAX_STRIDE];
        for (size_t t = 0; t < len; ++t) {
            const u64* term = m + t * S;
            if (shift) R.multiply(mono, term, shift);
            const uint32_t id = table.get(shift ? mono : term, shift_hash + hash_[src][t]);
            if (id >= pivot_of.size()) pivot_of.resize(id + 1, -1);
            row.ids.push_back(id);
        }
        return row;
    };
    for (int g : minimal) todo_rows.push_back(expand(static_cast<uint32_t>(ninputs_ + g), nullptr, 0));
    for (uint32_t id = 0; id < table.size(); ++id) {
        const u64* mono = table.monomial(id);
        const int g = reducer(mono);
        if (g < 0) continue;
        u64 shift[MAX_STRIDE];
        R.divide(shift, mono, lead(g));
        Pending row = expand(static_cast<uint32_t>(ninputs_ + g), shift, additive_hash(shift, S));
        pivot_of[id] = static_cast<int>(pivot_rows.size());
        pivot_rows.push_back(std::move(row));
    }
    std::vector<uint32_t> A, Bc;
    for (uint32_t id = 0; id < table.size(); ++id) (pivot_of[id] >= 0 ? A : Bc).push_back(id);
    auto larger = [&](uint32_t a, uint32_t b) { return R.compare(table.monomial(a), table.monomial(b)) > 0; };
    std::sort(A.begin(), A.end(), larger);
    std::sort(Bc.begin(), Bc.end(), larger);
    std::vector<uint32_t> position(table.size());
    for (uint32_t c = 0; c < A.size(); ++c) position[A[c]] = c;
    for (uint32_t c = 0; c < Bc.size(); ++c) position[Bc[c]] = static_cast<uint32_t>(A.size()) + c;
    final_.nA = static_cast<uint32_t>(A.size());
    final_.nB = static_cast<uint32_t>(Bc.size());
    auto emit = [&](Rows& rows, const Pending& row) {
        rows.src.push_back(row.src);
        for (uint32_t id : row.ids) rows.cols.push_back(position[id]);
        rows.start.push_back(static_cast<uint32_t>(rows.cols.size()));
    };
    for (uint32_t id : A) emit(final_.pivots, pivot_rows[pivot_of[id]]);
    for (const Pending& row : todo_rows) emit(final_.todo, row);
    words += final_.pivots.cols.size() + final_.todo.cols.size() + 2 * (final_.pivots.size() + final_.todo.size());
    if (words > budget) {
        too_large = true;
        return false;
    }
    const std::vector<std::vector<uint32_t>> tails = reduce_rows(final_, pointers(), F, threads_, true, 0, final_.todo.size());
    for (size_t k = 0; k < minimal.size(); ++k) {
        const uint32_t head = final_.todo.cols[final_.todo.start[k]];
        final_.lead.push_back(head);
        std::vector<u64> monomials(lead(minimal[k]), lead(minimal[k]) + S);
        output_residues.push_back(1);
        uint32_t count = 1;
        for (uint32_t q = 0; q < final_.nB; ++q) {
            if (!tails[k][q] || final_.nA + q == head) continue;
            final_.support.push_back(q);
            const u64* mono = table.monomial(Bc[q]);
            monomials.insert(monomials.end(), mono, mono + S);
            output_residues.push_back(tails[k][q]);
            ++count;
        }
        final_.support_start.push_back(static_cast<uint32_t>(final_.support.size()));
        output_lengths.push_back(count);
        output_monomials.push_back(std::move(monomials));
    }
    return true;
}

bool F4Trace::replay(const std::vector<Poly>& generators, uint32_t prime, std::vector<uint32_t>& residues, unsigned threads,
                     std::vector<uint32_t>* elements) const {
    const Field F(prime);
    std::vector<std::vector<uint32_t>> coefficients(ninputs_);
    for (size_t k = 0; k < ninputs_; ++k)
        if (!monic_residues(generators[k], F, coefficients[k])) return false;
    std::vector<const uint32_t*> pointers;
    auto refresh = [&] {
        pointers.resize(coefficients.size());
        for (size_t k = 0; k < coefficients.size(); ++k) pointers[k] = coefficients[k].data();
    };
    refresh();
    for (size_t s = 0; s < steps_.size(); ++s) {
        const Step& st = steps_[s];
        const long target = static_cast<long>(st.lead.size());
        const Elimination found = eliminate(st, pointers, F, target, true, false, workers(st, threads),
                                            (static_cast<u64>(prime) << 20) ^ s);
        if (found.leads.size() != st.lead.size()) return false;
        for (size_t k = 0; k < found.leads.size(); ++k) {
            if (st.nA + found.leads[k] != st.lead[k]) return false;
            const std::vector<uint32_t>& row = found.rows[k];
            std::vector<uint32_t> c;
            uint32_t q = found.leads[k];
            for (uint32_t i = st.support_start[k]; i < st.support_start[k + 1]; ++i) {
                for (; q < st.support[i]; ++q)
                    if (row[q]) return false;  // a term the trace did not have
                c.push_back(row[q++]);
            }
            for (; q < st.nB; ++q)
                if (row[q]) return false;
            coefficients.push_back(std::move(c));
        }
        refresh();
    }
    const std::vector<std::vector<uint32_t>> tails = reduce_rows(final_, pointers, F, threads, true, 0, final_.todo.size());
    if (elements) {
        elements->clear();
        for (size_t k = ninputs_; k < coefficients.size(); ++k)
            elements->insert(elements->end(), coefficients[k].begin(), coefficients[k].end());
    }
    residues.clear();
    for (size_t k = 0; k < tails.size(); ++k) {
        residues.push_back(1);
        const uint32_t head = final_.lead[k];
        uint32_t i = final_.support_start[k];
        for (uint32_t q = 0; q < final_.nB; ++q) {
            if (final_.nA + q == head) continue;
            if (i < final_.support_start[k + 1] && final_.support[i] == q) {
                residues.push_back(tails[k][q]);
                ++i;
            } else if (tails[k][q]) {
                return false;
            }
        }
    }
    return true;
}

std::vector<uint32_t> F4Trace::element_residues() const {
    std::vector<uint32_t> out;
    for (size_t k = ninputs_; k < coef_.size(); ++k) out.insert(out.end(), coef_[k].begin(), coef_[k].end());
    return out;
}

// One row over Q, reduced without fractions: the row is scaled by what the
// leading coefficient of a pivot row lacks, and its content is removed now
// and then. `new_of` gives the new element whose leading column a column
// without a pivot row is, or -1. With output >= 0 the row is, up to a factor,
// that element minus its reduced form `outputs[output]`.
bool F4Trace::exact_zero(const Step& st, const std::vector<int>& new_of, uint32_t row, long output,
                         const std::vector<Poly>& generators, const std::vector<Poly>& elements,
                         const std::vector<Poly>& outputs, std::vector<Z>& work) const {
    auto polynomial = [&](uint32_t id) -> const Poly& { return id < ninputs_ ? generators[id] : elements[id - ninputs_]; };
    const uint32_t nA = st.nA, nc = st.nA + st.nB;
    if (work.size() < nc) work.resize(nc);
    uint32_t last = 0;  // no entry beyond this column is nonzero
    {
        const Poly& f = polynomial(st.todo.src[row]);
        const uint32_t* cols = st.todo.cols.data() + st.todo.start[row];
        const uint32_t count = st.todo.start[row + 1] - st.todo.start[row];
        if (output < 0) {
            for (uint32_t t = 0; t < count; ++t) {
                fmpz_set(&work[cols[t]].v, &f.c[t].v);
                last = std::max(last, cols[t]);
            }
        } else {
            const Poly& reduced = outputs[output];
            for (uint32_t t = 0; t < count; ++t) {
                fmpz_mul(&work[cols[t]].v, &f.c[t].v, &reduced.c[0].v);
                last = std::max(last, cols[t]);
            }
            fmpz_submul(&work[cols[0]].v, &f.c[0].v, &reduced.c[0].v);
            for (uint32_t i = st.support_start[output], t = 1; i < st.support_start[output + 1]; ++i, ++t) {
                const uint32_t c = nA + st.support[i];
                fmpz_submul(&work[c].v, &f.c[0].v, &reduced.c[t].v);
                last = std::max(last, c);
            }
        }
    }
    Z multiple, scale, gcd;
    unsigned scalings = 0;
    bool zero = true;
    for (uint32_t c = 0; c <= last && zero; ++c) {
        fmpz* pivot = &work[c].v;
        if (fmpz_is_zero(pivot)) continue;
        const Poly* g;
        const uint32_t* cols = nullptr;   // the columns of the terms of a pivot row
        const uint32_t* support = nullptr;  // or those of a new element, after nA
        uint32_t count;
        if (c < nA) {
            g = &polynomial(st.pivots.src[c]);
            cols = st.pivots.cols.data() + st.pivots.start[c];
            count = st.pivots.start[c + 1] - st.pivots.start[c];
        } else if (new_of[c - nA] >= 0) {
            const int k = new_of[c - nA];
            g = &elements[st.first + k];
            support = st.support.data() + st.support_start[k];
            count = st.support_start[k + 1] - st.support_start[k];
        } else {
            zero = false;
            break;
        }
        const fmpz* head = &g->c[0].v;
        if (fmpz_is_one(head)) {
            fmpz_neg(&multiple.v, pivot);
        } else {
            fmpz_gcd(&gcd.v, pivot, head);
            fmpz_divexact(&multiple.v, pivot, &gcd.v);
            fmpz_neg(&multiple.v, &multiple.v);
            fmpz_divexact(&scale.v, head, &gcd.v);
            if (!fmpz_is_one(&scale.v)) {
                for (uint32_t left = c + 1; left <= last; ++left)
                    if (!fmpz_is_zero(&work[left].v)) fmpz_mul(&work[left].v, &work[left].v, &scale.v);
                ++scalings;
            }
        }
        for (uint32_t t = 1; t < count; ++t) {
            const uint32_t target = cols ? cols[t] : nA + support[t];
            fmpz_addmul(&work[target].v, &multiple.v, &g->c[t].v);
            last = std::max(last, target);
        }
        fmpz_zero(pivot);
        if (scalings == 8) {
            scalings = 0;
            fmpz_zero(&gcd.v);
            for (uint32_t left = c + 1; left <= last && !fmpz_is_one(&gcd.v); ++left) fmpz_gcd(&gcd.v, &gcd.v, &work[left].v);
            if (!fmpz_is_zero(&gcd.v) && !fmpz_is_one(&gcd.v))
                for (uint32_t left = c + 1; left <= last; ++left)
                    if (!fmpz_is_zero(&work[left].v)) fmpz_divexact(&work[left].v, &work[left].v, &gcd.v);
        }
    }
    if (!zero)
        for (uint32_t c = 0; c <= last; ++c) fmpz_zero(&work[c].v);  // leave the scratch row clean
    return zero;
}

bool F4Trace::rows_vanish(const std::vector<Poly>& generators, const std::vector<Poly>& elements,
                          const std::vector<Poly>& outputs, unsigned threads) const {
    if (!selected_ || elements.size() != G.size() || outputs.size() != final_.todo.size()) return false;
    struct Job { const Step* st; const std::vector<int>* new_of; uint32_t row; long output; };
    std::vector<std::vector<int>> leads(steps_.size() + 1);
    std::vector<Job> jobs;
    for (size_t s = 0; s < steps_.size(); ++s) {
        const Step& st = steps_[s];
        if (st.useful.size() != st.lead.size()) return false;  // the trace selected no rows
        leads[s].assign(st.nB, -1);
        for (size_t k = 0; k < st.lead.size(); ++k) leads[s][st.lead[k] - st.nA] = static_cast<int>(k);
        for (uint32_t row : st.useful) jobs.push_back({&st, &leads[s], row, -1});
    }
    leads.back().assign(final_.nB, -1);
    for (size_t k = 0; k < final_.todo.size(); ++k)
        jobs.push_back({&final_, &leads.back(), static_cast<uint32_t>(k), static_cast<long>(k)});
    std::atomic<bool> failed{false};
    std::atomic<size_t> cursor{0};
    const size_t workers = std::max<size_t>(1, std::min<size_t>(threads, jobs.size()));
    // Each worker keeps one scratch row; a job leaves it zero.
    run_parallel(workers, static_cast<unsigned>(workers), [&](size_t) {
        std::vector<Z> work;
        for (size_t j; !failed.load(std::memory_order_relaxed) && (j = cursor.fetch_add(1, std::memory_order_relaxed)) < jobs.size();)
            if (!exact_zero(*jobs[j].st, *jobs[j].new_of, jobs[j].row, jobs[j].output, generators, elements, outputs, work))
                failed.store(true, std::memory_order_relaxed);
        flint_cleanup();
    });
    return !failed.load();
}

}  // namespace atlas
