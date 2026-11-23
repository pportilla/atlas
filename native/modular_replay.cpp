#include "modular_replay.hpp"
#include "groebner.hpp"
#include <flint/ulong_extras.h>

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <system_error>
#include <thread>

namespace atlas {

// Run code[from, to) modulo p. Work entries stay below p*p.
bool execute_replay_block(const uint32_t* code, size_t from, size_t to, uint32_t p,
                          uint32_t* pool, uint64_t* work) {
    const uint64_t square = static_cast<uint64_t>(p) * p;
    size_t at = from;
    while (at < to) {
        switch (code[at]) {
        case OP_BEGIN:
            std::fill(work, work + code[at + 1], 0);
            at += 2;
            break;
        case OP_ADD: {
            const uint32_t* source = pool + code[at + 1];
            const uint32_t count = code[at + 2];
            const uint32_t* index = code + at + 3;
            for (uint32_t t = 0; t < count; ++t) {
                uint64_t& w = work[index[t]];
                w += source[t];
                if (w >= square) w -= square;
            }
            at += 3 + count;
            break;
        }
        case OP_SUB: {
            const uint32_t* source = pool + code[at + 1];
            const uint32_t count = code[at + 2];
            const uint32_t* index = code + at + 3;
            for (uint32_t t = 0; t < count; ++t) {
                uint64_t& w = work[index[t]];
                w += p - source[t];
                if (w >= square) w -= square;
            }
            at += 3 + count;
            break;
        }
        case OP_RED: {
            uint64_t& pivot = work[code[at + 1]];
            const uint32_t* source = pool + code[at + 2];
            const uint32_t count = code[at + 3];
            const uint32_t* index = code + at + 4;
            const uint64_t f = pivot % p;
            pivot = 0;
            if (f) {
                const uint64_t factor = p - f;
                for (uint32_t t = 0; t < count; ++t) {
                    uint64_t& w = work[index[t]];
                    w += factor * source[t];
                    if (w >= square) w -= square;
                }
            }
            at += 4 + count;
            break;
        }
        case OP_CHECK: {
            const uint32_t count = code[at + 1];
            const uint32_t* index = code + at + 2;
            for (uint32_t t = 0; t < count; ++t)
                if (work[index[t]] % p) return false;
            at += 2 + count;
            break;
        }
        case OP_STORE: {
            uint32_t* target = pool + code[at + 1];
            const uint32_t count = code[at + 2];
            const uint32_t* index = code + at + 3;
            const uint64_t lead = work[index[0]] % p;
            if (!lead) return false;
            const uint64_t inverse = n_invmod(lead, p);
            target[0] = 1;
            for (uint32_t t = 1; t < count; ++t) target[t] = static_cast<uint32_t>((work[index[t]] % p) * inverse % p);
            at += 3 + count;
            break;
        }
        default:
            throw std::logic_error("Corrupt replay program.");
        }
    }
    return true;
}

void replay_on_host(const std::vector<uint32_t>& code, size_t pool_size, size_t work_size,
                    const std::vector<uint32_t>& primes, std::vector<uint32_t>& pools,
                    std::vector<uint32_t>& status) {
    if (primes.empty() || code.empty()) return;
    const size_t lanes = std::min<size_t>(primes.size(), std::max(1u, std::thread::hardware_concurrency()));
    std::vector<uint64_t> scratch(lanes * work_size);
    std::atomic<size_t> cursor{0};
    // The workers stop once the caller's deadline passes; see check_deadline.
    const Deadline limit = thread_deadline();
    std::atomic<bool> stop{false};
    auto worker = [&](size_t lane) {
        uint64_t* work = scratch.data() + lane * work_size;
        for (size_t b; !stop.load(std::memory_order_relaxed) && (b = cursor.fetch_add(1)) < primes.size();)
            if (passed(limit)) stop.store(true, std::memory_order_relaxed);
            else if (status[b] && !execute_replay_block(code.data(), 0, code.size(), primes[b],
                                                   pools.data() + b * pool_size, work))
                status[b] = 0;
    };
    std::vector<std::thread> crew;
    crew.reserve(lanes - 1);
    try {
        for (size_t lane = 1; lane < lanes; ++lane) crew.emplace_back(worker, lane);
    } catch (const std::system_error&) {
        // Overlapping solves may exhaust the available worker threads. Those
        // already running and the calling thread consume the remaining primes.
    }
    worker(0);
    for (std::thread& thread : crew) thread.join();
    if (stop.load()) expire();
}

}  // namespace atlas
