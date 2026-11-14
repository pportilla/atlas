#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace atlas {

// The replay program is a sequence of blocks. Each block computes one
// polynomial in a work vector indexed by the monomials that can occur:
//   BEGIN width                          clear the work vector
//   ADD offset count index...            work[index[t]] += pool[offset + t]
//   SUB offset count index...            work[index[t]] -= pool[offset + t]
//   RED column offset count index...     f = work[column]; work[column] = 0;
//                                        work[index[t]] -= f * pool[offset + t]
//   CHECK count index...                 these entries must be zero
//   STORE offset count index...          pool[offset + t] = work[index[t]] / work[index[0]],
//                                        and work[index[0]] must not be zero
// The pool holds the coefficients of every polynomial of the computation,
// each one monic; offsets and indices are fixed, only the prime varies.
enum ReplayOpcode : uint32_t { OP_BEGIN = 1, OP_ADD, OP_SUB, OP_RED, OP_CHECK, OP_STORE };

// Replay one block while building a trace. A failed check rejects the prime.
bool execute_replay_block(const uint32_t* code, size_t from, size_t to, uint32_t prime,
                          uint32_t* pool, uint64_t* work);

// Replay the complete trace over the valid primes in parallel. Each prime
// owns pool_size consecutive entries in pools and one success flag in status.
void replay_on_host(const std::vector<uint32_t>& code, size_t pool_size, size_t work_size,
                    const std::vector<uint32_t>& primes, std::vector<uint32_t>& pools,
                    std::vector<uint32_t>& status);

}  // namespace atlas
