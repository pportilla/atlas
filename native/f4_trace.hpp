// F4 modulo a prime, recorded for replay modulo other primes (KERNEL.md,
// sections 7.5 and 7.6).
//
// Each degree is one matrix: the rows to reduce (S-pair halves and the
// generators of that degree) and the pivot rows found by symbolic
// preprocessing. Instead of reducing every row, the tracer reduces random
// linear combinations of them, as many as the degree needs. Each combination
// is a lane of one dense block, and every pivot row is applied to all lanes
// at once. When the number of generators is at most the number of variables,
// the Hilbert function of a complete intersection bounds the number of new
// leading monomials in each degree; reaching it proves the degree complete.
//
// Generators that are not homogeneous are traced the same way, with pairs
// taken by the degree of their lcm. Their trace can select, in every matrix,
// rows whose exact reduction over Q later proves that the lifted elements lie
// in the ideal (rows_vanish).
#pragma once

#include "groebner.hpp"

#include <cstdint>
#include <vector>

namespace atlas {

class F4Trace : public Engine {
public:
    // Rows of one matrix: row k is the polynomial `src[k]` (a generator, or
    // a basis element after the generators) times a monomial, given by the
    // columns of its terms, cols[start[k] .. start[k+1]).
    struct Rows {
        std::vector<uint32_t> src, start{0}, cols;
        size_t size() const { return src.size(); }
    };
    // The symbolic part of one matrix. Pivot row c reduces column c < nA.
    // Columns nA .. nA+nB-1 have no pivot row. New element k has its leading
    // monomial in column lead[k] and its terms in the columns
    // nA + support[support_start[k] .. support_start[k+1]).
    struct Step {
        Rows pivots, todo;
        uint32_t nA = 0, nB = 0;
        std::vector<uint32_t> lead, support_start{0}, support;
        uint32_t first = 0;            // the index of its first new element among all elements
        std::vector<uint32_t> useful;  // rows to reduce that are independent modulo the pivot rows
    };
    struct Options {
        // Give up at the first degree that misses its Hilbert bound, or at
        // once without a bound (missed_bound).
        bool stop_at_missed_bound = false;
        // Record in every matrix as many rows to reduce as it has new
        // elements, independent modulo the pivot rows: for rows_vanish.
        bool select_rows = false;
    };

    // The reduced basis at the trace prime: monomials (stride words each),
    // residues (monic) and the length of each element. Largest head first.
    std::vector<std::vector<u64>> output_monomials;
    std::vector<uint32_t> output_lengths;
    std::vector<uint32_t> output_residues;

    bool too_large = false;      // the recorded structure outgrew the budget
    bool homogeneous = true;     // every generator is homogeneous
    bool hilbert = false;        // the a priori Hilbert bound applies
    bool proved_degrees = true;  // every matrix met its Hilbert bound
    bool missed_bound = false;   // the trace stopped as Options::stop_at_missed_bound asks
    long matrices = 0, lanes = 0, dependent = 0;
    size_t words = 0;            // recorded structure, in 32-bit words

    // The reduced basis modulo `prime` of nonconstant generators. They need
    // not be homogeneous: pairs are always taken by the degree of their lcm.
    // False when the prime divides a leading coefficient, when the recorded
    // structure exceeds `budget` words (too_large), or as the options say.
    bool trace(const Ring& ring, const std::vector<Poly>& generators, uint32_t prime, size_t budget, unsigned threads,
               const Options& options);
    bool trace(const Ring& ring, const std::vector<Poly>& generators, uint32_t prime, size_t budget, unsigned threads);
    // The residues of the reduced basis modulo another prime, in the order of
    // the outputs, from the recorded matrices. False when the prime disagrees
    // with the trace: a leading coefficient vanishes, a matrix has a smaller
    // rank or other pivots, or a coefficient outside the traced terms is nonzero.
    // `elements`, when given, receives the residues of every element the
    // trace inserted, in the order and on the terms of G.
    bool replay(const std::vector<Poly>& generators, uint32_t prime, std::vector<uint32_t>& residues, unsigned threads,
                std::vector<uint32_t>* elements = nullptr) const;
    // The residues at the trace prime of every element inserted, as `elements` above.
    std::vector<uint32_t> element_residues() const;
    // Exact check over Q for a trace with Options::select_rows (KERNEL.md 7.6).
    // `generators` are those of the trace; `elements` and `outputs` are integer
    // multiples of the lifted elements and of the lifted reduced basis, on the
    // terms of G and of output_monomials. True when every selected row of
    // every matrix reduces to zero by the pivot rows and the new elements of
    // its matrix, and every output is its element reduced by the pivot rows of
    // the last matrix. Then all of them lie in the ideal of the generators.
    bool rows_vanish(const std::vector<Poly>& generators, const std::vector<Poly>& elements,
                     const std::vector<Poly>& outputs, unsigned threads) const;

private:
    uint32_t p_ = 0;
    size_t ninputs_ = 0;
    unsigned threads_ = 1;
    bool selected_ = false;                    // the trace recorded the rows for rows_vanish
    std::vector<std::vector<u64>> hash_;       // additive hash of each term of each polynomial
    std::vector<std::vector<uint32_t>> coef_;  // residues at the trace prime (generators, then basis elements)
    std::vector<Step> steps_;
    Step final_;                               // interreduction of the minimal elements

    const u64* terms(uint32_t id, const std::vector<Poly>& generators) const {
        return id < ninputs_ ? generators[id].m.data() : G[id - ninputs_].p.m.data();
    }
    size_t length(uint32_t id, const std::vector<Poly>& generators) const {
        return id < ninputs_ ? generators[id].size() : G[id - ninputs_].p.size();
    }
    bool exact_zero(const Step& st, const std::vector<int>& new_of, uint32_t row, long output,
                    const std::vector<Poly>& generators, const std::vector<Poly>& elements,
                    const std::vector<Poly>& outputs, std::vector<Z>& work) const;
};

}  // namespace atlas
