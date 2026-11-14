#pragma once

#include "local_basis.hpp"
#include <flint/fmpq.h>
#include <cstdint>

namespace atlas::quotient_ring {

// Eight exponent digits fit in one word. A guard bit per digit checks
// divisibility and detects products that must return to the GMP reducer.
class PackedReducer {
    using Word = std::uint64_t;
    struct Fraction {
        fmpq_t q;
        Fraction() { fmpq_init(q); }
        Fraction(Fraction&& other) noexcept : Fraction() { fmpq_swap(q,other.q); }
        Fraction(const Fraction& other) : Fraction() { fmpq_set(q,other.q); }
        ~Fraction() { fmpq_clear(q); }
    };
    using Terms = std::map<Word,Fraction>;
    struct Rule { Word head; std::vector<std::pair<Word,Fraction>> tail; };
    size_t width_;
    std::vector<Rule> rules_;
    bool lex_;
    bool signed_binomial_ = true;
    bool normal_word(Word& power, bool& negative) const;
    explicit PackedReducer(size_t width, bool lex) : width_(width), lex_(lex) {}
    std::optional<local::Polynomial> finish(Terms& f) const;
public:
    static std::unique_ptr<PackedReducer> make(const local::Basis& basis, const std::vector<local::Powers>& heads,
                                               size_t width, bool lex);
    bool pairs_reduce_to_zero() const;
    std::optional<local::Polynomial> reduce(const local::Polynomial& a, const local::Polynomial* b,
                                             bool subtract, bool multiply) const;
};
}
