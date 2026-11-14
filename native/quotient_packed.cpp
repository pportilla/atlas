#include "quotient_packed.hpp"
#include <algorithm>

namespace atlas::quotient_ring {
namespace {
using Word = std::uint64_t;
constexpr Word guard = 0x8080808080808080ULL;
constexpr Word invalid = ~Word(0);

Word pack(const local::Powers& e, bool lex) {
    Word result = 0;
    for (size_t j = 0; j < e.size(); ++j) {
        const auto& k = e[lex ? j : e.size()-1-j];
        if (k > 127) return invalid;
        result = (result << 8) | k.get_ui();
    }
    return result;
}
}

std::unique_ptr<PackedReducer> PackedReducer::make(const local::Basis& basis,
        const std::vector<local::Powers>& heads, size_t width, bool lex) {
    if (width > 8 || basis.empty()) return nullptr;
    // Normal form does not depend on which reducible term is visited first.
    // Use lex digit order for binomials to emit terms directly into the GMP map.
    const bool binomial = std::all_of(basis.begin(),basis.end(),[](const auto& p) { return p.size() <= 2; });
    lex |= binomial;
    auto reducer = std::unique_ptr<PackedReducer>(new PackedReducer(width,lex));
    for (size_t j = 0; j < basis.size(); ++j) {
        Word h = pack(heads[j],lex);
        if (h == invalid) return nullptr;
        Rule rule{h,{}};
        for (const auto& [e,c] : basis[j]) if (e != heads[j]) {
            Word p = pack(e,lex);
            if (p == invalid) return nullptr;
            auto& coefficient = rule.tail.emplace_back(p,Fraction{}).second;
            fmpq_set_mpq(coefficient.q,c.get_mpq_t());
            fmpq_neg(coefficient.q,coefficient.q);
            if (c != 1 && c != -1) reducer->signed_binomial_ = false;
        }
        if (rule.tail.size() > 1) reducer->signed_binomial_ = false;
        reducer->rules_.push_back(std::move(rule));
    }
    return reducer;
}

bool PackedReducer::normal_word(Word& power, bool& negative) const {
    for (;;) {
        const auto it = std::find_if(rules_.begin(),rules_.end(),[power](const Rule& r) {
            return (((power | guard)-r.head) & guard) == guard;
        });
        if (it == rules_.end()) return true;
        if (it->tail.empty()) { power = invalid; return true; } // zero modulo a monomial generator
        const auto& [tail,coefficient] = it->tail.front();
        power = power-it->head+tail;
        if (power & guard) return false;
        if (fmpq_sgn(coefficient.q) < 0) negative = !negative;
    }
}

bool PackedReducer::pairs_reduce_to_zero() const {
    if (!signed_binomial_) return false;
    for (size_t i = 0; i < rules_.size(); ++i) for (size_t j = 0; j < i; ++j) {
        const auto& a = rules_[i];
        const auto& b = rules_[j];
        Word lcm = 0;
        bool disjoint = true;
        for (size_t k = 0; k < width_; ++k) {
            const Word u = (a.head >> (8*k)) & 127, v = (b.head >> (8*k)) & 127;
            disjoint &= !u || !v;
            lcm |= std::max(u,v) << (8*k);
        }
        if (disjoint) continue;
        auto rewrite = [&](const Rule& rule, Word& power, bool& negative) {
            if (rule.tail.empty()) { power = invalid; return true; }
            power = lcm-rule.head+rule.tail[0].first;
            negative = fmpq_sgn(rule.tail[0].second.q) < 0;
            return !(power & guard) && normal_word(power,negative);
        };
        Word left, right;
        bool negate_left = false, negate_right = false;
        if (!rewrite(a,left,negate_left) || !rewrite(b,right,negate_right) || left != right ||
            (left != invalid && negate_left != negate_right)) return false;
    }
    return true;
}

std::optional<local::Polynomial> PackedReducer::finish(Terms& f) const {
    local::Polynomial result;
    // Consume the highest packed power in lex; in grevlex any reducible
    // term is valid because the supplied basis is Gröbner.
    while (!f.empty()) {
        auto node = f.extract(std::prev(f.end()));
        const Word e = node.key();
        const auto c = node.mapped().q;
        if (fmpq_is_zero(c)) continue;
        auto rule = std::find_if(rules_.begin(),rules_.end(),[e](const Rule& r) {
            return (((e | guard)-r.head) & guard) == guard;
        });
        if (rule == rules_.end()) {
            local::Powers powers(width_);
            for (size_t i = 0; i < width_; ++i) {
                const auto k = static_cast<unsigned long>((e >> (8*i)) & 127);
                if (k) powers[lex_ ? width_-1-i : i] = k;
            }
            auto [it,inserted] = result.try_emplace(std::move(powers));
            if (inserted) fmpq_get_mpq(it->second.get_mpq_t(),c);
            else {
                mpq_class addend;
                fmpq_get_mpq(addend.get_mpq_t(),c);
                it->second += addend;
                if (!it->second) result.erase(it);
            }
        } else {
            const Word shift = e-rule->head;
            for (const auto& [tail,d] : rule->tail) {
                const Word term = shift + tail;
                if (term & guard) return std::nullopt;
                fmpq_addmul(f[term].q,d.q,c);
            }
        }
    }
    return result;
}

std::optional<local::Polynomial> PackedReducer::reduce(const local::Polynomial& a, const local::Polynomial* b,
                                                       bool subtract, bool multiply) const {
    Terms f;
    if (multiply) {
        if (!b || b->empty() || a.empty()) return local::Polynomial{};
        if (a.size() > 4096/b->size()) return std::nullopt;
        std::vector<std::pair<Word,Fraction>> right;
        for (const auto& [e,c] : *b) {
            Word p = pack(e,lex_);
            if (p == invalid) return std::nullopt;
            fmpq_set_mpq(right.emplace_back(p,Fraction{}).second.q,c.get_mpq_t());
        }
        Fraction coefficient;
        for (const auto& [e,c] : a) {
            const Word p = pack(e,lex_);
            if (p == invalid) return std::nullopt;
            fmpq_set_mpq(coefficient.q,c.get_mpq_t());
            for (const auto& [s,d] : right) {
                Word monomial = s+p;
                if (monomial & guard) return std::nullopt;
                bool negative = false;
                if (signed_binomial_ && !normal_word(monomial,negative)) return std::nullopt;
                if (monomial == invalid) continue;
                auto& sum = f[monomial];
                if (negative) fmpq_submul(sum.q,coefficient.q,d.q);
                else fmpq_addmul(sum.q,coefficient.q,d.q);
            }
        }
    } else {
        for (const auto& [e,c] : a) {
            const Word p = pack(e,lex_);
            if (p == invalid) return std::nullopt;
            fmpq_set_mpq(f[p].q,c.get_mpq_t());
        }
        if (b) {
            Fraction coefficient;
            for (const auto& [e,c] : *b) {
                const Word p = pack(e,lex_);
                if (p == invalid) return std::nullopt;
                fmpq_set_mpq(coefficient.q,c.get_mpq_t());
                auto& sum = f[p];
                if (subtract) fmpq_sub(sum.q,sum.q,coefficient.q);
                else fmpq_add(sum.q,sum.q,coefficient.q);
            }
        }
    }
    return finish(f);
}
}
