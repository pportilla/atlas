#pragma once

#include <gmpxx.h>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace atlas { class Engine; }

namespace atlas::local {

using Powers = std::vector<mpz_class>;
using Polynomial = std::map<Powers, mpq_class>;
using Basis = std::vector<Polynomial>;

Powers local_head(const Polynomial& polynomial, bool lex);
bool global_less(const Powers& a, const Powers& b, bool lex);
const Powers& global_head(const Polynomial& polynomial, bool lex);
Basis standard_basis(const Basis& generators, const std::vector<std::string>& names, bool lex, bool prefer_modular = false);
std::optional<std::vector<std::vector<int>>> local_leading_supports(
    const Basis& centered, size_t width, bool lex, bool prefer_modular);

// A homogeneous rewrite preserves degree, so local division and global
// division in the same tie order have the same ordinary remainder, with unit 1.
class HomogeneousReducer {
    std::unique_ptr<Engine> engine_;
    std::optional<unsigned long> cutoff_;
    bool binomial_ = false;
    explicit HomogeneousReducer(std::unique_ptr<Engine> engine);
    friend std::optional<HomogeneousReducer> homogeneous_reducer(const Basis&, size_t, bool, bool);
public:
    HomogeneousReducer(HomogeneousReducer&&) noexcept;
    ~HomogeneousReducer();
    const std::optional<unsigned long>& finite_cutoff() const { return cutoff_; }
    Basis materialize() const;
    std::optional<Polynomial> reduce(const Polynomial& centered_query) const;
};

std::optional<HomogeneousReducer> homogeneous_reducer(const Basis& centered, size_t width,
                                                      bool lex, bool prefer_modular = false);
// A standard basis specialized for finite normal forms; decline if the
// finite-jet calculation cannot certify the result within its work budget.
std::optional<Basis> finite_standard_basis(const Basis& generators, const std::vector<std::string>& names, bool lex);

}
