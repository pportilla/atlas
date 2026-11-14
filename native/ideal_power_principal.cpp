// In the factorial ring QQ[x], a squarefree principal ideal has unmixed
// ordinary powers, hence the same ordinary and symbolic powers.
#include "ideal_power_principal.hpp"
#include "ideal_powers.hpp"
#include "local_io.hpp"
#include "polynomial_factors.hpp"
#include "sparse_flint.hpp"

namespace atlas::ideal_powers {
struct PrincipalBasis::Storage {
    local::SparseLexRing ring;
    PolyValue generator;
    explicit Storage(size_t width) : ring(width),generator(ring.ctx) {}
};

PrincipalBasis::PrincipalBasis(const local::Polynomial& f,size_t width,bool lex,
                               const mpz_class& n,bool symbolic)
    : storage(std::make_unique<Storage>(width)) {
    auto& ring = storage->ring;
    auto out = storage->generator.p;
    PolyValue input(ring.ctx);
    ring.write(input.p,f);
    if (symbolic) {
        rational_squarefree_part(out,input.p,ring.ctx);
        PolyValue monic(ring.ctx);
        fmpq_mpoly_make_monic(monic.p,input.p,ring.ctx);
        if (!fmpq_mpoly_equal(monic.p,out,ring.ctx))
            throw InvalidInput("Symbolic powers require a prime or radical ideal over QQ.");
    }
    if (n == 0) {
        fmpq_mpoly_one(out,ring.ctx);
        return;
    }
    const auto& lc = f.at(local::global_head(f,lex));
    if (lc != 1) {
        fmpq_set_mpq(ring.coefficient,lc.get_mpq_t());
        fmpq_mpoly_scalar_div_fmpq(input.p,input.p,ring.coefficient,ring.ctx);
    }
    Z exponent;
    fmpz_set_mpz(&exponent.v,n.get_mpz_t());
    if (!fmpq_mpoly_pow_fmpz(out,input.p,&exponent.v,ring.ctx))
        throw std::runtime_error("Unable to expand polynomial power.");
}

PrincipalBasis::~PrincipalBasis() = default;
PrincipalBasis::PrincipalBasis(PrincipalBasis&&) noexcept = default;
PrincipalBasis& PrincipalBasis::operator=(PrincipalBasis&&) noexcept = default;

local::Basis PrincipalBasis::materialize() const {
    local::Basis basis;
    basis.push_back(storage->ring.read(storage->generator.p));
    return basis;
}

std::vector<std::string> PrincipalBasis::expressions(const std::vector<std::string>& names) const {
    return {local::format_polynomial(storage->ring.read(storage->generator.p),names)};
}
}
