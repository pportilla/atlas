// Cofactors certify that a polynomial belongs to a rational ideal on a chart.
// A missing certificate leaves the full ideal computation to the caller.
#pragma once

#include "chart_expr.hpp"

namespace atlas::certificate {
using chart::Expr;
using chart::Ideal;
using chart::Variables;

struct CertificateCaps {
    unsigned degree_slack = 8, max_degree = 32;
    size_t max_rows = 512, max_cols = 512, max_nonzeros = 32768;
    size_t max_rank = 128;  // bound the dense solve for the proven submatrix
    void (*check_time)() = nullptr;
};
struct MemberCertificate {
    bool certified = false;
    Ideal cofactors;
    unsigned degree = 0;
    // Present on the localized and radical tests. The last certified generator
    // is u * t - 1, with t this variable.
    std::optional<chart::Coord> auxiliary;
};

bool verify_member(const Variables&, const Ideal&, const Expr&, const Ideal& cofactors);
MemberCertificate certify_member(const Variables&, const Ideal&, const Expr&, const CertificateCaps&);
MemberCertificate certify_unit(const Variables&, const Ideal&, const CertificateCaps&);
MemberCertificate certify_empty_on_open(const Variables&, const Ideal&, const Expr& unit, const CertificateCaps&);
MemberCertificate certify_radical_member(const Variables&, const Ideal&, const Expr&, const CertificateCaps&);

bool enabled();
bool trace(const char* site, bool certified);
}
