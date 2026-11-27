#pragma once

#include "finite_locus.hpp"

namespace atlas::certificate {
using JetPowers = std::vector<unsigned>;
using FieldCoefficient = std::vector<chart::Q>;

struct PointJet {
    unsigned degree = 0;
    chart::Variables parameters;
    std::map<JetPowers, FieldCoefficient> coefficients; // ascending powers of the orbit parameter
};

// A zero truncation gives the lower bound degree + 1.
unsigned jet_order(const PointJet& jet);
std::optional<PointJet> taylor_jet(const chart::Variables&, const chart::Expr&, const GaloisOrbit&,
                                   unsigned degree, const CertificateCaps&);
std::optional<PointJet> frame_jet(const chart::Variables&, const chart::Ideal& rows,
                                  const chart::Variables& dependent, const chart::Expr& f,
                                  const GaloisOrbit&, unsigned degree, const CertificateCaps&);
}
