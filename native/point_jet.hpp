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
// On a smooth coordinate-line germ through the orbit, test the full
// restriction of f. An unproved germ or an exceeded bound yields no answer.
std::optional<bool> nonzero_on_coordinate_line(const chart::Variables&, const chart::Ideal& rows,
                                              const chart::Coord& parameter, const chart::Expr& f,
                                              const GaloisOrbit&, const CertificateCaps&);
}
