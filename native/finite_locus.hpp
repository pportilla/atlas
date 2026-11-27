#pragma once

#include "membership_certificate.hpp"

namespace atlas::certificate {
struct GaloisOrbit {
    std::vector<chart::Q> mu;  // monic irreducible polynomial for the free parameter
    std::vector<std::vector<chart::Q>> coordinates;  // in user axis order, ascending powers
};
struct FiniteLocus {
    bool certified = false;
    std::vector<GaloisOrbit> orbits;
};
FiniteLocus finite_locus(const chart::Variables&, const chart::Ideal& equations,
                         const chart::Ideal& units, const CertificateCaps&);
}
