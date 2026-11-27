#include "finite_locus.hpp"
#include "groebner.hpp"
#include <iostream>

using namespace atlas::chart;
using namespace atlas::certificate;
namespace {
void ensure(bool test, const char* name) { if (!test) throw std::logic_error(name); }
}
int main() {
    const Variables vars{{"x", 0}, {"y", 0}};
    const auto x = symbol(vars[0]), y = symbol(vars[1]), one = constant(1);
    const CertificateCaps caps;
    auto empty = finite_locus(vars,{x, x-one}, {}, caps);
    ensure(empty.certified && empty.orbits.empty(), "the trivial frame is empty");
    auto split = finite_locus(vars, {power(x*y, 3), power(x-one, 2) * power(y-one, 2)}, {}, caps);
    ensure(split.certified && split.orbits.size() == 2, "computed right number of points after factorization");
    // A nontrivial orbit is a full QQ polynomial for its free variable.
    // The second coordinate stays rational over that same root.
    auto compound = finite_locus(vars, {(x-one)*(x*x+one), x*y-one}, {}, caps);
    ensure(compound.certified && compound.orbits.size()==2, "missing a geometric orbit");
    std::vector<unsigned> degrees;
    for (const auto& orbit : compound.orbits) {
        degrees.push_back(orbit.mu.size()-1);
        ensure(orbit.coordinates.size() == 2 && orbit.mu.back() == 1, "incorrect coordinate dimensions");
    }
    std::sort(degrees.begin(), degrees.end());
    ensure(degrees == std::vector<unsigned>({1,2}), "wrong Galois degrees");
    auto one_orbit = finite_locus(vars, {(x-one)*(x*x+one), x*y-one}, {x-one}, caps);
    ensure(one_orbit.certified && one_orbit.orbits.size()==1 && one_orbit.orbits[0].mu.size()==3, "variable on open");
    // The origin and (5,-3) both become zero under the first separating
    // direction 3x+5y. The prime test alone would only find one forbidden
    // point, so the linear form must get another chance to separate them.
    const auto moving = x - constant(5);
    auto closed = finite_locus(vars, {x * moving, x + scaled(y, Q(5,3))}, {}, caps);
    ensure(closed.certified && closed.orbits.size()==2, "nonseparating coordinate must not miss a point");
    auto orbit = finite_locus(vars, {x * (x-one), x*y}, {x}, caps);
    ensure(orbit.certified && orbit.orbits.size()==1 &&
           orbit.orbits[0].coordinates[0] == std::vector<Q>{1} && orbit.orbits[0].coordinates[1].empty(),
           "missed conditional point on open");
    ensure(!finite_locus(vars, {x*y}, {}, caps).certified, "a positive-dimensional locus is not finite");
    ensure(finite_locus(vars, {x}, {x}, caps).certified, "excluded hyperplane");
    try {
        atlas::DeadlineScope deadline(std::chrono::steady_clock::now()-std::chrono::seconds(1));
        finite_locus(vars, {x,y}, {}, caps);
        throw std::logic_error("ignored caller deadline");
    } catch (const atlas::Expired&) {}
    auto limited = caps;
    limited.max_nonzeros = 12;
    ensure(!finite_locus(vars, {x*x-one, y*y-one}, {}, limited).certified, "ignored numerical range");
    std::cout << "Finite-locus tests pass.\n";
}
