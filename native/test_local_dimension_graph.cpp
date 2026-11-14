#include "local_dimension_graph.hpp"
#include "local_io.hpp"
#include <iostream>
#include <stdexcept>

namespace {
using namespace atlas::local;

void check(const std::vector<std::string>& names, const std::vector<mpq_class>& point,
           const std::vector<std::string>& generators, int expected) {
    Basis ideal;
    for (const auto& f : generators) ideal.push_back(parse_expression(names, f));
    const int result = dimension_from_point_graph(ideal, point, false);
    if (result != expected) {
        std::cerr << "Expected graph dimension " << expected << ", got " << result << '\n';
        throw std::runtime_error("Incorrect point graph dimension.");
    }
}
}

int main() {
    const std::vector<std::string> xyz{"x", "y", "z"};
    // Plane and line intersect at (1,0,0). At x=0 only the line remains.
    const std::vector<std::string> components{"(x-1)*y", "(x-1)*z"};
    check(xyz, {1,0,0}, components, 2);
    check(xyz, {0,0,0}, components, 1);
    // The three equations have height two and no common factor. Their
    // tangent variety is a union of lines in non-coordinate directions.
    check(xyz, {1,1,0}, {"(x+y-2+z)*(x-y+z)", "(x+y-2+z)*(x-1+z)",
                          "(x-y+z)*(x-1+z)"}, 1);
    check({"x","y"}, {mpq_class(1,2),mpq_class(1,3)},
          {"(x-1/2)^2+(y-1/3)^3", "(y-1/3)^4"}, 0);
    check({"a","b","c","d","e","f"}, {0,0,0,0,0,0}, {"a*e-b*d", "a*f-c*d", "b*f-c*e"}, 4);
    // An exponent beyond the packed Gröbner range exercises the sparse
    // graph engine in a zero-centered axis, with a nonzero shifted axis.
    check({"x","y"}, {0,1}, {"x^18446744073709551623*(y-1)", "(y-1)^2"}, 1);
}
