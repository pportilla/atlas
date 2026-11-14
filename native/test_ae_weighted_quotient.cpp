#include "ae_graded.hpp"
#include "ae_weighted_ranks.hpp"
#include "local_io.hpp"
#include <iostream>

int main() {
    using namespace atlas::local;
    const std::vector<std::string> variables{"x","y"};
    // (xy, x²+y²) contains (x,y)^3. Thus k=3 is a valid contact
    // bound. The degree-two shift xy exercises both weight residues.
    for (const char* coefficient : {"1", "-7/3", "65521/18446744073709551617"}) {
        Basis map;
        for (const std::string& f : std::vector<std::string>{
                 std::string(coefficient)+"*x*y", "x^2+y^2", "x^3+y^3+x^2*y"})
            map.push_back(parse_expression(variables,f));
        const auto projected = ae_weighted_ranks(map,{1,1},3);
        const auto graph = ae_graded(map,variables);
        if (!projected || !graph || !(*projected)["finite"].asBool() ||
            (*projected)["value"].asString() != "12" || (*projected)["value"] != (*graph)["value"]) {
            std::cerr << "Weighted quotient disagrees with graph elimination at coefficient "
                      << coefficient << '\n';
            return 1;
        }
    }
}
