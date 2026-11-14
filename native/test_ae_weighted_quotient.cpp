#include "ae_graded.hpp"
#include "ae_weighted_quotient.hpp"
#include "ae_weighted_ranks.hpp"
#include "local_io.hpp"
#include <iostream>

// Each matrix has one free column. Only the second has a high minor
// bound: its diagonal has 65 large entries instead of one.
static bool check_height_budget(bool single_large_column) {
    using namespace atlas::local;
    size_t cells = 130000, updates = 100000;
    WeightedQuotient quotient(cells,updates);
    quotient.begin(65,66);
    const mpz_class large = mpz_class(1) << 1000;
    for (int i = 0; i < 65; ++i)
        quotient.add(i,i,single_large_column && i > 0 ? mpz_class(1) : large);
    try {
        return quotient.finish() == 1 && single_large_column;
    } catch (const WeightedRankLimit&) {
        return !single_large_column;
    }
}

int main() {
    using namespace atlas::local;
    if (!check_height_budget(true) || !check_height_budget(false)) {
        std::cerr << "Unexpected exact minor budget decision\n";
        return 1;
    }
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
