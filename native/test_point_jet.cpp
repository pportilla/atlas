#include "point_jet.hpp"
#include <iostream>

using namespace atlas::chart;
using namespace atlas::certificate;
namespace {
void ensure(bool ok, const char* msg) { if (!ok) throw std::logic_error(msg); }
}
int main() {
    Variables vars{{"x", 0}, {"y", 0}, {"z", 0}};
    const auto x = symbol(vars[0]), y = symbol(vars[1]), z = symbol(vars[2]);
    const auto a = constant(2), one = constant(1);
    CertificateCaps caps;
    GaloisOrbit rational{{0,1},{{2},{4},{8}}};
    // Two coupled graph rows, supplied in an order that needs a Jacobian swap.
    auto graph = frame_jet(vars, {z-x*y, y-x*x}, {vars[1],vars[2]}, z, rational, 4, caps);
    ensure(graph && graph->parameters == Variables{vars[0]} &&
           graph->coefficients == decltype(graph->coefficients){{{0},{8}},{{1},{12}},{{2},{6}},{{3},{1}}},
           "rational implicit jet differs from (2+h)^3");
    vars.pop_back();
    GaloisOrbit plus{{-2,0,1},{{2},{0,1}}};
    GaloisOrbit minus{{-2,0,1},{{2},{0,-1}}};
    Ideal rows{y*y-x};
    const auto square_root = frame_jet(vars, rows, {vars[1]}, y, plus, 4, caps);
    const auto conjugate = frame_jet(vars, rows, {vars[1]}, y, minus, 4, caps);
    ensure(square_root && conjugate, "smooth quadratic orbit incorrectly declined");
    using Terms = decltype(square_root->coefficients);
    Terms expected{{{0},{0,1}},{{1},{0,Q(1,4)}},{{2},{0,Q(-1,32)}},
                   {{3},{0,Q(1,128)}},{{4},{0,Q(-5,2048)}}};
    ensure(square_root->coefficients == expected, "positive root expansion");
    for (auto& [p,c] : expected) c[1] = -c[1];
    ensure(conjugate->coefficients == expected, "conjugate root expansion");
    const auto third = frame_jet(vars, rows, {vars[1]}, y*y-x + y*power(x-a,3), plus, 4, caps);
    ensure(third && jet_order(*third)==3 && third->coefficients.at({3}) == FieldCoefficient{0,1},
           "third-order cancellation in the quotient field");
    auto affine = taylor_jet(vars, y*x, plus, 2, caps);
    ensure(affine && affine->coefficients == Terms{{{0,0},{0,2}},{{1,0},{0,1}},{{0,1},{2}},{{1,1},{1}}},
           "algebraic affine jet");
    auto zero = frame_jet(vars, rows, {vars[1]}, rows.front(), plus, 4, caps);
    ensure(zero && zero->coefficients.empty() && jet_order(*zero)==5, "zero jet lower bound");
    // On x=2 with free y, a polynomial that first appears in degree 17
    // is nonzero on the germ even though all of its first 16 jets vanish.
    GaloisOrbit translated{{0,1},{{2},{3}}};
    auto line = [&](const Ideal& equations, const Expr& f, const GaloisOrbit& at) {
        return nonzero_on_coordinate_line(vars, equations, vars[1], f, at, caps);
    };
    ensure(line({x-a}, power(y-constant(3),17), translated) == true, "high-order nonzero line");
    ensure(line({x-a}, (x-a)*y, translated) == false, "zero on translated line");
    ensure(!line({power(x-a,2)}, x-a, translated), "singular line incorrectly certified");
    ensure(!line({x-a-power(y-constant(3),7)}, x-a, translated), "tangent line is not the frame germ");
    ensure(!line({x-a}, power(y,40), translated), "restriction crossed its degree bound");
    const GaloisOrbit imaginary{{1,0,1},{{0,1},{}}};
    ensure(line({x*x+one}, (x*x+one)*y, imaginary) == false, "quadratic orbit line restriction");
    ensure(line({x*x+one}, x*y, imaginary) == true, "nonzero on both conjugate lines");
    ensure(!frame_jet(vars, {power(rows.front(),2)}, {vars[1]}, y, plus, 4, caps), "singular frame");
    ensure(!frame_jet(vars, {rows.front()+one}, {vars[1]}, y, plus, 4, caps), "point off frame");
    auto malformed = plus;
    malformed.mu = {-1,0,1};
    ensure(!taylor_jet(vars,y,malformed,2,caps), "reducible modulus accepted");
    GaloisOrbit cubic{{1,1,0,1},{{Q(1,2),Q(1,2)}}};
    auto mixed = taylor_jet({vars[0]}, scaled(x*x,Q(1,3))+x, cubic, 2, caps);
    ensure(mixed && mixed->coefficients == Terms{{{0},{Q(7,12),Q(2,3),Q(1,12)}},
               {{1},{Q(4,3),Q(1,3)}},{{2},{Q(1,3)}}}, "mixed rational field coordinates");
    cubic.coordinates = {{-1},{0,1}};
    const auto cubic_row = power(y,3)+y-x;
    auto cubic_frame = frame_jet(vars,{cubic_row},{vars[1]},cubic_row+power(x+one,2),cubic,4,caps);
    ensure(cubic_frame && cubic_frame->coefficients == Terms{{{2},{1}}}, "cubic implicit frame");
    auto tiny = caps;
    tiny.max_nonzeros = 2;
    ensure(!frame_jet(vars,rows,{vars[1]},y,plus,4,tiny), "entry budget");
    GaloisOrbit origin{{0,1},{{},{}}};
    Z exponent = Z(1) << 90;
    auto wide = taylor_jet(vars, Expr({vars[0]},{{{exponent},Q(1)}}) + scaled(y,Q(exponent)), origin, 2,caps);
    ensure(wide && wide->coefficients == Terms{{{0,1},{Q(exponent)}}}, "wide exponent or coefficient narrowed");
    auto wide_field = taylor_jet(vars, scaled(x*y,Q(exponent)), plus, 2,caps);
    ensure(wide_field && wide_field->coefficients == Terms{{{0,0},{0,Q(2*exponent)}},
               {{1,0},{0,Q(exponent)}},{{0,1},{Q(2*exponent)}},{{1,1},{Q(exponent)}}},
           "wide binomial coordinate product narrowed");
    bool deadline = false;
    caps.check_time = []() { throw 17; };
    try { (void)taylor_jet(vars, y, plus, 2, caps); } catch (int) { deadline = true; }
    ensure(deadline, "deadline ignored");
    std::cout << "Rational and number-field point jets pass.\n";
}
