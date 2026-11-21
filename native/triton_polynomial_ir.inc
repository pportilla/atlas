// The coefficient planes are signed magnitudes in base 2^15. Short values
// take one signed 64-bit lane. For dense univariate products, each lane
// evaluates an entire output coefficient, avoiding a quadratic transfer.
std::string polynomial_ir(const std::string& name, int block) {
    LaneKernel ir(block);
    const auto lane=ir.offsets(), live=ir.less(lane,ir.splat("%count"));
    const std::string signature="%left: !tt.ptr<i64>, %right: !tt.ptr<i64>, %out: !tt.ptr<i64>, "
        "%n: i32, %m: i32, %aw: i32, %bw: i32, %ow: i32, %count: i32";
    if (name=="poly_add_words") {
        auto a=ir.load("%left",lane,live,"i64"), b=ir.load("%right",lane,live,"i64");
        ir.store("%out",lane,ir.binary("addi",a,b,"i64"),live,"i64");
        return ir.source(name,signature);
    }
    const bool dense=name=="poly_dense_convolution", point=name=="poly_point_product" || name=="poly_point_words";
    const bool fast=name=="poly_outer_product" || name=="poly_point_product";
    if (!dense && !point && !fast && name!="poly_outer_words")
        throw std::invalid_argument("Unknown Triton polynomial kernel.");
    const auto m=ir.splat("%m");
    if (fast) {
        auto x=point ? lane : ir.binary("divui",lane,m);
        auto y=point ? lane : ir.binary("remui",lane,m);
        auto a=ir.load("%left",x,live,"i64"), b=ir.load("%right",y,live,"i64");
        ir.store("%out",lane,ir.binary("muli",a,b,"i64"),live,"i64");
        return ir.source(name,signature);
    }

    auto stride=ir.splat("%ow");
    auto pair=dense ? lane : ir.binary("divui",lane,stride);
    auto digit=dense ? lane : ir.binary("remui",lane,stride);
    auto ai=dense ? ir.constant(0) : ir.binary("muli",point ? pair : ir.binary("divui",pair,m),ir.splat("%aw"));
    auto bi=dense ? ir.constant(0) : ir.binary("muli",point ? pair : ir.binary("remui",pair,m),ir.splat("%bw"));
    auto extent=dense ? m : ir.splat("%bw");
    auto zero=ir.constant(0,"i64");
    auto index_zero=ir.result("arith.constant 0 : i32"), one=ir.result("arith.constant 1 : i32");
    auto total=ir.begin_sum_loop(index_zero,dense ? "%n" : "%aw",one,zero,"i64");
    auto j=ir.splat("%j"), k=ir.binary("subi",digit,j);
    // Unsigned comparison masks both out-of-range and negative indices.
    auto mask=ir.binary("andi",live,ir.compare("ult",k,extent,"i32"),"i1");
    auto a=ir.load("%left",ir.binary("addi",ai,j),mask,"i64");
    auto b=ir.load("%right",ir.binary("addi",bi,k),mask,"i64");
    ir.end_sum_loop(ir.binary("addi","%partial",ir.binary("muli",a,b,"i64"),"i64"),"i64");
    ir.store("%out",lane,total,live,"i64");
    return ir.source(name,signature);
}
