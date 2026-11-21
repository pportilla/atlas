// Emit target-independent Triton IR. All lanes operate on exact integers.
// The selected Triton driver compiles it for its current CUDA, HIP or XPU device.
#include "groebner.hpp"
#include <sstream>

namespace atlas {
std::string triton_rank_ir(const std::string&, int prime, int rows_block, int cols_block, bool row_basis);
std::string triton_trace_ir(const std::string&, const std::vector<int>&);

namespace {

class LaneKernel {
    std::string body;
    int next = 0;
    int width;

    std::string signature(const std::string& element) const {
        return "tensor<" + std::to_string(width) + "x" + element + ">";
    }

public:
    explicit LaneKernel(int block) : width(block) {
        if (block < 1 || block > 1024 || (block & (block-1)))
            throw std::invalid_argument("Expected a power-of-two Triton block.");
    }

    void statement(const std::string& code) { body += code; }
    std::string result(const std::string& operation) {
        std::string name = "%v"+std::to_string(++next);
        body += "    " + name + " = " + operation + "\n";
        return name;
    }
    std::string constant(int value, const std::string& type="i32") {
        return result("arith.constant dense<"+std::to_string(value)+"> : "+signature(type));
    }
    std::string splat(const std::string& scalar, const std::string& type="i32") {
        return result("tt.splat "+scalar+" : "+type+" -> "+signature(type));
    }
    std::string binary(const std::string& op, const std::string& a, const std::string& b,
                       const std::string& type="i32") {
        return result("arith."+op+" "+a+", "+b+" : "+signature(type));
    }
    std::string offsets(const std::string& axis="x") {
        const auto pid = result("tt.get_program_id "+axis+" : i32");
        const auto start = binary("muli", splat(pid), constant(width));
        const auto lane = result("tt.make_range {end = "+std::to_string(width)+" : i32, start = 0 : i32} : "+signature("i32"));
        return binary("addi", start, lane);
    }
    std::string less(const std::string& a, const std::string& b) {
        return result("arith.cmpi slt, "+a+", "+b+" : "+signature("i32"));
    }
    std::string cast(const std::string& value, const std::string& op, const std::string& from, const std::string& to) {
        return result("arith."+op+" "+value+" : "+signature(from)+" to "+signature(to));
    }
    std::string compare(const std::string& predicate, const std::string& a, const std::string& b, const std::string& type) {
        return result("arith.cmpi "+predicate+", "+a+", "+b+" : "+signature(type));
    }
    std::string choose(const std::string& mask, const std::string& a, const std::string& b, const std::string& type) {
        return result("arith.select "+mask+", "+a+", "+b+" : "+signature("i1")+", "+signature(type));
    }
    std::string pointer(const std::string& base, const std::string& index, const std::string& type="i32",
                        const std::string& index_type="i32") {
        auto ptr = "!tt.ptr<"+type+">";
        return result("tt.addptr "+splat(base, ptr)+", "+index+" : "+signature(ptr)+", "+signature(index_type));
    }
    std::string load(const std::string& base, const std::string& index, const std::string& mask,
                     const std::string& type="i32", const std::string& index_type="i32", int other=0) {
        return result("tt.load "+pointer(base,index,type,index_type)+", "+mask+", "+constant(other,type)+
                      " : "+signature("!tt.ptr<"+type+">"));
    }
    void store(const std::string& base, const std::string& index, const std::string& value, const std::string& mask,
               const std::string& type="i32", const std::string& index_type="i32") {
        body += "    tt.store "+pointer(base,index,type,index_type)+", "+value+", "+mask+" : "+signature("!tt.ptr<"+type+">")+"\n";
    }
    std::string begin_sum_loop(const std::string& start, const std::string& stop, const std::string& step,
                               const std::string& initial, const std::string& type) {
        const std::string name="%v"+std::to_string(++next);
        body+="    "+name+" = scf.for %j = "+start+" to "+stop+" step "+step+" iter_args(%partial = "+
              initial+") -> ("+signature(type)+") : i32 {\n";
        return name;
    }
    void end_sum_loop(const std::string& value,const std::string& type) {
        body+="      scf.yield "+value+" : "+signature(type)+"\n    }\n";
    }
    std::string source(const std::string& name, const std::string& args) const {
        return "module {\n  tt.func public @"+name+"("+args+") {\n"+body+"    tt.return\n  }\n}\n";
    }
};

#include "triton_polynomial_ir.inc"
#include "triton_modular_batch.inc"
#include "triton_binary_field.inc"
#include "triton_finite_field.inc"

// One lane per residue of the Mersenne prime 8191. Reduced Horner products
// fit in unsigned 26 bits, so a shift and mask replace integer division.
std::string root_screen(int block) {
    LaneKernel ir(block);
    const auto x=ir.offsets(), prime=ir.constant(8191), zero=ir.constant(0);
    const auto live=ir.less(x,prime);
    const auto type="tensor<"+std::to_string(block)+"xi32>";
    const auto mask="tensor<"+std::to_string(block)+"xi1>";
    const auto begin=ir.result("arith.constant 0 : i32"), one=ir.result("arith.constant 1 : i32");
    const auto shift=ir.constant(13);
    const auto last=ir.result("arith.subi %length, "+one+" : i32");
    auto loop="scf.for %j = "+begin+" to %length step "+one+" iter_args(%acc = "+zero+") -> ("+type+") : i32 {\n";
    loop+="      %index = arith.subi "+last+", %j : i32\n";
    loop+="      %ptr = tt.addptr %coeffs, %index : !tt.ptr<i32>, i32\n      %scalar = tt.load %ptr : !tt.ptr<i32>\n";
    loop+="      %c = tt.splat %scalar : i32 -> "+type+"\n";
    loop+="      %product = arith.muli %acc, "+x+" : "+type+"\n      %sum = arith.addi %product, %c : "+type+"\n";
    loop+="      %low = arith.andi %sum, "+prime+" : "+type+"\n      %high = arith.shrui %sum, "+shift+" : "+type+"\n";
    loop+="      %fold = arith.addi %low, %high : "+type+"\n      %large = arith.cmpi uge, %fold, "+prime+" : "+type+"\n";
    loop+="      %diff = arith.subi %fold, "+prime+" : "+type+"\n      %reduced = arith.select %large, %diff, %fold : "+mask+", "+type+"\n";
    loop+="      scf.yield %reduced : "+type+"\n    }";
    auto root=ir.compare("eq",ir.result(loop),zero,"i32");
    ir.store("%out",x,ir.cast(root,"extui","i1","i32"),live);
    return ir.source("poly_root_screen","%coeffs: !tt.ptr<i32>, %out: !tt.ptr<i32>, %length: i32");
}

// In each coordinate with u_j > v_j, a rewrite consumes u_j-v_j.
// Other coordinates only constrain the first step. The minimum is exact.
std::string binomial_steps(int width, int block) {
    if (width<1 || width>32) throw std::invalid_argument("Unsupported binomial divisibility width.");
    LaneKernel ir(block);
    const auto index=ir.offsets(), live=ir.less(index,ir.splat("%terms"));
    const auto base=ir.binary("muli",ir.binary("addi",index,ir.constant(2)),ir.constant(width));
    auto bound=ir.constant(2147483647), zero=ir.constant(0), one=ir.constant(1);
    for (int j=0;j<width;++j) {
        const auto axis=ir.constant(j);
        auto u=ir.load("%powers",axis,live), v=ir.load("%powers",ir.constant(width+j),live);
        auto e=ir.load("%powers",ir.binary("addi",base,axis),live);
        auto dec=ir.binary("subi",u,v);
        auto gap=ir.binary("subi",e,u);
        auto div=ir.binary("divsi",gap,ir.binary("maxsi",dec,one));
        auto limit=ir.binary("addi",div,one);
        bound=ir.choose(ir.less(zero,dec),ir.binary("minsi",bound,limit),bound,"i32");
        bound=ir.choose(ir.less(e,u),zero,bound,"i32");
    }
    ir.store("%out",index,bound,live);
    return ir.source("poly_binomial_steps","%powers: !tt.ptr<i32>, %out: !tt.ptr<i32>, %terms: i32");
}

std::string replay_clear(int block) {
    LaneKernel ir(block);
    const auto column=ir.offsets("y"), row=ir.splat(ir.result("tt.get_program_id x : i32"));
    const auto base=ir.binary("muli",ir.cast(row,"extsi","i32","i64"),
                               ir.cast(ir.splat("%stride"),"extsi","i32","i64"),"i64");
    const auto index=ir.binary("addi",base,ir.cast(column,"extsi","i32","i64"),"i64");
    ir.store("%work",index,ir.constant(0,"i64"),ir.less(column,ir.splat("%width")),"i64","i64");
    return ir.source("replay_clear","%work: !tt.ptr<i64>, %stride: i32, %width: i32");
}

std::string charts(const std::string& op, int dimension, int block) {
    if (dimension < 1 || dimension > 64) throw std::invalid_argument("Unsupported chart dimension.");
    LaneKernel ir(block);
    const auto offset = ir.offsets(), live = ir.less(offset, ir.splat("%size"));
    const auto width = ir.constant(dimension), terms = ir.splat("%terms");
    std::string ray, term;
    if (op == "ray_pairs") {
        ray = ir.binary("divsi", offset, terms);
        term = ir.binary("remsi", offset, terms);
    } else {
        const auto pair = ir.binary("divsi", offset, width);
        const auto chart = ir.binary("divsi", pair, terms);
        const auto column = ir.binary("remsi", offset, width);
        const auto position = ir.binary("addi", ir.binary("muli", chart, width), column);
        ray = ir.load("%cones", position, live);
        term = ir.binary("remsi", pair, terms);
    }
    if (op == "gather_charts") {
        auto index = ir.binary("addi", ir.binary("muli", ray, terms), term);
        ir.store("%out", offset, ir.load("%table", index, live), live);
        return ir.source(op, "%table: !tt.ptr<i32>, %cones: !tt.ptr<i32>, %out: !tt.ptr<i32>, %terms: i32, %size: i32");
    }
    auto pairing = ir.constant(0);
    auto base = ir.binary("muli", term, width), direction = ir.binary("muli", ray, width);
    for (int k = 0; k < dimension; ++k) {
        auto axis = ir.constant(k);
        auto e = ir.load("%exponents", ir.binary("addi", base, axis), live);
        auto r = ir.load("%rays", ir.binary("addi", direction, axis), live);
        pairing = ir.binary("addi", pairing, ir.binary("muli", e, r));
    }
    ir.store("%out", offset, ir.binary("subi", pairing, ir.load("%orders", ray, live)), live);
    return ir.source(op, "%exponents: !tt.ptr<i32>, %rays: !tt.ptr<i32>, %cones: !tt.ptr<i32>, "
                         "%orders: !tt.ptr<i32>, %out: !tt.ptr<i32>, %terms: i32, %size: i32");
}

std::string divisors(int variables, int block) {
    if (variables < 1 || variables > 64) throw std::invalid_argument("Unsupported monomial dimension.");
    LaneKernel ir(block);
    const auto row = ir.offsets(), live = ir.less(row, ir.splat("%terms"));
    const auto type = "tensor<"+std::to_string(block)+"xi32>";
    const auto booltype = "tensor<"+std::to_string(block)+"xi1>";
    const auto base = ir.binary("muli", row, ir.constant(variables));
    const auto zero = ir.result("arith.constant 0 : i32"), one = ir.result("arith.constant 1 : i32");
    const auto width = ir.result("arith.constant "+std::to_string(variables)+" : i32");
    const auto none = ir.constant(-1);
    std::vector<std::string> powers;
    for (int k=0; k<variables; ++k)
        powers.push_back(ir.load("%powers", ir.binary("addi", base, ir.constant(k)), live));
    // The first dividing generator is the witness. Emit each comparison in
    // the loop over generators; coordinates are unrolled for this ring.
    std::string loop = "scf.for %j = "+zero+" to %divisors step "+one+" iter_args(%w = "+none+") -> ("+type+") : i32 {\n";
    loop += "      %base = arith.muli %j, "+width+" : i32\n";
    loop += "      %free = arith.cmpi eq, %w, "+none+" : "+type+"\n";
    std::string match = "%free";
    for (int k=0; k<variables; ++k) {
        const auto suffix = std::to_string(k);
        loop += "      %k"+suffix+" = arith.constant "+suffix+" : i32\n";
        loop += "      %idx"+suffix+" = arith.addi %base, %k"+suffix+" : i32\n";
        loop += "      %addr"+suffix+" = tt.addptr %basis, %idx"+suffix+" : !tt.ptr<i32>, i32\n";
        loop += "      %b"+suffix+" = tt.load %addr"+suffix+" : !tt.ptr<i32>\n";
        loop += "      %d"+suffix+" = tt.splat %b"+suffix+" : i32 -> "+type+"\n";
        loop += "      %cmp"+suffix+" = arith.cmpi sle, %d"+suffix+", "+powers[k]+" : "+type+"\n";
        loop += "      %m"+suffix+" = arith.andi "+match+", %cmp"+suffix+" : "+booltype+"\n";
        match = "%m"+suffix;
    }
    loop += "      %which = tt.splat %j : i32 -> "+type+"\n";
    loop += "      %chosen = arith.select "+match+", %which, %w : "+booltype+", "+type+"\n";
    loop += "      scf.yield %chosen : "+type+"\n    }";
    ir.store("%out", row, ir.result(loop), live);
    return ir.source("monomial_witness", "%powers: !tt.ptr<i32>, %basis: !tt.ptr<i32>, %out: !tt.ptr<i32>, "
                                         "%terms: i32, %divisors: i32");
}

// One coefficient per lane and one pair of primes per grid row.  Input
// primes are below 2^31, so all products and the combined residue fit i64.
std::string crt_pair(int block) {
    LaneKernel ir(block);
    auto pair = ir.result("tt.get_program_id y : i32");
    auto index = ir.offsets(), live = ir.less(index, ir.splat("%width"));
    auto pair_base = ir.result("arith.muli "+pair+", %width : i32");
    auto output = ir.binary("addi", ir.splat(pair_base), index);
    auto image_base = ir.binary("addi", ir.splat(pair_base), output);
    auto second_base = ir.binary("addi", image_base, ir.splat("%width"));
    auto wide = [&](const std::string& v) { return ir.cast(v,"extui","i32","i64"); };
    auto lhs = wide(ir.load("%images", image_base, live));
    auto rhs = wide(ir.load("%images", second_base, live));
    auto param = ir.binary("muli", ir.splat(pair), ir.constant(3));
    auto p = wide(ir.load("%moduli", param, live, "i32", "i32", 1));
    auto q = wide(ir.load("%moduli", ir.binary("addi", param, ir.constant(1)), live, "i32", "i32", 1));
    auto inverse = wide(ir.load("%moduli", ir.binary("addi", param, ir.constant(2)), live));
    // Reducing lhs first handles arbitrary ordering of the two primes.
    auto lmod = ir.binary("remui", lhs, q, "i64");
    auto delta = ir.binary("subi", ir.binary("addi", rhs, q, "i64"), lmod, "i64");
    auto correction = ir.binary("remui", ir.binary("muli", delta, inverse, "i64"), q, "i64");
    auto result = ir.binary("addi", lhs, ir.binary("muli", p, correction, "i64"), "i64");
    ir.store("%result", output, result, live, "i64");
    return ir.source("crt_pair", "%images: !tt.ptr<i32>, %moduli: !tt.ptr<i32>, %result: !tt.ptr<i64>, %width: i32");
}

// One lane computes one Leibniz product for a small exact determinant.
// A native bound on the product of absolute row sums proves that the
// products and every partial sum fit signed i64, including the reduction.
std::string determinant_small(int size) {
    if (size < 1 || size > 6) throw std::invalid_argument("Unsupported direct determinant dimension.");
    int factorial = 1;
    for (int i = 2; i <= size; ++i) factorial *= i;
    int width = 1;
    while (width < factorial) width *= 2;
    LaneKernel ir(width);
    const auto vec = "tensor<"+std::to_string(width)+"xi64>";
    const auto lane = ir.result("tt.make_range {end = "+std::to_string(width)+
                                " : i32, start = 0 : i32} : tensor<"+std::to_string(width)+"xi32>");
    const auto live = ir.less(lane, ir.constant(factorial));
    auto parity = ir.constant(0);
    std::vector<std::string> positions(size);
    // Lehmer codes enumerate permutations without a hardware-dependent table
    // of lane constants.  Every digit counts precisely its inversions.
    for (int i = 0; i < size; ++i) {
        factorial /= size-i;
        const auto quotient = ir.binary("divui", lane, ir.constant(factorial));
        positions[i] = ir.binary("remui", quotient, ir.constant(size-i));
        parity = ir.binary("xori", parity, positions[i]);
    }
    const auto odd = ir.compare("ne", ir.binary("andi", parity, ir.constant(1)), ir.constant(0),"i32");
    auto value = ir.choose(odd, ir.constant(-1,"i64"), ir.constant(1,"i64"), "i64");
    value = ir.choose(live, value, ir.constant(0,"i64"), "i64");
    for (int i = size-1; i >= 0; --i) {
        for (int j = i+1; j < size; ++j) {
            const auto bigger = ir.compare("sge", positions[j], positions[i], "i32");
            positions[j] = ir.binary("addi", positions[j], ir.cast(bigger,"extui","i1","i32"));
        }
    }
    for (int i = 0; i < size; ++i) {
        const auto index = ir.binary("addi", positions[i], ir.constant(i*size));
        value = ir.binary("muli", value, ir.load("%matrix", index, live, "i64"), "i64");
    }
    auto sum = ir.result("\"tt.reduce\"("+value+") <{axis = 0 : i32}> ({\n"
        "    ^bb0(%a: i64, %b: i64):\n      %v = arith.addi %a, %b : i64\n"
        "      tt.reduce.return %v : i64\n    }) : ("+vec+") -> i64");
    auto result = ir.splat(sum, "i64");
    ir.store("%det",lane,result,ir.compare("eq",lane,ir.constant(0),"i32"),"i64");
    return ir.source("determinant_small", "%matrix: !tt.ptr<i64>, %det: !tt.ptr<i64>");
}

std::string replay(const std::string& name, int block, int opcode = 4) {
    LaneKernel ir(block);
    const auto zero = ir.constant(0, "i64"), one = ir.constant(1, "i64");
    const bool by_prime = name == "replay_pivot" || name == "replay_inverse";
    const auto lane = ir.offsets(by_prime ? "x" : "y");
    const auto live = ir.less(lane, ir.splat(by_prime ? "%items" : "%count"));
    auto wide = [&](const std::string& v) { return ir.cast(v, "extsi", "i32", "i64"); };
    auto narrow = [&](const std::string& v) { return ir.cast(v, "trunci", "i64", "i32"); };
    auto mul = [&](const auto& a, const auto& b){ return ir.binary("muli", a, b, "i64"); };
    auto add = [&](const auto& a, const auto& b){ return ir.binary("addi", a, b, "i64"); };
    auto sub = [&](const auto& a, const auto& b){ return ir.binary("subi", a, b, "i64"); };
    const auto item = by_prime ? lane : ir.splat(ir.result("tt.get_program_id x : i32"));
    const auto prime = wide(ir.load("%prime", item, live, "i32", "i32", 3));
    auto mod = [&](const auto& v){ return ir.binary("remsi", v, prime, "i64"); };
    const auto at_work = mul(wide(item), wide(ir.splat("%work_size")));
    const auto term = by_prime ? ir.splat("%column") : ir.load("%indices", lane, live);
    auto work_index = add(at_work, wide(term));
    auto work = ir.load("%work", work_index, live, "i64", "i64");
    if (by_prime) {
        auto residue = mod(work);
        auto missing = ir.compare("eq", residue, zero, "i64");
        std::string signature = "%work: !tt.ptr<i64>, %factor: !tt.ptr<i32>, %prime: !tt.ptr<i32>, ";
        if (name == "replay_pivot") {
            ir.store("%work", work_index, zero, live, "i64", "i64");
            ir.store("%factor", item, narrow(ir.choose(missing, zero, sub(prime, residue), "i64")), live);
        } else {
            signature += "%status: !tt.ptr<i32>, ";
            ir.store("%status", item, ir.constant(0), ir.binary("andi", live, missing, "i1"));
            // Fermat inversion. The primes lie below 2^31 and each
            // multiplication is reduced before the next squaring.
            std::string power = residue, value = one, exponent = sub(prime, ir.constant(2, "i64"));
            for (int bit=0; bit<31; ++bit) {
                auto low = ir.binary("andi", exponent, one, "i64");
                auto odd = ir.compare("ne", low, zero, "i64");
                value = ir.choose(odd, mod(mul(value, power)), value, "i64");
                if (bit < 30) {
                    power = mod(mul(power, power));
                    exponent = ir.binary("shrui", exponent, one, "i64");
                }
            }
            ir.store("%factor", item, narrow(value), live);
        }
        return ir.source(name, signature+"%work_size: i32, %column: i32, %items: i32");
    }

    if (name == "replay_check") {
        auto wrong = ir.binary("andi", live, ir.compare("ne", mod(work), zero, "i64"), "i1");
        // Multiple blocks can reject the same prime, so the flag write
        // must be atomic. All live output lanes store the same zero.
        auto ptr = ir.pointer("%status", item);
        ir.result("tt.atomic_rmw and, relaxed, gpu, "+ptr+", "+ir.constant(0)+", "+wrong+
                  " : (tensor<"+std::to_string(block)+"x!tt.ptr<i32>>, tensor<"+std::to_string(block)+
                  "xi32>, tensor<"+std::to_string(block)+"xi1>) -> tensor<"+std::to_string(block)+"xi32>");
        return ir.source(name, "%work: !tt.ptr<i64>, %indices: !tt.ptr<i32>, %prime: !tt.ptr<i32>, "
                               "%status: !tt.ptr<i32>, %work_size: i32, %count: i32");
    }
    if (name != "replay_axpy" && name != "replay_scale") throw std::invalid_argument("Unknown replay operation.");
    auto pool_index = add(mul(wide(item), wide(ir.splat("%pool_size"))), add(wide(ir.splat("%base")), wide(lane)));
    // Additions and subtractions take their scalar directly from the
    // opcode. Only pivot updates and monic normalization need a buffer.
    auto factor = opcode == 2 ? one : opcode == 3 ? sub(prime, one) : wide(ir.load("%factor", item, live));
    if (name == "replay_axpy") {
        const auto pool = wide(ir.load("%pool", pool_index, live, "i32", "i64"));
        // Work entries stay below p squared between normalizations.
        auto sum = add(work, mul(factor, pool)), square = mul(prime, prime);
        auto result = ir.choose(ir.compare("sge", sum, square, "i64"), sub(sum, square), sum, "i64");
        ir.store("%work", work_index, result, live, "i64", "i64");
    } else {
        auto value = ir.choose(ir.compare("eq", lane, ir.constant(0), "i32"), one, mod(mul(mod(work), factor)), "i64");
        ir.store("%pool", pool_index, narrow(value), live, "i32", "i64");
    }
    return ir.source(name, "%work: !tt.ptr<i64>, %pool: !tt.ptr<i32>, %indices: !tt.ptr<i32>, "
                           "%factor: !tt.ptr<i32>, %prime: !tt.ptr<i32>, %work_size: i32, "
                           "%pool_size: i32, %base: i32, %count: i32");
}

} // namespace

extern "C" char* atlas_triton_ir(const char* input, size_t length) noexcept {
    try {
        std::istringstream request(std::string(input, length));
        std::string name;
        request >> name;
        std::vector<int> parameters;
        int value;
        while (request >> value) parameters.push_back(value);
        if (!request.eof()) throw std::invalid_argument("Invalid GPU kernel parameter.");
        if (name == "poly_root_screen" && parameters.size() == 1) return copy_text(root_screen(parameters[0]));
        if (name == "poly_binomial_steps" && parameters.size() == 2) return copy_text(binomial_steps(parameters[0],parameters[1]));
        if (name.rfind("poly_", 0)==0 && parameters.size()==1)
            return copy_text(polynomial_ir(name, parameters[0]));
        if (name.rfind("trace_", 0) == 0) return copy_text(triton_trace_ir(name, parameters));
        if (name == "finite_field_batch" && parameters.size()==3)
            return copy_text(finite_field_ir(parameters[0],parameters[1],parameters[2]));
        if (name == "residue_batch" && parameters.size()==2)
            return copy_text(residue_batch(parameters[0],parameters[1]));
        if (name == "crt_pair" && parameters.size() == 1)
            return copy_text(crt_pair(parameters[0]));
        if (name == "determinant_small" && parameters.size() == 1)
            return copy_text(determinant_small(parameters[0]));
        if (name.rfind("rank_", 0) == 0 && parameters.size() == 4)
            return copy_text(triton_rank_ir(name, parameters[0], parameters[1], parameters[2], parameters[3] != 0));
        if (name == "replay_clear" && parameters.size()==1) return copy_text(replay_clear(parameters[0]));
        if (name == "replay_axpy" && parameters.size() == 2 && parameters[1] >= 2 && parameters[1] <= 4)
            return copy_text(replay(name, parameters[0], parameters[1]));
        if (name.rfind("replay_", 0) == 0 && parameters.size() == 1)
            return copy_text(replay(name, parameters[0]));
        if (parameters.size() == 2) {
            if (name == "ray_pairs" || name == "strict_charts" || name == "gather_charts")
                return copy_text(charts(name, parameters[0], parameters[1]));
            if (name == "monomial_witness") return copy_text(divisors(parameters[0], parameters[1]));
        }
        throw std::invalid_argument("Unknown Triton kernel.");
    } catch (const std::exception& error) { set_error(error.what()); }
    catch (...) { set_error("Unable to emit Triton IR."); }
    return nullptr;
}

} // namespace atlas
