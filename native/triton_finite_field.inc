// Independent elements occupy lanes. Coefficients are canonical nonnegative residues.
std::string finite_field_ir(int action,int degree,int block) {
    if (action<0 || action>11 || degree<1 || degree>8)
        throw std::invalid_argument("Unsupported finite field kernel.");
    if (action>=6) return finite_binary_ir(action-6,degree,block);
    LaneKernel ir(block);
    using Vector=std::vector<std::string>;
    auto i=ir.offsets(), live=ir.less(i,ir.splat("%count"));
    auto p=ir.splat("%p","i64"),zero=ir.constant(0,"i64"),one=ir.constant(1,"i64");
    auto column=[&](int j) {
        return ir.binary("addi",i,ir.binary("muli",ir.constant(j),ir.splat("%count")));
    };
    auto op=[&](const char* name,const std::string& a,const std::string& b){return ir.binary(name,a,b,"i64");};
    auto add=[&](const std::string& a,const std::string& b) {
        auto sum=op("addi",a,b);
        return ir.choose(ir.compare("uge",sum,p,"i64"),op("subi",sum,p),sum,"i64");
    };
    auto sub=[&](const std::string& a,const std::string& b) {
        auto delta=op("subi",a,b);
        return ir.choose(ir.compare("slt",delta,zero,"i64"),op("addi",delta,p),delta,"i64");
    };
    auto mul=[&](const std::string& a,const std::string& b){return op("remui",op("muli",a,b),p);};
    Vector relation;
    for (int j=0;j<degree;++j) {
        auto scalar=ir.result("arith.constant "+std::to_string(j)+" : i32");
        auto ptr=ir.result("tt.addptr %relation, "+scalar+" : !tt.ptr<i64>, i32");
        relation.push_back(ir.splat(ir.result("tt.load "+ptr+" : !tt.ptr<i64>"),"i64"));
    }
    auto reduce=[&](Vector raw) {
        for (int k=static_cast<int>(raw.size())-1;k>=degree;--k)
            for (int j=0;j<degree;++j)
                raw[k-degree+j]=sub(raw[k-degree+j],mul(raw[k],relation[j]));
        raw.resize(degree);
        return raw;
    };
    auto multiply=[&](const Vector& a,const Vector& b) {
        Vector raw(2*degree-1,zero);
        for (int j=0;j<degree;++j) for (int k=0;k<degree;++k)
            raw[j+k]=add(raw[j+k],mul(a[j],b[k]));
        return reduce(std::move(raw));
    };
    Vector a,b,out;
    for (int k=0;k<(action==0?2*degree-1:degree);++k)
        a.push_back(ir.load("%values",column(k),live,"i64"));
    if (action>0 && action!=5)
        for (int k=0;k<degree;++k) b.push_back(ir.load("%values",column(degree+k),live,"i64"));
    if (action==0) out=reduce(std::move(a));
    else if (action<=2) {
        for (int k=0;k<degree;++k) out.push_back(action==1?add(a[k],b[k]):sub(a[k],b[k]));
    } else if (action==3) out=multiply(a,b);
    else {
        // In F_q, every nonzero element has inverse a^(q-2). Keep all squares and
        // partial products in registers throughout a single batched launch.
        auto start=ir.result("arith.constant 0 : i32"),step=ir.result("arith.constant 1 : i32");
        Vector base=action==5?a:b,acc,base_args;
        std::string vars, types, signature="tensor<"+std::to_string(block)+"xi64>";
        for (int k=0;k<2*degree;++k) {
            auto v="%state"+std::to_string(k);
            if (k) { vars+=", "; types+=", "; }
            vars+=v+" = "+(k>=degree ? base[k-degree] : (k==0 ? one : zero));
            types+=signature;
            (k<degree?acc:base_args).push_back(v);
        }
        ir.statement("    %powers:"+std::to_string(2*degree)+" = scf.for %bit = "+start+
                     " to %bits step "+step+" iter_args("+vars+") -> ("+types+") : i32 {\n");
        auto bit64=ir.result("arith.extui %bit : i32 to i64");
        auto shift=ir.result("arith.shrui %exponent, "+bit64+" : i64");
        auto odd=ir.compare("ne",op("andi",ir.splat(shift,"i64"),one),zero,"i64");
        auto prod=multiply(acc,base_args),sq=multiply(base_args,base_args);
        std::string yields;
        for (int k=0;k<degree;++k) {
            if (k) yields+=", ";
            yields+=ir.choose(odd,prod[k],acc[k],"i64");
        }
        for (const auto& coeff:sq) yields+=", "+coeff;
        ir.statement("      scf.yield "+yields+" : "+types+"\n    }\n");
        for (int k=0;k<degree;++k) out.push_back("%powers#"+std::to_string(k));
        if (action==4) out=multiply(a,out);
    }
    for (int k=0;k<degree;++k) ir.store("%out",column(k),out[k],live,"i64");
    return ir.source("finite_field_batch","%values: !tt.ptr<i64>, %relation: !tt.ptr<i64>, %out: !tt.ptr<i64>, "
                     "%count: i32, %p: i64, %exponent: i64, %bits: i32");
}
