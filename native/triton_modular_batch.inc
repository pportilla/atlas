// One modular operation per lane. EEA coefficients for 31-bit inputs fit in i64.
std::string residue_batch(int action, int block) {
    if (action<0 || action>8) throw std::invalid_argument("Unknown modular batch.");
    LaneKernel ir(block);
    const std::string ty="tensor<"+std::to_string(block)+"xi64>";
    const auto i=ir.offsets(), live=ir.less(i,ir.splat("%count"));
    const auto zero=ir.constant(0,"i64"), one=ir.constant(1,"i64");
    auto bin=[&](const std::string& op,const std::string& a,const std::string& b){return ir.binary(op,a,b,"i64");};
    auto mod=[&](const std::string& a,const std::string& m) {
        auto r=bin("remsi",a,m);
        return ir.choose(ir.compare("slt",r,zero,"i64"),bin("addi",r,m),r,"i64");
    };
    auto m=ir.splat("%modulus","i64"), n=ir.splat("%other_modulus","i64");
    auto a=ir.load("%left",i,live,"i64"), b=ir.load("%right",i,live,"i64");
    std::string answer, denominator=one;
    if (action<=3) {
        if (action==0) answer=mod(a,m);
        else answer=mod(bin(action==1?"addi":action==2?"subi":"muli",a,b),m);
    } else if (action==8) {
        const auto delta=ir.load("%aux",i,live,"i64");
        answer=bin("addi",a,bin("muli",m,mod(bin("muli",delta,b),n)));
    } else {
        const auto bound=action==6 ? ir.splat("%bound","i64") : zero;
        const auto first=action==6 ? a : b;
        auto start=ir.result("arith.constant 0 : i32"), end=ir.result("arith.constant 47 : i32");
        auto step=ir.result("arith.constant 1 : i32");
        // The Euclidean algorithm takes at most 45 iterations for these moduli.
        // Finished lanes keep their state; use 1 as their inactive divisor.
        const std::string types="("+ty+", "+ty+", "+ty+", "+ty+")";
        std::string loop="scf.for %k = "+start+" to "+end+" step "+step+
          " iter_args(%r0 = "+n+", %r1 = "+first+", %t0 = "+zero+", %t1 = "+one+") -> "+types+" : i32 {\n";
        auto select=[&](const char* name,const std::string& yes,const std::string& no){
            loop += "      "+std::string(name)+" = arith.select %active, "+yes+", "+no+" : tensor<"+
                    std::to_string(block)+"xi1>, "+ty+"\n";};
        loop+="      %active = arith.cmpi sgt, %r1, "+bound+" : "+ty+"\n";
        select("%safe","%r1",one);
        const std::string word="tensor<"+std::to_string(block)+"xi32>";
        loop+="      %old32 = arith.trunci %r0 : "+ty+" to "+word+"\n";
        loop+="      %safe32 = arith.trunci %safe : "+ty+" to "+word+"\n";
        loop+="      %quot32 = arith.divui %old32, %safe32 : "+word+"\n";
        loop+="      %q = arith.extui %quot32 : "+word+" to "+ty+"\n";
        loop+="      %qr = arith.muli %q, %r1 : "+ty+"\n";
        loop+="      %qt = arith.muli %q, %t1 : "+ty+"\n";
        loop+="      %nr = arith.subi %r0, %qr : "+ty+"\n";
        loop+="      %nt = arith.subi %t0, %qt : "+ty+"\n";
        select("%a0","%r1","%r0"); select("%a1","%nr","%r1");
        select("%b0","%t1","%t0"); select("%b1","%nt","%t1");
        loop+="      scf.yield %a0, %a1, %b0, %b1 : "+ty+", "+ty+", "+ty+", "+ty+"\n    }";
        // A four-result operation in textual MLIR uses one SSA name and numbered projections.
        ir.statement("    %eea:4 = "+loop+"\n");
        if (action==6) { answer="%eea#1"; denominator="%eea#3"; }
        else {
            auto good=ir.compare("eq","%eea#0",one,"i64");
            denominator=ir.choose(good,one,zero,"i64");
            if (action<=5) answer=mod(bin("muli",a,"%eea#2"),m);
            else {
                auto c=ir.load("%aux",i,live,"i64");
                answer=bin("addi",a,bin("muli",m,mod(bin("muli",c,"%eea#2"),n)));
            }
        }
    }
    ir.store("%out",i,answer,live,"i64");
    if (action>=4 && action!=8) ir.store("%den",i,denominator,live,"i64");
    return ir.source("residue_batch","%left: !tt.ptr<i64>, %right: !tt.ptr<i64>, %aux: !tt.ptr<i64>, "
                     "%out: !tt.ptr<i64>, %den: !tt.ptr<i64>, %count: i32, %modulus: i64, "
                     "%other_modulus: i64, %bound: i64");
}
