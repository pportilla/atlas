// Binary extension elements of degree <= 8 fit in one word. A carryless
// product has degree <= 14; reduce it by the monic defining polynomial.
std::string finite_binary_ir(int action,int degree,int block) {
    LaneKernel ir(block);
    const auto index=ir.offsets(), count=ir.splat("%count"), live=ir.less(index,count);
    const auto zero=ir.constant(0), one=ir.constant(1);
    auto column=[&](int j){return ir.binary("addi",index,ir.binary("muli",ir.constant(j),count));};
    auto shift=[&](const std::string& a,int n){return n==0?a:ir.binary("shli",a,ir.constant(n));};
    auto bit=[&](const std::string& a,int n) {
        return ir.binary("andi",n==0?a:ir.binary("shrui",a,ir.constant(n)),one);
    };
    auto mask=[&](const std::string& a,int n){return ir.binary("subi",zero,bit(a,n));};
    auto pack=[&](int start,int length){
        auto result=zero;
        for (int j=0;j<length;++j) {
            auto coeff=ir.cast(ir.load("%values",column(start+j),live,"i64"),"trunci","i64","i32");
            result=ir.binary("ori",result,shift(coeff,j));
        }
        return result;
    };
    auto relation=ir.constant(1<<degree);
    for (int j=0;j<degree;++j) {
        auto k=ir.result("arith.constant "+std::to_string(j)+" : i32");
        auto ptr=ir.result("tt.addptr %relation, "+k+" : !tt.ptr<i64>, i32");
        auto coeff=ir.result("tt.load "+ptr+" : !tt.ptr<i64>");
        relation=ir.binary("ori",relation,shift(ir.cast(ir.splat(coeff,"i64"),"trunci","i64","i32"),j));
    }
    auto reduce=[&](std::string a) {
        for (int j=2*degree-2;j>=degree;--j)
            a=ir.binary("xori",a,ir.binary("andi",mask(a,j),shift(relation,j-degree)));
        return a;
    };
    auto multiply=[&](const std::string& a,const std::string& b){
        auto result=zero;
        for (int j=0;j<degree;++j)
            result=ir.binary("xori",result,ir.binary("andi",mask(b,j),shift(a,j)));
        return reduce(result);
    };
    auto square=[&](const std::string& a) {
        auto result=zero;
        for (int j=0;j<degree;++j) result=ir.binary("ori",result,shift(bit(a,j),2*j));
        return reduce(result);
    };
    const auto a=pack(0,action==0 ? 2*degree-1 : degree);
    const auto b=action>0 && action!=5 ? pack(degree,degree) : a;
    std::string value;
    if (action==0) value=reduce(a);
    else if (action<=2) value=ir.binary("xori",a,b);
    else if (action==3) value=multiply(a,b);
    else {
        // After j iterations v=b^(2^(j+1)-1). One final square gives b^(2^d-2).
        auto inverse=b;
        for (int j=1;j<degree-1;++j) inverse=multiply(square(inverse),b);
        inverse=degree==1 ? one : square(inverse);
        value=action==4 ? multiply(a,inverse) : inverse;
    }
    for (int k=0;k<degree;++k)
        ir.store("%out",column(k),ir.cast(bit(value,k),"extui","i32","i64"),live,"i64");
    return ir.source("finite_field_batch","%values: !tt.ptr<i64>, %relation: !tt.ptr<i64>, %out: !tt.ptr<i64>, "
                     "%count: i32, %p: i64, %exponent: i64, %bits: i32");
}
