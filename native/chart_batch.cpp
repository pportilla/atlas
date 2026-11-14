// Exact toric chart exponents. The binding checks max(sum(exponent)) *
// max(ray coordinate) <= INT32_MAX and validates every cone index. Since
// all terms in the pairing are nonnegative, no intermediate sum overflows.
#include <array>
#include <cstdint>

namespace {
template<int N>
void transform(const int32_t* exps,const int32_t* rays,const int32_t* cones,
               const int32_t* orders,int terms,int charts,int32_t* out) noexcept {
    for (int chart=0;chart<charts;++chart) {
        const auto* cone=cones+N*chart;
        std::array<const int32_t*,N> directions;
        for (int axis=0;axis<N;++axis) directions[axis]=rays+N*cone[axis];
        for (int term=0;term<terms;++term) {
            const auto* e=exps+N*term;
            auto* dest=out+N*(chart*terms+term);
            for (int axis=0;axis<N;++axis) {
                const auto* r=directions[axis];
                int32_t value=0;
                for (int j=0;j<N;++j) value+=e[j]*r[j];
                dest[axis]=value-orders[cone[axis]];
            }
        }
    }
}
}

extern "C" void resolution_chart_cpu_nd(const int32_t* exps,const int32_t* rays,
                                        const int32_t* cones,const int32_t* orders,
                                        int terms,int charts,int dimension,int32_t* out) noexcept {
    using Kernel = decltype(&transform<2>);
    static const Kernel table[]={transform<2>,transform<3>,transform<4>,transform<5>,
                                 transform<6>,transform<7>,transform<8>};
    if(dimension>=2 && dimension<=8)
        table[dimension-2](exps,rays,cones,orders,terms,charts,out);
}

// The original three-dimensional ABI is retained for external callers.
extern "C" void resolution_chart_cpu(const int32_t* exps,const int32_t* rays,
                                     const int32_t* cones,const int32_t* orders,
                                     int terms,int charts,int32_t* out) noexcept {
    transform<3>(exps,rays,cones,orders,terms,charts,out);
}
extern "C" int resolution_chart_valid(const int32_t* output, int length) noexcept {
    for(int i=0;i<length;++i) if(output[i]<0) return 0;
    return 1;
}
