// Let the Triton backend use the same bounded, integer-only level plan as OpenCL.
#include "trace_levels.hpp"

extern "C" uint32_t atlas_groebner_trace_levels(uint32_t width, const uint32_t* offsets,
        const uint32_t* targets, const uint32_t* coefficients, uint32_t* starts, uint32_t* parents,
        uint32_t* factors, uint32_t* order, uint32_t* levels) noexcept {
    try {
        return atlas::trace_levels(width, offsets, targets, coefficients, starts, parents, factors, order, levels);
    } catch (...) {
        return 0;
    }
}
