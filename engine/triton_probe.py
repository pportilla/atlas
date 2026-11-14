"""A one-word launch to check that the selected GPU can run Atlas integer kernels."""
import triton
import triton.language as tl


@triton.jit
def square_modulo(value_ptr):
    value = tl.load(value_ptr)
    tl.store(value_ptr, (value * value) % 1_000_000_007)
