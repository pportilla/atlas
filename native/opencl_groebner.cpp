// Optional OpenCL 1.2 plugin for the modular Groebner route (modular.cpp):
// it runs one replay program for a batch of primes, one work item per prime.
// Every work item executes the same instructions on its own coefficients, so
// the items never diverge. The CPU library does not link against OpenCL; the
// host passes atlas_replay_opencl to atlas_groebner_use_batch.
#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {

// The instruction set of modular.cpp. Work entries stay below p*p < 2^62.
const char* source = R"CLC(
kernel void replay(global const uint *code, uint from, uint to, uint pool_size, uint work_size,
                   global const uint *primes, global uint *pools, global ulong *works, global uint *status) {
    const size_t item = get_global_id(0);
    if (!status[item]) return;
    const uint p = primes[item];
    const ulong square = (ulong)p * p;
    global uint *pool = pools + item * (size_t)pool_size;
    global ulong *work = works + item * (size_t)work_size;
    uint at = from;
    while (at < to) {
        const uint op = code[at];
        if (op == 1) {
            const uint width = code[at + 1];
            for (uint t = 0; t < width; ++t) work[t] = 0;
            at += 2;
        } else if (op == 2 || op == 3) {
            global const uint *src = pool + code[at + 1];
            const uint count = code[at + 2];
            global const uint *index = code + at + 3;
            for (uint t = 0; t < count; ++t) {
                ulong w = work[index[t]] + (op == 2 ? src[t] : p - src[t]);
                work[index[t]] = w >= square ? w - square : w;
            }
            at += 3 + count;
        } else if (op == 4) {
            const uint column = code[at + 1];
            global const uint *src = pool + code[at + 2];
            const uint count = code[at + 3];
            global const uint *index = code + at + 4;
            const ulong f = work[column] % p;
            work[column] = 0;
            if (f) {
                const ulong factor = p - f;
                for (uint t = 0; t < count; ++t) {
                    ulong w = work[index[t]] + factor * src[t];
                    work[index[t]] = w >= square ? w - square : w;
                }
            }
            at += 4 + count;
        } else if (op == 5) {
            const uint count = code[at + 1];
            global const uint *index = code + at + 2;
            for (uint t = 0; t < count; ++t)
                if (work[index[t]] % p) { status[item] = 0; return; }
            at += 2 + count;
        } else if (op == 6) {
            global uint *target = pool + code[at + 1];
            const uint count = code[at + 2];
            global const uint *index = code + at + 3;
            ulong base = work[index[0]] % p;
            if (!base) { status[item] = 0; return; }
            ulong inverse = 1;
            for (uint e = p - 2; e; e >>= 1) {
                if (e & 1) inverse = inverse * base % p;
                base = base * base % p;
            }
            target[0] = 1;
            for (uint t = 1; t < count; ++t) target[t] = (uint)((work[index[t]] % p) * inverse % p);
            at += 3 + count;
        } else {
            status[item] = 0;
            return;
        }
    }
}
)CLC";

void check(cl_int status) {
    if (status != CL_SUCCESS) throw std::runtime_error("OpenCL call failed");
}

template <typename Handle, cl_int (*Release)(Handle)> struct Owned {
    Handle value = nullptr;
    ~Owned() { if (value) Release(value); }
    Owned() = default;
    Owned(const Owned&) = delete;
    Owned& operator=(const Owned&) = delete;
};
using Buffer = Owned<cl_mem, clReleaseMemObject>;

struct Runtime {
    Owned<cl_context, clReleaseContext> context;
    Owned<cl_program, clReleaseProgram> program;
    Owned<cl_command_queue, clReleaseCommandQueue> queue;
    Owned<cl_kernel, clReleaseKernel> kernel;
    cl_device_id device = nullptr;

    Runtime() {
        std::vector<cl_device_id> found;
        cl_uint count = 0;
        if (clGetPlatformIDs(0, nullptr, &count) != CL_SUCCESS || !count) throw std::runtime_error("No OpenCL platform");
        std::vector<cl_platform_id> platforms(count);
        check(clGetPlatformIDs(count, platforms.data(), nullptr));
        for (cl_platform_id platform : platforms) {
            cl_uint n = 0;
            if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &n) != CL_SUCCESS || !n) continue;
            std::vector<cl_device_id> ids(n);
            check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, n, ids.data(), nullptr));
            found.insert(found.end(), ids.begin(), ids.end());
        }
        // The same choice of device as the other OpenCL kernels of Atlas.
        const char* wanted = std::getenv("RESOLUTION_GPU_INDEX");
        const size_t index = wanted ? static_cast<size_t>(std::strtoul(wanted, nullptr, 10)) : 0;
        if (index >= found.size()) throw std::runtime_error("No such OpenCL GPU");
        device = found[index];
        cl_int status;
        context.value = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
        check(status);
        program.value = clCreateProgramWithSource(context.value, 1, &source, nullptr, &status);
        check(status);
        check(clBuildProgram(program.value, 1, &device, "-cl-std=CL1.2", nullptr, nullptr));
        queue.value = clCreateCommandQueue(context.value, device, 0, &status);
        check(status);
        kernel.value = clCreateKernel(program.value, "replay", &status);
        check(status);
    }
};

std::mutex lock;  // one batch at a time: the kernel object holds its arguments

Runtime& runtime() {
    static std::unique_ptr<Runtime> instance;
    if (!instance) instance = std::make_unique<Runtime>();
    return *instance;
}

template <typename T> void arg(cl_kernel kernel, cl_uint index, T value) {
    check(clSetKernelArg(kernel, index, sizeof(T), &value));
}

}  // namespace

extern "C" {

// Run code[0, length) for `count` primes. `pools` holds count*pool words, the
// generators already in place; `status` receives 1 for every prime whose run
// passed all checks. Returns 0 when the batch was run on the device.
int atlas_replay_opencl(const uint32_t* code, size_t length, uint32_t pool, uint32_t work, const uint32_t* primes,
                        uint32_t count, uint32_t* pools, uint32_t* status) noexcept {
    try {
        if (!count || length > 0xFFFFFFFFu) return 1;
        std::lock_guard<std::mutex> guard(lock);
        Runtime& rt = runtime();
        cl_int error;
        const size_t pool_bytes = static_cast<size_t>(count) * std::max<uint32_t>(pool, 1) * sizeof(uint32_t);
        const size_t work_bytes = static_cast<size_t>(count) * std::max<uint32_t>(work, 1) * sizeof(cl_ulong);
        std::vector<uint32_t> ok(count, 1);
        Buffer program, moduli, coefficients, scratch, flags;
        program.value = clCreateBuffer(rt.context.value, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       std::max<size_t>(length, 1) * sizeof(uint32_t), const_cast<uint32_t*>(code), &error);
        check(error);
        moduli.value = clCreateBuffer(rt.context.value, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      count * sizeof(uint32_t), const_cast<uint32_t*>(primes), &error);
        check(error);
        coefficients.value = clCreateBuffer(rt.context.value, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pool_bytes, pools, &error);
        check(error);
        scratch.value = clCreateBuffer(rt.context.value, CL_MEM_READ_WRITE, work_bytes, nullptr, &error);
        check(error);
        flags.value = clCreateBuffer(rt.context.value, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     count * sizeof(uint32_t), ok.data(), &error);
        check(error);
        cl_kernel kernel = rt.kernel.value;
        arg(kernel, 0, program.value);
        arg<cl_uint>(kernel, 3, pool);
        arg<cl_uint>(kernel, 4, work);
        arg(kernel, 5, moduli.value);
        arg(kernel, 6, coefficients.value);
        arg(kernel, 7, scratch.value);
        arg(kernel, 8, flags.value);
        // Whole blocks at a time, so that no single launch runs for long.
        const size_t global = count;
        const size_t slice = 1u << 20;
        size_t from = 0;
        while (from < length) {
            size_t to = from, at = from;
            while (at < length) {
                const uint32_t op = code[at];
                size_t next;
                if (op == 1) {
                    if (at > from && at - from >= slice) break;  // a block starts here
                    next = at + 2;
                } else if (op == 2 || op == 3 || op == 6) next = at + 3 + code[at + 2];
                else if (op == 4) next = at + 4 + code[at + 3];
                else if (op == 5) next = at + 2 + code[at + 1];
                else return 1;
                at = to = next;
            }
            arg<cl_uint>(kernel, 1, static_cast<cl_uint>(from));
            arg<cl_uint>(kernel, 2, static_cast<cl_uint>(to));
            check(clEnqueueNDRangeKernel(rt.queue.value, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr));
            from = to;
        }
        check(clEnqueueReadBuffer(rt.queue.value, coefficients.value, CL_TRUE, 0, pool_bytes, pools, 0, nullptr, nullptr));
        check(clEnqueueReadBuffer(rt.queue.value, flags.value, CL_TRUE, 0, count * sizeof(uint32_t), status, 0, nullptr, nullptr));
        return 0;
    } catch (...) {
        return 1;
    }
}

// 1 when an OpenCL GPU is usable for replays.
int atlas_replay_opencl_available() noexcept {
    try {
        std::lock_guard<std::mutex> guard(lock);
        runtime();
        return 1;
    } catch (...) {
        return 0;
    }
}

}
